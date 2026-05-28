"""
清洗环境
========

训练和推理时使用的清洗环境。

训练时：用注入的错误，repair_value 用注入前的值（当作真值）
推理时：用真正的错误，repair_value 用提供的真值

回归/分类兼容：
  - 分数归一化: 分类用 accuracy [0,1], 回归用 1/(1+MSE) [0,1]
  - 标签替换: 分类用多数投票, 回归用 KNN 均值
  - 最低保留率: 删除上限 80%, 防止全删

Reward 设计:
  - 评估步 (每 N 步, N 自适应): 训练下游模型, perf_diff - repair_lambda * (action==1)
  - 中间步: 启发式 reward shaping (自动缩放到 perf_diff 量级)
  - final: score_improvement * 5 + keep_rate * 0.2 - repair_cost
  - N 自适应: 总评估开销 ≈ 常数, config.reward_eval_interval=0 启用自动模式
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import math
import random
import numpy as np

from ...config import DemandPrepPolicyConfig, TaskType
from ...models import ModelAdapter
from ..state import StateExtractor
from .value_estimation import ValueEstimator


class CleaningEnv:
    """
    清洗环境

    状态空间 (8维):
    - error_type: 错误类型 (0=missing, 1=semantic, 2=syntactic, 3=label_noise)
    - feature_importance: 特征重要性
    - distance_to_boundary: 到决策边界/均值的距离
    - row_position: 行位置
    - col_index: 列索引 (-1 表示标签错误)
    - col_error_rate: 当前列的错误率
    - sample_retention: 样本保留率
    - var_retention: 方差保留率

    标签错误特殊处理:
    - repair_value=None 时，action=1(repair) 自动降级为 action=2(delete)
    - reward shaping 对标签错误倾向 delete（标签估计值不可靠）
    """

    def __init__(self,
                 X_dirty: np.ndarray,
                 y: np.ndarray,
                 error_list: List[Dict[str, Any]],
                 model_adapter: ModelAdapter,
                 state_extractor: StateExtractor,
                 config: DemandPrepPolicyConfig,
                 X_base: Optional[np.ndarray] = None,
                 y_base: Optional[np.ndarray] = None,
                 value_estimator: Optional['ValueEstimator'] = None,
                 shaping_weight: float = 1.0):
        """
        初始化清洗环境

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            error_list: 错误列表
                [{'idx': int, 'col': int, 'type': int, 'repair_value': float|None}, ...]
                type: 0=missing, 1=semantic, 2=syntactic, 3=label_noise
                col: -1 表示标签错误
                repair_value: 训练时是注入前的值(或 None 表示无真值)，推理时是真值
            model_adapter: 模型适配器
            state_extractor: 状态提取器
            config: 配置对象
            X_base: 验证集特征矩阵（训练时传入，推理时为 None）
            y_base: 验证集标签向量（训练时传入，推理时为 None）
        """
        self.X_dirty_original = X_dirty.copy()
        self.y_original = y.copy()
        self.error_list = error_list
        self.model_adapter = model_adapter
        self.state_extractor = state_extractor
        self.config = config

        self.repair_lambda = config.repair_lambda
        self.min_truth_budget = config.min_truth_budget   # deprecated
        self.max_truth_budget = config.max_truth_budget   # deprecated

        # 修复率边界（替代旧的 min/max_truth_budget）
        n_errors = len(error_list)
        self.min_repair_ratio = config.min_repair_ratio
        self.max_repair_ratio = config.max_repair_ratio
        self.max_repair_count = int(n_errors * config.max_repair_ratio) if config.max_repair_ratio < 1.0 else n_errors
        # 兼容旧 max_truth_budget
        if config.max_truth_budget is not None:
            self.max_repair_count = min(self.max_repair_count, config.max_truth_budget)

        # 验证集（训练时传入，推理时为 None）
        self.X_base = X_base
        self.y_base = y_base
        self._eval_sample_indices = self._build_eval_sample_indices()

        # reward 评估缓存
        self._cached_score = None
        self._steps_since_eval = 0

        # 自适应评估间隔（config=0 时自动计算，>0 时用手动值）
        self._reward_eval_interval = self._compute_eval_interval(
            len(error_list), len(X_dirty), config.reward_eval_interval
        )

        # 启发式 reward 自动缩放状态
        self._perf_diff_ema = 0.0              # |perf_diff| 指数移动平均
        self._perf_diff_ema_initialized = False
        self._shaping_scale = 1.0              # 缩放因子（首次评估前=1.0 不缩放）
        self._shaping_weight = shaping_weight  # 训练进度衰减系数 [0,1]，由 trainer 传入

        # 状态变量
        self.X_current: Optional[np.ndarray] = None
        self.y_current: Optional[np.ndarray] = None
        self.current_error_idx = 0
        self.deleted_rows: Set[int] = set()
        self.action_counts = {
            'no_action': 0,
            'repair_value': 0,
            'delete': 0,
            'replace_nearby': 0
        }

        # 增量缓存（避免重复计算 _fill_nan 和 keep_mask）
        self._keep_mask = np.ones(len(X_dirty), dtype=bool)
        self._X_filled_cache: Optional[np.ndarray] = None

        # 修复记录
        self.repair_log: List[Dict] = []

        # 完整决策日志（记录所有4种动作的详情）
        self.decision_log: List[Dict] = []

        # 预计算统计量
        self._precompute_stats()

        # 值估计器（FD + KNN + DOMAIN）— 支持跨 episode 共享
        if value_estimator is not None:
            self.value_estimator = value_estimator
        else:
            self.value_estimator = ValueEstimator(config)

        # 特征重要性刷新间隔
        if config.importance_refresh_interval is not None:
            self.importance_refresh_interval = config.importance_refresh_interval
        else:
            self.importance_refresh_interval = max(20, len(error_list) // 10)

        # 初始化状态提取器（同时计算 X_filled 缓存）
        X_filled = self._init_state_extractor()

        # 基线性能（复用 _init_state_extractor 中已 fit 的模型，避免双重 fit）
        self.baseline_score = self._compute_baseline_score(X_filled)

        # episode 结束时的最终 score 缓存（供 trainer 复用，避免双重 evaluate）
        self.last_episode_score: Optional[float] = None

    def _build_eval_sample_indices(self) -> Optional[np.ndarray]:
        """构建验证集采样索引"""
        if self.X_base is None:
            return None
        n = len(self.X_base)
        ratio = getattr(self.config, 'eval_sample_ratio', 1.0)
        if ratio >= 1.0:
            return None  # 不采样，用全量
        k = max(10, int(n * ratio))
        return np.random.choice(n, k, replace=False)

    @staticmethod
    def _compute_eval_interval(n_errors: int, n_samples: int,
                                config_value: int) -> int:
        """计算自适应评估间隔

        config_value > 0: 用户手动指定，直接返回
        config_value == 0: 根据数据规模自动计算

        核心公式: 总评估开销 ∝ (n_errors / N) * n_samples ≈ 常数预算
        """
        if config_value > 0:
            return config_value
        if n_errors == 0:
            return 10

        BUDGET = 500_000       # 参考预算（beers级别开销）
        MIN_EVALS = 20         # 每 episode 最少评估次数
        MAX_INTERVAL = 200     # 间隔上限

        cost_based_n = max(1, int(n_errors * n_samples / BUDGET))
        max_n_for_min_evals = max(1, n_errors // MIN_EVALS)
        return max(5, min(cost_based_n, max_n_for_min_evals, MAX_INTERVAL))

    def _precompute_stats(self) -> None:
        """预计算统计量"""
        self.col_means = np.nanmean(self.X_dirty_original, axis=0)
        self.col_vars = np.nanvar(self.X_dirty_original, axis=0)
        self.col_stds = np.sqrt(self.col_vars)

        # 每列的有效值
        self.all_values = {}
        for col in range(self.X_dirty_original.shape[1]):
            valid = self.X_dirty_original[:, col][~np.isnan(self.X_dirty_original[:, col])]
            self.all_values[col] = valid

        # 计算每列错误率（col=-1 的标签错误不计入特征列）
        n_cols = self.X_dirty_original.shape[1]
        col_error_counts = np.zeros(n_cols)
        label_error_count = 0
        for error in self.error_list:
            col = error['col']
            if col == -1:
                label_error_count += 1
            elif col < n_cols:
                col_error_counts[col] += 1

        total_errors = len(self.error_list)
        if total_errors > 0:
            self.col_error_rate = col_error_counts / total_errors
        else:
            self.col_error_rate = np.zeros(n_cols)
        self.label_error_rate = label_error_count / max(total_errors, 1)

    def _init_state_extractor(self) -> np.ndarray:
        """初始化状态提取器，返回 X_filled 供后续复用

        使用去除检测错误行的干净子集 fit 模型，确保 distance_to_boundary
        在训练和推理时分布一致（都基于较干净的数据计算决策边界）。
        """
        # 填充 NaN 用于训练模型
        X_filled = self._fill_nan(self.X_dirty_original.copy())
        self._X_filled_cache = X_filled.copy()

        # 用干净子集（去除检测错误行）fit 模型，使状态分布跨训练/推理一致
        error_rows = set(e['idx'] for e in self.error_list)
        clean_mask = np.array([i not in error_rows for i in range(len(X_filled))])
        try:
            if clean_mask.sum() >= 20:
                self.model_adapter.fit(X_filled[clean_mask], self.y_original[clean_mask])
            else:
                self.model_adapter.fit(X_filled, self.y_original)
        except Exception:
            try:
                self.model_adapter.fit(X_filled, self.y_original)
            except Exception:
                pass

        # 计算特征重要性
        try:
            feature_importance = self.model_adapter.get_feature_importance()
        except Exception:
            feature_importance = np.ones(self.X_dirty_original.shape[1]) / self.X_dirty_original.shape[1]

        # 设置状态提取器
        self.state_extractor.set_model_adapter(self.model_adapter)
        self.state_extractor.set_feature_importance(feature_importance)
        self.state_extractor.set_col_error_rate(self.col_error_rate)
        self.state_extractor.set_col_stats(self.col_means, self.col_stds, self.col_vars)
        # 设置样本数，确保 compute_retention() 能正确计算 sample_retention
        self.state_extractor._n_samples = len(X_filled)

        return X_filled

    def _fill_nan(self, X: np.ndarray) -> np.ndarray:
        """填充 NaN 值"""
        X_filled = X.copy()
        for col in range(X_filled.shape[1]):
            col_mean = np.nanmean(X_filled[:, col])
            nan_mask = np.isnan(X_filled[:, col])
            if nan_mask.any():
                X_filled[nan_mask, col] = col_mean if not np.isnan(col_mean) else 0
        return X_filled

    def _normalize_raw_score(self, raw_score: float) -> float:
        """将模型原始评分归一化到 [0, 1]

        - 分类: accuracy 已在 [0, 1]
        - 回归: -MSE → 1/(1+log(1+MSE)) ∈ (0, 1]
          使用 log 压缩高 MSE 值，避免归一化后 perf_diff 信号过弱
        - 聚类: silhouette ∈ [-1, 1] → (s+1)/2 ∈ [0, 1]
        """
        if self.config.task_type == TaskType.REGRESSION:
            mse = abs(raw_score)
            if self.config.regression_log_normalize:
                return 1.0 / (1.0 + math.log(1.0 + mse))
            else:
                return 1.0 / (1.0 + mse)
        elif self.config.task_type == TaskType.CLUSTERING:
            return (raw_score + 1.0) / 2.0
        else:
            return raw_score

    def _compute_baseline_score(self, X_filled: np.ndarray) -> float:
        """复用 _init_state_extractor 中已 fit 的模型计算基线分数"""
        try:
            if self.X_base is not None and self.y_base is not None:
                if self._eval_sample_indices is not None:
                    X_eval = self.X_base[self._eval_sample_indices]
                    y_eval = self.y_base[self._eval_sample_indices]
                else:
                    X_eval = self.X_base
                    y_eval = self.y_base
                raw_score = self.model_adapter.evaluate(X_eval, y_eval)
            else:
                raw_score = self.model_adapter.evaluate(X_filled, self.y_original)
            return self._normalize_raw_score(raw_score)
        except Exception:
            return 0.0

    def _update_filled_cache(self, idx: int, col: int, value: float) -> None:
        """增量更新 _X_filled_cache 的单个元素"""
        if self._X_filled_cache is not None:
            if np.isnan(value):
                self._X_filled_cache[idx, col] = self.col_means[col]
            else:
                self._X_filled_cache[idx, col] = value

    def _evaluate_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估当前数据的模型性能

        有 X_base 时: fit(X_train) → evaluate(X_base_sample, y_base_sample)
        无 X_base 时: 保持原逻辑（向后兼容，推理时不用 reward）

        使用增量缓存 _X_filled_cache 和 _keep_mask 避免重复计算。
        """
        # 使用缓存（训练循环中始终可用）；fallback 到全量计算
        if self._X_filled_cache is not None:
            X_filled = self._X_filled_cache
        else:
            X_filled = self._fill_nan(X.copy())

        keep_mask = self._keep_mask
        if keep_mask.sum() < 10:
            return 0.0

        X_train = X_filled[keep_mask]
        y_train = y[keep_mask]

        try:
            self.model_adapter.fit(X_train, y_train)

            # 有验证集时用验证集评估
            if self.X_base is not None and self.y_base is not None:
                if self._eval_sample_indices is not None:
                    X_eval = self.X_base[self._eval_sample_indices]
                    y_eval = self.y_base[self._eval_sample_indices]
                else:
                    X_eval = self.X_base
                    y_eval = self.y_base
                raw_score = self.model_adapter.evaluate(X_eval, y_eval)
            else:
                # 向后兼容：无验证集时在全量数据上评估
                raw_score = self.model_adapter.evaluate(X_filled, y)

            score = self._normalize_raw_score(raw_score)

            # 顺便更新 feature importance（零额外 fit 成本，复用已 fit 的模型）
            try:
                new_imp = self.model_adapter.get_feature_importance()
                if new_imp is not None:
                    self.state_extractor.feature_importance = new_imp
            except Exception:
                pass

            return score
        except Exception:
            return 0.0

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.X_current = self.X_dirty_original.copy()
        self.y_current = self.y_original.copy()
        self.current_error_idx = 0
        self.deleted_rows = set()
        self.action_counts = {k: 0 for k in self.action_counts}
        self.repair_log = []
        self.decision_log = []
        self._cached_score = None
        self._steps_since_eval = 0
        # 重置增量缓存
        self._X_filled_cache = self._fill_nan(self.X_current.copy())
        self._keep_mask = np.ones(len(self.X_dirty_original), dtype=bool)
        self.last_episode_score = None
        random.shuffle(self.error_list)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态（8维错误级特征 + 2维全局特征）"""
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(self.config.state_size, dtype=np.float32)

        error = self.error_list[self.current_error_idx]
        base_state = self.state_extractor.extract(
            self.X_current,
            self.y_current,
            error,
            self.deleted_rows
        )

        # 追加全局特征：让 DQN 感知"还剩多少预算/还有多少错误待处理"
        # [8] remaining_budget_ratio: 剩余可用真值预算比例 [0,1]
        repair_used = self.action_counts['repair_value']
        remaining_budget = max(0, self.max_repair_count - repair_used)
        remaining_budget_ratio = remaining_budget / max(self.max_repair_count, 1)

        # [9] remaining_errors_ratio: 待处理错误占总数的比例 [0,1]
        total_errors = max(len(self.error_list), 1)
        remaining_errors = max(0, total_errors - self.current_error_idx)
        remaining_errors_ratio = remaining_errors / total_errors

        return np.concatenate([
            base_state,
            np.array([remaining_budget_ratio, remaining_errors_ratio], dtype=np.float32)
        ])

    def _get_nearby_value(self, idx: int, col: int) -> float:
        """获取临近值（多维 KNN + 规则优先）"""
        return self.value_estimator.estimate_feature_value(
            self.X_current, idx, col,
            self.deleted_rows, self.col_means
        )

    def _get_majority_label(self, idx: int, k: int = 5) -> float:
        """
        获取最近邻的标签估计

        分类任务: 多数投票 (majority vote)
        回归任务: KNN 均值 (weighted mean)

        Args:
            idx: 目标行索引
            k: 邻居数量

        Returns:
            估计的标签值
        """
        X_filled = self._X_filled_cache if self._X_filled_cache is not None else self._fill_nan(self.X_current.copy())
        target = X_filled[idx]

        # 计算距离
        distances = np.linalg.norm(X_filled - target, axis=1)
        distances[idx] = np.inf  # 排除自身

        # 排除已删除的行
        for d_idx in self.deleted_rows:
            distances[d_idx] = np.inf

        # 取前k个最近邻
        k = min(k, (distances < np.inf).sum())
        if k == 0:
            return self.y_current[idx]

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = self.y_current[nearest_indices]

        valid_labels = nearest_labels[~np.isnan(nearest_labels)]
        if len(valid_labels) == 0:
            return self.y_current[idx]

        # 回归: KNN 加权均值 (距离倒数权重)
        if self.config.task_type == TaskType.REGRESSION:
            nearest_dists = distances[nearest_indices]
            valid_mask = ~np.isnan(nearest_labels)
            valid_dists = nearest_dists[valid_mask]
            # 距离倒数权重, 防止除零
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid_labels, weights=weights))

        # 分类: 多数投票
        unique, counts = np.unique(valid_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        Args:
            action: 动作索引
                0: no_action
                1: repair_value
                2: delete
                3: replace_nearby

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(self.config.state_size, dtype=np.float32), 0, True, {}

        error = self.error_list[self.current_error_idx]
        idx, col, error_type = error['idx'], error['col'], error['type']
        current_state = self._get_state()  # 保存 state 用于 shaping reward
        repair_value = error['repair_value']

        # 跳过已删除的行
        if idx in self.deleted_rows:
            self.current_error_idx += 1
            return self._get_state(), 0, self.current_error_idx >= len(self.error_list), {}

        # 记录原始动作（降级前）
        original_action = action

        # 真值预算检查（统一使用 max_repair_count）
        # 超预算降级为 replace_nearby（KNN 替换本身效果好，避免丢数据）
        if action == 1 and self.action_counts['repair_value'] >= self.max_repair_count:
            action = 3  # replace_nearby

        # 判断是否为标签噪声错误 (col == -1)
        is_label_error = (col == -1)

        # 标签错误 + repair_value=None 时，自动降级为 delete
        # （自监督模式下，标签错误的真值不可靠，强制删除）
        if action == 1 and is_label_error and repair_value is None:
            action = 2  # 降级为 delete

        # 记录脏值
        if is_label_error:
            dirty_value = self.y_current[idx]
        else:
            dirty_value = self.X_current[idx, col]
        dirty_value_safe = dirty_value if not (isinstance(dirty_value, float) and np.isnan(dirty_value)) else None

        # 执行动作
        result_value = None
        if action == 0:
            self.action_counts['no_action'] += 1
        elif action == 1:
            if is_label_error:
                self.repair_log.append({
                    'idx': idx,
                    'col': -1,
                    'dirty_value': dirty_value_safe,
                    'clean_value': repair_value,
                    'error_type': error_type
                })
                self.y_current[idx] = repair_value
            else:
                self.repair_log.append({
                    'idx': idx,
                    'col': col,
                    'dirty_value': dirty_value_safe,
                    'clean_value': repair_value,
                    'error_type': error_type
                })
                self.X_current[idx, col] = repair_value
                self._update_filled_cache(idx, col, repair_value)
            self.action_counts['repair_value'] += 1
            result_value = repair_value
        elif action == 2:
            # 保护策略: 至少保留 20% 数据，超过上限则强制转为 no_action
            n_total = len(self.X_current)
            max_deletions = int(n_total * 0.8)
            if len(self.deleted_rows) >= max_deletions:
                # 已达到删除上限，退化为 no_action
                action = 0
                self.action_counts['no_action'] += 1
            else:
                self.deleted_rows.add(idx)
                self._keep_mask[idx] = False
                self.action_counts['delete'] += 1
        elif action == 3:
            if is_label_error:
                # 标签噪声：临近替换用多数投票
                nearby_label = self._get_majority_label(idx)
                self.y_current[idx] = nearby_label
                result_value = nearby_label
            else:
                nearby_val = self._get_nearby_value(idx, col)
                self.X_current[idx, col] = nearby_val
                self._update_filled_cache(idx, col, nearby_val)
                result_value = nearby_val
            self.action_counts['replace_nearby'] += 1

        # 记录完整决策日志
        self.decision_log.append({
            'error_idx': self.current_error_idx,
            'row_idx': idx,
            'col': col,
            'error_type': error_type,
            'action': action,
            'original_action': original_action,
            'dirty_value': dirty_value_safe,
            'result_value': result_value,
        })

        self.current_error_idx += 1
        done = self.current_error_idx >= len(self.error_list)

        # 计算奖励
        self._steps_since_eval += 1

        if done:
            # episode 结束：强制评估
            reward = self._calculate_final_reward()
            info = {'stage1_reward': reward, 'stage2_reward': reward}
        else:
            # Shaping bonus（始终计算，按 _shaping_weight 衰减）
            shaping_bonus = 0.0
            if self._shaping_weight > 0.01:
                shaping_bonus = self._get_shaping_reward(
                    action, error_type, current_state)

            if self._steps_since_eval >= self._reward_eval_interval or self._cached_score is None:
                # 评估步: 训练下游模型获取真实性能信号 + shaping bonus
                current_score = self._evaluate_score(self.X_current, self.y_current)
                self._cached_score = current_score
                self._steps_since_eval = 0

                perf_diff = current_score - self.baseline_score
                self._update_perf_diff_ema(perf_diff)

                repair_cost = self.repair_lambda if action == 1 else 0.0
                eval_reward = perf_diff - repair_cost

                # 融合: eval reward（主信号）+ shaping bonus（方向性引导）
                reward = eval_reward + shaping_bonus
                stage1_reward = reward
                stage2_reward = reward if action in (1, 3) else None
            else:
                # 中间步: 仅 shaping（不训练下游模型）
                reward = shaping_bonus
                stage1_reward = reward
                stage2_reward = reward if action in (1, 3) else None

            info = {'stage1_reward': stage1_reward, 'stage2_reward': stage2_reward}

        return self._get_state(), reward, done, info

    def _update_perf_diff_ema(self, perf_diff: float) -> None:
        """更新 perf_diff EMA 并重算 shaping 缩放因子"""
        abs_diff = abs(perf_diff)
        if not self._perf_diff_ema_initialized:
            self._perf_diff_ema = abs_diff
            self._perf_diff_ema_initialized = True
        else:
            self._perf_diff_ema = (
                0.3 * abs_diff + 0.7 * self._perf_diff_ema
            )
        SHAPING_REFERENCE = 0.03  # 启发式 reward 的典型绝对值
        if self._perf_diff_ema > 1e-6:
            self._shaping_scale = max(0.1, min(
                self._perf_diff_ema / SHAPING_REFERENCE, 10.0
            ))

    def _get_shaping_reward(self, action: int, error_type: int,
                            state: np.ndarray) -> float:
        """State-aware reward shaping（优先级引导）

        用连续 state 信号引导 DQN 学习优先级，不硬编码错误类型规则：
        - 高优先级（边界近 + 重要特征 + 高错误率）→ repair > replace
        - 低优先级 → replace > repair（别浪费预算）
        - 预算耗尽 → repair 自然不如 replace

        State 维度:
          [0] error_type, [1] feature_importance, [2] distance_to_boundary,
          [3] row_position, [4] col_index, [5] col_error_rate,
          [6] sample_retention, [7] var_retention,
          [8] remaining_budget_ratio, [9] remaining_errors_ratio
        """
        s = self._shaping_scale
        urgency = 1.0 - state[2]       # 边界紧迫度 [0,1]
        importance = state[1]           # 特征重要性 [0,1]
        error_rate = state[5]           # 列错误率 [0,1]

        # 综合优先级: 边界近 + 重要特征 + 高错误率 → 更值得用真值修复
        priority = (urgency + importance + error_rate) / 3.0

        # 预算感知: 直接用 state[8] (remaining_budget_ratio)
        # 预算充足时 repair 有吸引力；预算快用完时 repair 吸引力下降
        remaining_budget_ratio = float(state[8])
        remaining_errors_ratio = float(state[9])
        # 相对宽裕度: remaining_budget > remaining_errors → 可以大胆 repair
        budget_slack = remaining_budget_ratio - remaining_errors_ratio  # [-1, 1]
        # sigmoid: 宽裕时接近1，紧张时接近0
        budget_ok = 1.0 / (1.0 + math.exp(-6.0 * budget_slack))

        if action == 0:    # no_action: 不处理已知错误 → 惩罚
            reward = -0.03
        elif action == 1:  # repair: 高优先级时远超 replace，低优先级时不如 replace
            # 高priority: (0.02+0.04)=0.06  低priority: (0.02+0)=0.02
            # 对比 replace 固定 0.03 → 只有 priority>0.25 时 repair 才优于 replace
            reward = (0.02 + 0.04 * priority) * budget_ok
        elif action == 2:  # delete: 丢数据 → 惩罚（回归任务用更大惩罚避免过度删除）
            reward = self.config.delete_shaping_reward
        elif action == 3:  # replace: 免费且效果好 → 稳定正奖
            reward = 0.03
        else:
            reward = 0.0

        return reward * s * self._shaping_weight

    def _calculate_final_reward(self) -> float:
        """计算最终奖励（简化版: 性能提升 - 修复成本 + 保留率微奖）"""
        final_score = self._evaluate_score(self.X_current, self.y_current)
        self.last_episode_score = final_score  # 缓存供 trainer 复用
        score_improvement = final_score - self.baseline_score

        keep_rate = 1 - len(self.deleted_rows) / len(self.X_current)
        repair_cost = self.action_counts['repair_value'] * self.repair_lambda

        # 核心: 性能提升（主信号）+ 保留率奖励 - 修复成本
        reward = score_improvement * 5 + keep_rate * self.config.keep_rate_weight - repair_cost
        return reward

    def get_cleaned_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取清洗后的数据

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        X_result = self.X_current[keep_mask].copy()
        y_result = self.y_current[keep_mask]

        # 填充剩余 NaN (用 ValueEstimator 逐单元格精确估值)
        if np.isnan(X_result).any():
            col_means = np.nanmean(X_result, axis=0)
            # keep_mask 的原始行号映射
            original_indices = np.where(keep_mask)[0]
            for col in range(X_result.shape[1]):
                nan_mask = np.isnan(X_result[:, col])
                if nan_mask.any():
                    for i in np.where(nan_mask)[0]:
                        X_result[i, col] = self.value_estimator.estimate_feature_value(
                            X_result, i, col, set(), col_means,
                            dirty_df_row_indices=original_indices,
                        )

        return X_result, y_result, keep_mask

    def get_repair_log(self) -> List[Dict]:
        """获取修复记录"""
        return self.repair_log

    def get_action_counts(self) -> Dict[str, int]:
        """获取动作统计"""
        return self.action_counts.copy()

    def print_repair_log(self, max_rows: int = 20) -> None:
        """打印修复记录"""
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        print(f"\n{'='*70}")
        print(f"真值使用记录 (共 {len(self.repair_log)} 条)")
        print(f"{'='*70}")

        if len(self.repair_log) == 0:
            print("  (未使用任何真值)")
            return

        print(f"{'索引':<8} {'列':<6} {'脏数据':<15} {'干净数据':<15} {'错误类型':<10}")
        print("-" * 70)

        display_log = self.repair_log[:max_rows] if max_rows else self.repair_log

        for record in display_log:
            dirty_str = f"{record['dirty_value']:.4f}" if record['dirty_value'] is not None else "NaN"
            clean_str = f"{record['clean_value']:.4f}"
            error_type_str = error_type_names.get(record['error_type'], 'unknown')
            print(f"{record['idx']:<8} {record['col']:<6} {dirty_str:<15} {clean_str:<15} {error_type_str:<10}")

        if max_rows and len(self.repair_log) > max_rows:
            print(f"... 省略 {len(self.repair_log) - max_rows} 条 ...")

        # 统计
        print("-" * 70)
        by_type = {}
        for r in self.repair_log:
            t = error_type_names.get(r['error_type'], 'unknown')
            by_type[t] = by_type.get(t, 0) + 1

        print(f"统计: ", end="")
        print(", ".join([f"{k}={v}" for k, v in by_type.items()]))
        print(f"{'='*70}\n")

    def get_decision_log(self) -> List[Dict]:
        """获取完整决策日志（所有4种动作的详情）"""
        return self.decision_log

    def print_decision_summary(self, max_rows: int = 30) -> None:
        """
        打印分类汇总的决策日志

        按动作类型分组显示：repair → replace → delete → no_action
        """
        ACTION_NAMES = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
        ERROR_TYPE_NAMES = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        total = len(self.decision_log)
        if total == 0:
            print("  (无决策记录)")
            return

        # 动作分布
        print(f"\n  动作分布 (共 {total} 个错误):")
        for act_id, act_name in ACTION_NAMES.items():
            count = self.action_counts.get(act_name, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2.5)
            print(f"    {act_name:<16} {count:>5} ({pct:5.1f}%) {bar}")

        # 降级统计
        degraded = [d for d in self.decision_log if d['action'] != d['original_action']]
        if degraded:
            print(f"\n  动作降级: {len(degraded)} 次")
            for d in degraded[:5]:
                print(f"    行{d['row_idx']}: {ACTION_NAMES[d['original_action']]} → {ACTION_NAMES[d['action']]}")
            if len(degraded) > 5:
                print(f"    ... 共 {len(degraded)} 次降级")

        # 分组显示明细
        repairs = [d for d in self.decision_log if d['action'] == 1]
        replaces = [d for d in self.decision_log if d['action'] == 3]
        deletes = [d for d in self.decision_log if d['action'] == 2]

        def _fmt_val(v):
            if v is None:
                return 'NaN'
            return f'{v:.4f}'

        # 修复明细
        if repairs:
            n_show = min(max_rows, len(repairs))
            print(f"\n  修复明细 (repair_value): 共 {len(repairs)} 条")
            print(f"    {'行':<8} {'列':<6} {'脏值':<12} → {'修复值':<12} {'错误类型':<10}")
            print(f"    {'-'*55}")
            for d in repairs[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} → "
                      f"{_fmt_val(d['result_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(repairs) > n_show:
                print(f"    ... 省略 {len(repairs) - n_show} 条")

        # 替换明细
        if replaces:
            n_show = min(max_rows, len(replaces))
            print(f"\n  替换明细 (replace_nearby): 共 {len(replaces)} 条")
            print(f"    {'行':<8} {'列':<6} {'脏值':<12} → {'替换值':<12} {'错误类型':<10}")
            print(f"    {'-'*55}")
            for d in replaces[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} → "
                      f"{_fmt_val(d['result_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(replaces) > n_show:
                print(f"    ... 省略 {len(replaces) - n_show} 条")

        # 删除明细
        if deletes:
            n_show = min(max_rows, len(deletes))
            print(f"\n  删除明细 (delete): 共 {len(deletes)} 条")
            print(f"    {'行':<8} {'列':<6} {'脏值':<12} {'错误类型':<10}")
            print(f"    {'-'*40}")
            for d in deletes[:n_show]:
                print(f"    {d['row_idx']:<8} {d['col']:<6} "
                      f"{_fmt_val(d['dirty_value']):<12} "
                      f"{ERROR_TYPE_NAMES.get(d['error_type'], '?'):<10}")
            if len(deletes) > n_show:
                print(f"    ... 省略 {len(deletes) - n_show} 条")

    def _refresh_feature_importance(self) -> None:
        """周期性刷新特征重要性

        每处理 importance_refresh_interval 个错误后，
        用当前清洗数据重训模型并更新 feature_importance。
        """
        X_filled = self._X_filled_cache if self._X_filled_cache is not None else self._fill_nan(self.X_current.copy())
        keep_mask = self._keep_mask
        if keep_mask.sum() < 10:
            return
        try:
            self.model_adapter.fit(X_filled[keep_mask], self.y_current[keep_mask])
            new_importance = self.model_adapter.get_feature_importance()
            if new_importance is not None:
                self.state_extractor.feature_importance = new_importance
        except Exception:
            pass
