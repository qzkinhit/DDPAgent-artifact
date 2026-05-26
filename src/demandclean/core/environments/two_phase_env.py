"""
两阶段推理环境
==============

用于不提前知道真值的推理场景。

第一阶段：预测所有动作，repair_value 只加入 plan，其他动作立即执行
第二阶段：用户提供真值后，执行 plan 中的修复

保护策略:
  - 最低保留率: 删除上限 80%, 防止全删
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from ...config import DemandCleanConfig, TaskType
from ...models import ModelAdapter
from ..state import StateExtractor
from .value_estimation import ValueEstimator


class TwoPhaseCleaningEnv:
    """
    两阶段推理环境

    状态计算策略（与训练环境完全一致）：
    - 10 维状态 = 8 维错误级特征 + 2 维全局感知
    - 错误级特征委托 state_extractor.extract() 计算
    - 全局感知：[8] remaining_budget_ratio, [9] remaining_errors_ratio
    - action==1 时用 ValueEstimator 估计值写入 X_current，使状态计算和训练一致
    - action==3 时用 ValueEstimator 估计值（FD → KNN → DOMAIN）
    - 周期性刷新 feature_importance（与训练一致）
    """

    def __init__(self,
                 X_dirty: np.ndarray,
                 y: np.ndarray,
                 error_list: List[Dict[str, Any]],
                 model_adapter: ModelAdapter,
                 state_extractor: StateExtractor,
                 config: DemandCleanConfig):
        """
        初始化两阶段推理环境

        Args:
            X_dirty: 脏数据（不需要 X_clean）
            y: 标签
            error_list: 检测到的错误列表，此时 repair_value 可以是估计值或 None
            model_adapter: 模型适配器
            state_extractor: 状态提取器
            config: 配置对象
        """
        self.X_dirty_original = X_dirty.copy()
        self.y_original = y.copy()
        self.error_list = error_list
        self.model_adapter = model_adapter
        self.state_extractor = state_extractor
        self.config = config
        self.repair_lambda = config.repair_lambda

        # 状态变量
        self.X_current: Optional[np.ndarray] = None
        self.y_current: Optional[np.ndarray] = None
        self.current_error_idx = 0
        self.deleted_rows: Set[int] = set()

        # 动作计数
        self.action_counts = {
            'no_action': 0,
            'repair_value': 0,
            'delete': 0,
            'replace_nearby': 0
        }

        # 两阶段核心：修复计划
        self.repair_plan: List[Dict] = []
        self.planned_repairs: Set[Tuple[int, int]] = set()

        # 最大修复预算（用于全局感知 state）
        # 与 CleaningEnv 保持一致：基于 max_repair_ratio，兼容旧 max_truth_budget
        n_errors = len(error_list) if error_list else 1
        ratio_budget = int(n_errors * config.max_repair_ratio) if config.max_repair_ratio < 1.0 else n_errors
        if config.max_truth_budget is not None:
            ratio_budget = min(config.max_truth_budget, ratio_budget)
        self.max_repair_count = ratio_budget

        # 完整决策日志（记录所有4种动作的详情）
        self.decision_log: List[Dict] = []

        # 预计算
        self._precompute_stats()

        # 值估计器（FD + KNN + DOMAIN）
        self.value_estimator = ValueEstimator(config)

        # 特征重要性刷新间隔
        if config.importance_refresh_interval is not None:
            self.importance_refresh_interval = config.importance_refresh_interval
        else:
            self.importance_refresh_interval = max(20, len(error_list) // 10)

        self._init_state_extractor()

    def _precompute_stats(self) -> None:
        """预计算统计量"""
        n_cols = self.X_dirty_original.shape[1]

        self.col_means = np.nanmean(self.X_dirty_original, axis=0)
        self.col_stds = np.nanstd(self.X_dirty_original, axis=0)
        self.col_vars = np.nanvar(self.X_dirty_original, axis=0)

        # 处理 NaN
        for col in range(n_cols):
            if np.isnan(self.col_means[col]):
                self.col_means[col] = 0
            if np.isnan(self.col_stds[col]) or self.col_stds[col] == 0:
                self.col_stds[col] = 1
            if np.isnan(self.col_vars[col]) or self.col_vars[col] == 0:
                self.col_vars[col] = 1

        # 计算每列错误数（排除标签错误 col=-1）
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
            self.col_error_rates = col_error_counts / total_errors
            self.label_error_rate = label_error_count / total_errors
        else:
            self.col_error_rates = np.zeros(n_cols)
            self.label_error_rate = 0.0

        # 跟踪当前每列的剩余错误数
        self.col_remaining_errors = col_error_counts.copy()
        self.total_remaining_errors = total_errors

    def _init_state_extractor(self) -> None:
        """初始化状态提取器"""
        X_filled = self._fill_nan(self.X_dirty_original.copy())

        try:
            self.model_adapter.fit(X_filled, self.y_original)
        except Exception:
            pass

        try:
            feature_importance = self.model_adapter.get_feature_importance()
        except Exception:
            feature_importance = np.ones(self.X_dirty_original.shape[1]) / self.X_dirty_original.shape[1]

        self.state_extractor.set_model_adapter(self.model_adapter)
        self.state_extractor.set_feature_importance(feature_importance)
        self.state_extractor.set_col_error_rate(self.col_error_rates)
        self.state_extractor.set_col_stats(self.col_means, self.col_stds, self.col_vars)
        # 设置样本数，确保 compute_retention() 能正确计算 sample_retention
        self.state_extractor._n_samples = len(X_filled)

    def _fill_nan(self, X: np.ndarray) -> np.ndarray:
        """填充 NaN 值"""
        X_filled = X.copy()
        for col in range(X_filled.shape[1]):
            col_mean = np.nanmean(X_filled[:, col])
            nan_mask = np.isnan(X_filled[:, col])
            if nan_mask.any():
                X_filled[nan_mask, col] = col_mean if not np.isnan(col_mean) else 0
        return X_filled

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.X_current = self.X_dirty_original.copy()
        self.y_current = self.y_original.copy()
        self.current_error_idx = 0
        self.deleted_rows = set()
        self.action_counts = {k: 0 for k in self.action_counts}
        self.repair_plan = []
        self.planned_repairs = set()
        self.decision_log = []

        # 重置错误跟踪
        self._precompute_stats()

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态（10维 = 8维错误级特征 + 2维全局感知）"""
        if self.current_error_idx >= len(self.error_list):
            return np.zeros(self.config.state_size, dtype=np.float32)

        error = self.error_list[self.current_error_idx]
        # 8维错误级特征
        base_state = self.state_extractor.extract(
            self.X_current,
            self.y_current,
            error,
            self.deleted_rows
        )

        # [8] remaining_budget_ratio: 剩余可用真值预算比例 [0,1]
        repair_used = self.action_counts['repair_value']
        remaining_budget = max(0, self.max_repair_count - repair_used)
        remaining_budget_ratio = remaining_budget / max(self.max_repair_count, 1)

        # [9] remaining_errors_ratio: 待处理错误占总数的比例 [0,1]
        total_errors = max(len(self.error_list), 1)
        remaining_errors = max(0, total_errors - self.current_error_idx)
        remaining_errors_ratio = remaining_errors / total_errors

        # 拼接 10 维 state
        return np.concatenate([
            base_state,
            np.array([remaining_budget_ratio, remaining_errors_ratio], dtype=np.float32)
        ])

    def _get_majority_label(self, idx: int, k: int = 5) -> float:
        """
        获取最近邻的标签估计

        分类任务: 多数投票 (majority vote)
        回归任务: KNN 加权均值 (weighted mean)

        与 CleaningEnv._get_majority_label() 保持一致

        Args:
            idx: 目标行索引
            k: 邻居数量

        Returns:
            估计的标签值
        """
        X_filled = self._fill_nan(self.X_current.copy())
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
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid_labels, weights=weights))

        # 分类: 多数投票
        unique, counts = np.unique(valid_labels, return_counts=True)
        return unique[np.argmax(counts)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作

        对于 repair_value (action=1)：
        - 不实际修复，而是加入 repair_plan
        - 用估计值更新状态

        Args:
            action: 动作索引

        Returns:
            (next_state, reward, done, info)
        """
        if self.current_error_idx >= len(self.error_list):
            return self._get_state(), 0, True, {}

        error = self.error_list[self.current_error_idx]
        idx, col = error['idx'], error['col']
        error_type = error['type']
        is_label_error = (col == -1)

        # 记录原始动作（降级前）
        original_action = action

        # 记录脏值
        if is_label_error:
            dirty_value = self.y_current[idx]
        else:
            dirty_value = self.X_current[idx, col]
        dirty_value_safe = dirty_value if not (isinstance(dirty_value, float) and np.isnan(dirty_value)) else None

        result_value = None

        if action == 0:  # no_action
            self.action_counts['no_action'] += 1

        elif action == 1:  # repair_value -> 加入计划，用估计值写入 X_current
            self.action_counts['repair_value'] += 1

            if is_label_error:
                # 标签错误: KNN 多数投票估计
                estimated_value = self._get_majority_label(idx)
            else:
                # 特征错误: FD → 多维 KNN → DOMAIN 裁剪
                estimated_value = self.value_estimator.estimate_feature_value(
                    self.X_current, idx, col,
                    self.deleted_rows, self.col_means
                )

            # 加入修复计划
            self.repair_plan.append({
                'idx': idx,
                'col': col,
                'error_type': error_type,
                'estimated_value': estimated_value,
                'current_dirty_value': dirty_value_safe
            })

            # 标记为已计划修复
            self.planned_repairs.add((idx, col))

            # 用估计值更新当前数据
            if is_label_error:
                self.y_current[idx] = estimated_value
            else:
                self.X_current[idx, col] = estimated_value

            result_value = estimated_value

            # 更新错误计数
            if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
            self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

        elif action == 2:  # delete - 真实执行
            # 保护策略: 至少保留 20% 数据，超过上限则强制转为 no_action
            n_total = len(self.X_current)
            max_deletions = int(n_total * 0.8)
            if len(self.deleted_rows) >= max_deletions:
                # 已达到删除上限，退化为 no_action
                action = 0
                self.action_counts['no_action'] += 1
            else:
                self.action_counts['delete'] += 1
                self.deleted_rows.add(idx)

                # 更新错误计数（仅实际删除时才扣减）
                if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                    self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
                self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

        elif action == 3:  # replace_nearby - 真实执行
            self.action_counts['replace_nearby'] += 1

            if is_label_error:
                # 标签错误: 用多数投票/KNN替换
                nearby_val = self._get_majority_label(idx)
                self.y_current[idx] = nearby_val
                result_value = nearby_val
            else:
                # 特征错误: FD → 多维 KNN → DOMAIN 裁剪
                nearby_val = self.value_estimator.estimate_feature_value(
                    self.X_current, idx, col,
                    self.deleted_rows, self.col_means
                )
                self.X_current[idx, col] = nearby_val
                result_value = nearby_val

            # 更新错误计数
            if not is_label_error and 0 <= col < len(self.col_remaining_errors):
                self.col_remaining_errors[col] = max(0, self.col_remaining_errors[col] - 1)
            self.total_remaining_errors = max(0, self.total_remaining_errors - 1)

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

        # 周期性刷新 feature_importance
        if (self.current_error_idx % self.importance_refresh_interval == 0
                and not done):
            self._refresh_feature_importance()

        return self._get_state(), 0, done, {}

    def get_repair_plan(self) -> List[Dict]:
        """获取修复计划（第一阶段结果）"""
        return self.repair_plan

    def get_current_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取当前数据状态（未过滤，包含估计值）

        Returns:
            (X_current, y_current, keep_mask) - 未过滤的数据和掩码
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        return self.X_current.copy(), self.y_current.copy(), keep_mask

    def get_cleaned_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取清洗后的数据（已过滤、已填充 NaN）

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(self.X_current))])
        X_result = self.X_current[keep_mask].copy()
        y_result = self.y_current[keep_mask]

        # 填充剩余 NaN（no_action 选择的缺失值，用 ValueEstimator 逐单元格精确估值）
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

    def get_action_counts(self) -> Dict[str, int]:
        """获取动作统计"""
        return self.action_counts.copy()

    def execute_repair_plan(self,
                            X_dirty: np.ndarray,
                            true_values: Dict[Tuple[int, int], float],
                            y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行修复计划（第二阶段）

        Args:
            X_dirty: 原始脏数据
            true_values: 真值字典 {(idx, col): value}
                         col=-1 表示标签修复
            y_dirty: 原始脏标签（标签修复时需要）

        Returns:
            (X_result, y_result) - 修复后的特征和标签
        """
        X_result = X_dirty.copy()
        y_result = y_dirty.copy() if y_dirty is not None else None

        for plan_item in self.repair_plan:
            idx, col = plan_item['idx'], plan_item['col']
            repair_val = true_values.get((idx, col), plan_item['estimated_value'])

            if col == -1:
                # 标签修复
                if y_result is not None:
                    y_result[idx] = repair_val
            else:
                # 特征修复
                X_result[idx, col] = repair_val

        # 删除标记的行
        keep_mask = np.array([i not in self.deleted_rows for i in range(len(X_result))])
        X_result = X_result[keep_mask]
        if y_result is not None:
            y_result = y_result[keep_mask]

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

        return X_result, y_result

    def print_repair_plan(self, max_rows: int = 20) -> None:
        """打印修复计划"""
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        print(f"\n{'='*70}")
        print(f"修复计划 (共 {len(self.repair_plan)} 条，需要用户提供真值)")
        print(f"{'='*70}")

        if len(self.repair_plan) == 0:
            print("  (无需修复)")
            return

        print(f"{'索引':<8} {'列':<6} {'脏数据':<15} {'估计值':<15} {'错误类型':<10}")
        print("-" * 70)

        display_plan = self.repair_plan[:max_rows] if max_rows else self.repair_plan

        for record in display_plan:
            dirty_str = f"{record['current_dirty_value']:.4f}" if record['current_dirty_value'] is not None else "NaN"
            est_str = f"{record['estimated_value']:.4f}"
            error_type_str = error_type_names.get(record['error_type'], 'unknown')
            print(f"{record['idx']:<8} {record['col']:<6} {dirty_str:<15} {est_str:<15} {error_type_str:<10}")

        if max_rows and len(self.repair_plan) > max_rows:
            print(f"... 省略 {len(self.repair_plan) - max_rows} 条 ...")

        print(f"{'='*70}\n")

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """
        获取需要真值的位置列表

        Returns:
            [(idx, col), ...] 需要用户提供真值的位置
        """
        return [(p['idx'], p['col']) for p in self.repair_plan]

    def get_decision_log(self) -> List[Dict]:
        """获取完整决策日志（所有4种动作的详情）"""
        return self.decision_log

    def save_plan_csv(self, filepath: str) -> None:
        """
        将修复计划导出为 CSV 文件

        Args:
            filepath: CSV 文件路径
        """
        import csv
        error_type_names = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_idx', 'col', 'error_type', 'dirty_value', 'estimated_value'])
            for item in self.repair_plan:
                writer.writerow([
                    item['idx'],
                    item['col'],
                    error_type_names.get(item['error_type'], 'unknown'),
                    item['current_dirty_value'],
                    item['estimated_value']
                ])
        print(f"  修复计划已保存: {filepath} (共 {len(self.repair_plan)} 条)")

    def print_decision_summary(self, max_rows: int = 30) -> None:
        """
        打印分类汇总的决策日志

        按动作类型分组显示：repair(计划) → replace → delete → no_action
        """
        ACTION_NAMES = {0: 'no_action', 1: 'repair_value(计划)', 2: 'delete', 3: 'replace_nearby'}
        ERROR_TYPE_NAMES = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

        total = len(self.decision_log)
        if total == 0:
            print("  (无决策记录)")
            return

        # 动作分布
        print(f"\n  动作分布 (共 {total} 个错误):")
        ac_names = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
        for act_id in [0, 1, 2, 3]:
            act_name = ac_names[act_id]
            display_name = ACTION_NAMES[act_id]
            count = self.action_counts.get(act_name, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2.5)
            print(f"    {display_name:<20} {count:>5} ({pct:5.1f}%) {bar}")

        # 降级统计
        degraded = [d for d in self.decision_log if d['action'] != d['original_action']]
        if degraded:
            print(f"\n  动作降级: {len(degraded)} 次")

        # 替换明细
        replaces = [d for d in self.decision_log if d['action'] == 3]
        deletes = [d for d in self.decision_log if d['action'] == 2]

        def _fmt_val(v):
            if v is None:
                return 'NaN'
            return f'{v:.4f}'

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
        与 CleaningEnv._refresh_feature_importance 保持一致。
        """
        X_filled = self._fill_nan(self.X_current.copy())
        keep_mask = np.array([
            i not in self.deleted_rows for i in range(len(X_filled))
        ])
        if keep_mask.sum() < 10:
            return
        try:
            self.model_adapter.fit(X_filled[keep_mask], self.y_current[keep_mask])
            new_importance = self.model_adapter.get_feature_importance()
            if new_importance is not None:
                self.state_extractor.feature_importance = new_importance
        except Exception:
            pass
