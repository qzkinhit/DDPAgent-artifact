"""
三维度 Shapley 分析模块
========================

对 DemandClean 系统进行三个维度的 Shapley 值分析，量化各因素对下游任务性能的贡献：

维度1 - 动作重要性 Shapley:
    4个玩家: no_action, repair_value, delete, replace_nearby
    联盟值 v(S) = 只允许 S 中动作进行清洗后的下游性能
    精确枚举 4! = 24 个排列

维度2 - 特征重要性 Shapley:
    N个玩家 = 数据的各特征列
    联盟值 v(S) = 只对 S 中列的错误进行清洗后的下游性能
    蒙特卡洛采样（默认200次）

维度3 - 错误类型重要性 Shapley:
    3个玩家: missing, semantic, syntactic
    联盟值 v(S) = 只清洗 S 中类型的错误后的下游性能
    精确枚举 3! = 6 个排列

使用示例:
    >>> from demandclean.tools.shapley_analysis import run_full_shapley_analysis
    >>> results = run_full_shapley_analysis(
    ...     agent=agent,
    ...     X_dirty=X_dirty,
    ...     y=y,
    ...     X_clean=X_clean,
    ...     error_list=error_list,
    ...     config=config,
    ...     output_dir='shapley_results'
    ... )
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')  # 无头模式，避免弹窗
import matplotlib.pyplot as plt
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import BaseAgent
from ..core.environments import CleaningEnv
from ..core.state import (
    StateExtractor,
    ClassificationStateExtractor,
    RegressionStateExtractor,
    ClusteringStateExtractor,
)
from ..models import ModelAdapter, create_model_adapter


# ---------------------------------------------------------------------------
# 辅助：创建状态提取器
# ---------------------------------------------------------------------------

def _create_state_extractor(
    model_adapter: ModelAdapter,
    config: DemandCleanConfig,
) -> StateExtractor:
    """根据任务类型创建状态提取器"""
    if config.task_type == TaskType.REGRESSION:
        return RegressionStateExtractor(model_adapter, config)
    elif config.task_type == TaskType.CLUSTERING:
        return ClusteringStateExtractor(model_adapter, config)
    else:
        return ClassificationStateExtractor(model_adapter, config)


# ---------------------------------------------------------------------------
# 辅助：填充 NaN
# ---------------------------------------------------------------------------

def _fill_nan(X: np.ndarray) -> np.ndarray:
    """用列均值填充 NaN"""
    X_out = X.copy()
    for col in range(X_out.shape[1]):
        mask = np.isnan(X_out[:, col])
        if mask.any():
            col_mean = np.nanmean(X_out[:, col])
            X_out[mask, col] = col_mean if not np.isnan(col_mean) else 0.0
    return X_out


# ---------------------------------------------------------------------------
# 辅助：评估下游性能
# ---------------------------------------------------------------------------

def _evaluate_downstream(
    X: np.ndarray,
    y: np.ndarray,
    model_adapter: ModelAdapter,
    deleted_rows: Optional[Set[int]] = None,
) -> float:
    """
    在数据上训练模型并评估下游性能（使用 train/test split 避免过拟合偏差）

    Args:
        X: 特征矩阵（可能包含 NaN）
        y: 标签向量
        model_adapter: 模型适配器（会被 clone 后使用）
        deleted_rows: 需要排除的行索引集合

    Returns:
        性能得分（分类=accuracy, 回归=R2，由适配器决定）
    """
    X_filled = _fill_nan(X)

    if deleted_rows:
        keep = np.array([i not in deleted_rows for i in range(len(X_filled))])
        if keep.sum() < 10:
            return 0.0
        X_available = X_filled[keep]
        y_available = y[keep]
    else:
        X_available = X_filled
        y_available = y

    # 使用 80/20 train/test split 避免 train-on-test 偏差
    n = len(X_available)
    if n < 10:
        return 0.0

    rng = np.random.RandomState(42)
    n_test = max(2, int(n * 0.2))
    test_idx = rng.choice(n, size=n_test, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    X_train, y_train = X_available[train_mask], y_available[train_mask]
    X_test, y_test = X_available[test_idx], y_available[test_idx]

    adapter = model_adapter.clone()
    try:
        adapter.fit(X_train, y_train)
        return adapter.evaluate(X_test, y_test)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 辅助：使用 Agent 进行受限清洗
# ---------------------------------------------------------------------------

def _run_constrained_cleaning(
    agent: BaseAgent,
    X_dirty: np.ndarray,
    y: np.ndarray,
    error_list: List[Dict[str, Any]],
    config: DemandCleanConfig,
    allowed_actions: Optional[Set[int]] = None,
    allowed_columns: Optional[Set[int]] = None,
    allowed_error_types: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
    """
    使用 Agent 在约束条件下进行清洗

    通过过滤 error_list 或覆盖动作来实现"只允许部分动作/列/类型"的语义。

    策略:
      - allowed_actions: Agent 选出的动作若不在此集合中，则强制改为 no_action(0)
      - allowed_columns: 只保留 error_list 中列属于此集合的错误
      - allowed_error_types: 只保留 error_list 中类型属于此集合的错误

    Args:
        agent: 训练好的 DQN Agent
        X_dirty: 脏数据矩阵
        y: 标签向量
        error_list: 完整错误列表
        config: 配置对象
        allowed_actions: 允许的动作集合 (0/1/2/3)
        allowed_columns: 允许清洗的列索引集合
        allowed_error_types: 允许清洗的错误类型集合 (0/1/2)

    Returns:
        (X_cleaned, y_cleaned, deleted_rows)
    """
    # ---------- 按列/类型过滤 error_list ----------
    filtered_errors = error_list
    if allowed_columns is not None:
        filtered_errors = [e for e in filtered_errors if e['col'] in allowed_columns]
    if allowed_error_types is not None:
        filtered_errors = [e for e in filtered_errors if e['type'] in allowed_error_types]

    # 如果过滤后为空，直接返回原始脏数据
    if len(filtered_errors) == 0:
        return X_dirty.copy(), y.copy(), set()

    # ---------- 创建环境 ----------
    model_adapter = create_model_adapter(config.model_type, config.task_type)
    state_extractor = _create_state_extractor(model_adapter, config)

    env = CleaningEnv(
        X_dirty.copy(), y.copy(), copy.deepcopy(filtered_errors),
        model_adapter, state_extractor, config,
    )

    # ---------- 推理 ----------
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # 推理模式

    state = env.reset()
    while True:
        # 获取 Agent 动作
        if config.agent_type in (AgentType.TWO_STAGE, AgentType.DUELING_TWO_STAGE):
            raw_action, _, _ = agent.act(state, training=False)
        else:
            raw_action = agent.act(state, training=False)

        # 动作约束：若不在允许集合中则改为 no_action
        if allowed_actions is not None and raw_action not in allowed_actions:
            raw_action = 0  # 强制 no_action

        next_state, _, done, _ = env.step(raw_action)
        state = next_state
        if done:
            break

    agent.epsilon = old_epsilon  # 恢复

    X_result, y_result, _ = env.get_cleaned_data()
    # get_cleaned_data() 已经移除了 deleted 行，
    # 所以返回空集避免 _evaluate_downstream 双重删除
    return X_result, y_result, set()


# ===========================================================================
# 基类
# ===========================================================================

class _BaseShapleyAnalyzer(ABC):
    """Shapley 值分析器基类"""

    def __init__(
        self,
        agent: BaseAgent,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: np.ndarray,
        error_list: List[Dict[str, Any]],
        config: DemandCleanConfig,
    ):
        """
        Args:
            agent: 训练好的 DQN Agent
            X_dirty: 脏数据矩阵
            y: 标签向量
            X_clean: 干净数据（用于评估基线/参考）
            error_list: 完整错误列表
            config: 配置对象
        """
        self.agent = agent
        self.X_dirty = X_dirty.copy()
        self.y = y.copy()
        self.X_clean = X_clean.copy()
        self.error_list = copy.deepcopy(error_list)
        self.config = config

        # 缓存联盟值，避免重复计算
        self._coalition_cache: Dict[FrozenSet, float] = {}

        # 评估用的模型适配器
        self._eval_adapter = create_model_adapter(config.model_type, config.task_type)

    # ------ 需要子类实现 ------

    @abstractmethod
    def get_players(self) -> List[Any]:
        """返回玩家列表"""
        ...

    @abstractmethod
    def get_player_names(self) -> List[str]:
        """返回玩家名称列表（用于可视化）"""
        ...

    @abstractmethod
    def coalition_value(self, coalition: FrozenSet) -> float:
        """
        计算联盟值 v(S)

        Args:
            coalition: 玩家的冻结集合

        Returns:
            下游性能得分
        """
        ...

    # ------ 通用计算 ------

    def _cached_coalition_value(self, coalition: FrozenSet) -> float:
        """带缓存的联盟值计算"""
        if coalition not in self._coalition_cache:
            self._coalition_cache[coalition] = self.coalition_value(coalition)
        return self._coalition_cache[coalition]

    def compute_exact(self) -> Dict[Any, float]:
        """
        精确计算 Shapley 值（枚举所有排列）

        适用于玩家数量较少的情况（<= 6）

        Returns:
            {player: shapley_value} 字典
        """
        players = self.get_players()
        n = len(players)
        shapley = {p: 0.0 for p in players}
        n_perms = 0

        for perm in itertools.permutations(players):
            n_perms += 1
            current_set: Set = set()
            for player in perm:
                # v(S ∪ {i}) - v(S)
                s_without = frozenset(current_set)
                current_set.add(player)
                s_with = frozenset(current_set)

                v_with = self._cached_coalition_value(s_with)
                v_without = self._cached_coalition_value(s_without)

                shapley[player] += (v_with - v_without)

        # 取平均
        for p in players:
            shapley[p] /= n_perms

        return shapley

    def compute_monte_carlo(self, n_samples: int = 200, seed: int = 42) -> Dict[Any, float]:
        """
        蒙特卡洛近似 Shapley 值

        通过随机采样排列来近似。

        Args:
            n_samples: 采样排列数量
            seed: 随机种子

        Returns:
            {player: shapley_value} 字典
        """
        rng = np.random.RandomState(seed)
        players = self.get_players()
        shapley = {p: 0.0 for p in players}

        for _ in range(n_samples):
            perm = list(players)
            rng.shuffle(perm)

            current_set: Set = set()
            for player in perm:
                s_without = frozenset(current_set)
                current_set.add(player)
                s_with = frozenset(current_set)

                v_with = self._cached_coalition_value(s_with)
                v_without = self._cached_coalition_value(s_without)

                shapley[player] += (v_with - v_without)

        for p in players:
            shapley[p] /= n_samples

        return shapley

    def plot(
        self,
        shapley_values: Dict[Any, float],
        title: str,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制 Shapley 值柱状图

        Args:
            shapley_values: {player: value} 字典
            title: 图表标题
            save_path: 保存路径（PNG），为 None 则不保存
        """
        names = self.get_player_names()
        players = self.get_players()
        values = [shapley_values.get(p, 0.0) for p in players]

        # 按值降序排列
        sorted_indices = np.argsort(values)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # 颜色：正值用蓝色，负值用红色
        colors = ['#2196F3' if v >= 0 else '#F44336' for v in sorted_values]

        # 限制图表宽度，避免超大特征数导致像素溢出
        fig_width = min(max(8, len(names) * 1.2), 40)
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        bars = ax.bar(range(len(sorted_names)), sorted_values, color=colors, edgecolor='white', linewidth=0.8)

        ax.set_xticks(range(len(sorted_names)))
        # 特征数多时使用更小字体和更大旋转角度
        rotation = 90 if len(sorted_names) > 20 else 30
        fontsize = max(5, min(10, 200 // max(len(sorted_names), 1)))
        ax.set_xticklabels(sorted_names, rotation=rotation, ha='right', fontsize=fontsize)
        ax.set_ylabel('Shapley Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        # 保护 y 轴范围：当所有值为 0 或近似 0 时，设置合理的最小范围
        y_min_val = min(sorted_values) if sorted_values else 0
        y_max_val = max(sorted_values) if sorted_values else 0
        y_range = y_max_val - y_min_val
        if y_range < 1e-8:
            ax.set_ylim(-0.01, 0.01)

        # 在柱子上标注数值（值全为 0 时跳过标注避免渲染问题）
        if y_range >= 1e-8:
            for bar_obj, val in zip(bars, sorted_values):
                y_pos = bar_obj.get_height()
                offset = 0.002 if val >= 0 else -0.002
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    y_pos + offset,
                    f'{val:.4f}',
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=max(6, min(9, 180 // max(len(sorted_names), 1))),
                )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"[Shapley] 图表已保存: {save_path}")

        plt.close(fig)


# ===========================================================================
# 维度1: 动作重要性 Shapley
# ===========================================================================

class ActionShapleyAnalyzer(_BaseShapleyAnalyzer):
    """
    动作重要性 Shapley 分析

    4个玩家: no_action(0), repair_value(1), delete(2), replace_nearby(3)
    联盟值 v(S) = 只允许 S 中动作进行清洗后的下游性能
    精确枚举 4! = 24 个排列
    """

    # 动作名称映射
    ACTION_NAMES = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}

    def get_players(self) -> List[int]:
        return [0, 1, 2, 3]

    def get_player_names(self) -> List[str]:
        return [self.ACTION_NAMES[i] for i in self.get_players()]

    def coalition_value(self, coalition: FrozenSet) -> float:
        """
        联盟值: 只允许 coalition 中的动作

        空联盟 => 不做任何清洗，返回脏数据的下游性能
        """
        if len(coalition) == 0:
            # v(空集) = 脏数据直接下游评估
            return _evaluate_downstream(self.X_dirty, self.y, self._eval_adapter)

        allowed_actions = set(coalition)

        # 若联盟里没有 no_action(0)，仍然允许 no_action 作为"被迫不操作"的默认
        # 设计选择：当 Agent 选的动作不在 allowed 中时，我们强制它为 no_action
        # 但 no_action 本身的贡献需要单独衡量
        # 这里不额外加入 0，保证 Shapley 分解的语义正确

        X_cleaned, y_cleaned, deleted = _run_constrained_cleaning(
            self.agent, self.X_dirty, self.y,
            self.error_list, self.config,
            allowed_actions=allowed_actions,
        )
        return _evaluate_downstream(X_cleaned, y_cleaned, self._eval_adapter, deleted)

    def run(self) -> Dict[str, float]:
        """
        运行动作重要性 Shapley 分析（精确枚举）

        Returns:
            {动作名: Shapley值}
        """
        print("[Shapley] 维度1: 动作重要性 (精确枚举 4!=24 排列) ...")
        raw = self.compute_exact()
        return {self.ACTION_NAMES[k]: v for k, v in raw.items()}


# ===========================================================================
# 维度2: 特征重要性 Shapley
# ===========================================================================

class FeatureShapleyAnalyzer(_BaseShapleyAnalyzer):
    """
    特征重要性 Shapley 分析

    N个玩家 = 数据各特征列
    联盟值 v(S) = 只对 S 中列的错误进行清洗后的下游性能
    蒙特卡洛采样（默认200次，因为特征数可能很多）
    """

    def __init__(
        self,
        agent: BaseAgent,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: np.ndarray,
        error_list: List[Dict[str, Any]],
        config: DemandCleanConfig,
        column_names: Optional[List[str]] = None,
    ):
        super().__init__(agent, X_dirty, y, X_clean, error_list, config)
        n_features = X_dirty.shape[1]
        if column_names and len(column_names) == n_features:
            self._column_names = column_names
        else:
            self._column_names = [f'feature_{i}' for i in range(n_features)]

    def get_players(self) -> List[int]:
        return list(range(self.X_dirty.shape[1]))

    def get_player_names(self) -> List[str]:
        return self._column_names

    def coalition_value(self, coalition: FrozenSet) -> float:
        """
        联盟值: 只对 coalition 中列的错误进行清洗

        空联盟 => 不清洗任何列，返回脏数据的下游性能
        """
        if len(coalition) == 0:
            return _evaluate_downstream(self.X_dirty, self.y, self._eval_adapter)

        allowed_columns = set(coalition)

        X_cleaned, y_cleaned, deleted = _run_constrained_cleaning(
            self.agent, self.X_dirty, self.y,
            self.error_list, self.config,
            allowed_columns=allowed_columns,
        )
        return _evaluate_downstream(X_cleaned, y_cleaned, self._eval_adapter, deleted)

    def run(self, n_samples: int = 200, seed: int = 42) -> Dict[str, float]:
        """
        运行特征重要性 Shapley 分析（蒙特卡洛采样）

        Args:
            n_samples: 采样排列数量
            seed: 随机种子

        Returns:
            {特征名: Shapley值}
        """
        n_features = self.X_dirty.shape[1]
        print(f"[Shapley] 维度2: 特征重要性 (蒙特卡洛 {n_samples} 次, {n_features} 个特征) ...")
        raw = self.compute_monte_carlo(n_samples=n_samples, seed=seed)
        return {self._column_names[k]: v for k, v in raw.items()}


# ===========================================================================
# 维度2(高维): 特征分组 Shapley
# ===========================================================================

class GroupedFeatureShapleyAnalyzer(_BaseShapleyAnalyzer):
    """
    特征分组 Shapley 分析（高维加速版）

    当特征数过多时（>MAX_FEATURE_SHAPLEY），直接计算每列的 Shapley 值会导致
    计算量 ≈ mc_samples × n_features 次完整推理，不可接受。

    加速策略:
    1. 计算特征相关性矩阵
    2. 用层次聚类将 N 个特征分成 K 组 (K ≈ 20)
    3. 每组作为一个"玩家"，coalition_value = 只清洗该组所有列的错误
    4. 对 K 组计算 Shapley 值（精确枚举或少量 MC）
    5. 组的 Shapley 值均分到组内各特征

    计算量: mc_samples × K ≈ 200 × 20 = 4000 次（vs 200 × 376 = 75200）
    """

    DEFAULT_N_GROUPS = 20  # 默认分组数

    def __init__(
        self,
        agent: BaseAgent,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: np.ndarray,
        error_list: List[Dict[str, Any]],
        config: DemandCleanConfig,
        column_names: Optional[List[str]] = None,
        n_groups: int = DEFAULT_N_GROUPS,
    ):
        super().__init__(agent, X_dirty, y, X_clean, error_list, config)
        n_features = X_dirty.shape[1]
        if column_names and len(column_names) == n_features:
            self._column_names = column_names
        else:
            self._column_names = [f'feature_{i}' for i in range(n_features)]

        # 分组数不超过特征数
        self._n_groups = min(n_groups, n_features)
        # 执行分组
        self._groups, self._group_names = self._cluster_features()
        # 缓存组级 Shapley 结果（避免 run() 和 get_group_shapley() 双重计算）
        self._cached_group_shapley: Optional[Dict[int, float]] = None

    def _cluster_features(self) -> Tuple[List[List[int]], List[str]]:
        """
        基于相关性矩阵的层次聚类分组

        Returns:
            (groups, group_names):
                groups[i] = 第 i 组包含的特征列索引列表
                group_names[i] = 第 i 组的名称（包含的列名摘要）
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        n_features = self.X_dirty.shape[1]

        # 计算相关性矩阵，转为距离矩阵
        # 使用 clean 数据（无噪声）来计算相关性更准确
        try:
            corr = np.corrcoef(self.X_clean.T)
            # 处理 NaN（常量列相关系数为 NaN）
            corr = np.nan_to_num(corr, nan=0.0)
            # 相关性 -> 距离: d = 1 - |r|
            dist_matrix = 1.0 - np.abs(corr)
            np.fill_diagonal(dist_matrix, 0)
            # 确保对称和非负
            dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, 1)

            # 转为压缩距离向量
            dist_condensed = squareform(dist_matrix, checks=False)
            # 层次聚类
            Z = linkage(dist_condensed, method='average')
            labels = fcluster(Z, t=self._n_groups, criterion='maxclust')
        except Exception:
            # 聚类失败则均匀分组
            labels = np.array([(i % self._n_groups) + 1 for i in range(n_features)])

        # 按标签组织分组
        groups: List[List[int]] = [[] for _ in range(self._n_groups)]
        for col_idx, label in enumerate(labels):
            groups[label - 1].append(col_idx)

        # 移除空组
        groups = [g for g in groups if len(g) > 0]

        # 生成组名（取前 3 个特征名 + 数量）
        group_names = []
        for i, g in enumerate(groups):
            names = [self._column_names[j] for j in g[:3]]
            suffix = f'+{len(g)-3}' if len(g) > 3 else ''
            group_names.append(f'G{i}({",".join(names)}{suffix})')

        return groups, group_names

    def get_players(self) -> List[int]:
        """玩家 = 特征组索引"""
        return list(range(len(self._groups)))

    def get_player_names(self) -> List[str]:
        return self._group_names

    def coalition_value(self, coalition: FrozenSet) -> float:
        """
        联盟值: 只清洗 coalition 中组所包含的列的错误

        空联盟 => 不清洗任何列，返回脏数据的下游性能
        """
        if len(coalition) == 0:
            return _evaluate_downstream(self.X_dirty, self.y, self._eval_adapter)

        # 展开组索引 -> 列索引
        allowed_columns: Set[int] = set()
        for group_idx in coalition:
            allowed_columns.update(self._groups[group_idx])

        X_cleaned, y_cleaned, deleted = _run_constrained_cleaning(
            self.agent, self.X_dirty, self.y,
            self.error_list, self.config,
            allowed_columns=allowed_columns,
        )
        return _evaluate_downstream(X_cleaned, y_cleaned, self._eval_adapter, deleted)

    def _compute_group_shapley(self, n_samples: int = 200, seed: int = 42) -> Dict[int, float]:
        """计算组级 Shapley 值（带缓存）"""
        if self._cached_group_shapley is not None:
            return self._cached_group_shapley

        n_groups = len(self._groups)
        n_features = self.X_dirty.shape[1]

        # 组数 ≤ 6 则精确枚举，否则蒙特卡洛
        if n_groups <= 6:
            print(f"[Shapley] 维度2: 分组特征重要性 ({n_features} 特征 → {n_groups} 组, "
                  f"精确枚举 {n_groups}! 排列)")
            group_shapley = self.compute_exact()
        else:
            # 20 组用 50 次 MC 已足够收敛，不需要 200 次
            effective_mc = min(n_samples, max(30, 50))
            print(f"[Shapley] 维度2: 分组特征重要性 ({n_features} 特征 → {n_groups} 组, "
                  f"蒙特卡洛 {effective_mc} 次)")
            group_shapley = self.compute_monte_carlo(n_samples=effective_mc, seed=seed)

        self._cached_group_shapley = group_shapley
        return group_shapley

    def run(self, n_samples: int = 200, seed: int = 42) -> Dict[str, float]:
        """
        运行分组特征 Shapley 分析

        Returns:
            {特征名: Shapley值} — 组 Shapley 值均分到组内各特征
        """
        group_shapley = self._compute_group_shapley(n_samples, seed)

        # 将组 Shapley 值均分到组内各特征
        feature_shapley: Dict[str, float] = {}
        for group_idx, group_cols in enumerate(self._groups):
            group_value = group_shapley.get(group_idx, 0.0)
            per_feature = group_value / len(group_cols)
            for col_idx in group_cols:
                feature_shapley[self._column_names[col_idx]] = per_feature

        return feature_shapley

    def get_group_shapley(self, n_samples: int = 200, seed: int = 42) -> Dict[str, float]:
        """
        返回组级别的 Shapley 值（用于绘图）

        Returns:
            {组名: Shapley值}
        """
        group_shapley = self._compute_group_shapley(n_samples, seed)
        return {self._group_names[k]: v for k, v in group_shapley.items()}

    def get_groups_info(self) -> List[Dict[str, Any]]:
        """返回分组信息（用于报告）"""
        info = []
        for i, (cols, name) in enumerate(zip(self._groups, self._group_names)):
            info.append({
                'group_id': i,
                'group_name': name,
                'n_features': len(cols),
                'feature_indices': cols,
                'feature_names': [self._column_names[j] for j in cols],
            })
        return info


# ===========================================================================
# 维度3: 错误类型重要性 Shapley
# ===========================================================================

class ErrorTypeShapleyAnalyzer(_BaseShapleyAnalyzer):
    """
    错误类型重要性 Shapley 分析

    3个玩家: missing(0), semantic(1), syntactic(2)
    联盟值 v(S) = 只清洗 S 中类型的错误后的下游性能
    精确枚举 3! = 6 个排列
    """

    ERROR_TYPE_NAMES = {0: 'missing', 1: 'semantic', 2: 'syntactic', 3: 'label_noise'}

    def get_players(self) -> List[int]:
        # 只返回 error_list 中实际出现的类型
        present = set(e['type'] for e in self.error_list)
        return sorted(present)

    def get_player_names(self) -> List[str]:
        return [self.ERROR_TYPE_NAMES.get(p, f'type_{p}') for p in self.get_players()]

    def coalition_value(self, coalition: FrozenSet) -> float:
        """
        联盟值: 只清洗 coalition 中类型的错误

        空联盟 => 不清洗任何类型，返回脏数据的下游性能
        """
        if len(coalition) == 0:
            return _evaluate_downstream(self.X_dirty, self.y, self._eval_adapter)

        allowed_types = set(coalition)

        X_cleaned, y_cleaned, deleted = _run_constrained_cleaning(
            self.agent, self.X_dirty, self.y,
            self.error_list, self.config,
            allowed_error_types=allowed_types,
        )
        return _evaluate_downstream(X_cleaned, y_cleaned, self._eval_adapter, deleted)

    def run(self) -> Dict[str, float]:
        """
        运行错误类型重要性 Shapley 分析（精确枚举）

        Returns:
            {错误类型名: Shapley值}
        """
        players = self.get_players()
        n_fact = 1
        for i in range(1, len(players) + 1):
            n_fact *= i
        print(f"[Shapley] 维度3: 错误类型重要性 (精确枚举 {len(players)}!={n_fact} 排列) ...")
        raw = self.compute_exact()
        return {self.ERROR_TYPE_NAMES.get(k, f'type_{k}'): v for k, v in raw.items()}


# ===========================================================================
# 统一入口
# ===========================================================================

def run_full_shapley_analysis(
    agent: BaseAgent,
    X_dirty: np.ndarray,
    y: np.ndarray,
    X_clean: np.ndarray,
    error_list: List[Dict[str, Any]],
    config: DemandCleanConfig,
    output_dir: str = 'shapley_results',
    column_names: Optional[List[str]] = None,
    mc_samples: int = 200,
    mc_seed: int = 42,
    verbose: bool = True,
    max_rows_for_shapley: int = 5000,
) -> Dict[str, Any]:
    """
    运行全部三个维度的 Shapley 分析，生成可视化和 JSON 结果

    Args:
        agent: 训练好的 DQN Agent
        X_dirty: 脏数据矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        X_clean: 干净数据矩阵（同形状，用于参考评估）
        error_list: 完整错误列表
            [{'idx': int, 'col': int, 'type': int, 'repair_value': float}, ...]
        config: DemandCleanConfig 配置对象
        output_dir: 输出目录
        column_names: 特征列名称列表（可选）
        mc_samples: 蒙特卡洛采样数（特征维度使用）
        mc_seed: 蒙特卡洛随机种子
        verbose: 是否打印详细信息
        max_rows_for_shapley: 大数据集采样上限，超过此行数时自动采样

    Returns:
        汇总结果字典，包含三个维度的 Shapley 值和元信息
    """
    os.makedirs(output_dir, exist_ok=True)

    # 大数据集自动采样：避免 Shapley 计算超时
    n_rows = X_dirty.shape[0]
    sampled = False
    if n_rows > max_rows_for_shapley:
        sampled = True
        rng = np.random.RandomState(mc_seed)
        error_row_indices = set(e['idx'] for e in error_list if e['idx'] < n_rows)
        non_error_indices = [i for i in range(n_rows) if i not in error_row_indices]

        n_error = len(error_row_indices)
        n_non_error = len(non_error_indices)

        if n_error <= max_rows_for_shapley:
            # 错误行全保留，从非错误行中采样补足
            keep_error = list(error_row_indices)
            n_sample_non_error = max(0, max_rows_for_shapley - n_error)
            if n_sample_non_error < n_non_error:
                keep_non_error = list(rng.choice(non_error_indices, n_sample_non_error, replace=False))
            else:
                keep_non_error = non_error_indices
            sample_desc = f"保留全部 {n_error} 个错误行"
        else:
            # 错误行超过上限 → 按比例采样错误行和非错误行
            error_ratio = n_error / n_rows
            n_sample_error = int(max_rows_for_shapley * error_ratio)
            n_sample_non_error = max_rows_for_shapley - n_sample_error
            # 确保至少采样一些错误行和非错误行
            n_sample_error = max(n_sample_error, min(100, n_error))
            n_sample_non_error = max(n_sample_non_error, min(100, n_non_error))
            keep_error = list(rng.choice(list(error_row_indices), min(n_sample_error, n_error), replace=False))
            if n_sample_non_error < n_non_error:
                keep_non_error = list(rng.choice(non_error_indices, n_sample_non_error, replace=False))
            else:
                keep_non_error = non_error_indices
            sample_desc = f"采样 {len(keep_error)}/{n_error} 个错误行 ({error_ratio*100:.0f}% 错误率)"

        keep_indices = sorted(set(keep_error + keep_non_error))
        # 建立旧→新索引映射
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        X_dirty = X_dirty[keep_indices]
        y = y[keep_indices]
        X_clean = X_clean[keep_indices]
        # 重映射 error_list 中的 idx
        new_error_list = []
        for e in error_list:
            if e['idx'] in old_to_new:
                new_e = dict(e)
                new_e['idx'] = old_to_new[e['idx']]
                new_error_list.append(new_e)
        error_list = new_error_list
        # 大数据集同时减少蒙特卡洛次数
        mc_samples = min(mc_samples, 50)
        if verbose:
            print(f"[Shapley] 大数据集采样: {n_rows} -> {len(keep_indices)} 行"
                  f" ({sample_desc}, mc_samples={mc_samples})")

    # ---------- 错误数量采样：当 error_list 过大时，按类型分层采样 ----------
    MAX_ERRORS_FOR_SHAPLEY = 50000
    original_error_count = len(error_list)
    if original_error_count > MAX_ERRORS_FOR_SHAPLEY:
        rng_err = np.random.RandomState(mc_seed)
        by_type: Dict[int, list] = {}
        for e in error_list:
            by_type.setdefault(e['type'], []).append(e)
        sampled_errors: list = []
        for t, errs in by_type.items():
            ratio = len(errs) / original_error_count
            n_sample = max(100, int(MAX_ERRORS_FOR_SHAPLEY * ratio))
            if n_sample < len(errs):
                indices = rng_err.choice(len(errs), n_sample, replace=False)
                sampled_errors.extend([errs[i] for i in indices])
            else:
                sampled_errors.extend(errs)
        error_list = sampled_errors
        print(f"[Shapley] 错误数量采样: {original_error_count} -> {len(error_list)} 条（按类型分层）")

    if verbose:
        print("=" * 60)
        print("DemandClean - 三维度 Shapley 分析")
        print("=" * 60)
        print(f"  数据: {X_dirty.shape[0]} 行 x {X_dirty.shape[1]} 列")
        if sampled:
            print(f"  (原始数据 {n_rows} 行, 已采样至 {X_dirty.shape[0]} 行)")
        print(f"  错误数: {len(error_list)}")
        type_counts = {}
        for e in error_list:
            t = {0: 'missing', 1: 'semantic', 2: 'syntactic'}.get(e['type'], 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  错误分布: {type_counts}")
        print(f"  输出目录: {output_dir}")
        print()

    results: Dict[str, Any] = {
        'meta': {
            'n_samples': int(X_dirty.shape[0]),
            'n_features': int(X_dirty.shape[1]),
            'n_errors': len(error_list),
            'task_type': config.task_type.value,
            'model_type': config.model_type.value,
            'mc_samples': mc_samples,
        }
    }

    # ===================== 维度1: 动作重要性 =====================
    action_analyzer = ActionShapleyAnalyzer(
        agent, X_dirty, y, X_clean, error_list, config,
    )
    action_shapley = action_analyzer.run()
    results['action_shapley'] = action_shapley

    action_png = os.path.join(output_dir, 'shapley_action_importance.png')
    # 需要把字符串键转为整数键来 plot
    action_raw = {k: v for k, v in zip(action_analyzer.get_players(),
                                        [action_shapley[n] for n in action_analyzer.get_player_names()])}
    action_analyzer.plot(
        action_raw,
        title='Shapley Values - Action Importance',
        save_path=action_png,
    )

    if verbose:
        print(f"  动作重要性 Shapley 值:")
        for name, val in sorted(action_shapley.items(), key=lambda x: -x[1]):
            print(f"    {name:20s}: {val:+.6f}")
        print()

    # ===================== 维度2: 特征重要性（跳过） =====================
    n_features = X_dirty.shape[1]
    print(f"[Shapley] 维度2: 跳过特征 Shapley（计算量过大，仅保留维度1和维度3）")
    results['feature_shapley'] = {col: 0.0 for col in (column_names or [f'f{i}' for i in range(n_features)])}
    results['feature_shapley_skipped'] = True
    feature_png = os.path.join(output_dir, 'shapley_feature_importance.png')  # 占位，不生成图

    # ===================== 维度3: 错误类型重要性 =====================
    error_type_analyzer = ErrorTypeShapleyAnalyzer(
        agent, X_dirty, y, X_clean, error_list, config,
    )
    error_type_shapley = error_type_analyzer.run()
    results['error_type_shapley'] = error_type_shapley

    error_type_png = os.path.join(output_dir, 'shapley_error_type_importance.png')
    error_type_raw = {k: v for k, v in zip(error_type_analyzer.get_players(),
                                            [error_type_shapley[n] for n in error_type_analyzer.get_player_names()])}
    error_type_analyzer.plot(
        error_type_raw,
        title='Shapley Values - Error Type Importance',
        save_path=error_type_png,
    )

    if verbose:
        print(f"  错误类型重要性 Shapley 值:")
        for name, val in sorted(error_type_shapley.items(), key=lambda x: -x[1]):
            print(f"    {name:20s}: {val:+.6f}")
        print()

    # ===================== 保存 JSON 结果 =====================
    json_path = os.path.join(output_dir, 'shapley_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[Shapley] JSON 结果已保存: {json_path}")

    # ===================== 生成结构化 Markdown 报告 =====================
    report_lines = _generate_shapley_report(results, column_names)
    results['report_text'] = '\n'.join(report_lines)

    md_path = os.path.join(output_dir, 'shapley_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(results['report_text'])
    if verbose:
        print(f"[Shapley] Markdown 报告已保存: {md_path}")

    # ===================== 控制台汇总 =====================
    if verbose:
        print("\n" + "=" * 60)
        print("Shapley 分析完成!")
        print("=" * 60)
        # 打印精简版结论
        for line in report_lines:
            print(line)
        print(f"\n  输出文件:")
        print(f"    {action_png}")
        print(f"    {feature_png}")
        print(f"    {error_type_png}")
        print(f"    {json_path}")
        print(f"    {md_path}")
        print("=" * 60)

    return results


def _generate_shapley_report(results: Dict[str, Any],
                              column_names: Optional[List[str]] = None) -> List[str]:
    """
    根据 Shapley 分析结果生成结构化 Markdown 报告，包含排名表格和结论性解读。

    Args:
        results: run_full_shapley_analysis 的返回值
        column_names: 特征列名

    Returns:
        Markdown 行列表
    """
    lines: List[str] = []
    meta = results.get('meta', {})

    lines.append("# Shapley Value Analysis Report")
    lines.append("")
    lines.append(f"- 数据规模: {meta.get('n_samples', '?')} 行 × {meta.get('n_features', '?')} 列")
    lines.append(f"- 错误总数: {meta.get('n_errors', '?')}")
    lines.append(f"- 任务类型: {meta.get('task_type', '?')} ({meta.get('model_type', '?')})")
    lines.append(f"- MC 采样次数: {meta.get('mc_samples', '?')}")
    lines.append("")

    # ---- 维度 1: 动作重要性 ----
    action_sv = results.get('action_shapley', {})
    if action_sv:
        lines.append("## 维度 1: 动作重要性 (Action Importance)")
        lines.append("")
        sorted_actions = sorted(action_sv.items(), key=lambda x: -x[1])
        total_abs = sum(abs(v) for v in action_sv.values())

        lines.append("| 排名 | 动作 | Shapley 值 | 占比 | 方向 |")
        lines.append("|------|------|-----------|------|------|")
        for rank, (name, val) in enumerate(sorted_actions, 1):
            pct = abs(val) / total_abs * 100 if total_abs > 0 else 0
            direction = "↑ 正贡献" if val > 0.001 else ("↓ 负贡献" if val < -0.001 else "— 无影响")
            lines.append(f"| {rank} | {name} | {val:+.6f} | {pct:.1f}% | {direction} |")
        lines.append("")

        # 结论
        best_action = sorted_actions[0][0] if sorted_actions else "N/A"
        worst = [n for n, v in sorted_actions if v < -0.001]
        lines.append(f"**结论**: 最有效的清洗动作是 **{best_action}**")
        if worst:
            lines.append(f"（{', '.join(worst)} 对性能有负面影响，建议减少使用）")
        lines.append("")

    # ---- 维度 2: 特征重要性 ----
    feature_sv = results.get('feature_shapley', {})
    if feature_sv:
        is_grouped = results.get('feature_shapley_grouped', False)
        title_suffix = "（分组计算）" if is_grouped else ""
        lines.append(f"## 维度 2: 特征重要性 (Feature Importance){title_suffix}")
        lines.append("")
        if is_grouped:
            groups_info = results.get('feature_groups', [])
            n_groups = len(groups_info)
            lines.append(f"> 特征数较多，使用相关性聚类将 {len(feature_sv)} 个特征分为 {n_groups} 组，")
            lines.append(f"> 组级 Shapley 值均分到组内各特征。")
            lines.append("")
        sorted_features = sorted(feature_sv.items(), key=lambda x: -x[1])
        total_abs = sum(abs(v) for v in feature_sv.values())

        lines.append("| 排名 | 特征 | Shapley 值 | 占比 | 清洗价值 |")
        lines.append("|------|------|-----------|------|---------|")
        for rank, (name, val) in enumerate(sorted_features, 1):
            pct = abs(val) / total_abs * 100 if total_abs > 0 else 0
            if val > 0.01:
                worth = "★ 高价值"
            elif val > 0.001:
                worth = "○ 有价值"
            elif val > -0.001:
                worth = "— 可忽略"
            else:
                worth = "✗ 负价值"
            lines.append(f"| {rank} | {name} | {val:+.6f} | {pct:.1f}% | {worth} |")
        lines.append("")

        # 结论
        high_value = [n for n, v in sorted_features if v > 0.01]
        low_value = [n for n, v in sorted_features if abs(v) < 0.001]
        neg_value = [n for n, v in sorted_features if v < -0.001]
        lines.append(f"**结论**: ")
        if high_value:
            lines.append(f"- 最值得清洗的特征: **{', '.join(high_value[:3])}**（优先投入清洗预算）")
        if low_value:
            lines.append(f"- 清洗价值极低的特征: {', '.join(low_value[:3])}（清洗它们几乎不影响性能）")
        if neg_value:
            lines.append(f"- 清洗后反而降低性能: {', '.join(neg_value[:3])}（建议保留原值不清洗）")
        lines.append("")

    # ---- 维度 3: 错误类型重要性 ----
    error_sv = results.get('error_type_shapley', {})
    if error_sv:
        # 修正旧数据中的 type_N 键名
        _type_fix = {'type_0': 'missing', 'type_1': 'semantic', 'type_2': 'syntactic', 'type_3': 'label_noise'}
        error_sv = {_type_fix.get(k, k): v for k, v in error_sv.items()}

        lines.append("## 维度 3: 错误类型重要性 (Error Type Importance)")
        lines.append("")
        sorted_errors = sorted(error_sv.items(), key=lambda x: -x[1])
        total_abs = sum(abs(v) for v in error_sv.values())

        lines.append("| 排名 | 错误类型 | Shapley 值 | 占比 | 优先级 |")
        lines.append("|------|---------|-----------|------|--------|")
        for rank, (name, val) in enumerate(sorted_errors, 1):
            pct = abs(val) / total_abs * 100 if total_abs > 0 else 0
            priority = "🔴 高" if pct > 40 else ("🟡 中" if pct > 20 else "🟢 低")
            lines.append(f"| {rank} | {name} | {val:+.6f} | {pct:.1f}% | {priority} |")
        lines.append("")

        best_error = sorted_errors[0][0] if sorted_errors else "N/A"
        lines.append(f"**结论**: 对下游性能影响最大的错误类型是 **{best_error}**，检测器应优先识别此类错误。")
        lines.append("")

    # ---- 综合建议 ----
    lines.append("## 综合清洗建议")
    lines.append("")
    if action_sv and feature_sv:
        best_action_name = sorted(action_sv.items(), key=lambda x: -x[1])[0][0]
        top_features = [n for n, v in sorted(feature_sv.items(), key=lambda x: -x[1])[:3] if v > 0]
        if top_features and error_sv:
            best_error_type = sorted(error_sv.items(), key=lambda x: -x[1])[0][0]
            lines.append(
                f"在有限的真值预算下，建议优先对 **{', '.join(top_features)}** 列中的 "
                f"**{best_error_type}** 类型错误执行 **{best_action_name}** 操作，"
                f"以最大化下游 {meta.get('task_type', '')} 任务性能。"
            )
        elif top_features:
            lines.append(
                f"建议优先清洗 **{', '.join(top_features)}** 列，"
                f"使用 **{best_action_name}** 策略。"
            )
    lines.append("")

    return lines
