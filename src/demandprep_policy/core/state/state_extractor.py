"""
状态特征提取器基类
==================

从数据和错误信息中提取 DQN 状态向量。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ...models.base_adapter import ModelAdapter
    from ...config.config import DemandPrepPolicyConfig


class StateExtractor(ABC):
    """
    状态特征提取器抽象基类

    负责从数据和错误信息中提取 8 维状态向量:
        1. error_type: 错误类型 (归一化到 [0, 1])
        2. feature_importance: 特征重要性
        3. distance_to_boundary: 到决策边界的距离
        4. row_position: 行位置
        5. col_index: 列索引
        6. col_error_rate: 当前列的错误率
        7. sample_retention: 样本保留率
        8. var_retention: 方差保留率
    """

    def __init__(self, model_adapter: 'ModelAdapter', config: 'DemandPrepPolicyConfig'):
        """
        初始化状态提取器

        Args:
            model_adapter: 模型适配器
            config: 配置对象
        """
        self.model_adapter = model_adapter
        self.config = config

        # 列统计量
        self.col_stats: Dict[int, Dict[str, float]] = {}

        # 特征重要性
        self.feature_importance: Optional[np.ndarray] = None

        # 列错误率
        self.col_error_rate: Optional[np.ndarray] = None

        # 原始数据统计
        self._n_samples: int = 0
        self._n_features: int = 0

    def initialize(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   error_list: List[Dict[str, Any]]) -> None:
        """
        初始化状态提取器

        Args:
            X: 数据矩阵
            y: 标签
            error_list: 错误列表
        """
        self._n_samples = len(X)
        self._n_features = X.shape[1] if X.ndim > 1 else 1

        self._compute_col_stats(X)
        self._compute_col_error_rate(error_list)
        self._train_reference_model(X, y)

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """计算列统计量"""
        n_cols = X.shape[1] if X.ndim > 1 else 1

        for col in range(n_cols):
            col_data = X[:, col] if X.ndim > 1 else X
            valid = col_data[~np.isnan(col_data)]

            if len(valid) > 0:
                self.col_stats[col] = {
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid)) + 1e-6,
                    'var': float(np.var(valid)),
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'median': float(np.median(valid)),
                }
            else:
                self.col_stats[col] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'var': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'median': 0.0,
                }

    def _compute_col_error_rate(self, error_list: List[Dict[str, Any]]) -> None:
        """计算每列的错误率"""
        col_error_counts = np.zeros(self._n_features)

        for error in error_list:
            col = error.get('col', 0)
            if 0 <= col < self._n_features:
                col_error_counts[col] += 1
            # col == -1 (标签错误) 不计入特征列错误率

        total = len(error_list) if error_list else 1
        self.col_error_rate = col_error_counts / total

    def _train_reference_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练参考模型用于计算特征重要性和边界距离"""
        X_filled = self.fill_nan(X)

        try:
            self.model_adapter.fit(X_filled, y)
            self.feature_importance = self.model_adapter.get_feature_importance()
        except Exception as e:
            # 训练失败，使用均匀分布
            print(f"参考模型训练失败: {e}")
            self.feature_importance = np.ones(self._n_features) / self._n_features

    def fill_nan(self, X: np.ndarray) -> np.ndarray:
        """用列均值填充 NaN"""
        X_filled = X.copy()
        n_cols = X_filled.shape[1] if X_filled.ndim > 1 else 1

        for col in range(n_cols):
            col_data = X_filled[:, col] if X_filled.ndim > 1 else X_filled
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                mean_val = self.col_stats.get(col, {}).get('mean', 0)
                if X_filled.ndim > 1:
                    X_filled[nan_mask, col] = mean_val
                else:
                    X_filled[nan_mask] = mean_val

        return X_filled

    @abstractmethod
    def extract(self,
                X_current: np.ndarray,
                y: np.ndarray,
                error: Dict[str, Any],
                deleted_rows: Set[int]) -> np.ndarray:
        """
        提取状态特征向量

        Args:
            X_current: 当前数据矩阵
            y: 标签
            error: 当前错误信息
            deleted_rows: 已删除的行集合

        Returns:
            8 维状态向量
        """
        pass

    @abstractmethod
    def get_distance_to_boundary(self,
                                  X_current: np.ndarray,
                                  idx: int,
                                  col: int) -> float:
        """
        获取到决策边界的归一化距离

        Args:
            X_current: 当前数据矩阵
            idx: 行索引
            col: 列索引

        Returns:
            归一化距离 [0, 1]
        """
        pass

    def compute_retention(self,
                          X_current: np.ndarray,
                          col: int,
                          deleted_rows: Set[int]) -> tuple:
        """
        计算样本保留率和方差保留率

        Args:
            X_current: 当前数据矩阵
            col: 列索引
            deleted_rows: 已删除的行集合

        Returns:
            (sample_retention, var_retention)
        """
        keep_mask = np.array([i not in deleted_rows for i in range(len(X_current))])
        n_kept = keep_mask.sum()

        if n_kept < 2:
            return 0.0, 0.0

        # 避免除以零
        if self._n_samples == 0:
            sample_retention = 1.0
        else:
            sample_retention = n_kept / self._n_samples

        # 方差保留率
        X_kept = X_current[keep_mask]
        col_data = X_kept[:, col] if X_kept.ndim > 1 else X_kept
        valid_kept = ~np.isnan(col_data)

        if valid_kept.sum() < 2:
            return sample_retention, 1.0

        result_var = np.var(col_data[valid_kept])
        original_var = self.col_stats.get(col, {}).get('var', 1.0)

        if original_var > 1e-6:
            var_retention = np.clip(result_var / original_var, 0, 1.5)
        else:
            var_retention = 1.0

        return sample_retention, var_retention

    def get_nearby_value(self, X: np.ndarray, idx: int, col: int) -> float:
        """
        获取临近值

        Args:
            X: 数据矩阵
            idx: 行索引
            col: 列索引

        Returns:
            临近值
        """
        current_val = X[idx, col] if X.ndim > 1 else X[idx]
        col_stats = self.col_stats.get(col, {})

        if np.isnan(current_val):
            return col_stats.get('mean', 0.0)

        # 获取该列所有非 NaN 值
        col_vals = X[:, col] if X.ndim > 1 else X
        valid_vals = col_vals[~np.isnan(col_vals)]

        if len(valid_vals) == 0:
            return current_val

        # 找距离最近的不同值
        distances = np.abs(valid_vals - current_val)
        mask = distances > 0.01
        if mask.sum() > 0:
            min_idx = np.argmin(distances[mask])
            return float(valid_vals[mask][min_idx])

        return current_val

    # Setter methods for use by environments
    def set_model_adapter(self, model_adapter: 'ModelAdapter') -> None:
        """设置模型适配器"""
        self.model_adapter = model_adapter

    def set_feature_importance(self, importance: np.ndarray) -> None:
        """设置特征重要性"""
        self.feature_importance = importance

    def set_col_error_rate(self, error_rate: np.ndarray) -> None:
        """设置列错误率"""
        self.col_error_rate = error_rate

    def set_col_stats(self,
                      col_means: np.ndarray,
                      col_stds: np.ndarray,
                      col_vars: np.ndarray) -> None:
        """设置列统计量"""
        n_cols = len(col_means)
        self._n_features = n_cols

        for col in range(n_cols):
            self.col_stats[col] = {
                'mean': float(col_means[col]) if not np.isnan(col_means[col]) else 0.0,
                'std': float(col_stds[col]) if not np.isnan(col_stds[col]) else 1.0,
                'var': float(col_vars[col]) if not np.isnan(col_vars[col]) else 1.0,
            }
