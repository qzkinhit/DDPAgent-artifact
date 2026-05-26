"""
评估指标
========

分类和回归任务的评估指标计算。
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class Metrics:
    """评估指标计算工具类"""

    @staticmethod
    def classification_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算分类准确率

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            准确率 [0, 1]
        """
        return np.mean(y_true == y_pred)

    @staticmethod
    def regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算回归得分（负 MSE）

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            负 MSE（越接近 0 越好）
        """
        mse = np.mean((y_true - y_pred) ** 2)
        return -mse

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算 R2 决定系数

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            R2 值
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 1.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def authenticity(X_result: np.ndarray,
                     X_clean: np.ndarray,
                     keep_mask: Optional[np.ndarray] = None,
                     col: int = 1,
                     tolerance: float = 0.01) -> float:
        """
        计算数据真实性

        真实性 = 正确值数量 / 当前行数

        Args:
            X_result: 清洗后的数据
            X_clean: 干净数据
            keep_mask: 保留行的掩码
            col: 比较的列索引
            tolerance: 容差阈值

        Returns:
            真实性 [0, 1]
        """
        if len(X_result) == 0:
            return 0.0

        if keep_mask is not None:
            X_clean_kept = X_clean[keep_mask]
        else:
            X_clean_kept = X_clean

        n = min(len(X_result), len(X_clean_kept))
        correct = 0
        for i in range(n):
            if abs(X_result[i, col] - X_clean_kept[i, col]) < tolerance:
                correct += 1

        return correct / len(X_result)

    @staticmethod
    def diversity(X_result: np.ndarray,
                  X_clean: np.ndarray,
                  X_dirty: np.ndarray,
                  keep_mask: Optional[np.ndarray] = None,
                  col: int = 1) -> Tuple[float, float, float]:
        """
        计算数据多样性

        多样性 = 样本保留率 × 方差保留率

        Args:
            X_result: 清洗后的数据
            X_clean: 干净数据
            X_dirty: 脏数据
            keep_mask: 保留行的掩码
            col: 计算方差的列索引

        Returns:
            (diversity, sample_retention, var_retention)
        """
        n_total = len(X_clean)
        n_result = len(X_result)

        if n_result == 0:
            return 0.0, 0.0, 0.0

        # 样本保留率
        sample_retention = n_result / n_total

        # 方差保留率
        if n_result < 2:
            return 0.0, sample_retention, 0.0

        if keep_mask is not None:
            X_clean_kept = X_clean[keep_mask]
        else:
            X_clean_kept = X_clean[:n_result]

        result_var = np.var(X_result[:, col])
        clean_var = np.var(X_clean_kept[:, col])

        if clean_var > 1e-6:
            var_retention = np.clip(result_var / clean_var, 0, 1.5)
        else:
            var_retention = 1.0

        diversity = sample_retention * var_retention

        return diversity, sample_retention, var_retention

    @staticmethod
    def compute_all(X_result: np.ndarray,
                    y_result: np.ndarray,
                    X_clean: np.ndarray,
                    y_clean: np.ndarray,
                    X_dirty: np.ndarray,
                    keep_mask: Optional[np.ndarray] = None,
                    col: int = 1) -> Dict[str, Any]:
        """
        计算所有指标

        Args:
            X_result: 清洗后的特征
            y_result: 清洗后的标签
            X_clean: 干净特征
            y_clean: 干净标签
            X_dirty: 脏特征
            keep_mask: 保留行的掩码
            col: 计算指标的列索引

        Returns:
            包含所有指标的字典
        """
        # 真实性
        auth = Metrics.authenticity(X_result, X_clean, keep_mask, col)

        # 多样性
        div, sample_ret, var_ret = Metrics.diversity(
            X_result, X_clean, X_dirty, keep_mask, col
        )

        return {
            'authenticity': auth,
            'diversity': div,
            'sample_retention': sample_ret,
            'var_retention': var_ret,
            'n_samples': len(X_result),
            'n_original': len(X_clean),
        }

    @staticmethod
    def action_distribution(action_counts: Dict[str, int]) -> Dict[str, float]:
        """
        计算动作分布百分比

        Args:
            action_counts: 动作计数字典

        Returns:
            动作百分比字典
        """
        total = sum(action_counts.values())
        if total == 0:
            return {k: 0.0 for k in action_counts}

        return {k: v / total for k, v in action_counts.items()}
