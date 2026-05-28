"""
Ridge 回归适配器
================
"""

import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional

from ..base_adapter import ModelAdapter


class RidgeAdapter(ModelAdapter):
    """
    Ridge 回归适配器

    带 L2 正则化的线性回归。
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        初始化 Ridge 回归适配器

        Args:
            alpha: 正则化强度
            **kwargs: 传递给 Ridge 的其他参数
        """
        super().__init__()
        self.alpha = alpha
        self.kwargs = kwargs
        # 使用 'lsqr' solver 避免 scipy 版本兼容性问题
        self.model = Ridge(alpha=alpha, solver='lsqr', **kwargs)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeAdapter':
        """训练 Ridge 回归模型"""
        self.model.fit(X, y)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-6
        self._is_fitted = True

        # 特征重要性基于系数绝对值
        self._feature_importance = self._normalize_importance(self.model.coef_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self._is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算负 MSE

        返回负 MSE，越接近 0 越好
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return -mse

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        获取到"边界"的距离

        回归任务中，使用预测值偏离均值的程度作为"距离"
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            predictions = self.predict(X)
            influence = np.abs(predictions - self._y_mean) / (self._y_std * 2)
            return np.clip(influence, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self._feature_importance is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._feature_importance

    def clone(self) -> 'RidgeAdapter':
        """创建未训练的克隆"""
        return RidgeAdapter(alpha=self.alpha, **self.kwargs)
