"""
RandomForest 回归适配器
========================
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional

from ..base_adapter import ModelAdapter


class RandomForestRegressorAdapter(ModelAdapter):
    """
    随机森林回归适配器

    支持获取特征重要性和回归预测。
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        初始化 RandomForest 回归适配器

        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            random_state: 随机种子
            **kwargs: 传递给 RandomForestRegressor 的其他参数
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.extra_kwargs = kwargs.copy()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressorAdapter':
        """训练随机森林回归模型"""
        self.model.fit(X, y)
        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-6
        self._is_fitted = True

        # 特征重要性
        self._feature_importance = self._normalize_importance(
            self.model.feature_importances_
        )

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

    def clone(self) -> 'RandomForestRegressorAdapter':
        """创建未训练的克隆"""
        return RandomForestRegressorAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.extra_kwargs
        )
