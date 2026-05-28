"""
RandomForest 分类器适配器
=========================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

from ..base_adapter import ModelAdapter


class RandomForestAdapter(ModelAdapter):
    """
    随机森林分类器适配器

    支持获取特征重要性和预测概率。
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        初始化 RandomForest 适配器

        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            random_state: 随机种子
            **kwargs: 传递给 RandomForestClassifier 的其他参数
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.extra_kwargs = kwargs.copy()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self._y_classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestAdapter':
        """训练随机森林模型"""
        self.model.fit(X, y)
        self._y_classes = np.unique(y)
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
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        获取到决策边界的距离

        使用预测概率的最大值作为置信度
        置信度高 -> 远离边界 -> 距离大
        置信度低（接近0.5）-> 接近边界 -> 距离小
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            proba = self.model.predict_proba(X)
            # 最大概率作为置信度
            max_proba = np.max(proba, axis=1)
            # 置信度 0.5 -> 0, 置信度 1.0 -> 1
            # (max_proba - 0.5) * 2 将 [0.5, 1] 映射到 [0, 1]
            distances = (max_proba - 0.5) * 2
            return np.clip(distances, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self._feature_importance is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._feature_importance

    def clone(self) -> 'RandomForestAdapter':
        """创建未训练的克隆"""
        return RandomForestAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.extra_kwargs
        )
