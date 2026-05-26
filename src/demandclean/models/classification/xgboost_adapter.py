"""
XGBoost 分类器适配器
====================
"""

import numpy as np
from typing import Optional
import warnings

from ..base_adapter import ModelAdapter


class XGBoostClassifierAdapter(ModelAdapter):
    """
    XGBoost 分类器适配器

    支持梯度提升分类。
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        初始化 XGBoost 适配器

        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
            random_state: 随机种子
            **kwargs: 传递给 XGBClassifier 的其他参数
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self._y_classes: Optional[np.ndarray] = None

    def _create_model(self):
        """创建 XGBoost 模型"""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                **self.kwargs
            )
        except ImportError:
            warnings.warn("XGBoost 未安装，使用 RandomForest 替代")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifierAdapter':
        """训练 XGBoost 模型"""
        if self.model is None:
            self._create_model()

        self.model.fit(X, y)
        self._y_classes = np.unique(y)
        self._is_fitted = True

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self._feature_importance = self._normalize_importance(
                self.model.feature_importances_
            )
        else:
            self._feature_importance = np.ones(X.shape[1]) / X.shape[1]

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

        使用预测概率
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            proba = self.model.predict_proba(X)
            max_proba = np.max(proba, axis=1)
            distances = (max_proba - 0.5) * 2
            return np.clip(distances, 0, 1)
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self._feature_importance is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._feature_importance

    def clone(self) -> 'XGBoostClassifierAdapter':
        """创建未训练的克隆"""
        return XGBoostClassifierAdapter(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
