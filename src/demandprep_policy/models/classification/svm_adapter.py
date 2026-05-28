"""
SVM 分类器适配器
================
"""

import numpy as np
from sklearn.svm import SVC
from typing import Optional

from ..base_adapter import ModelAdapter


class SVMAdapter(ModelAdapter):
    """
    SVM 分类器适配器

    支持多种核函数，默认使用线性核。
    """

    def __init__(self, kernel: str = 'linear', C: float = 1.0,
                 max_iter: int = -1, **kwargs):
        """
        初始化 SVM 适配器

        Args:
            kernel: 核函数类型 ('linear', 'rbf', 'poly')
            C: 正则化参数
            max_iter: SMO 求解器最大迭代次数
                      -1 表示自适应：min(50000, max(10000, n_samples * 20))
                      正数表示固定上限
            **kwargs: 传递给 SVC 的其他参数
        """
        super().__init__()
        self.kernel = kernel
        self.C = C
        self._max_iter_config = max_iter  # 保存配置值，fit 时动态计算
        self.max_iter = max_iter
        self.model = SVC(kernel=kernel, C=C, max_iter=10000, **kwargs)  # 初始值，fit 时会更新
        self._y_classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMAdapter':
        """训练 SVM 模型"""
        # 自适应 max_iter：根据数据大小调整
        if self._max_iter_config == -1:
            # 自适应公式：小数据 10000，大数据按比例增加，上限 50000
            adaptive_iter = min(50000, max(10000, len(X) * 20))
            self.model.max_iter = adaptive_iter
            self.max_iter = adaptive_iter
        else:
            self.model.max_iter = self._max_iter_config
            self.max_iter = self._max_iter_config

        self.model.fit(X, y)
        self._y_classes = np.unique(y)
        self._is_fitted = True

        # 计算特征重要性
        if self.kernel == 'linear' and hasattr(self.model, 'coef_'):
            self._feature_importance = self._normalize_importance(self.model.coef_[0])
        else:
            # 非线性核，特征重要性均匀分布
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

        使用 decision_function 的绝对值，并通过 sigmoid 归一化到 [0, 1]
        距离越小表示越接近边界（越重要）
        """
        if not self._is_fitted:
            return np.ones(len(X)) * 0.5

        try:
            distances = np.abs(self.model.decision_function(X))
            # Sigmoid 归一化: 1 / (1 + exp(-d + 1))
            # 距离大 -> 值大（远离边界）
            # 距离小 -> 值小（接近边界）
            normalized = 1.0 / (1.0 + np.exp(-distances + 1))
            return normalized
        except Exception:
            return np.ones(len(X)) * 0.5

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self._feature_importance is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._feature_importance

    def clone(self) -> 'SVMAdapter':
        """创建未训练的克隆"""
        return SVMAdapter(kernel=self.kernel, C=self.C, max_iter=self.max_iter)
