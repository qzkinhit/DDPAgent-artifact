"""
模型适配器基类
==============

定义模型适配器的统一接口，用于支持多种机器学习模型。
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ModelAdapter(ABC):
    """
    模型适配器抽象基类

    统一分类和回归模型的接口，使得 DQN 环境可以与不同模型交互。

    核心方法:
        - fit: 训练模型
        - predict: 预测
        - evaluate: 评估模型性能
        - get_distance_to_boundary: 获取到决策边界的距离（核心！）
        - get_feature_importance: 获取特征重要性
    """

    def __init__(self):
        self.model = None
        self._feature_importance: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelAdapter':
        """
        训练模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测结果 (n_samples,)
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型性能

        Args:
            X: 特征矩阵
            y: 真实标签

        Returns:
            性能得分（分类返回准确率，回归返回负MSE）
        """
        pass

    @abstractmethod
    def get_distance_to_boundary(self, X: np.ndarray) -> np.ndarray:
        """
        获取到决策边界的距离

        这是 DQN 状态特征的关键组成部分。

        分类任务: 使用 decision_function 或预测概率
        回归任务: 使用预测值与整体均值的偏差

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            距离数组 (n_samples,)，归一化到 [0, 1]
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性

        Returns:
            特征重要性数组，归一化后各元素和为 1
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        return self._is_fitted

    def _normalize_importance(self, importance: np.ndarray) -> np.ndarray:
        """归一化特征重要性"""
        total = np.sum(np.abs(importance))
        if total < 1e-10:
            return np.ones_like(importance) / len(importance)
        return np.abs(importance) / total

    def clone(self) -> 'ModelAdapter':
        """
        创建一个未训练的克隆

        Returns:
            新的 ModelAdapter 实例
        """
        return type(self)()
