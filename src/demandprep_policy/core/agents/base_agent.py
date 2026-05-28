"""
DQN Agent 基类
==============

定义 DQN Agent 的基本接口。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np


class BaseAgent(ABC):
    """
    DQN Agent 抽象基类

    定义所有 Agent 必须实现的接口。
    """

    def __init__(self, state_size: int = 8):
        """
        初始化 Agent

        Args:
            state_size: 状态向量维度
        """
        self.state_size = state_size
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.learning_rate = 0.0005

        # 续训相关元数据
        self.total_episodes: int = 0
        self.best_score: float = -float('inf')
        self.best_episode: int = 0

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Any:
        """
        根据状态选择动作

        Args:
            state: 状态向量
            training: 是否在训练模式（影响探索）

        Returns:
            选择的动作
        """
        pass

    @abstractmethod
    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        存储经验到回放缓冲区

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        pass

    @abstractmethod
    def replay(self, batch_size: int = 64) -> None:
        """
        经验回放训练

        Args:
            batch_size: 批大小
        """
        pass

    @abstractmethod
    def update_target_model(self) -> None:
        """更新目标网络"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        pass

    def decay_epsilon(self) -> None:
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取状态对应的 Q 值（不选择动作）

        Args:
            state: 状态向量 (state_size,)

        Returns:
            Q 值数组，shape 取决于 Agent 类型:
            - 单阶段: (action_size,)
            - 两阶段: (stage1_action_size + stage2_action_size,) 拼接
        """
        raise NotImplementedError

    def get_weights(self) -> Any:
        """获取模型权重 (PyTorch state_dict 的深拷贝)"""
        pass

    def set_weights(self, weights: Any) -> None:
        """设置模型权重 (PyTorch state_dict)"""
        pass
