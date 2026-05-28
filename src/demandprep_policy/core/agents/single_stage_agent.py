"""
单阶段 DQN Agent (PyTorch)
===========================

直接输出 4 种动作的 DQN Agent。
"""

from typing import Optional, Any, Dict
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class _PlainQNetwork(nn.Module):
    """普通全连接 Q 网络: 64→64→32→action_size"""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleStageDQNAgent(BaseAgent):
    """
    单阶段 DQN Agent (PyTorch)

    直接输出 4 种动作:
        - 0: no_action (不操作)
        - 1: repair_value (用真值修复)
        - 2: delete (删除)
        - 3: replace_nearby (用临近值替换)

    使用 Double DQN 技术减少过估计。
    """

    def __init__(self,
                 state_size: int = 8,
                 action_size: int = 4,
                 memory_size: int = 10000,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.0005):
        """
        初始化单阶段 DQN Agent

        Args:
            state_size: 状态向量维度
            action_size: 动作空间大小
            memory_size: 经验回放缓冲区大小
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            learning_rate: 学习率
        """
        super().__init__(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device('cpu')

        # 构建网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def _build_model(self) -> _PlainQNetwork:
        """构建 Q 网络"""
        model = _PlainQNetwork(self.state_size, self.action_size).to(self.device)
        return model

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """numpy → torch tensor"""
        return torch.FloatTensor(arr).to(self.device)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        根据状态选择动作

        Args:
            state: 状态向量 (state_size,)
            training: 是否在训练模式

        Returns:
            动作索引 (0-3)
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 64) -> None:
        """
        经验回放训练 (Double DQN)
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = self._to_tensor(np.array([x[0] for x in minibatch]))
        actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
        rewards = self._to_tensor(np.array([x[2] for x in minibatch]))
        next_states = self._to_tensor(np.array([x[3] for x in minibatch]))
        dones = self._to_tensor(np.array([x[4] for x in minibatch], dtype=np.float32))

        # Double DQN: 主网络选动作，目标网络评估
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        self.decay_epsilon()

    def update_target_model(self) -> None:
        """更新目标网络（硬更新）"""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取状态对应的 Q 值 (action_size,)"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        """保存模型 (.pt 格式)，含续训元数据"""
        path = path.replace('.h5', '.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'target_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, path)

    def load(self, path: str) -> None:
        """加载模型（向后兼容旧 checkpoint）"""
        path = path.replace('.h5', '.pt')
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint.get('target_state', checkpoint['model_state']))
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.best_score = checkpoint.get('best_score', -float('inf'))
        self.best_episode = checkpoint.get('best_episode', 0)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取模型权重 (state_dict 的深拷贝)"""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """设置模型权重"""
        self.model.load_state_dict(weights)
        self.update_target_model()
