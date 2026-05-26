"""
Dueling Double DQN 单阶段 Agent (PyTorch)
==========================================

使用 Dueling 网络 + Double DQN + 软更新的单阶段 Agent。
"""

from typing import Any, Dict
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from .dueling_network import DuelingNetwork


class DuelingSingleStageAgent(BaseAgent):
    """
    Dueling Double DQN 单阶段 Agent

    直接输出 4 种动作:
        0: no_action
        1: repair_value
        2: delete
        3: replace_nearby

    相比普通 DQN:
        - 使用 Dueling 网络分离 V(s) 和 A(s,a)
        - 软更新目标网络 (tau)
        - 更大的隐藏层 (128)
    """

    def __init__(self,
                 state_size: int = 8,
                 action_size: int = 4,
                 memory_size: int = 5000,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.0005,
                 tau: float = 0.1,
                 hidden_size: int = 128):
        super().__init__(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.device = torch.device('cpu')

        # 构建网络
        self.model = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_model = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 初始同步目标网络
        self.target_model.load_state_dict(self.model.state_dict())

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(self.device)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 64) -> None:
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = self._to_tensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = self._to_tensor(np.array([x[2] for x in batch]))
        next_states = self._to_tensor(np.array([x[3] for x in batch]))
        dones = self._to_tensor(np.array([x[4] for x in batch], dtype=np.float32))

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

        self.decay_epsilon()

    def update_target_model(self) -> None:
        """软更新目标网络: target = tau * model + (1-tau) * target"""
        for t_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取状态对应的 Q 值 (action_size,)"""
        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
            return q.cpu().numpy().flatten()

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
        }, path)

    def load(self, path: str) -> None:
        """加载模型（向后兼容旧 checkpoint）"""
        path = path.replace('.h5', '.pt')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.target_model.load_state_dict(ckpt.get('target_state', ckpt['model_state']))
        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.best_score = ckpt.get('best_score', -float('inf'))
        self.best_episode = ckpt.get('best_episode', 0)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(self.model.state_dict())
