"""
两阶段 DQN Agent (PyTorch)
===========================

分两个阶段决策的 DQN Agent。
"""

from typing import Tuple, Optional, Any, Dict
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from .single_stage_agent import _PlainQNetwork


class TwoStageDQNAgent(BaseAgent):
    """
    两阶段 DQN Agent (PyTorch)

    Stage 1: 决定策略类型 (3 种动作)
        - 0: no_action (不操作)
        - 1: repair (修复)
        - 2: delete (删除)

    Stage 2: 如果选择 repair，决定如何修 (2 种动作)
        - 0: repair_with_value (用真值修复)
        - 1: replace_nearby (用临近值替换)

    最终动作映射:
        - 0: no_action
        - 1: repair_value (stage1=1, stage2=0)
        - 2: delete
        - 3: replace_nearby (stage1=1, stage2=1)
    """

    def __init__(self,
                 state_size: int = 8,
                 memory_size: int = 10000,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.0005):
        """
        初始化两阶段 DQN Agent

        Args:
            state_size: 状态向量维度
            memory_size: 经验回放缓冲区大小
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            learning_rate: 学习率
        """
        super().__init__(state_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device('cpu')

        # Stage 1: 策略选择网络
        self.stage1_action_size = 3
        self.stage1_memory = deque(maxlen=memory_size)
        self.stage1_model = _PlainQNetwork(state_size, self.stage1_action_size).to(self.device)
        self.stage1_target_model = _PlainQNetwork(state_size, self.stage1_action_size).to(self.device)
        self.stage1_optimizer = optim.Adam(self.stage1_model.parameters(), lr=learning_rate)

        # Stage 2: 修复方式网络
        self.stage2_action_size = 2
        self.stage2_memory = deque(maxlen=memory_size)
        self.stage2_model = _PlainQNetwork(state_size, self.stage2_action_size).to(self.device)
        self.stage2_target_model = _PlainQNetwork(state_size, self.stage2_action_size).to(self.device)
        self.stage2_optimizer = optim.Adam(self.stage2_model.parameters(), lr=learning_rate)

        self.update_target_models()

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """numpy → torch tensor"""
        return torch.FloatTensor(arr).to(self.device)

    def act_stage1(self, state: np.ndarray, training: bool = True) -> int:
        """Stage 1 动作选择"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.stage1_action_size)
        with torch.no_grad():
            q = self.stage1_model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def act_stage2(self, state: np.ndarray, training: bool = True) -> int:
        """Stage 2 动作选择"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.stage2_action_size)
        with torch.no_grad():
            q = self.stage2_model(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, int, Optional[int]]:
        """
        根据状态选择动作

        Args:
            state: 状态向量
            training: 是否在训练模式

        Returns:
            (final_action, stage1_action, stage2_action)
            - final_action: 最终动作 (0=no_action, 1=repair_value, 2=delete, 3=replace_nearby)
            - stage1_action: Stage 1 动作 (0, 1, 2)
            - stage2_action: Stage 2 动作 (0, 1) 或 None
        """
        stage1_action = self.act_stage1(state, training)

        if stage1_action == 0:  # no_action
            return 0, stage1_action, None
        elif stage1_action == 2:  # delete
            return 2, stage1_action, None
        else:  # repair
            stage2_action = self.act_stage2(state, training)
            if stage2_action == 0:  # repair_with_value
                return 1, stage1_action, stage2_action
            else:  # replace_nearby
                return 3, stage1_action, stage2_action

    def remember_stage1(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储 Stage 1 经验"""
        self.stage1_memory.append((state, action, reward, next_state, done))

    def remember_stage2(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储 Stage 2 经验"""
        self.stage2_memory.append((state, action, reward, next_state, done))

    def remember(self, state: np.ndarray, action: int,
                 reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验（兼容基类接口，存储到 Stage 1）"""
        self.remember_stage1(state, action, reward, next_state, done)

    def _replay_stage(self, memory: deque, model: nn.Module,
                      target_model: nn.Module, optimizer: optim.Optimizer,
                      batch_size: int) -> None:
        """单阶段经验回放 (Double DQN)"""
        if len(memory) < batch_size:
            return

        minibatch = random.sample(memory, batch_size)
        states = self._to_tensor(np.array([x[0] for x in minibatch]))
        actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
        rewards = self._to_tensor(np.array([x[2] for x in minibatch]))
        next_states = self._to_tensor(np.array([x[3] for x in minibatch]))
        dones = self._to_tensor(np.array([x[4] for x in minibatch], dtype=np.float32))

        # Double DQN
        current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = model(next_states).argmax(dim=1)
            next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def replay(self, batch_size: int = 64) -> None:
        """经验回放训练两个阶段"""
        self._replay_stage(self.stage1_memory, self.stage1_model,
                           self.stage1_target_model, self.stage1_optimizer, batch_size)
        self._replay_stage(self.stage2_memory, self.stage2_model,
                           self.stage2_target_model, self.stage2_optimizer, batch_size)

        # 衰减探索率
        self.decay_epsilon()

    def update_target_models(self) -> None:
        """更新两个阶段的目标网络（硬更新）"""
        self.stage1_target_model.load_state_dict(self.stage1_model.state_dict())
        self.stage2_target_model.load_state_dict(self.stage2_model.state_dict())

    def update_target_model(self) -> None:
        """兼容基类接口"""
        self.update_target_models()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取两阶段 Q 值拼接 (stage1_action_size + stage2_action_size,)"""
        with torch.no_grad():
            st = self._to_tensor(state).unsqueeze(0)
            q1 = self.stage1_model(st).cpu().numpy().flatten()
            q2 = self.stage2_model(st).cpu().numpy().flatten()
            return np.concatenate([q1, q2])

    def save(self, path: str) -> None:
        """保存两个阶段的模型 (.pt 格式)，含续训元数据"""
        base = path.replace('.h5', '').replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        stage2_path = base + '_stage2.pt'

        torch.save({
            'model_state': self.stage1_model.state_dict(),
            'target_state': self.stage1_target_model.state_dict(),
            'optimizer_state': self.stage1_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
        }, stage1_path)

        torch.save({
            'model_state': self.stage2_model.state_dict(),
            'target_state': self.stage2_target_model.state_dict(),
            'optimizer_state': self.stage2_optimizer.state_dict(),
        }, stage2_path)

    def load(self, path: str) -> None:
        """加载两个阶段的模型（向后兼容旧 checkpoint）"""
        base = path.replace('.h5', '').replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        stage2_path = base + '_stage2.pt'

        ckpt1 = torch.load(stage1_path, map_location=self.device, weights_only=False)
        self.stage1_model.load_state_dict(ckpt1['model_state'])
        self.stage1_target_model.load_state_dict(ckpt1.get('target_state', ckpt1['model_state']))
        if 'optimizer_state' in ckpt1:
            self.stage1_optimizer.load_state_dict(ckpt1['optimizer_state'])
        self.epsilon = ckpt1.get('epsilon', self.epsilon)
        self.total_episodes = ckpt1.get('total_episodes', 0)
        self.best_score = ckpt1.get('best_score', -float('inf'))
        self.best_episode = ckpt1.get('best_episode', 0)

        ckpt2 = torch.load(stage2_path, map_location=self.device, weights_only=False)
        self.stage2_model.load_state_dict(ckpt2['model_state'])
        self.stage2_target_model.load_state_dict(ckpt2.get('target_state', ckpt2['model_state']))
        if 'optimizer_state' in ckpt2:
            self.stage2_optimizer.load_state_dict(ckpt2['optimizer_state'])

    def get_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """获取两个阶段的模型权重"""
        s1 = {k: v.clone() for k, v in self.stage1_model.state_dict().items()}
        s2 = {k: v.clone() for k, v in self.stage2_model.state_dict().items()}
        return (s1, s2)

    def set_weights(self, weights: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        """设置两个阶段的模型权重"""
        stage1_weights, stage2_weights = weights
        self.stage1_model.load_state_dict(stage1_weights)
        self.stage2_model.load_state_dict(stage2_weights)
        self.update_target_models()
