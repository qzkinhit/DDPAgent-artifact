"""
Dueling DQN 网络
================

共享特征层 + Value/Advantage 双头结构。
Q(s,a) = V(s) + A(s,a) - mean(A)
"""

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """
    Dueling DQN 网络

    结构:
        共享层: state_size → hidden → hidden (ReLU)
        Value 头: hidden → 1
        Advantage 头: hidden → action_size
        输出: Q = V + A - mean(A)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value 头: 估计状态价值 V(s)
        self.value_head = nn.Linear(hidden_size, 1)

        # Advantage 头: 估计动作优势 A(s,a)
        self.advantage_head = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        value = self.value_head(features)                   # (batch, 1)
        advantage = self.advantage_head(features)           # (batch, action_size)
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
