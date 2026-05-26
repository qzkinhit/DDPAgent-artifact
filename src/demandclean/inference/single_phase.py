"""
单阶段推理
==========

直接使用训练好的 Agent 进行数据清洗。
需要提供真值用于修复。
"""

import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import BaseAgent
from ..core.environments import CleaningEnv
from ..core.state import (
    StateExtractor, ClassificationStateExtractor,
    RegressionStateExtractor, ClusteringStateExtractor,
)
from ..models import ModelAdapter, create_model_adapter

# Agent类型 → 算法名映射
_AGENT_ALGO_NAME = {
    AgentType.SINGLE_STAGE: 'DQN (Single Stage)',
    AgentType.DUELING_SINGLE_STAGE: 'Dueling DQN (Single Stage)',
    AgentType.TWO_STAGE: 'Double DQN (Two Stage)',
    AgentType.DUELING_TWO_STAGE: 'Dueling Double DQN (Two Stage)',
}


class SinglePhaseInference:
    """
    单阶段推理

    直接使用训练好的 Agent 对数据进行清洗。
    需要提供干净数据用于获取真值修复。
    """

    def __init__(self,
                 agent: BaseAgent,
                 config: DemandCleanConfig):
        """
        初始化推理器

        Args:
            agent: 训练好的 Agent
            config: 配置对象
        """
        self.agent = agent
        self.config = config

        # 创建模型适配器
        self.model_adapter = create_model_adapter(config.model_type, config.task_type)

        # 创建状态提取器
        self.state_extractor = self._create_state_extractor()

        # 推理后的环境引用（用于获取 decision_log）
        self._env: Optional[CleaningEnv] = None

    def _create_state_extractor(self) -> StateExtractor:
        """创建状态提取器"""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def clean(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              X_clean: np.ndarray,
              detected_errors: Dict[str, List],
              verbose: bool = True,
              y_clean: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[Dict]]:
        """
        使用检测到的错误进行数据清洗

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            X_clean: 干净数据（用于获取真值修复）
            detected_errors: 检测到的错误
            verbose: 是否打印详细信息
            y_clean: 干净标签向量（用于标签噪声修复）

        Returns:
            (X_clean_result, y_clean_result, keep_mask, action_counts, repair_log)
        """
        # 构建错误列表（用真值修复）
        error_list = self._build_error_list(detected_errors, X_clean, y_clean)

        n_missing = len(detected_errors.get('missing', []))
        n_semantic = len(detected_errors.get('semantic', []))
        n_syntactic = len(detected_errors.get('syntactic', []))
        n_label = len(detected_errors.get('label_noise', []))
        total_errors = len(error_list)

        if verbose:
            algo_name = _AGENT_ALGO_NAME.get(self.config.agent_type, self.config.agent_type.value)
            print(f"\n{'='*60}")
            print(f"单阶段推理")
            print(f"{'='*60}")
            print(f"  算法: {algo_name}")
            print(f"  任务类型: {self.config.task_type.value}")
            print(f"  下游模型: {self.config.model_type.value}")
            print(f"  检测到的错误: {total_errors} 个"
                  f" (missing={n_missing}, semantic={n_semantic},"
                  f" syntactic={n_syntactic}, label={n_label})")

        # 创建环境
        env = CleaningEnv(
            X_dirty, y, error_list,
            self.model_adapter, self.state_extractor, self.config
        )
        self._env = env

        # 设置为推理模式
        self.agent.epsilon = 0
        state = env.reset()

        # 进度条参数
        progress_total = 20
        progress_step = max(1, total_errors // progress_total)
        processed = 0

        if verbose:
            sys.stdout.write(f"\n  推理进度: [")
            sys.stdout.flush()

        # 推理
        while True:
            if self.config.agent_type in (AgentType.TWO_STAGE, AgentType.DUELING_TWO_STAGE):
                final_action, _, _ = self.agent.act(state, training=False)
            else:
                final_action = self.agent.act(state, training=False)

            next_state, _, done, _ = env.step(final_action)
            state = next_state
            processed += 1

            # 更新进度条
            if verbose and processed % progress_step == 0:
                sys.stdout.write("=")
                sys.stdout.flush()

            if done:
                break

        if verbose:
            # 补齐进度条
            bars_printed = processed // progress_step
            remaining = progress_total - bars_printed
            sys.stdout.write("=" * remaining + f"] {processed}/{total_errors}\n")
            sys.stdout.flush()

        X_result, y_result, keep_mask = env.get_cleaned_data()
        action_counts = env.get_action_counts()
        repair_log = env.get_repair_log()

        if verbose:
            env.print_decision_summary()

        return X_result, y_result, keep_mask, action_counts, repair_log

    def get_decision_log(self) -> List[Dict]:
        """获取推理后的完整决策日志"""
        if self._env is None:
            return []
        return self._env.get_decision_log()

    def _build_error_list(self,
                          detected_errors: Dict[str, List],
                          X_clean: np.ndarray,
                          y_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """将检测到的错误转换为环境需要的格式"""
        error_list = []

        # Missing errors (type=0)
        for item in detected_errors.get('missing', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': true_val
            })

        # Semantic errors (type=1)
        for item in detected_errors.get('semantic', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': true_val
            })

        # Syntactic errors (type=2)
        for item in detected_errors.get('syntactic', []):
            idx, col = item[0], item[1]
            true_val = X_clean[idx, col]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': true_val
            })

        # Label noise errors (type=3, col=-1)
        for item in detected_errors.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                # 优先使用 y_clean 获取真值
                if y_clean is not None and idx < len(y_clean):
                    repair_value = y_clean[idx]
                elif len(item) > 2:
                    repair_value = item[2]  # estimated_val
                else:
                    repair_value = float('nan')
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': repair_value
                })

        return error_list

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        return {
            'agent_type': self.config.agent_type.value,
            'task_type': self.config.task_type.value,
            'model_type': self.config.model_type.value
        }
