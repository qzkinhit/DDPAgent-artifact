"""
两阶段推理
==========

第一阶段：生成修复计划，不需要真值
第二阶段：用户提供真值后执行修复
"""

import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..config import DemandCleanConfig, TaskType, AgentType
from ..core.agents import BaseAgent
from ..core.environments import TwoPhaseCleaningEnv
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


class TwoPhaseInference:
    """
    两阶段推理

    第一阶段 (plan): 预测所有动作，repair_value 只加入 plan，其他动作立即执行
    第二阶段 (execute): 用户提供真值后，执行 plan 中的修复
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

        # 两阶段环境
        self._env: Optional[TwoPhaseCleaningEnv] = None
        self._repair_plan: List[Dict] = []

    def _create_state_extractor(self) -> StateExtractor:
        """创建状态提取器"""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def plan(self,
             X_dirty: np.ndarray,
             y: np.ndarray,
             detected_errors: Dict[str, List],
             verbose: bool = True,
             save_csv_path: Optional[str] = None) -> List[Dict]:
        """
        第一阶段：生成修复计划

        不需要真值，返回需要修复的位置列表。

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            detected_errors: 检测到的错误
            verbose: 是否打印详细信息
            save_csv_path: 可选，自动保存修复计划CSV路径

        Returns:
            repair_plan: 需要真值修复的位置列表
                [{'idx', 'col', 'error_type', 'estimated_value', 'current_dirty_value'}, ...]
        """
        # 构建错误列表（不需要真值）
        error_list = self._build_error_list_no_truth(detected_errors)

        n_missing = len(detected_errors.get('missing', []))
        n_semantic = len(detected_errors.get('semantic', []))
        n_syntactic = len(detected_errors.get('syntactic', []))
        n_label = len(detected_errors.get('label_noise', []))
        total_errors = len(error_list)

        if verbose:
            algo_name = _AGENT_ALGO_NAME.get(self.config.agent_type, self.config.agent_type.value)
            print(f"\n{'='*60}")
            print(f"两阶段推理 - 第一阶段 (Plan)")
            print(f"{'='*60}")
            print(f"  算法: {algo_name}")
            print(f"  任务类型: {self.config.task_type.value}")
            print(f"  下游模型: {self.config.model_type.value}")
            print(f"  检测到的错误: {total_errors} 个"
                  f" (missing={n_missing}, semantic={n_semantic},"
                  f" syntactic={n_syntactic}, label={n_label})")

        # 创建两阶段环境
        self._env = TwoPhaseCleaningEnv(
            X_dirty, y, error_list,
            self.model_adapter, self.state_extractor, self.config
        )

        # 设置为推理模式
        self.agent.epsilon = 0
        state = self._env.reset()

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

            next_state, _, done, _ = self._env.step(final_action)
            state = next_state
            processed += 1

            # 更新进度条
            if verbose and processed % progress_step == 0:
                sys.stdout.write("=")
                sys.stdout.flush()

            if done:
                break

        if verbose:
            bars_printed = processed // progress_step
            remaining = progress_total - bars_printed
            sys.stdout.write("=" * remaining + f"] {processed}/{total_errors}\n")
            sys.stdout.flush()

        self._repair_plan = self._env.get_repair_plan()

        if verbose:
            self._env.print_decision_summary()
            print(f"\n  需要用户提供 {len(self._repair_plan)} 个真值")

        # 自动保存计划CSV
        if save_csv_path:
            self._env.save_plan_csv(save_csv_path)
            if verbose:
                print(f"  修复计划已保存: {save_csv_path}")

        return self._repair_plan

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """
        获取需要真值的位置列表

        Returns:
            [(idx, col), ...] 需要用户提供真值的位置
        """
        if self._env is None:
            return []
        return self._env.get_plan_positions()

    def execute(self,
                X_dirty: np.ndarray,
                true_values: Dict[Tuple[int, int], float],
                verbose: bool = True,
                y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        第二阶段：执行修复

        Args:
            X_dirty: 原始脏数据
            true_values: 真值字典 {(idx, col): value}
            verbose: 是否打印详细信息
            y_dirty: 原始脏标签（标签修复时需要）

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        if self._env is None:
            raise ValueError("请先调用 plan() 方法生成修复计划")

        if verbose:
            print(f"\n{'='*60}")
            print(f"两阶段推理 - 第二阶段 (Execute)")
            print(f"{'='*60}")
            print(f"  提供的真值数量: {len(true_values)}")
            print(f"  计划修复数量: {len(self._repair_plan)}")

        # 获取 keep_mask 和标签（使用 get_cleaned_data 确保一致性）
        _, y_from_env, keep_mask = self._env.get_cleaned_data()

        # 执行修复（传入 y_dirty 以支持标签修复）
        X_result, y_result = self._env.execute_repair_plan(
            X_dirty, true_values, y_dirty=y_dirty
        )

        # 如果没有传入 y_dirty，使用环境中的 y 结果
        if y_result is None:
            y_result = y_from_env

        if verbose:
            matched = sum(1 for item in self._repair_plan
                          if (item['idx'], item['col']) in true_values)
            print(f"\n  执行结果:")
            print(f"    成功匹配真值: {matched} / {len(self._repair_plan)}")
            print(f"    删除行数: {int((~keep_mask).sum())}")
            print(f"    最终数据行数: {int(keep_mask.sum())}")

        return X_result, y_result, keep_mask

    def clean_with_reference(self,
                             X_dirty: np.ndarray,
                             y: np.ndarray,
                             X_clean: np.ndarray,
                             detected_errors: Dict[str, List],
                             verbose: bool = True,
                             y_clean: Optional[np.ndarray] = None,
                             save_csv_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[Dict]]:
        """
        使用参考数据进行两阶段清洗

        这是一个便捷方法，自动从 X_clean/y_clean 提取真值。

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            X_clean: 干净数据（用于获取真值）
            detected_errors: 检测到的错误
            verbose: 是否打印详细信息
            y_clean: 干净标签向量（用于标签噪声修复）
            save_csv_path: 可选，自动保存修复计划CSV路径

        Returns:
            (X_clean_result, y_clean_result, keep_mask, action_counts, repair_plan)
        """
        # 第一阶段
        repair_plan = self.plan(X_dirty, y, detected_errors, verbose,
                                save_csv_path=save_csv_path)

        # 从 X_clean / y_clean 提取真值
        true_values = {}
        for item in repair_plan:
            idx, col = item['idx'], item['col']
            if col == -1:
                # 标签噪声：从 y_clean 获取真值
                if y_clean is not None and idx < len(y_clean):
                    true_values[(idx, col)] = y_clean[idx]
            else:
                true_values[(idx, col)] = X_clean[idx, col]

        # 第二阶段
        X_result, y_result, keep_mask = self.execute(
            X_dirty, true_values, verbose, y_dirty=y
        )

        action_counts = self._env.get_action_counts() if self._env else {}

        return X_result, y_result, keep_mask, action_counts, repair_plan

    def _build_error_list_no_truth(self,
                                   detected_errors: Dict[str, List]) -> List[Dict]:
        """将检测到的错误转换为环境需要的格式（不需要真值）"""
        error_list = []

        # Missing errors (type=0)
        for item in detected_errors.get('missing', []):
            idx, col = item[0], item[1]
            estimated_val = item[2] if len(item) > 2 else 0
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': None  # 不需要真值
            })

        # Semantic errors (type=1)
        for item in detected_errors.get('semantic', []):
            idx, col = item[0], item[1]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': None
            })

        # Syntactic errors (type=2)
        for item in detected_errors.get('syntactic', []):
            idx, col = item[0], item[1]
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': None
            })

        # Label noise errors (type=3, col=-1)
        for item in detected_errors.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': None
                })

        return error_list

    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        stats = {
            'agent_type': self.config.agent_type.value,
            'task_type': self.config.task_type.value,
            'model_type': self.config.model_type.value,
            'plan_size': len(self._repair_plan)
        }

        if self._env:
            stats['action_counts'] = self._env.get_action_counts()

        return stats

    def get_decision_log(self) -> List[Dict]:
        """获取推理后的完整决策日志"""
        if self._env is None:
            return []
        return self._env.get_decision_log()
