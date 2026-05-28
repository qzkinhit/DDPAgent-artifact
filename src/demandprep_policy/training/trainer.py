"""
训练器
======

DQN Agent 训练器，支持两阶段和单阶段训练。

训练模式:
  - clean_base (默认): DeleteFix 策略构建基准数据，注入错误训练 Agent
  - self_supervised: 在脏数据全量上训练，已检测错误type固定，未检测单元格注入额外错误
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Set
import os
import numpy as np

from ..config import DemandPrepPolicyConfig, TaskType, AgentType
from ..core.agents import (
    BaseAgent, SingleStageDQNAgent, TwoStageDQNAgent,
    DuelingSingleStageAgent, DuelingTwoStageAgent,
)
from ..core.environments import CleaningEnv
from ..core.environments.value_estimation import ValueEstimator
from ..core.state import (
    StateExtractor, ClassificationStateExtractor,
    RegressionStateExtractor, ClusteringStateExtractor,
)
from ..models import ModelAdapter, create_model_adapter
from ..detectors import ErrorInjector
from ..detectors.error_injector import LabelErrorPattern, analyze_label_error_pattern
from ..utils.logger import DemandPrepPolicyLogger
from ..utils.model_io import ModelIO


class Trainer:
    """
    DQN Agent 训练器

    支持两种训练模式:
    - clean_base: 在去NaN的脏数据上自注入错误学习策略
    - self_supervised: 在脏数据全量上训练，融合已检测错误和新注入错误
    """

    def __init__(self, config: DemandPrepPolicyConfig):
        """
        初始化训练器

        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = DemandPrepPolicyLogger(config)
        self.model_io = ModelIO()

        # 创建模型适配器
        model_kwargs = config.model_kwargs or {}
        self.model_adapter = create_model_adapter(config.model_type, config.task_type, **model_kwargs)

        # 创建状态提取器
        self.state_extractor = self._create_state_extractor()

        # Agent
        self.agent: Optional[BaseAgent] = None

        # 训练历史
        self.history: Dict[str, List] = {}

    def _create_state_extractor(self) -> StateExtractor:
        """创建状态提取器"""
        if self.config.task_type == TaskType.REGRESSION:
            return RegressionStateExtractor(self.model_adapter, self.config)
        elif self.config.task_type == TaskType.CLUSTERING:
            return ClusteringStateExtractor(self.model_adapter, self.config)
        else:
            return ClassificationStateExtractor(self.model_adapter, self.config)

    def _create_agent(self) -> BaseAgent:
        """创建 Agent"""
        at = self.config.agent_type
        common = dict(
            state_size=self.config.state_size,
            memory_size=self.config.memory_size,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            epsilon_min=self.config.epsilon_min,
            epsilon_decay=self.config.epsilon_decay,
            learning_rate=self.config.learning_rate,
        )

        if at == AgentType.DUELING_TWO_STAGE:
            return DuelingTwoStageAgent(**common)
        elif at == AgentType.DUELING_SINGLE_STAGE:
            return DuelingSingleStageAgent(action_size=4, **common)
        elif at == AgentType.TWO_STAGE:
            return TwoStageDQNAgent(**common)
        else:
            return SingleStageDQNAgent(action_size=4, **common)

    # ====================================================================
    # 公共接口
    # ====================================================================

    def train(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              n_episodes: Optional[int] = None,
              verbose: bool = True,
              detected_errors: Optional[Dict[str, List]] = None,
              start_episode: int = 0,
              prev_history: Optional[Dict[str, List]] = None,
              agent: Optional[BaseAgent] = None,
              X_clean_val: Optional[np.ndarray] = None,
              y_clean_val: Optional[np.ndarray] = None,
              ) -> Tuple[BaseAgent, Dict[str, List]]:
        """
        训练 DQN Agent

        根据 config.training_mode 分派训练模式:
        - "clean_base": 在去NaN的脏数据上自注入错误
        - "self_supervised": 在脏数据全量上融合已检测错误 + 新注入错误

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            n_episodes: 训练轮数（默认使用配置）
            verbose: 是否打印详细信息
            detected_errors: 检测器结果（self_supervised 模式需要）
                {'missing': [...], 'semantic': [...], 'syntactic': [...], 'label_noise': [...]}
            start_episode: 续训起始轮号（0=从头训练）
            prev_history: 之前的训练历史（续训时拼接）
            agent: 外部传入的已有 Agent（续训时使用）
            X_clean_val: 干净验证集特征（Oracle 模式用于 reward 信号）
            y_clean_val: 干净验证集标签（Oracle 模式用于 reward 信号）

        Returns:
            (agent, history)
        """
        if (self.config.training_mode == 'self_supervised'
                and detected_errors is not None):
            return self._train_self_supervised(
                X_dirty, y, n_episodes, verbose, detected_errors,
                start_episode=start_episode,
                prev_history=prev_history,
                agent=agent)
        else:
            return self._train_clean_base(
                X_dirty, y, n_episodes, verbose, detected_errors,
                start_episode=start_episode,
                prev_history=prev_history,
                agent=agent,
                X_clean_val=X_clean_val,
                y_clean_val=y_clean_val)

    # ====================================================================
    # 训练模式 1: clean_base (默认，保持现有逻辑)
    # ====================================================================

    def _train_clean_base(self,
                          X_dirty: np.ndarray,
                          y: np.ndarray,
                          n_episodes: Optional[int] = None,
                          verbose: bool = True,
                          detected_errors: Optional[Dict[str, List]] = None,
                          start_episode: int = 0,
                          prev_history: Optional[Dict[str, List]] = None,
                          agent: Optional[BaseAgent] = None,
                          X_clean_val: Optional[np.ndarray] = None,
                          y_clean_val: Optional[np.ndarray] = None,
                          ) -> Tuple[BaseAgent, Dict[str, List]]:
        """clean_base 训练模式

        逻辑: X_base = DeleteFix(dirty, detected_errors) → 每 episode 注入错误 → 用注入前值当伪真值

        Oracle 模式 (use_clean_validation=True):
          - X_base_inject 仍从脏数据构建（用于错误注入）
          - CleaningEnv.X_base 使用 X_clean_val（干净验证集做 reward 信号）
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            mode_label = "续训" if start_episode > 0 else "开始训练"
            self.logger.log_info(f"{mode_label} ({self.config.agent_type.value} Agent, mode=clean_base)")
            self.logger.log_info(f"  任务类型: {self.config.task_type.value}")
            self.logger.log_info(f"  模型类型: {self.config.model_type.value}")
            self.logger.log_info(f"  原始数据: {len(X_dirty)} 行")
            if start_episode > 0:
                self.logger.log_info(f"  续训起始轮: {start_episode}")

        # 1. 准备基准数据（自动选择 DeleteFix 或 VE-Fill 策略）
        if getattr(self.config, 'auto_select_base', True) and detected_errors is not None:
            X_base, y_base, strategy = self._select_best_base(
                X_dirty, y, detected_errors, verbose)
        else:
            X_base, y_base = self._build_base_data(
                X_dirty, y, detected_errors=detected_errors,
                fill_nan=True, verbose=verbose)

        # 2. 分析标签错误模式（条件性标签注入）
        label_pattern, label_rate_range = self._analyze_label_injection(
            detected_errors, y, verbose)

        # 3. 计算自适应注入比例范围（基于检测结果）
        adaptive_ranges = self._compute_adaptive_rate_ranges(
            detected_errors, X_dirty, y, verbose)

        # 4. 初始化 Agent（续训时使用外部传入的 Agent）
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent()

        error_injector = ErrorInjector(
            X_base, y_base,
            fd_rules=self.config.fd_rules,
            column_names=self.config.column_names,
            rich_rules=self.config.rich_rules,
            label_encoders=self.config.label_encoders,
            scaler=self.config.scaler,
            categorical_cols=self.config.categorical_cols,
            dirty_df=self.config.dirty_df,
            label_col=self.config.label_col,
        )

        if verbose:
            stats = error_injector.get_stats()
            self.logger.log_info(
                f"  ErrorInjector: rich_rules={stats['has_rich_rules']}, "
                f"domain={stats['n_domain_rules']}, cfd={stats['n_cfd_rules']}, "
                f"fd={stats['n_fd_rules']}")

        # Oracle 分支: 确定 reward 评估目标
        if self.config.use_clean_validation and X_clean_val is not None:
            X_eval_target, y_eval_target = X_clean_val, y_clean_val
            if verbose:
                self.logger.log_info(
                    f"  Oracle 模式: reward 使用干净验证集 ({len(X_clean_val)} 行)")
        else:
            X_eval_target, y_eval_target = X_base, y_base

        # 5. 训练循环
        return self._run_training_loop(
            error_injector, n_episodes, verbose,
            label_pattern, label_rate_range,
            inject_mode='clean_base',
            adaptive_ranges=adaptive_ranges,
            start_episode=start_episode,
            prev_history=prev_history,
            X_base=X_eval_target, y_base=y_eval_target,
        )

    # ====================================================================
    # 训练模式 2: self_supervised (脏数据全量训练)
    # ====================================================================

    def _train_self_supervised(self,
                               X_dirty: np.ndarray,
                               y: np.ndarray,
                               n_episodes: Optional[int] = None,
                               verbose: bool = True,
                               detected_errors: Optional[Dict[str, List]] = None,
                               start_episode: int = 0,
                               prev_history: Optional[Dict[str, List]] = None,
                               agent: Optional[BaseAgent] = None,
                               ) -> Tuple[BaseAgent, Dict[str, List]]:
        """self_supervised 训练模式

        逻辑:
          - detected_errors 中的错误 → 固定 error_list（repair_value 用列均值/KNN估计）
          - 标签错误 → repair_value=None，训练时倾向 delete
          - 未检测位置 → 注入新错误，用注入前值当伪真值
          - 每 episode: 固定错误列表 + 新注入错误列表 → 合并 → CleaningEnv
        """
        n_episodes = n_episodes or self.config.n_episodes

        if verbose:
            mode_label = "续训" if start_episode > 0 else "开始训练"
            self.logger.log_info(f"{mode_label} ({self.config.agent_type.value} Agent, mode=self_supervised)")
            self.logger.log_info(f"  任务类型: {self.config.task_type.value}")
            self.logger.log_info(f"  模型类型: {self.config.model_type.value}")
            self.logger.log_info(f"  原始数据: {len(X_dirty)} 行")
            if start_episode > 0:
                self.logger.log_info(f"  续训起始轮: {start_episode}")

        # 1. 构建固定的已检测错误列表
        fixed_error_list, detected_cells = self._build_detected_error_list(
            detected_errors, X_dirty, y, verbose)

        # 2. 分析标签错误模式
        label_pattern, label_rate_range = self._analyze_label_injection(
            detected_errors, y, verbose)

        # 3. 计算自适应注入比例范围（基于检测结果）
        adaptive_ranges = self._compute_adaptive_rate_ranges(
            detected_errors, X_dirty, y, verbose)

        # 4. 初始化 Agent（续训时使用外部传入的 Agent）
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent()

        # 5. 使用 DeleteFix 策略构建 ErrorInjector 统计基
        X_base_for_stats, y_base_for_stats = self._build_base_data(
            X_dirty, y, detected_errors=detected_errors,
            fill_nan=True, verbose=verbose)

        error_injector = ErrorInjector(
            X_base_for_stats, y_base_for_stats,
            fd_rules=self.config.fd_rules,
            column_names=self.config.column_names,
            rich_rules=self.config.rich_rules,
            # 编码工具（用于 CSV 空间注入后重新编码）
            label_encoders=self.config.label_encoders,
            scaler=self.config.scaler,
            categorical_cols=self.config.categorical_cols,
            dirty_df=self.config.dirty_df,
            label_col=self.config.label_col,
        )

        if verbose:
            self.logger.log_info(
                f"  固定错误列表: {len(fixed_error_list)} 个, "
                f"检测位置: {len(detected_cells)} 个")

        # 6. 训练循环
        return self._run_training_loop(
            error_injector, n_episodes, verbose,
            label_pattern, label_rate_range,
            inject_mode='self_supervised',
            X_dirty_full=X_dirty,
            y_dirty_full=y,
            fixed_error_list=fixed_error_list,
            detected_cells=detected_cells,
            adaptive_ranges=adaptive_ranges,
            start_episode=start_episode,
            prev_history=prev_history,
            X_base=X_base_for_stats, y_base=y_base_for_stats,
        )

    # ====================================================================
    # 共用训练循环
    # ====================================================================

    def _run_training_loop(self,
                           error_injector: ErrorInjector,
                           n_episodes: int,
                           verbose: bool,
                           label_pattern: Optional[LabelErrorPattern],
                           label_rate_range: Tuple[float, float],
                           inject_mode: str = 'clean_base',
                           X_dirty_full: Optional[np.ndarray] = None,
                           y_dirty_full: Optional[np.ndarray] = None,
                           fixed_error_list: Optional[List[Dict]] = None,
                           detected_cells: Optional[Set[Tuple[int, int]]] = None,
                           adaptive_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                           start_episode: int = 0,
                           prev_history: Optional[Dict[str, List]] = None,
                           X_base: Optional[np.ndarray] = None,
                           y_base: Optional[np.ndarray] = None,
                           ) -> Tuple[BaseAgent, Dict[str, List]]:
        """共用训练循环（支持续训 + 不确定度记录）"""

        # 初始化训练历史
        history_keys = [
            'episode', 'score', 'reward', 'epsilon',
            'no_action', 'repair_value', 'delete', 'replace_nearby',
            'q_std', 'action_probs',
        ]
        self.history = {k: [] for k in history_keys}

        # 续训: 拼接之前的历史
        if prev_history:
            for k in self.history:
                self.history[k] = list(prev_history.get(k, []))

        # 最佳模型追踪
        # 注意: 续训时不恢复旧 best_score — 旧分数基于当时的编码/数据/注入,
        # 与当前训练环境不可比, 会导致新 episode 永远无法超越旧阈值,
        # best_weights 始终为 None, 最佳模型恢复机制失效。
        best_score = -float('inf') if self.config.task_type == TaskType.REGRESSION else 0
        best_episode = start_episode
        best_cost = float('inf')
        best_actions = None
        best_weights = None

        if verbose:
            self.logger.log_info(f"  开始训练 DQN ({n_episodes} episodes, 起始轮={start_episode})...")

        end_episode = start_episode + n_episodes

        # 共享 ValueEstimator（避免每个 episode 重建索引）
        shared_ve = ValueEstimator(self.config)

        for episode in range(start_episode, end_episode):
            # 自适应错误率（基于检测结果，没检测到的类型不注入）
            if adaptive_ranges is not None:
                missing_rate = np.random.uniform(*adaptive_ranges['missing'])
                semantic_rate = np.random.uniform(*adaptive_ranges['semantic'])
                syntactic_rate = np.random.uniform(*adaptive_ranges['syntactic'])
                label_rate = np.random.uniform(*adaptive_ranges['label'])
            else:
                # fallback: 无检测结果时使用 config 固定范围
                missing_rate = np.random.uniform(*self.config.missing_rate_range)
                semantic_rate = np.random.uniform(*self.config.semantic_rate_range)
                syntactic_rate = np.random.uniform(*self.config.syntactic_rate_range)
                label_rate = np.random.uniform(*label_rate_range)

            if inject_mode == 'self_supervised' and X_dirty_full is not None:
                # 自监督: 在脏数据未检测位置上注入额外错误
                X_injected, y_injected, new_injected = error_injector.inject_on_dirty(
                    X_dirty_full, y_dirty_full, detected_cells or set(),
                    missing_rate, semantic_rate, syntactic_rate,
                    label_rate, label_pattern,
                )
                # 合并: 固定错误列表 + 新注入错误列表
                new_error_list = error_injector.build_error_list(new_injected)
                error_list = (fixed_error_list or []) + new_error_list
            else:
                # clean_base: 在去NaN基准数据上注入
                X_injected, y_injected, injected = error_injector.inject_errors(
                    missing_rate, semantic_rate, syntactic_rate,
                    label_rate, label_pattern,
                )
                error_list = error_injector.build_error_list(injected)

            # 计算 shaping 衰减系数: warmup 阶段保持 1.0，之后线性衰减到 min_weight
            warmup_ratio = self.config.shaping_warmup_ratio
            min_weight = self.config.shaping_min_weight
            warmup_end = start_episode + int(n_episodes * warmup_ratio)
            if episode < warmup_end:
                shaping_weight = 1.0
            else:
                decay_progress = (episode - warmup_end) / max(start_episode + n_episodes - warmup_end, 1)
                shaping_weight = max(min_weight, 1.0 - (1.0 - min_weight) * decay_progress)

            # 创建环境（共享 ValueEstimator 避免重建索引）
            env = CleaningEnv(
                X_injected, y_injected, error_list,
                self.model_adapter, self.state_extractor, self.config,
                X_base=X_base, y_base=y_base,
                value_estimator=shared_ve,
                shaping_weight=shaping_weight,
            )

            # 首轮打印自适应评估间隔
            if episode == start_episode and verbose:
                n_errs = len(error_list)
                N = env._reward_eval_interval
                evals_per_ep = (n_errs // N + 1) if n_errs > 0 else 0
                self.logger.log_info(
                    f"  Reward 评估间隔: N={N} "
                    f"(错误数={n_errs}, 约 {evals_per_ep} 次评估/episode)"
                )

            state = env.reset()
            total_reward = 0

            # 收集本 episode 的 Q 值（用于不确定度统计）
            q_values_this_episode = []

            # Episode 训练
            while True:
                if self.config.agent_type in (AgentType.TWO_STAGE, AgentType.DUELING_TWO_STAGE):
                    # 收集 Q 值
                    try:
                        q_vals = self.agent.get_q_values(state)
                        q_values_this_episode.append(q_vals)
                    except Exception:
                        pass

                    final_action, stage1_action, stage2_action = self.agent.act(state, training=True)
                    next_state, reward, done, info = env.step(final_action)

                    # 分阶段 reward 路由
                    stage1_reward = info.get('stage1_reward', reward)
                    self.agent.remember_stage1(state, stage1_action, stage1_reward, next_state, done)

                    if stage2_action is not None:
                        stage2_reward = info.get('stage2_reward', reward)
                        if stage2_reward is not None:
                            self.agent.remember_stage2(state, stage2_action, stage2_reward, next_state, done)
                else:
                    # 收集 Q 值
                    try:
                        q_vals = self.agent.get_q_values(state)
                        q_values_this_episode.append(q_vals)
                    except Exception:
                        pass

                    action = self.agent.act(state, training=True)
                    next_state, reward, done, _ = env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                if done:
                    break

            # 经验回放
            self.agent.replay(batch_size=self.config.batch_size)

            # 更新目标网络
            if (episode + 1) % self.config.target_update_freq == 0:
                self.agent.update_target_model()

            # 记录历史（复用 _calculate_final_reward 中已评估的 score）
            final_score = getattr(env, 'last_episode_score', None)
            if final_score is None:
                final_score = env._evaluate_score(env.X_current, env.y_current)
            self.history['episode'].append(episode + 1)
            self.history['score'].append(final_score)
            self.history['reward'].append(total_reward)
            self.history['epsilon'].append(self.agent.epsilon)
            self.history['no_action'].append(env.action_counts['no_action'])
            self.history['repair_value'].append(env.action_counts['repair_value'])
            self.history['delete'].append(env.action_counts['delete'])
            self.history['replace_nearby'].append(env.action_counts['replace_nearby'])

            # 不确定度统计
            if q_values_this_episode:
                q_arr = np.array(q_values_this_episode)
                q_std = float(np.mean(np.std(q_arr, axis=1)))
                # 动作概率: softmax(Q) 的平均
                q_mean = np.mean(q_arr, axis=0)
                q_exp = np.exp(q_mean - np.max(q_mean))
                action_probs = (q_exp / q_exp.sum()).tolist()
            else:
                q_std = 0.0
                action_probs = []
            self.history['q_std'].append(q_std)
            self.history['action_probs'].append(action_probs)

            self.logger.log_episode(
                episode + 1, final_score, total_reward,
                self.agent.epsilon, env.action_counts
            )

            # 追踪最佳模型
            current_cost = env.action_counts['repair_value']
            is_better = False

            if self.config.task_type == TaskType.REGRESSION:
                if final_score > best_score or (final_score == best_score and current_cost <= best_cost):
                    is_better = True
            else:
                if final_score > best_score or (final_score == best_score and current_cost <= best_cost):
                    is_better = True

            if is_better:
                best_score = final_score
                best_cost = current_cost
                best_episode = episode + 1
                best_actions = env.action_counts.copy()
                best_weights = self.agent.get_weights()

            # 打印进度
            if verbose and (episode + 1) % self.config.log_interval == 0:
                self.logger.log_info(
                    f"  Episode {episode + 1}/{end_episode}: "
                    f"Score={final_score:.4f}, Best={best_score:.4f}, Eps={self.agent.epsilon:.3f}"
                )
                self.logger.log_info(f"    Actions: {env.action_counts}")

        # 恢复最佳模型权重
        if best_weights is not None:
            self.agent.set_weights(best_weights)
            if verbose:
                self.logger.log_info(
                    f"\n已恢复最佳模型权重 (Episode {best_episode}, "
                    f"Score={best_score:.4f}, Cost={best_cost})"
                )

        # 更新 Agent 续训元数据
        self.agent.total_episodes = end_episode
        self.agent.best_score = best_score
        self.agent.best_episode = best_episode

        if verbose:
            self.logger.log_info(f"\n训练完成! 最佳分数: {best_score:.4f} (共 {end_episode} episodes)")
            if best_actions:
                self.logger.log_info(f"  最佳动作分布: {best_actions}")

        return self.agent, self.history

    # ====================================================================
    # 辅助方法
    # ====================================================================

    def _analyze_label_injection(self,
                                  detected_errors: Optional[Dict[str, List]],
                                  y_dirty: np.ndarray,
                                  verbose: bool,
                                  ) -> Tuple[Optional[LabelErrorPattern], Tuple[float, float]]:
        """分析是否需要标签错误注入

        条件性: 如果检测器未发现标签错误，不注入。

        Returns:
            (label_pattern, label_rate_range)
        """
        if detected_errors is None:
            return None, (0.0, 0.0)

        detected_label = detected_errors.get('label_noise', [])
        if not detected_label:
            if verbose:
                self.logger.log_info("  标签注入: 检测器未发现标签错误，不注入")
            return None, (0.0, 0.0)

        # 分析标签错误模式
        task_type_str = self.config.task_type.value if hasattr(self.config.task_type, 'value') else str(self.config.task_type)
        pattern = analyze_label_error_pattern(detected_label, y_dirty, task_type=task_type_str)

        if verbose:
            self.logger.log_info(
                f"  标签注入: 检测到 {len(detected_label)} 个标签错误, "
                f"rate={pattern.error_rate:.3f}, symmetric={pattern.is_symmetric}")

        return pattern, self.config.label_rate_range

    def _compute_adaptive_rate_ranges(
        self,
        detected_errors: Optional[Dict[str, List]],
        X_dirty: np.ndarray,
        y: np.ndarray,
        verbose: bool,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """根据检测结果计算自适应注入比例范围

        原则:
          - 没检测到的错误类型 → rate_range = (0, 0)，训练时不注入
          - 检测到的错误类型 → 以检测率为中心, ±20% 浮动

        Returns:
            None 表示无检测结果（使用 config 默认值），否则返回自适应范围字典
        """
        if detected_errors is None:
            return None

        n_rows, n_features = X_dirty.shape
        n_cells = n_rows * n_features

        type_map = {
            'missing':   ('missing',     n_cells),
            'semantic':  ('semantic',    n_cells),
            'syntactic': ('syntactic',   n_cells),
            'label':     ('label_noise', n_rows),
        }

        result = {}
        for out_key, (det_key, denominator) in type_map.items():
            n_detected = len(detected_errors.get(det_key, []))
            if n_detected == 0:
                # 有 DC/FD/CFD 规则但原始数据未检测到语义错误时，给最小注入率
                # 原因: 规则可能覆盖注入后才出现的违规模式，不应完全跳过
                if out_key == 'semantic' and self._has_semantic_rules():
                    result[out_key] = (0.005, 0.02)
                else:
                    result[out_key] = (0.0, 0.0)
            else:
                rate = n_detected / max(denominator, 1)
                lo = max(rate * 0.8, 0.005)
                hi = max(rate * 1.2, lo + 0.005)
                result[out_key] = (round(lo, 6), round(hi, 6))

        if verbose:
            self.logger.log_info("  自适应注入比例范围 (基于检测结果):")
            for k, (lo, hi) in result.items():
                status = "不注入" if lo == 0.0 else f"({lo:.4f}, {hi:.4f})"
                self.logger.log_info(f"    {k}: {status}")

        return result

    def _has_semantic_rules(self) -> bool:
        """检查配置中是否存在语义类规则（FD/CFD/DC）

        用于 _compute_adaptive_rate_ranges(): 当原始数据无语义错误但有规则时，
        仍应给最小注入率以覆盖规则可检测的违规模式。

        Returns:
            True 如果存在 FD 规则、CFD 规则或 DC 规则
        """
        # FD 规则
        if self.config.fd_rules:
            return True
        # rich_rules 中的 CFD / DC
        rr = self.config.rich_rules
        if rr and rr.get('has_rich_rules'):
            if rr.get('cfd_rules'):
                return True
            if rr.get('dc_rules'):
                return True
        return False

    def _build_base_data(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]] = None,
        fill_nan: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建基准数据 (DeleteFix 策略)

        当有 detected_errors 时，只删除检测到有错误的行，保留其余行；
        无检测结果时 fallback 到原始 NaN-free 策略。

        Args:
            X_dirty: 原始脏数据
            y: 标签向量
            detected_errors: 检测结果 (None 时 fallback 到 NaN-free 策略)
            fill_nan: 是否填充残余 NaN (非错误行中的缺失值用列均值填充)
            verbose: 是否输出日志

        Returns:
            (X_base, y_base)
        """
        if detected_errors is not None:
            # DeleteFix 策略：只删除检测到有错误的行
            error_row_set = set()
            for err_type in ('missing', 'semantic', 'syntactic', 'label_noise'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        error_row_set.add(int(item[0]))
                    elif isinstance(item, dict) and 'idx' in item:
                        error_row_set.add(int(item['idx']))

            n_total = len(X_dirty)
            min_keep = max(10, int(n_total * 0.2))  # 至少保留 20%

            if len(error_row_set) <= n_total - min_keep:
                keep_mask = np.array([i not in error_row_set for i in range(n_total)])
            else:
                # 超过上限: 按错误数量降序，优先删多错误行
                from collections import Counter
                row_error_count = Counter()
                for err_type in ('label_noise', 'missing', 'syntactic', 'semantic'):
                    for item in detected_errors.get(err_type, []):
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            row_error_count[int(item[0])] += 1
                        elif isinstance(item, dict) and 'idx' in item:
                            row_error_count[int(item['idx'])] += 1
                max_delete = n_total - min_keep
                sorted_rows = sorted(row_error_count.keys(),
                                     key=lambda r: row_error_count[r], reverse=True)
                delete_set = set(sorted_rows[:max_delete])
                keep_mask = np.array([i not in delete_set for i in range(n_total)])

            X_base = X_dirty[keep_mask].copy()
            y_base = y[keep_mask].copy()
            n_deleted = n_total - int(keep_mask.sum())

            # 填充残余 NaN (用 ValueEstimator 逐单元格精确估值)
            n_residual_nan = 0
            if fill_nan:
                col_means = np.nanmean(X_base, axis=0)
                # 构建 idx 映射: X_base[i] → dirty_df 原始行号
                keep_indices = np.where(keep_mask)[0]
                # 用 ValueEstimator 逐单元格估值
                from demandprep_policy.core.environments.value_estimation import ValueEstimator
                ve = ValueEstimator(self.config)
                for j in range(X_base.shape[1]):
                    nan_mask_j = np.isnan(X_base[:, j])
                    n_residual_nan += int(nan_mask_j.sum())
                    for i in np.where(nan_mask_j)[0]:
                        X_base[i, j] = ve.estimate_feature_value(
                            X_base, i, j, set(), col_means,
                            dirty_df_row_indices=keep_indices,
                        )
                # y 标签残余 NaN: 用 KNN 多数投票
                y_nan = np.isnan(y_base)
                if y_nan.any():
                    n_residual_nan += int(y_nan.sum())
                    for i in np.where(y_nan)[0]:
                        y_base[i] = self._estimate_label_by_knn(X_base, y_base, i, k=5)

            if verbose:
                self.logger.log_info(
                    f"  基准数据: {len(X_base)} 行 "
                    f"(DeleteFix 策略: 删除 {n_deleted} 错误行, "
                    f"填充 {n_residual_nan} 个残余 NaN)")
        else:
            # Fallback: 无检测结果时仍用原策略 (删除所有含 NaN 行)
            no_nan_mask = ~np.isnan(X_dirty).any(axis=1)
            X_base = X_dirty[no_nan_mask].copy()
            y_base = y[no_nan_mask].copy()

            if verbose:
                self.logger.log_info(f"  基准数据: {len(X_base)} 行 (NaN-free)")

        return X_base, y_base

    def _build_ve_fill_data(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """VE-Fill 策略：保留全部行，用 VE 修复错误单元格

        与 DeleteFix 的区别:
          - 不删除任何行，保留完整样本量
          - 先批量填充所有 NaN（col_means），使矩阵 NaN-free
          - 再对检测到的错误单元格用 VE 覆盖（在 NaN-free 矩阵上 KNN 更快）
          - 标签错误：用 KNN 多数投票修复

        性能优化:
          旧实现: 每个 VE 调用在含 NaN 矩阵上做 KNN（内部重复 NaN fill），O(K × N × M)
          新实现: 先批量 NaN fill O(N×M)，再 VE 覆盖 O(K × N × M) 但矩阵已无 NaN

        Returns:
            (X_filled, y_filled) — 全量数据，所有检测错误已 VE 修复
        """
        X_out = X_dirty.copy()
        y_out = y.copy()

        # Step 0: 批量预填充所有 NaN（col_means），使矩阵 NaN-free
        col_means = np.nanmean(X_out, axis=0)
        n_total_nan = 0
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                n_total_nan += n_nan
                X_out[nan_mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

        # y 标签 NaN 也预填充
        y_nan = np.isnan(y_out)
        if y_nan.any():
            y_mean = np.nanmean(y_out)
            y_out[y_nan] = y_mean if not np.isnan(y_mean) else 0.0

        # Step 1: 收集错误单元格
        error_cells = set()  # (row, col) pairs
        label_error_rows = set()

        if detected_errors:
            for err_type in ('missing', 'semantic', 'syntactic'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        error_cells.add((int(item[0]), int(item[1])))
                    elif isinstance(item, dict):
                        row = int(item.get('idx', -1))
                        col = int(item.get('col', -1))
                        if row >= 0 and col >= 0:
                            error_cells.add((row, col))
            for item in detected_errors.get('label_noise', []):
                if isinstance(item, (list, tuple)):
                    label_error_rows.add(int(item[0]))
                elif isinstance(item, dict):
                    row = int(item.get('idx', -1))
                    if row >= 0:
                        label_error_rows.add(row)

        # Step 2: 用 VE 覆盖检测到的错误单元格（在 NaN-free 矩阵上 KNN 更快）
        ve = ValueEstimator(self.config)
        for row, col in error_cells:
            if row < X_out.shape[0] and col < X_out.shape[1]:
                X_out[row, col] = ve.estimate_feature_value(
                    X_out, row, col, set(), col_means)

        # Step 3: 标签错误用 KNN 多数投票修复
        for row in label_error_rows:
            if 0 <= row < len(y_out):
                y_out[row] = self._estimate_label_by_knn(X_out, y_out, row, k=5)

        if verbose:
            self.logger.log_info(
                f"  VE-Fill 策略: {len(X_out)} 行 "
                f"(预填充 {n_total_nan} NaN, "
                f"VE 覆盖 {len(error_cells)} 个错误单元格 + "
                f"{len(label_error_rows)} 个标签错误)")

        return X_out, y_out

    def _normalize_cv_score(self, raw_score: float) -> float:
        """将模型原始评分归一化到 [0, 1]（同 CleaningEnv._normalize_raw_score）"""
        if self.config.task_type == TaskType.REGRESSION:
            mse = abs(raw_score)
            return 1.0 / (1.0 + mse)
        elif self.config.task_type == TaskType.CLUSTERING:
            return (raw_score + 1.0) / 2.0
        else:
            return raw_score

    def _cv_evaluate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
        """对候选数据做 K-fold CV，返回平均归一化 score

        Args:
            X: 候选特征矩阵（已无 NaN）
            y: 候选标签向量
            n_folds: 交叉验证折数

        Returns:
            平均归一化 score
        """
        from sklearn.model_selection import KFold

        # 数据太少时退化为单次评估
        if len(X) < n_folds * 2:
            try:
                model_kwargs = self.config.model_kwargs or {}
                adapter = create_model_adapter(
                    self.config.model_type, self.config.task_type, **model_kwargs)
                adapter.fit(X, y)
                raw = adapter.evaluate(X, y)
                return self._normalize_cv_score(raw)
            except Exception:
                return 0.0

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        model_kwargs = self.config.model_kwargs or {}

        for train_idx, val_idx in kf.split(X):
            try:
                adapter = create_model_adapter(
                    self.config.model_type, self.config.task_type, **model_kwargs)
                adapter.fit(X[train_idx], y[train_idx])
                raw = adapter.evaluate(X[val_idx], y[val_idx])
                scores.append(self._normalize_cv_score(raw))
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _select_best_base(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        detected_errors: Optional[Dict[str, List]],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """自动选择最优 clean base 策略

        注意: X_dirty 传入时已经过 RAHA 真值替换（如果 apply_raha_truth=True），
        两个候选策略共享相同的 RAHA 预处理结果。

        流程:
          Step 1: 生成 DeleteFix 候选 — 调用 _build_base_data()
          Step 2: 生成 VE-Fill 候选 — 调用 _build_ve_fill_data()
          Step 3: 5-fold CV 评估，选择 score 更高的候选

        Returns:
            (X_base, y_base, strategy_name)
        """
        n_folds = getattr(self.config, 'base_cv_folds', 5)

        # Step 1: DeleteFix 候选
        X_delete, y_delete = self._build_base_data(
            X_dirty, y, detected_errors=detected_errors,
            fill_nan=True, verbose=False)

        # Step 2: VE-Fill 候选
        X_vefill, y_vefill = self._build_ve_fill_data(
            X_dirty, y, detected_errors, verbose=False)

        # Step 3: CV 评估 + 样本保留率加权
        # 无干净验证集时，self-CV 会高估小样本数据（内部一致性高但泛化差）
        # 用 √(n/N) 权重惩罚数据损失，确保样本充足性纳入考量
        score_delete = self._cv_evaluate(X_delete, y_delete, n_folds)
        score_vefill = self._cv_evaluate(X_vefill, y_vefill, n_folds)

        n_total = len(X_dirty)
        weight_delete = np.sqrt(len(X_delete) / max(n_total, 1))
        weight_vefill = np.sqrt(len(X_vefill) / max(n_total, 1))

        adj_score_delete = score_delete * weight_delete
        adj_score_vefill = score_vefill * weight_vefill

        # 选择加权后更优的策略
        if adj_score_vefill > adj_score_delete:
            chosen = 'VE-Fill'
            X_base, y_base = X_vefill, y_vefill
        else:
            chosen = 'DeleteFix'
            X_base, y_base = X_delete, y_delete

        # 关键决策：始终输出（不受 verbose 控制）
        self.logger.log_info(
            f"  Clean Base 自动选择 ({n_folds}-fold CV, 样本保留率加权):")
        self.logger.log_info(
            f"    DeleteFix: {len(X_delete)} 行, CV={score_delete:.4f}, "
            f"权重={weight_delete:.3f}, 加权={adj_score_delete:.4f}")
        self.logger.log_info(
            f"    VE-Fill:   {len(X_vefill)} 行, CV={score_vefill:.4f}, "
            f"权重={weight_vefill:.3f}, 加权={adj_score_vefill:.4f}")
        self.logger.log_info(
            f"    → 选择 {chosen} ({len(X_base)} 行)")

        return X_base, y_base, chosen

    def _build_detected_error_list(self,
                                    detected_errors: Optional[Dict[str, List]],
                                    X_dirty: np.ndarray,
                                    y_dirty: np.ndarray,
                                    verbose: bool,
                                    ) -> Tuple[List[Dict], Set[Tuple[int, int]]]:
        """构建已检测错误的固定 error_list（自监督模式用）

        - missing/semantic/syntactic: repair_value = ValueEstimator 估计（FD → KNN → DOMAIN）
        - label_noise: repair_value = KNN 多数投票估计

        Returns:
            (fixed_error_list, detected_cells)
        """
        fixed_error_list = []
        detected_cells: Set[Tuple[int, int]] = set()

        if not detected_errors:
            return fixed_error_list, detected_cells

        # 构建 ValueEstimator（FD + KNN + DOMAIN）
        value_estimator = ValueEstimator(self.config)
        col_means = np.nanmean(X_dirty, axis=0)

        type_map = {'missing': 0, 'semantic': 1, 'syntactic': 2, 'label_noise': 3}

        for err_type_name, err_type_id in type_map.items():
            for item in detected_errors.get(err_type_name, []):
                if isinstance(item, (list, tuple)):
                    idx = int(item[0])
                    col = int(item[1]) if len(item) > 1 else -1
                elif isinstance(item, dict):
                    idx = int(item.get('idx', -1))
                    col = int(item.get('col', -1))
                else:
                    continue

                if idx < 0:
                    continue

                # 构建 repair_value
                if err_type_id == 3:
                    # 标签错误: KNN 多数投票估计
                    repair_value = self._estimate_label_by_knn(
                        X_dirty, y_dirty, idx, k=5
                    )
                elif col >= 0 and col < X_dirty.shape[1]:
                    # 特征错误: FD → 多维 KNN → DOMAIN 裁剪
                    repair_value = value_estimator.estimate_feature_value(
                        X_dirty, idx, col, set(), col_means
                    )
                else:
                    repair_value = 0.0

                fixed_error_list.append({
                    'idx': idx,
                    'col': col,
                    'type': err_type_id,
                    'repair_value': repair_value,
                })

                if col >= 0:
                    detected_cells.add((idx, col))

        if verbose:
            from collections import Counter
            type_counts = Counter(e['type'] for e in fixed_error_list)
            self.logger.log_info(
                f"  固定错误列表: {dict(type_counts)} "
                f"(0=missing, 1=semantic, 2=syntactic, 3=label)")
            self.logger.log_info(f"  值估计器: {value_estimator.summary()}")

        return fixed_error_list, detected_cells

    def _estimate_label_by_knn(self, X: np.ndarray, y: np.ndarray,
                                idx: int, k: int = 5) -> float:
        """用 KNN 多数投票/加权均值估计标签值

        Args:
            X: 特征矩阵
            y: 标签向量
            idx: 目标行索引
            k: 邻居数量

        Returns:
            估计的标签值
        """
        # NaN 填充
        X_filled = X.copy()
        for c in range(X_filled.shape[1]):
            nan_mask = np.isnan(X_filled[:, c])
            if nan_mask.any():
                col_mean = np.nanmean(X_filled[:, c])
                X_filled[nan_mask, c] = col_mean if not np.isnan(col_mean) else 0.0

        target = X_filled[idx]
        distances = np.linalg.norm(X_filled - target, axis=1)
        distances[idx] = np.inf

        k = min(k, (distances < np.inf).sum())
        if k == 0:
            return y[idx]

        nearest = np.argsort(distances)[:k]
        nearest_labels = y[nearest]
        valid = nearest_labels[~np.isnan(nearest_labels)]

        if len(valid) == 0:
            return y[idx]

        # 回归: 距离倒数加权均值
        if self.config.task_type == TaskType.REGRESSION:
            valid_dists = distances[nearest][~np.isnan(nearest_labels)]
            weights = 1.0 / (valid_dists + 1e-8)
            weights /= weights.sum()
            return float(np.average(valid, weights=weights))

        # 分类: 多数投票
        unique, counts = np.unique(valid, return_counts=True)
        return float(unique[np.argmax(counts)])

    # ====================================================================
    # 模型管理
    # ====================================================================

    def save_agent(self, path: str) -> None:
        """保存 Agent"""
        if self.agent is None:
            raise ValueError("Agent 未训练，无法保存")
        self.model_io.save_agent(self.agent, path)

    def load_agent(self, path: str) -> BaseAgent:
        """加载 Agent"""
        self.agent = self.model_io.load_agent(path, self.config.agent_type)
        return self.agent

    def get_agent(self) -> Optional[BaseAgent]:
        """获取当前 Agent"""
        return self.agent

    def get_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return self.history
