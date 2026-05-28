"""
DemandPrepPolicy 高层 API
====================

提供简洁的接口用于数据清洗。
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Set
import os
import numpy as np

from ..config import (
    DemandPrepPolicyConfig, TaskType, ModelType, AgentType,
    DetectorMode, InferenceMode,
)
from ..core.agents import (
    BaseAgent,
    SingleStageDQNAgent, TwoStageDQNAgent,
    DuelingSingleStageAgent, DuelingTwoStageAgent,
)
from ..training import Trainer
from ..inference import SinglePhaseInference, TwoPhaseInference
from ..detectors import (
    ErrorInjector, AutoDetector, RahaBasedDetector, OracleDetector,
    parse_rules_file, load_rules, extract_fd_pairs, rules_to_dict,
)
from ..utils.model_io import ModelIO
from ..utils.logger import DemandPrepPolicyLogger


class DemandPrepPolicy:
    """
    DemandPrepPolicy 数据清洗系统

    使用示例:
    ```python
    from demandprep_policy import DemandPrepPolicy

    # 创建实例
    dc = DemandPrepPolicy(
        task_type='classification',
        model_type='random_forest',
        max_truth_budget=50
    )

    # 训练（不需要干净数据）
    dc.fit(X_dirty, y, semantic_errors=[(10, 1), (25, 1)])

    # 单阶段推理
    X_clean, y_clean, stats = dc.clean(X_dirty, y, X_clean_ref)

    # 两阶段推理
    plan = dc.plan(X_dirty, y)
    X_clean = dc.execute(X_dirty, true_values)
    ```
    """

    def __init__(self,
                 task_type: Optional[str] = None,
                 model_type: Optional[str] = None,
                 agent_type: Optional[str] = None,
                 detector_mode: Optional[str] = None,
                 inference_mode: Optional[str] = None,
                 training_mode: Optional[str] = None,
                 n_episodes: Optional[int] = None,
                 repair_lambda: Optional[float] = None,
                 min_truth_budget: Optional[int] = None,
                 max_truth_budget: Optional[int] = None,
                 rules_path: Optional[str] = None,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 column_names: Optional[List[str]] = None,
                 dirty_csv_path: Optional[str] = None,
                 clean_csv_path: Optional[str] = None,
                 csv_columns: Optional[List[str]] = None,
                 label_col: Optional[str] = None,
                 save_path: Optional[str] = None,
                 apply_raha_truth: Optional[bool] = None,
                 count_raha_cost: Optional[bool] = None,
                 **kwargs):
        """
        初始化 DemandPrepPolicy

        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            model_type: 模型类型 ('svm', 'random_forest', 'xgboost', 'linear', 'ridge')
            agent_type: Agent 类型 ('single', 'two_stage', 'dueling_single', 'dueling_two_stage')
            detector_mode: 检测器模式 ('auto' 或 'oracle')
            inference_mode: 推理模式 ('single_phase' 或 'two_phase')
            training_mode: 训练模式 ('clean_base' 或 'self_supervised')
            n_episodes: 训练轮数
            repair_lambda: 修复成本系数
            min_truth_budget: 最少需要使用的真值数量
            max_truth_budget: 最多可以使用的真值数量
            rules_path: FD规则文件路径
            fd_rules: 解析后的FD规则列表
            column_names: 数据特征列名列表（不含 index/label）
            dirty_csv_path: 原始脏数据 CSV 路径（auto 模式 RAHA 需要）
            clean_csv_path: 原始干净数据 CSV 路径（auto 模式 RAHA 需要）
            csv_columns: 原始 CSV 的所有列名（含 index/label，用于 RAHA 列映射）
            label_col: 标签列名
            save_path: 输出保存路径
            apply_raha_truth: 是否将 RAHA 标注行的真值应用到数据修复
            count_raha_cost: 是否将 RAHA 的标注成本计入真值总成本
            **kwargs: 其他配置参数
        """
        # ============================================================
        # 构建 config_kwargs：仅传入非 None 的参数
        # 所有默认值由 DemandPrepPolicyConfig (config.py) 统一定义
        # ============================================================

        # 枚举映射表
        _task_type_map = {
            'classification': TaskType.CLASSIFICATION,
            'regression': TaskType.REGRESSION,
            'clustering': TaskType.CLUSTERING,
        }
        _model_type_map = {
            'svm': ModelType.SVM,
            'random_forest': ModelType.RANDOM_FOREST,
            'xgboost': ModelType.XGBOOST,
            'linear': ModelType.LINEAR,
            'ridge': ModelType.RIDGE,
            'xgboost_reg': ModelType.XGBOOST_REG,
            'kmeans': ModelType.KMEANS,
        }
        _agent_type_map = {
            'single_stage': AgentType.SINGLE_STAGE,
            'single': AgentType.SINGLE_STAGE,
            'two_stage': AgentType.TWO_STAGE,
            'dueling_single': AgentType.DUELING_SINGLE_STAGE,
            'dueling_single_stage': AgentType.DUELING_SINGLE_STAGE,
            'dueling_two_stage': AgentType.DUELING_TWO_STAGE,
        }
        _detector_mode_map = {
            'auto': DetectorMode.AUTO,
            'oracle': DetectorMode.ORACLE,
        }
        _inference_mode_map = {
            'single_phase': InferenceMode.SINGLE_PHASE,
            'two_phase': InferenceMode.TWO_PHASE,
        }

        config_kwargs: Dict[str, Any] = {}

        # 枚举类参数：仅在非 None 时转换并加入
        if task_type is not None:
            config_kwargs['task_type'] = _task_type_map.get(
                task_type.lower(), TaskType.CLASSIFICATION)
        if model_type is not None:
            config_kwargs['model_type'] = _model_type_map.get(
                model_type.lower(), ModelType.SVM)
        if agent_type is not None:
            config_kwargs['agent_type'] = _agent_type_map.get(
                agent_type.lower(), AgentType.SINGLE_STAGE)
        if detector_mode is not None:
            config_kwargs['detector_mode'] = _detector_mode_map.get(
                detector_mode.lower(), DetectorMode.AUTO)
        if inference_mode is not None:
            config_kwargs['inference_mode'] = _inference_mode_map.get(
                inference_mode.lower(), InferenceMode.SINGLE_PHASE)

        # 标量参数：仅在非 None 时加入
        _optional_params = {
            'training_mode': training_mode,
            'n_episodes': n_episodes,
            'repair_lambda': repair_lambda,
            'min_truth_budget': min_truth_budget,
            'max_truth_budget': max_truth_budget,
            'rules_path': rules_path,
            'fd_rules': fd_rules,
            'column_names': column_names,
            'label_col': label_col,
            'save_path': save_path,
            'apply_raha_truth': apply_raha_truth,
            'count_raha_cost': count_raha_cost,
        }
        for key, val in _optional_params.items():
            if val is not None:
                config_kwargs[key] = val

        # 编码工具（通过 **kwargs 传入，仅非 None 时加入）
        _encoding_params = {
            'label_encoders': kwargs.pop('encoding_label_encoders', None),
            'scaler': kwargs.pop('encoding_scaler', None),
            'categorical_cols': kwargs.pop('encoding_categorical_cols', None),
            'dirty_df': kwargs.pop('encoding_dirty_df', None),
            'clean_df': kwargs.pop('encoding_clean_df', None),
        }
        for key, val in _encoding_params.items():
            if val is not None:
                config_kwargs[key] = val

        # 提取 AutoDetector 专用参数（不传入 DemandPrepPolicyConfig）
        self._disable_raha = kwargs.pop('disable_raha', False)

        # 剩余 kwargs 直接传入（向后兼容）
        config_kwargs.update(kwargs)

        self.config = DemandPrepPolicyConfig(**config_kwargs)

        # 保存 CSV 路径参数（用于 auto 模式 RAHA 检测）
        self.dirty_csv_path = dirty_csv_path
        self.clean_csv_path = clean_csv_path
        self.csv_columns = csv_columns
        self.label_col = label_col

        # 解析 FD 规则（如有 rules_path 且未直接提供 fd_rules）
        if rules_path and not fd_rules:
            self._parse_rules(rules_path)

        # 组件
        self.trainer = Trainer(self.config)
        self.detector: Optional[Union[AutoDetector, OracleDetector]] = None
        self.agent: Optional[BaseAgent] = None
        self.logger = DemandPrepPolicyLogger(self.config)
        self.model_io = ModelIO()

        # 推理器（延迟初始化）
        self._single_phase_inference: Optional[SinglePhaseInference] = None
        self._two_phase_inference: Optional[TwoPhaseInference] = None

        # 状态
        self._is_fitted = False

        # 检测结果缓存: 避免 fit() 和 clean()/plan() 对同一数据重复检测
        self._detected_cache: Optional[Dict[str, List]] = None
        self._detected_cache_fingerprint: Optional[int] = None

    def _parse_rules(self, rules_path: str):
        """从规则文件解析 FD 规则和丰富规则"""
        try:
            parsed = load_rules(rules_path)
            # 提取 FD 对
            fd_pairs = extract_fd_pairs(parsed)
            if fd_pairs:
                self.config.fd_rules = fd_pairs
            # 丰富规则（DOMAIN/CFD/DC）
            rich_dict = rules_to_dict(parsed)
            if rich_dict.get('has_rich_rules'):
                self.config.rich_rules = rich_dict
        except Exception as e:
            print(f"  [警告] 解析规则文件失败: {e}")

    def fit(self,
            X_dirty: np.ndarray,
            y: np.ndarray,
            X_clean: Optional[np.ndarray] = None,
            y_clean: Optional[np.ndarray] = None,
            semantic_errors: Optional[List[Tuple[int, int]]] = None,
            n_episodes: Optional[int] = None,
            verbose: bool = True,
            resume_from: Optional[str] = None,
            prev_history: Optional[Dict[str, List]] = None,
            X_clean_val: Optional[np.ndarray] = None,
            y_clean_val: Optional[np.ndarray] = None,
            ) -> 'DemandPrepPolicy':
        """
        训练模型

        支持利用 RAHA 标注的行（labeling_budget 条）预修复脏数据：
        当提供 X_clean/y_clean 且检测器为 Auto 模式时，会先运行 RAHA 检测
        获取 labeled_tuples，然后将脏数据中对应行替换为干净值，提升训练数据质量。

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量
            X_clean: 干净数据（可选，用于预修复 RAHA 标注行）
            y_clean: 干净标签（可选，用于预修复 RAHA 标注行的标签）
            semantic_errors: 语义错误位置列表 [(row, col), ...]
            n_episodes: 训练轮数（默认使用配置）
            verbose: 是否打印详细信息
            resume_from: 续训模型路径（None=从头训练）
            prev_history: 之前的训练历史（续训时拼接）
            X_clean_val: 干净验证集特征（Oracle 模式用于 reward 信号）
            y_clean_val: 干净验证集标签（Oracle 模式用于 reward 信号）

        Returns:
            self
        """
        if verbose:
            self.logger.log_info("=" * 50)
            self.logger.log_info("DemandPrepPolicy 训练开始")
            self.logger.log_info(f"  检测器模式: {self.config.detector_mode.value}")
            self.logger.log_info(f"  训练模式: {self.config.training_mode}")
            self.logger.log_info(f"  Agent类型: {self.config.agent_type.value}")
            self.logger.log_info(f"  推理模式: {self.config.inference_mode.value}")
            self.logger.log_info(f"  FD规则: {len(self.config.fd_rules or [])}")
            self.logger.log_info(f"  丰富规则: {'有' if self.config.rich_rules else '无'}")
            self.logger.log_info("=" * 50)

        # 1. 创建检测器
        if verbose:
            self.logger.log_info("\n[Step 1] 初始化错误检测器...")

        if self.config.is_oracle:
            self.detector = OracleDetector(
                column_names=self.config.column_names
            )
            self.detector.fit(verbose=verbose)
        else:
            self.detector = AutoDetector(
                dirty_csv_path=self.dirty_csv_path,
                clean_csv_path=self.clean_csv_path,
                dataset_name=os.path.basename(os.path.dirname(self.dirty_csv_path))
                    if self.dirty_csv_path else "data",
                label_col=self.label_col,
                csv_columns=self.csv_columns,
                column_names=self.config.column_names,
                fd_rules=self.config.fd_rules,
                labeling_budget=20,
                rules_path=self.config.rules_path,
            )
            nan_mask = ~np.isnan(X_dirty).any(axis=1)
            X_clean_subset = X_dirty[nan_mask]
            if len(X_clean_subset) > 0:
                self.detector.fit(X_clean_subset, verbose=verbose)
            else:
                self.detector.fit(verbose=verbose)

        # 2. 检测错误（训练阶段需要 detected_errors 供 self_supervised 模式使用）
        detected_errors = None

        if isinstance(self.detector, AutoDetector) and X_clean is not None:
            if not self.detector.labeled_tuples:
                # Auto 模式 + 提供 X_clean：运行 RAHA 获取 labeled_tuples
                if verbose:
                    self.logger.log_info("\n[Step 1.5] 运行 RAHA 检测获取标注行...")
                task_type_str = self.config.task_type.value
                detected_errors = self.detector.detect(
                    X_dirty, y_dirty=y, task_type=task_type_str,
                    semantic_positions=semantic_errors, verbose=verbose
                )
                if detected_errors:
                    for key in ['missing', 'semantic', 'syntactic', 'label_noise']:
                        if key not in detected_errors:
                            detected_errors[key] = []
                    detected_errors = self._cap_detected_errors(detected_errors)
                    # 缓存检测结果，供后续 clean()/plan() 复用
                    self._detected_cache = detected_errors
                    self._detected_cache_fingerprint = self._data_fingerprint(X_dirty, y)

            # 预修复 RAHA 标注行（受 apply_raha_truth 开关控制）
            if self.config.apply_raha_truth:
                X_dirty, y = self._prefix_labeled_rows(
                    X_dirty, y, X_clean, y_clean,
                    detected_errors=detected_errors, verbose=verbose
                )
            elif verbose:
                self.logger.log_info(
                    "  [跳过预修复] apply_raha_truth=False, RAHA 标注仅用于检测"
                )

        # self_supervised 模式下，所有检测器都需要 detected_errors
        if self.config.training_mode == 'self_supervised' and detected_errors is None:
            if verbose:
                self.logger.log_info("\n[Step 1.5] self_supervised 模式: 运行检测器获取错误分布...")
            detected_errors = self.detect_errors(
                X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
                semantic_errors=semantic_errors, verbose=verbose
            )
            detected_errors = self._cap_detected_errors(detected_errors)

        # 3. 续训: 加载已有模型（兼容两阶段模型文件命名）
        start_episode = 0
        resume_agent = None
        if resume_from and self.model_io.agent_model_exists(resume_from):
            if verbose:
                self.logger.log_info(f"\n[Step 1.8] 加载续训模型: {resume_from}")
            # 通过 Trainer 创建匹配配置的 Agent，再 load 权重
            resume_agent = self.trainer._create_agent()
            resume_agent.load(resume_from)
            start_episode = resume_agent.total_episodes
            if verbose:
                self.logger.log_info(
                    f"  已训练 {start_episode} episodes, "
                    f"best_score={resume_agent.best_score:.4f}, "
                    f"epsilon={resume_agent.epsilon:.4f}"
                )

        # 4. 训练 DQN Agent
        if verbose:
            self.logger.log_info("\n[Step 2] 训练 DQN Agent...")

        self.agent, history = self.trainer.train(
            X_dirty, y,
            n_episodes=n_episodes,
            verbose=verbose,
            detected_errors=detected_errors,
            start_episode=start_episode,
            prev_history=prev_history,
            agent=resume_agent,
            X_clean_val=X_clean_val,
            y_clean_val=y_clean_val,
        )

        self._is_fitted = True

        if verbose:
            self.logger.log_info("\n训练完成!")

        return self

    def clean(self,
              X_dirty: np.ndarray,
              y: np.ndarray,
              X_clean: np.ndarray,
              y_clean: Optional[np.ndarray] = None,
              semantic_errors: Optional[List[Tuple[int, int]]] = None,
              pre_detected: Optional[Dict[str, List]] = None,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        单阶段推理：直接清洗数据

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量（脏标签）
            X_clean: 干净数据（用于获取真值修复）
            y_clean: 干净标签向量（用于标签噪声检测和修复，可选）
            semantic_errors: 语义错误位置列表 [(row, col), ...]
            pre_detected: 预先提供的检测结果，跳过内置检测器（消融实验用，确保与基线公平比较）
            verbose: 是否打印详细信息

        Returns:
            (X_clean_result, y_clean_result, stats)
        """
        self._check_fitted()

        if verbose:
            self.logger.log_info("\n" + "=" * 50)
            self.logger.log_info("单阶段推理")
            self.logger.log_info("=" * 50)

        if pre_detected is not None:
            # 使用外部提供的检测结果（绕过 OracleDetector）
            detected = pre_detected
        else:
            # 检测错误（含标签噪声）
            detected = self.detect_errors(
                X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
                semantic_errors=semantic_errors, verbose=verbose
            )

        # 预修复 RAHA 标注行 + 从 detected 中移除已修复行的错误（受开关控制）
        if self.config.apply_raha_truth:
            X_dirty, y, detected = self._prefix_labeled_rows_for_inference(
                X_dirty, y, X_clean, y_clean, detected, verbose=verbose
            )

        # 创建推理器
        if self._single_phase_inference is None:
            self._single_phase_inference = SinglePhaseInference(self.agent, self.config)

        # 执行清洗
        X_result, y_result, keep_mask, action_counts, repair_log = \
            self._single_phase_inference.clean(
                X_dirty, y, X_clean, detected, verbose, y_clean=y_clean
            )

        stats = {
            'action_counts': action_counts,
            'repair_log': repair_log,
            'keep_mask': keep_mask,
            'truth_cost': action_counts['repair_value'],
            'deleted_count': action_counts['delete']
        }

        return X_result, y_result, stats

    def plan(self,
             X_dirty: np.ndarray,
             y: np.ndarray,
             X_clean: Optional[np.ndarray] = None,
             y_clean: Optional[np.ndarray] = None,
             semantic_errors: Optional[List[Tuple[int, int]]] = None,
             pre_detected: Optional[Dict[str, List]] = None,
             verbose: bool = True) -> List[Dict]:
        """
        两阶段推理 - 第一阶段：生成修复计划

        不需要真值，返回需要修复的位置列表。
        Oracle 模式下必须提供 X_clean 用于错误检测。

        Args:
            X_dirty: 脏数据矩阵
            y: 标签向量（脏标签）
            X_clean: 干净数据（Oracle 模式必须提供）
            y_clean: 干净标签向量（用于标签噪声检测，可选）
            semantic_errors: 语义错误位置列表
            pre_detected: 预先提供的检测结果，跳过内置检测器（用于外部单独计时）
            verbose: 是否打印详细信息

        Returns:
            repair_plan: 需要真值修复的位置列表
        """
        self._check_fitted()

        if verbose:
            self.logger.log_info("\n" + "=" * 50)
            self.logger.log_info("两阶段推理 - 第一阶段 (Plan)")
            self.logger.log_info("=" * 50)

        if pre_detected is not None:
            detected = pre_detected
        else:
            # 检测错误（含标签噪声，Oracle 模式需要 X_clean / y_clean）
            detected = self.detect_errors(
                X_dirty, X_clean, y_dirty=y, y_clean=y_clean,
                semantic_errors=semantic_errors, verbose=verbose
            )

        # 创建推理器
        if self._two_phase_inference is None:
            self._two_phase_inference = TwoPhaseInference(self.agent, self.config)

        # 生成计划
        repair_plan = self._two_phase_inference.plan(X_dirty, y, detected, verbose)

        return repair_plan

    def get_plan_positions(self) -> List[Tuple[int, int]]:
        """获取需要真值的位置列表"""
        if self._two_phase_inference is None:
            return []
        return self._two_phase_inference.get_plan_positions()

    def execute(self,
                X_dirty: np.ndarray,
                true_values: Dict[Tuple[int, int], float],
                verbose: bool = True,
                y_dirty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        两阶段推理 - 第二阶段：执行修复

        Args:
            X_dirty: 原始脏数据
            true_values: 真值字典 {(idx, col): value}
            verbose: 是否打印详细信息
            y_dirty: 原始脏标签（标签修复时需要）

        Returns:
            (X_clean, y_clean, keep_mask)
        """
        if self._two_phase_inference is None:
            raise ValueError("请先调用 plan() 方法生成修复计划")

        return self._two_phase_inference.execute(X_dirty, true_values, verbose, y_dirty=y_dirty)

    @staticmethod
    def _data_fingerprint(X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """计算数据指纹（用于检测缓存命中判断）

        基于形状 + 前/后/随机采样值的哈希，O(1) 且碰撞概率极低。
        """
        parts = [X.shape]
        # 头 / 尾行
        if len(X) > 0:
            parts.append(X[0].tobytes())
            parts.append(X[-1].tobytes())
        if len(X) > 2:
            parts.append(X[len(X) // 2].tobytes())
        if y is not None:
            parts.append(y.shape)
            if len(y) > 0:
                parts.append(y[0])
                parts.append(y[-1])
        return hash(tuple(str(p) for p in parts))

    def detect_errors(self,
                      X_dirty: np.ndarray,
                      X_clean: Optional[np.ndarray] = None,
                      y_dirty: Optional[np.ndarray] = None,
                      y_clean: Optional[np.ndarray] = None,
                      semantic_errors: Optional[List[Tuple[int, int]]] = None,
                      verbose: bool = True) -> Dict[str, List]:
        """
        检测错误

        Oracle 模式下直接对比 dirty/clean（含特征 + 标签）；
        Auto 模式下使用 RAHA(原始CSV) + FD + Confident Learning。

        Args:
            X_dirty: 脏数据
            X_clean: 干净数据（Oracle 模式必须提供，Auto 模式可选）
            y_dirty: 脏标签向量
            y_clean: 干净标签向量（Oracle 模式用于标签噪声检测）
            semantic_errors: 语义错误位置列表（Auto 模式且无FD规则时使用）
            verbose: 是否打印详细信息

        Returns:
            detected: {'missing': [...], 'semantic': [...], 'syntactic': [...], 'label_noise': [...]}
        """
        # 缓存命中判断: 同一数据不重复检测
        fingerprint = self._data_fingerprint(X_dirty, y_dirty)
        if (self._detected_cache is not None
                and self._detected_cache_fingerprint == fingerprint):
            if verbose:
                total = sum(len(v) for v in self._detected_cache.values())
                print(f"  [检测缓存命中] 复用上次检测结果 ({total} cells)")
            # 返回深拷贝，防止调用方修改（如 _prefix_labeled_rows_for_inference）影响缓存
            return {k: list(v) for k, v in self._detected_cache.items()}

        if self.detector is None:
            if self.config.is_oracle:
                self.detector = OracleDetector(column_names=self.config.column_names)
            else:
                self.detector = AutoDetector(
                    dirty_csv_path=self.dirty_csv_path,
                    clean_csv_path=self.clean_csv_path,
                    dataset_name="data",
                    label_col=self.label_col,
                    csv_columns=self.csv_columns,
                    column_names=self.config.column_names,
                    fd_rules=self.config.fd_rules,
                    disable_raha=self._disable_raha,
                )
                self.detector._compute_col_stats(X_dirty)

        if isinstance(self.detector, OracleDetector):
            if X_clean is None:
                raise ValueError("Oracle 模式下必须提供 X_clean")
            detected = self.detector.detect(
                X_dirty, X_clean,
                y_dirty=y_dirty, y_clean=y_clean,
                verbose=verbose
            )
        else:
            task_type_str = self.config.task_type.value if hasattr(self.config.task_type, 'value') else str(self.config.task_type)
            detected = self.detector.detect(
                X_dirty,
                y_dirty=y_dirty,
                task_type=task_type_str,
                semantic_positions=semantic_errors,
                verbose=verbose
            )

        # 确保所有类型的错误列表都存在
        for key in ['missing', 'semantic', 'syntactic', 'label_noise']:
            if key not in detected:
                detected[key] = []
        detected = self._cap_detected_errors(detected)

        # 写入缓存
        self._detected_cache = detected
        self._detected_cache_fingerprint = fingerprint

        return detected

    def _cap_detected_errors(self, detected: Dict[str, List]) -> Dict[str, List]:
        """Stratified deterministic cap for large detected-error sets."""
        max_total = int(getattr(self.config, 'max_detected_errors', 0) or 0)
        if max_total <= 0:
            return detected
        keys = ['missing', 'semantic', 'syntactic', 'label_noise']
        counts = {key: len(detected.get(key, [])) for key in keys}
        total = sum(counts.values())
        if total <= max_total:
            return detected

        quotas: Dict[str, int] = {}
        remainders = []
        assigned = 0
        for key in keys:
            count = counts[key]
            if count == 0:
                quotas[key] = 0
                continue
            raw = max_total * count / total
            quota = max(1, int(raw))
            quota = min(quota, count)
            quotas[key] = quota
            assigned += quota
            remainders.append((raw - int(raw), key))

        while assigned > max_total:
            candidates = [key for key in keys if quotas.get(key, 0) > 1]
            if not candidates:
                break
            key = min(candidates, key=lambda k: quotas[k] / max(counts[k], 1))
            quotas[key] -= 1
            assigned -= 1

        for _, key in sorted(remainders, reverse=True):
            if assigned >= max_total:
                break
            if quotas.get(key, 0) < counts[key]:
                quotas[key] += 1
                assigned += 1

        capped = {}
        for key in keys:
            values = list(detected.get(key, []))
            quota = quotas.get(key, 0)
            capped[key] = self._even_sample(values, quota) if quota and len(values) > quota else values
        return capped

    @staticmethod
    def _even_sample(values: List, quota: int) -> List:
        if quota <= 0:
            return []
        if len(values) <= quota:
            return values
        if quota == 1:
            return [values[0]]
        step = (len(values) - 1) / (quota - 1)
        return [values[round(i * step)] for i in range(quota)]

    def save(self,
             model_path: str,
             detector_path: Optional[str] = None) -> None:
        """
        保存模型

        Args:
            model_path: Agent 模型路径
            detector_path: 检测器路径（可选）
        """
        self._check_fitted()

        # 保存 Agent
        self.model_io.save_agent(self.agent, model_path)

        # 保存检测器
        if detector_path and self.detector:
            self.detector.save(detector_path)

        self.logger.log_info(f"模型已保存: {model_path}")

    def load(self,
             model_path: str,
             detector_path: Optional[str] = None) -> 'DemandPrepPolicy':
        """
        加载模型

        Args:
            model_path: Agent 模型路径
            detector_path: 检测器路径（可选）

        Returns:
            self
        """
        # 加载 Agent
        _AGENT_CLASS_MAP = {
            AgentType.SINGLE_STAGE: SingleStageDQNAgent,
            AgentType.TWO_STAGE: TwoStageDQNAgent,
            AgentType.DUELING_SINGLE_STAGE: DuelingSingleStageAgent,
            AgentType.DUELING_TWO_STAGE: DuelingTwoStageAgent,
        }
        agent_cls = _AGENT_CLASS_MAP.get(self.config.agent_type, SingleStageDQNAgent)
        self.agent = self.model_io.load_agent(agent_cls, model_path)

        # 加载检测器
        if detector_path and os.path.exists(detector_path):
            self.detector = AutoDetector.load(detector_path)

        self._is_fitted = True
        self.logger.log_info(f"模型已加载: {model_path}")

        return self

    def _get_labeled_tuples(self) -> Set[int]:
        """获取 RAHA 标注的行索引集合"""
        if isinstance(self.detector, AutoDetector) and self.detector.labeled_tuples:
            return self.detector.labeled_tuples
        return set()

    def _prefix_labeled_rows(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: Optional[np.ndarray],
        y_clean: Optional[np.ndarray],
        detected_errors: Optional[Dict[str, List]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用 RAHA 标注行的干净值预修复脏数据（训练阶段）

        只替换标注行中被检测到错误的单元格，不替换整行。

        Returns:
            (X_dirty_prefixed, y_prefixed)  —— 副本，不修改原数组
        """
        labeled_indices = self._get_labeled_tuples()
        if not labeled_indices or X_clean is None:
            return X_dirty, y

        # 从 detected_errors 提取所有错误单元格 (idx, col)
        error_cells: Set[Tuple[int, int]] = set()
        label_error_rows: Set[int] = set()
        if detected_errors:
            for key in ['missing', 'semantic', 'syntactic']:
                for item in detected_errors.get(key, []):
                    error_cells.add((item[0], item[1]))
            for item in detected_errors.get('label_noise', []):
                label_error_rows.add(item[0])

        X_out = X_dirty.copy()
        y_out = y.copy()
        fixed_feature_cells = 0
        fixed_label_cells = 0

        for idx in sorted(labeled_indices):
            if idx >= len(X_out) or idx >= len(X_clean):
                continue
            # 只替换该行中被检测为错误的特征单元格
            for col in range(X_out.shape[1]):
                if (idx, col) in error_cells:
                    X_out[idx, col] = X_clean[idx, col]
                    fixed_feature_cells += 1
            # 标签错误：只替换被检测为标签噪声的行
            if idx in label_error_rows and y_clean is not None and idx < len(y_clean):
                y_out[idx] = y_clean[idx]
                fixed_label_cells += 1

        total_fixed = fixed_feature_cells + fixed_label_cells
        if verbose and total_fixed > 0:
            self.logger.log_info(
                f"  [预修复] 利用 RAHA 标注行修复 {fixed_feature_cells} 个特征单元格"
                f" + {fixed_label_cells} 个标签"
            )
            self.logger.log_info(
                f"  标注行: {sorted(labeled_indices)[:10]}"
                + (f"... (共 {len(labeled_indices)} 行)" if len(labeled_indices) > 10 else "")
            )

        return X_out, y_out

    def _prefix_labeled_rows_for_inference(
        self,
        X_dirty: np.ndarray,
        y: np.ndarray,
        X_clean: Optional[np.ndarray],
        y_clean: Optional[np.ndarray],
        detected: Dict[str, List],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """
        利用 RAHA 标注行的干净值预修复脏数据（推理阶段）
        只替换标注行中被检测到错误的单元格，同时从 detected_errors 中移除已修复的错误。

        Returns:
            (X_dirty_prefixed, y_prefixed, detected_filtered)
        """
        labeled_indices = self._get_labeled_tuples()
        if not labeled_indices or X_clean is None:
            return X_dirty, y, detected

        # 从 detected 提取所有错误单元格 (idx, col) 和标签错误行
        error_cells: Set[Tuple[int, int]] = set()
        label_error_rows: Set[int] = set()
        for key in ['missing', 'semantic', 'syntactic']:
            for item in detected.get(key, []):
                error_cells.add((item[0], item[1]))
        for item in detected.get('label_noise', []):
            label_error_rows.add(item[0])

        X_out = X_dirty.copy()
        y_out = y.copy()
        fixed_feature_cells: Set[Tuple[int, int]] = set()
        fixed_label_rows: Set[int] = set()

        for idx in sorted(labeled_indices):
            if idx >= len(X_out) or idx >= len(X_clean):
                continue
            # 只替换该行中被检测为错误的特征单元格
            for col in range(X_out.shape[1]):
                if (idx, col) in error_cells:
                    X_out[idx, col] = X_clean[idx, col]
                    fixed_feature_cells.add((idx, col))
            # 标签错误
            if idx in label_error_rows and y_clean is not None and idx < len(y_clean):
                y_out[idx] = y_clean[idx]
                fixed_label_rows.add(idx)

        # 从 detected_errors 中移除已修复的单元格
        removed_counts = {}
        detected_filtered = {}
        for key in ['missing', 'semantic', 'syntactic']:
            original = detected.get(key, [])
            filtered = [e for e in original if (e[0], e[1]) not in fixed_feature_cells]
            detected_filtered[key] = filtered
            removed = len(original) - len(filtered)
            if removed > 0:
                removed_counts[key] = removed
        # 标签噪声：移除已修复行
        original_label = detected.get('label_noise', [])
        filtered_label = [e for e in original_label if e[0] not in fixed_label_rows]
        detected_filtered['label_noise'] = filtered_label
        removed_label = len(original_label) - len(filtered_label)
        if removed_label > 0:
            removed_counts['label_noise'] = removed_label

        total_fixed = len(fixed_feature_cells) + len(fixed_label_rows)
        if verbose and total_fixed > 0:
            self.logger.log_info(
                f"  [预修复] 利用 RAHA 标注行修复 {len(fixed_feature_cells)} 个特征单元格"
                f" + {len(fixed_label_rows)} 个标签"
            )
            if removed_counts:
                parts = [f"{k}={v}" for k, v in removed_counts.items()]
                self.logger.log_info(
                    f"  移除已修复错误: {', '.join(parts)}"
                )

        return X_out, y_out, detected_filtered

    def _check_fitted(self) -> None:
        """检查是否已训练"""
        if not self._is_fitted or self.agent is None:
            raise ValueError("模型未训练或加载，请先调用 fit() 或 load()")

    def get_config(self) -> DemandPrepPolicyConfig:
        """获取配置"""
        return self.config

    def get_training_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return self.trainer.get_history()

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted
