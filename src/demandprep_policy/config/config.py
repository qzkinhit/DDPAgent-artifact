"""
DemandPrepPolicy 配置管理
====================

定义系统配置的数据类和枚举类型。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import os


class TaskType(Enum):
    """任务类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class ModelType(Enum):
    """模型类型枚举"""
    # 分类模型
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    # 回归模型
    LINEAR = "linear"
    RIDGE = "ridge"
    XGBOOST_REG = "xgboost_reg"
    # 聚类模型
    KMEANS = "kmeans"


class AgentType(Enum):
    """DQN Agent 类型枚举"""
    SINGLE_STAGE = "single"
    TWO_STAGE = "two_stage"
    DUELING_SINGLE_STAGE = "dueling_single"
    DUELING_TWO_STAGE = "dueling_two_stage"


class DetectorMode(Enum):
    """检测器模式枚举"""
    AUTO = "auto"       # 自动检测（FD + RAHA + isnan）
    ORACLE = "oracle"   # 外部提供完整错误标签（消融实验用）


class InferenceMode(Enum):
    """推理模式枚举"""
    SINGLE_PHASE = "single_phase"
    TWO_PHASE = "two_phase"


@dataclass
class DemandPrepPolicyConfig:
    """
    DemandPrepPolicy 主配置类

    Attributes:
        task_type: 任务类型（分类/回归）
        model_type: 模型类型
        agent_type: DQN Agent 类型（单阶段/两阶段）
        detector_mode: 检测器模式（auto/oracle）
        inference_mode: 推理模式（single_phase/two_phase）

        n_episodes: 训练轮数
        repair_lambda: 真值修复的成本系数
        min_truth_budget: 最少使用真值数量
        max_truth_budget: 最多使用真值数量

        state_size: 状态向量维度
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_min: 最小探索率
        epsilon_decay: 探索率衰减
        learning_rate: 学习率
        batch_size: 批大小
        memory_size: 经验回放缓冲区大小

        missing_rate_range: 缺失值注入率范围
        semantic_rate_range: 语义错误注入率范围
        syntactic_rate_range: 句法错误注入率范围

        rules_path: 规则文件路径（data/{dataset}/rules.txt）
        fd_rules: 解析后的FD规则列表 [(lhs_str, rhs_str), ...]
        column_names: 数据列名列表

        save_path: 输出保存路径
        model_path: 模型保存路径
        detector_path: 检测器保存路径

        verbose: 是否输出详细日志
        log_interval: 日志输出间隔（每多少轮）
    """
    # 任务配置
    task_type: TaskType = TaskType.CLASSIFICATION
    model_type: ModelType = ModelType.SVM
    agent_type: AgentType = AgentType.SINGLE_STAGE
    detector_mode: DetectorMode = DetectorMode.AUTO
    inference_mode: InferenceMode = InferenceMode.SINGLE_PHASE

    # 训练配置
    n_episodes: int = 300
    repair_lambda: float = 0.03
    min_truth_budget: Optional[int] = None   # deprecated: 使用 min_repair_ratio 替代
    max_truth_budget: Optional[int] = None   # deprecated: 使用 max_repair_ratio 替代

    # 修复率控制（仅 max_repair_count 仍生效，ratio_penalty 和 dynamic_modifier 已移除）
    min_repair_ratio: float = 0.1       # deprecated: 不再用于 ratio_penalty
    max_repair_ratio: float = 0.3       # 修复预算上限: max_repair_count = int(n_errors * ratio)
    repair_sensitivity: float = 10.0    # deprecated: dynamic_modifier 已移除

    # Oracle 模式
    use_clean_validation: bool = True    # 用干净验证集做 reward 信号

    # DQN 配置
    state_size: int = 10  # 8 error-level + 2 global (remaining_budget_ratio, remaining_errors_ratio)
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    learning_rate: float = 0.0005
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 5  # 目标网络更新频率

    # 错误注入配置
    missing_rate_range: Tuple[float, float] = (0.02, 0.08)
    semantic_rate_range: Tuple[float, float] = (0.05, 0.15)
    syntactic_rate_range: Tuple[float, float] = (0.1, 0.25)
    label_rate_range: Tuple[float, float] = (0.0, 0.05)  # 默认 0, 由检测结果决定是否启用

    # FD规则配置
    rules_path: Optional[str] = None
    fd_rules: Optional[List[Tuple[str, str]]] = None
    column_names: Optional[List[str]] = None
    rich_rules: Optional[Dict[str, Any]] = None  # 丰富规则字典（DOMAIN/CFD/DC）

    # 训练模式
    training_mode: str = "clean_base"  # "clean_base" 或 "self_supervised"

    # 编码工具（用于 ErrorInjector 在 CSV 空间注入后重新编码）
    label_encoders: Optional[Dict[str, Any]] = None    # {col_name: LabelEncoder}
    scaler: Optional[Any] = None                        # StandardScaler
    categorical_cols: Optional[set] = None              # 分类列集合
    dirty_df: Optional[Any] = None                      # 原始 CSV dirty DataFrame
    clean_df: Optional[Any] = None                      # 原始 CSV clean DataFrame
    label_col: Optional[str] = None                     # 标签列名
    protected_cols: Optional[set] = None                # 受保护列名（排除句法注入）

    # Clean Base 自动选择
    auto_select_base: bool = True        # 是否自动选择最优 clean base 策略（DeleteFix vs VE-Fill）
    base_cv_folds: int = 5              # CV 折数
    max_detected_errors: int = 0         # 0=不限制；>0 时分层采样 detected error list

    # 路径配置
    save_path: str = "output"
    model_path: Optional[str] = None
    detector_path: Optional[str] = None

    # 日志配置
    verbose: bool = True
    log_interval: int = 50

    # RAHA 检测器配置
    raha_n_runs: int = 15
    raha_threshold_m: int = 4
    semantic_rate: float = 0.9  # 语义错误检测率（用于模拟）

    # 特征重要性刷新间隔（None 表示由环境自动计算 = max(20, n_errors // 10)）
    importance_refresh_interval: Optional[int] = None

    # Reward 评估配置
    reward_eval_interval: int = 0          # 0=自适应（按数据规模自动计算），>0 手动指定
    eval_sample_ratio: float = 1.0         # 验证集采样比例 (1.0 = 全量, 0.3 = 30% 采样)
    model_kwargs: Optional[Dict[str, Any]] = None  # 传递给模型适配器的额外参数 (如 n_estimators=10)

    # Shaping 衰减配置
    shaping_warmup_ratio: float = 0.5      # 前 50% episode 保持全量 shaping
    shaping_min_weight: float = 0.1        # 衰减下限（不完全关闭，保留微弱引导）

    # Reward 差异化参数（按任务类型设置，回归任务用优化值，分类/聚类用默认值）
    delete_shaping_reward: float = -0.02       # delete 的 shaping reward（回归建议 -0.05）
    keep_rate_weight: float = 0.2              # final reward 中 keep_rate 权重（回归建议 1.0）
    regression_log_normalize: bool = False     # 回归任务是否用 log 压缩归一化 1/(1+log(1+MSE))

    # RAHA 真值开关
    apply_raha_truth: bool = True   # 是否将 RAHA 标注行的真值应用到数据修复
    count_raha_cost: bool = True    # 是否将 RAHA 的标注成本计入真值总成本

    def __post_init__(self):
        """初始化后处理"""
        # 确保 save_path 存在
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # 字符串转枚举
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        if isinstance(self.agent_type, str):
            self.agent_type = AgentType(self.agent_type)
        if isinstance(self.detector_mode, str):
            self.detector_mode = DetectorMode(self.detector_mode)
        if isinstance(self.inference_mode, str):
            self.inference_mode = InferenceMode(self.inference_mode)

    @property
    def is_classification(self) -> bool:
        """是否为分类任务"""
        return self.task_type == TaskType.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """是否为回归任务"""
        return self.task_type == TaskType.REGRESSION

    @property
    def is_clustering(self) -> bool:
        """是否为聚类任务"""
        return self.task_type == TaskType.CLUSTERING

    @property
    def is_oracle(self) -> bool:
        """是否使用 Oracle 检测器"""
        return self.detector_mode == DetectorMode.ORACLE

    @property
    def is_two_phase(self) -> bool:
        """是否使用两阶段推理"""
        return self.inference_mode == InferenceMode.TWO_PHASE

    @property
    def epsilon(self) -> float:
        """探索率 (alias for epsilon_start)"""
        return self.epsilon_start

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_type': self.task_type.value,
            'model_type': self.model_type.value,
            'agent_type': self.agent_type.value,
            'detector_mode': self.detector_mode.value,
            'inference_mode': self.inference_mode.value,
            'n_episodes': self.n_episodes,
            'repair_lambda': self.repair_lambda,
            'min_truth_budget': self.min_truth_budget,
            'max_truth_budget': self.max_truth_budget,
            'state_size': self.state_size,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'rules_path': self.rules_path,
            'save_path': self.save_path,
            'verbose': self.verbose,
            'apply_raha_truth': self.apply_raha_truth,
            'count_raha_cost': self.count_raha_cost,
            'reward_eval_interval': self.reward_eval_interval,
            'eval_sample_ratio': self.eval_sample_ratio,
            'model_kwargs': self.model_kwargs,
            'auto_select_base': self.auto_select_base,
            'base_cv_folds': self.base_cv_folds,
            'max_detected_errors': self.max_detected_errors,
            'min_repair_ratio': self.min_repair_ratio,
            'max_repair_ratio': self.max_repair_ratio,
            'repair_sensitivity': self.repair_sensitivity,
            'use_clean_validation': self.use_clean_validation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DemandPrepPolicyConfig':
        """从字典创建"""
        return cls(**d)

    def copy(self, **updates) -> 'DemandPrepPolicyConfig':
        """创建副本，可选地更新部分字段"""
        d = self.to_dict()
        d.update(updates)
        return self.from_dict(d)
