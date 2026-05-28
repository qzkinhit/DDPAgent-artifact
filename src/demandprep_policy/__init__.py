"""
DemandPrepPolicy - 按需数据清洗系统
==============================

基于深度强化学习的自监督数据清洗框架。

核心特性:
    - 训练时不接触干净数据，完全自监督学习
    - 支持分类和回归任务
    - 支持多种机器学习模型（SVM、RandomForest、XGBoost等）
    - 提供单阶段和两阶段推理模式

使用示例:
    >>> from demandprep_policy import DemandPrepPolicy
    >>>
    >>> # 创建实例
    >>> dc = DemandPrepPolicy(
    ...     task_type='classification',
    ...     model_type='random_forest',
    ...     max_truth_budget=50
    ... )
    >>>
    >>> # 训练（不需要干净数据）
    >>> dc.fit(X_dirty, y, semantic_errors=[(10, 1), (25, 1)])
    >>>
    >>> # 单阶段推理
    >>> X_clean, y_clean, stats = dc.clean(X_dirty, y, X_clean_ref)
    >>>
    >>> # 或两阶段推理
    >>> plan = dc.plan(X_dirty, y)
    >>> X_clean = dc.execute(X_dirty, plan, true_values)
"""

__version__ = '1.0.0'
__author__ = 'DemandPrepPolicy Team'

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == 'DemandPrepPolicy':
        from .api.demand_prep_policy import DemandPrepPolicy
        return DemandPrepPolicy
    elif name == 'DemandPrepPolicyConfig':
        from .config.config import DemandPrepPolicyConfig
        return DemandPrepPolicyConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DemandPrepPolicy',
    'DemandPrepPolicyConfig',
]
