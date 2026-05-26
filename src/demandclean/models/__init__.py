"""模型适配器模块"""

from .base_adapter import ModelAdapter
from .classification import SVMAdapter, RandomForestAdapter, XGBoostClassifierAdapter
from .regression import LinearAdapter, RidgeAdapter, XGBoostRegressorAdapter, RandomForestRegressorAdapter
from .clustering import KMeansAdapter

__all__ = [
    'ModelAdapter',
    'SVMAdapter',
    'RandomForestAdapter',
    'XGBoostClassifierAdapter',
    'LinearAdapter',
    'RidgeAdapter',
    'XGBoostRegressorAdapter',
    'RandomForestRegressorAdapter',
    'KMeansAdapter',
    'get_adapter',
    'create_model_adapter',
]


def get_adapter(model_type: str, task_type: str = 'classification', **kwargs):
    """
    根据模型类型获取适配器

    Args:
        model_type: 模型类型 ('svm', 'random_forest', 'xgboost', 'linear', 'ridge', 'xgboost_reg', 'kmeans')
        task_type: 任务类型 ('classification', 'regression', 'clustering')
        **kwargs: 传递给适配器的参数

    Returns:
        ModelAdapter 实例
    """
    adapters = {
        # 分类
        ('svm', 'classification'): SVMAdapter,
        ('random_forest', 'classification'): RandomForestAdapter,
        ('xgboost', 'classification'): XGBoostClassifierAdapter,
        # 回归
        ('linear', 'regression'): LinearAdapter,
        ('ridge', 'regression'): RidgeAdapter,
        ('xgboost_reg', 'regression'): XGBoostRegressorAdapter,
        ('random_forest', 'regression'): RandomForestRegressorAdapter,
        # 聚类
        ('kmeans', 'clustering'): KMeansAdapter,
    }

    key = (model_type, task_type)
    if key not in adapters:
        raise ValueError(f"不支持的模型类型: {model_type} for {task_type}")

    return adapters[key](**kwargs)


def create_model_adapter(model_type, task_type, **kwargs):
    """
    根据枚举类型创建模型适配器

    Args:
        model_type: ModelType 枚举或字符串
        task_type: TaskType 枚举或字符串
        **kwargs: 传递给适配器的参数

    Returns:
        ModelAdapter 实例
    """
    from ..config import ModelType, TaskType

    # 处理枚举类型
    if hasattr(model_type, 'value'):
        model_type_str = model_type.value
    else:
        model_type_str = str(model_type)

    if hasattr(task_type, 'value'):
        task_type_str = task_type.value
    else:
        task_type_str = str(task_type)

    # 模型类型映射
    model_map = {
        'svm': 'svm',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'linear': 'linear',
        'ridge': 'ridge',
        'xgboost_reg': 'xgboost_reg',
        'kmeans': 'kmeans',
    }

    model_key = model_map.get(model_type_str, model_type_str)

    return get_adapter(model_key, task_type_str, **kwargs)

