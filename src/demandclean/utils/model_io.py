"""
模型 I/O 工具 (PyTorch)
=======================

DQN Agent 和检测器的保存/加载功能。
"""

import os
import pickle
from typing import Any, Optional, Type
import warnings
import torch


class ModelIO:
    """模型保存和加载工具类"""

    @staticmethod
    def agent_model_exists(path: str) -> bool:
        """检查 Agent 模型文件是否存在（兼容单阶段/两阶段）

        单阶段 Agent: 文件直接保存在 {base}.pt
        两阶段 Agent: 文件保存在 {base}_stage1.pt + {base}_stage2.pt

        Args:
            path: 模型路径（调用方传入的 base path）

        Returns:
            True 如果找到单阶段 .pt 文件或两阶段 _stage1.pt 文件
        """
        pt_path = path.replace('.h5', '.pt')
        # 单阶段: 直接检查 .pt 文件
        if os.path.exists(pt_path):
            return True
        # 两阶段: 检查 _stage1.pt
        base = pt_path.replace('.pt', '')
        stage1_path = base + '_stage1.pt'
        return os.path.exists(stage1_path)

    @staticmethod
    def is_two_stage_model(path: str) -> bool:
        """判断路径对应的模型是否为两阶段（通过文件存在性推断）

        Returns:
            True 如果 _stage1.pt 存在且 base .pt 不存在
        """
        pt_path = path.replace('.h5', '.pt')
        if os.path.exists(pt_path):
            return False  # base .pt 存在 → 单阶段
        base = pt_path.replace('.pt', '')
        return os.path.exists(base + '_stage1.pt')

    @staticmethod
    def save_agent(agent: Any, path: str) -> None:
        """
        保存 DQN Agent (.pt 格式)

        Args:
            agent: DQN Agent 实例
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        agent.save(path)

    @staticmethod
    def load_agent(agent_class: Type, path: str, **kwargs) -> Any:
        """
        加载 DQN Agent

        从 checkpoint 自动提取 state_size / action_size 等初始化参数。
        外部传入的 kwargs 优先级更高。

        兼容两阶段模型: 当 base .pt 不存在但 _stage1.pt 存在时，
        从 _stage1.pt 读取元数据。

        Args:
            agent_class: Agent 类
            path: 模型路径
            **kwargs: Agent 初始化参数（可覆盖 checkpoint 中保存的值）
        """
        pt_path = path.replace('.h5', '.pt')
        base = pt_path.replace('.pt', '')

        # 确定实际的 checkpoint 路径（用于提取初始化参数）
        if os.path.exists(pt_path):
            ckpt_path = pt_path
        elif os.path.exists(base + '_stage1.pt'):
            ckpt_path = base + '_stage1.pt'
        else:
            raise FileNotFoundError(
                f"Agent 模型文件不存在: {pt_path} 或 {base}_stage1.pt")

        # 读 checkpoint 获取构建参数
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        init_kwargs = {}
        for key in ('state_size', 'action_size'):
            if key in checkpoint:
                init_kwargs[key] = checkpoint[key]

        # 向后兼容：旧 checkpoint 未存 state_size，从第一层 Linear 权重 shape 推断
        if 'state_size' not in init_kwargs and 'model_state' in checkpoint:
            for k, v in checkpoint['model_state'].items():
                if 'weight' in k and hasattr(v, 'dim') and v.dim() == 2:
                    init_kwargs['state_size'] = v.shape[1]
                    print(f"[load_agent] state_size 从 weight 推断 = {v.shape[1]} (key={k})")
                    break

        init_kwargs.update(kwargs)

        agent = agent_class(**init_kwargs)
        agent.load(pt_path)
        print(f"Agent 已加载: {pt_path}")
        return agent

    @staticmethod
    def save_detector(detector: Any, path: str) -> None:
        """
        保存错误检测器

        Args:
            detector: 检测器实例
            path: 保存路径 (.pkl)
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(detector, f)
        print(f"检测器已保存: {path}")

    @staticmethod
    def load_detector(path: str) -> Any:
        """
        加载错误检测器

        Args:
            path: 检测器文件路径 (.pkl)

        Returns:
            检测器实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"检测器文件不存在: {path}")

        with open(path, 'rb') as f:
            detector = pickle.load(f)
        print(f"检测器已加载: {path}")
        return detector

    @staticmethod
    def save_config(config: Any, path: str) -> None:
        """
        保存配置

        Args:
            config: 配置对象
            path: 保存路径 (.json)
        """
        import json
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"配置已保存: {path}")

    @staticmethod
    def load_config(config_class: Type, path: str) -> Any:
        """
        加载配置

        Args:
            config_class: 配置类
            path: 配置文件路径 (.json)

        Returns:
            配置对象
        """
        import json

        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        if hasattr(config_class, 'from_dict'):
            return config_class.from_dict(config_dict)
        return config_class(**config_dict)

    @staticmethod
    def exists(path: str) -> bool:
        """检查文件是否存在（兼容两阶段模型）"""
        return ModelIO.agent_model_exists(path)

    @staticmethod
    def ensure_dir(path: str) -> str:
        """确保目录存在，返回目录路径"""
        dir_path = os.path.dirname(path) or '.'
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
