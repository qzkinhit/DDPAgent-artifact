"""
日志工具
========

统一的日志管理和训练历史记录。
"""

import logging
import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional


class DemandPrepPolicyLogger:
    """
    DemandPrepPolicy 日志管理器

    提供:
        - 控制台和文件日志输出
        - 训练历史记录
        - JSON 格式的历史导出
    """

    def __init__(self,
                 config_or_name = "demandprep_policy",
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 to_file: bool = True,
                 to_console: bool = True):
        """
        初始化日志管理器

        Args:
            config_or_name: 配置对象或日志名称字符串
            log_dir: 日志文件目录
            level: 日志级别
            to_file: 是否输出到文件
            to_console: 是否输出到控制台
        """
        # 处理配置对象
        if hasattr(config_or_name, 'save_path'):
            # 传入的是配置对象
            config = config_or_name
            name = "demandprep_policy"
            log_dir = log_dir or config.save_path
        else:
            # 传入的是字符串名称
            name = str(config_or_name)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        self.logger.handlers = []  # 清除已有的 handler
        self.logger.propagate = False

        # 日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
        )

        # 控制台输出
        if to_console:
            console_handler = logging.StreamHandler(sys.__stderr__)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件输出
        self.log_file = None
        if to_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(log_dir, f'demandprep_policy_{timestamp}.log')
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # 训练历史记录
        self.history: Dict[str, List] = {
            'episode': [],
            'score': [],
            'reward': [],
            'epsilon': [],
            'no_action': [],
            'repair_value': [],
            'delete': [],
            'replace_nearby': []
        }

        # 最佳模型记录
        self.best_score = float('-inf')
        self.best_episode = 0

    def info(self, msg: str) -> None:
        """输出 INFO 级别日志"""
        self.logger.info(msg)

    def log_info(self, msg: str) -> None:
        """输出 INFO 级别日志（别名）"""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """输出 WARNING 级别日志"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """输出 ERROR 级别日志"""
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """输出 DEBUG 级别日志"""
        self.logger.debug(msg)

    def log_episode(self,
                    episode: int,
                    score: float,
                    reward: float,
                    epsilon: float,
                    action_counts: Dict[str, int]) -> None:
        """
        记录单轮训练结果

        Args:
            episode: 轮次
            score: 模型得分（准确率或负MSE）
            reward: 累积奖励
            epsilon: 当前探索率
            action_counts: 动作统计
        """
        self.history['episode'].append(episode)
        self.history['score'].append(score)
        self.history['reward'].append(reward)
        self.history['epsilon'].append(epsilon)

        for action in ['no_action', 'repair_value', 'delete', 'replace_nearby']:
            self.history[action].append(action_counts.get(action, 0))

        # 更新最佳记录
        if score > self.best_score:
            self.best_score = score
            self.best_episode = episode

    def log_training_start(self, config: Any) -> None:
        """记录训练开始"""
        self.info("=" * 60)
        self.info("DemandPrepPolicy 训练开始")
        self.info("=" * 60)
        self.info(f"任务类型: {config.task_type.value}")
        self.info(f"模型类型: {config.model_type.value}")
        self.info(f"Agent类型: {config.agent_type.value}")
        self.info(f"训练轮数: {config.n_episodes}")
        self.info(f"真值预算: [{config.min_truth_budget}, {config.max_truth_budget}]")
        self.info("-" * 60)

    def log_training_end(self) -> None:
        """记录训练结束"""
        self.info("-" * 60)
        self.info("训练完成!")
        self.info(f"最佳得分: {self.best_score:.4f} (Episode {self.best_episode})")
        self.info("=" * 60)

    def log_inference(self,
                      action_counts: Dict[str, int],
                      repair_log: List[Dict]) -> None:
        """
        记录推理结果

        Args:
            action_counts: 动作统计
            repair_log: 修复日志
        """
        self.info("=" * 50)
        self.info("推理完成")
        self.info("=" * 50)
        self.info(f"动作统计:")
        self.info(f"  不操作: {action_counts.get('no_action', 0)}")
        self.info(f"  真值修复: {action_counts.get('repair_value', 0)}")
        self.info(f"  删除: {action_counts.get('delete', 0)}")
        self.info(f"  临近值替换: {action_counts.get('replace_nearby', 0)}")
        self.info(f"使用真值数: {len(repair_log)}")
        self.info("-" * 50)

    def log_two_phase_plan(self, repair_plan: List[Dict]) -> None:
        """记录两阶段推理计划"""
        self.info("=" * 50)
        self.info("两阶段推理 - 第一阶段: 修复计划")
        self.info("=" * 50)
        self.info(f"需要真值修复的位置: {len(repair_plan)} 个")
        for i, item in enumerate(repair_plan[:10]):  # 最多显示10个
            self.info(f"  [{i+1}] 位置({item['idx']}, {item['col']}): "
                     f"估计值={item.get('estimated_value', 'N/A'):.4f}")
        if len(repair_plan) > 10:
            self.info(f"  ... 还有 {len(repair_plan) - 10} 个")
        self.info("-" * 50)

    def log_two_phase_execute(self, repair_count: int) -> None:
        """记录两阶段推理执行"""
        self.info("=" * 50)
        self.info("两阶段推理 - 第二阶段: 执行修复")
        self.info("=" * 50)
        self.info(f"已修复: {repair_count} 个位置")
        self.info("-" * 50)

    def save_history(self, path: str) -> None:
        """
        保存训练历史到 JSON 文件

        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        self.info(f"训练历史已保存到: {path}")

    def load_history(self, path: str) -> None:
        """
        加载训练历史

        Args:
            path: 历史文件路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        self.info(f"训练历史已加载: {path}")

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.history['episode']:
            return {}

        return {
            'total_episodes': len(self.history['episode']),
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'final_score': self.history['score'][-1],
            'final_epsilon': self.history['epsilon'][-1],
            'avg_score_last_50': sum(self.history['score'][-50:]) / min(50, len(self.history['score'])),
        }
