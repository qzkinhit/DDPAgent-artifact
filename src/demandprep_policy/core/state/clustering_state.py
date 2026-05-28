"""
聚类任务状态提取器
==================

专为聚类任务（KMeans等）设计的状态特征提取器。

与分类/回归的核心差异：
- distance_to_boundary: 使用簇间 margin（最近 vs 次近中心距离差）
- feature_importance: 基于聚类中心对各特征维度的区分度
- 聚类无"标签"概念，col==-1 的错误按照特征重要性均值处理
"""

from typing import Dict, Any, Set
import numpy as np

from .state_extractor import StateExtractor


class ClusteringStateExtractor(StateExtractor):
    """
    聚类任务的状态特征提取器

    关键语义差异：
    1. distance_to_boundary 表示样本到簇边界的距离（margin）
       - margin 大 → 深入某簇内部 → 删除/修改风险较小
       - margin 小 → 接近簇边界 → 修改可能导致簇分配翻转
    2. 聚类中标签错误（col==-1）意味着真实簇标签与聚类结果不一致
       - 但这类错误在聚类中不直接"修复"标签，而是通过清洗特征来改善聚类质量
    """

    def extract(self,
                X_current: np.ndarray,
                y: np.ndarray,
                error: Dict[str, Any],
                deleted_rows: Set[int]) -> np.ndarray:
        """
        提取 8 维状态特征向量

        Args:
            X_current: 当前数据矩阵
            y: 簇标签（聚类中可能是真实标签或聚类预测标签）
            error: 当前错误信息 {'idx', 'col', 'type', ...}
            deleted_rows: 已删除的行集合

        Returns:
            8 维状态向量
        """
        idx = error.get('idx', 0)
        col = error.get('col', 0)
        error_type = error.get('type', 0)

        # 如果该行已删除，返回 8 维零向量（全局特征由 _get_state 追加）
        if idx in deleted_rows:
            return np.zeros(8, dtype=np.float32)

        # 标签错误标记 (col == -1, type == 3)
        is_label_error = (col == -1)

        # 1. error_type (归一化到 [0, 1])
        # type: 0=missing, 1=semantic, 2=syntactic, 3=label_noise
        error_type_norm = min(error_type / 3.0, 1.0)

        # 2. feature_importance
        if is_label_error:
            # 聚类中"标签错误"意味着该样本在错误的簇中
            # 重要性取所有特征的最大值（因为不知道哪个特征导致了错误分配）
            if self.feature_importance is not None and len(self.feature_importance) > 0:
                feat_imp = float(np.max(self.feature_importance))
            else:
                feat_imp = 0.5
        elif self.feature_importance is not None and 0 <= col < len(self.feature_importance):
            feat_imp = self.feature_importance[col]
        else:
            feat_imp = 0.5

        # 3. distance_to_boundary (聚类: 簇间 margin)
        if is_label_error:
            # 标签错误: 用所有特征维度的平均边界距离
            distance_norm = self._get_avg_boundary_distance(X_current, idx)
        else:
            distance_norm = self.get_distance_to_boundary(X_current, idx, col)

        # 4. row_position
        n_rows = len(X_current)
        row_pos = idx / (n_rows - 1) if n_rows > 1 else 0

        # 5. col_index
        n_cols = X_current.shape[1] if X_current.ndim > 1 else 1
        if is_label_error:
            col_norm = 1.0
        else:
            col_norm = col / (n_cols - 1) if n_cols > 1 else 0

        # 6. col_error_rate
        if is_label_error:
            col_err_rate = 0.5
        elif self.col_error_rate is not None and 0 <= col < len(self.col_error_rate):
            col_err_rate = self.col_error_rate[col]
        else:
            col_err_rate = 0.0

        # 7-8. sample_retention 和 var_retention
        if is_label_error:
            keep_mask = np.array([i not in deleted_rows for i in range(len(X_current))])
            sample_retention = keep_mask.sum() / max(len(X_current), 1)
            var_retention = 1.0
        else:
            sample_retention, var_retention = self.compute_retention(
                X_current, col, deleted_rows
            )

        return np.array([
            error_type_norm,
            feat_imp,
            distance_norm,
            row_pos,
            col_norm,
            col_err_rate,
            sample_retention,
            var_retention
        ], dtype=np.float32)

    def get_distance_to_boundary(self,
                                  X_current: np.ndarray,
                                  idx: int,
                                  col: int) -> float:
        """
        获取到聚类边界的归一化距离

        聚类中"边界"= 簇间分界面：
        - 使用 KMeansAdapter.get_distance_to_boundary()
          返回 (次近中心距离 - 最近中心距离) / max_margin
        - 值大 → 深入簇内部，修改该特征不太可能改变簇分配
        - 值小 → 接近簇边界，修改该特征可能导致簇分配翻转
        """
        if not self.model_adapter.is_fitted:
            return 0.5

        try:
            X_point = X_current[idx:idx+1].copy()
            X_point = self.fill_nan(X_point)
            distance = self.model_adapter.get_distance_to_boundary(X_point)[0]
            return float(np.clip(distance, 0, 1))
        except Exception:
            return 0.5

    def _get_avg_boundary_distance(self,
                                    X_current: np.ndarray,
                                    idx: int) -> float:
        """
        获取样本到簇边界的平均距离

        用于标签错误场景，因为标签错误不对应特定特征列。
        """
        return self.get_distance_to_boundary(X_current, idx, 0)
