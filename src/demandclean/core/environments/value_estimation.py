"""
值估计器
========

统一的值估计算法，被 CleaningEnv 和 TwoPhaseCleaningEnv 共享。

优先级链: FD 规则推导 → DC 规则推导 → CFD 基线回归 → 数值提取 → 编辑距离估值 → 多维 KNN(k=5) 加权均值 → DOMAIN 范围裁剪

降级策略:
  - 无 fd_rules / column_names → 跳过 FD 推导
  - 无 rich_rules['dc_rules'] → 跳过 DC 推导
  - 无 rich_rules['cfd_rules'] → 跳过 CFD 推导
  - 无 dirty_df / label_encoders → 跳过编辑距离估值
  - 无 rich_rules → 跳过 DOMAIN 裁剪
  - 无 scaler → DOMAIN/DC/CFD 用原始空间值
  - KNN 找不到邻居 (k=0) → fallback 到 col_means
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import re
import logging
import numpy as np
import pandas as pd

from ...config import DemandCleanConfig
from ...utils.edit_distance import find_nearest_known


class ValueEstimator:
    """
    统一值估计器

    根据配置中的 FD 规则、DOMAIN 规则和 scaler 信息，
    对给定位置的值进行最优估计。

    使用方式:
        estimator = ValueEstimator(config)
        estimated = estimator.estimate_feature_value(X, idx, col, deleted_rows, col_means)
    """

    def __init__(self, config: DemandCleanConfig):
        """
        初始化值估计器

        Args:
            config: 包含 fd_rules, column_names, rich_rules, scaler 等的配置对象
        """
        self.config = config

        # 分类列索引集合（列名 → 列索引映射）
        self.categorical_col_indices: Set[int] = set()
        self._build_categorical_index()

        # FD 索引: {rhs_col_idx: [(lhs_col_idx_list), ...]}
        self.fd_index: Dict[int, List[List[int]]] = {}
        self._build_fd_index()

        # DOMAIN 范围: {col_idx: (encoded_min, encoded_max)} 或 None
        self.domain_ranges: Dict[int, Tuple[float, float]] = {}
        self._build_domain_index()

        # DC 索引: {col_idx: [dc_rule_dict, ...]}
        self.dc_index: Dict[int, List[Dict]] = {}
        self._build_dc_index()

        # CFD 索引: {col_idx: [cfd_rule_dict, ...]}
        self.cfd_index: Dict[int, List[Dict]] = {}
        self._build_cfd_index()

        # 分类列原始值映射: col_idx → [str, ...]（用于编辑距离估值）
        self._cat_col_original_values: Dict[int, List[str]] = {}
        self._build_cat_col_original_values()

    # ====================================================================
    # 索引构建
    # ====================================================================

    def _build_categorical_index(self) -> None:
        """构建分类列索引: 列名集合 → 列索引集合"""
        if not self.config.categorical_cols or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for col_name in self.config.categorical_cols:
            idx = col_name_to_idx.get(col_name)
            if idx is not None:
                self.categorical_col_indices.add(idx)

    def _is_categorical(self, col: int) -> bool:
        """判断列是否为分类列"""
        return col in self.categorical_col_indices

    def _build_fd_index(self) -> None:
        """构建 FD 规则索引: rhs_col_idx → [lhs_col_idx_list, ...]"""
        if not self.config.fd_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }

        for lhs_str, rhs_str in self.config.fd_rules:
            # LHS 可能是逗号分隔的多列
            lhs_cols = [c.strip() for c in lhs_str.split(',')]
            lhs_indices = []
            for c in lhs_cols:
                if c in col_name_to_idx:
                    lhs_indices.append(col_name_to_idx[c])

            rhs_idx = col_name_to_idx.get(rhs_str.strip())

            if lhs_indices and rhs_idx is not None:
                self.fd_index.setdefault(rhs_idx, []).append(lhs_indices)

    def _build_domain_index(self) -> None:
        """构建 DOMAIN 范围索引: col_idx → (encoded_min, encoded_max)"""
        if not self.config.rich_rules or not self.config.column_names:
            return

        domain_rules = self.config.rich_rules.get('domain_rules', [])
        if not domain_rules:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }

        for rule in domain_rules:
            col_name = rule.get('column', '')
            dtype = rule.get('dtype', '')
            min_val = rule.get('min_val')
            max_val = rule.get('max_val')

            # 仅对 INT/FLOAT 类型建立范围约束（ENUM 不做裁剪）
            if dtype not in ('INT', 'FLOAT') or min_val is None or max_val is None:
                continue

            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                continue

            # 转换到编码空间
            encoded_min, encoded_max = self._transform_to_encoded_space(
                col_idx, min_val, max_val
            )
            if encoded_min is not None:
                self.domain_ranges[col_idx] = (encoded_min, encoded_max)

    def _build_dc_index(self) -> None:
        """构建 DC 规则索引: col_idx → [dc_rule_dict, ...]

        DC 规则的序列化字典包含:
        - raw: 原始字符串
        - clauses: 条件子句列表
        - mark_cols: MARK 标记的目标列
        - involved_cols: 所有涉及的列

        索引逻辑:
        - 有 mark_cols: mark_cols 列是目标列
        - 无 mark_cols: involved_cols 都是潜在目标列
        """
        if not self.config.rich_rules:
            return
        dc_rules = self.config.rich_rules.get('dc_rules', [])
        if not dc_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for rule in dc_rules:
            target_cols = rule.get('mark_cols', []) or rule.get('involved_cols', [])
            for col_name in target_cols:
                if col_name in col_name_to_idx:
                    col_idx = col_name_to_idx[col_name]
                    self.dc_index.setdefault(col_idx, []).append(rule)

    def _build_cfd_index(self) -> None:
        """构建 CFD 规则索引: col_idx → [cfd_rule_dict, ...]

        CFD 规则的序列化字典包含:
        - conditions: [(col, op, val), ...]
        - target_col: 目标列名
        - direction: 'EXCESS' 或 'DEFICIT'
        - threshold: 偏差阈值
        - baseline: 基线值
        """
        if not self.config.rich_rules:
            return
        cfd_rules = self.config.rich_rules.get('cfd_rules', [])
        if not cfd_rules or not self.config.column_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(self.config.column_names)
        }
        for rule in cfd_rules:
            col_name = rule.get('target_col', '')
            if col_name in col_name_to_idx:
                col_idx = col_name_to_idx[col_name]
                self.cfd_index.setdefault(col_idx, []).append(rule)

    def _build_cat_col_original_values(self) -> None:
        """构建分类列 col_idx → 原始字符串值列表映射

        前置条件: config.label_encoders, config.categorical_cols, config.column_names
        """
        label_encoders = getattr(self.config, 'label_encoders', None)
        cat_cols = self.config.categorical_cols
        col_names = self.config.column_names

        if not label_encoders or not cat_cols or not col_names:
            return

        col_name_to_idx = {
            name: idx for idx, name in enumerate(col_names)
        }
        for col_name in cat_cols:
            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                continue
            le = label_encoders.get(col_name)
            if le is None or not hasattr(le, 'classes_'):
                continue
            self._cat_col_original_values[col_idx] = list(le.classes_)

    def _transform_to_encoded_space(
            self, col_idx: int, raw_min: float, raw_max: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """将原始空间 [min, max] 转换到编码空间"""
        scaler = self.config.scaler
        if scaler is None:
            # 无 scaler，直接用原始值
            return raw_min, raw_max

        try:
            # scaler 是 StandardScaler: encoded = (raw - mean) / scale
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                mean = scaler.mean_[col_idx]
                scale = scaler.scale_[col_idx]
                if scale == 0 or np.isnan(scale):
                    return raw_min, raw_max
                encoded_min = (raw_min - mean) / scale
                encoded_max = (raw_max - mean) / scale
                return float(encoded_min), float(encoded_max)
        except (IndexError, AttributeError):
            pass

        # scaler 不兼容，用原始值
        return raw_min, raw_max

    # ====================================================================
    # 核心估计方法
    # ====================================================================

    def estimate_feature_value(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> float:
        """
        估计指定位置的特征值

        优先级链: FD 规则推导 → DC 规则推导 → CFD 基线回归 → 数值提取 → 编辑距离估值 → 多维 KNN → DOMAIN 裁剪

        Args:
            X: 当前数据矩阵
            idx: 目标行索引
            col: 目标列索引
            deleted_rows: 已删除行集合
            col_means: 列均值 (fallback 用)
            dirty_df_row_indices: X 行号 → dirty_df 原始行号的映射数组。
                当 X 是 dirty_df 的子集（如 DeleteFix 后的 X_base）时必须提供，
                以便编辑距离和数值提取策略能正确定位原始脏字符串。
                None 时行为不变（要求 len(X) == len(dirty_df)）。

        Returns:
            估计的特征值
        """
        # Step 1: 尝试 FD 推导
        fd_result = self._try_fd_derivation(X, idx, col, deleted_rows)
        if fd_result is not None:
            return self._clip_domain(col, fd_result)

        # Step 2: 尝试 DC 推导
        dc_result = self._try_dc_derivation(X, idx, col)
        if dc_result is not None:
            return self._clip_domain(col, dc_result)

        # Step 3: 尝试 CFD 推导
        cfd_result = self._try_cfd_derivation(col)
        if cfd_result is not None:
            return self._clip_domain(col, cfd_result)

        # Step 3.5: 数值提取（句法错误 → 从原始字符串提取数字）
        numeric_result = self._try_numeric_extraction(X, idx, col, dirty_df_row_indices)
        if numeric_result is not None:
            return self._clip_domain(col, numeric_result)

        # Step 3.6: 编辑距离估值（分类列 typo 修复）
        edit_dist_result = self._try_edit_distance_estimation(X, idx, col, dirty_df_row_indices)
        if edit_dist_result is not None:
            return self._clip_domain(col, edit_dist_result)

        # Step 4: 多维 KNN
        knn_result = self._multidim_knn(X, idx, col, deleted_rows, col_means)
        if knn_result is not None:
            return self._clip_domain(col, knn_result)

        # Step 5: Fallback
        # 分类列: 用该列出现最多的值（众数）
        # 数值列: 用列均值
        if self._is_categorical(col):
            fallback = self._col_mode(X, col, deleted_rows)
        else:
            fallback = col_means[col] if not np.isnan(col_means[col]) else 0.0
        return self._clip_domain(col, fallback)

    # ====================================================================
    # FD 推导
    # ====================================================================

    def _try_fd_derivation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
    ) -> Optional[float]:
        """
        基于 FD 规则推导目标列的值

        检查 col 是否是某条 FD 的 RHS，如果是：
        1. 取当前行的 LHS 值
        2. 在数据中找所有 LHS 值相同的行（编码空间 |a-b| < 1e-6）
        3. 排除自身和已删除行
        4. 至少需要 2 个匹配行才有信心

        Returns:
            推导值 或 None (降级到 KNN)
        """
        if col not in self.fd_index:
            return None

        for lhs_cols in self.fd_index[col]:
            # 获取目标行的 LHS 值
            lhs_values = X[idx, lhs_cols]

            # 跳过含 NaN 的 LHS
            if np.any(np.isnan(lhs_values)):
                continue

            # 查找所有 LHS 值相同的行
            candidates = []
            for i in range(len(X)):
                if i == idx or i in deleted_rows:
                    continue

                row_lhs = X[i, lhs_cols]
                if np.any(np.isnan(row_lhs)):
                    continue

                # 编码空间容差匹配
                if np.all(np.abs(row_lhs - lhs_values) < 1e-6):
                    rhs_val = X[i, col]
                    if not np.isnan(rhs_val):
                        candidates.append(rhs_val)

            # 至少 2 个匹配才有信心
            if len(candidates) < 2:
                continue

            candidates = np.array(candidates)

            # 分类列: 始终多数投票（保证输出合法编码值）
            # 数值列: 少量唯一值 → 多数投票；多唯一值 → 均值
            if self._is_categorical(col):
                unique, counts = np.unique(candidates, return_counts=True)
                return float(unique[np.argmax(counts)])

            unique_vals = np.unique(candidates)
            if len(unique_vals) <= 5:
                unique, counts = np.unique(candidates, return_counts=True)
                return float(unique[np.argmax(counts)])

            # 多唯一值 → 均值
            return float(np.mean(candidates))

        return None

    # ====================================================================
    # DC 推导
    # ====================================================================

    def _try_dc_derivation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
    ) -> Optional[float]:
        """
        基于 DC 规则推导目标列的修复值

        DC 使用 denial 语义: 当所有 clauses 成立时 = 约束被违反。
        估值目标 = 让约束不被违反（让至少一个 clause 不成立）。

        策略:
        - 找到涉及当前列 (MARK 列) 的 clause (target_clause)
        - 检查其他 clause 是否都成立（如果其他 clause 有不成立的，规则不适用）
        - 反推 target_clause: 让该 clause 不成立的值就是修复值

        Returns:
            编码空间的修复值 或 None (降级到下一策略)
        """
        if col not in self.dc_index:
            return None

        col_name_to_idx = {
            name: i for i, name in enumerate(self.config.column_names)
        }

        for rule in self.dc_index[col]:
            col_name = self.config.column_names[col]
            clauses = rule.get('clauses', [])
            mark_cols = rule.get('mark_cols', [])

            # 只对 MARK 列做推导（如果有 MARK 定义）
            if mark_cols and col_name not in mark_cols:
                continue

            # 分离: 涉及当前列的 clause (target) vs 其他 clause
            target_clause = None
            other_clauses_all_met = True

            for clause in clauses:
                clause_cols = clause.get('columns', [])
                if col_name in clause_cols:
                    # 这是涉及当前列的 clause
                    target_clause = clause
                    continue

                # 评估非目标列的 clause 是否成立
                if not self._evaluate_dc_clause_encoded(clause, X, idx, col_name_to_idx):
                    other_clauses_all_met = False
                    break

            if not other_clauses_all_met or target_clause is None:
                continue

            # 根据 target_clause 反推修复值
            result = self._derive_from_dc_clause(
                target_clause, X, idx, col, col_name, col_name_to_idx
            )
            if result is not None:
                return result

        return None

    def _evaluate_dc_clause_encoded(
            self,
            clause: Dict,
            X: np.ndarray,
            row_idx: int,
            col_name_to_idx: Dict[str, int],
    ) -> bool:
        """
        在编码空间中评估单个 DC 子句是否成立

        对于 simple/simple_str 类型: 将规则中的原始值转到编码空间后比较
        对于 abs_diff 类型: 直接在编码空间比较（编码后差值方向不变）

        Returns:
            True 表示子句条件成立（约束被违反的一个条件）
        """
        clause_type = clause.get('type', '')

        if clause_type == 'abs_diff':
            col1_name, col2_name = clause['col1'], clause['col2']
            col1_idx = col_name_to_idx.get(col1_name)
            col2_idx = col_name_to_idx.get(col2_name)
            if col1_idx is None or col2_idx is None:
                return False
            try:
                v1 = float(X[row_idx, col1_idx])
                v2 = float(X[row_idx, col2_idx])
            except (ValueError, TypeError, IndexError):
                return False
            if np.isnan(v1) or np.isnan(v2):
                return False
            # abs_diff 阈值需要转换到编码空间
            threshold = clause['value']
            encoded_threshold = self._transform_threshold_to_encoded_space(
                col1_name, threshold
            )
            actual = abs(v1 - v2)
            return self._dc_compare(actual, clause['op'], encoded_threshold)

        elif clause_type in ('simple', 'simple_str'):
            col_name = clause['col']
            col_idx = col_name_to_idx.get(col_name)
            if col_idx is None:
                return False
            try:
                actual_encoded = float(X[row_idx, col_idx])
            except (ValueError, TypeError, IndexError):
                return False
            if np.isnan(actual_encoded):
                return False
            # 将规则中的原始值转换到编码空间
            expected_encoded = self._transform_single_value_to_encoded_space(
                col_name, clause['value']
            )
            if expected_encoded is None:
                return False
            return self._dc_compare(actual_encoded, clause['op'], expected_encoded)

        return False

    @staticmethod
    def _dc_compare(actual: float, op: str, expected: float) -> bool:
        """DC 数值比较（与 auto_detector 中的逻辑一致）"""
        if op == 'GT':
            return actual > expected
        elif op == 'GTE':
            return actual >= expected
        elif op == 'LT':
            return actual < expected
        elif op == 'LTE':
            return actual <= expected
        elif op in ('EQ', 'IQ'):
            return abs(actual - expected) < 1e-6
        elif op == 'NEQ':
            return abs(actual - expected) > 1e-6
        return False

    def _derive_from_dc_clause(
            self,
            clause: Dict,
            X: np.ndarray,
            idx: int,
            col: int,
            col_name: str,
            col_name_to_idx: Dict[str, int],
    ) -> Optional[float]:
        """
        根据 DC 的 target clause 反推修复值

        Denial 语义: clause 成立 = 约束被违反
        修复 = 让 clause 不成立

        推导逻辑:
        - NEQ(mark_col, val): mark_col != val 时成立 → 修复为 val（让 clause 不成立）
        - GT(mark_col, threshold): mark_col > threshold 时成立 → 修复为 threshold
        - LT(mark_col, threshold): mark_col < threshold 时成立 → 修复为 threshold
        - EQ(mark_col, val): mark_col == val 时成立 → 无法精确推导，跳过
        - abs_diff GT(ABS(col1-col2), t): 修另一列的值使差值为 0

        Returns:
            编码空间的修复值 或 None
        """
        clause_type = clause.get('type', '')
        op = clause.get('op', '')

        if clause_type in ('simple', 'simple_str'):
            clause_col = clause.get('col', '')
            raw_value = clause.get('value')

            if clause_col != col_name:
                return None

            if op == 'NEQ':
                # NEQ(col, val): col != val 时成立 → 修复为 val
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            elif op in ('GT', 'GTE'):
                # GT(col, threshold): col > threshold 时成立 → 修复为 threshold
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            elif op in ('LT', 'LTE'):
                # LT(col, threshold): col < threshold 时成立 → 修复为 threshold
                return self._transform_single_value_to_encoded_space(col_name, raw_value)

            # EQ: col == val 时成立，修复需要 col != val，无精确推导
            return None

        elif clause_type == 'abs_diff':
            # GT(ABS(col1 - col2), threshold)
            col1_name = clause.get('col1', '')
            col2_name = clause.get('col2', '')

            # 当前列是 col1 → 修复为 X[idx, col2_idx]（让差值 = 0，在阈值内）
            if col1_name == col_name:
                other_idx = col_name_to_idx.get(col2_name)
                if other_idx is not None:
                    other_val = X[idx, other_idx]
                    if not np.isnan(other_val):
                        return float(other_val)

            # 当前列是 col2 → 修复为 X[idx, col1_idx]
            if col2_name == col_name:
                other_idx = col_name_to_idx.get(col1_name)
                if other_idx is not None:
                    other_val = X[idx, other_idx]
                    if not np.isnan(other_val):
                        return float(other_val)

        return None

    # ====================================================================
    # CFD 推导
    # ====================================================================

    def _try_cfd_derivation(self, col: int) -> Optional[float]:
        """
        基于 CFD 规则推导目标列的修复值

        CFD 检测标记了偏离基线的列，估值 = 回到基线值。
        baseline 是规则中定义的标准值（原始 CSV 空间），
        需要转换到编码空间后返回。

        Returns:
            编码空间的修复值 或 None (降级到下一策略)
        """
        if col not in self.cfd_index:
            return None

        for rule in self.cfd_index[col]:
            baseline = rule.get('baseline')
            if baseline is not None:
                # 将 baseline 从原始空间转换到编码空间
                col_name = self.config.column_names[col]
                encoded_baseline = self._transform_single_value_to_encoded_space(
                    col_name, baseline
                )
                if encoded_baseline is not None:
                    return encoded_baseline

        return None

    # ====================================================================
    # 数值提取（句法错误修复）
    # ====================================================================

    # 匹配字符串中的数字部分（整数或浮点数，含可选负号）
    _NUMERIC_RE = re.compile(r'-?\d+(?:\.\d+)?')

    def _try_numeric_extraction(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """从原始脏字符串中提取数值部分（句法错误修复）

        适用场景: 编码时 pd.to_numeric 失败产生 NaN，但原始值包含数字。
        例: "12.0 oz" → 12.0, "0.05%" → 0.05

        安全条件:
        - 需要能将 idx 映射到 dirty_df 行号（通过 dirty_df_row_indices 或行数一致）
        - 仅当当前单元格为 NaN 时触发（编码成功的不需要提取）
        - 仅对数值列（非分类列）生效
        - 原始值是 NaN/None/空串 → 跳过（真正的缺失值）

        Args:
            X: 当前数据矩阵
            idx: 目标行索引
            col: 目标列索引
            dirty_df_row_indices: X 行号 → dirty_df 原始行号的映射数组（可选）

        Returns:
            编码空间的提取值，或 None（不适用）
        """
        # 前置条件检查
        dirty_df = self.config.dirty_df
        if dirty_df is None or not self.config.column_names:
            return None

        # 确定 dirty_df 中的行号
        if dirty_df_row_indices is not None:
            if idx >= len(dirty_df_row_indices):
                return None
            dirty_idx = int(dirty_df_row_indices[idx])
        else:
            # 行数不匹配 → idx 无法映射到 dirty_df
            if len(X) != len(dirty_df):
                return None
            dirty_idx = idx

        # 分类列不做数值提取
        if self._is_categorical(col):
            return None

        # 当前值不是 NaN → 编码已成功，无需提取
        current_val = X[idx, col]
        if not np.isnan(current_val):
            return None

        # 获取原始脏字符串
        col_name = self.config.column_names[col]
        if col_name not in dirty_df.columns:
            return None

        raw_value = dirty_df.iloc[dirty_idx][col_name]

        # 真正的缺失值（NaN/None/空串）→ 跳过
        if pd.isna(raw_value):
            return None
        raw_str = str(raw_value).strip()
        if raw_str == '':
            return None

        # 尝试从字符串中提取数字
        match = self._NUMERIC_RE.search(raw_str)
        if match is None:
            return None

        try:
            extracted = float(match.group())
        except ValueError:
            return None

        # 转换到编码空间
        encoded = self._transform_single_value_to_encoded_space(col_name, extracted)
        if encoded is not None:
            logger = logging.getLogger('demandclean.value_estimation')
            logger.debug(
                f"NumericExtraction: row={idx}, col={col_name}, "
                f"raw='{raw_str}' → {extracted} → encoded={encoded:.4f}"
            )
        return encoded

    # ====================================================================
    # 编辑距离估值（分类列 typo 修复）
    # ====================================================================

    def _try_edit_distance_estimation(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """基于编辑距离修复分类列的拼写错误

        仅对分类列生效。从 dirty_df 取原始脏字符串，
        用 find_nearest_known() 找最近已知类别，转换到编码空间。

        安全条件:
        - 仅对分类列生效
        - 需要能将 idx 映射到 dirty_df 行号（通过 dirty_df_row_indices 或行数一致）
        - 原始脏字符串必须与最近已知值不同（确实是 typo）
        - 当前编码值不是众数（众数 = 很可能正确，不需修复）

        阈值 0.6（严格）: 估值侧需要高置信度，只修复明显的 typo。

        Args:
            X: 当前数据矩阵
            idx: 目标行索引
            col: 目标列索引
            dirty_df_row_indices: X 行号 → dirty_df 原始行号的映射数组（可选）

        Returns:
            编码空间的修复值，或 None（不适用）
        """
        # 前置条件: 仅分类列
        if not self._is_categorical(col):
            return None

        # 需要分类列原始值映射
        if col not in self._cat_col_original_values:
            return None

        known_values = self._cat_col_original_values[col]
        if not known_values:
            return None

        # 需要 dirty_df 和列名
        dirty_df = self.config.dirty_df
        if dirty_df is None or not self.config.column_names:
            return None

        # 确定 dirty_df 中的行号
        if dirty_df_row_indices is not None:
            if idx >= len(dirty_df_row_indices):
                return None
            dirty_idx = int(dirty_df_row_indices[idx])
        else:
            # 行数不匹配 → idx 无法映射到 dirty_df
            if len(X) != len(dirty_df):
                return None
            dirty_idx = idx

        # 获取原始脏字符串
        col_name = self.config.column_names[col]
        if col_name not in dirty_df.columns:
            return None

        raw_value = dirty_df.iloc[dirty_idx][col_name]
        if pd.isna(raw_value):
            return None
        raw_str = str(raw_value).strip()
        if not raw_str:
            return None

        # 编辑距离匹配（严格阈值）
        nearest = find_nearest_known(raw_str, known_values, threshold=0.6)
        if nearest is None:
            return None

        # 如果最近值 == 原始值 → 没有 typo，跳过
        if nearest == raw_str:
            return None

        # 众数检查: 如果当前编码值本身就是该列众数 → 跳过
        # （众数 = 最常出现的值，通常是正确的，不需要修复）
        current_val = X[idx, col]
        if not np.isnan(current_val):
            col_vals = X[:, col]
            valid_vals = col_vals[~np.isnan(col_vals)]
            if len(valid_vals) > 0:
                unique, counts = np.unique(valid_vals, return_counts=True)
                mode_val = unique[np.argmax(counts)]
                if abs(current_val - mode_val) < 1e-6:
                    return None

        # 将最近值转换到编码空间
        label_encoders = getattr(self.config, 'label_encoders', None)
        if not label_encoders or col_name not in label_encoders:
            return None

        le = label_encoders[col_name]
        scaler = self.config.scaler

        try:
            le_val = le.transform([nearest])[0]
        except (ValueError, KeyError):
            return None

        # 转到 LE+SS 空间
        if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            try:
                mean = scaler.mean_[col]
                scale = scaler.scale_[col]
                if scale > 0 and not np.isnan(scale):
                    encoded = float((le_val - mean) / scale)
                else:
                    encoded = float(le_val)
            except (IndexError, AttributeError):
                encoded = float(le_val)
        else:
            encoded = float(le_val)

        logger = logging.getLogger('demandclean.value_estimation')
        logger.debug(
            f"EditDistance: row={idx}, col={col_name}, "
            f"raw='{raw_str}' → nearest='{nearest}' → encoded={encoded:.4f}"
        )
        return encoded

    # ====================================================================
    # 编码空间转换辅助方法
    # ====================================================================

    def _transform_single_value_to_encoded_space(
            self, col_name: str, raw_value: Any
    ) -> Optional[float]:
        """
        将单个原始 CSV 值转换到 LE+SS 编码空间

        流程: raw_value → float → (val - mean) / scale

        Args:
            col_name: 列名
            raw_value: 原始值（可能是 float/int/str）

        Returns:
            编码空间的值，或 None（无法转换）
        """
        try:
            val = float(raw_value)
        except (ValueError, TypeError):
            return None

        scaler = self.config.scaler
        if scaler is None:
            return val  # 无 scaler 时直接返回原始值

        if not self.config.column_names:
            return val

        try:
            col_idx = list(self.config.column_names).index(col_name)
            # StandardScaler: encoded = (val - mean) / scale
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                mean = scaler.mean_[col_idx]
                scale = scaler.scale_[col_idx]
                if scale > 0 and not np.isnan(scale):
                    return float((val - mean) / scale)
            return val
        except (ValueError, IndexError):
            return val

    def _transform_threshold_to_encoded_space(
            self, col_name: str, raw_threshold: float
    ) -> float:
        """
        将 abs_diff 阈值转换到编码空间

        对于 StandardScaler，abs_diff 的阈值只需要除以 scale（不减 mean，
        因为 |encoded_a - encoded_b| = |raw_a - raw_b| / scale）。

        Args:
            col_name: 参考列名（用于获取 scale）
            raw_threshold: 原始空间的阈值

        Returns:
            编码空间的阈值
        """
        scaler = self.config.scaler
        if scaler is None or not self.config.column_names:
            return raw_threshold

        try:
            col_idx = list(self.config.column_names).index(col_name)
            if hasattr(scaler, 'scale_'):
                scale = scaler.scale_[col_idx]
                if scale > 0 and not np.isnan(scale):
                    return raw_threshold / scale
        except (ValueError, IndexError):
            pass

        return raw_threshold

    # ====================================================================
    # 多维 KNN
    # ====================================================================

    def _multidim_knn(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            k: int = 5,
    ) -> Optional[float]:
        """
        多维 KNN 估计（排除目标列）

        用除目标列外的所有特征计算欧氏距离，
        取 k 个最近邻的距离倒数加权均值。

        Args:
            X: 数据矩阵
            idx: 目标行索引
            col: 目标列索引
            deleted_rows: 已删除行集合
            col_means: 列均值 (NaN 填充用)
            k: 邻居数量

        Returns:
            加权均值 或 None
        """
        n_rows, n_cols = X.shape
        if n_rows < 2:
            return None

        # 构建特征掩码（排除目标列）
        feature_mask = np.ones(n_cols, dtype=bool)
        feature_mask[col] = False

        # 准备填充后的特征矩阵（仅用于距离计算）
        X_feat = X[:, feature_mask].copy()
        for c in range(X_feat.shape[1]):
            nan_mask = np.isnan(X_feat[:, c])
            if nan_mask.any():
                # 用对应列的均值填充
                orig_col = np.where(feature_mask)[0][c]
                fill_val = col_means[orig_col] if not np.isnan(col_means[orig_col]) else 0.0
                X_feat[nan_mask, c] = fill_val

        target_feat = X_feat[idx]

        # 计算距离
        distances = np.linalg.norm(X_feat - target_feat, axis=1)
        distances[idx] = np.inf

        # 排除已删除行
        for d_idx in deleted_rows:
            if d_idx < n_rows:
                distances[d_idx] = np.inf

        # 排除目标列为 NaN 的行
        target_col_vals = X[:, col]
        nan_in_target = np.isnan(target_col_vals)
        distances[nan_in_target] = np.inf

        # 取 k 个最近邻
        valid_count = (distances < np.inf).sum()
        k_actual = min(k, valid_count)
        if k_actual == 0:
            return None

        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_dists = distances[nearest_indices]
        nearest_vals = target_col_vals[nearest_indices]

        # 距离倒数加权
        weights = 1.0 / (nearest_dists + 1e-8)
        weights /= weights.sum()

        # 分类列: 加权多数投票（保证输出是已有的合法编码值）
        if self._is_categorical(col):
            return self._weighted_majority_vote(nearest_vals, weights)

        # 数值列: 加权均值
        return float(np.average(nearest_vals, weights=weights))

    # ====================================================================
    # 分类列投票
    # ====================================================================

    @staticmethod
    def _weighted_majority_vote(values: np.ndarray, weights: np.ndarray) -> float:
        """
        加权多数投票

        对每个唯一值累加其对应的权重，返回总权重最大的值。
        结果保证是 values 中实际存在的合法编码值。

        Args:
            values: 邻居的值数组
            weights: 对应的距离倒数权重

        Returns:
            得票最高的合法编码值
        """
        unique_vals = np.unique(values)
        if len(unique_vals) == 1:
            return float(unique_vals[0])

        best_val = unique_vals[0]
        best_weight = -1.0

        for v in unique_vals:
            mask = np.abs(values - v) < 1e-8
            total_w = weights[mask].sum()
            if total_w > best_weight:
                best_weight = total_w
                best_val = v

        return float(best_val)

    def _col_mode(self, X: np.ndarray, col: int, deleted_rows: Set[int]) -> float:
        """
        分类列众数（fallback 用）

        取该列中非 NaN、非已删除行的最频繁值。

        Args:
            X: 数据矩阵
            col: 列索引
            deleted_rows: 已删除行集合

        Returns:
            众数编码值（合法值）
        """
        vals = X[:, col].copy()
        # 排除已删除行
        for d_idx in deleted_rows:
            if d_idx < len(vals):
                vals[d_idx] = np.nan
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            return 0.0
        unique, counts = np.unique(valid, return_counts=True)
        return float(unique[np.argmax(counts)])

    # ====================================================================
    # DOMAIN 裁剪
    # ====================================================================

    def _clip_domain(self, col: int, value: float) -> float:
        """
        DOMAIN 范围裁剪

        如果该列有 DOMAIN 规则（INT/FLOAT 类型），
        将值裁剪到编码空间的合法范围内。

        Args:
            col: 列索引
            value: 待裁剪的值

        Returns:
            裁剪后的值
        """
        if col in self.domain_ranges:
            encoded_min, encoded_max = self.domain_ranges[col]
            value = float(np.clip(value, encoded_min, encoded_max))
        return value

    # ====================================================================
    # 诊断方法
    # ====================================================================

    def get_estimation_source(
            self,
            X: np.ndarray,
            idx: int,
            col: int,
            deleted_rows: Set[int],
            col_means: np.ndarray,
            dirty_df_row_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, str]:
        """
        估计值并返回来源信息（用于测试和诊断）

        Returns:
            (estimated_value, source) — source 为 'fd', 'dc', 'cfd', 'numeric_extraction',
            'edit_distance', 'knn', 'mode', 'mean'
        """
        # Step 1: FD
        fd_result = self._try_fd_derivation(X, idx, col, deleted_rows)
        if fd_result is not None:
            return self._clip_domain(col, fd_result), 'fd'

        # Step 2: DC
        dc_result = self._try_dc_derivation(X, idx, col)
        if dc_result is not None:
            return self._clip_domain(col, dc_result), 'dc'

        # Step 3: CFD
        cfd_result = self._try_cfd_derivation(col)
        if cfd_result is not None:
            return self._clip_domain(col, cfd_result), 'cfd'

        # Step 3.5: 数值提取
        numeric_result = self._try_numeric_extraction(X, idx, col, dirty_df_row_indices)
        if numeric_result is not None:
            return self._clip_domain(col, numeric_result), 'numeric_extraction'

        # Step 3.6: 编辑距离估值
        edit_dist_result = self._try_edit_distance_estimation(X, idx, col, dirty_df_row_indices)
        if edit_dist_result is not None:
            return self._clip_domain(col, edit_dist_result), 'edit_distance'

        # Step 4: KNN
        knn_result = self._multidim_knn(X, idx, col, deleted_rows, col_means)
        if knn_result is not None:
            return self._clip_domain(col, knn_result), 'knn'

        # Step 5: Fallback
        if self._is_categorical(col):
            fallback = self._col_mode(X, col, deleted_rows)
        else:
            fallback = col_means[col] if not np.isnan(col_means[col]) else 0.0
        return self._clip_domain(col, fallback), 'mode' if self._is_categorical(col) else 'mean'

    def summary(self) -> str:
        """返回估计器配置摘要"""
        parts = []
        if self.fd_index:
            total_rules = sum(len(v) for v in self.fd_index.values())
            parts.append(f"FD={total_rules} rules ({len(self.fd_index)} RHS cols)")
        if self.dc_index:
            total_dc = sum(len(v) for v in self.dc_index.values())
            parts.append(f"DC={total_dc} rules ({len(self.dc_index)} target cols)")
        if self.cfd_index:
            total_cfd = sum(len(v) for v in self.cfd_index.values())
            parts.append(f"CFD={total_cfd} rules ({len(self.cfd_index)} target cols)")
        if self.domain_ranges:
            parts.append(f"DOMAIN={len(self.domain_ranges)} cols")
        if self.categorical_col_indices:
            parts.append(f"CAT={len(self.categorical_col_indices)} cols")
        if not parts:
            parts.append("no rules (KNN only)")
        return f"ValueEstimator({', '.join(parts)})"
