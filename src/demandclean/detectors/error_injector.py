"""
错误注入器
==========

在数据上注入各类错误用于自监督训练。

设计理念: 错误注入 = 检测的逆过程
  - 语义错误: 基于规则（DOMAIN/CFD/FD）反向注入，模拟 RAHA 检测不到的逻辑违规
  - 句法错误: RAHA-aware 统计驱动（OD-Gaussian/Histogram/PVD），不使用规则
  - 标签错误: 条件性注入，镜像检测器发现的标签翻转模式
  - 缺失值:   直接设为 NaN

错误类型编码:
  0 = missing, 1 = semantic, 2 = syntactic, 3 = label_noise
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field

from ..utils.edit_distance import generate_typo, find_nearest_known, find_top_k_nearest


# ============================================================================
# 标签错误模式分析
# ============================================================================

@dataclass
class LabelErrorPattern:
    """检测到的标签错误模式"""
    flip_matrix: Dict[Tuple, int] = field(default_factory=dict)  # (from_class, to_class) -> count
    error_rate: float = 0.0
    is_symmetric: bool = True
    unique_classes: List = field(default_factory=list)
    # 回归任务扩展字段
    is_regression: bool = False
    noise_std: float = 0.0      # 估计的标签噪声标准差
    label_mean: float = 0.0     # 标签均值
    label_std: float = 1.0      # 标签标准差


def analyze_label_error_pattern(
    detected_label_errors: List,
    y_dirty: np.ndarray,
    task_type: str = 'classification',
) -> LabelErrorPattern:
    """分析检测到的标签错误模式

    Args:
        detected_label_errors: 检测器返回的标签错误列表
            格式: [(row_idx, col=-1, ...), ...] 或 [{'idx': ..., 'col': -1}, ...]
        y_dirty: 脏标签向量
        task_type: 任务类型 ('classification' 或 'regression')

    Returns:
        LabelErrorPattern 描述翻转分布（分类）或噪声分布（回归）
    """
    pattern = LabelErrorPattern()
    valid_y = y_dirty[~np.isnan(y_dirty)]
    unique_classes = list(np.unique(valid_y))
    pattern.unique_classes = unique_classes

    if not detected_label_errors:
        return pattern

    # 提取错误行索引
    error_indices = set()
    for item in detected_label_errors:
        if isinstance(item, (list, tuple)):
            error_indices.add(int(item[0]))
        elif isinstance(item, dict):
            error_indices.add(int(item.get('idx', item.get(0, -1))))

    if not error_indices:
        return pattern

    total_errors = len(error_indices)
    pattern.error_rate = total_errors / max(1, len(y_dirty))

    # ============================================================
    # 回归任务: 估计噪声标准差而非构建 flip_matrix
    # ============================================================
    if task_type == 'regression':
        pattern.is_regression = True
        pattern.label_mean = float(np.mean(valid_y))
        pattern.label_std = float(np.std(valid_y)) + 1e-8

        # 用被标记为错误的标签与全局分布的偏差估计噪声量级
        error_labels = np.array([y_dirty[i] for i in error_indices
                                  if i < len(y_dirty) and not np.isnan(y_dirty[i])])
        if len(error_labels) > 0:
            # 估计: 错误标签偏离全局均值的标准差作为噪声量级
            deviations = np.abs(error_labels - pattern.label_mean)
            pattern.noise_std = float(np.mean(deviations)) * 0.5  # 保守估计
            # 下限: 至少是标签标准差的 10%
            pattern.noise_std = max(pattern.noise_std, pattern.label_std * 0.1)
        else:
            pattern.noise_std = pattern.label_std * 0.2

        return pattern

    # ============================================================
    # 分类任务: 构建 flip_matrix（原有逻辑）
    # ============================================================
    # 统计各类的错误占比
    class_error_count = defaultdict(int)
    for idx in error_indices:
        if idx < len(y_dirty) and not np.isnan(y_dirty[idx]):
            class_error_count[y_dirty[idx]] += 1

    if len(unique_classes) == 2:
        c0, c1 = unique_classes[0], unique_classes[1]
        n_c0_err = class_error_count.get(c0, 0)
        n_c1_err = class_error_count.get(c1, 0)
        if n_c0_err > 0:
            pattern.flip_matrix[(c0, c1)] = n_c0_err
        if n_c1_err > 0:
            pattern.flip_matrix[(c1, c0)] = n_c1_err
        pattern.is_symmetric = abs(n_c0_err - n_c1_err) < max(1, total_errors * 0.3)
    else:
        # 多分类: 均匀翻转到其他类
        for cls_from, count in class_error_count.items():
            others = [c for c in unique_classes if c != cls_from]
            per_other = max(1, count // len(others)) if others else 0
            for cls_to in others:
                pattern.flip_matrix[(cls_from, cls_to)] = per_other

    return pattern


# ============================================================================
# ErrorInjector 主类
# ============================================================================

class ErrorInjector:
    """
    错误注入器

    在基准数据上注入四类错误用于训练：
    - 缺失值 (type=0): 将值设为 NaN
    - 语义错误 (type=1): 基于规则反向注入（DOMAIN/CFD/FD），无规则时随机替换
    - 句法错误 (type=2): RAHA-aware 统计驱动（OD-Gaussian/Histogram/PVD 模拟）
    - 标签错误 (type=3): 条件性标签翻转，镜像检测到的模式
    """

    def __init__(self, X_base: np.ndarray, y_base: np.ndarray,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 column_names: Optional[List[str]] = None,
                 rich_rules: Optional[Dict[str, Any]] = None,
                 label_encoders: Optional[Dict[str, Any]] = None,
                 scaler: Optional[Any] = None,
                 categorical_cols: Optional[set] = None,
                 dirty_df: Optional[Any] = None,
                 label_col: Optional[str] = None):
        """
        初始化错误注入器

        Args:
            X_base: 基准数据（删除空缺值后的脏数据，当作相对干净的）
            y_base: 标签
            fd_rules: FD规则列表 [("lhs_col", "rhs_col"), ...]
            column_names: 数据列名列表
            rich_rules: 丰富规则字典（来自 rule_parser.rules_to_dict()）
                        包含 domain_rules, cfd_rules 等
            label_encoders: {col_name: LabelEncoder} 编码工具
            scaler: StandardScaler 标准化工具
            categorical_cols: 分类列名集合
            dirty_df: 原始 CSV 空间 dirty DataFrame
            label_col: 标签列名
        """
        self.X_base = X_base.copy()
        self.y_base = y_base.copy()
        self.fd_rules = fd_rules or []
        self.column_names = column_names or []
        self.rich_rules = rich_rules

        # 编码工具
        self.label_encoders = label_encoders or {}
        self.scaler = scaler
        self.categorical_cols = categorical_cols or set()
        self.dirty_df = dirty_df
        self.label_col = label_col
        self._has_encoding_tools = bool(self.scaler is not None)

        # 计算统计量
        self.col_means = np.nanmean(X_base, axis=0)
        self.col_stds = np.nanstd(X_base, axis=0)
        self.col_percentiles: Dict[int, Tuple[float, float]] = {}
        self.all_values: Dict[int, np.ndarray] = {}

        for col in range(X_base.shape[1]):
            valid = X_base[:, col][~np.isnan(X_base[:, col])]
            self.all_values[col] = valid
            if len(valid) > 0:
                self.col_percentiles[col] = (
                    np.percentile(valid, 1),
                    np.percentile(valid, 99),
                )

        # 构建 FD 列索引映射
        self.fd_col_pairs: List[Tuple[List[int], int]] = []
        self._build_fd_index()

        # FD 主键列集合（高频 LHS：出现在 ≥2 条 FD 规则中的列）
        # 理论上句法注入应避开这些列以防止 FD 检测级联误报，
        # 但实测发现排除后错误集中在分类列上，RAHA 过度检测（8→150+），
        # 导致总体 FP 反而增加。暂时禁用，保留架构供后续优化。
        from collections import Counter
        lhs_counter = Counter()
        for lhs_indices, _rhs_idx in self.fd_col_pairs:
            for li in lhs_indices:
                lhs_counter[li] += 1
        self._fd_lhs_cols: Set[int] = set()  # 禁用: 启用会恶化 RAHA
        # 启用版本: {col for col, cnt in lhs_counter.items() if cnt >= 2}

        # 构建 DOMAIN/CFD/DC 列索引映射
        self._domain_col_map: Dict[int, Dict] = {}   # col_idx -> domain_rule_dict
        self._cfd_col_map: Dict[str, List[Dict]] = {} # class_val -> [cfd_rule_dict, ...]
        self._dc_rule_list: List[Dict] = []            # DC 规则字典列表
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            self._build_rich_rule_index()

        # 如果有编码工具，将规则值预转换到 LE+SS 空间
        if self._has_encoding_tools:
            self._convert_rules_to_encoded_space()

        # 分类列: col_idx → 原始字符串值列表（来自 LabelEncoder.classes_）
        # 用于分类列 typo 注入（逆变换 + typo + 正变换）
        self._cat_col_original_values: Dict[int, List[str]] = {}
        self._cat_col_idx_set: Set[int] = set()
        self._build_categorical_col_map()

    def _build_categorical_col_map(self):
        """构建分类列索引 → 原始字符串值列表的映射

        前置条件: self.label_encoders, self.categorical_cols, self.column_names
        """
        if not self.label_encoders or not self.categorical_cols or not self.column_names:
            return

        for col_name in self.categorical_cols:
            if col_name not in self.column_names:
                continue
            col_idx = self.column_names.index(col_name)

            le = self.label_encoders.get(col_name)
            if le is None or not hasattr(le, 'classes_'):
                continue

            known_values = list(le.classes_)
            if len(known_values) >= 2:  # 至少 2 个类别才有意义
                self._cat_col_original_values[col_idx] = known_values
                self._cat_col_idx_set.add(col_idx)

    def _generate_categorical_typo_encoded(
            self, col: int, current_encoded: float
    ) -> Optional[float]:
        """为分类列生成句法异常：随机替换为另一个合法 LE 类别

        训练-推理一致性: dirty-fit LE 下，推理时 typo 是合法 LE 整数，
        训练时注入的错误也应该是合法 LE 整数（不再用 OOV 极端值）。

        Args:
            col: 列索引
            current_encoded: 当前 LE+SS 编码值

        Returns:
            新编码值，或 None
        """
        if col not in self._cat_col_original_values:
            return None
        col_name = self.column_names[col]
        le = self.label_encoders.get(col_name)
        if le is None or self.scaler is None or col >= len(self.scaler.mean_):
            return None

        n_classes = len(le.classes_)
        if n_classes < 2:
            return None

        # 逆变换获取当前 LE 整数
        scaler_mean = self.scaler.mean_[col]
        scaler_scale = self.scaler.scale_[col]
        current_le = int(round(current_encoded * scaler_scale + scaler_mean))
        current_le = max(0, min(current_le, n_classes - 1))

        # 随机选一个不同的合法 LE 整数
        new_le = current_le
        for _ in range(20):
            new_le = np.random.randint(0, n_classes)
            if new_le != current_le:
                break
        if new_le == current_le:
            return None

        # LE → SS 编码
        new_encoded = (new_le - scaler_mean) / scaler_scale
        return float(new_encoded)

    def _build_fd_index(self):
        """构建 FD 规则的列索引映射"""
        if not self.fd_rules or not self.column_names:
            return

        for rule in self.fd_rules:
            if isinstance(rule, (list, tuple)) and len(rule) == 2:
                lhs_str, rhs_str = rule
                lhs_cols = [c.strip() for c in str(lhs_str).split(',')]
                rhs_col = str(rhs_str).strip()

                lhs_indices = []
                for c in lhs_cols:
                    if c in self.column_names:
                        lhs_indices.append(self.column_names.index(c))

                if rhs_col in self.column_names and lhs_indices:
                    rhs_idx = self.column_names.index(rhs_col)
                    self.fd_col_pairs.append((lhs_indices, rhs_idx))

    def _build_rich_rule_index(self):
        """构建 DOMAIN/CFD 规则的列索引映射"""
        if not self.rich_rules:
            return

        # DOMAIN 规则 → 列索引
        for rule in self.rich_rules.get('domain_rules', []):
            col_name = rule.get('column', '')
            if col_name in self.column_names:
                col_idx = self.column_names.index(col_name)
                self._domain_col_map[col_idx] = rule

        # CFD 规则 → 按标签值分组（支持任意标签列名）
        label_names = {'class'}
        if self.label_col:
            label_names.add(self.label_col)
        for rule in self.rich_rules.get('cfd_rules', []):
            for col, op, val in rule.get('conditions', []):
                if col in label_names and op == '=':
                    self._cfd_col_map.setdefault(val, []).append(rule)
                    break

        # DC 规则（已是序列化后的字典列表）
        self._dc_rule_list = self.rich_rules.get('dc_rules', [])

    def _convert_rules_to_encoded_space(self):
        """将规则值从 CSV 原始空间预转换到 LE+SS 编码空间

        解决核心问题: rules.txt 中的 DOMAIN/CFD 规则值是原始 CSV 字符串空间，
        但 ErrorInjector 操作的数据是经过 LabelEncoder + StandardScaler 编码的。

        转换策略:
          - DOMAIN ENUM: 用 LabelEncoder 转换枚举值列表 → LE 整数，
                        再用 StandardScaler 转到 LE+SS 空间
          - DOMAIN INT/FLOAT: 用 StandardScaler 转换 min_val/max_val 到 LE+SS 空间
          - CFD class_val: 用 LabelEncoder 转换 class 标签值到 LE 空间
        """
        if not self.scaler:
            return

        scaler_mean = self.scaler.mean_
        scaler_scale = self.scaler.scale_

        # --- 转换 DOMAIN 规则 ---
        for col_idx, rule in self._domain_col_map.items():
            col_name = rule.get('column', '')

            if rule.get('dtype') == 'ENUM':
                # ENUM: 如果是分类列（有 LabelEncoder），转换枚举值
                if col_name in self.label_encoders and col_name in self.categorical_cols:
                    le = self.label_encoders[col_name]
                    enum_vals = rule.get('enum_vals', [])
                    if enum_vals:
                        try:
                            # 将原始字符串枚举值转为 LE 编码整数
                            le_vals = le.transform(enum_vals)
                            # 再通过 StandardScaler 转到 LE+SS 空间
                            if col_idx < len(scaler_mean):
                                ss_vals = (le_vals - scaler_mean[col_idx]) / scaler_scale[col_idx]
                                # 存储编码后的枚举范围（用于注入时生成越界值）
                                rule['_encoded_enum_max'] = float(np.max(ss_vals))
                                rule['_encoded_enum_min'] = float(np.min(ss_vals))
                                rule['_encoded_enum_step'] = scaler_scale[col_idx]  # 1 个原始单位在 SS 空间的步长
                                rule['_encoding_converted'] = True
                        except (ValueError, KeyError):
                            # 某些枚举值不在 LabelEncoder 类别中（如新出现的错误值）
                            rule['_encoding_converted'] = False
                else:
                    # 纯数值 ENUM（不需要 LabelEncoder）
                    try:
                        num_vals = [float(v) for v in rule.get('enum_vals', [])]
                        if num_vals and col_idx < len(scaler_mean):
                            ss_vals = [(v - scaler_mean[col_idx]) / scaler_scale[col_idx] for v in num_vals]
                            rule['_encoded_enum_max'] = max(ss_vals)
                            rule['_encoded_enum_min'] = min(ss_vals)
                            rule['_encoded_enum_step'] = 1.0 / scaler_scale[col_idx]
                            rule['_encoding_converted'] = True
                    except (ValueError, TypeError):
                        rule['_encoding_converted'] = False

            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                # INT/FLOAT: 转换 min_val, max_val 到 LE+SS 空间
                if col_idx < len(scaler_mean):
                    original_min = rule['min_val']
                    original_max = rule['max_val']
                    rule['_encoded_min'] = (original_min - scaler_mean[col_idx]) / scaler_scale[col_idx]
                    rule['_encoded_max'] = (original_max - scaler_mean[col_idx]) / scaler_scale[col_idx]
                    # 原始空间 1 个单位对应的 SS 空间步长
                    rule['_encoded_unit_step'] = 1.0 / scaler_scale[col_idx]
                    rule['_encoding_converted'] = True

        # --- 转换 CFD 规则的 class_val ---
        if self.label_col and self.label_col in self.label_encoders:
            le = self.label_encoders[self.label_col]
            new_cfd_map: Dict[str, List[Dict]] = {}
            for class_val, rules in self._cfd_col_map.items():
                try:
                    encoded_val = le.transform([class_val])[0]
                    encoded_key = str(int(encoded_val))
                    for rule in rules:
                        rule['_original_class_val'] = class_val
                        rule['_encoded_class_val'] = encoded_key
                    new_cfd_map.setdefault(encoded_key, []).extend(rules)
                except (ValueError, KeyError):
                    # class_val 不在 LabelEncoder 中，保持原样
                    new_cfd_map.setdefault(class_val, []).extend(rules)
            self._cfd_col_map = new_cfd_map

        # --- 转换 DC 规则的 clause value 到 LE+SS 空间 ---
        for dc_rule in self._dc_rule_list:
            if dc_rule.get('_encoding_converted'):
                continue  # 已转换过

            all_cols_valid = True
            for clause in dc_rule.get('clauses', []):
                ctype = clause.get('type', '')

                if ctype == 'simple':
                    col_name = clause.get('col', '')
                    if col_name not in self.column_names:
                        all_cols_valid = False
                        break
                    col_idx = self.column_names.index(col_name)
                    raw_val = clause.get('value', 0.0)

                    if col_idx < len(scaler_mean):
                        # 分类列先通过 LabelEncoder
                        if col_name in self.label_encoders and col_name in self.categorical_cols:
                            try:
                                le = self.label_encoders[col_name]
                                le_val = le.transform([str(int(raw_val))])[0]
                                encoded_val = (le_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                            except (ValueError, KeyError):
                                encoded_val = (raw_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                        else:
                            encoded_val = (raw_val - scaler_mean[col_idx]) / scaler_scale[col_idx]
                        clause['_encoded_value'] = encoded_val
                        clause['_col_idx'] = col_idx
                        clause['_scaler_scale'] = scaler_scale[col_idx]

                elif ctype == 'abs_diff':
                    col1_name = clause.get('col1', '')
                    col2_name = clause.get('col2', '')
                    if col1_name not in self.column_names or col2_name not in self.column_names:
                        all_cols_valid = False
                        break
                    col1_idx = self.column_names.index(col1_name)
                    col2_idx = self.column_names.index(col2_name)
                    raw_threshold = clause.get('value', 0.0)

                    # abs_diff 的 threshold 是差值，需要考虑两列的 scale
                    # 近似处理: 使用两列 scale 的平均值
                    if col1_idx < len(scaler_scale) and col2_idx < len(scaler_scale):
                        avg_scale = (scaler_scale[col1_idx] + scaler_scale[col2_idx]) / 2.0
                        encoded_threshold = raw_threshold / avg_scale if avg_scale > 1e-10 else raw_threshold
                    else:
                        encoded_threshold = raw_threshold

                    clause['_encoded_value'] = encoded_threshold
                    clause['_col1_idx'] = col1_idx
                    clause['_col2_idx'] = col2_idx

            dc_rule['_encoding_converted'] = all_cols_valid

    def _has_cfd_for_label(self) -> bool:
        """检查是否有 CFD 规则覆盖标签列（用于决定是否从语义预算中分配标签注入）

        只要 CFD 规则的 conditions 中包含标签列（如 class=X => ...），
        AutoDetector 就能通过 CFD 推断检测到标签翻转。
        """
        if not self.rich_rules or not self.rich_rules.get('has_rich_rules'):
            return False
        if not self.label_col:
            return False
        for rule in self.rich_rules.get('cfd_rules', []):
            for col, op, val in rule.get('conditions', []):
                if col == self.label_col or col == 'class':
                    return True
        return False

    # ====================================================================
    # 公共接口
    # ====================================================================

    def inject_errors(self,
                      missing_rate: float = 0.05,
                      semantic_rate: float = 0.1,
                      syntactic_rate: float = 0.15,
                      label_rate: float = 0.0,
                      label_pattern: Optional[LabelErrorPattern] = None,
                      strict_semantic: bool = False,
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """
        在基准数据上注入错误

        错误分类（与 AutoDetector 检测通道一致）:
          - syntactic: 值域/格式异常 = DOMAIN 违规 + RAHA-aware 统计异常
          - semantic: 逻辑关系违反 = FD + CFD + DC 违规
          - missing: 缺失值
          - label_noise: 标签翻转

        Args:
            missing_rate: 缺失值比例
            semantic_rate: 语义错误比例（包含标签预算，如果有 CFD 标签规则）
            syntactic_rate: 句法错误比例（包含 DOMAIN 违规 + RAHA-aware）
            label_rate: 标签错误比例（已废弃，标签预算现从 semantic_rate 中分配）
            label_pattern: 标签错误模式（来自检测器分析或均匀翻转矩阵）
            strict_semantic: 严格模式 — 不 fallback 到不可检测的随机语义注入

        Returns:
            (X_dirty, y_dirty, injected_errors)
            - X_dirty: 注入错误后的特征数据
            - y_dirty: 注入错误后的标签
            - injected_errors: 注入的错误信息
                {
                    'missing': [(idx, col, original_val), ...],
                    'semantic': [(idx, col, original_val, new_val), ...],
                    'syntactic': [(idx, col, original_val, noise), ...],
                    'label_noise': [(idx, -1, original_val, new_val), ...]
                }
        """
        X_dirty = self.X_base.copy()
        y_dirty = self.y_base.copy()
        n_samples, n_features = X_dirty.shape

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        used_indices: Set[Tuple[int, int]] = set()

        # 1. 注入缺失值
        n_missing = int(n_samples * missing_rate)
        self._inject_missing(X_dirty, n_missing, used_indices, injected)

        # 2. 注入语义错误 + 标签错误
        #    语义 = CFD + DC + FD（逻辑关系违反），不含 DOMAIN
        n_semantic_total = int(n_samples * semantic_rate)

        # 标签预算策略:
        #   - label_rate > 0（旧调用方式，如 trainer.py）: 独立于语义预算
        #   - label_rate == 0（新调用方式）: 从语义预算中分配（需有 CFD 标签规则）
        has_label_rules = self._has_cfd_for_label()
        n_label = 0
        label_from_semantic = False
        if label_rate > 0 and label_pattern is not None:
            # 向后兼容: 使用显式 label_rate，独立于语义预算
            n_label = int(n_samples * label_rate)
        elif has_label_rules and label_pattern is not None:
            # 新逻辑: 从语义预算中分配 ~20%
            n_label = max(1, int(n_semantic_total * 0.2))
            label_from_semantic = True

        n_semantic = n_semantic_total - n_label if label_from_semantic else n_semantic_total

        # 特征列语义注入（不含 DOMAIN，仅 CFD + DC + FD）
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            count = self._inject_semantic_no_domain(X_dirty, y_dirty, n_semantic, used_indices, injected)
            remaining = n_semantic - count
            if remaining > 0 and self.fd_col_pairs:
                self._inject_fd_violations(X_dirty, remaining, used_indices, injected,
                                           strict=strict_semantic)
            elif remaining > 0 and not strict_semantic:
                self._inject_random_semantic(X_dirty, remaining, used_indices, injected)
        elif self.fd_col_pairs:
            self._inject_fd_violations(X_dirty, n_semantic, used_indices, injected,
                                       strict=strict_semantic)
        elif not strict_semantic:
            self._inject_random_semantic(X_dirty, n_semantic, used_indices, injected)

        # 3. 注入句法错误 = DOMAIN 违规 + RAHA-aware 统计异常
        n_syntactic = int(n_samples * syntactic_rate)

        # DOMAIN 违规（占句法预算的 30%，有 DOMAIN 规则时）
        n_domain = 0
        if self._domain_col_map:
            n_domain = int(n_syntactic * 0.3)
            self._inject_domain_violations(X_dirty, n_domain, used_indices, injected)

        # RAHA-aware 统计异常（剩余句法预算）
        n_raha_syntactic = n_syntactic - n_domain
        self._inject_raha_aware_syntactic(X_dirty, n_raha_syntactic, used_indices, injected)

        # 4. 标签错误（从语义预算分配，仅 CFD 标签规则存在时）
        if n_label > 0 and label_pattern is not None:
            # 收集已有特征错误的行（标签注入应避开这些行，
            # 否则检测端的"特征受损行排除"会过滤掉这些 TP）
            feature_damaged_rows: Set[int] = set()
            for item in injected['missing']:
                feature_damaged_rows.add(item[0])
            for item in injected['semantic']:
                feature_damaged_rows.add(item[0])
            for item in injected['syntactic']:
                feature_damaged_rows.add(item[0])
            self._inject_label_noise(y_dirty, n_label, label_pattern, injected,
                                     exclude_rows=feature_damaged_rows)

        return X_dirty, y_dirty, injected

    def inject_on_dirty(self,
                        X_dirty: np.ndarray,
                        y_dirty: np.ndarray,
                        detected_cells: Set[Tuple[int, int]],
                        missing_rate: float = 0.05,
                        semantic_rate: float = 0.1,
                        syntactic_rate: float = 0.15,
                        label_rate: float = 0.0,
                        label_pattern: Optional[LabelErrorPattern] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
        """在脏数据上注入额外错误（自监督训练用）

        只在未被检测器标记且非 NaN 的单元格上注入。

        Args:
            X_dirty: 原始脏数据（全量，含已检测错误）
            y_dirty: 原始脏标签
            detected_cells: 检测器已标记的位置集合 {(row, col), ...}
            其他参数同 inject_errors

        Returns:
            (X_augmented, y_augmented, injected_new)
        """
        X_aug = X_dirty.copy()
        y_aug = y_dirty.copy()
        n_samples, n_features = X_aug.shape

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        # 已检测位置 + NaN 位置都不可注入
        used_indices = set(detected_cells)
        for i in range(n_samples):
            for j in range(n_features):
                if np.isnan(X_aug[i, j]):
                    used_indices.add((i, j))

        # 注入逻辑与 inject_errors 完全一致
        n_missing = int(n_samples * missing_rate)
        self._inject_missing(X_aug, n_missing, used_indices, injected)

        # 语义 = CFD + DC + FD（不含 DOMAIN）
        n_semantic = int(n_samples * semantic_rate)
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            count = self._inject_semantic_no_domain(X_aug, y_aug, n_semantic, used_indices, injected)
            remaining = n_semantic - count
            if remaining > 0 and self.fd_col_pairs:
                self._inject_fd_violations(X_aug, remaining, used_indices, injected)
            elif remaining > 0:
                self._inject_random_semantic(X_aug, remaining, used_indices, injected)
        elif self.fd_col_pairs:
            self._inject_fd_violations(X_aug, n_semantic, used_indices, injected)
        else:
            self._inject_random_semantic(X_aug, n_semantic, used_indices, injected)

        # 句法 = DOMAIN + RAHA-aware
        n_syntactic = int(n_samples * syntactic_rate)
        n_domain = 0
        if self._domain_col_map:
            n_domain = int(n_syntactic * 0.3)
            self._inject_domain_violations(X_aug, n_domain, used_indices, injected)
        self._inject_raha_aware_syntactic(X_aug, n_syntactic - n_domain, used_indices, injected)

        if label_rate > 0 and label_pattern is not None:
            n_label = int(n_samples * label_rate)
            self._inject_label_noise(y_aug, n_label, label_pattern, injected)

        return X_aug, y_aug, injected

    # ====================================================================
    # 1. 缺失值注入
    # ====================================================================

    def _inject_missing(self, X_dirty: np.ndarray, n_missing: int,
                        used_indices: Set[Tuple[int, int]],
                        injected: Dict[str, List]):
        """注入缺失值 (type=0)"""
        n_samples, n_features = X_dirty.shape
        for _ in range(n_missing):
            idx = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                X_dirty[idx, col] = np.nan
                injected['missing'].append((idx, col, original_val))
                used_indices.add((idx, col))

    # ====================================================================
    # 2. 语义错误注入（基于规则 — DOMAIN/CFD/DC/FD）
    # ====================================================================

    def _inject_rule_based_semantic(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                     n_semantic: int,
                                     used_indices: Set[Tuple[int, int]],
                                     injected: Dict[str, List]) -> int:
        """基于 DOMAIN + CFD + DC 规则反向注入语义错误（向后兼容）

        注意: 新的 inject_errors() 不再调用此方法，改用 _inject_semantic_no_domain()。
        保留此方法用于 inject_on_dirty() 等旧接口。

        优先级: DOMAIN 违规 (40%) > CFD 违规 (30%) > DC 违规 (30%) > FD 补充

        Returns:
            实际注入数量
        """
        total_injected = 0

        # DOMAIN 违规注入（占比 40%）
        n_domain = int(n_semantic * 0.4)
        if self._domain_col_map:
            total_injected += self._inject_domain_violations(
                X_dirty, n_domain, used_indices, injected)

        # CFD 违规注入（占比 30%）
        n_cfd = int(n_semantic * 0.3)
        if self._cfd_col_map:
            total_injected += self._inject_cfd_violations(
                X_dirty, y_dirty, n_cfd, used_indices, injected)

        # DC 违规注入（剩余全给 DC）
        n_dc = n_semantic - total_injected
        if self._dc_rule_list:
            total_injected += self._inject_dc_violations(
                X_dirty, n_dc, used_indices, injected)

        return total_injected

    def _inject_semantic_no_domain(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                    n_semantic: int,
                                    used_indices: Set[Tuple[int, int]],
                                    injected: Dict[str, List]) -> int:
        """仅 CFD + DC 的语义注入（DOMAIN 已移到 syntactic）

        与 AutoDetector 的 semantic 通道对齐:
          - AutoDetector semantic = FD + CFD + DC
          - ErrorInjector semantic = CFD + DC + FD

        CFD 50% > DC 50% > FD 补充

        Returns:
            实际注入数量
        """
        total_injected = 0

        # CFD 违规注入（占比 50%）
        n_cfd = int(n_semantic * 0.5)
        if self._cfd_col_map:
            total_injected += self._inject_cfd_violations(
                X_dirty, y_dirty, n_cfd, used_indices, injected)

        # DC 违规注入（剩余全给 DC）
        n_dc = n_semantic - total_injected
        if self._dc_rule_list:
            total_injected += self._inject_dc_violations(
                X_dirty, n_dc, used_indices, injected)

        return total_injected

    def _inject_domain_violations(self, X_dirty: np.ndarray, n_target: int,
                                   used_indices: Set[Tuple[int, int]],
                                   injected: Dict[str, List]) -> int:
        """注入 DOMAIN 违规（超出合法值域）→ 归类为 syntactic

        与 AutoDetector 一致: DOMAIN 检测在 syntactic 通道 (Stage 2c)，
        因此注入也放入 injected['syntactic']。

        如果有编码工具，使用预转换到 LE+SS 空间的边界值生成越界值。
        否则（无编码工具），使用原始规则值（向后兼容）。

        INT [1, 10] → 在 LE+SS 空间中注入超出 [encoded_min, encoded_max] 的值
        ENUM {a, b, c} → 在 LE+SS 空间中注入超出 [encoded_enum_min, encoded_enum_max] 的值
        """
        n_samples = len(X_dirty)
        count = 0
        domain_cols = list(self._domain_col_map.keys())

        if not domain_cols:
            return 0

        for _ in range(n_target * 3):  # 多尝试
            if count >= n_target:
                break

            col_idx = np.random.choice(domain_cols)
            idx = np.random.randint(0, n_samples)

            if (idx, col_idx) in used_indices or np.isnan(X_dirty[idx, col_idx]):
                continue

            rule = self._domain_col_map[col_idx]
            original_val = X_dirty[idx, col_idx]
            new_val = None

            if rule.get('dtype') == 'ENUM':
                if rule.get('_encoding_converted'):
                    # 使用预转换的编码空间范围
                    max_encoded = rule['_encoded_enum_max']
                    # 在 LE+SS 空间中，一个原始枚举单位 ≈ 1/scaler_scale
                    step = rule.get('_encoded_enum_step', 1.0)
                    unit_step = 1.0 / step if step > 0 else 0.5
                    new_val = max_encoded + np.random.choice([1, 2, 3]) * unit_step
                else:
                    # 无编码工具或转换失败：用数据矩阵中该列的实际值范围
                    col_vals = self.X_base[:, col_idx]
                    valid_vals = col_vals[~np.isnan(col_vals)]
                    if len(valid_vals) > 0:
                        max_data = float(np.max(valid_vals))
                        std_data = float(np.std(valid_vals)) if len(valid_vals) > 1 else 1.0
                        new_val = max_data + np.random.choice([1, 2, 3]) * max(std_data * 0.5, 0.1)
                    else:
                        new_val = np.random.choice([1, 2, 3])

            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                if rule.get('_encoding_converted'):
                    # 使用预转换的编码空间边界
                    encoded_min = rule['_encoded_min']
                    encoded_max = rule['_encoded_max']
                    unit_step = rule.get('_encoded_unit_step', 0.5)
                    if np.random.random() < 0.5:
                        # 超出上界
                        new_val = encoded_max + np.random.randint(1, 6) * unit_step
                    else:
                        # 超出下界
                        new_val = encoded_min - np.random.randint(1, 6) * unit_step
                else:
                    # 无编码工具：使用原始值（向后兼容，可能不准确）
                    min_v = rule['min_val']
                    max_v = rule['max_val']
                    if np.random.random() < 0.5:
                        new_val = max_v + np.random.randint(1, 6)
                    else:
                        new_val = min_v - np.random.randint(1, 6)

            if new_val is not None and abs(new_val - original_val) > 1e-6:
                X_dirty[idx, col_idx] = new_val
                noise = new_val - original_val
                # DOMAIN 违规归类为 syntactic（与 AutoDetector Stage 2c 一致）
                injected['syntactic'].append((idx, col_idx, original_val, noise))
                used_indices.add((idx, col_idx))
                count += 1

        return count

    def _inject_cfd_violations(self, X_dirty: np.ndarray, y_dirty: np.ndarray,
                                n_target: int,
                                used_indices: Set[Tuple[int, int]],
                                injected: Dict[str, List]) -> int:
        """注入 CFD 违规（条件函数依赖违反）

        Example: class=2, n_anomaly<=2 => CT EXCESS >= 5 FROM_BASELINE 5
        → 在 class=2 的行中，将 CT 注入为 baseline + threshold + rand(0,2)
        """
        n_samples = len(X_dirty)
        count = 0

        # 按 class 值分组行索引
        class_row_map: Dict[str, List[int]] = defaultdict(list)
        for i in range(n_samples):
            if not np.isnan(y_dirty[i]):
                # 使用通用字符串转换（回归标签可能是浮点值）
                try:
                    class_key = str(int(y_dirty[i])) if y_dirty[i] == int(y_dirty[i]) else str(y_dirty[i])
                except (ValueError, OverflowError):
                    class_key = str(y_dirty[i])
                class_row_map[class_key].append(i)

        # 遍历所有 CFD 规则
        all_cfd_rules = []
        for class_val, rules in self._cfd_col_map.items():
            for rule in rules:
                all_cfd_rules.append((class_val, rule))

        if not all_cfd_rules:
            return 0

        per_rule = max(1, n_target // len(all_cfd_rules))

        for class_val, rule in all_cfd_rules:
            if count >= n_target:
                break

            target_col_name = rule.get('target_col', '')
            if target_col_name not in self.column_names:
                continue
            col_idx = self.column_names.index(target_col_name)

            direction = rule.get('direction', 'EXCESS')
            threshold = rule.get('threshold', 5.0)
            baseline = rule.get('baseline', 0.0)

            candidate_rows = class_row_map.get(class_val, [])
            if not candidate_rows:
                continue

            np.random.shuffle(candidate_rows)
            rule_count = 0

            for row_idx in candidate_rows:
                if rule_count >= per_rule or count >= n_target:
                    break
                if (row_idx, col_idx) in used_indices or np.isnan(X_dirty[row_idx, col_idx]):
                    continue

                original_val = X_dirty[row_idx, col_idx]

                # threshold 和 baseline 需要转换到 LE+SS 空间
                if rule.get('_encoding_converted') or self._has_encoding_tools:
                    # 使用编码空间的值
                    if self.scaler is not None and col_idx < len(self.scaler.mean_):
                        # 将 baseline 和 threshold 从原始空间转到 LE+SS 空间
                        mean_j = self.scaler.mean_[col_idx]
                        scale_j = self.scaler.scale_[col_idx]
                        encoded_baseline = (baseline - mean_j) / scale_j
                        encoded_threshold = threshold / scale_j  # threshold 是差值，只需除以 scale
                        encoded_delta = np.random.uniform(0, 2) / scale_j
                    else:
                        encoded_baseline = baseline
                        encoded_threshold = threshold
                        encoded_delta = np.random.uniform(0, 2)
                else:
                    encoded_baseline = baseline
                    encoded_threshold = threshold
                    encoded_delta = np.random.uniform(0, 2)

                if direction == 'EXCESS':
                    new_val = encoded_baseline + encoded_threshold + encoded_delta
                elif direction == 'DEFICIT':
                    new_val = encoded_baseline - encoded_threshold - encoded_delta
                else:
                    continue

                # DOMAIN 范围限制: 确保原始空间值在 DOMAIN 内
                # 防止被 DOMAIN 通道截获为句法错误（应为语义错误）
                if col_idx in self._domain_col_map:
                    domain_rule = self._domain_col_map[col_idx]
                    enc_min = domain_rule.get('_encoded_min')
                    enc_max = domain_rule.get('_encoded_max')
                    if enc_min is not None and enc_max is not None:
                        # 留一点余量避免边界精度问题
                        # margin = 1% DOMAIN 范围，避免边界精度问题
                        margin = abs(enc_max - enc_min) * 0.01
                        clamped = max(enc_min + margin, min(enc_max - margin, new_val))
                        # 确保 clamped 后仍满足 CFD 检测阈值
                        if direction == 'EXCESS':
                            if clamped - encoded_baseline >= encoded_threshold:
                                new_val = clamped
                            else:
                                continue  # 无法在 DOMAIN 内满足 CFD 阈值
                        elif direction == 'DEFICIT':
                            if encoded_baseline - clamped >= encoded_threshold:
                                new_val = clamped
                            else:
                                continue

                if abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, col_idx] = new_val
                    injected['semantic'].append((row_idx, col_idx, original_val, new_val))
                    used_indices.add((row_idx, col_idx))
                    rule_count += 1
                    count += 1

        return count

    def _inject_dc_violations(self, X_dirty: np.ndarray, n_target: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List]) -> int:
        """注入 DC (Denial Constraint) 违规

        DC 使用 denial 语义: 当所有 clauses 都成立时 = 约束被违反。
        注入 = 使约束被违反 = 让所有 clauses 都成立。

        策略:
          1. 找到当前不违反约束的行（至少有一个 clause 不成立）
          2. 对 MARK 列: 修改 MARK 列的值使所有 clause 都成立
          3. 对 abs_diff 类型（无 MARK）: 修改其中一列使差值超过阈值

        所有操作在编码空间(numpy数组)上进行。
        """
        n_samples = len(X_dirty)
        count = 0

        if not self._dc_rule_list:
            return 0

        # 过滤出已成功编码转换的 DC 规则
        valid_dc_rules = [r for r in self._dc_rule_list if r.get('_encoding_converted')]

        # 调试日志: DC 规则注入状态
        import logging
        _dc_logger = logging.getLogger('demandclean.error_injector')
        _dc_logger.debug(
            f"DC inject: total={len(self._dc_rule_list)}, "
            f"valid(encoded)={len(valid_dc_rules)}, target={n_target}")

        if not valid_dc_rules:
            return 0

        # 规则数多于预算时随机抽样，避免每条规则分不到 1 个
        if len(valid_dc_rules) > n_target:
            valid_dc_rules = list(np.random.choice(
                valid_dc_rules, size=n_target, replace=False))

        per_rule = max(1, n_target // len(valid_dc_rules))

        for dc_rule in valid_dc_rules:
            if count >= n_target:
                break

            clauses = dc_rule.get('clauses', [])
            mark_cols = dc_rule.get('mark_cols', [])

            if not clauses:
                continue

            # 分发到不同的注入子策略
            if mark_cols:
                injected_count = self._inject_dc_mark_violation(
                    X_dirty, clauses, mark_cols,
                    min(per_rule, n_target - count),
                    used_indices, injected)
                count += injected_count
            else:
                # 无 MARK 列: 检查是否有 abs_diff 类型
                abs_diff_clauses = [c for c in clauses if c.get('type') == 'abs_diff']
                if abs_diff_clauses:
                    injected_count = self._inject_dc_abs_diff_violation(
                        X_dirty, abs_diff_clauses,
                        min(per_rule, n_target - count),
                        used_indices, injected)
                    count += injected_count

        return count

    def _inject_dc_mark_violation(self, X_dirty: np.ndarray,
                                   clauses: List[Dict],
                                   mark_cols: List[str],
                                   n_target: int,
                                   used_indices: Set[Tuple[int, int]],
                                   injected: Dict[str, List]) -> int:
        """DC 注入: 有 MARK 列的情况

        找到满足非 MARK 条件但不满足 MARK 条件的行（当前合法行），
        然后修改 MARK 列使所有条件都成立（制造违规）。

        示例: EQ(holiday, 1) & NEQ(workingday, 0) & MARK(workingday)
          - 找 holiday=1 且 workingday=0 的行（当前合法，因为 NEQ(workingday,0) 不成立）
          - 把 workingday 改为非0值（如1），使 NEQ(workingday,0) 成立 → 约束被违反
        """
        n_samples = len(X_dirty)
        count = 0

        # 分离非 MARK 子句和 MARK 子句
        non_mark_clauses = []
        mark_clauses = []  # MARK 列对应的条件子句

        for clause in clauses:
            col_name = clause.get('col', '')
            if col_name in mark_cols:
                mark_clauses.append(clause)
            else:
                non_mark_clauses.append(clause)

        # 解析 MARK 列的列索引
        mark_col_indices = []
        for mc in mark_cols:
            if mc in self.column_names:
                mark_col_indices.append(self.column_names.index(mc))
            else:
                return 0  # MARK 列不在特征中，跳过

        if not mark_col_indices:
            return 0

        # 找到满足所有非 MARK 条件的候选行
        candidate_rows = []
        for i in range(n_samples):
            all_non_mark_satisfied = True
            for clause in non_mark_clauses:
                if not self._evaluate_clause(X_dirty, i, clause):
                    all_non_mark_satisfied = False
                    break

            if all_non_mark_satisfied:
                # 检查 MARK 条件是否不满足（当前合法 = 不违反约束）
                any_mark_unsatisfied = False
                for clause in mark_clauses:
                    if not self._evaluate_clause(X_dirty, i, clause):
                        any_mark_unsatisfied = True
                        break

                # 如果没有 mark_clauses（MARK 列没有对应的条件子句），
                # 也将其作为候选（可以直接注入使约束违反）
                if any_mark_unsatisfied or not mark_clauses:
                    candidate_rows.append(i)

        if not candidate_rows:
            return 0

        np.random.shuffle(candidate_rows)

        for row_idx in candidate_rows:
            if count >= n_target:
                break

            # 对每个 MARK 列，生成使对应 clause 成立的值
            for mc_idx, mark_col_name in zip(mark_col_indices, mark_cols):
                if (row_idx, mc_idx) in used_indices or np.isnan(X_dirty[row_idx, mc_idx]):
                    continue

                original_val = X_dirty[row_idx, mc_idx]
                new_val = self._compute_dc_mark_value(
                    X_dirty, row_idx, mc_idx, mark_col_name, clauses)

                if new_val is not None and abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, mc_idx] = new_val
                    injected['semantic'].append((row_idx, mc_idx, original_val, new_val))
                    used_indices.add((row_idx, mc_idx))
                    count += 1
                    break  # 每行只注入一个 MARK 列

        return count

    def _inject_dc_abs_diff_violation(self, X_dirty: np.ndarray,
                                       abs_diff_clauses: List[Dict],
                                       n_target: int,
                                       used_indices: Set[Tuple[int, int]],
                                       injected: Dict[str, List]) -> int:
        """DC 注入: abs_diff 类型（无 MARK 列）

        示例: GT(ABS(t1.col1 - t1.col2), threshold)
          - 找 |col1-col2| <= threshold 的行（当前合法）
          - 修改 col1 使 |col1-col2| > threshold（制造违规）
        """
        n_samples = len(X_dirty)
        count = 0

        for clause in abs_diff_clauses:
            if count >= n_target:
                break

            col1_idx = clause.get('_col1_idx')
            col2_idx = clause.get('_col2_idx')
            encoded_threshold = clause.get('_encoded_value')
            op = clause.get('op', 'GT')

            if col1_idx is None or col2_idx is None or encoded_threshold is None:
                continue

            # 找当前不违反约束的行（clause 不成立 = 合法）
            candidate_rows = []
            for i in range(n_samples):
                if (i, col1_idx) in used_indices or (i, col2_idx) in used_indices:
                    continue
                if np.isnan(X_dirty[i, col1_idx]) or np.isnan(X_dirty[i, col2_idx]):
                    continue

                abs_diff = abs(X_dirty[i, col1_idx] - X_dirty[i, col2_idx])

                # clause 不成立 = 当前合法（不违反约束）
                clause_holds = self._eval_comparison(abs_diff, op, encoded_threshold)
                if not clause_holds:
                    candidate_rows.append(i)

            if not candidate_rows:
                continue

            np.random.shuffle(candidate_rows)
            per_clause = max(1, (n_target - count) // max(1, len(abs_diff_clauses)))

            clause_count = 0
            for row_idx in candidate_rows:
                if clause_count >= per_clause or count >= n_target:
                    break

                # 选择修改 col1（随机也可以选 col2）
                target_col = col1_idx if np.random.random() < 0.5 else col2_idx
                other_col = col2_idx if target_col == col1_idx else col1_idx

                if (row_idx, target_col) in used_indices:
                    continue

                original_val = X_dirty[row_idx, target_col]
                other_val = X_dirty[row_idx, other_col]

                # 使 |target - other| > threshold
                # 设 target = other + threshold + delta (或 other - threshold - delta)
                delta = encoded_threshold * np.random.uniform(0.1, 0.5)
                if np.random.random() < 0.5:
                    new_val = other_val + encoded_threshold + delta
                else:
                    new_val = other_val - encoded_threshold - delta

                # DOMAIN 范围限制: 确保注入值在 DOMAIN 内
                # 防止被 DOMAIN 通道截获为句法错误
                if target_col in self._domain_col_map:
                    domain_rule = self._domain_col_map[target_col]
                    enc_min = domain_rule.get('_encoded_min')
                    enc_max = domain_rule.get('_encoded_max')
                    if enc_min is not None and enc_max is not None:
                        # margin = 1% DOMAIN 范围，避免边界精度问题
                        margin = abs(enc_max - enc_min) * 0.01
                        clamped = max(enc_min + margin, min(enc_max - margin, new_val))
                        # 确保 clamped 后仍满足 DC 违规条件
                        if abs(clamped - other_val) > encoded_threshold:
                            new_val = clamped
                        else:
                            continue  # 无法在 DOMAIN 内违反 DC 约束 → 跳过

                if abs(new_val - original_val) > 1e-6:
                    X_dirty[row_idx, target_col] = new_val
                    injected['semantic'].append((row_idx, target_col, original_val, new_val))
                    used_indices.add((row_idx, target_col))
                    clause_count += 1
                    count += 1

        return count

    def _evaluate_clause(self, X_dirty: np.ndarray, row_idx: int,
                          clause: Dict) -> bool:
        """评估单个 DC clause 在指定行上是否成立

        Args:
            X_dirty: 数据矩阵（编码空间）
            row_idx: 行索引
            clause: DC clause 字典

        Returns:
            True 如果 clause 条件成立
        """
        ctype = clause.get('type', '')

        if ctype == 'simple':
            col_idx = clause.get('_col_idx')
            encoded_val = clause.get('_encoded_value')
            if col_idx is None or encoded_val is None:
                return False
            if np.isnan(X_dirty[row_idx, col_idx]):
                return False

            cell_val = X_dirty[row_idx, col_idx]
            op = clause.get('op', 'EQ')
            return self._eval_comparison(cell_val, op, encoded_val)

        elif ctype == 'abs_diff':
            col1_idx = clause.get('_col1_idx')
            col2_idx = clause.get('_col2_idx')
            encoded_threshold = clause.get('_encoded_value')
            if col1_idx is None or col2_idx is None or encoded_threshold is None:
                return False
            if np.isnan(X_dirty[row_idx, col1_idx]) or np.isnan(X_dirty[row_idx, col2_idx]):
                return False

            abs_diff = abs(X_dirty[row_idx, col1_idx] - X_dirty[row_idx, col2_idx])
            op = clause.get('op', 'GT')
            return self._eval_comparison(abs_diff, op, encoded_threshold)

        return False

    @staticmethod
    def _eval_comparison(val: float, op: str, threshold: float,
                          tol: float = 1e-4) -> bool:
        """评估比较操作

        Args:
            val: 左值
            op: 操作符 (EQ, NEQ, GT, GTE, LT, LTE)
            threshold: 右值
            tol: EQ/NEQ 的容差（编码空间浮点比较）

        Returns:
            比较结果
        """
        if op == 'EQ':
            return abs(val - threshold) < tol
        elif op == 'NEQ':
            return abs(val - threshold) >= tol
        elif op == 'GT':
            return val > threshold
        elif op == 'GTE':
            return val >= threshold
        elif op == 'LT':
            return val < threshold
        elif op == 'LTE':
            return val <= threshold
        return False

    def _compute_dc_mark_value(self, X_dirty: np.ndarray, row_idx: int,
                                mark_col_idx: int, mark_col_name: str,
                                clauses: List[Dict]) -> Optional[float]:
        """计算 MARK 列应注入的值，使得所有 clauses 都成立

        找到 MARK 列对应的 clause，计算满足该 clause 的值。

        Args:
            X_dirty: 数据矩阵
            row_idx: 行索引
            mark_col_idx: MARK 列的列索引
            mark_col_name: MARK 列名
            clauses: 所有条件子句

        Returns:
            应注入的编码空间值，或 None
        """
        # 找到 MARK 列对应的 clause
        target_clause = None
        for clause in clauses:
            if clause.get('type') == 'simple' and clause.get('col') == mark_col_name:
                target_clause = clause
                break

        if target_clause is None:
            # MARK 列没有对应的条件子句
            # 尝试使用该列的其他值来制造某种违规
            col_vals = self.all_values.get(mark_col_idx, np.array([]))
            if len(col_vals) > 1:
                current_val = X_dirty[row_idx, mark_col_idx]
                # 随机选一个不同的值
                new_val = np.random.choice(col_vals)
                attempts = 0
                while abs(new_val - current_val) < 1e-4 and attempts < 10:
                    new_val = np.random.choice(col_vals)
                    attempts += 1
                if abs(new_val - current_val) > 1e-4:
                    return float(new_val)
            return None

        encoded_val = target_clause.get('_encoded_value')
        op = target_clause.get('op', 'EQ')
        scaler_scale = target_clause.get('_scaler_scale', 1.0)
        # 一个原始单位在编码空间中的步长
        unit_step = 1.0 / scaler_scale if scaler_scale > 1e-10 else 0.5

        if encoded_val is None:
            return None

        current_val = X_dirty[row_idx, mark_col_idx]

        # 根据 op 计算使条件成立的值
        if op == 'EQ':
            # 需要让 col == val → 直接设为 encoded_val
            return float(encoded_val)

        elif op == 'NEQ':
            # 需要让 col != val → 设为 encoded_val + offset
            offset = unit_step * np.random.choice([1, 2, -1, -2])
            new_val = encoded_val + offset
            # 确保确实 != encoded_val
            if abs(new_val - encoded_val) < 1e-4:
                new_val = encoded_val + unit_step
            return float(new_val)

        elif op == 'GT':
            # 需要让 col > val → 设为 val + delta
            delta = unit_step * np.random.uniform(1, 3)
            return float(encoded_val + delta)

        elif op == 'GTE':
            # 需要让 col >= val → 设为 val + small delta
            delta = unit_step * np.random.uniform(0, 2)
            return float(encoded_val + delta)

        elif op == 'LT':
            # 需要让 col < val → 设为 val - delta
            delta = unit_step * np.random.uniform(1, 3)
            return float(encoded_val - delta)

        elif op == 'LTE':
            # 需要让 col <= val → 设为 val - small delta
            delta = unit_step * np.random.uniform(0, 2)
            return float(encoded_val - delta)

        return None

    def _inject_fd_violations(self, X_dirty: np.ndarray, n_semantic: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List],
                               strict: bool = False):
        """注入违反 FD 规则的语义错误（组间交换 RHS 值）

        关键设计: 每个 FD 组最多注入严格少于半数的行，保证原始值仍是多数投票
        中的 majority，使注入的行能被 detect_fd_violations() 正确检出。

        Args:
            strict: 严格模式 — 剩余预算不 fallback 到 _inject_random_semantic
        """
        if not self.fd_col_pairs:
            return

        per_rule_budget = max(1, n_semantic // len(self.fd_col_pairs))
        total_injected = 0

        for lhs_indices, rhs_idx in self.fd_col_pairs:
            if total_injected >= n_semantic:
                break

            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(X_dirty)):
                if (i, rhs_idx) in used_indices or np.isnan(X_dirty[i, rhs_idx]):
                    continue
                lhs_vals = X_dirty[i, lhs_indices]
                if np.isnan(lhs_vals).any():
                    continue
                key = tuple(lhs_vals.tolist())
                groups[key].append(i)

            # 只选 ≥3 的组（确保注入 1 行后原始值仍严格多数）
            group_keys = [k for k in groups if len(groups[k]) >= 3]
            if len(group_keys) < 2:
                if not strict:
                    self._inject_random_semantic_for_col(
                        X_dirty, per_rule_budget, rhs_idx, used_indices, injected)
                    total_injected += per_rule_budget
                continue

            rule_injected = 0
            np.random.shuffle(group_keys)
            for i, gk in enumerate(group_keys):
                if rule_injected >= per_rule_budget:
                    break
                rows = groups[gk]
                # 每组最多注入 floor((n-1)/2) 行，保证原始值严格多数
                max_per_group = max(1, (len(rows) - 1) // 2)

                other_key = group_keys[(i + 1) % len(group_keys)]
                other_rows = groups[other_key]
                donor_row = np.random.choice(other_rows)
                donor_val = X_dirty[donor_row, rhs_idx]

                group_injected = 0
                for row_idx in rows:
                    if (rule_injected >= per_rule_budget
                            or total_injected >= n_semantic
                            or group_injected >= max_per_group):
                        break
                    if (row_idx, rhs_idx) in used_indices:
                        continue
                    original_val = X_dirty[row_idx, rhs_idx]
                    if abs(donor_val - original_val) > 1e-6:
                        X_dirty[row_idx, rhs_idx] = donor_val
                        injected['semantic'].append(
                            (row_idx, rhs_idx, original_val, donor_val))
                        used_indices.add((row_idx, rhs_idx))
                        rule_injected += 1
                        total_injected += 1
                        group_injected += 1

        remaining = n_semantic - total_injected
        if remaining > 0 and not strict:
            self._inject_random_semantic(X_dirty, remaining, used_indices, injected)

    def _inject_random_semantic_for_col(self, X_dirty: np.ndarray, n: int,
                                         col: int,
                                         used_indices: Set[Tuple[int, int]],
                                         injected: Dict[str, List]):
        """对指定列进行随机语义错误注入"""
        n_samples = len(X_dirty)
        count = 0
        for _ in range(n * 3):
            if count >= n:
                break
            idx = np.random.randint(0, n_samples)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                candidates = self.all_values.get(col, np.array([]))
                if len(candidates) > 1:
                    new_val = np.random.choice(candidates)
                    attempts = 0
                    while abs(new_val - original_val) < 0.01 and attempts < 10:
                        new_val = np.random.choice(candidates)
                        attempts += 1
                    X_dirty[idx, col] = new_val
                    injected['semantic'].append((idx, col, original_val, new_val))
                    used_indices.add((idx, col))
                    count += 1

    def _inject_random_semantic(self, X_dirty: np.ndarray, n_semantic: int,
                                 used_indices: Set[Tuple[int, int]],
                                 injected: Dict[str, List]):
        """无规则时的随机语义错误注入（用同列其他值替换）"""
        n_samples, n_features = X_dirty.shape
        for _ in range(n_semantic):
            idx = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                original_val = X_dirty[idx, col]
                candidates = self.all_values.get(col, np.array([]))
                if len(candidates) > 1:
                    new_val = np.random.choice(candidates)
                    attempts = 0
                    while abs(new_val - original_val) < 0.01 and attempts < 10:
                        new_val = np.random.choice(candidates)
                        attempts += 1
                    X_dirty[idx, col] = new_val
                    injected['semantic'].append((idx, col, original_val, new_val))
                    used_indices.add((idx, col))

    # ====================================================================
    # 3. 句法错误注入（RAHA-aware 统计驱动，不使用规则）
    # ====================================================================

    def _inject_raha_aware_syntactic(self, X_dirty: np.ndarray, n_syntactic: int,
                                      used_indices: Set[Tuple[int, int]],
                                      injected: Dict[str, List]):
        """RAHA-aware 句法错误注入

        三种子策略模拟 RAHA 检测的逆过程:
          A (40%): OD-Gaussian 可检测 → 注入 2~4σ 偏离
          B (30%): OD-Histogram 可检测 → 注入极端分位数外的值
          C (30%): PVD 可检测 → 量级异常模拟（*10, *11, 符号翻转）

        避开 FD LHS 列以防止 FD 检测产生级联误报。
        """
        n_samples, n_features = X_dirty.shape

        # 句法注入可用列 = 全部列 - FD LHS 列
        eligible_cols = [c for c in range(n_features) if c not in self._fd_lhs_cols]
        if not eligible_cols:
            eligible_cols = list(range(n_features))  # 全是 LHS 时退化

        # 按比例分配
        n_gaussian = int(n_syntactic * 0.4)
        n_histogram = int(n_syntactic * 0.3)
        n_pvd = n_syntactic - n_gaussian - n_histogram

        # 策略 A: OD-Gaussian (3~5 sigma 偏离)
        self._inject_syntactic_gaussian(X_dirty, n_gaussian, used_indices, injected, eligible_cols)

        # 策略 B: OD-Histogram (0.3~1.0 倍 IQR99 偏移)
        self._inject_syntactic_histogram(X_dirty, n_histogram, used_indices, injected, eligible_cols)

        # 策略 C: PVD (量级异常模拟)
        self._inject_syntactic_pvd(X_dirty, n_pvd, used_indices, injected, eligible_cols)

    def _inject_syntactic_gaussian(self, X_dirty: np.ndarray, n: int,
                                    used_indices: Set[Tuple[int, int]],
                                    injected: Dict[str, List],
                                    eligible_cols: Optional[List[int]] = None):
        """策略 A: 注入 3~5σ 偏离值（让 RAHA 的 OD-Gaussian 模型能可靠检测到）

        量级说明: 3-5σ 在标准正态下的概率 < 0.3%，足以被 z-score 检测器发现，
        同时不会过于极端导致 RAHA 元分类器对标注样本过拟合。
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # 分类列走 typo 路径
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # 跳过数值列逻辑

                original_val = X_dirty[idx, col]
                std = self.col_stds[col] if not np.isnan(self.col_stds[col]) else 1.0
                if std < 1e-10:
                    std = 1.0
                # 3~5 sigma 偏移（适中量级，避免 RAHA 过拟合）
                sigma_mult = np.random.uniform(3.0, 5.0)
                direction = np.random.choice([-1, 1])
                noise = direction * sigma_mult * std
                X_dirty[idx, col] = self.col_means[col] + noise
                injected['syntactic'].append((idx, col, original_val, noise))
                used_indices.add((idx, col))

    def _inject_syntactic_histogram(self, X_dirty: np.ndarray, n: int,
                                     used_indices: Set[Tuple[int, int]],
                                     injected: Dict[str, List],
                                     eligible_cols: Optional[List[int]] = None):
        """策略 B: 注入极端分位数外的值（让 RAHA 的 OD-Histogram 模型能可靠检测到）

        量级: 0.3~1.0 倍 IQR99 偏移（适中，确保在分位数边界外但不会过于极端）
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # 分类列走 typo 路径
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # 跳过数值列逻辑

                original_val = X_dirty[idx, col]

                if col in self.col_percentiles:
                    p1, p99 = self.col_percentiles[col]
                    prange = max(abs(p99 - p1), 1e-6)
                    # 0.3~1.0 倍 IQR99 偏移（超出分位数但不过于极端）
                    offset = prange * np.random.uniform(0.3, 1.0)

                    if np.random.random() < 0.5:
                        new_val = p99 + offset  # 超出 99% 分位
                    else:
                        new_val = p1 - offset   # 低于 1% 分位
                else:
                    # 无分位数信息时回退到 Gaussian
                    std = self.col_stds[col] if not np.isnan(self.col_stds[col]) else 1.0
                    new_val = original_val + np.random.choice([-1, 1]) * 3 * std

                noise = new_val - original_val
                X_dirty[idx, col] = new_val
                injected['syntactic'].append((idx, col, original_val, noise))
                used_indices.add((idx, col))

    def _inject_syntactic_pvd(self, X_dirty: np.ndarray, n: int,
                               used_indices: Set[Tuple[int, int]],
                               injected: Dict[str, List],
                               eligible_cols: Optional[List[int]] = None):
        """策略 C: 量级异常模拟（模拟字符级异常在数值空间的效果）

        - val * 10 (多一位数字)
        - round(val) * 11 (双位重复 33, 55, 88)
        - -abs(val) (符号翻转)
        """
        n_samples, n_features = X_dirty.shape
        if eligible_cols is None:
            eligible_cols = list(range(n_features))
        for _ in range(n):
            idx = np.random.randint(0, n_samples)
            col = eligible_cols[np.random.randint(0, len(eligible_cols))]
            if (idx, col) not in used_indices and not np.isnan(X_dirty[idx, col]):
                # 分类列走 typo 路径
                if col in self._cat_col_idx_set:
                    original_val = X_dirty[idx, col]
                    new_val = self._generate_categorical_typo_encoded(col, original_val)
                    if new_val is not None and abs(new_val - original_val) > 1e-6:
                        noise = new_val - original_val
                        X_dirty[idx, col] = new_val
                        injected['syntactic'].append((idx, col, original_val, noise))
                        used_indices.add((idx, col))
                    continue  # 跳过数值列逻辑

                original_val = X_dirty[idx, col]

                strategy = np.random.choice(['mul10', 'double_digit', 'sign_flip'])
                if strategy == 'mul10':
                    new_val = original_val * 10
                elif strategy == 'double_digit':
                    base = max(1, abs(int(round(original_val))))
                    new_val = base * 11.0  # e.g., 3 → 33, 5 → 55
                else:  # sign_flip
                    new_val = -abs(original_val) if original_val > 0 else abs(original_val) + 1

                noise = new_val - original_val
                if abs(noise) > 1e-6:  # 确保确实改变了值
                    X_dirty[idx, col] = new_val
                    injected['syntactic'].append((idx, col, original_val, noise))
                    used_indices.add((idx, col))

    # ====================================================================
    # 4. 标签错误注入（条件性，镜像检测模式）
    # ====================================================================

    def _inject_label_noise(self, y_dirty: np.ndarray, n_label: int,
                             label_pattern: LabelErrorPattern,
                             injected: Dict[str, List],
                             exclude_rows: Optional[Set[int]] = None):
        """根据检测到的标签错误模式注入标签噪声（规则感知）

        条件性注入: 只在检测器发现标签错误时才调用此方法。
        - 分类: 优先翻转规则覆盖的行，再按 flip_matrix 翻转
        - 回归: 加高斯噪声 (std = 估计的 noise_std)

        Args:
            exclude_rows: 应排除的行索引集合（已有特征错误的行）
        """
        n_samples = len(y_dirty)
        if n_label <= 0:
            return

        # 回归任务: 高斯噪声注入
        if label_pattern.is_regression:
            valid_indices = [i for i in range(n_samples)
                             if not np.isnan(y_dirty[i])
                             and (exclude_rows is None or i not in exclude_rows)]
            if not valid_indices:
                return
            np.random.shuffle(valid_indices)
            count = 0
            noise_std = label_pattern.noise_std if label_pattern.noise_std > 0 else label_pattern.label_std * 0.2
            for idx in valid_indices:
                if count >= n_label:
                    break
                original_val = y_dirty[idx]
                # 高斯噪声
                noise = np.random.normal(0, noise_std)
                new_val = original_val + noise
                if abs(noise) > 1e-8:
                    y_dirty[idx] = new_val
                    injected['label_noise'].append((idx, -1, original_val, new_val))
                    count += 1
            return

        # 分类任务: 规则感知翻转
        if not label_pattern.unique_classes:
            return

        valid_indices = [i for i in range(n_samples)
                         if not np.isnan(y_dirty[i])
                         and (exclude_rows is None or i not in exclude_rows)]
        if not valid_indices:
            return

        # 找到规则覆盖的候选行（编码空间）
        rule_aware = self._find_encoded_rule_aware_candidates(y_dirty, valid_indices)

        # 优先翻转规则覆盖的行
        priority_order = rule_aware + [
            i for i in valid_indices if i not in set(rule_aware)]
        np.random.shuffle(valid_indices)  # 非规则部分随机化

        count = 0
        for idx in priority_order:
            if count >= n_label:
                break

            current_label = y_dirty[idx]
            new_label = self._sample_flip(current_label, label_pattern)

            if new_label is not None and new_label != current_label:
                original_val = y_dirty[idx]
                y_dirty[idx] = new_label
                injected['label_noise'].append((idx, -1, original_val, new_label))
                count += 1

    def _find_encoded_rule_aware_candidates(
        self,
        y: np.ndarray,
        valid_indices: List[int],
    ) -> List[int]:
        """在编码空间中找到规则感知的标签翻转候选

        遍历 CFD 规则，找到当前标签与规则期望不同但满足特征条件的行。
        翻转后这些行满足完整规则条件，能被检测到。

        编码空间无法方便地检查全部特征条件（分类列需逆编码），
        因此仅检查标签方向是否有规则覆盖，保证翻转方向正确。

        Returns:
            候选行索引列表（已随机打乱）
        """
        candidates = set()

        # 收集所有有标签规则覆盖的方向
        covered_directions = set()  # 规则期望的标签值（翻转后的目标值）
        for class_val_str in self._cfd_col_map.keys():
            try:
                covered_directions.add(float(class_val_str))
            except (ValueError, TypeError):
                continue

        # DC 标签规则
        label_names = {'class'}
        if self.label_col:
            label_names.add(self.label_col)

        for dc_rule in self._dc_rule_list:
            mark_cols = dc_rule.get('mark_cols', [])
            if not mark_cols:
                continue
            is_label_dc = any(mc in label_names for mc in mark_cols)
            if not is_label_dc:
                continue

            clauses = dc_rule.get('clauses', [])
            for clause in clauses:
                col = clause.get('col', '')
                if col in label_names and clause.get('op') == 'EQ':
                    val = clause.get('_encoded_value')
                    if val is not None:
                        covered_directions.add(float(val))
                    break

        if not covered_directions:
            return []

        # 对每个有规则覆盖的方向，收集当前标签不等于该值的行（翻转后匹配）
        for idx in valid_indices:
            current_label = y[idx]
            for target_val in covered_directions:
                if abs(current_label - target_val) > 1e-6:
                    # 当前标签 ≠ target_val，翻转后 = target_val，规则能覆盖
                    candidates.add(idx)
                    break

        result = list(candidates)
        np.random.shuffle(result)
        return result

    def _sample_flip(self, current_label: float,
                     pattern: LabelErrorPattern) -> Optional[float]:
        """按检测到的翻转分布采样目标类别"""
        if pattern.flip_matrix:
            # 收集从 current_label 出发的翻转目标
            targets = []
            weights = []
            for (from_cls, to_cls), cnt in pattern.flip_matrix.items():
                if abs(from_cls - current_label) < 1e-6:
                    targets.append(to_cls)
                    weights.append(cnt)

            if targets:
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
                return np.random.choice(targets, p=weights)

        # 无 flip_matrix 时：随机翻转到其他类
        others = [c for c in pattern.unique_classes if abs(c - current_label) > 1e-6]
        if others:
            return np.random.choice(others)
        return None

    # ====================================================================
    # 错误列表构建
    # ====================================================================

    def build_error_list(self,
                         injected: Dict[str, List]) -> List[Dict]:
        """
        将注入的错误转换为清洗环境需要的格式

        Args:
            injected: inject_errors 返回的错误信息

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
        """
        error_list = []

        # Missing errors (type=0)
        for idx, col, original_val in injected.get('missing', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': original_val
            })

        # Semantic errors (type=1)
        for idx, col, original_val, new_val in injected.get('semantic', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': original_val
            })

        # Syntactic errors (type=2)
        for idx, col, original_val, noise in injected.get('syntactic', []):
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': original_val
            })

        # Label noise (type=3, col=-1)
        for idx, col, original_val, new_val in injected.get('label_noise', []):
            error_list.append({
                'idx': idx,
                'col': -1,
                'type': 3,
                'repair_value': original_val
            })

        return error_list

    # ====================================================================
    # 统计信息
    # ====================================================================

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'n_samples': len(self.X_base),
            'n_features': self.X_base.shape[1],
            'col_means': self.col_means,
            'col_stds': self.col_stds,
            'n_fd_rules': len(self.fd_col_pairs),
            'has_rich_rules': bool(self.rich_rules and self.rich_rules.get('has_rich_rules')),
            'n_domain_rules': len(self._domain_col_map),
            'n_cfd_rules': sum(len(v) for v in self._cfd_col_map.values()),
            'n_dc_rules': len(self._dc_rule_list),
        }

    # ====================================================================
    # CSV 空间注入（直接操作 DataFrame 字符串）
    # ====================================================================

    def inject_csv_space(
        self,
        clean_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        categorical_cols: Set[str],
        missing_rate: float = 0.05,
        semantic_rate: float = 0.1,
        syntactic_rate: float = 0.15,
        protected_cols: Optional[Set[str]] = None,
        label_pattern: Optional['LabelErrorPattern'] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """在原始 CSV 字符串空间注入错误

        与编码空间版 inject_errors() 对齐，但直接操作 DataFrame 字符串值，
        消除 LE+SS 编码/逆编码引入的浮点精度损失和格式差异。

        Args:
            clean_df: 干净 CSV DataFrame（原始格式）
            feature_cols: 特征列名列表
            label_col: 标签列名
            categorical_cols: 分类列名集合
            missing_rate / semantic_rate / syntactic_rate: 各类错误注入率
            protected_cols: 受保护列（排除句法注入，如 FD 高频 LHS 列）
            label_pattern: 标签错误模式

        Returns:
            (dirty_df, injected) — injected 格式:
            {
                'missing': [(row_idx, col_name, original_str), ...],
                'semantic': [(row_idx, col_name, original_str, new_str), ...],
                'syntactic': [(row_idx, col_name, original_str, new_str), ...],
                'label_noise': [(row_idx, label_col, original_str, new_str), ...],
            }
        """
        dirty_df = clean_df.copy()
        n_samples = len(dirty_df)
        protected_cols = protected_cols or set()

        injected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        # 已注入位置: (row_idx, col_name)
        used: Set[Tuple[int, str]] = set()

        # 1. 缺失值注入
        n_missing = int(n_samples * missing_rate)
        self._csv_inject_missing(dirty_df, n_missing, feature_cols, used, injected)

        # 2. 语义错误注入 (FD/CFD 违规)
        n_semantic_total = int(n_samples * semantic_rate)

        # 标签预算策略（与编码空间版一致）
        has_label_rules = self._has_cfd_for_label()
        n_label = 0
        label_from_semantic = False
        if has_label_rules and label_pattern is not None:
            n_label = max(1, int(n_semantic_total * 0.2))
            label_from_semantic = True
        n_semantic = n_semantic_total - n_label if label_from_semantic else n_semantic_total

        self._csv_inject_semantic(
            dirty_df, n_semantic, feature_cols, label_col,
            categorical_cols, used, injected)

        # 3. 句法错误注入
        n_syntactic = int(n_samples * syntactic_rate)
        self._csv_inject_syntactic(
            dirty_df, n_syntactic, feature_cols,
            categorical_cols, protected_cols, used, injected)

        # 4. 标签错误注入
        if n_label > 0 and label_pattern is not None:
            self._csv_inject_label(dirty_df, n_label, label_col, label_pattern, injected)

        return dirty_df, injected

    def _csv_inject_missing(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV 空间缺失值注入：将单元格设为空字符串"""
        n_samples = len(df)
        n_cols = len(feature_cols)
        for _ in range(n * 3):
            if len(injected['missing']) >= n:
                break
            idx = np.random.randint(0, n_samples)
            col_name = feature_cols[np.random.randint(0, n_cols)]
            if (idx, col_name) in used:
                continue
            original = str(df.at[df.index[idx], col_name])
            if original == '' or original.lower() in ('nan', 'none', ''):
                continue
            df.at[df.index[idx], col_name] = ''
            injected['missing'].append((idx, col_name, original))
            used.add((idx, col_name))

    def _csv_inject_semantic(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        label_col: str,
        categorical_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV 空间语义错误注入

        三种注入源按优先级依次使用:
          1. FD 规则: 组间交换 RHS 值
          2. DC abs_diff 规则: 修改单列使 |col1-col2| > threshold
          3. CFD 特征规则: 修改特征值使之偏离类内基线
        """
        count = 0

        # --- 1. FD 规则注入 ---
        if self.fd_rules and self.column_names:
            count += self._csv_inject_semantic_fd(df, n, feature_cols, used, injected)

        # --- 2. DC abs_diff 规则注入 ---
        remaining = n - count
        if remaining > 0 and self.rich_rules and self.rich_rules.get('dc_rules'):
            count += self._csv_inject_semantic_dc(
                df, remaining, feature_cols, used, injected)

        return count

    def _csv_inject_semantic_fd(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ) -> int:
        """FD 规则: 组间交换 RHS 字符串值"""
        count = 0
        if not self.fd_rules:
            return 0

        per_rule_budget = max(1, n // max(len(self.fd_rules), 1))

        for lhs_str, rhs_str in self.fd_rules:
            if count >= n:
                break
            lhs_cols = [c.strip() for c in str(lhs_str).split(',')]
            rhs_col = str(rhs_str).strip()

            if rhs_col not in feature_cols:
                continue
            if not all(c in df.columns for c in lhs_cols):
                continue
            if rhs_col not in df.columns:
                continue

            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(df)):
                if (i, rhs_col) in used:
                    continue
                lhs_vals = tuple(str(df.at[df.index[i], c]) for c in lhs_cols)
                if any(v == '' or v.lower() in ('nan', 'none') for v in lhs_vals):
                    continue
                rhs_val = str(df.at[df.index[i], rhs_col])
                if rhs_val == '' or rhs_val.lower() in ('nan', 'none'):
                    continue
                groups[lhs_vals].append(i)

            group_keys = [k for k in groups if len(groups[k]) >= 3]
            if len(group_keys) < 2:
                continue

            rule_injected = 0
            np.random.shuffle(group_keys)

            for gi, gk in enumerate(group_keys):
                if rule_injected >= per_rule_budget or count >= n:
                    break
                rows = groups[gk]
                max_per_group = max(1, (len(rows) - 1) // 2)

                other_key = group_keys[(gi + 1) % len(group_keys)]
                other_rows = groups[other_key]
                donor_row = np.random.choice(other_rows)
                donor_val = str(df.at[df.index[donor_row], rhs_col])

                group_injected = 0
                for row_idx in rows:
                    if (rule_injected >= per_rule_budget
                            or count >= n
                            or group_injected >= max_per_group):
                        break
                    if (row_idx, rhs_col) in used:
                        continue
                    original = str(df.at[df.index[row_idx], rhs_col])
                    if original != donor_val:
                        df.at[df.index[row_idx], rhs_col] = donor_val
                        injected['semantic'].append((row_idx, rhs_col, original, donor_val))
                        used.add((row_idx, rhs_col))
                        rule_injected += 1
                        count += 1
                        group_injected += 1

        return count

    def _csv_inject_semantic_dc(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ) -> int:
        """DC abs_diff 规则在 CSV 空间注入语义违规

        对 GT(ABS(t1.col1 - t1.col2), threshold) 类型规则:
          - 找当前 |col1-col2| <= threshold 的行（合法行）
          - 修改 col1 或 col2 使差值 > threshold（制造违规）
          - 尊重 DOMAIN 约束（如 INT [1,10]）
        """
        dc_rules = self.rich_rules.get('dc_rules', [])
        if not dc_rules:
            return 0

        # 预构建 DOMAIN 范围 (col_name → (min, max))
        # rich_rules 中的 domain_rules 是 dict 列表（经 rules_to_dict 转换）
        domain_bounds: Dict[str, Tuple[float, float]] = {}
        if self.rich_rules.get('domain_rules'):
            for dr in self.rich_rules['domain_rules']:
                col_name = dr['column'] if isinstance(dr, dict) else dr.column
                min_v = dr.get('min_val') if isinstance(dr, dict) else getattr(dr, 'min_val', None)
                max_v = dr.get('max_val') if isinstance(dr, dict) else getattr(dr, 'max_val', None)
                if min_v is not None and max_v is not None:
                    domain_bounds[col_name] = (min_v, max_v)

        count = 0
        per_rule_budget = max(1, n // max(len(dc_rules), 1))

        for dc_rule in dc_rules:
            if count >= n:
                break

            # rich_rules 中的 dc_rules 是 dict 列表（经 rules_to_dict 转换）
            clauses = dc_rule['clauses'] if isinstance(dc_rule, dict) else dc_rule.clauses
            # 只处理无 MARK 的纯 abs_diff 规则
            if len(clauses) != 1 or clauses[0].get('type') != 'abs_diff':
                continue
            mark_cols = dc_rule.get('mark_cols', []) if isinstance(dc_rule, dict) else dc_rule.mark_cols
            if mark_cols:
                continue

            clause = clauses[0]
            col1, col2 = clause['col1'], clause['col2']
            threshold = clause['value']

            if col1 not in df.columns or col2 not in df.columns:
                continue

            # 收集合法行
            candidates = []
            for i in range(len(df)):
                if (i, col1) in used or (i, col2) in used:
                    continue
                try:
                    v1 = float(str(df.at[df.index[i], col1]).strip())
                    v2 = float(str(df.at[df.index[i], col2]).strip())
                except (ValueError, TypeError):
                    continue
                if abs(v1 - v2) <= threshold:
                    candidates.append((i, v1, v2))

            np.random.shuffle(candidates)
            rule_count = 0

            for idx, v1, v2 in candidates:
                if rule_count >= per_rule_budget or count >= n:
                    break

                # 随机选择修改 col1 或 col2
                target_col = col1 if np.random.random() < 0.5 else col2
                other_val = v2 if target_col == col1 else v1

                # 计算新值: 使 |new - other| > threshold
                delta = threshold * np.random.uniform(0.2, 0.8)
                if np.random.random() < 0.5:
                    new_val = other_val + threshold + delta
                else:
                    new_val = other_val - threshold - delta

                # DOMAIN 约束
                if target_col in domain_bounds:
                    lo, hi = domain_bounds[target_col]
                    new_val = max(lo, min(hi, new_val))
                    # 检查 clamp 后是否仍违规
                    if abs(new_val - other_val) <= threshold:
                        # 尝试另一个方向
                        new_val = other_val + threshold + delta
                        new_val = max(lo, min(hi, new_val))
                        if abs(new_val - other_val) <= threshold:
                            new_val = other_val - threshold - delta
                            new_val = max(lo, min(hi, new_val))
                            if abs(new_val - other_val) <= threshold:
                                continue  # 无法在 DOMAIN 内制造违规

                # 判断是否为整数列（看原始值格式）
                original_str = str(df.at[df.index[idx], target_col]).strip()
                try:
                    int(original_str)
                    new_val = int(round(new_val))
                except (ValueError, TypeError):
                    new_val = round(new_val, 6)

                new_str = str(new_val)
                if new_str != original_str:
                    df.at[df.index[idx], target_col] = new_str
                    injected['semantic'].append((idx, target_col, original_str, new_str))
                    used.add((idx, target_col))
                    rule_count += 1
                    count += 1

        return count

    def _csv_inject_syntactic(
        self,
        df: pd.DataFrame,
        n: int,
        feature_cols: List[str],
        categorical_cols: Set[str],
        protected_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV 空间句法错误注入

        数值列: 3-5σ 偏离 / ×10 / 符号翻转
        分类列: 直接 generate_typo()
        DOMAIN 违规: 超出规则定义的值域
        """
        n_samples = len(df)
        eligible_cols = [c for c in feature_cols if c not in protected_cols]
        if not eligible_cols:
            eligible_cols = list(feature_cols)

        # 收集每列的 CSV 空间统计量（数值列）
        col_stats: Dict[str, Dict] = {}
        for col_name in eligible_cols:
            if col_name in categorical_cols:
                continue
            vals = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(vals) > 1:
                col_stats[col_name] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'p1': float(vals.quantile(0.01)),
                    'p99': float(vals.quantile(0.99)),
                }

        # DOMAIN 违规部分（30% 预算）
        n_domain = 0
        domain_cols_csv = {}  # col_name → domain_rule
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            for rule in self.rich_rules.get('domain_rules', []):
                col_name = rule.get('column', '')
                if col_name in eligible_cols:
                    domain_cols_csv[col_name] = rule
            if domain_cols_csv:
                n_domain = int(n * 0.3)
                self._csv_inject_domain(df, n_domain, domain_cols_csv, categorical_cols,
                                        used, injected)

        # 统计异常部分（剩余预算）
        n_stat = n - n_domain
        count = 0

        # 按策略分配: 40% gaussian, 30% histogram, 30% pvd
        n_gaussian = int(n_stat * 0.4)
        n_histogram = int(n_stat * 0.3)
        n_pvd = n_stat - n_gaussian - n_histogram

        for sub_n, strategy in [(n_gaussian, 'gaussian'),
                                 (n_histogram, 'histogram'),
                                 (n_pvd, 'pvd')]:
            for _ in range(sub_n * 3):
                if count >= n_stat:
                    break
                idx = np.random.randint(0, n_samples)
                col_name = eligible_cols[np.random.randint(0, len(eligible_cols))]
                if (idx, col_name) in used:
                    continue
                original = str(df.at[df.index[idx], col_name])
                if original == '' or original.lower() in ('nan', 'none'):
                    continue

                # 分类列: 格式异常注入（短字符串）或 typo（长字符串）
                if col_name in categorical_cols:
                    new_val = self._generate_csv_categorical_anomaly(original, col_name)
                    if new_val != original:
                        df.at[df.index[idx], col_name] = new_val
                        injected['syntactic'].append((idx, col_name, original, new_val))
                        used.add((idx, col_name))
                        count += 1
                    continue

                # 数值列
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue

                stats = col_stats.get(col_name)
                if stats is None:
                    continue

                new_float = None

                if strategy == 'gaussian':
                    std = stats['std'] if stats['std'] > 1e-10 else 1.0
                    sigma_mult = np.random.uniform(3.0, 5.0)
                    direction = np.random.choice([-1, 1])
                    new_float = stats['mean'] + direction * sigma_mult * std

                elif strategy == 'histogram':
                    p1, p99 = stats['p1'], stats['p99']
                    prange = max(abs(p99 - p1), 1e-6)
                    offset = prange * np.random.uniform(0.3, 1.0)
                    if np.random.random() < 0.5:
                        new_float = p99 + offset
                    else:
                        new_float = p1 - offset

                elif strategy == 'pvd':
                    pvd_strat = np.random.choice(['mul10', 'double_digit', 'sign_flip'])
                    if pvd_strat == 'mul10':
                        new_float = orig_float * 10
                    elif pvd_strat == 'double_digit':
                        base = max(1, abs(int(round(orig_float))))
                        new_float = float(base * 11)
                    else:
                        new_float = -abs(orig_float) if orig_float > 0 else abs(orig_float) + 1

                if new_float is not None and abs(new_float - orig_float) > 1e-6:
                    # 保持原始列的整数/浮点格式
                    if '.' not in original and original.lstrip('-').isdigit():
                        new_str = str(int(round(new_float)))
                    else:
                        new_str = f"{new_float:.6g}"
                    df.at[df.index[idx], col_name] = new_str
                    injected['syntactic'].append((idx, col_name, original, new_str))
                    used.add((idx, col_name))
                    count += 1

    def _generate_csv_categorical_anomaly(
        self, original: str, col_name: str
    ) -> str:
        """为分类列生成格式异常值（CSV 空间）

        对短字符串(len<=2): 使用格式异常注入，确保 RAHA/DOMAIN 能检测:
          - 数字混入: "a" → "a1", "az" → "a2z"
          - 特殊字符: "a" → "a_", "az" → "a-z"
          - 重复字符: "a" → "aa", "az" → "azz"

        对长字符串(len>=3): 标准 generate_typo()，但验证结果不在 ENUM 中。
        如果 typo 后仍是合法值，则 fallback 到格式异常注入。

        Args:
            original: 原始字符串值
            col_name: 列名（用于查 DOMAIN 规则中的 ENUM 列表）

        Returns:
            异常值字符串（保证与 original 不同）
        """
        # 获取该列的 ENUM 合法值列表（用于验证注入值确实不在合法范围内）
        enum_vals = set()
        if self.rich_rules and self.rich_rules.get('has_rich_rules'):
            for rule in self.rich_rules.get('domain_rules', []):
                if rule.get('column') == col_name and rule.get('dtype') == 'ENUM':
                    enum_vals = set(str(v) for v in rule.get('enum_vals', []))
                    break

        # 短字符串策略: 直接格式异常注入
        if len(original) <= 2:
            return self._format_anomaly(original, enum_vals)

        # 长字符串策略: 先尝试 generate_typo
        new_val = generate_typo(original)
        # 验证: typo 结果不应在 ENUM 中（否则 DOMAIN 检测不到）
        if new_val != original and (not enum_vals or new_val not in enum_vals):
            return new_val

        # Fallback: 格式异常注入
        return self._format_anomaly(original, enum_vals)

    @staticmethod
    def _format_anomaly(original: str, enum_vals: set) -> str:
        """生成格式异常值

        四种策略随机选择，确保结果不在 ENUM 中:
          1. 数字混入: 在随机位置插入数字
          2. 特殊字符: 添加下划线/连字符
          3. 重复字符: 重复末尾字符
          4. 空格混入: 中间加空格

        Args:
            original: 原始字符串
            enum_vals: ENUM 合法值集合

        Returns:
            异常值（保证与 original 不同，且尽量不在 enum_vals 中）
        """
        import random as _rng

        strategies = ['digit', 'special', 'repeat']
        if len(original) >= 2:
            strategies.append('space')

        _rng.shuffle(strategies)

        for strategy in strategies:
            if strategy == 'digit':
                # 在末尾插入数字
                digit = str(_rng.randint(1, 9))
                candidate = original + digit
            elif strategy == 'special':
                # 添加下划线
                candidate = original + '_'
            elif strategy == 'repeat':
                # 重复末尾字符
                candidate = original + original[-1]
            else:  # space
                # 中间加空格
                pos = _rng.randint(1, len(original) - 1)
                candidate = original[:pos] + ' ' + original[pos:]

            if candidate != original and candidate not in enum_vals:
                return candidate

        # 兜底: 原值 + 数字后缀
        return original + '1'

    def _csv_inject_domain(
        self,
        df: pd.DataFrame,
        n: int,
        domain_cols: Dict[str, Dict],
        categorical_cols: Set[str],
        used: Set[Tuple[int, str]],
        injected: Dict[str, List],
    ):
        """CSV 空间 DOMAIN 违规注入（超出规则定义的合法值域）"""
        n_samples = len(df)
        col_names = list(domain_cols.keys())
        count = 0

        for _ in range(n * 3):
            if count >= n:
                break
            col_name = np.random.choice(col_names)
            idx = np.random.randint(0, n_samples)
            if (idx, col_name) in used:
                continue
            original = str(df.at[df.index[idx], col_name])
            if original == '' or original.lower() in ('nan', 'none'):
                continue

            rule = domain_cols[col_name]
            new_str = None

            if rule.get('dtype') == 'ENUM':
                enum_vals = rule.get('enum_vals', [])
                if enum_vals and col_name in categorical_cols:
                    # 生成不在枚举中的 typo 值
                    new_str = generate_typo(original)
                    if new_str in enum_vals:
                        new_str = original + '_invalid'
            elif rule.get('min_val') is not None and rule.get('max_val') is not None:
                min_v = rule['min_val']
                max_v = rule['max_val']
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue
                if np.random.random() < 0.5:
                    new_float = max_v + np.random.randint(1, 6)
                else:
                    new_float = min_v - np.random.randint(1, 6)
                # 保持整数/浮点格式
                if '.' not in original and original.lstrip('-').isdigit():
                    new_str = str(int(round(new_float)))
                else:
                    new_str = f"{new_float:.6g}"

            if new_str is not None and new_str != original:
                df.at[df.index[idx], col_name] = new_str
                injected['syntactic'].append((idx, col_name, original, new_str))
                used.add((idx, col_name))
                count += 1

    def _csv_inject_label(
        self,
        df: pd.DataFrame,
        n: int,
        label_col: str,
        label_pattern: 'LabelErrorPattern',
        injected: Dict[str, List],
    ):
        """CSV 空间标签错误注入（规则感知）

        分类任务: 优先翻转满足 CFD/DC 标签规则条件的行（确保翻转后规则能检测到）
        回归: 加高斯噪声（不变）
        """
        if label_col not in df.columns:
            return

        n_samples = len(df)
        valid_indices = [i for i in range(n_samples)
                         if str(df.at[df.index[i], label_col]).strip() not in
                         ('', 'nan', 'none', 'None', 'NaN')]
        if not valid_indices:
            return

        np.random.shuffle(valid_indices)
        count = 0

        if label_pattern.is_regression:
            noise_std = label_pattern.noise_std if label_pattern.noise_std > 0 else label_pattern.label_std * 0.2
            for idx in valid_indices:
                if count >= n:
                    break
                original = str(df.at[df.index[idx], label_col])
                try:
                    orig_float = float(original)
                except (ValueError, TypeError):
                    continue
                noise = np.random.normal(0, noise_std)
                new_float = orig_float + noise
                if abs(noise) > 1e-8:
                    if '.' not in original and original.lstrip('-').isdigit():
                        new_str = str(int(round(new_float)))
                    else:
                        new_str = f"{new_float:.6g}"
                    df.at[df.index[idx], label_col] = new_str
                    injected['label_noise'].append((idx, label_col, original, new_str))
                    count += 1
        else:
            # 分类任务: 规则感知标签注入
            label_values = [str(df.at[df.index[i], label_col])
                            for i in valid_indices]
            unique_labels = list(set(label_values))
            if len(unique_labels) < 2:
                return

            # 收集规则覆盖的行（翻转后至少有一条规则能检测到）
            rule_aware_indices = self._find_rule_aware_label_candidates(
                df, label_col, valid_indices, unique_labels)

            # 优先翻转规则覆盖的行，再随机翻转其余行
            priority_order = rule_aware_indices + [
                i for i in valid_indices if i not in set(rule_aware_indices)]

            for idx in priority_order:
                if count >= n:
                    break
                original = str(df.at[df.index[idx], label_col])
                others = [l for l in unique_labels if l != original]
                if not others:
                    continue
                new_label = np.random.choice(others)
                df.at[df.index[idx], label_col] = new_label
                injected['label_noise'].append((idx, label_col, original, new_label))
                count += 1

    def _find_rule_aware_label_candidates(
        self,
        df: pd.DataFrame,
        label_col: str,
        valid_indices: List[int],
        unique_labels: List[str],
    ) -> List[int]:
        """找到翻转后至少有一条 CFD/DC 标签规则能检测到的行

        对于每条 CFD 标签规则 (conditions 中包含标签列):
          - 找满足非标签条件的行
          - 翻转标签后，该行满足完整条件 → 规则能检测到

        Example:
          规则: income=1, capital_gain>=5000 => income EXCESS >= 1
          当前行: income=0 (高收入), capital_gain=8000
          → 翻转后 income=1，满足条件，规则能检测

        Returns:
            规则覆盖的候选行索引列表（已随机打乱）
        """
        candidates = set()

        if not self.rich_rules or not self.rich_rules.get('has_rich_rules'):
            return []

        # 遍历 CFD 规则
        for rule in self.rich_rules.get('cfd_rules', []):
            conditions = rule.get('conditions', [])
            # 找标签条件和非标签条件
            label_conds = []
            feature_conds = []
            for col, op, val in conditions:
                if col == label_col or col == 'class':
                    label_conds.append((col, op, val))
                else:
                    feature_conds.append((col, op, val))

            if not label_conds:
                continue  # 不涉及标签列的规则跳过

            # 确定规则期望的标签值
            rule_label_val = None
            for col, op, val in label_conds:
                if op == '=':
                    rule_label_val = val
                    break

            if rule_label_val is None:
                continue

            # 找当前标签不等于规则期望值（翻转后等于）且满足特征条件的行
            for idx in valid_indices:
                current_label = str(df.at[df.index[idx], label_col])
                if current_label == rule_label_val:
                    continue  # 当前已满足标签条件 — 翻转反而会移出覆盖范围

                # 检查非标签特征条件
                if self._csv_check_feature_conditions(df, idx, feature_conds):
                    candidates.add(idx)

        # DC 标签规则 (MARK 列是标签列)
        for dc_rule in self.rich_rules.get('dc_rules', []):
            mark_cols = dc_rule.get('mark_cols', [])
            if label_col not in mark_cols and 'class' not in mark_cols:
                continue

            clauses = dc_rule.get('clauses', [])
            # 找标签 EQ 条件
            label_eq_val = None
            non_label_clauses = []
            for clause in clauses:
                col = clause.get('col', '')
                if (col == label_col or col == 'class') and clause.get('op') == 'EQ':
                    label_eq_val = clause.get('value')
                elif col not in mark_cols:
                    non_label_clauses.append(clause)

            if label_eq_val is None:
                continue

            label_eq_str = str(int(label_eq_val)) if isinstance(label_eq_val, (int, float)) else str(label_eq_val)

            for idx in valid_indices:
                current_label = str(df.at[df.index[idx], label_col])
                if current_label == label_eq_str:
                    continue

                # 检查非标签条件
                all_ok = True
                for clause in non_label_clauses:
                    col = clause.get('col', '')
                    if col not in df.columns:
                        all_ok = False
                        break
                    try:
                        cell_val = float(df.at[df.index[idx], col])
                    except (ValueError, TypeError):
                        all_ok = False
                        break
                    op = clause.get('op', 'EQ')
                    val = clause.get('value', 0)
                    if not self._eval_comparison(cell_val, op, float(val)):
                        all_ok = False
                        break

                if all_ok:
                    candidates.add(idx)

        result = list(candidates)
        np.random.shuffle(result)
        return result

    @staticmethod
    def _csv_check_feature_conditions(
        df: pd.DataFrame, idx: int, conditions: List[Tuple]
    ) -> bool:
        """检查一行是否满足所有特征条件 (CSV 空间)

        Args:
            df: DataFrame
            idx: 行索引
            conditions: [(col, op, val), ...]

        Returns:
            True 如果所有条件都满足
        """
        for col, op, val in conditions:
            if col == 'n_anomaly':
                continue  # n_anomaly 是动态计算的伪列，跳过
            if col not in df.columns:
                return False
            try:
                cell_val = float(df.at[df.index[idx], col])
                threshold = float(val)
            except (ValueError, TypeError):
                # 字符串比较
                cell_str = str(df.at[df.index[idx], col])
                if op == '=':
                    if cell_str != val:
                        return False
                elif op == '!=':
                    if cell_str == val:
                        return False
                continue

            if op == '=' and abs(cell_val - threshold) > 1e-6:
                return False
            elif op == '!=' and abs(cell_val - threshold) < 1e-6:
                return False
            elif op == '<=' and cell_val > threshold + 1e-6:
                return False
            elif op == '>=' and cell_val < threshold - 1e-6:
                return False
            elif op == '<' and cell_val >= threshold:
                return False
            elif op == '>' and cell_val <= threshold:
                return False

        return True
