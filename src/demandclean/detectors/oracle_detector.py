"""
Oracle 检测器
=============

通过直接对比 X_dirty 和 X_clean 生成完整错误标签，跳过自动检测。
用于消融实验，提供错误检测的上界基线。

错误分类逻辑：
1. 缺失值: NaN 或字符串 "empty"
2. 句法错误: 差异 > 3 * std
3. 语义错误: 其余值差异
"""

from typing import Dict, List, Optional
import os
import pickle
import numpy as np


class OracleDetector:
    """
    Oracle 检测器（消融实验用）

    直接对比 X_dirty 和 X_clean，得到完整的错误标签。
    接口与 AutoDetector 完全兼容。
    """

    def __init__(self, column_names: Optional[List[str]] = None):
        """
        初始化 Oracle 检测器

        Args:
            column_names: 列名列表（可选，仅用于日志输出）
        """
        self.column_names = column_names
        self.col_stats: Dict[int, Dict[str, float]] = {}
        self.is_fitted = True  # Oracle 不需要训练

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """根据干净数据计算每列统计量（mean / std）"""
        for col in range(X.shape[1]):
            valid = X[:, col][~np.isnan(X[:, col])]
            if len(valid) > 0:
                self.col_stats[col] = {
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid) + 1e-6),
                    'q1': float(np.percentile(valid, 25)),
                    'q3': float(np.percentile(valid, 75)),
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'median': float(np.median(valid))
                }

    @staticmethod
    def _is_missing(value) -> bool:
        """
        判断是否为缺失值

        判定条件：
        - np.isnan（数值 NaN）
        - 字符串 "empty"（不区分大小写）
        """
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and value.strip().lower() == "empty":
            return True
        return False

    # ------------------------------------------------------------------
    # 公开接口（与 AutoDetector 对齐）
    # ------------------------------------------------------------------

    def fit(self, X_clean_subset: np.ndarray = None, verbose: bool = True) -> 'OracleDetector':
        """
        训练（无操作）

        Oracle 检测器不需要训练，此方法仅为接口兼容。

        Args:
            X_clean_subset: 干净数据子集（忽略）
            verbose: 是否打印详细信息

        Returns:
            self
        """
        if verbose:
            print("[OracleDetector] fit() 无操作，Oracle 不需要训练")
        self.is_fitted = True
        return self

    def detect(self,
               X_dirty: np.ndarray,
               X_clean: np.ndarray,
               y_dirty: Optional[np.ndarray] = None,
               y_clean: Optional[np.ndarray] = None,
               verbose: bool = True) -> Dict[str, List]:
        """
        逐单元格对比 X_dirty 和 X_clean，生成完整的错误标签。
        同时检测标签噪声（y_dirty vs y_clean）。

        Args:
            X_dirty: 脏数据，shape = (n, d)
            X_clean: 干净数据，shape = (n, d)
            y_dirty: 脏标签向量（可选）
            y_clean: 干净标签向量（可选）
            verbose: 是否打印详细信息

        Returns:
            detected: {
                'missing':     [(idx, col, clean_val), ...],
                'semantic':    [(idx, col, clean_val, dirty_val), ...],
                'syntactic':   [(idx, col, clean_val, noise), ...],
                'label_noise': [(idx, -1, clean_label, dirty_label), ...]
            }
        """
        assert X_dirty.shape == X_clean.shape, (
            f"X_dirty {X_dirty.shape} 与 X_clean {X_clean.shape} 维度不一致"
        )

        n_rows, n_cols = X_dirty.shape

        # 用干净数据计算列统计量
        self._compute_col_stats(X_clean)

        detected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': []
        }

        # ---- 特征错误检测 ----
        for i in range(n_rows):
            for col in range(n_cols):
                dirty_val = X_dirty[i, col]
                clean_val = X_clean[i, col]

                # ---- 1. 缺失值 ----
                if self._is_missing(dirty_val):
                    estimated_val = clean_val if not np.isnan(clean_val) else \
                        self.col_stats.get(col, {}).get('mean', 0)
                    detected['missing'].append((i, col, estimated_val))
                    continue

                # ---- 跳过无差异的单元格 ----
                if not np.isnan(dirty_val) and not np.isnan(clean_val):
                    if dirty_val == clean_val:
                        continue
                else:
                    if np.isnan(clean_val):
                        continue

                # ---- 2. 值存在差异，区分句法/语义 ----
                diff = abs(dirty_val - clean_val)
                col_std = self.col_stats.get(col, {}).get('std', 1.0)

                if diff > 3 * col_std:
                    noise = dirty_val - clean_val
                    detected['syntactic'].append((i, col, clean_val, noise))
                else:
                    detected['semantic'].append((i, col, clean_val, dirty_val))

        # ---- 标签噪声检测 ----
        if y_dirty is not None and y_clean is not None:
            assert len(y_dirty) == len(y_clean), (
                f"y_dirty ({len(y_dirty)}) 与 y_clean ({len(y_clean)}) 长度不一致"
            )
            for i in range(len(y_dirty)):
                d_val = y_dirty[i]
                c_val = y_clean[i]
                # 跳过两者都是 NaN
                if np.isnan(d_val) and np.isnan(c_val):
                    continue
                # 标签不同
                if np.isnan(d_val) or np.isnan(c_val) or d_val != c_val:
                    detected['label_noise'].append((i, -1, c_val, d_val))

        if verbose:
            feat_total = (len(detected['missing'])
                          + len(detected['semantic'])
                          + len(detected['syntactic']))
            label_total = len(detected['label_noise'])
            total = feat_total + label_total
            total_cells = n_rows * n_cols
            print(f"\n[OracleDetector] 检测完成:")
            print(f"  缺失值:     {len(detected['missing'])} 个")
            print(f"  语义错误:   {len(detected['semantic'])} 个")
            print(f"  句法错误:   {len(detected['syntactic'])} 个")
            print(f"  标签噪声:   {label_total} 个")
            print(f"  共计:       {total} 个错误  "
                  f"(特征: {feat_total}/{total_cells}={feat_total/max(total_cells,1):.2%}, "
                  f"标签: {label_total}/{n_rows}={label_total/max(n_rows,1):.2%})")

        return detected

    def build_error_list(self,
                         detected: Dict[str, List],
                         X_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """
        将检测到的错误转换为清洗环境需要的格式

        Args:
            detected: detect() 返回的错误字典
            X_clean: 干净数据（用于获取真值；Oracle 模式下 detected 已包含真值，
                     若提供则优先使用）

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
                type: 0=missing, 1=semantic, 2=syntactic
        """
        error_list = []

        # Missing errors (type=0)
        for item in detected['missing']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 0,
                'repair_value': repair_value
            })

        # Semantic errors (type=1)
        for item in detected['semantic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 1,
                'repair_value': repair_value
            })

        # Syntactic errors (type=2)
        for item in detected['syntactic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx,
                'col': col,
                'type': 2,
                'repair_value': repair_value
            })

        # Label noise errors (type=3, col=-1)
        for item in detected.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                # item 格式: (idx, -1, clean_label, dirty_label)
                clean_val = item[2] if len(item) > 2 else float('nan')
                repair_value = clean_val if not (isinstance(clean_val, float) and np.isnan(clean_val)) else float('nan')
                error_list.append({
                    'idx': idx,
                    'col': -1,
                    'type': 3,
                    'repair_value': repair_value
                })

        return error_list

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存检测器参数"""
        data = {
            'column_names': self.column_names,
            'col_stats': self.col_stats,
            'is_fitted': self.is_fitted
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  [OracleDetector] 已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'OracleDetector':
        """加载检测器参数"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(column_names=data.get('column_names'))
        detector.col_stats = data.get('col_stats', {})
        detector.is_fitted = data.get('is_fitted', True)
        print(f"  [OracleDetector] 已加载: {path}")
        return detector
