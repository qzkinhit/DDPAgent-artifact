"""
自动错误检测器 (AutoDetector)
==============================

多策略分层错误检测器，检测流程清晰分工：

1. 缺失值: np.isnan 直接检测（确定性）
2. 句法错误: RAHA + REGEX + DOMAIN 规则（格式/模式/值域异常）
3. 语义错误: FD + CFD(特征) + DC 规则（值的逻辑关系错误）
4. 标签噪声: CFD 标签规则（完全替代 Confident Learning）

设计原则：
- 句法检测: RAHA 为主 + REGEX/DOMAIN 规则补充（超出值域 = 句法异常）
- 语义检测: 规则驱动（FD/CFD/DC），不用统计离群值
- 标签噪声: 纯 CFD 规则，不用 Confident Learning（CL 误报太多）
- 检测率按总体 TP/FP/FN 计算，不分错误类型求均值
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import os
import re
import sys
import pickle
import shutil
import tempfile
import numpy as np
import pandas as pd

from .rule_parser import (
    ParsedRules, DomainRule, RegexRule, CFDRule, DCRule,
    parse_rules_file, load_rules,
    parse_dc_clause,
)

# 尝试导入 RAHA（优先使用项目本地 Baran_Raha，其次 pip 版本）
RAHA_AVAILABLE = False
_RAHA_SOURCE = "none"

try:
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    _baran_raha_path = os.path.join(_project_root, 'Baran_Raha')
    if os.path.isdir(_baran_raha_path):
        if _baran_raha_path not in sys.path:
            sys.path.insert(0, _baran_raha_path)
        from detection import Detection as _RahaDetection
        RAHA_AVAILABLE = True
        _RAHA_SOURCE = "Baran_Raha/detection.py"
except ImportError:
    pass

if not RAHA_AVAILABLE:
    try:
        from raha.detection import Detection as _RahaDetection
        RAHA_AVAILABLE = True
        _RAHA_SOURCE = "pip:raha"
    except ImportError:
        pass


class RahaHelper:
    """
    RAHA 辅助类 —— 直接在原始 CSV 上运行 RAHA Detection

    - 接受原始 CSV 路径，不做 numpy→CSV 转换
    - 单次运行 RAHA（内部已有 20 轮迭代标注）
    - 支持列分块（列 > max_cols 时）和行分块（行 > max_rows 时）
    """

    # 分块阈值
    MAX_COLS_PER_CHUNK = 20
    MAX_ROWS_PER_CHUNK = 10000

    @staticmethod
    def run_on_csv(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str = "data",
        labeling_budget: int = 20,
        verbose: bool = False,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """
        在原始 CSV 上运行 RAHA Detection。

        自动判断是否需要分块：
        - 列数 > MAX_COLS_PER_CHUNK → 列分块
        - 行数 > MAX_ROWS_PER_CHUNK → 行分块
        - 两者都超 → 先分列再分行（目前仅实现一种维度的分块）

        Returns:
            (detected_cells, labeled_tuple_indices, chunk_info)
            detected_cells: {(row, col): "JUST A DUMMY VALUE", ...}
            labeled_tuple_indices: RAHA 标注的行索引集合
            chunk_info: {'n_col_chunks': int, 'n_row_chunks': int,
                         'labeling_budget': int, 'raha_total_cost': int}
        """
        if not RAHA_AVAILABLE:
            if verbose:
                print("  [警告] RAHA 不可用，跳过检测")
            return {}, set(), {'n_col_chunks': 0, 'n_row_chunks': 0,
                               'labeling_budget': labeling_budget, 'raha_total_cost': 0}

        # 读取数据判断是否需要分块
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False, nrows=5)
        n_cols = len(dirty_df.columns)
        # 行数用更高效的方式获取
        with open(dirty_csv_path, 'r') as f:
            n_rows = sum(1 for _ in f) - 1  # 减去 header

        if verbose:
            print(f"  [RAHA] 数据规模: {n_rows} 行 × {n_cols} 列")

        if n_cols > RahaHelper.MAX_COLS_PER_CHUNK:
            if verbose:
                print(f"  [RAHA] 列数 {n_cols} > {RahaHelper.MAX_COLS_PER_CHUNK}，启用列分块模式")
            return RahaHelper._run_column_split(
                dirty_csv_path, clean_csv_path, dataset_name,
                labeling_budget, verbose,
                fd_rules=fd_rules, primary_key=primary_key, label_col=label_col,
            )
        elif n_rows > RahaHelper.MAX_ROWS_PER_CHUNK:
            if verbose:
                print(f"  [RAHA] 行数 {n_rows} > {RahaHelper.MAX_ROWS_PER_CHUNK}，启用行分块模式")
            # 尝试聚类分块
            grouping_key = RahaHelper._infer_grouping_key(
                dirty_df=pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False),
                fd_rules=fd_rules,
                primary_key=primary_key,
                label_col=label_col,
            )
            if grouping_key:
                if verbose:
                    print(f"  [RAHA] 推断聚类键: {grouping_key}，启用聚类分块模式")
                return RahaHelper._run_row_split_clustered(
                    dirty_csv_path, clean_csv_path,
                    dataset_name, labeling_budget, verbose,
                    grouping_key=grouping_key,
                )
            else:
                if verbose:
                    print(f"  [RAHA] 未找到聚类键，回退到等间隔分块")
                return RahaHelper._run_row_split(
                    dirty_csv_path, clean_csv_path,
                    dataset_name, labeling_budget, verbose, n_rows,
                )
        else:
            detected, labeled = RahaHelper._run_single(
                dirty_csv_path, clean_csv_path, dataset_name,
                labeling_budget, verbose,
            )
            chunk_info = {
                'n_col_chunks': 1,
                'n_row_chunks': 1,
                'labeling_budget': labeling_budget,
                'raha_total_cost': labeling_budget,
            }
            return detected, labeled, chunk_info

    @staticmethod
    def _run_column_split(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """列分块运行 RAHA：每批 ≤ MAX_COLS_PER_CHUNK 列

        策略：
        - 将特征列分成多批，每批不超过 MAX_COLS_PER_CHUNK 列
        - 每批生成子 CSV，运行 RAHA
        - 合并 detected_cells（列索引需映射回原始列索引）
        - labeled_tuples 取所有批次并集
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)

        all_cols = list(dirty_df.columns)
        max_per_chunk = RahaHelper.MAX_COLS_PER_CHUNK

        # 将列分成多批
        n_col_chunks = (len(all_cols) + max_per_chunk - 1) // max_per_chunk
        if verbose:
            print(f"  [列分块] 总列数 {len(all_cols)}, 分为 {n_col_chunks} 批 (每批≤{max_per_chunk}列)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_colsplit_{dataset_name}_')
        total_row_chunks = 0

        try:
            for chunk_idx in range(n_col_chunks):
                start = chunk_idx * max_per_chunk
                end = min(start + max_per_chunk, len(all_cols))
                chunk_cols = all_cols[start:end]

                if verbose:
                    print(f"  [列分块 {chunk_idx+1}/{n_col_chunks}] 列 {start}~{end-1}: {chunk_cols[0]}...{chunk_cols[-1]}")

                # 生成子 CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_chunk{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_chunk{chunk_idx}.csv')
                dirty_df[chunk_cols].to_csv(chunk_dirty_path, index=False)
                clean_df[chunk_cols].to_csv(chunk_clean_path, index=False)

                # 运行 RAHA（如果行数也超阈值，嵌套行分块）
                n_rows_chunk = len(dirty_df)
                if n_rows_chunk > RahaHelper.MAX_ROWS_PER_CHUNK:
                    if verbose:
                        print(f"  [列分块 {chunk_idx+1}] 行数 {n_rows_chunk} 仍超阈值，嵌套行分块")
                    # 尝试聚类分块（对子 CSV 推断聚类键）
                    chunk_dirty_df = pd.read_csv(chunk_dirty_path, dtype=str, keep_default_na=False)
                    chunk_grouping_key = RahaHelper._infer_grouping_key(
                        dirty_df=chunk_dirty_df,
                        fd_rules=fd_rules,
                        primary_key=primary_key,
                        label_col=label_col,
                    )
                    if chunk_grouping_key:
                        if verbose:
                            print(f"  [列分块 {chunk_idx+1}] 聚类键: {chunk_grouping_key}")
                        chunk_detected, chunk_labeled, sub_chunk_info = RahaHelper._run_row_split_clustered(
                            chunk_dirty_path, chunk_clean_path,
                            f"{dataset_name}_chunk{chunk_idx}",
                            labeling_budget, verbose,
                            grouping_key=chunk_grouping_key,
                        )
                    else:
                        chunk_detected, chunk_labeled, sub_chunk_info = RahaHelper._run_row_split(
                            chunk_dirty_path, chunk_clean_path,
                            f"{dataset_name}_chunk{chunk_idx}",
                            labeling_budget, verbose, n_rows_chunk,
                        )
                    total_row_chunks = max(total_row_chunks, sub_chunk_info['n_row_chunks'])
                else:
                    chunk_detected, chunk_labeled = RahaHelper._run_single(
                        chunk_dirty_path, chunk_clean_path,
                        f"{dataset_name}_chunk{chunk_idx}",
                        labeling_budget, verbose,
                    )
                    total_row_chunks = max(total_row_chunks, 1)

                # 映射列索引回原始列索引
                for (row, chunk_col), val in chunk_detected.items():
                    original_col = start + chunk_col
                    merged_detected[(row, original_col)] = val

                merged_labeled |= chunk_labeled

                if verbose:
                    print(f"  [列分块 {chunk_idx+1}] 检测: {len(chunk_detected)} 个, 标注行: {len(chunk_labeled)} 个")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [列分块] 合并结果: {len(merged_detected)} 个检测, {len(merged_labeled)} 个标注行")

        n_row_chunks_final = max(total_row_chunks, 1)
        raha_total_cost = labeling_budget * n_col_chunks * n_row_chunks_final
        chunk_info = {
            'n_col_chunks': n_col_chunks,
            'n_row_chunks': n_row_chunks_final,
            'labeling_budget': labeling_budget,
            'raha_total_cost': raha_total_cost,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _run_row_split(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        n_rows: int,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """行分块运行 RAHA：每批 ≤ MAX_ROWS_PER_CHUNK 行

        策略：
        - 等间隔分块（不按主键聚类，因为数据集不一定有主键）
        - 每批生成子 CSV，运行 RAHA
        - 合并 detected_cells（行索引需映射回原始行索引）
        - labeled_tuples 取所有批次并集（映射回原始行索引）
        - 尽量少分块（避免真值浪费）
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)

        max_per_chunk = RahaHelper.MAX_ROWS_PER_CHUNK
        n_chunks = (n_rows + max_per_chunk - 1) // max_per_chunk

        # 用户要求不要分太多块（每块消耗 20 个真值）
        # 如果块数过多，增大每块大小
        while n_chunks > 4 and max_per_chunk < n_rows:
            max_per_chunk = int(max_per_chunk * 1.5)
            n_chunks = (n_rows + max_per_chunk - 1) // max_per_chunk

        if verbose:
            print(f"  [行分块] 总行数 {n_rows}, 分为 {n_chunks} 批 (每批≤{max_per_chunk}行)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_rowsplit_{dataset_name}_')

        try:
            for chunk_idx in range(n_chunks):
                start = chunk_idx * max_per_chunk
                end = min(start + max_per_chunk, n_rows)

                if verbose:
                    print(f"  [行分块 {chunk_idx+1}/{n_chunks}] 行 {start}~{end-1} ({end-start} 行)")

                # 生成子 CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_row{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_row{chunk_idx}.csv')
                dirty_df.iloc[start:end].to_csv(chunk_dirty_path, index=False)
                clean_df.iloc[start:end].to_csv(chunk_clean_path, index=False)

                # 运行 RAHA
                chunk_detected, chunk_labeled = RahaHelper._run_single(
                    chunk_dirty_path, chunk_clean_path,
                    f"{dataset_name}_row{chunk_idx}",
                    labeling_budget, verbose,
                )

                # 映射行索引回原始行索引
                for (chunk_row, col), val in chunk_detected.items():
                    original_row = start + chunk_row
                    merged_detected[(original_row, col)] = val

                # 映射 labeled_tuples 回原始行索引
                for chunk_row in chunk_labeled:
                    merged_labeled.add(start + chunk_row)

                if verbose:
                    print(f"  [行分块 {chunk_idx+1}] 检测: {len(chunk_detected)} 个, 标注行: {len(chunk_labeled)} 个")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [行分块] 合并结果: {len(merged_detected)} 个检测, {len(merged_labeled)} 个标注行")

        chunk_info = {
            'n_col_chunks': 1,
            'n_row_chunks': n_chunks,
            'labeling_budget': labeling_budget,
            'raha_total_cost': labeling_budget * n_chunks,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _infer_grouping_key(
        dirty_df: pd.DataFrame,
        fd_rules: Optional[List] = None,
        primary_key: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Optional[List[str]]:
        """推断最适合做聚类分块的列

        优先级:
        1. 显式指定的主键 (rules.txt [PRIMARY_KEY])
        2. FD 规则的 LHS 列 (出现频次最高的)
        3. 自动推断 (cardinality ratio 在 0.01~0.5 之间的分类列)
        4. 回退: 返回 None (使用等间隔分块)
        """
        # 1. 显式主键
        if primary_key:
            valid = [c for c in primary_key if c in dirty_df.columns]
            if valid:
                return valid

        # 2. FD 的 LHS 列
        if fd_rules:
            from collections import Counter
            lhs_counter = Counter()
            for lhs_str, rhs_str in fd_rules:
                for col in lhs_str.split(','):
                    col = col.strip()
                    if col in dirty_df.columns:
                        lhs_counter[col] += 1
            if lhs_counter:
                # 取出现频次最高的 LHS 列
                best_col = lhs_counter.most_common(1)[0][0]
                # 检查 cardinality
                nunique = dirty_df[best_col].nunique()
                nrows = len(dirty_df)
                ratio = nunique / nrows if nrows > 0 else 1
                if 0.001 <= ratio <= 0.8:
                    return [best_col]

        # 3. 自动推断
        candidates = []
        nrows = len(dirty_df)
        for col in dirty_df.columns:
            if col == label_col:
                continue
            nunique = dirty_df[col].nunique()
            ratio = nunique / nrows if nrows > 0 else 1
            # 理想: 不太少(常量列) 也不太多(唯一标识符)
            if 0.01 <= ratio <= 0.5:
                # 优先选看起来是分类的列 (nunique < 100)
                score = 0
                if nunique <= 100:
                    score += 2
                if ratio <= 0.1:
                    score += 1
                # 优先非数值列
                try:
                    dirty_df[col].astype(float)
                except (ValueError, TypeError):
                    score += 1
                candidates.append((col, score, nunique))

        if candidates:
            candidates.sort(key=lambda x: (-x[1], x[2]))
            return [candidates[0][0]]

        return None  # 回退到等间隔分块

    @staticmethod
    def _run_row_split_clustered(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
        grouping_key: List[str],
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int], Dict[str, Any]]:
        """按聚类键分组后合并为块，运行 RAHA

        策略:
        - 按 grouping_key 分组
        - First-Fit Decreasing Bin Packing 合并小组
        - 每块 <= MAX_ROWS_PER_CHUNK
        - 最多 4 块
        """
        dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
        clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)
        max_size = RahaHelper.MAX_ROWS_PER_CHUNK

        # 按 grouping_key 分组（NaN 填充避免分组丢失）
        if len(grouping_key) == 1:
            group_col = grouping_key[0]
            grouped = dirty_df.fillna({group_col: '__NA__'}).groupby(group_col)
        else:
            fill_dict = {c: '__NA__' for c in grouping_key}
            grouped = dirty_df.fillna(fill_dict).groupby(grouping_key)

        group_info = [(key, list(indices)) for key, indices in grouped.groups.items()]
        group_info.sort(key=lambda x: len(x[1]), reverse=True)

        # First-Fit Decreasing Bin Packing
        chunks = []  # 每个 chunk = list of (key, [indices])
        chunk_sizes = []

        for key, indices in group_info:
            size = len(indices)
            if size > max_size:
                # 单组超限 -> 独立为一块
                chunks.append([(key, indices)])
                chunk_sizes.append(size)
            else:
                placed = False
                for i in range(len(chunks)):
                    if chunk_sizes[i] + size <= max_size:
                        chunks[i].append((key, indices))
                        chunk_sizes[i] += size
                        placed = True
                        break
                if not placed:
                    chunks.append([(key, indices)])
                    chunk_sizes.append(size)

        # 保证块数 <= 4
        while len(chunks) > 4:
            # 合并最小的两个块
            min_idx = sorted(range(len(chunks)), key=lambda i: chunk_sizes[i])[:2]
            i1, i2 = min_idx[0], min_idx[1]
            chunks[i1].extend(chunks[i2])
            chunk_sizes[i1] += chunk_sizes[i2]
            del chunks[i2]
            del chunk_sizes[i2]

        if verbose:
            print(f"  [聚类分块] 按 {grouping_key} 分组, 共 {len(group_info)} 组, 合并为 {len(chunks)} 块")
            for ci, (chunk, size) in enumerate(zip(chunks, chunk_sizes)):
                print(f"  [聚类分块 {ci+1}/{len(chunks)}] {size} 行 ({len(chunk)} 组)")

        merged_detected: Dict[Tuple[int, int], str] = {}
        merged_labeled: Set[int] = set()
        tmp_dir = tempfile.mkdtemp(prefix=f'raha_clustered_{dataset_name}_')

        try:
            for chunk_idx, chunk in enumerate(chunks):
                # 收集该块所有行索引
                all_indices = []
                for key, indices in chunk:
                    all_indices.extend(indices)
                all_indices.sort()

                if verbose:
                    print(f"  [聚类分块 {chunk_idx+1}/{len(chunks)}] {len(all_indices)} 行")

                # 生成子 CSV
                chunk_dirty_path = os.path.join(tmp_dir, f'dirty_cluster{chunk_idx}.csv')
                chunk_clean_path = os.path.join(tmp_dir, f'clean_cluster{chunk_idx}.csv')
                dirty_df.iloc[all_indices].to_csv(chunk_dirty_path, index=False)
                clean_df.iloc[all_indices].to_csv(chunk_clean_path, index=False)

                # 运行 RAHA
                chunk_detected, chunk_labeled = RahaHelper._run_single(
                    chunk_dirty_path, chunk_clean_path,
                    f"{dataset_name}_cluster{chunk_idx}",
                    labeling_budget, verbose,
                )

                # 映射行索引回原始行索引
                for (chunk_row, col), val in chunk_detected.items():
                    original_row = all_indices[chunk_row]
                    merged_detected[(original_row, col)] = val

                for chunk_row in chunk_labeled:
                    if chunk_row < len(all_indices):
                        merged_labeled.add(all_indices[chunk_row])

                if verbose:
                    print(f"  [聚类分块 {chunk_idx+1}] 检测: {len(chunk_detected)} 个, 标注行: {len(chunk_labeled)} 个")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            print(f"  [聚类分块] 合并结果: {len(merged_detected)} 个检测, {len(merged_labeled)} 个标注行")

        chunk_info = {
            'n_col_chunks': 1,
            'n_row_chunks': len(chunks),
            'labeling_budget': labeling_budget,
            'raha_total_cost': labeling_budget * len(chunks),
            'grouping_key': grouping_key,
        }
        return merged_detected, merged_labeled, chunk_info

    @staticmethod
    def _run_single(
        dirty_csv_path: str,
        clean_csv_path: str,
        dataset_name: str,
        labeling_budget: int,
        verbose: bool,
    ) -> Tuple[Dict[Tuple[int, int], str], Set[int]]:
        """单次运行 RAHA（不分块）

        会对 dirty 和 clean CSV 做逐单元格格式归一化，
        消除 "1.0" vs "1" 等数值格式差异导致的虚假错误检测。
        """
        # results_dir 也用 pid 后缀，与 raha_name 保持一致
        raha_name = f"{dataset_name}_p{os.getpid()}"
        results_dir = os.path.join(
            os.path.dirname(dirty_csv_path),
            f"raha-baran-results-{raha_name}"
        )
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)

        normalized_dirty_path = None
        normalized_clean_path = None
        try:
            from demandprep_policy.tools.csv_normalizer import normalize_cell_format
            dirty_df = pd.read_csv(dirty_csv_path, dtype=str, keep_default_na=False)
            dirty_norm = normalize_cell_format(dirty_df, verbose=False)
            fd, normalized_dirty_path = tempfile.mkstemp(
                suffix='.csv', prefix='raha_norm_dirty_'
            )
            os.close(fd)
            dirty_norm.to_csv(normalized_dirty_path, index=False)

            clean_df = pd.read_csv(clean_csv_path, dtype=str, keep_default_na=False)
            clean_norm = normalize_cell_format(clean_df, verbose=False)
            fd, normalized_clean_path = tempfile.mkstemp(
                suffix='.csv', prefix='raha_norm_clean_'
            )
            os.close(fd)
            clean_norm.to_csv(normalized_clean_path, index=False)

            actual_dirty_path = normalized_dirty_path
            actual_clean_path = normalized_clean_path
        except Exception:
            actual_dirty_path = dirty_csv_path
            actual_clean_path = clean_csv_path

        # RAHA 内部用 name 拼接 /tmp/{name}-{hash}.csv，同名 = 同路径 = 跨进程互删
        dataset_dictionary = {
            "name": raha_name,
            "path": actual_dirty_path,
            "clean_path": actual_clean_path,
        }

        try:
            app = _RahaDetection()
            app.LABELING_BUDGET = labeling_budget
            app.VERBOSE = verbose
            app.SAVE_RESULTS = False

            # 覆写 run() 以保留 dataset 对象 d（原版只返回 detected_cells，d 是局部变量会丢失）
            d = app.initialize_dataset(dataset_dictionary)
            app.run_strategies(d)
            app.generate_features(d)
            app.build_clusters(d)
            while len(d.labeled_tuples) < app.LABELING_BUDGET:
                app.sample_tuple(d)
                if d.has_ground_truth:
                    app.label_with_ground_truth(d)
            app.propagate_labels(d)
            app.predict_labels(d)
            detected_cells = d.detected_cells

            labeled_tuple_indices = set()
            if hasattr(d, 'labeled_tuples') and d.labeled_tuples:
                labeled_tuple_indices = set(d.labeled_tuples.keys())
            try:
                eval_result = d.get_data_cleaning_evaluation(detected_cells)
                p, r, f1 = eval_result[:3]
                print(f"  RAHA 检测结果: {len(detected_cells)} 个单元格")
                print(f"  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
            except Exception:
                print(f"  RAHA 检测结果: {len(detected_cells)} 个单元格")

            if labeled_tuple_indices:
                print(f"  RAHA 标注行数: {len(labeled_tuple_indices)} (labeling_budget={labeling_budget})")

            return (detected_cells if detected_cells else {}, labeled_tuple_indices)

        except Exception as e:
            print(f"  [错误] RAHA 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return {}, set()
        finally:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir, ignore_errors=True)
            for tmp_path in [normalized_dirty_path, normalized_clean_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    @staticmethod
    def map_raha_cells_to_feature_indices(
        detected_cells: Dict[Tuple[int, int], str],
        csv_columns: List[str],
        feature_columns: List[str],
        label_col: str,
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        将 RAHA 检测的 (row, csv_col) 映射到特征列索引 (row, feat_col)

        Returns:
            feature_errors: [(row, feat_col_idx), ...]
            label_error_rows: [row, ...]
        """
        feature_errors = []
        label_error_rows = []

        feat_name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}

        for (row, csv_col_idx), _ in detected_cells.items():
            if csv_col_idx < len(csv_columns):
                col_name = csv_columns[csv_col_idx]
                if col_name in feat_name_to_idx:
                    feature_errors.append((row, feat_name_to_idx[col_name]))
                elif col_name == label_col:
                    label_error_rows.append(row)

        return feature_errors, label_error_rows


class RuleBasedDetector:
    """
    规则检测器 —— 在原始 CSV(dirty_df) 上执行 REGEX/DOMAIN/CFD/DC 规则

    所有方法都是静态方法，接受 dirty_df 和规则对象，返回错误单元格集合。
    """

    # ------------------------------------------------------------------
    # (a) REGEX 检测 — 句法错误补充
    # ------------------------------------------------------------------
    @staticmethod
    def detect_regex(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        regex_rules: List[RegexRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Set[Tuple[int, int]]:
        """
        REGEX 规则检测句法错误

        Args:
            dirty_df: 原始脏数据 DataFrame (str dtype)
            feature_cols: 特征列名列表
            regex_rules: REGEX 规则列表
            excluded_cells: 已检出的单元格（不重复标记）
            feat_name_to_idx: 特征列名 → 索引的映射
            verbose: 是否输出日志

        Returns:
            Set[(row, col_idx)] — 归入句法错误
        """
        detected: Set[Tuple[int, int]] = set()

        for rule in regex_rules:
            # 确定目标列
            if rule.column == 'ALL_FEATURES':
                target_cols = feature_cols
            elif rule.column in feat_name_to_idx:
                target_cols = [rule.column]
            else:
                continue

            try:
                pattern = re.compile(rule.pattern)
            except re.error:
                continue

            for col_name in target_cols:
                col_idx = feat_name_to_idx.get(col_name)
                if col_idx is None:
                    continue

                if col_name not in dirty_df.columns:
                    continue

                for row_idx in range(len(dirty_df)):
                    if (row_idx, col_idx) in excluded_cells or (row_idx, col_idx) in detected:
                        continue
                    val = dirty_df.iloc[row_idx][col_name]
                    if pd.isna(val) or str(val).strip() == '':
                        continue
                    if pattern.search(str(val)):
                        detected.add((row_idx, col_idx))

        if verbose and detected:
            print(f"  REGEX 规则检测: {len(detected)} 个句法错误")

        return detected

    # ------------------------------------------------------------------
    # (b) DOMAIN 检测 — 句法错误（超出值域 = 句法异常）
    # ------------------------------------------------------------------
    @staticmethod
    def detect_domain(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        domain_rules: List[DomainRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        DOMAIN 规则检测句法错误（超出值域 = 句法异常）

        Returns:
            (syntactic_cells, label_cells)
            syntactic_cells: Set[(row, col_idx)] — 特征列域外错误
            label_cells: Set[(row, -1)] — 标签列域外错误
        """
        semantic: Set[Tuple[int, int]] = set()
        label: Set[Tuple[int, int]] = set()

        for rule in domain_rules:
            col_name = rule.column
            is_label = (col_name == label_col)

            if col_name not in dirty_df.columns:
                continue

            col_idx = feat_name_to_idx.get(col_name) if not is_label else -1

            for row_idx in range(len(dirty_df)):
                cell_key = (row_idx, col_idx)
                if cell_key in excluded_cells:
                    continue
                if not is_label and cell_key in semantic:
                    continue
                if is_label and cell_key in label:
                    continue

                val = dirty_df.iloc[row_idx][col_name]
                if pd.isna(val) or str(val).strip() == '':
                    continue

                val_str = str(val).strip()

                # 跳过常见占位符（与 preprocess_data 的替换列表一致）
                if val_str.lower() in (
                    'empty', 'nan', 'null', 'none', 'n/a', 'na', '?', '-',
                ):
                    continue

                is_violation = False

                if rule.dtype in ('INT', 'FLOAT'):
                    try:
                        num_val = float(val_str)
                        if num_val < rule.min_val or num_val > rule.max_val:
                            is_violation = True
                    except (ValueError, TypeError):
                        # 不可解析为数值 → 跳过（非范围违规，属于数据类型问题）
                        continue
                elif rule.dtype == 'ENUM':
                    if val_str not in rule.enum_vals:
                        # 浮点数据中整数 ENUM: str(0.0)='0.0' 不匹配 '0'
                        # 尝试数值比较: 若 val 可转为数值且四舍五入后匹配则不算违反
                        try:
                            num_val = float(val_str)
                            # 检查是否有某个 enum 值在数值上相等
                            matched = any(
                                abs(num_val - float(ev)) < 1e-9
                                for ev in rule.enum_vals
                                if ev.replace('.', '', 1).replace('-', '', 1).isdigit()
                            )
                            if not matched:
                                is_violation = True
                        except (ValueError, TypeError):
                            is_violation = True

                if is_violation:
                    if is_label:
                        label.add(cell_key)
                    else:
                        semantic.add(cell_key)

        if verbose:
            if semantic:
                print(f"  DOMAIN 规则检测: {len(semantic)} 个句法错误 (特征列, 超出值域)")
            if label:
                print(f"  DOMAIN 规则检测: {len(label)} 个标签错误")

        return semantic, label

    # ------------------------------------------------------------------
    # (c) CFD 检测 — 语义/标签错误
    # ------------------------------------------------------------------
    @staticmethod
    def detect_cfd(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        cfd_rules: List[CFDRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        CFD 规则检测

        Returns:
            (semantic_cells, label_cells)
            semantic_cells: 特征列 CFD 违规
            label_cells: 标签列 CFD 违规 (row, -1)
        """
        semantic: Set[Tuple[int, int]] = set()
        label: Set[Tuple[int, int]] = set()

        if not cfd_rules:
            return semantic, label

        n_rows = len(dirty_df)

        # 预计算 n_anomaly（breast_cancer 专用: 该行超出良性/恶性典型范围的列数）
        # 用于支持 CFD 条件中的 n_anomaly 变量
        # n_anomaly 只在条件引用了 n_anomaly 时才需要计算
        need_n_anomaly = any(
            any(col == 'n_anomaly' for col, _, _ in rule.conditions)
            for rule in cfd_rules
        )

        n_anomaly_map: Dict[int, int] = {}
        if need_n_anomaly:
            n_anomaly_map = RuleBasedDetector._compute_n_anomaly(
                dirty_df, feature_cols, label_col
            )

        for rule in cfd_rules:
            is_label_rule = (rule.target_col == label_col)
            target_col_idx = -1 if is_label_rule else feat_name_to_idx.get(rule.target_col)

            if target_col_idx is None:
                continue

            for row_idx in range(n_rows):
                cell_key = (row_idx, target_col_idx)
                if cell_key in excluded_cells:
                    continue
                if is_label_rule and cell_key in label:
                    continue
                if not is_label_rule and cell_key in semantic:
                    continue

                # 检查所有条件是否满足
                conditions_met = True
                for cond_col, cond_op, cond_val in rule.conditions:
                    # n_anomaly 特殊处理
                    if cond_col == 'n_anomaly':
                        actual_val = n_anomaly_map.get(row_idx, 0)
                        if not RuleBasedDetector._check_numeric_condition(
                            actual_val, cond_op, float(cond_val)
                        ):
                            conditions_met = False
                            break
                        continue

                    if cond_col not in dirty_df.columns:
                        conditions_met = False
                        break

                    row_val = dirty_df.iloc[row_idx][cond_col]
                    if pd.isna(row_val):
                        conditions_met = False
                        break

                    row_val_str = str(row_val).strip()

                    # 数值条件
                    if cond_op in ('<=', '>=', '<', '>', '!='):
                        try:
                            actual_num = float(row_val_str)
                            cond_num = float(cond_val)
                            if not RuleBasedDetector._check_numeric_condition(
                                actual_num, cond_op, cond_num
                            ):
                                conditions_met = False
                                break
                        except (ValueError, TypeError):
                            # 字符串比较
                            if cond_op == '!=' and row_val_str == cond_val:
                                conditions_met = False
                                break
                            elif cond_op == '=' and row_val_str != cond_val:
                                conditions_met = False
                                break
                    elif cond_op == '=':
                        if row_val_str != cond_val:
                            conditions_met = False
                            break

                if not conditions_met:
                    continue

                # 检查目标列偏差
                target_col_name = rule.target_col
                if target_col_name not in dirty_df.columns:
                    continue

                target_val = dirty_df.iloc[row_idx][target_col_name]
                if pd.isna(target_val):
                    continue

                try:
                    target_num = float(str(target_val).strip())
                except (ValueError, TypeError):
                    continue

                deviation = target_num - rule.baseline
                is_violation = False

                if rule.direction == 'EXCESS':
                    if deviation >= rule.threshold:
                        is_violation = True
                elif rule.direction == 'DEFICIT':
                    if -deviation >= rule.threshold:
                        is_violation = True

                if is_violation:
                    if is_label_rule:
                        label.add(cell_key)
                    else:
                        semantic.add(cell_key)

        # CFD 条件含标签等值约束时推断标签错误
        # 策略: 只有当该行 CFD 触发了多条规则(>=2 列偏差)但该行没有
        #        大量特征被标记为语义/句法错误时, 才推断为标签错误
        # 原理: 如果 class=2 但多个特征都像恶性, 且这些特征本身不是注入错误
        #        (即没被其他检测器标为语义/句法错误), 说明标签可能真的错了
        if label_col:
            # 统计每行被 CFD 触发的语义错误数
            cfd_row_counts: Dict[int, int] = {}
            for (r, c) in semantic:
                cfd_row_counts[r] = cfd_row_counts.get(r, 0) + 1

            for row_idx, cfd_count in cfd_row_counts.items():
                # 该行被CFD标记了至少2列特征偏差
                if cfd_count < 2:
                    continue
                label_key = (row_idx, -1)
                if label_key in label or label_key in excluded_cells:
                    continue
                # 检查该行 CFD 是否条件含标签列
                # (所有 breast_cancer CFD 都含 class=X)
                # 这里直接检查任何 CFD 规则是否含标签条件
                has_label_cond = any(
                    any(c == label_col and op == '='
                        for c, op, _ in rule.conditions)
                    for rule in cfd_rules
                )
                if has_label_cond:
                    label.add(label_key)

        if verbose:
            if semantic:
                print(f"  CFD 规则检测: {len(semantic)} 个语义错误 (特征列)")
            if label:
                print(f"  CFD 规则检测: {len(label)} 个标签错误")

        return semantic, label

    # ------------------------------------------------------------------
    # (d) DC 检测 — 语义错误
    # ------------------------------------------------------------------
    @staticmethod
    def detect_dc(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        dc_rules: List[DCRule],
        excluded_cells: Set[Tuple[int, int]],
        feat_name_to_idx: Dict[str, int],
        verbose: bool = True,
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, int]]:
        """
        DC (Denial Constraint) 规则检测

        接受结构化 DCRule 对象（已由 rule_parser 解析完成）。
        DC 使用 denial 语义：当所有子句条件都成立时表示约束被违反。

        对 abs_diff 类型且无 MARK 的规则，只标记偏离列中位数更大的列
        （用脏数据自身的统计量推断，不依赖干净数据）。

        Returns:
            (detected, rule_hit_counts)
            - detected: Set[(row, col_idx)] — 归入语义错误
            - rule_hit_counts: {rule_index: hit_count} — 每条规则触发行数
        """
        detected: Set[Tuple[int, int]] = set()
        rule_hit_counts: Dict[int, int] = {}

        # 预计算列中位数（用于 abs_diff 单列标记）
        col_medians: Dict[str, float] = {}
        for col_name in feature_cols:
            if col_name in dirty_df.columns:
                try:
                    vals = pd.to_numeric(dirty_df[col_name], errors='coerce')
                    col_medians[col_name] = float(vals.median())
                except Exception:
                    pass

        for rule_idx, dc_rule in enumerate(dc_rules):
            parsed_clauses = dc_rule.clauses
            if not parsed_clauses:
                continue

            # 判断是否为无 MARK 的 abs_diff 规则
            is_abs_diff_no_mark = (
                not dc_rule.mark_cols
                and len(parsed_clauses) == 1
                and parsed_clauses[0].get('type') == 'abs_diff'
            )

            # 如果有 MARK 子句，只标记 MARK 指定的列；否则标记所有涉及的列
            target_cols = dc_rule.mark_cols if dc_rule.mark_cols else dc_rule.involved_cols
            hit_count = 0

            # 对每行检查约束是否被违反
            for row_idx in range(len(dirty_df)):
                all_clauses_met = True
                for clause in parsed_clauses:
                    if not RuleBasedDetector._evaluate_dc_clause(
                        clause, dirty_df, row_idx
                    ):
                        all_clauses_met = False
                        break

                if all_clauses_met:
                    hit_count += 1

                    if is_abs_diff_no_mark:
                        # abs_diff 无 MARK: 只标记偏离中位数更大的列
                        clause = parsed_clauses[0]
                        col1_name, col2_name = clause['col1'], clause['col2']
                        try:
                            v1 = float(str(dirty_df.iloc[row_idx][col1_name]).strip())
                            v2 = float(str(dirty_df.iloc[row_idx][col2_name]).strip())
                        except (ValueError, TypeError):
                            continue

                        med1 = col_medians.get(col1_name, v1)
                        med2 = col_medians.get(col2_name, v2)
                        dev1 = abs(v1 - med1)
                        dev2 = abs(v2 - med2)

                        # 标记偏离更大的列
                        flag_col = col1_name if dev1 >= dev2 else col2_name
                        col_idx = feat_name_to_idx.get(flag_col)
                        if col_idx is not None:
                            cell_key = (row_idx, col_idx)
                            if cell_key not in excluded_cells:
                                detected.add(cell_key)
                    else:
                        # 有 MARK 或其他类型: 标记所有目标列
                        for col_name in target_cols:
                            if col_name == label_col:
                                col_idx = -1
                            else:
                                col_idx = feat_name_to_idx.get(col_name)
                            if col_idx is not None:
                                cell_key = (row_idx, col_idx)
                                if cell_key not in excluded_cells:
                                    detected.add(cell_key)

            rule_hit_counts[rule_idx] = hit_count

        if verbose and detected:
            print(f"  DC 规则检测: {len(detected)} 个语义错误")

        return detected, rule_hit_counts

    # ------------------------------------------------------------------
    # DC 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dc_clause(clause_str: str) -> Optional[Dict[str, Any]]:
        """解析单个 DC 子句（委托给 rule_parser.parse_dc_clause 共享实现）"""
        return parse_dc_clause(clause_str)

    @staticmethod
    def _evaluate_dc_clause(
        clause: Dict[str, Any],
        dirty_df: pd.DataFrame,
        row_idx: int,
    ) -> bool:
        """对指定行评估一个 DC 子句，返回 True 表示子句条件成立"""
        clause_type = clause['type']

        if clause_type == 'abs_diff':
            col1, col2 = clause['col1'], clause['col2']
            if col1 not in dirty_df.columns or col2 not in dirty_df.columns:
                return False
            try:
                v1 = float(str(dirty_df.iloc[row_idx][col1]).strip())
                v2 = float(str(dirty_df.iloc[row_idx][col2]).strip())
            except (ValueError, TypeError):
                return False
            actual = abs(v1 - v2)
            return RuleBasedDetector._dc_compare(actual, clause['op'], clause['value'])

        elif clause_type == 'simple':
            col = clause['col']
            if col not in dirty_df.columns:
                return False
            val = dirty_df.iloc[row_idx][col]
            if pd.isna(val):
                return False
            try:
                actual = float(str(val).strip())
            except (ValueError, TypeError):
                return False
            return RuleBasedDetector._dc_compare(actual, clause['op'], clause['value'])

        elif clause_type == 'simple_str':
            col = clause['col']
            if col not in dirty_df.columns:
                return False
            val = dirty_df.iloc[row_idx][col]
            if pd.isna(val):
                return False
            actual = str(val).strip()
            expected = str(clause['value']).strip()
            op = clause['op']
            if op in ('EQ', 'IQ'):
                return actual == expected
            elif op == 'NEQ':
                return actual != expected
            return False

        return False

    @staticmethod
    def _dc_compare(actual: float, op: str, expected: float) -> bool:
        """DC 数值比较"""
        if op == 'GT':
            return actual > expected
        elif op == 'GTE':
            return actual >= expected
        elif op == 'LT':
            return actual < expected
        elif op == 'LTE':
            return actual <= expected
        elif op in ('EQ', 'IQ'):
            return abs(actual - expected) < 1e-9
        elif op == 'NEQ':
            return abs(actual - expected) > 1e-9
        return False

    # ------------------------------------------------------------------
    # CFD 辅助: 计算 n_anomaly
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_n_anomaly(
        dirty_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
    ) -> Dict[int, int]:
        """计算每行超出典型范围的列数 (n_anomaly)

        用于 CFD 规则中 n_anomaly 条件。
        典型范围基于 class 值分组的 P5/P95。
        目前为 breast_cancer 硬编码，后续可从 clean 数据自动计算。
        """
        # breast_cancer 的典型范围（从 clean 数据的 P5/P95 计算）
        benign_upper = {
            'Clump Thickness': 5, 'Uniformity of Cell Size': 3,
            'Uniformity of Cell Shape': 3, 'Marginal Adhesion': 3,
            'Single Epithelial Cell Size': 3, 'Bare Nuclei': 3,
            'Bland Chromatin': 3, 'Normal Nucleoli': 3, 'Mitoses': 1,
        }
        malignant_lower = {
            'Clump Thickness': 4, 'Uniformity of Cell Size': 3,
            'Uniformity of Cell Shape': 3, 'Bland Chromatin': 3,
        }

        n_anomaly_map: Dict[int, int] = {}

        if label_col and label_col in dirty_df.columns:
            for row_idx in range(len(dirty_df)):
                class_val = str(dirty_df.iloc[row_idx].get(label_col, '')).strip()
                count = 0

                if class_val == '2':
                    for col, upper in benign_upper.items():
                        if col in dirty_df.columns and col in feature_cols:
                            try:
                                v = float(str(dirty_df.iloc[row_idx][col]).strip())
                                if v > upper:
                                    count += 1
                            except (ValueError, TypeError):
                                pass
                elif class_val == '4':
                    for col, lower in malignant_lower.items():
                        if col in dirty_df.columns and col in feature_cols:
                            try:
                                v = float(str(dirty_df.iloc[row_idx][col]).strip())
                                if v < lower:
                                    count += 1
                            except (ValueError, TypeError):
                                pass

                n_anomaly_map[row_idx] = count

        return n_anomaly_map

    @staticmethod
    def _check_numeric_condition(actual: float, op: str, expected: float) -> bool:
        """通用数值条件检查"""
        if op == '=':
            return abs(actual - expected) < 1e-9
        elif op == '!=':
            return abs(actual - expected) > 1e-9
        elif op == '<=':
            return actual <= expected
        elif op == '>=':
            return actual >= expected
        elif op == '<':
            return actual < expected
        elif op == '>':
            return actual > expected
        return False


class SemanticErrorDetector:
    """
    语义错误检测器

    核心思想：RAHA 擅长检测句法错误（格式异常、模式违规等），
    但对语义错误（值的格式/分布正常、但逻辑关系错误）无能为力。

    因此：RAHA 已检测的 = 句法错误；RAHA 未覆盖的真实错误 = 语义错误。

    检测策略：
    1. FD/CFD 规则多数投票：LHS → RHS 约束违规
    """

    @staticmethod
    def detect_fd_violations(
        X_dirty: np.ndarray,
        fd_col_pairs: List[Tuple[List[int], int]],
        excluded_cells: Set[Tuple[int, int]],
        verbose: bool = True,
    ) -> List[Tuple]:
        """
        基于 FD 规则的多数投票检测

        Returns:
            list of (idx, col, estimated_val, current_val)
        """
        violations = []
        seen_cells: Set[Tuple[int, int]] = set()

        for lhs_indices, rhs_idx in fd_col_pairs:
            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(X_dirty)):
                if (i, rhs_idx) in excluded_cells or (i, rhs_idx) in seen_cells:
                    continue
                lhs_vals = X_dirty[i, lhs_indices]
                if np.isnan(lhs_vals).any() or np.isnan(X_dirty[i, rhs_idx]):
                    continue
                key = tuple(lhs_vals.tolist())
                groups[key].append(i)

            for key, rows in groups.items():
                if len(rows) < 2:
                    continue
                rhs_values = [X_dirty[r, rhs_idx] for r in rows]
                val_counts: Dict[float, int] = defaultdict(int)
                for v in rhs_values:
                    val_counts[v] += 1
                majority_val = max(val_counts, key=val_counts.get)
                majority_count = val_counts[majority_val]

                if majority_count < len(rows):
                    for r in rows:
                        current_val = X_dirty[r, rhs_idx]
                        if (abs(current_val - majority_val) > 1e-6
                                and (r, rhs_idx) not in excluded_cells
                                and (r, rhs_idx) not in seen_cells):
                            violations.append((r, rhs_idx, majority_val, current_val))
                            seen_cells.add((r, rhs_idx))

        if verbose and violations:
            print(f"  FD 规则检测: {len(violations)} 个语义错误")

        return violations

class AutoDetector:
    """
    自动错误检测器

    检测策略分工清晰：
    1. 缺失值: np.isnan 直接检测
    2. 句法错误: RAHA + REGEX + DOMAIN 规则（超出值域 = 句法异常）
    3. 语义错误: FD + CFD(特征) + DC 规则（值的逻辑关系错误）
    4. 标签噪声: CFD 标签规则（完全替代 CL）
    """

    def __init__(self,
                 dirty_csv_path: Optional[str] = None,
                 clean_csv_path: Optional[str] = None,
                 dataset_name: str = "data",
                 label_col: Optional[str] = None,
                 csv_columns: Optional[List[str]] = None,
                 column_names: Optional[List[str]] = None,
                 fd_rules: Optional[List[Tuple[str, str]]] = None,
                 labeling_budget: int = 20,
                 enable_confident_learning: bool = False,
                 rules_path: Optional[str] = None,
                 disable_raha: bool = False):
        """
        Args:
            dirty_csv_path: 原始脏数据 CSV 路径
            clean_csv_path: 原始干净数据 CSV 路径
            dataset_name: 数据集名称
            label_col: 标签列名
            csv_columns: 原始 CSV 的所有列名（含 index/label）
            column_names: 特征列名列表（不含 index/label）
            fd_rules: FD 规则列表 [("lhs_col", "rhs_col"), ...]
            labeling_budget: RAHA 标注预算
            enable_confident_learning: 已弃用，保留参数兼容性（永远不启用 CL）
            rules_path: 规则文件路径 (data/{dataset}/rules.txt)
        """
        self.dirty_csv_path = dirty_csv_path
        self.clean_csv_path = clean_csv_path
        self.dataset_name = dataset_name
        self.label_col = label_col
        self.csv_columns = csv_columns
        self.column_names = column_names
        self.fd_rules = fd_rules or []
        self.labeling_budget = labeling_budget
        # CL 已被完全移除，此参数仅保留接口兼容性
        self.enable_confident_learning = False
        self.disable_raha = disable_raha
        self.col_stats: Dict[int, Dict[str, float]] = {}
        self.is_fitted = False

        # 规则解析
        self.rules_path = rules_path
        self.parsed_rules: ParsedRules = ParsedRules()
        if rules_path:
            self.parsed_rules = parse_rules_file(rules_path)

        # COL_STATS 已废弃: 不再从 rules.txt [STATISTICAL] 段读取预计算统计量。
        # 原因: generate_col_stats.py 基于 dirty 数据 fit scaler，
        #        而主 pipeline 基于 clean 数据 fit，两者编码空间不一致。
        # 现在 col_stats 完全依赖运行时计算:
        #   1. fit(X_clean_subset) 时计算 (最优先)
        #   2. detect() 时用 X_dirty 回退计算

        # FD 列索引映射（合并 __init__ 传入的 + 规则文件中的）
        self.fd_col_pairs: List[Tuple[List[int], int]] = []
        self._build_fd_index()

        # RAHA 检测缓存（同一数据只运行一次）
        self._raha_cache: Optional[Dict[Tuple[int, int], str]] = None

        # RAHA 标注的行索引（labeling_budget 条，用于预修复）
        self.labeled_tuples: Set[int] = set()

        # RAHA 检测成本信息（分块信息 + 总标注成本）
        self.raha_cost_info: Dict[str, Any] = {}

        # dirty_df 缓存（规则检测需要在原始 CSV 上运行）
        self._dirty_df_cache: Optional[pd.DataFrame] = None

        # 特征列名 → 索引映射
        self._feat_name_to_idx: Dict[str, int] = {}
        if self.column_names:
            self._feat_name_to_idx = {
                name: idx for idx, name in enumerate(self.column_names)
            }

    def _build_fd_index(self):
        """构建 FD 规则的列索引映射

        合并两个来源: __init__ 传入的 fd_rules + 规则文件中的 FD rules
        """
        if not self.column_names:
            return

        # 合并 FD 来源
        all_fd_pairs = list(self.fd_rules)  # 原始传入的
        if self.parsed_rules and self.parsed_rules.fd_rules:
            for lhs, rhs in self.parsed_rules.fd_rules:
                pair = (lhs, rhs)
                if pair not in all_fd_pairs:
                    all_fd_pairs.append(pair)

        seen = set()
        for rule in all_fd_pairs:
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
                    key = (tuple(lhs_indices), rhs_idx)
                    if key not in seen:
                        seen.add(key)
                        self.fd_col_pairs.append((lhs_indices, rhs_idx))

    def _compute_col_stats(self, X: np.ndarray) -> None:
        """计算列统计量"""
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

    def fit(self, X_clean_subset: np.ndarray = None, verbose: bool = True) -> 'AutoDetector':
        """
        初始化检测器（计算统计量 + 预计算干净数据基线）

        Args:
            X_clean_subset: 干净数据子集，用于计算列统计量
            verbose: 是否打印日志
        """
        if verbose:
            print(f"\n[AutoDetector] 初始化")
            print(f"  RAHA 来源: {_RAHA_SOURCE}")
            print(f"  RAHA 可用: {RAHA_AVAILABLE}")
            print(f"  CSV 路径: {self.dirty_csv_path}")
            print(f"  FD 规则: {len(self.fd_col_pairs)} 条")
            print(f"  规则文件: {self.rules_path or '无'}")
            if self.parsed_rules.has_any_rules:
                print(f"  规则摘要: {self.parsed_rules.summary()}")

        if X_clean_subset is not None and not self.col_stats:
            self._compute_col_stats(X_clean_subset)

        self.is_fitted = True
        return self

    def _run_raha_once(self, verbose: bool = True) -> Dict[Tuple[int, int], str]:
        """运行 RAHA 并缓存结果（同时缓存 labeled_tuples）"""
        if self._raha_cache is not None:
            if verbose:
                print(f"  [RAHA] 使用缓存结果: {len(self._raha_cache)} 个单元格")
            return self._raha_cache

        if not self.dirty_csv_path or not self.clean_csv_path:
            if verbose:
                print("  [RAHA] 无 CSV 路径，跳过")
            self._raha_cache = {}
            return self._raha_cache

        if not RAHA_AVAILABLE:
            if verbose:
                print("  [RAHA] 库不可用，跳过")
            self._raha_cache = {}
            return self._raha_cache

        if verbose:
            print(f"\n[检测] 在原始 CSV 上运行 RAHA (labeling_budget={self.labeling_budget})...")

        detected_cells, labeled_indices, chunk_info = RahaHelper.run_on_csv(
            dirty_csv_path=self.dirty_csv_path,
            clean_csv_path=self.clean_csv_path,
            dataset_name=self.dataset_name,
            labeling_budget=self.labeling_budget,
            verbose=verbose,
            fd_rules=self.fd_rules,
            primary_key=self.parsed_rules.primary_key if self.parsed_rules else None,
            label_col=self.label_col,
        )
        self._raha_cache = detected_cells
        self.labeled_tuples = labeled_indices
        self.raha_cost_info = {
            **chunk_info,
            'total_labeled_rows': len(labeled_indices),
        }
        return self._raha_cache

    def _load_dirty_df(self) -> Optional[pd.DataFrame]:
        """加载原始 CSV 为 DataFrame（缓存，规则检测用）"""
        if self._dirty_df_cache is not None:
            return self._dirty_df_cache

        if not self.dirty_csv_path or not os.path.exists(self.dirty_csv_path):
            return None

        self._dirty_df_cache = pd.read_csv(
            self.dirty_csv_path, dtype=str, keep_default_na=False
        )
        return self._dirty_df_cache

    def detect(self,
               X_dirty: np.ndarray,
               y_dirty: Optional[np.ndarray] = None,
               task_type: str = 'classification',
               semantic_positions: Optional[List[Tuple[int, int]]] = None,
               verbose: bool = True) -> Dict[str, List]:
        """
        检测错误（核心方法）

        流程:
          1. 缺失值检测（NaN）
          2. 句法错误检测
             2a. RAHA（原始 CSV 上运行）
             2b. REGEX 规则（补充 RAHA 漏检的句法模式）
             2c. DOMAIN 规则（超出值域 = 句法异常）
          3. 语义错误检测
             3a. FD 规则（多数投票）
             3b. CFD 特征规则（条件依赖违规）
             3c. DC 规则（跨列完整性约束）
          4. 标签噪声检测
             4a. CFD 标签规则（完全替代 Confident Learning）

        Returns:
            detected: {
                'missing': [(idx, col, estimated_val), ...],
                'semantic': [(idx, col, estimated_val, current_val), ...],
                'syntactic': [(idx, col, estimated_val, noise), ...],
                'label_noise': [(idx, -1, estimated_val, dirty_val), ...]
            }
        """
        detected: Dict[str, List] = {
            'missing': [],
            'semantic': [],
            'syntactic': [],
            'label_noise': [],
        }
        n = len(X_dirty)

        # 确保已计算统计量（优先级：fit(X_clean_subset) > X_dirty 回退）
        if not self.col_stats:
            self._compute_col_stats(X_dirty)

        # 加载原始 dirty_df（规则检测需要）
        dirty_df = self._load_dirty_df()
        has_rules = self.parsed_rules.has_any_rules
        feat_name_to_idx = self._feat_name_to_idx

        # ================================================================
        # 1. 缺失值检测
        # ================================================================
        missing_cells: Set[Tuple[int, int]] = set()
        for i in range(n):
            for col in range(X_dirty.shape[1]):
                val = X_dirty[i, col]
                if isinstance(val, float) and np.isnan(val):
                    estimated_val = self.col_stats.get(col, {}).get('mean', 0)
                    detected['missing'].append((i, col, estimated_val))
                    missing_cells.add((i, col))

        if verbose:
            print(f"  缺失值: {len(detected['missing'])} 个")

        # 标签 NaN 检测
        label_missing_count = 0
        if y_dirty is not None:
            for i in range(len(y_dirty)):
                if isinstance(y_dirty[i], float) and np.isnan(y_dirty[i]):
                    detected['label_noise'].append((i, -1, float('nan'), float('nan')))
                    label_missing_count += 1
            if verbose and label_missing_count > 0:
                print(f"  标签缺失(NaN): {label_missing_count} 个")

        # ================================================================
        # 2. 句法错误检测
        # ================================================================
        syntactic_cells: Set[Tuple[int, int]] = set()

        # 2a. RAHA 检测（可通过 disable_raha=True 禁用）
        raha_detected = {} if self.disable_raha else self._run_raha_once(verbose=verbose)

        if raha_detected and self.csv_columns and self.column_names:
            feature_errors, raha_label_error_rows = RahaHelper.map_raha_cells_to_feature_indices(
                detected_cells=raha_detected,
                csv_columns=self.csv_columns,
                feature_columns=self.column_names,
                label_col=self.label_col or "",
            )

            for row, feat_col in feature_errors:
                if ((row, feat_col) not in missing_cells
                        and (row, feat_col) not in syntactic_cells
                        and row < n and feat_col < X_dirty.shape[1]):
                    estimated_val = self.col_stats.get(feat_col, {}).get('mean', 0)
                    current_val = X_dirty[row, feat_col] if not np.isnan(X_dirty[row, feat_col]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, feat_col, estimated_val, noise))
                    syntactic_cells.add((row, feat_col))

            # RAHA 检测到的标签列错误也记录
            if raha_label_error_rows and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for row_idx in raha_label_error_rows:
                    if row_idx < len(y_dirty) and row_idx not in existing_label_rows:
                        dirty_label = y_dirty[row_idx]
                        detected['label_noise'].append((row_idx, -1, float('nan'), dirty_label))
                if verbose:
                    print(f"  RAHA 标签错误: {len(raha_label_error_rows)} 行")

        if verbose:
            print(f"  句法错误(RAHA): {len(detected['syntactic'])} 个")

        # 2b. REGEX 规则检测（补充 RAHA 漏检）
        if dirty_df is not None and self.parsed_rules.regex_rules and self.column_names:
            excluded_for_regex = missing_cells | syntactic_cells
            regex_cells = RuleBasedDetector.detect_regex(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                regex_rules=self.parsed_rules.regex_rules,
                excluded_cells=excluded_for_regex,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in regex_cells:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, col_idx, estimated_val, noise))
                    syntactic_cells.add((row, col_idx))

        if verbose and self.parsed_rules.regex_rules:
            print(f"  句法错误(RAHA+REGEX): {len(detected['syntactic'])} 个")

        # 2c. DOMAIN 规则检测（超出值域 = 句法异常，在原始 CSV 上运行）
        if dirty_df is not None and self.parsed_rules.domain_rules and self.column_names:
            domain_syntactic, domain_label = RuleBasedDetector.detect_domain(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                domain_rules=self.parsed_rules.domain_rules,
                excluded_cells=missing_cells | syntactic_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in domain_syntactic:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else 0
                    noise = current_val - estimated_val
                    detected['syntactic'].append((row, col_idx, estimated_val, noise))
                    syntactic_cells.add((row, col_idx))
            # DOMAIN 标签错误
            if domain_label and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for (row, _) in domain_label:
                    if row < len(y_dirty) and row not in existing_label_rows:
                        dirty_label = y_dirty[row]
                        detected['label_noise'].append((row, -1, float('nan'), dirty_label))

        if verbose and self.parsed_rules.domain_rules:
            print(f"  句法错误(RAHA+REGEX+DOMAIN): {len(detected['syntactic'])} 个")

        # ================================================================
        # 3. 语义错误检测（规则驱动）
        # ================================================================
        excluded_cells = missing_cells | syntactic_cells
        semantic_errors = []

        # 3a. FD 规则检测（多数投票，在 numpy 数组上运行）
        if self.fd_col_pairs:
            fd_violations = SemanticErrorDetector.detect_fd_violations(
                X_dirty, self.fd_col_pairs, excluded_cells, verbose=verbose
            )
            semantic_errors.extend(fd_violations)
            for item in fd_violations:
                excluded_cells.add((item[0], item[1]))

        # 3b. CFD 特征规则检测（在原始 CSV 上运行）  [原 3c]
        if dirty_df is not None and self.parsed_rules.cfd_rules and self.column_names:
            cfd_semantic, cfd_label = RuleBasedDetector.detect_cfd(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                cfd_rules=self.parsed_rules.cfd_rules,
                excluded_cells=excluded_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in cfd_semantic:
                if row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else estimated_val
                    semantic_errors.append((row, col_idx, estimated_val, current_val))
                    excluded_cells.add((row, col_idx))
            # CFD 标签错误（完全替代 CL）
            if cfd_label and y_dirty is not None:
                existing_label_rows = {item[0] for item in detected['label_noise']}
                for (row, _) in cfd_label:
                    if row < len(y_dirty) and row not in existing_label_rows:
                        dirty_label = y_dirty[row]
                        detected['label_noise'].append((row, -1, float('nan'), dirty_label))

        # 3c. DC 规则检测（在原始 CSV 上运行）  [原 3d]
        if dirty_df is not None and self.parsed_rules.dc_rules and self.column_names:
            dc_cells, _dc_hits = RuleBasedDetector.detect_dc(
                dirty_df=dirty_df,
                feature_cols=self.column_names,
                label_col=self.label_col,
                dc_rules=self.parsed_rules.dc_rules,
                excluded_cells=excluded_cells,
                feat_name_to_idx=feat_name_to_idx,
                verbose=verbose,
            )
            for (row, col_idx) in dc_cells:
                if col_idx == -1:
                    # DC 标记的标签错误
                    if y_dirty is not None:
                        existing_label_rows = {item[0] for item in detected['label_noise']}
                        if row < len(y_dirty) and row not in existing_label_rows:
                            dirty_label = y_dirty[row]
                            detected['label_noise'].append((row, -1, float('nan'), dirty_label))
                elif row < n and col_idx < X_dirty.shape[1]:
                    estimated_val = self.col_stats.get(col_idx, {}).get('mean', 0)
                    current_val = X_dirty[row, col_idx] if not np.isnan(X_dirty[row, col_idx]) else estimated_val
                    semantic_errors.append((row, col_idx, estimated_val, current_val))
                    excluded_cells.add((row, col_idx))

        # 3e. 外部提供的语义错误位置（保留向后兼容）
        if semantic_positions:
            for pos in semantic_positions:
                if len(pos) >= 2:
                    idx, col = pos[0], pos[1]
                    if (idx, col) not in excluded_cells:
                        estimated_val = self.col_stats.get(col, {}).get('mean', 0)
                        current_val = X_dirty[idx, col] if not np.isnan(X_dirty[idx, col]) else estimated_val
                        semantic_errors.append((idx, col, estimated_val, current_val))
                        excluded_cells.add((idx, col))

        detected['semantic'] = semantic_errors

        if verbose:
            sources = []
            if self.fd_col_pairs:
                sources.append("FD")
            if self.parsed_rules.cfd_rules:
                sources.append("CFD")
            if self.parsed_rules.dc_rules:
                sources.append("DC")
            if semantic_positions:
                sources.append("外部")
            mode = "+".join(sources) if sources else "无来源"
            print(f"  语义错误: {len(detected['semantic'])} 个 ({mode})")

        # ================================================================
        # 4. 标签噪声检测 — 汇总
        # ================================================================
        # CFD/DC 标签错误已在阶段 3b/3c 中检测并添加到 label_noise
        if verbose:
            label_from_nan = label_missing_count
            label_total = len(detected['label_noise'])
            label_from_rules = label_total - label_from_nan
            print(f"  标签噪声: {label_total} 个 (NaN={label_from_nan}, 规则={label_from_rules})")

        # ================================================================
        # 4b. 特征受损行标签排除
        # ================================================================
        # 原则: CFD/DC 标签规则依赖特征条件。如果一行的特征被检测到有
        # 语义/句法/缺失错误, 则基于这些错误特征的标签判断不可信。
        # 排除这些行可大幅降低标签检测的 FP。
        feature_damaged_removed = 0
        if detected['label_noise']:
            damaged_rows: Set[int] = set()
            for item in detected['semantic']:
                damaged_rows.add(item[0])
            for item in detected['syntactic']:
                damaged_rows.add(item[0])
            for item in detected['missing']:
                damaged_rows.add(item[0])

            if damaged_rows:
                filtered_label = []
                for item in detected['label_noise']:
                    row_idx = item[0]
                    if row_idx in damaged_rows:
                        feature_damaged_removed += 1
                        continue
                    filtered_label.append(item)
                detected['label_noise'] = filtered_label
                if verbose and feature_damaged_removed > 0:
                    print(f"  特征受损行标签排除: {feature_damaged_removed} 个")

        # ================================================================
        # 总结
        # ================================================================
        if verbose:
            total = (len(detected['missing']) + len(detected['syntactic'])
                     + len(detected['semantic']) + len(detected['label_noise']))
            print(f"\n  检测总结: {total} 个错误 "
                  f"(缺失={len(detected['missing'])}, "
                  f"句法={len(detected['syntactic'])}, "
                  f"语义={len(detected['semantic'])}, "
                  f"标签噪声={len(detected['label_noise'])})")

        return detected

    def build_error_list(self,
                         detected: Dict[str, List],
                         X_clean: Optional[np.ndarray] = None) -> List[Dict]:
        """
        将检测到的错误转换为清洗环境需要的格式

        Returns:
            error_list: [{'idx', 'col', 'type', 'repair_value'}, ...]
        """
        error_list = []

        # Missing errors (type=0)
        for item in detected['missing']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 0, 'repair_value': repair_value
            })

        # Semantic errors (type=1)
        for item in detected['semantic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 1, 'repair_value': repair_value
            })

        # Syntactic errors (type=2)
        for item in detected['syntactic']:
            idx, col, estimated_val = item[0], item[1], item[2]
            repair_value = X_clean[idx, col] if X_clean is not None else estimated_val
            error_list.append({
                'idx': idx, 'col': col, 'type': 2, 'repair_value': repair_value
            })

        # Label noise errors (type=3, col=-1)
        for item in detected.get('label_noise', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx = item[0]
                clean_val = item[2] if len(item) > 2 else float('nan')
                repair_value = clean_val if not np.isnan(clean_val) else float('nan')
                error_list.append({
                    'idx': idx, 'col': -1, 'type': 3, 'repair_value': repair_value
                })

        return error_list

    def save(self, path: str) -> None:
        """保存检测器参数"""
        data = {
            'dirty_csv_path': self.dirty_csv_path,
            'clean_csv_path': self.clean_csv_path,
            'dataset_name': self.dataset_name,
            'label_col': self.label_col,
            'csv_columns': self.csv_columns,
            'column_names': self.column_names,
            'fd_rules': self.fd_rules,
            'labeling_budget': self.labeling_budget,
            'rules_path': self.rules_path,
            'col_stats': self.col_stats,
            'is_fitted': self.is_fitted,
            'labeled_tuples': list(self.labeled_tuples),
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  检测器已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'AutoDetector':
        """加载检测器参数"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        detector = cls(
            dirty_csv_path=data.get('dirty_csv_path'),
            clean_csv_path=data.get('clean_csv_path'),
            dataset_name=data.get('dataset_name', 'data'),
            label_col=data.get('label_col'),
            csv_columns=data.get('csv_columns'),
            column_names=data.get('column_names'),
            fd_rules=data.get('fd_rules'),
            labeling_budget=data.get('labeling_budget', 20),
            rules_path=data.get('rules_path'),
        )
        detector.col_stats = data.get('col_stats', {})
        detector.is_fitted = data.get('is_fitted', True)
        detector.labeled_tuples = set(data.get('labeled_tuples', []))
        print(f"  检测器已加载: fd_rules={len(detector.fd_col_pairs)}, "
              f"rules={detector.parsed_rules.summary()}, "
              f"labeled_tuples={len(detector.labeled_tuples)}")
        return detector


# 向后兼容别名
RahaBasedDetector = AutoDetector


def is_raha_available() -> bool:
    """检查 RAHA 是否可用"""
    return RAHA_AVAILABLE
