"""
CSV 格式预清洗工具
==================

数据传入 DemandClean 后的第一步预处理：统一数值格式。

问题背景:
    原始 dirty CSV 中数值列可能存在格式不统一的问题，例如:
    - 整数值以浮点形式存储: "1.0", "4.0", "10.0"
    - 同一列中混杂 "1" 和 "1.0" 两种写法

    RAHA 在原始 CSV 上做字符串级比较，这类格式差异会被误判为错误。

解决方案:
    自包含的格式预清洗（不参考 clean 数据）:
    - 对每一列，检测是否为整数值列（所有有效值都是整数）
    - 如果是，将 "1.0" → "1" 统一为整数格式
    - 这是标准的数据预处理步骤，不涉及任何"偷看答案"

    后续评测（getScore）时，格式统一视为清洗效果的一部分。

用法:
    from demandclean.tools.csv_normalizer import normalize_dirty_format

    # 只传入 dirty，返回格式统一后的 DataFrame
    dirty_normalized = normalize_dirty_format(dirty_df)

    # 或者直接处理文件
    norm_path = normalize_dirty_to_file(dirty_path)
"""

import os
import tempfile
from typing import Optional, Set

import pandas as pd

# 常见缺失值/占位符标记，遇到时跳过（不影响整列判断）
_NA_LIKE_VALUES: Set[str] = {
    '', 'nan', 'NaN', 'NAN', 'null', 'NULL', 'none', 'None', 'NONE',
    'empty', 'Empty', 'EMPTY', 'na', 'NA', 'N/A', 'n/a', '?', '-', '.',
}


def normalize_dirty_format(
    dirty_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """对 dirty DataFrame 做自包含的数值格式预清洗。

    规则（逐列判断）:
      1. 跳过 NaN / 空值 / 常见缺失标记 (如 "?", "empty", "nan")
      2. 对每列的所有有效数值，检查是否全为整数
      3. 如果是，将 "1.0" → "1" 统一为整数格式
      4. 非数值列 / 含真实小数的列不做处理

    Args:
        dirty_df: 脏数据 DataFrame
        verbose: 是否打印统计信息

    Returns:
        格式统一后的 dirty DataFrame（深拷贝，不修改原始数据）
    """
    result = dirty_df.copy()
    total_normalized = 0

    for col in result.columns:
        col_loc = result.columns.get_loc(col)
        col_normalized = 0

        # 收集该列所有非空值，判断是否为"全整数列"
        non_null_mask = result[col].notna()
        if non_null_mask.sum() == 0:
            continue

        values = result.loc[non_null_mask, col]
        all_integer = True
        has_numeric = False

        for val in values:
            s = str(val).strip()
            # 跳过缺失标记
            if s in _NA_LIKE_VALUES:
                continue
            try:
                f = float(s)
                if f != f:  # NaN
                    continue
                has_numeric = True
                if f != int(f):
                    # 存在真实小数（如 3.14），整列不处理
                    all_integer = False
                    break
            except (ValueError, TypeError):
                # 非数值（如字符串 "Beer"），跳过该值继续
                # 但不阻断整列判断——允许混杂少量非数值脏数据
                continue

        if not all_integer or not has_numeric:
            continue

        # 该列数值部分全是整数，统一格式: "1.0" → "1"
        for i in result.index[non_null_mask]:
            s = str(result.at[i, col]).strip()
            if s in _NA_LIKE_VALUES:
                continue
            try:
                f = float(s)
                if f != f:
                    continue
                if f != int(f):
                    continue
                int_s = str(int(f))
                if s != int_s:
                    result.iat[result.index.get_loc(i), col_loc] = int_s
                    col_normalized += 1
            except (ValueError, TypeError):
                pass

        total_normalized += col_normalized

    if verbose and total_normalized > 0:
        print(f"  [格式预清洗] 统一 {total_normalized} 个整数浮点格式 (如 \"1.0\" → \"1\")")

    return result


def normalize_cell_format(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """对 DataFrame 做逐单元格的数值格式归一化。

    与 normalize_dirty_format（逐列判断）不同，此方法逐单元格处理：
    - "1.0" → "1"（整数值的浮点表示统一为整数格式）
    - "3.14" 保持不变（真实小数不动）
    - 非数值字符串保持不变

    适用场景：dirty CSV 中某列混杂了整数和注入的浮点脏数据，
    逐列方法会因为检测到小数而跳过整列，但逐单元格方法可以
    将正确行的 "1.0" 归一化为 "1"，同时保留异常行的 "142.3" 不变。

    Args:
        df: 输入 DataFrame（dtype=str）
        verbose: 是否打印统计信息

    Returns:
        归一化后的 DataFrame（深拷贝）
    """
    result = df.copy()
    total_normalized = 0

    for col_idx in range(result.shape[1]):
        for row_idx in range(result.shape[0]):
            val = result.iat[row_idx, col_idx]
            if not isinstance(val, str) or val in _NA_LIKE_VALUES:
                continue
            try:
                f = float(val)
                if f != f:  # NaN
                    continue
                # 仅在值是整数（如 1.0, -343.0）且字符串含 "." 时归一化
                if f == int(f) and '.' in val:
                    int_str = str(int(f))
                    if val != int_str:
                        result.iat[row_idx, col_idx] = int_str
                        total_normalized += 1
            except (ValueError, TypeError, OverflowError):
                continue

    if verbose and total_normalized > 0:
        print(f"  [逐单元格格式归一化] 统一 {total_normalized} 个值 (如 \"1.0\" → \"1\")")

    return result


def normalize_dirty_to_file(
    dirty_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """读取 dirty CSV，格式预清洗后写入文件。

    Args:
        dirty_path: 脏数据 CSV 路径
        output_path: 输出路径。None 则写入临时文件。
        verbose: 是否打印统计信息

    Returns:
        预清洗后的 CSV 文件路径
    """
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)

    normalized = normalize_dirty_format(dirty_df, verbose=verbose)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.csv', prefix='normalized_dirty_')
        os.close(fd)

    normalized.to_csv(output_path, index=False)

    if verbose:
        print(f"  [格式预清洗] 已写入: {output_path}")

    return output_path
