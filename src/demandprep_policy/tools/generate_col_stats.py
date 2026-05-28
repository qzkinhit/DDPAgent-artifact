"""
[已废弃] 为 9 个数据集生成 [STATISTICAL] 段写入 rules.txt
=====================================================

⚠️ 本脚本已废弃，不再使用。

原因: generate_col_stats 基于 dirty 数据 fit StandardScaler 计算统计量，
但主 pipeline (shared_preprocess) 基于 clean 数据 fit StandardScaler，
导致两者编码空间不一致。

替代方案: AutoDetector 现在完全依赖运行时计算 col_stats:
  1. fit(X_clean_subset) 时从实际数据计算 (最优先)
  2. detect() 时用 X_dirty 回退计算
所有 rules.txt 中的 [STATISTICAL] 段已被移除。

如需重新生成，请确保 scaler 与主 pipeline 的 fit 基准一致（clean 数据）。

旧用法（不再推荐）:
    python -m demandprep_policy.tools.generate_col_stats
    python -m demandprep_policy.tools.generate_col_stats --datasets beers adult
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 数据集配置（与 run_demandprep_policy_base.py 一致）
DATASETS = {
    'beers':         {'label_col': 'style'},
    'adult':         {'label_col': 'income'},
    'bike':          {'label_col': 'cnt'},
    'breast_cancer': {'label_col': 'class'},
    'har':           {'label_col': 'gt'},
    'mercedes':      {'label_col': 'y'},
    'nasa':          {'label_col': 'sound_pressure_level'},
    'smartfactory':  {'label_col': 'labels'},
    'soilmoisture':  {'label_col': 'soil_moisture'},
}


def load_dirty_encoded(dataset_name: str):
    """加载脏数据并编码到数值空间（与 run_demandprep_policy_base.py 一致）

    Returns:
        (X_dirty, column_names)
    """
    label_col = DATASETS[dataset_name]['label_col']
    data_dir = os.path.join(PROJECT_ROOT, 'data', dataset_name)

    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(data_dir, 'clean_with_index.csv')

    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    # 清洗列名
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]
    dirty_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)
    clean_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # 识别分类列
    categorical_cols = set()
    for col in feature_cols:
        combined = pd.concat([dirty_df[col], clean_df[col]]).dropna()
        combined = combined[~combined.astype(str).str.strip().isin(['?', '', 'N/A'])]
        try:
            pd.to_numeric(combined, errors='raise')
        except (ValueError, TypeError):
            categorical_cols.add(col)

    # LabelEncoder（基于 dirty + clean 的 union）
    label_encoders = {}
    X_df = dirty_df[feature_cols].copy()
    for col in feature_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            all_vals = pd.concat([
                dirty_df[col].dropna().astype(str),
                clean_df[col].dropna().astype(str),
            ]).unique()
            le.fit(all_vals)
            label_encoders[col] = le
            nan_mask = X_df[col].isna()
            if not nan_mask.all():
                X_df.loc[~nan_mask, col] = le.transform(
                    X_df.loc[~nan_mask, col].astype(str)
                )
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    X_dirty_raw = X_df.values.astype(float)

    # StandardScaler（基于 dirty 中非 NaN 行拟合，与 run_demandprep_policy_base.py 一致）
    nan_mask_row = np.isnan(X_dirty_raw).any(axis=1)
    X_for_fit = X_dirty_raw[~nan_mask_row]
    if len(X_for_fit) == 0:
        X_for_fit = X_dirty_raw.copy()
        col_means = np.nanmean(X_for_fit, axis=0)
        for c in range(X_for_fit.shape[1]):
            m = np.isnan(X_for_fit[:, c])
            if m.any():
                X_for_fit[m, c] = col_means[c] if not np.isnan(col_means[c]) else 0

    scaler = StandardScaler()
    scaler.fit(X_for_fit)

    # scale（保留 NaN）
    X_out = X_dirty_raw.copy()
    nan_mask = np.isnan(X_out)
    X_out[nan_mask] = 0
    X_scaled = scaler.transform(X_out)
    X_scaled[nan_mask] = np.nan

    return X_scaled, feature_cols


def compute_col_stats(X: np.ndarray) -> dict:
    """计算编码空间中每列的统计量（与 auto_detector._compute_col_stats 完全一致）"""
    col_stats = {}
    for col in range(X.shape[1]):
        valid = X[:, col][~np.isnan(X[:, col])]
        if len(valid) > 0:
            col_stats[col] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid) + 1e-6),
                'q1': float(np.percentile(valid, 25)),
                'q3': float(np.percentile(valid, 75)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'median': float(np.median(valid)),
            }
    return col_stats


def write_stats_to_rules(rules_path: str, column_names: list, col_stats: dict):
    """将 [STATISTICAL] 段追加/替换到 rules.txt"""
    # 读取现有内容
    existing_lines = []
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()

    # 移除已有的 [STATISTICAL] 段
    new_lines = []
    in_statistical = False
    for line in existing_lines:
        stripped = line.strip()
        if stripped == '[STATISTICAL]':
            in_statistical = True
            continue
        if in_statistical:
            if stripped.startswith('[') and stripped.endswith(']'):
                in_statistical = False
                new_lines.append(line)
            continue
        new_lines.append(line)

    # 确保末尾有换行
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    # 追加 [STATISTICAL] 段
    new_lines.append('\n[STATISTICAL]\n')
    new_lines.append('# 按列统计量（编码空间，基于脏数据计算）\n')
    for col_idx, stats in sorted(col_stats.items()):
        if col_idx < len(column_names):
            col_name = column_names[col_idx]
            parts = [f"{k}={v:.6f}" for k, v in stats.items()]
            new_lines.append(f"COL_STATS: {col_name} | {' | '.join(parts)}\n")

    with open(rules_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(description='Generate [STATISTICAL] section for rules.txt')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    args = parser.parse_args()

    print("=" * 60)
    print("为 rules.txt 生成 [STATISTICAL] 段")
    print("=" * 60)

    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"  [跳过] 未知数据集: {ds}")
            continue

        try:
            rules_path = os.path.join(PROJECT_ROOT, 'data', ds, 'rules.txt')
            if not os.path.exists(rules_path):
                print(f"  [跳过] {ds}: rules.txt 不存在")
                continue

            X_dirty, column_names = load_dirty_encoded(ds)
            col_stats = compute_col_stats(X_dirty)

            write_stats_to_rules(rules_path, column_names, col_stats)
            print(f"  {ds}: {len(col_stats)} 列统计量已写入 {rules_path}")

        except Exception as e:
            print(f"  [错误] {ds}: {e}")
            import traceback
            traceback.print_exc()

    print("\n完成!")


if __name__ == '__main__':
    main()
