"""
增强版容忍度分析模块
====================

提供四大核心功能：
1. 模型容忍度计算 - 先验 / 后验容忍度，支持多模型批量评估与可视化
2. 检测器准确率评估 - 基于真实错误位置，按 missing / syntactic / semantic 分类评估
3. 跨版本对比分析 - 生成对比表格 CSV 和柱状图
4. Shapley + 容忍度联合分析 - 对各策略计算相对容忍度

依赖：
- sklearn 用于模型训练和评估
- matplotlib 用于可视化
- tools.getScoreML 中的 get_classifier / get_regressor
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sklearn 相关
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score,
)

# ---------------------------------------------------------------------------
# 从项目已有模块导入 get_classifier / get_regressor
# 兼容从包内或包外两种运行方式
# ---------------------------------------------------------------------------
try:
    # 作为 demandclean 包的子模块运行
    from demandclean.tools._compat import get_classifier, get_regressor
except ImportError:
    pass

# 直接引用 tools/getScoreML.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_tools_dir = os.path.join(_project_root, 'tools')
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

try:
    from getScoreML import get_classifier, get_regressor  # type: ignore[no-redef]
except ImportError:
    # 如果都导入不了，则内联实现一份轻量版
    def get_classifier(model_name: str):
        """内联的分类器工厂（备用）"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        _map = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(random_state=42),
            'knn': KNeighborsClassifier(),
            'dt': DecisionTreeClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
        }
        return _map.get(model_name, _map['rf'])

    def get_regressor(model_name: str):
        """内联的回归器工厂（备用）"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        _map = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression(),
            'ridge': Ridge(solver='lsqr', random_state=42),
            'lasso': Lasso(random_state=42),
            'svm': SVR(),
            'knn': KNeighborsRegressor(),
            'gb': GradientBoostingRegressor(random_state=42),
        }
        return _map.get(model_name, _map['rf'])

# ---------------------------------------------------------------------------
# matplotlib 可视化
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端，避免弹窗
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# 尝试设置中文字体
if _MPL_AVAILABLE:
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


# ============================================================================
# 内部辅助函数
# ============================================================================

def _safe_print(msg: str) -> None:
    """安全打印，处理 stdout 被重定向或关闭的情况"""
    try:
        print(msg)
    except (ValueError, IOError):
        if sys.__stdout__ is not None:
            sys.__stdout__.write(str(msg) + '\n')
            sys.__stdout__.flush()


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    model_name: str,
) -> float:
    """
    训练模型并返回统一的性能分数。

    - 分类: 返回 accuracy
    - 回归: 返回 R2（越大越好）
    - 聚类: 返回 silhouette score

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        task_type: 'classification' / 'regression' / 'clustering'
        model_name: 模型简称

    Returns:
        性能分数（越大越好）
    """
    # 防御性处理: 填充 NaN / Inf
    def _sanitize(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j]) | np.isinf(arr[:, j])
            arr[mask, j] = col_means[j]
        return arr

    X_train = _sanitize(X_train)
    X_test = _sanitize(X_test)

    # 清理 y 中的 NaN
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    # 去掉训练集中 y 为 NaN 的行
    train_valid = ~(np.isnan(y_train) | np.isinf(y_train))
    if not train_valid.all():
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]

    # 去掉测试集中 y 为 NaN 的行
    test_valid = ~(np.isnan(y_test) | np.isinf(y_test))
    if not test_valid.all():
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

    if len(X_train) < 5 or len(X_test) < 2:
        return 0.0

    if task_type == 'classification':
        model = get_classifier(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    elif task_type == 'regression':
        model = get_regressor(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return float(r2_score(y_test, y_pred))
    elif task_type == 'clustering':
        from sklearn.cluster import KMeans
        n_clusters = max(2, len(np.unique(y_test)))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        if len(set(labels)) < 2:
            return 0.0
        return float(silhouette_score(X_train, labels))
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")


def _ensure_dir(path: Optional[str]) -> None:
    """确保目录存在"""
    if path is not None:
        os.makedirs(path, exist_ok=True)


# ============================================================================
# 功能1: 模型容忍度计算
# ============================================================================

def compute_model_tolerance(
    X_dirty: np.ndarray,
    y_dirty: np.ndarray,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_result: np.ndarray,
    y_result: np.ndarray,
    task_type: str,
    models_list: List[str],
    save_dir: Optional[str] = None,
    task_name: str = '',
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    计算多模型的先验 / 后验容忍度。

    指标定义
    --------
    - P_clean   : 在完全干净数据上训练 → 测试性能
    - P_dirty   : 在脏数据上训练 → 测试性能
    - P_dc      : 在 DemandClean 清洗后数据上训练 → 测试性能

    - 先验容忍度 tolerance_prior = P_dc / P_dirty
      含义: 清洗后相对于"什么都不做"的倍率。>1 表示清洗有正向作用。

    - 后验容忍度 tolerance_post = (P_dc - P_dirty) / (P_clean - P_dirty)
      含义: 清洗恢复了多大比例的性能差距。1 代表完全恢复，0 代表没有恢复。

    Args:
        X_dirty: 脏数据特征，shape=(n_samples, n_features)
        y_dirty: 脏数据标签，shape=(n_samples,)
        X_clean: 干净数据特征
        y_clean: 干净数据标签
        X_result: DemandClean 清洗后的特征
        y_result: DemandClean 清洗后的标签
        task_type: 'classification' / 'regression' / 'clustering'
        models_list: 模型简称列表，如 ['rf', 'lr', 'svm']
        save_dir: 可视化图片保存目录（None 则不保存）
        task_name: 任务名称，用于图片标题和文件名前缀
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        字典: {
            model_name: {
                'P_clean': float,
                'P_dirty': float,
                'P_dc': float,
                'tolerance_prior': float,
                'tolerance_post': float,
            },
            ...
        }
    """
    # 统一使用干净标签进行评估（保证公平性）
    # 划分统一的测试集索引
    n = len(X_clean)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    results: Dict[str, Dict[str, Any]] = {}

    _safe_print("=" * 60)
    _safe_print(f"模型容忍度计算{f' - {task_name}' if task_name else ''}")
    _safe_print(f"任务类型: {task_type}  |  模型列表: {models_list}")
    _safe_print("=" * 60)

    for model_name in models_list:
        _safe_print(f"\n--- 模型: {model_name.upper()} ---")

        try:
            # P_clean: 在完全干净数据上训练
            P_clean = _train_and_evaluate(
                X_clean[train_idx], y_clean[train_idx],
                X_clean[test_idx], y_clean[test_idx],
                task_type, model_name,
            )

            # P_dirty: 在脏数据上训练，在干净测试集上评估
            # 确保 dirty 数据量足够
            n_dirty = min(len(X_dirty), n)
            dirty_train_idx = train_idx[train_idx < n_dirty]
            P_dirty = _train_and_evaluate(
                X_dirty[dirty_train_idx], y_clean[dirty_train_idx],
                X_clean[test_idx], y_clean[test_idx],
                task_type, model_name,
            )

            # P_dc: 在 DemandClean 结果上训练
            n_result = min(len(X_result), n)
            result_train_idx = train_idx[train_idx < n_result]
            P_dc = _train_and_evaluate(
                X_result[result_train_idx], y_clean[result_train_idx],
                X_clean[test_idx], y_clean[test_idx],
                task_type, model_name,
            )

            # 先验容忍度
            if abs(P_dirty) > 1e-12:
                tolerance_prior = P_dc / P_dirty
            else:
                tolerance_prior = float('inf') if P_dc > 0 else 0.0

            # 后验容忍度
            gap = P_clean - P_dirty
            if abs(gap) > 1e-12:
                tolerance_post = (P_dc - P_dirty) / gap
            else:
                # P_clean == P_dirty 说明脏数据没有损害性能，容忍度无意义
                tolerance_post = 1.0

            results[model_name] = {
                'P_clean': P_clean,
                'P_dirty': P_dirty,
                'P_dc': P_dc,
                'tolerance_prior': tolerance_prior,
                'tolerance_post': tolerance_post,
            }

            _safe_print(f"  P_clean  = {P_clean:.4f}")
            _safe_print(f"  P_dirty  = {P_dirty:.4f}")
            _safe_print(f"  P_dc     = {P_dc:.4f}")
            _safe_print(f"  先验容忍度 (P_dc / P_dirty)            = {tolerance_prior:.4f}")
            _safe_print(f"  后验容忍度 (P_dc-P_dirty)/(P_clean-P_dirty) = {tolerance_post:.4f}")

        except Exception as e:
            _safe_print(f"  [错误] 模型 {model_name} 评估失败: {e}")
            results[model_name] = {
                'P_clean': None,
                'P_dirty': None,
                'P_dc': None,
                'tolerance_prior': None,
                'tolerance_post': None,
                'error': str(e),
            }

    # 可视化
    if save_dir is not None and _MPL_AVAILABLE:
        _plot_tolerance_bar(results, save_dir, task_name)

    return results


def _plot_tolerance_bar(
    results: Dict[str, Dict[str, Any]],
    save_dir: str,
    task_name: str,
) -> None:
    """
    绘制容忍度柱状图并保存。

    包含两个子图：
    - 左图：各模型的 P_clean / P_dirty / P_dc 对比
    - 右图：各模型的先验容忍度与后验容忍度
    """
    _ensure_dir(save_dir)

    # 过滤掉出错的模型
    valid = {k: v for k, v in results.items() if v.get('P_clean') is not None}
    if not valid:
        _safe_print("  [警告] 没有有效结果，跳过可视化")
        return

    model_names = list(valid.keys())
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + 2 * n_models, 5))

    # --- 左图: 性能对比 ---
    p_clean_vals = [valid[m]['P_clean'] for m in model_names]
    p_dirty_vals = [valid[m]['P_dirty'] for m in model_names]
    p_dc_vals = [valid[m]['P_dc'] for m in model_names]

    bars1 = ax1.bar(x - width, p_clean_vals, width, label='P_clean (FullFix)', color='#2ecc71')
    bars2 = ax1.bar(x, p_dirty_vals, width, label='P_dirty (NoFix)', color='#e74c3c')
    bars3 = ax1.bar(x + width, p_dc_vals, width, label='P_dc (DemandClean)', color='#3498db')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Performance')
    title_prefix = f'{task_name} - ' if task_name else ''
    ax1.set_title(f'{title_prefix}Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in model_names])
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', fontsize=7)

    # --- 右图: 容忍度 ---
    prior_vals = [valid[m]['tolerance_prior'] for m in model_names]
    post_vals = [valid[m]['tolerance_post'] for m in model_names]

    bars4 = ax2.bar(x - width / 2, prior_vals, width, label='Tolerance Prior', color='#f39c12')
    bars5 = ax2.bar(x + width / 2, post_vals, width, label='Tolerance Post', color='#9b59b6')

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Tolerance')
    ax2.set_title(f'{title_prefix}Tolerance')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in model_names])
    ax2.legend(fontsize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline=1')
    ax2.grid(axis='y', alpha=0.3)

    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax2.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'{task_name}_tolerance_bar.png' if task_name else 'tolerance_bar.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _safe_print(f"  容忍度柱状图已保存: {fname}")


# ============================================================================
# 功能2: 检测器准确率评估
# ============================================================================

def evaluate_detector_accuracy(
    detected_errors: Dict[str, List],
    X_dirty: np.ndarray,
    X_clean: np.ndarray,
    verbose: bool = True,
    y_dirty: Optional[np.ndarray] = None,
    y_clean: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    评估检测器的准确率。

    通过对比 X_dirty 和 X_clean 得到真实错误位置 (ground truth),
    再与 detected_errors 进行 Precision / Recall / F1 的计算。

    同时按照错误类型 (missing / syntactic / semantic) 分别评估。

    Args:
        detected_errors: 检测器输出的错误字典，格式如::

            {
                'missing':   [(idx, col, estimated_val), ...],
                'syntactic': [(idx, col, estimated_val, noise), ...],
                'semantic':  [(idx, col, estimated_val, current_val), ...],
            }

        X_dirty: 脏数据特征矩阵，shape=(n_samples, n_features)
        X_clean: 干净数据特征矩阵，shape=(n_samples, n_features)
        verbose: 是否打印详细信息

    Returns:
        accuracy_dict: {
            'overall': {'precision': ..., 'recall': ..., 'f1': ..., 'tp': ..., 'fp': ..., 'fn': ...},
            'missing': {'precision': ..., 'recall': ..., 'f1': ..., ...},
            'syntactic': {'precision': ..., 'recall': ..., 'f1': ..., ...},
            'semantic': {'precision': ..., 'recall': ..., 'f1': ..., ...},
            'ground_truth_total': int,
            'detected_total': int,
        }
    """
    assert X_dirty.shape == X_clean.shape, (
        f"X_dirty 和 X_clean 形状不一致: {X_dirty.shape} vs {X_clean.shape}"
    )

    n_rows, n_cols = X_dirty.shape

    # ------------------------------------------------------------------
    # 1. 构建真实错误位置集合（ground truth），并按类型分类
    # ------------------------------------------------------------------
    gt_missing: set = set()      # 缺失值: dirty 中为 NaN，clean 中不为 NaN
    gt_value_error: set = set()  # 值错误（含 syntactic + semantic）

    for i in range(n_rows):
        for j in range(n_cols):
            d_val = X_dirty[i, j]
            c_val = X_clean[i, j]

            d_is_nan = (np.isnan(d_val) if isinstance(d_val, float) else False)
            c_is_nan = (np.isnan(c_val) if isinstance(c_val, float) else False)

            if d_is_nan and not c_is_nan:
                gt_missing.add((i, j))
            elif not d_is_nan and not c_is_nan:
                # 值不相等视为值错误
                try:
                    if abs(float(d_val) - float(c_val)) > 1e-9:
                        gt_value_error.add((i, j))
                except (TypeError, ValueError):
                    if str(d_val).strip() != str(c_val).strip():
                        gt_value_error.add((i, j))

    gt_all = gt_missing | gt_value_error

    # 标签错误 ground truth: y_dirty != y_clean
    gt_label_noise: set = set()
    if y_dirty is not None and y_clean is not None:
        for i in range(len(y_dirty)):
            try:
                if abs(float(y_dirty[i]) - float(y_clean[i])) > 1e-9:
                    gt_label_noise.add((i, -1))
            except (TypeError, ValueError):
                if str(y_dirty[i]).strip() != str(y_clean[i]).strip():
                    gt_label_noise.add((i, -1))

    gt_all_with_label = gt_all | gt_label_noise

    # ------------------------------------------------------------------
    # 2. 构建检测器报告的错误位置集合
    # ------------------------------------------------------------------
    det_missing: set = set()
    det_syntactic: set = set()
    det_semantic: set = set()
    det_label_noise: set = set()

    for item in detected_errors.get('missing', []):
        det_missing.add((item[0], item[1]))
    for item in detected_errors.get('syntactic', []):
        det_syntactic.add((item[0], item[1]))
    for item in detected_errors.get('semantic', []):
        det_semantic.add((item[0], item[1]))
    for item in detected_errors.get('label_noise', []):
        det_label_noise.add((item[0], item[1]))

    det_all = det_missing | det_syntactic | det_semantic
    det_all_with_label = det_all | det_label_noise

    # ------------------------------------------------------------------
    # 3. 计算指标
    # ------------------------------------------------------------------
    def _calc_prf(gt_set: set, det_set: set) -> Dict[str, Any]:
        tp = len(gt_set & det_set)
        fp = len(det_set - gt_set)
        fn = len(gt_set - det_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': len(gt_set),
            'det_count': len(det_set),
        }

    # overall 包含标签错误
    overall = _calc_prf(gt_all_with_label, det_all_with_label)

    # 按类型: missing 直接对比
    missing_eval = _calc_prf(gt_missing, det_missing)

    # syntactic: 因为 ground truth 中不区分 syntactic / semantic，
    # 这里用 gt_value_error 作为这两类的联合 ground truth
    syntactic_eval = _calc_prf(gt_value_error, det_syntactic)
    semantic_eval = _calc_prf(gt_value_error, det_semantic)

    # syntactic + semantic 合并后与 gt_value_error 比较
    det_value_combined = det_syntactic | det_semantic
    value_combined_eval = _calc_prf(gt_value_error, det_value_combined)

    # 标签噪声评估
    label_noise_eval = _calc_prf(gt_label_noise, det_label_noise)

    result = {
        'overall': overall,
        'missing': missing_eval,
        'syntactic': syntactic_eval,
        'semantic': semantic_eval,
        'value_error_combined': value_combined_eval,
        'label_noise': label_noise_eval,
        'ground_truth_total': len(gt_all_with_label),
        'detected_total': len(det_all_with_label),
    }

    # ------------------------------------------------------------------
    # 4. 打印
    # ------------------------------------------------------------------
    if verbose:
        _safe_print("\n" + "=" * 60)
        _safe_print("检测器准确率评估")
        _safe_print("=" * 60)
        _safe_print(f"真实错误总数: {len(gt_all)} (缺失: {len(gt_missing)}, 值错误: {len(gt_value_error)})")
        _safe_print(f"检测报告总数: {len(det_all)} (missing: {len(det_missing)}, "
                     f"syntactic: {len(det_syntactic)}, semantic: {len(det_semantic)})")
        _safe_print("-" * 60)

        for category, label in [
            ('overall', '总体'),
            ('missing', '缺失值 (missing)'),
            ('syntactic', '句法错误 (syntactic)'),
            ('semantic', '语义错误 (semantic)'),
            ('value_error_combined', '值错误合并 (syntactic+semantic)'),
            ('label_noise', '标签噪声 (label_noise)'),
        ]:
            m = result[category]
            _safe_print(f"\n  [{label}]")
            _safe_print(f"    Precision = {m['precision']:.4f}  |  Recall = {m['recall']:.4f}  |  F1 = {m['f1']:.4f}")
            _safe_print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
                         f"(GT={m['gt_count']}, Det={m['det_count']})")

    return result


# ============================================================================
# 功能3: 跨版本对比分析
# ============================================================================

def compare_versions(
    all_results: Dict[str, Dict[str, Any]],
    save_dir: Optional[str] = None,
    task_name: str = '',
) -> pd.DataFrame:
    """
    跨版本（或跨策略 / 跨方法）的对比分析。

    将所有版本的结果生成:
    1. 对比表格 (DataFrame + CSV)
    2. 对比柱状图

    Args:
        all_results: 所有版本的结果字典，格式::

            {
                'NoFix': {model: {P_clean, P_dirty, P_dc, tolerance_prior, tolerance_post}, ...},
                'FullFix': {...},
                'DemandClean_v1': {...},
                'DemandClean_v2': {...},
                ...
            }

            也支持扁平格式::

            {
                'NoFix': {P_dirty: ..., tolerance_prior: ..., ...},
                ...
            }

        save_dir: 结果保存目录
        task_name: 任务名称前缀

    Returns:
        对比表格 DataFrame
    """
    _safe_print("\n" + "=" * 60)
    _safe_print(f"跨版本对比分析{f' - {task_name}' if task_name else ''}")
    _safe_print("=" * 60)

    rows = []

    for version_name, version_data in all_results.items():
        # 判断是 nested（每个 model 一个子字典）还是 flat
        sample_val = next(iter(version_data.values()), None)
        is_nested = isinstance(sample_val, dict)

        if is_nested:
            for model_name, metrics in version_data.items():
                if isinstance(metrics, dict):
                    row = {'version': version_name, 'model': model_name}
                    row.update({
                        k: v for k, v in metrics.items()
                        if k != 'error' and v is not None
                    })
                    rows.append(row)
        else:
            # 扁平格式
            row = {'version': version_name, 'model': 'default'}
            row.update({
                k: v for k, v in version_data.items()
                if v is not None
            })
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        _safe_print("  [警告] 没有数据可对比")
        return df

    _safe_print(f"\n对比表格 ({len(df)} 行):")
    _safe_print(df.to_string(index=False))

    # 保存 CSV
    if save_dir is not None:
        _ensure_dir(save_dir)
        csv_path = os.path.join(
            save_dir,
            f'{task_name}_version_comparison.csv' if task_name else 'version_comparison.csv'
        )
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        _safe_print(f"\n  对比表格已保存: {csv_path}")

    # 绘制对比柱状图
    if save_dir is not None and _MPL_AVAILABLE:
        _plot_version_comparison(df, save_dir, task_name)

    return df


def _plot_version_comparison(
    df: pd.DataFrame,
    save_dir: str,
    task_name: str,
) -> None:
    """绘制跨版本对比柱状图"""
    _ensure_dir(save_dir)

    # 选择要展示的指标列
    metric_cols = [c for c in df.columns if c not in ('version', 'model', 'error')]
    if not metric_cols:
        return

    # 为每个数值指标画一个子图
    numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return

    # 最多展示 6 个指标
    show_cols = numeric_cols[:6]
    n_plots = len(show_cols)
    n_cols_fig = min(3, n_plots)
    n_rows_fig = (n_plots + n_cols_fig - 1) // n_cols_fig

    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4 * n_rows_fig))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 版本-模型 组合标签
    df['label'] = df['version'] + ' / ' + df['model']

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, col in enumerate(show_cols):
        ax = axes[i]
        vals = df[col].astype(float).values
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors[:len(vals)])
        ax.set_title(col, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(df['label'].values, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6)

    # 隐藏多余子图
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    title_prefix = f'{task_name} - ' if task_name else ''
    fig.suptitle(f'{title_prefix}Version Comparison', fontsize=12, y=1.02)
    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f'{task_name}_version_comparison.png' if task_name else 'version_comparison.png'
    )
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _safe_print(f"  对比柱状图已保存: {fname}")


# ============================================================================
# 功能4: Shapley + 容忍度联合分析
# ============================================================================

def compute_strategy_tolerance(
    strategies_results: Dict[str, Dict[str, Any]],
    baseline_results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    对各策略计算相对容忍度（Shapley + 容忍度联合分析）。

    将每种策略的清洗效果与 baseline（通常是 NoFix）进行比较，
    计算相对容忍度和 Shapley 风格的边际贡献。

    指标定义
    --------
    对于每个策略 s 和每个模型 m：

    - relative_tolerance = P_s(m) / P_baseline(m)
      相对于 baseline 的性能倍率

    - marginal_gain = P_s(m) - P_baseline(m)
      边际性能增益

    - normalized_gain = (P_s - P_baseline) / (max_P_all - P_baseline)
      归一化后的增益（0~1 之间，1 表示达到最佳策略水平）

    Args:
        strategies_results: 各策略的结果字典::

            {
                'FullFix': {
                    'rf': {'P_dc': 0.92, ...},
                    'lr': {'P_dc': 0.88, ...},
                },
                'DemandClean': {
                    'rf': {'P_dc': 0.90, ...},
                    'lr': {'P_dc': 0.86, ...},
                },
                'RelaxFix': {...},
                ...
            }

            也支持扁平格式::

            {
                'FullFix': {'P_dc': 0.92, ...},
                'DemandClean': {'P_dc': 0.90, ...},
            }

        baseline_results: baseline 的结果字典（通常是 NoFix）::

            {
                'rf': {'P_dc': 0.80, ...},
                ...
            }

            或扁平格式::

            {'P_dc': 0.80, ...}

    Returns:
        策略容忍度字典::

            {
                strategy_name: {
                    'model_name:relative_tolerance': float,
                    'model_name:marginal_gain': float,
                    'model_name:normalized_gain': float,
                    'avg_relative_tolerance': float,
                    'avg_marginal_gain': float,
                    'avg_normalized_gain': float,
                },
                ...
            }
    """
    _safe_print("\n" + "=" * 60)
    _safe_print("Shapley + 容忍度联合分析")
    _safe_print("=" * 60)

    # 判断 baseline 是 nested 还是 flat
    def _extract_perf(data: Dict, model_key: Optional[str] = None) -> Optional[float]:
        """从结果字典中提取性能值 P_dc"""
        if model_key and model_key in data and isinstance(data[model_key], dict):
            return data[model_key].get('P_dc')
        if 'P_dc' in data:
            return data['P_dc']
        # 尝试取第一个子字典的 P_dc
        for v in data.values():
            if isinstance(v, dict) and 'P_dc' in v:
                return v['P_dc']
        return None

    # 确定所有模型名
    def _get_models(data: Dict) -> List[str]:
        models = []
        for k, v in data.items():
            if isinstance(v, dict) and 'P_dc' in v:
                models.append(k)
        return models if models else ['default']

    baseline_is_nested = any(
        isinstance(v, dict) and 'P_dc' in v
        for v in baseline_results.values()
    )

    if baseline_is_nested:
        model_names = _get_models(baseline_results)
    else:
        model_names = ['default']

    # 收集所有策略在各模型上的 P_dc
    all_perf: Dict[str, Dict[str, float]] = {}  # {strategy: {model: P_dc}}

    for strategy_name, strategy_data in strategies_results.items():
        all_perf[strategy_name] = {}
        for m in model_names:
            if m == 'default':
                p = _extract_perf(strategy_data)
            else:
                p = _extract_perf(strategy_data, m)
            if p is not None:
                all_perf[strategy_name][m] = p

    # baseline 性能
    baseline_perf: Dict[str, float] = {}
    for m in model_names:
        if m == 'default':
            p = _extract_perf(baseline_results)
        else:
            p = _extract_perf(baseline_results, m)
        if p is not None:
            baseline_perf[m] = p

    # 各模型上所有策略的最大性能（用于 normalized_gain）
    max_perf: Dict[str, float] = {}
    for m in model_names:
        vals = [all_perf[s].get(m, 0) for s in all_perf]
        base = baseline_perf.get(m, 0)
        vals.append(base)
        max_perf[m] = max(vals) if vals else 0

    # 计算各策略的相对容忍度
    output: Dict[str, Dict[str, float]] = {}

    for strategy_name in strategies_results:
        entry: Dict[str, float] = {}
        rel_tol_list = []
        gain_list = []
        norm_gain_list = []

        for m in model_names:
            p_s = all_perf.get(strategy_name, {}).get(m)
            p_base = baseline_perf.get(m)

            if p_s is None or p_base is None:
                continue

            # 相对容忍度
            if abs(p_base) > 1e-12:
                rel_tol = p_s / p_base
            else:
                rel_tol = float('inf') if p_s > 0 else 0.0

            # 边际增益
            marginal = p_s - p_base

            # 归一化增益
            max_gap = max_perf[m] - p_base
            if abs(max_gap) > 1e-12:
                norm_g = marginal / max_gap
            else:
                norm_g = 0.0

            key_prefix = m if m != 'default' else ''
            sep = ':' if key_prefix else ''
            entry[f'{key_prefix}{sep}relative_tolerance'] = rel_tol
            entry[f'{key_prefix}{sep}marginal_gain'] = marginal
            entry[f'{key_prefix}{sep}normalized_gain'] = norm_g

            rel_tol_list.append(rel_tol)
            gain_list.append(marginal)
            norm_gain_list.append(norm_g)

        # 均值
        if rel_tol_list:
            entry['avg_relative_tolerance'] = float(np.mean(rel_tol_list))
            entry['avg_marginal_gain'] = float(np.mean(gain_list))
            entry['avg_normalized_gain'] = float(np.mean(norm_gain_list))
        else:
            entry['avg_relative_tolerance'] = 0.0
            entry['avg_marginal_gain'] = 0.0
            entry['avg_normalized_gain'] = 0.0

        output[strategy_name] = entry

    # 打印汇总
    _safe_print("\n策略容忍度汇总:")
    _safe_print(f"{'策略':<20s} {'平均相对容忍度':>14s} {'平均边际增益':>12s} {'平均归一化增益':>14s}")
    _safe_print("-" * 62)
    for sname, sdata in output.items():
        _safe_print(
            f"{sname:<20s} "
            f"{sdata.get('avg_relative_tolerance', 0):>14.4f} "
            f"{sdata.get('avg_marginal_gain', 0):>12.4f} "
            f"{sdata.get('avg_normalized_gain', 0):>14.4f}"
        )

    return output


# ============================================================================
# 便捷函数: 从 DataFrame / CSV 出发的端到端调用
# ============================================================================

def tolerance_from_dataframes(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    result_df: pd.DataFrame,
    label_column: str,
    task_type: str = 'classification',
    models_list: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    task_name: str = '',
) -> Dict[str, Dict[str, Any]]:
    """
    从 DataFrame 出发计算容忍度（便捷封装）。

    自动进行预处理（编码、填充、标准化），然后调用 compute_model_tolerance。

    Args:
        dirty_df: 脏数据 DataFrame
        clean_df: 干净数据 DataFrame
        result_df: DemandClean 清洗后的 DataFrame
        label_column: 标签列名
        task_type: 任务类型
        models_list: 模型列表
        save_dir: 保存目录
        task_name: 任务名称

    Returns:
        compute_model_tolerance 的结果
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if models_list is None:
        models_list = ['rf', 'lr']

    def _preprocess(df: pd.DataFrame):
        X = df.drop(columns=[label_column]).copy()
        y = df[label_column].copy()

        # 编码分类特征
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # 填充缺失值
        X = X.fillna(X.mean())

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 编码标签
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        return X_scaled, np.array(y, dtype=float)

    X_dirty, y_dirty = _preprocess(dirty_df)
    X_clean, y_clean = _preprocess(clean_df)
    X_result, y_result = _preprocess(result_df)

    return compute_model_tolerance(
        X_dirty=X_dirty,
        y_dirty=y_dirty,
        X_clean=X_clean,
        y_clean=y_clean,
        X_result=X_result,
        y_result=y_result,
        task_type=task_type,
        models_list=models_list,
        save_dir=save_dir,
        task_name=task_name,
    )


def tolerance_from_csv(
    dirty_path: str,
    clean_path: str,
    result_path: str,
    label_column: str,
    task_type: str = 'classification',
    models_list: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    task_name: str = '',
) -> Dict[str, Dict[str, Any]]:
    """
    从 CSV 文件路径出发计算容忍度（便捷封装）。

    Args:
        dirty_path: 脏数据 CSV 路径
        clean_path: 干净数据 CSV 路径
        result_path: DemandClean 清洗后 CSV 路径
        label_column: 标签列名
        task_type: 任务类型
        models_list: 模型列表
        save_dir: 保存目录
        task_name: 任务名称

    Returns:
        compute_model_tolerance 的结果
    """
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)
    result_df = pd.read_csv(result_path)

    return tolerance_from_dataframes(
        dirty_df=dirty_df,
        clean_df=clean_df,
        result_df=result_df,
        label_column=label_column,
        task_type=task_type,
        models_list=models_list,
        save_dir=save_dir,
        task_name=task_name,
    )


# ============================================================================
# 自测入口
# ============================================================================

def _self_test():
    """模块自测：使用合成数据验证四大功能"""
    np.random.seed(42)
    n = 500
    n_features = 5

    # --- 生成合成数据 ---
    X_clean = np.random.randn(n, n_features)
    coef = np.random.randn(n_features)
    y_clean = (X_clean @ coef + np.random.randn(n) * 0.3 > 0).astype(float)

    # 脏数据: 10% 缺失 + 15% 噪声
    X_dirty = X_clean.copy()
    missing_mask = np.random.rand(n, n_features) < 0.10
    X_dirty[missing_mask] = np.nan
    noise_mask = np.random.rand(n, n_features) < 0.15
    X_dirty[noise_mask] += np.random.randn(np.sum(noise_mask)) * 3

    # DemandClean 结果: 修复了大部分缺失值 + 部分噪声
    X_result = X_dirty.copy()
    # 用均值填充缺失
    col_means = np.nanmean(X_dirty, axis=0)
    for j in range(n_features):
        nan_idx = np.isnan(X_result[:, j])
        X_result[nan_idx, j] = col_means[j]
    # 部分修复噪声
    repair_mask = noise_mask & (np.random.rand(n, n_features) < 0.6)
    X_result[repair_mask] = X_clean[repair_mask]

    y_dirty = y_clean.copy()
    y_result = y_clean.copy()

    _safe_print("\n" + "#" * 70)
    _safe_print("# tolerance_analysis.py 自测")
    _safe_print("#" * 70)

    # 功能 1: 模型容忍度
    _safe_print("\n\n>>> 功能1: 模型容忍度计算")
    tol = compute_model_tolerance(
        X_dirty, y_dirty, X_clean, y_clean, X_result, y_result,
        task_type='classification',
        models_list=['rf', 'lr'],
        task_name='self_test',
    )

    # 功能 2: 检测器准确率
    _safe_print("\n\n>>> 功能2: 检测器准确率评估")
    # 模拟一个检测器的输出
    X_dirty_filled = X_dirty.copy()
    for j in range(n_features):
        nan_idx = np.isnan(X_dirty_filled[:, j])
        X_dirty_filled[nan_idx, j] = col_means[j]

    detected_errors: Dict[str, List] = {
        'missing': [],
        'syntactic': [],
        'semantic': [],
    }
    for i in range(n):
        for j in range(n_features):
            if missing_mask[i, j]:
                detected_errors['missing'].append((i, j, col_means[j]))
            elif noise_mask[i, j] and np.random.rand() < 0.7:
                detected_errors['syntactic'].append((i, j, col_means[j], X_dirty_filled[i, j] - col_means[j]))

    det_acc = evaluate_detector_accuracy(detected_errors, X_dirty_filled, X_clean)

    # 功能 3: 跨版本对比
    _safe_print("\n\n>>> 功能3: 跨版本对比分析")
    all_results = {
        'NoFix': {m: {'P_dc': v['P_dirty']} for m, v in tol.items()},
        'DemandClean': {m: {'P_dc': v['P_dc']} for m, v in tol.items()},
        'FullFix': {m: {'P_dc': v['P_clean']} for m, v in tol.items()},
    }
    compare_df = compare_versions(all_results, task_name='self_test')

    # 功能 4: Shapley + 容忍度联合分析
    _safe_print("\n\n>>> 功能4: Shapley + 容忍度联合分析")
    baseline = {m: {'P_dc': v['P_dirty']} for m, v in tol.items()}
    strategies = {
        'FullFix': {m: {'P_dc': v['P_clean']} for m, v in tol.items()},
        'DemandClean': {m: {'P_dc': v['P_dc']} for m, v in tol.items()},
    }
    strat_tol = compute_strategy_tolerance(strategies, baseline)

    _safe_print("\n\n>>> 自测完成!")


if __name__ == '__main__':
    _self_test()
