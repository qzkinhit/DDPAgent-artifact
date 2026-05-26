"""
编辑距离工具模块
================

提供基于编辑距离（SequenceMatcher）的字符串相似度、最近值查找和 typo 生成功能。
零外部依赖，仅使用标准库 difflib + random + string。

三个核心场景:
  1. ErrorInjector: generate_typo() 为分类列生成真实拼写错误
  2. encode_df(): find_nearest_known() 将脏值映射到已知类别（替代 NaN）
  3. ValueEstimator: find_nearest_known() 编辑距离估值（修复明显 typo）
"""

import random
import string
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


def edit_distance_ratio(a: str, b: str) -> float:
    """计算两个字符串的相似度比率

    基于 SequenceMatcher.ratio()，返回值域 [0, 1]。
    1.0 = 完全相同，0.0 = 完全不同。

    Args:
        a: 第一个字符串
        b: 第二个字符串

    Returns:
        相似度 0~1
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_nearest_known(
    value: str,
    known_values: List[str],
    threshold: float = 0.6,
) -> Optional[str]:
    """在已知值列表中找编辑距离最近的值

    遍历 known_values，计算与 value 的相似度，
    返回相似度最高且 >= threshold 的值。

    Args:
        value: 待匹配的字符串
        known_values: 已知合法值列表
        threshold: 最低相似度阈值，低于此值返回 None

    Returns:
        最近的已知值，或 None（无匹配超过阈值）
    """
    if not value or not known_values:
        return None

    best_match: Optional[str] = None
    best_ratio: float = -1.0

    for known in known_values:
        ratio = SequenceMatcher(None, value, known).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = known

    if best_ratio >= threshold:
        return best_match
    return None


def find_top_k_nearest(
    value: str,
    known_values: List[str],
    k: int = 3,
    threshold: float = 0.3,
) -> List[Tuple[str, float]]:
    """在已知值列表中找编辑距离最近的 top-k 值

    Args:
        value: 待匹配的字符串
        known_values: 已知合法值列表
        k: 返回的最大数量
        threshold: 最低相似度阈值

    Returns:
        [(known_value, ratio), ...] 按相似度降序排列
    """
    if not value or not known_values:
        return []

    scored = []
    for known in known_values:
        ratio = SequenceMatcher(None, value, known).ratio()
        if ratio >= threshold:
            scored.append((known, ratio))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def generate_typo(value: str) -> str:
    """对字符串施加一个随机 typo

    四种等概率策略:
      - char_swap:   交换相邻字符      "Colorado" -> "Clorado"
      - char_delete: 删除随机字符        "Colorado" -> "Colordo"
      - char_insert: 插入随机字符        "Colorado" -> "Coloradoo"
      - case_change: 大小写变化          "Colorado" -> "cOlorado"

    对于长度 <= 1 的字符串，仅使用 char_insert 策略。

    Args:
        value: 原始字符串

    Returns:
        施加 typo 后的字符串（保证与原始值不同）
    """
    if not value:
        return value

    chars = list(value)

    if len(chars) <= 1:
        # 短字符串只能插入
        strategies = ['char_insert', 'case_change']
    else:
        strategies = ['char_swap', 'char_delete', 'char_insert', 'case_change']

    strategy = random.choice(strategies)

    if strategy == 'char_swap' and len(chars) >= 2:
        # 交换相邻字符
        pos = random.randint(0, len(chars) - 2)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        # 如果交换后相同（如 "aa" 中交换），换另一个位置
        result = ''.join(chars)
        if result == value and len(chars) >= 3:
            pos = (pos + 1) % (len(chars) - 1)
            chars = list(value)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    elif strategy == 'char_delete' and len(chars) >= 2:
        # 删除随机字符（避免删到只剩空串）
        pos = random.randint(0, len(chars) - 1)
        chars.pop(pos)

    elif strategy == 'char_insert':
        # 在随机位置插入一个随机字母
        pos = random.randint(0, len(chars))
        insert_char = random.choice(string.ascii_lowercase)
        chars.insert(pos, insert_char)

    elif strategy == 'case_change':
        # 随机切换 1~2 个字母的大小写
        alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
        if alpha_positions:
            n_changes = min(random.choice([1, 2]), len(alpha_positions))
            positions = random.sample(alpha_positions, n_changes)
            for pos in positions:
                chars[pos] = chars[pos].swapcase()

    result = ''.join(chars)

    # 保证与原始值不同
    if result == value:
        # fallback: 在末尾插入一个随机字符
        result = value + random.choice(string.ascii_lowercase)

    return result


