"""
规则解析器 (DemandClean 用)
============================

解析 data/{dataset}/rules.txt 的所有 Section，
为语义错误注入和检测提供结构化规则数据。

支持的 Section:
  [REGEX]       — 正则句法检测规则（仅存储，注入不用）
  [DOMAIN]      — 值域约束（语义注入: 注入超域值）
  [FD]          — 函数依赖（语义注入: FD 违规）
  [HORIZON_FD]  — 等价于 [FD]
  [CFD]         — 条件函数依赖（语义注入: 条件违规）
  [DC]          — 跨列完整性约束（语义注入参考）
  [STATISTICAL] — 统计阈值参数（仅存储，供检测用）

设计原则:
  - 所有规则注入产生的都是语义错误（规则违反）
  - 句法错误由 RAHA-aware 统计方法注入，不使用规则
  - 无丰富规则的数据集自动回退到 FD/随机注入
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class DomainRule:
    """值域约束规则

    Examples:
        INT [1, 10]   → dtype='INT', min_val=1, max_val=10, enum_vals=None
        ENUM {2, 4}   → dtype='ENUM', min_val=None, max_val=None, enum_vals={2, 4}
    """
    column: str
    dtype: str          # 'INT', 'FLOAT', 'ENUM'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    enum_vals: Optional[Set[str]] = None


@dataclass
class RegexRule:
    """正则检测规则（仅存储，注入不使用）"""
    column: str         # 列名或 'ALL_FEATURES'
    pattern: str        # 正则表达式


@dataclass
class CFDRule:
    """条件函数依赖规则

    Example:
        class=2, n_anomaly<=2 => Clump Thickness EXCESS >= 5 FROM_BASELINE 5
        → conditions = [('class', '=', '2'), ('n_anomaly', '<=', '2')]
          target_col = 'Clump Thickness'
          direction = 'EXCESS'   # 或 'DEFICIT'
          threshold = 5
          baseline = 5
    """
    conditions: List[Tuple[str, str, str]]  # [(col, op, val), ...]
    target_col: str
    direction: str      # 'EXCESS' 或 'DEFICIT'
    threshold: float
    baseline: float


@dataclass
class DCRule:
    """结构化的 DC (Denial Constraint) 规则

    DC 使用 denial 语义：当所有子句条件都成立时表示约束被违反。
    MARK 子句指定违反时标记的目标列。

    Examples:
        t1&EQ(t1.holiday, 1)&NEQ(t1.workingday, 0)&MARK(t1.workingday)
        → clauses = [{'type':'simple','op':'EQ','col':'holiday','value':1.0},
                      {'type':'simple','op':'NEQ','col':'workingday','value':0.0}]
          mark_cols = ['workingday']
          involved_cols = ['holiday', 'workingday']

        t1&GT(ABS(t1.454 - t1.458), 0.03)
        → clauses = [{'type':'abs_diff','op':'GT','col1':'454','col2':'458','value':0.03}]
          mark_cols = []
          involved_cols = ['454', '458']
    """
    raw: str                                    # 原始字符串
    clauses: List[Dict[str, Any]]               # 解析后的条件子句列表
    mark_cols: List[str]                        # MARK 标记的目标列（可能为空）
    involved_cols: List[str]                    # 所有涉及的列（不含 MARK 列自身）


@dataclass
class ParsedRules:
    """解析后的完整规则集合"""
    # 语义注入用
    domain_rules: List[DomainRule] = field(default_factory=list)
    fd_rules: List[Tuple[str, str]] = field(default_factory=list)       # [(lhs, rhs)]
    cfd_rules: List[CFDRule] = field(default_factory=list)
    dc_rules: List[DCRule] = field(default_factory=list)                # 结构化 DC 规则

    # 可选
    primary_key: Optional[List[str]] = None     # 主键列名列表（分块聚类用）

    # 仅存储，供检测用
    regex_rules: List[RegexRule] = field(default_factory=list)
    statistical: Dict[str, Any] = field(default_factory=dict)

    # 元信息
    raw_sections: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def has_rich_rules(self) -> bool:
        """是否有丰富规则（DOMAIN / CFD / DC）

        有丰富规则时优先用规则注入语义错误；
        否则回退到 FD 注入或随机注入。
        """
        return bool(self.domain_rules or self.cfd_rules or self.dc_rules)

    @property
    def has_any_rules(self) -> bool:
        """是否有任何规则"""
        return bool(
            self.domain_rules or self.fd_rules or self.cfd_rules
            or self.dc_rules or self.regex_rules
        )

    def summary(self) -> str:
        """规则摘要"""
        parts = []
        if self.regex_rules:
            parts.append(f"REGEX={len(self.regex_rules)}")
        if self.domain_rules:
            parts.append(f"DOMAIN={len(self.domain_rules)}")
        if self.fd_rules:
            parts.append(f"FD={len(self.fd_rules)}")
        if self.cfd_rules:
            parts.append(f"CFD={len(self.cfd_rules)}")
        if self.dc_rules:
            parts.append(f"DC={len(self.dc_rules)}")
        if self.statistical:
            parts.append(f"STAT={len(self.statistical)}")
        return f"ParsedRules({', '.join(parts) or 'empty'})"


# ============================================================================
# 解析函数
# ============================================================================

def parse_rules_file(rules_path: str) -> ParsedRules:
    """解析 rules.txt 的所有 Section

    Args:
        rules_path: 规则文件路径

    Returns:
        ParsedRules 结构化规则对象
    """
    if not rules_path or not os.path.exists(rules_path):
        return ParsedRules()

    # 第一步：按 section 分组读取原始行
    raw_sections: Dict[str, List[str]] = {}
    current_section = None

    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].upper()
                raw_sections.setdefault(current_section, [])
            elif current_section:
                raw_sections[current_section].append(line)

    # 第二步：逐 section 解析
    result = ParsedRules(raw_sections=raw_sections)

    # REGEX
    for line in raw_sections.get('REGEX', []):
        rule = _parse_regex_line(line)
        if rule:
            result.regex_rules.append(rule)

    # DOMAIN
    for line in raw_sections.get('DOMAIN', []):
        rule = _parse_domain_line(line)
        if rule:
            result.domain_rules.append(rule)

    # FD (合并 [FD] 和 [HORIZON_FD])
    for section_name in ('FD', 'HORIZON_FD'):
        for line in raw_sections.get(section_name, []):
            pair = _parse_fd_line(line)
            if pair:
                result.fd_rules.append(pair)

    # CFD
    for line in raw_sections.get('CFD', []):
        rule = _parse_cfd_line(line)
        if rule:
            result.cfd_rules.append(rule)

    # DC（结构化解析）
    for line in raw_sections.get('DC', []):
        rule = _parse_dc_line(line)
        if rule:
            result.dc_rules.append(rule)

    # PRIMARY_KEY（可选，分块聚类用）
    pk_lines = raw_sections.get('PRIMARY_KEY', [])
    if pk_lines:
        pk_cols = []
        for line in pk_lines:
            for col in line.split(','):
                col = col.strip()
                if col:
                    pk_cols.append(col)
        if pk_cols:
            result.primary_key = pk_cols

    # STATISTICAL
    for line in raw_sections.get('STATISTICAL', []):
        key, val = _parse_statistical_line(line)
        if key == 'COL_STATS' and isinstance(val, tuple):
            # 按列统计量: val = (col_name, {mean, std, ...})
            col_stats_dict = result.statistical.setdefault('col_stats', {})
            col_name, stats = val
            col_stats_dict[col_name] = stats
        elif key:
            result.statistical[key] = val

    return result


# ============================================================================
# Section 级解析器
# ============================================================================

def _parse_regex_line(line: str) -> Optional[RegexRule]:
    """解析 REGEX 规则行

    格式: COLUMN: pattern
    Example: ALL_FEATURES: ^(\\d)\\1$
    """
    if ':' not in line:
        return None
    col, pattern = line.split(':', 1)
    col = col.strip()
    pattern = pattern.strip()
    if col and pattern:
        return RegexRule(column=col, pattern=pattern)
    return None


def _parse_domain_line(line: str) -> Optional[DomainRule]:
    """解析 DOMAIN 规则行

    格式:
        Column Name: INT [min, max]
        Column Name: FLOAT [min, max]
        Column Name: ENUM {val1, val2, ...}
    """
    if ':' not in line:
        return None

    col, spec = line.split(':', 1)
    col = col.strip()
    spec = spec.strip()

    if not col or not spec:
        return None

    # INT/FLOAT 范围
    range_match = re.match(r'(INT|FLOAT)\s*\[([^,]+),\s*([^\]]+)\]', spec)
    if range_match:
        dtype = range_match.group(1)
        min_val = float(range_match.group(2).strip())
        max_val = float(range_match.group(3).strip())
        return DomainRule(column=col, dtype=dtype, min_val=min_val, max_val=max_val)

    # ENUM 集合
    enum_match = re.match(r'ENUM\s*\{([^}]+)\}', spec)
    if enum_match:
        vals = {v.strip() for v in enum_match.group(1).split(',')}
        return DomainRule(column=col, dtype='ENUM', enum_vals=vals)

    return None


def _parse_fd_line(line: str) -> Optional[Tuple[str, str]]:
    """解析 FD 规则行

    格式: LHS => RHS  或  LHS ⇒ RHS
    """
    for sep in ('=>', '⇒'):
        if sep in line:
            parts = line.split(sep)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                if lhs and rhs:
                    return (lhs, rhs)
    return None


def _parse_cfd_line(line: str) -> Optional[CFDRule]:
    """解析 CFD 规则行

    格式:
        class=2, n_anomaly<=2 => Clump Thickness EXCESS >= 5 FROM_BASELINE 5
        class=4, n_anomaly<=1 => Clump Thickness DEFICIT >= 3 FROM_BASELINE 4
    """
    if '=>' not in line:
        return None

    lhs, rhs = line.split('=>', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    # 解析条件部分 (逗号分隔的 col op val)
    conditions = []
    for cond_str in lhs.split(','):
        cond_str = cond_str.strip()
        # 匹配 col op val，支持 =, <=, >=, <, >, !=
        m = re.match(r'(\w+)\s*(<=|>=|!=|=|<|>)\s*(.+)', cond_str)
        if m:
            conditions.append((m.group(1).strip(), m.group(2), m.group(3).strip()))

    if not conditions:
        return None

    # 解析右侧:  ColName EXCESS/DEFICIT >= threshold FROM_BASELINE baseline
    rhs_match = re.match(
        r'(.+?)\s+(EXCESS|DEFICIT)\s*>=\s*(\d+(?:\.\d+)?)\s+FROM_BASELINE\s+(\d+(?:\.\d+)?)',
        rhs
    )
    if not rhs_match:
        return None

    target_col = rhs_match.group(1).strip()
    direction = rhs_match.group(2)
    threshold = float(rhs_match.group(3))
    baseline = float(rhs_match.group(4))

    return CFDRule(
        conditions=conditions,
        target_col=target_col,
        direction=direction,
        threshold=threshold,
        baseline=baseline,
    )


# ---- DC 子句解析（共享逻辑，auto_detector 也使用） ----

def parse_dc_clause(clause_str: str) -> Optional[Dict[str, Any]]:
    """解析单个 DC 子句

    支持格式:
        EQ(t1.col, val)     → {'type':'simple', 'op':'EQ', 'col':'col', 'value':val}
        GTE(t1.col, val)    → {'type':'simple', 'op':'GTE', 'col':'col', 'value':val}
        GT(ABS(t1.c1-t1.c2), val) → {'type':'abs_diff', 'op':'GT', 'col1','col2','value'}
        MARK(t1.col)        → {'type':'mark', 'col':'col'}
    """
    # MARK 格式: MARK(t1.col)
    mark_match = re.match(r'MARK\(t1\.(.+?)\)', clause_str)
    if mark_match:
        col = mark_match.group(1).strip()
        return {
            'type': 'mark',
            'col': col,
            'columns': [col],
        }

    # ABS 差值格式: GT(ABS(t1.col1 - t1.col2), val)
    abs_match = re.match(
        r'(GT|GTE|LT|LTE|EQ|NEQ)\(ABS\(t1\.(.+?)\s*-\s*t1\.(.+?)\)\s*,\s*(.+?)\)',
        clause_str
    )
    if abs_match:
        op = abs_match.group(1)
        col1 = abs_match.group(2).strip()
        col2 = abs_match.group(3).strip()
        val = abs_match.group(4).strip()
        try:
            val = float(val)
        except ValueError:
            return None
        return {
            'type': 'abs_diff',
            'op': op,
            'col1': col1,
            'col2': col2,
            'value': val,
            'columns': [col1, col2],
        }

    # 简单格式: OP(t1.col, val)
    simple_match = re.match(
        r'(GT|GTE|LT|LTE|EQ|NEQ|IQ)\(t1\.(.+?)\s*,\s*(.+?)\)',
        clause_str
    )
    if simple_match:
        op = simple_match.group(1)
        col = simple_match.group(2).strip()
        val = simple_match.group(3).strip()
        try:
            val_num = float(val)
            return {
                'type': 'simple',
                'op': op,
                'col': col,
                'value': val_num,
                'columns': [col],
            }
        except ValueError:
            return {
                'type': 'simple_str',
                'op': op,
                'col': col,
                'value': val,
                'columns': [col],
            }

    return None


def _parse_dc_line(line: str) -> Optional[DCRule]:
    """解析 DC 规则行

    格式: t1&CLAUSE1&CLAUSE2&...&MARK(t1.col)
    Example:
        t1&EQ(t1.holiday, 1)&NEQ(t1.workingday, 0)&MARK(t1.workingday)
        t1&GT(ABS(t1.454 - t1.458), 0.03)
    """
    line = line.strip()
    if not line or not line.startswith('t1'):
        return None

    parts = line.split('&')
    if len(parts) < 2:
        return None

    clauses = []
    mark_cols = []
    involved_cols = []

    for part in parts[1:]:  # 跳过 "t1"
        clause = parse_dc_clause(part.strip())
        if clause:
            if clause['type'] == 'mark':
                mark_cols.append(clause['col'])
            else:
                clauses.append(clause)
                involved_cols.extend(clause.get('columns', []))

    if not clauses:
        return None

    # 去重但保序
    seen = set()
    unique_cols = []
    for c in involved_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return DCRule(
        raw=line,
        clauses=clauses,
        mark_cols=mark_cols,
        involved_cols=unique_cols,
    )


def _parse_statistical_line(line: str) -> Tuple[Optional[str], Any]:
    """解析 STATISTICAL 配置行

    格式1 (全局参数): KEY: value
    Example:
        IQR_MULTIPLIER: 2.5
        ZSCORE_THRESHOLD: 3.0

    格式2 (按列统计量): COL_STATS: col_name | mean=0.123 | std=0.456 | ...
    Example:
        COL_STATS: abv | mean=0.059 | std=0.014 | q1=0.050 | q3=0.068 | min=0.001 | max=0.128 | median=0.056
    """
    if ':' not in line:
        return None, None

    key, val = line.split(':', 1)
    key = key.strip()
    val = val.strip()

    if not key:
        return None, None

    # 按列统计量
    if key == 'COL_STATS' and '|' in val:
        parts = [p.strip() for p in val.split('|')]
        col_name = parts[0]
        stats = {}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                try:
                    stats[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        if col_name and stats:
            return 'COL_STATS', (col_name, stats)
        return None, None

    # 尝试解析为数值
    try:
        return key, float(val)
    except ValueError:
        pass

    # 逗号分隔的列名列表
    if ',' in val:
        return key, [v.strip() for v in val.split(',')]

    return key, val


# ============================================================================
# 便捷函数
# ============================================================================

def load_rules(rules_path: Optional[str]) -> ParsedRules:
    """加载并解析规则文件（带空值保护）

    Args:
        rules_path: 规则文件路径，None 时返回空规则集

    Returns:
        ParsedRules 对象
    """
    if not rules_path:
        return ParsedRules()
    return parse_rules_file(rules_path)


def extract_fd_pairs(parsed: ParsedRules) -> List[Tuple[str, str]]:
    """提取 FD 规则对（兼容 HORIZON_FD 和 FD section）

    返回去重后的 (lhs, rhs) 列表
    """
    seen = set()
    result = []
    for lhs, rhs in parsed.fd_rules:
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def get_domain_range(parsed: ParsedRules, column: str) -> Optional[Tuple[float, float]]:
    """获取指定列的值域范围

    Returns:
        (min_val, max_val) 或 None
    """
    for rule in parsed.domain_rules:
        if rule.column == column and rule.min_val is not None:
            return (rule.min_val, rule.max_val)
    return None


def get_domain_enum(parsed: ParsedRules, column: str) -> Optional[Set[str]]:
    """获取指定列的枚举合法值集合

    Returns:
        set of valid values 或 None
    """
    for rule in parsed.domain_rules:
        if rule.column == column and rule.enum_vals is not None:
            return rule.enum_vals
    return None


def get_cfd_rules_for_class(parsed: ParsedRules, class_val: str) -> List[CFDRule]:
    """获取指定 class 值的所有 CFD 规则"""
    result = []
    for rule in parsed.cfd_rules:
        for col, op, val in rule.conditions:
            if col == 'class' and op == '=' and val == str(class_val):
                result.append(rule)
                break
    return result


def get_col_stats_from_rules(
    parsed: ParsedRules,
    column_names: List[str],
) -> Dict[int, Dict[str, float]]:
    """从 parsed_rules.statistical['col_stats'] 提取按列索引的统计量

    Args:
        parsed: 解析后的规则
        column_names: 特征列名列表

    Returns:
        {col_idx: {'mean': ..., 'std': ..., 'q1': ..., 'q3': ..., 'min': ..., 'max': ..., 'median': ...}}
    """
    col_stats_by_name = parsed.statistical.get('col_stats', {})
    if not col_stats_by_name or not column_names:
        return {}

    name_to_idx = {name: idx for idx, name in enumerate(column_names)}
    result = {}
    for col_name, stats in col_stats_by_name.items():
        if col_name in name_to_idx:
            result[name_to_idx[col_name]] = stats
    return result


def rules_to_dict(parsed: ParsedRules) -> Dict[str, Any]:
    """将 ParsedRules 转为可序列化的字典（供 config 传递）"""
    return {
        'has_rich_rules': parsed.has_rich_rules,
        'has_any_rules': parsed.has_any_rules,
        'domain_rules': [
            {
                'column': r.column,
                'dtype': r.dtype,
                'min_val': r.min_val,
                'max_val': r.max_val,
                'enum_vals': list(r.enum_vals) if r.enum_vals else None,
            }
            for r in parsed.domain_rules
        ],
        'fd_rules': parsed.fd_rules,
        'cfd_rules': [
            {
                'conditions': r.conditions,
                'target_col': r.target_col,
                'direction': r.direction,
                'threshold': r.threshold,
                'baseline': r.baseline,
            }
            for r in parsed.cfd_rules
        ],
        'dc_rules': [
            {
                'raw': r.raw,
                'clauses': r.clauses,
                'mark_cols': r.mark_cols,
                'involved_cols': r.involved_cols,
            }
            for r in parsed.dc_rules
        ],
        'primary_key': parsed.primary_key,
        'statistical': parsed.statistical,
    }


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'breast_cancer', 'rules.txt'
        )

    parsed = parse_rules_file(path)
    print(parsed.summary())

    if parsed.domain_rules:
        print(f"\nDOMAIN ({len(parsed.domain_rules)}):")
        for r in parsed.domain_rules:
            print(f"  {r}")

    if parsed.fd_rules:
        print(f"\nFD ({len(parsed.fd_rules)}):")
        for lhs, rhs in parsed.fd_rules:
            print(f"  {lhs} => {rhs}")

    if parsed.cfd_rules:
        print(f"\nCFD ({len(parsed.cfd_rules)}):")
        for r in parsed.cfd_rules:
            print(f"  {r.conditions} => {r.target_col} {r.direction} >= {r.threshold}")

    if parsed.dc_rules:
        print(f"\nDC ({len(parsed.dc_rules)}):")
        for r in parsed.dc_rules:
            mark_str = f" MARK={r.mark_cols}" if r.mark_cols else ""
            print(f"  {r.raw}  →  clauses={len(r.clauses)}, cols={r.involved_cols}{mark_str}")

    if parsed.statistical:
        print(f"\nSTATISTICAL: {parsed.statistical}")
