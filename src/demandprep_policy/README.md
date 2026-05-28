# DemandPrepPolicy：基于 DQN 的按需数据清洗框架

> **核心思想**：将数据清洗建模为 MDP 序贯决策问题。Agent 学会识别哪些错误对下游任务影响大（需花预算用真值修复），哪些用免费的值估计即可，哪些可以跳过不管。

---

## 全景流程图

```
 dirty_index.csv (2410行, CSV字符串)
       │
  ┌────┴────┐
  │ 数据预处理 │ LabelEncoder + StandardScaler
  └────┬────┘
       │ (2410行, LE+SS numpy 含NaN)
  ┌────┴────┐
  │ 三路划分  │ train=1446(60%) / val=482(20%) / test=482(20%)
  └────┬────┘
       │
  ┌────┴────┐
  │ 错误检测  │ AutoDetector: 缺失→句法(RAHA+DOMAIN)→语义(FD)→标签
  └────┬────┘
       │ ~3256 个错误单元格
  ┌────┴────┐
  │ RAHA预修复│ 20行标注行直接修复
  └────┬────┘
       │
  ┌────┴────────┐
  │ Clean Base  │ DeleteFix vs VE-Fill (5-fold CV + √(n/N) 加权)
  └────┬────────┘
       │
  ┌────┴────┐
  │ DQN训练  │ 300 episodes: 注入→Agent决策→reward→经验回放
  └────┬────┘
       │
  ┌────┴────┐
  │ 两阶段推理│ Phase1: Plan(用户只需提供~30个真值) → Phase2: Execute
  └────┬────┘
       │
  ┌────┴────┐
  │   评估   │ 6种Baseline + EDR + Shapley + 成本核算
  └─────────┘
```

---

## Quick Start

```bash
# 1. 安装依赖
pip install numpy pandas scikit-learn torch

# 2. 运行 beers v6（主版本）
python run_demandprep_policy/run_demandprep_policy_base.py --dataset beers --versions v6 --n_episodes 300

# 3. 查看结果
ls results/demandprep_policy/beers/v6_auto_dueling_two/
```

### Python API

```python
from demandprep_policy import DemandPrepPolicy

# 创建 v6 配置
dc = DemandPrepPolicy(
    task_type='classification',
    model_type='random_forest',
    agent_type='dueling_two_stage',
    detector_mode='auto',
    inference_mode='two_phase',
    n_episodes=300,
    count_raha_cost=True,
)

# 训练
dc.fit(X_dirty, y, X_clean_val=X_clean_val, y_clean_val=y_clean_val)

# 两阶段推理
plan = dc.plan(X_dirty, y)               # Phase1: 生成修复计划
X_cleaned, y_cleaned, mask = dc.execute(  # Phase2: 执行修复
    X_dirty, true_values, y_dirty=y
)
```

---

## 版本矩阵

| 版本 | 检测器 | Agent 类型 | 推理模式 | 消融维度 |
|------|--------|-----------|---------|---------|
| v3 | oracle | plain_single | single | 基准：oracle + 最简 Agent |
| v4 | oracle | plain_two | two_phase | 两阶段 vs 单阶段 |
| v5 | auto | dueling_single | single | Dueling 单阶段 |
| **v6** | **auto** | **dueling_two_stage** | **two_phase** | **★ 主版本** |
| v7 | auto | plain_single | single | 消融：无 Dueling |
| v8 | auto | plain_two | two_phase | 消融：无 Dueling + 两阶段 |

消融映射：**v6 vs v5** = 两阶段增益 | **v6 vs v7** = Dueling 增益 | **v6 vs v3** = 自动检测 vs Oracle

---

## 数据集列表

| 数据集 | 任务 | 模型 | 行数 | 特征 | 标签列 | 特点 |
|--------|------|------|------|------|--------|------|
| beers | 分类 | RF | 2410 | 8 | style | FD规则、ibu大量缺失 |
| adult | 分类 | RF | 32561 | 14 | income | 大数据集、多分类列 |
| bike | 回归 | RF | 8645 | 11 | cnt | 全数值、DC规则 |
| breast_cancer | 分类 | RF | 699 | 9 | class | 小数据集 |
| har | 聚类 | KMeans | 10299 | 3 | gt | 加速度数据 |
| mercedes | 回归 | Ridge | 4209 | 16 | y | 多分类+数值混合 |
| nasa | 回归 | Ridge | 1503 | 5 | sound_pressure_level | 全FLOAT |
| smartfactory | 分类 | RF | 2000 | 10 | machine_status | 工业场景 |
| soilmoisture | 回归 | Ridge | 1500 | 8 | soil_moisture | 传感器数据 |

---

## 技术文档索引

> 完整文档见 [`docs/`](docs/README.md) 目录，共 15 章。

| 章节 | 标题 | 一句话摘要 |
|------|------|-----------|
| [01](docs/01-系统总览.md) | 系统总览 | 核心思想、10步流水线、v6配置、模块依赖 |
| [02](docs/02-数据预处理与编码.md) | 数据预处理与编码 | CSV→LE→LE+SS 编码管道，NaN保留，编辑距离容错 |
| [03](docs/03-数据划分与CleanBase.md) | 数据划分与Clean Base | Oracle三路划分、DeleteFix vs VE-Fill + √(n/N)加权 |
| [04](docs/04-规则系统.md) | 规则系统 | rules.txt 6种规则类型的格式、语义与三重用途 |
| [05](docs/05-错误检测.md) | 错误检测 | 4阶段管道、RAHA分块(20列/10000行)、FD多数投票 |
| [06](docs/06-错误注入.md) | 错误注入 | 注入=检测逆过程、FD严格<半数、RAHA-aware句法注入 |
| [07](docs/07-配置系统.md) | 配置系统 | 5种枚举、DemandPrepPolicyConfig全部字段、版本映射 |
| [08](docs/08-状态空间.md) | 状态空间 | 8维向量逐维定义：error_type/importance/distance/... |
| [09](docs/09-DQN-Agent.md) | DQN Agent | 4种变体、Dueling V+A分解、两阶段(3+2)决策 |
| [10](docs/10-清洗环境与Reward.md) | 清洗环境与Reward | 4种动作+3种降级、tanh动态修复调节器、三层修复率控制 |
| [11](docs/11-值估计链.md) | 值估计链 | 7级优先级：FD→DC→CFD→数值提取→编辑距离→KNN→Fallback |
| [12](docs/12-训练流程.md) | 训练流程 | clean_base/self_supervised、自适应注入比例、共享VE |
| [13](docs/13-推理流程.md) | 推理流程 | 两阶段Plan+Execute、用户只需提供plan中的真值 |
| [14](docs/14-评估体系.md) | 评估体系 | 6种Baseline、EDR成本效益、Shapley三维度、成本核算 |
| [15](docs/15-模型适配器.md) | 模型适配器 | 工厂模式、分类/回归/聚类统一接口、分数归一化 |

---

## CLI 参数速查

```bash
python run_demandprep_policy/run_demandprep_policy_base.py [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集名称 (beers/adult/bike/...) |
| `--versions` | v6 | 版本列表 (v3,v5,v6,v7,v8) |
| `--n_episodes` | 300 | 训练轮数 |
| `--oracle` | False | 使用 Oracle 三路划分 |
| `--verbose` | True | 详细日志 |
| `--resume` | auto | 续训模式 (auto/force_new) |
| `--all_datasets` | False | 运行全部 9 个数据集 |
| `--visualize_only` | False | 跳过训练，仅重新生成可视化 |
| `--apply_raha_truth` | True | 是否用 RAHA 标注行预修复 |
| `--count_raha_cost` | True | 是否计入 RAHA 标注成本 |

---

## 消融实验说明

v6 为主版本，其他版本为消融对照组：

```
v6 (完整版)
 ├── 去掉两阶段推理  → v5 (单阶段 Dueling)
 ├── 去掉 Dueling    → v8 (两阶段 Plain)
 ├── 两者都去掉      → v7 (单阶段 Plain)
 └── 换 Oracle 检测   → v3 (Oracle 上界)
```

| 对比 | 消融维度 | 验证假设 |
|------|---------|---------|
| v6 vs v5 | 两阶段推理 | Plan+Execute 比直接清洗更能节省真值成本 |
| v6 vs v7 | Dueling 网络 | V+A 分解加速学习收敛 |
| v6 vs v8 | Dueling 网络(保留两阶段) | 网络结构对两阶段的影响 |
| v6 vs v3 | 自动检测 | 自动检测 vs Oracle 作弊的差距 |

---

## License

MIT License
