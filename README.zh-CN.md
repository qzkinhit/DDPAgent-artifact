# DDPAgent 开源复现仓库

[English README](README.md)

本仓库包含 ADS 2026 workshop 论文所使用的代码、数据集、缓存清洗结果、论文文件和交互式演示：

**DDPAgent for Demand-Driven Data Preparation via Agentic Action Allocation and Operator-Grounded Execution**

仓库范围严格对应论文和演示。这里只包含论文中使用的数据集、baseline、缓存清洗结果和实验汇总。

## 核心思想

DDPAgent 将数据准备看作一个数据治理 agent，而不是一条固定的数据清洗流水线。给定下游任务、候选模型、预算、候选动作和可用的数据算子后，系统先进行零人工真值成本的动作分配训练。在推理阶段，策略会为每个检测到的错误或数据对象选择动作，例如不操作、修复、删除或近邻值替换。动作确定后，再进入该动作对应的算子空间，由算子编排层生成具体的数据操作和可追溯记录。

论文中的实现以数据清洗为例。修复和替换动作会调用多信号清洗算子编排层，生成真实修复规则、执行顺序和反馈权重。这个框架本身不是只能做清洗。后续可以扩展到数据增强、数据选择、是否调模型、模型调参等动作。每个动作下面也可以维护自己的算子库和编排策略。

## 真实性与溯源

仓库中没有模拟出来的 workflow trace。默认演示为了效率读取已经真实跑出的缓存结果。如果某个运行只保存了缓存清洗表，没有运行时算子 trace，界面会明确提示，并且只展示真实存在的动作 trace 和数据操作 trace。

Streamlit 页面中也提供真实运行按钮。该按钮实际调用：

```bash
python -m ads_clean.cli run ... --force-uniclean-run --trace-operators
```

也就是说，它会重新运行 UniClean 和 DDPAgent，不会伪造算子覆盖范围、修复规则数量、权重或数据操作。

## 目录结构

- `src/ads_clean`：DDPAgent 主流程，包括数据集加载、动作分配执行、trace 输出、baseline 评估和结果汇总。
- `src/demandclean`：动作分配控制器使用的强化学习实现。
- `src/SampleScrubber`、`src/AnalyticsCache`、`src/CoreSetSample`：清洗算子执行和编排相关代码。
- `data/uniclean`：Beers、Flights、Hospitals、Rayyan 和 Tax-10K 的原生错误表。
- `result_assets/UnicleanResult`：论文使用的缓存原生错误表、人工注入错误表、FullOps 输出和固定清洗 baseline 输出。
- `outputs/experiments_20260519_final` 和 `outputs/experiments_20260520_hospital_measurecode`：论文实验汇总和运行产物。
- `outputs/demo_trace_runs`：交互式演示使用的真实 trace 运行结果。
- `paper`：论文 LaTeX 源码、PDF、参考文献和生成图片。
- `streamlit_app.py`：交互式工作流可视化页面。

## 数据集和 Baseline 范围

仓库包含的数据集为 Beers、Flights、Hospitals、Rayyan 和 Tax-10K。

固定清洗 baseline 包括 Baran、BigDansing、Holistic、HoloClean 和 Horizon。论文中还使用了 No-op、FullOps、OracleDel 和 GTRepair 作为诊断控制。其中 OracleDel 和 GTRepair 需要干净参考表，不是可部署的固定清洗 baseline。

缓存的固定清洗表用于复现论文中的下游机器学习评估。本仓库不从零重新运行 Baran、BigDansing、Holistic、HoloClean 或 Horizon。

## 环境配置

推荐使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果 `torch` 或 `pyspark` 安装较慢，可以先根据本机 Python 版本安装官方 wheel，再执行上面的依赖安装命令。

## 缓存复现

最快的复现路径是使用仓库中已经包含的真实缓存结果：

```bash
bash scripts/reproduce_cached.sh
```

该命令会运行单元测试、检查 artifact 范围、验证论文汇总结果，并根据已有实验表重新生成论文图。

## 完整实验复跑

如果要重新跑 DDPAgent 实验，可以执行：

```bash
bash scripts/reproduce_full.sh
```

该脚本会在五个原生错误数据集和 40 个人工注入设置上训练动作分配器，并用相同下游任务评估固定清洗 baseline。运行时间取决于 CPU 资源。Tax 使用仓库中按主键聚类得到的 10K 子集。

## 生成带 Trace 的演示结果

为了让前端展示算子编排和数据操作溯源，可以运行：

```bash
bash scripts/run_demo_trace.sh
```

该脚本会在论文数据集上运行原生错误和人工注入错误设置，并启用运行时算子 trace。结果会写入 `outputs/demo_trace_runs`。

也可以手动运行一个小实验：

```bash
PYTHONPATH=src python -m ads_clean.cli run \
  --dataset beers \
  --scenario original \
  --output-root outputs/demo_trace_runs \
  --episodes 50 \
  --model-type-override random_forest \
  --force-uniclean-run \
  --trace-operators
```

## 启动交互式 Agent 控制台

执行：

```bash
bash scripts/run_streamlit_demo.sh
```

然后打开 Streamlit 输出的地址，通常是：

```text
http://127.0.0.1:8501
```

页面不再是静态结果浏览器，而是按照 agent 控制台组织。用户先配置数据治理任务：

1. 选择数据集
2. 选择原生错误或人工注入错误设置
3. 选择候选下游模型和当前模型
4. 选择查看缓存 artifact 或真实运行

页面默认展示已经真实跑出的缓存 artifact。在执行步骤里选择 **真实运行流程** 后，可以点击按钮启动新的真实运行。侧边栏中的全局按钮可以在中文和英文之间切换。

页面展示内容包括：

- 下游任务和模型配置
- 检测错误上的动作分配
- 存在 trace 时展示运行时算子编排
- 生成的修复规则和反馈权重
- 带验证器接受标记的数据操作记录
- 数据准备前后的下游模型收益

## 结果溯源

论文主要结果表：

- `outputs/experiments_20260519_final/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260519_final/baseline_eval/artificial/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/adsclean/adsclean_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/original/baseline_ml_summary.csv`
- `outputs/experiments_20260520_hospital_measurecode/baseline_eval/artificial/baseline_ml_summary.csv`

带 trace 的运行会输出：

- `action_trace.csv`
- `operation_trace.csv`
- `operator_trace.csv`
- `operation_rule_trace.csv`
- `operator_weight_trace.csv`
- `model_trace.csv`
- `workflow_trace.json`

如果运行使用缓存清洗表，可能没有运行时算子 trace，因为缓存表本身不保存原始算子级规则溯源。

## License

代码使用 MIT License。数据集和缓存 baseline 输出仅用于复现论文实验，原始上游数据集许可仍然适用。
