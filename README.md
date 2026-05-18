# OpenRCA Meta Controller

本仓库包含一个用于 OpenRCA 数据集的 “Meta-Controller + 策略池” 评测实现，入口在 `openrca_meta_controller/main.py`。

## 主要功能

- 从 OpenRCA 数据集加载 metrics / logs / traces
- 运行反事实引擎生成候选根因（candidate）
- 由 Meta-Controller 在多种策略之间进行选择与迭代，逐步收集证据并输出预测
- 对预测结果进行打分与汇总

## 目录结构

- `main.py`：命令行入口与批量评测流程
- `config.py`：全局参数、数据类定义、数据集路径配置
- `data/`：数据加载与统一化
- `counterfactual/`：初始反事实分析
- `strategies/`：策略池（A/B/C/D）
- `meta_controller/`：Meta-Controller（包含情绪控制器与 LLM 控制器）
- `evaluation/`：评测与聚合
- `utils/`：工具（含 LLM 客户端）

## 运行方式

在仓库目录下运行：

```bash
python -m openrca_meta_controller.main --option B --max-queries 5
```

常用参数示例：

```bash
python -m openrca_meta_controller.main --option B --systems Bank
python -m openrca_meta_controller.main --option C --systems Bank --max-queries 5
```

其中：

- `--option B`：使用情绪驱动的 Meta-Controller（不依赖外部 LLM）
- `--option C`：使用 LLM Meta-Controller（需要配置 API Key）

## 数据集路径

默认 OpenRCA 数据根目录在 `config.py` 的 `OPENRCA_ROOT`：

- `OPENRCA_ROOT = "/home/dell2/RCA513/yyx/OpenRCA"`

如果你的数据路径不同，请修改该值或将其改为环境变量/配置项再运行。

## LLM 配置（仅 option C）

`utils/llm_client.py` 支持：

- DeepSeek（OpenAI 兼容接口）：设置 `DEEPSEEK_API_KEY`
- Claude（Anthropic）：设置 `ANTHROPIC_API_KEY`（且需要 `anthropic` Python 包）

例如：

```bash
export DEEPSEEK_API_KEY="YOUR_KEY"
python -m openrca_meta_controller.main --option C --systems Bank
```

