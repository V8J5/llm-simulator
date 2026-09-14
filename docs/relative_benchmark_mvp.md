# 相对性能 Benchmark MVP（协议 v0.1）

## 本次完成的闭环

页面现在同时呈现两类信息：

1. Qwen3-32B 的版本化推荐 P/D 负载；
2. 同模型、同 dtype、同 TP、同 Batch、同长度的硬件相对性能。

相对评分不再混合 Prefill 和 Decode：

- P-Score：所有可比较 Prefill case 吞吐倍率的几何平均；
- D-Score：所有可比较 Decode case 吞吐倍率的几何平均；
- 单点倍率：`候选硬件吞吐 / 基准硬件吞吐`。

倍率大于 1 表示候选硬件更快。比如 1.20× 表示快约 20%。

## 公平性规则

主榜使用 `iso_workload`：模型、dtype、TP、Batch、Seq/KV 长度和协议必须一致。某张卡 OOM 时，该 case 记为不可行，不能自动降低 Batch 后继续进入主榜。

硬件各自选择最大可行 Batch 属于 `best_feasible`，用于容量和性价比分析，但不能与固定负载排名混在一起。

## 可信状态

- `verified`：每个配对 case 都有独立的真实 A/B 端到端倍率；
- `provisional`：负载可以公平配对，但还没有独立真实 A/B 倍率；
- `invalid`：模型、TP、协议不一致，或没有相同且可行的 case。

当前旧页面报告会在内存中修正历史吞吐公式，并标记为 `legacy_v0_adapted_in_memory`。历史文件不会被覆盖，结果仍是 `provisional`。

## 推荐负载生成

对新模型，先读取本地 Hugging Face `config.json`：

```bash
python scripts/recommend_benchmark_workload.py \
  --model-config /path/to/model/config.json \
  --model-id organization/model-name \
  --dtype bfloat16 \
  --output configs/benchmark_specs/model_name_v1.json
```

规则会自动识别 MHA、MQA、GQA，读取层数、隐藏维度、KV 头数和上下文上限。遇到线性注意力、混合架构、MoE 或短上下文裁剪，会设置 `review_required=true`，要求人工确认。LLM/Agent 的角色是解释和提出候选调整，不允许静默修改已经用于评分的负载。

## 启动页面

在服务器项目环境中：

```bash
python web/app.py
```

浏览 `http://localhost:5000`。页面顶部会显示推荐负载；选择相同 TP 的基准和候选硬件后点击“比较”。

## 命令行相对比较

协议 v0.1 原生报告可以直接比较：

```bash
python scripts/generate_relative_benchmark.py \
  --reference data/benchmark_reports/reference.json \
  --candidate data/benchmark_reports/candidate.json \
  --output-dir data/relative_benchmark/candidate_vs_reference
```

如果已采集独立真实 A/B 倍率，可提供一个 JSON，例如：

```json
{"P_S": 1.18, "P_M": 1.21, "P_L": 1.20, "P_XL": 1.19,
 "D_S": 1.16, "D_M": 1.17, "D_L": 1.15, "D_XL": 1.14}
```

然后追加：

```bash
--ground-truth-ratios data/ground_truth_ratios.json
```

只有覆盖全部有效配对 case 后，状态才会升级为 `verified`，并逐点输出相对误差。

## 已修复的问题

- Prefill 吞吐由 `seq/time` 修正为 `batch*seq/time`；
- Decode 吞吐由 `1/time` 修正为 `batch/time`；
- 不再把 Prefill 与 Decode 的绝对吞吐各 50% 相加作为科学排名；旧综合评分仅保留兼容展示；
- 报告选择由不确定的第一个文件改为最新文件；
- 推荐负载成为单一版本化 JSON 来源，评分脚本和页面共用。

## 仍然存在的关键限制

当前 `prefill_layer_bench.py`、`decode_layer_bench.py` 仍是合成 PyTorch 代理实现，不是忠实的 vLLM/Qwen kernel 路径；因此现阶段相对倍率只能作为工程候选结果，不能宣称已可信复现真实硬件差距。下一步最重要的验证是：选两种硬件，在相同 TP 与相同 case 上采集真实 P/D 端到端结果，检查排序和倍率误差。

真实 A/B 采集与验证已经由 `scripts/collect_relative_ground_truth.py` 和 `scripts/compare_relative_ground_truth.py` 支持，完整命令见 `docs/relative_ground_truth_workflow.md`。
