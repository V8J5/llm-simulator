# 跨硬件相对性能真实验证流程

目标：在完全相同的 Qwen3-32B、dtype、TP 和 P/D Case 下，分别采集两种硬件的真实 vLLM fixed-batch 数据，然后验证平台是否保持正确排序，以及预测倍率与真实倍率的差距。

## 1. 两台机器使用相同运行协议

本轮以 TP4 为例。两台机器的 vLLM 启动参数应保持一致：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/home/public/weight/Qwen3-32B \
  --tensor-parallel-size 4 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --enforce-eager
```

每台机器都要准备对应的 deployment JSON。可以复制 `configs/fixed_batch_tp4.json`，修改模型本地路径和 `cuda_visible_devices`；其余调度参数应保持一致。

确认服务：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models
```

## 2. 在基准硬件 L20 上采集

```bash
python scripts/collect_relative_ground_truth.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/home/public/weight/Qwen3-32B \
  --model-id Qwen/Qwen3-32B \
  --tokenizer /data/home/public/weight/Qwen3-32B \
  --hardware-id L20 \
  --tp-size 4 \
  --dtype bfloat16 \
  --benchmark-spec configs/benchmark_specs/qwen3_32b_v1.json \
  --deployment-config configs/fixed_batch_tp4.json \
  --warmup 1 \
  --repeats 3 \
  --resume \
  --output-dir data/relative_ground_truth/runs/L20/tp4
```

脚本会采集协议中的 4 个 Prefill 和 4 个 Decode case。每个 case 首先预热一次，再记录三次有效重复。

## 3. 在候选硬件上采集

以 RTX5090 为例，在 RTX5090 所在机器运行：

```bash
python scripts/collect_relative_ground_truth.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3-32B \
  --model-id Qwen/Qwen3-32B \
  --tokenizer /path/to/Qwen3-32B \
  --hardware-id RTX5090 \
  --tp-size 4 \
  --dtype bfloat16 \
  --benchmark-spec configs/benchmark_specs/qwen3_32b_v1.json \
  --deployment-config configs/fixed_batch_tp4_rtx5090.json \
  --warmup 1 \
  --repeats 3 \
  --resume \
  --output-dir data/relative_ground_truth/runs/RTX5090/tp4
```

`--model` 可以是不同的本地路径；`--model-id` 必须相同，用它表示两台机器运行的是同一个逻辑模型。

## 4. 汇总到同一台机器

把两台机器的整个运行目录复制到负责分析的项目中：

```text
data/relative_ground_truth/runs/L20/tp4/
data/relative_ground_truth/runs/RTX5090/tp4/
```

必须复制 `manifest.json` 和旁边全部 `prefill_*.json`、`decode_*.json`。即使 manifest 中保存的是原机器绝对路径，分析脚本也会回退到 manifest 同目录查找文件。

## 5. 验证已有平台预测

将下面两个报告路径替换成实际最新文件：

```bash
python scripts/compare_relative_ground_truth.py \
  --reference-manifest data/relative_ground_truth/runs/L20/tp4/manifest.json \
  --candidate-manifest data/relative_ground_truth/runs/RTX5090/tp4/manifest.json \
  --benchmark-spec configs/benchmark_specs/qwen3_32b_v1.json \
  --reference-report data/benchmark_reports/benchmark_L20_TP4_YYYYMMDD_HHMMSS.json \
  --candidate-report data/benchmark_reports/benchmark_RTX5090_TP4_YYYYMMDD_HHMMSS.json \
  --relative-mape-threshold-pct 15 \
  --max-cv-pct 10 \
  --output-dir data/relative_ground_truth/comparisons/RTX5090_vs_L20_tp4
```

输出包括：

- `relative_ground_truth_report.json`：完整身份核验、真实倍率、预测误差和验收结论；
- `ground_truth_ratios.json`：供其他评分命令使用的逐 case 真实倍率；
- `relative_ground_truth_cases.csv`：适合汇报展示的明细表。

默认验收条件：

1. 两套数据完整覆盖相同的 8 个 Case；
2. vLLM 调度配置一致；
3. 每个 Case 两端重复测量 CV 均不超过 10%；
4. 所有 Case 均不排反，排序一致率为 100%；
5. 8 个 Case 的相对倍率 MAPE 不超过 15%。

## 6. 页面查看

```bash
python web/app.py
```

打开 `http://localhost:5000`，选择同 TP 的 L20 和 RTX5090。后端会自动寻找 `data/relative_ground_truth/**/relative_ground_truth_report.json`：找到完整真实数据后，状态显示 `verified`，并展示真实 P/D 倍率、阶段误差、逐点 MAPE 和排序一致率。

## 注意

- 命令中的 URL 必须是纯文本，不要复制成 `[http://...](http://...)`。
- 如果某个 case 输出 `valid=False`，不要手工修改 JSON；应先检查失败请求或客户端启动偏斜，再重采该点。
- 两端不能一边开启 prefix caching、另一边关闭，也不能使用不同的 TP 或不同调度上限。
- `verified` 表示已经拿真实 A/B 数据完成校验；是否达到可信目标仍以 `acceptance.passed` 为准。
