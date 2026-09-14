# TP4 端到端校准与独立验证

本流程只拟合 Decode 的 TP4 端到端缩放因子。原来的
`data/trace/validation_trace.csv` 是最终留出验证集，严禁参与拟合。

## 1. 保持 vLLM 配置固定

继续使用已记录的 Qwen3-32B、4×L20、TP=4、BF16、block size=16、
`num_gpu_blocks=22136`、`max_num_batched_tokens=8192`、
`max_num_seqs=32`、Eager、关闭 prefix caching 的配置。

## 2. 一次性采集三类校准负载

采集脚本会依次运行 isolated、medium、saturated 三条 trace，每条重复
3 次。底层 collector 在每次正式采集前都执行 tokenizer、API 和模型预热，
因此生成的三份文件都是正式样本，不需要再丢弃第一份。

```bash
python scripts/collect_calibration_runs.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/home/public/weight/Qwen3-32B \
  --tokenizer /data/home/public/weight/Qwen3-32B
```

若中途失败，修复后直接重跑即可：已存在且通过完整性检查的文件会复用。
只有需要主动覆盖旧结果时才加 `--overwrite`。

## 3. 只用校准集拟合

```bash
python scripts/fit_calibration.py \
  --num-gpu-blocks 22136 \
  --block-size-tokens 16 \
  --tp-size 4 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32
```

输出为 `configs/calibration_tp4.json` 和
`data/calibration/tp4/fit_report.json`。拟合目标是三类负载上按请求数加权的
TPOT MAPE；Prefill 因子固定为 1.0。

## 4. 最后只运行一次留出验证

```bash
python scripts/run_validation.py \
  --num-gpu-blocks 22136 \
  --block-size-tokens 16 \
  --tp-size 4 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --calibration-file configs/calibration_tp4.json \
  --enable-calibration \
  --output-dir data/validation/tp4_calibrated
```

重点对比 baseline 与 calibrated 的 TPOT、E2E、吞吐率误差。TTFT 的逐请求
MAPE 还会受模拟器批次边界与真实 vLLM 调度差异影响，因此同时观察 TTFT
均值和 P50/P90，而不要用单个比例强行修正 Prefill。
