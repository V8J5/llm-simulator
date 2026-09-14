# llm-simulator

一个由小规模 profiling 数据驱动的 LLM 在线推理离散事件仿真器。当前主线聚焦
Dense FP16/BF16、单节点 TP、continuous batching；PD 分离作为第二条实验路径。

## 当前执行链

```text
model / hardware / TP / workload
              │
              ▼
 operator + collective profiles ──► profile-driven CostModel
              │                              │
              ▼                              ▼
      memory/KV constraints ◄── scheduler creates current batch shape
                                             │
                                             ▼
                              discrete-event simulation
                                             │
                                             ▼
                       TTFT / TPOT / E2E / throughput / SLA goodput
```

成本模型不会假装自己是指令级模拟器。测量网格内采用 log-bilinear 插值；网格外采用
有界的边缘斜率外推，并在结果的 `extrapolated` 和 `warnings` 字段中显式标记。
端到端补偿只有在提供真实校准 JSON 后才会生效，不再内置某张卡或某个 TP 的经验倍率。

## 快速验证

在项目根目录运行：

```bash
python -m unittest discover -s tests -v
python -m core.qwen3_cost_model
python -m core.simulator
```

核心 Python 仿真不依赖 GPU。profiling 与 ground-truth 采集需要对应 CUDA/CANN、通信库、
模型权重和推理运行时。

## 可信的 vLLM ground truth

先启动 OpenAI-compatible vLLM 服务，再运行：

```bash
python scripts/collect_vllm_ground_truth.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/model \
  --tokenizer /path/to/model \
  --trace data/trace/validation_trace.csv \
  --output data/ground_truth_co_located/vllm_validation.json
```

该脚本按 trace 的绝对时间并发启动请求，TTFT 来自流式首包，token 数来自 vLLM usage，
TPOT 使用 `(最后 token 时间 - 首 token 时间) / (输出 token 数 - 1)`。请勿再使用串行
循环或把 HTTP 文本 chunk 个数当作 token 个数。

## Profiling 数据约定

当前兼容已有三个表：

- Prefill：`batch_size`, `prompt_length`, `ffn_gemm_ms`, `attention_prefill_ms`
- Decode：`batch_size`, `kv_length`, `ffn_gemv_ms`, `attention_decode_ms`
- Collective：`world_size`, `msg_size_mb`, `allreduce_ms`

不同后端（CUDA/NCCL、CANN/HCCL）应导出相同字段，并把设备、runtime/driver、dtype、
模型、TP、warmup、重复次数、均值/分位数作为伴随 metadata 保存。成本模型只消费统一
schema，不直接依赖某个运行时。

可选端到端校准文件示例：

```json
{
  "prefill": {"1": 1.08, "2": 1.03},
  "decode": {"1": 1.11, "2": 1.05}
}
```

通过 `LLMCostModel(..., calibration_file=..., enable_e2e_compensation=True)` 使用。系数必须
来自独立校准集；验证集不可参与拟合。

## 当前边界与下一步

1. 先用 1/2/4/8 卡 holdout 数据验证 fixed-batch prediction；如果误差不达标，不应做大规模配置外推。
2. 在线仿真已经按 query-token budget、并发数和增量 KV block 闭环调度，并支持 chunked prefill；尚未实现抢占和 prefix cache。
3. PD 路径具备独立队列、TP replica、KV 池和有限并发传输链路；下一步是用独立 PD 实测校准传输与两侧运行时开销。
4. PP、MoE/EP、offload 应在真实依赖图和独立 profiling 数据就绪后逐项加入，不能用单一倍率近似。
5. 跨硬件结论必须分别采集 CUDA/NCCL 或 CANN/HCCL 数据；roofline 只能作为范围外 sanity check。

## SLA 容量与配置搜索

完成单点校准和 holdout 验证后，可以在同一条 co-located 仿真链路上扫描请求率与
服务参数。容量点要求每次重复实验都满足完成率门槛，并按“满足 TTFT 和 TPOT 的请求数 /
全部到达请求数”计算 SLA 成功率，避免只统计已完成请求造成幸存者偏差。

```bash
python scripts/run_capacity_search.py \
  --arrival-rates-rps 0.5 1 2 4 8 \
  --refine-iterations 3 \
  --tp-sizes 4 \
  --dp-sizes 1 2 4 \
  --num-gpu-blocks 22136 \
  --max-num-batched-tokens 4096 8192 \
  --max-num-seqs 16 32 \
  --chunked-prefill on \
  --mode poisson \
  --prompt-len 512 \
  --output-len 128 \
  --num-requests 200 \
  --repeats 3 \
  --ttft-sla-ms 500 \
  --tpot-sla-ms 50 \
  --min-sla-success-ratio 0.9 \
  --enable-calibration \
  --calibration-file configs/calibration_tp4_chunked.json \
  --output-dir data/capacity_search/tp4_baseline
```

如需长度分布而不是固定 shape，可用 `--prompt-len-range 128 2048` 和
`--output-len-range 32 256`；每个重复实验使用独立但可复现的随机种子。

`--dp-sizes` 表示独立推理副本数。全局请求流按 round-robin 分发，每个副本拥有独立的
continuous-batching 调度器和 KV block 池；推理 DP 不引入训练式 AllReduce。输出同时提供
总容量排名和按 GPU 归一化的效率排名。

输出包括逐点 CSV、配置容量排名 CSV、GPU 效率排名 CSV 和保留全部重复实验的 JSON。报告同时记录 profile
外推 batch 比例、峰值 KV block 利用率和估计瓶颈；如果 TP 缺少通信 profiling，默认拒绝
扫描，只有显式传入 `--allow-unprofiled-tp` 才会继续。若所有测试请求率均满足 SLA，容量
会标记为 `capacity_censored=true`，表示当前只获得容量下界，应继续提高请求率而不能把该点
误认为真正的容量上限。若粗粒度请求率中同时存在通过点和失败点，默认执行三轮二分细化，
并输出 `capacity_lower_bound_rps`、`capacity_upper_bound_rps` 和区间宽度。

## PD 分离仿真

PD 模拟器将 Prefill 和 Decode 建模为可并行工作的独立设备池。每侧 GPU 数必须能被该侧
TP 整除，商为独立 replica 数；请求完成 Prefill 后产生 KV 传输事件，传输完成后才能进入
Decode。`kv_transfer_concurrency` 个传输 lane 是显式共享资源，因此传输会排队。

```bash
python scripts/run_pd_simulation.py \
  --prefill-gpus 4 --prefill-tp 4 --prefill-num-gpu-blocks 22136 \
  --decode-gpus 4 --decode-tp 4 --decode-num-gpu-blocks 22136 \
  --prefill-max-num-batched-tokens 4096 --prefill-max-num-seqs 16 \
  --decode-max-num-batched-tokens 4096 --decode-max-num-seqs 32 \
  --kv-transfer-mode pcie --kv-transfer-bw-gb-s 12.5 \
  --kv-transfer-latency-ms 0.1 --kv-transfer-concurrency 1 \
  --mode poisson --arrival-rate-rps 1 --prompt-len 512 --output-len 128 \
  --num-requests 200 --enable-calibration \
  --calibration-file configs/calibration_tp4_chunked.json \
  --output-dir data/pd_simulation/p4_d4_tp4
```

输出包含汇总报告、逐请求结果、P/D batch 日志和 KV 传输日志，便于分别诊断计算、排队、
KV 容量和传输链路。

### PD 容量与资源配比搜索

`pd_config_scan.py` 在同一个 PD 仿真闭环上扫描物理 GPU 划分、两侧 TP/replica 数、
P/D 调度参数以及 KV 传输链路。容量仍采用所有到达请求为分母，并要求每次重复实验都达到
完成率门槛；报告同时给出吞吐容量排名和按总 GPU 数归一化的效率排名。

下面的第一轮实验固定 TP=4，对 8、12、16 张总 GPU 的几种 P/D 配比进行比较：

```bash
python scripts/pd_config_scan.py \
  --gpu-splits 4:4 4:8 8:4 8:8 \
  --prefill-tp-sizes 4 --decode-tp-sizes 4 \
  --prefill-num-gpu-blocks 22136 --decode-num-gpu-blocks 22136 \
  --prefill-max-num-batched-tokens 4096 \
  --prefill-max-num-seqs 8 16 \
  --decode-max-num-batched-tokens 4096 \
  --decode-max-num-seqs 16 32 \
  --chunked-prefill on \
  --kv-transfer-modes pcie --kv-transfer-bandwidths-gb-s 12.5 \
  --kv-transfer-latency-ms 0.1 --kv-transfer-concurrencies 1 \
  --arrival-rates-rps 0.5 1 2 4 8 12 16 \
  --refine-iterations 3 \
  --mode poisson --prompt-len 512 --output-len 128 \
  --num-requests 200 --repeats 3 \
  --ttft-sla-ms 500 --tpot-sla-ms 50 \
  --min-completion-ratio 1 --min-sla-success-ratio 0.9 \
  --enable-calibration \
  --calibration-file configs/calibration_tp4_chunked.json \
  --output-dir data/pd_capacity_search/tp4_ratio_search
```

`P:D` 表示两侧的物理 GPU 数，必须分别能被对应 TP 整除；商是该侧的独立 replica
数。输出包括 `pd_capacity_points.csv`、`pd_capacity_ranking.csv`、
`pd_efficiency_ranking.csv`、完整 JSON，以及被 TP 整除规则拒绝的配置。瓶颈分类为
`prefill_compute`、`decode_compute`、`kv_transfer`、`prefill_memory` 或
`decode_memory`，并附带 P/D 计算利用率、传输利用率、传输排队、两侧峰值 KV 利用率和
profile 外推比例。报告还记录 P/D batch size 的均值、P95、最大值和各 replica 的最小/
最大利用率，用来识别多副本导致的组批稀释或负载不均衡。

不同总 GPU 数可以用效率排名比较，但确定部署配比时还应在固定 GPU 预算内比较，例如总计
12 卡只比较 `4:8` 与 `8:4`。如果 `capacity_censored=true`，该结果只是容量下界，需把
请求率网格继续向上扩展；如果 `non_monotonic_points=true`，应增加请求数或重复次数后复核。
当前 PCIe 带宽和时延仍是配置假设，形成最终硬件结论前需要用独立 PD 实测替换。

### 真实 PD Ground Truth

真实 PD 数据必须发送到 Prefill/Decode 前面的代理入口，不能直接请求 Decode worker。
当前项目的八卡基线严格对应 `4P+4D、TP4+TP4`。项目内代理遵循 vLLM 0.19.0 官方
P2pNcclConnector 参考代理的请求协议，因此直接启动三个组件：

```bash
pip install -r requirements-ground-truth.txt
bash scripts/start_vllm_pd_tp4.sh
```

启动脚本固定使用 GPU `0,1,2,3` 作为 Prefill、`4,5,6,7` 作为 Decode；worker HTTP
端口为 8100/8200，代理端口为 8000，日志写入
`data/ground_truth_pd/runtime_logs/tp4_1p1d`。脚本保持在前台，按 Ctrl+C 只终止它启动的
三个进程，不会使用 `pkill` 影响其他用户任务。若实际 GPU 编号或端口不同，使用脚本开头
列出的环境变量覆盖，并同步修改部署清单。TP4 下 Prefill 默认占用 KV 端口
`14579–14582`，Decode 占用 `14589–14592`；启动脚本会拒绝重叠的端口区间。

先检查整个链路而不是直接运行正式 trace：

```bash
cp configs/pd_ground_truth_tp4_template.json configs/pd_ground_truth_tp4.json
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
curl --noproxy '*' -fsS http://127.0.0.1:8100/metrics >/dev/null
curl --noproxy '*' -fsS http://127.0.0.1:8200/metrics >/dev/null
python scripts/collect_pd_ground_truth.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/home/public/weight/Qwen3-32B \
  --tokenizer /data/home/public/weight/Qwen3-32B \
  --trace data/trace/pd_smoke.csv \
  --deployment-config configs/pd_ground_truth_tp4.json \
  --component-metrics-url prefill=http://127.0.0.1:8100/metrics \
  --component-metrics-url decode=http://127.0.0.1:8200/metrics \
  --repeat-id smoke \
  --output data/ground_truth_pd/tp4_1p1d/smoke.json
```

只有终端显示 `0 failed` 时，才进入正式三次采集。如果失败，优先发送 `smoke.json` 和
`runtime_logs/tp4_1p1d` 下三个日志，而不是反复重启服务。

先复制并按实际启动参数修改 `configs/pd_ground_truth_tp4_template.json`，确保 GPU、TP、
调度上限、connector、dtype 和 vLLM 版本均与实际服务一致。然后用同一 trace 独立采集三次：

```bash
python scripts/collect_pd_ground_truth.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/home/public/weight/Qwen3-32B \
  --tokenizer /data/home/public/weight/Qwen3-32B \
  --trace data/trace/validation_trace.csv \
  --deployment-config configs/pd_ground_truth_tp4.json \
  --component-metrics-url prefill=http://127.0.0.1:8100/metrics \
  --component-metrics-url decode=http://127.0.0.1:8200/metrics \
  --repeat-id repeat_1 \
  --output data/ground_truth_pd/tp4_1p1d/repeat_1.json
```

将 `repeat_1` 和输出文件名依次改为 2、3。每次调用都会先做独立 warmup；正式请求按
trace 的绝对到达时间并发发送。采集器使用精确 prompt token ids，并根据服务端 usage
计算输出 token 数；若请求失败、没有返回 token 或 token 数不符合 trace，仍会保存诊断
文件，但进程以非零状态退出，该次数据不可进入聚合。

```bash
python scripts/aggregate_vllm_runs.py \
  data/ground_truth_pd/tp4_1p1d/repeat_1.json \
  data/ground_truth_pd/tp4_1p1d/repeat_2.json \
  data/ground_truth_pd/tp4_1p1d/repeat_3.json \
  --output-json data/ground_truth_pd/tp4_1p1d/aggregate.json \
  --output-csv data/ground_truth_pd/tp4_1p1d/aggregate.csv
```

三个文件的 deployment manifest 哈希必须相同，避免把不同 P/D 启动配置误聚合为重复实验。
组件 `/metrics` 不可访问时，客户端延迟仍可采集，但必须在结论中注明缺少两侧运行时证据。

聚合完成后，用同一 trace 验证 PD 仿真。P/D 的 TP、replica、KV block 和调度上限会从
真实数据内嵌的 deployment manifest 自动读取，避免手工参数与真实部署不一致：

```bash
python scripts/run_pd_validation.py \
  --ground-truth data/ground_truth_pd/tp4_1p1d/aggregate.json \
  --trace data/trace/validation_trace.csv \
  --enable-calibration \
  --calibration-file configs/calibration_tp4_chunked.json \
  --output-dir data/validation/pd_tp4_1p1d
```

如果 deployment manifest 尚未填写实测 NCCL 带宽和固定延迟，验证器会显式标记假设值，
默认使用 12.5 GB/s 和 0.1 ms。此时结果可用于首轮误差诊断，但不能声称 KV 传输模型已经
经过真实带宽校准。

共置校准参数不能默认迁移到 PD。需要保持同一 P/D 服务运行，用独立 calibration traces
采集三次真实 PD 数据：

```bash
python scripts/collect_pd_calibration_runs.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/home/public/weight/Qwen3-32B \
  --tokenizer /data/home/public/weight/Qwen3-32B \
  --deployment-config configs/pd_ground_truth_tp4.json \
  --component-metrics-url prefill=http://127.0.0.1:8100/metrics \
  --component-metrics-url decode=http://127.0.0.1:8200/metrics
```

随后只使用这些独立数据拟合 PD 专用阶段参数：

```bash
python scripts/fit_pd_calibration.py \
  --output-config configs/calibration_pd_tp4_1p1d.json \
  --output-report data/calibration/pd_tp4_1p1d/fit_report.json
```

冻结配置后，重新运行 `run_pd_validation.py`，将 calibration file 改为
`configs/calibration_pd_tp4_1p1d.json`。拟合器会拒绝 validation/holdout trace，并要求所有
输入具有完全相同的 deployment manifest 哈希。

## 重要代码

- `core/schema.py`：硬件无关 schema
- `core/qwen3_cost_model.py`：profiling 插值、通信和校准
- `core/memory_estimator.py`：权重、KV、workspace、runtime/graph/通信缓冲记账
- `core/kv_block_pool.py`：PagedAttention 风格的增量 block 生命周期
- `core/simulator.py`：continuous batching + DES + 指标
- `core/pd_simulator.py`：prefill/decode 池与 KV 传输
- `core/pd_experiment_runner.py`：PD 配置网格、SLA 容量和瓶颈排名
- `scripts/collect_vllm_ground_truth.py`：并发 vLLM 验证数据采集
- `scripts/collect_pd_ground_truth.py`：PD 代理端到端采集和 P/D 指标采样
- `tests/test_core.py`：关键不变量回归测试
