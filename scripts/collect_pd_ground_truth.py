#!/usr/bin/env python3
"""
采集 PD 分离真实 Ground Truth
用法: python scripts/collect_pd_ground_truth.py

运行命令：
    export NCCL_P2P_DISABLE=1
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /data/home/public/weight/Qwen3-32B \
    --tensor-parallel-size 2 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":5e9,"kv_port":14581,"engine_id":"my_pd_cluster"}'

    export NCCL_P2P_DISABLE=1
    CUDA_VISIBLE_DEVICES=2,3,4,5 python -m vllm.entrypoints.openai.api_server \
    --model /data/home/public/weight/Qwen3-32B \
    --tensor-parallel-size 2 \
    --port 8001 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":5e9,"kv_ip":"10.199.6.2","kv_port":14581,"engine_id":"my_pd_cluster"}'


    http_proxy="" https_proxy="" all_proxy="" curl -v -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/home/public/weight/Qwen3-32B",
        "prompt": "Hello",
        "max_tokens": 5,
        "request_id": "___decode_addr_10.199.6.2:14581_test001"
    }'

"""

import csv
import time
import json
import os
from openai import OpenAI
from datetime import datetime

# ================= 强制禁用代理（覆盖环境变量） =================
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''

# ================= 配置 =================
VLLM_URL = "http://localhost:8001"  # Decode 节点地址
TRACE_FILE = "/data/home/lihaozhe/llm-simulator/data/trace/validation_trace.csv"
OUTPUT_DIR = "/data/home/lihaozhe/llm-simulator/data/ground_truth_pd/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 读取 Trace =================
requests_data = []
with open(TRACE_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        requests_data.append({
            "arrival_time_ms": float(row["arrival_time_ms"]),
            "prompt_len": int(row["prompt_len"]),
            "output_len": int(row["output_len"])
        })

print(f"📋 加载了 {len(requests_data)} 个请求")
print(f"🔗 连接 vLLM: {VLLM_URL}")

# ================= 初始化 Client =================
client = OpenAI(
    base_url=f"{VLLM_URL}/v1",
    api_key="EMPTY",  # vLLM 默认不需要 API Key
    timeout=600.0
)

# ================= 发送请求，逐条记录 =================
results = []
start_time = time.time()

# Prefill 节点的 IP 和端口（与你的实际配置一致）
PREFILL_IP = "10.199.6.2"
PREFILL_PORT = 8000

# ===== 新增：Decode 节点的 IP 和端口（就是你 curl 里的 14581，但注意是 NCCL 通信端口） =====
# 注意：这里的 DECODE_PORT 是 vLLM 启动命令里 --kv-transfer-config 中的 kv_port（即 14581），
# 而不是 HTTP 服务端口 8001。NCCL 通信走的是这个端口。
DECODE_IP = "10.199.6.2"    # 如果你 Consumer 和 Producer 在同一台机器，填同一个 IP
DECODE_KV_PORT = 14581      # 这个端口在 Producer 的 curl 命令里出现过（对应 decode_addr）

for idx, req in enumerate(requests_data):
    prompt = "请用一句话回答：" + "假设你是一个AI助手。" * (req["prompt_len"] // 10)  # 凑够 token 数
    
    # 记录发送时间（模拟 arrival_time 对齐）
    send_time = start_time + req["arrival_time_ms"] / 1000.0
    current_time = time.time()
    if current_time < send_time:
        time.sleep(send_time - current_time)
    
    request_start = time.time()
    
    try:
        # ===== 修复：使用完整格式的 request_id =====
        # 格式: cmpl-___prefill_addr_IP:PORT___decode_addr_IP:PORT___suffix
        custom_request_id = (
            f"cmpl-"
            f"___prefill_addr_{PREFILL_IP}:{PREFILL_PORT}___"
            f"decode_addr_{DECODE_IP}:{DECODE_KV_PORT}___"
            f"req_{idx+1}"
        )
        
        stream = client.chat.completions.create(
            model="/data/home/public/weight/Qwen3-32B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=req["output_len"],
            temperature=0.0,
            stream=True,
            extra_body={"request_id": custom_request_id}
        )
        
        ttft = None
        first_token_time = None
        tokens_received = 0
        last_token_time = None
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = (first_token_time - request_start) * 1000  # 转 ms
                tokens_received += 1
                last_token_time = time.time()
        
        end_time = time.time()
        e2e_ms = (end_time - request_start) * 1000
        tpot_ms = (e2e_ms - ttft) / tokens_received if tokens_received > 0 else 0
        
        results.append({
            "request_id": custom_request_id,
            "arrival_time_ms": req["arrival_time_ms"],
            "ttft_ms": round(ttft or 0, 2),
            "tpot_ms": round(tpot_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
            "status": "success"
        })
        
        print(f"✅ req_{idx+1}: TTFT={ttft:.1f}ms, TPOT={tpot_ms:.1f}ms")
        
    except Exception as e:
        print(f"❌ req_{idx+1} 失败: {e}")
        results.append({
            "request_id": f"req_{idx+1}",
            "arrival_time_ms": req["arrival_time_ms"],
            "ttft_ms": 0,
            "tpot_ms": 0,
            "e2e_ms": 0,
            "status": f"failed: {e}"
        })

# ================= 保存结果 =================
output_file = os.path.join(OUTPUT_DIR, f"pd_ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# 同时保存 CSV 方便对比
csv_file = os.path.join(OUTPUT_DIR, "pd_ground_truth_latest.csv")
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["request_id", "arrival_time_ms", "ttft_ms", "tpot_ms", "e2e_ms", "status"])
    for r in results:
        writer.writerow([r["request_id"], r["arrival_time_ms"], r["ttft_ms"], r["tpot_ms"], r["e2e_ms"], r["status"]])

print(f"\n💾 结果已保存至: {output_file}")
print(f"💾 CSV 已保存至: {csv_file}")

# ================= 统计摘要 =================
success_results = [r for r in results if r["status"] == "success"]
if success_results:
    ttfts = [r["ttft_ms"] for r in success_results]
    tpots = [r["tpot_ms"] for r in success_results]
    print("\n" + "="*60)
    print("📊 Ground Truth 统计摘要")
    print("="*60)
    print(f"成功请求: {len(success_results)}/{len(results)}")
    print(f"TTFT: avg={sum(ttfts)/len(ttfts):.1f}ms, p50={sorted(ttfts)[len(ttfts)//2]:.1f}ms")
    print(f"TPOT: avg={sum(tpots)/len(tpots):.1f}ms, p50={sorted(tpots)[len(tpots)//2]:.1f}ms")
else:
    print("\n❌ 没有成功请求，请检查 PD 服务状态。")