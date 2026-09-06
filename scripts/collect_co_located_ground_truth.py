#!/usr/bin/env python3
"""
采集混部（Co-located）推理的 Ground Truth
用法:
    # 1. 先清理可能残留的进程
    pkill -f vllm
    pkill -f EngineCore

    # 2. 设置环境变量（禁用代理 + NCCL 优化）
    export http_proxy=""
    export https_proxy=""
    export HTTP_PROXY=""
    export HTTPS_PROXY=""
    export all_proxy=""
    export ALL_PROXY=""
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1

    # 3. 启动混部 vLLM 服务（TP=4，使用 4 张 L20 GPU）
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model /data/home/public/weight/Qwen3-32B \
        --tensor-parallel-size 4 \
        --port 8000 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --enforce-eager

    python scripts/collect_co_located_ground_truth.py
"""

import csv
import time
import json
import os
from openai import OpenAI
from datetime import datetime

# 禁用代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''

VLLM_URL = "http://localhost:8000"  # 混部服务地址
# TRACE_FILE = "/data/home/lihaozhe/llm-simulator/data/trace/validation_trace.csv"
TRACE_FILE = "/data/home/lihaozhe/llm-simulator/data/trace/trace_random.csv"
OUTPUT_DIR = "/data/home/lihaozhe/llm-simulator/data/ground_truth_co_located/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 trace
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

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="EMPTY", timeout=600.0)

results = []
start_time = time.time()

for idx, req in enumerate(requests_data):
    prompt = "请用一句话回答：" + "假设你是一个AI助手。" * (req["prompt_len"] // 10)
    
    send_time = start_time + req["arrival_time_ms"] / 1000.0
    current_time = time.time()
    if current_time < send_time:
        time.sleep(send_time - current_time)
    
    request_start = time.time()
    
    try:
        # 混部不需要特殊 request_id，直接用普通 ID
        stream = client.chat.completions.create(
            model="/data/home/public/weight/Qwen3-32B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=req["output_len"],
            temperature=0.0,
            stream=True,
            extra_body={"request_id": f"req_{idx+1}"}
        )
        
        ttft = None
        first_token_time = None
        tokens_received = 0
        last_token_time = None
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = (first_token_time - request_start) * 1000
                tokens_received += 1
                last_token_time = time.time()
        
        end_time = time.time()
        e2e_ms = (end_time - request_start) * 1000
        tpot_ms = (e2e_ms - ttft) / tokens_received if tokens_received > 0 else 0
        
        results.append({
            "request_id": f"req_{idx+1}",
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

# 保存结果
output_file = os.path.join(OUTPUT_DIR, f"co_located_ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

csv_file = os.path.join(OUTPUT_DIR, "co_located_ground_truth_latest.csv")
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["request_id", "arrival_time_ms", "ttft_ms", "tpot_ms", "e2e_ms", "status"])
    for r in results:
        writer.writerow([r["request_id"], r["arrival_time_ms"], r["ttft_ms"], r["tpot_ms"], r["e2e_ms"], r["status"]])

print(f"\n💾 结果已保存至: {output_file}")
print(f"💾 CSV 已保存至: {csv_file}")

# 统计摘要
success_results = [r for r in results if r["status"] == "success"]
if success_results:
    ttfts = [r["ttft_ms"] for r in success_results]
    tpots = [r["tpot_ms"] for r in success_results]
    print("\n" + "="*60)
    print("📊 混部 Ground Truth 统计摘要")
    print("="*60)
    print(f"成功请求: {len(success_results)}/{len(results)}")
    print(f"TTFT: avg={sum(ttfts)/len(ttfts):.1f}ms, p50={sorted(ttfts)[len(ttfts)//2]:.1f}ms")
    print(f"TPOT: avg={sum(tpots)/len(tpots):.1f}ms, p50={sorted(tpots)[len(tpots)//2]:.1f}ms")