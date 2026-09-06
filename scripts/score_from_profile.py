#!/usr/bin/env python3
"""
独立评分工具：用户提供 Profiling 数据（CSV），自动输出评分

用法：
    python scripts/score_from_profile.py --input /path/to/profiling.csv --gpu-type L20 --tp 1
    python scripts/score_from_profile.py --input /path/to/profiling.csv --gpu-type A100 --tp 1 --price 20000
"""

import sys
import os
import csv
import json
import argparse
from typing import Dict, List
from datetime import datetime

# ================= 模型固定参数 =================
NUM_LAYERS = 64
BENCHMARK_CASES = {
    "prefill": [
        {"name": "P_S",  "batch": 1,  "seq": 512},
        {"name": "P_M",  "batch": 4,  "seq": 1024},
        {"name": "P_L",  "batch": 8,  "seq": 4096},
        {"name": "P_XL", "batch": 16, "seq": 4096},
    ],
    "decode": [
        {"name": "D_S",  "batch": 1,  "kv": 512},
        {"name": "D_M",  "batch": 8,  "kv": 1024},
        {"name": "D_L",  "batch": 16, "kv": 4096},
        {"name": "D_XL", "batch": 32, "kv": 4096},
    ]
}


def parse_profiling_csv(filepath: str) -> Dict[str, float]:
    """解析 Profiling CSV，返回 {key: layer_ms}"""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 根据列名自动识别
            batch = int(row.get('batch_size', row.get('batch', 0)))
            seq = int(row.get('seq_len', row.get('seq', row.get('kv', row.get('kv_len', 0)))))
            layer_ms = float(row.get('prefill_layer_ms', row.get('decode_layer_ms', row.get('layer_ms', 0))))
            key = f"{batch}_{seq}"
            data[key] = layer_ms
    return data


def main():
    parser = argparse.ArgumentParser(description="从 Profiling 数据生成硬件评分")
    parser.add_argument("--input", required=True, help="Profiling CSV 文件路径")
    parser.add_argument("--gpu-type", default="Unknown", help="GPU 型号标识")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel 大小")
    parser.add_argument("--price", type=float, default=None, help="GPU 价格（元）")
    parser.add_argument("--output-dir", default="./", help="输出目录")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return 1
    
    # 解析数据
    data = parse_profiling_csv(args.input)
    print(f"📊 加载了 {len(data)} 条 Profiling 记录")
    
    # 计算 Prefill 评分
    prefill_throughputs = []
    prefill_results = []
    for case in BENCHMARK_CASES["prefill"]:
        key = f"{case['batch']}_{case['seq']}"
        if key not in data:
            continue
        layer_ms = data[key]
        # 简单估算：假设没有通信（因为是纯 Profiling）
        total_ms = layer_ms * NUM_LAYERS / args.tp
        throughput = case['seq'] / (total_ms / 1000)
        prefill_throughputs.append(throughput)
        prefill_results.append({
            "case": case['name'],
            "batch": case['batch'],
            "seq": case['seq'],
            "throughput_tok_s": round(throughput, 2)
        })
    
    # 计算 Decode 评分
    decode_throughputs = []
    decode_results = []
    for case in BENCHMARK_CASES["decode"]:
        key = f"{case['batch']}_{case['kv']}"
        if key not in data:
            continue
        layer_ms = data[key]
        total_ms = layer_ms * NUM_LAYERS / args.tp
        throughput = 1 / (total_ms / 1000)
        decode_throughputs.append(throughput)
        decode_results.append({
            "case": case['name'],
            "batch": case['batch'],
            "kv": case['kv'],
            "throughput_tok_s": round(throughput, 2)
        })
    
    # 综合评分
    avg_prefill = sum(prefill_throughputs) / len(prefill_throughputs) if prefill_throughputs else 0
    avg_decode = sum(decode_throughputs) / len(decode_throughputs) if decode_throughputs else 0
    combined = avg_prefill * 0.5 + avg_decode * 0.5
    
    # 输出报告
    print("\n" + "=" * 70)
    print(f"📊 硬件评分报告: {args.gpu_type}")
    print("=" * 70)
    print(f"Prefill 平均吞吐: {avg_prefill:.2f} tok/s")
    print(f"Decode 平均吞吐:  {avg_decode:.2f} tok/s")
    print(f"综合评分:          {combined:.2f}")
    if args.price:
        print(f"性价比:            {combined/args.price:.4f} 分/元")
    
    # 保存 JSON
    output = {
        "gpu_type": args.gpu_type,
        "tp": args.tp,
        "price": args.price,
        "timestamp": datetime.now().isoformat(),
        "prefill": prefill_results,
        "decode": decode_results,
        "summary": {
            "avg_prefill_throughput": round(avg_prefill, 2),
            "avg_decode_throughput": round(avg_decode, 2),
            "combined_score": round(combined, 2)
        }
    }
    if args.price:
        output["summary"]["性价比"] = round(combined / args.price, 4)
    
    out_path = os.path.join(args.output_dir, f"score_{args.gpu_type}_TP{args.tp}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 结果已保存: {out_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())