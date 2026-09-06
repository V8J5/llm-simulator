#!/usr/bin/env python3
"""
硬件推理能力标准化评测工具（集成内存建模 + 性价比）

功能：
1. 检查/运行整层 Profiling（如果数据不存在）
2. 检查/运行通信 Profiling（如果数据不存在）
3. 用标准 Case 计算各硬件评分
4. 输出标准化报告（CSV + JSON）

用法：
    # 在当前机器上运行完整评测
    python scripts/generate_hardware_benchmark.py --gpu-type L20

    # 在另一台机器上运行（数据自动存到 data/RTX5090/ 目录）
    python scripts/generate_hardware_benchmark.py --gpu-type RTX5090 --tp 1

    # 如果已经有数据，只生成报告
    python scripts/generate_hardware_benchmark.py \
        --gpu-type L20 \
        --tp 1 \
        --price 28000 \
        --gpu-memory-mb 46000 \
        --memory-mode inference

    查看汇总文件 data/benchmark_reports/benchmark_summary.csv
"""

import sys
import os
import json
import csv
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 添加 core 目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from memory_estimator import MemoryEstimator

# ================= 模型固定参数 =================
HIDDEN_SIZE = 5120
NUM_LAYERS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_ELEM = 2

# ================= 标准化测试用例 =================
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


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir(gpu_type: str) -> str:
    """获取指定 GPU 的数据目录"""
    base_dir = os.path.join(get_project_root(), "data")
    if gpu_type and gpu_type != "Unknown":
        data_dir = os.path.join(base_dir, gpu_type)
    else:
        data_dir = base_dir
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_layer_table(data_dir: str, filename: str) -> Dict[str, Dict]:
    """从指定目录加载层数据"""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return {f"{d['batch_size']}_{d.get('seq_len', d.get('kv_len', 0))}": d for d in data}


def load_comm_table(data_dir: str, tp_size: int) -> Dict[float, float]:
    """从指定目录加载通信表"""
    path = os.path.join(data_dir, f"comm_lookup_table_tp{tp_size}.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    # 跳过 0.01 MB 的异常点（冷启动）
    return {d['msg_size_mb']: d['allreduce_ms'] for d in data if d['msg_size_mb'] > 0.01}


def get_comm_time(comm_table: Dict[float, float], msg_size_mb: float) -> float:
    if not comm_table:
        return 0.0
    points = sorted(comm_table.items())
    if msg_size_mb <= points[0][0]:
        slope = points[0][1] / points[0][0] if points[0][0] > 0 else 0
        return max(slope * msg_size_mb, 0.001)
    if msg_size_mb >= points[-1][0]:
        if points[-1][0] > points[-2][0]:
            slope = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])
        else:
            slope = points[-1][1] / points[-1][0] if points[-1][0] > 0 else 0
        return slope * msg_size_mb
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= msg_size_mb <= x1:
            ratio = (msg_size_mb - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + ratio * (y1 - y0)
    return 0.0


def calc_prefill_time(layer_ms: float, tp: int, batch: int, seq: int, comm_table: Dict) -> float:
    compute_ms = layer_ms * NUM_LAYERS / tp
    msg_size_bytes = batch * seq * (HIDDEN_SIZE / tp) * BYTES_PER_ELEM
    msg_size_mb = msg_size_bytes / (1024 * 1024)
    comm_per_layer = get_comm_time(comm_table, msg_size_mb)
    comm_ms = comm_per_layer * 2 * NUM_LAYERS
    
    if tp == 2:
        sync_overhead = 1.05
    elif tp == 4:
        sync_overhead = 1.12
    else:
        sync_overhead = 1.0
    return (compute_ms + comm_ms) * sync_overhead


def calc_decode_time(layer_ms: float, tp: int, batch: int, kv: int, comm_table: Dict) -> float:
    compute_ms = layer_ms * NUM_LAYERS / tp
    msg_size_bytes = batch * 1 * (HIDDEN_SIZE / tp) * BYTES_PER_ELEM
    msg_size_mb = msg_size_bytes / (1024 * 1024)
    comm_per_layer = get_comm_time(comm_table, msg_size_mb)
    comm_ms = comm_per_layer * 2 * NUM_LAYERS
    
    if tp == 2:
        sync_overhead = 1.05
    elif tp == 4:
        sync_overhead = 1.12
    else:
        sync_overhead = 1.0
    return (compute_ms + comm_ms) * sync_overhead


def find_max_batch_for_prefill(tp: int, seq: int, layer_ms: float, comm_table: Dict,
                               max_batch: int = 64, gpu_memory_mb: int = 46000,mode: str = 'test') -> Tuple[int, float]:
    """在显存约束下找最大可运行 Batch"""
    memory_estimator = MemoryEstimator(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_params=32_000_000_000,
        bytes_per_param=2,
        safety_margin=0.10
    )
    
    for batch in range(max_batch, 0, -1):
        mem_info = memory_estimator.estimate_total(batch, seq, tp, stage="prefill", mode=mode)
        total_mb = mem_info.get("total_with_margin_mb", 0)
        if total_mb < gpu_memory_mb:
            time_ms = calc_prefill_time(layer_ms, tp, batch, seq, comm_table)
            throughput = seq / (time_ms / 1000)
            return batch, throughput
    return 1, 0.0


def find_max_batch_for_decode(tp: int, kv: int, layer_ms: float, comm_table: Dict,
                              max_batch: int = 64, gpu_memory_mb: int = 46000,mode: str = 'test') -> Tuple[int, float]:
    """在显存约束下找最大可运行 Batch"""
    memory_estimator = MemoryEstimator(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_params=32_000_000_000,
        bytes_per_param=2,
        safety_margin=0.10
    )
    
    for batch in range(max_batch, 0, -1):
        mem_info = memory_estimator.estimate_total(batch, kv, tp, stage="decode", mode=mode)
        total_mb = mem_info.get("total_with_margin_mb", 0)
        if total_mb < gpu_memory_mb:
            time_ms = calc_decode_time(layer_ms, tp, batch, kv, comm_table)
            throughput = 1 / (time_ms / 1000)
            return batch, throughput
    return 1, 0.0


def run_script(script_name: str, data_dir: str) -> bool:
    """运行指定的 Python 脚本"""
    script_path = os.path.join(get_project_root(), "scripts", script_name)
    if not os.path.exists(script_path):
        print(f"⚠️ 脚本不存在: {script_path}")
        return False
    
    print(f"🔄 运行: python {script_name}")
    result = subprocess.run([sys.executable, script_path], cwd=get_project_root())
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="硬件推理能力标准化评测")
    parser.add_argument("--no-run", action="store_true", help="不运行 Profiling，只生成报告")
    parser.add_argument("--gpu-type", required=True, help="GPU 型号标识（如 L20, RTX5090, A100）")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel 大小")
    parser.add_argument("--price", type=float, default=None, help="GPU 价格（元），用于计算性价比")
    parser.add_argument("--gpu-memory-mb", type=int, default=46000, help="GPU 显存大小（MB）")
    parser.add_argument("--memory-mode", choices=["test", "inference"], default="test",help="显存估算模式: test=测试模式(不含权重), inference=真实推理模式(含权重)")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()
    
    # ========== 按 GPU 型号隔离数据目录 ==========
    DATA_DIR = get_data_dir(args.gpu_type)
    OUTPUT_DIR = args.output_dir or os.path.join(get_project_root(), "data", "benchmark_reports")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    GPU_TYPE = args.gpu_type
    TP = args.tp
    PRICE = args.price
    GPU_MEMORY_MB = args.gpu_memory_mb
    MEMORY_MODE = args.memory_mode
    
    print("=" * 70)
    print(f"🔬 硬件推理能力标准化评测（内存建模 + 性价比）")
    print("=" * 70)
    print(f"GPU 型号: {GPU_TYPE}")
    print(f"TP 配置: {TP}")
    print(f"显存: {GPU_MEMORY_MB} MB")
    print(f"价格: {PRICE if PRICE else '未提供'}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # ========== 检查/运行 Profiling ==========
    prefill_layer_path = os.path.join(DATA_DIR, "qwen3_32b_prefill_layer_lookup.json")
    decode_layer_path = os.path.join(DATA_DIR, "qwen3_32b_decode_layer_lookup.json")
    comm_path = os.path.join(DATA_DIR, f"comm_lookup_table_tp{TP}.json")
    
    if not args.no_run:
        # 检查 Prefill 整层数据
        if not os.path.exists(prefill_layer_path):
            print("📊 Prefill 整层数据不存在，开始 Profiling...")
            if not run_script("prefill_layer_bench.py", DATA_DIR):
                print("❌ Prefill Profiling 失败")
                return 1
        else:
            print(f"✅ Prefill 整层数据已存在: {os.path.basename(prefill_layer_path)}")
        
        # 检查 Decode 整层数据
        if not os.path.exists(decode_layer_path):
            print("📊 Decode 整层数据不存在，开始 Profiling...")
            if not run_script("decode_layer_bench.py", DATA_DIR):
                print("❌ Decode Profiling 失败")
                return 1
        else:
            print(f"✅ Decode 整层数据已存在: {os.path.basename(decode_layer_path)}")
        
        # 检查通信数据
        if not os.path.exists(comm_path):
            print(f"📊 通信数据 TP={TP} 不存在，请手动运行:")
            print(f"   torchrun --nproc_per_node={TP} scripts/comm_micro_bench.py")
            print("   或将已有通信表复制到当前数据目录")
        else:
            print(f"✅ 通信数据已存在: {os.path.basename(comm_path)}")
    
    # ========== 加载数据 ==========
    prefill_layer = load_layer_table(DATA_DIR, "qwen3_32b_prefill_layer_lookup.json")
    decode_layer = load_layer_table(DATA_DIR, "qwen3_32b_decode_layer_lookup.json")
    
    if not prefill_layer or not decode_layer:
        print("❌ 未找到整层 Profiling 数据，请先运行:")
        print(f"   python scripts/prefill_layer_bench.py")
        print(f"   python scripts/decode_layer_bench.py")
        print(f"   并确保数据存放在: {DATA_DIR}")
        return 1
    
    # 加载通信表
    comm_table = load_comm_table(DATA_DIR, TP)
    if not comm_table:
        # 尝试从父目录加载
        fallback_dir = os.path.join(get_project_root(), "data")
        comm_table = load_comm_table(fallback_dir, TP)
        if comm_table:
            print(f"📡 从父目录使用通信表 TP={TP}")
        else:
            print(f"⚠️ 未找到通信表 TP={TP}，将忽略通信开销")
    
    print(f"📡 使用通信表 TP={TP}" if comm_table else "⚠️ 无通信表，通信开销为 0")
    
    # ========== 评分计算 ==========
    results = {
        "meta": {
            "gpu_type": GPU_TYPE,
            "tp": TP,
            "price": PRICE,
            "gpu_memory_mb": GPU_MEMORY_MB,
            "timestamp": datetime.now().isoformat(),
        },
        "prefill": [],
        "decode": [],
        "memory_constrained": [],
        "summary": {}
    }
    
    # Prefill
    print("\n📈 Prefill 评分（标准 Case）:")
    print(f"{'Case':<8} {'Batch':<8} {'Seq':<10} {'单层(ms)':<12} {'整模型(ms)':<14} {'吞吐(tok/s)':<14}")
    print("-" * 70)
    
    prefill_throughputs = []
    for case in BENCHMARK_CASES["prefill"]:
        name = case["name"]
        batch = case["batch"]
        seq = case["seq"]
        key = f"{batch}_{seq}"
        
        if key not in prefill_layer:
            print(f"{name:<8} {'N/A':<8} {seq:<10} {'N/A':<12} {'N/A':<14} {'N/A':<14}")
            continue
        
        layer_ms = prefill_layer[key]["prefill_layer_ms"]
        
        # 检查显存
        memory_estimator = MemoryEstimator(
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            num_params=32_000_000_000,
            bytes_per_param=2,
            safety_margin=0.10
        )
        mem_info = memory_estimator.estimate_total(batch, seq, TP, stage="prefill", mode=MEMORY_MODE)
        total_mb = mem_info.get("total_with_margin_mb", 0)
        
        if total_mb > GPU_MEMORY_MB:
            new_batch, throughput = find_max_batch_for_prefill(TP, seq, layer_ms, comm_table,
                                                               gpu_memory_mb=GPU_MEMORY_MB,mode=MEMORY_MODE)
            total_ms = calc_prefill_time(layer_ms, TP, new_batch, seq, comm_table)
            results["memory_constrained"].append({
                "stage": "Prefill",
                "case": name,
                "original_batch": batch,
                "adjusted_batch": new_batch,
                "seq": seq,
                "throughput_tok_s": round(throughput, 2)
            })
            print(f"{name:<8} {new_batch:<8} {seq:<10} {layer_ms:<12.4f} {total_ms:<14.2f} {throughput:<14.2f} ⚠️ 降Batch")
            prefill_throughputs.append(throughput)
            results["prefill"].append({
                "case": name, "batch": new_batch, "seq": seq,
                "layer_ms": round(layer_ms, 4), "total_ms": round(total_ms, 2),
                "throughput_tok_s": round(throughput, 2),
                "original_batch": batch, "memory_oom": True
            })
        else:
            total_ms = calc_prefill_time(layer_ms, TP, batch, seq, comm_table)
            throughput = seq / (total_ms / 1000)
            prefill_throughputs.append(throughput)
            results["prefill"].append({
                "case": name, "batch": batch, "seq": seq,
                "layer_ms": round(layer_ms, 4), "total_ms": round(total_ms, 2),
                "throughput_tok_s": round(throughput, 2),
                "original_batch": batch, "memory_oom": False
            })
            print(f"{name:<8} {batch:<8} {seq:<10} {layer_ms:<12.4f} {total_ms:<14.2f} {throughput:<14.2f}")
    
    # Decode
    print("\n📈 Decode 评分（标准 Case）:")
    print(f"{'Case':<8} {'Batch':<8} {'KV':<10} {'单层(ms)':<12} {'整模型(ms)':<14} {'吞吐(tok/s)':<14}")
    print("-" * 70)
    
    decode_throughputs = []
    for case in BENCHMARK_CASES["decode"]:
        name = case["name"]
        batch = case["batch"]
        kv = case["kv"]
        key = f"{batch}_{kv}"
        
        if key not in decode_layer:
            print(f"{name:<8} {'N/A':<8} {kv:<10} {'N/A':<12} {'N/A':<14} {'N/A':<14}")
            continue
        
        layer_ms = decode_layer[key]["decode_layer_ms"]
        
        memory_estimator = MemoryEstimator(
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            num_params=32_000_000_000,
            bytes_per_param=2,
            safety_margin=0.10
        )
        mem_info = memory_estimator.estimate_total(batch, kv, TP, stage="decode", mode=MEMORY_MODE)
        total_mb = mem_info.get("total_with_margin_mb", 0)
        
        if total_mb > GPU_MEMORY_MB:
            new_batch, throughput = find_max_batch_for_decode(TP, kv, layer_ms, comm_table,
                                                              gpu_memory_mb=GPU_MEMORY_MB,mode=MEMORY_MODE)
            total_ms = calc_decode_time(layer_ms, TP, new_batch, kv, comm_table)
            results["memory_constrained"].append({
                "stage": "Decode",
                "case": name,
                "original_batch": batch,
                "adjusted_batch": new_batch,
                "kv": kv,
                "throughput_tok_s": round(throughput, 2)
            })
            print(f"{name:<8} {new_batch:<8} {kv:<10} {layer_ms:<12.4f} {total_ms:<14.2f} {throughput:<14.2f} ⚠️ 降Batch")
            decode_throughputs.append(throughput)
            results["decode"].append({
                "case": name, "batch": new_batch, "kv": kv,
                "layer_ms": round(layer_ms, 4), "total_ms": round(total_ms, 2),
                "throughput_tok_s": round(throughput, 2),
                "original_batch": batch, "memory_oom": True
            })
        else:
            total_ms = calc_decode_time(layer_ms, TP, batch, kv, comm_table)
            throughput = 1 / (total_ms / 1000)
            decode_throughputs.append(throughput)
            results["decode"].append({
                "case": name, "batch": batch, "kv": kv,
                "layer_ms": round(layer_ms, 4), "total_ms": round(total_ms, 2),
                "throughput_tok_s": round(throughput, 2),
                "original_batch": batch, "memory_oom": False
            })
            print(f"{name:<8} {batch:<8} {kv:<10} {layer_ms:<12.4f} {total_ms:<14.2f} {throughput:<14.2f}")
    
    # 综合评分
    if prefill_throughputs and decode_throughputs:
        avg_prefill = sum(prefill_throughputs) / len(prefill_throughputs)
        avg_decode = sum(decode_throughputs) / len(decode_throughputs)
        combined = avg_prefill * 0.5 + avg_decode * 0.5
        
        results["summary"] = {
            "avg_prefill_throughput": round(avg_prefill, 2),
            "avg_decode_throughput": round(avg_decode, 2),
            "combined_score": round(combined, 2),
            "prefill_cases": len(prefill_throughputs),
            "decode_cases": len(decode_throughputs)
        }
        
        if PRICE:
            results["summary"]["price"] = PRICE
            results["summary"]["性价比"] = round(combined / PRICE, 4)
        
        print("\n" + "=" * 70)
        print("📊 综合评分")
        print("=" * 70)
        print(f"  Prefill 平均吞吐: {avg_prefill:.2f} tokens/s")
        print(f"  Decode 平均吞吐:  {avg_decode:.2f} tokens/s")
        print(f"  综合评分 (各50%):  {combined:.2f}")
        if PRICE:
            print(f"  性价比:           {combined/PRICE:.4f} 分/元")
    
    if results["memory_constrained"]:
        print("\n" + "=" * 70)
        print("📊 显存降级记录")
        print("=" * 70)
        for item in results["memory_constrained"]:
            print(f"  {item['stage']} {item['case']}: {item['original_batch']}→{item['adjusted_batch']} batch, 吞吐 {item['throughput_tok_s']:.2f} tok/s")
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_slug = GPU_TYPE.replace(" ", "_").replace("/", "_")
    report_filename = f"benchmark_{gpu_slug}_TP{TP}_{timestamp}"
    
    json_path = os.path.join(OUTPUT_DIR, f"{report_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 JSON 报告: {json_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, f"{report_filename}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["GPU", "TP", "价格", "阶段", "Case", "Batch", "Seq/KV", "单层(ms)", "整模型(ms)", "吞吐(tok/s)"])
        for item in results["prefill"]:
            writer.writerow([GPU_TYPE, TP, PRICE or "", "Prefill", item["case"], item["batch"], item["seq"],
                            item["layer_ms"], item["total_ms"], item["throughput_tok_s"]])
        for item in results["decode"]:
            writer.writerow([GPU_TYPE, TP, PRICE or "", "Decode", item["case"], item["batch"], item["kv"],
                            item["layer_ms"], item["total_ms"], item["throughput_tok_s"]])
    print(f"💾 CSV 报告: {csv_path}")
    
    summary_csv = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    file_exists = os.path.exists(summary_csv)
    with open(summary_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["GPU", "TP", "价格", "Prefill_Avg(tok/s)", "Decode_Avg(tok/s)", "综合评分", "性价比"])
        writer.writerow([
            GPU_TYPE, TP, PRICE or "",
            results["summary"].get("avg_prefill_throughput", 0),
            results["summary"].get("avg_decode_throughput", 0),
            results["summary"].get("combined_score", 0),
            results["summary"].get("性价比", "")
        ])
    print(f"💾 汇总已追加: {summary_csv}")
    
    print("\n✅ 评测完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())