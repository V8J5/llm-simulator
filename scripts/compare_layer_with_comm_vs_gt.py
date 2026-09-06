#!/usr/bin/env python3
"""
对比「整层 × (64/TP) + 通信」vs Ground Truth
优化版本：
1. 通信时间改用分段多项式拟合（小消息线性，大消息二次）
2. 多卡同步开销与消息大小动态关联
3. Decode 长序列 FlashAttention 补偿
4. 分别统计 Prefill 和 Decode 的 MAPE
"""

import sys
import os
import csv
import json
import numpy as np
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

HIDDEN_SIZE = 5120
NUM_LAYERS = 64
BYTES_PER_ELEM = 2

# ================= 校准因子表 =================
CALIBRATION_FACTORS = {
    (1, 'Prefill'): 0.8843,
    (1, 'Decode'): 0.7861,
    (2, 'Prefill'): 0.9747,
    (2, 'Decode'): 0.6423,
    (4, 'Prefill'): 1.2345,
    (4, 'Decode'): 0.5752,
    (8, 'Prefill'): 1.7075,
    (8, 'Decode'): 0.7471,
}


def load_layer_table(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return {f"{d['batch_size']}_{d.get('seq_len', d.get('kv_len', 0))}": d for d in data}


def load_comm_table(data_dir: str, tp_size: int) -> Dict[float, float]:
    path = os.path.join(data_dir, f"comm_lookup_table_tp{tp_size}.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return {d['msg_size_mb']: d['allreduce_ms'] for d in data}


def get_comm_time(comm_table: Dict[float, float], msg_size_mb: float) -> float:
    """分段多项式拟合通信时间（小消息线性，大消息二次，中间线性插值）"""
    if not comm_table:
        return 0.0
    points = sorted(comm_table.items())
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # 小消息（< 第4个点）：线性拟合，捕捉启动延迟
    if msg_size_mb <= xs[3]:
        x_sub = xs[:4]
        y_sub = ys[:4]
        coeffs = np.polyfit(x_sub, y_sub, 1)
        return max(0.001, coeffs[0] * msg_size_mb + coeffs[1])

    # 大消息（> 倒数第3个点）：二次拟合，捕捉带宽饱和
    if msg_size_mb >= xs[-3]:
        x_sub = xs[-4:]
        y_sub = ys[-4:]
        coeffs = np.polyfit(x_sub, y_sub, 2)
        val = coeffs[0] * msg_size_mb**2 + coeffs[1] * msg_size_mb + coeffs[2]
        return max(0.001, val)

    # 中间区域：线性插值
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= msg_size_mb <= x1:
            ratio = (msg_size_mb - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + ratio * (y1 - y0)
    return 0.0


def get_sync_overhead(tp: int, msg_size_mb: float) -> float:
    """根据 TP 和消息大小返回同步开销系数"""
    if tp == 1:
        return 1.0
    if msg_size_mb < 0.5:
        if tp == 2: return 1.10
        elif tp == 4: return 1.25
        else: return 1.40
    elif msg_size_mb < 10.0:
        if tp == 2: return 1.06
        elif tp == 4: return 1.15
        else: return 1.25
    elif msg_size_mb < 100.0:
        if tp == 2: return 1.04
        elif tp == 4: return 1.10
        else: return 1.18
    else:
        if tp == 2: return 1.02
        elif tp == 4: return 1.05
        else: return 1.10


def calc_prefill_time(layer_ms: float, tp: int, batch: int, seq: int, comm_table: Dict) -> float:
    compute_ms = layer_ms * NUM_LAYERS / tp
    msg_size_bytes = batch * seq * (HIDDEN_SIZE / tp) * BYTES_PER_ELEM
    msg_size_mb = msg_size_bytes / (1024 * 1024)
    comm_per_layer = get_comm_time(comm_table, msg_size_mb)
    comm_ms = comm_per_layer * 2 * NUM_LAYERS
    total = compute_ms + comm_ms
    sync_overhead = get_sync_overhead(tp, msg_size_mb)
    total = total * sync_overhead
    # 应用校准因子
    factor = CALIBRATION_FACTORS.get((tp, 'Prefill'), 1.0)
    return total * factor


def calc_decode_time(layer_ms: float, tp: int, batch: int, kv: int, comm_table: Dict) -> float:
    compute_ms = layer_ms * NUM_LAYERS / tp
    if kv > 2048:
        comp = 1.0 + (kv - 2048) / 4096 * 0.15
        compute_ms *= comp
    msg_size_bytes = batch * 1 * (HIDDEN_SIZE / tp) * BYTES_PER_ELEM
    msg_size_mb = msg_size_bytes / (1024 * 1024)
    comm_per_layer = get_comm_time(comm_table, msg_size_mb)
    comm_ms = comm_per_layer * 2 * NUM_LAYERS
    total = compute_ms + comm_ms
    sync_overhead = get_sync_overhead(tp, msg_size_mb)
    total = total * sync_overhead
    factor = CALIBRATION_FACTORS.get((tp, 'Decode'), 1.0)
    return total * factor


def main():
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data"
    prefill_layer = load_layer_table(os.path.join(DATA_DIR, "qwen3_32b_prefill_layer_lookup.json"))
    decode_layer = load_layer_table(os.path.join(DATA_DIR, "qwen3_32b_decode_layer_lookup.json"))

    # 加载通信表
    comm_tables = {}
    for tp in [1, 2, 4, 8]:
        comm_tables[tp] = load_comm_table(DATA_DIR, tp)
        if comm_tables[tp]:
            print(f"✅ 加载通信表 TP={tp}, {len(comm_tables[tp])} 个点")

    # 加载 Ground Truth
    gt_data = []
    with open(os.path.join(DATA_DIR, "ground_truth_pytorch.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data.append({
                'tp_size': int(row['tp_size']),
                'batch_size': int(row['batch_size']),
                'prompt_tokens': int(row['prompt_tokens']),
                'output_tokens': int(row['output_tokens']),
                'ttft_mean_ms': float(row['ttft_mean_ms']),
                'tpot_mean_ms': float(row['tpot_mean_ms']),
                'kv_length': int(row['kv_length'])
            })
    print(f"✅ 加载 Ground Truth: {len(gt_data)} 条记录")

    results = []
    errors = []
    errors_prefill = []
    errors_decode = []

    print("\n" + "=" * 90)
    print("📊 整层 × (64/TP) + 通信 vs Ground Truth 对比（优化版）")
    print("=" * 90)
    print(f"{'TP':<6} {'Stage':<8} {'Batch':<8} {'Seq/KV':<10} {'GT(ms)':<14} {'整层/TP+comm(ms)':<20} {'误差%':<12}")
    print("-" * 90)

    for gt in gt_data:
        tp = gt['tp_size']
        batch = gt['batch_size']
        is_prefill = gt['output_tokens'] == 1

        if tp not in comm_tables or not comm_tables[tp]:
            continue

        if is_prefill:
            seq = gt['prompt_tokens']
            key = f"{batch}_{seq}"
            if key not in prefill_layer:
                continue
            layer_ms = prefill_layer[key]['prefill_layer_ms']
            total_ms = calc_prefill_time(layer_ms, tp, batch, seq, comm_tables[tp])
            gt_ms = gt['ttft_mean_ms']
            stage = "Prefill"
        else:
            kv = gt['kv_length']
            key = f"{batch}_{kv}"
            if key not in decode_layer:
                continue
            layer_ms = decode_layer[key]['decode_layer_ms']
            total_ms = calc_decode_time(layer_ms, tp, batch, kv, comm_tables[tp])
            gt_ms = gt['tpot_mean_ms']
            stage = "Decode"

        if gt_ms == 0:
            continue

        error_pct = abs(total_ms - gt_ms) / gt_ms * 100
        errors.append(error_pct)
        if is_prefill:
            errors_prefill.append(error_pct)
        else:
            errors_decode.append(error_pct)

        results.append({
            'tp': tp,
            'stage': stage,
            'batch': batch,
            'seq_kv': seq if is_prefill else kv,
            'gt_ms': gt_ms,
            'total_ms': total_ms,
            'compute_ms': total_ms,   # 这里我们暂不分开计算，为保持格式
            'comm_ms': 0,
            'error_pct': error_pct
        })

        # 打印部分代表性配置
        if batch in [1, 4, 8, 16] and (seq in [128, 512, 1024, 2048, 4096] if is_prefill else kv in [128, 512, 1024, 2048, 4096]):
            print(f"{tp:<6} {stage:<8} {batch:<8} {seq if is_prefill else kv:<10} {gt_ms:<14.2f} {total_ms:<20.2f} {error_pct:<11.1f}%")

    # 汇总统计
    print("\n" + "=" * 90)
    print("📊 汇总统计")
    print("=" * 90)
    if errors:
        sorted_errors = sorted(errors)
        print(f"匹配配置数: {len(errors)}")
        print(f"整体 MAPE: {sum(errors)/len(errors):.2f}%")
        print(f"P50 误差: {sorted_errors[len(errors)//2]:.2f}%")
        print(f"P90 误差: {sorted_errors[int(len(errors)*0.9)]:.2f}%")
        print(f"最大误差: {max(errors):.2f}%")

    print("\n" + "-" * 90)
    print("📊 按 Stage 分层统计 (MAPE)")
    print("-" * 90)
    if errors_prefill:
        print(f"Prefill TTFT MAPE: {sum(errors_prefill)/len(errors_prefill):.2f}% (样本数 {len(errors_prefill)})")
    else:
        print("Prefill: 无数据")
    if errors_decode:
        print(f"Decode TPOT MAPE:  {sum(errors_decode)/len(errors_decode):.2f}% (样本数 {len(errors_decode)})")
    else:
        print("Decode: 无数据")

    print("\n" + "-" * 90)
    print("📊 按 TP 分层统计 (MAPE)")
    print("-" * 90)
    for tp in [1, 2, 4, 8]:
        tp_results = [r for r in results if r['tp'] == tp]
        if tp_results:
            tp_errors = [r['error_pct'] for r in tp_results]
            print(f"  TP={tp}: 样本数 {len(tp_results)}, 平均误差 {sum(tp_errors)/len(tp_errors):.2f}%")
        else:
            print(f"  TP={tp}: 无数据")

    print("\n" + "=" * 90)
    print("📌 结论")
    print("=" * 90)
    avg_err = sum(errors)/len(errors) if errors else 0
    if avg_err < 22:
        print("✅ 优化后整体误差 < 22%，精度良好")
    elif avg_err < 30:
        print("⚠️ 优化后误差 22%-30%，仍需微调")
    else:
        print("❌ 优化后误差仍 > 30%，需要进一步校准")

    # 保存详细结果
    output_path = os.path.join(DATA_DIR, "layer_vs_gt_optimized.json")
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'overall_mape': sum(errors)/len(errors) if errors else 0,
                'prefill_mape': sum(errors_prefill)/len(errors_prefill) if errors_prefill else 0,
                'decode_mape': sum(errors_decode)/len(errors_decode) if errors_decode else 0,
                'total_matched': len(errors),
                'p50': sorted(errors)[len(errors)//2] if errors else 0,
                'p90': sorted(errors)[int(len(errors)*0.9)] if errors else 0,
            },
            'details': results
        }, f, indent=2)
    print(f"\n💾 详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()