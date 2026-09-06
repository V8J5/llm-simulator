#!/usr/bin/env python3
"""
TP 扩展分析工具

输入 (batch, seq_len)，输出不同 TP 下的计算/通信对比
用法:
    python analyze_tp_scaling.py --batch 8 --seq 4096
    python analyze_tp_scaling.py -b 8 -s 4096
    python analyze_tp_scaling.py -b 8 -s 4096 --stage prefill
"""

import csv
import argparse
import os
from typing import Dict, List, Tuple


def load_data(csv_path: str) -> List[Dict]:
    """加载 validation_report.csv"""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_config(data: List[Dict], batch: int, seq_len: int, stage: str = None) -> List[Dict]:
    """筛选指定配置的数据"""
    rows = []
    for row in data:
        if row['Batch'] == str(batch) and row['Seq_Len'] == str(seq_len):
            if stage is None or row['Stage'].lower() == stage.lower():
                rows.append(row)
    return rows


def parse_float(value: str) -> float:
    """安全转换为 float"""
    if value == '-' or value == '':
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def analyze_tp_scaling(rows: List[Dict]) -> Dict:
    """
    分析 TP 扩展趋势
    
    返回:
        {
            "tps": [1, 2, 4, 8],
            "compute_ms": [...],
            "comm_ms": [...],
            "total_ms": [...],
            "bottlenecks": [...],
            "feasible": [...]
        }
    """
    # 按 TP 排序
    rows_sorted = sorted(rows, key=lambda r: int(r['TP']))
    
    tps = [int(r['TP']) for r in rows_sorted]
    
    # 计算总时间 = Actual_ms (真实) 或 Predicted_ms
    # 这里用 Actual_ms 作为真实对比，也可以切换到 Predicted_ms
    total_ms = [parse_float(r['Actual_ms']) for r in rows_sorted]
    
    # 从 breakdown 恢复 compute 和 comm
    # 方法：用 Actual_ms * Compute_Ratio
    compute_ratio = [parse_float(r['Compute_Ratio']) for r in rows_sorted]
    comm_ratio = [parse_float(r['Comm_Ratio']) for r in rows_sorted]
    
    compute_ms = [total_ms[i] * compute_ratio[i] if total_ms[i] > 0 else 0 for i in range(len(total_ms))]
    comm_ms = [total_ms[i] * comm_ratio[i] if total_ms[i] > 0 else 0 for i in range(len(total_ms))]
    
    bottlenecks = [r['Bottleneck'] for r in rows_sorted]
    feasible = [r['Feasible'] == 'True' for r in rows_sorted]
    
    return {
        "tps": tps,
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "total_ms": total_ms,
        "bottlenecks": bottlenecks,
        "feasible": feasible
    }


def print_analysis(result: Dict, batch: int, seq_len: int, stage: str):
    """打印分析结果"""
    tps = result["tps"]
    compute_ms = result["compute_ms"]
    comm_ms = result["comm_ms"]
    total_ms = result["total_ms"]
    bottlenecks = result["bottlenecks"]
    feasible = result["feasible"]
    
    if not tps:
        print(f"❌ 未找到配置: Batch={batch}, Seq={seq_len}, Stage={stage}")
        return
    
    print("=" * 70)
    print(f"📊 TP 扩展分析: Batch={batch}, Seq={seq_len}, Stage={stage}")
    print("=" * 70)
    
    # 表头
    print(f"\n{'TP':>6} | {'计算时间 (ms)':>14} | {'通信时间 (ms)':>14} | {'总时间 (ms)':>14} | {'瓶颈':>12} | {'可行'}")
    print("-" * 80)
    
    for i, tp in enumerate(tps):
        feasible_str = "✅" if feasible[i] else "❌ OOM"
        print(f"{tp:>6} | {compute_ms[i]:>14.2f} | {comm_ms[i]:>14.2f} | {total_ms[i]:>14.2f} | {bottlenecks[i]:>12} | {feasible_str}")
    
    # 如果所有 TP 都可行，分析变化
    if all(feasible) and len(tps) >= 2:
        print("\n" + "=" * 70)
        print("📈 变化分析")
        print("=" * 70)
        
        base_compute = compute_ms[0]
        base_comm = comm_ms[0]
        
        for i in range(1, len(tps)):
            tp_prev = tps[i-1]
            tp_curr = tps[i]
            
            compute_delta = compute_ms[i] - compute_ms[i-1]
            comm_delta = comm_ms[i] - comm_ms[i-1]
            total_delta = total_ms[i] - total_ms[i-1]
            total_pct = (total_delta / total_ms[i-1] * 100) if total_ms[i-1] > 0 else 0
            
            compute_pct = (compute_delta / compute_ms[i-1] * 100) if compute_ms[i-1] > 0 else 0
            comm_pct = (comm_delta / comm_ms[i-1] * 100) if comm_ms[i-1] > 0 else 0
            
            print(f"\n  TP {tp_prev} → {tp_curr}:")
            print(f"    计算: {compute_delta:+.2f} ms ({compute_pct:+.1f}%)")
            print(f"    通信: {comm_delta:+.2f} ms ({comm_pct:+.1f}%)")
            print(f"    总时间: {total_delta:+.2f} ms ({total_pct:+.1f}%)")
            
            if compute_delta < 0 and comm_delta > 0:
                print(f"    💡 计算减少 {abs(compute_delta):.0f}ms，通信增加 {comm_delta:.0f}ms，收益递减")
            elif compute_delta < 0 and comm_delta < 0:
                print(f"    ✅ 计算和通信同时减少，TP 扩展有效")
            elif compute_delta > 0 and comm_delta > 0:
                print(f"    ⚠️ 计算和通信都增加，TP 扩展无效")
        
        # 找出最优 TP
        best_idx = min(range(len(total_ms)), key=lambda i: total_ms[i])
        print(f"\n  🏆 最优 TP: {tps[best_idx]} (总时间 {total_ms[best_idx]:.2f} ms)")
        
        # 如果最佳 TP 不是最大 TP，说明通信开始成为瓶颈
        if tps[best_idx] < max(tps):
            print(f"    💡 超过 TP={tps[best_idx]} 后，通信增加超过了计算减少，总时间变长")
    else:
        # 有配置不可行
        print("\n" + "=" * 70)
        print("⚠️ 部分配置显存不足:")
        print("=" * 70)
        for i, tp in enumerate(tps):
            if not feasible[i]:
                print(f"  TP={tp}: 显存不足 (OOM)")


def main():
    parser = argparse.ArgumentParser(
        description="分析 TP 扩展对计算/通信的影响",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_tp_scaling.py -b 8 -s 4096
  python analyze_tp_scaling.py -b 8 -s 4096 --stage prefill
  python analyze_tp_scaling.py -b 8 -s 4096 --stage decode
        """
    )
    parser.add_argument("-b", "--batch", type=int, required=True, help="Batch 大小")
    parser.add_argument("-s", "--seq", type=int, required=True, help="序列长度 (Prefill 用 Seq_Len, Decode 用 KV_Len)")
    parser.add_argument("--stage", type=str, default=None, choices=["prefill", "decode"], help="阶段 (prefill/decode)，不指定则自动查找")
    parser.add_argument("--csv", type=str, 
                        default="/data/home/lihaozhe/llm-simulator/data/validation_report.csv",
                        help="validation_report.csv 路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"❌ CSV 文件不存在: {args.csv}")
        return
    
    # 加载数据
    data = load_data(args.csv)
    
    # 尝试查找匹配的配置
    rows = filter_config(data, args.batch, args.seq, args.stage)
    
    # 如果没有找到且 stage 是 None，尝试在 Prefill 和 Decode 中分别查找
    if not rows and args.stage is None:
        rows_prefill = filter_config(data, args.batch, args.seq, "prefill")
        rows_decode = filter_config(data, args.batch, args.seq, "decode")
        
        if rows_prefill and rows_decode:
            # 同时找到两个，询问用户
            print(f"⚠️ 配置 Batch={args.batch}, Seq={args.seq} 在 Prefill 和 Decode 中都存在")
            print("  使用 --stage prefill 或 --stage decode 指定")
            print("\nPrefill 数据:")
            analyze_result = analyze_tp_scaling(rows_prefill)
            print_analysis(analyze_result, args.batch, args.seq, "prefill")
            print("\nDecode 数据:")
            analyze_result = analyze_tp_scaling(rows_decode)
            print_analysis(analyze_result, args.batch, args.seq, "decode")
            return
        elif rows_prefill:
            rows = rows_prefill
            args.stage = "prefill"
        elif rows_decode:
            rows = rows_decode
            args.stage = "decode"
    
    if rows:
        result = analyze_tp_scaling(rows)
        print_analysis(result, args.batch, args.seq, args.stage)
    else:
        print(f"❌ 未找到配置: Batch={args.batch}, Seq={args.seq}, Stage={args.stage}")
        print(f"   可用的配置示例:")
        # 显示前 10 个配置
        seen = set()
        count = 0
        for row in data:
            key = (row['Batch'], row['Seq_Len'])
            if key not in seen:
                seen.add(key)
                print(f"     Batch={row['Batch']}, Seq={row['Seq_Len']}")
                count += 1
                if count >= 10:
                    print("     ...")
                    break


if __name__ == "__main__":
    main()