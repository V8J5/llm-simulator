#!/usr/bin/env python3
"""
分析验证数据，找出实际/预测比值与工作负载的关系
"""

import csv
import os
import sys
from collections import defaultdict

CSV_PATH = "/data/home/lihaozhe/llm-simulator/data/validation_report.csv"

def load_data():
    """从 validation_report.csv 加载数据，只取 TP=1 的单卡数据"""
    prefill_data = []
    decode_data = []
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row['Stage']
            tp = int(row['TP'])
            batch = int(row['Batch'])
            seq_len = int(row['Seq_Len']) if row['Seq_Len'] != '-' else 0
            kv_len = int(row['KV_Len']) if row['KV_Len'] != '-' else 0
            actual = float(row['Actual_ms'])
            predicted = float(row['Predicted_ms'])
            error_str = row['Error_%'].replace('%', '')
            error = float(error_str)
            
            # 只取单卡数据
            if tp != 1:
                continue
            if actual == 0 or predicted == 0:
                continue
            
            # 计算比值 (实际/预测)
            ratio = actual / predicted
            
            if stage == 'Prefill':
                workload = batch * seq_len
                prefill_data.append({
                    'batch': batch,
                    'seq_len': seq_len,
                    'workload': workload,
                    'actual': actual,
                    'predicted': predicted,
                    'ratio': ratio,
                    'error': error
                })
            elif stage == 'Decode':
                workload = batch * kv_len if kv_len > 0 else batch * seq_len
                decode_data.append({
                    'batch': batch,
                    'kv_len': kv_len,
                    'workload': workload,
                    'actual': actual,
                    'predicted': predicted,
                    'ratio': ratio,
                    'error': error
                })
    
    return prefill_data, decode_data

def analyze_ratio(data, stage_name):
    """分析比值随 workload 的变化"""
    if not data:
        print(f"\n⚠️ {stage_name} 无数据")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 {stage_name} 单卡 实际/预测 比值分析")
    print(f"{'='*70}")
    
    # 按 workload 排序
    sorted_data = sorted(data, key=lambda x: x['workload'])
    
    # 分组显示
    groups = defaultdict(list)
    for d in sorted_data:
        if d['workload'] < 100:
            groups['< 100'] = d
        elif d['workload'] < 500:
            groups['< 500'] = d
        elif d['workload'] < 1000:
            groups['< 1000'] = d
        elif d['workload'] < 2000:
            groups['< 2000'] = d
        elif d['workload'] < 5000:
            groups['< 5000'] = d
        else:
            groups['>= 5000'] = d
    
    # 打印每个数据点
    print(f"\n{'Workload':>10} | {'Batch':>6} | {'Seq/KV':>8} | {'Actual':>10} | {'Pred':>10} | {'Ratio':>8} | {'Error':>8}")
    print("-" * 80)
    for d in sorted_data:
        if 'seq_len' in d:
            seq_kv = d['seq_len']
        else:
            seq_kv = d['kv_len']
        print(f"{d['workload']:>10} | {d['batch']:>6} | {seq_kv:>8} | {d['actual']:>10.2f} | {d['predicted']:>10.2f} | {d['ratio']:>8.2f} | {d['error']:>7.1f}%")
    
    # 统计不同 workload 区间
    print(f"\n📈 按 workload 区间统计:")
    print(f"{'区间':>12} | {'样本数':>6} | {'平均比值':>10} | {'最小比值':>10} | {'最大比值':>10}")
    print("-" * 65)
    
    intervals = [
        (0, 100, '< 100'),
        (100, 500, '100-500'),
        (500, 1000, '500-1000'),
        (1000, 2000, '1000-2000'),
        (2000, 5000, '2000-5000'),
        (5000, float('inf'), '>= 5000')
    ]
    
    for low, high, label in intervals:
        filtered = [d for d in sorted_data if low <= d['workload'] < high]
        if filtered:
            ratios = [d['ratio'] for d in filtered]
            print(f"{label:>12} | {len(filtered):>6} | {sum(ratios)/len(ratios):>10.2f} | {min(ratios):>10.2f} | {max(ratios):>10.2f}")
    
    # 拟合指数衰减公式: ratio = 1 + (base - 1) * exp(-workload / tau)
    print(f"\n🧮 建议修正公式:")
    print("   ratio = 1 + (base - 1) * exp(-workload / tau)")
    
    # 从数据估算 base 和 tau
    # 用最小 workload 的比值估算 base
    min_workload = min(d['workload'] for d in sorted_data)
    max_workload = max(d['workload'] for d in sorted_data)
    
    # 取最小 workload 的比值作为 base 的近似
    min_ratio = min(d['ratio'] for d in sorted_data if d['workload'] == min_workload)
    # 取最大 workload 的比值作为稳定值
    max_ratio = max(d['ratio'] for d in sorted_data if d['workload'] == max_workload)
    
    # 如果数据足够，用最后几个点的平均值作为稳定值
    stable_ratio = 1.0
    large_workload_data = [d for d in sorted_data if d['workload'] >= 5000]
    if large_workload_data:
        stable_ratio = sum(d['ratio'] for d in large_workload_data) / len(large_workload_data)
    else:
        stable_ratio = max_ratio
    
    # 用第一个点的比值作为 base
    base_ratio = min_ratio
    
    # 估算 tau：当 ratio 降到 (base + stable) / 2 时的 workload
    target_ratio = (base_ratio + stable_ratio) / 2
    tau = None
    for d in sorted_data:
        if d['ratio'] <= target_ratio:
            tau = d['workload']
            break
    
    if tau is None:
        tau = (max_workload + min_workload) / 2
    
    print(f"   base ≈ {base_ratio:.2f} (小 workload 时)")
    print(f"   stable ≈ {stable_ratio:.2f} (大 workload 时)")
    print(f"   tau ≈ {tau:.0f} (衰减到中间值的 workload)")
    print(f"\n   💡 推荐参数: base={base_ratio:.2f}, tau={tau:.0f}, stable={stable_ratio:.2f}")

def main():
    print("🔍 分析验证数据...")
    prefill, decode = load_data()
    
    analyze_ratio(prefill, "Prefill")
    analyze_ratio(decode, "Decode")

if __name__ == "__main__":
    main()