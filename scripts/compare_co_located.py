#!/usr/bin/env python3
"""
对比混部仿真器输出与真实 vLLM 数据
用法: python scripts/compare_co_located.py
"""

import csv
import json
import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = "/data/home/lihaozhe/llm-simulator"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "ground_truth_co_located", "co_located_ground_truth_latest.csv")
SIMULATION_OUTPUT_FILE = os.path.join(DATA_DIR, "simulation_results", "co_located_sim_output.csv")

# 确保目录存在
os.makedirs(os.path.dirname(SIMULATION_OUTPUT_FILE), exist_ok=True)


def load_ground_truth():
    """加载真实 vLLM 数据"""
    results = []
    with open(GROUND_TRUTH_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'success':
                results.append({
                    "request_id": row['request_id'],
                    "arrival_time_ms": float(row['arrival_time_ms']),
                    "ttft_ms": float(row['ttft_ms']),
                    "tpot_ms": float(row['tpot_ms']),
                    "e2e_ms": float(row['e2e_ms'])
                })
    return results


def load_simulation():
    """加载仿真器输出（先检查是否存在）"""
    if not os.path.exists(SIMULATION_OUTPUT_FILE):
        print(f"⚠️ 仿真输出文件不存在: {SIMULATION_OUTPUT_FILE}")
        print("💡 请先运行 simulator.py 并保存结果")
        return None
    
    results = []
    with open(SIMULATION_OUTPUT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "request_id": row['request_id'],
                "arrival_time_ms": float(row['arrival_time_ms']),
                "ttft_ms": float(row['ttft_ms']),
                "tpot_ms": float(row['tpot_ms']),
                "e2e_ms": float(row['e2e_ms'])
            })
    return results


def calculate_errors(gt, sim):
    """计算逐条误差"""
    # 按 request_id 匹配
    gt_dict = {r['request_id']: r for r in gt}
    sim_dict = {r['request_id']: r for r in sim}
    
    common_ids = set(gt_dict.keys()) & set(sim_dict.keys())
    
    errors = []
    for rid in common_ids:
        g = gt_dict[rid]
        s = sim_dict[rid]
        
        ttft_err = (s['ttft_ms'] - g['ttft_ms']) / g['ttft_ms'] * 100 if g['ttft_ms'] > 0 else 0
        tpot_err = (s['tpot_ms'] - g['tpot_ms']) / g['tpot_ms'] * 100 if g['tpot_ms'] > 0 else 0
        e2e_err = (s['e2e_ms'] - g['e2e_ms']) / g['e2e_ms'] * 100 if g['e2e_ms'] > 0 else 0
        
        errors.append({
            "request_id": rid,
            "gt_ttft": g['ttft_ms'],
            "sim_ttft": s['ttft_ms'],
            "ttft_error_pct": ttft_err,
            "gt_tpot": g['tpot_ms'],
            "sim_tpot": s['tpot_ms'],
            "tpot_error_pct": tpot_err,
            "gt_e2e": g['e2e_ms'],
            "sim_e2e": s['e2e_ms'],
            "e2e_error_pct": e2e_err
        })
    
    return errors


def percentile(data, p):
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def print_report(errors, gt, sim):
    """打印统计报告"""
    print("\n" + "=" * 70)
    print("📊 混部仿真 vs 真实 vLLM 对比报告")
    print("=" * 70)
    
    print(f"\n📌 基础统计:")
    print(f"  真实 vLLM 成功请求数: {len(gt)}")
    print(f"  仿真器成功请求数: {len(sim)}")
    print(f"  匹配到的请求数: {len(errors)}")
    
    if not errors:
        print("\n⚠️ 没有匹配到任何请求，请检查 request_id 是否一致")
        return
    
    # TTFT 误差统计
    ttft_errors = [e['ttft_error_pct'] for e in errors]
    tpot_errors = [e['tpot_error_pct'] for e in errors]
    e2e_errors = [e['e2e_error_pct'] for e in errors]
    
    print("\n" + "-" * 70)
    print("🎯 TTFT 误差分析 (正=仿真偏大, 负=仿真偏小):")
    print(f"  MAPE (平均绝对百分比误差): {sum(abs(e) for e in ttft_errors)/len(ttft_errors):.2f}%")
    print(f"  平均误差: {sum(ttft_errors)/len(ttft_errors):.2f}%")
    print(f"  P50 误差: {percentile(ttft_errors, 50):.2f}%")
    print(f"  P90 误差: {percentile(ttft_errors, 90):.2f}%")
    print(f"  P95 误差: {percentile(ttft_errors, 95):.2f}%")
    print(f"  最大误差: {max(ttft_errors):.2f}%")
    print(f"  最小误差: {min(ttft_errors):.2f}%")
    
    print("\n" + "-" * 70)
    print("🎯 TPOT 误差分析 (正=仿真偏大, 负=仿真偏小):")
    print(f"  MAPE (平均绝对百分比误差): {sum(abs(e) for e in tpot_errors)/len(tpot_errors):.2f}%")
    print(f"  平均误差: {sum(tpot_errors)/len(tpot_errors):.2f}%")
    print(f"  P50 误差: {percentile(tpot_errors, 50):.2f}%")
    print(f"  P90 误差: {percentile(tpot_errors, 90):.2f}%")
    print(f"  P95 误差: {percentile(tpot_errors, 95):.2f}%")
    print(f"  最大误差: {max(tpot_errors):.2f}%")
    print(f"  最小误差: {min(tpot_errors):.2f}%")
    
    print("\n" + "-" * 70)
    print("🎯 E2E 端到端误差分析:")
    print(f"  MAPE (平均绝对百分比误差): {sum(abs(e) for e in e2e_errors)/len(e2e_errors):.2f}%")
    print(f"  平均误差: {sum(e2e_errors)/len(e2e_errors):.2f}%")
    print(f"  P50 误差: {percentile(e2e_errors, 50):.2f}%")
    print(f"  P90 误差: {percentile(e2e_errors, 90):.2f}%")
    
    # 打印误差最大的 5 条
    sorted_by_ttft = sorted(errors, key=lambda x: abs(x['ttft_error_pct']), reverse=True)
    print("\n" + "-" * 70)
    print("🔍 TTFT 误差最大的 5 条请求:")
    print(f"{'Request ID':<15} {'GT TTFT':<12} {'Sim TTFT':<12} {'误差':<10}")
    for e in sorted_by_ttft[:5]:
        print(f"{e['request_id']:<15} {e['gt_ttft']:<12.2f} {e['sim_ttft']:<12.2f} {e['ttft_error_pct']:<10.1f}%")
    
    # 真实 vs 仿真的 P50/P90 对比
    gt_ttfts = [e['gt_ttft'] for e in errors]
    sim_ttfts = [e['sim_ttft'] for e in errors]
    gt_tpots = [e['gt_tpot'] for e in errors]
    sim_tpots = [e['sim_tpot'] for e in errors]
    
    print("\n" + "-" * 70)
    print("📊 分位数对比:")
    print(f"{'指标':<15} {'真实 P50':<12} {'仿真 P50':<12} {'真实 P90':<12} {'仿真 P90':<12}")
    print(f"{'TTFT (ms)':<15} {percentile(gt_ttfts, 50):<12.2f} {percentile(sim_ttfts, 50):<12.2f} {percentile(gt_ttfts, 90):<12.2f} {percentile(sim_ttfts, 90):<12.2f}")
    print(f"{'TPOT (ms)':<15} {percentile(gt_tpots, 50):<12.2f} {percentile(sim_tpots, 50):<12.2f} {percentile(gt_tpots, 90):<12.2f} {percentile(sim_tpots, 90):<12.2f}")
    
    # 保存详细误差到 CSV
    detail_file = os.path.join(DATA_DIR, "simulation_results", "co_located_error_detail.csv")
    os.makedirs(os.path.dirname(detail_file), exist_ok=True)
    with open(detail_file, 'w') as f:
        fieldnames = ["request_id", "gt_ttft", "sim_ttft", "ttft_error_pct", 
                      "gt_tpot", "sim_tpot", "tpot_error_pct",
                      "gt_e2e", "sim_e2e", "e2e_error_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    print(f"\n💾 详细误差已保存: {detail_file}")
    
    print("\n" + "=" * 70)


def main():
    print("📋 加载真实 vLLM 数据...")
    gt_data = load_ground_truth()
    print(f"   ✅ 加载了 {len(gt_data)} 条成功请求")
    
    print("📋 加载仿真器输出...")
    sim_data = load_simulation()
    if sim_data is None:
        print("\n🚨 请先运行仿真器并保存输出!")
        print("💡 提示: 在 simulator.py 的 demo_simulation 中添加保存 CSV 的逻辑")
        return
    
    print(f"   ✅ 加载了 {len(sim_data)} 条仿真结果")
    
    errors = calculate_errors(gt_data, sim_data)
    print_report(errors, gt_data, sim_data)


if __name__ == "__main__":
    main()