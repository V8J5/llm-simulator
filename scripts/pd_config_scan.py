#!/usr/bin/env python3
"""
PD 分离配置扫描工具（完善版）

遍历多种 P/D 配置，输出对比报告
用法: python scripts/pd_config_scan.py

修复内容：
1. 仿真时长从 10 秒改为 30 秒，确保所有请求都能完成
2. 增加 TP ≤ 卡数的合法性校验，自动跳过无效配置
3. 统一请求数为 20 个
4. 增加配置合法性检查的详细输出
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 添加 core 目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pd_simulator import PDSimulator, PDConfig
from core.qwen3_cost_model import LLMCostModel
from core.simulator import RequestGenerator


def is_config_valid(p_gpus: int, d_gpus: int, p_tp: int, d_tp: int) -> Tuple[bool, str]:
    """
    检查 PD 配置是否合法
    
    Returns:
        (is_valid, reason)
    """
    if p_tp > p_gpus:
        return False, f"Prefill TP({p_tp}) > Prefill 卡数({p_gpus})"
    if d_tp > d_gpus:
        return False, f"Decode TP({d_tp}) > Decode 卡数({d_gpus})"
    if p_gpus <= 0 or d_gpus <= 0:
        return False, "卡数必须大于 0"
    if p_tp <= 0 or d_tp <= 0:
        return False, "TP 必须大于 0"
    return True, "OK"


def run_single_config(pd_config: PDConfig, 
                      trace_file: str = None,  # 新增参数
                      sim_duration_ms: float = 30000.0) -> Dict[str, Any]:
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data/trace"
    cost_model = LLMCostModel(DATA_DIR, peak_tflops=119.5, mem_bw_gb_s=864.0)
    
    # 使用 trace 模式加载固定 trace
    generator = RequestGenerator(
        mode="trace",           # 改为 trace 模式
        trace_file=trace_file,  # 传入固定 trace 路径
        trace_repeat=False
    )
    
    simulator = PDSimulator(
        pd_config=pd_config,
        cost_model=cost_model,
        request_generator=generator,
        resource_checker=None,
        max_concurrent_requests=8,
        max_batch_token=4096
    )
    
    stats = simulator.run(
        simulation_duration_ms=sim_duration_ms,
        ttft_sla_ms=500.0,
        tpot_sla_ms=50.0
    )
    
    return stats


def scan_configs():
    """扫描多种 PD 配置"""
    
    # 定义要扫描的配置: (name, P卡数, D卡数, P_TP, D_TP)
    all_configs = [
        ("P2_D2_TP2", 2, 2, 2, 2),
        ("P2_D4_TP2_TP4", 2, 4, 2, 4),
        ("P4_D4_TP4", 4, 4, 4, 4),
        ("P4_D8_TP4_TP8", 4, 8, 4, 8),
        ("P2_D8_TP2_TP8", 2, 8, 2, 8),
        ("P4_D8_TP8_TP8", 4, 8, 8, 8),  # 此配置将在校验中被标记为无效
    ]
    
    # ========== 修复 2: 过滤掉无效配置 ==========
    valid_configs = []
    invalid_configs = []
    
    for name, p_gpus, d_gpus, p_tp, d_tp in all_configs:
        is_valid, reason = is_config_valid(p_gpus, d_gpus, p_tp, d_tp)
        if is_valid:
            valid_configs.append((name, p_gpus, d_gpus, p_tp, d_tp))
        else:
            invalid_configs.append((name, p_gpus, d_gpus, p_tp, d_tp, reason))
    
    results = []
    total_kv = 30000
    
    print("=" * 80)
    print("PD 分离配置扫描 (完善版)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总配置数: {len(all_configs)}")
    print(f"有效配置: {len(valid_configs)}")
    print(f"无效配置: {len(invalid_configs)}")
    
    # 打印无效配置
    if invalid_configs:
        print("\n⚠️ 以下配置因不合法将被跳过:")
        for name, p_gpus, d_gpus, p_tp, d_tp, reason in invalid_configs:
            print(f"  - {name}: {reason}")
    
    print("-" * 80)
    print(f"仿真时长: 30000ms (30秒)，确保所有请求完成")
    print(f"请求数量: 20 个 (固定间隔 100ms 到达)")
    print("-" * 80)
    
    # ========== 修复 1: 仿真时长统一为 30 秒 ==========
    SIM_DURATION_MS = 30000.0
    
    for idx, (name, p_gpus, d_gpus, p_tp, d_tp) in enumerate(valid_configs):
        print(f"\n[{idx+1}/{len(valid_configs)}] 运行配置: {name}")
        print(f"  P: {p_gpus} 卡, TP={p_tp}")
        print(f"  D: {d_gpus} 卡, TP={d_tp}")
        
        # KV 池按比例分配 (总 KV 30GB，P/D 按卡数比例分配)
        p_kv = int(total_kv * p_gpus / (p_gpus + d_gpus))
        d_kv = total_kv - p_kv
        
        pd_config = PDConfig(
            num_prefill_gpus=p_gpus,
            prefill_tp=p_tp,
            prefill_kv_memory_mb=p_kv,
            num_decode_gpus=d_gpus,
            decode_tp=d_tp,
            decode_kv_memory_mb=d_kv,
            kv_transfer_bw_gb_s=12.5,
            kv_transfer_latency_ms=0.1,
            kv_transfer_mode="pcie"  # 可改为 "nvlink" 或 "roce"
        )
        
        try:
            # stats = run_single_config(pd_config, mode="fixed", 
            #                           num_requests=20, 
            #                           sim_duration_ms=SIM_DURATION_MS)
            stats = run_single_config(
                pd_config, 
                trace_file="/data/home/lihaozhe/llm-simulator/data/trace/validation_trace.csv",
                sim_duration_ms=SIM_DURATION_MS
            )
            
            result = {
                "config_name": name,
                "p_gpus": p_gpus,
                "d_gpus": d_gpus,
                "p_tp": p_tp,
                "d_tp": d_tp,
                "p_kv_mb": p_kv,
                "d_kv_mb": d_kv,
                "total_requests": stats.get("total_requests", 0),
                "completed_requests": stats.get("completed_requests", 0),
                "ttft_avg": stats.get("ttft_ms", {}).get("avg", 0),
                "ttft_p50": stats.get("ttft_ms", {}).get("p50", 0),
                "ttft_p90": stats.get("ttft_ms", {}).get("p90", 0),
                "ttft_p99": stats.get("ttft_ms", {}).get("p99", 0),
                "tpot_avg": stats.get("tpot_ms", {}).get("avg", 0),
                "tpot_p50": stats.get("tpot_ms", {}).get("p50", 0),
                "tpot_p90": stats.get("tpot_ms", {}).get("p90", 0),
                "tpot_p99": stats.get("tpot_ms", {}).get("p99", 0),
                "throughput": stats.get("throughput_tokens_per_s", 0),
                "goodput_ttft": stats.get("goodput", {}).get("ttft_ratio", 0) * 100,
                "goodput_tpot": stats.get("goodput", {}).get("tpot_ratio", 0) * 100,
                "goodput_both": stats.get("goodput", {}).get("both_ratio", 0) * 100,
            }
            
            results.append(result)
            
            # 计算完成率
            complete_rate = f"{result['completed_requests']}/{result['total_requests']}"
            print(f"  ✅ 完成: {complete_rate} 个请求")
            print(f"  📊 Goodput (Both): {result['goodput_both']:.1f}%")
            print(f"  📊 吞吐: {result['throughput']:.1f} tokens/s")
            print(f"  📊 TTFT P50: {result['ttft_p50']:.1f} ms")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append({
                "config_name": name,
                "p_gpus": p_gpus,
                "d_gpus": d_gpus,
                "p_tp": p_tp,
                "d_tp": d_tp,
                "p_kv_mb": p_kv,
                "d_kv_mb": d_kv,
                "error": str(e)
            })
    
    return results, invalid_configs


def print_report(results: List[Dict[str, Any]], invalid_configs: List[Tuple]):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("📊 PD 分离配置扫描报告")
    print("=" * 80)
    
    # 如果有无效配置，先显示
    if invalid_configs:
        print("\n⚠️ 跳过的无效配置:")
        for name, p_gpus, d_gpus, p_tp, d_tp, reason in invalid_configs:
            print(f"  - {name}: {reason}")
        print()
    
    # 有效配置的结果
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("❌ 没有有效的配置结果")
        return
    
    # 表头
    print(f"\n{'配置':<16} {'P卡/D卡':<10} {'P_TP/D_TP':<12} {'Goodput':<10} {'吞吐':<12} {'TTFT P50':<12} {'TTFT P90':<12} {'TPOT avg':<12} {'完成率':<10}")
    print("-" * 120)
    
    for r in valid_results:
        complete_rate = f"{r['completed_requests']}/{r['total_requests']}"
        print(f"{r['config_name']:<16} {r['p_gpus']}/{r['d_gpus']:<7} {r['p_tp']}/{r['d_tp']:<9} {r['goodput_both']:<9.1f}% {r['throughput']:<11.1f} {r['ttft_p50']:<11.1f} {r['ttft_p90']:<11.1f} {r['tpot_avg']:<11.1f} {complete_rate:<10}")
    
    # 找出最优配置
    best_goodput = max(valid_results, key=lambda x: x['goodput_both'])
    best_throughput = max(valid_results, key=lambda x: x['throughput'])
    best_ttft = min(valid_results, key=lambda x: x['ttft_p50'])
    
    print("\n" + "=" * 80)
    print("🏆 最优配置")
    print("=" * 80)
    print(f"  最佳 Goodput: {best_goodput['config_name']} ({best_goodput['goodput_both']:.1f}%)")
    print(f"  最佳吞吐: {best_throughput['config_name']} ({best_throughput['throughput']:.1f} tokens/s)")
    print(f"  最佳 TTFT P50: {best_ttft['config_name']} ({best_ttft['ttft_p50']:.1f} ms)")
    
    # 保存结果
    output_dir = "/data/home/lihaozhe/llm-simulator/data/pd_config_scan"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"pd_config_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 保存时包含完整信息
    full_result = {
        "scan_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "sim_duration_ms": 30000,
        "num_requests": 20,
        "invalid_configs": [
            {"name": name, "p_gpus": p_gpus, "d_gpus": d_gpus, 
             "p_tp": p_tp, "d_tp": d_tp, "reason": reason}
            for name, p_gpus, d_gpus, p_tp, d_tp, reason in invalid_configs
        ],
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(full_result, f, indent=2)
    print(f"\n💾 结果已保存至: {output_file}")


def main():
    results, invalid_configs = scan_configs()
    print_report(results, invalid_configs)


if __name__ == "__main__":
    main()