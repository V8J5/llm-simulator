#!/usr/bin/env python3
"""
生成模拟 Trace 文件

支持两种模式:
  1. 确定性模式 (--fixed): 固定参数，用于 Ground Truth 验证
  2. 随机模式 (默认): 模拟真实负载波动
"""

import csv
import random
import os
import argparse

def generate_fixed_trace(num_requests=20, 
                         interval_ms=100.0,
                         prompt_len=512,
                         output_len=128):
    """
    生成固定配置的 Trace（用于 Ground Truth 验证）
    """
    trace = []
    current_time = 0.0
    
    for i in range(num_requests):
        trace.append({
            'arrival_time_ms': current_time,
            'prompt_len': prompt_len,
            'output_len': output_len
        })
        current_time += interval_ms
    
    return trace

def generate_random_trace(num_requests=50, 
                          arrival_rate=0.01,      # 平均到达率 (1/ms)
                          prompt_range=(128, 2048),
                          output_range=(64, 256)):
    """
    生成随机 Trace（模拟真实负载波动）
    """
    trace = []
    current_time = 0.0
    
    for i in range(num_requests):
        interval = random.expovariate(arrival_rate)
        current_time += interval
        
        prompt_len = random.randint(prompt_range[0], prompt_range[1])
        output_len = random.randint(output_range[0], output_range[1])
        
        trace.append({
            'arrival_time_ms': current_time,
            'prompt_len': prompt_len,
            'output_len': output_len
        })
    
    return trace

def save_trace(trace, filename):
    """保存 trace 到 CSV"""
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arrival_time_ms', 'prompt_len', 'output_len'])
        writer.writeheader()
        writer.writerows(trace)
    
    print(f"✅ Trace 已保存至: {filename}")

def print_stats(trace, mode="随机"):
    arrivals = [t['arrival_time_ms'] for t in trace]
    print(f"\n📊 {mode} Trace 统计:")
    print(f"  总请求数: {len(trace)}")
    print(f"  总时长: {arrivals[-1]:.0f} ms")
    print(f"  平均到达间隔: {(arrivals[-1] - arrivals[0]) / len(trace):.1f} ms")
    print(f"  Prompt 长度: min={min(t['prompt_len'] for t in trace)}, max={max(t['prompt_len'] for t in trace)}")
    print(f"  Output 长度: min={min(t['output_len'] for t in trace)}, max={max(t['output_len'] for t in trace)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 LLM 仿真 Trace")
    parser.add_argument("--mode", choices=["fixed", "random"], default="fixed",
                        help="Trace 模式: fixed(确定性,用于验证) | random(随机,用于模拟)")
    parser.add_argument("--num", type=int, default=20,
                        help="请求数量")
    parser.add_argument("--interval", type=float, default=100.0,
                        help="到达间隔 (ms), 仅 fixed 模式有效")
    parser.add_argument("--prompt", type=int, default=512,
                        help="Prompt 长度, 仅 fixed 模式有效")
    parser.add_argument("--output", type=int, default=128,
                        help="Output 长度, 仅 fixed 模式有效")
    parser.add_argument("--output-path", type=str,
                        default="/data/home/lihaozhe/llm-simulator/data/trace/trace_example.csv",
                        help="输出文件路径")
    parser.add_argument("--arrival-rate", type=float, default=0.008,
                        help="到达率 (1/ms), 仅 random 模式有效")
    parser.add_argument("--prompt-range", type=int, nargs=2, default=[128, 4096],
                        help="Prompt 长度范围, 仅 random 模式有效")
    parser.add_argument("--output-range", type=int, nargs=2, default=[64, 256],
                        help="Output 长度范围, 仅 random 模式有效")
    
    args = parser.parse_args()
    
    print("📊 生成 Trace...")
    
    if args.mode == "fixed":
        trace = generate_fixed_trace(
            num_requests=args.num,
            interval_ms=args.interval,
            prompt_len=args.prompt,
            output_len=args.output
        )
        mode_label = "确定性 (验证用)"
    else:
        trace = generate_random_trace(
            num_requests=args.num,
            arrival_rate=args.arrival_rate,
            prompt_range=tuple(args.prompt_range),
            output_range=tuple(args.output_range)
        )
        mode_label = "随机 (模拟用)"
    
    save_trace(trace, args.output_path)
    print_stats(trace, mode_label)