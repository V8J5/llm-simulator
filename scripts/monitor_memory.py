#!/usr/bin/env python3
"""
实时监控 GPU 显存占用，输出到 CSV 文件
用法：
    python monitor_memory.py --interval 0.1 --duration 300
    # 监控 300 秒，每 0.1 秒采样一次
"""

import time
import csv
import argparse
import sys
import os
from datetime import datetime

try:
    from pynvml import *
except ImportError:
    print("❌ 请先安装: pip install nvidia-ml-py3")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1, 
                        help="采样间隔 (秒)，默认 0.1")
    parser.add_argument("--duration", type=int, default=300,
                        help="监控时长 (秒)，默认 300")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径，默认自动生成")
    args = parser.parse_args()

    # 初始化 NVML
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    print(f"🔍 检测到 {device_count} 张 GPU")

    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]

    # 获取 GPU 名称
    gpu_names = []
    for h in handles:
        try:
            gpu_names.append(nvmlDeviceGetName(h).decode('utf-8'))
        except:
            gpu_names.append(f"GPU_{i}")

    # 生成输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/data/home/lihaozhe/llm-simulator/data/memory_trace_{timestamp}.csv"
    else:
        output_file = args.output

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 创建 CSV
    fieldnames = ["timestamp", "elapsed_sec"] + [f"gpu{i}_mem_mb" for i in range(device_count)]
    
    print(f"📁 输出文件: {output_file}")
    print(f"⏱️  采样间隔: {args.interval}s，监控时长: {args.duration}s")
    print("按 Ctrl+C 提前停止")
    print("=" * 60)

    start_time = time.time()

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > args.duration:
                    break

                row = {
                    "timestamp": time.time(),
                    "elapsed_sec": round(elapsed, 3)
                }

                for i, h in enumerate(handles):
                    info = nvmlDeviceGetMemoryInfo(h)
                    row[f"gpu{i}_mem_mb"] = round(info.used / 1024 / 1024, 2)  # MiB

                writer.writerow(row)
                f.flush()  # 立即写入

                # 每 10 秒打印一次状态
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    mems = [row[f"gpu{i}_mem_mb"] for i in range(device_count)]
                    print(f"[{elapsed:.1f}s] GPU 显存: {mems} MiB")

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        except Exception as e:
            print(f"❌ 错误: {e}")

    print(f"✅ 监控结束，数据已保存至: {output_file}")

    # 打印统计信息
    print("\n📊 统计信息:")
    print("-" * 60)
    
    # 读取并分析 CSV
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            for i in range(device_count):
                col = f"gpu{i}_mem_mb"
                values = [float(r[col]) for r in rows]
                print(f"GPU {i} (MiB): 最小 {min(values):.0f}, 最大 {max(values):.0f}, "
                      f"平均 {sum(values)/len(values):.0f}")
            
            # 找出峰值时间
            for i in range(device_count):
                col = f"gpu{i}_mem_mb"
                max_row = max(rows, key=lambda r: float(r[col]))
                print(f"GPU {i} 峰值: {max_row[col]} MiB @ {float(max_row['elapsed_sec']):.1f}s")

    nvmlShutdown()

if __name__ == "__main__":
    main()