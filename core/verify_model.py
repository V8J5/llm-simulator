import csv
import os
import sys
from collections import defaultdict

# 将 core 目录加入系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_estimator import MemoryEstimator
from qwen3_cost_model import LLMCostModel
from kv_block_pool import KVBlockPool
from communication_model import CommunicationModel
from resource_checker import ResourceChecker


def main():
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data"
    CSV_PATH = "/data/home/lihaozhe/llm-simulator/data/ground_truth_pytorch.csv"
    OUTPUT_CSV_PATH = "/data/home/lihaozhe/llm-simulator/data/validation_report.csv"

    # ============ 初始化各个模块 ============
    # 1. 显存估算器
    estimator = MemoryEstimator(
        num_layers=64,
        hidden_size=5120,
        num_kv_heads=8,
        head_dim=128,
        num_params=32_000_000_000
    )
    
    # 2. 成本模型
    engine = LLMCostModel(DATA_DIR)
    
    # 3. KV Cache 资源池。若能获取 vLLM metrics，应直接传入
    # num_gpu_blocks；该旧验证脚本只有显存预算，因此用 TP=4 反推。
    kv_pool = KVBlockPool.from_memory_budget(
        total_kv_memory_mib=20000,
        block_size_tokens=16,
        kv_bytes_per_token_per_rank=estimator.kv_bytes_per_token_per_rank(4),
    )
    
    # 4. 通信模型
    comm_model = CommunicationModel(
        node_internal_bandwidth_gb_s=600,   # NVLink
        node_external_bandwidth_gb_s=25,    # InfiniBand
        node_internal_latency_ms=0.1,
        node_external_latency_ms=2.0
    )
    
    # 5. 资源检查器
    checker = ResourceChecker(kv_pool=kv_pool, comm_model=comm_model)
    
    # ============ 统计变量 ============
    prefill_errors = []
    decode_errors = []
    report_data = []
    
    # 用于排序验证的数据结构
    prefill_sort_data = defaultdict(dict)
    decode_sort_data = defaultdict(dict)
    
    # 资源统计
    total_configs = 0
    feasible_configs = 0
    oom_configs = 0

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tp = int(row['tp_size'])
            batch = int(row['batch_size'])
            prompt = int(row['prompt_tokens'])
            output = int(row['output_tokens'])
            total_configs += 1
            
            # ============ Prefill 验证 ============
            if output == 1:
                # ----- 1. 显存估算（不再单独打印，改为只在误差>50%时显示） -----
                mem_info = estimator.estimate_total(batch, prompt, tp)
                
                # ----- 2. 资源检查 (KV Cache) -----
                needed = checker.calculate_kv_blocks_needed(
                    batch_size=batch,
                    seq_len=prompt,
                    num_layers=64,
                    num_kv_heads=8,
                    head_dim=128,
                    tp_size=tp
                )
                can_allocate, msg, needed, available = checker.simulate_allocate(
                    batch_size=batch,
                    seq_len=prompt,
                    num_layers=64,
                    num_kv_heads=8,
                    head_dim=128,
                    tp_size=tp
                )
                
                # ----- 3. 时间预测 -----
                actual = float(row['ttft_mean_ms'])
                if actual == 0:
                    continue
                
                pred = engine.predict_prefill(batch, prompt, tp)
                pred_time = pred['total_time_ms']
                err = abs(actual - pred_time) / actual * 100
                prefill_errors.append(err)
                
                # 存储用于排序验证
                prefill_sort_data[(batch, prompt)][tp] = actual
                
                # 统计资源可行性
                if can_allocate:
                    feasible_configs += 1
                    # 模拟分配和释放
                    kv_pool.allocate(f"prefill_{batch}_{prompt}_{tp}", needed)
                    kv_pool.release(f"prefill_{batch}_{prompt}_{tp}")
                else:
                    oom_configs += 1
                
                # ----- 4. 瓶颈分析 -----
                compute_time = pred['breakdown']['compute_ms']
                comm_time = pred['breakdown']['comm_ms']
                check_result = checker.check_batch(
                    batch_size=batch,
                    seq_len=prompt,
                    compute_time_ms=compute_time,
                    comm_time_ms=comm_time,
                    kv_blocks_needed=needed,
                    tp_size=tp
                )
                
                # ----- 5. 打印高误差（包含显存估算） -----
                if err > 50:
                    print(f"\n🚨 [Prefill 高误差] Batch={batch}, Seq={prompt}, TP={tp}")
                    print(f"   真实: {actual:.2f} ms | 预测: {pred_time:.2f} ms (误差: {err:.1f}%)")
                    print(f"   🔧 拆解 -> 计算: {pred['breakdown']['compute_ms']:.2f} ms | 通信: {pred['breakdown']['comm_ms']:.2f} ms")
                    print(f"   💾 显存估算: {mem_info['actual_total_mb']:.0f} MB / {mem_info['total_with_margin_mb']:.0f} MB (含余量)")
                    print(f"   📊 资源 -> 可行: {can_allocate} | 瓶颈: {check_result.bottleneck.value}")
                    print(f"   📦 KV块 -> 需要: {needed}, 可用: {available}")
                
                report_data.append({
                    "Stage": "Prefill",
                    "TP": tp,
                    "Batch": batch,
                    "Seq_Len": prompt,
                    "KV_Len": "-",
                    "Actual_ms": round(actual, 2),
                    "Predicted_ms": round(pred_time, 2),
                    "Error_%": f"{err:.2f}%",
                    "Feasible": can_allocate,
                    "KV_Blocks_Needed": needed,
                    "KV_Blocks_Available": available,
                    "Bottleneck": check_result.bottleneck.value,
                    "Compute_Ratio": f"{check_result.compute_ratio:.2f}",
                    "Comm_Ratio": f"{check_result.comm_ratio:.2f}",
                    "Memory_Ratio": f"{check_result.memory_ratio:.2f}"
                })
            
            # ============ Decode 验证 ============
            elif output == 128:
                kv_len = int(row.get('kv_length', prompt + output - 1))
                
                # ----- 1. 资源检查 (KV Cache) -----
                needed = checker.calculate_kv_blocks_needed(
                    batch_size=batch,
                    seq_len=kv_len,
                    num_layers=64,
                    num_kv_heads=8,
                    head_dim=128,
                    tp_size=tp
                )
                can_allocate, msg, needed, available = checker.simulate_allocate(
                    batch_size=batch,
                    seq_len=kv_len,
                    num_layers=64,
                    num_kv_heads=8,
                    head_dim=128,
                    tp_size=tp
                )
                
                # ----- 2. 时间预测 -----
                actual = float(row['tpot_mean_ms'])
                if actual == 0:
                    continue
                
                pred = engine.predict_decode(batch, kv_len, tp)
                pred_time = pred['total_time_ms']
                err = abs(actual - pred_time) / actual * 100
                decode_errors.append(err)
                
                # 存储用于排序验证
                decode_sort_data[(batch, kv_len)][tp] = actual
                
                # 统计资源可行性
                if can_allocate:
                    feasible_configs += 1
                    kv_pool.allocate(f"decode_{batch}_{kv_len}_{tp}", needed)
                    kv_pool.release(f"decode_{batch}_{kv_len}_{tp}")
                else:
                    oom_configs += 1
                
                # ----- 3. 瓶颈分析 -----
                compute_time = pred['breakdown']['compute_ms']
                comm_time = pred['breakdown']['comm_ms']
                check_result = checker.check_batch(
                    batch_size=batch,
                    seq_len=kv_len,
                    compute_time_ms=compute_time,
                    comm_time_ms=comm_time,
                    kv_blocks_needed=needed,
                    tp_size=tp
                )
                
                # ----- 4. 打印高误差（包含显存估算） -----
                if err > 50:
                    print(f"\n🚨 [Decode 高误差] Batch={batch}, KV_Len={kv_len}, TP={tp}")
                    print(f"   真实: {actual:.2f} ms | 预测: {pred_time:.2f} ms (误差: {err:.1f}%)")
                    print(f"   🔧 拆解 -> 计算: {pred['breakdown']['compute_ms']:.2f} ms | 通信: {pred['breakdown']['comm_ms']:.2f} ms")
                    print(f"   📊 资源 -> 可行: {can_allocate} | 瓶颈: {check_result.bottleneck.value}")
                    print(f"   📦 KV块 -> 需要: {needed}, 可用: {available}")
                
                report_data.append({
                    "Stage": "Decode",
                    "TP": tp,
                    "Batch": batch,
                    "Seq_Len": prompt,
                    "KV_Len": kv_len,
                    "Actual_ms": round(actual, 2),
                    "Predicted_ms": round(pred_time, 2),
                    "Error_%": f"{err:.2f}%",
                    "Feasible": can_allocate,
                    "KV_Blocks_Needed": needed,
                    "KV_Blocks_Available": available,
                    "Bottleneck": check_result.bottleneck.value,
                    "Compute_Ratio": f"{check_result.compute_ratio:.2f}",
                    "Comm_Ratio": f"{check_result.comm_ratio:.2f}",
                    "Memory_Ratio": f"{check_result.memory_ratio:.2f}"
                })

    # ============ 写入详细报告 ============
    if report_data:
        with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
            fieldnames = [
                "Stage", "TP", "Batch", "Seq_Len", "KV_Len",
                "Actual_ms", "Predicted_ms", "Error_%",
                "Feasible", "KV_Blocks_Needed", "KV_Blocks_Available",
                "Bottleneck", "Compute_Ratio", "Comm_Ratio", "Memory_Ratio"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_data)
        print(f"\n✅ 详细验证结果已保存至: {OUTPUT_CSV_PATH}")
    else:
        print("\n⚠️ 无有效验证数据")

    # ============ 资源统计 ============
    print("\n" + "="*60)
    print("📊 资源可行性统计")
    print("="*60)
    print(f"  总配置数: {total_configs}")
    print(f"  可行配置: {feasible_configs}")
    print(f"  显存不足: {oom_configs}")
    print(f"  可行性率: {feasible_configs/total_configs*100:.1f}%")

    # ============ 排序一致性检查 ============
    print("\n" + "="*60)
    print("📊 配置排序一致性检查 (TP Scaling 是否正确)")
    print("="*60)
    
    def check_sort_consistency(data_dict, stage_name):
        correct = 0
        total = 0
        for key, tp_map in data_dict.items():
            if len(tp_map) < 2:
                continue
            sorted_tps = sorted(tp_map.keys())
            actual_times = [tp_map[t] for t in sorted_tps]
            is_monotonic = all(actual_times[i] >= actual_times[i+1] * 0.9 for i in range(len(actual_times)-1))
            total += 1
            if is_monotonic:
                correct += 1
            else:
                print(f"   ⚠️ {stage_name} 排序异常: {key}, TPs={sorted_tps}, 耗时={actual_times}")
        if total > 0:
            print(f"   ✅ {stage_name} 排序一致性: {correct}/{total} ({correct/total*100:.1f}%)")
        else:
            print(f"   ⚠️ {stage_name} 无足够数据检查排序")

    check_sort_consistency(prefill_sort_data, "Prefill")
    check_sort_consistency(decode_sort_data, "Decode")

    # ============ MAPE 汇总 ============
    print("\n" + "="*60)
    print("🎯 MAPE (平均绝对百分比误差) 汇总")
    print("="*60)
    if prefill_errors:
        print(f"Prefill TTFT MAPE: {sum(prefill_errors)/len(prefill_errors):.2f}% ({len(prefill_errors)} 个样本)")
    else:
        print("Prefill: 无有效数据点")
        
    if decode_errors:
        print(f"Decode TPOT MAPE: {sum(decode_errors)/len(decode_errors):.2f}% ({len(decode_errors)} 个样本)")
    else:
        print("Decode: 无有效数据点")


if __name__ == "__main__":
    main()
