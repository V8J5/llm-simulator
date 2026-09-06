#!/usr/bin/env python3
"""
分析校准因子，找出规律
"""

import json
import numpy as np

def main():
    with open("/data/home/lihaozhe/llm-simulator/data/layer_vs_gt_optimized.json", 'r') as f:
        data = json.load(f)
    
    details = data['details']
    
    # 按 TP 分组
    for tp in [1, 2, 4, 8]:
        tp_data = [d for d in details if d['tp'] == tp]
        if not tp_data:
            continue
        
        ratios = [d['gt_ms'] / d['total_ms'] for d in tp_data]
        
        print(f"\nTP={tp}")
        print(f"  样本数: {len(tp_data)}")
        print(f"  校准因子平均值: {np.mean(ratios):.4f}")
        print(f"  校准因子中位数: {np.median(ratios):.4f}")
        print(f"  校准因子 P90: {np.percentile(ratios, 90):.4f}")
        
        # 按 stage 细分
        for stage in ['Prefill', 'Decode']:
            stage_data = [d for d in tp_data if d['stage'] == stage]
            if not stage_data:
                continue
            stage_ratios = [d['gt_ms'] / d['total_ms'] for d in stage_data]
            print(f"    {stage}: 平均 {np.mean(stage_ratios):.4f}")

if __name__ == "__main__":
    main()