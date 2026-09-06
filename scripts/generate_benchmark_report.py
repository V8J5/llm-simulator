#!/usr/bin/env python3
"""
标准化评测报告生成器

从 PD 配置扫描结果生成标准化的卡评测报告
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# 添加 core 目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scan_results(file_path: str) -> List[Dict]:
    """加载 PD 配置扫描结果"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_standard_cases(file_path: str) -> Dict:
    """加载标准 Case 配置"""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_report(scan_results: List[Dict], cases: Dict) -> Dict:
    """
    生成标准化评测报告
    
    报告结构:
    1. 环境信息
    2. 各配置下的性能指标
    3. 最优配置
    4. 评分（相对基准）
    5. 结论
    """
    
    # 评分规则
    # 对于每个配置，基于 Goodput、吞吐、TTFT 综合评分
    # 满分 10 分：Goodput 占 40%，吞吐占 30%，TTFT 占 30%
    
    report = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": "PD 分离评测框架 v1.0",
            "hardware": "L20",
            "model": "Qwen3-32B"
        },
        "configs": [],
        "ranking": [],
        "recommendations": []
    }
    
    valid_results = [r for r in scan_results if "error" not in r]
    
    # 计算各配置的评分
    for r in valid_results:
        # 计算综合评分
        goodput_score = r.get('goodput_both', 0) / 100  # 0-1
        throughput_score = min(r.get('throughput', 0) / 30, 1.0)  # 基准 30 tokens/s
        ttft_score = max(0, 1 - (r.get('ttft_p50', 500) / 500))  # 500ms 以内得满分
        
        # 权重: Goodput 40%, 吞吐 30%, TTFT 30%
        combined_score = goodput_score * 0.4 + throughput_score * 0.3 + ttft_score * 0.3
        score_10 = combined_score * 10
        
        config_report = {
            "name": r.get('config_name'),
            "hardware": "L20",
            "p_gpus": r.get('p_gpus'),
            "d_gpus": r.get('d_gpus'),
            "p_tp": r.get('p_tp'),
            "d_tp": r.get('d_tp'),
            "metrics": {
                "goodput": r.get('goodput_both', 0),
                "throughput_tokens_s": r.get('throughput', 0),
                "ttft_p50_ms": r.get('ttft_p50', 0),
                "ttft_p90_ms": r.get('ttft_p90', 0),
                "ttft_avg_ms": r.get('ttft_avg', 0),
                "tpot_avg_ms": r.get('tpot_avg', 0),
                "completed_requests": r.get('completed_requests', 0),
                "total_requests": r.get('total_requests', 0)
            },
            "score": {
                "total": round(score_10, 1),
                "goodput": round(goodput_score * 10, 1),
                "throughput": round(throughput_score * 10, 1),
                "ttft": round(ttft_score * 10, 1)
            },
            "summary": ""
        }
        
        report["configs"].append(config_report)
    
    # 按评分排序
    report["configs"].sort(key=lambda x: x["score"]["total"], reverse=True)
    
    # 生成排名
    for i, c in enumerate(report["configs"]):
        report["ranking"].append({
            "rank": i + 1,
            "config": c["name"],
            "score": c["score"]["total"],
            "goodput": c["metrics"]["goodput"],
            "throughput": c["metrics"]["throughput_tokens_s"]
        })
    
    # 生成推荐
    best = report["configs"][0] if report["configs"] else None
    if best:
        report["recommendations"].append({
            "type": "best_performance",
            "config": best["name"],
            "reason": f"综合评分 {best['score']['total']:.1f}/10，Goodput {best['metrics']['goodput']:.1f}%，吞吐 {best['metrics']['throughput_tokens_s']:.1f} tokens/s"
        })
    
    # 找出性价比最高的（2卡配置）
    two_card_configs = [c for c in report["configs"] if c.get("p_gpus", 0) + c.get("d_gpus", 0) <= 4]
    if two_card_configs:
        best_value = max(two_card_configs, key=lambda x: x["score"]["total"])
        report["recommendations"].append({
            "type": "best_value",
            "config": best_value["name"],
            "reason": f"仅需 {best_value.get('p_gpus',0) + best_value.get('d_gpus',0)} 卡，综合评分 {best_value['score']['total']:.1f}/10"
        })
    
    return report


def print_report(report: Dict):
    """打印报告到控制台"""
    print("\n" + "=" * 80)
    print("📊 推理卡标准化评测报告")
    print("=" * 80)
    print(f"生成时间: {report['meta']['generated_at']}")
    print(f"硬件: {report['meta']['hardware']}")
    print(f"模型: {report['meta']['model']}")
    print(f"工具: {report['meta']['tool']}")
    
    print("\n" + "-" * 80)
    print("📈 配置性能排行")
    print("-" * 80)
    
    # 表头
    print(f"\n{'排名':<6} {'配置':<18} {'Goodput':<12} {'吞吐(t/s)':<14} {'TTFT P50':<14} {'综合评分':<12}")
    print("-" * 80)
    
    for item in report["ranking"]:
        config = next((c for c in report["configs"] if c["name"] == item["config"]), None)
        if config:
            print(f"{item['rank']:<6} {item['config']:<18} {item['goodput']:<11.1f}% {item['throughput']:<13.1f} {config['metrics']['ttft_p50_ms']:<13.1f} {item['score']:<11.1f}")
    
    # 推荐
    print("\n" + "=" * 80)
    print("🏆 推荐配置")
    print("=" * 80)
    for rec in report["recommendations"]:
        if rec["type"] == "best_performance":
            print(f"\n  🚀 最佳性能: {rec['config']}")
        elif rec["type"] == "best_value":
            print(f"\n  💰 最佳性价比: {rec['config']}")
        print(f"     {rec['reason']}")
    
    # 详细指标
    print("\n" + "=" * 80)
    print("📋 详细指标")
    print("=" * 80)
    
    for config in report["configs"]:
        print(f"\n  {config['name']}:")
        print(f"    配置: P={config['p_gpus']}卡, TP={config['p_tp']}  |  D={config['d_gpus']}卡, TP={config['d_tp']}")
        print(f"    Goodput: {config['metrics']['goodput']:.1f}%")
        print(f"    吞吐: {config['metrics']['throughput_tokens_s']:.1f} tokens/s")
        print(f"    TTFT P50: {config['metrics']['ttft_p50_ms']:.1f} ms")
        print(f"    TTFT P90: {config['metrics']['ttft_p90_ms']:.1f} ms")
        print(f"    TPOT avg: {config['metrics']['tpot_avg_ms']:.1f} ms")
        print(f"    评分: {config['score']['total']:.1f}/10 (Goodput:{config['score']['goodput']:.1f}, 吞吐:{config['score']['throughput']:.1f}, TTFT:{config['score']['ttft']:.1f})")


def save_report(report: Dict, output_path: str):
    """保存报告到文件"""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 报告已保存至: {output_path}")


def main():
    # 加载扫描结果
    scan_dir = "/data/home/lihaozhe/llm-simulator/data"
    scan_files = [f for f in os.listdir(scan_dir) if f.startswith("pd_config_scan_") and f.endswith(".json")]
    
    if not scan_files:
        print("❌ 未找到 PD 配置扫描结果，请先运行 pd_config_scan.py")
        return
    
    # 使用最新的扫描结果
    latest_scan = sorted(scan_files)[-1]
    scan_path = os.path.join(scan_dir, latest_scan)
    print(f"📂 加载扫描结果: {latest_scan}")
    
    scan_results = load_scan_results(scan_path)
    
    # 加载标准 Case
    cases_path = "/data/home/lihaozhe/llm-simulator/data/standard_cases.json"
    if os.path.exists(cases_path):
        cases = load_standard_cases(cases_path)
        print(f"📂 加载标准 Case: {len(cases['cases'])} 个")
    else:
        print("⚠️ 未找到标准 Case 配置文件，使用默认配置")
        cases = {"cases": []}
    
    # 生成报告
    report = generate_report(scan_results, cases)
    
    # 打印报告
    print_report(report)
    
    # 保存报告
    output_path = os.path.join(scan_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_report(report, output_path)


if __name__ == "__main__":
    main()