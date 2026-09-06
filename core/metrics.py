# core/metrics.py
import numpy as np

class MetricsCalculator:
    def __init__(self, sla_ttft_ms=500, sla_tpot_ms=100):
        self.sla_ttft = sla_ttft_ms
        self.sla_tpot = sla_tpot_ms

    def calculate_all(self, requests_data):
        """
        输入格式: List[Dict] containing 'ttft', 'tpot', 'total_time', 'input_tokens', 'output_tokens'
        """
        if not requests_data:
            return {}

        ttfts = [r['ttft'] for r in requests_data]
        tpots = [r['tpot'] for r in requests_data]
        
        # 基础指标
        avg_ttft = np.mean(ttfts)
        p99_ttft = np.percentile(ttfts, 99)
        avg_tpot = np.mean(tpots)
        
        # 吞吐量计算
        total_input = sum(r['prompt_tokens'] for r in requests_data)
        total_output = sum(r['output_tokens'] for r in requests_data)
        total_time_s = max(r['total_time'] for r in requests_data) / 1000.0
        
        input_throughput = total_input / total_time_s if total_time_s > 0 else 0
        output_throughput = total_output / total_time_s if total_time_s > 0 else 0

        # Goodput 计算 (满足SLA的请求占比)
        good_requests = [r for r in requests_data if r['ttft'] <= self.sla_ttft and r['tpot'] <= self.sla_tpot]
        goodput_tokens = sum(r['output_tokens'] for r in good_requests)
        goodput_ratio = len(good_requests) / len(requests_data)

        return {
            "avg_ttft_ms": round(avg_ttft, 2),
            "p99_ttft_ms": round(p99_ttft, 2),
            "avg_tpot_ms": round(avg_tpot, 2),
            "input_throughput": round(input_throughput, 2),
            "output_throughput": round(output_throughput, 2),
            "goodput_tokens": goodput_tokens,
            "goodput_ratio": round(goodput_ratio, 4)
        }