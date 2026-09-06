#!/usr/bin/env python3
"""
没有用的代码


显存估算器 - 基于实测数据校准
校准配置: TP=4, Batch=8, Seq=4096, Prefill
实测峰值: 4480 MiB ≈ 4.48 GB
校准后 workspace 系数: 12.4
"""

class MemoryEstimator:
    def __init__(self, 
                 num_layers: int,
                 hidden_size: int,
                 num_kv_heads: int,
                 head_dim: int,
                 num_params: int = 32_000_000_000,
                 bytes_per_param: int = 2,
                 safety_margin: float = 0.10):
        """
        初始化显存估算器
        
        Args:
            num_layers: Transformer 层数
            hidden_size: 隐藏层维度
            num_kv_heads: KV 头数
            head_dim: 每个头的维度
            num_params: 模型参数量 (默认 32B)
            bytes_per_param: 每个参数的字节数 (FP16/BF16 = 2)
            safety_margin: 安全余量比例 (默认 10%)
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_params = num_params
        self.bytes_per_param = bytes_per_param
        self.safety_margin = safety_margin
    
    def estimate_weight_memory(self, tp_size: int) -> float:
        """
        估算模型权重显存 (MB)
        公式: 参数量 × 字节数 / TP
        """
        total_bytes = self.num_params * self.bytes_per_param # 模型参数量 x 每个参数的字节数
        per_rank_bytes = total_bytes / tp_size # 每个 rank 的显存占用
        return per_rank_bytes / (1024 * 1024)
    
    def estimate_kv_cache(self, batch_size: int, seq_len: int, tp_size: int) -> float:
        """
        估算 KV Cache 显存 (MB)
        公式: num_layers × batch × seq_len × kv_heads × head_dim × 2 (K+V) × bytes_per_param / TP
        """
        bytes_per_token = self.num_kv_heads * self.head_dim * 2 * self.bytes_per_param # 每个 token 的 KV Cache 大小 = kv_heads × head_dim × 2 (K+V) × bytes_per_param
        total_bytes = self.num_layers * batch_size * seq_len * bytes_per_token / tp_size # 总的 KV Cache = 层数 × batch × seq_len × 每 token 大小 / TP
        return total_bytes / (1024 * 1024)
    
    def estimate_workspace(self, batch_size: int, seq_len: int, stage: str = 'prefill') -> float:
        if stage == 'prefill':
            factor = 12.4
        else:
            factor = 4.0
        
        bytes_approx = batch_size * seq_len * self.hidden_size * factor * self.bytes_per_param
        workspace_mb = bytes_approx / (1024 * 1024)
        
        # ✅ 增加最小 workspace 下限（实测校准）
        min_workspace_mb = 4096  # 4GB，来自实测峰值
        return max(workspace_mb, min_workspace_mb)
        
    
    def estimate_total(self, batch_size: int, seq_len: int, tp_size: int, 
                    stage: str = 'prefill',
                    mode: str = 'test') -> dict:
        """
        估算总显存占用
        
        Args:
            batch_size: Batch 大小
            seq_len: 序列长度
            tp_size: Tensor Parallel 大小
            stage: 'prefill' 或 'decode'
            mode: 'test' - 测试模式（权重不计入，用于 generate_ground_truth 验证）
                'inference' - 真实推理模式（权重计入，用于硬件选型判断）
        """
        weight = self.estimate_weight_memory(tp_size)

        if stage == 'prefill':
            kv = 0.0
        else:
            kv = self.estimate_kv_cache(batch_size, seq_len, tp_size)

        workspace = self.estimate_workspace(batch_size, seq_len, stage)

        if mode == 'inference':
            actual_total = weight + workspace + kv
        else:
            actual_total = workspace + kv  # 测试模式，权重不计入

        total_with_margin = actual_total * (1 + self.safety_margin)

        return {
            "stage": stage,
            "mode": mode,
            "weight_mb": round(weight, 2),
            "kv_cache_mb": round(kv, 2),
            "workspace_mb": round(workspace, 2),
            "actual_total_mb": round(actual_total, 2),
            "total_with_margin_mb": round(total_with_margin, 2)
        }

        
    def can_fit(self, batch_size: int, seq_len: int, tp_size: int,
                stage: str = 'prefill',
                mode: str = 'test',
                gpu_memory_mb: int = 46068) -> bool:
        """
        判断是否能装入显存 (默认 L20: 46068 MiB ≈ 46 GB)
        """
        total = self.estimate_total(batch_size, seq_len, tp_size, stage, mode)
        return total["total_with_margin_mb"] < gpu_memory_mb


if __name__ == "__main__":
    # 测试配置: TP=4, Batch=8, Seq=4096, Prefill (与实测对比)
    estimator = MemoryEstimator(
        num_layers=64,
        hidden_size=5120,
        num_kv_heads=8,
        head_dim=80,
        num_params=32_000_000_000
    )
    
    print("=" * 60)
    print("🔍 显存估算验证 (TP=4, Batch=8, Seq=4096, Prefill)")
    print("=" * 60)
    
    result = estimator.estimate_total(
        batch_size=8, 
        seq_len=4096, 
        tp_size=4,
        stage='prefill'
    )
    
    for k, v in result.items():
        print(f"  {k}: {v} MB")
    
    print("\n" + "=" * 60)
    print("📊 与实测对比")
    print("=" * 60)
    print(f"  实测峰值: 4480 MiB ≈ 4.48 GB")
    print(f"  估算实际占用: {result['actual_total_mb']:.0f} MB ≈ {result['actual_total_mb']/1024:.2f} GB")
    print(f"  估算含余量: {result['total_with_margin_mb']:.0f} MB ≈ {result['total_with_margin_mb']/1024:.2f} GB")
    print(f"  误差: {abs(result['actual_total_mb'] - 4480) / 4480 * 100:.1f}%")
    
    can_fit = estimator.can_fit(8, 4096, 4, 'prefill')
    print(f"\n  能否装入 (L20 46GB): {can_fit}")
    
    print("\n" + "=" * 60)
    print("🔍 测试 Decode 阶段估算 (TP=4, Batch=8, Seq=4096)")
    print("=" * 60)
    
    result_decode = estimator.estimate_total(
        batch_size=8, 
        seq_len=4096, 
        tp_size=4,
        stage='decode'
    )
    
    for k, v in result_decode.items():
        print(f"  {k}: {v} MB")
    
    can_fit_decode = estimator.can_fit(8, 4096, 4, 'decode')
    print(f"\n  能否装入 (L20 46GB): {can_fit_decode}")