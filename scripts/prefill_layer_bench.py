#!/usr/bin/env python3
"""
整层 Prefill Profiling - 完整单层 Transformer 前向传播
包含: LayerNorm → QKV → Attention → o_proj → 残差 → LayerNorm → FFN1 → SiLU → FFN2 → 残差

用法: python prefill_layer_bench.py
"""

import os
import torch
import torch.nn.functional as F
import json
import gc

# ============================================================
# 1. Qwen3-32B 物理骨架
# ============================================================
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 80
NUM_LAYERS = 64

# ============================================================
# 2. 测试矩阵（完整 10x9 = 90 配置，与 L20 保持一致）
# ============================================================
BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
SEQ_LENS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]


# ============================================================
# 3. 计时器（无显存清理）
# ============================================================
def benchmark(func, *args, runs=20, **kwargs):
    """
    精确测量函数执行时间（无显存清理）
    与 L20 原始测试脚本保持一致
    """
    # 预热
    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    if len(times) > 2:
        times = sorted(times)[1:-1]
    return round(sum(times) / len(times), 4) if times else 0.0


# ============================================================
# 4. 单层 Prefill 前向（完整，禁用梯度）
# ============================================================
def single_prefill_layer(x):
    """
    完整的单层 Prefill 前向
    包含: LN → QKV → Attention → o_proj → 残差 → LN → FFN1 → SiLU → FFN2 → 残差
    """
    with torch.no_grad():
        batch, seq_len, hidden = x.shape
        
        # 1. LayerNorm (前)
        ln_weight = torch.ones(hidden, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden, dtype=torch.float16, device='cuda')
        x = F.layer_norm(x, (hidden,), ln_weight, ln_bias)
        
        # 2. QKV 投影
        w_qkv = torch.randn(hidden, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                            dtype=torch.float16, device='cuda')
        qkv = torch.matmul(x, w_qkv)
        
        # 3. 拆分 Q/K/V
        q_dim = NUM_Q_HEADS * HEAD_DIM
        kv_dim = NUM_KV_HEADS * HEAD_DIM
        
        q = qkv[:, :, :q_dim].view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[:, :, q_dim:q_dim+kv_dim].view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[:, :, q_dim+kv_dim:q_dim+2*kv_dim].view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        # 4. GQA 扩头 (8 → 64)
        rep = NUM_Q_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        
        # 5. FlashAttention (Prefill, causal)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0/(HEAD_DIM**0.5))
        
        # 6. o_proj
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, NUM_Q_HEADS * HEAD_DIM)
        w_o = torch.randn(NUM_Q_HEADS * HEAD_DIM, hidden, dtype=torch.float16, device='cuda')
        attn_out = torch.matmul(attn, w_o)
        
        # 7. 残差连接
        x = x + attn_out
        
        # 8. LayerNorm (后)
        x = F.layer_norm(x, (hidden,), ln_weight, ln_bias)
        
        # 9. FFN1 (带 SiLU)
        w_ffn1 = torch.randn(hidden, INTERMEDIATE_SIZE, dtype=torch.float16, device='cuda')
        ffn1 = torch.matmul(x, w_ffn1)
        ffn1 = F.silu(ffn1)
        
        # 10. FFN2
        w_ffn2 = torch.randn(INTERMEDIATE_SIZE, hidden, dtype=torch.float16, device='cuda')
        ffn_out = torch.matmul(ffn1, w_ffn2)
        
        # 11. 残差连接
        x = x + ffn_out
        
        return x


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("🚀 开始生成整层 Prefill 查找表")
    print("=" * 60)
    print(f"   Batch 范围: {BATCH_SIZES}")
    print(f"   Seq 范围: {SEQ_LENS}")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-" * 60)
    
    lookup_table = []
    total_configs = len(BATCH_SIZES) * len(SEQ_LENS)
    completed = 0
    failed = 0
    
    for batch in BATCH_SIZES:
        for seq_len in SEQ_LENS:
            print(f"\n🔄 测试 [{completed+1}/{total_configs}]: Batch={batch}, Seq={seq_len}")
            
            try:
                x = torch.randn(batch, seq_len, HIDDEN_SIZE, dtype=torch.float16, device='cuda')
                
                layer_time = benchmark(single_prefill_layer, x, runs=20)
                
                lookup_table.append({
                    "batch_size": batch,
                    "seq_len": seq_len,
                    "prefill_layer_ms": layer_time,
                    "num_layers": NUM_LAYERS,
                    "estimated_full_time_ms": round(layer_time * NUM_LAYERS, 2),
                    "device": torch.cuda.get_device_name(0)
                })
                
                print(f"   ✅ 单层: {layer_time:.4f} ms → 64层: {layer_time * NUM_LAYERS:.2f} ms")
                completed += 1
                
                del x
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"   ❌ OOM: {e}")
                failed += 1
                lookup_table.append({
                    "batch_size": batch,
                    "seq_len": seq_len,
                    "prefill_layer_ms": -1,
                    "num_layers": NUM_LAYERS,
                    "estimated_full_time_ms": -1,
                    "device": torch.cuda.get_device_name(0),
                    "error": "OOM"
                })
                # 清理显存后继续
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            except Exception as e:
                print(f"   ❌ 错误: {e}")
                failed += 1
                torch.cuda.empty_cache()
                continue
    
    # 保存结果
    output_path = "./qwen3_32b_prefill_layer_lookup.json"
    with open(output_path, "w") as f:
        json.dump(lookup_table, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 Profiling 完成统计")
    print("=" * 60)
    print(f"   总配置数: {total_configs}")
    print(f"   成功: {completed}")
    print(f"   失败: {failed}")
    print(f"   成功率: {completed/total_configs*100:.1f}%")
    print(f"   📁 输出: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()