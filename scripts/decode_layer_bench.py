#!/usr/bin/env python3
"""
整层 Decode Profiling - 测量完整的单层 Decode 前向传播
包含: LayerNorm → QKV → Attention (with KV Cache) → o_proj → 残差 → LayerNorm → FFN1 → SiLU → FFN2 → 残差

用法: python scripts/decode_layer_bench.py
"""

import torch
import torch.nn.functional as F
import json
import os

# ================= 1. Qwen3-32B 物理骨架 =================
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 80
NUM_LAYERS = 64

# ================= 2. 测试矩阵 =================
BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
KV_LENS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
# ================= 3. 计时器 =================
def benchmark(func, *args, runs=30, **kwargs):
    for _ in range(5):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(runs):
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    if len(times) > 2:
        times = sorted(times)[1:-1]
    return round(sum(times) / len(times), 4) if times else 0.0


# ================= 4. 单层 Decode 前向（完整，含 KV Cache） =================
def single_decode_layer(x, kv_cache_k, kv_cache_v):
    """
    完整的单层 Decode 前向
    包含: LN → QKV → Attention (with KV Cache) → o_proj → 残差 → LN → FFN1 → SiLU → FFN2 → 残差
    """
    batch, _, hidden = x.shape
    kv_len = kv_cache_k.shape[2]
    
    ln_weight = torch.ones(hidden, dtype=torch.float16, device='cuda')
    ln_bias = torch.zeros(hidden, dtype=torch.float16, device='cuda')
    
    # 1. LayerNorm
    x = F.layer_norm(x, (hidden,), ln_weight, ln_bias)
    
    # 2. QKV 投影
    w_qkv = torch.randn(hidden, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                        dtype=torch.float16, device='cuda')
    qkv = torch.matmul(x, w_qkv)  # [B, 1, (Q+2K)*head_dim]
    
    # 3. 拆分 Q/K/V
    q_dim = NUM_Q_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    
    q = qkv[:, :, :q_dim].view(batch, 1, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = qkv[:, :, q_dim:q_dim+kv_dim].view(batch, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = qkv[:, :, q_dim+kv_dim:q_dim+2*kv_dim].view(batch, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    
    # 4. 先扩头（KV Cache 和当前 token 的 K/V 都扩到 64 头）
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    
    # KV Cache 扩头: [B, 8, kv_len, 80] -> [B, 64, kv_len, 80]
    kv_cache_k_exp = kv_cache_k.repeat_interleave(rep, dim=1)
    kv_cache_v_exp = kv_cache_v.repeat_interleave(rep, dim=1)
    
    # 当前 token 的 K/V 扩头: [B, 8, 1, 80] -> [B, 64, 1, 80]
    k_exp = k.repeat_interleave(rep, dim=1)
    v_exp = v.repeat_interleave(rep, dim=1)
    
    # 5. 拼接 KV Cache 和当前 token
    k_full = torch.cat([kv_cache_k_exp, k_exp], dim=2)
    v_full = torch.cat([kv_cache_v_exp, v_exp], dim=2)
    
    # 6. FlashAttention (Decode, no causal)
    attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False, scale=1.0/(HEAD_DIM**0.5))
    
    # 7. o_proj
    attn = attn.transpose(1, 2).contiguous().view(batch, 1, NUM_Q_HEADS * HEAD_DIM)
    w_o = torch.randn(NUM_Q_HEADS * HEAD_DIM, hidden, dtype=torch.float16, device='cuda')
    attn_out = torch.matmul(attn, w_o)
    
    # 8. 残差连接
    x = x + attn_out
    
    # 9. LayerNorm
    x = F.layer_norm(x, (hidden,), ln_weight, ln_bias)
    
    # 10. FFN1 + SiLU
    w_ffn1 = torch.randn(hidden, INTERMEDIATE_SIZE, dtype=torch.float16, device='cuda')
    ffn1 = torch.matmul(x, w_ffn1)
    ffn1 = F.silu(ffn1)
    
    # 11. FFN2
    w_ffn2 = torch.randn(INTERMEDIATE_SIZE, hidden, dtype=torch.float16, device='cuda')
    ffn_out = torch.matmul(ffn1, w_ffn2)
    
    # 12. 残差连接
    x = x + ffn_out
    
    return x


# ================= 5. 主流程 =================
def main():
    print("🚀 开始生成整层 Decode 查找表...")
    print(f"   Batch 范围: {BATCH_SIZES}")
    print(f"   KV 范围: {KV_LENS}")
    print("-" * 60)
    
    lookup_table = []
    
    for batch in BATCH_SIZES:
        for kv_len in KV_LENS:
            print(f"🔄 测试: Batch={batch}, KV_Len={kv_len}")
            
            # 输入: [B, 1, H]
            x = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.float16, device='cuda')
            # KV Cache: [B, kv_heads, kv_len, head_dim]
            kv_cache_k = torch.randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=torch.float16, device='cuda')
            kv_cache_v = torch.randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=torch.float16, device='cuda')
            
            layer_time = benchmark(single_decode_layer, x, kv_cache_k, kv_cache_v, runs=20)
            
            lookup_table.append({
                "batch_size": batch,
                "kv_len": kv_len,
                "decode_layer_ms": layer_time,
                "num_layers": NUM_LAYERS,
                "estimated_full_time_ms": round(layer_time * NUM_LAYERS, 2)
            })
            
            print(f"   ✅ 单层: {layer_time:.4f} ms → 64层: {layer_time * NUM_LAYERS:.2f} ms")
    
            del x
            torch.cuda.empty_cache()
            
    output_path = "./qwen3_32b_decode_layer_lookup.json"
    with open(output_path, "w") as f:
        json.dump(lookup_table, f, indent=2)
    
    print(f"\n✅ 整层 Decode 查找表已保存至: {output_path}")
    print(f"   共 {len(lookup_table)} 条记录")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# 整层 Decode Profiling - 【使用真实 Qwen3-32B 权重】测量完整的单层 Decode 前向传播
# 包含: RMSNorm → QKV (分离) → Attention with KV Cache → o_proj → 残差 → RMSNorm → FFN (gate+up) → 残差
# 支持 OOM 自动跳过

# 用法: 
#     python scripts/decode_layer_bench.py --model-path /data/home/public/weight/Qwen3-32B
#     python scripts/decode_layer_bench.py --model-path /data/home/public/weight/Qwen3-32B --output-dir data/L20
# """

# import torch
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoConfig
# import json
# import os
# import argparse
# from tqdm import tqdm

# # ================= 1. 模型加载 =================
# def load_model(model_path: str):
#     print(f"🔄 加载真实模型: {model_path}")
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         config=config,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     model.eval()
#     return model, config

# # ================= 2. 测试矩阵 =================
# BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
# KV_LENS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]

# # ================= 3. 计时器 =================
# def benchmark(func, *args, runs=10, **kwargs):
#     for _ in range(3):
#         func(*args, **kwargs)
#     torch.cuda.synchronize()
    
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
    
#     times = []
#     for _ in range(runs):
#         start.record()
#         func(*args, **kwargs)
#         end.record()
#         torch.cuda.synchronize()
#         times.append(start.elapsed_time(end))
    
#     if len(times) > 2:
#         times = sorted(times)[1:-1]
#     return round(sum(times) / len(times), 4) if times else 0.0

# # ================= 4. 单层 Decode 前向（真实权重） =================
# def single_decode_layer_with_real_weights(x, kv_cache_k, kv_cache_v, layer, num_heads, num_kv_heads, head_dim):
#     """
#     使用真实权重执行单层 Decode 前向
#     x: [B, 1, H] 当前 token 的输入
#     kv_cache_k, kv_cache_v: [B, num_kv_heads, kv_len, head_dim] 已有的 KV Cache
#     """
#     batch, _, hidden = x.shape
#     kv_len = kv_cache_k.shape[2]
    
#     # 1. RMSNorm (前)
#     x = layer.input_layernorm(x)
    
#     # 2. QKV 投影
#     q = layer.self_attn.q_proj(x)  # [B, 1, num_heads * head_dim]
#     k = layer.self_attn.k_proj(x)  # [B, 1, num_kv_heads * head_dim]
#     v = layer.self_attn.v_proj(x)  # [B, 1, num_kv_heads * head_dim]
    
#     # 3. 重塑为多头格式
#     q = q.view(batch, 1, num_heads, head_dim).transpose(1, 2)          # [B, num_heads, 1, head_dim]
#     k = k.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)       # [B, num_kv_heads, 1, head_dim]
#     v = v.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)       # [B, num_kv_heads, 1, head_dim]
    
#     # 4. KV Cache 扩头（GQA）
#     rep = num_heads // num_kv_heads
#     # 扩头 KV Cache
#     kv_cache_k_exp = kv_cache_k.repeat_interleave(rep, dim=1)          # [B, num_heads, kv_len, head_dim]
#     kv_cache_v_exp = kv_cache_v.repeat_interleave(rep, dim=1)
#     # 扩头当前 token 的 K/V
#     k_exp = k.repeat_interleave(rep, dim=1)                            # [B, num_heads, 1, head_dim]
#     v_exp = v.repeat_interleave(rep, dim=1)
    
#     # 5. 拼接 KV Cache 和当前 token
#     k_full = torch.cat([kv_cache_k_exp, k_exp], dim=2)                 # [B, num_heads, kv_len+1, head_dim]
#     v_full = torch.cat([kv_cache_v_exp, v_exp], dim=2)
    
#     # 6. FlashAttention (Decode, no causal)
#     attn = F.scaled_dot_product_attention(
#         q, k_full, v_full,
#         is_causal=False,
#         scale=1.0 / (head_dim ** 0.5)
#     )
    
#     # 7. o_proj
#     attn = attn.transpose(1, 2).contiguous().view(batch, 1, num_heads * head_dim)
#     attn_out = layer.self_attn.o_proj(attn)
    
#     # 8. 残差
#     x = x + attn_out
    
#     # 9. RMSNorm (后)
#     x = layer.post_attention_layernorm(x)
    
#     # 10. FFN (gate + up)
#     gate = layer.mlp.gate_proj(x)
#     up = layer.mlp.up_proj(x)
#     ffn1 = F.silu(gate) * up
    
#     # 11. FFN down
#     ffn_out = layer.mlp.down_proj(ffn1)
    
#     # 12. 残差
#     x = x + ffn_out
    
#     return x

# # ================= 5. 主流程 =================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", required=True, help="真实 Qwen3-32B 权重路径")
#     parser.add_argument("--output-dir", default="/data/home/lihaozhe/llm-simulator/data", help="输出目录")
#     parser.add_argument("--layer-index", type=int, default=0, help="测试哪一层（默认第0层）")
#     args = parser.parse_args()
    
#     # 加载模型
#     model, config = load_model(args.model_path)
    
#     # 获取指定层
#     layer = model.model.layers[args.layer_index]
#     layer = layer.to(torch.float16).to('cuda')
    
#     # 从模型提取真实维度
#     actual_hidden_size = layer.input_layernorm.weight.shape[0]
#     q_out_features = layer.self_attn.q_proj.out_features
#     num_heads = config.num_attention_heads
#     num_kv_heads = config.num_key_value_heads
#     head_dim = q_out_features // num_heads
#     hidden_size = actual_hidden_size
#     num_layers = config.num_hidden_layers
    
#     print(f"📊 模型参数 (直接从模型层提取):")
#     print(f"   Hidden Size (实际): {hidden_size}")
#     print(f"   q_proj.out_features: {q_out_features}")
#     print(f"   Num Heads: {num_heads}")
#     print(f"   Num KV Heads: {num_kv_heads}")
#     print(f"   Head Dim: {head_dim}")
#     print(f"   Num Layers: {num_layers}")
    
#     output_dir = args.output_dir
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("\n🚀 开始生成真实权重整层 Decode 查找表...")
#     print(f"   Batch 范围: {BATCH_SIZES}")
#     print(f"   KV 范围: {KV_LENS}")
#     print(f"   测试层: Layer {args.layer_index} (共 {num_layers} 层)")
#     print(f"   输出目录: {output_dir}")
#     print(f"   ⚠️ 遇到 OOM 会自动跳过该配置")
#     print("-" * 60)
    
#     lookup_table = []
#     skipped_configs = []
#     total_configs = len(BATCH_SIZES) * len(KV_LENS)
#     pbar = tqdm(total=total_configs, desc="Profiling")
    
#     for batch in BATCH_SIZES:
#         for kv_len in KV_LENS:
#             try:
#                 # 创建输入：当前 token 的输入 [B, 1, H]
#                 x = torch.randn(batch, 1, hidden_size, dtype=torch.float16, device='cuda')
#                 # 创建 KV Cache
#                 kv_cache_k = torch.randn(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
#                 kv_cache_v = torch.randn(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16, device='cuda')
                
#                 def forward_fn():
#                     return single_decode_layer_with_real_weights(
#                         x, kv_cache_k, kv_cache_v, layer, num_heads, num_kv_heads, head_dim
#                     )
                
#                 layer_time = benchmark(forward_fn, runs=10)
                
#                 lookup_table.append({
#                     "batch_size": batch,
#                     "kv_len": kv_len,
#                     "decode_layer_ms": layer_time,
#                     "num_layers": num_layers,
#                     "estimated_full_time_ms": round(layer_time * num_layers, 2),
#                     "layer_index": args.layer_index,
#                     "model_path": args.model_path,
#                     "head_dim": head_dim,
#                     "num_heads": num_heads,
#                     "num_kv_heads": num_kv_heads,
#                     "hidden_size": hidden_size
#                 })
                
#                 # 清理显存
#                 del x, kv_cache_k, kv_cache_v
#                 torch.cuda.empty_cache()
                
#             except torch.cuda.OutOfMemoryError as e:
#                 print(f"\n⚠️ 跳过 Batch={batch}, KV={kv_len} (OOM)")
#                 skipped_configs.append({"batch": batch, "kv": kv_len, "error": str(e)})
#                 torch.cuda.empty_cache()
#                 continue
            
#             pbar.update(1)
    
#     pbar.close()
    
#     # 保存结果
#     output_path = os.path.join(output_dir, "qwen3_32b_decode_layer_lookup_real.json")
#     with open(output_path, "w") as f:
#         json.dump(lookup_table, f, indent=2)
    
#     if skipped_configs:
#         skip_path = os.path.join(output_dir, "skipped_configs_decode.json")
#         with open(skip_path, "w") as f:
#             json.dump(skipped_configs, f, indent=2)
#         print(f"\n⚠️ 跳过了 {len(skipped_configs)} 个配置（OOM），详见 {skip_path}")
    
#     print(f"\n✅ 真实权重整层 Decode 查找表已保存至: {output_path}")
#     print(f"   共 {len(lookup_table)} 条记录")


# if __name__ == "__main__":
#     main()