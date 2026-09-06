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

# ================= 2. 测试矩阵配置 =================
# Decode 阶段的核心变量：Batch Size 和 KV Cache 长度
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
KV_LENGTHS = [128, 512, 1024, 2048, 4096]

# ================= 3. 核心测试工具 =================
def benchmark(func, *args, runs=50, **kwargs):
    # 预热
    for _ in range(10): func(*args, **kwargs)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs): func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    
    return round(start.elapsed_time(end) / runs, 4)

# ================= 4. 自动化遍历测试 =================
lookup_table = []
print("🚀 开始生成 Qwen3-32B Decode 阶段微基准测试查找表...")

for batch in BATCH_SIZES:
    for kv_len in KV_LENGTHS:
        print(f"🔄 正在测试: Batch={batch}, KV_Len={kv_len}")
        
        # 1. 测试 FFN GEMV (向量乘矩阵)
        # 形状推导: [Batch * 1, Hidden] x [Hidden, Intermediate]
        gemm_input = torch.randn(batch * 1, HIDDEN_SIZE, dtype=torch.float16, device='cuda')
        gemm_weight = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.float16, device='cuda')
        ffn_time = benchmark(torch.matmul, gemm_input, gemm_weight)
        
        # 2. 测试 Attention Decode (GQA 架构)
        # Q 的形状: [Batch, Num_Q_Heads, 1, Head_Dim]  <-- 注意 Seq=1
        q = torch.randn(batch, NUM_Q_HEADS, 1, HEAD_DIM, dtype=torch.float16, device='cuda')
        
        # KV 的形状: [Batch, Num_KV_Heads, KV_Len, Head_Dim]
        k = torch.randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=torch.float16, device='cuda')
        v = torch.randn(batch, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=torch.float16, device='cuda')
        
        # 扩展 KV 头数以匹配 Q
        k_exp = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
        v_exp = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
        
        # Decode 阶段不需要 Causal Mask
        attn_time = benchmark(F.scaled_dot_product_attention, q, k_exp, v_exp, is_causal=False)
        
        # 3. 记录数据
        lookup_table.append({
            "batch_size": batch,
            "kv_length": kv_len,
            "ffn_gemv_ms": ffn_time,
            "attention_decode_ms": attn_time
        })

# ================= 5. 保存为 JSON 查找表 =================
target_dir = "/data/home/lihaozhe/llm-simulator/data"
os.makedirs(target_dir, exist_ok=True)
output_path = os.path.join(target_dir, "qwen3_32b_decode_lookup_table.json")

with open(output_path, "w") as f:
    json.dump(lookup_table, f, indent=4)

print(f"\n✅ Decode 查找表生成完毕！已保存至: {output_path}")