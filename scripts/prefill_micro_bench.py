"""
    生成 Prefill 阶段的微基准查找表
    在单卡 L20 上，用纯 PyTorch 测量不同 batch size 和 prompt length 下，单层 Attention 和单层 FFN 的纯算子耗时，生成成本模型的查找表。
    
    微基准（本代码）测的是单层时间，成本模型将其乘以 64 层 + 通信时间，得到端到端预测。
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
NUM_LAYERS = 64 # 虽然定义了 NUM_LAYERS = 64，但这里没有循环 64 层，而是只测单层。64 层的累加是在 qwen3_cost_model.py 中做的

# ================= 2. 测试矩阵配置 =================
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]

# ================= 3. 核心测试工具（计时器） =================
def benchmark(func, *args, runs=50, **kwargs): # 预热 10 次，测量 50 次，取平均值。
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

print("🚀 开始生成 Qwen3-32B 微基准测试查找表...")
for batch in BATCH_SIZES:
    for seq_len in PROMPT_LENGTHS:
        print(f"🔄 正在测试: Batch={batch}, Seq_Len={seq_len}")
        
        # 1. 测试 FFN GEMM
        gemm_input = torch.randn(batch * seq_len, HIDDEN_SIZE, dtype=torch.float16, device='cuda')
        gemm_weight = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.float16, device='cuda')
        ffn_time = benchmark(torch.matmul, gemm_input, gemm_weight)
        
        # 2. 测试 Attention Prefill (GQA)
        q = torch.randn(batch, NUM_Q_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device='cuda')
        k = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device='cuda')
        v = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device='cuda')

        # GQA 扩头：把 8 个 KV 头复制 8 次，变成 64 个
        k_exp = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
        v_exp = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)

        attn_time = benchmark(F.scaled_dot_product_attention, q, k_exp, v_exp, is_causal=True)
        
        # 3. 记录数据
        lookup_table.append({
            "batch_size": batch,
            "prompt_length": seq_len,
            "ffn_gemm_ms": ffn_time,
            "attention_prefill_ms": attn_time
        })

# ================= 5. 保存为 JSON 查找表 =================
output_path = "/data/home/lihaozhe/llm-simulator/data/qwen3_32b_prefill_lookup_table.json"
with open(output_path, "w") as f:
    json.dump(lookup_table, f, indent=4)

print(f"\n✅ 查找表生成完毕！已保存至: {os.path.abspath(output_path)}")