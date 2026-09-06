#!/usr/bin/env python3
"""
在真实 L20 GPU 上用纯 PyTorch 模拟 Qwen3-32B 的完整 64 层前向传播，测量不同配置下的真实耗时，作为成本模型的 Ground Truth，输出格式与 verify_model.py 兼容。

用法（依次执行）:
    torchrun --nproc_per_node=1 scripts/generate_ground_truth.py
    torchrun --nproc_per_node=2 scripts/generate_ground_truth.py
    torchrun --nproc_per_node=4 scripts/generate_ground_truth.py
    torchrun --nproc_per_node=8 scripts/generate_ground_truth.py
"""

import os
import sys
import csv
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist

# ================= 1. Qwen3-32B 物理骨架 =================
HIDDEN_SIZE = 5120          # 模型隐藏层维度
INTERMEDIATE_SIZE = 25600   # FFN 中间层维度
NUM_Q_HEADS = 64            # Query 头数
NUM_KV_HEADS = 8            # Key/Value 头数 (GQA)
HEAD_DIM = 80   #这个数据不对。官方 Qwen3-32B 的 head_dim=128，但那样需要额外实现 o_proj 层，否则维度不匹配。用 80 可以让 64 × 80 = 5120 = HIDDEN_SIZE，省去 o_proj 层。
NUM_LAYERS = 64             # Transformer 层数

# # ================= 2. 测试矩阵 =================
# BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# PREFILL_SEQ_LENS = [128, 512, 1024, 2048, 4096]
# DECODE_KV_LENS = [128, 512, 1024, 2048, 4096]

BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]   # 网格点 + 非网格点
PREFILL_SEQ_LENS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]# 网格点 + 非网格点
DECODE_KV_LENS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]# 网格点 + 非网格点

# # 临时修改为只测一个配置
# BATCH_SIZES = [8]
# PREFILL_SEQ_LENS = [4096]
# DECODE_KV_LENS = []  # 先不测 Decode

# BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# PREFILL_SEQ_LENS = []              # ← 清空，不测 Prefill
# DECODE_KV_LENS = [128, 512, 1024, 2048, 4096]

# ================= 3. 分布式工具 =================
def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ================= 4. 基准测量函数 （计时器）=================
def benchmark_iteration(forward_fn, *args, runs=5, **kwargs):   # 用于精确测量 模型前向传播耗时 的基准测试（Benchmark）函数
    for _ in range(3):  # 预热（Warmup） 3 次，避免首次调用的额外开销影响测量
        forward_fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):   # 精确计时阶段
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward_fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    if len(times) > 2: # 剔除异常值并求平均
        times = sorted(times)[1:-1] # 将收集到的时间从小到大排序，然后去掉一个最快和一个最慢的值
    return sum(times) / len(times) if times else 0.0 # 最后，对剩下的时间求平均值并返回

# ================= 5. Prefill 前向 (每层 2 次 AllReduce) =================
def run_prefill_forward(rank, world_size, batch_size, seq_len):
    tp_size = world_size
    hidden = HIDDEN_SIZE
    inter = INTERMEDIATE_SIZE
    num_q_heads = NUM_Q_HEADS
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM

    x = torch.randn(batch_size, seq_len, hidden, dtype=torch.float16, device=f'cuda:{rank}')
    
    for _ in range(NUM_LAYERS):
        # ========== 1. QKV 投影（列切分）==========
        qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim // tp_size
        
        w_qkv = torch.randn(hidden, qkv_dim, dtype=torch.float16, device=f'cuda:{rank}')
        qkv = torch.matmul(x, w_qkv)

        q_dim = num_q_heads * head_dim // tp_size
        kv_dim = num_kv_heads * head_dim // tp_size
        
        q = qkv[:, :, :q_dim].view(batch_size, seq_len, num_q_heads // tp_size, head_dim).transpose(1, 2)
        k = qkv[:, :, q_dim:q_dim+kv_dim].view(batch_size, seq_len, num_kv_heads // tp_size, head_dim).transpose(1, 2)
        v = qkv[:, :, q_dim+kv_dim:q_dim+2*kv_dim].view(batch_size, seq_len, num_kv_heads // tp_size, head_dim).transpose(1, 2)

        # GQA 扩头
        if num_q_heads // tp_size > num_kv_heads // tp_size:
            rep = (num_q_heads // tp_size) // (num_kv_heads // tp_size)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # ========== 2. Attention ==========
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0/(head_dim**0.5))
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, num_q_heads // tp_size * head_dim)

        # ========== 3. AllGather 拼接各 rank 的注意力输出 ==========
        gather_list = [torch.zeros_like(attn) for _ in range(world_size)]
        dist.all_gather(gather_list, attn)
        attn_full = torch.cat(gather_list, dim=-1)  # [batch, seq, num_heads * head_dim]

        # ========== 4. FFN1（列切分，不通信）==========
        w_ffn1 = torch.randn(hidden, inter // tp_size, dtype=torch.float16, device=f'cuda:{rank}')
        ffn1_part = torch.matmul(attn_full, w_ffn1)  # [batch, seq, inter // tp]

        # ========== 5. FFN2（行切分，权重 (inter//tp, hidden)）==========
        w_ffn2 = torch.randn(inter // tp_size, hidden, dtype=torch.float16, device=f'cuda:{rank}')
        ffn2_part = torch.matmul(ffn1_part, w_ffn2)  # [batch, seq, hidden]

        # ========== 6. AllReduce 聚合 FFN2 输出 ==========
        dist.all_reduce(ffn2_part, op=dist.ReduceOp.SUM)
        x = ffn2_part

    torch.cuda.synchronize()
    return 0.0

# ================= 6. Decode 前向 =================
def run_decode_forward(rank, world_size, batch_size, kv_len):
    tp_size = world_size
    hidden = HIDDEN_SIZE
    inter = INTERMEDIATE_SIZE
    num_q_heads = NUM_Q_HEADS
    num_kv_heads = NUM_KV_HEADS
    head_dim = HEAD_DIM

    x = torch.randn(batch_size, 1, hidden, dtype=torch.float16, device=f'cuda:{rank}') #  在 Decode 阶段，每次推理的输入序列长度固定为 1
    
    for _ in range(NUM_LAYERS):
        # ========== 1. QKV 投影 ==========
        qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim // tp_size
        w_qkv = torch.randn(hidden, qkv_dim, dtype=torch.float16, device=f'cuda:{rank}')
        qkv = torch.matmul(x, w_qkv)

        q_dim = num_q_heads * head_dim // tp_size
        kv_dim = num_kv_heads * head_dim // tp_size

        q = qkv[:, :, :q_dim].view(batch_size, 1, num_q_heads // tp_size, head_dim).transpose(1, 2)
        k = qkv[:, :, q_dim:q_dim+kv_dim].view(batch_size, 1, num_kv_heads // tp_size, head_dim).transpose(1, 2)
        v = qkv[:, :, q_dim+kv_dim:q_dim+2*kv_dim].view(batch_size, 1, num_kv_heads // tp_size, head_dim).transpose(1, 2)

        # 模拟 KV Cache , 为了避免在生成每个新词时重复计算历史 token 的 K 和 V，推理引擎会将历史 K、V 缓存在显存中
        k_cache = torch.randn(batch_size, num_kv_heads // tp_size, kv_len, head_dim, dtype=torch.float16, device=f'cuda:{rank}')
        v_cache = torch.randn(batch_size, num_kv_heads // tp_size, kv_len, head_dim, dtype=torch.float16, device=f'cuda:{rank}')
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2) # 这里模拟了将当前 token 计算出的 K、V 与历史 k_cache、v_cache 在序列维度（dim=2）上进行拼接，得到完整的 KV 矩阵供注意力机制使用。

        # GQA 扩头
        if num_q_heads // tp_size > num_kv_heads // tp_size:
            rep = (num_q_heads // tp_size) // (num_kv_heads // tp_size)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # ========== 2. Attention ==========
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0/(head_dim**0.5)) # 在 Decode 阶段，Q 只有 1 个 token，而 K/V 包含了所有历史 token。当前 token 本来就可以关注所有历史 token，因此不需要因果掩码
        attn = attn.transpose(1, 2).contiguous().view(batch_size, 1, num_q_heads // tp_size * head_dim)

        # ========== 3. AllGather 拼接 ==========
        gather_list = [torch.zeros_like(attn) for _ in range(world_size)]
        dist.all_gather(gather_list, attn)
        attn_full = torch.cat(gather_list, dim=-1)

        # ========== 4. FFN1（列切分）==========
        w_ffn1 = torch.randn(hidden, inter // tp_size, dtype=torch.float16, device=f'cuda:{rank}')
        ffn1_part = torch.matmul(attn_full, w_ffn1)

        # ========== 5. FFN2（行切分）==========
        w_ffn2 = torch.randn(inter // tp_size, hidden, dtype=torch.float16, device=f'cuda:{rank}')
        ffn2_part = torch.matmul(ffn1_part, w_ffn2)

        # ========== 6. AllReduce 聚合 ==========
        dist.all_reduce(ffn2_part, op=dist.ReduceOp.SUM)
        x = ffn2_part

    torch.cuda.synchronize()
    return 0.0

# ================= 7. 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/home/lihaozhe/llm-simulator/data")
    args = parser.parse_args()

    rank, world_size = init_distributed()
    tp_size = world_size
    output_csv = os.path.join(args.output_dir, "ground_truth_pytorch.csv")
    
    file_exists = os.path.isfile(output_csv)
    
    if rank == 0:
        print(f"🚀 TP={tp_size}, 总卡数={world_size}")
        print(f"📁 输出: {output_csv}")

    dist.barrier()

    results = []

    # ---------- Prefill (output_tokens=1) ----------
    for batch in BATCH_SIZES:
        for seq_len in PREFILL_SEQ_LENS:  # 双重循环：遍历不同的 Batch Size 和 Prompt 长度（Seq Len）组合
            if rank == 0:
                print(f"⏳ Prefill: TP={tp_size}, Batch={batch}, Seq={seq_len}")
            
            # 调用 benchmark_iteration 对 run_prefill_forward 进行 5 次循环计时，返回平均耗时（毫秒）
            time_ms = benchmark_iteration(run_prefill_forward, rank, world_size, batch, seq_len, runs=5) 
            
            
            if rank == 0:
                results.append({
                    "tp_size": tp_size,
                    "batch_size": batch,
                    "prompt_tokens": seq_len,
                    "output_tokens": 1,           # 标记为 Prefill
                    "ttft_mean_ms": round(time_ms, 3),
                    "tpot_mean_ms": 0,
                    "kv_length": 0
                })
            dist.barrier()

    # ---------- Decode (output_tokens=128) ----------
    for batch in BATCH_SIZES:
        for kv_len in DECODE_KV_LENS:  # 双重循环：遍历不同的 Batch Size 和 KV Cache 长度（KV Len）组合。
            if rank == 0:
                print(f"⏳ Decode: TP={tp_size}, Batch={batch}, KV_Len={kv_len}")
            
            time_ms = benchmark_iteration(run_decode_forward, rank, world_size, batch, kv_len, runs=5)
            
            if rank == 0:
                results.append({
                    "tp_size": tp_size,
                    "batch_size": batch,
                    "prompt_tokens": 1,            # Decode 时 prompt_tokens 填 1
                    "output_tokens": 128,          # 标记为 Decode
                    "ttft_mean_ms": 0,
                    "tpot_mean_ms": round(time_ms, 3),
                    "kv_length": kv_len
                })
            dist.barrier()

    # ---------- 写入 CSV ----------
    if rank == 0:
        fieldnames = ["tp_size", "batch_size", "prompt_tokens", "output_tokens", 
                      "ttft_mean_ms", "tpot_mean_ms", "kv_length"]
        mode = 'a' if file_exists else 'w'
        with open(output_csv, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"✅ TP={tp_size} 已追加")

    cleanup_distributed()

if __name__ == "__main__":
    main()