"""
    在多卡（1/2/4/8 卡）上，用纯 PyTorch + NCCL 测量不同数据量下 AllReduce 通信的耗时，生成成本模型的通信查找表。
"""

import torch
import torch.distributed as dist
import os
import json

# ================= 1. 分布式环境初始化 =================
# 注意：运行此脚本时需要使用 torchrun 启动多卡环境
# 例如: torchrun --nproc_per_node=8 /data/home/lihaozhe/llm-simulator/scripts/comm_micro_bench.py
def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()

# ================= 2. 通信测试配置 =================
# 模拟不同 Batch 和 Sequence Length 下，AllReduce 需要传输的数据量
# 公式: 元素数量 = Batch * Seq_Len * Hidden_Size
# 这里我们直接测试不同大小的数据块（单位：MB）
MSG_SIZES_MB = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256] #关键修改：包含了 0.01, 0.02, 0.05 等小消息点，覆盖 Decode 阶段的通信量（约 0.01~0.5 MB），也覆盖 Prefill 阶段的通信量（1~256 MB）

# ================= 3. 核心测试工具 =================
def benchmark_comm(op, tensor, runs=50):
    # 预热
    for _ in range(10): op(tensor)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs): op(tensor)  # 测量50次
    end.record()
    torch.cuda.synchronize()
    
    return round(start.elapsed_time(end) / runs, 4)

# ================= 4. 自动化遍历测试 =================
def run_comm_benchmark():
    rank, world_size = init_distributed()
    lookup_table = []
    
    if rank == 0:
        print(f"🚀 开始生成通信查找表 (当前卡数: {world_size})...")
        
    for size_mb in MSG_SIZES_MB:
        # 根据 MB 计算需要分配的 FP16 元素数量 (1 MB = 524288 个 FP16 元素)
        num_elements = int(size_mb * 524288)
        tensor = torch.randn(num_elements, dtype=torch.float16, device=f'cuda:{rank}') # 创建对应大小的 随机张量
        
        # 测试 AllReduce (TP 架构下最核心的通信原语)
        ar_time = benchmark_comm(lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM), tensor)
        
        # 记录数据
        lookup_table.append({
            "world_size": world_size,
            "msg_size_mb": size_mb,
            "allreduce_ms": ar_time
        })
        
        if rank == 0:
            print(f"🔄 消息大小: {size_mb} MB | AllReduce 耗时: {ar_time} ms")
            
    # 只在主进程保存文件
    if rank == 0:
        # 指定目标保存目录
        target_dir = "./"
        
        # 如果目录不存在，则自动创建（防止报错）
        os.makedirs(target_dir, exist_ok=True)
        
        # 拼接完整的文件路径
        output_path = os.path.join(target_dir, f"comm_lookup_table_tp{world_size}.json")
        
        with open(output_path, "w") as f:
            json.dump(lookup_table, f, indent=4)
            
        print(f"\n✅ 通信查找表已保存至: {output_path}")

if __name__ == "__main__":
    run_comm_benchmark()