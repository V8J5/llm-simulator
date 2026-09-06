#!/usr/bin/env python3
"""
资源约束检查器 - 判断 batch 能否运行，并分析瓶颈类型
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from kv_block_pool import KVBlockPool
from communication_model import CommunicationModel, Topology, CommType


class BottleneckType(Enum):
    """瓶颈类型"""
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    BALANCED = "balanced"


class FeasibilityStatus(Enum):
    """可行性状态"""
    OK = "ok"
    KV_CACHE_OOM = "kv_cache_oom"
    WORKSPACE_OOM = "workspace_oom"
    INVALID_CONFIG = "invalid_config"


@dataclass
class ResourceCheckResult:
    """资源检查结果"""
    feasible: bool
    status: FeasibilityStatus
    bottleneck: BottleneckType
    kv_blocks_needed: int
    kv_blocks_available: int
    compute_ratio: float
    comm_ratio: float
    memory_ratio: float
    details: Dict


class ResourceChecker:
    """
    资源约束检查器
    
    集成 KV Block 池和通信模型，判断给定 batch 能否运行，
    并分析瓶颈类型。
    """
    
    def __init__(self,
                 kv_pool: KVBlockPool,
                 comm_model: Optional[CommunicationModel] = None,
                 workspace_reserve_mb: int = 4096):
        """
        初始化资源检查器
        
        Args:
            kv_pool: KV Cache 资源池实例
            comm_model: 通信模型实例
            workspace_reserve_mb: workspace 预留显存 (MB)
        """
        self.kv_pool = kv_pool
        self.comm_model = comm_model or CommunicationModel()
        self.workspace_reserve_mb = workspace_reserve_mb
    
    def calculate_kv_blocks_needed(self, 
                                   batch_size: int, 
                                   seq_len: int,
                                   num_layers: int,
                                   num_kv_heads: int,
                                   head_dim: int,
                                   bytes_per_param: int = 2,
                                   tp_size: int = 1) -> int:
        """
        计算一个 batch 需要的 KV block 数量
        
        Args:
            batch_size: batch 大小
            seq_len: 序列长度
            num_layers: Transformer 层数
            num_kv_heads: KV 头数
            head_dim: 每个头的维度
            bytes_per_param: 每个参数的字节数
            tp_size: TP 大小
        
        Returns:
            需要的 KV block 数量
        """
        # 计算 KV Cache 总字节数
        bytes_per_token = num_kv_heads * head_dim * 2 * bytes_per_param / tp_size
        total_bytes = num_layers * batch_size * seq_len * bytes_per_token
        
        # 转换为 MB
        kv_mb = total_bytes / (1024 * 1024)
        
        # 计算需要的 block 数量 (向上取整)
        block_size_mb = self.kv_pool.block_size_mb
        num_blocks = int((kv_mb + block_size_mb - 1) // block_size_mb)
        
        return max(num_blocks, 1)
    
    def check_batch(self,
                    batch_size: int,
                    seq_len: int,
                    compute_time_ms: float,
                    comm_time_ms: float,
                    kv_blocks_needed: int,
                    tp_size: int = 1) -> ResourceCheckResult:
        """
        检查一个 batch 是否可运行
        
        Args:
            batch_size: batch 大小
            seq_len: 序列长度
            compute_time_ms: 计算时间 (ms)
            comm_time_ms: 通信时间 (ms)
            kv_blocks_needed: 需要的 KV block 数量
            tp_size: TP 大小
        
        Returns:
            资源检查结果
        """
        details = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tp_size": tp_size,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": comm_time_ms,
            "kv_blocks_needed": kv_blocks_needed,
        }
        
        # 1. 检查 KV Cache 是否足够
        kv_blocks_available = self.kv_pool.get_free_blocks()
        details["kv_blocks_available"] = kv_blocks_available
        
        if kv_blocks_needed > kv_blocks_available:
            return ResourceCheckResult(
                feasible=False,
                status=FeasibilityStatus.KV_CACHE_OOM,
                bottleneck=BottleneckType.MEMORY,
                kv_blocks_needed=kv_blocks_needed,
                kv_blocks_available=kv_blocks_available,
                compute_ratio=0,
                comm_ratio=0,
                memory_ratio=1.0,
                details=details
            )
        
        # 2. 计算各维度占比
        total_time = compute_time_ms + comm_time_ms
        compute_ratio = compute_time_ms / total_time if total_time > 0 else 0
        comm_ratio = comm_time_ms / total_time if total_time > 0 else 0
        
        # 显存占比：已分配 + 需要的
        used_memory = self.kv_pool.get_used_memory_mb()
        needed_memory = kv_blocks_needed * self.kv_pool.block_size_mb
        total_kv_memory = self.kv_pool.total_kv_memory_mb
        memory_ratio = (used_memory + needed_memory) / total_kv_memory if total_kv_memory > 0 else 0
        
        details["compute_ratio"] = compute_ratio
        details["comm_ratio"] = comm_ratio
        details["memory_ratio"] = memory_ratio
        details["used_kv_memory_mb"] = used_memory
        details["needed_kv_memory_mb"] = needed_memory
        
        # 3. 判断瓶颈
        if compute_ratio > 0.6 and comm_ratio < 0.3 and memory_ratio < 0.5:
            bottleneck = BottleneckType.COMPUTE
        elif comm_ratio > 0.5:
            bottleneck = BottleneckType.COMMUNICATION
        elif memory_ratio > 0.7:
            bottleneck = BottleneckType.MEMORY
        else:
            bottleneck = BottleneckType.BALANCED
        
        return ResourceCheckResult(
            feasible=True,
            status=FeasibilityStatus.OK,
            bottleneck=bottleneck,
            kv_blocks_needed=kv_blocks_needed,
            kv_blocks_available=kv_blocks_available,
            compute_ratio=compute_ratio,
            comm_ratio=comm_ratio,
            memory_ratio=memory_ratio,
            details=details
        )
    
    def simulate_allocate(self, batch_size: int, seq_len: int, 
                          num_layers: int, num_kv_heads: int, head_dim: int,
                          tp_size: int = 1) -> Tuple[bool, str, int, int]:
        """
        模拟分配：计算需要的 block 数量，并检查是否足够
        
        Returns:
            (can_allocate, message, needed, available)
        """
        needed = self.calculate_kv_blocks_needed(
            batch_size, seq_len, num_layers, num_kv_heads, head_dim, tp_size=tp_size
        )
        available = self.kv_pool.get_free_blocks()
        
        if needed <= available:
            return True, "OK", needed, available
        else:
            return False, f"需要 {needed} 个 block，可用 {available} 个", needed, available


# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始化资源池 (20GB = 20480 MB)
    pool = KVBlockPool(total_kv_memory_mb=20480, block_size_mb=64)
    checker = ResourceChecker(kv_pool=pool)
    
    # 模拟参数
    num_layers = 64
    num_kv_heads = 8
    head_dim = 80
    
    # 测试一个中等 batch
    batch_size = 8
    seq_len = 4096
    tp_size = 4
    
    # 计算需要的 block 数量
    needed = checker.calculate_kv_blocks_needed(
        batch_size, seq_len, num_layers, num_kv_heads, head_dim, tp_size=tp_size
    )
    print(f"Batch {batch_size} x Seq {seq_len}, TP={tp_size}")
    print(f"  需要的 KV blocks: {needed} ({needed * 64} MB)")
    print(f"  可用 blocks: {pool.get_free_blocks()}")
    
    # 模拟分配
    can, msg, needed, avail = checker.simulate_allocate(
        batch_size, seq_len, num_layers, num_kv_heads, head_dim, tp_size
    )
    print(f"  能否分配: {can}, {msg}")