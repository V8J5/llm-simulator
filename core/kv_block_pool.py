#!/usr/bin/env python3
"""
KV Cache 资源池 - 管理显存中 KV Cache 的分配和释放
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class KVBlock:
    """单个 KV Block 的元数据"""
    block_id: int
    request_id: Optional[str] = None
    allocated: bool = False


class KVBlockPool:
    """
    KV Cache 资源池
    
    核心概念：显存被划分为固定大小的 KV Block，每个请求占用若干个 block。
    这与 vLLM 的 PagedAttention 思想一致，但简化了实现。
    """
    
    def __init__(self, 
                 total_kv_memory_mb: int,
                 block_size_mb: int = 64):
        """
        初始化 KV Cache 资源池
        
        Args:
            total_kv_memory_mb: KV Cache 可用总显存 (MB)
            block_size_mb: 每个 KV Block 的大小 (MB)，默认 64MB
        """
        self.total_kv_memory_mb = total_kv_memory_mb
        self.block_size_mb = block_size_mb
        self.total_blocks = total_kv_memory_mb // block_size_mb
        
        # 初始化所有 block
        self.blocks: List[KVBlock] = [
            KVBlock(block_id=i) for i in range(self.total_blocks)
        ]
        
        # 记录每个请求占用的 block IDs
        self.request_blocks: Dict[str, List[int]] = {}
        
        # 统计信息
        self.allocated_blocks = 0
        self.free_blocks = self.total_blocks
        
        print(f"✅ KVBlockPool 初始化: {self.total_blocks} blocks, "
              f"每块 {self.block_size_mb} MB, 总计 {total_kv_memory_mb} MB")
    
    def get_free_blocks(self) -> int:
        """获取当前可用的 block 数量"""
        return self.free_blocks
    
    def get_used_blocks(self) -> int:
        """获取已分配的 block 数量"""
        return self.allocated_blocks
    
    def get_total_blocks(self) -> int:
        """获取总 block 数量"""
        return self.total_blocks
    
    def get_available_memory_mb(self) -> int:
        """获取当前可用显存 (MB)"""
        return self.free_blocks * self.block_size_mb
    
    def get_used_memory_mb(self) -> int:
        """获取已分配显存 (MB)"""
        return self.allocated_blocks * self.block_size_mb
    
    def get_utilization(self) -> float:
        """获取显存利用率 (0-1)"""
        return self.allocated_blocks / self.total_blocks if self.total_blocks > 0 else 0
    
    def can_allocate(self, num_blocks: int) -> bool:
        """判断是否能分配指定数量的 block"""
        return self.free_blocks >= num_blocks
    
    def allocate(self, request_id: str, num_blocks: int) -> bool:
        """
        为请求分配 KV Cache
        
        Args:
            request_id: 请求 ID
            num_blocks: 需要的 block 数量
        
        Returns:
            是否分配成功
        """
        if not self.can_allocate(num_blocks):
            return False
        
        # 找到空闲的 block
        allocated_block_ids = []
        for block in self.blocks:
            if not block.allocated and len(allocated_block_ids) < num_blocks:
                block.allocated = True
                block.request_id = request_id
                allocated_block_ids.append(block.block_id)
        
        # 更新状态
        self.request_blocks[request_id] = allocated_block_ids
        self.allocated_blocks += num_blocks
        self.free_blocks -= num_blocks
        
        return True
    
    def release(self, request_id: str) -> bool:
        """
        释放请求占用的 KV Cache
        
        Args:
            request_id: 请求 ID
        
        Returns:
            是否释放成功
        """
        if request_id not in self.request_blocks:
            return False
        
        block_ids = self.request_blocks[request_id]
        num_blocks = len(block_ids)
        
        # 释放 block
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.allocated = False
            block.request_id = None
        
        # 更新状态
        del self.request_blocks[request_id]
        self.allocated_blocks -= num_blocks
        self.free_blocks += num_blocks
        
        return True
    
    def release_all(self):
        """释放所有 block (重置状态)"""
        for block in self.blocks:
            block.allocated = False
            block.request_id = None
        self.request_blocks.clear()
        self.allocated_blocks = 0
        self.free_blocks = self.total_blocks
    
    def get_blocks_for_request(self, request_id: str) -> List[int]:
        """获取某个请求占用的 block IDs"""
        return self.request_blocks.get(request_id, [])
    
    def get_status(self) -> dict:
        """获取资源池状态"""
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "used_blocks": self.allocated_blocks,
            "total_memory_mb": self.total_kv_memory_mb,
            "free_memory_mb": self.free_blocks * self.block_size_mb,
            "used_memory_mb": self.allocated_blocks * self.block_size_mb,
            "utilization": self.get_utilization(),
            "active_requests": len(self.request_blocks),
            "block_size_mb": self.block_size_mb
        }


# ============ 使用示例 ============
if __name__ == "__main__":
    # 假设 KV Cache 可用 20GB = 20480 MB
    pool = KVBlockPool(total_kv_memory_mb=20480, block_size_mb=64)
    
    # 分配 5 个 block (320MB)
    print(f"\n分配 5 个 block:")
    success = pool.allocate("req_1", 5)
    print(f"  成功: {success}")
    print(f"  状态: {pool.get_status()}")
    
    # 分配 10 个 block (640MB)
    print(f"\n分配 10 个 block:")
    success = pool.allocate("req_2", 10)
    print(f"  成功: {success}")
    print(f"  状态: {pool.get_status()}")
    
    # 释放 req_1
    print(f"\n释放 req_1:")
    pool.release("req_1")
    print(f"  状态: {pool.get_status()}")
    
    # 尝试分配 200 个 block (12.8GB)，应该成功
    print(f"\n分配 200 个 block (12.8GB):")
    success = pool.allocate("req_3", 200)
    print(f"  成功: {success}")
    print(f"  状态: {pool.get_status()}")
    
    # 尝试分配 2000 个 block (128GB)，应该失败
    print(f"\n分配 2000 个 block (128GB):")
    success = pool.allocate("req_4", 2000)
    print(f"  成功: {success}")