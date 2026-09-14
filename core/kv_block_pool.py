"""vLLM-compatible logical KV token-block pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class KVBlock:
    block_id: int
    request_id: Optional[str] = None
    allocated: bool = False


class KVBlockPool:
    """Manage logical PagedAttention blocks.

    ``total_blocks`` and ``block_size_tokens`` map directly to vLLM's
    ``num_gpu_blocks`` and ``block_size`` metrics. Memory is diagnostic only;
    scheduling and admission operate exclusively on logical block counts.
    """

    def __init__(self, total_blocks: int, block_size_tokens: int = 16,
                 kv_bytes_per_token_per_rank: float = 0.0,
                 verbose: bool = False):
        if not isinstance(total_blocks, int) or total_blocks <= 0:
            raise ValueError("total_blocks must be a positive integer")
        if not isinstance(block_size_tokens, int) or block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be a positive integer")
        if kv_bytes_per_token_per_rank < 0:
            raise ValueError("kv_bytes_per_token_per_rank must be non-negative")
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.kv_bytes_per_token_per_rank = float(kv_bytes_per_token_per_rank)
        self.blocks = [KVBlock(block_id=index) for index in range(total_blocks)]
        self.request_blocks: Dict[str, List[int]] = {}
        self.allocated_blocks = 0
        self.free_blocks = total_blocks
        if verbose:
            print(
                f"KVBlockPool: {total_blocks} blocks, "
                f"{block_size_tokens} tokens/block, "
                f"{self.block_size_mib:.4f} MiB/block"
            )

    @classmethod
    def from_memory_budget(cls, total_kv_memory_mib: float,
                           block_size_tokens: int,
                           kv_bytes_per_token_per_rank: float,
                           verbose: bool = False) -> "KVBlockPool":
        """Compatibility factory when runtime ``num_gpu_blocks`` is unavailable."""
        if total_kv_memory_mib <= 0 or kv_bytes_per_token_per_rank <= 0:
            raise ValueError("memory budget and KV bytes/token must be positive")
        block_mib = (block_size_tokens * kv_bytes_per_token_per_rank / 1024 ** 2)
        total_blocks = math.floor(total_kv_memory_mib / block_mib)
        if total_blocks <= 0:
            raise ValueError("memory budget cannot hold one KV block")
        return cls(total_blocks, block_size_tokens,
                   kv_bytes_per_token_per_rank, verbose)

    @property
    def block_size_mib(self) -> float:
        return self.block_size_tokens * self.kv_bytes_per_token_per_rank / 1024 ** 2

    @property
    def total_kv_memory_mib(self) -> float:
        return self.total_blocks * self.block_size_mib

    def blocks_for_tokens(self, token_count: int) -> int:
        """Logical blocks for one request, including per-sequence fragmentation."""
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return math.ceil(token_count / self.block_size_tokens) if token_count else 0

    def get_free_blocks(self) -> int:
        return self.free_blocks

    def get_used_blocks(self) -> int:
        return self.allocated_blocks

    def get_total_blocks(self) -> int:
        return self.total_blocks

    def get_available_memory_mb(self) -> float:
        return self.free_blocks * self.block_size_mib

    def get_used_memory_mb(self) -> float:
        return self.allocated_blocks * self.block_size_mib

    def get_utilization(self) -> float:
        return self.allocated_blocks / self.total_blocks

    def can_allocate(self, num_blocks: int) -> bool:
        return isinstance(num_blocks, int) and 0 <= num_blocks <= self.free_blocks

    def allocate(self, request_id: str, num_blocks: int) -> bool:
        if request_id in self.request_blocks:
            return self.ensure_capacity(request_id, num_blocks)
        return self.ensure_capacity(request_id, num_blocks)

    def ensure_capacity(self, request_id: str, total_blocks: int) -> bool:
        """Atomically grow one request to the requested logical block count."""
        if not isinstance(total_blocks, int) or total_blocks < 0:
            raise ValueError("total_blocks must be a non-negative integer")
        current = len(self.request_blocks.get(request_id, []))
        additional = total_blocks - current
        if additional <= 0:
            return True
        if not self.can_allocate(additional):
            return False
        allocated = self.request_blocks.setdefault(request_id, [])
        for block in self.blocks:
            if not block.allocated:
                block.allocated = True
                block.request_id = request_id
                allocated.append(block.block_id)
                self.allocated_blocks += 1
                self.free_blocks -= 1
                additional -= 1
                if additional == 0:
                    return True
        raise RuntimeError("KV pool counters disagree with physical block state")

    def release(self, request_id: str) -> bool:
        block_ids = self.request_blocks.pop(request_id, None)
        if block_ids is None:
            return False
        for block_id in block_ids:
            self.blocks[block_id].allocated = False
            self.blocks[block_id].request_id = None
        count = len(block_ids)
        self.allocated_blocks -= count
        self.free_blocks += count
        return True

    def release_all(self):
        for block in self.blocks:
            block.allocated = False
            block.request_id = None
        self.request_blocks.clear()
        self.allocated_blocks = 0
        self.free_blocks = self.total_blocks

    def get_blocks_for_request(self, request_id: str) -> List[int]:
        return list(self.request_blocks.get(request_id, []))

    def get_status(self) -> Dict:
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "used_blocks": self.allocated_blocks,
            "block_size_tokens": self.block_size_tokens,
            "token_capacity": self.total_blocks * self.block_size_tokens,
            "kv_bytes_per_token_per_rank": self.kv_bytes_per_token_per_rank,
            "block_size_mib": self.block_size_mib,
            "total_memory_mib": self.total_kv_memory_mib,
            "free_memory_mib": self.get_available_memory_mb(),
            "used_memory_mib": self.get_used_memory_mb(),
            "utilization": self.get_utilization(),
            "active_requests": len(self.request_blocks),
        }
