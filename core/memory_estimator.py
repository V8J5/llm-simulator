"""Inference memory accounting shared by feasibility checks and cost reports."""

from __future__ import annotations

from typing import Dict


class MemoryEstimator:
    def __init__(self, num_layers: int, hidden_size: int, num_kv_heads: int,
                 head_dim: int, num_params: int = 32_000_000_000,
                 bytes_per_param: int = 2, safety_margin: float = 0.10,
                 runtime_overhead_mb: float = 1024.0,
                 communication_buffer_mb: float = 256.0,
                 graph_pool_mb: float = 512.0,
                 workspace_factor_prefill: float = 4.0,
                 workspace_factor_decode: float = 2.0):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_params = num_params
        self.bytes_per_param = bytes_per_param
        self.safety_margin = safety_margin
        self.runtime_overhead_mb = runtime_overhead_mb
        self.communication_buffer_mb = communication_buffer_mb
        self.graph_pool_mb = graph_pool_mb
        self.workspace_factors = {
            "prefill": workspace_factor_prefill,
            "decode": workspace_factor_decode,
        }

    @property
    def kv_bytes_per_token(self) -> float:
        """Unsharded K+V bytes for one sequence token over every layer."""
        return (self.num_layers * self.num_kv_heads * self.head_dim * 2
                * self.bytes_per_param)

    def kv_bytes_per_token_per_rank(self, tp_size: int) -> float:
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        return self.kv_bytes_per_token / tp_size

    def kv_block_memory_mib(self, block_size_tokens: int, tp_size: int) -> float:
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        return (block_size_tokens * self.kv_bytes_per_token_per_rank(tp_size)
                / 1024 ** 2)

    def kv_capacity_from_blocks(self, num_gpu_blocks: int,
                                block_size_tokens: int,
                                tp_size: int) -> Dict[str, float]:
        if num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks must be positive")
        block_mib = self.kv_block_memory_mib(block_size_tokens, tp_size)
        return {
            "num_gpu_blocks": num_gpu_blocks,
            "block_size_tokens": block_size_tokens,
            "token_capacity": num_gpu_blocks * block_size_tokens,
            "block_memory_mib_per_rank": block_mib,
            "total_kv_memory_mib_per_rank": num_gpu_blocks * block_mib,
        }

    def estimate_weight_memory(self, tp_size: int) -> float:
        return self.num_params * self.bytes_per_param / tp_size / (1024 ** 2)

    def estimate_kv_cache(self, batch_size: int, seq_len: int, tp_size: int) -> float:
        return (self.kv_bytes_per_token_per_rank(tp_size) * batch_size * seq_len
                / 1024 ** 2)

    def estimate_workspace(self, batch_size: int, seq_len: int,
                           stage: str = "prefill") -> float:
        factor = self.workspace_factors.get(stage, self.workspace_factors["decode"])
        activation = batch_size * seq_len * self.hidden_size * self.bytes_per_param
        return activation * factor / (1024 ** 2)

    def estimate_total(self, batch_size: int, seq_len: int, tp_size: int,
                       stage: str = "prefill", mode: str = "inference") -> Dict:
        if min(batch_size, seq_len, tp_size) <= 0:
            raise ValueError("batch_size, seq_len and tp_size must be positive")
        weights = self.estimate_weight_memory(tp_size)
        kv_cache = self.estimate_kv_cache(batch_size, seq_len, tp_size)
        workspace = self.estimate_workspace(batch_size, seq_len, stage)
        fixed = self.runtime_overhead_mb + self.communication_buffer_mb + self.graph_pool_mb
        subtotal = workspace + kv_cache + fixed
        if mode == "inference":
            subtotal += weights
        total = subtotal * (1 + self.safety_margin)
        return {
            "stage": stage,
            "mode": mode,
            "weight_mb": round(weights, 2),
            "kv_cache_mb": round(kv_cache, 2),
            "workspace_mb": round(workspace, 2),
            "runtime_overhead_mb": round(fixed, 2),
            "actual_total_mb": round(subtotal, 2),
            "total_with_margin_mb": round(total, 2),
        }

    def available_kv_memory(self, gpu_memory_mb: float, tp_size: int,
                            utilization: float = 0.9) -> float:
        """Memory left for KV after weights and fixed runtime reservations."""
        usable = gpu_memory_mb * utilization
        fixed = (self.estimate_weight_memory(tp_size) + self.runtime_overhead_mb
                 + self.communication_buffer_mb + self.graph_pool_mb)
        return max(usable - fixed, 0.0)

    def can_fit(self, batch_size: int, seq_len: int, tp_size: int,
                stage: str = "prefill", mode: str = "inference",
                gpu_memory_mb: int = 46068) -> bool:
        return self.estimate_total(batch_size, seq_len, tp_size, stage, mode)[
            "total_with_margin_mb"] <= gpu_memory_mb
