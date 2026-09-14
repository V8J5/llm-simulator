"""Hardware-independent configuration and cost-result schema."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    num_parameters: int
    bytes_per_element: int = 2


@dataclass(frozen=True)
class HardwareSpec:
    accelerator: str
    vendor: str = "unknown"
    memory_gb: float = 0.0
    peak_tflops: float = 0.0 # 当前主线不是完全依靠 peak_tflops 和带宽解析计算，而是优先读取真实 profiling 表。
                            # HardwareSpec 主要提供硬件身份；profiling 缺失时的 fallback；后续多硬件扩展接口。
    memory_bandwidth_gb_s: float = 0.0
    interconnect: str = "unknown"
    interconnect_bandwidth_gb_s: float = 0.0


@dataclass(frozen=True)
class ParallelSpec:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1 # 目前只有字段，尚未有执行语义。
    data_parallel: int = 1


@dataclass(frozen=True)
class KVCacheSpec:
    """Runtime KV-cache geometry reported by vLLM."""
    num_gpu_blocks: int
    block_size_tokens: int = 16
    cache_dtype: str = "auto"
    enable_prefix_caching: bool = False

    def __post_init__(self):
        if self.num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks must be positive")
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")


@dataclass
class CostEstimate:
    stage: str
    total_time_ms: float
    compute_ms: float
    communication_ms: float
    memory_mb: Dict[str, float]
    extrapolated: bool = False
    profile_source: str = "operator_profile"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["breakdown"] = {
            "compute_ms": round(self.compute_ms, 4),
            "comm_ms": round(self.communication_ms, 4),
            "comm_ratio": (f"{100 * self.communication_ms / self.total_time_ms:.1f}%"
                           if self.total_time_ms > 0 else "0%"),
        }
        result["total_time_ms"] = round(self.total_time_ms, 4)
        return result
