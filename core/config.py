# core/config.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class HardwareConfig:
    """Accelerator-independent hardware description (L20 defaults retained)."""
    gpu_name: str = "L20"
    vendor: str = "NVIDIA"
    memory_gb: float = 48
    tflops_fp16: float = 119.5  
    memory_bandwidth_gbps: float = 864.0
    interconnect: str = "PCIe Gen4 x16" 
    interconnect_bandwidth_gbps: float = 64.0
    runtime: str = "CUDA"

# 定义所有支持的模型参数库
MODEL_REGISTRY = {
    "Qwen3-32B": {
        "hidden_size": 5120,       
        "intermediate_size": 25600, 
        "num_hidden_layers": 64,   
        "num_attention_heads": 64,
        "num_key_value_heads": 8,  
        "head_dim": 128,
        "num_parameters": 32_000_000_000,
        "vocab_size": 151936,      
        "max_position_embeddings": 40960,
        "description": "Target Model (Dense)"
    },
    "Qwen2.5-7B-Instruct": {
        "hidden_size": 3584,       # 7B 的隐藏层维度
        "intermediate_size": 18944, # 7B 的 FFN 中间层维度
        "num_hidden_layers": 28,   # 7B 只有 28 层
        "num_attention_heads": 28,
        "num_key_value_heads": 4,  # GQA 比例不同
        "head_dim": 128,
        "num_parameters": 7_600_000_000,
        "vocab_size": 152064,      # Qwen2.5 词表略大
        "max_position_embeddings": 32768,
        "description": "Proxy Model for Profiling (Dense)"
    }
}

@dataclass
class ModelConfig:
    """模型配置加载器"""
    model_name: str = "Qwen3-32B" # 默认还是看 32B
    
    def __post_init__(self):
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"未知模型: {self.model_name}. 可选: {list(MODEL_REGISTRY.keys())}")
        
        params = MODEL_REGISTRY[self.model_name]
        # 动态注入属性
        for k, v in params.items():
            setattr(self, k, v)

    @property
    def total_params(self):
        return self.num_parameters

# 使用示例：
# config_32b = ModelConfig(model_name="Qwen3-32B")
# config_7b = ModelConfig(model_name="Qwen2.5-7B-Instruct")
