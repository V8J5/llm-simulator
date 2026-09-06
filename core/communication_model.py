#!/usr/bin/env python3
"""
通信模型细化 - 区分节点内/节点间链路和不同通信组，估算通信耗时

当前的 qwen3_cost_model.py 是直接查 comm_lookup_table_tp*.json 的实测值，而不是用 CommunicationModel 的公式计算。
CommunicationModel 目前是为后续扩展预留的：
    PP（流水线并行）通信需要跨节点，需要区分拓扑
    DP（数据并行）通信与 TP 不同
    未来多节点通信需要区分节点内/节点间

"""

from typing import Dict, Optional
from enum import Enum


class CommType(Enum):
    """通信类型"""
    TP_ALLREDUCE = "tp_allreduce"
    TP_ALLGATHER = "tp_allgather"
    PP_SEND = "pp_send"
    PP_RECV = "pp_recv"
    DP_ALLREDUCE = "dp_allreduce"


class Topology(Enum):
    """拓扑类型"""
    NODE_INTERNAL = "node_internal"   # 节点内 (NVLink/PCIe)
    NODE_EXTERNAL = "node_external"   # 节点间 (InfiniBand/RoCE)


class CommunicationModel:
    """
    通信模型
    
    根据通信类型、数据量和拓扑位置，返回通信耗时。
    支持区分节点内和节点间链路。
    """
    
    def __init__(self, 
                 node_internal_bandwidth_gb_s: float = 600.0,   # NVLink 4.0
                 node_external_bandwidth_gb_s: float = 25.0,    # InfiniBand HDR
                 node_internal_latency_ms: float = 0.1,
                 node_external_latency_ms: float = 2.0):
        """
        初始化通信模型
        
        Args:
            node_internal_bandwidth_gb_s: 节点内带宽 (GB/s)
            node_external_bandwidth_gb_s: 节点间带宽 (GB/s)
            node_internal_latency_ms: 节点内延迟 (ms)
            node_external_latency_ms: 节点间延迟 (ms)
        """
        self.bandwidth = {
            Topology.NODE_INTERNAL: node_internal_bandwidth_gb_s,
            Topology.NODE_EXTERNAL: node_external_bandwidth_gb_s,
        }
        self.latency = {
            Topology.NODE_INTERNAL: node_internal_latency_ms,
            Topology.NODE_EXTERNAL: node_external_latency_ms,
        }
        
        # 通信类型开销系数 (经验值)
        self.comm_overhead = {
            CommType.TP_ALLREDUCE: 1.0,
            CommType.TP_ALLGATHER: 0.8,
            CommType.PP_SEND: 0.3,
            CommType.PP_RECV: 0.3,
            CommType.DP_ALLREDUCE: 1.0,
        }
    
    def get_bandwidth(self, topology: Topology) -> float:
        """获取指定拓扑的带宽 (GB/s)"""
        return self.bandwidth.get(topology, self.bandwidth[Topology.NODE_INTERNAL])
    
    def get_latency(self, topology: Topology) -> float:
        """获取指定拓扑的延迟 (ms)"""
        return self.latency.get(topology, self.latency[Topology.NODE_INTERNAL])
    
    def estimate_time(self, 
                      comm_type: CommType,
                      data_size_mb: float,
                      topology: Topology = Topology.NODE_INTERNAL) -> float:
        """
        估算通信耗时 (ms)
        
        Args:
            comm_type: 通信类型
            data_size_mb: 数据量 (MB)
            topology: 拓扑类型
        
        Returns:
            通信耗时 (ms)
        """
        bandwidth = self.get_bandwidth(topology)
        latency = self.get_latency(topology)
        overhead = self.comm_overhead.get(comm_type, 1.0)
        
        # 数据传输时间 (MB / GB/s = ms)
        transfer_time = data_size_mb / (bandwidth * 1000) * 1000  # 转换为 ms
        
        # 总时间 = 延迟 + 传输时间 + 协议开销
        total = latency + transfer_time * overhead
        
        return max(total, 0.01)  # 至少 0.01ms
    
    def estimate_tp_allreduce(self, data_size_mb: float, 
                              topology: Topology = Topology.NODE_INTERNAL) -> float:
        """估算 TP AllReduce 耗时"""
        return self.estimate_time(CommType.TP_ALLREDUCE, data_size_mb, topology)
    
    def estimate_tp_allgather(self, data_size_mb: float,
                              topology: Topology = Topology.NODE_INTERNAL) -> float:
        """估算 TP AllGather 耗时"""
        return self.estimate_time(CommType.TP_ALLGATHER, data_size_mb, topology)
    
    def estimate_pp_send(self, data_size_mb: float,
                         topology: Topology = Topology.NODE_EXTERNAL) -> float:
        """估算 PP Send 耗时"""
        return self.estimate_time(CommType.PP_SEND, data_size_mb, topology)
    
    def get_status(self) -> dict:
        """获取通信模型状态"""
        return {
            "node_internal_bandwidth_gb_s": self.bandwidth[Topology.NODE_INTERNAL],
            "node_external_bandwidth_gb_s": self.bandwidth[Topology.NODE_EXTERNAL],
            "node_internal_latency_ms": self.latency[Topology.NODE_INTERNAL],
            "node_external_latency_ms": self.latency[Topology.NODE_EXTERNAL],
        }


# ============ 使用示例 ============
if __name__ == "__main__":
    comm_model = CommunicationModel()
    
    # 测试各种通信场景
    data_size = 100  # 100 MB
    
    print("通信模型测试 (数据量 100MB):")
    print(f"  TP AllReduce (节点内): {comm_model.estimate_tp_allreduce(data_size):.4f} ms")
    print(f"  TP AllReduce (节点间): {comm_model.estimate_tp_allreduce(data_size, Topology.NODE_EXTERNAL):.4f} ms")
    print(f"  TP AllGather (节点内): {comm_model.estimate_tp_allgather(data_size):.4f} ms")
    print(f"  PP Send (节点间): {comm_model.estimate_pp_send(data_size):.4f} ms")
    
    # 小消息场景 (Decode)
    small_data = 1  # 1 MB
    print(f"\n小消息 (1MB):")
    print(f"  TP AllReduce (节点内): {comm_model.estimate_tp_allreduce(small_data):.4f} ms")