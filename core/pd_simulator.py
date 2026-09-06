#!/usr/bin/env python3
"""
第五阶段：PD 分离在线推理仿真器（完善版）

核心概念：
- Prefill 和 Decode 部署在独立的设备池上
- 请求先经过 Prefill 池处理，然后 KV Cache 传输到 Decode 池
- 两个池各自拥有独立的调度器、队列和 KV Cache 资源池
- 新增：支持网络拓扑配置、Decode OOM 重试机制、修复 Batch Token 计数
"""

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import csv
import os
import sys
# 将当前文件所在目录（core）添加到 sys.path，确保能导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入已有模块
from qwen3_cost_model import LLMCostModel
from kv_block_pool import KVBlockPool
from resource_checker import ResourceChecker
from communication_model import CommunicationModel
from simulator import Request, RequestStatus, Batch, MetricsCollector, RequestGenerator


# ================================================================
# Step 1: PD 分离配置和请求数据结构
# ================================================================

@dataclass
class PDConfig:
    """PD 分离硬件配置（支持网络拓扑扩展）"""
    # Prefill 池
    num_prefill_gpus: int = 2
    prefill_tp: int = 2
    prefill_kv_memory_mb: int = 10000
    
    # Decode 池
    num_decode_gpus: int = 4
    decode_tp: int = 4
    decode_kv_memory_mb: int = 20000
    
    # ===== 修复 D: 网络拓扑配置 =====
    kv_transfer_mode: str = "pcie"          # "pcie" | "nvlink" | "roce"
    kv_transfer_bw_gb_s: float = 12.5       # 默认 PCIe 3.0 x16 (备用)
    kv_transfer_latency_ms: float = 0.1     # 默认延迟 (备用)
    
    # 跨节点/高级拓扑配置
    cross_node_bw_gb_s: float = 25.0        # RoCE v2 典型带宽
    cross_node_latency_ms: float = 2.0      # 跨节点典型延迟


class PDRequestStatus(Enum):
    """PD 分离请求的扩展状态"""
    WAITING_PREFILL = "waiting_prefill"
    PREFILLING = "prefilling"
    WAITING_TRANSFER = "waiting_transfer"
    TRANSFERRING = "transferring"
    WAITING_DECODE = "waiting_decode"
    DECODING = "decoding"
    COMPLETED = "completed"


@dataclass
class PDRequest(Request):
    """PD 分离请求 (继承自 Request)"""
    pd_status: PDRequestStatus = PDRequestStatus.WAITING_PREFILL
    kv_cache_size_mb: float = 0.0
    kv_transfer_start_time: float = 0.0
    kv_transfer_end_time: float = 0.0
    assigned_prefill_gpu: int = -1
    assigned_decode_gpu: int = -1
    
    @property
    def kv_len(self) -> int:
        if self.status in [RequestStatus.WAITING, RequestStatus.COMPLETED]:
            return 0
        return self.prompt_len + self.generated_tokens
    
    @property
    def ttft(self) -> float:
        # PD 分离中，TTFT 定义为请求到达 -> 解码开始（包含传输时间）
        if self.prefill_end_time > 0 and self.kv_transfer_end_time > 0 and self.decode_start_time > 0:
            return self.decode_start_time - self.arrival_time
        return 0.0
    
    @property
    def tpot(self) -> float:
        # 解码阶段平均每 token 耗时（扣除传输等待时间）
        if self.generated_tokens > 0 and self.decode_start_time > 0:
            return (self.last_decode_end_time - self.prefill_end_time - 
                    (self.kv_transfer_end_time - self.kv_transfer_start_time)) / self.generated_tokens
        return 0.0


# ================================================================
# Step 2: KV 传输模型（支持拓扑感知）
# ================================================================

class KVTransferModel:
    def __init__(self, bandwidth_gb_s: float = 12.5, latency_ms: float = 0.1):
        self.bandwidth_gb_s = bandwidth_gb_s
        self.latency_ms = latency_ms
        
        # ===== 修复 D: 定义不同拓扑的带宽/延迟映射 =====
        self.topology_map = {
            "pcie": {"bw": 12.5, "lat": 0.1},    # PCIe 3.0 x16
            "nvlink": {"bw": 100.0, "lat": 0.01}, # NVLink 3.0
            "roce": {"bw": 25.0, "lat": 2.0},     # RoCE v2 跨节点
            "infiniband": {"bw": 50.0, "lat": 1.0} # InfiniBand EDR
        }
    
    def estimate_transfer_time(self, kv_cache_size_mb: float, mode: str = "pcie") -> float:
        """
        估算 KV 传输时间
        :param kv_cache_size_mb: 传输大小 (MB)
        :param mode: 拓扑模式 ("pcie", "nvlink", "roce", "infiniband")
        """
        if kv_cache_size_mb <= 0:
            return 0.0
        
        # 获取对应拓扑的带宽和延迟，若未找到则使用默认值
        topo = self.topology_map.get(mode, {"bw": self.bandwidth_gb_s, "lat": self.latency_ms})
        bw_gb_s = topo["bw"]
        lat_ms = topo["lat"]
        
        # 传输时间 = 固定延迟 + 数据量 / 带宽
        # 注意：带宽单位 GB/s，数据量 MB，需要统一单位
        transfer_ms = kv_cache_size_mb / (bw_gb_s * 1000) * 1000
        return lat_ms + transfer_ms
    
    def calculate_kv_cache_size(self, prompt_len: int, num_layers: int = 64,
                                num_kv_heads: int = 8, head_dim: int = 80,
                                bytes_per_param: int = 2) -> float:
        """计算理论 KV Cache 大小 (MB) - 保留用于参考，实际传输以 Block 为准"""
        bytes_per_token = num_kv_heads * head_dim * 2 * bytes_per_param
        total_bytes = num_layers * prompt_len * bytes_per_token
        return total_bytes / (1024 * 1024)


# ================================================================
# Step 3: PD 分离调度器（修复 A: Batch Token 计数）
# ================================================================

class PDScheduler:
    def __init__(self, name: str, kv_pool: KVBlockPool, cost_model: LLMCostModel,
                 tp_size: int, max_concurrent_requests: int = 8,
                 max_batch_token: int = 4096, resource_checker: ResourceChecker = None,
                 num_layers: int = 64, num_kv_heads: int = 8, head_dim: int = 80):
        self.name = name
        self.kv_pool = kv_pool
        self.cost_model = cost_model
        self.tp_size = tp_size
        self.max_concurrent_requests = max_concurrent_requests
        self.max_batch_token = max_batch_token
        self.resource_checker = resource_checker
        # 存储模型参数用于计算 blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.total_batches = 0
    
    def schedule(self, queue: List[PDRequest], current_time: float, stage: str) -> Optional[Batch]:
        if not queue:
            return None
        
        sorted_requests = sorted(queue, key=lambda r: r.arrival_time)
        selected = []
        # ===== 修复 A: 正确统计 Batch 总 Token 数 =====
        total_tokens = 0
        available_blocks = self.kv_pool.get_free_blocks()
        
        for req in sorted_requests:
            # 1. 检查显存
            needed = self._calculate_kv_blocks(req, stage)
            if needed > available_blocks:
                continue
            
            # 2. 计算当前请求的 token 数（Prefill 用 prompt_len，Decode 用 kv_len）
            req_tokens = req.prompt_len if stage == "prefill" else req.kv_len
            
            # 3. 检查加入后是否超过 Batch Token 上限（这才是正确的做法）
            if total_tokens + req_tokens > self.max_batch_token:
                continue
            
            # 4. 检查并发数
            if len(selected) >= self.max_concurrent_requests:
                break
            
            # 5. 选定请求
            selected.append(req)
            total_tokens += req_tokens
            available_blocks -= needed
        
        if not selected:
            return None
        
        self.total_batches += 1
        batch_id = f"{stage}_{self.total_batches}"
        
        return Batch(
            batch_id=batch_id,
            requests=selected,
            created_time=current_time,
            stage=stage
        )
    
    def _calculate_kv_blocks(self, req: PDRequest, stage: str) -> int:
        """计算请求需要的 KV blocks（优先使用 ResourceChecker）"""
        seq_len = req.prompt_len if stage == "prefill" else req.kv_len
        
        if self.resource_checker is not None:
            # 使用精确计算
            return self.resource_checker.calculate_kv_blocks_needed(
                batch_size=1, 
                seq_len=seq_len, 
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads, 
                head_dim=self.head_dim, 
                tp_size=self.tp_size
            )
        
        # fallback 估算（当 resource_checker 不可用时）
        # 注意：这里的硬编码 64 是 old bug 的遗留，但作为 fallback 保留
        bytes_per_token = self.num_kv_heads * self.head_dim * 2 * 2 / self.tp_size
        total_bytes = self.num_layers * seq_len * bytes_per_token
        block_size_mb = self.kv_pool.block_size_mb
        blocks = max(1, int(total_bytes / (block_size_mb * 1024 * 1024)) + 1)
        return min(blocks, 100)


# ================================================================
# Step 4: PD 分离仿真器（主循环，修复 B/C/D）
# ================================================================

class PDSimulator:
    def __init__(self, pd_config: PDConfig, cost_model: LLMCostModel,
                 request_generator: RequestGenerator, resource_checker: ResourceChecker = None,
                 max_concurrent_requests: int = 8, max_batch_token: int = 4096):
        
        self.config = pd_config
        self.cost_model = cost_model
        self.request_generator = request_generator
        self.num_layers = 64
        self.num_kv_heads = 8
        self.head_dim = 80
        self.bytes_per_elem = 2
        
        # --- Prefill 池 ---
        self.prefill_kv_pool = KVBlockPool(
            total_kv_memory_mb=pd_config.prefill_kv_memory_mb, block_size_mb=64
        )
        # 创建 Prefill 专用的 ResourceChecker
        self.prefill_resource_checker = ResourceChecker(
            kv_pool=self.prefill_kv_pool, 
            comm_model=CommunicationModel(),
            workspace_reserve_mb=1024
        )
        self.prefill_scheduler = PDScheduler(
            name="prefill", kv_pool=self.prefill_kv_pool, cost_model=cost_model,
            tp_size=pd_config.prefill_tp, 
            max_concurrent_requests=max_concurrent_requests,
            max_batch_token=max_batch_token, 
            resource_checker=self.prefill_resource_checker,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        self.prefill_queue: List[PDRequest] = []
        
        # --- Decode 池 ---
        self.decode_kv_pool = KVBlockPool(
            total_kv_memory_mb=pd_config.decode_kv_memory_mb, block_size_mb=64
        )
        # 创建 Decode 专用的 ResourceChecker
        self.decode_resource_checker = ResourceChecker(
            kv_pool=self.decode_kv_pool,
            comm_model=CommunicationModel(),
            workspace_reserve_mb=1024
        )
        self.decode_scheduler = PDScheduler(
            name="decode", kv_pool=self.decode_kv_pool, cost_model=cost_model,
            tp_size=pd_config.decode_tp,
            max_concurrent_requests=max_concurrent_requests,
            max_batch_token=max_batch_token,
            resource_checker=self.decode_resource_checker,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        self.decode_queue: List[PDRequest] = []
        
        # --- 传输模型（初始化时传入拓扑配置）---
        self.transfer_model = KVTransferModel(
            bandwidth_gb_s=pd_config.kv_transfer_bw_gb_s,
            latency_ms=pd_config.kv_transfer_latency_ms
        )
        self.transfer_mode = pd_config.kv_transfer_mode
        
        # --- 仿真状态 ---
        self.current_time = 0.0
        self.running_prefill_batch: Optional[Batch] = None
        self.running_decode_batch: Optional[Batch] = None
        
        # 事件队列 (time, event_type, counter, data)
        self.event_queue: List[Tuple[float, str, int, Any]] = []
        self._event_counter = 0
        
        # 统计
        self.total_requests = 0
        self.finished_requests = 0
        self.completed_requests: List[PDRequest] = []
    
    def reset(self):
        self.current_time = 0.0
        self.prefill_queue = []
        self.decode_queue = []
        self.running_prefill_batch = None
        self.running_decode_batch = None
        self.event_queue = []
        self._event_counter = 0
        self.total_requests = 0
        self.finished_requests = 0
        self.completed_requests = []
        self.prefill_kv_pool.release_all()
        self.decode_kv_pool.release_all()
        self.request_generator.reset()
    
    def add_request(self, req: PDRequest):
        self.prefill_queue.append(req)
        self.total_requests += 1
        print(f"[{self.current_time:.1f}ms] 📥 请求 {req.request_id} 到达 "
              f"(prompt={req.prompt_len}, output={req.output_len})")
    
    def _add_event(self, time: float, event_type: str, data: Any):
        heapq.heappush(self.event_queue, (time, event_type, self._event_counter, data))
        self._event_counter += 1
    
    def _schedule_prefill_batch(self) -> bool:
        if self.running_prefill_batch is not None:
            return False
        
        batch = self.prefill_scheduler.schedule(self.prefill_queue, self.current_time, "prefill")
        if batch is None:
            return False
        
        selected_ids = {r.request_id for r in batch.requests}
        self.prefill_queue = [r for r in self.prefill_queue if r.request_id not in selected_ids]
        
        for req in batch.requests:
            req.pd_status = PDRequestStatus.PREFILLING
            req.status = RequestStatus.PREFILLING
            req.prefill_start_time = self.current_time
        
        batch.start_time = self.current_time
        estimated_time = self._estimate_batch_time(batch, "prefill")
        batch.estimated_time_ms = estimated_time
        batch.end_time = self.current_time + estimated_time
        
        self.running_prefill_batch = batch
        self._add_event(batch.end_time, "prefill_complete", batch)
        
        print(f"[{self.current_time:.1f}ms] 🚀 调度 Prefill Batch {batch.batch_id} "
              f"(requests={batch.batch_size}, est={estimated_time:.1f}ms)")
        return True
    
    def _schedule_decode_batch(self) -> bool:
        if self.running_decode_batch is not None:
            return False
        
        batch = self.decode_scheduler.schedule(self.decode_queue, self.current_time, "decode")
        if batch is None:
            return False
        
        selected_ids = {r.request_id for r in batch.requests}
        self.decode_queue = [r for r in self.decode_queue if r.request_id not in selected_ids]
        
        for req in batch.requests:
            req.pd_status = PDRequestStatus.DECODING
            req.status = RequestStatus.DECODING
            if req.decode_start_time == 0:
                req.decode_start_time = self.current_time
        
        batch.start_time = self.current_time
        estimated_time = self._estimate_batch_time(batch, "decode")
        batch.estimated_time_ms = estimated_time
        batch.end_time = self.current_time + estimated_time
        
        self.running_decode_batch = batch
        self._add_event(batch.end_time, "decode_complete", batch)
        
        print(f"[{self.current_time:.1f}ms] 🚀 调度 Decode Batch {batch.batch_id} "
              f"(requests={batch.batch_size}, est={estimated_time:.1f}ms)")
        return True
    
    def _estimate_batch_time(self, batch: Batch, stage: str) -> float:
        batch_size = batch.batch_size
        if stage == "prefill":
            seq_len = max(r.prompt_len for r in batch.requests)
            result = self.cost_model.predict_prefill(batch_size, seq_len, self.config.prefill_tp)
            return result["total_time_ms"]
        else:
            kv_len = max(r.kv_len for r in batch.requests)
            result = self.cost_model.predict_decode(batch_size, kv_len, self.config.decode_tp)
            return result["total_time_ms"]
    
    def _process_events(self):
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            time, event_type, counter, data = heapq.heappop(self.event_queue)
            if event_type == "prefill_complete":
                self._process_prefill_complete(data)
            elif event_type == "decode_complete":
                self._process_decode_complete(data)
            elif event_type == "kv_transfer_complete":
                self._process_kv_transfer_complete(data)
    
    def _process_prefill_complete(self, batch: Batch):
        print(f"[{self.current_time:.1f}ms] ✅ Prefill Batch {batch.batch_id} 完成 "
              f"(实际={batch.end_time - batch.start_time:.1f}ms)")
        
        for req in batch.requests:
            req.prefill_end_time = self.current_time
            req.pd_status = PDRequestStatus.WAITING_TRANSFER
            
            # ===== 修复 C: 计算实际需要的 Blocks，并以此作为传输大小 =====
            needed_blocks = self.prefill_scheduler._calculate_kv_blocks(req, "prefill")
            # 在 Prefill 池中实际分配 KV Cache（保留直到传输完成）
            success = self.prefill_kv_pool.allocate(req.request_id, needed_blocks)
            if not success:
                print(f"⚠️ Prefill 池分配失败: {req.request_id} (需要 {needed_blocks} blocks)")
                # 如果 Prefill 池都分配失败，说明配置有问题，直接标记失败（不继续）
                req.status = RequestStatus.COMPLETED # 或标记为失败，此处简化处理
                continue
            
            # 根据实际分配的 Block 数量计算传输大小（保证了传输大小与显存占用一致）
            kv_size_mb = needed_blocks * self.prefill_kv_pool.block_size_mb
            req.kv_cache_size_mb = kv_size_mb
            
            # 使用配置的拓扑模式估算传输时间
            transfer_time = self.transfer_model.estimate_transfer_time(kv_size_mb, self.transfer_mode)
            req.kv_transfer_start_time = self.current_time
            req.kv_transfer_end_time = self.current_time + transfer_time
            
            # 调度 KV 传输完成事件
            self._add_event(req.kv_transfer_end_time, "kv_transfer_complete", req)
            
            print(f"   └─ {req.request_id}: Prefill 完成, KV={kv_size_mb:.1f}MB "
                  f"(需 {needed_blocks} blocks), 传输={transfer_time:.1f}ms "
                  f"(@{self.transfer_mode})")
        
        self.running_prefill_batch = None
        # 尝试调度下一个 Prefill Batch
        self._schedule_prefill_batch()
    
    def _process_kv_transfer_complete(self, req: PDRequest):
        # ===== 修复 B: Decode OOM 重试机制 =====
        # 1. 从 Prefill 池释放（传输完成，源端可以释放了）
        self.prefill_kv_pool.release(req.request_id)
        
        # 2. 尝试在 Decode 池分配
        needed_blocks = self.decode_scheduler._calculate_kv_blocks(req, "decode")
        success = self.decode_kv_pool.allocate(req.request_id, needed_blocks)
        
        if not success:
            # Decode 池显存不足：不要丢弃请求！重新放回传输等待队列，稍后重试
            req.pd_status = PDRequestStatus.WAITING_TRANSFER
            # 设置重试时间（例如 5ms 后，模拟等待资源释放）
            retry_time = self.current_time + 5.0
            self._add_event(retry_time, "kv_transfer_complete", req)
            print(f"⏳ [{self.current_time:.1f}ms] Decode池满，{req.request_id} 延迟传输重试 (5ms后)")
            return
        
        # 3. 分配成功，请求进入 Decode 队列
        req.pd_status = PDRequestStatus.WAITING_DECODE
        self.decode_queue.append(req)
        print(f"[{self.current_time:.1f}ms] 📦 KV 传输完成: {req.request_id} "
              f"(分配 {needed_blocks} blocks), 进入 Decode 队列")
    
    def _process_decode_complete(self, batch: Batch):
        print(f"[{self.current_time:.1f}ms] ✅ Decode Batch {batch.batch_id} 完成 "
              f"(实际={batch.end_time - batch.start_time:.1f}ms)")
        
        for req in batch.requests:
            req.generated_tokens += 1
            req.last_decode_end_time = self.current_time
            
            if req.remaining_tokens <= 0:
                # 请求完成
                req.status = RequestStatus.COMPLETED
                req.pd_status = PDRequestStatus.COMPLETED
                req.completion_time = self.current_time
                self.finished_requests += 1
                self.completed_requests.append(req)
                # 释放 Decode 池中的 KV Cache
                self.decode_kv_pool.release(req.request_id)
                print(f"   └─ {req.request_id}: 完成 (总 tokens={req.output_len}, "
                      f"e2e={req.e2e_latency:.1f}ms)")
            else:
                # 未完成，重新加入 Decode 队列
                self.decode_queue.append(req)
                print(f"   └─ {req.request_id}: 生成 token {req.generated_tokens}/{req.output_len}")
        
        self.running_decode_batch = None
        # 尝试调度下一个 Decode Batch
        self._schedule_decode_batch()
    
    def run(self, simulation_duration_ms: float = 30000.0,
            ttft_sla_ms: float = 500.0, tpot_sla_ms: float = 50.0) -> Dict:
        self.reset()
        self.request_generator.reset()
        
        print("=" * 60)
        print("🚀 PD 分离仿真开始 (完善版)")
        print("=" * 60)
        print(f"📊 Prefill 池: {self.config.num_prefill_gpus} 卡, TP={self.config.prefill_tp}")
        print(f"📊 Decode 池: {self.config.num_decode_gpus} 卡, TP={self.config.decode_tp}")
        print(f"📊 KV 传输拓扑: {self.transfer_mode}")
        
        # 生成请求
        # 注意：这里为了方便 demo，生成 20 个请求；实际使用时可调整
        requests = self.request_generator.generate_requests(20)
        req_index = 0
        
        while self.current_time <= simulation_duration_ms:
            # 1. 处理请求到达
            while req_index < len(requests) and requests[req_index].arrival_time <= self.current_time:
                req = requests[req_index]
                pd_req = PDRequest(
                    request_id=req.request_id,
                    prompt_len=req.prompt_len,
                    output_len=req.output_len,
                    arrival_time=req.arrival_time
                )
                self.add_request(pd_req)
                req_index += 1
            
            # 2. 调度（尝试调度 Prefill 和 Decode）
            self._schedule_prefill_batch()
            self._schedule_decode_batch()
            
            # 3. 计算下一个事件时间
            next_events = []
            if req_index < len(requests):
                next_events.append(requests[req_index].arrival_time)
            if self.running_prefill_batch is not None:
                next_events.append(self.running_prefill_batch.end_time)
            if self.running_decode_batch is not None:
                next_events.append(self.running_decode_batch.end_time)
            # 检查事件队列中所有的 kv_transfer_complete 事件
            for item in self.event_queue:
                if item[1] == "kv_transfer_complete":
                    next_events.append(item[0])
            
            if next_events:
                next_time = min(next_events)
                if next_time > simulation_duration_ms:
                    break
                self.current_time = next_time
            else:
                # 没有事件，但还有未到达的请求 -> 跳到下一个请求到达时间
                if req_index < len(requests):
                    self.current_time = requests[req_index].arrival_time
                else:
                    break
            
            # 4. 处理到期事件
            self._process_events()
            
            # 5. 检查是否所有请求都已完成
            if self.finished_requests >= len(requests) and \
               self.running_prefill_batch is None and \
               self.running_decode_batch is None and \
               not self.decode_queue and not self.prefill_queue:
                print(f"[{self.current_time:.1f}ms] 🏁 所有请求已完成")
                break
        
        print("=" * 60)
        print("🏁 PD 分离仿真结束")
        print("=" * 60)
        
        # 收集统计
        return self._collect_metrics(self.completed_requests, ttft_sla_ms, tpot_sla_ms)

    def _collect_metrics(self, completed_requests: List[PDRequest],
                        ttft_sla_ms: float, tpot_sla_ms: float) -> Dict:
        """收集统计指标"""
        return MetricsCollector.collect(completed_requests, ttft_sla_ms, tpot_sla_ms)


# ================================================================
# Step 5: 使用示例（已适配完善版）
# ================================================================

def demo_pd_simulation(mode="fixed"):
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data"
    
    cost_model = LLMCostModel(DATA_DIR, peak_tflops=119.5, mem_bw_gb_s=864.0)
    
    # ===== 完善版配置：可以轻松切换拓扑 =====
    pd_config = PDConfig(
        num_prefill_gpus=2,
        prefill_tp=2,
        prefill_kv_memory_mb=10000,
        num_decode_gpus=4,
        decode_tp=4,
        decode_kv_memory_mb=20000,
        # 尝试切换不同的拓扑模式看效果
        kv_transfer_mode="pcie",      # "pcie" | "nvlink" | "roce"
        kv_transfer_bw_gb_s=12.5,
        kv_transfer_latency_ms=0.1
    )
    
    generator = RequestGenerator(
        mode=mode,
        arrival_interval_ms=100.0,
        prompt_len=512,
        output_len=128
    )
    
    # 实例化 PDSimulator（内部自动创建 ResourceChecker）
    simulator = PDSimulator(
        pd_config=pd_config,
        cost_model=cost_model,
        request_generator=generator,
        resource_checker=None,  # 这里传 None 即可，内部已自动创建
        max_concurrent_requests=8,
        max_batch_token=4096
    )
    
    stats = simulator.run(
        simulation_duration_ms=20000.0,
        ttft_sla_ms=500.0,
        tpot_sla_ms=50.0
    )
    
    print("\n" + "=" * 60)
    print("📊 PD 分离仿真统计结果")
    print("=" * 60)
    print(f"总请求数: {stats['total_requests']}")
    print(f"完成请求数: {stats['completed_requests']}")
    print(f"\nTTFT (ms):")
    print(f"  avg: {stats['ttft_ms']['avg']:.1f}, p50: {stats['ttft_ms']['p50']:.1f}, "
          f"p90: {stats['ttft_ms']['p90']:.1f}, p99: {stats['ttft_ms']['p99']:.1f}")
    print(f"\nTPOT (ms):")
    print(f"  avg: {stats['tpot_ms']['avg']:.1f}, p50: {stats['tpot_ms']['p50']:.1f}, "
          f"p90: {stats['tpot_ms']['p90']:.1f}, p99: {stats['tpot_ms']['p99']:.1f}")
    print(f"\n吞吐量: {stats.get('throughput_tokens_per_s', 0):.1f} tokens/s")
    
    goodput = stats.get('goodput', {})
    if goodput:
        print(f"\n🎯 Goodput (TTFT<{goodput['ttft_sla_ms']:.0f}ms, TPOT<{goodput['tpot_sla_ms']:.0f}ms):")
        print(f"  TTFT 满足: {goodput['ttft_satisfied']}/{stats['completed_requests']} ({goodput['ttft_ratio']*100:.1f}%)")
        print(f"  TPOT 满足: {goodput['tpot_satisfied']}/{stats['completed_requests']} ({goodput['tpot_ratio']*100:.1f}%)")
        print(f"  两者都满足: {goodput['both_satisfied']}/{stats['completed_requests']} ({goodput['both_ratio']*100:.1f}%)")
    
    return stats


if __name__ == "__main__":
    stats = demo_pd_simulation(mode="fixed")