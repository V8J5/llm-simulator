#!/usr/bin/env python3
"""
第四阶段：在线推理仿真器

包含：
  Step 1: Request / Batch 数据结构
  Step 2: RequestGenerator (请求生成器)
  Step 3: Scheduler (调度器)
  Step 4: Simulator (主循环)
  Step 5: MetricsCollector (指标统计)
"""

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import csv
import os


# 导入已有模块
from qwen3_cost_model import LLMCostModel
from kv_block_pool import KVBlockPool
from resource_checker import ResourceChecker
from communication_model import CommunicationModel



# ================================================================
# Step 1: Request 和 Batch 数据结构
# ================================================================

class RequestStatus(Enum):
    """请求生命周期状态"""
    WAITING = "waiting"          # 等待调度
    PREFILLING = "prefilling"    # 正在 Prefill
    DECODING = "decoding"        # 正在 Decode
    COMPLETED = "completed"      # 已完成


@dataclass
class Request:
    """单个请求"""
    # ---- 静态属性 ----
    request_id: str
    prompt_len: int
    output_len: int
    arrival_time: float          # 到达时间 (虚拟时间, ms)
    
    # ---- 动态状态 ----
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: int = 0
    
    # ---- 时间戳 (用于计算延迟) ----
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0   # 首 token 时间 (TTFT)
    decode_start_time: float = 0.0
    last_decode_end_time: float = 0.0
    completion_time: float = 0.0
    
    @property
    def kv_len(self) -> int:
        """当前 KV Cache 长度 (用于查询成本模型)"""
        if self.status in [RequestStatus.WAITING, RequestStatus.COMPLETED]:
            return 0
        return self.prompt_len + self.generated_tokens
    
    @property
    def remaining_tokens(self) -> int:
        return self.output_len - self.generated_tokens
    
    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.COMPLETED
    
    @property
    def ttft(self) -> float:
        """首 token 延迟 (Time To First Token)"""
        if self.prefill_end_time > 0:
            return self.prefill_end_time - self.arrival_time
        return 0.0
    
    @property
    def tpot(self) -> float:
        """每 token 平均时间 (Time Per Output Token)"""
        if self.generated_tokens > 0 and self.decode_start_time > 0:
            return (self.last_decode_end_time - self.prefill_end_time) / self.generated_tokens
        return 0.0
    
    @property
    def e2e_latency(self) -> float:
        """端到端延迟"""
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0


@dataclass
class Batch:
    """一轮执行的 Batch"""
    batch_id: str
    requests: List[Request]
    created_time: float          # 调度创建时间
    stage: str                   # "prefill" 或 "decode"
    
    # 运行时填充
    estimated_time_ms: float = 0.0
    actual_time_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def batch_size(self) -> int:
        return len(self.requests)
    
    @property
    def total_tokens(self) -> int:
        """batch 中所有请求的 KV 长度总和 (用于成本模型)"""
        if self.stage == "prefill":
            return sum(r.prompt_len for r in self.requests)
        else:
            return sum(r.kv_len for r in self.requests)
    
    @property
    def max_kv_len(self) -> int:
        """最大 KV 长度 (用于成本模型)"""
        return max(r.kv_len for r in self.requests) if self.requests else 0


# ================================================================
# Step 2: RequestGenerator (请求生成器)
# ================================================================

class RequestGenerator:
    def __init__(self, 
                 mode: str = "fixed",
                 arrival_interval_ms: float = 100.0,
                 prompt_len: int = 512,
                 output_len: int = 128,
                 prompt_len_range: Tuple[int, int] = None,
                 output_len_range: Tuple[int, int] = None,
                 trace_file: str = None,          # 新增：trace 文件路径
                 trace_repeat: bool = False):     # 新增：是否循环回放
        """
        Args:
            mode: "fixed" | "poisson" | "trace"
            trace_file: CSV 文件路径 (mode="trace" 时必需)
            trace_repeat: 是否在 trace 结束后从头循环
        """
        self.mode = mode
        self.arrival_interval_ms = arrival_interval_ms
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.prompt_len_range = prompt_len_range
        self.output_len_range = output_len_range
        self.trace_file = trace_file
        self.trace_repeat = trace_repeat
        
        self.counter = 0
        self.next_arrival_time = 0.0
        
        # 加载 trace 数据
        self.trace_data = []
        self.trace_index = 0
        if self.trace_file and os.path.exists(self.trace_file):
            import csv
            with open(self.trace_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.trace_data.append({
                        'arrival_time': float(row['arrival_time_ms']),
                        'prompt_len': int(row['prompt_len']),
                        'output_len': int(row['output_len'])
                    })
            print(f"✅ 已加载 Trace 文件: {self.trace_file} ({len(self.trace_data)} 个请求)")
        elif self.mode == "trace":
            print(f"⚠️ Trace 文件不存在: {self.trace_file}，将使用固定模式")
            self.mode = "fixed"
    
    def reset(self):
        self.counter = 0
        self.next_arrival_time = 0.0
        self.trace_index = 0
    
    def get_next_arrival(self) -> Tuple[float, Request]:
        if self.mode == "fixed":
            arrival_time = self.next_arrival_time
            self.next_arrival_time += self.arrival_interval_ms
            prompt_len = self.prompt_len
            output_len = self.output_len
            
        elif self.mode == "poisson":
            interval = random.expovariate(1.0 / self.arrival_interval_ms)
            arrival_time = self.next_arrival_time
            self.next_arrival_time += interval
            prompt_len = self.prompt_len
            output_len = self.output_len
            
        elif self.mode == "trace":
            if self.trace_index >= len(self.trace_data):
                if self.trace_repeat:
                    self.trace_index = 0
                    # 调整时间偏移，使循环连接平滑
                    last_time = self.trace_data[-1]['arrival_time']
                    self.next_arrival_time = self.next_arrival_time - last_time
                else:
                    # 没有更多请求了
                    return None, None
            
            entry = self.trace_data[self.trace_index]
            self.trace_index += 1
            arrival_time = self.next_arrival_time + entry['arrival_time']
            prompt_len = entry['prompt_len']
            output_len = entry['output_len']
            self.next_arrival_time = arrival_time
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # 如果支持随机范围，覆盖（仅 fixed/poisson 模式生效）
        if self.mode in ["fixed", "poisson"] and self.prompt_len_range:
            prompt_len = random.randint(self.prompt_len_range[0], self.prompt_len_range[1])
        if self.mode in ["fixed", "poisson"] and self.output_len_range:
            output_len = random.randint(self.output_len_range[0], self.output_len_range[1])
        
        self.counter += 1
        request_id = f"req_{self.counter}"
        
        return arrival_time, Request(
            request_id=request_id,
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=arrival_time
        )
    
    def generate_requests(self, num_requests: int = None) -> List[Request]:
        """生成请求列表"""
        requests = []
        self.reset()
        
        if self.mode == "trace":
            # Trace 模式：生成所有 trace 请求
            for _ in range(len(self.trace_data)):
                _, req = self.get_next_arrival()
                if req is None:
                    break
                requests.append(req)
            return requests
        
        # 其他模式：生成 num_requests 个请求
        if num_requests is None:
            num_requests = 20
        for _ in range(num_requests):
            _, req = self.get_next_arrival()
            requests.append(req)
        return requests


# ================================================================
# Step 3: Scheduler (调度器)
# ================================================================

class Scheduler:
    """
    调度器: 从等待队列中选择请求组成 Batch
    
    策略: 简单的 Continuous Batching
    - 优先选择等待时间最长的请求
    - 受 max_batch_token 和 max_concurrent_requests 限制
    - 受 KV Cache 可用 block 数量限制
    """
    
    def __init__(self, 
                 max_batch_token: int = 4096,
                 max_concurrent_requests: int = 32,
                 kv_pool: KVBlockPool = None,
                 cost_model: LLMCostModel = None,
                 resource_checker: ResourceChecker = None):
        """
        Args:
            max_batch_token: 单 batch 最大 token 数
            max_concurrent_requests: 单 batch 最大请求数
            kv_pool: KV 资源池 (用于检查可用 block)
            cost_model: 成本模型 (用于估算 batch token 数)
        """
        self.max_batch_token = max_batch_token
        self.max_concurrent_requests = max_concurrent_requests
        self.kv_pool = kv_pool
        self.cost_model = cost_model
        self.resource_checker = resource_checker  # 存储引用
        
        # 统计
        self.total_batches = 0
    
    def schedule(self, 
                 waiting_queue: List[Request],
                 current_time: float,
                 tp_size: int = 1) -> Optional[Batch]:
        """
        从等待队列中调度一个 Batch
        
        Args:
            waiting_queue: 等待队列 (按到达时间排序)
            current_time: 当前虚拟时间
            tp_size: TP 配置
        
        Returns:
            Batch 或 None (如果没有可调度的请求)
        """
        if not waiting_queue:
            return None
        
        # 按到达时间排序 (先到先服务)
        sorted_requests = sorted(waiting_queue, key=lambda r: r.arrival_time)
        
        selected = []
        total_kv_len = 0
        available_blocks = self.kv_pool.get_free_blocks() if self.kv_pool else 999999
        
        for req in sorted_requests:
            # 检查请求需要的 KV blocks
            if self.kv_pool:
                # 计算这个请求需要多少 block
                needed = self._calculate_kv_blocks(req, tp_size)
                if needed > available_blocks:
                    # 显存不够，跳过这个请求 (它继续等待)
                    continuezz
            
            # 检查 batch 限制
            if len(selected) >= self.max_concurrent_requests:
                break
            
            # 计算当前 batch 的 token 数
            if selected:
                # 取最大 KV 长度作为 batch 的代表 (保守)
                max_kv = max(req.kv_len for req in selected + [req])
                """
                这里判断的是“单个请求的 KV 长度是否超过 max_batch_token”，而不是“batch 中所有 token 的总和”。
                这是一个偏保守的实现，等价于“任意一个请求的 KV 长度都不能超过 4096 token”。实际 continuous batching 通常限制的是 batch 总 token 数。
                """

                # 如果超过限制，不加入
                if max_kv > self.max_batch_token:
                    continue
            
            selected.append(req)
            total_kv_len += req.kv_len if req.kv_len > 0 else req.prompt_len
            
            # 更新可用 block
            if self.kv_pool:
                available_blocks -= self._calculate_kv_blocks(req, tp_size)
        
        if not selected:
            return None
        
        # 确定阶段: 如果所有请求都完成了 Prefill -> Decode batch
        all_waiting = all(r.status == RequestStatus.WAITING for r in selected)
        all_decoding = all(r.status == RequestStatus.DECODING for r in selected)

        if all_waiting:
            stage = "prefill"
        elif all_decoding:
            stage = "decode"
        else:
            stage = "mixed"
        
        self.total_batches += 1
        batch = Batch(
            batch_id=f"batch_{self.total_batches}",
            requests=selected,
            created_time=current_time,
            stage=stage
        )
        
        return batch
    
    def _calculate_kv_blocks(self, req: Request, tp_size: int = 1) -> int:
        """计算请求需要的 KV blocks"""
        # 关键修复：DECODING 请求的 KV 块已分配，调度时不再需要额外分配
        if req.status == RequestStatus.DECODING:
            return 0
        # WAITING 请求才需要分配
        seq_len = req.prompt_len
        # 简化: 根据 kv_len 估算
        # 优先使用 ResourceChecker 的精确计算，否则 fallback 到估算
        if self.resource_checker is not None:
            # 使用 ResourceChecker 的精确计算
            # 注意：calculate_kv_blocks_needed 需要 (batch_size, seq_len, num_layers, num_kv_heads, head_dim, tp_size)
            # 这里 batch_size 固定为 1（单个请求），seq_len 用当前 KV 长度
            return self.resource_checker.calculate_kv_blocks_needed(
                batch_size=1,
                seq_len=seq_len,
                num_layers=64,
                num_kv_heads=8,
                head_dim=80,
                tp_size=tp_size  # Scheduler 层面不知道 tp_size，需要从外部传入或使用默认值
            )
        
        # fallback: 如果 resource_checker 不可用，使用原有的硬编码估算
        kv_len = req.kv_len if req.kv_len > 0 else req.prompt_len
        # 粗略估算：每个 token 的 KV Cache 大小 (bytes)
        bytes_per_token = 64 * 8 * 80 * 2 * 2 / 64
        total_bytes = kv_len * bytes_per_token
        blocks = max(1, int(total_bytes / (64 * 1024 * 1024)) + 1)
        return min(blocks, 100)


# ================================================================
# Step 4: Simulator (主循环)
# ================================================================

class Simulator:
    """
    离散事件仿真器
    
    维护:
    - 请求队列 (waiting_queue)
    - 当前正在执行的 Batch
    - 虚拟时间 (current_time)
    - KV Cache 状态 (kv_pool)
    - 事件队列 (event_queue)
    - 完成的请求 (completed_requests)
    """
    
    def __init__(self,
                 cost_model: LLMCostModel,
                 kv_pool: KVBlockPool,
                 resource_checker: ResourceChecker,
                 scheduler: Scheduler,
                 request_generator: RequestGenerator,
                 tp_size: int = 1,
                 overhead_scale: float = 1.0):
        """
        Args:
            cost_model: 成本模型 (预测执行时间)
            kv_pool: KV Cache 资源池
            resource_checker: 资源检查器
            scheduler: 调度器
            request_generator: 请求生成器
            tp_size: TP 并行度
        """
        self.cost_model = cost_model
        self.kv_pool = kv_pool
        self.resource_checker = resource_checker
        self.scheduler = scheduler
        self.request_generator = request_generator
        self.tp_size = tp_size
        self.overhead_scale = overhead_scale

        # 仿真状态
        self.current_time = 0.0
        self.waiting_queue: List[Request] = []
        self.running_batch: Optional[Batch] = None
        self.completed_requests: List[Request] = []
        
        # 事件队列 (事件驱动)
        self.event_queue: List[Tuple[float, str, Any]] = []  # (time, type, data)
        
        # 统计
        self.total_requests = 0
        self.finished_requests = 0
    
    def reset(self):
        """重置仿真状态"""
        self.current_time = 0.0
        self.waiting_queue = []
        self.running_batch = None
        self.completed_requests = []
        self.event_queue = []
        self.total_requests = 0
        self.finished_requests = 0
        self.kv_pool.release_all()
        self.request_generator.reset()
    
    def add_request(self, req: Request):
        """添加一个请求到系统"""
        self.waiting_queue.append(req)
        self.total_requests += 1
        print(f"[{self.current_time:.1f}ms] 📥 请求 {req.request_id} 到达 "
              f"(prompt={req.prompt_len}, output={req.output_len})")
    
    def _schedule_next_batch(self) -> bool:
        """调度下一批请求"""
        if self.running_batch is not None:
            return False
        
        batch = self.scheduler.schedule(
            self.waiting_queue,
            self.current_time,
            self.tp_size
        )
        
        if batch is None:
            return False
        
        # 从等待队列中移除被选中的请求
        selected_ids = {r.request_id for r in batch.requests}
        self.waiting_queue = [r for r in self.waiting_queue if r.request_id not in selected_ids]
        
        # 更新请求状态
        for req in batch.requests:
            if req.status == RequestStatus.WAITING:
                req.status = RequestStatus.PREFILLING
                req.prefill_start_time = self.current_time
        
        # 估算执行时间
        batch.start_time = self.current_time
        estimated_time = self._estimate_batch_time(batch)
        batch.estimated_time_ms = estimated_time
        batch.end_time = self.current_time + estimated_time
        
        self.running_batch = batch
        
        # 调度 Batch 完成事件
        self._add_event(batch.end_time, "batch_complete", batch)
        
        print(f"[{self.current_time:.1f}ms] 🚀 调度 Batch {batch.batch_id} "
              f"(stage={batch.stage}, requests={batch.batch_size}, "
              f"est={estimated_time:.1f}ms)")
        
        return True
    
    def _estimate_batch_time(self, batch: Batch) -> float:
        if batch.stage == "prefill":
            seq_len = max(r.prompt_len for r in batch.requests)
            result = self.cost_model.predict_prefill(batch.batch_size, seq_len, self.tp_size)
            return result["total_time_ms"]
        elif batch.stage == "decode":
            # 用总 KV 长度除以 Batch 大小，得到平均长度
            avg_kv_len = sum(r.kv_len for r in batch.requests) / batch.batch_size
            result = self.cost_model.predict_decode(batch.batch_size, int(avg_kv_len), self.tp_size)
            total_time = result["total_time_ms"]
        else:  # mixed
            prefill_reqs = [r for r in batch.requests if r.status == RequestStatus.WAITING]
            decode_reqs = [r for r in batch.requests if r.status == RequestStatus.DECODING]
            total_time = 0.0
            if prefill_reqs:
                seq_len = max(r.prompt_len for r in prefill_reqs)
                result = self.cost_model.predict_prefill(len(prefill_reqs), seq_len, self.tp_size)
                total_time += result["total_time_ms"]
            if decode_reqs:
                kv_len = max(r.kv_len for r in decode_reqs)
                result = self.cost_model.predict_decode(len(decode_reqs), kv_len, self.tp_size)
                total_time += result["total_time_ms"]
        total_time *= self.overhead_scale
        # ===== 冷启动开销 =====
        if batch.batch_id == "batch_1" and batch.stage in ("prefill", "mixed"):
            total_time += 500.0  # 模拟 CUDA Graph 编译 + KV Cache 初始化
    
        return total_time
    
    def _add_event(self, time: float, event_type: str, data: Any):
        """添加事件到事件队列"""
        heapq.heappush(self.event_queue, (time, event_type, data))
    
    def _process_batch_complete(self, batch: Batch):
        """处理 Batch 完成事件"""
        batch.actual_time_ms = batch.end_time - batch.start_time
        
        print(f"[{self.current_time:.1f}ms] ✅ Batch {batch.batch_id} 完成 "
              f"(模拟耗时={batch.actual_time_ms:.1f}ms)")
        
        # 更新请求状态
        for req in batch.requests:
            if req.status == RequestStatus.PREFILLING:
                # Prefill 完成 → 进入 Decode
                req.status = RequestStatus.DECODING
                req.prefill_end_time = self.current_time
                req.decode_start_time = self.current_time
                req.generated_tokens = 1  # 首 token 已生成
                
                print(f"   └─ {req.request_id}: TTFT={req.ttft:.1f}ms")
                
                # 如果还需要更多 token，重新加入等待队列
                if req.remaining_tokens > 0:
                    self.waiting_queue.append(req)
                else:
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = self.current_time
                    self.finished_requests += 1
                    self.completed_requests.append(req)
            
            elif req.status == RequestStatus.DECODING:
                # Decode 完成了一轮 token
                req.generated_tokens += 1
                req.last_decode_end_time = self.current_time
                
                if req.remaining_tokens <= 0:
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = self.current_time
                    self.finished_requests += 1
                    self.completed_requests.append(req)
                    
                    print(f"   └─ {req.request_id}: 完成 (tokens={req.output_len}, "
                          f"latency={req.e2e_latency:.1f}ms)")
                else:
                    # 还需要继续 Decode，重新加入等待队列
                    self.waiting_queue.append(req)
        
        # 释放 KV Cache
        for req in batch.requests:
            if req.status == RequestStatus.COMPLETED:
                # 释放该请求占用的 KV blocks
                self.kv_pool.release(req.request_id)
        
        self.running_batch = None
        
        # 尝试调度下一个 Batch
        self._schedule_next_batch()
    
    def _process_events(self):
        """处理事件队列中的所有到期事件"""
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            time, event_type, data = heapq.heappop(self.event_queue)
            if event_type == "batch_complete":
                self._process_batch_complete(data)
    
    def run(self, simulation_duration_ms: float = 10000.0,        
            ttft_sla_ms: float = 500.0,
            tpot_sla_ms: float = 50.0) -> Dict:
        """
        运行仿真
        
        Args:
            simulation_duration_ms: 仿真时长 (ms)
        
        Returns:
            统计结果字典
        """
        self.reset()
        
        # 记录下一个请求到达时间
        self.request_generator.reset()
        
        print("=" * 60)
        print("🚀 仿真开始")
        print("=" * 60)

        # ========== 根据模式生成请求 ==========
        if self.request_generator.mode == "trace":
            # Trace 模式：预生成所有 trace 请求
            requests = self.request_generator.generate_requests()
            # 由于 trace 已经包含所有请求，num_requests 由 trace 决定
            req_index = 0
            next_arrival = requests[0].arrival_time if requests else float('inf')
            
            print(f"📋 Trace 模式: {len(requests)} 个请求")
        else:
            # 固定/Poisson 模式：生成指定数量的请求
            requests = self.request_generator.generate_requests(20)
            req_index = 0
            next_arrival = requests[0].arrival_time if requests else float('inf')
        
        # # 预生成所有请求 (简化)
        # num_requests = 20
        # requests = self.request_generator.generate_requests(num_requests)
        
        # # 把请求放入队列 (按到达时间)
        # req_index = 0
        # next_arrival = requests[0].arrival_time if requests else float('inf')
        
        while self.current_time <= simulation_duration_ms:
            # 1. 处理到期的请求到达
            while req_index < len(requests) and requests[req_index].arrival_time <= self.current_time:
                self.add_request(requests[req_index])
                req_index += 1
            
            # 2. 如果没有正在执行的 Batch，尝试调度
            if self.running_batch is None:
                self._schedule_next_batch()
            
            # 3. 推进时间到下一个事件
            next_events = []
            
            # 下一个请求到达
            if req_index < len(requests):
                next_events.append(requests[req_index].arrival_time)
            
            # 下一个 Batch 完成
            if self.running_batch is not None:
                next_events.append(self.running_batch.end_time)
            
            # 如果有事件，推进到最早的事件
            if next_events:
                next_time = min(next_events)
                # 不要超过仿真时长
                if next_time > simulation_duration_ms:
                    break
                self.current_time = next_time
            else:
                # 没有事件了
                break
            
            # 4. 处理事件
            self._process_events()
            
            # 5. 检查是否所有请求都完成了
            if self.finished_requests >= len(requests) and self.running_batch is None:
                print(f"[{self.current_time:.1f}ms] 🏁 所有请求已完成")
                break
        
        print("=" * 60)
        print("🏁 仿真结束")
        print("=" * 60)
        
        return self._collect_metrics(requests, ttft_sla_ms, tpot_sla_ms)
    
    def _collect_metrics(self, all_requests: List[Request], ttft_sla_ms: float, tpot_sla_ms: float) -> Dict:
        """收集统计指标"""
        return MetricsCollector.collect(all_requests, ttft_sla_ms, tpot_sla_ms)


# ================================================================
# Step 5: MetricsCollector (指标统计)
# ================================================================

class MetricsCollector:
    """统计指标收集器"""
    
    @staticmethod
    def collect(requests: List[Request], 
                ttft_sla_ms: float = 500.0,   # TTFT SLA 上限 (ms)
                tpot_sla_ms: float = 50.0) -> Dict:   # TPOT SLA 上限 (ms)
        """收集所有请求的统计指标"""
        # 始终返回包含基本信息的字典
        total = len(requests)
        completed = [r for r in requests if r.status == RequestStatus.COMPLETED]
        completed_count = len(completed)
        
        # 基础统计
        stats = {
            "total_requests": total,
            "completed_requests": completed_count,
        }
        
        # 如果没有完成的请求，直接返回基础信息
        if completed_count == 0:
            stats.update({
                "ttft_ms": {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0},
                "tpot_ms": {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0},
                "e2e_latency_ms": {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0},
                "prompt_len": {"avg": 0, "max": 0, "min": 0},
                "output_len": {"avg": 0, "max": 0, "min": 0},
                "throughput_tokens_per_s": 0,
                # ========== 新增 Goodput ==========
                "goodput": {
                    "ttft_sla_ms": ttft_sla_ms,
                    "tpot_sla_ms": tpot_sla_ms,
                    "ttft_satisfied": 0,
                    "tpot_satisfied": 0,
                    "both_satisfied": 0,
                    "ttft_ratio": 0.0,
                    "tpot_ratio": 0.0,
                    "both_ratio": 0.0,
                }
            })
            return stats
        
        ttfts = [r.ttft for r in completed if r.ttft > 0]
        tpots = [r.tpot for r in completed if r.tpot > 0]
        latencies = [r.e2e_latency for r in completed if r.e2e_latency > 0]
        prompt_lens = [r.prompt_len for r in completed]
        output_lens = [r.output_len for r in completed]
        
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        stats.update({
            "ttft_ms": {
                "avg": sum(ttfts) / len(ttfts) if ttfts else 0,
                "p50": percentile(ttfts, 50),
                "p90": percentile(ttfts, 90),
                "p99": percentile(ttfts, 99),
                "max": max(ttfts) if ttfts else 0,
                "min": min(ttfts) if ttfts else 0,
            },
            "tpot_ms": {
                "avg": sum(tpots) / len(tpots) if tpots else 0,
                "p50": percentile(tpots, 50),
                "p90": percentile(tpots, 90),
                "p99": percentile(tpots, 99),
                "max": max(tpots) if tpots else 0,
                "min": min(tpots) if tpots else 0,
            },
            "e2e_latency_ms": {
                "avg": sum(latencies) / len(latencies) if latencies else 0,
                "p50": percentile(latencies, 50),
                "p90": percentile(latencies, 90),
                "p99": percentile(latencies, 99),
                "max": max(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0,
            },
            "prompt_len": {
                "avg": sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0,
                "max": max(prompt_lens) if prompt_lens else 0,
                "min": min(prompt_lens) if prompt_lens else 0,
            },
            "output_len": {
                "avg": sum(output_lens) / len(output_lens) if output_lens else 0,
                "max": max(output_lens) if output_lens else 0,
                "min": min(output_lens) if output_lens else 0,
            }
        })
        
        # 计算吞吐量
        if latencies:
            total_latency = sum(latencies)
            total_output_tokens = sum(output_lens)
            stats["throughput_tokens_per_s"] = (total_output_tokens / total_latency) * 1000
        else:
            stats["throughput_tokens_per_s"] = 0
        
        # ============================================================
        # 新增：计算 Goodput (满足 SLA 的请求比例)
        # ============================================================
        if completed_count > 0:
            # 满足 TTFT SLA 的请求数
            ttft_ok = sum(1 for r in completed if r.ttft > 0 and r.ttft <= ttft_sla_ms)
            # 满足 TPOT SLA 的请求数
            tpot_ok = sum(1 for r in completed if r.tpot > 0 and r.tpot <= tpot_sla_ms)
            # 两者都满足的请求数
            both_ok = sum(1 for r in completed 
                        if r.ttft > 0 and r.ttft <= ttft_sla_ms 
                        and r.tpot > 0 and r.tpot <= tpot_sla_ms)
            
            stats["goodput"] = {
                "ttft_sla_ms": ttft_sla_ms,
                "tpot_sla_ms": tpot_sla_ms,
                "ttft_satisfied": ttft_ok,
                "tpot_satisfied": tpot_ok,
                "both_satisfied": both_ok,
                "ttft_ratio": ttft_ok / completed_count,
                "tpot_ratio": tpot_ok / completed_count,
                "both_ratio": both_ok / completed_count,
            }
        else:
            stats["goodput"] = {
                "ttft_sla_ms": ttft_sla_ms,
                "tpot_sla_ms": tpot_sla_ms,
                "ttft_satisfied": 0,
                "tpot_satisfied": 0,
                "both_satisfied": 0,
                "ttft_ratio": 0.0,
                "tpot_ratio": 0.0,
                "both_ratio": 0.0,
            }
        
        return stats


# ================================================================
# 使用示例
# ================================================================

def demo_simulation(mode="poisson"):
    """演示仿真流程"""
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data"
    
    # 1. 初始化成本模型
    cost_model = LLMCostModel(DATA_DIR, peak_tflops=119.5, mem_bw_gb_s=864.0)
    
    # 2. 初始化 KV 资源池
    kv_pool = KVBlockPool(total_kv_memory_mb=40000, block_size_mb=32)

    # 3. 初始化资源检查器 (用于瓶颈分析)
    comm_model = CommunicationModel()
    resource_checker = ResourceChecker(kv_pool=kv_pool, comm_model=comm_model)
    
    # 4. 初始化调度器
    scheduler = Scheduler(
        max_batch_token=524288,
        max_concurrent_requests=128,
        kv_pool=kv_pool,
        cost_model=cost_model,
        resource_checker=resource_checker
    )
    print(f"🔧 调度器 max_batch_token: {scheduler.max_batch_token}")
    # 5. 初始化请求生成器
    if mode == "trace":
        generator = RequestGenerator(
            mode="trace",
            # trace_file="/data/home/lihaozhe/llm-simulator/data/trace/validation_trace.csv",
            trace_file="/data/home/lihaozhe/llm-simulator/data/trace/trace_random.csv",
            trace_repeat=False
        )
    elif mode == "poisson":
        generator = RequestGenerator(
            mode="poisson",
            arrival_interval_ms=100.0,
            prompt_len=512,
            output_len=128
        )
    else:  # fixed
        generator = RequestGenerator(
            mode="fixed",
            arrival_interval_ms=100.0,
            prompt_len=512,
            output_len=128
        )
    
    
    # 6. 创建仿真器
    simulator = Simulator(
        cost_model=cost_model,
        kv_pool=kv_pool,
        resource_checker=resource_checker,
        scheduler=scheduler,
        request_generator=generator,
        tp_size=4,
        overhead_scale=0.92
    )
    
    # 7. 运行仿真，传入 SLA
    sla_list = [
        (500, 50, "严格对话"),
        (2000, 100, "对话-宽松"),
        (5000, 200, "代码/文档"),
        (30000, 500, "离线批处理"),
    ]
    
    results = []
    for ttft_sla, tpot_sla, label in sla_list:
        stats = simulator.run(
            simulation_duration_ms=300000.0,
            ttft_sla_ms=ttft_sla,
            tpot_sla_ms=tpot_sla
        )
        results.append({
            "SLA": label,
            "TTFT_SLA": ttft_sla,
            "TPOT_SLA": tpot_sla,
            "TTFT_Goodput": stats['goodput']['ttft_ratio'] * 100,
            "TPOT_Goodput": stats['goodput']['tpot_ratio'] * 100,
            "Both_Goodput": stats['goodput']['both_ratio'] * 100,
        })
    # 打印对比表
    print("\n" + "=" * 80)
    print("📊 Goodput vs SLA 对比")
    print("=" * 80)
    print(f"{'SLA场景':<20} {'TTFT SLA':<12} {'TPOT SLA':<12} {'TTFT满足':<10} {'TPOT满足':<10} {'两者满足':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['SLA']:<20} {r['TTFT_SLA']:<12} {r['TPOT_SLA']:<12} {r['TTFT_Goodput']:<10.1f}% {r['TPOT_Goodput']:<10.1f}% {r['Both_Goodput']:<10.1f}%")
    
    # 8. 打印统计结果
    print("\n" + "=" * 60)
    print("📊 仿真统计结果")
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

        # ===== 新增：保存逐条请求结果到 CSV（用于与 vLLM 对比）=====
    sim_output_dir = "/data/home/lihaozhe/llm-simulator/data/simulation_results"
    os.makedirs(sim_output_dir, exist_ok=True)
    sim_csv_path = os.path.join(sim_output_dir, "co_located_sim_output.csv")
    
    with open(sim_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "arrival_time_ms", "ttft_ms", "tpot_ms", "e2e_ms"])
        for req in simulator.completed_requests:
            writer.writerow([
                req.request_id,
                req.arrival_time,
                req.ttft,
                req.tpot,
                req.e2e_latency
            ])
    print(f"💾 仿真逐条结果已保存: {sim_csv_path}")
    
    return stats



if __name__ == "__main__":
    stats = demo_simulation(mode="trace")   # 可以切换模式: "fixed" | "poisson" | "trace"