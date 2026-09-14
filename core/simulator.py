"""Closed-loop continuous-batching discrete-event simulator.

Static execution cost comes from ``LLMCostModel``; the scheduler creates the
dynamic execution shape from queue, request progress and KV block availability.
"""

from __future__ import annotations

import csv
import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .kv_block_pool import KVBlockPool
    from .qwen3_cost_model import LLMCostModel
    from .resource_checker import ResourceChecker
except ImportError:  # direct script compatibility
    from kv_block_pool import KVBlockPool
    from qwen3_cost_model import LLMCostModel
    from resource_checker import ResourceChecker


class RequestStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"


@dataclass
class Request:  # Request 既是输入请求，也是离散事件仿真中的状态对象。
    request_id: str
    prompt_len: int # 静态输入
    output_len: int # 静态输入
    arrival_time: float # 静态输入
    status: RequestStatus = RequestStatus.WAITING # 仿真过程中不断变化的状态
    generated_tokens: int = 0   # 仿真过程中不断变化的状态，指已经生成的输出 token 数量
    prefilled_tokens: int = 0  # 仿真过程中不断变化的状态，指已经 prefill 的 token 数量
    prefill_start_time: float = 0.0 # 仿真过程中不断变化的状态，指 prefill 开始时间
    prefill_end_time: float = 0.0   # 仿真过程中不断变化的状态，指 prefill 结束时间
    first_token_time: float = 0.0   # 仿真过程中不断变化的状态，指第一个输出 token 生成完成时间
    decode_start_time: float = 0.0  # 仿真过程中不断变化的状态，指 decode 开始时间
    last_decode_end_time: float = 0.0   # 仿真过程中不断变化的状态，指最后一个输出 token 生成完成时间
    completion_time: float = 0.0    

    def __post_init__(self):
        if self.prompt_len <= 0 or self.output_len < 0 or self.arrival_time < 0:
            raise ValueError("prompt_len must be positive; output_len/arrival_time non-negative")

    @property
    def kv_len(self) -> int:
        if self.status in {RequestStatus.WAITING, RequestStatus.PREFILLING}:
            return self.prefilled_tokens
        return self.prompt_len + self.generated_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(self.prompt_len - self.prefilled_tokens, 0)

    @property
    def remaining_tokens(self) -> int:
        return max(self.output_len - self.generated_tokens, 0)

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.COMPLETED

    @property
    def ttft(self) -> float:
        return max(self.first_token_time - self.arrival_time, 0.0) if self.first_token_time else 0.0
    # first_token_time - arrival_time：包含了排队等待时间 + Prefill+ 首次 Decode 调度+ 首 token 生成。

    @property
    def tpot(self) -> float:
        # The first output token belongs to TTFT, hence N-1 decode intervals.
        intervals = self.generated_tokens - 1
        if intervals <= 0:
            return 0.0
        return max(self.last_decode_end_time - self.first_token_time, 0.0) / intervals
    # 分母是 N-1？因为第一个输出 token 已经被 TTFT 覆盖。TPOT 只统计第一个 token 之后的 token 间隔。

    @property
    def e2e_latency(self) -> float:
        return max(self.completion_time - self.arrival_time, 0.0) if self.completion_time else 0.0
    # 用户观察到的请求总延迟。

@dataclass
class Batch:
    batch_id: str
    requests: List[Request]
    created_time: float
    stage: str
    estimated_time_ms: float = 0.0
    actual_time_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    prediction: Optional[Dict] = None
    scheduled_token_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def prefill_requests(self) -> List[Request]:
        return [item for item in self.requests if item.status == RequestStatus.PREFILLING]

    @property
    def decode_requests(self) -> List[Request]:
        return [item for item in self.requests if item.status == RequestStatus.DECODING]

    @property
    def total_tokens(self) -> int:
        # vLLM's scheduling budget counts query tokens, not historical KV tokens.
        return sum(self.scheduled_token_counts.get(
            item.request_id, item.remaining_prefill_tokens)
                   for item in self.prefill_requests) + len(self.decode_requests)

    @property
    def prefill_chunk_lengths(self) -> List[int]:
        return [self.scheduled_token_counts.get(
            item.request_id, item.remaining_prefill_tokens)
                for item in self.prefill_requests]

    @property
    def prefill_context_lengths(self) -> List[int]:
        return [item.prefilled_tokens + self.scheduled_token_counts.get(
            item.request_id, item.remaining_prefill_tokens)
                for item in self.prefill_requests]

    @property
    def max_kv_len(self) -> int:
        return max((item.kv_len for item in self.requests), default=0)


class RequestGenerator: 
    def __init__(self, mode: str = "fixed", arrival_interval_ms: float = 100.0,
                 prompt_len: int = 512, output_len: int = 128,
                 prompt_len_range: Optional[Tuple[int, int]] = None,
                 output_len_range: Optional[Tuple[int, int]] = None,
                 trace_file: Optional[str] = None, trace_repeat: bool = False,
                 seed: int = 0):
        if mode not in {"fixed", "poisson", "trace"}:# 按照固定时间间隔产生请求：fixed；
                                                    # 按照泊松分布产生请求：poisson，用来模拟在线服务中随机到达的请求；
                                                    # 按照 trace 文件产生请求：trace。
            raise ValueError(f"unknown request mode: {mode}")
        if arrival_interval_ms <= 0:
            raise ValueError("arrival_interval_ms must be positive")
        self.mode, self.arrival_interval_ms = mode, arrival_interval_ms
        self.prompt_len, self.output_len = prompt_len, output_len
        self.prompt_len_range, self.output_len_range = prompt_len_range, output_len_range
        self.trace_file, self.trace_repeat, self.seed = trace_file, trace_repeat, seed
        self.trace_data = self._load_trace(trace_file) if mode == "trace" else []
        if mode == "trace" and not self.trace_data:
            raise ValueError("trace mode requires a non-empty trace_file")
        self.reset()

    @staticmethod
    def _load_trace(path: Optional[str]) -> List[Dict]:
        if not path or not Path(path).exists():
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = [{"arrival_time": float(row["arrival_time_ms"]),
                     "prompt_len": int(row["prompt_len"]),
                     "output_len": int(row["output_len"])}
                    for row in csv.DictReader(handle)]
        if any(rows[index]["arrival_time"] > rows[index + 1]["arrival_time"]
               for index in range(len(rows) - 1)):
            raise ValueError("trace arrival_time_ms must be non-decreasing")
        return rows

    def reset(self):
        self.counter = 0
        self.next_arrival_time = 0.0
        self.trace_index = 0
        self.trace_cycle_offset = 0.0
        self.random = random.Random(self.seed)

    def get_next_arrival(self) -> Tuple[Optional[float], Optional[Request]]:
        if self.mode == "trace":
            if self.trace_index >= len(self.trace_data):
                if not self.trace_repeat:
                    return None, None
                cycle_span = max(self.trace_data[-1]["arrival_time"], self.arrival_interval_ms)
                self.trace_cycle_offset += cycle_span
                self.trace_index = 0
            row = self.trace_data[self.trace_index]
            self.trace_index += 1
            arrival = self.trace_cycle_offset + row["arrival_time"]
            prompt_len, output_len = row["prompt_len"], row["output_len"]
        else:
            arrival = self.next_arrival_time
            interval = (self.arrival_interval_ms if self.mode == "fixed" else
                        self.random.expovariate(1.0 / self.arrival_interval_ms))
            self.next_arrival_time += interval
            prompt_len, output_len = self.prompt_len, self.output_len
            if self.prompt_len_range:
                prompt_len = self.random.randint(*self.prompt_len_range)
            if self.output_len_range:
                output_len = self.random.randint(*self.output_len_range)
        self.counter += 1
        request = Request(f"req_{self.counter}", prompt_len, output_len, arrival)
        return arrival, request

    def generate_requests(self, num_requests: Optional[int] = None) -> List[Request]:
        self.reset()
        count = len(self.trace_data) if self.mode == "trace" and not self.trace_repeat else (num_requests or 20)
        requests = []
        for _ in range(count):
            _, request = self.get_next_arrival()
            if request is None:
                break
            requests.append(request)
        return requests


class Scheduler:
    """Decode-priority FCFS continuous batching with token and KV constraints."""

    def __init__(self, max_batch_token: int = 4096,
                 max_concurrent_requests: int = 32,
                 kv_pool: Optional[KVBlockPool] = None,
                 cost_model: Optional[LLMCostModel] = None,
                 resource_checker: Optional[ResourceChecker] = None,
                 num_layers: int = 64, num_kv_heads: int = 8,
                 head_dim: int = 128, bytes_per_element: int = 2,
                 enable_chunked_prefill: bool = False):
        self.max_batch_token = max_batch_token
        self.max_concurrent_requests = max_concurrent_requests
        self.kv_pool, self.cost_model = kv_pool, cost_model
        self.resource_checker = resource_checker
        self.num_layers, self.num_kv_heads = num_layers, num_kv_heads
        self.head_dim, self.bytes_per_element = head_dim, bytes_per_element
        self.enable_chunked_prefill = enable_chunked_prefill
        self.total_batches = 0

    def _total_blocks(self, request: Request, scheduled_tokens: int) -> int:
        if not self.kv_pool:
            return 0
        if request.status == RequestStatus.DECODING:
            target_tokens = request.kv_len
        else:
            target_tokens = request.prefilled_tokens + scheduled_tokens
        return self.kv_pool.blocks_for_tokens(target_tokens)

    def schedule(self, waiting_queue: List[Request], current_time: float,
                 tp_size: int = 1) -> Optional[Batch]:
        if not waiting_queue:
            return None
        ordered = sorted(waiting_queue, key=lambda item:
                         (item.status != RequestStatus.DECODING, item.arrival_time))
        selected, scheduled_tokens = [], 0
        token_counts: Dict[str, int] = {}
        free_blocks = self.kv_pool.get_free_blocks() if self.kv_pool else math.inf
        for request in ordered:
            remaining_budget = self.max_batch_token - scheduled_tokens
            if remaining_budget <= 0:
                break
            if request.status == RequestStatus.DECODING:
                query_tokens = 1
            else:
                query_tokens = request.remaining_prefill_tokens
                if self.enable_chunked_prefill:
                    query_tokens = min(query_tokens, remaining_budget)
            if query_tokens <= 0 or query_tokens > remaining_budget:
                continue
            current_blocks = (len(self.kv_pool.get_blocks_for_request(request.request_id))
                              if self.kv_pool else 0)
            needed_blocks = self._total_blocks(request, query_tokens)
            additional = max(needed_blocks - current_blocks, 0)
            if additional > free_blocks:
                continue
            selected.append(request)
            token_counts[request.request_id] = query_tokens
            scheduled_tokens += query_tokens
            free_blocks -= additional
            if len(selected) >= self.max_concurrent_requests:
                break
        if not selected:
            return None
        for request in selected:
            if self.kv_pool and not self.kv_pool.ensure_capacity(
                    request.request_id,
                    self._total_blocks(request, token_counts[request.request_id])):
                raise RuntimeError("KV allocation changed after scheduling")
            if request.status == RequestStatus.WAITING:
                request.status = RequestStatus.PREFILLING
                if request.prefilled_tokens == 0:
                    request.prefill_start_time = current_time
        prefill = any(item.status == RequestStatus.PREFILLING for item in selected)
        decode = any(item.status == RequestStatus.DECODING for item in selected)
        stage = "mixed" if prefill and decode else ("prefill" if prefill else "decode")
        self.total_batches += 1
        return Batch(f"batch_{self.total_batches}", selected, current_time, stage,
                     scheduled_token_counts=token_counts)


class Simulator:
    def __init__(self, cost_model: LLMCostModel, kv_pool: KVBlockPool,
                 resource_checker: ResourceChecker, scheduler: Scheduler,
                 request_generator: RequestGenerator, tp_size: int = 1,
                 overhead_scale: float = 1.0, fixed_iteration_overhead_ms: float = 0.0,
                 verbose: bool = False):
        self.cost_model, self.kv_pool = cost_model, kv_pool
        self.resource_checker, self.scheduler = resource_checker, scheduler
        self.request_generator, self.tp_size = request_generator, tp_size
        self.overhead_scale = overhead_scale
        self.fixed_iteration_overhead_ms = fixed_iteration_overhead_ms
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.waiting_queue: List[Request] = []
        self.running_batch: Optional[Batch] = None
        self.completed_requests: List[Request] = []
        self.batch_history: List[Dict] = []
        self.event_queue: List[Tuple[float, int, str, Any]] = []
        self.event_counter = 0
        self.total_requests = self.finished_requests = 0
        self.kv_pool.release_all()
        self.request_generator.reset()

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def add_request(self, request: Request):
        self.waiting_queue.append(request)
        self.total_requests += 1

    def _add_event(self, time: float, event_type: str, data: Any):
        self.event_counter += 1
        heapq.heappush(self.event_queue, (time, self.event_counter, event_type, data))

    def _estimate_batch_time(self, batch: Batch) -> float:
        prediction = self.cost_model.predict_batch(
            batch.prefill_chunk_lengths,
            [item.kv_len for item in batch.decode_requests], self.tp_size,
            prefill_context_lengths=batch.prefill_context_lengths)
        batch.prediction = prediction
        return prediction["total_time_ms"] * self.overhead_scale + self.fixed_iteration_overhead_ms

    @staticmethod
    def _part(prediction: Dict, stage: str) -> Dict:
        return next((part for part in prediction.get("parts", [])
                     if part.get("stage") == stage), {})

    def _record_batch(self, batch: Batch) -> None:
        prediction = batch.prediction or {}
        breakdown = prediction.get("breakdown", {})
        compute_ms = float(breakdown.get("compute_ms", 0.0))
        communication_ms = float(breakdown.get("comm_ms", 0.0))
        modeled_ms = compute_ms + communication_ms
        prefill_part = self._part(prediction, "prefill")
        decode_part = self._part(prediction, "decode")
        decode_lengths = [item.kv_len for item in batch.decode_requests]
        self.batch_history.append({
            "batch_id": batch.batch_id,
            "start_time_ms": batch.start_time,
            "end_time_ms": batch.end_time,
            "stage": batch.stage,
            "batch_size": batch.batch_size,
            "prefill_count": len(batch.prefill_requests),
            "decode_count": len(batch.decode_requests),
            "prefill_request_ids": "|".join(
                item.request_id for item in batch.prefill_requests),
            "decode_request_ids": "|".join(
                item.request_id for item in batch.decode_requests),
            "prefill_tokens": sum(batch.prefill_chunk_lengths),
            "prefill_context_len_max": max(
                batch.prefill_context_lengths, default=0),
            "decode_kv_len_mean": (sum(decode_lengths) / len(decode_lengths)
                                   if decode_lengths else 0.0),
            "decode_kv_len_max": max(decode_lengths, default=0),
            "prefill_factor": prefill_part.get("calibration_factor", 1.0),
            "decode_factor": decode_part.get("calibration_factor", 1.0),
            "iteration_overhead_ms": prediction.get(
                "breakdown", {}).get("iteration_overhead_ms", 0.0),
            "compute_ms": compute_ms,
            "communication_ms": communication_ms,
            "compute_ratio": compute_ms / modeled_ms if modeled_ms else 0.0,
            "communication_ratio": (
                communication_ms / modeled_ms if modeled_ms else 0.0),
            "kv_blocks_used": self.kv_pool.get_used_blocks(),
            "kv_blocks_total": self.kv_pool.get_total_blocks(),
            "kv_block_utilization": self.kv_pool.get_utilization(),
            "estimated_time_ms": batch.estimated_time_ms,
            "extrapolated": prediction.get("extrapolated", False),
        })

    def _schedule_next_batch(self) -> bool:
        if self.running_batch is not None:
            return False
        batch = self.scheduler.schedule(self.waiting_queue, self.current_time, self.tp_size)
        if batch is None:
            return False
        ids = {item.request_id for item in batch.requests}
        self.waiting_queue = [item for item in self.waiting_queue if item.request_id not in ids]
        batch.start_time = self.current_time
        batch.estimated_time_ms = self._estimate_batch_time(batch)
        batch.end_time = self.current_time + batch.estimated_time_ms
        self._record_batch(batch)
        self.running_batch = batch
        self._add_event(batch.end_time, "batch_complete", batch)
        return True

    def _complete_request(self, request: Request):
        request.status = RequestStatus.COMPLETED
        request.completion_time = self.current_time
        if request.generated_tokens > 1 and not request.last_decode_end_time:
            request.last_decode_end_time = self.current_time
        self.finished_requests += 1
        self.completed_requests.append(request)
        self.kv_pool.release(request.request_id)

    def _process_batch_complete(self, batch: Batch):
        batch.actual_time_ms = self.current_time - batch.start_time
        for request in batch.requests:
            if request.status == RequestStatus.PREFILLING:
                request.prefilled_tokens += batch.scheduled_token_counts.get(
                    request.request_id, request.remaining_prefill_tokens)
                if request.prefilled_tokens < request.prompt_len:
                    self.waiting_queue.append(request)
                    continue
                request.prefill_end_time = self.current_time
                request.first_token_time = self.current_time
                request.decode_start_time = self.current_time
                if request.output_len == 0:
                    self._complete_request(request)
                    continue
                request.generated_tokens = 1
                if request.output_len == 1:
                    self._complete_request(request)
                else:
                    request.status = RequestStatus.DECODING
                    self.waiting_queue.append(request)
            elif request.status == RequestStatus.DECODING:
                request.generated_tokens += 1
                request.last_decode_end_time = self.current_time
                if request.generated_tokens >= request.output_len:
                    self._complete_request(request)
                else:
                    self.waiting_queue.append(request)
        self.running_batch = None

    def _process_events(self):
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            _, _, event_type, data = heapq.heappop(self.event_queue)
            if event_type == "batch_complete":
                self._process_batch_complete(data)

    def run(self, simulation_duration_ms: float = 10000.0,
            ttft_sla_ms: float = 500.0, tpot_sla_ms: float = 50.0,
            num_requests: int = 20) -> Dict:
        self.reset()
        requests = self.request_generator.generate_requests(num_requests)
        request_index = 0
        while self.current_time <= simulation_duration_ms:
            while (request_index < len(requests) and
                   requests[request_index].arrival_time <= self.current_time):
                self.add_request(requests[request_index])
                request_index += 1
            if self.running_batch is None:
                scheduled = self._schedule_next_batch()
                if not scheduled and self.waiting_queue and request_index >= len(requests):
                    break  # infeasible request; report it as incomplete instead of spinning
            next_times = []
            if request_index < len(requests):
                next_times.append(requests[request_index].arrival_time)
            if self.event_queue:
                next_times.append(self.event_queue[0][0])
            if not next_times:
                break
            next_time = min(next_times)
            if next_time > simulation_duration_ms:
                break
            self.current_time = next_time
            self._process_events()
            if self.finished_requests == len(requests) and not self.running_batch:
                break
        return MetricsCollector.collect(requests, ttft_sla_ms, tpot_sla_ms,
                                        observation_end_ms=self.current_time)


class MetricsCollector:
    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile / 100
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    @classmethod
    def _summary(cls, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {key: 0.0 for key in ("avg", "p50", "p90", "p99", "max", "min")}
        return {"avg": sum(values) / len(values),
                "p50": cls._percentile(values, 50),
                "p90": cls._percentile(values, 90),
                "p99": cls._percentile(values, 99),
                "max": max(values), "min": min(values)}

    @classmethod
    def collect(cls, requests: List[Request], ttft_sla_ms: float = 500.0,
                tpot_sla_ms: float = 50.0,
                observation_end_ms: Optional[float] = None) -> Dict:
        completed = [item for item in requests if item.status == RequestStatus.COMPLETED]
        first_arrival = min((item.arrival_time for item in requests), default=0.0)
        last_completion = max((item.completion_time for item in completed), default=first_arrival)
        end = max(observation_end_ms or last_completion, last_completion)
        makespan_s = max((end - first_arrival) / 1000.0, 0.0)
        input_tokens = sum(item.prompt_len for item in completed)
        output_tokens = sum(item.generated_tokens for item in completed)
        good = [item for item in completed
                if item.ttft <= ttft_sla_ms and
                (item.generated_tokens <= 1 or item.tpot <= tpot_sla_ms)]
        ttft_good = [item for item in completed if item.ttft <= ttft_sla_ms]
        tpot_good = [item for item in completed
                     if item.generated_tokens <= 1 or item.tpot <= tpot_sla_ms]
        good_tokens = sum(item.generated_tokens for item in good)
        stats = {
            "total_requests": len(requests),
            "completed_requests": len(completed),
            "completion_ratio": len(completed) / len(requests) if requests else 0.0,
            "ttft_ms": cls._summary([item.ttft for item in completed]),
            "tpot_ms": cls._summary([item.tpot for item in completed if item.generated_tokens > 1]),
            "e2e_latency_ms": cls._summary([item.e2e_latency for item in completed]),
            "input_throughput_tokens_per_s": input_tokens / makespan_s if makespan_s else 0.0,
            "output_throughput_tokens_per_s": output_tokens / makespan_s if makespan_s else 0.0,
            "throughput_tokens_per_s": output_tokens / makespan_s if makespan_s else 0.0,
            "goodput_tokens_per_s": good_tokens / makespan_s if makespan_s else 0.0,#只有同时满足两个 SLA 的请求才进入 goodput
            "goodput": {
                "ttft_sla_ms": ttft_sla_ms, "tpot_sla_ms": tpot_sla_ms,
                "ttft_satisfied": len(ttft_good),
                "tpot_satisfied": len(tpot_good),
                "both_satisfied": len(good),
                "ttft_ratio": len(ttft_good) / len(completed) if completed else 0.0,
                "tpot_ratio": len(tpot_good) / len(completed) if completed else 0.0,
                "both_ratio": len(good) / len(completed) if completed else 0.0,
                "token_goodput": good_tokens,
            },
            "simulation_time_ms": max(end - first_arrival, 0.0),
        }
        return stats


def demo_simulation(mode: str = "poisson") -> Dict:
    try:
        from .communication_model import CommunicationModel
    except ImportError:
        from communication_model import CommunicationModel
    data_dir = Path(__file__).resolve().parents[1] / "data"
    model = LLMCostModel(str(data_dir))
    # Replace this demo value with vLLM cache_config_info.num_gpu_blocks.
    pool = KVBlockPool(total_blocks=4096, block_size_tokens=16)
    checker = ResourceChecker(pool, CommunicationModel())
    scheduler = Scheduler(8192, 32, pool, model, checker)
    generator = RequestGenerator(mode=mode, arrival_interval_ms=100,
                                 prompt_len=512, output_len=32)
    return Simulator(model, pool, checker, scheduler, generator).run()


if __name__ == "__main__":
    import json
    print(json.dumps(demo_simulation(), indent=2, ensure_ascii=False))
