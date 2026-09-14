"""Prefill/decode-disaggregated discrete-event simulation.

Prefill and decode use independent TP replicas, schedulers and KV pools. A
request can enter decode only after its KV transfer has completed. Transfer
lanes are explicit serialized resources, so concurrent transfers contend for
finite bandwidth instead of receiving an unlimited link independently.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .communication_model import CommunicationModel
from .kv_block_pool import KVBlockPool
from .resource_checker import ResourceChecker
from .simulator import (
    Batch, MetricsCollector, Request, RequestGenerator, RequestStatus, Scheduler,
)


@dataclass
class PDConfig:
    """PD hardware, runtime and KV-transfer configuration."""

    num_prefill_gpus: int = 2
    prefill_tp: int = 2
    prefill_num_gpu_blocks: Optional[int] = None
    prefill_kv_memory_mb: float = 10000.0
    prefill_max_num_batched_tokens: int = 4096
    prefill_max_num_seqs: int = 8
    num_decode_gpus: int = 4
    decode_tp: int = 4
    decode_num_gpu_blocks: Optional[int] = None
    decode_kv_memory_mb: float = 20000.0
    decode_max_num_batched_tokens: int = 4096
    decode_max_num_seqs: int = 8
    block_size_tokens: int = 16
    enable_chunked_prefill: bool = True
    kv_transfer_mode: str = "pcie"
    kv_transfer_bw_gb_s: float = 12.5
    kv_transfer_latency_ms: float = 0.1
    kv_transfer_concurrency: int = 1
    decode_kv_load_policy: str = "post_transfer"

    def __post_init__(self) -> None:
        positive = {
            "num_prefill_gpus": self.num_prefill_gpus,
            "prefill_tp": self.prefill_tp,
            "num_decode_gpus": self.num_decode_gpus,
            "decode_tp": self.decode_tp,
            "block_size_tokens": self.block_size_tokens,
            "prefill_max_num_batched_tokens": self.prefill_max_num_batched_tokens,
            "prefill_max_num_seqs": self.prefill_max_num_seqs,
            "decode_max_num_batched_tokens": self.decode_max_num_batched_tokens,
            "decode_max_num_seqs": self.decode_max_num_seqs,
            "kv_transfer_concurrency": self.kv_transfer_concurrency,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"PD configuration values must be positive: {invalid}")
        if self.num_prefill_gpus % self.prefill_tp:
            raise ValueError("num_prefill_gpus must be divisible by prefill_tp")
        if self.num_decode_gpus % self.decode_tp:
            raise ValueError("num_decode_gpus must be divisible by decode_tp")
        if self.kv_transfer_bw_gb_s <= 0 or self.kv_transfer_latency_ms < 0:
            raise ValueError("KV transfer bandwidth must be positive and latency non-negative")
        if self.decode_kv_load_policy not in {"post_transfer", "blocking_receive"}:
            raise ValueError(
                "decode_kv_load_policy must be post_transfer or blocking_receive")
        for name, value in (
                ("prefill_num_gpu_blocks", self.prefill_num_gpu_blocks),
                ("decode_num_gpu_blocks", self.decode_num_gpu_blocks)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def prefill_replicas(self) -> int:
        return self.num_prefill_gpus // self.prefill_tp

    @property
    def decode_replicas(self) -> int:
        return self.num_decode_gpus // self.decode_tp

    @property
    def total_gpus(self) -> int:
        return self.num_prefill_gpus + self.num_decode_gpus


class PDRequestStatus(Enum):
    WAITING_PREFILL = "waiting_prefill"
    PREFILLING = "prefilling"
    WAITING_TRANSFER = "waiting_transfer"
    TRANSFERRING = "transferring"
    WAITING_DECODE = "waiting_decode"
    DECODING = "decoding"
    COMPLETED = "completed"


@dataclass
class PDRequest(Request):
    pd_status: PDRequestStatus = PDRequestStatus.WAITING_PREFILL
    kv_cache_size_mb: float = 0.0
    kv_transfer_start_time: float = 0.0
    kv_transfer_end_time: float = 0.0
    assigned_prefill_replica: int = -1
    assigned_decode_replica: int = -1
    kv_ready: bool = False
    decode_receive_start_time: float = 0.0
    decode_receive_end_time: float = 0.0


class KVTransferModel:
    """Latency plus payload/bandwidth transfer model."""

    def __init__(self, bandwidth_gb_s: float = 12.5,
                 latency_ms: float = 0.1, mode: str = "custom"):
        if bandwidth_gb_s <= 0 or latency_ms < 0:
            raise ValueError("invalid transfer bandwidth or latency")
        self.bandwidth_gb_s = bandwidth_gb_s
        self.latency_ms = latency_ms
        self.mode = mode

    def estimate_transfer_time(self, kv_cache_size_mb: float,
                               mode: Optional[str] = None) -> float:
        if kv_cache_size_mb < 0:
            raise ValueError("KV transfer size cannot be negative")
        # MB / (GB/s) has the same numeric value in milliseconds.
        return self.latency_ms + kv_cache_size_mb / self.bandwidth_gb_s


@dataclass
class _StageReplica:
    replica_id: int
    tp_size: int
    pool: KVBlockPool
    scheduler: Scheduler
    running_batch: Optional[Batch] = None
    blocked_kv_requests: List[PDRequest] = field(default_factory=list)
    # A blocking P2P receive belongs to the decode iteration that admitted it.
    # Once that cohort becomes ready, that iteration must run before another
    # receive cohort is admitted; otherwise successive receives collapse into
    # one ever-growing barrier and all first tokens appear at the same time.
    decode_iteration_ready: bool = False


class PDSimulator:
    def __init__(self, pd_config: PDConfig, cost_model: Any,
                 request_generator: RequestGenerator,
                 resource_checker: Optional[ResourceChecker] = None,
                 max_concurrent_requests: Optional[int] = None,
                 max_batch_token: Optional[int] = None,
                 verbose: bool = False):
        self.config = pd_config
        self.cost_model = cost_model
        self.request_generator = request_generator
        self.verbose = verbose
        # Backward-compatible constructor overrides used by older callers.
        if max_concurrent_requests is not None:
            self.config.prefill_max_num_seqs = max_concurrent_requests
            self.config.decode_max_num_seqs = max_concurrent_requests
        if max_batch_token is not None:
            self.config.prefill_max_num_batched_tokens = max_batch_token
            self.config.decode_max_num_batched_tokens = max_batch_token
        self.transfer_model = KVTransferModel(
            pd_config.kv_transfer_bw_gb_s,
            pd_config.kv_transfer_latency_ms,
            pd_config.kv_transfer_mode)
        self.prefill_replicas = self._build_replicas("prefill")
        self.decode_replicas = self._build_replicas("decode")
        self.prefill_kv_pool = self.prefill_replicas[0].pool
        self.decode_kv_pool = self.decode_replicas[0].pool
        self.reset()

    def _pool(self, stage: str, tp_size: int) -> KVBlockPool:
        blocks = getattr(self.config, f"{stage}_num_gpu_blocks")
        bytes_per_token = self.cost_model.memory_estimator.kv_bytes_per_token_per_rank(
            tp_size)
        if blocks is not None:
            return KVBlockPool(blocks, self.config.block_size_tokens, bytes_per_token)
        memory_mb = getattr(self.config, f"{stage}_kv_memory_mb")
        return KVBlockPool.from_memory_budget(
            memory_mb, self.config.block_size_tokens, bytes_per_token)

    def _build_replicas(self, stage: str) -> List[_StageReplica]:
        tp_size = getattr(self.config, f"{stage}_tp")
        count = getattr(self.config, f"{stage}_replicas")
        max_tokens = getattr(self.config, f"{stage}_max_num_batched_tokens")
        max_seqs = getattr(self.config, f"{stage}_max_num_seqs")
        replicas = []
        for replica_id in range(count):
            pool = self._pool(stage, tp_size)
            checker = ResourceChecker(pool, CommunicationModel())
            scheduler = Scheduler(
                max_tokens, max_seqs, pool, self.cost_model, checker,
                enable_chunked_prefill=(
                    self.config.enable_chunked_prefill if stage == "prefill" else False))
            replicas.append(_StageReplica(replica_id, tp_size, pool, scheduler))
        return replicas

    def reset(self) -> None:
        self.current_time = 0.0
        self.prefill_queue: List[PDRequest] = []
        self.decode_queue: List[PDRequest] = []
        self.decode_receive_queue: List[PDRequest] = []
        self.completed_requests: List[PDRequest] = []
        self.requests: List[PDRequest] = []
        self.event_queue: List[Tuple[float, int, str, int, Any]] = []
        self.event_counter = 0
        self.finished_requests = 0
        self.batch_history: List[Dict[str, Any]] = []
        self.transfer_history: List[Dict[str, Any]] = []
        self.decode_receive_history: List[Dict[str, Any]] = []
        self.transfer_lane_available = [0.0] * self.config.kv_transfer_concurrency
        for replica in self.prefill_replicas + self.decode_replicas:
            replica.running_batch = None
            replica.blocked_kv_requests.clear()
            replica.decode_iteration_ready = False
            replica.pool.release_all()
            replica.scheduler.total_batches = 0
        self.request_generator.reset()

    def _add_event(self, time: float, event_type: str,
                   replica_id: int, data: Any) -> None:
        self.event_counter += 1
        heapq.heappush(self.event_queue, (
            time, self.event_counter, event_type, replica_id, data))

    @staticmethod
    def _eligible(queue: Sequence[PDRequest], stage: str,
                  replica_id: int) -> List[PDRequest]:
        attribute = ("assigned_prefill_replica" if stage == "prefill"
                     else "assigned_decode_replica")
        return [request for request in queue
                if getattr(request, attribute) in {-1, replica_id}]

    def _prediction(self, batch: Batch, stage: str, tp_size: int) -> Dict[str, Any]:
        if stage == "prefill":
            return self.cost_model.predict_batch(
                batch.prefill_chunk_lengths, [], tp_size,
                prefill_context_lengths=batch.prefill_context_lengths)
        return self.cost_model.predict_batch(
            [], [request.kv_len for request in batch.requests], tp_size)

    def _record_batch(self, batch: Batch, stage: str,
                      replica: _StageReplica) -> None:
        prediction = batch.prediction or {}
        breakdown = prediction.get("breakdown", {})
        self.batch_history.append({
            "batch_id": f"{stage}_r{replica.replica_id}_{batch.batch_id}",
            "stage": stage, "replica_id": replica.replica_id,
            "tp_size": replica.tp_size, "start_time_ms": batch.start_time,
            "end_time_ms": batch.end_time, "batch_size": batch.batch_size,
            "query_tokens": batch.total_tokens,
            "prefill_tokens": sum(batch.prefill_chunk_lengths),
            "decode_count": len(batch.decode_requests),
            "compute_ms": float(breakdown.get("compute_ms", 0.0)),
            "communication_ms": float(breakdown.get("comm_ms", 0.0)),
            "iteration_overhead_ms": float(
                breakdown.get("iteration_overhead_ms", 0.0)),
            "estimated_time_ms": batch.estimated_time_ms,
            "kv_blocks_used": replica.pool.get_used_blocks(),
            "kv_blocks_total": replica.pool.get_total_blocks(),
            "kv_block_utilization": replica.pool.get_utilization(),
            "extrapolated": bool(prediction.get("extrapolated", False)),
        })

    def _schedule_stage(self, stage: str) -> bool:
        replicas = self.prefill_replicas if stage == "prefill" else self.decode_replicas
        queue = self.prefill_queue if stage == "prefill" else self.decode_queue
        scheduled_any = False
        for replica in replicas:
            if (replica.running_batch is not None or
                    (stage == "decode" and replica.blocked_kv_requests)):
                continue
            batch = replica.scheduler.schedule(
                self._eligible(queue, stage, replica.replica_id),
                self.current_time, replica.tp_size)
            if batch is None:
                continue
            ids = {request.request_id for request in batch.requests}
            queue[:] = [request for request in queue if request.request_id not in ids]
            for request in batch.requests:
                if stage == "prefill":
                    request.assigned_prefill_replica = replica.replica_id
                    request.pd_status = PDRequestStatus.PREFILLING
                    if request.prefilled_tokens == 0:
                        request.prefill_start_time = self.current_time
                else:
                    request.assigned_decode_replica = replica.replica_id
                    request.pd_status = PDRequestStatus.DECODING
                    request.status = RequestStatus.DECODING
                    if request.generated_tokens == 0:
                        request.decode_start_time = self.current_time
            batch.start_time = self.current_time
            batch.prediction = self._prediction(batch, stage, replica.tp_size)
            batch.estimated_time_ms = float(batch.prediction["total_time_ms"])
            batch.end_time = self.current_time + batch.estimated_time_ms
            replica.running_batch = batch
            if stage == "decode":
                replica.decode_iteration_ready = False
            self._record_batch(batch, stage, replica)
            self._add_event(batch.end_time, f"{stage}_complete",
                            replica.replica_id, batch)
            scheduled_any = True
        return scheduled_any

    def _release_blocked_decode(self, replica: _StageReplica) -> bool:
        blocked = replica.blocked_kv_requests
        if not blocked or not all(request.kv_ready for request in blocked):
            return False
        for request in blocked:
            request.decode_receive_end_time = self.current_time
            request.status = RequestStatus.DECODING
            request.pd_status = PDRequestStatus.WAITING_DECODE
            self.decode_queue.append(request)
            self.decode_receive_history.append({
                "request_id": request.request_id,
                "replica_id": replica.replica_id,
                "start_time_ms": request.decode_receive_start_time,
                "end_time_ms": self.current_time,
                "blocked_time_ms": (
                    self.current_time - request.decode_receive_start_time),
            })
        blocked.clear()
        replica.decode_iteration_ready = True
        return True

    def _schedule_decode_receives(self) -> bool:
        """Model P2P's synchronous worker-side recv_tensor admission."""
        if self.config.decode_kv_load_policy != "blocking_receive":
            return False
        scheduled_any = False
        for replica in self.decode_replicas:
            if (replica.running_batch is not None or
                    replica.blocked_kv_requests or
                    replica.decode_iteration_ready):
                continue
            active = sum(
                request.assigned_decode_replica == replica.replica_id
                for request in self.decode_queue)
            slots = replica.scheduler.max_concurrent_requests - active
            if slots <= 0:
                continue
            selected: List[PDRequest] = []
            for request in self.decode_receive_queue:
                if request.assigned_decode_replica != -1:
                    continue
                required = replica.pool.blocks_for_tokens(request.prompt_len)
                if not replica.pool.ensure_capacity(request.request_id, required):
                    continue
                request.assigned_decode_replica = replica.replica_id
                request.decode_receive_start_time = self.current_time
                selected.append(request)
                if len(selected) >= slots:
                    break
            if not selected:
                continue
            selected_ids = {request.request_id for request in selected}
            self.decode_receive_queue[:] = [
                request for request in self.decode_receive_queue
                if request.request_id not in selected_ids]
            replica.blocked_kv_requests.extend(selected)
            self._release_blocked_decode(replica)
            scheduled_any = True
        return scheduled_any

    def _schedule_transfer(self, request: PDRequest,
                           source_replica: _StageReplica) -> None:
        blocks = source_replica.pool.blocks_for_tokens(request.prompt_len)
        size_mb = blocks * source_replica.pool.block_size_mib * source_replica.tp_size
        duration = self.transfer_model.estimate_transfer_time(size_mb)
        lane = min(range(len(self.transfer_lane_available)),
                   key=self.transfer_lane_available.__getitem__)
        start = max(self.current_time, self.transfer_lane_available[lane])
        end = start + duration
        self.transfer_lane_available[lane] = end
        request.kv_cache_size_mb = size_mb
        request.kv_transfer_start_time = start
        request.kv_transfer_end_time = end
        request.pd_status = PDRequestStatus.TRANSFERRING
        self.transfer_history.append({
            "request_id": request.request_id,
            "source_replica_id": source_replica.replica_id,
            "lane_id": lane, "mode": self.config.kv_transfer_mode,
            "size_mb": size_mb, "queued_time_ms": start - self.current_time,
            "start_time_ms": start, "end_time_ms": end,
            "duration_ms": duration,
        })
        self._add_event(end, "transfer_complete", source_replica.replica_id, request)

    def _complete_request(self, request: PDRequest,
                          decode_replica: Optional[_StageReplica] = None) -> None:
        request.status = RequestStatus.COMPLETED
        request.pd_status = PDRequestStatus.COMPLETED
        request.completion_time = self.current_time
        self.completed_requests.append(request)
        self.finished_requests += 1
        if decode_replica is not None:
            decode_replica.pool.release(request.request_id)

    def _prefill_complete(self, replica: _StageReplica, batch: Batch) -> None:
        replica.running_batch = None
        for request in batch.requests:
            request.prefilled_tokens += batch.scheduled_token_counts.get(
                request.request_id, request.remaining_prefill_tokens)
            if request.prefilled_tokens < request.prompt_len:
                request.pd_status = PDRequestStatus.WAITING_PREFILL
                self.prefill_queue.append(request)
                continue
            request.prefill_end_time = self.current_time
            if request.output_len == 0:
                replica.pool.release(request.request_id)
                self._complete_request(request)
            else:
                request.pd_status = PDRequestStatus.WAITING_TRANSFER
                self._schedule_transfer(request, replica)

    def _transfer_complete(self, source_replica: _StageReplica,
                           request: PDRequest) -> None:
        source_replica.pool.release(request.request_id)
        request.kv_ready = True
        if self.config.decode_kv_load_policy == "blocking_receive":
            if request.assigned_decode_replica >= 0:
                self._release_blocked_decode(
                    self.decode_replicas[request.assigned_decode_replica])
        else:
            request.status = RequestStatus.DECODING
            request.pd_status = PDRequestStatus.WAITING_DECODE
            self.decode_queue.append(request)

    def _decode_complete(self, replica: _StageReplica, batch: Batch) -> None:
        replica.running_batch = None
        for request in batch.requests:
            request.generated_tokens += 1
            request.last_decode_end_time = self.current_time
            if request.generated_tokens == 1:
                request.first_token_time = self.current_time
            if request.generated_tokens >= request.output_len:
                self._complete_request(request, replica)
            else:
                request.pd_status = PDRequestStatus.WAITING_DECODE
                self.decode_queue.append(request)

    def _process_events(self) -> None:
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            _, _, event_type, replica_id, data = heapq.heappop(self.event_queue)
            if event_type == "prefill_complete":
                self._prefill_complete(self.prefill_replicas[replica_id], data)
            elif event_type == "transfer_complete":
                self._transfer_complete(self.prefill_replicas[replica_id], data)
            elif event_type == "decode_complete":
                self._decode_complete(self.decode_replicas[replica_id], data)

    def run(self, simulation_duration_ms: float = 30000.0,
            ttft_sla_ms: float = 500.0, tpot_sla_ms: float = 50.0,
            num_requests: int = 20) -> Dict[str, Any]:
        self.reset()
        base_requests = self.request_generator.generate_requests(num_requests)
        self.requests = [PDRequest(
            request.request_id, request.prompt_len, request.output_len,
            request.arrival_time) for request in base_requests]
        request_index = 0
        while self.current_time <= simulation_duration_ms:
            while (request_index < len(self.requests) and
                   self.requests[request_index].arrival_time <= self.current_time):
                request = self.requests[request_index]
                self.prefill_queue.append(request)
                if (self.config.decode_kv_load_policy == "blocking_receive" and
                        request.output_len > 0):
                    self.decode_receive_queue.append(request)
                request_index += 1
            self._schedule_stage("prefill")
            self._schedule_decode_receives()
            self._schedule_stage("decode")
            if self.finished_requests == len(self.requests) and not self.event_queue:
                break
            next_times = []
            if request_index < len(self.requests):
                next_times.append(self.requests[request_index].arrival_time)
            if self.event_queue:
                next_times.append(self.event_queue[0][0])
            if not next_times:
                break
            next_time = min(next_times)
            if next_time > simulation_duration_ms:
                break
            self.current_time = next_time
            self._process_events()
        metrics = MetricsCollector.collect(
            self.requests, ttft_sla_ms, tpot_sla_ms,
            observation_end_ms=self.current_time)
        metrics["pd"] = {
            "total_gpus": self.config.total_gpus,
            "prefill_replicas": self.config.prefill_replicas,
            "decode_replicas": self.config.decode_replicas,
            "transfer_count": len(self.transfer_history),
            "transfer_size_mb": sum(row["size_mb"] for row in self.transfer_history),
            "transfer_queue_time_ms": sum(
                row["queued_time_ms"] for row in self.transfer_history),
            "decode_kv_load_policy": self.config.decode_kv_load_policy,
            "decode_blocking_receive_count": len(self.decode_receive_history),
            "decode_blocking_receive_time_ms": sum(
                row["blocked_time_ms"] for row in self.decode_receive_history),
            "prefill_batch_count": sum(
                row["stage"] == "prefill" for row in self.batch_history),
            "decode_batch_count": sum(
                row["stage"] == "decode" for row in self.batch_history),
            "prefill_peak_kv_utilization": max(
                (row["kv_block_utilization"] for row in self.batch_history
                 if row["stage"] == "prefill"), default=0.0),
            "decode_peak_kv_utilization": max(
                (row["kv_block_utilization"] for row in self.batch_history
                 if row["stage"] == "decode"), default=0.0),
        }
        return metrics


def demo_pd_simulation(mode: str = "fixed") -> Dict[str, Any]:
    from pathlib import Path
    from .qwen3_cost_model import LLMCostModel

    data_dir = Path(__file__).resolve().parents[1] / "data"
    model = LLMCostModel(str(data_dir))
    config = PDConfig(
        num_prefill_gpus=2, prefill_tp=2, prefill_num_gpu_blocks=4096,
        num_decode_gpus=4, decode_tp=4, decode_num_gpu_blocks=8192)
    generator = RequestGenerator(
        mode=mode, arrival_interval_ms=100, prompt_len=512, output_len=32)
    return PDSimulator(config, model, generator).run(
        simulation_duration_ms=100000)


if __name__ == "__main__":
    import json
    print(json.dumps(demo_pd_simulation(), indent=2, ensure_ascii=False))
