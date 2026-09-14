"""Reproducible configuration and SLA-capacity experiments.

This module deliberately builds on the validated co-located simulator.  It
does not contain another scheduling model: every point creates a fresh
``Simulator`` and changes only declared serving/workload parameters.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .communication_model import CommunicationModel
from .kv_block_pool import KVBlockPool
from .resource_checker import ResourceChecker
from .simulator import MetricsCollector, Request, RequestGenerator, Scheduler, Simulator


@dataclass(frozen=True)
class ServingConfig:
    tp_size: int
    num_gpu_blocks: int
    block_size_tokens: int
    max_num_batched_tokens: int
    max_num_seqs: int
    chunked_prefill: bool
    data_parallel: int = 1 # DP 在这里指多个独立推理副本，不是训练 DP。因此不会额外产生训练式 AllReduce。

    @property
    def config_id(self) -> str:
        chunked = "chunked" if self.chunked_prefill else "unchunked"
        return (f"tp{self.tp_size}_dp{self.data_parallel}_kv{self.num_gpu_blocks}_"
                f"bt{self.max_num_batched_tokens}_seq{self.max_num_seqs}_{chunked}")

    @property
    def total_gpus(self) -> int:
        return self.tp_size * self.data_parallel


@dataclass(frozen=True)
class WorkloadSpec:
    mode: str = "poisson"
    prompt_len: int = 512
    output_len: int = 128
    prompt_len_range: Optional[Tuple[int, int]] = None
    output_len_range: Optional[Tuple[int, int]] = None
    num_requests: int = 200
    simulation_duration_ms: float = 1_000_000.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "poisson"}:
            raise ValueError("capacity search supports fixed or poisson workloads")
        if self.prompt_len <= 0 or self.output_len < 0:
            raise ValueError("prompt_len must be positive and output_len non-negative")
        if (self.prompt_len_range and
                (self.prompt_len_range[0] <= 0 or
                 self.prompt_len_range[0] > self.prompt_len_range[1])):
            raise ValueError("prompt_len_range must be an increasing positive range")
        if (self.output_len_range and
                (self.output_len_range[0] < 0 or
                 self.output_len_range[0] > self.output_len_range[1])):
            raise ValueError("output_len_range must be an increasing non-negative range")
        if self.num_requests <= 0 or self.simulation_duration_ms <= 0:
            raise ValueError("num_requests and simulation_duration_ms must be positive")


@dataclass(frozen=True)
class CapacityPolicy:
    ttft_sla_ms: float = 500.0
    tpot_sla_ms: float = 50.0
    min_completion_ratio: float = 1.0 # 最低完成率
    min_sla_success_ratio: float = 0.9 # 最低 SLA 成功率。只有完成率和 SLA 成功率同时满足，某个请求率才会被判定为可持续容量点。

    def __post_init__(self) -> None:
        if self.ttft_sla_ms <= 0 or self.tpot_sla_ms <= 0:
            raise ValueError("SLA thresholds must be positive")
        if not 0 <= self.min_completion_ratio <= 1:
            raise ValueError("min_completion_ratio must be in [0, 1]")
        if not 0 <= self.min_sla_success_ratio <= 1:
            raise ValueError("min_sla_success_ratio must be in [0, 1]")


def serving_grid(tp_sizes: Sequence[int], num_gpu_blocks: Sequence[int],
                 block_sizes: Sequence[int], batched_token_limits: Sequence[int],
                 sequence_limits: Sequence[int],
                 chunked_prefill_values: Sequence[bool],
                 data_parallel_sizes: Sequence[int] = (1,)) -> List[ServingConfig]:
    """Return the Cartesian product in a stable, reproducible order."""
    configs = []
    for values in itertools.product(
            tp_sizes, num_gpu_blocks, block_sizes, batched_token_limits,
            sequence_limits, chunked_prefill_values, data_parallel_sizes):
        config = ServingConfig(*values)
        if min(config.tp_size, config.num_gpu_blocks, config.block_size_tokens,
               config.max_num_batched_tokens, config.max_num_seqs,
               config.data_parallel) <= 0:
            raise ValueError("all serving configuration values must be positive")
        configs.append(config)
    return configs


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


class _ReplayGenerator:
    """Create fresh Request objects from an in-memory, absolute-time trace."""

    def __init__(self, rows: Sequence[Request]):
        self.rows = list(rows)
        self.generated: List[Request] = []

    def reset(self) -> None:
        self.generated = []

    def generate_requests(self, num_requests: Optional[int] = None) -> List[Request]:
        selected = self.rows if num_requests is None else self.rows[:num_requests]
        self.generated = [Request(
            row.request_id, row.prompt_len, row.output_len, row.arrival_time)
            for row in selected]
        return self.generated


class CapacityExperimentRunner:
    def __init__(self, cost_model: Any, workload: WorkloadSpec,
                 policy: CapacityPolicy, repeats: int = 3):
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.cost_model = cost_model
        self.workload = workload
        self.policy = policy
        self.repeats = repeats

    def _run_once(self, config: ServingConfig, arrival_rate_rps: float,
                  repeat: int) -> Dict[str, Any]:
        if arrival_rate_rps <= 0:
            raise ValueError("arrival rates must be positive")
        source_generator = RequestGenerator(
            mode=self.workload.mode,
            arrival_interval_ms=1000.0 / arrival_rate_rps,
            prompt_len=self.workload.prompt_len,
            output_len=self.workload.output_len,
            prompt_len_range=self.workload.prompt_len_range,
            output_len_range=self.workload.output_len_range,
            seed=self.workload.seed + repeat,
        )
        offered_requests = source_generator.generate_requests(
            self.workload.num_requests)
        # Round-robin request routing models independent DP serving replicas.
        # Each replica has its own scheduler and KV pool; TP cost stays local to
        # the replica and no training-style DP all-reduce is added for inference.
        replica_rows = [offered_requests[index::config.data_parallel]
                        for index in range(config.data_parallel)]
        all_requests: List[Request] = []
        batches: List[Dict[str, Any]] = []
        observation_end_ms = 0.0
        memory_estimator = getattr(self.cost_model, "memory_estimator", None)
        kv_bytes = (memory_estimator.kv_bytes_per_token_per_rank(config.tp_size)
                    if memory_estimator else 0.0)
        for replica_index, rows in enumerate(replica_rows):
            if not rows:
                continue
            pool = KVBlockPool(
                total_blocks=config.num_gpu_blocks,
                block_size_tokens=config.block_size_tokens,
                kv_bytes_per_token_per_rank=kv_bytes,
            )
            checker = ResourceChecker(pool, CommunicationModel())
            scheduler = Scheduler(
                config.max_num_batched_tokens,
                config.max_num_seqs,
                pool,
                self.cost_model,
                checker,
                enable_chunked_prefill=config.chunked_prefill,
            )
            generator = _ReplayGenerator(rows)
            simulator = Simulator(
                self.cost_model, pool, checker, scheduler, generator,
                tp_size=config.tp_size,
            )
            simulator.run(
                simulation_duration_ms=self.workload.simulation_duration_ms,
                ttft_sla_ms=self.policy.ttft_sla_ms,
                tpot_sla_ms=self.policy.tpot_sla_ms,
                num_requests=len(rows),
            )
            all_requests.extend(generator.generated)
            observation_end_ms = max(observation_end_ms, simulator.current_time)
            for batch in simulator.batch_history:
                batches.append({**batch, "replica_id": replica_index})
        metrics = MetricsCollector.collect(
            all_requests, self.policy.ttft_sla_ms, self.policy.tpot_sla_ms,
            observation_end_ms=observation_end_ms)
        extrapolated = sum(bool(row["extrapolated"]) for row in batches)
        modeled_time = sum(float(row["estimated_time_ms"]) for row in batches)
        compute_time = sum(float(row["compute_ms"]) for row in batches)
        communication_time = sum(
            float(row["communication_ms"]) for row in batches)
        max_kv_utilization = max(
            (float(row["kv_block_utilization"]) for row in batches), default=0.0)
        total_requests = int(metrics["total_requests"])
        completed = int(metrics["completed_requests"])
        both_satisfied = int(metrics["goodput"]["both_satisfied"])
        simulation_s = float(metrics["simulation_time_ms"]) / 1000.0
        return {
            "repeat": repeat + 1,
            "seed": self.workload.seed + repeat,
            "total_requests": total_requests,
            "completed_requests": completed,
            "completion_ratio": completed / total_requests if total_requests else 0.0,
            # Denominator is all offered requests, not only completed requests.
            "sla_success_ratio": (
                both_satisfied / total_requests if total_requests else 0.0),
            "ttft_p50_ms": metrics["ttft_ms"]["p50"],
            "ttft_p99_ms": metrics["ttft_ms"]["p99"],
            "tpot_p50_ms": metrics["tpot_ms"]["p50"],
            "tpot_p99_ms": metrics["tpot_ms"]["p99"],
            "e2e_p99_ms": metrics["e2e_latency_ms"]["p99"],
            "output_throughput_tokens_per_s": metrics[
                "output_throughput_tokens_per_s"],
            "goodput_tokens_per_s": metrics["goodput_tokens_per_s"],
            "completed_requests_per_s": (
                completed / simulation_s if simulation_s else 0.0),
            "batch_count": len(batches),
            "extrapolated_batch_ratio": (
                extrapolated / len(batches) if batches else 0.0),
            "max_kv_block_utilization": max_kv_utilization,
            "compute_time_share": compute_time / modeled_time if modeled_time else 0.0,
            "communication_time_share": (
                communication_time / modeled_time if modeled_time else 0.0),
        }

    def run_point(self, config: ServingConfig,
                  arrival_rate_rps: float) -> Dict[str, Any]:
        runs = [self._run_once(config, arrival_rate_rps, repeat)
                for repeat in range(self.repeats)]
        completion_ratio = _mean(runs, "completion_ratio")
        sla_success_ratio = _mean(runs, "sla_success_ratio")
        sustainable = (
            min(row["completion_ratio"] for row in runs) >=
            self.policy.min_completion_ratio and
            sla_success_ratio >= self.policy.min_sla_success_ratio)
        max_kv_utilization = max(
            row["max_kv_block_utilization"] for row in runs)
        compute_share = _mean(runs, "compute_time_share")
        communication_share = _mean(runs, "communication_time_share")
        if max_kv_utilization >= 0.9:
            estimated_bottleneck = "memory"
        elif communication_share >= 0.5:
            estimated_bottleneck = "communication"
        elif compute_share >= 0.5:
            estimated_bottleneck = "compute"
        else:
            estimated_bottleneck = "balanced"
        mean_prompt_len = (sum(self.workload.prompt_len_range) / 2
                           if self.workload.prompt_len_range
                           else self.workload.prompt_len)
        mean_output_len = (sum(self.workload.output_len_range) / 2
                           if self.workload.output_len_range
                           else self.workload.output_len)
        return {
            "config_id": config.config_id,
            **asdict(config),
            "arrival_rate_rps": arrival_rate_rps,
            "total_gpus": config.total_gpus,
            "offered_input_tokens_per_s": arrival_rate_rps * mean_prompt_len,
            "offered_output_tokens_per_s": arrival_rate_rps * mean_output_len,
            "sustainable": sustainable,
            "completion_ratio": completion_ratio,
            "worst_completion_ratio": min(row["completion_ratio"] for row in runs),
            "sla_success_ratio": sla_success_ratio,
            "ttft_p50_ms": _mean(runs, "ttft_p50_ms"),
            "ttft_p99_ms": _mean(runs, "ttft_p99_ms"),
            "tpot_p50_ms": _mean(runs, "tpot_p50_ms"),
            "tpot_p99_ms": _mean(runs, "tpot_p99_ms"),
            "e2e_p99_ms": _mean(runs, "e2e_p99_ms"),
            "output_throughput_tokens_per_s": _mean(
                runs, "output_throughput_tokens_per_s"),
            "goodput_tokens_per_s": _mean(runs, "goodput_tokens_per_s"),
            "goodput_tokens_per_s_per_gpu": (
                _mean(runs, "goodput_tokens_per_s") / config.total_gpus),
            "completed_requests_per_s": _mean(runs, "completed_requests_per_s"),
            "extrapolated_batch_ratio": _mean(runs, "extrapolated_batch_ratio"),
            "max_kv_block_utilization": max_kv_utilization,
            "compute_time_share": compute_share,
            "communication_time_share": communication_share,
            "estimated_bottleneck": estimated_bottleneck,
            "runs": runs,
        }

    def run(self, configs: Iterable[ServingConfig],
            arrival_rates_rps: Sequence[float],
            refine_iterations: int = 0) -> Dict[str, Any]:
        if refine_iterations < 0:
            raise ValueError("refine_iterations must be non-negative")
        config_list = list(configs)
        rates = sorted(set(float(rate) for rate in arrival_rates_rps))
        if not rates or rates[0] <= 0:
            raise ValueError("at least one positive arrival rate is required")
        points = [self.run_point(config, rate)
                  for config in config_list for rate in rates]

        # Refine the first pass/fail transition for every configuration. The
        # same repeat seeds are used at every midpoint.
        for config in config_list:
            for _ in range(refine_iterations):
                config_points = sorted(
                    (point for point in points
                     if point["config_id"] == config.config_id),
                    key=lambda row: row["arrival_rate_rps"])
                lower = None
                upper = None
                for point in config_points:
                    if point["sustainable"] and upper is None:
                        lower = point
                    elif not point["sustainable"]:
                        upper = point
                        break
                if lower is None or upper is None:
                    break
                midpoint = (lower["arrival_rate_rps"] +
                            upper["arrival_rate_rps"]) / 2.0
                if any(abs(point["arrival_rate_rps"] - midpoint) < 1e-12
                       for point in config_points):
                    break
                points.append(self.run_point(config, midpoint))

        points.sort(key=lambda row: (row["config_id"], row["arrival_rate_rps"]))
        capacities = []
        for config_id in dict.fromkeys(point["config_id"] for point in points):
            config_points = sorted(
                (point for point in points if point["config_id"] == config_id),
                key=lambda row: row["arrival_rate_rps"])
            raw_states = [bool(point["sustainable"]) for point in config_points]
            non_monotonic = any(
                raw_states[index] and not all(raw_states[:index])
                for index in range(1, len(raw_states)))
            # Capacity is the contiguous sustainable prefix. A noisy high-rate
            # pass after a lower-rate failure must not become the capacity.
            candidates = []
            for point in config_points:
                if not point["sustainable"]:
                    break
                candidates.append(point)
            if candidates:
                best = candidates[-1]
                first_failure = next(
                    (point for point in config_points[len(candidates):]
                     if not point["sustainable"]), None)
                upper_bound = (first_failure["arrival_rate_rps"]
                               if first_failure else None)
                capacities.append({
                    "rank": 0,
                    "config_id": config_id,
                    "max_sustainable_arrival_rate_rps": best["arrival_rate_rps"],
                    "goodput_tokens_per_s_at_capacity": best["goodput_tokens_per_s"],
                    "total_gpus": best["total_gpus"],
                    "capacity_rps_per_gpu": (
                        best["arrival_rate_rps"] / best["total_gpus"]),
                    "goodput_tokens_per_s_per_gpu_at_capacity": best[
                        "goodput_tokens_per_s_per_gpu"],
                    "sla_success_ratio_at_capacity": best["sla_success_ratio"],
                    "ttft_p99_ms_at_capacity": best["ttft_p99_ms"],
                    "tpot_p99_ms_at_capacity": best["tpot_p99_ms"],
                    "max_kv_block_utilization_at_capacity": best[
                        "max_kv_block_utilization"],
                    "extrapolated_batch_ratio_at_capacity": best[
                        "extrapolated_batch_ratio"],
                    "estimated_bottleneck_at_capacity": best["estimated_bottleneck"],
                    "capacity_lower_bound_rps": best["arrival_rate_rps"],
                    "capacity_upper_bound_rps": upper_bound,
                    "capacity_interval_width_rps": (
                        upper_bound - best["arrival_rate_rps"]
                        if upper_bound is not None else None),
                    "capacity_censored": upper_bound is None,
                    "non_monotonic_points": non_monotonic,
                })
            else:
                upper_bound = (config_points[0]["arrival_rate_rps"]
                               if config_points else None)
                capacities.append({
                    "rank": 0,
                    "config_id": config_id,
                    "max_sustainable_arrival_rate_rps": None,
                    "goodput_tokens_per_s_at_capacity": 0.0,
                    "total_gpus": config_points[0]["total_gpus"],
                    "capacity_rps_per_gpu": None,
                    "goodput_tokens_per_s_per_gpu_at_capacity": 0.0,
                    "sla_success_ratio_at_capacity": 0.0,
                    "ttft_p99_ms_at_capacity": None,
                    "tpot_p99_ms_at_capacity": None,
                    "max_kv_block_utilization_at_capacity": None,
                    "extrapolated_batch_ratio_at_capacity": None,
                    "estimated_bottleneck_at_capacity": None,
                    "capacity_lower_bound_rps": None,
                    "capacity_upper_bound_rps": upper_bound,
                    "capacity_interval_width_rps": None,
                    "capacity_censored": False,
                    "non_monotonic_points": non_monotonic,
                })
        capacities.sort(
            key=lambda row: (row["max_sustainable_arrival_rate_rps"] is not None,
                             row["max_sustainable_arrival_rate_rps"] or 0.0,
                             row["goodput_tokens_per_s_at_capacity"]),
            reverse=True,
        )
        for rank, row in enumerate(capacities, 1):
            row["rank"] = rank
        efficiency_ranking = sorted(
            (dict(row) for row in capacities),
            key=lambda row: (row["capacity_rps_per_gpu"] is not None,
                             row["capacity_rps_per_gpu"] or 0.0,
                             row["goodput_tokens_per_s_per_gpu_at_capacity"]),
            reverse=True)
        for rank, row in enumerate(efficiency_ranking, 1):
            row["efficiency_rank"] = rank
        return {
            "schema_version": 1,
            "experiment_type": "co_located_sla_capacity_search",
            "workload": asdict(self.workload),
            "capacity_policy": asdict(self.policy),
            "repeats": self.repeats,
            "initial_arrival_rates_rps": rates,
            "evaluated_arrival_rates_rps": sorted(set(
                point["arrival_rate_rps"] for point in points)),
            "refine_iterations": refine_iterations,
            "points": points,
            "capacity_ranking": capacities,
            "efficiency_ranking": efficiency_ranking,
        }
