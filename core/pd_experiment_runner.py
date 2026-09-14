"""Capacity experiments for prefill/decode-disaggregated serving."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .experiment_runner import CapacityPolicy, WorkloadSpec
from .pd_simulator import PDConfig, PDSimulator
from .simulator import RequestGenerator


@dataclass(frozen=True)
class PDServingConfig:
    prefill_gpus: int
    prefill_tp: int
    decode_gpus: int
    decode_tp: int
    prefill_num_gpu_blocks: int
    decode_num_gpu_blocks: int
    block_size_tokens: int
    prefill_max_num_batched_tokens: int
    prefill_max_num_seqs: int
    decode_max_num_batched_tokens: int
    decode_max_num_seqs: int
    chunked_prefill: bool
    kv_transfer_mode: str
    kv_transfer_bw_gb_s: float
    kv_transfer_latency_ms: float
    kv_transfer_concurrency: int
    decode_kv_load_policy: str = "post_transfer"

    @property
    def prefill_replicas(self) -> int:
        return self.prefill_gpus // self.prefill_tp

    @property
    def decode_replicas(self) -> int:
        return self.decode_gpus // self.decode_tp

    @property
    def total_gpus(self) -> int:
        return self.prefill_gpus + self.decode_gpus

    @property
    def config_id(self) -> str:
        chunked = "chunked" if self.chunked_prefill else "unchunked"
        return (
            f"p{self.prefill_gpus}tp{self.prefill_tp}r{self.prefill_replicas}_"
            f"d{self.decode_gpus}tp{self.decode_tp}r{self.decode_replicas}_"
            f"pkv{self.prefill_num_gpu_blocks}_dkv{self.decode_num_gpu_blocks}_"
            f"blk{self.block_size_tokens}_"
            f"pbt{self.prefill_max_num_batched_tokens}_"
            f"ps{self.prefill_max_num_seqs}_"
            f"dbt{self.decode_max_num_batched_tokens}_"
            f"ds{self.decode_max_num_seqs}_{chunked}_"
            f"{self.kv_transfer_mode}{self.kv_transfer_bw_gb_s:g}_"
            f"lat{self.kv_transfer_latency_ms:g}_"
            f"x{self.kv_transfer_concurrency}_"
            f"kvload-{self.decode_kv_load_policy}")

    def to_pd_config(self) -> PDConfig:
        return PDConfig(
            num_prefill_gpus=self.prefill_gpus,
            prefill_tp=self.prefill_tp,
            prefill_num_gpu_blocks=self.prefill_num_gpu_blocks,
            prefill_max_num_batched_tokens=self.prefill_max_num_batched_tokens,
            prefill_max_num_seqs=self.prefill_max_num_seqs,
            num_decode_gpus=self.decode_gpus,
            decode_tp=self.decode_tp,
            decode_num_gpu_blocks=self.decode_num_gpu_blocks,
            decode_max_num_batched_tokens=self.decode_max_num_batched_tokens,
            decode_max_num_seqs=self.decode_max_num_seqs,
            block_size_tokens=self.block_size_tokens,
            enable_chunked_prefill=self.chunked_prefill,
            kv_transfer_mode=self.kv_transfer_mode,
            kv_transfer_bw_gb_s=self.kv_transfer_bw_gb_s,
            kv_transfer_latency_ms=self.kv_transfer_latency_ms,
            kv_transfer_concurrency=self.kv_transfer_concurrency,
            decode_kv_load_policy=self.decode_kv_load_policy)


def pd_serving_grid(
        gpu_splits: Sequence[Tuple[int, int]],
        prefill_tp_sizes: Sequence[int], decode_tp_sizes: Sequence[int],
        prefill_num_gpu_blocks: int, decode_num_gpu_blocks: int,
        block_size_tokens: int,
        prefill_batched_token_limits: Sequence[int],
        prefill_sequence_limits: Sequence[int],
        decode_batched_token_limits: Sequence[int],
        decode_sequence_limits: Sequence[int],
        chunked_prefill_values: Sequence[bool],
        transfer_modes: Sequence[str], transfer_bandwidths_gb_s: Sequence[float],
        transfer_latency_ms: float,
        transfer_concurrencies: Sequence[int],
        decode_kv_load_policies: Sequence[str] = ("post_transfer",),
        ) -> Tuple[List[PDServingConfig], List[Dict[str, Any]]]:
    """Build valid configurations and return rejected combinations separately."""
    configs: List[PDServingConfig] = []
    rejected: List[Dict[str, Any]] = []
    for values in itertools.product(
            gpu_splits, prefill_tp_sizes, decode_tp_sizes,
            prefill_batched_token_limits, prefill_sequence_limits,
            decode_batched_token_limits, decode_sequence_limits,
            chunked_prefill_values, transfer_modes,
            transfer_bandwidths_gb_s, transfer_concurrencies,
            decode_kv_load_policies):
        (split, prefill_tp, decode_tp, prefill_tokens, prefill_seqs,
         decode_tokens, decode_seqs, chunked, mode, bandwidth,
         concurrency, decode_kv_load_policy) = values
        prefill_gpus, decode_gpus = split
        reason = None
        if min(prefill_gpus, decode_gpus, prefill_tp, decode_tp) <= 0:
            reason = "GPU and TP counts must be positive"
        elif prefill_gpus % prefill_tp:
            reason = "prefill_gpus is not divisible by prefill_tp"
        elif decode_gpus % decode_tp:
            reason = "decode_gpus is not divisible by decode_tp"
        if reason:
            rejected.append({
                "prefill_gpus": prefill_gpus, "prefill_tp": prefill_tp,
                "decode_gpus": decode_gpus, "decode_tp": decode_tp,
                "reason": reason})
            continue
        configs.append(PDServingConfig(
            prefill_gpus, prefill_tp, decode_gpus, decode_tp,
            prefill_num_gpu_blocks, decode_num_gpu_blocks, block_size_tokens,
            prefill_tokens, prefill_seqs, decode_tokens, decode_seqs,
            chunked, mode, bandwidth, transfer_latency_ms, concurrency,
            decode_kv_load_policy))
    unique = {config.config_id: config for config in configs}
    return list(unique.values()), rejected


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


class PDCapacityExperimentRunner:
    def __init__(self, cost_model: Any, workload: WorkloadSpec,
                 policy: CapacityPolicy, repeats: int = 3):
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.cost_model = cost_model
        self.workload = workload
        self.policy = policy
        self.repeats = repeats

    def _run_once(self, config: PDServingConfig, arrival_rate_rps: float,
                  repeat: int) -> Dict[str, Any]:
        generator = RequestGenerator(
            mode=self.workload.mode,
            arrival_interval_ms=1000.0 / arrival_rate_rps,
            prompt_len=self.workload.prompt_len,
            output_len=self.workload.output_len,
            prompt_len_range=self.workload.prompt_len_range,
            output_len_range=self.workload.output_len_range,
            seed=self.workload.seed + repeat)
        simulator = PDSimulator(config.to_pd_config(), self.cost_model, generator)
        metrics = simulator.run(
            simulation_duration_ms=self.workload.simulation_duration_ms,
            ttft_sla_ms=self.policy.ttft_sla_ms,
            tpot_sla_ms=self.policy.tpot_sla_ms,
            num_requests=self.workload.num_requests)
        total = int(metrics["total_requests"])
        completed = int(metrics["completed_requests"])
        both = int(metrics["goodput"]["both_satisfied"])
        duration_ms = max(float(metrics["simulation_time_ms"]), 1e-9)
        prefill_batches = [row for row in simulator.batch_history
                           if row["stage"] == "prefill"]
        decode_batches = [row for row in simulator.batch_history
                          if row["stage"] == "decode"]
        transfers = simulator.transfer_history
        prefill_utilization = sum(
            float(row["estimated_time_ms"]) for row in prefill_batches
        ) / (duration_ms * config.prefill_replicas)
        decode_utilization = sum(
            float(row["estimated_time_ms"]) for row in decode_batches
        ) / (duration_ms * config.decode_replicas)
        transfer_utilization = sum(
            float(row["duration_ms"]) for row in transfers
        ) / (duration_ms * config.kv_transfer_concurrency)
        prefill_kv = float(metrics["pd"]["prefill_peak_kv_utilization"])
        decode_kv = float(metrics["pd"]["decode_peak_kv_utilization"])
        resources = {
            "prefill_compute": prefill_utilization,
            "decode_compute": decode_utilization,
            "kv_transfer": transfer_utilization,
        }
        if max(prefill_kv, decode_kv) >= 0.9:
            bottleneck = "prefill_memory" if prefill_kv >= decode_kv else "decode_memory"
        else:
            bottleneck = max(resources, key=resources.get)
        transfer_queues = [float(row["queued_time_ms"]) for row in transfers]
        prefill_batch_sizes = [float(row["batch_size"]) for row in prefill_batches]
        decode_batch_sizes = [float(row["batch_size"]) for row in decode_batches]
        prefill_replica_utilizations = [
            sum(float(row["estimated_time_ms"]) for row in prefill_batches
                if int(row["replica_id"]) == replica_id) / duration_ms
            for replica_id in range(config.prefill_replicas)]
        decode_replica_utilizations = [
            sum(float(row["estimated_time_ms"]) for row in decode_batches
                if int(row["replica_id"]) == replica_id) / duration_ms
            for replica_id in range(config.decode_replicas)]
        simulation_s = duration_ms / 1000.0
        return {
            "repeat": repeat + 1,
            "seed": self.workload.seed + repeat,
            "total_requests": total,
            "completed_requests": completed,
            "completion_ratio": completed / total if total else 0.0,
            "sla_success_ratio": both / total if total else 0.0,
            "ttft_p50_ms": metrics["ttft_ms"]["p50"],
            "ttft_p99_ms": metrics["ttft_ms"]["p99"],
            "tpot_p50_ms": metrics["tpot_ms"]["p50"],
            "tpot_p99_ms": metrics["tpot_ms"]["p99"],
            "e2e_p99_ms": metrics["e2e_latency_ms"]["p99"],
            "output_throughput_tokens_per_s": metrics[
                "output_throughput_tokens_per_s"],
            "goodput_tokens_per_s": metrics["goodput_tokens_per_s"],
            "completed_requests_per_s": completed / simulation_s,
            "prefill_utilization": prefill_utilization,
            "decode_utilization": decode_utilization,
            "transfer_utilization": transfer_utilization,
            "prefill_peak_kv_utilization": prefill_kv,
            "decode_peak_kv_utilization": decode_kv,
            "transfer_queue_mean_ms": (
                statistics.fmean(transfer_queues) if transfer_queues else 0.0),
            "transfer_queue_p99_ms": _percentile(transfer_queues, 99),
            "prefill_batch_size_mean": (
                statistics.fmean(prefill_batch_sizes)
                if prefill_batch_sizes else 0.0),
            "prefill_batch_size_p95": _percentile(prefill_batch_sizes, 95),
            "prefill_batch_size_max": max(prefill_batch_sizes, default=0.0),
            "decode_batch_size_mean": (
                statistics.fmean(decode_batch_sizes)
                if decode_batch_sizes else 0.0),
            "decode_batch_size_p95": _percentile(decode_batch_sizes, 95),
            "decode_batch_size_max": max(decode_batch_sizes, default=0.0),
            "prefill_replica_utilization_min": min(
                prefill_replica_utilizations, default=0.0),
            "prefill_replica_utilization_max": max(
                prefill_replica_utilizations, default=0.0),
            "decode_replica_utilization_min": min(
                decode_replica_utilizations, default=0.0),
            "decode_replica_utilization_max": max(
                decode_replica_utilizations, default=0.0),
            "prefill_extrapolated_batch_ratio": (
                sum(bool(row["extrapolated"]) for row in prefill_batches) /
                len(prefill_batches) if prefill_batches else 0.0),
            "decode_extrapolated_batch_ratio": (
                sum(bool(row["extrapolated"]) for row in decode_batches) /
                len(decode_batches) if decode_batches else 0.0),
            "estimated_bottleneck": bottleneck,
        }

    def run_point(self, config: PDServingConfig,
                  arrival_rate_rps: float) -> Dict[str, Any]:
        if arrival_rate_rps <= 0:
            raise ValueError("arrival rates must be positive")
        runs = [self._run_once(config, arrival_rate_rps, repeat)
                for repeat in range(self.repeats)]
        completion = _mean(runs, "completion_ratio")
        sla_success = _mean(runs, "sla_success_ratio")
        sustainable = (
            min(row["completion_ratio"] for row in runs) >=
            self.policy.min_completion_ratio and
            sla_success >= self.policy.min_sla_success_ratio)
        resource_keys = ("prefill_utilization", "decode_utilization",
                         "transfer_utilization")
        resource_means = {key: _mean(runs, key) for key in resource_keys}
        p_kv = max(row["prefill_peak_kv_utilization"] for row in runs)
        d_kv = max(row["decode_peak_kv_utilization"] for row in runs)
        if max(p_kv, d_kv) >= 0.9:
            bottleneck = "prefill_memory" if p_kv >= d_kv else "decode_memory"
        else:
            bottleneck = {
                "prefill_utilization": "prefill_compute",
                "decode_utilization": "decode_compute",
                "transfer_utilization": "kv_transfer",
            }[max(resource_means, key=resource_means.get)]
        goodput = _mean(runs, "goodput_tokens_per_s")
        return {
            "config_id": config.config_id,
            **asdict(config),
            "prefill_replicas": config.prefill_replicas,
            "decode_replicas": config.decode_replicas,
            "total_gpus": config.total_gpus,
            "arrival_rate_rps": arrival_rate_rps,
            "sustainable": sustainable,
            "completion_ratio": completion,
            "worst_completion_ratio": min(row["completion_ratio"] for row in runs),
            "sla_success_ratio": sla_success,
            "ttft_p50_ms": _mean(runs, "ttft_p50_ms"),
            "ttft_p99_ms": _mean(runs, "ttft_p99_ms"),
            "tpot_p50_ms": _mean(runs, "tpot_p50_ms"),
            "tpot_p99_ms": _mean(runs, "tpot_p99_ms"),
            "e2e_p99_ms": _mean(runs, "e2e_p99_ms"),
            "output_throughput_tokens_per_s": _mean(
                runs, "output_throughput_tokens_per_s"),
            "goodput_tokens_per_s": goodput,
            "goodput_tokens_per_s_per_gpu": goodput / config.total_gpus,
            **resource_means,
            "prefill_peak_kv_utilization": p_kv,
            "decode_peak_kv_utilization": d_kv,
            "transfer_queue_mean_ms": _mean(runs, "transfer_queue_mean_ms"),
            "transfer_queue_p99_ms": _mean(runs, "transfer_queue_p99_ms"),
            "prefill_batch_size_mean": _mean(runs, "prefill_batch_size_mean"),
            "prefill_batch_size_p95": _mean(runs, "prefill_batch_size_p95"),
            "prefill_batch_size_max": max(
                row["prefill_batch_size_max"] for row in runs),
            "decode_batch_size_mean": _mean(runs, "decode_batch_size_mean"),
            "decode_batch_size_p95": _mean(runs, "decode_batch_size_p95"),
            "decode_batch_size_max": max(
                row["decode_batch_size_max"] for row in runs),
            "prefill_replica_utilization_min": min(
                row["prefill_replica_utilization_min"] for row in runs),
            "prefill_replica_utilization_max": max(
                row["prefill_replica_utilization_max"] for row in runs),
            "decode_replica_utilization_min": min(
                row["decode_replica_utilization_min"] for row in runs),
            "decode_replica_utilization_max": max(
                row["decode_replica_utilization_max"] for row in runs),
            "prefill_extrapolated_batch_ratio": _mean(
                runs, "prefill_extrapolated_batch_ratio"),
            "decode_extrapolated_batch_ratio": _mean(
                runs, "decode_extrapolated_batch_ratio"),
            "estimated_bottleneck": bottleneck,
            "runs": runs,
        }

    def run(self, configs: Iterable[PDServingConfig],
            arrival_rates_rps: Sequence[float],
            refine_iterations: int = 3) -> Dict[str, Any]:
        if refine_iterations < 0:
            raise ValueError("refine_iterations must be non-negative")
        configs = list(configs)
        rates = sorted(set(float(rate) for rate in arrival_rates_rps))
        if not configs or not rates or rates[0] <= 0:
            raise ValueError("valid configurations and positive rates are required")
        points = [self.run_point(config, rate)
                  for config in configs for rate in rates]
        for config in configs:
            for _ in range(refine_iterations):
                rows = sorted((row for row in points
                               if row["config_id"] == config.config_id),
                              key=lambda row: row["arrival_rate_rps"])
                lower = None
                upper = None
                for row in rows:
                    if row["sustainable"] and upper is None:
                        lower = row
                    elif not row["sustainable"]:
                        upper = row
                        break
                if lower is None or upper is None:
                    break
                midpoint = (lower["arrival_rate_rps"] + upper["arrival_rate_rps"]) / 2
                points.append(self.run_point(config, midpoint))
        points.sort(key=lambda row: (row["config_id"], row["arrival_rate_rps"]))
        ranking = []
        for config in configs:
            rows = [row for row in points if row["config_id"] == config.config_id]
            raw_states = [bool(row["sustainable"]) for row in rows]
            non_monotonic = any(
                raw_states[index] and not all(raw_states[:index])
                for index in range(1, len(raw_states)))
            passed = []
            for row in rows:
                if not row["sustainable"]:
                    break
                passed.append(row)
            best = passed[-1] if passed else None
            failure = next((row for row in rows[len(passed):]
                            if not row["sustainable"]), None)
            lower = best["arrival_rate_rps"] if best else None
            upper = failure["arrival_rate_rps"] if failure else None
            ranking.append({
                "rank": 0, "config_id": config.config_id,
                "total_gpus": config.total_gpus,
                "capacity_lower_bound_rps": lower,
                "capacity_upper_bound_rps": upper,
                "capacity_interval_width_rps": (
                    upper - lower if lower is not None and upper is not None else None),
                "capacity_censored": best is not None and upper is None,
                "non_monotonic_points": non_monotonic,
                "capacity_rps_per_gpu": (
                    lower / config.total_gpus if lower is not None else None),
                "goodput_tokens_per_s_at_capacity": (
                    best["goodput_tokens_per_s"] if best else 0.0),
                "goodput_tokens_per_s_per_gpu_at_capacity": (
                    best["goodput_tokens_per_s_per_gpu"] if best else 0.0),
                "sla_success_ratio_at_capacity": (
                    best["sla_success_ratio"] if best else 0.0),
                "ttft_p99_ms_at_capacity": best["ttft_p99_ms"] if best else None,
                "tpot_p99_ms_at_capacity": best["tpot_p99_ms"] if best else None,
                "prefill_utilization_at_capacity": (
                    best["prefill_utilization"] if best else None),
                "decode_utilization_at_capacity": (
                    best["decode_utilization"] if best else None),
                "transfer_utilization_at_capacity": (
                    best["transfer_utilization"] if best else None),
                "estimated_bottleneck_at_capacity": (
                    best["estimated_bottleneck"] if best else None),
                "prefill_batch_size_mean_at_capacity": (
                    best["prefill_batch_size_mean"] if best else None),
                "prefill_batch_size_p95_at_capacity": (
                    best["prefill_batch_size_p95"] if best else None),
                "prefill_batch_size_max_at_capacity": (
                    best["prefill_batch_size_max"] if best else None),
                "decode_batch_size_mean_at_capacity": (
                    best["decode_batch_size_mean"] if best else None),
                "decode_batch_size_p95_at_capacity": (
                    best["decode_batch_size_p95"] if best else None),
                "decode_batch_size_max_at_capacity": (
                    best["decode_batch_size_max"] if best else None),
                "prefill_replica_utilization_min_at_capacity": (
                    best["prefill_replica_utilization_min"] if best else None),
                "prefill_replica_utilization_max_at_capacity": (
                    best["prefill_replica_utilization_max"] if best else None),
                "decode_replica_utilization_min_at_capacity": (
                    best["decode_replica_utilization_min"] if best else None),
                "decode_replica_utilization_max_at_capacity": (
                    best["decode_replica_utilization_max"] if best else None),
            })
        ranking.sort(key=lambda row: (
            row["capacity_lower_bound_rps"] is not None,
            row["capacity_lower_bound_rps"] or 0,
            row["goodput_tokens_per_s_at_capacity"]), reverse=True)
        for index, row in enumerate(ranking, 1):
            row["rank"] = index
        efficiency = sorted((dict(row) for row in ranking), key=lambda row: (
            row["capacity_rps_per_gpu"] is not None,
            row["capacity_rps_per_gpu"] or 0,
            row["goodput_tokens_per_s_per_gpu_at_capacity"]), reverse=True)
        for index, row in enumerate(efficiency, 1):
            row["efficiency_rank"] = index
        return {
            "schema_version": 1,
            "experiment_type": "pd_sla_capacity_search",
            "workload": asdict(self.workload),
            "capacity_policy": asdict(self.policy),
            "repeats": self.repeats,
            "initial_arrival_rates_rps": rates,
            "evaluated_arrival_rates_rps": sorted({
                float(row["arrival_rate_rps"]) for row in points}),
            "refine_iterations": refine_iterations,
            "points": points,
            "capacity_ranking": ranking,
            "efficiency_ranking": efficiency,
        }
