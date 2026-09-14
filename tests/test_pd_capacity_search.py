import unittest

from core.experiment_runner import CapacityPolicy, WorkloadSpec
from core.pd_experiment_runner import (
    PDCapacityExperimentRunner,
    PDServingConfig,
    pd_serving_grid,
)


class FakeMemoryEstimator:
    def kv_bytes_per_token_per_rank(self, tp_size):
        return 64.0


class FakeCostModel:
    memory_estimator = FakeMemoryEstimator()

    def predict_batch(self, prefill_lengths, decode_lengths, tp_size, **kwargs):
        return {
            "total_time_ms": 1.0,
            "breakdown": {"compute_ms": 0.8, "comm_ms": 0.2,
                          "iteration_overhead_ms": 0.0},
            "parts": [], "extrapolated": False, "warnings": [],
        }


def make_config(**overrides):
    values = {
        "prefill_gpus": 1, "prefill_tp": 1,
        "decode_gpus": 1, "decode_tp": 1,
        "prefill_num_gpu_blocks": 128, "decode_num_gpu_blocks": 128,
        "block_size_tokens": 16,
        "prefill_max_num_batched_tokens": 32,
        "prefill_max_num_seqs": 4,
        "decode_max_num_batched_tokens": 32,
        "decode_max_num_seqs": 4,
        "chunked_prefill": True, "kv_transfer_mode": "test",
        "kv_transfer_bw_gb_s": 100.0, "kv_transfer_latency_ms": 0.0,
        "kv_transfer_concurrency": 1,
    }
    values.update(overrides)
    return PDServingConfig(**values)


class PDCapacitySearchTests(unittest.TestCase):
    def test_grid_rejects_fractional_replicas(self):
        configs, rejected = pd_serving_grid(
            gpu_splits=[(4, 4), (3, 4)],
            prefill_tp_sizes=[2], decode_tp_sizes=[4],
            prefill_num_gpu_blocks=100, decode_num_gpu_blocks=100,
            block_size_tokens=16, prefill_batched_token_limits=[32],
            prefill_sequence_limits=[4], decode_batched_token_limits=[32],
            decode_sequence_limits=[4], chunked_prefill_values=[True],
            transfer_modes=["pcie"], transfer_bandwidths_gb_s=[12.5],
            transfer_latency_ms=0.1, transfer_concurrencies=[1])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].prefill_replicas, 2)
        self.assertEqual(len(rejected), 1)
        self.assertIn("not divisible", rejected[0]["reason"])

    def test_decode_kv_load_policy_is_part_of_config_identity(self):
        common = dict(
            gpu_splits=[(4, 4)], prefill_tp_sizes=[4], decode_tp_sizes=[4],
            prefill_num_gpu_blocks=100, decode_num_gpu_blocks=100,
            block_size_tokens=16, prefill_batched_token_limits=[32],
            prefill_sequence_limits=[4], decode_batched_token_limits=[32],
            decode_sequence_limits=[4], chunked_prefill_values=[True],
            transfer_modes=["nccl"], transfer_bandwidths_gb_s=[12.5],
            transfer_latency_ms=0.1, transfer_concurrencies=[1])
        configs, rejected = pd_serving_grid(
            **common,
            decode_kv_load_policies=["post_transfer", "blocking_receive"])
        self.assertFalse(rejected)
        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.to_pd_config().decode_kv_load_policy for config in configs},
            {"post_transfer", "blocking_receive"})
        self.assertEqual(len({config.config_id for config in configs}), 2)

    def test_report_ranks_capacity_and_efficiency(self):
        runner = PDCapacityExperimentRunner(
            FakeCostModel(),
            WorkloadSpec(mode="fixed", prompt_len=16, output_len=2,
                         num_requests=5, simulation_duration_ms=10000),
            CapacityPolicy(ttft_sla_ms=100, tpot_sla_ms=100,
                           min_completion_ratio=1,
                           min_sla_success_ratio=1),
            repeats=2)
        report = runner.run([make_config()], [1, 2], refine_iterations=1)
        self.assertEqual(len(report["points"]), 2)
        self.assertEqual(report["evaluated_arrival_rates_rps"], [1.0, 2.0])
        self.assertTrue(all(row["sustainable"] for row in report["points"]))
        capacity = report["capacity_ranking"][0]
        self.assertEqual(capacity["capacity_lower_bound_rps"], 2.0)
        self.assertTrue(capacity["capacity_censored"])
        self.assertFalse(capacity["non_monotonic_points"])
        self.assertIn(capacity["estimated_bottleneck_at_capacity"], {
            "prefill_compute", "decode_compute", "kv_transfer"})
        self.assertGreater(capacity["decode_batch_size_mean_at_capacity"], 0)
        self.assertLessEqual(
            capacity["decode_replica_utilization_min_at_capacity"],
            capacity["decode_replica_utilization_max_at_capacity"])
        self.assertEqual(report["efficiency_ranking"][0]["efficiency_rank"], 1)

    def test_slow_link_is_reported_as_transfer_bottleneck(self):
        runner = PDCapacityExperimentRunner(
            FakeCostModel(),
            WorkloadSpec(mode="fixed", prompt_len=16, output_len=1,
                         num_requests=2, simulation_duration_ms=10000),
            CapacityPolicy(ttft_sla_ms=10000, tpot_sla_ms=10000,
                           min_completion_ratio=1,
                           min_sla_success_ratio=1), repeats=1)
        point = runner.run_point(
            make_config(kv_transfer_bw_gb_s=0.00001,
                        kv_transfer_latency_ms=1.0), 100)
        self.assertEqual(point["estimated_bottleneck"], "kv_transfer")
        self.assertGreater(point["transfer_utilization"],
                           point["prefill_utilization"])


if __name__ == "__main__":
    unittest.main()
