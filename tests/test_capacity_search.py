import unittest

from core.experiment_runner import (
    CapacityExperimentRunner,
    CapacityPolicy,
    ServingConfig,
    WorkloadSpec,
    serving_grid,
)


class FakeCostModel:
    def predict_batch(self, prefill_lengths, decode_lengths, tp_size, **kwargs):
        return {
            "total_time_ms": 1.0,
            "breakdown": {"compute_ms": 0.75, "comm_ms": 0.25,
                          "iteration_overhead_ms": 0.0},
            "parts": [], "extrapolated": False, "warnings": [],
        }


class CapacitySearchTests(unittest.TestCase):
    def test_serving_grid_is_cartesian(self):
        configs = serving_grid([2, 4], [100], [16], [1024, 2048], [8], [True])
        self.assertEqual(len(configs), 4)
        self.assertEqual(len({item.config_id for item in configs}), 4)

    def test_data_parallel_grid_reports_gpu_efficiency(self):
        configs = serving_grid(
            [2], [100], [16], [32], [4], [True], [1, 2])
        self.assertEqual([item.total_gpus for item in configs], [2, 4])
        runner = CapacityExperimentRunner(
            FakeCostModel(),
            WorkloadSpec(mode="fixed", prompt_len=16, output_len=2,
                         num_requests=8, simulation_duration_ms=10000),
            CapacityPolicy(ttft_sla_ms=100, tpot_sla_ms=100,
                           min_completion_ratio=1,
                           min_sla_success_ratio=1),
            repeats=1)
        report = runner.run(configs, [1])
        self.assertEqual(len(report["efficiency_ranking"]), 2)
        self.assertEqual({point["data_parallel"] for point in report["points"]},
                         {1, 2})
        self.assertTrue(all("goodput_tokens_per_s_per_gpu" in point
                            for point in report["points"]))

    def test_capacity_report_uses_all_offered_requests_for_sla(self):
        runner = CapacityExperimentRunner(
            FakeCostModel(),
            WorkloadSpec(mode="fixed", prompt_len=16, output_len=2,
                         num_requests=5, simulation_duration_ms=10000, seed=7),
            CapacityPolicy(ttft_sla_ms=100, tpot_sla_ms=100,
                           min_completion_ratio=1.0,
                           min_sla_success_ratio=1.0),
            repeats=2,
        )
        config = ServingConfig(1, 100, 16, 32, 4, True)
        report = runner.run([config], [1.0, 2.0])
        self.assertEqual(len(report["points"]), 2)
        self.assertTrue(all(point["sustainable"] for point in report["points"]))
        self.assertEqual(
            report["capacity_ranking"][0]["max_sustainable_arrival_rate_rps"],
            2.0)
        self.assertTrue(report["capacity_ranking"][0]["capacity_censored"])
        self.assertEqual(
            report["capacity_ranking"][0]["capacity_lower_bound_rps"], 2.0)
        self.assertIsNone(
            report["capacity_ranking"][0]["capacity_upper_bound_rps"])
        self.assertEqual(report["points"][0]["sla_success_ratio"], 1.0)
        self.assertIn("max_kv_block_utilization", report["points"][0])
        self.assertEqual(report["points"][0]["estimated_bottleneck"], "compute")

    def test_refinement_reports_a_bounded_capacity_interval(self):
        runner = CapacityExperimentRunner(
            FakeCostModel(),
            WorkloadSpec(mode="fixed", prompt_len=1, output_len=2,
                         num_requests=20, simulation_duration_ms=1000),
            CapacityPolicy(ttft_sla_ms=1, tpot_sla_ms=1,
                           min_completion_ratio=1,
                           min_sla_success_ratio=1),
            repeats=1,
        )
        config = ServingConfig(1, 100, 16, 8, 1, True)
        report = runner.run([config], [100, 1000], refine_iterations=2)
        capacity = report["capacity_ranking"][0]
        self.assertFalse(capacity["capacity_censored"])
        self.assertLess(capacity["capacity_lower_bound_rps"],
                        capacity["capacity_upper_bound_rps"])
        self.assertEqual(len(report["points"]), 4)


if __name__ == "__main__":
    unittest.main()
