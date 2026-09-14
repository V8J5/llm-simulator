import unittest

from core.pd_simulator import KVTransferModel, PDConfig, PDSimulator
from core.simulator import RequestGenerator


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


class PDSimulatorTests(unittest.TestCase):
    def test_config_rejects_fractional_replica(self):
        with self.assertRaises(ValueError):
            PDConfig(num_prefill_gpus=3, prefill_tp=2)

    def test_transfer_model_uses_configured_link(self):
        model = KVTransferModel(bandwidth_gb_s=10, latency_ms=2)
        self.assertAlmostEqual(model.estimate_transfer_time(100), 12.0)

    def test_pd_pipeline_includes_transfer_and_releases_kv(self):
        config = PDConfig(
            num_prefill_gpus=1, prefill_tp=1, prefill_num_gpu_blocks=32,
            num_decode_gpus=1, decode_tp=1, decode_num_gpu_blocks=32,
            prefill_max_num_batched_tokens=16,
            decode_max_num_batched_tokens=16,
            kv_transfer_bw_gb_s=1, kv_transfer_latency_ms=5)
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=100,
            prompt_len=16, output_len=2)
        simulator = PDSimulator(config, FakeCostModel(), generator)
        result = simulator.run(simulation_duration_ms=100, num_requests=1,
                               ttft_sla_ms=100, tpot_sla_ms=100)
        self.assertEqual(result["completed_requests"], 1)
        self.assertGreaterEqual(result["ttft_ms"]["avg"], 7.0)
        self.assertEqual(result["pd"]["transfer_count"], 1)
        self.assertEqual(simulator.prefill_kv_pool.get_used_blocks(), 0)
        self.assertEqual(simulator.decode_kv_pool.get_used_blocks(), 0)

    def test_prefill_replicas_execute_concurrently(self):
        config = PDConfig(
            num_prefill_gpus=2, prefill_tp=1, prefill_num_gpu_blocks=32,
            num_decode_gpus=2, decode_tp=1, decode_num_gpu_blocks=32,
            prefill_max_num_batched_tokens=16,
            decode_max_num_batched_tokens=16,
            kv_transfer_bw_gb_s=100, kv_transfer_latency_ms=0,
            kv_transfer_concurrency=2)
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=0.001,
            prompt_len=16, output_len=1)
        simulator = PDSimulator(config, FakeCostModel(), generator)
        result = simulator.run(simulation_duration_ms=100, num_requests=2,
                               ttft_sla_ms=100, tpot_sla_ms=100)
        prefill = [row for row in simulator.batch_history
                   if row["stage"] == "prefill"]
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual({row["replica_id"] for row in prefill}, {0, 1})
        self.assertLessEqual(max(row["start_time_ms"] for row in prefill), 0.001)

    def test_blocking_receive_stalls_decode_until_new_request_kv_is_ready(self):
        config = PDConfig(
            num_prefill_gpus=1, prefill_tp=1, prefill_num_gpu_blocks=32,
            num_decode_gpus=1, decode_tp=1, decode_num_gpu_blocks=32,
            prefill_max_num_batched_tokens=16,
            decode_max_num_batched_tokens=16,
            decode_max_num_seqs=2,
            kv_transfer_bw_gb_s=1, kv_transfer_latency_ms=5,
            decode_kv_load_policy="blocking_receive")
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=2,
            prompt_len=16, output_len=3)
        simulator = PDSimulator(config, FakeCostModel(), generator)
        result = simulator.run(
            simulation_duration_ms=100, num_requests=2,
            ttft_sla_ms=100, tpot_sla_ms=100)
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(result["pd"]["decode_blocking_receive_count"], 2)
        blocked = [row["blocked_time_ms"]
                   for row in simulator.decode_receive_history]
        self.assertGreater(min(blocked), 0.0)
        self.assertGreaterEqual(max(blocked), 6.0)
        # Each ready receive cohort must execute a decode iteration before the
        # next cohort can block the worker.  In particular, the first request
        # must not be held until the second request's KV is ready.
        self.assertLess(
            simulator.requests[0].first_token_time,
            simulator.requests[1].first_token_time)


if __name__ == "__main__":
    unittest.main()
