import unittest

from core.communication_model import CommunicationModel
from core.kv_block_pool import KVBlockPool
from core.memory_estimator import MemoryEstimator
from core.resource_checker import ResourceChecker
from core.simulator import Request, RequestGenerator, RequestStatus, Scheduler, Simulator


class FakeCostModel:
    def predict_batch(self, prefill_lengths, decode_lengths, tp_size, **kwargs):
        return {"total_time_ms": 1.0}


class TokenBlockTests(unittest.TestCase):
    def test_token_block_boundaries(self):
        pool = KVBlockPool(total_blocks=8, block_size_tokens=16)
        self.assertEqual(pool.blocks_for_tokens(0), 0)
        self.assertEqual(pool.blocks_for_tokens(1), 1)
        self.assertEqual(pool.blocks_for_tokens(16), 1)
        self.assertEqual(pool.blocks_for_tokens(17), 2)
        self.assertEqual(pool.blocks_for_tokens(32), 2)

    def test_batch_fragmentation_is_per_request(self):
        pool = KVBlockPool(total_blocks=8, block_size_tokens=16)
        checker = ResourceChecker(pool, CommunicationModel())
        needed = checker.calculate_kv_blocks_needed(
            batch_size=2,
            seq_len=17,
            num_layers=64,
            num_kv_heads=8,
            head_dim=128,
            tp_size=4,
        )
        self.assertEqual(needed, 4)

    def test_qwen3_32b_tp4_block_memory(self):
        estimator = MemoryEstimator(
            num_layers=64,
            hidden_size=5120,
            num_kv_heads=8,
            head_dim=128,
        )
        self.assertEqual(estimator.kv_bytes_per_token_per_rank(4), 65536)
        self.assertAlmostEqual(estimator.kv_block_memory_mib(16, 4), 1.0)
        capacity = estimator.kv_capacity_from_blocks(100, 16, 4)
        self.assertEqual(capacity["token_capacity"], 1600)
        self.assertAlmostEqual(capacity["total_kv_memory_mib_per_rank"], 100.0)

    def test_scheduler_grows_at_token_boundary(self):
        pool = KVBlockPool(total_blocks=3, block_size_tokens=16)
        scheduler = Scheduler(100, 2, pool, None, None)
        decoding = Request(
            "decode", 16, 2, 0,
            status=RequestStatus.DECODING,
            generated_tokens=1,
        )
        waiting = Request("prefill", 17, 1, 1)
        batch = scheduler.schedule([waiting, decoding], 2, 4)
        self.assertEqual([request.request_id for request in batch.requests], ["decode"])
        self.assertEqual(pool.get_used_blocks(), 2)

    def test_closed_loop_releases_token_blocks(self):
        pool = KVBlockPool(total_blocks=16, block_size_tokens=16)
        checker = ResourceChecker(pool, CommunicationModel())
        model = FakeCostModel()
        scheduler = Scheduler(64, 4, pool, model, checker)
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=1, prompt_len=17, output_len=3
        )
        result = Simulator(model, pool, checker, scheduler, generator).run(
            simulation_duration_ms=100, num_requests=4
        )
        self.assertEqual(result["completed_requests"], 4)
        self.assertEqual(result["goodput"]["token_goodput"], 12)
        self.assertEqual(pool.get_used_blocks(), 0)

    def test_long_prefill_completes_in_multiple_chunks(self):
        pool = KVBlockPool(total_blocks=16, block_size_tokens=16)
        checker = ResourceChecker(pool, CommunicationModel())
        model = FakeCostModel()
        scheduler = Scheduler(
            8, 4, pool, model, checker, enable_chunked_prefill=True)
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=1,
            prompt_len=20, output_len=1)
        simulator = Simulator(model, pool, checker, scheduler, generator)
        result = simulator.run(simulation_duration_ms=100, num_requests=1)
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(
            [row["prefill_tokens"] for row in simulator.batch_history],
            [8, 8, 4])
        self.assertEqual(simulator.completed_requests[0].first_token_time, 3.0)
        self.assertEqual(pool.get_used_blocks(), 0)


if __name__ == "__main__":
    unittest.main()
