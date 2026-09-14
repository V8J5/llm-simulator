import csv
import tempfile
import unittest
from pathlib import Path

from core.communication_model import CommunicationModel
from core.kv_block_pool import KVBlockPool
from core.qwen3_cost_model import LLMCostModel
from core.resource_checker import ResourceChecker
from core.simulator import Request, RequestGenerator, RequestStatus, Scheduler, Simulator


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cost = LLMCostModel(str(ROOT / "data"))

    def test_trace_arrivals_are_absolute(self):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", delete=False) as handle:
            writer = csv.writer(handle)
            writer.writerow(["arrival_time_ms", "prompt_len", "output_len"])
            writer.writerows([[0, 16, 2], [10, 16, 2], [25, 16, 2]])
            name = handle.name
        try:
            generated = RequestGenerator(mode="trace", trace_file=name).generate_requests()
            self.assertEqual([item.arrival_time for item in generated], [0, 10, 25])
        finally:
            Path(name).unlink(missing_ok=True)

    def test_scheduler_counts_query_tokens_and_prioritizes_decode(self):
        pool = KVBlockPool(1024, 16)
        checker = ResourceChecker(pool, CommunicationModel())
        scheduler = Scheduler(17, 2, pool, self.cost, checker)
        waiting = Request("new", 17, 2, 0)
        decoding = Request("running", 16, 2, 1, status=RequestStatus.DECODING,
                           generated_tokens=1)
        batch = scheduler.schedule([waiting, decoding], 2, 1)
        self.assertEqual([item.request_id for item in batch.requests], ["running"])
        self.assertEqual(batch.total_tokens, 1)
        # Only decode fits once the 17-token prefill would exceed the remaining 16.
        self.assertEqual(batch.stage, "decode")

    def test_chunked_prefill_fills_budget_after_decode(self):
        pool = KVBlockPool(1024, 16)
        checker = ResourceChecker(pool, CommunicationModel())
        scheduler = Scheduler(
            17, 2, pool, self.cost, checker,
            enable_chunked_prefill=True)
        waiting = Request("new", 17, 2, 0)
        decoding = Request(
            "running", 16, 2, 1, status=RequestStatus.DECODING,
            generated_tokens=1)
        batch = scheduler.schedule([waiting, decoding], 2, 1)
        self.assertEqual(
            [item.request_id for item in batch.requests], ["running", "new"])
        self.assertEqual(batch.scheduled_token_counts["new"], 16)
        self.assertEqual(batch.total_tokens, 17)
        self.assertEqual(batch.stage, "mixed")

    def test_kv_pool_grows_without_double_allocation(self):
        pool = KVBlockPool(64, 16)
        self.assertTrue(pool.ensure_capacity("r", 2))
        self.assertTrue(pool.ensure_capacity("r", 3))
        self.assertEqual(pool.get_used_blocks(), 3)
        self.assertTrue(pool.ensure_capacity("r", 3))
        self.assertEqual(pool.get_used_blocks(), 3)

    def test_cost_model_reports_extrapolation(self):
        inside = self.cost.predict_prefill(4, 1024, 2)
        outside = self.cost.predict_prefill(64, 8192, 2)
        self.assertFalse(inside["extrapolated"])
        self.assertTrue(outside["extrapolated"])
        self.assertGreater(outside["total_time_ms"], inside["total_time_ms"])

    def test_closed_loop_metrics(self):
        pool = KVBlockPool(2048, 16)
        checker = ResourceChecker(pool, CommunicationModel())
        scheduler = Scheduler(512, 4, pool, self.cost, checker)
        generator = RequestGenerator(mode="fixed", arrival_interval_ms=5,
                                     prompt_len=32, output_len=3)
        result = Simulator(self.cost, pool, checker, scheduler, generator).run(
            simulation_duration_ms=20000, num_requests=4)
        self.assertEqual(result["completed_requests"], 4)
        self.assertGreater(result["output_throughput_tokens_per_s"], 0)
        self.assertEqual(result["goodput"]["token_goodput"], 12)


if __name__ == "__main__":
    unittest.main()
