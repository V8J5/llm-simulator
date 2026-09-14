import json
import tempfile
import unittest
from pathlib import Path

from core.communication_model import CommunicationModel
from core.kv_block_pool import KVBlockPool
from core.qwen3_cost_model import LLMCostModel
from core.resource_checker import ResourceChecker
from core.simulator import RequestGenerator, Scheduler, Simulator
from scripts.fit_shape_calibration import coordinate_search


ROOT = Path(__file__).resolve().parents[1]


class FakeCostModel:
    def predict_batch(self, prefill_lengths, decode_lengths, tp_size, **kwargs):
        return {"total_time_ms": 1.0}


class ShapeCalibrationTests(unittest.TestCase):
    def _model(self, payload):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = LLMCostModel(
            str(ROOT / "data"), calibration_file=str(path),
            enable_e2e_compensation=True)
        self.addCleanup(directory.cleanup)
        return model

    def test_legacy_scalar_calibration_remains_supported(self):
        model = self._model({"decode": {"4": 0.6}})
        result = model.predict_decode(8, 1024, 4)
        self.assertAlmostEqual(result["calibration_factor"], 0.6)

    def test_shape_factor_uses_batch_and_kv_length(self):
        model = self._model({"decode": {"4": {
            "base_factor": 0.6,
            "reference_batch_size": 8,
            "reference_kv_len": 1024,
            "batch_exponent": 0.5,
            "kv_exponent": 0.25,
            "min_factor": 0.2,
            "max_factor": 2.0,
        }}})
        reference = model.predict_decode(8, 1024, 4)
        larger = model.predict_decode(32, 4096, 4)
        self.assertAlmostEqual(reference["calibration_factor"], 0.6)
        self.assertGreater(larger["calibration_factor"],
                           reference["calibration_factor"])

    def test_shape_factor_clamps_outside_measured_calibration_support(self):
        model = self._model({"decode": {"4": {
            "base_factor": 1.0,
            "reference_batch_size": 4,
            "reference_kv_len": 512,
            "batch_exponent": 0.5,
            "kv_exponent": 0.25,
            "batch_size_min": 2,
            "batch_size_max": 6,
            "kv_len_min": 80,
            "kv_len_max": 784,
            "min_factor": 0.2,
            "max_factor": 2.0,
        }}})
        boundary = model.predict_decode(6, 784, 4)
        outside = model.predict_decode(20, 4096, 4)
        self.assertAlmostEqual(
            outside["calibration_factor"], boundary["calibration_factor"])
        self.assertTrue(outside["calibration_shape_clamped"])
        self.assertIn("clamped", " ".join(outside["warnings"]))

    def test_iteration_overhead_is_added_once_per_batch(self):
        model = self._model({
            "decode": {"4": 1.0},
            "iteration_overhead_ms": {"4": 3.5},
        })
        result = model.predict_batch([], [512, 512], 4)
        parts_total = sum(part["total_time_ms"] for part in result["parts"])
        self.assertAlmostEqual(result["total_time_ms"] - parts_total, 3.5)

    def test_chunked_prefill_accounts_for_existing_context(self):
        model = self._model({"prefill": {"4": 1.0}})
        no_context = model.predict_batch(
            [128], [], 4, prefill_context_lengths=[128])
        with_context = model.predict_batch(
            [128], [], 4, prefill_context_lengths=[512])
        self.assertGreater(with_context["total_time_ms"],
                           no_context["total_time_ms"])

    def test_simulator_records_batch_shapes(self):
        pool = KVBlockPool(64, 16)
        checker = ResourceChecker(pool, CommunicationModel())
        model = FakeCostModel()
        scheduler = Scheduler(64, 4, pool, model, checker)
        generator = RequestGenerator(
            mode="fixed", arrival_interval_ms=1, prompt_len=16, output_len=3)
        simulator = Simulator(model, pool, checker, scheduler, generator)
        simulator.run(simulation_duration_ms=100, num_requests=2)
        self.assertTrue(simulator.batch_history)
        self.assertIn("decode_kv_len_mean", simulator.batch_history[0])
        self.assertIn("estimated_time_ms", simulator.batch_history[0])

    def test_coordinate_search_finds_multivariate_minimum(self):
        ranges = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}
        best, score, _ = coordinate_search(
            lambda item: (item["x"] - 0.7) ** 2 + (item["y"] + 0.4) ** 2,
            starts=[{"x": 0.0, "y": 0.0}], ranges=ranges,
            points=9, rounds=4)
        self.assertAlmostEqual(best["x"], 0.7, places=1)
        self.assertAlmostEqual(best["y"], -0.4, places=1)
        self.assertLess(score, 0.01)


if __name__ == "__main__":
    unittest.main()
