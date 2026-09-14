import argparse
import csv
import unittest
from pathlib import Path

from scripts.collect_calibration_runs import build_command, discover_traces
from scripts.fit_calibration import grid_search


ROOT = Path(__file__).resolve().parents[1]


class CalibrationPipelineTests(unittest.TestCase):
    def test_calibration_traces_are_varied_and_not_validation(self):
        trace_dir = ROOT / "data" / "trace" / "calibration"
        paths = discover_traces(trace_dir, None)
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertNotEqual(path.name, "validation_trace.csv")
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            arrivals = [float(row["arrival_time_ms"]) for row in rows]
            self.assertEqual(arrivals, sorted(arrivals))
            self.assertGreater(len({int(row["prompt_len"]) for row in rows}), 1)
            self.assertGreater(len({int(row["output_len"]) for row in rows}), 1)

    def test_collection_command_has_distinct_trace_and_output(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8000/v1", model="model",
            tokenizer="tokenizer", api_key="EMPTY", max_concurrency=32,
            timeout=600.0)
        trace = Path("calibration.csv").resolve()
        output = Path("repeat_1.json").resolve()
        command = build_command(args, trace, output)
        self.assertIn(str(trace), command)
        self.assertIn(str(output), command)
        self.assertIn("collect_vllm_ground_truth.py", " ".join(command))

    def test_grid_search_finds_synthetic_optimum(self):
        best, value, curve = grid_search(
            lambda factor: (factor - 0.73) ** 2,
            0.4, 1.2, points=9, rounds=4)
        self.assertAlmostEqual(best, 0.73, places=2)
        self.assertLess(value, 0.0001)
        self.assertGreater(len(curve), 9)


if __name__ == "__main__":
    unittest.main()
