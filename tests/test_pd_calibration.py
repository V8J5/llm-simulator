import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.collect_pd_calibration_runs import (
    discover_traces,
    validate_output,
)
from scripts.fit_pd_calibration import candidate_payload


class PDCalibrationTests(unittest.TestCase):
    def test_discovery_rejects_holdout_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "holdout_pd.csv"
            trace.write_text(
                "arrival_time_ms,prompt_len,output_len\n0,16,2\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "holdout"):
                discover_traces(Path(directory), [trace])

    def test_collection_validation_checks_deployment_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({
                "collector": "vllm_pd_proxy_streaming",
                "deployment_config_sha256": "hash-a",
                "requests": [{"request_id": "req_1", "status": "success"}],
            }), encoding="utf-8")
            self.assertEqual(validate_output(path, 1), "hash-a")
            with self.assertRaisesRegex(RuntimeError, "deployment changed"):
                validate_output(path, 1, "hash-b")

    def test_candidate_has_independent_stage_factors(self):
        args = Namespace(
            reference_batch_size=8.0, reference_kv_len=1024.0,
            kv_transfer_latency_ms=None, kv_transfer_concurrency=1)
        payload = candidate_payload(4, args, {
            "prefill_factor": 1.2,
            "decode_base_factor": 0.9,
            "batch_exponent": 0.1,
            "kv_exponent": -0.1,
            "iteration_overhead_ms": 2.0,
            "transfer_ms_per_128mb": 64.0,
        })
        self.assertEqual(payload["prefill"]["4"], 1.2)
        self.assertEqual(payload["decode"]["4"]["base_factor"], 0.9)
        self.assertEqual(payload["iteration_overhead_ms"]["4"], 2.0)
        self.assertEqual(payload["pd_transfer"]["effective_bandwidth_gb_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
