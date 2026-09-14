import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_pd_ground_truth import (
    load_deployment,
    parse_metrics_endpoints,
    parse_prometheus,
    summarize,
)


class PDGroundTruthCollectorTests(unittest.TestCase):
    def test_deployment_manifest_requires_auditable_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            path.write_text(json.dumps({"vllm_version": "0.19.0"}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_deployment(path)
            value = {
                "vllm_version": "0.19.0", "prefill": {}, "decode": {},
                "router": {}, "kv_connector": {},
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded, digest = load_deployment(path)
            self.assertEqual(loaded, value)
            self.assertEqual(len(digest), 64)

    def test_metrics_parser_preserves_labeled_vllm_series(self):
        parsed = parse_prometheus(
            '# HELP ignored ignored\n'
            'vllm:num_requests_running{model_name="qwen"} 3\n'
            'process_cpu_seconds_total 12\n'
            'vllm:gpu_cache_usage_perc 0.25\n',
            ["vllm:"])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed['vllm:num_requests_running{model_name="qwen"}'], 3)
        self.assertEqual(parsed["vllm:gpu_cache_usage_perc"], 0.25)

    def test_component_endpoint_names_must_be_unique(self):
        self.assertEqual(
            parse_metrics_endpoints([
                "prefill=http://127.0.0.1:8100/metrics",
                "decode=http://127.0.0.1:8200/metrics",
            ]),
            {"prefill": "http://127.0.0.1:8100/metrics",
             "decode": "http://127.0.0.1:8200/metrics"})
        with self.assertRaises(ValueError):
            parse_metrics_endpoints(["prefill=a", "prefill=b"])

    def test_summary_does_not_hide_failed_requests(self):
        rows = [
            {"status": "success", "ttft_ms": 10, "tpot_ms": 2,
             "e2e_ms": 20, "client_backpressure_ms": 0},
            {"status": "failed", "ttft_ms": None, "tpot_ms": None,
             "e2e_ms": 1, "client_backpressure_ms": 0},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["success_ratio"], 0.5)
        self.assertEqual(summary["ttft_ms"]["mean"], 10)


if __name__ == "__main__":
    unittest.main()
