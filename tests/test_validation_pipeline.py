import json
import tempfile
import unittest
from pathlib import Path

from scripts.validation_utils import aggregate_vllm_runs, compare_requests


def make_run(path: Path, delta: float = 0.0, request_ids=("req_1", "req_2")):
    requests = []
    for index, request_id in enumerate(request_ids):
        requests.append({
            "request_id": request_id,
            "arrival_time_ms": index * 100.0,
            "prompt_len": 16,
            "output_len": 2,
            "actual_prompt_tokens": 16,
            "actual_output_tokens": 2,
            "request_start_ms": index * 100.0 + 1.0,
            "ttft_ms": 10.0 + index + delta,
            "tpot_ms": 2.0 + delta,
            "e2e_ms": 12.0 + index + delta,
            "status": "success",
            "error": None,
        })
    path.write_text(json.dumps({
        "schema_version": 1,
        "collector": "vllm_openai_streaming",
        "model": "model",
        "trace": "/tmp/trace.csv",
        "requests": requests,
    }), encoding="utf-8")


class ValidationPipelineTests(unittest.TestCase):
    def test_aggregate_aligns_requests_and_reports_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "run1.json"
            second = Path(directory) / "run2.json"
            make_run(first, 0.0)
            make_run(second, 2.0)
            result = aggregate_vllm_runs([first, second])
        self.assertEqual(result["repeat_count"], 2)
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(result["requests"][0]["ttft_ms_mean"], 11.0)
        self.assertGreater(result["summary"]["repeatability"]["ttft_ms"]["cv_pct"], 0)

    def test_aggregate_rejects_misaligned_request_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "run1.json"
            second = Path(directory) / "run2.json"
            make_run(first)
            make_run(second, request_ids=("req_1", "req_3"))
            with self.assertRaisesRegex(ValueError, "request IDs differ"):
                aggregate_vllm_runs([first, second])

    def test_aggregate_accepts_matching_pd_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "run1.json"
            second = Path(directory) / "run2.json"
            make_run(first)
            make_run(second, 1.0)
            for path in (first, second):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.update({
                    "schema_version": 2,
                    "collector": "vllm_pd_proxy_streaming",
                    "deployment_config_sha256": "same-deployment",
                    "deployment": {"prefill": {}, "decode": {}},
                })
                path.write_text(json.dumps(payload), encoding="utf-8")
            result = aggregate_vllm_runs([first, second])
        self.assertEqual(result["collector"],
                         "vllm_pd_proxy_streaming_aggregate")
        self.assertEqual(result["deployment_config_sha256"], "same-deployment")

    def test_compare_reports_signed_and_absolute_error(self):
        aggregate = {
            "requests": [{
                "request_id": "req_1", "arrival_time_ms": 0,
                "prompt_len": 16, "output_len": 2,
                "ttft_ms_mean": 10.0, "tpot_ms_mean": 2.0,
                "e2e_ms_mean": 12.0,
            }]
        }
        detail, summary = compare_requests(aggregate, [{
            "request_id": "req_1", "ttft_ms": 12.0,
            "tpot_ms": 1.0, "e2e_ms": 12.0,
        }])
        self.assertEqual(detail[0]["ttft_ms_error_pct"], 20.0)
        self.assertEqual(detail[0]["tpot_ms_error_pct"], -50.0)
        self.assertEqual(summary["e2e_ms"]["mape_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
