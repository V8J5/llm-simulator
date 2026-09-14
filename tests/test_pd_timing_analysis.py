import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_pd_timing import (
    analyze_run,
    load_kv_event_log,
    load_timing_log,
)


class PDTimingAnalysisTests(unittest.TestCase):
    def _files(self, directory: Path):
        run = {
            "deployment_config_sha256": "abc",
            "requests": [
                {"request_id": "req_1", "status": "success",
                 "arrival_time_ms": 0, "prompt_len": 16, "output_len": 2,
                 "ttft_ms": 19, "tpot_ms": 4, "e2e_ms": 23},
                {"request_id": "req_2", "status": "success",
                 "arrival_time_ms": 10, "prompt_len": 16, "output_len": 2,
                 "ttft_ms": 13, "tpot_ms": 4, "e2e_ms": 17},
            ],
        }
        run_path = directory / "run.json"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        fields = (
            "prefill_start_ms=1 prefill_complete_ms={prefill} "
            "decode_start_ms=1 decode_headers_ms=2 "
            "decode_first_byte_ms={first} decode_complete_ms=22 "
            "proxy_complete_ms=23")
        log_path = directory / "timing.log"
        log_path.write_text(
            "INFO PD_TIMING client_request_id=req_1 internal_request_id=a " +
            fields.format(prefill=5, first=15) + "\n" +
            "INFO PD_TIMING client_request_id=req_2 internal_request_id=b " +
            fields.format(prefill=7, first=9) + "\n", encoding="utf-8")
        return run_path, log_path

    def test_aligns_and_groups_absolute_first_token_times(self):
        with tempfile.TemporaryDirectory() as name:
            run, log = self._files(Path(name))
            summary, rows, cohorts = analyze_run(
                "test", run, log, cohort_threshold_ms=5)
        self.assertEqual(summary["aligned_timing_records"], 2)
        self.assertEqual(summary["readiness_cohort_count"], 1)
        self.assertEqual(summary["largest_readiness_cohort"], 2)
        self.assertAlmostEqual(rows[0]["post_prefill_to_first_byte_ms"], 10)
        self.assertEqual(cohorts[0]["request_ids"], "req_1,req_2")

    def test_rejects_duplicate_records(self):
        with tempfile.TemporaryDirectory() as name:
            _, log = self._files(Path(name))
            text = log.read_text(encoding="utf-8")
            log.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_timing_log(log)

    def test_rejects_misaligned_request_ids(self):
        with tempfile.TemporaryDirectory() as name:
            run, log = self._files(Path(name))
            log.write_text(
                log.read_text(encoding="utf-8").replace("req_2", "req_3"),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alignment failed"):
                analyze_run("test", run, log)

    def test_splits_blocking_connector_receive_from_surrounding_wait(self):
        with tempfile.TemporaryDirectory() as name:
            run, log = self._files(Path(name))
            text = log.read_text(encoding="utf-8")
            text = text.replace(
                "internal_request_id=a ",
                "internal_request_id=a proxy_received_monotonic_s=100.0 ")
            text = text.replace(
                "internal_request_id=b ",
                "internal_request_id=b proxy_received_monotonic_s=200.0 ")
            log.write_text(text, encoding="utf-8")
            events = Path(name) / "decode.log"
            events.write_text(
                "PD_KV_EVENT event=recv_start request_id=a monotonic_s=100.006\n"
                "PD_KV_EVENT event=recv_complete request_id=a monotonic_s=100.010\n"
                "PD_KV_EVENT event=recv_start request_id=b monotonic_s=200.004\n"
                "PD_KV_EVENT event=recv_complete request_id=b monotonic_s=200.008\n",
                encoding="utf-8")
            summary, rows, _ = analyze_run(
                "test", run, log, cohort_threshold_ms=5,
                kv_event_path=events)
        self.assertAlmostEqual(rows[0]["kv_receive_duration_ms"], 4.0)
        self.assertAlmostEqual(
            rows[0]["kv_receive_complete_to_first_byte_ms"], 5.0)
        self.assertEqual(summary["kv_event_source"],
                         "p2p_worker_blocking_receive")
        self.assertIn("kv_state_timing", summary)

    def test_kv_event_parser_rejects_duplicate_event(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "decode.log"
            line = "PD_KV_EVENT event=promoted request_id=a monotonic_s=1.0\n"
            path.write_text(line + line, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_kv_event_log(path)


if __name__ == "__main__":
    unittest.main()
