import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_pd_validation import (
    config_from_deployment,
    load_pd_aggregate,
    validate_trace_identity,
)


def deployment():
    return {
        "prefill": {
            "replicas": 1, "tensor_parallel_size": 4,
            "num_gpu_blocks_from_log": 22486,
            "max_num_batched_tokens": 4096, "max_num_seqs": 8,
        },
        "decode": {
            "replicas": 1, "tensor_parallel_size": 4,
            "num_gpu_blocks_from_log": 22486,
            "max_num_batched_tokens": 4096, "max_num_seqs": 32,
        },
        "runtime": {"block_size_tokens": 16},
        "kv_connector": {
            "name": "P2pNcclConnector",
            "transport": "NCCL", "measured_bandwidth_gb_s": None,
            "measured_fixed_latency_ms": None,
        },
    }


class PDValidationTests(unittest.TestCase):
    def test_configuration_is_derived_from_manifest(self):
        config, provenance = config_from_deployment(
            deployment(), chunked_prefill=True)
        self.assertEqual(config.num_prefill_gpus, 4)
        self.assertEqual(config.prefill_max_num_seqs, 8)
        self.assertEqual(config.decode_max_num_seqs, 32)
        self.assertEqual(config.prefill_num_gpu_blocks, 22486)
        self.assertEqual(config.kv_transfer_mode, "nccl")
        self.assertEqual(config.decode_kv_load_policy, "blocking_receive")
        self.assertEqual(config.kv_transfer_bw_gb_s, 12.5)
        self.assertTrue(provenance["kv_transfer_bandwidth_assumed"])

    def test_explicit_transfer_measurement_overrides_fallback(self):
        config, provenance = config_from_deployment(
            deployment(), chunked_prefill=True,
            bandwidth_gb_s=23.0, latency_ms=0.25)
        self.assertEqual(config.kv_transfer_bw_gb_s, 23.0)
        self.assertEqual(config.kv_transfer_latency_ms, 0.25)
        self.assertFalse(provenance["kv_transfer_bandwidth_assumed"])
        self.assertFalse(provenance["kv_transfer_latency_assumed"])

    def test_loader_rejects_non_pd_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps({
                "collector": "vllm_openai_streaming_aggregate",
                "requests": [{}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PD"):
                load_pd_aggregate(path)

    def test_trace_identity_rejects_changed_workload(self):
        class Request:
            request_id = "req_1"
            arrival_time = 0.0
            prompt_len = 513
            output_len = 128

        aggregate = {"requests": [{
            "request_id": "req_1", "arrival_time_ms": 0.0,
            "prompt_len": 512, "output_len": 128,
        }]}
        with self.assertRaisesRegex(ValueError, "trace differs"):
            validate_trace_identity(aggregate, [Request()])


if __name__ == "__main__":
    unittest.main()
