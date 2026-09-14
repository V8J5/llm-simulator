import unittest

from scripts.collect_vllm_ground_truth import parse_scheduler_metrics


class VLLMGroundTruthMetricTests(unittest.TestCase):
    def test_only_scheduler_metrics_are_retained(self):
        text = """
# HELP vllm:num_requests_running Number running
vllm:num_requests_running{engine="0",model_name="m"} 20.0
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
vllm:num_preemptions_total{engine="0",model_name="m"} 1.0
vllm:iteration_tokens_total_bucket{engine="0",le="32.0"} 9.0
vllm:request_prompt_tokens_sum{engine="0"} 512.0
"""
        metrics = parse_scheduler_metrics(text)
        self.assertEqual(len(metrics), 4)
        self.assertIn(
            'vllm:num_requests_running{engine="0",model_name="m"}', metrics)
        self.assertNotIn('vllm:request_prompt_tokens_sum{engine="0"}', metrics)


if __name__ == "__main__":
    unittest.main()
