import json
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI

from samba_bench.client import MeasurementError, run_loop, run_turn
from samba_bench.fixtures import demo_client, sse_events


class MeasurementTests(unittest.TestCase):
    def client(self, *, body=None, status=200):
        def handler(request):
            return httpx2.Response(status, headers={"Content-Type": "text/event-stream"},
                                   content=sse_events() if body is None else body)
        return OpenAI(api_key="synthetic-demo-key", base_url="https://fixture.invalid/v1",
                      http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0)

    def run_sample(self, client):
        times = iter([1.0, 1.5])
        with client:
            return run_turn(client, [{"role": "user", "content": "Explain decode"}], clock=lambda: next(times))

    def test_actual_sdk_parses_extended_usage(self):
        result = self.run_sample(self.client())
        self.assertEqual(result.text, "Synthetic fixture response.")
        self.assertEqual(result.server_time_to_first_token_seconds, 0.25)
        self.assertEqual(result.server_completion_tokens_after_first_per_second, 80)
        self.assertEqual(result.client_elapsed_seconds, 0.5)
        self.assertEqual(result.completion_tokens, 5)

    def test_native_sse_flow_uses_no_sockets(self):
        with patch("socket.socket.connect", side_effect=AssertionError("Unexpected network")):
            result = self.run_sample(self.client())
        self.assertEqual(result.total_tokens, 13)

    def test_loop_preserves_context_and_request_options(self):
        requests = []
        with demo_client(requests) as client:
            result = run_loop(client, "Explain decode", steps=10)
        self.assertEqual(len(requests), 10)
        self.assertEqual([len(request["messages"]) for request in requests], list(range(1, 20, 2)))
        self.assertEqual(requests[-1]["messages"][1]["role"], "assistant")
        self.assertTrue(all(request["stream_options"] == {"include_usage": True} for request in requests))
        self.assertTrue(all(request["max_tokens"] == 300 for request in requests))
        self.assertEqual(result["totals"]["reported_completion_tokens"], 50)
        self.assertEqual(result["totals"]["server_time_to_first_token_seconds_sum"], 2.5)
        self.assertIsNone(result["totals"]["server_decode_duration_seconds"])

    def test_missing_usage_is_failure(self):
        with self.assertRaisesRegex(MeasurementError, "final usage"):
            self.run_sample(self.client(body=sse_events(include_usage=False)))

    def test_missing_provider_timing_is_not_estimated(self):
        usage = {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13}
        with self.assertRaisesRegex(MeasurementError, "omitted"):
            self.run_sample(self.client(body=sse_events(usage=usage)))

    def test_invalid_rates_and_nonfinite_metrics(self):
        for field, value in [("time_to_first_token", -1), ("time_to_first_token", float("nan")),
                             ("completion_tokens_after_first_per_sec", 0),
                             ("completion_tokens_after_first_per_sec", float("inf")),
                             ("time_to_first_token", 10 ** 400),
                             ("completion_tokens_after_first_per_sec", True)]:
            usage = {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13,
                     "time_to_first_token": 0.25, "completion_tokens_after_first_per_sec": 80}
            usage[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(MeasurementError):
                self.run_sample(self.client(body=sse_events(usage=usage)))

    def test_malformed_text_is_a_controlled_measurement_error(self):
        for text in [7, ["unexpected"], {"unexpected": "object"}]:
            with self.subTest(text=text), self.assertRaisesRegex(MeasurementError, "text must"):
                self.run_sample(self.client(body=sse_events(text=text)))

    def test_event_deadline_is_checked(self):
        with patch("samba_bench.client.time.monotonic", side_effect=[0, 31]):
            with self.assertRaisesRegex(MeasurementError, "time budget"):
                self.run_sample(self.client())

    def test_inconsistent_token_counts(self):
        usage = {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 999,
                 "time_to_first_token": 0.25, "completion_tokens_after_first_per_sec": 80}
        with self.assertRaisesRegex(MeasurementError, "inconsistent"):
            self.run_sample(self.client(body=sse_events(usage=usage)))

    def test_truncated_finish_is_failure(self):
        with self.assertRaisesRegex(MeasurementError, "finish"):
            self.run_sample(self.client(body=sse_events(finish_reason=None)))

    def test_length_limited_completion_is_explicit(self):
        result = self.run_sample(self.client(body=sse_events(finish_reason="length")))
        self.assertEqual(result.finish_reason, "length")

    def test_stream_body_budget(self):
        with self.assertRaisesRegex(MeasurementError, "text budget"):
            self.run_sample(self.client(body=sse_events(text="x" * 100001)))

    def test_step_prompt_and_token_budgets(self):
        with demo_client() as client:
            for steps in [0, 11, True]:
                with self.assertRaises(MeasurementError):
                    run_loop(client, "x", steps=steps)
            for prompt in ["", "x" * 4001]:
                with self.assertRaises(MeasurementError):
                    run_loop(client, prompt)
            for tokens in [0, 4097, True]:
                with self.assertRaises(MeasurementError):
                    run_turn(client, [{"role": "user", "content": "x"}], max_tokens=tokens)

    def test_context_budget_precedes_network(self):
        with demo_client() as client, self.assertRaisesRegex(MeasurementError, "character budget"):
            run_turn(client, [{"role": "user", "content": "x" * 20001}])

    def test_cli_demo_and_missing_live_key(self):
        result = subprocess.run([sys.executable, "-B", "-m", "samba_bench", "--demo", "--steps", "2"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["performance_evidence"])
        self.assertEqual(payload["steps"], 2)
        environment = os.environ.copy()
        environment.pop("SAMBANOVA_API_KEY", None)
        result = subprocess.run([sys.executable, "-B", "-m", "samba_bench", "--live"],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("SAMBANOVA_API_KEY", result.stderr)


if __name__ == "__main__":
    unittest.main()
