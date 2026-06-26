import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import request


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, json_error=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


class ModelParsingTests(unittest.TestCase):
    def test_free_models_require_zero_prompt_and_completion(self):
        payload = {
            "data": [
                {"id": "vendor/model-7b:free", "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "paid-output", "pricing": {"prompt": "0", "completion": "0.1"}},
                {"id": "missing-price", "pricing": {"prompt": "0"}},
                {"id": "numeric-zero-1.5B", "pricing": {"prompt": 0, "completion": 0.0}},
                {"pricing": {"prompt": "0", "completion": "0"}},
            ]
        }

        self.assertEqual(
            request.parse_free_models(payload),
            [
                request.ModelInfo("vendor/model-7b:free", "7B"),
                request.ModelInfo("numeric-zero-1.5B", "1.5B"),
            ],
        )

    def test_invalid_model_payloads_are_rejected(self):
        for payload in (None, {}, {"data": {}}, {"data": None}):
            with self.subTest(payload=payload), self.assertRaises(request.ScannerError):
                request.parse_free_models(payload)

    def test_get_models_uses_connect_and_read_timeout(self):
        session = Mock()
        session.get.return_value = FakeResponse(
            payload={"data": [{"id": "free", "pricing": {"prompt": "0", "completion": "0"}}]}
        )
        self.assertEqual(
            request.get_free_models(session, {"Authorization": "secret"}, 12),
            [request.ModelInfo("free")],
        )
        self.assertEqual(session.get.call_args.kwargs["timeout"], (5, 12))

    def test_get_models_maps_http_errors(self):
        cases = [(401, "API-ключ"), (403, "API-ключ"), (429, "лимит"), (503, "HTTP 503"), (418, "HTTP 418")]
        for status, expected in cases:
            with self.subTest(status=status):
                session = Mock()
                session.get.return_value = FakeResponse(status, {}, {"Retry-After": "4"})
                with self.assertRaisesRegex(request.ScannerError, expected):
                    request.get_free_models(session, {}, 10)

    def test_get_models_maps_network_errors(self):
        cases = [(requests.Timeout(), "время ожидания"), (requests.ConnectionError(), "подключиться")]
        for exception, expected in cases:
            with self.subTest(exception=exception):
                session = Mock()
                session.get.side_effect = exception
                with self.assertRaisesRegex(request.ScannerError, expected):
                    request.get_free_models(session, {}, 10)


class AttemptTests(unittest.TestCase):
    def test_response_statuses_are_classified(self):
        cases = [
            (FakeResponse(200, {"choices": []}), "OK"),
            (FakeResponse(200, json_error=ValueError("bad")), "INVALID_JSON"),
            (FakeResponse(200, {"id": "no choices"}), "INVALID_JSON"),
            (FakeResponse(401, {"error": {"message": "bad key"}}), "AUTH_ERROR"),
            (FakeResponse(404, {}), "NOT_FOUND"),
            (FakeResponse(429, {}, {"Retry-After": "8"}), "RATE_LIMITED"),
            (FakeResponse(502, {}), "SERVER_ERROR"),
            (FakeResponse(422, {}), "HTTP_422"),
        ]
        for response, expected in cases:
            with self.subTest(expected=expected):
                session = Mock()
                session.post.return_value = response
                result = request.run_attempt(session, {}, "model", "hello", 5, 10)
                self.assertEqual(result.status, expected)
                if expected == "RATE_LIMITED":
                    self.assertEqual(result.retry_after, "8")

    def test_request_exceptions_are_classified(self):
        cases = [
            (requests.Timeout(), "TIMEOUT"),
            (requests.ConnectionError("offline"), "CONNECTION_ERROR"),
            (requests.RequestException("broken"), "REQUEST_ERROR"),
        ]
        for exception, expected in cases:
            with self.subTest(expected=expected):
                session = Mock()
                session.post.side_effect = exception
                result = request.run_attempt(session, {}, "model", "hello", 5, 10)
                self.assertEqual(result.status, expected)


class ResultTests(unittest.TestCase):
    def test_result_uses_success_median_and_partial_status(self):
        result = request.ModelResult(
            request.ModelInfo("m"),
            [
                request.AttemptResult("OK", 0.8),
                request.AttemptResult("TIMEOUT"),
                request.AttemptResult("OK", 0.2),
            ],
        )
        self.assertEqual(result.successful_runs, 2)
        self.assertEqual(result.median_latency, 0.5)
        self.assertEqual(result.status, "PARTIAL")

    def test_sort_and_fastest_are_deterministic(self):
        slow = request.ModelResult(request.ModelInfo("z"), [request.AttemptResult("OK", 2.0)])
        fast_b = request.ModelResult(request.ModelInfo("b"), [request.AttemptResult("OK", 1.0)])
        fast_a = request.ModelResult(request.ModelInfo("a"), [request.AttemptResult("OK", 1.0)])
        failed = request.ModelResult(request.ModelInfo("failed"), [request.AttemptResult("TIMEOUT")])
        results = request.sort_results([slow, failed, fast_b, fast_a], "latency")
        self.assertEqual([item.model.id for item in results], ["a", "b", "z", "failed"])
        self.assertEqual(request.fastest_result(results).model.id, "a")
        self.assertIsNone(request.fastest_result([failed]))

    def test_exit_codes_distinguish_failures_from_api_errors(self):
        timeout = request.ModelResult(
            request.ModelInfo("slow"), [request.AttemptResult("TIMEOUT")]
        )
        auth = request.ModelResult(
            request.ModelInfo("private"), [request.AttemptResult("AUTH_ERROR")]
        )
        success = request.ModelResult(
            request.ModelInfo("ok"), [request.AttemptResult("OK", 0.2)]
        )
        self.assertEqual(request.exit_code_for_results([timeout]), 1)
        self.assertEqual(request.exit_code_for_results([auth]), 2)
        self.assertEqual(request.exit_code_for_results([auth, success]), 0)

    def test_json_export_contains_summary_and_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "results.json"
            result = request.ModelResult(
                request.ModelInfo("m", "7B"), [request.AttemptResult("OK", 0.1)]
            )
            request.export_results(output, [result])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["model"], {"id": "m", "params": "7B"})
            self.assertEqual(payload[0]["median_latency"], 0.1)


class CliTests(unittest.TestCase):
    def test_missing_key_returns_configuration_error(self):
        stderr = io.StringIO()
        with patch.object(request, "load_dotenv"), patch.object(
            request.os, "getenv", return_value=None
        ), redirect_stderr(stderr):
            self.assertEqual(request.main([]), 2)
        self.assertIn("OPENROUTER_API_KEY", stderr.getvalue())

    def test_empty_filter_result_returns_one(self):
        session = Mock()
        session.get.return_value = FakeResponse(
            payload={"data": [{"id": "vendor/free", "pricing": {"prompt": "0", "completion": "0"}}]}
        )
        stdout = io.StringIO()
        with patch.object(request, "load_dotenv"), patch.object(
            request.os, "getenv", return_value="test-key"
        ), patch.object(request.requests, "Session", return_value=session), redirect_stdout(stdout):
            self.assertEqual(request.main(["--model", "missing"]), 1)
        self.assertIn("не найдены", stdout.getvalue())

    def test_success_and_json_export_return_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            session = Mock()
            session.get.return_value = FakeResponse(
                payload={"data": [{"id": "vendor/free", "pricing": {"prompt": "0", "completion": "0"}}]}
            )
            session.post.return_value = FakeResponse(payload={"choices": []})
            with patch.object(request, "load_dotenv"), patch.object(
                request.os, "getenv", return_value="test-key"
            ), patch.object(request.requests, "Session", return_value=session), redirect_stdout(io.StringIO()):
                self.assertEqual(request.main(["--json", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())[0]["status"], "OK")

    def test_invalid_positive_number_exits_with_code_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as context:
            request.main(["--workers", "0"])
        self.assertEqual(context.exception.code, 2)

    def test_non_finite_timeout_exits_with_code_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as context:
            request.main(["--timeout", "nan"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
