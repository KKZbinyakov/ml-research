import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import run
import summarize
from types import SimpleNamespace


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_use_nearest_rank(self):
        self.assertEqual(run.percentile(list(range(1, 9)), .95), 8)
        self.assertEqual(run.percentile(list(range(1, 9)), .50), 4)

    def request(self, body, status=200, done=True):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(status)
                self.end_headers()
                if status == 200:
                    for event in body:
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    if done:
                        self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        result = run.stream_request(f"http://127.0.0.1:{server.server_port}", "test", "secret", "hello", 4)
        payload = getattr(server, "payload", None)
        server.shutdown()
        server.server_close()
        return result, payload

    def test_ignores_role_chunk_and_measures_content_usage(self):
        result, payload = self.request([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 7}},
        ])
        self.assertTrue(result["ok"])
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (13, 7))
        self.assertTrue(result["ttft_seconds"] <= result["latency_seconds"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertNotIn("secret", json.dumps(result))

    def test_rejects_missing_done_or_usage(self):
        result, _ = self.request([{"choices": [{"delta": {"content": "x"}}]}], done=False)
        self.assertFalse(result["ok"])
        result, _ = self.request([{"choices": [{"delta": {"content": "x"}}]}])
        self.assertFalse(result["ok"])

    def test_reports_http_status_without_body(self):
        result, _ = self.request([], status=401)
        self.assertEqual(result, {"ok": False, "status": 401, "error": "http_error"})

    def test_malformed_sse_is_fixed_error_without_payload(self):
        original = run.urllib.request.build_opener
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self): return iter([b"data: {broken json}\n\n"])
        class Opener:
            def open(self, *args, **kwargs): return Response()
        run.urllib.request.build_opener = lambda *args, **kwargs: Opener()
        try:
            result = run.stream_request("http://localhost", "m", "secret", "p", 1)
        finally:
            run.urllib.request.build_opener = original
        self.assertEqual(result["error"], "malformed_sse")

    def test_warmup_failure_skips_measurement(self):
        args = SimpleNamespace(base_url="", model="", token="", requests=3, label="x", repeat=2)
        calls = []
        def fake(*_):
            calls.append(1)
            return {"ok": False, "error": "connection_or_protocol_error"}
        original = run.stream_request
        run.stream_request = fake
        try:
            case = run.run_case(args, 2, "short")
        finally:
            run.stream_request = original
        self.assertEqual(len(calls), 2)
        self.assertFalse(case["warmup"]["ok"])
        self.assertEqual(case["errors"], 3)
        self.assertEqual(case["requests_detail"], [])

    def test_summary_pools_tokens_and_wall_not_repeat_rates(self):
        docs = [{"results": [
            {"label": "x", "workload": "short", "concurrency": 1, "repeat": 1, "requests": 2, "successes": 2,
             "wall_seconds": 4, "output_tokens": 10, "requests_detail": []},
            {"label": "x", "workload": "short", "concurrency": 1, "repeat": 2, "requests": 2, "successes": 1,
             "wall_seconds": 1, "output_tokens": 6, "requests_detail": []}]}]
        row = summarize.summarize(docs)[0]
        self.assertEqual(row["requests_per_second"], .6)
        self.assertEqual(row["output_tokens_per_second"], 16 / 5)
        self.assertEqual((row["repeat_throughput_min"], row["repeat_throughput_max"]), (.5, 1))


if __name__ == "__main__":
    unittest.main()
