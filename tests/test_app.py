import contextlib
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import privacy


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_model", APP / "download_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrivacyTests(unittest.TestCase):
    def test_only_exact_safe_audit_schema_passes(self):
        record = {"event": "request_finished", "request_id": "12345678-1234-4234-8234-123456789012",
                  "method": "POST", "route": "/v1/completions", "status": 200, "duration_ms": 1.5}
        self.assertEqual(json.loads(privacy.safe_audit_line(json.dumps(record))), record)
        for bad in (dict(record, prompt="secret prompt"), dict(record, route="/v1?q=secret"),
                    dict(record, request_id="secret"), dict(record, status="200"),
                    {"message": "secret prompt"}, "not json", "[DONE]"):
            self.assertIsNone(privacy.safe_audit_line(json.dumps(bad) if isinstance(bad, dict) else bad))

    def test_malicious_values_are_rejected_or_normalized(self):
        record = privacy.audit_record("request_started", "12345678-1234-4234-8234-123456789012", "POST /secret", "/v1?prompt=x", 100, 0)
        self.assertEqual(record["method"], "OTHER")
        self.assertEqual(record["route"], "other")
        self.assertNotIn("secret", json.dumps(record))

    def test_streaming_output_filters_prompt_text_even_across_chunks(self):
        allowed = {"event": "request_finished", "request_id": "12345678-1234-4234-8234-123456789012",
                   "method": "POST", "route": "/v1/completions", "status": 200, "duration_ms": 1}
        safe = (privacy.signed_line(allowed, b"unit-test-key") + "\n").encode()
        stream = privacy.AuditFilter(b"unit-test-key")
        self.assertEqual(stream.feed(b"generated PRIVATE PROMPT\n" + safe[:17]), [])
        accepted = stream.feed(safe[17:] + b"prompt PRIVATE OUTPUT\n")
        self.assertEqual(len(accepted), 1)
        self.assertNotIn("PRIVATE", "".join(accepted))

    def test_filter_rejects_unsigned_forged_json_and_wrong_signature(self):
        record = {"event": "request_finished", "request_id": "12345678-1234-4234-8234-123456789012",
                  "method": "POST", "route": "/v1/completions", "status": 200, "duration_ms": 1}
        unsigned = json.dumps(record, separators=(",", ":")).encode() + b"\n"
        forged = {"record": record, "signature": "0" * 64}
        stream = privacy.AuditFilter(b"unit-test-key")
        self.assertEqual(stream.feed(unsigned), [])
        self.assertEqual(stream.feed(json.dumps(forged).encode() + b"\n"), [])

    def test_filter_discards_recursively_nested_untrusted_record(self):
        stream = privacy.AuditFilter(b"unit-test-key")
        self.assertEqual(stream.feed((b"[" * 1500) + b"0" + (b"]" * 1500) + b"\n"), [])

    def test_huge_line_tail_cannot_become_a_forwarded_record(self):
        stream = privacy.AuditFilter(b"unit-test-key")
        line = json.dumps({"event": "startup", "status": 0}).encode()
        self.assertEqual(stream.feed(b"x" * 65537), [])
        signed = (privacy.signed_line(json.loads(line), b"unit-test-key") + "\n").encode()
        self.assertEqual(stream.feed(signed), [])
        self.assertEqual(stream.dropped_lines, 1)

    def test_safe_handler_bypasses_wrapped_stream_and_discards_message(self):
        handler = privacy.SafeHandler()
        handler.setFormatter(privacy.SafeFormatter())
        written = []
        with patch.dict(os.environ, {"ML_AUDIT_KEY": "unit-test-key"}), \
                patch("privacy.os.write", side_effect=lambda fd, data: written.append(data.decode())):
            handler.emit(logging.makeLogRecord({"levelname": "ERROR", "msg": "private prompt"}))
        accepted = privacy.AuditFilter(b"unit-test-key").feed(written[0].encode())
        self.assertEqual(json.loads(accepted[0]), {"event": "runtime_log", "level": "ERROR",
            "component": "runtime", "stage": "runtime", "error_type": "none"})

    def test_middleware_auth_protects_invocations_and_every_unknown_path(self):
        calls = []
        async def app(scope, receive, send):
            calls.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        for path, headers in (("/invocations", []), ("/unknown-inference", []),
                              ("/invocations", [(b"authorization", b"Bearer wrong")]),
                              ("/invocations", [(b"authorization", b"Bearer test-key"),
                                                 (b"authorization", b"Bearer test-key")])):
            sent, writes = [], []
            async def send(message):
                sent.append(message)
            with patch.dict(os.environ, {"VLLM_API_KEY": "test-key", "ML_AUDIT_KEY": "audit-key"}), \
                    patch("privacy.os.write", side_effect=lambda fd, data: writes.append(data)):
                coro = privacy.AuditMiddleware(app)(
                    {"type": "http", "method": "POST", "path": path,
                     "headers": headers}, receive, send)
                with self.assertRaises(StopIteration):
                    coro.send(None)
            self.assertEqual(sent[0]["status"], 401)
            self.assertEqual(sent[0]["headers"][-1], (b"www-authenticate", b"Bearer"))
            self.assertNotIn(b"test-key", sent[1]["body"])
            audit = privacy.AuditFilter("audit-key")
            records = [json.loads(line) for raw in writes for line in audit.feed(raw)]
            self.assertEqual([r["event"] for r in records], ["request_started", "request_finished"])
            self.assertEqual(records[-1]["status"], 401)
        self.assertEqual(calls, [])

    def test_middleware_accepts_exact_bearer_and_keeps_health_public(self):
        seen = []
        async def app(scope, receive, send):
            seen.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        for scope, env in (({"type": "http", "method": "POST", "path": "/invocations",
                            "headers": [(b"authorization", b"Bearer test-key")]},
                           {"VLLM_API_KEY": "test-key", "ML_AUDIT_KEY": "audit-key"}),
                           ({"type": "http", "method": "GET", "path": "/health", "headers": []},
                            {"ML_AUDIT_KEY": "audit-key"})):
            sent = []
            async def send(message):
                sent.append(message)
            with patch.dict(os.environ, env, clear=True), patch("privacy.os.write", return_value=1):
                coro = privacy.AuditMiddleware(app)(scope, receive, send)
                with self.assertRaises(StopIteration):
                    coro.send(None)
            self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(seen, ["/invocations", "/health"])

    def test_websocket_is_closed_without_authentication(self):
        sent = []
        async def send(message):
            sent.append(message)
        async def unused(*args):
            self.fail("WebSocket inference must not reach the app")
        with patch.dict(os.environ, {}, clear=True):
            coro = privacy.AuditMiddleware(unused)(
                {"type": "websocket", "path": "/v1/chat/completions"}, unused, send)
            with self.assertRaises(StopIteration):
                coro.send(None)
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])

    def test_asgi_middleware_audits_success_stream_and_exception_without_text(self):
        outputs = []
        async def send(message):
            outputs.append(message)
        async def receive():
            return {"type": "http.request", "body": b"private prompt", "more_body": False}
        async def ok_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"private generated text", "more_body": True})
            await send({"type": "http.response.body", "body": b"tail", "more_body": False})
        writes = []
        stdout = io.StringIO()
        key = "unit-test-key"
        with patch.dict(os.environ, {"ML_AUDIT_KEY": key, "VLLM_API_KEY": "test-key"}), \
                patch("privacy.os.write", side_effect=lambda fd, data: writes.append(data.decode())):
            with contextlib.redirect_stdout(stdout):
                coroutine = privacy.AuditMiddleware(ok_app)(
                    {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                     "headers": [(b"authorization", b"Bearer test-key")]}, receive, send)
                with self.assertRaises(StopIteration):
                    coroutine.send(None)
        stream = privacy.AuditFilter(key)
        lines = []
        for data in writes:
            lines.extend(stream.feed(data.encode()))
        self.assertEqual([json.loads(line)["event"] for line in lines], ["request_started", "request_finished"])
        self.assertEqual(json.loads(lines[-1])["status"], 200)
        self.assertEqual(stdout.getvalue(), "")

        async def broken(scope, receive, send):
            raise RuntimeError("private exception details")
        stdout = io.StringIO()
        writes = []
        with patch.dict(os.environ, {"ML_AUDIT_KEY": key, "VLLM_API_KEY": "test-key"}), \
                patch("privacy.os.write", side_effect=lambda fd, data: writes.append(data.decode())):
            with contextlib.redirect_stdout(stdout), self.assertRaises(RuntimeError):
                privacy.AuditMiddleware(broken)(
                    {"type": "http", "method": "POST", "path": "/v1/completions",
                     "headers": [(b"authorization", b"Bearer test-key")]}, receive, send).send(None)
        stream = privacy.AuditFilter(key)
        lines = []
        for data in writes:
            lines.extend(stream.feed(data.encode()))
        self.assertEqual(json.loads(lines[-1])["status"], 500)
        self.assertEqual(stdout.getvalue(), "")


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.model_dir = Path(self.temp.name) / "model"
        self.module = load_downloader()
        self.patches = [patch.object(self.module, "MODEL_REVISION", "a" * 40),
                        patch.object(self.module, "MODEL_DIR", self.model_dir),
                        patch.object(self.module, "STAGING", self.model_dir.with_name("model.staging")),
                        patch.object(self.module, "BACKUP", self.model_dir.with_name("model.previous"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.temp.cleanup()

    def valid_files(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (path / name).write_text("{}")
        return {name: (path / name).stat().st_size for name in
                ("weights.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")}

    def mark_valid(self, path):
        files, hashes = self.module.snapshot_files(path)
        (path / self.module.MARKER).write_text(json.dumps({"model_id": self.module.MODEL_ID,
            "revision": "a" * 40, "files": files, "sha256": hashes}))

    def test_warm_path_skips_huggingface_import_and_token(self):
        self.valid_files(self.model_dir)
        self.mark_valid(self.model_dir)
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertEqual(self.module.main(), 0)

    def test_downloader_cli_warm_path_exits_successfully_without_token(self):
        self.valid_files(self.model_dir)
        self.mark_valid(self.model_dir)
        env = os.environ.copy()
        env.update(MODEL_DIR=str(self.model_dir), MODEL_ID=self.module.MODEL_ID,
                   MODEL_REVISION="a" * 40)
        env.pop("HF_TOKEN", None)
        result = subprocess.run([sys.executable, str(APP / "download_model.py")],
                                cwd=APP.parent, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"event": "model_cache", "stage": "ready", "code": "local_ready"})
        self.assertEqual(result.stderr, "")

    def test_cold_path_cleans_interrupted_staging_and_commits_valid_snapshot(self):
        self.module.STAGING.mkdir()
        (self.module.STAGING / "partial").write_text("partial")
        hub = types.ModuleType("huggingface_hub")
        def download(**kwargs):
            self.assertEqual(kwargs["revision"], "a" * 40)
            self.assertIsNone(kwargs["token"])
            self.assertIn("*.safetensors", kwargs["allow_patterns"])
            self.valid_files(self.module.STAGING)
            lock = self.module.STAGING / ".cache" / "huggingface" / "download" / "metadata.lock"
            lock.parent.mkdir(parents=True)
            lock.touch()
        hub.snapshot_download = download
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True), patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(self.module.main(), 0)
        self.assertTrue(self.module.valid(self.model_dir))
        self.assertNotIn(".cache/huggingface/download/metadata.lock",
                         json.loads((self.model_dir / self.module.MARKER).read_text())["files"])
        self.assertFalse((self.module.STAGING / "partial").exists())

    def test_identity_mismatch_redownloads(self):
        self.valid_files(self.model_dir)
        (self.model_dir / self.module.MARKER).write_text("{}")
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kwargs: self.valid_files(self.module.STAGING)
        with patch.dict(os.environ, {"HF_TOKEN": "secret"}), patch.dict(sys.modules, {"huggingface_hub": hub}):
            self.assertEqual(self.module.main(), 0)
        self.assertTrue(self.module.valid(self.model_dir))

    def test_corrupt_files_redownload_and_failed_download_preserves_old_directory(self):
        self.valid_files(self.model_dir)
        self.mark_valid(self.model_dir)
        (self.model_dir / "weights.safetensors").write_text("")
        hub = types.ModuleType("huggingface_hub")
        def fail(**kwargs):
            raise RuntimeError("secret token and URL")
        hub.snapshot_download = fail
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaises(RuntimeError):
                self.module.main()
        self.assertTrue(self.model_dir.exists())
        self.assertEqual((self.model_dir / "weights.safetensors").read_text(), "")

    def test_failed_revision_update_preserves_complete_previous_snapshot(self):
        self.valid_files(self.model_dir)
        self.mark_valid(self.model_dir)
        marker = self.model_dir / self.module.MARKER
        data = json.loads(marker.read_text())
        data["revision"] = "b" * 40
        marker.write_text(json.dumps(data))
        before = marker.read_bytes()
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("download failed"))
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaises(RuntimeError):
                self.module.main()
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual((self.model_dir / "weights.safetensors").stat().st_size, 10)


if __name__ == "__main__":
    unittest.main()
