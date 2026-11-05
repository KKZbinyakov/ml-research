"""Safe request audit middleware. It emits only fixed-shape metadata."""
import json
import hashlib
import hmac
import logging
import os
import time
import uuid

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
COMPONENTS = ("api", "engine", "model", "compute", "runtime", "supervisor")
ERROR_TYPES = ("none", "ValueError", "MemoryError", "PermissionError", "TimeoutError",
               "OSError", "RuntimeError", "OtherError")
STAGES = ("runtime", "exception", "configuration", "spawn", "supervision", "shutdown")


def error_name(exception_type):
    if exception_type is None:
        return "none"
    if isinstance(exception_type, type):
        for candidate in (MemoryError, PermissionError, TimeoutError, OSError, ValueError, RuntimeError):
            if issubclass(exception_type, candidate):
                return candidate.__name__
    return "OtherError"


def audit_record(event, request_id, method, route, status, duration_ms):
    if event not in ("request_started", "request_finished"):
        return None
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        method = "OTHER"
    if route not in ("/v1/chat/completions", "/v1/completions"):
        route = "other"
    try:
        if str(uuid.UUID(request_id)) != request_id:
            return None
    except (ValueError, TypeError, AttributeError):
        return None
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "OTHER"):
        return None
    if type(status) is not int or not 100 <= status <= 599:
        return None
    if type(duration_ms) not in (int, float) or not 0 <= duration_ms < 1e12:
        return None
    return {"event": event, "request_id": request_id, "method": method,
            "route": route, "status": status, "duration_ms": duration_ms}


def safe_audit_line(line):
    """Return canonical JSON only for a strictly valid audit object."""
    try:
        data = json.loads(line)
        if type(data) is not dict or set(data) != {
                "event", "request_id", "method", "route", "status", "duration_ms"}:
            return None
        record = audit_record(**data)
        return json.dumps(record, separators=(",", ":"), allow_nan=False) if record == data else None
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


class AuditFilter:
    """Filter arbitrary child output incrementally without forwarding raw fragments."""
    def __init__(self, key):
        self.buffer = bytearray()
        self.discarding = False
        self.dropped_lines = 0
        self.key = key.encode("utf-8") if isinstance(key, str) else key

    def feed(self, chunk):
        if self.discarding:
            end = chunk.find(b"\n")
            if end < 0:
                return []
            chunk = chunk[end + 1:]
            self.discarding = False
        self.buffer.extend(chunk)
        if len(self.buffer) > 65536 and b"\n" not in self.buffer:
            self.buffer.clear()
            self.discarding = True
            self.dropped_lines += 1
            return []
        accepted = []
        while b"\n" in self.buffer:
            raw, _, rest = self.buffer.partition(b"\n")
            self.buffer[:] = rest
            try:
                decoded = raw.decode("utf-8")
                envelope = json.loads(decoded)
                if (type(envelope) is not dict or set(envelope) != {"record", "signature"}
                        or type(envelope["signature"]) is not str):
                    line = None
                else:
                    plain = json.dumps(envelope["record"], separators=(",", ":"), allow_nan=False)
                    signature = hmac.new(self.key, plain.encode("utf-8"), hashlib.sha256).hexdigest()
                    line = (safe_output_line(plain) if hmac.compare_digest(signature, envelope["signature"])
                            else None)
            except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError, RecursionError):
                line = None
            if line:
                accepted.append(line)
            else:
                self.dropped_lines += 1
        return accepted

    def finish(self):
        if self.buffer:
            self.dropped_lines += 1
            self.buffer.clear()


def safe_output_line(line):
    audit = safe_audit_line(line)
    if audit:
        return audit
    try:
        data = json.loads(line)
        if type(data) is not dict:
            return None
        if data.get("event") == "runtime_log" and set(data) == {
                "event", "level", "component", "stage", "error_type"}:
            if (data["level"] not in LEVELS or data["component"] not in COMPONENTS
                    or data["stage"] not in STAGES or data["error_type"] not in ERROR_TYPES):
                return None
        elif data.get("event") == "startup" and set(data) == {"event", "status"}:
            if type(data["status"]) is not int or data["status"] not in (0, 1):
                return None
        elif data.get("event") == "exit" and set(data) == {"event", "status"}:
            if type(data["status"]) is not int or not 0 <= data["status"] <= 255:
                return None
        elif data.get("event") == "child_output_dropped" and set(data) == {"event", "count"}:
            if type(data["count"]) is not int or data["count"] < 1:
                return None
        else:
            return None
        return json.dumps(data, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None


def emit_safe(record):
    key = os.environ.get("ML_AUDIT_KEY", "").encode("utf-8")
    plain = json.dumps(record, separators=(",", ":"), allow_nan=False)
    line = signed_line(record, key) if key and safe_output_line(plain) else None
    if line:
        os.write(1, (line + "\n").encode("utf-8"))


def signed_line(record, key):
    plain = json.dumps(record, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(key, plain.encode("utf-8"), hashlib.sha256).hexdigest()
    return json.dumps({"record": record, "signature": signature}, separators=(",", ":"))


class SafeFormatter:
    """Classify the source and exception without ever formatting their text."""
    def __init__(self, fmt=None, datefmt=None, style="%"):
        pass

    def format(self, record):
        level = record.levelname if record.levelname in LEVELS else "ERROR"
        name = str(record.name)
        component = "runtime"
        for prefixes, value in (
                (("vllm.v1.engine", "vllm.v1.worker"), "engine"),
                (("vllm.entrypoints", "uvicorn", "starlette", "fastapi"), "api"),
                (("huggingface_hub", "transformers"), "model"),
                (("torch", "triton"), "compute")):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                component = value
                break
        error_type = error_name(record.exc_info[0] if record.exc_info else None)
        return json.dumps({"event": "runtime_log", "level": level, "component": component,
                           "stage": "exception" if error_type != "none" else "runtime",
                           "error_type": error_type}, separators=(",", ":"))


class SafeHandler(logging.Handler):
    """Write only schema-validated records around vLLM's stdout/stderr wrappers."""
    def emit(self, record):
        try:
            plain = self.format(record)
            data = json.loads(plain)
            key = os.environ.get("ML_AUDIT_KEY", "").encode("utf-8")
            line = signed_line(data, key) if key and safe_output_line(plain) else None
            if line:
                os.write(1, (line + "\n").encode("utf-8"))
        except BaseException:
            pass


class AuditMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = str(uuid.uuid4())
        method = scope.get("method", "OTHER")
        route = scope.get("path", "")
        route = route if route in ("/v1/chat/completions", "/v1/completions") else "other"
        started = time.monotonic()
        status = 500
        record = audit_record("request_started", rid, method, route, 100, 0)
        emit_safe(record)

        async def tracked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            if not self.authorized(scope):
                status = 401
                body = b'{"error":"Unauthorized"}'
                await tracked_send({"type": "http.response.start", "status": 401,
                                    "headers": [(b"content-type", b"application/json"),
                                                (b"www-authenticate", b"Bearer")]})
                await tracked_send({"type": "http.response.body", "body": body})
                return
            await self.app(scope, receive, tracked_send)
        finally:
            record = audit_record("request_finished", rid, method, route, status,
                                  max(0, (time.monotonic() - started) * 1000))
            emit_safe(record)

    @staticmethod
    def authorized(scope):
        method = scope.get("method", "").upper()
        if method in ("GET", "HEAD") and scope.get("path") == "/health":
            return True
        api_key = os.environ.get("VLLM_API_KEY")
        if not api_key:
            return False
        values = [value for name, value in scope.get("headers", [])
                  if name.lower() == b"authorization"]
        if len(values) != 1:
            return False
        try:
            expected = b"Bearer " + api_key.encode("utf-8")
            return hmac.compare_digest(values[0], expected)
        except (AttributeError, UnicodeError, TypeError):
            return False
