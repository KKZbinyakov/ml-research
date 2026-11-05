#!/usr/bin/env python3
"""Run vLLM behind a signed stdout/stderr privacy boundary."""
import json
import os
import selectors
import secrets
import signal
import subprocess
import sys
import time

from privacy import AuditFilter, error_name

SHUTDOWN_SECONDS = 10
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


def emit(event, **fields):
    sys.stdout.write(json.dumps({"event": event, **fields}, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def diagnostic(stage, exception_type):
    emit("runtime_log", level="ERROR", component="supervisor", stage=stage,
         error_type=error_name(exception_type))


def send_signal(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def supervise(args):
    child = None
    selector = None
    termination = 0
    deadline = None
    previous = {}
    phase = "spawn"

    def forward_signal(signum, _frame):
        nonlocal termination, deadline
        termination = signum
        if deadline is None:
            deadline = time.monotonic() + SHUTDOWN_SECONDS
        if child is not None:
            send_signal(child, signum)

    try:
        # A signal received during Popen is remembered until its handle exists.
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward_signal)
        child = subprocess.Popen(["vllm", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 bufsize=0, start_new_session=True)
        if termination:
            send_signal(child, termination)
        emit("startup", status=0)
        phase = "supervision"
        selector = selectors.DefaultSelector()
        filters = {child.stdout: AuditFilter(os.environ["ML_AUDIT_KEY"]),
                   child.stderr: AuditFilter(os.environ["ML_AUDIT_KEY"])}
        for stream in filters:
            selector.register(stream, selectors.EVENT_READ)
        child_exit_deadline = None
        while selector.get_map():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                send_signal(child, KILL_SIGNAL)
                break
            if child.poll() is not None:
                if child_exit_deadline is None:
                    child_exit_deadline = now + SHUTDOWN_SECONDS
                elif now >= child_exit_deadline:
                    # Descendants cannot retain the inherited pipes indefinitely.
                    send_signal(child, KILL_SIGNAL)
                    break
            for key, _ in selector.select(timeout=0.5):
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    selector.unregister(stream)
                    filters[stream].finish()
                    continue
                for line in filters[stream].feed(chunk):
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
        dropped = sum(f.dropped_lines for f in filters.values())
        if dropped:
            emit("child_output_dropped", count=dropped)
        phase = "shutdown"
        try:
            status = child.wait(timeout=SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            send_signal(child, KILL_SIGNAL)
            status = child.wait(timeout=SHUTDOWN_SECONDS)
        exit_code = status if status >= 0 else 128 + abs(status)
        emit("exit", status=min(exit_code, 255))
        return exit_code
    except Exception as exc:
        diagnostic(phase, type(exc))
        return 1
    finally:
        if child is not None:
            send_signal(child, signal.SIGTERM)
            try:
                child.wait(timeout=SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                send_signal(child, KILL_SIGNAL)
                child.wait(timeout=SHUTDOWN_SECONDS)
            # Also terminate any remaining group members on every error path.
            send_signal(child, KILL_SIGNAL)
            for stream in (child.stdout, child.stderr):
                stream.close()
        if selector is not None:
            selector.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
                      VLLM_LOGGING_LEVEL="WARNING", VLLM_REQUEST_BODY_LOG_LEVEL="CRITICAL",
                      VLLM_LOGGING_CONFIG_PATH="/opt/ml/app/logging.json")
    os.environ["ML_AUDIT_KEY"] = secrets.token_hex(32)
    if not os.environ.get("VLLM_API_KEY"):
        diagnostic("configuration", ValueError)
        emit("exit", status=2)
        return 2
    return supervise(["serve", "--model", os.environ.get("MODEL_DIR", "/models/model"),
                      "--config", os.environ.get("VLLM_CONFIG", "/config/serve.yaml"),
                      "--middleware", "app.privacy.AuditMiddleware", "--no-enable-log-requests",
                      "--no-enable-log-outputs", "--disable-uvicorn-access-log"])


if __name__ == "__main__":
    try:
        result = main()
    except BaseException as exc:
        diagnostic("supervision", type(exc))
        result = 1
    sys.exit(result)
