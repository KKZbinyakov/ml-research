"""Restart the actual Pod with networking disabled in the Python init process."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def kube(*args, data=None):
    config = os.environ.get("KUBECONFIG", str(ROOT / ".tools/kubeconfig"))
    return subprocess.run(["kubectl", "--kubeconfig", config, "-n", "ml-research", *args], input=data,
                          capture_output=True, text=True, check=True).stdout


def marker_state():
    code = ("from pathlib import Path;import hashlib,json;"
            "p=Path('/models/model/.download-complete.json');"
            "print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),"
            "'mtime_ns':p.stat().st_mtime_ns}))")
    return json.loads(kube("exec", "deployment/vllm", "-c", "vllm", "--", "python", "-c", code))


def main():
    before = marker_state()
    # A warm init must return before importing HF; any attempted Python network
    # connection or HF SDK import makes this replacement initContainer fail.
    offline_command = """
import builtins, runpy, socket
original_import = builtins.__import__
def no_network(*args, **kwargs):
    raise RuntimeError('network_disabled_for_cache_verification')
def guarded_import(name, *args, **kwargs):
    if name == 'huggingface_hub' or name.startswith('huggingface_hub.'):
        raise RuntimeError('hf_import_forbidden_on_warm_start')
    return original_import(name, *args, **kwargs)
socket.socket = no_network
socket.create_connection = no_network
builtins.__import__ = guarded_import
runpy.run_path('/opt/ml/app/download_model.py', run_name='__main__')
"""
    patch = {"spec": {"template": {"spec": {"initContainers": [{
        "name": "download-model", "command": ["python", "-c", offline_command],
        "env": [{"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "HF_ENDPOINT", "value": "http://127.0.0.1:9"}]}]}}}}
    started = time.perf_counter()
    # Remove only test-only fields and restore the standard command in one
    # rollout. The restored init also reuses PVC.
    restore = {"spec": {"template": {"spec": {"initContainers": [{
        "name": "download-model", "command": ["python", "/opt/ml/app/download_model.py"],
        "env": [{"name": "HF_HUB_OFFLINE", "$patch": "delete"},
                {"name": "HF_ENDPOINT", "$patch": "delete"}]}]}}}}
    try:
        kube("patch", "deployment/vllm", "--type=strategic", "--patch", json.dumps(patch))
        kube("rollout", "status", "deployment/vllm", "--timeout=15m")
        after = marker_state()
        pods = json.loads(kube("get", "pods", "-l", "app=vllm", "-o", "json"))
        pod = next(p for p in pods["items"] if not p["metadata"].get("deletionTimestamp"))
        init = pod["status"]["initContainerStatuses"][0]["state"]["terminated"]
        record = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "restart_wall_seconds": time.perf_counter() - started,
                  "init_exit_code": init["exitCode"], "init_started_at": init["startedAt"],
                  "init_finished_at": init["finishedAt"], "marker_unchanged": before == after,
                  "marker_sha256": after["sha256"], "hf_import_forbidden": True,
                  "python_socket_disabled": True, "hf_offline": True, "invalid_hf_endpoint": True}
        (ROOT / "artifacts/cache-verification.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)
        if not record["marker_unchanged"] or record["init_exit_code"] != 0:
            raise SystemExit(1)
    finally:
        kube("patch", "deployment/vllm", "--type=strategic", "--patch", json.dumps(restore))
        kube("rollout", "status", "deployment/vllm", "--timeout=15m")


if __name__ == "__main__":
    main()
