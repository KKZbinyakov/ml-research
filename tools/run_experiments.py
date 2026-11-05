#!/usr/bin/env python3
"""Run serial, counterbalanced short/long benchmark conditions on the local cluster."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import record_environment

CONDITIONS = {
    "short-seqs4": {"max-num-seqs": 4, "max-num-batched-tokens": 2048, "workload": "short", "concurrency": 8},
    "short-seqs8": {"max-num-seqs": 8, "max-num-batched-tokens": 2048, "workload": "short", "concurrency": 8},
    "long-batch2048": {"max-num-seqs": 4, "max-num-batched-tokens": 2048, "workload": "long", "concurrency": 2},
    "long-batch4096": {"max-num-seqs": 4, "max-num-batched-tokens": 4096, "workload": "long", "concurrency": 2},
}


def condition_order(repeats):
    return [(repeat, label) for repeat in range(1, repeats + 1)
            for label in (list(CONDITIONS) if repeat % 2 else list(reversed(CONDITIONS)))]


def update_serve_yaml(text, values):
    for key in ("max-num-seqs", "max-num-batched-tokens"):
        pattern = rf"(?m)^(\s*{re.escape(key)}:\s*)[^#\n]+"
        text, count = re.subn(pattern, lambda m: m.group(1) + str(values[key]), text)
        if count != 1:
            raise ValueError("serve_config_key_missing_or_duplicated")
    return text


def metric_counter(text, name):
    values = []
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except ValueError:
                continue
    return sum(values) if values else None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def portable_path(path):
    return Path(os.path.relpath(path, ROOT)).as_posix()


class Experiment:
    def __init__(self, kubeconfig, port, timeout, token):
        self.kubeconfig = str(kubeconfig)
        self.port = port
        self.timeout = timeout
        self.token = token
        self.base = ["kubectl", "--kubeconfig", self.kubeconfig, "-n", "ml-research"]
        self.forward = None

    def kube(self, *args, data=None):
        return subprocess.run(self.base + list(args), input=data, capture_output=True, text=True, check=True).stdout

    def get_config(self):
        return json.loads(self.kube("get", "configmap", "vllm-config", "-o", "json"))

    def set_serve_yaml(self, text):
        patch = json.dumps({"data": {"serve.yaml": text}})
        self.kube("patch", "configmap", "vllm-config", "--type=merge", "--patch", patch)
        self.kube("rollout", "restart", "deployment/vllm")
        self.kube("rollout", "status", "deployment/vllm", f"--timeout={self.timeout}s")

    def validate_current_pod(self, expected_config):
        deployment = json.loads(self.kube("get", "deployment", "vllm", "-o", "json"))
        revision = deployment.get("metadata", {}).get("annotations", {}).get("deployment.kubernetes.io/revision")
        if not revision:
            raise RuntimeError("deployment_revision_missing")
        items = json.loads(self.kube("get", "pods", "-l", "app=vllm", "-o", "json")).get("items", [])
        live = [p for p in items if not p.get("metadata", {}).get("deletionTimestamp")]
        if len(live) != 1:
            raise RuntimeError("unexpected_live_pod_count")
        pod = live[0]
        owners = pod.get("metadata", {}).get("ownerReferences", [])
        rs = next((o for o in owners if o.get("kind") == "ReplicaSet"), None)
        if not rs:
            raise RuntimeError("pod_replicaset_missing")
        rs_doc = json.loads(self.kube("get", "replicaset", rs["name"], "-o", "json"))
        rs_revision = rs_doc.get("metadata", {}).get("annotations", {}).get("deployment.kubernetes.io/revision")
        statuses = pod.get("status", {}).get("containerStatuses", [])
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in pod.get("status", {}).get("conditions", []))
        if rs_revision != revision or not ready or not statuses or not all(c.get("ready") for c in statuses):
            raise RuntimeError("current_pod_not_ready_at_revision")
        uid = pod.get("metadata", {}).get("uid")
        if not uid:
            raise RuntimeError("pod_uid_missing")
        mounted = self.kube("exec", pod["metadata"]["name"], "-c", "vllm", "--", "cat", "/config/serve.yaml")
        if mounted != expected_config:
            raise RuntimeError("mounted_config_mismatch")
        return {"name": pod["metadata"]["name"], "uid": uid,
                "replicaset": rs["name"], "revision": revision}

    def start_forward(self):
        self.stop_forward()
        self.forward = subprocess.Popen(["kubectl", "--kubeconfig", self.kubeconfig, "-n", "ml-research",
                                         "port-forward", "--address", "127.0.0.1", "service/vllm",
                                         f"{self.port}:8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.forward.poll() is not None:
                raise RuntimeError("port_forward_failed")
            try:
                opener.open(f"http://127.0.0.1:{self.port}/health", timeout=2).close()
                return
            except Exception:
                time.sleep(.5)
        raise RuntimeError("server_health_timeout")

    def stop_forward(self):
        if self.forward:
            self.forward.terminate()
            try:
                self.forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.forward.kill()
                self.forward.wait(timeout=5)
            self.forward = None

    def metrics(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/metrics",
                                         headers={"Authorization": f"Bearer {self.token}"})
        with opener.open(request, timeout=10) as response:
            return response.read().decode("utf-8", "replace")


def token_from_file(token_file=None):
    if os.environ.get("API_TOKEN"):
        return os.environ["API_TOKEN"]
    path = Path(token_file) if token_file else ROOT / ".secrets/api-token"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def run(args):
    token = token_from_file(args.token_file)
    if not token:
        raise RuntimeError("api_token_missing")
    exp = Experiment(args.kubeconfig, args.port, args.rollout_timeout, token)
    original = exp.get_config()
    original_text = original.get("data", {}).get("serve.yaml", "")
    if not original_text:
        raise RuntimeError("serve_config_missing")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    progress_path = output_dir / "manifest.json"
    manifest = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "requests": args.requests,
                "repeats": args.repeats, "conditions": [], "status": "running", "phase": "starting"}
    atomic_json(progress_path, manifest)
    config_changed = False
    try:
        for repeat, label in condition_order(args.repeats):
            condition = CONDITIONS[label]
            desired = update_serve_yaml(original_text, condition)
            exp.stop_forward()
            config_changed = True
            manifest["phase"] = "condition_rollout"
            atomic_json(progress_path, manifest)
            exp.set_serve_yaml(desired)
            pod_identity = exp.validate_current_pod(desired)
            exp.start_forward()
            before = metric_counter(exp.metrics(), "vllm:prefix_cache_hits_total")
            environment_path = output_dir / f"environment-{repeat}-{label}.json"
            env_doc = record_environment.record(args.kubeconfig)
            if env_doc.get("pod", {}).get("uid") != pod_identity["uid"]:
                raise RuntimeError("environment_pod_identity_changed")
            atomic_json(environment_path, env_doc)
            result_path = output_dir / f"repeat-{repeat}-{label}.json"
            command = [sys.executable, str(ROOT / "bench/run.py"), "--base-url", f"http://127.0.0.1:{args.port}",
                       "--requests", str(args.requests), "--repeats", "1", "--workload", condition["workload"],
                       "--concurrency", str(condition["concurrency"]), "--label", label,
                       "--max-tokens", "32", "--output", str(result_path), "--environment", str(environment_path)]
            child_env = dict(os.environ, API_TOKEN=token)
            subprocess.run(command, cwd=ROOT, env=child_env, check=True)
            with result_path.open(encoding="utf-8") as f:
                result_doc = json.load(f)
            result_doc["experiment_repeat"] = repeat
            atomic_json(result_path, result_doc)
            after = metric_counter(exp.metrics(), "vllm:prefix_cache_hits_total")
            if before is None or after is None:
                raise RuntimeError("prefix_cache_metric_missing")
            if after != 0:
                raise RuntimeError("prefix_cache_hit_detected")
            entry = {"repeat": repeat, "label": label, "result": portable_path(result_path),
                     "environment": portable_path(environment_path), "prefix_cache_hits_before": before,
                     "prefix_cache_hits_after": after, "pod": pod_identity}
            manifest["conditions"].append(entry)
            atomic_json(progress_path, manifest)
            print(f"completed repeat={repeat} condition={label}", flush=True)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = type(exc).__name__
        raise
    finally:
        exp.stop_forward()
        if config_changed:
            try:
                exp.set_serve_yaml(original_text)
            except Exception as exc:
                manifest["restored_original_config"] = False
                manifest["restore_error"] = type(exc).__name__
                manifest["status"] = "failed"
                manifest["phase"] = "restoration_failed"
                manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(progress_path, manifest)
                raise
            else:
                manifest["restored_original_config"] = True
                if manifest.get("status") == "running":
                    manifest["status"] = "complete"
                manifest["phase"] = "finished"
                manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(progress_path, manifest)
        else:
            if manifest.get("status") == "running":
                manifest["status"] = "complete"
            manifest["phase"] = "finished"
            manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(progress_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--kubeconfig", default=os.environ.get("KUBECONFIG", str(ROOT / ".tools/kubeconfig")))
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--rollout-timeout", type=int, default=900)
    parser.add_argument("--output-dir", default=str(ROOT / "bench/experiments"))
    parser.add_argument("--token-file", help="API token file (defaults to .secrets/api-token)")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.requests < 1 or args.repeats < 1 or not (1 <= args.port <= 65535):
        parser.error("requests, repeats, and port must be positive")
    if args.plan_only:
        print(json.dumps([{"repeat": r, "label": label, **CONDITIONS[label]} for r, label in condition_order(args.repeats)], indent=2))
        return
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        print(f"experiment_failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
