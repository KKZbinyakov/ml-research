#!/usr/bin/env python3
"""Record benchmark host and Kubernetes configuration without secret values."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check).stdout.strip()


def sha256_files():
    paths = list((ROOT / "app").glob("*.py")) + list((ROOT / "bench").glob("*.py")) + [ROOT / "app/logging.json"]
    paths += list(ROOT.glob("Dockerfile*")) + list(ROOT.glob("requirements*"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file()}


def parse_yaml_config(text):
    return {key: value.strip().strip('"\'') for key, value in
            re.findall(r"^\s*(max-num-seqs|max-num-batched-tokens|enable-chunked-prefill|no-enable-prefix-caching):\s*([^#\n]+)", text, re.M)}


def record(kubeconfig):
    docker = {}
    try:
        info = json.loads(command(["docker", "info", "--format", "{{json .}}"] ))
        docker = {"cpus": info.get("NCPU"), "memory_bytes": info.get("MemTotal"),
                  "architecture": info.get("Architecture"), "kernel": info.get("KernelVersion"),
                  "server_version": info.get("ServerVersion")}
        ver = str(info.get("CgroupVersion", ""))
        docker["cgroup_version"] = ver or None
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        docker = {"error": "docker_info_unavailable"}
    cpu = []
    try:
        cpu = [{"field": line.split(":", 1)[0], "data": line.split(":", 1)[1].strip()}
               for line in command(["lscpu"]).splitlines() if ":" in line]
    except (OSError, subprocess.CalledProcessError):
        pass
    images = []
    for name in ("ml-research-vllm:0.27.0", "ml-research-init:0.27.0"):
        try:
            image = json.loads(command(["docker", "image", "inspect", name]))[0]
            images.append({"name": name, "id": image.get("Id"), "size_bytes": image.get("Size"),
                           "user": image.get("Config", {}).get("User")})
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
            images.append({"name": name, "error": "image_unavailable"})
    base = ["kubectl", "--kubeconfig", str(kubeconfig), "-n", "ml-research"]
    try:
        cm = json.loads(command(base + ["get", "configmap", "vllm-config", "-o", "json"]))
        config = cm.get("data", {})
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError("configmap_unavailable") from exc
    try:
        deployment = json.loads(command(base + ["get", "deployment", "vllm", "-o", "json"]))
        revision = deployment.get("metadata", {}).get("annotations", {}).get("deployment.kubernetes.io/revision")
        items = json.loads(command(base + ["get", "pods", "-l", "app=vllm", "-o", "json"])).get("items", [])
        live = [p for p in items if not p.get("metadata", {}).get("deletionTimestamp")]
        if len(live) != 1:
            raise RuntimeError("unexpected_live_pod_count")
        current = live[0]
        spec = current.get("spec", {})
        pod_status = current.get("status", {})
        owner = next((o for o in current.get("metadata", {}).get("ownerReferences", [])
                      if o.get("kind") == "ReplicaSet"), None)
        if not revision or not owner:
            raise RuntimeError("deployment_pod_revision_missing")
        rs = json.loads(command(base + ["get", "replicaset", owner["name"], "-o", "json"]))
        rs_revision = rs.get("metadata", {}).get("annotations", {}).get("deployment.kubernetes.io/revision")
        if rs_revision != revision:
            raise RuntimeError("pod_revision_mismatch")
        if pod_status.get("phase") != "Running" or not any(
                c.get("type") == "Ready" and c.get("status") == "True" for c in pod_status.get("conditions", [])):
            raise RuntimeError("pod_not_ready")
        statuses = []
        for status in pod_status.get("containerStatuses", []):
            state = status.get("state", {})
            running = state.get("running", {})
            terminated = state.get("terminated", {})
            statuses.append({"name": status.get("name"), "ready": status.get("ready"),
                             "restart_count": status.get("restartCount"),
                             "image": status.get("image"), "image_id": status.get("imageID"),
                             "state": "running" if running else (terminated.get("reason") or "waiting"),
                             "exit_code": terminated.get("exitCode")})
        container = next((c for c in spec.get("containers", []) if c.get("name") == "vllm"), {})
        if not statuses or not all(c["ready"] for c in statuses):
            raise RuntimeError("container_not_ready")
        pod = {"name": current.get("metadata", {}).get("name"), "uid": current.get("metadata", {}).get("uid"),
               "replicaset": owner["name"], "revision": revision, "phase": pod_status.get("phase"),
               "conditions": [{"type": c.get("type"), "status": c.get("status")} for c in pod_status.get("conditions", [])],
               "containers": statuses, "resources": container.get("resources"),
               "security_context": container.get("securityContext"),
               "pod_security_context": spec.get("securityContext")}
        if not cpu and spec.get("nodeName"):
            try:
                cpu = [{"field": line.split(":", 1)[0], "data": line.split(":", 1)[1].strip()}
                       for line in command(["docker", "exec", spec["nodeName"], "lscpu"]).splitlines()
                       if ":" in line]
            except (OSError, subprocess.CalledProcessError):
                pass
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, StopIteration, KeyError) as exc:
        raise RuntimeError("pod_state_unavailable") from exc
    return {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "docker": docker,
            "cpu": {"lscpu": cpu}, "images": images, "config": config,
            "serve_config": parse_yaml_config(config.get("serve.yaml", "")), "pod": pod,
            "source_sha256": sha256_files()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", default=os.environ.get("KUBECONFIG", str(ROOT / ".tools/kubeconfig")))
    parser.add_argument("--output", default=str(ROOT / "artifacts/environment.json"))
    args = parser.parse_args()
    result = record(args.kubeconfig)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
