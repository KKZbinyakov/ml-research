"""Deploy the submission archive in a new kind cluster with a fresh PVC."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile

import local

ROOT = Path(__file__).resolve().parents[1]


def extract_archive(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            name = member.filename.replace("\\", "/")
            path = PurePosixPath(name)
            target = (destination / name).resolve()
            if (path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name)
                    or not target.is_relative_to(destination)
                    or {".secrets", ".tools", ".git"}.intersection(path.parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("unsafe_submission_archive")
        package.extractall(destination)
    project = destination / "ml_research"
    if not (project / "tools/local.py").is_file():
        raise ValueError("submission_project_missing")
    return project


def runtime_sources(project):
    paths = []
    for folder in ("app", "bench", "tools", "k8s"):
        paths.extend(p for p in (project / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".json")
                     and "__pycache__" not in p.parts
                     and (folder != "bench" or p.suffix == ".py"))
    # Results are evidence, not runtime source. Include only the app logging JSON.
    paths = [p for p in paths if p.suffix != ".json" or p == project / "app/logging.json"]
    paths += list(project.glob("Dockerfile*")) + list(project.glob("requirements*")) + [project / "kind.yaml"]
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cluster", default="ml-research-acceptance")
    parser.add_argument("--port", type=int, default=18002)
    args = parser.parse_args()
    local.validate_cluster(args.cluster)
    if args.cluster == local.DEFAULT_CLUSTER:
        parser.error("the acceptance cluster must differ from the working cluster")
    binary = local.kind_binary()
    clusters = local.run([binary, "get", "clusters"], capture_output=True, text=True).stdout.splitlines()
    if args.cluster in clusters:
        parser.error("acceptance cluster already exists; choose a new name")
    work = ROOT / ".tools" / ("cleanroom-" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True)
    project = extract_archive(args.archive, work)
    sources = runtime_sources(project)
    if sources != runtime_sources(ROOT):
        raise RuntimeError("archive_runtime_sources_differ_from_workspace")
    kubeconfig = project / ".tools" / f"kubeconfig-{args.cluster}"
    original_kube = ROOT / ".tools/kubeconfig"
    base = ["kubectl", "--kubeconfig", str(original_kube), "-n", "ml-research"]
    original = json.loads(subprocess.check_output(base + ["get", "deployment", "vllm", "-o", "json"], text=True))
    replicas = original["spec"].get("replicas", 1)
    record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "cluster_absent_before": True, "runtime_source_sha256": sources,
              "archive_sources_match_workspace": True, "cluster": args.cluster,
              "status": "running"}
    output = ROOT / "artifacts/cleanroom-verification.json"
    forward = None
    log = (work / "deployment.log").open("w", encoding="utf-8")
    try:
        print("cleanroom: pausing the working deployment to free RAM", flush=True)
        subprocess.run(base + ["scale", "deployment/vllm", "--replicas=0"], check=True, stdout=log, stderr=log)
        subprocess.run(base + ["wait", "--for=delete", "pod", "-l", "app=vllm", "--timeout=120s"],
                       check=True, stdout=log, stderr=log)
        print("cleanroom: starting the unpacked submission in a new cluster", flush=True)
        child_env = os.environ.copy()
        child_env.pop("API_TOKEN", None)
        subprocess.run([sys.executable, "tools/local.py", "up", "--cluster", args.cluster],
                       cwd=project, env=child_env, check=True, stdout=log, stderr=log)
        clean = ["kubectl", "--kubeconfig", str(kubeconfig), "-n", "ml-research"]
        pod = json.loads(subprocess.check_output(clean + ["get", "pods", "-l", "app=vllm", "-o", "json"], text=True))["items"][0]
        pvc = json.loads(subprocess.check_output(clean + ["get", "pvc", "model-cache", "-o", "json"], text=True))
        init_logs = subprocess.check_output(clean + ["logs", pod["metadata"]["name"], "-c", "download-model"], text=True)
        events = [json.loads(line) for line in init_logs.splitlines() if line.strip()]
        record["cold_download_observed"] = any(e.get("code") == "download_started" for e in events)
        record["download_completed"] = any(e.get("code") == "download_complete" for e in events)
        record["pvc_created_at"] = pvc["metadata"]["creationTimestamp"]
        record["pvc_volume"] = pvc["spec"]["volumeName"]
        if not record["cold_download_observed"] or not record["download_completed"]:
            raise RuntimeError("cold_download_not_proven")
        print("cleanroom: cold download succeeded; checking the authenticated API", flush=True)
        forward = subprocess.Popen(clean + ["port-forward", "service/vllm", f"{args.port}:8000", "--address=127.0.0.1"],
                                   stdout=log, stderr=log)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30
        while True:
            try:
                opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2).close()
                break
            except OSError:
                if time.monotonic() > deadline or forward.poll() is not None:
                    raise RuntimeError("cleanroom_port_forward_failed") from None
                time.sleep(.5)
        subprocess.run([sys.executable, "tools/local.py", "verify", "--cluster", args.cluster,
                        "--port", str(args.port)], cwd=project, env=child_env, check=True, stdout=log, stderr=log)
        verification = json.loads((project / "artifacts/verification.json").read_text(encoding="utf-8"))
        record["api_checks"] = verification["checks"]
        record["runtime"] = verification["runtime"]
        print("cleanroom: checking warm PVC with HF imports and Python sockets forbidden", flush=True)
        cache_env = dict(child_env, KUBECONFIG=str(kubeconfig))
        subprocess.run([sys.executable, "tools/verify_cache.py"], cwd=project, env=cache_env,
                       check=True, stdout=log, stderr=log)
        cache_proof = json.loads((project / "artifacts/cache-verification.json").read_text(encoding="utf-8"))
        record["offline_cache"] = cache_proof
        if cache_proof["init_exit_code"] != 0 or not cache_proof["marker_unchanged"]:
            raise RuntimeError("offline_cache_verification_failed")
        (ROOT / "artifacts/cache-verification.json").write_text(json.dumps(cache_proof, indent=2), encoding="utf-8")
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        if forward is not None:
            forward.terminate()
            try:
                forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                forward.kill()
                forward.wait()
        print("cleanroom: cleaning the temporary cluster and restoring the working deployment", flush=True)
        try:
            try:
                subprocess.run([sys.executable, "tools/local.py", "down", "--cluster", args.cluster], cwd=project,
                               check=True, stdout=log, stderr=log)
                record["temporary_cluster_removed"] = True
            except Exception as exc:
                record["status"] = "failed"
                record["cleanup_error_type"] = type(exc).__name__
                record["temporary_cluster_removed"] = False
                # Only this absent-before, newly created test cluster is eligible.
                subprocess.run([str(binary), "delete", "cluster", "--name", args.cluster],
                               check=True, stdout=log, stderr=log)
                record["temporary_cluster_removed"] = True
        finally:
            try:
                subprocess.run(base + ["scale", "deployment/vllm", f"--replicas={replicas}"], check=True, stdout=log, stderr=log)
                if replicas:
                    subprocess.run(base + ["rollout", "status", "deployment/vllm", "--timeout=15m"],
                                   check=True, stdout=log, stderr=log)
                record["working_deployment_restored"] = True
            except Exception as exc:
                record["status"] = "failed"
                record["working_deployment_restored"] = False
                record["restoration_error_type"] = type(exc).__name__
                raise
            finally:
                record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                log.close()
    if record["status"] != "passed":
        raise SystemExit(1)
    print(json.dumps({"status": record["status"], "api_checks": len(record["api_checks"]),
                      "cold_download_observed": record["cold_download_observed"]}), flush=True)


if __name__ == "__main__":
    main()
