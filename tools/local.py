"""Local kind workflow. Run from any directory with Python 3.12+ and Docker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".secrets"
KIND_VERSION = "v0.33.0"
DEFAULT_CLUSTER = "ml-research"


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def _kind_version(binary):
    try:
        output = subprocess.run([str(binary), "version"], capture_output=True, text=True,
                                check=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bkind\s+(v\S+)", output)
    return match.group(1) if match else None


def kind_binary():
    installed = shutil.which("kind")
    if installed and _kind_version(installed) == KIND_VERSION:
        return installed
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}[platform.system()]
    arch = {"AMD64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    destination = ROOT / ".tools" / ("kind.exe" if os_name == "windows" else "kind")
    if destination.exists() and _kind_version(destination) == KIND_VERSION:
        return destination
    destination.parent.mkdir(exist_ok=True)
    url = f"https://kind.sigs.k8s.io/dl/{KIND_VERSION}/kind-{os_name}-{arch}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with urllib.request.urlopen(url + ".sha256sum", timeout=30) as response:
        expected = response.read().decode().split()[0]
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("kind checksum mismatch")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.chmod(0o755)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if _kind_version(destination) != KIND_VERSION:
        destination.unlink(missing_ok=True)
        raise RuntimeError("downloaded kind version mismatch")
    return destination


def kube(kubeconfig, *args, **kwargs):
    return run(["kubectl", "--kubeconfig", kubeconfig, "-n", "ml-research", *args], **kwargs)


def client_env(token_file):
    env = os.environ.copy()
    env["API_TOKEN"] = read_token(token_file)
    return env


def kubeconfig_path(cluster, explicit=None):
    if explicit:
        return Path(explicit).expanduser().resolve()
    suffix = "" if cluster == DEFAULT_CLUSTER else f"-{cluster}"
    return ROOT / ".tools" / f"kubeconfig{suffix}"


def token_path(cluster):
    suffix = "" if cluster == DEFAULT_CLUSTER else f"-{cluster}"
    return PRIVATE / f"api-token{suffix}"


def read_token(token_file):
    token = token_file.read_text().strip()
    if not token:
        raise ValueError("API token must not be empty or whitespace")
    return token


def validate_cluster(cluster):
    if not re.fullmatch(r"ml-research(?:-[a-z0-9][a-z0-9-]{0,48})?", cluster):
        raise ValueError("cluster name must be 'ml-research' or start with 'ml-research-'")


def up(cluster=DEFAULT_CLUSTER, kubeconfig=None):
    validate_cluster(cluster)
    kubeconfig = kubeconfig_path(cluster, kubeconfig)
    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".tools").mkdir(exist_ok=True)
    token_file = token_path(cluster)
    binary = kind_binary()
    clusters = run([binary, "get", "clusters"], capture_output=True, text=True).stdout.splitlines()
    if cluster not in clusters:
        config = (ROOT / "kind.yaml").read_text()
        config = re.sub(r"(?m)^name:\s*.*$", f"name: {cluster}", config, count=1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8",
                                         dir=ROOT / ".tools", delete=False) as handle:
            handle.write(config)
            config_path = Path(handle.name)
        try:
            run([binary, "create", "cluster", "--name", cluster, "--config", config_path,
                 "--kubeconfig", kubeconfig, "--wait", "120s"])
        finally:
            config_path.unlink(missing_ok=True)
    else:
        run([binary, "export", "kubeconfig", "--name", cluster, "--kubeconfig", kubeconfig])
    run(["docker", "build", "-t", "ml-research-vllm:0.27.0", "-f", "Dockerfile", "."])
    run(["docker", "build", "-t", "ml-research-init:0.27.0", "-f", "Dockerfile.init", "."])
    run([binary, "load", "docker-image", "--name", cluster,
         "ml-research-vllm:0.27.0", "ml-research-init:0.27.0"])
    PRIVATE.mkdir(exist_ok=True)
    PRIVATE.chmod(0o700)
    if not token_file.exists():
        supplied_token = os.environ.get("API_TOKEN")
        if supplied_token is not None and not supplied_token.strip():
            raise ValueError("API_TOKEN must not be empty or whitespace")
        token = supplied_token.strip() if supplied_token is not None else secrets.token_urlsafe(32)
        token_file.write_text(token)
        token_file.chmod(0o600)
    read_token(token_file)
    kube(kubeconfig, "apply", "-f", "k8s/base/namespace.yaml")
    secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {
        "name": "ml-research-secrets", "namespace": "ml-research"}, "type": "Opaque",
        "stringData": {"API_TOKEN": read_token(token_file),
                       "HF_TOKEN": os.environ.get("HF_TOKEN", "")}}
    # stdin keeps token values out of argv and terminal output.
    kube(kubeconfig, "apply", "-f", "-", input=json.dumps(secret), text=True)
    kube(kubeconfig, "apply", "-k", "k8s")
    kube(kubeconfig, "rollout", "restart", "deployment/vllm")
    kube(kubeconfig, "rollout", "status", "deployment/vllm", "--timeout=20m")


def down(cluster=DEFAULT_CLUSTER, kubeconfig=None):
    validate_cluster(cluster)
    run([kind_binary(), "delete", "cluster", "--name", cluster])
    path = kubeconfig_path(cluster, kubeconfig)
    expected = ROOT / ".tools" / ("kubeconfig" if cluster == DEFAULT_CLUSTER else f"kubeconfig-{cluster}")
    managed_parent = expected.parent.resolve()
    if (managed_parent.is_relative_to(ROOT.resolve()) and path.parent.resolve() == managed_parent
            and path.name == expected.name and (path.exists() or path.is_symlink())):
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("up", "down", "status", "forward", "benchmark", "verify"))
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--port", type=int, default=18000)
    args, rest = parser.parse_known_args()
    validate_cluster(args.cluster)
    kubeconfig = kubeconfig_path(args.cluster, args.kubeconfig)
    if args.action == "up":
        up(args.cluster, kubeconfig)
    elif args.action == "down":
        down(args.cluster, kubeconfig)
    elif args.action == "status":
        kube(kubeconfig, "get", "pods,pvc,svc")
    elif args.action == "forward":
        kube(kubeconfig, "port-forward", "service/vllm", f"{args.port}:8000", "--address=127.0.0.1")
    else:
        script = "bench/run.py" if args.action == "benchmark" else "tools/verify_runtime.py"
        env = client_env(token_path(args.cluster))
        env["KUBECONFIG"] = str(kubeconfig)
        run([sys.executable, script, "--base-url", f"http://127.0.0.1:{args.port}", *rest], env=env)


if __name__ == "__main__":
    main()
