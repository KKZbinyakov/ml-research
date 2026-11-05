"""Exercise the deployed endpoint; record only safe verification metadata."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("privacy", ROOT / "app/privacy.py")
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def kube(*args):
    return subprocess.run(["kubectl", "-n", "ml-research", *args], check=True,
                          capture_output=True, text=True).stdout


def request(base, path, token=None, payload=None, headers=None):
    hdr = {"Content-Type": "application/json", **(headers or {})}
    if token is not None:
        hdr["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers=hdr)
    try:
        with opener.open(req, timeout=180) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", default="artifacts/verification.json")
    args = parser.parse_args()
    token = os.environ["API_TOKEN"]
    private_prompt = "PRIVATE_PROMPT_" + uuid.uuid4().hex
    private_query = "PRIVATE_QUERY_" + uuid.uuid4().hex
    private_header = "PRIVATE_HEADER_" + uuid.uuid4().hex
    base = args.base_url.rstrip("/")
    checks = {}
    checks["health"] = request(base, "/health")[0] == 200
    checks["unauthenticated_401"] = request(base, "/v1/models")[0] == 401
    checks["wrong_bearer_401"] = request(base, "/v1/models", "wrong-token")[0] == 401
    checks["authenticated_200"] = request(base, "/v1/models", token)[0] == 200
    for path in ("/invocations", "/generative_scoring", "/pause", "/abort_requests"):
        checks[path.strip("/") + "_unauthenticated_401"] = request(base, path, payload={})[0] == 401
    payload = {"model": "qwen-cpu", "messages": [{"role": "user", "content":
        f"Say the word pineapple. This is a privacy test {private_prompt}."}],
        "max_tokens": 16, "temperature": 0}
    status, body = request(base, "/v1/chat/completions?private=" + private_query,
                           token, payload, {"X-Request-Id": private_header})
    checks["inference_200"] = status == 200
    output = json.loads(body)["choices"][0]["message"]["content"]
    payload["stream"] = True
    status, stream = request(base, "/v1/chat/completions", token, payload)
    checks["stream_200_done"] = status == 200 and b"data: [DONE]" in stream
    streamed_output = ""
    for line in stream.splitlines():
        if line.startswith(b"data: ") and line != b"data: [DONE]":
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                streamed_output += choices[0].get("delta", {}).get("content") or ""
    payload["model"] = private_prompt
    checks["invalid_model_404"] = request(base, "/v1/chat/completions", token, payload)[0] == 404
    logs = kube("logs", "deployment/vllm", "-c", "vllm")
    checks["no_prompt_response_header_query_or_token_in_logs"] = all(
        text not in logs for text in (private_prompt, private_query, private_header, token))
    if output.strip():
        checks["no_generated_response_in_logs"] = output.strip() not in logs
    if streamed_output.strip():
        checks["no_streamed_response_in_logs"] = streamed_output.strip() not in logs
    records = [json.loads(line) for line in logs.splitlines() if line.strip()]
    checks["logs_match_allowlist"] = all(privacy.safe_output_line(json.dumps(r)) for r in records)
    checks["request_fact_logged"] = any(r.get("event") == "request_finished"
                                         and r.get("route") == "/v1/chat/completions" for r in records)
    runtime_code = (
        "import os,json,torch,importlib.metadata as m;from vllm.platforms import current_platform;"
        "print(json.dumps({'uid':os.getuid(),'vllm':m.version('vllm'),"
        "'torch':torch.__version__,'cuda':torch.version.cuda,'device':current_platform.device_type,"
        "'hf_token_absent':not bool(os.getenv('HF_TOKEN')),'offline':os.getenv('HF_HUB_OFFLINE')}))")
    runtime = json.loads(kube("exec", "deployment/vllm", "-c", "vllm", "--", "python", "-c", runtime_code))
    checks["rootless_cpu_offline_no_hf_token"] = (
        runtime["uid"] == 10001 and runtime["device"] == "cpu" and runtime["cuda"] is None
        and runtime["hf_token_absent"] and runtime["offline"] == "1")
    document = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "checks": checks, "runtime": runtime, "log_records_checked": len(records)}
    destination = ROOT / args.output
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
