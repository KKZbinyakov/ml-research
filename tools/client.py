"""Send one authenticated request through the local port forward."""
import json
import os
from pathlib import Path
import sys
import urllib.request

root = Path(__file__).resolve().parents[1]
token = os.environ.get("API_TOKEN") or (root / ".secrets/api-token").read_text().strip()
request = urllib.request.Request(
    "http://127.0.0.1:18000/v1/chat/completions",
    json.dumps({"model": "qwen-cpu", "messages": [{"role": "user", "content":
        " ".join(sys.argv[1:]) or "Explain what a Kubernetes initContainer does."}],
        "max_tokens": 64, "temperature": 0}).encode(),
    {"Content-Type": "application/json", "Authorization": "Bearer " + token})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=180) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
