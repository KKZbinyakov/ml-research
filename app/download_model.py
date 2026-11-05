#!/usr/bin/env python3
"""Fetch one immutable Hugging Face snapshot into the shared model volume."""
import json
import hashlib
import logging
import os
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/model"))
MARKER = ".download-complete.json"
STAGING = MODEL_DIR.with_name(MODEL_DIR.name + ".staging")
BACKUP = MODEL_DIR.with_name(MODEL_DIR.name + ".previous")
CACHE_STAGE = "validate"


def snapshot_files(path: Path):
    if path.is_symlink():
        return None
    files = {}
    hashes = {}
    for p in path.rglob("*"):
        rel = p.relative_to(path)
        if p.is_symlink():
            return None
        if not p.is_file() or p.name == MARKER or ".cache" in rel.parts:
            continue
        name = rel.as_posix()
        files[name] = p.stat().st_size
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if not all(name in files and files[name] > 0 for name in required):
        return None
    try:
        for name in required:
            json.loads((path / name).read_text(encoding="utf-8"))
        shards = [name for name in files if name.endswith(".safetensors")]
        if not shards:
            return None
        for name in shards:
            with (path / name).open("rb") as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    return None
                header_size = int.from_bytes(prefix, "little")
                if not 0 < header_size <= files[name] - 8 or type(json.loads(f.read(header_size))) is not dict:
                    return None
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            expected_shards = set(weight_map.values())
            if not expected_shards or any(name not in files for name in expected_shards):
                return None
        return files, hashes
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def valid(path: Path) -> bool:
    try:
        marker = json.loads((path / MARKER).read_text(encoding="utf-8"))
        if type(marker) is not dict:
            return False
        if marker.get("model_id") != MODEL_ID or marker.get("revision") != MODEL_REVISION:
            return False
        sizes = marker.get("files")
        snapshot = snapshot_files(path)
        if type(sizes) is not dict or not sizes or snapshot is None or snapshot[0] != sizes:
            return False
        hashes = marker.get("sha256")
        if type(hashes) is not dict or hashes != snapshot[1]:
            return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def diagnostic(stage: str, code: str) -> None:
    print(json.dumps({"event": "model_cache", "stage": stage, "code": code}, separators=(",", ":")))


def safe_sibling(path: Path) -> bool:
    return path.parent == MODEL_DIR.parent and path.name in (MODEL_DIR.name, STAGING.name, BACKUP.name)


def remove_sibling(path: Path) -> None:
    if not safe_sibling(path):
        raise ValueError("invalid cache sibling")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def main() -> int:
    global CACHE_STAGE
    CACHE_STAGE = "validate"
    if not re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION):
        diagnostic(CACHE_STAGE, "invalid_configuration")
        return 1
    if not MODEL_ID or not MODEL_DIR.is_absolute() or MODEL_DIR == MODEL_DIR.parent:
        diagnostic(CACHE_STAGE, "invalid_configuration")
        return 1
    if valid(MODEL_DIR):
        diagnostic("ready", "local_ready")
        return 0
    if valid(STAGING):
        CACHE_STAGE = "recover"
        if MODEL_DIR.exists():
            if valid(BACKUP):
                remove_sibling(MODEL_DIR)
            else:
                remove_sibling(BACKUP)
                MODEL_DIR.replace(BACKUP)
        try:
            STAGING.replace(MODEL_DIR)
        except BaseException:
            if BACKUP.exists() and not MODEL_DIR.exists():
                BACKUP.replace(MODEL_DIR)
            raise
        remove_sibling(BACKUP)
        diagnostic("ready", "staging_recovered")
        return 0
    if valid(BACKUP):
        CACHE_STAGE = "recover"
        remove_sibling(MODEL_DIR) if MODEL_DIR.exists() else None
        BACKUP.replace(MODEL_DIR)
        diagnostic("ready", "backup_recovered")
        return 0
    token = (os.environ.get("HF_TOKEN") or "").strip() or None

    CACHE_STAGE = "download"
    remove_sibling(STAGING)
    STAGING.mkdir(parents=True)
    diagnostic("download", "download_started")
    # Delayed import is intentional: a warm PVC path must never initialize HF networking.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    patterns = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "model.safetensors.index.json", "*.safetensors",
                "tokenizer.model", "merges.txt", "vocab.json"]
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        from huggingface_hub import snapshot_download
        logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(STAGING),
                          token=token, allow_patterns=patterns)
    snapshot = snapshot_files(STAGING)
    if snapshot is None:
        diagnostic("validate", "validation_failed")
        return 1
    files, hashes = snapshot
    (STAGING / MARKER).write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION,
                                             "files": files, "sha256": hashes}, separators=(",", ":")), encoding="utf-8")
    CACHE_STAGE = "commit"
    remove_sibling(BACKUP)
    if MODEL_DIR.exists():
        MODEL_DIR.replace(BACKUP)
    try:
        STAGING.replace(MODEL_DIR)
    except BaseException:
        if BACKUP.exists() and not MODEL_DIR.exists():
            BACKUP.replace(MODEL_DIR)
        raise
    remove_sibling(BACKUP)
    diagnostic("ready", "download_complete")
    return 0


if __name__ == "__main__":
    try:
        result = main()
    except BaseException as exc:
        # Never expose exception strings: HF exceptions can contain URLs or credentials.
        code = "network_failed" if exc.__class__.__module__.startswith("huggingface_hub") else "io_failed"
        diagnostic(CACHE_STAGE, code)
        sys.exit(1)
    sys.exit(result)
