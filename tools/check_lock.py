"""Fail a build if any installed distribution differs from the complete lock."""
import importlib.metadata
import re
import sys
from pathlib import Path


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check(path):
    expected = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, version = line.split("==")
        expected[normalize(name)] = version
    actual = {normalize(d.metadata["Name"]): d.version
              for d in importlib.metadata.distributions()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise SystemExit(f"Dependency lock mismatch: missing={missing}, extra={extra}, changed={changed}")
    print(f"Verified complete lock: {len(expected)} distributions")


if __name__ == "__main__":
    check(sys.argv[1])
