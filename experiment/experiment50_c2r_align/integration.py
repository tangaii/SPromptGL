"""Static integration checks for Experiment 50.

This file intentionally contains no alternate model implementation.  It
documents the immutable B0-B boundary and gives an auditable, dependency-free
entry point for checking the source snapshot and model parameter counts.
"""
from pathlib import Path
import hashlib
import json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(path="manifest.json"):
    """Return whether the recorded source/config hashes are still present."""
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / path).read_text())
    checks = {"config": sha256(root / "config.json") == manifest["config_sha"]}
    for name, digest in manifest["code_sha"].items():
        checks[name] = sha256(root / name) == digest
    return all(checks.values()), checks


if __name__ == "__main__":
    ok, checks = verify_manifest()
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)
