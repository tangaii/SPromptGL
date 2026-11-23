#!/usr/bin/env python3
"""Verify or stream the frozen DPSL bundle on an authorized system."""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INVENTORY = "SHA256SUMS"


def verified_files():
    rows = (ROOT / INVENTORY).read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError("Empty DPSL SHA inventory")
    seen = set()
    for row in rows:
        digest, name = row.split("  ", 1)
        relative = Path(name)
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or name == INVENTORY
                or path in seen or path.resolve() != path or not path.is_file()):
            raise ValueError(f"Invalid frozen file path {name}")
        seen.add(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {name}")
        yield path, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "pack"))
    args = parser.parse_args()
    files = list(verified_files())
    if args.action == "verify":
        print(f"PASS {len(files)} frozen files")
        return
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as bundle:
        for path, name in [*files, (ROOT / INVENTORY, INVENTORY)]:
            info = bundle.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                bundle.addfile(info, stream)


if __name__ == "__main__":
    main()
