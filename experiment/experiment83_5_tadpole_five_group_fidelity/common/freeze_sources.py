"""Create the startup source manifest for the sealed E83.5 release."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for base_name in ("sources", "common", "inputs"):
        base = ROOT / base_name
        for path in sorted(item for item in base.rglob("*") if item.is_file()
                           and "__pycache__" not in item.parts and item.suffix != ".pyc"):
            relative = path.relative_to(ROOT).as_posix()
            if relative == "sources/source_manifest.json":
                continue
            entries.append({"path": relative, "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
    manifest = {
        "experiment": "Experiment 83.5",
        "status": "FROZEN_AT_PREFLIGHT_START",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "actual_root": str(ROOT),
        "project_repository": "https://github.com/tangaii/SPromptGL",
        "project_commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
        "author_repository": "https://github.com/wanxixi11/SPromptGL",
        "author_commit": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
        "historical_sources": {
            "E83.4": {"commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
                      "working_root": "/model/spromptgl/experiment/experiment83_4_tadpole_four_group_complete_evidence"},
            "E43.4": {"commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
                      "path": "sources/E43_4_original",
                      "checkpoint_artifacts_in_GitHub": False},
            "E43.2": {"commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
                      "path": "sources/E43_2_original",
                      "untrained_anchor_artifacts_in_GitHub": False},
        },
        "data_sha256": "0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc",
        "split_sha256": "7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5",
        "prompt_manifest_sha256": sha256(ROOT / "prompt_manifest.json"),
        "files": entries,
    }
    target = ROOT / "sources/source_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"files": len(entries), "manifest": str(target),
                      "sha256": sha256(target)}, indent=2))


if __name__ == "__main__":
    main()
