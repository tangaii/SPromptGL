"""Refresh E83.6 provenance manifests after local source assembly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(roots):
    result = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                result.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                               "sha256": digest(path)})
    return result


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    source_path = ROOT / "source_manifest.json"
    source = json.loads(source_path.read_text())
    source["files"] = inventory((ROOT / "code", ROOT / "sources"))
    source["source_roots"] = ["code", "sources/author", "sources/e83_5_reference_4649c5d",
                              "sources/e83_5_sources", "sources/sp"]
    source["mmgpl_figure6_paper"] = {
        "local_copy": "sources/sp/2024 MIA (MMGPL)Multimodal Medical Data Analysis with Graph Prompt Learning.pdf",
        "sha256": digest(ROOT / "sources/sp/2024 MIA (MMGPL)Multimodal Medical Data Analysis with Graph Prompt Learning.pdf"),
    }
    source["s0_preprocessing"] = {
        "source_commit": "f4534f8d6cc9288dd1d08f85470c4eb78e64789c",
        "local_preprocessing_manifest": "groups/S0/sp/data/preprocessing.json",
        "preprocessing_manifest_sha256": digest(ROOT / "groups/S0/sp/data/preprocessing.json"),
        "output_hashes": json.loads((ROOT / "groups/S0/sp/data/preprocessing.json").read_text())["output_hashes"],
    }
    write_json(source_path, source)

    data_path = ROOT / "data_manifest.json"
    data = json.loads(data_path.read_text())
    s0 = ROOT / "groups/S0/sp/data"
    s0_features = json.loads((s0 / "features.json").read_text())
    data["cross_repository_positive_control"] = {
        "source_commit": "f4534f8d6cc9288dd1d08f85470c4eb78e64789c",
        "preprocessing_manifest_sha256": digest(s0 / "preprocessing.json"),
        "files": {name: digest(s0 / name) for name in (
            "features.json", "modal_feat_dict.npy", "processed_standard_data.csv", "samples.csv")},
        "feature_order": [feature for columns in
                          __import__("numpy").load(s0 / "modal_feat_dict.npy", allow_pickle=True).item().values()
                          for feature in columns],
        "preprocessing": s0_features,
    }
    write_json(data_path, data)


if __name__ == "__main__":
    main()
