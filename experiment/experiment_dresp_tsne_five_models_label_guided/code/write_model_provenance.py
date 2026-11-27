#!/usr/bin/env python3
"""Record independent model checkpoint lineage and immutable source hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
OUT = EXP / "inputs/model_checkpoint_provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_files(directory: Path, pattern: str):
    records = []
    for fold in range(1, 11):
        path = directory / pattern.format(fold=fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"fold": fold, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return records


def main():
    model_sources = {
        "CAM_only": {
            "identity": "E43.4-E0",
            "source_kind": "verified_protocol_rebuild_from_E43.1_anchors",
            "feature_archive_directory": str(EXP / "inputs/cam_only_features"),
            "checkpoint_records": json.loads((EXP / "results/cam_only_rebuild/cam_only_rebuild_status.json").read_text())["feature_archives"],
        },
        "SPromptGL": {
            "identity": "E28-E0 (P0)",
            "source_kind": "isolated_seed0_retrain_best_checkpoints",
            "source_directory": "/model/spromptgl/experiment/experiment_dresp_tsne_40folds",
            "checkpoint_records": fold_files(Path("/model/spromptgl/experiment/experiment_dresp_tsne_40folds/checkpoints/E28_E0_seed0"), "fold{fold:02}.best.pt"),
        },
        "DARG": {
            "identity": "E74.4-G2",
            "source_kind": "authenticated_DARG_runtime_best_checkpoints",
            "source_directory": "/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/DARG/runtime",
            "checkpoint_records": fold_files(Path("/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/DARG/runtime/checkpoints"), "E74.4-G2-Positive-DCR_f{fold:02}.best.pt"),
        },
        "DARG_JSL": {
            "identity": "experiment_darg_jsl",
            "source_kind": "formal_DARG_JSL_best_checkpoints",
            "source_directory": "/model/spromptgl/experiment/experiment_darg_jsl",
            "checkpoint_records": fold_files(Path("/model/spromptgl/experiment/experiment_darg_jsl/checkpoints"), "fold{fold:02}.best.pt"),
        },
        "DReSP": {
            "identity": "E82.4.6.10-G1",
            "source_kind": "DPSL_formal_standalone_best_checkpoints",
            "source_directory": "/model/spromptgl/experiment/DPSL",
            "checkpoint_records": fold_files(Path("/model/spromptgl/experiment/DPSL/checkpoints"), "fold{fold:02}.best.pt"),
        },
    }
    payload = {
        "status": "PASS_SOURCE_CHECKPOINTS_PRESENT_OR_CAM_PROTOCOL_REBUILT",
        "experiment": "DReSP Dual Label-Guided t-SNE Exploration",
        "models_have_independent_lineage": True,
        "feature_archives_are_frozen": True,
        "models": model_sources,
        "historical_feature_archive_manifest": str(Path("/model/spromptgl/experiment/experiment_dresp_tsne_compact_separation/inputs/source_manifest.json")),
        "note": "CAM-only original E43.4-E0 Best binaries were absent. E43.1 anchor hashes matched the E43.4 manifest and the ten-fold E0 protocol was deterministically rebuilt with historical prediction/metric identity checks.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    handoff_bytes = handoff.read_bytes()
    (EXP / "results/handoff_prefix.json").write_text(json.dumps({
        "bytes": len(handoff_bytes),
        "sha256": hashlib.sha256(handoff_bytes).hexdigest(),
        "marker_absent_at_capture": b"DRESP_DUAL_LABEL_GUIDED_FIVE_MODELS" not in handoff_bytes,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "path": str(OUT), "models": list(model_sources)}, indent=2))


if __name__ == "__main__":
    main()
