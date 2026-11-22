"""Read-only proof that the restored Fold-2 checkpoint reproduces E82.4.6.6."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/model/spromptgl/experiment/experiment82_4_6_6_g1_fold4_add712")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    evaluated = json.loads((ROOT / "runtime/independent_fold_02.json").read_text())
    with (SOURCE / "results/patient_level.csv").open(newline="") as stream:
        source_rows = [row for row in csv.DictReader(stream) if int(row["fold"]) == 2]
    expected_ids = [int(x) + 1 for x in evaluated["test_indices_0based"]]
    source_ids = [int(row["patient_id_1based"]) for row in source_rows]
    assert expected_ids == source_ids
    labels = [int(x) for x in evaluated["labels"]]
    assert labels == [int(row["true_label"]) for row in source_rows]

    fields = {
        "full": ("prediction_full", "p_asd_full", "true_margin_full", "margin_full"),
        "off": ("prediction_residual_off", "p_asd_residual_off", "true_margin_residual_off", "margin_off"),
        "anonymous": ("prediction_anonymous", "p_asd_anonymous", "true_margin_anonymous", "margin_anonymous"),
    }
    comparisons = {}
    for name, (prediction_field, probability_field, margin_field, margin_key) in fields.items():
        observed = evaluated[name]
        source_predictions = [int(row[prediction_field]) for row in source_rows]
        source_probabilities = np.asarray([float(row[probability_field]) for row in source_rows], dtype=np.float64)
        source_margins = np.asarray([float(row[margin_field]) for row in source_rows], dtype=np.float64)
        predictions = np.asarray(observed["prediction"], dtype=np.int64)
        probabilities = np.asarray(observed["probability"], dtype=np.float64)[:, 0]
        margins = np.asarray(evaluated[margin_key], dtype=np.float64)
        assert np.array_equal(predictions, source_predictions)
        assert np.array_equal(probabilities, source_probabilities)
        assert np.array_equal(margins, source_margins)
        comparisons[name] = {
            "prediction_exact": True,
            "probability_exact": True,
            "true_margin_exact": True,
        }

    manifest = json.loads((SOURCE / "manifest.json").read_text())
    identity = manifest["reuse"]["2"]
    chain = Path(identity["source_directory"])
    chain_manifest = json.loads((chain / "manifest.json").read_text())
    checkpoint = chain_manifest["checkpoint_sources"]["2"]
    underlying = Path(identity["underlying_source_directory"])
    task = json.loads((underlying / "runtime/G1_f02.training.json").read_text())
    best, latest, journal = (Path(checkpoint[key]) for key in ("best", "latest", "journal"))
    assert sha(chain / "manifest.json") == identity["source_manifest_sha"]
    assert sha(underlying / "manifest.json") == identity["underlying_source_manifest_sha"]
    assert evaluated["source_manifest_sha"] == identity["underlying_source_manifest_sha"]
    assert evaluated["checkpoint"] == str(best)
    assert evaluated["checkpoint_sha"] == identity["best_sha"] == checkpoint["best_sha"] == sha(best)
    assert identity["latest_sha"] == checkpoint["latest_sha"] == sha(latest)
    assert checkpoint["journal_sha"] == sha(journal)
    assert task["best_sha"] == sha(best) and task["latest_sha"] == sha(latest) and task["journal_sha"] == sha(journal)
    assert manifest["status"] == "COMPLETE_AUDITED" and evaluated["status"] == "PASS"
    result = {
        "status": "PASS",
        "source_directory": str(SOURCE),
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "checkpoint_source": {
            "best": str(best), "best_sha256": sha(best),
            "latest": str(latest), "latest_sha256": sha(latest),
            "journal": str(journal), "journal_sha256": sha(journal),
        },
        "test_rows": 87,
        "darg_train_count": evaluated["darg_train_count"],
        "legal_count": evaluated["legal_count"],
        "overlap_ids": evaluated["overlap_ids"],
        "patient540_in_train_or_relation_pairs": evaluated["patient540_in_train_or_relation_pairs"],
        "patient540_residual_allowed": evaluated["patient540_residual_allowed"],
        "patient540_joint_residual_l2": evaluated["patient540_joint_residual_l2"],
        "test_order_exact": True,
        "labels_exact": True,
        "patient_540": next(item for item in source_rows if int(item["patient_id_1based"]) == 540),
        "patient_18": next(item for item in source_rows if int(item["patient_id_1based"]) == 18),
        "interventions": comparisons,
        "checkpoint_manifest_and_task_exact": True,
    }
    (ROOT / "runtime/fold2_reuse_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"status": "PASS", "test_rows": 87, "best_sha": sha(best), "patients": [540, 18]}, indent=2))


if __name__ == "__main__":
    main()
