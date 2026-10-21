"""Build and validate the frozen SCORE report bank.

The bank is deliberately independent of patients, labels and fold metrics.  The
only inputs are the verified ROI mapping and the frozen RN50 text encoder.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ORIGINAL = (
    "Nromal brain volume", "Uniform cortical thickness",
    "Symmetrical brain structure", "Normal ventricular morphology",
    "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
)
TEMPLATES = (
    "Resting-state functional connectivity between {left} and {right} is substantially below the training reference.",
    "Resting-state functional connectivity between {left} and {right} is moderately below the training reference.",
    "Resting-state functional connectivity between {left} and {right} is close to the training reference.",
    "Resting-state functional connectivity between {left} and {right} is moderately above the training reference.",
    "Resting-state functional connectivity between {left} and {right} is substantially above the training reference.",
)
EXCLUDED = (195, 225, 232, 236)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _mapping(root: Path):
    folder = root / "experiment/experiment11_roi_mapping_recovery/results"
    labels_path = folder / "roi_111_labels.csv"
    mapping_path = folder / "roi_256_named_mapping.csv"
    import csv
    with labels_path.open(newline="", encoding="utf-8") as f:
        labels = {int(r["roi_index_zero_based"]): r for r in csv.DictReader(f)}
    with mapping_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(labels) == 111 and len(rows) == 256
    tri_i, tri_j = np.triu_indices(111, 1)
    valid = []
    names = []
    for e, r in enumerate(rows):
        assert int(r["correlation_feature_index"]) == e
        assert r["verification_status"] == "VERIFIED_NUMERIC_UNIQUE"
        old = int(r["original_connection_index"])
        assert (int(r["roi_i_zero_based"]), int(r["roi_j_zero_based"])) == (int(tri_i[old]), int(tri_j[old]))
        usable = r["usable_for_roi_semantic_prompt"] == "True"
        if not usable:
            assert e in EXCLUDED
            continue
        i, j = int(r["roi_i_zero_based"]), int(r["roi_j_zero_based"])
        assert labels[i]["name_verified"] == "True" and labels[j]["name_verified"] == "True"
        left, right = labels[i]["roi_name"], labels[j]["roi_name"]
        assert "UNKNOWN" not in left + right
        assert r["connection_name"] == left + " — " + right
        valid.append(e)
        names.append((left, right))
    assert valid == [e for e in range(256) if e not in EXCLUDED]
    assert len(valid) == 252
    return np.asarray(valid, dtype=np.int64), names, mapping_path


def _encode(sentences):
    import sys
    import torch
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from CLIP.clip import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("RN50", device=device, jit=False)
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(sentences), 64):
            tok = clip.tokenize(sentences[start:start + 64], truncate=False).to(device)
            assert tok.shape[1] == 77
            values.append(model.encode_text(tok).float().cpu().numpy())
    del model
    raw = np.concatenate(values, axis=0).astype(np.float64)
    assert raw.shape[1] == 1024 and np.isfinite(raw).all()
    return raw


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    while np.any(p == np.arange(n)):
        p = rng.permutation(n)
    return p.astype(np.int64)


def _site_permutations(sites: np.ndarray, folds) -> np.ndarray:
    all_perm = []
    for fold_i, (train, test) in enumerate(folds):
        rng = np.random.default_rng(31031 + fold_i)
        p = np.arange(len(sites), dtype=np.int64)
        for part in (np.asarray(train), np.asarray(test)):
            for site in np.unique(sites[part]):
                ids = part[sites[part] == site]
                if len(ids) > 1:
                    q = rng.permutation(ids)
                    if np.any(q == ids):
                        q = np.roll(ids, 1)
                    p[ids] = q
        assert np.array_equal(sites[p], sites)
        assert all(set(p[part]) == set(part) for part in (train, test))
        assert np.any(p != np.arange(len(sites)))
        all_perm.append(p)
    return np.stack(all_perm)


def build(root: str | Path, exp: str | Path, sites: np.ndarray, folds) -> dict:
    root, exp = Path(root), Path(exp)
    valid, names, mapping_path = _mapping(root)
    sentences = list(ORIGINAL)
    for left, right in names:
        sentences.extend(t.format(left=left, right=right) for t in TEMPLATES)
    assert len(sentences) == 8 + 252 * 5
    reverse_sentences = []
    for left, right in names:
        reverse_sentences.extend(t.format(left=right, right=left) for t in TEMPLATES)
    encoded = _encode(sentences + reverse_sentences)
    raw, reverse_raw = encoded[:len(sentences)], encoded[len(sentences):]
    normalized = unit(raw)
    common = normalized.mean(axis=0)
    centered = normalized - common
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    projected = unit(centered @ vt[:64].T).astype(np.float32)
    assert projected.shape == (1268, 64) and np.isfinite(projected).all()
    text_bank = projected[8:].reshape(252, 5, 64)
    original8 = projected[:8]
    reverse_projected = unit((unit(reverse_raw) - common) @ vt[:64].T).astype(np.float32).reshape(252, 5, 64)

    # Frozen non-medical controls.  They are generated before formal training
    # and never fitted to a patient, label or result.
    rng = np.random.default_rng(31032)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    numeric_match = (text_bank @ orthogonal.astype(np.float32)).astype(np.float32)
    state_axis = np.zeros((252, 5, 64), np.float32)
    edge_basis = rng.normal(size=(252, 64))
    edge_basis = unit(edge_basis).astype(np.float32)
    for e in range(252):
        for s in range(5):
            state_axis[e, s] = unit(edge_basis[e] + (s - 2) * 0.1 * np.eye(64, dtype=np.float32)[e % 64])
    site_perms = _site_permutations(np.asarray(sites, dtype=np.int64), folds)
    roi_perm = _derangement(252, 31033)
    controls = {
        "state_reverse": [4, 3, 2, 1, 0],
        "state_degree_swap": [1, 0, 2, 4, 3],
        "roi_permutation": roi_perm.tolist(),
        "site_permutation_folds_sha256": hashlib.sha256(site_perms.tobytes()).hexdigest(),
    }
    wrong_projected = text_bank[roi_perm].copy()

    np.savez_compressed(
        exp / "report_bank.npz",
        text_bank=text_bank,
        original8=original8,
        p0_original8=raw[:8].astype(np.float32),
        numeric_match=numeric_match,
        numeric_state=state_axis,
        wrong_roi=wrong_projected,
        valid_indices=valid,
        roi_names=np.asarray(names, dtype="U160"),
        site_permutation_folds=site_perms,
        roi_permutation=roi_perm,
    )
    metadata = {
        "encoding": "RN50 frozen text pass; unit normalization, text-only centering/PCA64, row L2 normalization",
        "sentence_count": len(sentences),
        "valid_indices": valid.tolist(),
        "excluded_indices": list(EXCLUDED),
        "templates": list(TEMPLATES),
        "original": list(ORIGINAL),
        "roi_names": [list(x) for x in names],
        "controls": controls,
        "text_raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "reverse_text_raw_sha256": hashlib.sha256(reverse_raw.tobytes()).hexdigest(),
        "text_bank_sha256": hashlib.sha256(text_bank.tobytes()).hexdigest(),
        "original8_sha256": hashlib.sha256(original8.tobytes()).hexdigest(),
        "wrong_roi_sha256": hashlib.sha256(wrong_projected.tobytes()).hexdigest(),
        "pca_effective_rank": int(np.sum(singular[:64] > 1e-8)),
        "pca_energy": float(np.sum(singular[:64] ** 2) / max(np.sum(singular ** 2), 1e-30)),
        "state_adjacent_cosine_mean": float(np.mean(np.sum(text_bank[:, 1:] * text_bank[:, :-1], axis=-1))),
        "state_nonzero_fraction": float(np.mean(np.linalg.norm(text_bank - text_bank[:, 0:1], axis=-1) > 1e-6)),
        "state_distance_min": float(np.linalg.norm(text_bank - text_bank[:, 0:1], axis=-1)[:, 1:].min()),
        "reverse_direction_cosine_mean": float(np.mean(np.sum(text_bank * reverse_projected, axis=-1))),
        "cross_roi_cosine_mean": float(np.mean(np.sum(text_bank[:-1] * text_bank[1:], axis=-1))),
        "mapping_sha256": sha(mapping_path),
        "clip_weight_sha256": sha(Path.home() / ".cache/clip/RN50.pt"),
        "sentences_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in sentences],
        "no_patient_data": True,
    }
    (exp / "report_bank.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def load(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


if __name__ == "__main__":
    print("SCORE report_bank module ready")
