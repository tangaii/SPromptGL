"""Subject-row audit and deterministic, label-blind fold text construction."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


QC_ANAT = ["anat_cnr", "anat_efc", "anat_fber", "anat_fwhm", "anat_qi1", "anat_snr"]
QC_FUNC = ["func_efc", "func_fber", "func_fwhm", "func_dvars", "func_outlier",
           "func_quality", "func_mean_fd", "func_num_fd", "func_perc_fd", "func_gsr"]
CLINICAL = ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ", "HANDEDNESS_CATEGORY"]
SITES = ["SITE_ID", "SITE_ID"]
SENTINELS = {-9999.0, -9999.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _valid_numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number in SENTINELS:
        return None
    return number


def _valid_string(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"-9999", "nan", "NaN", "N/A"}:
        return None
    return text


def load_aligned_inputs(root: Path, exp11: Path | None = None):
    root = Path(root).resolve()
    exp11 = exp11 or root / "experiment/experiment11_roi_mapping_recovery"
    processed_path = root / "data/ABIDE/processed_standard_data.csv"
    modal_path = root / "data/ABIDE/modal_feat_dict.npy"
    phenotype_path = root / "dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    mapping_path = exp11 / "results/subject_row_mapping.csv"
    feature_validation_path = exp11 / "results/feature_validation.csv"
    for path in (processed_path, modal_path, phenotype_path, mapping_path, feature_validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    processed = pd.read_csv(processed_path, low_memory=False)
    phenotype = pd.read_csv(phenotype_path, low_memory=False)
    mapping = pd.read_csv(mapping_path, low_memory=False)
    features_verified = pd.read_csv(feature_validation_path, low_memory=False)
    modal = np.load(modal_path, allow_pickle=True).item()
    dims = {"PHENO": 48, "ANAT": 6, "FUNC": 10, "CORR": 256}
    assert processed.shape == (871, 321)
    assert list(map(len, [modal["PHENO"], modal["ANAT"], modal["FUNC"], modal["Correlation"]])) == list(dims.values())
    assert mapping.shape[0] == 871 and mapping["verification_status"].eq("VERIFIED").all()
    assert mapping["subject_SUB_ID"].nunique() == 871 and mapping["csv_row_zero_based"].is_unique
    mapping = mapping.sort_values("csv_row_zero_based").reset_index(drop=True)
    assert np.array_equal(mapping.csv_row_zero_based.to_numpy(), np.arange(871))
    ids = pd.to_numeric(mapping.subject_SUB_ID, errors="raise").astype(int).to_numpy()
    assert phenotype.SUB_ID.is_unique and set(ids) == set(pd.to_numeric(phenotype.SUB_ID, errors="raise").astype(int)) & set(ids)
    by_id = phenotype.assign(SUB_ID=pd.to_numeric(phenotype.SUB_ID, errors="raise").astype(int)).set_index("SUB_ID", drop=False)
    aligned = by_id.loc[ids].reset_index(drop=True)
    assert np.array_equal(aligned.SUB_ID.to_numpy(dtype=np.int64), ids)
    assert len(features_verified) == 256
    assert features_verified.iloc[:, 0].is_unique
    if "status" in features_verified.columns:
        assert features_verified.status.astype(str).str.contains("PASS|VERIFIED|MATCH", case=False).all()

    y_raw = pd.to_numeric(processed["label"], errors="raise").to_numpy(dtype=np.int64)
    assert set(np.unique(y_raw)) == {1, 2}
    y = y_raw - 1
    dx = pd.to_numeric(aligned.DX_GROUP, errors="raise").to_numpy(dtype=np.int64)
    assert np.array_equal(dx - 1, y), "identity mapping failed the label-only after-the-fact audit"
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468

    # Reconstruct the original 48 PHENO + 16 QC block in the saved processed order.
    ph = phenotype.copy()
    ph["SUB_ID"] = pd.to_numeric(ph.SUB_ID, errors="raise").astype(int)
    subset = ph.loc[ph.SUB_ID.isin(ids)].copy().reset_index(drop=True)
    age = pd.to_numeric(subset.AGE_AT_SCAN, errors="coerce")
    bins, edge = [float(age.min())], float(age.min())
    while edge < float(age.max()):
        edge += 2
        bins.append(edge)
    encoded = subset.drop(columns=["AGE_AT_SCAN"]).copy()
    encoded["AGE_AT_SCAN"] = pd.cut(age, bins=bins, right=False)
    encoded = pd.get_dummies(encoded, columns=["AGE_AT_SCAN"])
    encoded = pd.get_dummies(encoded, columns=["SITE_ID"])
    encoded = pd.get_dummies(encoded, columns=["SEX"])
    pheno_columns = [c for c in encoded.columns if re.search(r"^SITE|^AGE_AT_SCAN|^SEX", str(c))]
    assert len(pheno_columns) == 48
    qc_columns = QC_ANAT + QC_FUNC
    qc = subset[qc_columns].apply(pd.to_numeric, errors="coerce")
    assert qc.notna().all().all(), "selected 871 subjects have missing QC input values"
    encoded.loc[:, qc_columns] = StandardScaler().fit_transform(qc)
    expected_by_id = encoded.assign(SUB_ID=subset.SUB_ID.to_numpy()).set_index("SUB_ID")
    expected_ids = expected_by_id.loc[ids, pheno_columns + qc_columns].to_numpy(dtype=np.float64)
    observed = processed.iloc[:, :64].to_numpy(dtype=np.float64)
    max_abs = float(np.max(np.abs(expected_ids - observed)))
    assert max_abs <= 1e-8, f"PHENO/QC row/value reconstruction mismatch: {max_abs}"
    x = processed.iloc[:, :320].to_numpy(dtype=np.float32)
    assert x.shape == (871, 320) and np.isfinite(x).all()

    identity_digest = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    field_audit = {}
    for name in CLINICAL + qc_columns:
        raw = aligned[name]
        if name in {"FIQ", "VIQ", "PIQ"}:
            values = pd.to_numeric(raw, errors="coerce").replace([-9999, -9999.0], np.nan)
        elif name == "HANDEDNESS_CATEGORY":
            values = raw.map(_valid_string)
        else:
            values = pd.to_numeric(raw, errors="coerce")
        field_audit[name] = {"valid": int(pd.Series(values).notna().sum()),
                             "missing": int(pd.Series(values).isna().sum())}
    audit = {
        "subjects": int(len(ids)), "unique_ids": int(len(set(ids))),
        "ordered_subject_id_sha256": identity_digest,
        "mapping_status_counts": mapping.verification_status.value_counts().to_dict(),
        "mapping_join_source": "Experiment11 verified subject_row_mapping.csv; no row-order assumption",
        "label_audit_only": {"ASD_label0": int((y == 0).sum()), "TD_label1": int((y == 1).sum()),
                             "dx_group_matches_processed_label": True},
        "processed_shape": list(processed.shape), "feature_shape": list(x.shape),
        "modal_dims": dims, "pheno_qc_max_abs_reconstruction_error": max_abs,
        "clinical_field_coverage": field_audit,
        "forbidden_fields_used_in_text": [],
        "inputs_sha256": {str(processed_path): sha256_file(processed_path),
                          str(modal_path): sha256_file(modal_path),
                          str(phenotype_path): sha256_file(phenotype_path),
                          str(mapping_path): sha256_file(mapping_path),
                          str(feature_validation_path): sha256_file(feature_validation_path)}
    }
    return {"root": root, "x": x, "y": y, "ids": ids, "phenotype": aligned,
            "processed": processed, "audit": audit}


def _clinical_text(row):
    parts = []
    age = _valid_numeric(row.get("AGE_AT_SCAN"))
    sex = _valid_numeric(row.get("SEX"))
    if age is not None:
        parts.append(f"age {age:.1f} years")
    if sex in (1.0, 2.0):
        parts.append("recorded sex male" if sex == 1.0 else "recorded sex female")
    for key, label in (("FIQ", "full-scale IQ"), ("VIQ", "verbal IQ"), ("PIQ", "performance IQ")):
        value = _valid_numeric(row.get(key))
        if value is not None:
            parts.append(f"recorded {label} score {value:.0f}")
    handedness = _valid_string(row.get("HANDEDNESS_CATEGORY"))
    if handedness is not None:
        parts.append(f"recorded handedness category {handedness.lower()}")
    return "Participant profile: " + "; ".join(parts) + "."


def _qc_z(train_rows, all_rows, columns):
    train = train_rows[:, columns]
    whole = all_rows[:, columns]
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    std[std < 1e-8] = 1.0
    return (whole - mean) / std


def _clinical_numeric(frame):
    n = len(frame)
    out = np.zeros((n, 11), dtype=np.float32)
    age = pd.to_numeric(frame.AGE_AT_SCAN, errors="coerce").to_numpy(dtype=float)
    out[:, 0] = np.nan_to_num(age / 100.0, nan=0.0)
    sex = pd.to_numeric(frame.SEX, errors="coerce").to_numpy(dtype=float)
    out[:, 1] = np.nan_to_num((sex - 1.0), nan=0.0)
    for col, target in zip(("FIQ", "VIQ", "PIQ"), (2, 3, 4)):
        arr = pd.to_numeric(frame[col], errors="coerce").replace([-9999, -9999.0], np.nan).to_numpy(dtype=float)
        out[:, target] = np.nan_to_num(arr / 200.0, nan=0.0)
    hand_map = {"R": 0, "L": 1, "Ambi": 2, "Mixed": 3, "Other": 4}
    for i, value in enumerate(frame.HANDEDNESS_CATEGORY):
        value = _valid_string(value)
        idx = hand_map.get(value, None)
        if idx is not None:
            out[i, 6 + idx] = 1.0
    # Site and missingness masks are intentionally excluded from the numeric condition.
    return out


def make_fold_payload(data, train_idx, fold: int):
    """Build same-source numeric vectors and deterministic text, using train stats only."""
    frame = data["phenotype"].reset_index(drop=True)
    x = data["x"].astype(np.float64)
    n = len(frame)
    clinical = [_clinical_text(row) for _, row in frame.iterrows()]
    clinical_num = _clinical_numeric(frame)

    anat_z = _qc_z(x[train_idx], x, np.arange(48, 54))
    func_z = _qc_z(x[train_idx], x, np.arange(54, 64))
    corr_train = x[train_idx, 64:320]
    corr = x[:, 64:320]
    corr_mean = corr_train.mean(axis=0)
    corr_std = corr_train.std(axis=0, ddof=0)
    corr_std[corr_std < 1e-8] = 1.0
    corr_z = np.clip((corr - corr_mean) / corr_std, -5.0, 5.0)
    corr_blocks = np.stack([np.stack([corr_z[:, j:j + 16].mean(1),
                                      corr_z[:, j:j + 16].std(1, ddof=0)], axis=1)
                            for j in range(0, 256, 16)], axis=1).reshape(n, 32)

    anat_text = []
    func_text = []
    corr_text = []
    for i in range(n):
        anat_items = [f"{name} z {value:+.2f}" for name, value in zip(QC_ANAT, anat_z[i])]
        func_items = [f"{name} z {value:+.2f}" for name, value in zip(QC_FUNC, func_z[i])]
        blocks = [f"feature block {b + 1} mean {corr_blocks[i, 2*b]:+.2f} SD {corr_blocks[i, 2*b+1]:.2f}"
                  for b in range(16)]
        anat_text.append("Anatomical scan quality measurements, standardized within the outer-training fold: " +
                         "; ".join(anat_items) + ".")
        func_text.append("Functional scan quality and motion measurements, standardized within the outer-training fold: " +
                         "; ".join(func_items) + ".")
        corr_text.append("Selected functional-connectivity feature-slot summaries, in processed feature order, standardized within the outer-training fold: " +
                         "; ".join(blocks) + ". These feature blocks are not anatomical regions.")

    modal_full = np.empty((n, 4), dtype=object)
    modal_only = np.empty((n, 4), dtype=object)
    for i in range(n):
        age = _valid_numeric(frame.iloc[i].get("AGE_AT_SCAN"))
        sex = _valid_numeric(frame.iloc[i].get("SEX"))
        pheno_evidence = "Participant age and recorded sex: " + "; ".join(
            value for value in ([f"age {age:.1f} years"] if age is not None else []) +
            (["male" if sex == 1.0 else "female"] if sex in (1.0, 2.0) else [])) + "."
        modal_only[i, 0] = pheno_evidence
        modal_only[i, 1] = anat_text[i]
        modal_only[i, 2] = func_text[i]
        modal_only[i, 3] = corr_text[i]
        modal_full[i, 0] = clinical[i]
        modal_full[i, 1] = clinical[i] + " " + anat_text[i]
        modal_full[i, 2] = clinical[i] + " " + func_text[i]
        modal_full[i, 3] = clinical[i] + " " + corr_text[i]

    all_full = []
    all_modal_only = []
    for i in range(n):
        all_full.append(clinical[i] + " " + anat_text[i] + " " + func_text[i] + " " + corr_text[i])
        all_modal_only.append(anat_text[i] + " " + func_text[i] + " " + corr_text[i] + " " + modal_only[i, 0])

    modal_numeric = np.empty((n, 4, 768), dtype=np.float32)
    modal_numeric[:] = 0.0
    modal_numeric[:, 0, :11] = clinical_num
    for m, values in ((1, anat_z), (2, func_z), (3, corr_blocks)):
        modal_numeric[:, m, :11] = clinical_num
        modal_numeric[:, m, 11:11 + values.shape[1]] = values.astype(np.float32)
    all_numeric = np.zeros((n, 768), dtype=np.float32)
    all_numeric[:, :11] = clinical_num
    cursor = 11
    for values in (anat_z, func_z, corr_blocks):
        width = values.shape[1]
        all_numeric[:, cursor:cursor + width] = values.astype(np.float32)
        cursor += width

    constant_modal = np.asarray([
        "Participant profile for the PHENO relation; no individual values are supplied.",
        "Anatomical scan-quality record; no individual measurements are supplied. These are image-quality measures, not structural lesions.",
        "Functional scan-quality record; no individual measurements are supplied. These are image-quality measures, not clinical symptoms.",
        "Selected functional-connectivity feature record; no individual measurements are supplied and feature slots are not anatomical regions."
    ], dtype=object)
    constant_all = "Participant record combining demographic, anatomical quality, functional quality, and selected connectivity sources; no individual values are supplied."

    return {
        "fold": int(fold), "clinical": np.asarray(clinical, dtype=object),
        "modal_full": modal_full, "all_full": np.asarray(all_full, dtype=object),
        "modal_only": modal_only, "all_modal_only": np.asarray(all_modal_only, dtype=object),
        "constant_modal": constant_modal, "constant_all": constant_all,
        "modal_numeric": modal_numeric, "all_numeric": all_numeric,
        "anat_z": anat_z.astype(np.float32), "func_z": func_z.astype(np.float32),
        "corr_block_summary": corr_blocks.astype(np.float32),
        "outer_train_stats": {
            "anat_mean": x[train_idx, 48:54].mean(0).tolist(),
            "anat_std": np.maximum(x[train_idx, 48:54].std(0, ddof=0), 1e-8).tolist(),
            "func_mean": x[train_idx, 54:64].mean(0).tolist(),
            "func_std": np.maximum(x[train_idx, 54:64].std(0, ddof=0), 1e-8).tolist(),
            "corr_mean": corr_mean.tolist(), "corr_std": corr_std.tolist()
        }
    }


def text_summary(payload, tokenizer=None):
    all_text = []
    for key in ("clinical", "modal_full", "all_full", "modal_only", "all_modal_only"):
        all_text.extend(np.asarray(payload[key], dtype=object).reshape(-1).tolist())
    all_text.extend(np.asarray(payload["constant_modal"], dtype=object).tolist())
    all_text.append(str(payload["constant_all"]))
    lengths = np.asarray([len(t.split()) for t in all_text], dtype=np.int64)
    truncation_count = 0
    if tokenizer is not None:
        unique = list(dict.fromkeys(all_text))
        truncation_count = sum(len(tokenizer.encode(t, add_special_tokens=True)) > 512 for t in unique)
    return {"text_instances": len(all_text), "unique_texts": len(set(all_text)),
            "unique_rate": len(set(all_text)) / max(len(all_text), 1),
            "mean_word_count": float(lengths.mean()), "max_word_count": int(lengths.max()),
            "token_truncated_unique_count": int(truncation_count)}


def missingness_audit(data):
    frame = data["phenotype"].reset_index(drop=True)
    y = data["y"]
    rows = []
    for field in CLINICAL:
        raw = frame[field]
        if field in {"FIQ", "VIQ", "PIQ", "AGE_AT_SCAN", "SEX"}:
            values = pd.to_numeric(raw, errors="coerce").replace([-9999, -9999.0], np.nan)
        else:
            values = raw.map(_valid_string)
        for label, name in ((0, "ASD"), (1, "TD")):
            mask = y == label
            rows.append({"field": field, "group": name, "n": int(mask.sum()),
                         "valid": int(pd.Series(values[mask]).notna().sum()),
                         "missing": int(pd.Series(values[mask]).isna().sum())})
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load_aligned_inputs(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"audit": data["audit"], "field_missingness_by_label_audit_only": missingness_audit(data)}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
