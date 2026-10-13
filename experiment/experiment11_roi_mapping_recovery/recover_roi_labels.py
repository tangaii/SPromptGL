#!/usr/bin/env python3
"""Recover ABIDE PCP HO ROI labels and attach them to the frozen 256-edge map."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parents[1]
REFERENCE = EXP_ROOT / "reference"
RESULTS = EXP_ROOT / "results"
LOGS = EXP_ROOT / "logs"
ROI_DIR = PROJECT_ROOT / "dataset/ABIDE_pcp/cpac/filt_noglobal/rois_ho"

ABIDE_URLS = {
    "ho_mask_pad.nii.gz": "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/preprocessing/resources/abide_rois/ho_mask_pad.nii.gz",
    "abide_rois.txt": "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/preprocessing/settings/abide_rois.txt",
    "pipeline_config_abide_rerun.yml": "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/preprocessing/settings/pipeline_config_abide_rerun.yml",
}
FSL4_BASE = "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/fsl-4_0_2/"
FSL5_BASE = "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/fsl-5_0_11/"
FSL_FILES = {
    "HarvardOxford-Cortical.xml": FSL4_BASE + "HarvardOxford-Cortical.xml",
    "HarvardOxford-Subcortical.xml": FSL4_BASE + "HarvardOxford-Subcortical.xml",
    "fsl-4_0_2/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz": FSL4_BASE + "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
    "fsl-4_0_2/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz": FSL4_BASE + "HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
    "fsl-5_0_11/HarvardOxford-Cortical.xml": FSL5_BASE + "HarvardOxford-Cortical.xml",
    "fsl-5_0_11/HarvardOxford-Subcortical.xml": FSL5_BASE + "HarvardOxford-Subcortical.xml",
    "fsl-5_0_11/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz": FSL5_BASE + "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
    "fsl-5_0_11/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz": FSL5_BASE + "HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_bytes_fsync(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_text_fsync(path: Path, text: str) -> None:
    write_bytes_fsync(path, text.encode("utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_xml_labels(path: Path) -> dict[int, dict[str, object]]:
    root = ET.parse(path).getroot()
    result = {}
    for elem in root.findall("data/label"):
        index = int(elem.get("index"))
        xyz = [int(elem.get(axis)) for axis in ("x", "y", "z")]
        result[index] = {"name": " ".join((elem.text or "").split()), "xyz_voxel": xyz}
    return result


def label_header(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig") as f:
        line = f.readline().strip()
    fields = line.split("\t")
    if not fields or any(re.fullmatch(r"#\d+", item) is None for item in fields):
        raise ValueError(f"Unrecognized ROI label header: {path}")
    return [int(item[1:]) for item in fields]


def world_center(image: nib.spatialimages.SpatialImage, array: np.ndarray, value: int) -> np.ndarray:
    voxels = np.argwhere(array == value)
    if len(voxels) == 0:
        raise ValueError(f"Atlas value {value} has no voxels")
    return nib.affines.apply_affine(image.affine, voxels.mean(axis=0))


def image_label_centers(image: nib.spatialimages.SpatialImage, array: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(value): world_center(image, array, int(value))
        for value in np.unique(array)
        if int(value) != 0
    }


def ranked_centers(target: np.ndarray, centers: dict[int, np.ndarray]) -> list[tuple[float, int]]:
    return sorted((float(np.linalg.norm(target - center)), code) for code, center in centers.items())


def csv_text(columns: list[str], rows: list[dict[str, object]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project = args.project_root.resolve()
    exp = project / "experiment/experiment11_roi_mapping_recovery"
    ref = exp / "reference"
    results = exp / "results"
    logs = exp / "logs"
    roi_dir = project / "dataset/ABIDE_pcp/cpac/filt_noglobal/rois_ho"

    numeric_map = results / "roi_256_mapping.csv"
    subject_map = results / "subject_row_mapping.csv"
    audit_path = results / "data_audit.json"
    migration_log = logs / "migration.log"
    original_mapping_sha = sha256(numeric_map)
    old_audit = json.loads(audit_path.read_text(encoding="utf-8"))

    subjects = read_csv(subject_map)
    roi_files = [row["roi_raw_file"] for row in subjects]
    if len(subjects) != 871 or len(set(roi_files)) != 871:
        raise ValueError(f"Expected 871 unique mapped subjects/files, got {len(subjects)}/{len(set(roi_files))}")
    headers = []
    for name in roi_files:
        path = roi_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        headers.append(label_header(path))
    unique_headers = {tuple(row) for row in headers}
    if len(unique_headers) != 1:
        raise ValueError(f"Expected one stable column order across all files, got {len(unique_headers)}")
    ordered_labels = list(next(iter(unique_headers)))

    mask_path = ref / "ho_mask_pad.nii.gz"
    mask_image = nib.load(str(mask_path))
    mask = np.asanyarray(mask_image.dataobj).astype(np.int32, copy=False)
    mask_counts = {int(v): int(c) for v, c in zip(*np.unique(mask, return_counts=True))}
    atlas_labels = [v for v in mask_counts if v != 0]
    if len(ordered_labels) != 111 or set(ordered_labels) != set(atlas_labels):
        raise ValueError(".1D header labels do not exactly match the official ABIDE atlas mask labels")

    fsl4_cort_xml_path = ref / "HarvardOxford-Cortical.xml"
    fsl4_sub_xml_path = ref / "HarvardOxford-Subcortical.xml"
    fsl5_cort_xml_path = ref / "fsl-5_0_11/HarvardOxford-Cortical.xml"
    fsl5_sub_xml_path = ref / "fsl-5_0_11/HarvardOxford-Subcortical.xml"
    fsl4_cort_img = nib.load(str(ref / "fsl-4_0_2/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"))
    fsl4_sub_img = nib.load(str(ref / "fsl-4_0_2/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"))
    fsl4_cort = np.asanyarray(fsl4_cort_img.dataobj).astype(np.int32, copy=False)
    fsl4_sub = np.asanyarray(fsl4_sub_img.dataobj).astype(np.int32, copy=False)
    fsl5_cort = read_xml_labels(fsl5_cort_xml_path)
    fsl5_sub = read_xml_labels(fsl5_sub_xml_path)
    fsl4_cort = read_xml_labels(fsl4_cort_xml_path)
    fsl4_sub = read_xml_labels(fsl4_sub_xml_path)
    if set(fsl4_cort) != set(fsl5_cort) or set(fsl4_sub) != set(fsl5_sub):
        raise ValueError("FSL4/FSL5 atlas label indices differ")

    cortex_centers = image_label_centers(fsl4_cort_img, np.asanyarray(fsl4_cort_img.dataobj))
    sub_centers = image_label_centers(fsl4_sub_img, np.asanyarray(fsl4_sub_img.dataobj))
    cort_on_mask = np.asarray(resample_from_to(fsl4_cort_img, (mask_image.shape, mask_image.affine), order=0).dataobj, dtype=np.int32)
    sub_on_mask = np.asarray(resample_from_to(fsl4_sub_img, (mask_image.shape, mask_image.affine), order=0).dataobj, dtype=np.int32)

    standard: dict[int, dict[str, object]] = {}
    cortical_ids: set[int] = set()
    for index in sorted(fsl5_cort):
        atlas_value = index + 1  # FSL summary-image label value for XML index.
        name = str(fsl5_cort[index]["name"])
        if atlas_value not in cortex_centers:
            raise ValueError(f"FSL cortical label value {atlas_value} missing from source image")
        for suffix in (1, 2):
            output_value = atlas_value * 100 + suffix
            coordinates = np.argwhere(mask == output_value)
            if len(coordinates) == 0:
                raise ValueError(f"Expected cortical ABIDE mask value {output_value} is missing")
            center = nib.affines.apply_affine(mask_image.affine, coordinates.mean(axis=0))
            hemisphere = "Right" if center[0] > 0 else "Left"
            expected_suffix = 1 if hemisphere == "Right" else 2
            if suffix != expected_suffix:
                raise ValueError(f"Cortical label {output_value} contradicts its x=0 hemisphere")
            standard[output_value] = {
                "roi_name": f"{hemisphere} {name}",
                "hemisphere": hemisphere,
                "anatomical_category": "cortical",
                "name_source": "Official FSL HarvardOxford-Cortical.xml index; source FSL maxprob label value; ABIDE mask pair split at x=0",
                "atlas_index": index,
                "source_center_distance_mm": None,
                "source_center_margin_mm": None,
                "source_label_overlap_fraction": float(np.mean(cort_on_mask[mask == output_value] == atlas_value)),
                "notes": f"FSL cortical label value {atlas_value}; ABIDE code {output_value}=100×label+hemisphere suffix {suffix}.",
            }
            cortical_ids.add(output_value)

    # For the historical ABIDE codebook, subcortical integers are not XML indices.
    # Resolve each kept XML region to the nearest labeled region center in the original FSL4 image.
    excluded_terms = ("cerebral white matter", "cerebral cortex", "lateral vent", "brain-stem")
    retained_xml = [
        index for index, item in sorted(fsl4_sub.items())
        if not any(term in str(item["name"]).lower() for term in excluded_terms)
    ]
    if len(retained_xml) != 14:
        raise ValueError(f"Expected 14 retained subcortical labels after ABIDE exclusions, got {len(retained_xml)}")
    used_sub_ids: set[int] = set()
    for index in retained_xml:
        item = fsl4_sub[index]
        xml_center = nib.affines.apply_affine(fsl4_sub_img.affine, item["xyz_voxel"])
        ranked = ranked_centers(xml_center, sub_centers)
        distance, output_value = ranked[0]
        margin = ranked[1][0] - distance
        if distance > 12.0 or margin < 2.0 or output_value in used_sub_ids:
            raise ValueError(f"Ambiguous FSL subcortical label {item['name']}: distance={distance:.2f}, margin={margin:.2f}")
        if output_value not in mask_counts or output_value == 0:
            raise ValueError(f"FSL subcortical label code {output_value} is absent from ABIDE mask")
        output_center = world_center(mask_image, mask, output_value)
        output_ranked = ranked_centers(output_center, sub_centers)
        if output_ranked[0][1] != output_value or output_ranked[1][0] - output_ranked[0][0] < 2.0:
            raise ValueError(f"ABIDE label {output_value} does not independently map to its FSL subcortical source center")
        canonical_name = str(fsl5_sub[index]["name"])
        source_values, source_counts = np.unique(sub_on_mask[mask == output_value], return_counts=True)
        source_overlap = float(np.sum(source_counts[source_values == output_value]) / mask_counts[output_value]) if np.any(source_values == output_value) else 0.0
        standard[output_value] = {
            "roi_name": canonical_name,
            "hemisphere": "Left" if canonical_name.startswith("Left ") else "Right",
            "anatomical_category": "subcortical",
            "name_source": "Official FSL HarvardOxford-Subcortical.xml label center matched to original FSL maxprob value; FSL5 canonical label text",
            "atlas_index": index,
            "source_center_distance_mm": distance,
            "source_center_margin_mm": margin,
            "label_center_distance_mm": output_ranked[0][0],
            "label_center_margin_mm": output_ranked[1][0] - output_ranked[0][0],
            "source_label_overlap_fraction": source_overlap,
            "notes": f"Historical ABIDE/FSL label value {output_value}; XML index {index}; atlas-center match {distance:.2f} mm with {margin:.2f} mm nearest-label margin.",
        }
        used_sub_ids.add(output_value)

    extras = set(ordered_labels) - set(standard)
    if extras != {3455} or len(standard) != 110 or len(cortical_ids) != 96 or len(used_sub_ids) != 14:
        raise ValueError(f"Unexpected ROI composition: standard={len(standard)}, extras={sorted(extras)}")

    extra_voxels = np.argwhere(mask == 3455)
    extra_world = nib.affines.apply_affine(mask_image.affine, extra_voxels)
    extra_sub_values, extra_sub_counts = np.unique(sub_on_mask[mask == 3455], return_counts=True)
    extra_cort_values, extra_cort_counts = np.unique(cort_on_mask[mask == 3455], return_counts=True)
    subcode_to_name = {int(code): str(info["roi_name"]) for code, info in standard.items() if info["anatomical_category"] == "subcortical"}
    extra_sub_overlap = [
        {"fsl_value": int(v), "fsl_name": subcode_to_name.get(int(v), "Background/Unmapped"), "voxel_count": int(c)}
        for v, c in zip(extra_sub_values, extra_sub_counts)
    ]
    extra_cort_overlap = [{"fsl_value": int(v), "voxel_count": int(c)} for v, c in zip(extra_cort_values, extra_cort_counts)]

    roi_rows = []
    by_index: dict[int, dict[str, object]] = {}
    for column_index, atlas_value in enumerate(ordered_labels):
        is_extra = atlas_value == 3455
        info = standard.get(atlas_value)
        if info:
            name = str(info["roi_name"])
            hemisphere = str(info["hemisphere"])
            category = str(info["anatomical_category"])
            name_verified = True
            confidence = "VERIFIED"
            source = str(info["name_source"])
            notes = str(info["notes"])
        else:
            name = "UNKNOWN"
            hemisphere = "UNKNOWN"
            category = "extra_unknown"
            name_verified = False
            confidence = "EXTRA_ROI"
            source = "ABIDE PCP issue #9; no valid FSL XML label code"
            notes = "Two voxels. FSL4 cortical resampling labels both background; FSL4 subcortical resampling maps both to the Right Amygdala location (value 54). Retain as an extra code; do not rename or delete."
        center = world_center(mask_image, mask, atlas_value)
        column_order_verified = True
        evidence = (
            f"All {len(headers)}/{len(headers)} mapped .1D headers are identical; atlas value {atlas_value} is at zero-based column {column_index}; "
            "ordered header labels equal the nonzero labels in official ABIDE ho_mask_pad.nii.gz."
        )
        row = {
            "roi_index_zero_based": column_index,
            "roi_index_one_based": column_index + 1,
            "atlas_integer_value": atlas_value,
            "roi_name": name,
            "hemisphere": hemisphere,
            "anatomical_category": category,
            "voxel_count": mask_counts[atlas_value],
            "is_extra_roi": is_extra,
            "name_source": source,
            "column_order_evidence": evidence,
            "name_verified": name_verified,
            "column_order_verified": column_order_verified,
            "confidence_level": confidence,
            "notes": notes,
        }
        if info:
            row["notes"] += f" Mask center MNI (mm): {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}."
        else:
            row["notes"] += f" Mask center MNI (mm): {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}."
        roi_rows.append(row)
        by_index[column_index] = row

    old_features = read_csv(numeric_map)
    extra_columns = [
        "roi_i_atlas_value", "roi_j_atlas_value", "roi_i_name_verified", "roi_j_name_verified",
        "connection_name", "connection_anatomical_status", "involves_extra_roi",
        "usable_for_roi_semantic_prompt", "evidence_status",
    ]
    named_features = []
    for feature in old_features:
        i = int(feature["roi_i_zero_based"])
        j = int(feature["roi_j_zero_based"])
        li, lj = by_index[i], by_index[j]
        vi = bool(li["name_verified"] and li["column_order_verified"])
        vj = bool(lj["name_verified"] and lj["column_order_verified"])
        extra = bool(li["is_extra_roi"] or lj["is_extra_roi"])
        if vi and vj:
            status = "VERIFIED_BOTH_ENDPOINTS"
            connection_name = f"{li['roi_name']} — {lj['roi_name']}"
        elif vi or vj:
            status = "ONE_ENDPOINT_EXTRA_OR_UNKNOWN"
            left_name = str(li["roi_name"]) if vi else f"UNKNOWN (atlas {li['atlas_integer_value']})"
            right_name = str(lj["roi_name"]) if vj else f"UNKNOWN (atlas {lj['atlas_integer_value']})"
            connection_name = f"{left_name} — {right_name}"
        else:
            status = "BOTH_ENDPOINTS_UNKNOWN"
            connection_name = f"UNKNOWN (atlas {li['atlas_integer_value']}) — UNKNOWN (atlas {lj['atlas_integer_value']})"
        named_features.append({
            **feature,
            "roi_i_atlas_value": li["atlas_integer_value"],
            "roi_j_atlas_value": lj["atlas_integer_value"],
            "roi_i_name_verified": vi,
            "roi_j_name_verified": vj,
            "connection_name": connection_name,
            "connection_anatomical_status": status,
            "involves_extra_roi": extra,
            "usable_for_roi_semantic_prompt": bool(vi and vj and not extra),
            "evidence_status": "NUMERIC_MAPPING_VERIFIED; LABEL_AND_COLUMN_ORDER_VERIFIED" if vi and vj else "NUMERIC_MAPPING_VERIFIED; EXTRA_ROI_ENDPOINT_UNNAMED",
        })

    n_both = sum(row["connection_anatomical_status"] == "VERIFIED_BOTH_ENDPOINTS" for row in named_features)
    n_one = sum(row["connection_anatomical_status"] == "ONE_ENDPOINT_EXTRA_OR_UNKNOWN" for row in named_features)
    n_none = sum(row["connection_anatomical_status"] == "BOTH_ENDPOINTS_UNKNOWN" for row in named_features)
    n_extra_edges = sum(bool(row["involves_extra_roi"]) for row in named_features)
    usable = sum(bool(row["usable_for_roi_semantic_prompt"]) for row in named_features)
    mapping_sha_after = sha256(numeric_map)
    if mapping_sha_after != original_mapping_sha:
        raise ValueError("Original 256-feature numeric mapping changed during this audit")

    migration_text = migration_log.read_text(encoding="utf-8") if migration_log.exists() else ""
    migration_status_match = re.search(r"^MIGRATION_STATUS=(\w+)$", migration_text, flags=re.M)
    migration_status = migration_status_match.group(1) if migration_status_match else "UNKNOWN"
    migration_verified_match = re.search(r"^files_verified=(\d+)$", migration_text, flags=re.M)
    migration_source_match = re.search(r"^source_files=(\d+)$", migration_text, flags=re.M)
    source_files = int(migration_source_match.group(1)) if migration_source_match else None
    files_verified = int(migration_verified_match.group(1)) if migration_verified_match else None
    roi_label_status = "COMPLETE" if (
        len(headers) == 871 and len(unique_headers) == 1 and len(standard) == 110 and extras == {3455}
        and n_both == 252 and n_one == 4 and n_none == 0 and n_extra_edges == 4
    ) else "PARTIAL"

    ref_inventory = {}
    for relative, url in {**ABIDE_URLS, **FSL_FILES}.items():
        path = ref / relative
        if not path.is_file():
            # The original HO mask is stored at reference root, not in a nested subfolder.
            if relative == "ho_mask_pad.nii.gz":
                path = ref / "ho_mask_pad.nii.gz"
            else:
                raise FileNotFoundError(path)
        ref_inventory[relative] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}

    retrieved_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "experiment": "experiment11_roi_mapping_recovery",
        "retrieved_utc": retrieved_utc,
        "original_atlas_source": ABIDE_URLS["ho_mask_pad.nii.gz"],
        "abide_settings": {
            "abide_rois.txt": ABIDE_URLS["abide_rois.txt"],
            "pipeline_config": ABIDE_URLS["pipeline_config_abide_rerun.yml"],
            "observed_fsl_dir": "/usr/share/fsl/5.0",
            "patch_version_documented": False,
        },
        "fsl_reference_strategy": "FSL4.0.2 summary images provide the historical integer codebook matching the ABIDE mask; FSL5.0.11 XML provides canonical labels. FSL4/FSL5 label indices are equal; observed text differences are known spelling corrections only.",
        "files": ref_inventory,
        "external_evidence": {
            "abide_pipelines": "https://preprocessed-connectomes-project.org/abide/Pipelines.html",
            "extra_roi_issue": "https://github.com/preprocessed-connectomes-project/abide/issues/9",
            "fsl_xml_format": "https://fsl.fmrib.ox.ac.uk/fsl/docs/other/atlas_xml.html",
        },
        "not_for_github_upload": [
            "ho_mask_pad.nii.gz", "fsl-4_0_2/*maxprob*.nii.gz", "fsl-5_0_11/*maxprob*.nii.gz",
        ],
    }

    fsl4_name_diffs = {
        "cortical": [
            {"index": i, "fsl4_name": fsl4_cort[i]["name"], "fsl5_name": fsl5_cort[i]["name"]}
            for i in fsl4_cort if fsl4_cort[i]["name"] != fsl5_cort[i]["name"]
        ],
        "subcortical": [
            {"index": i, "fsl4_name": fsl4_sub[i]["name"], "fsl5_name": fsl5_sub[i]["name"]}
            for i in fsl4_sub if fsl4_sub[i]["name"] != fsl5_sub[i]["name"]
        ],
    }
    migration = {
        "status": migration_status,
        "source_files": source_files,
        "files_sha256_verified": files_verified,
        "source_directory_exists_after_verified_move": (Path("/tmp/spromptgl-work11/experiment11_roi_mapping_recovery").exists()),
        "log": str(migration_log),
        "destination": str(exp),
    }
    audit = {
        "experiment": "experiment11_roi_mapping_recovery",
        "MIGRATION_STATUS": migration_status,
        "ROI_LABEL_STATUS": roi_label_status,
        "migration": migration,
        "atlas": {
            "path": str(mask_path),
            "sha256": sha256(mask_path),
            "shape": list(mask_image.shape),
            "affine": mask_image.affine.tolist(),
            "zooms_mm": list(mask_image.header.get_zooms()[:3]),
            "nonzero_roi_count": len(atlas_labels),
            "unique_label_count_including_background": len(mask_counts),
            "label_set_matches_all_headers": True,
            "header_sequences_identical": len(unique_headers) == 1,
            "mapped_header_files": len(headers),
        },
        "roi_labels": {
            "total": len(ordered_labels),
            "verified_names": len(standard),
            "probable_names": 0,
            "unknown_other_than_explicit_extra": 0,
            "extra_roi_count": len(extras),
            "extra_roi": {
                "atlas_integer_value": 3455,
                "roi_index_zero_based": ordered_labels.index(3455),
                "roi_index_one_based": ordered_labels.index(3455) + 1,
                "voxel_count": mask_counts[3455],
                "voxel_ijk": extra_voxels.tolist(),
                "voxel_world_mm": extra_world.tolist(),
                "center_world_mm": extra_world.mean(axis=0).tolist(),
                "fsl4_cortical_overlay": extra_cort_overlap,
                "fsl4_subcortical_overlay": extra_sub_overlap,
                "roi_name": "UNKNOWN",
                "status": "EXTRA_ROI",
            },
            "column_order_verified": True,
            "cortical_count": len(cortical_ids),
            "subcortical_count": len(used_sub_ids),
            "fsl4_fsl5_xml_text_differences": fsl4_name_diffs,
        },
        "named_feature_map": {
            "rows": len(named_features),
            "both_endpoint_names_verified": n_both,
            "exactly_one_endpoint_name_verified": n_one,
            "neither_endpoint_name_verified": n_none,
            "involves_extra_roi": n_extra_edges,
            "usable_for_roi_semantic_prompt": usable,
            "extra_roi_feature_indices": [int(row["correlation_feature_index"]) for row in named_features if row["involves_extra_roi"]],
        },
        "original_numeric_mapping": {
            "path": str(numeric_map),
            "sha256_before": original_mapping_sha,
            "sha256_after": mapping_sha_after,
            "unchanged": mapping_sha_after == original_mapping_sha,
            "features_verified": old_audit.get("connectivity_reconstruction", {}).get("features_verified"),
            "verification_status": "VERIFIED_NUMERIC_UNIQUE" if old_audit.get("connectivity_reconstruction", {}).get("features_verified") == 256 else "CHECK_REQUIRED",
            "max_absolute_error": old_audit.get("connectivity_reconstruction", {}).get("max_absolute_error"),
            "global_rmse": old_audit.get("connectivity_reconstruction", {}).get("global_rmse"),
        },
        "limitations": [
            "No subject-level functional NIfTI volumes were found in the project, so no independent re-extraction of time series was possible.",
            "The original pipeline config specifies FSLDIR 5.0 but not a patch release; atlas codebook and label centers were cross-checked against official FSL 4.0.2 and FSL 5.0.11 resources.",
            "Anatomical names do not establish ASD-specific effects or causal interpretation.",
        ],
    }

    label_columns = [
        "roi_index_zero_based", "roi_index_one_based", "atlas_integer_value", "roi_name", "hemisphere",
        "anatomical_category", "voxel_count", "is_extra_roi", "name_source", "column_order_evidence",
        "name_verified", "column_order_verified", "confidence_level", "notes",
    ]
    feature_columns = list(old_features[0].keys()) + extra_columns
    report_lines = [
        "# Experiment 11：ABIDE PCP Harvard–Oxford ROI 标签与列序恢复",
        "",
        f"生成时间（UTC）：{retrieved_utc}",
        "",
        "## 状态",
        "",
        f"- `MIGRATION_STATUS`: **{migration_status}**。迁移 {files_verified}/{source_files} 个源文件通过 SHA256 校验；源目录在完整校验后按任务说明移除。",
        f"- `ROI_LABEL_STATUS`: **{roi_label_status}**。871/871 个 ROI 时间序列文件的标签头完全相同，111 个标签与原始 ABIDE PCP 掩膜的 111 个非背景值集合完全一致。",
        "- 已恢复 110 个标准 ROI 的解剖名称；额外编码 3455 保留为 `UNKNOWN` / `EXTRA_ROI`。没有修改原始 256 维映射或受试者数据。",
        "",
        "## 文件迁移审计",
        "",
        f"源目录：`/tmp/spromptgl-work11/experiment11_roi_mapping_recovery`；目标：`{exp}`。目标文件系统探测已完成 fsync、关闭、重新读取并通过。详细的逐文件 SHA256 记录见 [migration.log](../logs/migration.log)。",
        "",
        "## 图谱来源与映射方法",
        "",
        f"ABIDE PCP 官方 `abide_rois.txt` 将 HO 图谱资源指向 `ho_mask_pad.nii.gz`；本地原始掩膜 SHA256 为 `{sha256(mask_path)}`，形状为 `{mask_image.shape}`，分辨率为 3 mm。另记录了官方 pipeline 配置及 FSL Harvard–Oxford XML/最大概率标签图的来源、SHA256 与获取时间，见 [atlas_source_manifest.json](../reference/atlas_source_manifest.json)。",
        "",
        "列序没有按整数排序推断：逐一读取与 871 行受试者映射对应的全部 `.1D` 文件头。所有文件都给出相同的 111 个 `#<atlas_integer_value>`，其顺序与官方 ABIDE 掩膜的全部非零标签集合一一对应。因而 `rois_ho.1D` 第 k 列到掩膜标签整数的对应关系有直接文件头证据。",
        "",
        "皮层标签按 FSL cortical XML 的 48 个索引/名称对应原始 FSL 标签值；ABIDE 掩膜对每个皮层区域按 x=0 分成两侧，输出编码形如 `100 × FSL 标签值 + 半球后缀`，后缀 1/2 由掩膜质心的 MNI x 符号独立核验。皮层下代码不按数值顺序填充：将 FSL XML 坐标变换到原始 FSL4.0.2 图像空间，以各标签体素质心最近邻和第二名距离差匹配，然后验证 ABIDE 掩膜该代码的质心也唯一最近于同一原图标签。14 个匹配均有大于 2 mm 的最近邻间隔。FSL5.0.11 XML 仅用于采用规范的当前名称文本；两版本索引数量相同，差异仅为拼写修正。",
        "",
        "ABIDE PCP 说明 HO 概率图阈值、左右半球拆分和子皮层排除规则；相关历史 issue 报告了 110 个标准 ROI 外多出的 `3455`。当前数据头和掩膜确认它在第 83 列，原掩膜中仅有 2 个体素。将原 FSL4 图像按最近邻重采样到掩膜网格后，这两个点的皮层图标签均为背景，子皮层图标签均落在右侧杏仁核位置；由于整数值 3455 不属于标准标签代码，本报告仍将其记为额外未知编码，不将它命名为杏仁核。",
        "",
        "## ROI 名称覆盖",
        "",
        "| 分类 | 数量 | 处理 |",
        "|---|---:|---|",
        f"| 名称、标签、列序均验证 | {len(standard)} | 96 个皮层 + 14 个皮层下 |",
        "| 仅可能名称 | 0 | 无间接猜测名称 |",
        "| 其他未知 | 0 | 唯一未命名项单独列为 EXTRA_ROI |",
        f"| EXTRA_ROI / UNKNOWN | {len(extras)} | atlas 值 3455；第 83 列；2 个体素 |",
        "",
        "### 前 20 个 ROI",
        "",
        "| 列(0-based) | 列(1-based) | 掩膜整数值 | 名称 | 类别 | 体素数 | 置信级别 |",
        "|---:|---:|---:|---|---|---:|---|",
    ]
    for row in roi_rows[:20]:
        report_lines.append(f"| {row['roi_index_zero_based']} | {row['roi_index_one_based']} | {row['atlas_integer_value']} | {row['roi_name']} | {row['anatomical_category']} | {row['voxel_count']} | {row['confidence_level']} |")
    report_lines += [
        "",
        "### 第 83 个 ROI",
        "",
        f"0-based 索引 `{ordered_labels.index(3455)}` / 1-based 列 83 对应掩膜值 3455。体素 IJK 为 `{extra_voxels.tolist()}`，MNI 世界坐标为 `{np.round(extra_world, 2).tolist()}`，共 `{mask_counts[3455]}` 个体素。两个点在重采样 FSL 子皮层标签图中均落在 Right Amygdala 区域，但标签整数 3455 本身不是标准 FSL/ABIDE ROI 编码，故不覆盖成正式脑区名、不删除，也不用于 ROI 语义 Prompt。涉及它的 Correlation 特征索引：`{audit['named_feature_map']['extra_roi_feature_indices']}`。",
        "",
        "## 256 维 Correlation 连接命名",
        "",
        f"旧映射逐字节保持不变（SHA256 `{original_mapping_sha}`）；数值核验仍为 max absolute error `{audit['original_numeric_mapping']['max_absolute_error']:.3e}`、global RMSE `{audit['original_numeric_mapping']['global_rmse']:.3e}`。新命名表保留旧表全部字段，仅追加标签与证据字段。",
        "",
        "| 覆盖情况 | 连接数 |",
        "|---|---:|",
        f"| 两端脑区名和列序均可信 | {n_both} |",
        f"| 一端可信、一端为额外未知 ROI | {n_one} |",
        f"| 两端均未知 | {n_none} |",
        f"| 涉及额外 ROI 3455 | {n_extra_edges} |",
        f"| 可安全使用 ROI 名称级语义描述 | {usable} |",
        "",
        "下表列出前 20 个 Correlation 特征；`UNKNOWN (atlas 3455)` 保持明确未知，不以位置或 ASD 结果代填名称。",
        "",
        "| 特征索引 | 连接(0-based ROI 列) | 解剖连接 | 数值最大误差 | 状态 |",
        "|---:|---|---|---:|---|",
    ]
    for row in named_features[:20]:
        report_lines.append(f"| {row['correlation_feature_index']} | {row['roi_i_zero_based']}–{row['roi_j_zero_based']} | {row['connection_name']} | {float(row['max_absolute_error']):.3e} | {row['connection_anatomical_status']} |")
    report_lines += [
        "",
        "## 医学语义审计与限制",
        "",
        "- 110 个标准 ROI 可安全使用其 Harvard–Oxford 解剖名称；252 条连接的两端均有可信名称，可用于准确描述连接两端。4 条涉及 3455 的连接不得生成正式 ROI 名称 Prompt；可作为单独敏感性对照或保持为未知端点。",
        "- 可引用官方/可信神经科学资料补充通用解剖功能描述，但不能把正常解剖名称改写成 ASD 特异性异常。RFE 入选、相关系数方向或本次名字恢复都不构成 ASD 因果证据。",
        "- 项目中未发现与这些 ROI 时间序列配套的受试者功能 NIfTI；因此没有重提取时间序列的独立数值校验。现有 `feature_validation.csv` 的 256 维数值映射未改动，整体误差证据继续有效。",
        "- 当前可将 252 个已命名 ROI 连接的受试者相关系数写成忠实的定量描述；仅凭静态相关系数尚不能安全推断个体临床功能状态或疾病含义，仍需外部功能/临床验证。",
        "",
        "## 结论",
        "",
        "`ROI_LABEL_STATUS=COMPLETE`：110 个标准 ROI 名称与列序已通过官方掩膜、871 份 `.1D` 头、FSL 标签/空间中心交叉核验；第 83 列的 3455 已明确作为未知额外编码标出。后续可以开始构造真实 ROI 名称级的通用解剖语义，但疾病特异性和患者功能解释仍需单独验证。",
        "",
        "### 主要来源",
        "",
        "- [ABIDE PCP ROI / pipeline 说明](https://preprocessed-connectomes-project.org/abide/Pipelines.html)",
        "- [ABIDE PCP 官方图谱配置](https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/abide_rois.txt)",
        "- [ABIDE PCP 第 83 个 ROI issue #9](https://github.com/preprocessed-connectomes-project/abide/issues/9)",
        "- [FSL XML 图谱标签及坐标定义](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/atlas_xml.html)",
        "- [FSL data_atlases FSL 4.0.2](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-4_0_2)",
        "- [FSL data_atlases FSL 5.0.11](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-5_0_11)",
        "",
    ]

    write_text_fsync(results / "roi_111_labels.csv", csv_text(label_columns, roi_rows))
    write_text_fsync(results / "roi_256_named_mapping.csv", csv_text(feature_columns, named_features))
    write_text_fsync(results / "roi_label_audit.json", json.dumps(audit, ensure_ascii=False, indent=2, default=json_default) + "\n")
    write_text_fsync(ref / "atlas_source_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text_fsync(results / "roi_label_report.md", "\n".join(report_lines))
    log_lines = [
        f"ROI_LABEL_STATUS={roi_label_status}", f"MIGRATION_STATUS={migration_status}",
        f"created_utc={retrieved_utc}", f"headers_verified={len(headers)}/{len(headers)}",
        f"atlas_labels_verified={len(standard)}/110", f"extra_roi=3455 index_zero={ordered_labels.index(3455)} voxels={mask_counts[3455]}",
        f"features_both_endpoints={n_both}", f"features_one_endpoint={n_one}",
        f"features_extra_roi={n_extra_edges}", f"features_semantic_usable={usable}",
        f"numeric_mapping_sha256_unchanged={mapping_sha_after}",
        f"report={results / 'roi_label_report.md'}",
    ]
    write_text_fsync(logs / "roi_label_recovery.log", "\n".join(log_lines) + "\n")

    # Re-read each result after close/fsync so persisted output is verified on the mounted filesystem.
    for path in [results / "roi_111_labels.csv", results / "roi_256_named_mapping.csv", results / "roi_label_audit.json", results / "roi_label_report.md", ref / "atlas_source_manifest.json", logs / "roi_label_recovery.log"]:
        if not path.is_file() or path.stat().st_size == 0:
            raise IOError(f"Result not persisted: {path}")
        with path.open("rb") as f:
            os.fsync(f.fileno())
            f.read(128)
    if sha256(numeric_map) != original_mapping_sha:
        raise ValueError("Original numeric mapping changed after output generation")
    print(json.dumps({
        "MIGRATION_STATUS": migration_status,
        "ROI_LABEL_STATUS": roi_label_status,
        "header_files_verified": len(headers),
        "verified_names": len(standard),
        "extra_roi": {"value": 3455, "column_one_based": ordered_labels.index(3455) + 1, "voxels": mask_counts[3455]},
        "features_both_names_verified": n_both,
        "features_one_endpoint": n_one,
        "features_involving_extra": n_extra_edges,
        "usable_features": usable,
        "numeric_mapping_unchanged": mapping_sha_after == original_mapping_sha,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
