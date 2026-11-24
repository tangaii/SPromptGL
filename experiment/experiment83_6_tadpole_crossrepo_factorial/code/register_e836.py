"""Register immutable E83.6 inputs, sources, protocol and group identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
E835 = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
DATA_SHA = "0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc"
SPLIT_SHA = "7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"Frozen registration differs at {path}")
    else:
        path.write_bytes(raw)


def group_specs():
    historical = {
        "H1": ("G1_E28_E0", "Original SPromptGL-MMGPL12"),
        "H2": ("G2_E43_1_E00", "Physical No-PEN"),
        "H3": ("G3_E74_4_G2", "DARG Positive-DCR"),
        "H4": ("G4_E82_4_6_10_G1", "DARG+DPSL"),
        "H5": ("G5_E43_4_E0_ANCHORED_PHYSICAL_CAM", "Anchored physical CAM-only"),
    }
    specs = {}
    for group, (source_group, label) in historical.items():
        specs[group] = {
            "kind": "HISTORICAL_REPLAY", "source_group": source_group,
            "label": label, "source_experiment": str(E835), "training": False,
        }

    def add(group, base, *, prefix=False, graph=False, res=False, low_wd=False,
            attention_dropout=None, warmup=None, note=""):
        specs[group] = {
            "kind": "NEW_FORMAL_TRAINING", "base_reference": base,
            "class_prefix_prompts": prefix, "pair_mean_graph_supervision": graph,
            "res_only": res, "low_base_weight_decay": low_wd,
            "encoder_attention_dropout": attention_dropout,
            "warmup_reference": warmup, "note": note,
            "max_epochs": 450, "early_stop_wait_gt": 200, "training_seed": 0,
            "training": True,
        }

    add("A1", "H1", prefix=True, note="Prompt text only")
    add("A2", "H1", graph=True, note="Pair-mean graph loss only")
    add("A3", "H1", prefix=True, graph=True, note="Prompt and pair-mean graph loss")
    add("B1", "H2", graph=True, note="Pair-mean graph loss only")
    add("B2", "H2", res=True, note="Original Eq. 6 RES only")
    add("B3", "H2", graph=True, res=True, note="RES and pair-mean graph loss")
    add("C1", "H3", graph=True, note="DARG plus pair-mean graph loss")
    add("C2", "H3", res=True, note="DARG plus original Eq. 6 RES")
    add("C3", "H3", graph=True, res=True, note="DARG plus RES and graph loss")
    add("C4", "H3", low_wd=True, note="GFR/CAM base WD 0.0036")
    add("D1", "H4", prefix=True, warmup="H3", note="DPSL class-prefixed medical queries")
    add("D2", "H4", graph=True, warmup="C1", note="DPSL plus C1-matched graph warmup")
    add("D3", "H4", prefix=True, graph=True, warmup="C1", note="DPSL class prefix plus C1-matched graph warmup")
    add("D4", "H4", res=True, warmup="C2", note="DPSL plus C2-matched RES warmup")
    add("D5", "H4", graph=True, res=True, warmup="C3", note="DPSL plus C3-matched RES/graph warmup")
    add("D6", "H4", prefix=True, graph=True, res=True, warmup="C3", note="DPSL class prefix plus C3-matched RES/graph warmup")
    add("D7", "H4", low_wd=True, warmup="C4", note="DPSL plus C4-matched low-WD warmup")
    add("E1", "H2", attention_dropout=0.0, note="Only CAM Encoder attention dropout 0.1 to 0")
    specs["S0"] = {
        "kind": "CROSS_REPOSITORY_POSITIVE_CONTROL", "base_reference": "sp/exp/tadpole_repair",
        "variant": "mmgpl_class_graph_sup", "class_prefix_prompts": True,
        "pair_mean_graph_supervision": True, "feature_order": "sp deterministic lexical order",
        "epochs": 300, "training_seed": 0, "split_seed": 10,
        "rng": "sp original continuous ten-fold RNG propagation",
        "attention_dropout": 0.0, "optimizer": "sp original Adam",
        "training": True,
    }
    return specs


def source_inventory():
    roots = (ROOT / "code", ROOT / "sources/sp/exp/tadpole_repair")
    entries = []
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "sha256": sha(path)})
    return entries


def main():
    data_dir = ROOT / "inputs"
    data_path = data_dir / "processed_standard_data.csv"
    split_path = data_dir / "split_indices.json"
    if sha(data_path) != DATA_SHA or sha(split_path) != SPLIT_SHA:
        raise RuntimeError("E83.6 data or split digest differs from the registered values")
    frame = pd.read_csv(data_path)
    samples = pd.read_csv(data_dir / "sample_ids.csv")
    features = np.load(data_dir / "modal_feat_dict.npy", allow_pickle=True).item()
    ordered = [column for columns in features.values() for column in columns]
    if frame.shape != (598, 366) or frame.columns.tolist() != ordered + ["LABEL"]:
        raise RuntimeError("TADPOLE feature matrix order differs from its modality dictionary")
    if samples.RID.nunique() != 598 or frame.LABEL.value_counts().sort_index().tolist() != [209, 315, 74]:
        raise RuntimeError("TADPOLE row identity or class distribution differs")

    prompt_rn = torch.load(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt", map_location="cpu",
                           weights_only=False)
    prompt_scmr = torch.load(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt", map_location="cpu",
                             weights_only=False)
    prompt_manifest = {
        "encoder": "OpenAI CLIP RN50, frozen official weights",
        "weights_path": "/root/.cache/clip/RN50.pt",
        "weights_sha256": prompt_rn["clip_weights_sha256"],
        "class_prompts": prompt_rn["texts"], "concepts": prompt_rn["concepts"],
        "class_mapping": prompt_rn["class_mapping"], "groups": prompt_rn["groups"],
        "mci_mapping_note": "MMGPL Figure 6 uses LMCI; the four MCI concepts are semantic proxies for TADPOLE MCI.",
        "rn50_tensor_shape": list(prompt_rn["medical"].shape),
        "rn50_tensor_sha256": tensor_sha(prompt_rn["medical"]),
        "rn50_file_sha256": sha(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt"),
        "query_shape": list(prompt_scmr["q_medical"].shape),
        "query_mapping_sha256": tensor_sha(prompt_scmr["q_medical"]),
        "scmr_file_sha256": sha(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt"),
        "gram_sha256": prompt_scmr["gram_sha"],
        "gram_reconstruction_max_abs": prompt_scmr["gram_reconstruction_max_abs"],
        "source": "tangaii/sp exp/tadpole_repair/config.py MMGPL_CLASS_PROMPTS, Figure 6 concepts",
    }
    write_once(ROOT / "prompt_manifest.json", prompt_manifest)

    groups = group_specs()
    protocol = {
        "experiment": "Experiment 83.6",
        "title": "TADPOLE Cross-Repository Reproduction, Fidelity Repair and DARG-DPSL Improvement",
        "status": "REGISTERED_PRETRAINING",
        "actual_root": str(ROOT),
        "requested_reference_project_commit": "4649c5d",
        "sealed_e83_5_project_commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
        "author_source_commit": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
        "sp_project_commit": "f4534f8",
        "paper": "https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
        "repositories": {
            "author": "https://github.com/wanxixi11/SPromptGL",
            "tadpole_reproduction": "https://github.com/tangaii/sp",
            "project_history": "https://github.com/tangaii/SPromptGL",
        },
        "dataset": {
            "n_samples": 598, "label_mapping": {"NC": 0, "MCI": 1, "AD": 2},
            "label_counts": {"NC": 209, "MCI": 315, "AD": 74},
            "modalities": list(features), "dimensions": [len(v) for v in features.values()],
            "total_input_dimensions": 365, "cam_output_dimensions": 240,
            "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
            "folds": "StratifiedKFold(n_splits=10, shuffle=True, random_state=10)",
            "preserved_protocol": ["full-data RFE", "mean imputation", "StandardScaler before outer CV",
                                   "transductive patient graph", "val_idx == test_idx",
                                   "first strict test-fold ACC improvement saves Best",
                                   "450 epoch cap", "stop when wait > 200"],
        },
        "training": {"device": "cuda:0", "gpu_workers_global_max": 8, "seed": 0,
                     "per_epoch_reseed": False, "new_formal_folds": 190,
                     "historical_replay_folds": 50},
        "metrics": {"ACC": "argmax accuracy", "AUC_original": "macro OVR AUC from hard argmax one-hot predictions",
                     "AUC_probability": "supplemental only"},
        "groups": groups,
        "paper_references": {"CAM_only": [0.8912, 0.9084], "CAM_GFR": [0.9380, 0.9437],
                             "CAM_GFR_RES": [0.9431, 0.9446],
                             "full": [0.9481, 0.9559]},
        "limitations": ["val=test checkpoint selection", "full-data RFE and scaling before outer CV",
                        "transductive patient graph", "multi-candidate comparison on one fixed CV"],
    }
    write_once(ROOT / "protocol.json", protocol)
    for group, spec in groups.items():
        config = {"experiment": "Experiment 83.6", "group": group, **spec,
                  "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA}
        write_once(ROOT / "configs" / f"{group}.json", config)

    data_manifest = {
        "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "files": {name: sha(data_dir / name) for name in (
            "processed_standard_data.csv", "sample_ids.csv", "split_indices.json", "modal_feat_dict.npy")},
        "rows": len(samples), "columns": frame.shape[1], "feature_order": ordered,
        "modalities": list(features), "dimensions": [len(v) for v in features.values()],
    }
    write_once(ROOT / "data_manifest.json", data_manifest)
    source_manifest = {
        "e83_5_project_repository": "https://github.com/tangaii/SPromptGL",
        "requested_e83_5_reference_commit": "4649c5d",
        "sealed_e83_5_commit": "5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11",
        "author_repository": "https://github.com/wanxixi11/SPromptGL",
        "author_commit": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
        "sp_repository": "https://github.com/tangaii/sp",
        "sp_local_snapshot_commit": "f4534f8",
        "author_source_paths": ["main.py", "model.py", "network.py", "layers.py", "utils.py",
                                "result_Tadpole.txt"],
        "files": source_inventory(),
        "source_copy_note": "All runtime and research source copies are local to E83.6. Checkpoints are inherited from the server's E83.5 directory, not claimed to exist on GitHub.",
    }
    write_once(ROOT / "source_manifest.json", source_manifest)
    print(json.dumps({"groups": len(groups), "data_sha256": DATA_SHA,
                      "split_sha256": SPLIT_SHA, "prompt_sha256": prompt_manifest["rn50_file_sha256"],
                      "source_files": len(source_manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
