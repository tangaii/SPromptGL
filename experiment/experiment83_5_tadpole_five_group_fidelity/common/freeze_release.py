"""Write the five immutable method configurations and fidelity criteria."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fold_checkpoint import ROOT, atomic_write, sha_file, write_json


GROUPS = (
    "G1_E28_E0", "G2_E43_1_E00", "G3_E74_4_G2",
    "G4_E82_4_6_10_G1", "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
)


def file_hashes(directory: Path, pattern: str):
    return {str(path.relative_to(ROOT)): sha_file(path)
            for path in sorted(directory.glob(pattern)) if path.is_file()}


def build_configs(workers: int):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol_sha = sha_file(ROOT / "protocol.json")
    configs = ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    inherited = {
        "G2_E43_1_E00": "G2_E43_1_E00",
        "G3_E74_4_G2": "G3_E74_4_G2",
    }
    for group in GROUPS:
        config = {
            "experiment": protocol["experiment"], "group": group,
            "method": protocol["groups"][group], "protocol_sha256": protocol_sha,
            "data_sha256": protocol["dataset"]["data_sha256"],
            "split_sha256": protocol["dataset"]["split_sha256"],
            "seed": protocol["folds"]["training_seed"], "folds": 10,
            "actual_gpu_workers": workers,
            "status": "NEW_FORMAL_TRAINING" if group in (GROUPS[0], GROUPS[3], GROUPS[4])
                      else "INHERITED_CHECKPOINT_REPLAY",
        }
        if group == GROUPS[0]:
            config.update({
                "model_type": "FidelityModel", "prompt_file": "inputs/prompts/mmGPL12_scmr.pt",
                "prompt_file_sha256": sha_file(ROOT / "inputs/prompts/mmGPL12_scmr.pt"),
                "class_prompt_groups": {"NC": [0, 1, 2, 3], "MCI": [4, 5, 6, 7],
                                        "AD": [8, 9, 10, 11]},
                "formal_retraining": True,
            })
        elif group in inherited:
            old = inherited[group]
            config.update({
                "source_experiment": "Experiment 83.4",
                "source_group": old,
                "formal_retraining": False,
                "best_checkpoint_sha256s": file_hashes(ROOT / old / "best_checkpoints", "fold*.pt"),
                "latest_checkpoint_sha256s": file_hashes(ROOT / old / "latest_checkpoints", "fold*.pt"),
            })
        elif group == GROUPS[3]:
            config.update({
                "model_type": "FidelityModel", "prompt_file": "inputs/prompts/mmGPL12_scmr.pt",
                "prompt_file_sha256": sha_file(ROOT / "inputs/prompts/mmGPL12_scmr.pt"),
                "warmup_source_group": "G3_E74_4_G2", "warmup_epochs": 10,
                "formal_retraining": True,
            })
        else:
            config.update({
                "model_type": "PhysicalCAMOnly", "formal_retraining": True,
                "history_source": "sources/E43_4_original/E0_ANCHORED_PHYSICAL_CAM",
                "anchor_source": "untrained TADPOLE CAM constructor rebuilt from E83.4 G2",
                "no_trained_checkpoint_initialization": True,
                "optimizer": {"class": "Adam", "lr": 0.012, "weight_decay": 0.0036},
                "prompt_input": None,
            })
        target = configs / f"{group}.json"
        if target.exists() and json.loads(target.read_text()) != config:
            raise RuntimeError(f"Refusing to change frozen group configuration {target}")
        if not target.exists():
            write_json(target, config)

    matrix = """# Experiment 83.5 Method-Fidelity Criteria

## Source checks

- The MICCAI 2025 SPromptGL paper, the author repository and project commit `5ebc4f5` were checked against the frozen E83.4 source copies.
- The author implementation uses eight prompt rows and a two-class semantic competition. G1 preserves that eight-row branch and adds only a 12-row three-class PEN route for NC, MCI and AD.
- E43.4 E0 is an anchored, physically instantiated CAM-only experiment on ABIDE. Its checkpoint and reported score are not treated as TADPOLE weights or results.
- E43.2 is an initialization-method source only. It supplies no assumption that excluded checkpoints exist on GitHub.

## G1

- Historical eight-query forward parity and its gradient path must pass.
- The 12-query PEN must use the frozen RN50 features, the original 4096-128-hidden MLP, three-way semantic competition and original local/global fusion.
- Every four-prompt class group must have a nonzero gradient in the probe. The PEN forward signature accepts only token features, so test labels cannot route semantic attention.

## G2 and G3

- E83.4 Best and Latest checkpoints are read-only inputs. Each fold must independently restore Best and Latest, verify output and optimizer/RNG state, and deterministically replay the next step.
- G2 must have no PEN state or optimizer parameter. G3 must preserve the registered Positive-DCR source behavior and pass Direct-Off and Disease-Neutral interventions.

## G4

- The query is L2-normalized before both spectral attention and pair context. A frozen E83.4 G4 checkpoint is used only for a numerical before/after diagnostic.
- The 12 frozen RN50 features are normalized in float64. Their Gram matrix must be symmetric, positive definite and rank 12. Deterministic Cholesky rows are padded to width 32 and reproduce the original RN50 Gram matrix within the stated tolerance.
- All three class pairs remain active with the E83.4 weights, losses, optimizer, residual cap and shared gradient budget. The differential-query branch is disabled in the registered source configuration.
- Each fold begins from the matched G3 ten-epoch warmup state. E83.4 G4 final weights are not an initialization source.

## G5

- Each fold is anchored to a fresh, untrained TADPOLE CAM constructor state, initial CAM-only Adam state and post-construction RNG. No trained checkpoint initializes G5.
- The formal model contains only HistoricalCAM. GFR, graph learning, GCN, message passing, PEN, SPGS, CLIP, prompt embedding and graph loss are absent from its module tree, parameters, state and optimizer.
- A functional direct-CAM reference and the registered physical model must match logits, loss, every CAM gradient, updated parameter, Adam state and RNG for two steps from the same anchor.

## Shared protocol and release gate

- The frozen 598-patient TADPOLE matrix, 365 input features, six modality widths, 240-dimensional CAM and ten provided stratified folds must match their registered SHA-256 values.
- Historical whole-dataset RFE, pre-outer-CV scaling, mean imputation, transductive patient graphs for G1-G4, `val_idx == test_idx`, first strict test-fold ACC improvement, original losses, original optimizer behavior and early stopping remain fixed.
- G5 does not build a patient graph. No per-epoch seed reset, test-driven selection or threshold search is permitted.
- The formal release starts only when all preflight checks, the throughput probe and all five group configuration hashes pass. The start preflight and seal are immutable during training.
"""
    matrix_path = ROOT / "audits/method_fidelity_matrix.md"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_path.exists():
        if matrix_path.read_text() != matrix:
            raise RuntimeError("Refusing to change the frozen method-fidelity matrix")
    else:
        atomic_write(matrix_path, lambda stream: stream.write(matrix.encode()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.workers not in (1, 2, 4, 8, 10):
        parser.error("--workers must be a measured setting")
    build_configs(args.workers)
    print(json.dumps({
        "configs": {group: sha_file(ROOT / "configs" / f"{group}.json") for group in GROUPS},
        "method_fidelity_matrix_sha256": sha_file(ROOT / "audits/method_fidelity_matrix.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
