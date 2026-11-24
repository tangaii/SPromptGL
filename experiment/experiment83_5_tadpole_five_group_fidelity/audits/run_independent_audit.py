"""Run the frozen independent audit with path-keyed inherited SHA manifests."""
from __future__ import annotations

import json
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "common"
sys.path.insert(0, str(COMMON))

import audit_e835
from fold_checkpoint import sha_file
from run_e833 import (capture_rng_state, restore_rng, seed_everything)


def verify_inherited_artifact_inventory(group: str, fold: int):
    config_path = ROOT / "configs" / f"{group}.json"
    config = json.loads(config_path.read_text())
    source_group = config["source_group"]
    e834_root = Path("/model/spromptgl/experiment/experiment83_4_tadpole_four_group_complete_evidence")
    source_dir = e834_root / source_group
    if not source_dir.is_dir():
        raise FileNotFoundError(f"E83.4 source fold directory missing for {group}")
    verified = {}
    for kind, directory_key in (("best", "best_checkpoint_sha256s"),
                                ("latest", "latest_checkpoint_sha256s")):
        relative_path = f"{source_group}/{kind}_checkpoints/fold{fold:02d}.pt"
        expected_sha = config[directory_key][relative_path]
        source_path = e834_root / relative_path
        local_path = ROOT / relative_path
        if sha_file(source_path) != expected_sha or sha_file(local_path) != expected_sha:
            raise RuntimeError(f"Inherited {kind} checkpoint copy/hash mismatch for {group} fold {fold}")
        verified[kind] = expected_sha
    source_config = source_dir / "config.json"
    return {"source_config_sha256": sha_file(source_config),
            "release_config_sha256": sha_file(config_path),
            "inherited_checkpoint_sha256s": verified}


audit_e835._verify_inherited_artifact_inventory = verify_inherited_artifact_inventory


def g5_next_step(fold, latest, split, x_np, y_np, device):
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    config_path = ROOT / "configs" / f"{audit_e835.GROUPS[4]}.json"
    identity = audit_e835.run_g5.checkpoint_identity(
        fold, train_idx, np.asarray(split["test_idx"], dtype=np.int64), sha_file(config_path))
    for key, value in identity.items():
        if latest.get(key) != value:
            raise RuntimeError(f"G5 fold {fold} Latest identity mismatch at {key}")
    anchor_path = ROOT / audit_e835.GROUPS[4] / "anchors" / f"fold{fold:02d}.pt"
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    x = torch.as_tensor(x_np, device=device)
    labels = torch.as_tensor(y_np, device=device, dtype=torch.long)
    rows = torch.as_tensor(train_idx, device=device, dtype=torch.long)

    def step_signature():
        seed_everything()
        model, optimizer = audit_e835.run_g5.restore_anchor(anchor, device)
        model.load_state_dict(latest["model"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(latest["optimizer"]))
        restore_rng(latest["rng"])
        restored_rng = audit_e835.run_g5.digest(capture_rng_state())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits, _ = model(x)
        loss = audit_e835.run_g5.weighted_nll(logits, labels, rows)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logp, _, _ = model(x)
        return {
            "loss": float(loss.detach()),
            "model_sha256": audit_e835.run_g5.digest(model.state_dict()),
            "optimizer_sha256": audit_e835.run_g5.digest(optimizer.state_dict()),
            "rng_sha256": audit_e835.run_g5.digest(capture_rng_state()),
            "eval_sha256": hashlib.sha256(logp.detach().cpu().numpy().tobytes()).hexdigest(),
            "restored_rng_sha256": restored_rng,
        }

    first, second = step_signature(), step_signature()
    if first != second:
        raise RuntimeError(f"G5 fold {fold} deterministic Latest next-step differs")
    latest_path = ROOT / audit_e835.GROUPS[4] / "latest_checkpoints" / f"fold{fold:02d}.pt"
    return {"latest_sha256": sha_file(latest_path), "best_sha256": latest["best_sha256"],
            "latest_epoch": int(latest["epoch"]), "optimizer_restored": True,
            "rng_restored": first["restored_rng_sha256"] == audit_e835.run_g5.digest(latest["rng"]),
            "finite_latest_forward": True, "next_step_signature": first}


audit_e835._g5_next_step = g5_next_step
raise SystemExit(audit_e835.run())
