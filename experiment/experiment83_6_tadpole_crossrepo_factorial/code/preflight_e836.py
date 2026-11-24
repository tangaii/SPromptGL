"""Run the GPU and identity preflight for Experiment 83.6."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE / "common"))
sys.path.insert(0, str(CODE))

import run_e836 as runner
from fidelity_model import OriginalRES, normalize_adj
from run_g5 import PhysicalCAMOnly


EXPECTED_GROUPS = {
    "H1", "H2", "H3", "H4", "H5", "A1", "A2", "A3", "B1", "B2", "B3",
    "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "S0",
}
HISTORICAL_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
HISTORICAL_GROUPS = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    return sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def file_hash(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def direct_res(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, modalities, width = value.shape
    hidden = width // 4
    local = value.view(n, modalities, 4, hidden)
    global_value = F.avg_pool1d(value.reshape(n * modalities, 1, width), kernel_size=4, stride=4)
    global_value = global_value.view(n, modalities, hidden)
    pieces = []
    for part_index in range(4):
        part = local[:, :, part_index, :]
        relation = F.softmax(part * global_value, dim=-1)
        pieces.append(relation * part)
    return value + torch.cat(pieces, dim=2), global_value


def data_and_split_audit(x_np, y_np, ids, splits):
    frame = pd.read_csv(ROOT / "inputs/processed_standard_data.csv")
    feature_map = np.load(ROOT / "inputs/modal_feat_dict.npy", allow_pickle=True).item()
    ordered = [column for columns in feature_map.values() for column in columns]
    if frame.columns.tolist() != ordered + ["LABEL"]:
        raise RuntimeError("Main TADPOLE column order differs from its modality dictionary")
    if ids.RID.nunique() != 598:
        raise RuntimeError("Main TADPOLE patient identity validation failed")
    if x_np.shape != (598, 365) or np.bincount(y_np, minlength=3).tolist() != [209, 315, 74]:
        raise RuntimeError("Main TADPOLE cohort validation failed")
    expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
                    .split(np.zeros(len(y_np)), y_np))
    if len(expected) != len(splits):
        raise RuntimeError("The locked split count is not ten")
    test_occurrences = np.zeros(len(y_np), dtype=np.int64)
    fold_sizes = []
    for split, (train, test) in zip(splits, expected, strict=True):
        if (int(split["fold"]) < 1 or not np.array_equal(split["train_idx"], train)
                or not np.array_equal(split["test_idx"], test)):
            raise RuntimeError(f"Locked fold {split.get('fold')} differs from StratifiedKFold")
        if split.get("val_equals_test") is not True:
            raise RuntimeError(f"Fold {split['fold']} validation/test identity differs")
        test_occurrences[test] += 1
        fold_sizes.append(len(test))
    if not np.all(test_occurrences == 1):
        raise RuntimeError("OOF fold partition does not cover each patient exactly once")

    s0 = ROOT / "groups/S0/sp/data"
    s0_files = {name: file_hash(s0 / name) for name in (
        "features.json", "modal_feat_dict.npy", "processed_standard_data.csv", "samples.csv")}
    sp_frame = pd.read_csv(s0 / "processed_standard_data.csv")
    sp_features = np.load(s0 / "modal_feat_dict.npy", allow_pickle=True).item()
    sp_order = [column for columns in sp_features.values() for column in columns]
    if sp_frame.columns.tolist() != sp_order + ["LABEL"]:
        raise RuntimeError("S0 deterministic feature order differs from its prepared dictionary")
    sp_labels = sp_frame.LABEL.to_numpy(dtype=np.int64) - 1
    if (sp_frame.shape != (598, 366) or np.bincount(sp_labels, minlength=3).tolist() != [209, 315, 74]
            or s0_files != json.loads((s0 / "preprocessing.json").read_text())["output_hashes"]):
        raise RuntimeError("S0 prepared data does not match its preprocessing seal")
    s0_split = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
                    .split(np.zeros(len(sp_labels)), sp_labels))
    order_differences = sum(left != right for left, right in zip(ordered, sp_order, strict=True))
    s0_splits_match = all(
        np.array_equal(train, split["train_idx"]) and np.array_equal(test, split["test_idx"])
        for (train, test), split in zip(s0_split, splits, strict=True))
    if not s0_splits_match:
        raise RuntimeError("S0 StratifiedKFold indices differ from the registered common ten-fold split")
    return {
        "main_data_sha256": runner.sha_file(ROOT / "inputs/processed_standard_data.csv"),
        "main_split_sha256": runner.sha_file(ROOT / "inputs/split_indices.json"),
        "main_feature_order_sha256": sha_bytes("\n".join(ordered).encode()),
        "main_modalities": list(feature_map), "main_dimensions": [len(v) for v in feature_map.values()],
        "rows": 598, "input_dimensions": 365, "class_counts": [209, 315, 74],
        "fold_count": len(splits), "fold_test_sizes": fold_sizes, "val_equals_test": True,
        "test_oof_coverage_once": True, "s0_prepared_hashes": s0_files,
        "s0_feature_order_sha256": sha_bytes("\n".join(sp_order).encode()),
        "s0_feature_order_differences_from_main": int(order_differences),
        "s0_split_seed": 10, "s0_class_counts": np.bincount(sp_labels, minlength=3).tolist(),
        "s0_splits_match_locked_indices": s0_splits_match,
    }


def prompt_audit():
    old = torch.load(ROOT / "inputs/prompts/mmGPL12_rn50.pt", map_location="cpu", weights_only=False)
    prefixed = torch.load(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt", map_location="cpu", weights_only=False)
    query = torch.load(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt", map_location="cpu", weights_only=False)
    if old["clip_weights_sha256"] != prefixed["clip_weights_sha256"]:
        raise RuntimeError("Original and class-prefixed prompt vectors use different RN50 weights")
    if tuple(old["medical"].shape) != (12, 1024) or tuple(prefixed["medical"].shape) != (12, 1024):
        raise RuntimeError("MMGPL12 prompt embedding dimensions differ")
    class_map = prefixed["class_mapping"]
    concepts = prefixed["concepts"]
    expected_text = [f"{name}: {concept}" for name in ("NC", "MCI", "AD")
                     for concept in concepts[name]]
    if prefixed["texts"] != expected_text or len(expected_text) != 12:
        raise RuntimeError("Class prompt text or Figure 6 concept ordering differs")
    old_vec = F.normalize(old["medical"].float(), dim=-1)
    new_vec = F.normalize(prefixed["medical"].float(), dim=-1)
    paired_cosines = (old_vec * new_vec).sum(-1)
    centers = {}
    for class_id, name in enumerate(("NC", "MCI", "AD")):
        left, right = class_id * 4, (class_id + 1) * 4
        old_center = F.normalize(old_vec[left:right].mean(0), dim=0)
        new_center = F.normalize(new_vec[left:right].mean(0), dim=0)
        centers[name] = float(old_center @ new_center)
    if tuple(query["q_medical"].shape) != (12, 32):
        raise RuntimeError("SCMR prompt query map does not preserve all 12 concepts")
    return {
        "texts": prefixed["texts"], "class_mapping": class_map,
        "mci_proxy_note": "MMGPL Figure 6 uses LMCI. The four MCI prompts are semantic proxies for TADPOLE MCI.",
        "rn50_weights_sha256": prefixed["clip_weights_sha256"],
        "rn50_original_sha256": runner.sha_tensor(old["medical"]),
        "rn50_prefixed_sha256": runner.sha_tensor(prefixed["medical"]),
        "paired_prompt_cosine_mean": float(paired_cosines.mean()),
        "paired_prompt_cosine_min": float(paired_cosines.min()),
        "paired_prompt_l2_delta_mean": float((old["medical"] - prefixed["medical"]).norm(dim=-1).mean()),
        "class_centroid_cosine_before_after": centers,
        "scmr_query_shape": list(query["q_medical"].shape),
        "scmr_query_sha256": runner.sha_tensor(query["q_medical"]),
        "prefix_changes_representation": bool((paired_cosines < 1 - 1e-7).any()),
    }


def graph_audit(device, x_np, y_np, split):
    config = runner.group_config("A2")
    model = runner.build_model(config, int(split["fold"]), device)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.as_tensor(split["train_idx"], device=device).long()
    test = torch.as_tensor(split["test_idx"], device=device).long()
    model.eval()
    _, aux = model(x, labels, rows)
    loss, target = runner.pair_mean_graph_supervision(aux["continuous_scores"], labels, rows)
    changed = labels.clone()
    changed[test] = changed[test].roll(1)
    changed_loss, changed_target = runner.pair_mean_graph_supervision(aux["continuous_scores"], changed, rows)
    if len(aux["continuous_scores"]) != 6 or not loss.requires_grad:
        raise RuntimeError("Graph supervision did not receive six differentiable pre-threshold scores")
    expected_target = labels[rows, None].eq(labels[rows][None, :]).to(target.dtype)
    if (not torch.equal(target, expected_target) or not torch.equal(changed_target, target)
            or not torch.equal(loss, changed_loss)):
        raise RuntimeError("Graph supervision target includes non-training labels")
    if not torch.isfinite(loss.detach()) or float(loss.detach()) <= 0:
        raise RuntimeError("Graph supervision loss is zero or non-finite")
    for score, adjacency, normalized in zip(aux["continuous_scores"], aux["adjacencies"],
                                            aux["normalized_adjacencies"], strict=True):
        expected = score * (score > model.gfr.th).detach().to(score.dtype)
        torch.testing.assert_close(adjacency, expected, rtol=0, atol=0)
        eye = torch.eye(len(x), device=device, dtype=score.dtype)
        torch.testing.assert_close(normalized, normalize_adj(adjacency + eye), rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    loss.backward()
    graph_grad = [graph.p.weight.grad for graph in model.gfr.graphs]
    if any(value is None or not torch.isfinite(value).all() or float(value.norm()) <= 0 for value in graph_grad):
        raise RuntimeError("At least one GraphLearn module has no finite nonzero supervision gradient")

    parts = tuple(x.split(runner.historical.DIMS, dim=1))
    fused, details = model.gfr(parts, labels, rows)
    raw = torch.cat(parts, dim=1)
    expected_fused = raw
    coefficients = [0.0] * len(details["messages"])
    for index, message in enumerate(details["messages"]):
        coefficient = torch.sigmoid(model.gfr.fusion_weight[index])
        if index < 2:
            expected_fused = coefficient * message + raw
        else:
            expected_fused = coefficient * message + expected_fused
        if index >= 1:
            coefficients[index] = float(coefficient.detach())
    torch.testing.assert_close(fused, expected_fused, rtol=0, atol=0)
    if not torch.isfinite(torch.stack(aux["continuous_scores"])).all():
        raise RuntimeError("Continuous modality cosine scores contain NaN or Inf")
    return {
        "loss": float(loss.detach()), "target_shape": list(target.shape),
        "training_pair_rows": int(len(rows)), "six_modalities": len(aux["continuous_scores"]),
        "per_modality_loss": [float((score[rows[:, None], rows[None, :]] - target).square().mean())
                               for score in aux["continuous_scores"]],
        "graphlearn_gradient_norms": [float(value.norm()) for value in graph_grad],
        "continuous_scores_differentiable": True, "test_label_permutation_invariant": True,
        "test_nodes_excluded_from_target": int(target.shape[0]) == len(rows),
        "thresholded_adjacency_unchanged": True, "normalized_propagation_unchanged": True,
        "gfr_message_fusion_coefficients": coefficients,
        "first_modality_message_overwritten": coefficients[0] == 0.0,
        "fused_message_norms": [float(item.norm()) for item in details["messages"]],
        "all_finite": True,
    }


def res_dropout_wd_audit(device, x_np, y_np, split):
    token = torch.randn(7, 6, 40, device=device, requires_grad=True)
    actual, global_value = OriginalRES()(token)
    reference, reference_global = direct_res(token)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(global_value, reference_global, rtol=0, atol=0)
    actual.square().mean().backward()
    if not torch.isfinite(token.grad).all() or float(token.grad.norm()) <= 0:
        raise RuntimeError("RES-only backward is not finite and nonzero")

    base_config = runner.group_config("B1")
    res_config = runner.group_config("B2")
    base = runner.build_model(base_config, int(split["fold"]), device)
    res = runner.build_model(res_config, int(split["fold"]), device)
    res.load_state_dict(base.state_dict(), strict=True)
    base.eval()
    res.eval()
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.as_tensor(split["train_idx"], device=device).long()
    _, base_aux = base(x, labels, rows)
    _, res_aux = res(x, labels, rows)
    delta = res_aux["cam"] - base_aux["cam"]
    if not torch.isfinite(delta).all() or float(delta.norm()) <= 0:
        raise RuntimeError("Original RES has no measurable input/output effect in the forward probe")
    if sum(parameter.numel() for parameter in res.cam.RES.parameters()) != 0:
        raise RuntimeError("RES-only added learnable parameters")

    no_pen = runner.build_model(runner.group_config("B1"), int(split["fold"]), device)
    e1 = runner.build_model(runner.group_config("E1"), int(split["fold"]), device)
    if sum(parameter.numel() for parameter in no_pen.parameters()) != sum(
        parameter.numel() for parameter in e1.parameters()
    ):
        raise RuntimeError("E1 changed the parameter count")
    dropout = {name: float(module.attention.dropout.p) for name, module in e1.cam.named_modules()
               if name.startswith("encode_") and hasattr(module, "attention")}
    other_dropout = {name: float(module.p) for name, module in e1.cam.named_modules()
                     if isinstance(module, torch.nn.Dropout) and not name.endswith("attention.dropout")}
    if dropout != {"encode_0": 0.0, "encode_1": 0.0}:
        raise RuntimeError("E1 did not set only CAM Encoder attention dropout to zero")
    if any(value != 0.0 for value in other_dropout.values()):
        raise RuntimeError("E1 changed a non-Encoder CAM dropout")

    historical_h4 = torch.load(HISTORICAL_ROOT / HISTORICAL_GROUPS["H4"] /
                               "best_checkpoints/fold01.pt", map_location="cpu", weights_only=False)
    decay_contract = {}
    for name, config_name in (("H2", "B1"), ("H3", "C1"), ("C4", "C4"), ("D7", "D7")):
        cfg = runner.group_config(config_name)
        stats = historical_h4["spectral_stats"] if cfg["base_reference"] == "H4" else None
        model = runner.build_model(cfg, int(split["fold"]), device, spectral_stats=stats)
        optimizers, optimizer_config = runner.make_optimizers(model, cfg)
        decay_contract[name] = {
            "base": optimizers["base"].param_groups[0]["weight_decay"],
            "cam": optimizers["cam"].param_groups[0]["weight_decay"],
            "darg": (optimizers["darg"].param_groups[0]["weight_decay"]
                     if "darg" in optimizers else None),
            "dpsl": (optimizers["dpsl"].param_groups[0]["weight_decay"]
                     if "dpsl" in optimizers else None),
            "optimizer_config": optimizer_config,
        }
    expected = {"H2": (0.0036, 0.0036, None, None), "H3": (0.11, 0.11, 0.01, None),
                "C4": (0.0036, 0.0036, 0.01, None), "D7": (0.0036, 0.0036, 0.01, 0.01)}
    for name, values in expected.items():
        observed = decay_contract[name]
        actual_values = (observed["base"], observed["cam"], observed["darg"], observed["dpsl"])
        if actual_values != values:
            raise RuntimeError(f"{name} optimizer weight decay differs: {actual_values} != {values}")
    return {
        "res_formula_exact_to_author_local_global_operation": True,
        "res_token_shape": list(actual.shape), "res_global_shape": list(global_value.shape),
        "res_trainable_parameter_count": 0, "res_gradient_norm": float(token.grad.norm()),
        "res_cam_delta_norm": float(delta.norm()), "res_off_is_identity_by_branch": True,
        "e1_encoder_attention_dropout": dropout, "e1_other_cam_dropout": other_dropout,
        "e1_parameter_count_unchanged": True, "optimizer_weight_decay": decay_contract,
    }


def history_checkpoint_audit(splits):
    rows = []
    for logical, source_group in HISTORICAL_GROUPS.items():
        source = HISTORICAL_ROOT / source_group
        for split in splits:
            fold = int(split["fold"])
            best = source / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest = source / "latest_checkpoints" / f"fold{fold:02d}.pt"
            audit = source / "audits" / f"fold{fold:02d}.json"
            if not best.is_file() or not latest.is_file() or not audit.is_file():
                raise RuntimeError(f"Missing server-side E83.5 Best/Latest/audit for {logical} fold {fold}")
            best_payload = torch.load(best, map_location="cpu", weights_only=False)
            latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
            audit_payload = json.loads(audit.read_text())
            if (best_payload.get("fold") != fold or latest_payload.get("fold") != fold
                    or best_payload.get("train_idx") != split["train_idx"]
                    or best_payload.get("test_idx") != split["test_idx"]
                    or latest_payload.get("train_idx") != split["train_idx"]
                    or latest_payload.get("test_idx") != split["test_idx"]
                    or latest_payload.get("best_sha256") != file_hash(best)):
                raise RuntimeError(f"Historical checkpoint identity mismatch for {logical} fold {fold}")
            rows.append({"group": logical, "source_group": source_group, "fold": fold,
                         "best_sha256": file_hash(best), "latest_sha256": file_hash(latest),
                         "best_epoch": int(best_payload["epoch"]),
                         "latest_epoch": int(latest_payload["epoch"]),
                         "source_status": audit_payload.get("status"),
                         "source": "server E83.5 checkpoint, not GitHub artifact"})
    return rows


def main() -> None:
    seal_path = ROOT / "audits/training_seal.json"
    if not seal_path.is_file():
        raise RuntimeError("Create the pre-training source seal before running this preflight")
    runner.configure_worker()
    device = torch.device("cuda:0")
    x_np, y_np, ids, splits = runner.load_data()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    config_groups = {path.stem for path in (ROOT / "configs").glob("*.json")}
    if config_groups != EXPECTED_GROUPS or set(protocol["groups"]) != EXPECTED_GROUPS:
        raise RuntimeError("The registered 24-group matrix differs")
    if not list((ROOT / "sources/sp/exp/tadpole_repair").glob("*.py")):
        raise RuntimeError("The tangaii/sp source snapshot is missing")

    seal = json.loads(seal_path.read_text())
    current_hashes = runner.current_implementation_hashes()
    if (seal.get("status") != "SEALED_BEFORE_TRAINING" or seal.get("files") != current_hashes
            or seal.get("implementation_sha256") != runner.implementation_fingerprint(current_hashes)):
        raise RuntimeError("The pre-training source seal differs from the current registered files")

    start = {
        "experiment": "Experiment 83.6", "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "torch": str(torch.__version__),
        "formal_training_allowed": False,
        "training_seal_sha256": file_hash(seal_path),
        "implementation_sha256": json.loads(seal_path.read_text())["implementation_sha256"],
        "data_and_order": data_and_split_audit(x_np, y_np, ids, splits),
        "prompts": prompt_audit(),
        "graph_supervision": graph_audit(device, x_np, y_np, splits[0]),
        "res_dropout_weight_decay": res_dropout_wd_audit(device, x_np, y_np, splits[0]),
        "historical_checkpoint_inventory": history_checkpoint_audit(splits),
        "historical_checkpoint_count": 50,
    }
    physical_cam = PhysicalCAMOnly().to(device)
    forbidden = ("gfr", "graphlearn", "gcn", "messagepassing", "pen", "spgs", "res",
                 "clip", "prompt", "graphconstructloss")
    physical_names = [name.lower() for name, _ in physical_cam.named_parameters()]
    if any(term in name for name in physical_names for term in forbidden):
        raise RuntimeError("H5 physical CAM-only preflight found a forbidden module")
    start["h5_physical_cam_only"] = {
        "parameter_count": sum(p.numel() for p in physical_cam.parameters()),
        "named_parameters": physical_names, "forbidden_absent": True,
    }
    start["status"] = "PASS"
    start["formal_training_allowed"] = True
    path = ROOT / "audits/preflight_start.json"
    path.write_text(json.dumps(start, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": start["status"], "gpu": start["gpu"],
                      "main_data_sha256": start["data_and_order"]["main_data_sha256"],
                      "split_sha256": start["data_and_order"]["main_split_sha256"],
                      "historical_checkpoint_count": start["historical_checkpoint_count"]}, indent=2))


if __name__ == "__main__":
    main()
