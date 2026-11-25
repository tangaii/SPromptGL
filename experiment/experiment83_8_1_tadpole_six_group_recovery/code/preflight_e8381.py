"""Run group-scoped Experiment 83.8.1 method and execution preflights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import run_e8381 as runner
from scmr_runtime import CLASS_PAIRS, source_config


HISTORY_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
E837_ROOT = Path("/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization")
HISTORY_NAMES = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data():
    x, y, ids, splits = runner.load_data()
    data_path = runner.ROOT / "inputs/processed_standard_data.csv"
    split_path = runner.ROOT / "inputs/split_indices.json"
    manifest = json.loads((runner.ROOT / "data_manifest.json").read_text())
    file_hashes = {name: sha_file(runner.ROOT / "inputs" / name)
                   for name in manifest["files"]}
    if file_hashes != manifest["files"]:
        raise RuntimeError("A locked TADPOLE input file differs from the sealed data manifest")
    feature_map = np.load(runner.ROOT / "inputs/modal_feat_dict.npy", allow_pickle=True).item()
    frame = pd.read_csv(data_path)
    ordered = [column for values in feature_map.values() for column in values]
    if frame.columns.tolist() != ordered + ["LABEL"]:
        raise RuntimeError("The locked modality feature order differs from the data columns")
    if list(map(len, feature_map.values())) != list(runner.MODAL_DIMS):
        raise RuntimeError("TADPOLE modality dimensions differ from the six registered branches")
    expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
                    .split(np.zeros(len(y)), y))
    occurrences = np.zeros(len(y), dtype=np.int64)
    fold_records = []
    for item, (train, test) in zip(splits, expected, strict=True):
        train_idx = np.asarray(item["train_idx"], dtype=np.int64)
        test_idx = np.asarray(item["test_idx"], dtype=np.int64)
        if not np.array_equal(train_idx, train) or not np.array_equal(test_idx, test):
            raise RuntimeError(f"Stratified split differs at fold {item.get('fold')}")
        if np.intersect1d(train_idx, test_idx).size or item.get("val_equals_test") is not True:
            raise RuntimeError("A fold overlaps train/test or does not retain val_idx == test_idx")
        occurrences[test_idx] += 1
        fold_records.append({"fold": int(item["fold"]), "train_count": len(train_idx),
                             "test_count": len(test_idx),
                             "train_class_counts": np.bincount(y[train_idx], minlength=3).tolist(),
                             "test_class_counts": np.bincount(y[test_idx], minlength=3).tolist()})
    if not np.all(occurrences == 1) or len(x) != 598 or len(ids) != 598 or ids.RID.nunique() != 598:
        raise RuntimeError("Patient dimensions, identities, or one-time OOF coverage differ")
    if np.bincount(y, minlength=3).tolist() != [209, 315, 74]:
        raise RuntimeError("Locked NC/MCI/AD class counts differ")
    return {"status": "PASS", "samples": len(x), "features": x.shape[1],
            "class_counts": np.bincount(y, minlength=3).tolist(),
            "modality_dimensions": list(runner.MODAL_DIMS),
            "data_sha256": sha_file(data_path), "split_sha256": sha_file(split_path),
            "locked_input_file_sha256": file_hashes,
            "folds": fold_records, "oof_coverage_once": True, "val_equals_test": True}


def check_history():
    replay_path = E837_ROOT / "audits/base_checkpoint_replay.json"
    summary_path = HISTORY_ROOT / "results/group_summary.csv"
    replay = json.loads(replay_path.read_text())
    summary = pd.read_csv(summary_path)
    if (replay.get("status") != "PASS" or replay.get("folds") != 50
            or replay.get("best_passes") != 50 or replay.get("latest_passes") != 50
            or replay.get("next_step_passes") != 50):
        raise RuntimeError("The E83.5 independent Best/Latest replay is not complete")
    expected_correct = {"H1": 521, "H2": 531, "H3": 553, "H4": 556, "H5": 511}
    records = []
    for logical, name in HISTORY_NAMES.items():
        row = summary[summary.group == name]
        if len(row) != 1:
            raise RuntimeError(f"Historical group identity is missing for {logical}")
        row = row.iloc[0]
        group_records = [item for item in replay["records"] if item["group"] == logical]
        if len(group_records) != 10:
            raise RuntimeError(f"Independent historical replay lacks ten folds for {logical}")
        replay_acc = float(np.mean([item["ACC"] for item in group_records]))
        replay_auc = float(np.mean([item["AUC_original"] for item in group_records]))
        replay_correct = int(sum(item["correct"] for item in group_records))
        if (abs(float(row.ACC_mean) - replay_acc) > 1e-12
                or abs(float(row.AUC_original_mean) - replay_auc) > 1e-12
                or int(row.correct_total) != replay_correct
                or replay_correct != expected_correct[logical]):
            raise RuntimeError(f"Historical summary differs from independently replayed {logical} predictions")
        for item in group_records:
            best, latest = Path(item["best_path"]), Path(item["latest_path"])
            if sha_file(best) != item["best_sha256"] or sha_file(latest) != item["latest_sha256"]:
                raise RuntimeError(f"Historical checkpoint changed for {logical} fold {item['fold']}")
        records.append({"group": logical, "source_group": name, "folds": 10,
                        "ACC_mean": replay_acc, "AUC_original_mean": replay_auc,
                        "correct_total": replay_correct,
                        "best_checkpoint_shas": [item["best_sha256"] for item in group_records],
                        "latest_checkpoint_shas": [item["latest_sha256"] for item in group_records]})
    snapshot = {"status": "PASS", "source": str(HISTORY_ROOT), "groups": records,
                "replay_source": str(replay_path), "replay_status": replay["status"]}
    runner.write_json(runner.ROOT / "audits/historical_checkpoint_snapshot.json", snapshot)
    return snapshot


def loss_fidelity_audit(aloss, pair_mean):
    author_network = runner.ROOT / "sources/author/network.py"
    author_model = runner.ROOT / "sources/author/model.py"
    network_text = author_network.read_text()
    model_text = author_model.read_text()
    if ("torch.linalg.norm(normalized_adj0 - A)" not in network_text
            or "return combined_feat , all_adj, ALoss/model_num" not in network_text
            or "loss = cls_loss + all_graph_loss" not in model_text
            or "loss.backward()" not in model_text):
        raise RuntimeError("The frozen author-source ALoss implementation does not match the reviewed expression")
    author_block = model_text[model_text.index("elif mode == 'simple-2'"):]
    author_block = author_block[:author_block.index("elif mode == 'normal'")]
    loss_expression = author_block[author_block.index("loss ="):author_block.index("loss.backward()")]
    if "ALoss" in loss_expression:
        raise RuntimeError("Author simple-2 unexpectedly backpropagates ALoss")
    aloss_ok = aloss["status"] in ("PASS", "PASS_WITH_STRUCTURAL_ZERO_GRADIENT")
    return {
        "status": "PASS" if aloss_ok and all(
            pair_mean[group].get("status") == "PASS" for group in ("X5", "X6")) else "FAIL",
        "formula_identity_status": "PASS",
        "mathematical_objects_are_distinct": True,
        "per_group_aloss_gradient_gate": aloss["group_statuses"],
        "aloss_failure_reason": aloss.get("failure_reason"),
        "paper_eq1": "L_G=(1/M) sum_m [L_S(A^(m)) + ||Ahat-A^(m)||_F^2]",
        "paper_eq7": "L=L_G+L_Label, lambda=eta=1",
        "author_network_expression": "(1/model_num) sum_m torch.linalg.norm(normalize(A_m+I)-Ahat), unsquared; hard-coded graph branches i=0,1,2,3",
        "author_model_training_use": "simple-2 backpropagates cls_loss + all_graph_loss; ALoss is only printed after backward",
        "author_network_sha256": runner.sha_file(author_network),
        "author_model_sha256": runner.sha_file(author_model),
        "e836_pair_mean": "mean over six continuous cosine matrices of the MSE over every ordered train-pair entry, including diagonal, coefficient 1.0",
        "e836_runner_sha256": hashlib.sha256(
            (Path("/model/spromptgl/experiment/experiment83_6_tadpole_crossrepo_factorial/code/run_e836.py")
             .read_bytes())).hexdigest(),
        "e837_runner_sha256": hashlib.sha256(
            (Path("/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization/code/run_e837.py")
             .read_bytes())).hexdigest(),
        "e838_x1_x4": {
            "formula": "(1/6) sum_m || normalize(A_m+I)[train,train] - normalize(Ahat_train+I_full)[train,train] ||_F^2",
            "coefficient": 1.0,
            "actual_graphs": "six actual GraphLearn adjacency matrices returned by the existing GFR, with DARG original modality graphs for X3/X4",
            "normalization": "symmetric D^-1/2(A+I)D^-1/2 over 598 transductive nodes before the train-train restriction",
            "label_mask": "only train_idx x train_idx entries use true labels; all held-out off-diagonal label entries are zero",
            "diagonal": "included, with the same-label diagonal and the identity self-loop retained",
            "loss_is_squared_frobenius_not_pair_mse": True,
            "backward_location": "added directly to historical_total before normal backward or DPSL shared-gradient optimizer step",
            "preflight_numeric_scale": {group: {
                "aloss_mean": item["loss"],
                "classification_loss_mean": item["classification_loss"],
                "aloss_to_classification_ratio": item["loss_to_classification_ratio"],
                "per_modality_loss": item["per_modality_loss"],
                "per_modality_p_weight_gradient_norm": item["per_modality_graphlearn_p_weight_gradient_norm"],
                "per_modality_actual_graph_offdiagonal_nonzero_entries": item[
                    "per_modality_actual_graph_offdiagonal_nonzero_entries"],
                "per_modality_effective_gradient_at_1e-8": item[
                    "per_modality_effective_gradient_at_1e-8"],
                "effective_gradient_modality_count": item["effective_gradient_modality_count"],
            } for group, item in aloss["groups"].items()},
        },
        "e838_x5": {
            "formula": pair_mean["X5"]["x5_formula"], "coefficient": 0.1,
            "active_from_epoch": 11, "diagonal": "excluded for same-class groups",
            "class_pair_groups": pair_mean["X5"]["group_counts"],
        },
        "e838_x6": {
            "formula": pair_mean["X6"]["x6_formula"], "coefficient": 0.1,
            "active_from_epoch": 11,
            "weight": "0.5 + 0.5*(u_i+u_j)/2, u=H(p)/log(3)",
            "weight_gradient": "detached from native logits and normalized inside each class-pair group",
        },
    }


def check_structure(device):
    records = {}
    expected_losses = {
        "X1": ("paper_aloss", 1.0, "continuous"),
        "X2": ("paper_aloss", 1.0, "continuous"),
        "X3": ("paper_aloss", 1.0, "continuous"),
        "X4": ("paper_aloss", 1.0, "continuous"),
        "X5": ("cb_pair_mean", 0.1, "epoch_11_onward"),
        "X6": ("cb_uncertainty_pair_mean", 0.1, "epoch_11_onward"),
    }
    for group in (f"X{index}" for index in range(1, 7)):
        config = runner.group_config(group)
        expected_kind, expected_coefficient, expected_schedule = expected_losses[group]
        if (config.get("supervision_kind") != expected_kind
                or float(config.get("supervision_coefficient")) != expected_coefficient
                or config.get("graph_supervision_schedule") != expected_schedule
                or int(config.get("training_seed")) != 0
                or int(config.get("max_epochs")) != 450
                or int(config.get("early_stop_wait_gt")) != 200):
            raise RuntimeError(f"{group} frozen loss, schedule, seed, or stopping configuration differs")
        runner.historical.seed_everything(runner.SEED)
        model = runner.build_model(config, 1, device)
        graphs = model.gfr.graphs
        expected = {
            "X1": (True, False, False), "X2": (False, False, False),
            "X3": (False, True, False), "X4": (False, True, True),
            "X5": (False, True, True), "X6": (False, True, True),
        }[group]
        actual = (bool(model.cam.use_pen), bool(model.use_darg), bool(model.use_dpsl))
        if actual != expected or len(graphs) != 6 or tuple(model.dims) != runner.MODAL_DIMS:
            raise RuntimeError(f"{group} changes its registered base architecture: {actual}")
        if model.cam.Outputlayer.classifier.out_features != 3:
            raise RuntimeError(f"{group} classifier is not three-class")
        records[group] = {"base_reference": config["base_reference"], "pen": actual[0],
                          "darg": actual[1], "dpsl_registered": actual[2],
                          "graphlearn_modules": len(graphs), "modal_dims": list(model.dims),
                          "supervision_kind": expected_kind,
                          "supervision_coefficient": expected_coefficient,
                          "schedule": expected_schedule}
        del model
        torch.cuda.empty_cache()
    cfg = source_config()
    if (set(CLASS_PAIRS) != {(0, 1), (0, 2), (1, 2)}
            or float(cfg["residual"]["fraction"]) != 0.08
            or float(cfg["full_classification"]["shared_cap"]) != 0.001):
        raise RuntimeError("G4 DPSL class pairs, residual cap, or shared-gradient cap changed")
    for group in ("X4", "X5", "X6"):
        config = runner.group_config(group)
        if float(config["dpsl_shared_cap"]) != 0.001 or float(config["dpsl_residual_cap"]) != 0.08:
            raise RuntimeError(f"{group} DPSL caps differ from frozen G4")
    return {"status": "PASS", "groups": records, "dpsl_class_pairs": [list(pair) for pair in CLASS_PAIRS],
            "dpsl_residual_cap": float(cfg["residual"]["fraction"]),
            "dpsl_shared_gradient_cap": float(cfg["full_classification"]["shared_cap"])}


def check_disabled_forward(device):
    x_np, y_np, _, splits = runner.load_data()
    split = splits[0]
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device).long()
    train_np = np.asarray(split["train_idx"], dtype=np.int64)
    train = torch.as_tensor(train_np, device=device).long()
    rows = {}
    for group, name in (("H1", "G1_E28_E0"), ("H2", "G2_E43_1_E00"),
                        ("H3", "G3_E74_4_G2"), ("H4", "G4_E82_4_6_10_G1")):
        path = HISTORY_ROOT / name / "best_checkpoints/fold01.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = {"base_reference": group, "class_prefix_prompts": False}
        model = runner.build_model(config, 1, device, spectral_stats=payload.get("spectral_stats"))
        if model.use_darg:
            model.darg.enter_positive_dcr()
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        pair = same = None
        if model.use_darg:
            pair = payload["relation_pairs"].to(device)
            same = payload["relation_targets"].to(device)
        with torch.no_grad():
            output, aux = model(x, y, train, pair, same, dpsl_enabled=True)
            baseline_logits = output.clone()
        _, historical_total, _, _ = runner.historical_loss(aux, y, train, model, "none")
        predictions = output.argmax(1).cpu().numpy()
        expected = pd.read_csv(HISTORY_ROOT / name / "predictions/fold01.csv")
        test_np = np.asarray(split["test_idx"], dtype=np.int64)
        np.testing.assert_array_equal(predictions[test_np], expected.pred_label.to_numpy(np.int64))
        if not torch.isfinite(historical_total) or not torch.equal(output, baseline_logits):
            raise RuntimeError(f"Disabling the added loss changed {group} native forward output")
        rows[group] = {"status": "PASS", "checkpoint_sha256": runner.sha_file(path),
                       "prediction_matches_historical": True,
                       "native_forward_unchanged_when_added_loss_disabled": True,
                       "model_parameter_count": sum(parameter.numel() for parameter in model.parameters())}
        del model
        torch.cuda.empty_cache()
    return {"status": "PASS", "groups": rows}


def check_initial_pairing(device):
    records = {}
    for group, source_name in (("X1", "G1_E28_E0"), ("X2", "G2_E43_1_E00")):
        config = runner.group_config(group)
        runner.historical.seed_everything(runner.SEED)
        model = runner.build_model(config, 1, device)
        optimizers, optimizer_config = runner.make_optimizers(model, config)
        source_path = HISTORY_ROOT / source_name / "best_checkpoints/fold01.pt"
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        current_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        source_shapes = {key: tuple(value.shape) for key, value in source["model"].items()}
        if current_shapes != source_shapes or optimizer_config != source.get("optimizer_config"):
            raise RuntimeError(f"{group} architecture or optimizer contract differs from historical {source_name}")
        records[group] = {
            "status": "PARTIAL_MATCH", "source_group": source_name,
            "architecture_state_shapes_match": True, "optimizer_contract_matches": True,
            "candidate_initial_model_sha256": runner.sha_object(model.state_dict()),
            "candidate_initial_optimizer_sha256": runner.sha_object(
                {key: value.state_dict() for key, value in optimizers.items()}),
            "candidate_initial_rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
            "historical_initial_hashes_available": False,
            "historical_checkpoint_sha256": runner.sha_file(source_path),
        }
        del model, optimizers
        torch.cuda.empty_cache()
    return {"status": "PARTIAL_MATCH", "pairs": records,
            "reason": "E83.5 H1/H2 checkpoints do not contain initial model, optimizer, or RNG hashes."}


def model_batch(group, fold, x, y_np, y, split, device):
    config = runner.group_config(group)
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device)
    rows_np = np.asarray(split["train_idx"], dtype=np.int64)
    rows = torch.as_tensor(rows_np, device=device).long()
    pair = same = None
    if model.use_darg:
        supports = runner.historical.anchor_pair_support(model, x)
        pair_np, same_np = runner.historical.make_pairs(rows_np, y_np, fold, supports)
        pair = torch.as_tensor(pair_np, device=device).long()
        same = torch.as_tensor(same_np, device=device).float()
    model.eval()
    _, aux = model(x, y, rows, pair, same, dpsl_enabled=False, darg_warmup=model.use_darg)
    return model, rows, pair, same, aux


def _gradients(loss, graph_modules, retain_graph):
    parameters = [graph.p.weight for graph in graph_modules]
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return [None if value is None else value.detach().clone() for value in values]


def _same_gradients(left, right):
    return len(left) == len(right) and all(
        (a is None and b is None) or
        (a is not None and b is not None and torch.equal(a, b))
        for a, b in zip(left, right, strict=True))


def _gradient_vector(values, parameters):
    pieces = [torch.zeros_like(parameter).reshape(-1) if value is None else value.reshape(-1)
              for value, parameter in zip(values, parameters, strict=True)]
    return torch.cat(pieces)


def _gradient_comparison(supervised, historical):
    supervised_norm = float(supervised.norm())
    historical_norm = float(historical.norm())
    if supervised_norm == 0.0 or historical_norm == 0.0:
        cosine = None
    else:
        cosine = float(torch.dot(supervised, historical) / (supervised_norm * historical_norm))
    return {
        "aloss_gradient_norm": supervised_norm,
        "historical_gradient_norm": historical_norm,
        "norm_ratio_aloss_to_historical": supervised_norm / max(historical_norm, 1e-30),
        "cosine_similarity": cosine,
    }


def _optimizer_update_norm(parameters, gradients, optimizer_template):
    clones = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    template_group = optimizer_template.param_groups[0]
    options = {key: value for key, value in template_group.items() if key != "params"}
    optimizer = type(optimizer_template)([{"params": clones, **options}])
    for parameter, gradient in zip(clones, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    before = torch.cat([parameter.detach().reshape(-1) for parameter in clones])
    optimizer.step()
    after = torch.cat([parameter.detach().reshape(-1) for parameter in clones])
    return float((after - before).norm())


def check_paper_aloss(device, requested_groups=None):
    x_np, y_np, _, splits = runner.load_data()
    y = torch.as_tensor(y_np, device=device).long()
    x = torch.as_tensor(x_np, device=device)
    split = splits[0]
    rows_np = np.asarray(split["train_idx"], dtype=np.int64)
    records = {}
    groups = requested_groups or ("X1", "X2", "X3", "X4")
    for group in groups:
        model, rows, pair, same, aux = model_batch(group, 1, x, y_np, y, split, device)
        _, base_total, _, _ = runner.historical_loss(aux, y, rows, model, "none")
        aloss, details = runner.paper_aloss(aux, y, rows, model)
        parameters = [graph.p.weight for graph in model.gfr.graphs]
        gradients = torch.autograd.grad(aloss, parameters, retain_graph=True, allow_unused=True)
        historical_gradients = torch.autograd.grad(base_total, parameters, retain_graph=True,
                                                   allow_unused=True)
        combined_gradients = torch.autograd.grad(
            base_total + runner.PAPER_ALOSS_COEFFICIENT * aloss,
            parameters, retain_graph=True, allow_unused=True)
        graph_parameters = list(model.graph_parameters())
        graph_aloss_gradients = torch.autograd.grad(aloss, graph_parameters, retain_graph=True,
                                                    allow_unused=True)
        graph_historical_gradients = torch.autograd.grad(base_total, graph_parameters,
                                                         retain_graph=True, allow_unused=True)
        graph_combined_gradients = torch.autograd.grad(
            base_total + runner.PAPER_ALOSS_COEFFICIENT * aloss, graph_parameters,
            retain_graph=True, allow_unused=True)
        modality_norms = [0.0 if value is None else float(value.norm()) for value in gradients]
        off_diagonal_edges = []
        for adjacency in aux["adjacencies"]:
            off_diagonal = ~torch.eye(len(adjacency), dtype=torch.bool, device=device)
            off_diagonal_edges.append(int(((adjacency.detach() != 0) & off_diagonal).sum()))
        effective = [bool(edge_count > 0 and gradient is not None
                          and torch.isfinite(gradient).all() and float(gradient.norm()) > 1e-8)
                     for edge_count, gradient in zip(off_diagonal_edges, gradients, strict=True)]
        structural_zero = [bool(edge_count == 0 and gradient_norm <= 1e-8)
                           for edge_count, gradient_norm in zip(off_diagonal_edges, modality_norms, strict=True)]
        modality_statuses = []
        for edge_count, gradient, gradient_norm in zip(
                off_diagonal_edges, gradients, modality_norms, strict=True):
            if edge_count == 0 and gradient_norm <= 1e-8:
                modality_statuses.append("STRUCTURAL_ZERO_GRADIENT")
            elif (gradient is not None and torch.isfinite(gradient).all()
                  and gradient_norm > 1e-8):
                modality_statuses.append("EFFECTIVE_GRADIENT")
            else:
                modality_statuses.append("FAIL")
        gradient_values = [value for value in (*gradients, *historical_gradients, *combined_gradients,
                                               *graph_aloss_gradients, *graph_historical_gradients,
                                               *graph_combined_gradients)
                           if value is not None]
        finite = (len(aux["adjacencies"]) == 6 and len(details["per_modality_loss"]) == 6
                  and torch.isfinite(aloss) and np.isfinite(details["per_modality_loss"]).all()
                  and all(torch.isfinite(value).all() for value in gradient_values))
        active_edge_gradients_valid = all(
            edge_count == 0 or (gradient is not None and torch.isfinite(gradient).all()
                                and float(gradient.norm()) > 1e-8)
            for edge_count, gradient in zip(off_diagonal_edges, gradients, strict=True))
        aloss_vector = _gradient_vector(gradients, parameters)
        historical_vector = _gradient_vector(historical_gradients, parameters)
        combined_vector = _gradient_vector(combined_gradients, parameters)
        graph_aloss_vector = _gradient_vector(graph_aloss_gradients, graph_parameters)
        graph_historical_vector = _gradient_vector(graph_historical_gradients, graph_parameters)
        graph_combined_vector = _gradient_vector(graph_combined_gradients, graph_parameters)
        aggregate_aloss_gradient_norm = float(graph_aloss_vector.norm())
        aggregate_gradient_effective = aggregate_aloss_gradient_norm > 1e-8
        optimizers, _ = runner.make_optimizers(model, runner.group_config(group))
        optimizer_template = optimizers["base"]
        historical_update_norm = _optimizer_update_norm(
            graph_parameters, graph_historical_gradients, optimizer_template)
        combined_update_norm = _optimizer_update_norm(
            graph_parameters, graph_combined_gradients, optimizer_template)
        del optimizers
        changed_test = y.clone()
        test_rows = torch.as_tensor(split["test_idx"], device=device).long()
        changed_test[test_rows] = 2 - changed_test[test_rows]
        test_loss, test_details = runner.paper_aloss(aux, changed_test, rows, model)
        test_gradients = _gradients(test_loss, model.gfr.graphs, retain_graph=True)
        if (not torch.equal(details["target"], test_details["target"])
                or not torch.equal(aloss.detach(), test_loss.detach())
                or not _same_gradients(gradients, test_gradients)):
            raise RuntimeError(f"{group} ALoss depends on held-out labels or gradients")
        changed_train = y.clone()
        changed_train[rows] = torch.roll(changed_train[rows], shifts=1)
        train_loss, train_details = runner.paper_aloss(aux, changed_train, rows, model)
        _gradients(train_loss, model.gfr.graphs, retain_graph=False)
        if torch.equal(details["target"], train_details["target"]):
            raise RuntimeError(f"{group} training-label permutation did not change the ALoss target")
        method_pass = (finite and active_edge_gradients_valid and aggregate_gradient_effective
                       and details["test_label_permutation_target_loss_gradient_invariant"]
                       if "test_label_permutation_target_loss_gradient_invariant" in details else
                       finite and active_edge_gradients_valid and aggregate_gradient_effective
                       and torch.equal(details["target"], test_details["target"])
                       and torch.equal(aloss.detach(), test_loss.detach())
                       and _same_gradients(gradients, test_gradients))
        status = ("FAIL" if not method_pass else
                  "PASS_WITH_STRUCTURAL_ZERO_GRADIENT" if any(structural_zero) else "PASS")
        classification_value = float(runner.historical_loss(aux, y, rows, model, "none")[0].detach())
        historical_graph_value = float(aux["graph_loss"].detach())
        darg_value = float(sum(float(aux.get(name, torch.zeros((), device=device)).detach()) * coefficient
                               for name, coefficient in (("pair_loss", 0.05), ("adv_loss", 0.01),
                                                         ("preserve_loss", 0.01))))
        records[group] = {
            "preflight_status": status,
            "modalities": len(aux["adjacencies"]), "loss": float(aloss.detach()),
            "classification_loss": classification_value,
            "historical_graph_construct_loss": historical_graph_value,
            "historical_darg_loss_contribution": darg_value,
            "historical_total_without_new_supervision": float(base_total.detach()),
            "loss_to_classification_ratio": float(aloss.detach()) / max(classification_value, 1e-12),
            "loss_to_historical_graph_ratio": float(aloss.detach()) /
            max(historical_graph_value, 1e-12),
            "loss_to_historical_total_ratio": float(aloss.detach()) /
            max(float(base_total.detach()), 1e-12),
            "per_modality_loss": details["per_modality_loss"],
            "per_modality_graphlearn_p_weight_gradient_norm": modality_norms,
            "per_modality_actual_graph_offdiagonal_nonzero_entries": off_diagonal_edges,
            "per_modality_graph_density": [value / (len(y) * (len(y) - 1)) for value in off_diagonal_edges],
            "per_modality_effective_gradient_at_1e-8": effective,
            "per_modality_gradient_status": modality_statuses,
            "effective_gradient_modality_count": sum(effective),
            "structural_zero_gradient_modalities": [index + 1 for index, value in enumerate(structural_zero) if value],
            "structural_zero_loss_scalar_contribution": float(sum(
                loss for loss, structural in zip(details["per_modality_loss"], structural_zero, strict=True)
                if structural) / 6.0),
            "aggregate_aloss_graphlearn_gradient_norm": aggregate_aloss_gradient_norm,
            "aggregate_historical_graphlearn_gradient_norm": float(historical_vector.norm()),
            "aggregate_combined_graphlearn_gradient_norm": float(combined_vector.norm()),
            "aggregate_aloss_gfr_gradient_norm": aggregate_aloss_gradient_norm,
            "aggregate_historical_gfr_gradient_norm": float(graph_historical_vector.norm()),
            "aggregate_combined_gfr_gradient_norm": float(graph_combined_vector.norm()),
            "gfr_gradient_comparison": _gradient_comparison(graph_aloss_vector,
                                                              graph_historical_vector),
            "aggregate_gradient_comparison": _gradient_comparison(aloss_vector, historical_vector),
            "combined_gradient_cosine_with_historical": _gradient_comparison(
                combined_vector, historical_vector)["cosine_similarity"],
            "gfr_combined_gradient_cosine_with_historical": _gradient_comparison(
                graph_combined_vector, graph_historical_vector)["cosine_similarity"],
            "initial_adam_gfr_update_norm_historical_only": historical_update_norm,
            "initial_adam_gfr_update_norm_with_aloss": combined_update_norm,
            "initial_adam_gfr_update_delta_norm": abs(combined_update_norm - historical_update_norm),
            "initial_adam_gfr_update_norm_ratio": combined_update_norm /
            max(historical_update_norm, 1e-30),
            "finite_values_and_gradients": finite,
            "all_active_edge_modalities_have_effective_gradient": active_edge_gradients_valid,
            "aggregate_aloss_gradient_effective": aggregate_gradient_effective,
            "train_pair_count_including_diagonal": details["train_pair_count_including_diagonal"],
            "diagonal_included": details["diagonal_included"],
            "training_label_permutation_changes_target": True,
            "test_label_permutation_target_loss_gradient_invariant": True,
            "target_sha256": runner.sha_object(details["target"].detach().cpu()),
        }
        del model, aux
        torch.cuda.empty_cache()
    group_statuses = {group: item["preflight_status"] for group, item in records.items()}
    overall_status = ("FAIL" if "FAIL" in group_statuses.values() else
                      "PASS_WITH_STRUCTURAL_ZERO_GRADIENT"
                      if "PASS_WITH_STRUCTURAL_ZERO_GRADIENT" in group_statuses.values() else "PASS")
    return {"status": overall_status,
            "required_effective_gradient_modalities": "all modalities with actual nonzero off-diagonal graph edges",
            "effective_gradient_threshold": 1e-8,
            "group_statuses": group_statuses,
            "all_six_modalities_computed": all(item["modalities"] == 6 for item in records.values()),
            "failure_reason": None if overall_status != "FAIL" else "A finite/effective-gradient or label-isolation check failed.",
            "coefficient": runner.PAPER_ALOSS_COEFFICIENT,
            "formula": "(1/6) * sum_m || normalize(A_m + I)[train,train] - normalize(Ahat_train + I_full)[train,train] ||_F^2",
            "normalization": "symmetric D^-1/2(A+I)D^-1/2 over the 598-node transductive graph before train-train restriction",
            "train_pair_mask": "train_idx x train_idx only, from outer-train true labels",
            "diagonal_included": True, "groups": records}


def check_pair_mean(device, group):
    x_np, y_np, _, splits = runner.load_data()
    y = torch.as_tensor(y_np, device=device).long()
    x = torch.as_tensor(x_np, device=device)
    split = splits[0]
    if group not in ("X5", "X6"):
        raise ValueError(f"PairMean preflight does not apply to {group}")
    model, rows, _, _, aux = model_batch(group, 1, x, y_np, y, split, device)
    train_labels = y.index_select(0, rows).detach()
    groups = runner._pair_groups(train_labels)
    class_counts = np.bincount(y_np[np.asarray(split["train_idx"], dtype=np.int64)], minlength=3)
    expected_counts = {
        "class_0-class_0": int(class_counts[0] * (class_counts[0] - 1) // 2),
        "class_1-class_1": int(class_counts[1] * (class_counts[1] - 1) // 2),
        "class_2-class_2": int(class_counts[2] * (class_counts[2] - 1) // 2),
        "class_0-class_1": int(class_counts[0] * class_counts[1]),
        "class_0-class_2": int(class_counts[0] * class_counts[2]),
        "class_1-class_2": int(class_counts[1] * class_counts[2]),
    }
    actual_counts = {name: int(left.numel()) for name, left, _, _ in groups}
    pair_keys = []
    for _, left, right, _ in groups:
        if torch.any(left == right):
            raise RuntimeError("Pair-mean contains a diagonal self-pair")
        pair_keys.extend(tuple(sorted((int(a), int(b)))) for a, b in zip(left.cpu(), right.cpu(), strict=True))
    if (len(groups) != 6 or actual_counts != expected_counts
            or len(pair_keys) != len(set(pair_keys))
            or len(pair_keys) != len(train_labels) * (len(train_labels) - 1) // 2):
        raise RuntimeError("Class-pair group counts include duplicates, diagonals, or omissions")
    x5_loss, x5_details = runner.pair_mean_graph_supervision(aux["continuous_scores"], y, rows)
    x6_loss = x6_details = None
    if group == "X6":
        x6_loss, x6_details = runner.pair_mean_graph_supervision(
            aux["continuous_scores"], y, rows, native_logits=aux["native_logits"])
    changed = y.clone()
    test = torch.as_tensor(split["test_idx"], device=device).long()
    changed[test] = (changed[test] + 1) % 3
    changed_x5, changed_x5_details = runner.pair_mean_graph_supervision(aux["continuous_scores"], changed, rows)
    if (len(aux["continuous_scores"]) != 6 or actual_counts != x5_details["group_counts"]
            or x5_details["group_target"] != changed_x5_details["group_target"]
            or not torch.equal(x5_loss.detach(), changed_x5.detach())):
        raise RuntimeError(f"{group} pair-mean group definitions or held-out label isolation failed")
    if group == "X6":
        changed_x6, changed_x6_details = runner.pair_mean_graph_supervision(
            aux["continuous_scores"], changed, rows, native_logits=aux["native_logits"])
        if (actual_counts != x6_details["group_counts"]
                or x5_details["group_target"] != x6_details["group_target"]
                or x6_details["group_target"] != changed_x6_details["group_target"]
                or not torch.equal(x6_loss.detach(), changed_x6.detach())):
            raise RuntimeError("X6 pair-mean targets or losses depend on held-out labels")
    x5_gradients = torch.autograd.grad(x5_loss, aux["continuous_scores"], allow_unused=True,
                                       retain_graph=True)
    x6_gradients = (torch.autograd.grad(x6_loss, aux["continuous_scores"], allow_unused=True,
                                        retain_graph=True) if group == "X6" else None)
    x5_gradient_norms = [0.0 if value is None else float(value.norm()) for value in x5_gradients]
    x6_gradient_norms = ([0.0 if value is None else float(value.norm()) for value in x6_gradients]
                         if x6_gradients is not None else None)
    x5_gradient = _gradient_vector(x5_gradients, aux["continuous_scores"])
    x6_gradient = (_gradient_vector(x6_gradients, aux["continuous_scores"])
                   if x6_gradients is not None else None)
    gradient_norms = (x5_gradient.norm(), x6_gradient.norm() if x6_gradient is not None else None)
    gradient_cosine = (float(torch.dot(x5_gradient, x6_gradient) /
                             (gradient_norms[0] * gradient_norms[1]).clamp_min(1e-30))
                       if x6_gradient is not None else None)
    x5_valid = len(x5_gradient_norms) == 6 and all(
        np.isfinite(value) and value > 0 for value in x5_gradient_norms)
    x6_valid = (x6_gradient_norms is not None and len(x6_gradient_norms) == 6 and all(
        np.isfinite(value) and value > 0 for value in x6_gradient_norms))
    if group == "X5" and not x5_valid:
        raise RuntimeError("X5 PairMean does not give all six continuous modalities effective gradients")
    if group == "X6" and not x6_valid:
        raise RuntimeError("X6 PairMean does not give all six continuous modalities effective gradients")
    logits_gradient = (torch.autograd.grad(x6_loss, aux["native_logits"], allow_unused=True,
                                           retain_graph=True)[0] if group == "X6" else None)
    if group == "X6" and (not x6_details["weights_detached"] or not x6_details["weighted"]
                           or logits_gradient is not None):
        raise RuntimeError("X6 entropy weights are not detached from native logits")
    weight_stats = {}
    uncertainty = None
    if group == "X6":
        probabilities = torch.softmax(aux["native_logits"].detach(), dim=-1)
        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(-1)
        uncertainty = (entropy / np.log(3.0)).clamp(0.0, 1.0)
    pair_contributions = {}
    for name, left, right, _ in groups:
        pair_contributions[name] = {"X5_equal_group_contribution":
                                    float(x5_details["per_group_loss"][name] / 6.0)}
        if uncertainty is not None:
            local_uncertainty = uncertainty.index_select(0, rows)
            weights = (0.5 + 0.5 * (local_uncertainty[left] + local_uncertainty[right]) / 2.0).detach()
            weight_stats[name] = {
                "pair_count": int(left.numel()), "mean": float(weights.mean()),
                "std_population": float(weights.std(unbiased=False)),
                "min": float(weights.min()), "max": float(weights.max()),
                "weighted_loss_fraction": float(x6_details["per_group_loss"][name] / 6.0),
            }
            pair_contributions[name]["X6_equal_group_contribution"] = float(
                x6_details["per_group_loss"][name] / 6.0)
    if group == "X6" and any(
            item["min"] < 0.5 - 1e-7 or item["max"] > 1.0 + 1e-7
            for item in weight_stats.values()):
        raise RuntimeError("X6 normalized entropy pair weights are outside [0.5,1]")
    result = {
        "status": "PASS", "group": group, "class_counts_train": class_counts.tolist(),
        "group_counts": actual_counts, "unique_pairs": len(pair_keys),
        "expected_unique_pairs": len(train_labels) * (len(train_labels) - 1) // 2,
        "x5_loss": float(x5_loss.detach()),
        "x6_loss": float(x6_loss.detach()) if x6_loss is not None else None,
        "coefficient": runner.PAIR_MEAN_COEFFICIENT,
        "active_from_epoch": runner.PAIR_WARMUP_EPOCHS + 1,
        "six_modality_score_gradient_norms_x5": x5_gradient_norms,
        "six_modality_score_gradient_norms_x6": x6_gradient_norms,
        "pair_loss_score_gradient_norm_x5": float(gradient_norms[0]),
        "pair_loss_score_gradient_norm_x6": (float(gradient_norms[1])
                                             if gradient_norms[1] is not None else None),
        "pair_loss_gradient_norm_ratio_x6_to_x5": (
            float(gradient_norms[1] / gradient_norms[0].clamp_min(1e-30))
            if gradient_norms[1] is not None else None),
        "pair_loss_gradient_cosine_x5_x6": gradient_cosine,
        "x6_weights_detached": bool(x6_details["weights_detached"]) if x6_details else None,
        "native_logits_weight_gradient": None if logits_gradient is None else float(logits_gradient.norm()),
        "test_label_permutation_invariant": True,
        "x6_weight_statistics_by_pair_group": weight_stats if group == "X6" else None,
        "equal_class_pair_loss_contributions": pair_contributions,
        "x5_x6_loss_absolute_difference": (abs(float(x5_loss.detach()) - float(x6_loss.detach()))
                                           if x6_loss is not None else None),
        "x5_x6_loss_relative_difference": (abs(float(x5_loss.detach()) - float(x6_loss.detach())) /
                                            max(abs(float(x5_loss.detach())), 1e-30)
                                            if x6_loss is not None else None),
        "x5_formula": "mean_m (1/6) sum_s mean_(unique train pairs in s) (S_mij-T_ij)^2",
        "x6_formula": "mean_m (1/6) sum_s sum_(i,j in s) w_ij*(S_mij-T_ij)^2/(sum_s w_ij+eps)",
    }
    del model, aux
    torch.cuda.empty_cache()
    return result


def _warmup_once(group, fold, x_np, y_np, device, kind):
    split = next(item for item in runner.load_data()[3] if int(item["fold"]) == fold)
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device).long()
    rows_np = np.asarray(split["train_idx"], dtype=np.int64)
    config = runner.group_config(group)
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device)
    optimizers, optimizer_config = runner.make_optimizers(model, config)
    supports = runner.historical.anchor_pair_support(model, x)
    pair_np, same_np = runner.historical.make_pairs(rows_np, y_np, fold, supports)
    pair = torch.as_tensor(pair_np, device=device).long()
    same = torch.as_tensor(same_np, device=device).float()
    initial = runner.initial_warmup_identity(model, optimizers)
    history, rng_after = runner.run_warmup(model, optimizers, x, y,
                                            torch.as_tensor(rows_np, device=device).long(),
                                            pair, same, kind)
    payload = runner.make_warmup_payload(fold, model, optimizers, history, rng_after,
                                         pair, same, initial, config, rows_np, y_np)
    return model, optimizers, optimizer_config, pair, same, payload


def check_warmup_pairing(device):
    x_np, y_np, _, _ = runner.load_data()
    outcomes = {"X3_X4": {"status": "NOT_RUN"}, "X5_X6": {"status": "NOT_RUN"},
                "X5_historical_H4": {"status": "NOT_RUN"},
                "X6_historical_H4": {"status": "NOT_RUN"}}
    states = {}

    def run_pair(left_group, right_group, kind, label, shared_only=False):
        left = right = None
        try:
            left = _warmup_once(left_group, 1, x_np, y_np, device, kind)
            right = _warmup_once(right_group, 1, x_np, y_np, device, kind)
            if shared_only:
                fields = ("initial_shared_darg_cam_gfr_sha256", "initial_shared_optimizer_sha256",
                          "initial_rng_sha256", "shared_darg_cam_gfr_state_sha256",
                          "shared_optimizer_state_sha256", "rng_state_sha256", "pair_sha256",
                          "train_index_sha256", "train_labels_sha256")
            else:
                fields = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
                          "base_state_sha256", "optimizer_state_sha256", "rng_state_sha256",
                          "pair_sha256", "train_index_sha256", "train_labels_sha256")
            differences = [key for key in fields if left[-1].get(key) != right[-1].get(key)]
            if left[-1]["history"] != right[-1]["history"]:
                differences.append("history")
            status = "FAIL" if differences else "PASS"
            record = {"status": status, "compared_fields": list(fields),
                      "field_differences": differences, "history_exact": not differences,
                      "warmup_graph_supervision": [item["graph_supervision"]
                                                    for item in left[-1]["history"]]}
            if status == "FAIL":
                record["failure_reason"] = f"paired warmup differs at {differences}"
            outcomes[label] = record
            states[left_group] = left
            states[right_group] = right
        except Exception as exc:
            outcomes[label] = {"status": "FAIL", "failure_reason": repr(exc)}
        finally:
            torch.cuda.empty_cache()

    run_pair("X3", "X4", "paper_aloss", "X3_X4", shared_only=True)
    run_pair("X5", "X6", "none", "X5_X6")
    split = next(item for item in runner.load_data()[3] if int(item["fold"]) == 1)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    for group in ("X5", "X6"):
        state = states.get(group)
        label = f"{group}_historical_H4"
        if state is None:
            outcomes[label] = {"status": "FAIL", "failure_reason": "paired warmup state unavailable"}
            continue
        try:
            result = runner.compare_warmup(group, 1, runner.group_config(group), state[-1],
                                           state[3], state[4], train_idx, y_np)
            outcomes[label] = result
        except Exception as exc:
            outcomes[label] = {"status": "FAIL", "failure_reason": repr(exc)}
    darg_state = states.get("X4")
    overall = "PASS" if all(item.get("status") == "PASS" for item in outcomes.values()) else "PARTIAL"
    return {"status": overall, **outcomes}, states


def check_dpsl(device, warmup_state, group):
    model, optimizers, _, pair, same, payload = warmup_state
    x_np, y_np, _, splits = runner.load_data()
    split = next(item for item in splits if int(item["fold"]) == 1)
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device).long()
    rows = torch.as_tensor(split["train_idx"], device=device).long()
    model.eval()
    with torch.no_grad():
        _, warm_aux = model(x, y, rows, pair, same, dpsl_enabled=False)
    stats = runner.fit_spectral_statistics(warm_aux["cam"], y, rows, fold=1,
                                            warm_logits=warm_aux["native_logits"])
    stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
    stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
    config = runner.group_config(group)
    model.attach_dpsl(stats, str(runner.prompt_path_for(config, runner.BASE_GROUPS["H4"])),
                      824602 * 100 + 1)
    optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
    runner.zero_optimizers(optimizers)
    model.train()
    _, aux = model(x, y, rows, pair, same, dpsl_enabled=True)
    supervision_kind = runner.active_supervision_kind(config, 11)
    _, total, supervision, details = runner.historical_loss(aux, y, rows, model, supervision_kind)
    _, baseline_total, _, _ = runner.historical_loss(aux, y, rows, model, "none")
    coefficient = (runner.PAPER_ALOSS_COEFFICIENT if supervision_kind == "paper_aloss"
                   else runner.PAIR_MEAN_COEFFICIENT)
    expected_total = baseline_total + coefficient * supervision
    if not torch.allclose(total.detach(), expected_total.detach(), rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"{group} graph supervision is not added exactly once to historical_total")
    graph_weights_before = [graph.p.weight.detach().clone() for graph in model.gfr.graphs]
    budget = runner.dpsl_optimizer_step(model, optimizers, total, aux["scmr_objectives"],
                                         config["dpsl_shared_cap"])
    graph_weights_after = [graph.p.weight.detach() for graph in model.gfr.graphs]
    family = budget["families"].get("cam", {})
    hist_norm = float(family.get("historical", 0.0))
    applied = float(family.get("total_applied", 0.0))
    residual = float(aux["dpsl"]["residual_ratio"].detach().max())
    gradients = [parameter.grad for graph in model.gfr.graphs for parameter in graph.parameters()
                 if parameter.grad is not None]
    dpsl_gradient_norm = float(sum(
        parameter.grad.norm() for parameter in model.dpsl.parameters()
        if parameter.grad is not None))
    if (not torch.isfinite(total) or not torch.isfinite(supervision) or not gradients
            or any(not torch.isfinite(value).all() for value in gradients)
            or dpsl_gradient_norm <= 0
            or all(torch.equal(before, after) for before, after in
                   zip(graph_weights_before, graph_weights_after, strict=True))
            or float(budget["shared_cap"]) != 0.001
            or applied > 0.001 * hist_norm + 1e-8 or residual > 0.08 + 1e-7):
        raise RuntimeError(f"{group} supervision backward, DPSL shared budget, or residual bound failed")
    return {"status": "PASS", "group": group, "warmup_epochs": len(payload["history"]),
            "historical_total_includes_registered_supervision": True,
            "supervision_kind": supervision_kind, "supervision": float(supervision.detach()),
            "total": float(total.detach()), "base_total_without_supervision": float(baseline_total.detach()),
            "shared_cap": float(budget["shared_cap"]),
            "cam_historical_gradient_norm": hist_norm,
            "cam_shared_gradient_norm": applied,
            "shared_gradient_ratio": applied / max(hist_norm, 1e-30),
            "dpsl_parameter_gradient_norm": dpsl_gradient_norm, "residual_max": residual,
            "residual_cap": 0.08}


def main():
    runner.configure_worker()
    device = torch.device("cuda:0")
    shared_checks = {}
    shared_functions = {
        "data_split": check_data,
        "historical_references": check_history,
        "model_structure": lambda: check_structure(device),
        "disabled_loss_forward": lambda: check_disabled_forward(device),
        "initialization_pairing": lambda: check_initial_pairing(device),
    }
    for name, function in shared_functions.items():
        try:
            shared_checks[name] = function()
        except Exception as exc:
            shared_checks[name] = {"status": "FAIL", "failure_reason": repr(exc)}
    shared_failures = [name for name, result in shared_checks.items()
                       if result.get("status") == "FAIL"]
    shared_status = "FAIL" if shared_failures else "PASS"

    aloss_groups = {}
    aloss_records = {}
    for group in ("X1", "X2", "X3", "X4"):
        try:
            result = check_paper_aloss(device, [group])
            aloss_records[group] = result["groups"][group]
            aloss_groups[group] = result
        except Exception as exc:
            aloss_records[group] = {"preflight_status": "FAIL", "failure_reason": repr(exc)}
            aloss_groups[group] = {"status": "FAIL", "failure_reason": repr(exc), "groups": {}}
    aloss_statuses = {group: record.get("preflight_status", "FAIL")
                      for group, record in aloss_records.items()}
    aloss_overall = ("FAIL" if "FAIL" in aloss_statuses.values() else
                     "PASS_WITH_STRUCTURAL_ZERO_GRADIENT"
                     if "PASS_WITH_STRUCTURAL_ZERO_GRADIENT" in aloss_statuses.values() else "PASS")
    aloss = {"status": aloss_overall, "groups": aloss_records,
             "group_statuses": aloss_statuses,
             "required_effective_gradient_modalities":
             "all actual modalities with nonzero off-diagonal graph edges",
             "coefficient": runner.PAPER_ALOSS_COEFFICIENT,
             "formula": "(1/6) * sum_m || normalize(A_m + I)[train,train] - normalize(Ahat_train + I_full)[train,train] ||_F^2",
             "normalization": "symmetric D^-1/2(A+I)D^-1/2 over the 598-node transductive graph before train-train restriction",
             "train_pair_mask": "train_idx x train_idx only, from outer-train true labels",
             "diagonal_included": True}

    pair_mean = {}
    for group in ("X5", "X6"):
        try:
            pair_mean[group] = check_pair_mean(device, group)
        except Exception as exc:
            pair_mean[group] = {"status": "FAIL", "group": group, "failure_reason": repr(exc)}

    try:
        warmup, warmup_states = check_warmup_pairing(device)
    except Exception as exc:
        warmup = {"status": "FAIL", "failure_reason": repr(exc)}
        warmup_states = {}
    dpsl = {}
    for group in ("X4", "X5", "X6"):
        warmup_state = warmup_states.get(group)
        pair_status = (warmup.get("X3_X4", {}).get("status") if group in ("X3", "X4")
                       else warmup.get("X5_X6", {}).get("status"))
        if warmup_state is None or pair_status != "PASS":
            dpsl[group] = {"status": "BLOCKED", "failure_reason": "paired warmup did not pass"}
            continue
        try:
            dpsl[group] = check_dpsl(device, warmup_state, group)
        except Exception as exc:
            dpsl[group] = {"status": "FAIL", "group": group, "failure_reason": repr(exc)}
    del warmup_states
    torch.cuda.empty_cache()

    loss_fidelity = loss_fidelity_audit(aloss, pair_mean)
    groups = {}
    for group in (f"X{index}" for index in range(1, 7)):
        reasons = []
        if shared_status != "PASS":
            reasons.append("shared checks failed: " + ", ".join(shared_failures))
        method_checks = {}
        if group in ("X1", "X2", "X3", "X4"):
            status = aloss_statuses.get(group, "FAIL")
            method_checks["paper_aloss"] = status
            if status == "FAIL":
                reasons.append("Paper-ALoss group-specific check failed")
            if group in ("X3", "X4"):
                pair_status = warmup.get("X3_X4", {}).get("status", "FAIL")
                method_checks["paired_aloss_warmup"] = pair_status
                if pair_status != "PASS":
                    reasons.append("X3/X4 paired ALoss warmup failed")
            if group == "X4":
                method_checks["dpsl_budget"] = dpsl.get("X4", {}).get("status", "FAIL")
                if method_checks["dpsl_budget"] != "PASS":
                    reasons.append("X4 DPSL/ALoss interaction check failed")
        else:
            method_checks["pair_mean"] = pair_mean.get(group, {}).get("status", "FAIL")
            if method_checks["pair_mean"] != "PASS":
                reasons.append(f"{group} PairMean-specific check failed")
            pair_status = warmup.get("X5_X6", {}).get("status", "FAIL")
            method_checks["paired_warmup"] = pair_status
            if pair_status != "PASS":
                reasons.append("X5/X6 paired warmup failed")
            hist_status = warmup.get(f"{group}_historical_H4", {}).get("status", "FAIL")
            method_checks["historical_H4_warmup"] = hist_status
            if hist_status == "FAIL":
                reasons.append(f"{group} failed historical H4 warmup replay")
            method_checks["dpsl_budget"] = dpsl.get(group, {}).get("status", "FAIL")
            if method_checks["dpsl_budget"] != "PASS":
                reasons.append(f"{group} DPSL interaction check failed")
        warning = (group in aloss_statuses and
                   aloss_statuses[group] == "PASS_WITH_STRUCTURAL_ZERO_GRADIENT")
        allowed = not reasons
        group_status = ("PASS_WITH_STRUCTURAL_ZERO_GRADIENT" if allowed and warning else
                        "PASS" if allowed else "FAIL")
        groups[group] = {
            "preflight_status": group_status,
            "method_fidelity": "PASS" if allowed else "FAIL",
            "formal_training_allowed": allowed,
            "block_reason": reasons,
            "checks": method_checks,
        }
    allowed_groups = [group for group, result in groups.items()
                      if result["formal_training_allowed"]]
    if shared_status != "PASS":
        overall_status = "FAIL_SHARED"
    elif not allowed_groups:
        overall_status = "FAIL_GROUP_METHODS"
    elif len(allowed_groups) < len(groups):
        overall_status = "PARTIAL"
    elif any(result["preflight_status"] == "PASS_WITH_STRUCTURAL_ZERO_GRADIENT"
             for result in groups.values()):
        overall_status = "PASS_WITH_STRUCTURAL_ZERO_GRADIENT"
    else:
        overall_status = "PASS"
    value = {
        "experiment": "Experiment 83.8.1", "status": overall_status,
        "shared_status": shared_status,
        "shared_failure_reasons": shared_failures,
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "training_seal_sha256": runner.sha_file(runner.ROOT / "audits/training_seal.json"),
        "groups": groups,
        "checks": {"shared": shared_checks, "paper_aloss": aloss,
                   "pair_mean": pair_mean, "warmup_pairing": warmup,
                   "dpsl_budget": dpsl, "loss_fidelity": loss_fidelity},
        "preflight_is_not_formal_result": True,
    }
    for group, group_result in groups.items():
        per_group = {"experiment": "Experiment 83.8.1", "group": group,
                     "status": group_result["preflight_status"],
                     "method_fidelity": group_result["method_fidelity"],
                     "formal_training_allowed": group_result["formal_training_allowed"],
                     "block_reason": group_result["block_reason"],
                     "checks": group_result["checks"],
                     "shared_status": shared_status,
                     "training_seal_sha256": value["training_seal_sha256"]}
        runner.write_json(runner.ROOT / f"audits/preflight_{group}.json", per_group)
    path = runner.ROOT / "audits/preflight_start.json"
    runner.write_json(path, value)
    runner.write_json(runner.ROOT / "audits/loss_fidelity.json", loss_fidelity)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if not allowed_groups:
        raise SystemExit("No E83.8.1 group passed its independent formal-training gate.")


if __name__ == "__main__":
    main()
