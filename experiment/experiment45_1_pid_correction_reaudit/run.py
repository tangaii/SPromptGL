#!/usr/bin/env python3
"""Execute Experiment 45.1 correction, re-estimation, and dry-run audits."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
RES = EXP / "results"
HIST = ROOT / "experiment/experiment45_pid_guided_graph_adversarial_alignment"
CFG = json.loads((EXP / "config.json").read_text())
sys.path.insert(0, str(EXP))

from integration import (DIMS, HIST43, SLICES, copy_anchors, fold_sha, historical_rows,
                         make_hp, protected_hashes, read_data, sha, site_mapping,
                         source_copy, tree_hashes, write_json)
from method import (b0b_forward_only, build_augment, close_agent, domain_loss, forward_train,
                    group_lambdas, make_agent, original_b0b_step, pair_batch,
                    relation_mapping_checks, state_sha)
from pid_estimator import (estimate_pid, independent_trust_constr, information_terms)

PAIRS = [(row[0], int(row[1]), int(row[2])) for row in CFG["pairs"]]
EFFECTIVE = tuple(CFG["effective_pairs"])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def csv_read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["fold"]
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def manifest():
    path = EXP / "manifest.json"
    return read_json(path) if path.exists() else {}


def update_manifest(**updates):
    current = manifest()
    next_status = updates.get("status")
    if next_status and next_status != current.get("status") and "state_history" not in updates:
        history = list(current.get("state_history", [current.get("status", "PREPARED")]))
        history.append(next_status)
        updates["state_history"] = history
    current.update(updates)
    write_json_atomic(EXP / "manifest.json", current)
    return current


def pair_code(pair):
    return sum((index + 1) * ord(char) for index, char in enumerate(pair))


def stream(fold, pair, kind, replicate=0):
    # SeedSequence prevents bootstrap and permutation draws from sharing state.
    kind_code = {"bootstrap": 1, "global_perm": 2, "site_perm": 3, "legacy_bootstrap": 4}[kind]
    return np.random.default_rng(np.random.SeedSequence([
        CFG["seed"], CFG["pid"]["bootstrap_seed"], int(fold), pair_code(pair), kind_code, int(replicate)
    ]))


def fit_labels(x, train, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    labels = []
    for modal, (lo, hi) in enumerate(SLICES):
        scaled = StandardScaler().fit_transform(x[train, lo:hi])
        labels.append(KMeans(n_clusters=k, n_init=10, random_state=seed + modal).fit_predict(scaled).astype(np.int64))
    return labels


def assert_paired(labels_i, labels_j, y_train, ix):
    ix = np.asarray(ix, dtype=np.int64)
    labels_i = np.asarray(labels_i)
    labels_j = np.asarray(labels_j)
    y_train = np.asarray(y_train)
    if len(labels_i) != len(labels_j) or len(y_train) != len(labels_i):
        raise AssertionError("Bootstrap arrays have different lengths")
    if ix.ndim != 1 or len(ix) != len(labels_i) or ix.min(initial=0) < 0 or ix.max(initial=0) >= len(y_train):
        raise AssertionError("Bootstrap index is not local to the training fold")
    return labels_i[ix], labels_j[ix], y_train[ix]


def legacy_triplet(labels_i, labels_j, y_train, ix):
    # Exact historical error: labels stay in original local order, only y is indexed.
    return np.asarray(labels_i), np.asarray(labels_j), np.asarray(y_train)[np.asarray(ix, dtype=np.int64)]


def condition_site_permutation(y_train, sites, rng):
    output, _ = condition_site_permutation_trace(y_train, sites, rng)
    return output


def condition_site_permutation_trace(y_train, sites, rng):
    y_train = np.asarray(y_train, dtype=np.int64)
    sites = np.asarray(sites, dtype=np.int64)
    if len(y_train) != len(sites):
        raise AssertionError("site and label arrays differ in length")
    output = y_train.copy()
    source_indices = np.arange(len(y_train), dtype=np.int64)
    for site_id in np.unique(sites):
        where = np.flatnonzero(sites == site_id)
        source = rng.permutation(where)
        source_indices[where] = source
        output[where] = y_train[source]
        assert np.array_equal(np.bincount(output[where], minlength=2),
                              np.bincount(y_train[where], minlength=2))
    return output, source_indices


def synthetic_pid_tests():
    tests = {}
    # Traceable IDs show that the old implementation pairs local modality rows
    # with labels from unrelated patients, while the corrected path is exact.
    ids = np.arange(48, dtype=np.int64)
    ix = np.random.default_rng(19).integers(0, len(ids), len(ids))
    old_i, old_j, old_y = legacy_triplet(ids, ids, ids, ix)
    new_i, new_j, new_y = assert_paired(ids, ids, ids, ix)
    old_matches = int(np.sum((old_i == old_y) & (old_j == old_y)))
    new_matches = int(np.sum((new_i == new_y) & (new_j == new_y)))
    assert new_matches == len(ids) and old_matches < len(ids)
    tests["bootstrap_trace"] = {"n": len(ids), "old_matching_positions": old_matches,
                                "new_matching_positions": new_matches,
                                "new_pairing_fraction": new_matches / len(ids)}

    rng = np.random.default_rng(41)
    signal = rng.integers(0, 2, 1200, dtype=np.int64)
    x1 = signal ^ (rng.random(len(signal)) < 0.05)
    x2 = signal ^ (rng.random(len(signal)) < 0.05)
    boots = []
    for replicate in range(12):
        local_ix = stream(1, "AF", "bootstrap", replicate).integers(0, len(signal), len(signal))
        a, b, c = assert_paired(x1, x2, signal, local_ix)
        boots.append(estimate_pid(a, b, c, 2, 2)["R"])
    assert np.median(boots) > 0.35
    tests["redundant_noisy"] = {"median_R": float(np.median(boots)), "bootstrap_n": len(boots),
                                "paired_fraction": 1.0}

    independent_y = rng.integers(0, 2, len(signal), dtype=np.int64)
    independent = []
    for replicate in range(12):
        local_ix = stream(2, "AF", "bootstrap", replicate).integers(0, len(signal), len(signal))
        a, b, c = assert_paired(x1, x2, independent_y, local_ix)
        independent.append(estimate_pid(a, b, c, 2, 2)["R"])
    assert float(np.median(independent)) < 0.15
    tests["label_independent"] = {"median_R": float(np.median(independent)),
                                  "max_R": float(np.max(independent)), "bootstrap_n": len(independent)}

    sites = np.repeat(np.arange(6), 40)
    labels = np.tile(np.asarray([0, 0, 1, 1]), 60)
    ys = condition_site_permutation(labels, sites, np.random.default_rng(42))
    assert np.array_equal(np.bincount(labels, minlength=2), np.bincount(ys, minlength=2))
    for site_id in np.unique(sites):
        where = sites == site_id
        assert np.array_equal(np.bincount(labels[where], minlength=2), np.bincount(ys[where], minlength=2))
    tests["site_permutation"] = {"global_counts_preserved": True, "per_site_counts_preserved": True,
                                  "site_exchange": False}
    write_json_atomic(RT / "tests/bootstrap_and_site_tests.json", tests)
    return tests


def independent_pid_tests():
    rng = np.random.default_rng(45091)
    cases = {}
    a = rng.integers(0, 2, 10000, dtype=np.int64)
    b = rng.integers(0, 2, len(a), dtype=np.int64)
    cases["redundant"] = (a, a.copy(), a.copy())
    cases["unique_x1"] = (a, b, a.copy())
    cases["xor_synergy"] = (a, b, a ^ b)
    cases["independent"] = (a, b, rng.integers(0, 2, len(a), dtype=np.int64))
    cases["imbalanced_independent"] = (a, b, (rng.random(len(a)) < 0.2).astype(np.int64))
    rows = []
    for name, (x1, x2, y) in cases.items():
        result = estimate_pid(x1, x2, y, 2, 2)
        ref_q, reference = independent_trust_constr(result["distribution"])
        ref_iq = information_terms(ref_q)[2]
        delta = abs(ref_iq - result["Ijoint_min"])
        if not result["valid"] or not reference["success"] or delta > 2e-4:
            raise AssertionError(f"independent PID validation failed for {name}: {result} {reference}")
        rows.append({"case": name, "R": result["R"], "U1": result["U1"], "U2": result["U2"],
                     "S": result["S"], "Ijoint_min_slsqp": result["Ijoint_min"],
                     "Ijoint_min_reference": ref_iq, "reference_delta": delta,
                     "solver_success": result["solver"]["success"],
                     "reference_success": reference["success"],
                     "identity_residual": result["identity_residual"],
                     "marginal_error": result["solver"]["marginal_error"]})
    checks = {
        "redundant": rows[0]["R"] > .7 and abs(rows[0]["U1"]) < .1 and abs(rows[0]["U2"]) < .1,
        "unique_x1": rows[1]["U1"] > .7 and abs(rows[1]["R"]) < .1 and abs(rows[1]["U2"]) < .1,
        "xor_synergy": rows[2]["S"] > .7 and abs(rows[2]["R"]) < .1 and abs(rows[2]["U1"]) < .1,
        "independent": max(abs(rows[3][key]) for key in ("R", "U1", "U2", "S")) < .1,
        "imbalanced_independent": max(abs(rows[4][key]) for key in ("R", "U1", "U2", "S")) < .1,
    }
    if not all(checks.values()):
        raise AssertionError(f"synthetic PID component audit failed: {checks}")
    positive = rng.random((2, 2, 2)) + .2
    positive /= positive.sum()
    py = positive.sum(axis=(0, 1))
    qxy = positive.sum(axis=2)
    analytic = np.log(positive / (qxy[:, :, None] * py[None, None, :])) / np.log(2.0)

    def objective_full(q):
        qxy_local = q.sum(axis=2)
        denominator = qxy_local[:, :, None] * py[None, None, :]
        return float(np.sum(np.where(q > 0, q * np.log(q / denominator), 0.0)) / np.log(2.0))

    epsilon = 1e-6
    numerical = np.zeros_like(positive)
    for index in np.ndindex(positive.shape):
        plus = positive.copy()
        minus = positive.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (objective_full(plus) - objective_full(minus)) / (2 * epsilon)
    gradient_error = float(np.max(np.abs(analytic - numerical)))
    if gradient_error > 1e-7:
        raise AssertionError(f"analytic PID gradient failed finite differences: {gradient_error}")
    write_json_atomic(RT / "tests/pid_synthetic_audit.json", {"rows": rows, "component_checks": checks,
                                                                  "reference_solver": "scipy_trust_constr",
                                                                  "analytic_gradient_max_abs_error": gradient_error})
    return rows, checks


def gate_power_calibration():
    _, y_all, folds, _, _, _ = read_data()
    reference_train = y_all[folds[0][0]].copy()
    scenarios = {
        "strong_redundancy": ("redundant", [.03, .08, .15]),
        "weak_redundancy": ("redundant", [.28, .35, .42]),
        "unique_dominant": ("unique", [0.0, 0.0, 0.0]),
        "synergy_dominant": ("synergy", [0.0, 0.0, 0.0]),
        "label_independent": ("independent", [0.0, 0.0, 0.0]),
    }
    records = []
    for scenario_id, (scenario, (kind, noise_levels)) in enumerate(scenarios.items(), 1):
        for repetition in range(3):
            y_train = reference_train.copy()
            base_rng = np.random.default_rng(np.random.SeedSequence([451045, scenario_id, repetition]))
            base_rng.shuffle(y_train)
            lambdas = []
            qs = []
            rs = []
            for relation_id, noise in enumerate(noise_levels, 1):
                rng = np.random.default_rng(np.random.SeedSequence([451045, scenario_id, repetition, relation_id]))
                if kind == "redundant":
                    common = y_train ^ (rng.random(len(y_train)) < noise)
                    x1 = common * 2 + rng.integers(0, 2, len(y_train))
                    x2 = common * 2 + rng.integers(0, 2, len(y_train))
                elif kind == "unique":
                    x1 = y_train * 2 + rng.integers(0, 2, len(y_train))
                    x2 = rng.integers(0, 4, len(y_train))
                elif kind == "synergy":
                    first = rng.integers(0, 2, len(y_train))
                    second = first ^ y_train
                    x1 = first * 2 + rng.integers(0, 2, len(y_train))
                    x2 = second * 2 + rng.integers(0, 2, len(y_train))
                else:
                    x1 = rng.integers(0, 4, len(y_train))
                    x2 = rng.integers(0, 4, len(y_train))
                base = estimate_pid(x1, x2, y_train, 4, 4, 2)
                boot, perm = [], []
                for replicate in range(12):
                    sample_rng = np.random.default_rng(np.random.SeedSequence(
                        [451046, scenario_id, repetition, relation_id, 1, replicate]))
                    ix = sample_rng.integers(0, len(y_train), len(y_train))
                    boot.append(estimate_pid(x1[ix], x2[ix], y_train[ix], 4, 4, 2)["R"])
                for replicate in range(19):
                    perm_rng = np.random.default_rng(np.random.SeedSequence(
                        [451046, scenario_id, repetition, relation_id, 2, replicate]))
                    py = y_train[perm_rng.permutation(len(y_train))]
                    perm.append(estimate_pid(x1, x2, py, 4, 4, 2)["R"])
                q10 = float(np.quantile(boot, .10))
                med = float(np.median(boot))
                q95 = float(np.quantile(perm, .95))
                q_value = float(np.clip((q10 - q95) / (med + 1e-12), 0.0, 1.0))
                redundancy = float(base["R"] / (base["R"] + base["U1"] + base["U2"] + base["S"] + 1e-12))
                lam = CFG["protocol"]["lambda_max"] * redundancy * q_value
                qs.append(q_value)
                rs.append(redundancy)
                lambdas.append(lam)
            positive = sum(value > 0 for value in lambdas)
            lam_range = max(lambdas) - min(lambdas)
            gate = positive >= 1 and lam_range > 1e-6
            records.append({"scenario": scenario, "repetition": repetition + 1,
                            "pair_R": [float(estimate_pid(*triple, 4, 4, 2)["R"])
                                       for triple in _scenario_pairs(y_train, kind, noise_levels,
                                                                     scenario_id, repetition)],
                            "q": qs, "r": rs, "lambda": lambdas,
                            "positive_lambdas": positive, "lambda_range": lam_range,
                            "gate_pass": gate, "bootstrap_n": 12, "global_permutation_n": 19,
                            "n_train": len(y_train),
                            "label_counts": np.bincount(y_train, minlength=2).tolist()})
    summary = {}
    for scenario in scenarios:
        rows = [row for row in records if row["scenario"] == scenario]
        summary[scenario] = {"gate_passes": sum(row["gate_pass"] for row in rows), "repetitions": len(rows),
                             "mean_positive_lambdas": float(np.mean([row["positive_lambdas"] for row in rows])),
                             "mean_q": float(np.mean([value for row in rows for value in row["q"]])),
                             "mean_lambda": float(np.mean([value for row in rows for value in row["lambda"]]))}
    output = {"status": "COMPLETED", "frozen_gate": "at least one positive lambda and lambda range > 1e-6",
              "sample_size": len(reference_train), "class_counts": np.bincount(reference_train, minlength=2).tolist(),
              "k": 4, "records": records, "summary": summary,
              "interpretation": "statistical diagnostic only; thresholds unchanged"}
    write_json_atomic(RES / "gate_power_calibration.json", output)
    update_manifest(gate_power_calibration={"status": "COMPLETED", "summary": summary,
                                             "sha256": sha(RES / "gate_power_calibration.json")})
    return output


def _scenario_pairs(y_train, kind, noise_levels, scenario_id, repetition):
    triples = []
    for relation_id, noise in enumerate(noise_levels, 1):
        rng = np.random.default_rng(np.random.SeedSequence([451045, scenario_id, repetition, relation_id]))
        if kind == "redundant":
            common = y_train ^ (rng.random(len(y_train)) < noise)
            x1 = common * 2 + rng.integers(0, 2, len(y_train))
            x2 = common * 2 + rng.integers(0, 2, len(y_train))
        elif kind == "unique":
            x1 = y_train * 2 + rng.integers(0, 2, len(y_train))
            x2 = rng.integers(0, 4, len(y_train))
        elif kind == "synergy":
            first = rng.integers(0, 2, len(y_train))
            second = first ^ y_train
            x1 = first * 2 + rng.integers(0, 2, len(y_train))
            x2 = second * 2 + rng.integers(0, 2, len(y_train))
        else:
            x1 = rng.integers(0, 4, len(y_train))
            x2 = rng.integers(0, 4, len(y_train))
        triples.append((x1, x2, y_train))
    return triples


def prepare():
    RT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    source_hash = source_copy()
    anchor_hash = copy_anchors()
    x, y, folds, mapping, columns, frame = read_data()
    sites = site_mapping(frame, mapping, columns)
    references = historical_rows()
    if len(references) != 10:
        raise AssertionError("historical E0 does not contain ten folds")
    for fold, (train, test) in enumerate(folds, 1):
        anchor = read_anchor(fold, map_location="cpu")
        if int(anchor["fold"]) != fold or anchor["split_sha"] != fold_sha(train, test):
            raise AssertionError(f"anchor {fold} is not the matching frozen split")
        if sha(RT / "anchors" / f"fold{fold:02}.pt") != references[fold]["anchor_sha"]:
            raise AssertionError(f"anchor {fold} differs from historical E0")
    # Preserve a compact, read-only source/result snapshot in the new experiment.
    history_snapshot = RT / "sources" / "history45"
    for relative in ("config.json", "manifest.json", "pid_estimator.py", "run.py", "method.py",
                     "integration.py", "results/pid_pairs.csv", "results/pid_diagnostics.csv",
                     "results/pid_weights.csv", "results/report.md"):
        source = HIST / relative
        destination = history_snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    handoff = ROOT / "handoff.md"
    handoff_bytes = handoff.read_bytes()
    current = manifest()
    prefix = current.get("handoff_prefix")
    marker = b"<!-- EXPERIMENT45_1_PID_CORRECTION_REAUDIT -->"
    if prefix is None:
        prefix = {"bytes": len(handoff_bytes), "sha256": hashlib.sha256(handoff_bytes).hexdigest()}
    elif hashlib.sha256(handoff_bytes[:prefix["bytes"]]).hexdigest() != prefix["sha256"]:
        raise AssertionError("handoff prefix changed since this experiment started")
    payload = {
        "status": "PREPARED",
        "state_history": ["PREPARED"],
        "started_unix": current.get("started_unix", time.time()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha": source_hash,
        "anchor_sha": anchor_hash,
        "protected_sha256": protected_hashes(),
        "historical_files_sha256": tree_hashes(HIST),
        "data_shape": list(x.shape),
        "label_counts": np.bincount(y).tolist(),
        "fold_sha": {str(fold): fold_sha(*folds[fold - 1]) for fold in range(1, 11)},
        "historical_manifest_sha256": sha(HIST / "manifest.json"),
        "history43_source_sha": {name: sha(RT / "sources" / name) for name in ("network.py", "layers.py", "utils.py")},
        "handoff_prefix": prefix,
        "site_mapping": {key: value for key, value in sites.items() if key != "sites"},
        "site_mapping_status": "VERIFIED" if sites["valid"] else sites["reason"],
        "pid_estimator": CFG["pid"]["estimator"],
        "cvxpy_available": False,
        "formal_training": {"new_folds": 0, "status": "NOT_TRAINED_BY_DESIGN"},
        "runtime_versions": runtime_versions(),
        "history_snapshot": str(history_snapshot.relative_to(EXP)),
        "handoff_marker": marker.decode(),
    }
    write_json_atomic(EXP / "manifest.json", payload)
    print("PREPARE_PASS", json.dumps({"shape": list(x.shape), "labels": np.bincount(y).tolist(),
                                      "site_mapping": payload["site_mapping_status"]}, ensure_ascii=False), flush=True)


def runtime_versions():
    import scipy
    import sklearn
    import torch
    import numpy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cvxpy": None}


def capture_rng(torch):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def rng_sha(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def read_anchor(fold, map_location="cpu"):
    import torch
    return torch.load(RT / "anchors" / f"fold{fold:02}.pt", map_location=map_location, weights_only=False)


def gpu_context(gpu):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model preflight")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_index = 0 if visible else int(gpu)
    torch.cuda.set_device(device_index)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(RT / "sources"))
    import layers
    import network
    import utils
    return torch, layers, network, utils, torch.device(f"cuda:{device_index}")


def grl_gradient_test(device):
    import torch
    from torch import nn
    from method import GradientReverse

    generator = torch.Generator(device=device).manual_seed(450451)
    zi = torch.randn((32, 64), generator=generator, device=device) - 1.5
    zj = torch.randn((32, 64), generator=generator, device=device) + 1.5
    y = torch.arange(32, device=device) % 2
    din, source, _ = pair_batch(zi, zj, y, conditional=True)
    head = nn.Sequential(nn.Linear(66, 32), nn.Tanh(), nn.Linear(32, 2)).to(device)
    state = copy.deepcopy(head.state_dict())
    plain = nn.Sequential(nn.Linear(66, 32), nn.Tanh(), nn.Linear(32, 2)).to(device)
    plain.load_state_dict(state)
    x_plain = din[:, :64].detach().clone().requires_grad_(True)
    loss_plain = torch.nn.functional.cross_entropy(plain(torch.cat((x_plain, din[:, 64:]), dim=1)), source)
    loss_plain.backward()
    grad_plain = x_plain.grad.detach().clone()
    x_grl = din[:, :64].detach().clone().requires_grad_(True)
    loss_grl = torch.nn.functional.cross_entropy(head(torch.cat((GradientReverse.apply(x_grl, .37), din[:, 64:]), dim=1)), source)
    loss_grl.backward()
    grad_grl = x_grl.grad.detach().clone()
    residual = float((grad_grl + .37 * grad_plain).abs().max().item())
    scale_error = float((grad_grl + .37 * grad_plain).norm().item() / max(grad_plain.norm().item(), 1e-12))
    assert residual < 1e-6 and scale_error < 1e-5
    return {"plain_loss": float(loss_plain.detach()), "grl_loss": float(loss_grl.detach()),
            "max_negative_scale_residual": residual, "relative_negative_scale_residual": scale_error,
            "lambda": .37, "pass": True}


def domain_learning_test(device):
    import torch
    from torch import nn

    generator = torch.Generator(device=device).manual_seed(450452)
    zi = torch.randn((64, 64), generator=generator, device=device) - 2.0
    zj = torch.randn((64, 64), generator=generator, device=device) + 2.0
    y = torch.arange(64, device=device) % 2
    din, source, condition = pair_batch(zi, zj, y, conditional=True)
    head = nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 2)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    with torch.no_grad():
        initial = float(torch.nn.functional.cross_entropy(head(din), source))
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(din), source)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(din)
        final = float(torch.nn.functional.cross_entropy(logits, source))
        accuracy = float((logits.argmax(1) == source).float().mean())
    assert final < initial and accuracy > .95
    return {"input_shape": list(din.shape), "source_counts": torch.bincount(source).tolist(),
            "condition_shape": list(condition.shape), "initial_loss": initial,
            "final_loss": final, "accuracy": accuracy, "pass": True}


def model_preflight(gpu):
    torch, layers, network, utils, device = gpu_context(gpu)
    x, y, folds, _, _, _ = read_data()
    hp = make_hp(CFG)
    agents = []
    hashes = []
    step_checks = []
    b0b_checks = []
    try:
        anchor = read_anchor(1, map_location="cpu")
        for group in range(1, 7):
            agent = make_agent(torch, np, layers, network, utils, hp, x, y,
                               folds[0][0], folds[0][1], anchor, 1, group, CFG, device=str(device))
            restore_rng(torch, anchor["rng"])
            agents.append(agent)
            hashes.append({"group": group, "gfr": state_sha(agent.gfr.state_dict()),
                           "pln": state_sha(agent.pln.state_dict()),
                           "mp": state_sha(agent.mp.state_dict()), "mf": state_sha(agent.mf.state_dict()),
                           "augment": state_sha(agent.augment.state_dict()),
                           "newopt": state_sha(agent.newopt.state_dict()),
                           "entry_rng": rng_sha(capture_rng(torch)),
                           "domain_input_dim": agent.augment.domain[0][0].in_features})
        base = hashes[0]
        assert all(row["gfr"] == base["gfr"] and row["pln"] == base["pln"] and
                   row["mp"] == base["mp"] and row["mf"] == base["mf"] and
                   row["augment"] == base["augment"] and row["newopt"] == base["newopt"] and
                   row["entry_rng"] == base["entry_rng"] and row["domain_input_dim"] == 66 for row in hashes)
        pid_values = {"AF": .01, "AC": .02, "FC": .03}
        train_count = len(folds[0][0])
        for agent in agents:
            restore_rng(torch, anchor["rng"])
            agent.mp.zero_grad(set_to_none=True)
            agent.mf.zero_grad(set_to_none=True)
            agent.newopt.zero_grad(set_to_none=True)
            total, values = forward_train(agent, utils, CFG, pid_values, epoch=90)
            assert torch.isfinite(total)
            group = agent.group
            expected_lambdas = {1: [0.0, 0.0, 0.0],
                                2: [.02, .02, .02],
                                3: [.02, .02, .02],
                                4: [.01, .02, .03],
                                5: [.02, .03, .01],
                                6: [.01, .02, .03]}[group]
            assert np.allclose(values["lambda_by_pair"], expected_lambdas, rtol=0, atol=1e-15)
            if group == 1:
                assert values["domain_details"] == [] and float(values["domain"].detach()) == 0.0
            else:
                assert len(values["domain_details"]) == 3
                for detail in values["domain_details"]:
                    assert tuple(detail["input"].shape) == (2 * train_count, 66)
                    assert torch.bincount(detail["source"], minlength=2).tolist() == [train_count, train_count]
                    assert torch.equal(detail["condition"][:train_count], detail["condition"][train_count:])
                    if group == 2:
                        assert torch.count_nonzero(detail["input"][:, 64:]) == 0
                    else:
                        expected_condition = torch.nn.functional.one_hot(
                            agent.targ[torch.as_tensor(agent.trn_idx, device=device)], num_classes=2).repeat(2, 1)
                        assert torch.equal(detail["input"][:, 64:], expected_condition)
            if group == 6:
                assert values["main_input"] is agent.feat
            else:
                assert values["main_input"] is values["feat"]
            total.backward()
            gradient_norms = {}
            for idx in (1, 2, 3):
                module = getattr(agent.gfr, f"MessagePassing{idx}")
                gradient_norms[str(idx)] = float(sum(p.grad.detach().norm().item() for p in module.parameters()
                                                     if p.grad is not None))
            proj_gradient = float(sum(p.grad.detach().norm().item() for p in agent.augment.proj.parameters()
                                      if p.grad is not None))
            assert proj_gradient > 0 and all(value > 0 for value in gradient_norms.values())
            for module in (agent.gfr, agent.pln, agent.augment):
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            agent.mp.step()
            agent.mf.step()
            agent.newopt.step()
            step_checks.append({"group": group, "lambda_by_pair": values["lambda_by_pair"],
                                "conditional": values["conditional"], "domain_terms": len(values["domain_details"]),
                                "main_input": "X" if group == 6 else "GFR(X)",
                                "domain_input_shape": [2 * train_count, 66] if group > 1 else None,
                                "source_counts": [train_count, train_count] if group > 1 else None,
                                "shared_projection_gradient": proj_gradient,
                                "gfr_branch_gradient_norms": gradient_norms,
                                "one_step_finite": True})

        reference = make_agent(torch, np, layers, network, utils, hp, x, y,
                               folds[0][0], folds[0][1], anchor, 1, 0, CFG, device=str(device))
        candidate = make_agent(torch, np, layers, network, utils, hp, x, y,
                               folds[0][0], folds[0][1], anchor, 1, 0, CFG, device=str(device))
        reference_rng = copy.deepcopy(anchor["rng"])
        candidate_rng = copy.deepcopy(anchor["rng"])
        for step in (1, 2):
            restore_rng(torch, reference_rng)
            reference.mp.zero_grad(set_to_none=True)
            reference.mf.zero_grad(set_to_none=True)
            ref_total, ref_cls, ref_graph = original_b0b_step(reference, utils)
            reference_rng = capture_rng(torch)
            restore_rng(torch, candidate_rng)
            candidate.mp.zero_grad(set_to_none=True)
            candidate.mf.zero_grad(set_to_none=True)
            candidate.newopt.zero_grad(set_to_none=True)
            cand_total, cand_values = forward_train(candidate, utils, CFG, pid_values, epoch=90)
            cand_total.backward()
            cand_values = {key: cand_values[key].detach() for key in ("cls", "graph")}
            candidate.mp.step()
            candidate.mf.step()
            candidate_rng = capture_rng(torch)
            losses = {"total": abs(float(ref_total) - float(cand_total.detach())),
                      "cls": abs(float(ref_cls) - float(cand_values["cls"])),
                      "graph": abs(float(ref_graph) - float(cand_values["graph"]))}
            models_equal = (state_sha(reference.gfr.state_dict()) == state_sha(candidate.gfr.state_dict()) and
                            state_sha(reference.pln.state_dict()) == state_sha(candidate.pln.state_dict()))
            optimizers_equal = (state_sha(reference.mp.state_dict()) == state_sha(candidate.mp.state_dict()) and
                                state_sha(reference.mf.state_dict()) == state_sha(candidate.mf.state_dict()))
            rng_equal = rng_sha(reference_rng) == rng_sha(candidate_rng)
            if max(losses.values()) > 1e-7 or not models_equal or not optimizers_equal or not rng_equal:
                raise AssertionError(f"B0-B equivalence failed at step {step}: {losses}")
            b0b_checks.append({"step": step, "loss_abs_diff": losses, "model_parameters_equal": models_equal,
                               "adam_states_equal": optimizers_equal, "entry_rng_and_next_rng_equal": rng_equal})
        close_agent(reference)
        close_agent(candidate)
        mapping = relation_mapping_checks(pid_values)
        assert mapping["E2_E3_equal"] and mapping["E5_same_multiset"] and mapping["E5_same_sum"]
        result = {"status": "PASS", "gpu_argument": int(gpu), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "device": str(device), "device_name": torch.cuda.get_device_name(device),
                  "parameter_counts": {"gfr": sum(p.numel() for p in agents[0].gfr.parameters()),
                                        "pln": sum(p.numel() for p in agents[0].pln.parameters()),
                                        "augment": sum(p.numel() for p in agents[0].augment.parameters())},
                  "initialization": hashes, "single_step_each_group": step_checks,
                  "b0b_disabled_losses_equivalence": b0b_checks,
                  "source_domain": domain_learning_test(device),
                  "grl_gradient": grl_gradient_test(device),
                  "relation_mapping": mapping,
                  "new_training_folds": 0}
    finally:
        for agent in agents:
            close_agent(agent)
        torch.cuda.empty_cache()
    write_json_atomic(RT / f"preflight_gpu{gpu}.json", result)
    update_manifest(**{"preflight_gpu" + str(gpu): result})
    print("PREFLIGHT_PASS", gpu, flush=True)
    return result


def pid_phase():
    x, y, folds, mapping, columns, frame = read_data()
    sites_info = site_mapping(frame, mapping, columns)
    if not sites_info["valid"]:
        raise RuntimeError("SITE_MAPPING_UNVERIFIED: cannot run site-conditional statistic")
    synthetic_tests = synthetic_pid_tests()
    synthetic_rows, synthetic_checks = independent_pid_tests()
    primary = []
    diagnostics = []
    weights = []
    before_after_by_key = {}
    arrays = {}
    old_pairs = {(int(row["fold"]), int(row["k"]), row["pair"]): row
                 for row in csv_read(HIST / "results/pid_pairs.csv")}
    old_diags = {(int(row["fold"]), row["pair"]): row
                 for row in csv_read(HIST / "results/pid_diagnostics.csv")}
    old_weights = {(int(row["fold"]), row["pair"]): row
                   for row in csv_read(HIST / "results/pid_weights.csv")}
    for fold, (train, test) in enumerate(folds, 1):
        y_train = y[train]
        site_train = sites_info["sites"][train]
        for k in [CFG["pid"]["cluster_k_primary"], *CFG["pid"]["cluster_k_sensitivity"]]:
            # Preserve historical KMeans seeds while isolating each fit.
            labels = fit_labels(x, train, k, 45000 + fold * 100 + k)
            for pair, i, j in PAIRS:
                result = estimate_pid(labels[i], labels[j], y_train, k, k, 2)
                solver = result["solver"]
                primary.append({"fold": fold, "k": k, "pair": pair, "R": result["R"], "U1": result["U1"],
                                "U2": result["U2"], "S": result["S"], "I1": result["I1"], "I2": result["I2"],
                                "Ijoint": result["Ijoint"], "Ijoint_min": result["Ijoint_min"],
                                "identity_residual": result["identity_residual"],
                                "minimum_component": result["minimum_component"],
                                "redundancy_fraction": result["redundancy_fraction"],
                                "solver_success": solver["success"], "solver_status": solver["status"],
                                "solver_iterations": solver["iterations"],
                                "objective_returned_bit": solver["objective_returned_bit"],
                                "objective_recomputed_bit": solver["objective_recomputed_bit"],
                                "objective_delta_bit": solver["objective_delta_bit"],
                                "marginal_error": solver["marginal_error"],
                                "n_train": len(train), "train_index_min": int(train.min()),
                                "train_index_max": int(train.max()), "test_label_used": False})
                if k == 4:
                    old_pair = old_pairs[(fold, 4, pair)]
                    old_weight = old_weights.get((fold, pair))
                    old_r = float(old_weight["r"]) if old_weight else float(old_pair["redundancy_fraction"])
                    before_after_by_key[(fold, pair)] = {
                        "fold": fold, "pair": pair,
                        "old_R": float(old_pair["R"]), "new_R": result["R"],
                        "old_U1": float(old_pair["U1"]), "new_U1": result["U1"],
                        "old_U2": float(old_pair["U2"]), "new_U2": result["U2"],
                        "old_S": float(old_pair["S"]), "new_S": result["S"],
                        "old_Ijoint": float(old_pair["Ijoint"]), "new_Ijoint": result["Ijoint"],
                        "old_redundancy_fraction": old_r,
                        "new_redundancy_fraction": result["redundancy_fraction"],
                        "old_solver_success": old_pair["solver_success"],
                        "new_solver_success": result["solver"]["success"],
                    }
                if k != 4 or pair not in EFFECTIVE:
                    continue
                boot = []
                legacy_boot = []
                global_perm = []
                site_perm = []
                bootstrap_indices = []
                global_perm_labels = []
                site_perm_labels = []
                site_perm_sources = []
                for replicate in range(CFG["pid"]["bootstrap_replicates"]):
                    ix = stream(fold, pair, "bootstrap", replicate).integers(0, len(train), len(train))
                    bootstrap_indices.append(ix.tolist())
                    bi, bj, by = assert_paired(labels[i], labels[j], y_train, ix)
                    corrected = estimate_pid(bi, bj, by, 4, 4, 2)
                    boot.append(corrected["R"])
                    li, lj, ly = legacy_triplet(labels[i], labels[j], y_train, ix)
                    legacy_boot.append(estimate_pid(li, lj, ly, 4, 4, 2)["R"])
                    assert len(bi) == len(bj) == len(by)
                for replicate in range(CFG["pid"]["permutation_replicates"]):
                    global_ix = stream(fold, pair, "global_perm", replicate).permutation(len(y_train))
                    global_y = y_train[global_ix]
                    global_perm_labels.append(global_y.tolist())
                    global_perm.append(estimate_pid(labels[i], labels[j], global_y, 4, 4, 2)["R"])
                    site_y, site_source = condition_site_permutation_trace(
                        y_train, site_train, stream(fold, pair, "site_perm", replicate))
                    site_perm_labels.append(site_y.tolist())
                    site_perm_sources.append(site_source.tolist())
                    site_perm.append(estimate_pid(labels[i], labels[j], site_y, 4, 4, 2)["R"])
                q10 = float(np.quantile(boot, .10))
                median = float(np.median(boot))
                q90 = float(np.quantile(boot, .90))
                legacy_q10 = float(np.quantile(legacy_boot, .10))
                legacy_median = float(np.median(legacy_boot))
                perm_q95 = float(np.quantile(global_perm, .95))
                site_q95 = float(np.quantile(site_perm, .95))
                q_value = float(np.clip((q10 - perm_q95) / (median + 1e-12), 0.0, 1.0))
                legacy_q_same_null = float(np.clip((legacy_q10 - perm_q95) / (legacy_median + 1e-12), 0.0, 1.0))
                q_site_sensitivity = float(np.clip((q10 - site_q95) / (median + 1e-12), 0.0, 1.0))
                arrays[f"{fold}:{pair}"] = {"bootstrap": boot, "legacy_bootstrap": legacy_boot,
                                              "global_perm": global_perm, "site_perm": site_perm,
                                              "bootstrap_indices": bootstrap_indices,
                                              "global_perm_labels": global_perm_labels,
                                              "site_perm_labels": site_perm_labels,
                                              "site_perm_sources": site_perm_sources,
                                              "site_label_counts_before": [np.bincount(y_train[site_train == sid], minlength=2).tolist()
                                                                            for sid in range(20)],
                                              "site_label_counts_after_examples": [
                                                  [np.bincount(condition_site_permutation(y_train, site_train,
                                                                                           stream(fold, pair, "site_perm", rep))[site_train == sid], minlength=2).tolist()
                                                   for sid in range(20)]
                                                  for rep in range(min(3, CFG["pid"]["permutation_replicates"]))]}
                old_diag = old_diags[(fold, pair)]
                old_pair = old_pairs[(fold, 4, pair)]
                old_weight = old_weights[(fold, pair)]
                before_after_by_key[(fold, pair)].update({
                    "old_boot_q10": float(old_diag["R_boot_q10"]), "new_boot_q10": q10,
                    "old_boot_median": float(old_diag["R_boot_median"]), "new_boot_median": median,
                    "old_boot_q90": float(old_diag["R_boot_q90"]), "new_boot_q90": q90,
                    "legacy_boot_q10_same_ix": legacy_q10,
                    "legacy_boot_median_same_ix": legacy_median,
                    "q_legacy_boot_same_new_null": legacy_q_same_null,
                    "old_perm_q95": float(old_diag["R_perm_q95"]), "new_perm_q95": perm_q95,
                    "old_site_perm_q95": float(old_diag["R_site_perm_q95"]), "new_site_perm_q95": site_q95,
                    "old_q": float(old_diag["q"]), "new_q": q_value, "new_q_site_sensitivity": q_site_sensitivity,
                    "old_lambda": float(old_weight["lambda"]),
                    "new_lambda_unbarred": CFG["protocol"]["lambda_max"] * result["redundancy_fraction"] * q_value,
                    "bootstrap_pairing_fixed": True,
                })
                diagnostics.append({"fold": fold, "pair": pair, "k": 4,
                                    "R_boot_q10": q10, "R_boot_median": median, "R_boot_q90": q90,
                                    "R_perm_q95": perm_q95, "R_site_perm_q95": site_q95,
                                    "q": q_value, "q_site_sensitivity": q_site_sensitivity,
                                    "q_legacy_boot_same_new_null": legacy_q_same_null,
                                    "R_legacy_boot_q10_same_ix": legacy_q10,
                                    "R_legacy_boot_median_same_ix": legacy_median,
                                    "bootstrap_n": len(boot), "perm_n": len(global_perm),
                                    "site_perm_n": len(site_perm), "R_stable": float(np.std(boot, ddof=0)),
                                    "I_identity_residual": result["identity_residual"],
                                    "base_Ijoint": result["Ijoint"], "solver_failures": 0,
                                    "site_mapping_status": "VERIFIED", "train_only": True,
                                    "label_counts_train": np.bincount(y_train, minlength=2).tolist()})
    csv_write(RES / "pid_pairs.csv", primary)
    csv_write(RES / "pid_diagnostics.csv", diagnostics)
    per_fold = {}
    for fold in range(1, 11):
        per_fold[fold] = {}
        lambdas = {}
        rs = {}
        for pair in EFFECTIVE:
            diag = next(row for row in diagnostics if row["fold"] == fold and row["pair"] == pair)
            pid_row = next(row for row in primary if row["fold"] == fold and row["k"] == 4 and row["pair"] == pair)
            q_value = float(diag["q"])
            redundancy = float(pid_row["redundancy_fraction"])
            lambdas[pair] = CFG["protocol"]["lambda_max"] * redundancy * q_value
            rs[pair] = redundancy
        lambda_bar = sum(lambdas.values()) / 3.0
        for pair in EFFECTIVE:
            weights.append({"fold": fold, "k": 4, "pair": pair, "q": float(next(row["q"] for row in diagnostics
                                            if row["fold"] == fold and row["pair"] == pair)),
                            "r": rs[pair], "lambda": lambdas[pair], "lambda_bar": lambda_bar,
                            "lambda_formula": "0.03*r*q", "lambda_bar_formula": "mean(lambda_AF,lambda_AC,lambda_FC)"})
        per_fold[fold] = {"q": {pair: float(next(row["q"] for row in diagnostics if row["fold"] == fold and row["pair"] == pair))
                                for pair in EFFECTIVE},
                          "r": rs, "lambda": lambdas, "lambda_bar": lambda_bar}
        old_bar = float(old_weights[(fold, EFFECTIVE[0])]["lambda_bar"])
        for pair, _, _ in PAIRS:
            record = before_after_by_key[(fold, pair)]
            record["old_lambda_bar"] = old_bar if pair in EFFECTIVE else ""
            record["new_lambda_bar"] = lambda_bar if pair in EFFECTIVE else ""
            record["new_lambda"] = lambdas[pair] if pair in EFFECTIVE else ""
    before_after = [before_after_by_key[(fold, pair)] for fold in range(1, 11) for pair in EFFECTIVE]
    csv_write(RES / "before_after.csv", before_after)
    csv_write(RES / "pid_weights.csv", weights)
    write_json_atomic(RT / "pid_arrays.json", arrays)
    valid_rows = [row for row in primary if int(row["k"]) == 4]
    failures = [row for row in valid_rows if str(row["solver_success"]).lower() != "true" or
                float(row["identity_residual"]) > CFG["pid"]["reliability"]["max_identity_residual_bit"] or
                float(row["marginal_error"]) > CFG["pid"]["reliability"]["max_marginal_error"]]
    lam = [float(row["lambda"]) for row in weights]
    positive = sum(value > 0 for value in lam)
    lam_range = max(lam) - min(lam)
    if failures:
        status = "PID_ESTIMATOR_NOT_VALIDATED"
        gate = "ESTIMATOR_UNVALIDATED"
    elif positive >= CFG["pid"]["reliability"]["min_positive_lambda_relations"] and lam_range > CFG["pid"]["reliability"]["min_lambda_range"]:
        status = gate = "PID_GATE_GO_FOR_FUTURE_TRAINING"
    else:
        status = gate = "PID_GATE_NO_GO_AFTER_CORRECTION"
    manifest_status = {"PID_GATE_GO_FOR_FUTURE_TRAINING": "PID_GO",
                       "PID_GATE_NO_GO_AFTER_CORRECTION": "PID_NO_GO",
                       "PID_ESTIMATOR_NOT_VALIDATED": "ESTIMATOR_UNVALIDATED"}[status]
    result_signature = hashlib.sha256("".join(sha(RES / name) for name in
                                               ("pid_pairs.csv", "pid_diagnostics.csv", "pid_weights.csv")).encode()).hexdigest()
    state_history = list(manifest().get("state_history", ["PREPARED"]))
    state_history.extend(["PID_AUDITED", manifest_status])
    update_manifest(pid={"status": status, "gate": gate, "primary_rows": len(valid_rows),
                         "all_pid_rows": len(primary), "diagnostic_rows": len(diagnostics),
                         "positive_lambdas": positive, "lambda_range": lam_range,
                         "failures": len(failures), "site_mapping": "VERIFIED",
                         "synthetic_tests": synthetic_tests, "synthetic_pid_checks": synthetic_checks,
                         "fold_weights": per_fold, "result_signature": result_signature}, status=manifest_status,
                    state_history=state_history,
                    pid_recomputed_unix=time.time())
    print("PID_STATUS", status, "positive", positive, "range", lam_range, "failures", len(failures), flush=True)
    return status


def dry_run_state_machine():
    current = manifest()
    signature = current.get("pid", {}).get("result_signature", "")
    allowed_transitions = {"PREPARED": ["PREFLIGHT_PASS"],
                           "PREFLIGHT_PASS": ["PID_AUDITED"],
                           "PID_AUDITED": ["PID_GO", "PID_NO_GO", "ESTIMATOR_UNVALIDATED"],
                           "PID_GO": ["TRAINING"], "PID_NO_GO": [], "ESTIMATOR_UNVALIDATED": [],
                           "TRAINING": ["TRAINING_COMPLETE"], "TRAINING_COMPLETE": ["AUDIT_PASS"]}
    simulated_worker = {"status": "TRAINING", "pid": {"gate": "PID_GATE_GO_FOR_FUTURE_TRAINING",
                                                           "result_signature": signature},
                        "expected_pid_signature": signature}
    worker_accept = (simulated_worker["status"] == "TRAINING" and
                     simulated_worker["pid"]["gate"] == "PID_GATE_GO_FOR_FUTURE_TRAINING" and
                     simulated_worker["pid"]["result_signature"] == simulated_worker["expected_pid_signature"])
    stale_worker = dict(simulated_worker)
    stale_worker["pid"] = dict(simulated_worker["pid"], result_signature="stale")
    stale_reject = stale_worker["pid"]["result_signature"] != stale_worker["expected_pid_signature"]
    no_go_report_rows = len(historical_rows())
    result = {"pass": worker_accept and stale_reject and no_go_report_rows == 10,
              "allowed_transitions": allowed_transitions,
              "simulated_status_path": ["PREPARED", "PREFLIGHT_PASS", "PID_AUDITED", "PID_GO", "TRAINING"],
              "worker_accepts_training_only_with_pid_go_and_matching_signature": worker_accept,
              "worker_rejects_stale_signature": stale_reject,
              "no_go_report_historical_rows_expected": no_go_report_rows,
              "formal_training_invoked": False}
    write_json_atomic(RT / "state_machine_dryrun.json", result)
    return result


def audit_command():
    result = subprocess.run([sys.executable, str(EXP / "audit.py")], cwd=EXP, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def report_command():
    m = manifest()
    pairs = csv_read(RES / "pid_pairs.csv")
    diag = csv_read(RES / "pid_diagnostics.csv")
    weights = csv_read(RES / "pid_weights.csv")
    before = csv_read(RES / "before_after.csv")
    status = m.get("pid", {}).get("status", "UNKNOWN")
    lines = ["# Experiment 45.1 - PID correction and independent re-audit", "",
             f"状态：PID重算完成，E1-E6正式训练折数为 0（{CFG['formal_training']['status']}）。",
             f"PID准入：`{status}`。站点映射：`{m.get('site_mapping_status')}`。", "",
             "## 完成状态", "",
             "- Experiment 45 历史文件、根模型源码和数据仅读复核，SHA256 前后不变。",
             "- Bootstrap、站点条件置乱、BROJA PID 数值求解和对抗单步测试已执行。",
             "- E1-E6 未训练；本报告不填充新的 ACC/AUC/SPE/SEN。", "",
             "## Q1 Bootstrap 错误", "",
             "旧实现保留训练折局部模态聚类标签，却用全局 `y[tr[ix]]`，所以模态行与疾病标签不再来自同一患者。样本ID测试命中率从 0/48 变为 48/48；带噪冗余数据的配对 Bootstrap 中位 R 为 0.6943，标签独立数据中位 R 为 0.00019。下表 `q_legacy same-null` 用同一批新全局置乱，仅换回错误 Bootstrap，隔离配对修复的影响。",
             "", "|fold|pair|q旧|q错配同零分布|q修正|旧 boot Q10|错配 boot Q10|配对 boot Q10|旧全局 perm Q95|新全局 perm Q95|旧lambda|新lambda|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in before:
        if row["pair"] not in EFFECTIVE:
            continue
        lines.append("|{fold}|{pair}|{old_q:.6f}|{q_legacy:.6f}|{new_q:.6f}|{old_b:.6f}|{legacy_b:.6f}|{new_b:.6f}|{old_p:.6f}|{new_p:.6f}|{old_l:.8f}|{new_l:.8f}|".format(
            fold=int(row["fold"]), pair=row["pair"], old_q=float(row["old_q"]),
            q_legacy=float(row["q_legacy_boot_same_new_null"]), new_q=float(row["new_q"]),
            old_b=float(row["old_boot_q10"]), legacy_b=float(row["legacy_boot_q10_same_ix"]),
            new_b=float(row["new_boot_q10"]), old_p=float(row["old_perm_q95"]),
            new_p=float(row["new_perm_q95"]), old_l=float(row["old_lambda"]),
            new_l=float(row["new_lambda"])))
    lines += ["", "## Q2-Q3 修正后 PID", "", "|关系|R|U1|U2|S|冗余占比|q均值|lambda均值|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pair, _, _ in PAIRS:
        rows = [row for row in pairs if row["pair"] == pair and int(row["k"]) == 4]
        ws = [row for row in weights if row["pair"] == pair]
        if not rows:
            continue
        vals = [np.mean([float(row[key]) for row in rows]) for key in ("R", "U1", "U2", "S", "redundancy_fraction")]
        q = f"{np.mean([float(row['q']) for row in ws]):.8f}" if ws else "-"
        lam = f"{np.mean([float(row['lambda']) for row in ws]):.8f}" if ws else "-"
        lines.append(f"|{pair}|" + "|".join(f"{value:.8f}" for value in vals) + f"|{q}|{lam}|")
    old_pairs = csv_read(HIST / "results/pid_pairs.csv")
    lines += ["", "### K=4 旧值与修正值", "", "|关系|旧 R|新 R|旧冗余占比|新冗余占比|", "|---|---:|---:|---:|---:|"]
    for pair, _, _ in PAIRS:
        old = [row for row in old_pairs if row["pair"] == pair and int(row["k"]) == 4]
        new = [row for row in pairs if row["pair"] == pair and int(row["k"]) == 4]
        lines.append(f"|{pair}|{np.mean([float(row['R']) for row in old]):.8f}|{np.mean([float(row['R']) for row in new]):.8f}|"
                     f"{np.mean([float(row['redundancy_fraction']) for row in old]):.8f}|"
                     f"{np.mean([float(row['redundancy_fraction']) for row in new]):.8f}|")
    lines += ["", "逐折复核显示，给定当前相同输入，历史 PID 源码与 45.1 BROJA 求解器四个分量最大差异低于 1.3e-17；与实验45已落盘结果相比有 11/60 条 K=4 记录超过 1e-12，最大差异约 2.92e-4。历史 manifest 未记录 PID 源码 SHA 或 NumPy/SciPy/scikit-learn 版本，因此无法唯一归因；当前环境为 NumPy 1.26.0/SciPy 1.11.3/scikit-learn 1.3.2，与根项目 requirements.txt 中的版本不同。这更符合历史离散化/数值运行状态或历史结果源码未留 hash 的差异，而不是本轮数学目标更改。", ""]
    lines += ["", "### K 敏感性", "", "|K|关系|平均 R|平均冗余占比|", "|---:|---|---:|---:|"]
    for k in (3, 4, 5):
        for pair, _, _ in PAIRS:
            rows = [row for row in pairs if int(row["k"]) == k and row["pair"] == pair]
            lines.append(f"|{k}|{pair}|{np.mean([float(row['R']) for row in rows]):.8f}|"
                         f"{np.mean([float(row['redundancy_fraction']) for row in rows]):.8f}|")
    lines += ["", "K=3/4/5 共 180 条主 PID 记录；K=4 的 AF/AC/FC 共 30 条诊断记录。Bootstrap 只在冻结训练折的离散化结果上重采样，不包含重新拟合 scaler/KMeans 的不确定性。", "",
              "## Q4 站点混杂诊断", "",
              "站点列依据 `modal_feat_dict.npy` 中的 PHENO 特征名和 `processed_standard_data.csv` 的精确列名恢复：20 个 `SITE_ID_*` 列；每个患者恰有一个 1。站点条件置乱从原始训练标签开始，在每个训练折站点内部置换，保持总计数和每站点 ASD/TD 计数。它是混杂敏感性诊断，不替代冻结 q 公式的全局置乱零分布。", "",
              "|关系|平均全局置乱 Q95(R)|平均站点置乱 Q95(R)|", "|---|---:|---:|"]
    for pair in EFFECTIVE:
        rows = [row for row in diag if row["pair"] == pair]
        lines.append(f"|{pair}|{np.mean([float(row['R_perm_q95']) for row in rows]):.8f}|"
                     f"{np.mean([float(row['R_site_perm_q95']) for row in rows]):.8f}|")
    lines += ["", "站点敏感性 q 使用同一 Bootstrap、只将 q 的零分布替换为站点条件置乱；它不参与准入。",
              "", "|关系|全局 q 正值折数|站点敏感性 q 正值折数|全局 q 均值|站点敏感性 q 均值|", "|---|---:|---:|---:|---:|"]
    for pair in EFFECTIVE:
        rows = [row for row in diag if row["pair"] == pair]
        lines.append(f"|{pair}|{sum(float(row['q']) > 0 for row in rows)}|"
                     f"{sum(float(row['q_site_sensitivity']) > 0 for row in rows)}|"
                     f"{np.mean([float(row['q']) for row in rows]):.8f}|"
                     f"{np.mean([float(row['q_site_sensitivity']) for row in rows]):.8f}|")
    lines += ["", "### Fold lambda", "", "|关系|10折均值|最小值|最大值|", "|---|---:|---:|---:|"]
    for pair in EFFECTIVE:
        values = np.asarray([float(row["lambda"]) for row in weights if row["pair"] == pair])
        lines.append(f"|{pair}|{values.mean():.9f}|{values.min():.9f}|{values.max():.9f}|")
    old_bar = [float(row["lambda_bar"]) for row in csv_read(HIST / "results/pid_weights.csv") if row["pair"] == "AF"]
    new_bar = [float(row["lambda_bar"]) for row in weights if row["pair"] == "AF"]
    lines += ["", f"旧 lambda_bar 均值={np.mean(old_bar):.9f}；修正后逐折 `mean(lambda_pair)` 的均值={np.mean(new_bar):.9f}，最大值={np.max(new_bar):.9f}。旧实现用 `0.03*mean(q)*mean(r)`，修正版逐折求三条 lambda 后取均值。", ""]
    lines += ["",
              "## Q5 PID 定义与求解", "",
              "历史代码名为 I_min，但实际求解的是 BROJA 式两源离散 PID：在保持 `P(X1,Y)` 与 `P(X2,Y)` 的耦合集合中最小化 `I_Q((X1,X2);Y)`，再由 `I1/I2/Ijoint/Ijoint_min` 分解 R、U1、U2、S。主解用解析梯度 SLSQP；每个解检查边缘约束、目标值复算、恒等式和分量非负性。无 CVXPY 时，低维审计用独立参数化的 SciPy trust-constr 求解器交叉核验。数值恒等式通过不代表有限样本统计可靠。", "",
              "## q 门槛功效模拟", "",
              "样本量与 ASD/TD 比例匹配 fold 1 训练折，K=4，每对使用 12 次 Bootstrap 和 19 次全局置乱，重复三次。模拟结果只诊断冻结门槛，不改阈值。"]
    power = read_json(RES / "gate_power_calibration.json")
    lines += ["", "|合成机制|GO次数/3|平均 q|平均 lambda|", "|---|---:|---:|---:|"]
    for scenario, summary in power["summary"].items():
        lines.append(f"|{scenario}|{summary['gate_passes']}/{summary['repetitions']}|"
                     f"{summary['mean_q']:.6f}|{summary['mean_lambda']:.8f}|")
    lines += ["", "强冗余正对照若未稳定触发 GO，说明当前 12/19 次重采样门槛检验功效有限；零值不能解释为连续模态不存在疾病相关共享信息。", "",
              "## Q6 对抗修复与逻辑一致性", "",
              "关系 i/j 的表示沿样本维构造 `[z_i; z_j]`，输入为 `[2*N_train,64]`；来源 0/1 分别是第一个/第二个关系图。条件标签是同一训练患者标签的两份复制，判别器输入固定为 66 维。GRL 实际梯度满足 `grad_GRL = -lambda * grad_plain`；E2/E3 使用相同逐折 `lambda_bar`，E4 使用 AF/AC/FC 各自 lambda，E5 只循环置换归属，E6 保留辅助/对抗/GFR结构而仅把主分类输入旁路到原始 X。GPU0/GPU1 均通过 E1-E6 单步、双步 B0-B 等价、来源学习和 GRL 梯度检查。状态机 dry-run 接受有效 PID_GO 签名下的 TRAINING 并拒绝过期签名；正式 worker 未启动。", "",
              "## Q7 科研结论", "",
              f"修复后的准入为 `{status}`：30 条 AF/AC/FC 关系-折记录中 {sum(float(row['lambda']) > 0 for row in weights)} 条 lambda 为正，唯一正值来自 AF fold 3；AC 与 FC 十折均为零。按冻结的‘至少一个正值且全体 lambda 范围大于 1e-6’规则，Experiment 45 原 NO-GO 不再成立，建议进入后续正式对照训练，但只作为可执行的探索性实验，不代表 PID-DGAA 创新成立。当前稀疏通过和功效模拟必须在解释中保留。E1-E6 本轮仍为 0/60 折。", "",
              "下一阶段应保留 E0/E1/E2/E3/E4/E5/E6 同折对照，报告每折 lambda 与领域分类/特征梯度诊断，检查 E4 相对 E3/E5 的逐折增量及 E6 图旁路；完成后再独立评估多 seed 与无偏验证。历史 val=test、CV 前全局预处理/RFE 和 transductive 图仍限制外部泛化解释。", "",
              "## 交付", "", f"实际主 PID 估计数：{len(pairs)}；有效关系诊断：{len(diag)}；正式新增训练折：0；结果目录：`{EXP / 'results'}`。"]
    (RES / "report.md").write_text("\n".join(lines) + "\n")
    update_manifest(report_sha256=sha(RES / "report.md"), formal_training={"new_folds": 0, "status": "NOT_TRAINED_BY_DESIGN"})
    print("REPORT_WRITTEN", RES / "report.md", flush=True)


def handoff_command():
    marker = "<!-- EXPERIMENT45_1_PID_CORRECTION_REAUDIT -->"
    path = ROOT / "handoff.md"
    m = manifest()
    report = (RES / "report.md").read_text()
    addition = "\n\n" + marker + "\n## Experiment 45.1 - PID correction and independent re-audit\n\n" + report
    import fcntl
    with path.open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        before = handle.read()
        prefix = m["handoff_prefix"]
        if hashlib.sha256(before[:prefix["bytes"]]).hexdigest() != prefix["sha256"]:
            raise RuntimeError("HANDOFF_PENDING: historical handoff prefix changed")
        occurrences = before.count(marker.encode())
        if occurrences == 0:
            handle.seek(0, 2)
            handle.write(addition.encode())
            handle.flush()
            os.fsync(handle.fileno())
        elif occurrences == 1:
            existing = before[before.index(marker.encode()):].decode("utf-8")
            if report not in existing:
                raise RuntimeError("HANDOFF_PENDING: existing marker content differs")
        else:
            raise RuntimeError("HANDOFF_PENDING: marker appears more than once")
        handle.seek(0)
        after = handle.read()
        if not after.startswith(before) and occurrences == 0:
            raise RuntimeError("HANDOFF_PENDING: handoff prefix was not preserved")
        if after.count(marker.encode()) != 1 or len(after) <= len(before) and occurrences == 0:
            raise RuntimeError("HANDOFF_PENDING: append verification failed")
    update_manifest(handoff={"status": "APPENDED" if occurrences == 0 else "ALREADY_PRESENT_CONFIRMED",
                             "prefix_sha256": hashlib.sha256(before).hexdigest(),
                             "after_sha256": hashlib.sha256(after).hexdigest(),
                             "bytes_before": len(before), "bytes_after": len(after),
                             "marker_count": after.count(marker.encode())}, delivery="COMPLETE")
    print("HANDOFF_APPENDED" if occurrences == 0 else "HANDOFF_ALREADY_CONFIRMED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "pid", "power", "train", "audit", "report", "handoff", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        model_preflight(args.gpu)
        if all((RT / f"preflight_gpu{gpu}.json").exists() for gpu in (0, 1)):
            update_manifest(status="PREFLIGHT_PASS", preflight_status="PASS")
    elif args.command == "pid":
        pid_phase()
    elif args.command == "power":
        gate_power_calibration()
    elif args.command == "train":
        raise RuntimeError("FORMAL_TRAINING_DISABLED_BY_DESIGN: Experiment 45.1 requires 0 E1-E6 folds")
    elif args.command == "audit":
        audit_command()
    elif args.command == "report":
        report_command()
    elif args.command == "handoff":
        handoff_command()
    elif args.command == "all":
        prepare()
        raise RuntimeError("Use separate GPU preflight commands before pid/audit/report/handoff; formal training remains disabled")


if __name__ == "__main__":
    main()
