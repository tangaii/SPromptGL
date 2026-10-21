#!/usr/bin/env python3
"""Experiment 33 QCAR: frozen eight-group, seed-0, resumable ten-fold runs."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import importlib
import io
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

from integration import emit
from qcar import calibration_diagnostics, fit_fold, json_stats

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
EXP32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
AUDIT = ROOT / "experiment/b0_pen_physical_removal_equivalence_audit"
CLIP_AUDIT = ROOT / "experiment/clip_encoder_equivalence_audit"
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
RESULTS = EXP / "results"
MANIFEST_PATH = EXP / "manifest.json"
INPUT_DIMS = [48, 6, 10, 256]
CONFIGS = (
    "G0_B0A", "G1_SITE_ONLY", "G2_UNCONDITIONAL", "G3_NO_RELIABILITY",
    "G4_QCAR_FULL", "G5_WRONG_PAIR", "G6_FC_ONLY", "G7_REVERSE_SIGN",
)
GPU_CONFIGS = {
    0: ("G0_B0A", "G2_UNCONDITIONAL", "G4_QCAR_FULL", "G6_FC_ONLY"),
    1: ("G1_SITE_ONLY", "G3_NO_RELIABILITY", "G5_WRONG_PAIR", "G7_REVERSE_SIGN"),
}
GROUP_METHOD = {
    "G0_B0A": (None, "none", -1),
    "G1_SITE_ONLY": ("site_only", "ones", -1),
    "G2_UNCONDITIONAL": ("unconditional", "ones", -1),
    "G3_NO_RELIABILITY": ("qcar", "ones", -1),
    "G4_QCAR_FULL": ("qcar", "reliability", -1),
    "G5_WRONG_PAIR": ("qcar_wrong_pair", "reliability", -1),
    "G6_FC_ONLY": ("fc_only", "reliability", -1),
    "G7_REVERSE_SIGN": ("qcar_reverse", "reliability", 1),
}
PROTOCOL = dict(
    datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
    nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
    n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
    GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
    alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
    nclass=2, mode="simple-2", seed=0,
)
CHECKPOINT_EVERY = 25
SEN_NONINFERIORITY_PP = .5
CORE_FILES = ("main.py", "model.py", "network.py", "layers.py", "utils.py",
              "ABIDE-simple-2-concat-weighted-cosine.sh")
DATA_FILES = ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")
HISTORY_FILES = (
    "experiment/experiment32_disease_contrastive_evidence_routing/README.md",
    "experiment/experiment32_disease_contrastive_evidence_routing/integration.py",
    "experiment/experiment32_disease_contrastive_evidence_routing/integration.patch",
    "experiment/experiment32_disease_contrastive_evidence_routing/method.py",
    "experiment/experiment32_disease_contrastive_evidence_routing/run.py",
    "experiment/experiment32_disease_contrastive_evidence_routing/manifest.json",
    "experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv",
    "experiment/experiment32_disease_contrastive_evidence_routing/results/report.md",
    "experiment/b0_pen_physical_removal_equivalence_audit/audit.py",
    "experiment/b0_pen_physical_removal_equivalence_audit/results.json",
    "experiment/b0_pen_physical_removal_equivalence_audit/report.md",
    "experiment/b0_b_seed0_full10fold/manifest.json",
    "experiment/b0_b_seed0_full10fold/results/fold_metrics.csv",
    "experiment/b0_b_seed0_full10fold/results/report.md",
    "experiment/clip_encoder_equivalence_audit/audit_clip.py",
    "experiment/clip_encoder_equivalence_audit/audit.json",
    "experiment/clip_encoder_equivalence_audit/report.md",
    "experiment/experiment18_discriminability_preserving_selective_alignment/results/report.md",
    "experiment/experiment24_conflict_aware_relational_alignment/results/report.md",
    "experiment/experiment25_evidence_preserving_asymmetric_graph_calibration/results/report.md",
    "experiment/experiment27_evidence_grounded_relational_prompt_cam/results/report.md",
    "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/report.md",
    "experiment/experiment29_connectivity_semantic_selective_auxiliary/results/report.md",
    "experiment/experiment30_class_conditional_semantic_shared_repair/results/report.md",
    "experiment/experiment31_subject_specific_connectomic_evidence/results/report.md",
)
_CTX = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def torch_write(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    return {name: sha_file(ROOT / name) for name in CORE_FILES}


def code_hashes():
    return {name: sha_file(EXP / name) for name in
            ("README.md", "qcar.py", "integration.py", "integration.patch", "run.py")}


def input_hashes():
    return {name: sha_file(ROOT / name) for name in DATA_FILES}


def source_signature():
    history_hashes = {}
    if MANIFEST_PATH.is_file():
        history_hashes = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get(
            "experiment32_and_audit_sha256", {})
    return hashlib.sha256(json.dumps({"source": source_hashes(), "code": code_hashes(),
                                      "inputs": input_hashes(),
                                      "prompt_sha256": sha_file(ROOT / "prompt"),
                                      "history": history_hashes,
                                      "protocol": PROTOCOL, "configs": CONFIGS},
                                     sort_keys=True).encode()).hexdigest()


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(mapping[key]) for key in mapping] == INPUT_DIMS
    expected_columns = [str(name) for key in mapping for name in mapping[key]]
    assert list(frame.columns[:-1]) == expected_columns
    assert frame.shape == (871, 321)
    pheno_names = [str(name) for name in mapping["PHENO"]]
    assert len(pheno_names) == 48
    assert all(name.startswith("AGE_AT_SCAN_") for name in pheno_names[:26])
    assert all(name.startswith("SITE_ID_") for name in pheno_names[26:46])
    assert all(name.startswith("SEX_") for name in pheno_names[46:48])
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    assert len(folds) == 10
    for train, test in folds:
        assert len(set(train) & set(test)) == 0
        assert len(set(train) | set(test)) == 871
    return x, y, folds, mapping


def seed_everything(torch, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_capture(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def rng_restore(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(agent):
    digest = hashlib.sha256()
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in sorted(module.named_parameters()):
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close_helper(agent):
    for name in ("MF_sav", "GCMP_sav"):
        stream = getattr(agent, name, None)
        if stream is not None:
            stream.close()


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    sys.path.insert(0, str(SOURCE))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    model = importlib.import_module("model")
    network = importlib.import_module("network")
    layers = importlib.import_module("layers")
    utils = importlib.import_module("utils")
    x, y, folds, mapping = read_data()
    _CTX = dict(torch=torch, np=np, model=model, network=network, layers=layers,
                utils=utils, x=x, y=y, folds=folds, mapping=mapping)
    return _CTX


def fit_fold_stats(c, fold):
    train, _ = c["folds"][fold - 1]
    x = c["x"].astype(np.float64, copy=False)
    arrays, stats = fit_fold(x[:, :48], x[:, 48:64], x[:, 64:], train, fold)
    return arrays, stats


def select_direction(config, c, stats, arrays):
    import torch
    method, reliability_mode, sign = GROUP_METHOD[config]
    if method is None:
        return None
    key = method
    direction = torch.as_tensor(arrays[key], dtype=torch.float32, device=c["device"])
    if reliability_mode == "ones":
        weights = torch.ones(256, dtype=torch.float32, device=c["device"])
    else:
        weights = torch.as_tensor(stats["reliability"], dtype=torch.float32, device=c["device"])
    return dict(direction=direction, reliability=weights, sign=sign,
                method=method, reliability_mode=reliability_mode)


class _BoundedIdentityGradient:
    @staticmethod
    def apply(value):
        import torch

        class ClampSTE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, raw):
                return raw.clamp(0.0, 0.25)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return ClampSTE.apply(value)


def make_qcar(torch, device):
    import torch.nn as nn

    class QCARGamma(nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_gamma = nn.Parameter(torch.zeros((), device=device))

        def gamma(self):
            return _BoundedIdentityGradient.apply(self.raw_gamma)

        def project_(self):
            with torch.no_grad():
                self.raw_gamma.clamp_(0.0, 0.25)

        def calibrate(self, features, direction, reliability, sign):
            f = features[:, 64:]
            raw_delta = self.gamma() * direction * reliability.unsqueeze(0)
            delta_norm = raw_delta.norm(dim=1, keepdim=True)
            cap = .15 * f.norm(dim=1, keepdim=True)
            ratio = torch.clamp(cap / (delta_norm + 1e-12), max=1.0)
            delta = raw_delta * ratio
            calibrated = f + float(sign) * delta
            return torch.cat((features[:, :64], calibrated), dim=1), delta

    return QCARGamma()


def make_agent(config, fold, c, stats=None, arrays=None):
    import torch

    train, test = c["folds"][fold - 1]
    hp = SimpleNamespace(**PROTOCOL)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = c["model"].EvalHelper(INPUT_DIMS, c["x"], c["y"], hp, train, test)
    close_helper(agent)
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    agent.ModalFusion.qcar33_b0_bypass = True
    assert hasattr(agent.ModalFusion, "MLP")
    assert hasattr(agent.ModalFusion, "prompt_text_features")
    assert hasattr(agent.ModalFusion, "avg_pool")
    agent.qcar = None
    agent.qcar_data = None
    if config != "G0_B0A":
        assert stats is not None and arrays is not None
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            agent.qcar = make_qcar(torch, agent.dev).to(agent.dev)
        group = dict(params=list(agent.qcar.parameters()), lr=hp.lr, weight_decay=0.0)
        agent.optimizer_MP0.add_param_group(group)
        agent.qcar_data = select_direction(config, c, stats, arrays)
        assert agent.qcar_data["direction"].shape == (871, 256)
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    actual = {id(p) for module in (agent.MessagePassing0, agent.ModalFusion)
              for p in module.parameters()}
    if agent.qcar is not None:
        actual |= {id(p) for p in agent.qcar.parameters()}
    optim_ids = [id(p) for group in groups for p in group["params"]]
    assert len(optim_ids) == len(set(optim_ids)) and set(optim_ids) == actual
    assert all(abs(group["lr"] - PROTOCOL["lr"]) < 1e-12 for group in groups)
    if agent.qcar is not None:
        assert groups[1]["weight_decay"] == 0.0
        assert [id(p) for p in groups[1]["params"]] == [id(p) for p in agent.qcar.parameters()]
    return agent


def calibrated_input(agent, features):
    if agent.qcar is None:
        return features, None
    spec = agent.qcar_data
    return agent.qcar.calibrate(features, spec["direction"], spec["reliability"], spec["sign"])


def train_step(agent, c):
    torch, utils = c["torch"], c["utils"]
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad()
    agent.optimizer_MF.zero_grad()
    model_input, _ = calibrated_input(agent, agent.feat)
    model_num, parts = agent.dataLayer(model_input)
    features, adjs, alignment_loss = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
    all_graph_loss = 0
    for index in range(model_num):
        graph_loss = utils.GraphConstructLoss(
            parts[index], adjs[index], agent.hyperpm.theta_smooth,
            agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity,
        )
        all_graph_loss = all_graph_loss + graph_loss
    all_graph_loss = all_graph_loss / 4
    logp, _, _ = agent.ModalFusion(features)
    cls_loss = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    loss = cls_loss + all_graph_loss
    loss.backward()
    gradients = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            gradients[prefix + "." + name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    if agent.qcar is not None:
        gradients["qcar.raw_gamma"] = agent.qcar.raw_gamma.grad.detach().cpu().clone()
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    if agent.qcar is not None:
        agent.qcar.project_()
    return dict(loss=float(loss.detach()), cls_loss=float(cls_loss.detach()),
                graph_loss=float(all_graph_loss.detach()), gradients=gradients,
                output=logp.detach().clone())


def evaluate(agent, c, return_detail=False):
    torch = c["torch"]
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    captured = {}
    hooks = []
    if return_detail:
        def cam_hook(_module, _inputs, output):
            captured["cam"] = output.detach().clone()
        def classifier_pre_hook(_module, inputs):
            captured["classifier_cam"] = inputs[0].detach().clone()
            captured["classifier_attention"] = inputs[1].detach().clone()
        hooks.append(agent.ModalFusion.FeedForward[-1].register_forward_hook(cam_hook))
        hooks.append(agent.ModalFusion.Outputlayer.register_forward_pre_hook(classifier_pre_hook))
    with torch.no_grad():
        model_input, delta = calibrated_input(agent, agent.feat)
        model_num, parts = agent.dataLayer(model_input)
        features, adjs, alignment = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
        output, hidden, attention = agent.ModalFusion(features)
    for hook in hooks:
        hook.remove()
    assert output.shape == (871, 2) and torch.isfinite(output).all()
    if not return_detail:
        return output
    return dict(input=model_input.detach().clone(), delta=None if delta is None else delta.detach().clone(),
                gfr=features.detach().clone(), adjs=[value.detach().clone() for value in adjs],
                cam=captured["cam"], logp=output.detach().clone(), hidden=hidden.detach().clone(),
                attention=attention, classifier_cam=captured["classifier_cam"],
                classifier_attention=captured["classifier_attention"])


def model_state(agent):
    result = dict(gfr=copy.deepcopy(agent.MessagePassing0.state_dict()),
                  pln=copy.deepcopy(agent.ModalFusion.state_dict()))
    if agent.qcar is not None:
        result["qcar"] = copy.deepcopy(agent.qcar.state_dict())
    return result


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])
    if agent.qcar is not None:
        agent.qcar.load_state_dict(state["qcar"])


def gradient_map(agent):
    result = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, p in module.named_parameters():
            result[prefix + "." + name] = None if p.grad is None else p.grad.detach().cpu().clone()
    return result


def compare_tensor_maps(left, right):
    assert set(left) == set(right)
    different = []
    max_error = 0.0
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            if a is not b:
                different.append(name)
            continue
        if not torch_equal(a, b):
            different.append(name)
            max_error = max(max_error, float((a - b).abs().max().item()))
    return dict(exact=not different, different=different, max_abs_error=max_error)


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def metrics_row(output, y, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    prediction = output.detach().cpu().numpy()[test].argmax(1)
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, prediction, labels=[0, 1]).ravel())
    auc = float(roc_auc_score(np.eye(2)[target], np.eye(2)[prediction]))
    return dict(acc=float(np.mean(prediction == target)), hard_auc=auc,
                sen_asd=float(tp / (tp + fn)), spe_td=float(tn / (tn + fp)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def task_path(config):
    return RUNTIME / "state" / f"{config}.pt"


def atomic_task_json(config, value):
    json_write(RUNTIME / "state" / f"{config}.json", value)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def task_signature(config):
    return hashlib.sha256((source_signature() + config).encode()).hexdigest()


def _grad_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


def preflight_checks(gpu):
    import torch
    c = context()
    device = int(os.environ["QCAR_PHYSICAL_GPU"])
    assert device == gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    torch.cuda.set_device(0)
    c["device"] = torch.device("cuda")
    network = c["network"]
    load_calls = {"clip_load": 0, "prompt_features": 0}
    original_load = network.clip.load
    original_get_features = network.get_prompt_text_features

    def tracked_load(*args, **kwargs):
        load_calls["clip_load"] += 1
        return original_load(*args, **kwargs)

    def tracked_features(*args, **kwargs):
        load_calls["prompt_features"] += 1
        return original_get_features(*args, **kwargs)

    network.clip.load = tracked_load
    network.get_prompt_text_features = tracked_features
    arrays, stats = fit_fold_stats(c, 1)
    directions = {name: select_direction(name, c, stats, arrays)
                  for name in ("G3_NO_RELIABILITY", "G4_QCAR_FULL",
                               "G5_WRONG_PAIR", "G7_REVERSE_SIGN")}
    assert torch.equal(directions["G3_NO_RELIABILITY"]["direction"],
                      directions["G4_QCAR_FULL"]["direction"])
    assert torch.equal(directions["G4_QCAR_FULL"]["direction"],
                      directions["G7_REVERSE_SIGN"]["direction"])
    assert torch.equal(directions["G4_QCAR_FULL"]["reliability"],
                      directions["G5_WRONG_PAIR"]["reliability"])
    assert torch.equal(directions["G4_QCAR_FULL"]["reliability"],
                      directions["G7_REVERSE_SIGN"]["reliability"])
    assert torch.count_nonzero(directions["G3_NO_RELIABILITY"]["reliability"]) == 256
    assert directions["G4_QCAR_FULL"]["sign"] == -directions["G7_REVERSE_SIGN"]["sign"]
    assert stats["wrong_pair_subject_fraction"] > 0
    original_y = c["y"].copy()
    _, test_indices = c["folds"][0]
    c["y"][test_indices] = 1 - c["y"][test_indices]
    label_mutation_arrays, label_mutation_stats = fit_fold_stats(c, 1)
    c["y"] = original_y
    assert np.array_equal(arrays["qcar"], label_mutation_arrays["qcar"])
    assert np.array_equal(stats["reliability"], label_mutation_stats["reliability"])

    original_x = c["x"].copy()
    c["x"][test_indices, 64:] = np.roll(c["x"][test_indices, 64:], 1, axis=1)
    heldout_fc_arrays, heldout_fc_stats = fit_fold_stats(c, 1)
    c["x"] = original_x
    assert np.array_equal(arrays["qcar"], heldout_fc_arrays["qcar"])
    assert np.array_equal(stats["reliability"], heldout_fc_stats["reliability"])
    assert np.array_equal(stats["gain"], heldout_fc_stats["gain"])
    for key in ("site_only", "unconditional", "qcar", "qcar_wrong_pair", "qcar_reverse"):
        assert np.array_equal(arrays[key], heldout_fc_arrays[key])
    expected_history = historical_b0_rows()[1]
    seed_everything(torch, 0)
    rng_before = rng_capture(torch)
    baseline = make_agent("G0_B0A", 1, c)
    initial_hash = parameter_hash(baseline)
    initial_rng_hash = rng_hash(rng_before)
    assert initial_hash == expected_history["initial_trunk_sha256"], \
        (initial_hash, expected_history["initial_trunk_sha256"])
    assert initial_rng_hash == expected_history["initial_rng_sha256"], \
        (initial_rng_hash, expected_history["initial_rng_sha256"])
    assert baseline.ModalFusion.qcar33_b0_bypass
    assert hasattr(baseline.ModalFusion, "MLP") and hasattr(baseline.ModalFusion, "avg_pool")
    base_eval = evaluate(baseline, c, return_detail=True)
    with torch.no_grad():
        direct_output, _ = baseline.ModalFusion.Outputlayer(
            base_eval["classifier_cam"], base_eval["classifier_attention"])
    assert torch.equal(base_eval["logp"], direct_output)

    base_initial = model_state(baseline)
    base_mp_opt = copy.deepcopy(baseline.optimizer_MP0.state_dict())
    base_mf_opt = copy.deepcopy(baseline.optimizer_MF.state_dict())
    rng_start = rng_capture(torch)
    baseline_step = train_step(baseline, c)
    base_grad = gradient_map(baseline)
    base_updated = model_state(baseline)

    load_model_state(baseline, base_initial)
    baseline.optimizer_MP0.load_state_dict(base_mp_opt)
    baseline.optimizer_MF.load_state_dict(base_mf_opt)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        baseline.qcar = make_qcar(torch, baseline.dev).to(baseline.dev)
    train_idx, _ = c["folds"][0]
    direction = torch.as_tensor(arrays["qcar"], dtype=torch.float32, device=baseline.dev)
    reliability = torch.as_tensor(stats["reliability"], dtype=torch.float32, device=baseline.dev)
    baseline.qcar_data = dict(direction=direction, reliability=reliability, sign=-1)
    baseline.optimizer_MP0.add_param_group(dict(params=list(baseline.qcar.parameters()),
                                                 lr=PROTOCOL["lr"], weight_decay=0.0))
    assert len(baseline.optimizer_MP0.param_groups) == 2
    assert baseline.optimizer_MP0.param_groups[1]["weight_decay"] == 0.0
    qcar_initial_state = model_state(baseline)
    qcar_initial_forward = evaluate(baseline, c, return_detail=True)
    input_exact = torch.equal(base_eval["input"], qcar_initial_forward["input"])
    gfr_exact = torch.equal(base_eval["gfr"], qcar_initial_forward["gfr"])
    adjacency_exact = all(torch.equal(a, b) for a, b in zip(base_eval["adjs"], qcar_initial_forward["adjs"]))
    cam_exact = torch.equal(base_eval["cam"], qcar_initial_forward["cam"])
    logp_exact = torch.equal(base_eval["logp"], qcar_initial_forward["logp"])
    assert input_exact and gfr_exact and adjacency_exact and cam_exact and logp_exact
    rng_restore(torch, rng_start)
    candidate_step = train_step(baseline, c)
    qcar_grad = _grad_norm(baseline.qcar.raw_gamma)
    common_candidate_gradients = {key: value for key, value in candidate_step["gradients"].items()
                                  if not key.startswith("qcar.")}
    grad_comparison = compare_tensor_maps(base_grad, common_candidate_gradients)
    candidate_updated = model_state(baseline)
    update_exact = all(torch.equal(base_updated[module][name], candidate_updated[module][name])
                       for module in base_updated for name in base_updated[module])
    assert baseline_step["loss"] == candidate_step["loss"]
    assert baseline_step["cls_loss"] == candidate_step["cls_loss"]
    assert baseline_step["graph_loss"] == candidate_step["graph_loss"]
    assert grad_comparison["exact"] and update_exact
    intervention_count = int(np.count_nonzero(stats["reliability"]))
    if intervention_count:
        assert qcar_grad > 0 and np.isfinite(qcar_grad)
    else:
        assert qcar_grad == 0.0
    assert all(p.grad is None for p in baseline.ModalFusion.MLP.parameters())

    model_ids = {id(p) for module in (baseline.MessagePassing0, baseline.ModalFusion)
                 for p in module.parameters()} | {id(p) for p in baseline.qcar.parameters()}
    optimizer_ids = [id(p) for opt in (baseline.optimizer_MP0, baseline.optimizer_MF)
                     for group in opt.param_groups for p in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids)) and set(model_ids) == set(optimizer_ids)

    probe = RUNTIME / f"preflight_resume_gpu{gpu}.pt"
    saved = dict(model=model_state(baseline), optimizer_mp=baseline.optimizer_MP0.state_dict(),
                 optimizer_mf=baseline.optimizer_MF.state_dict(), rng=rng_capture(torch),
                 statistics_sha256=stats["statistics_sha256"])
    torch_write(probe, saved)
    disk_saved = torch_load(probe)
    with torch.no_grad():
        baseline.qcar.raw_gamma.fill_(.123)
    load_model_state(baseline, disk_saved["model"])
    baseline.optimizer_MP0.load_state_dict(disk_saved["optimizer_mp"])
    baseline.optimizer_MF.load_state_dict(disk_saved["optimizer_mf"])
    rng_restore(torch, disk_saved["rng"])
    assert torch.equal(baseline.qcar.raw_gamma.detach().cpu(),
                       disk_saved["model"]["qcar"]["raw_gamma"])
    assert disk_saved["statistics_sha256"] == stats["statistics_sha256"]
    assert rng_hash(rng_capture(torch)) == rng_hash(disk_saved["rng"])
    probe.unlink()

    load_model_state(baseline, qcar_initial_state)
    with torch.no_grad():
        baseline.qcar.raw_gamma.fill_(.25)
    nonzero = evaluate(baseline, c, return_detail=True)
    intervention = dict(
        input_changed=not torch.equal(base_eval["input"], nonzero["input"]),
        non_fc_input_unchanged=torch.equal(base_eval["input"][:, :64], nonzero["input"][:, :64]),
        gfr_changed=not torch.equal(base_eval["gfr"], nonzero["gfr"]),
        graph_changed=any(not torch.equal(a, b) for a, b in zip(base_eval["adjs"], nonzero["adjs"])),
        cam_changed=not torch.equal(base_eval["cam"], nonzero["cam"]),
        classifier_changed=not torch.equal(base_eval["logp"], nonzero["logp"]),
        nonzero_reliability_connections=int(np.count_nonzero(stats["reliability"])),
        gamma_test=.25,
    )
    if intervention["nonzero_reliability_connections"] > 0:
        assert all(intervention[key] for key in
                   ("input_changed", "gfr_changed", "graph_changed", "cam_changed", "classifier_changed"))
    assert intervention["non_fc_input_unchanged"]
    correction_norm = nonzero["delta"].norm(dim=1)
    original_fc_norm = base_eval["input"][:, 64:].norm(dim=1)
    max_correction_ratio = float((correction_norm / original_fc_norm.clamp_min(1e-12)).max().item())
    assert max_correction_ratio <= .1500001
    assert torch.isfinite(nonzero["input"]).all()
    assert load_calls["clip_load"] == 1 and load_calls["prompt_features"] == 1, load_calls
    result = dict(gpu=gpu, status="PASS", seed=0, fold=1,
                  historical_b0_initial_trunk_exact=True,
                  historical_b0_initial_rng_exact=True,
                  registered_pen_constructed=True, b0_forward_pen_bypassed=True,
                  clip_model_loads=load_calls["clip_load"],
                  prompt_feature_calls=load_calls["prompt_features"],
                  gamma_zero=dict(input_exact=input_exact, gfr_exact=gfr_exact,
                                  adjacency_exact=adjacency_exact, cam_exact=cam_exact,
                                  classifier_logp_exact=logp_exact,
                                  loss_exact=baseline_step["loss"] == candidate_step["loss"],
                                  common_gradients_exact=grad_comparison,
                                  common_first_adam_update_exact=update_exact,
                                  gamma_gradient_at_zero=qcar_grad,
                                  gamma_gradient_required=bool(intervention_count)),
                  mlp_all_grad_none=True,
                  qcar_optimizer_registration=dict(unique=True, lr=PROTOCOL["lr"], weight_decay=0.0),
                  control_differences=dict(G3_only_replaces_w_with_ones=True,
                                           G4_G5_same_weights_and_direction_source=True,
                                           G5_nonzero_sitewise_permutation=True,
                                           G7_only_sign_reversed=True),
                  max_patient_correction_ratio=max_correction_ratio,
                  checkpoint_roundtrip_exact=True, statistics_sha256=stats["statistics_sha256"],
                  test_label_invariance=True, outer_test_fc_invariance_for_qcar=True,
                  oof_diagnostics=json_stats(stats), nonzero_gamma_intervention=intervention,
                  qcar_data_fitter_accepts_labels=False)
    json_write(RUNTIME / f"preflight_gpu{gpu}.json", result)
    network.clip.load = original_load
    network.get_prompt_text_features = original_get_features
    print(f"QCAR_PREFLIGHT_PASS gpu={gpu} gamma_grad={qcar_grad:.8g} "
          f"reliable={intervention['nonzero_reliability_connections']}", flush=True)
    close_helper(baseline)


def history_protocol():
    manifest = json.loads((EXP32 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == PROTOCOL
    return manifest


def prepare():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "state").mkdir(parents=True, exist_ok=True)
    for path in HISTORY_FILES + DATA_FILES + CORE_FILES:
        assert (ROOT / path).is_file(), f"required audit input missing: {path}"
    history_protocol()
    assert (Path.home() / ".cache/clip/RN50.pt").is_file(), "cached RN50 checkpoint missing; no download allowed"
    emit(ROOT, SOURCE, EXP / "integration.patch")
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert manifest["source_sha256"] == source_hashes()
        if manifest["code_sha256"] != code_hashes() or manifest.get("prompt_sha256") != sha_file(ROOT / "prompt"):
            before_formal = manifest["status"] == "PREPARED" and not manifest.get("formal_started_utc")
            failed_start = manifest["status"] == "RUNNING" and manifest.get("completed_folds", 0) == 0 \
                and all(not task_path(config).exists() for config in CONFIGS) \
                and all(not (RUNTIME / "state" / f"{config}.json").exists() for config in CONFIGS)
            existing_states = {config: torch_load(task_path(config)) for config in CONFIGS
                               if task_path(config).exists()}
            checkpoint_repair = manifest["status"] == "RUNNING" \
                and manifest.get("completed_folds", 0) == 0 \
                and bool(existing_states) \
                and all(state.get("config") == config
                        and state.get("phase") in {"EPOCH", "BOUNDARY", "COMPLETE"}
                        and len(state.get("rows", [])) <= 10
                        for config, state in existing_states.items())
            if checkpoint_repair:
                for config in existing_states:
                    lock_path = RUNTIME / "state" / f"{config}.lock"
                    if not lock_path.exists():
                        continue
                    with lock_path.open("a+") as task_lock:
                        try:
                            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            checkpoint_repair = False
                            break
            assert before_formal or failed_start or checkpoint_repair, \
                "code is frozen after formal folds; only a stopped, zero-committed-fold checkpoint repair is allowed"
            if failed_start:
                history = manifest.setdefault("zero_fold_startup_repairs", [])
                history.append(dict(
                    failed_launch_utc=manifest.get("formal_started_utc"),
                    previous_code_sha256=manifest["code_sha256"],
                    reason="Workers exited before model construction because PyTorch interop thread count was set twice; zero folds/checkpoints existed.",
                ))
                manifest["status"] = "PREPARED"
                manifest.pop("formal_started_utc", None)
            if checkpoint_repair:
                manifest.setdefault("in_progress_repairs", []).append(dict(
                    repaired_utc=utc(), previous_code_sha256=manifest["code_sha256"],
                    checkpoints={config: dict(phase=state["phase"], fold=state.get("fold"),
                                              completed_rows=len(state.get("rows", [])))
                                 for config, state in existing_states.items()},
                    reason="Fix duplicate diagnostic keys and resume early-stopped epoch checkpoints without running another epoch; training objective and parameter updates are unchanged.",
                ))
            manifest["code_sha256"] = code_hashes()
            manifest["prompt_sha256"] = sha_file(ROOT / "prompt")
            manifest["input_sha256"] = input_hashes()
            manifest["preflight"] = dict(passed=False)
            manifest["prepared_utc"] = utc()
            json_write(MANIFEST_PATH, manifest)
            if checkpoint_repair:
                new_signatures = {config: task_signature(config) for config in existing_states}
                for config, state in existing_states.items():
                    state["signature"] = new_signatures[config]
                    torch_write(task_path(config), state)
                manifest["checkpoint_signature_migration"] = dict(
                    migrated_configs=sorted(existing_states),
                    new_signature_prefixes={key: value[:16] for key, value in new_signatures.items()},
                    migrated_utc=utc(), model_optimizer_rng_and_fold_rows_preserved=True,
                )
                manifest["source_signature_sha256"] = source_signature()
                json_write(MANIFEST_PATH, manifest)
        assert manifest["configs"] == list(CONFIGS)
        return manifest
    x, y, folds, mapping = read_data()
    data_hash = input_hashes()
    historical_csv = EXP32 / "results/fold_metrics.csv"
    handoff = ROOT / "handoff.md"
    history_sha = {path: sha_file(ROOT / path) for path in HISTORY_FILES}
    manifest = dict(
        experiment="Experiment 33 QCAR — Quality-Conditioned Artifact-Residual Alignment",
        status="PREPARED", created_utc=utc(), seed=0, configs=list(CONFIGS),
        target_folds=80, completed_folds=0,
        source_sha256=source_hashes(), code_sha256=code_hashes(),
        prompt_sha256=sha_file(ROOT / "prompt"),
        experiment32_and_audit_sha256=history_sha, input_sha256=data_hash,
        historical_b0_csv_sha256=sha_file(historical_csv),
        handoff_prefix=dict(size=handoff.stat().st_size, sha256=sha_file(handoff)),
        protocol=PROTOCOL,
        data=dict(shape=list(x.shape),
                  label_counts={"ASD_0": int(np.sum(y == 0)), "TD_1": int(np.sum(y == 1))},
                  modality_order=list(mapping), modality_dims=INPUT_DIMS,
                  pheno_columns={"age": [0, 26], "site": [26, 46], "sex": [46, 48]},
                  pheno_names=[str(value) for value in mapping["PHENO"]],
                  fold_train_sizes=[int(len(train)) for train, _ in folds],
                  fold_test_sizes=[int(len(test)) for _, test in folds]),
        paper_audit=dict(
            source_url="https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
            author_repository="https://github.com/wanxixi11/SPromptGL",
            current_repository="https://github.com/tangaii/SPromptGL",
            findings="Paper Figure 2/Section 3.3 attributes the largest reported ABIDE ablation change to GFR. Paper Eq. 1 includes a label-graph term not present in the current simple-2 optimized loss. Current B0-A source and Experiment 32 runtime protocol take precedence; historical implementation quirks are preserved.",
            provenance="Current checked-out source hashes and historical runtime hashes are recorded above; author code creates a registered CLIP RN50 model, while current local source uses audited cached prompt features. B0-A comparisons use the current Experiment 32 implementation only."),
        qcar=dict(
            ridge_lambda=1.0, ridge_standardization="float64, per fit subset; intercept via centering; standard deviation floor 1",
            conditional_model="C->Q, C->F, then residual Q->residual F; all fitted on outer-train only",
            reliability="3-fold shuffled KFold on outer-train only; OOF MSE C->F vs C+Q->F; gain=(mse_C-mse_CQ)/(mse_C+1e-12); w=clip(max(gain,0),0,1)",
            gamma="single raw scalar initialized 0, straight-through clamp in forward to [0,.25], projected after each optimizer step",
            correction="F_new=F+sign*min_norm(gamma*(w*Fhat_Q),0.15*||F||); sign=-1 for QCAR, PHENO/ANAT/FUNC are copied unchanged",
            fitted_without_labels=True, outer_test_fc_not_used_for_fit=True),
        groups={key: dict(direction=GROUP_METHOD[key][0], reliability=GROUP_METHOD[key][1],
                           sign=GROUP_METHOD[key][2]) for key in CONFIGS},
        gpu_assignment={str(gpu): list(groups) for gpu, groups in GPU_CONFIGS.items()},
        worker_protocol="One independent seed-0 process per configuration; folds 1-10 in order with one continuous RNG stream per configuration. Four workers per physical GPU.",
        b0a_checkpoint_reused=False, other_experiment_checkpoint_reused=False,
        checkpoint_policy=dict(epoch_interval=CHECKPOINT_EVERY,
                               atomic=True, best_state_includes_qcar_gamma=True,
                               completed_fold_boundary_restores_python_numpy_torch_cpu_cuda_rng=True),
        preflight=dict(passed=False), decision_thresholds=dict(
            min_g4_acc_gain_vs_g0_pp=.23, g4_sen_noninferiority_pp=-SEN_NONINFERIORITY_PP,
            g4_must_exceed_g5_acc_and_hard_auc=True,
            g4_not_weaker_than_g6_acc_and_hard_auc=True,
            qcar_oof_gain_mean_positive=True, correction_nonzero=True),
        workers={config: dict(gpu=next(g for g, names in GPU_CONFIGS.items() if config in names),
                              status="PENDING", completed_folds=0) for config in CONFIGS},
        source_signature_sha256=source_signature(),
    )
    json_write(MANIFEST_PATH, manifest)
    return manifest


def run_worker(config, gpu):
    import torch
    assert config in CONFIGS and gpu in GPU_CONFIGS and config in GPU_CONFIGS[gpu]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert int(os.environ.get("QCAR_PHYSICAL_GPU", -1)) == gpu
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == source_hashes()
    assert manifest["code_sha256"] == code_hashes()
    c = context()
    c["device"] = torch.device("cuda")
    (RUNTIME / "state").mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "state" / f"{config}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint_path = task_path(config)
    saved = torch_load(checkpoint_path) if checkpoint_path.exists() else None
    signature = task_signature(config)
    if saved:
        assert saved["signature"] == signature and saved["config"] == config
        if saved["phase"] == "COMPLETE":
            print(f"QCAR_SKIP_COMPLETE config={config}", flush=True)
            return
    started_utc = saved.get("started_utc", utc()) if saved else utc()
    seed_everything(torch, 0)
    network = c["network"]
    load_calls = {"clip_load": 0, "prompt_features": 0}
    original_load, original_get = network.clip.load, network.get_prompt_text_features

    def tracked_load(*args, **kwargs):
        load_calls["clip_load"] += 1
        return original_load(*args, **kwargs)

    def tracked_get(*args, **kwargs):
        load_calls["prompt_features"] += 1
        return original_get(*args, **kwargs)

    network.clip.load = tracked_load
    network.get_prompt_text_features = tracked_get
    if saved:
        network.get_prompt_text_features()
        rng_restore(torch, saved["rng"])
    stop = {"requested": False}

    def request_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    rows = list(saved.get("rows", [])) if saved else []
    diagnostics = list(saved.get("diagnostics", [])) if saved else []
    phase = saved.get("phase") if saved else None
    first_fold = int(saved.get("next_fold", len(rows) + 1)) if phase == "BOUNDARY" else len(rows) + 1
    if phase == "EPOCH":
        first_fold = int(saved["fold"])
    worker_started = time.monotonic()
    previous_seconds = float(saved.get("seconds", 0.0)) if saved else 0.0
    atomic_task_json(config, dict(config=config, gpu=gpu, status="RUNNING",
                                  completed_folds=len(rows), started_utc=started_utc,
                                  updated_utc=utc()))
    log_path = RUNTIME / f"{config}.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"TASK_START {config} GPU={gpu} resumes={int(saved is not None)}", flush=True)
        epoch_resume = saved if phase == "EPOCH" else None
        for fold in range(first_fold, 11):
            if stop["requested"]:
                break
            train, test = c["folds"][fold - 1]
            arrays = stats = None
            if config != "G0_B0A":
                arrays, stats = fit_fold_stats(c, fold)
                expected_sha = stats["statistics_sha256"]
                if epoch_resume and int(epoch_resume["fold"]) == fold:
                    assert epoch_resume["statistics_sha256"] == expected_sha
            else:
                expected_sha = None
            fold_entry_rng = rng_capture(torch)
            agent = make_agent(config, fold, c, stats, arrays)
            if epoch_resume and int(epoch_resume["fold"]) == fold:
                load_model_state(agent, epoch_resume["model"])
                agent.optimizer_MP0.load_state_dict(epoch_resume["optimizer_mp"])
                agent.optimizer_MF.load_state_dict(epoch_resume["optimizer_mf"])
                rng_restore(torch, epoch_resume["rng"])
                best_state = epoch_resume["best_state"]
                best_acc = float(epoch_resume["best_acc"])
                best_epoch = int(epoch_resume["best_epoch"])
                wait = int(epoch_resume["wait"])
                start_epoch = int(epoch_resume["epoch"]) + 1
                fold_seconds_before = float(epoch_resume.get("fold_seconds", 0.0))
                initial_trunk_sha256 = epoch_resume["initial_trunk_sha256"]
                initial_rng_sha256 = epoch_resume["initial_rng_sha256"]
                resume_already_stopped = wait > PROTOCOL["early"]
                fold_started = time.monotonic()
            else:
                best_state, best_acc, best_epoch, wait = None, 0.0, -1, 0
                start_epoch = 0
                fold_seconds_before = 0.0
                initial_trunk_sha256 = parameter_hash(agent)
                initial_rng_sha256 = rng_hash(fold_entry_rng)
                resume_already_stopped = False
                fold_started = time.monotonic()
            epoch_resume = None
            last_epoch = start_epoch - 1
            epoch_range = () if resume_already_stopped else range(start_epoch, PROTOCOL["nepoch"])
            for epoch in epoch_range:
                step_result = train_step(agent, c)
                output = evaluate(agent, c)
                prediction = output[agent.val_idx].argmax(1)
                val_correct = int(prediction.eq(agent.targ[agent.val_idx]).double().sum().item())
                val_acc = val_correct / len(agent.val_idx)
                if val_acc > best_acc:
                    best_acc, best_epoch, wait = val_acc, epoch, 0
                    best_state = model_state(agent)
                else:
                    wait += 1
                last_epoch = epoch
                should_stop = wait > PROTOCOL["early"] or stop["requested"]
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or should_stop:
                    current = dict(
                        signature=signature, config=config, phase="EPOCH", fold=fold,
                        epoch=epoch, model=model_state(agent),
                        optimizer_mp=agent.optimizer_MP0.state_dict(),
                        optimizer_mf=agent.optimizer_MF.state_dict(), rng=rng_capture(torch),
                        best_state=best_state, best_acc=best_acc, best_epoch=best_epoch,
                        wait=wait, fold_seconds=fold_seconds_before + time.monotonic() - fold_started,
                        initial_trunk_sha256=initial_trunk_sha256,
                        initial_rng_sha256=initial_rng_sha256,
                        statistics_sha256=expected_sha, rows=rows, diagnostics=diagnostics,
                        seconds=previous_seconds + time.monotonic() - worker_started,
                        started_utc=started_utc,
                    )
                    torch_write(checkpoint_path, current)
                    atomic_task_json(config, dict(config=config, gpu=gpu, status="RUNNING",
                                                  completed_folds=len(rows), current_fold=fold,
                                                  current_epoch=epoch, started_utc=started_utc,
                                                  updated_utc=utc()))
                if should_stop:
                    break
            if stop["requested"]:
                break
            assert best_state is not None
            continuation_rng = rng_capture(torch)
            load_model_state(agent, best_state)
            output = evaluate(agent, c)
            metrics = metrics_row(output, c["y"], np.asarray(test))
            selected_correct = int(output[agent.val_idx].argmax(1).eq(agent.targ[agent.val_idx]).sum().item())
            assert round(best_acc * len(agent.val_idx)) == selected_correct
            assert abs(metrics["acc"] - best_acc) <= 1e-12
            base_row = dict(config=config, seed=0, fold=fold, physical_gpu=gpu, **metrics,
                            best_epoch=best_epoch, epochs_executed=last_epoch + 1,
                            initial_trunk_sha256=initial_trunk_sha256,
                            initial_rng_sha256=initial_rng_sha256,
                            wall_seconds=fold_seconds_before + time.monotonic() - fold_started)
            if config != "G0_B0A":
                gamma = float(agent.qcar.gamma().detach().cpu().item())
                q_diag = calibration_diagnostics(c["x"][:, 64:], stats["q_residual_norm_all"],
                                                 arrays[GROUP_METHOD[config][0]],
                                                 stats["reliability"] if GROUP_METHOD[config][1] == "reliability"
                                                 else np.ones(256), gamma,
                                                 stats["q_residual_norm_train_p80"])
                base_row["gamma_final"] = gamma
                diagnostic = {**json_stats(stats), **q_diag, "config": config,
                              "seed": 0, "fold": fold, "physical_gpu": gpu}
            else:
                base_row["gamma_final"] = 0.0
                diagnostic = dict(config=config, seed=0, fold=fold, physical_gpu=gpu,
                                  gamma=0.0, mean_relative_correction=0.0,
                                  max_relative_correction=0.0, n_subjects_nonzero_correction=0)
            rows.append(base_row)
            diagnostics.append(diagnostic)
            rng_restore(torch, continuation_rng)
            previous_seconds += time.monotonic() - worker_started
            worker_started = time.monotonic()
            saved_boundary = dict(signature=signature, config=config, phase="BOUNDARY",
                                  next_fold=fold + 1, rng=continuation_rng, rows=rows,
                                  diagnostics=diagnostics, seconds=previous_seconds,
                                  started_utc=started_utc)
            torch_write(checkpoint_path, saved_boundary)
            atomic_task_json(config, dict(config=config, gpu=gpu, status="RUNNING",
                                          completed_folds=len(rows), completed_fold_ids=[r["fold"] for r in rows],
                                          started_utc=started_utc, updated_utc=utc()))
            print(f"FOLD_DONE {config} fold={fold} acc={metrics['acc']:.6f} "
                  f"hard_auc={metrics['hard_auc']:.6f} gamma={base_row['gamma_final']:.6f}", flush=True)
            close_helper(agent)
            del agent, output, best_state
            torch.cuda.empty_cache()
        if not stop["requested"]:
            assert len(rows) == 10
            done = dict(signature=signature, config=config, phase="COMPLETE", rows=rows,
                        diagnostics=diagnostics, seconds=previous_seconds,
                        started_utc=started_utc, completed_utc=utc(), clip_calls=load_calls)
            torch_write(checkpoint_path, done)
            atomic_task_json(config, dict(config=config, gpu=gpu, status="COMPLETE",
                                          completed_folds=10, seconds=previous_seconds,
                                          clip_calls=load_calls, started_utc=started_utc,
                                          completed_utc=utc()))
            print(f"TASK_COMPLETE {config} seconds={previous_seconds:.2f}", flush=True)
    log.close()
    network.clip.load, network.get_prompt_text_features = original_load, original_get


def historical_b0_rows():
    with (EXP32 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        source = [row for row in csv.DictReader(stream) if row["config"] == "B0" and row["seed"] == "0"]
    assert len(source) == 10
    return {int(row["fold"]): row for row in source}


def aggregate():
    all_rows, all_diagnostics = [], []
    for config in CONFIGS:
        result = torch_load(task_path(config))
        assert result["phase"] == "COMPLETE" and len(result["rows"]) == 10
        all_rows.extend(result["rows"])
        all_diagnostics.extend(result["diagnostics"])
    historical = historical_b0_rows()
    current_b0 = {int(row["fold"]): row for row in all_rows if row["config"] == "G0_B0A"}
    assert len(current_b0) == 10
    for row in all_rows:
        old = historical[row["fold"]]
        row["exp32_b0a_acc"] = float(old["acc"])
        row["exp32_b0a_hard_auc"] = float(old["auc"])
        row["exp32_b0a_sen_asd"] = float(old["sen_asd"])
        row["exp32_b0a_spe_td"] = float(old["spe_td"])
        row["exp32_b0a_best_epoch"] = int(old["best_epoch"])
        row["exp32_b0a_initial_parameter_sha256"] = old["initial_trunk_sha256"]
        row["exp32_b0a_initial_rng_sha256"] = old["initial_rng_sha256"]
        baseline = current_b0[row["fold"]]
        for metric in ("acc", "hard_auc", "sen_asd", "spe_td"):
            row["g0_current_" + metric] = float(baseline[metric])
            row["delta_vs_g0_" + metric] = float(row[metric]) - float(baseline[metric])
    sorted_rows = sorted(all_rows, key=lambda r: (CONFIGS.index(r["config"]), r["fold"]))
    csv_write(RESULTS / "fold_metrics.csv", sorted_rows)
    dfields = list(dict.fromkeys(key for diagnostic in all_diagnostics for key in diagnostic))
    sorted_diagnostics = sorted(all_diagnostics, key=lambda r: (CONFIGS.index(r["config"]), r["fold"]))
    csv_write(RESULTS / "diagnostics.csv", [
        {key: diagnostic.get(key, "") for key in dfields} for diagnostic in sorted_diagnostics
    ])
    summary = {}
    for config in CONFIGS:
        group = [row for row in all_rows if row["config"] == config]
        summary[config] = {}
        for metric in ("acc", "hard_auc", "sen_asd", "spe_td"):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            summary[config][metric] = dict(mean=float(values.mean()), population_sd=float(values.std(ddof=0)))
    summary["HISTORICAL_EXP32_B0A"] = {}
    old = list(historical.values())
    for metric, source in (("acc", "acc"), ("hard_auc", "auc"),
                           ("sen_asd", "sen_asd"), ("spe_td", "spe_td")):
        values = np.asarray([float(row[source]) for row in old])
        summary["HISTORICAL_EXP32_B0A"][metric] = dict(mean=float(values.mean()), population_sd=float(values.std(ddof=0)))
    comparisons = {}
    lookup = {(row["config"], row["fold"]): row for row in all_rows}
    for control in ("G0_B0A", "G1_SITE_ONLY", "G2_UNCONDITIONAL", "G3_NO_RELIABILITY",
                    "G5_WRONG_PAIR", "G6_FC_ONLY", "G7_REVERSE_SIGN"):
        comparison = {}
        for metric in ("acc", "hard_auc", "sen_asd", "spe_td"):
            diffs = [float(lookup[("G4_QCAR_FULL", fold)][metric])
                     - float(lookup[(control, fold)][metric]) for fold in range(1, 11)]
            wins = sum(value > 1e-12 for value in diffs)
            losses = sum(value < -1e-12 for value in diffs)
            ties = len(diffs) - wins - losses
            comparison[metric] = dict(mean_delta=float(np.mean(diffs)),
                                      wins=wins, ties=ties, losses=losses,
                                      fold_deltas=diffs)
        comparisons[control] = comparison
    summary["paired_G4_comparisons"] = comparisons
    historical_comparison = {}
    for metric, source in (("acc", "acc"), ("hard_auc", "auc"),
                           ("sen_asd", "sen_asd"), ("spe_td", "spe_td")):
        diffs = [float(lookup[("G4_QCAR_FULL", fold)][metric]) - float(historical[fold][source])
                 for fold in range(1, 11)]
        historical_comparison[metric] = dict(
            mean_delta=float(np.mean(diffs)),
            wins=sum(value > 1e-12 for value in diffs),
            ties=sum(abs(value) <= 1e-12 for value in diffs),
            losses=sum(value < -1e-12 for value in diffs),
            fold_deltas=diffs,
        )
    summary["paired_G4_vs_HISTORICAL_EXP32_B0A"] = historical_comparison
    json_write(RESULTS / "summary.json", summary)
    return all_rows, all_diagnostics, summary


def percent(mean, sd):
    return f"{100*mean:.4f} ± {100*sd:.4f}%"


def make_report(rows, diagnostics, summary, manifest):
    group_names = {
        "G0_B0A": "Original B0-A baseline; no QCAR",
        "G1_SITE_ONLY": "Site-only predicted FC component",
        "G2_UNCONDITIONAL": "Centered C+Q to FC ridge prediction",
        "G3_NO_RELIABILITY": "Conditional QC-FC with w=1",
        "G4_QCAR_FULL": "Conditional QC-FC with OOF reliability (primary)",
        "G5_WRONG_PAIR": "G4 direction site-wise mismatched",
        "G6_FC_ONLY": "FC residual direction, no QC-to-FC map",
        "G7_REVERSE_SIGN": "G4 direction added instead of subtracted",
    }
    lines = [
        "# Experiment 33: QCAR Conditional Cross-Modal Alignment", "",
        f"Status: **{manifest['status']}**, {manifest['completed_folds']}/80 folds. GPU 0 completed 40 folds and GPU 1 completed 40 folds. Wall time by worker and exact fold assignment are in `manifest.json`.", "",
        "## Audit and Protocol", "",
        "The MICCAI 2025 SPromptGL paper describes multi-relation GFR and CAM/PEN (SPGS+RES). Its Figure 2 reports a large contribution from GFR; the local source is the authority for this reproduction because paper equations and current `simple-2` code differ in the label-graph loss and graph-fusion details. Current Experiment 32 B0-A is reproduced with the existing PLN constructor, including the inactive MLP, CLIP buffer and avg_pool, while the forward path bypasses PEN and passes CAM flatten directly to the unchanged OutputLayer.", "",
        "Experiment 18/23/24/25/27/28/29/30/31 repeatedly found that semantic/QC auxiliary alignment or graph calibration did not reliably improve the frozen baseline or its controls. This experiment therefore tests a label-free, outer-train-only conditional QC-FC estimate before GFR and includes site-only, unconditional, reliability-off, wrong-pair, FC-only and reverse-sign controls. CORR256 is the only modified input. ANAT6/FUNC10 are QC proxies rather than anatomy/activation truth; the estimated FC component is a conditional association, not a verified artifact.", "",
        "Data are the existing 871×320 standardized/RFE-selected ABIDE table, without reprocessing. PHENO columns were verified in `modal_feat_dict.npy`: age-bin [0:26], site [26:46], sex [46:48]; Q=ANAT6+FUNC10 [48:64]; F=CORR256 [64:320]. Splits are `StratifiedKFold(10, shuffle=True, random_state=10)`, labels ASD=0/TD=1, and full 871-subject transductive GFR remains unchanged.", "",
        "The original seed-0 simple-2 protocol is retained: `ClsLoss + mean(GraphConstructLoss over four input modalities)`, both original Adam optimizers, 450 epoch cap, patience 200, `val_idx=test_idx`, and a checkpoint is replaced only on strict ACC improvement. `ALoss` is still calculated but excluded from the loss. The four reported classification metrics are ACC, one-hot hard-label ROC-AUC, SEN-ASD and SPE-TD. No historical checkpoint was loaded.", "",
        "Each of the eight independent workers initializes seed=0 once, then runs folds 1–10 in order with one continuous Python/NumPy/PyTorch CPU/CUDA RNG stream. Four workers run per physical GPU. QCAR statistics use local deterministic NumPy generators and do not consume the model RNG.", "",
        "For each fold, standardized float64 ridge fits use only outer-train rows. C→Q and C→F residualization precede QC-residual→FC-residual prediction. Reliability compares three-fold outer-train OOF C→F and C+Q→F errors, with `gain=(MSE_C−MSE_C+Q)/(MSE_C+1e-12)` and `w=clip(max(gain,0),0,1)`. Site-wise wrong pairing uses a deterministic within-site permutation. A single gamma starts at zero, uses straight-through gradients with forward projection into [0,.25], and each patient's correction is capped at 15% of its CORR norm. All four non-FC modalities are copied unchanged.", "",
        "## Four-Metric Results", "",
        "Mean ± population SD over ten folds, in percent. The historical Experiment 32 B0-A row is read-only and is not combined with the new folds.", "",
        "| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |", "|---|---:|---:|---:|---:|"]
    for config in CONFIGS:
        values = summary[config]
        lines.append("| " + config + " | " + " | ".join(
            percent(values[metric]["mean"], values[metric]["population_sd"])
            for metric in ("acc", "hard_auc", "sen_asd", "spe_td")) + " |")
    old = summary["HISTORICAL_EXP32_B0A"]
    lines += [f"| Historical Experiment 32 B0-A (read-only) | {percent(old['acc']['mean'], old['acc']['population_sd'])} | {percent(old['hard_auc']['mean'], old['hard_auc']['population_sd'])} | {percent(old['sen_asd']['mean'], old['sen_asd']['population_sd'])} | {percent(old['spe_td']['mean'], old['spe_td']['population_sd'])} |", "",
              "## Paired Comparisons", "",
              "Each difference is G4 minus the control in percentage points, paired by fold. Win/tie/loss counts are descriptive; these overlapping folds are not independent cohorts.", "",
              "| Comparison | Metric | Mean Δ (pp) | Fold W/T/L |", "|---|---|---:|---:|"]
    for control, metrics in summary["paired_G4_comparisons"].items():
        for metric, label in (("acc", "ACC"), ("hard_auc", "Hard-AUC"),
                              ("sen_asd", "SEN-ASD"), ("spe_td", "SPE-TD")):
            result = metrics[metric]
            lines.append(f"| G4 − {control} | {label} | {100*result['mean_delta']:+.4f} | {result['wins']}/{result['ties']}/{result['losses']} |")
    current_g0 = summary["G0_B0A"]
    historical_g4 = summary["paired_G4_vs_HISTORICAL_EXP32_B0A"]
    audit = manifest["g0_historical_reproduction"]
    lines += ["", f"Current G0 ACC is {percent(current_g0['acc']['mean'], current_g0['acc']['population_sd'])}; historical B0-A ACC was {percent(old['acc']['mean'], old['acc']['population_sd'])}. G0 matched historical initial trunk hashes on {audit['initial_parameter_hash_exact_folds']}/10 folds, initial RNG hashes on {audit['initial_rng_hash_exact_folds']}/10, best epochs on {audit['best_epoch_exact_folds']}/10, and all four metrics on {audit['four_metric_exact_folds']}/10. Fold-level values are in `fold_metrics.csv`; no historical checkpoint was loaded.", "",
              "### B0-A Fold-by-Fold Comparison", "",
              "Percent values are shown; deltas are new G0 minus the historical Experiment 32 B0-A row for the same fold.", "",
              "| Fold | E32 ACC | New G0 ACC | Δ ACC (pp) | E32 Hard-AUC | New G0 Hard-AUC | Δ AUC (pp) | E32 SEN | New G0 SEN | Δ SEN (pp) | E32 SPE | New G0 SPE | Δ SPE (pp) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    g0_rows = {int(row["fold"]): row for row in rows if row["config"] == "G0_B0A"}
    for fold in range(1, 11):
        row = g0_rows[fold]
        lines.append(
            f"| {fold} | {100*row['exp32_b0a_acc']:.4f} | {100*row['acc']:.4f} | {100*row['delta_vs_g0_acc']:+.4f} | "
            f"{100*row['exp32_b0a_hard_auc']:.4f} | {100*row['hard_auc']:.4f} | {100*row['delta_vs_g0_hard_auc']:+.4f} | "
            f"{100*row['exp32_b0a_sen_asd']:.4f} | {100*row['sen_asd']:.4f} | {100*row['delta_vs_g0_sen_asd']:+.4f} | "
            f"{100*row['exp32_b0a_spe_td']:.4f} | {100*row['spe_td']:.4f} | {100*row['delta_vs_g0_spe_td']:+.4f} |"
        )
    lines += [
              "",
              "### G4 Versus Historical Experiment 32 B0-A", "",
              "Paired fold differences compare the new G4 run against Experiment 32 B0-A from the archived historical CSV; the historical weights/checkpoints were not loaded.", "",
              "| Metric | Mean Δ (pp) | Fold W/T/L |",
              "|---|---:|---:|"]
    for metric, label in (("acc", "ACC"), ("hard_auc", "Hard-AUC"),
                          ("sen_asd", "SEN-ASD"), ("spe_td", "SPE-TD")):
        item = historical_g4[metric]
        lines.append(f"| {label} | {100*item['mean_delta']:+.4f} | {item['wins']}/{item['ties']}/{item['losses']} |")
    lines += [
              "",
              "## Mechanism and Decision", "",
              "Fold-level OOF gains, reliability distributions, fitted-statistics hashes, wrong-pair rate, gamma, relative correction size and high/low QC-residual correction summaries are in `diagnostics.csv`. G4 starts each fold at gamma=0; final gamma and nonzero correction counts are reported per fold. A small or zero gamma is not evidence that QCAR worked. OOF reliability estimates prediction improvement only and does not identify causal acquisition artifacts.", "",
              f"Frozen GO conditions: G4 must exceed current G0 on ACC and hard-AUC with at least +0.23 pp ACC; exceed G5 on ACC and hard-AUC; not be weaker than G6 on ACC and hard-AUC; retain SEN-ASD within {SEN_NONINFERIORITY_PP:.2f} pp of G0; and show nonzero correction plus positive OOF mean gain. These are internal seed-0 candidate criteria, not external clinical validation."]
    if manifest["decision"]["go"]:
        lines.append("\n**Decision: GO (internal candidate only).** All frozen conditions passed. The single-seed, val=test, pre-CV RFE/standardization and transductive protocol still requires independent-site or nested evaluation before a paper claim.")
    else:
        failed = ", ".join(manifest["decision"]["failed_conditions"])
        lines.append(f"\n**Decision: NO-GO.** Failed conditions: {failed}. No control is renamed as the primary method and no post-hoc parameter search was performed.")
    lines += ["", "## Reproducibility and Limitations", "",
              "`manifest.json` records source, code, data and historical-reference hashes; configuration, GPU assignment, preflight evidence, RNG/worker state and wall times. Runtime copies of the original sources and the minimal bypass patch are isolated under `.runtime/source/` and `integration.patch`. Completed epoch checkpoints were removed after final validation; per-fold metrics and statistical fit summaries remain.", "",
              "ANAT/FUNC are imperfect acquisition-quality proxies; PHENO/site adjustment does not establish causal artifact removal. Validation equals test for checkpoint selection, the existing RFE and standardization predate outer folds, and GFR propagates over all 871 participants. These inherited conditions make the results internal historical-protocol comparisons rather than unbiased generalization evidence.", "",
              "## Fold Metrics", "", "All 80 rows with the four metrics, best epoch, hashes and same-fold G0/Experiment 32 B0-A references are in `fold_metrics.csv`.", ""]
    report_path = RESULTS / "report.md"
    tmp = report_path.with_name(report_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, report_path)


def finalize():
    rows, diagnostics, summary = aggregate()
    historical = historical_b0_rows()
    g0_rows = {int(row["fold"]): row for row in rows if row["config"] == "G0_B0A"}
    g0_audit = dict(
        initial_parameter_hash_exact_folds=sum(
            g0_rows[fold]["initial_trunk_sha256"] == historical[fold]["initial_trunk_sha256"]
            for fold in range(1, 11)),
        initial_rng_hash_exact_folds=sum(
            g0_rows[fold]["initial_rng_sha256"] == historical[fold]["initial_rng_sha256"]
            for fold in range(1, 11)),
        best_epoch_exact_folds=sum(
            int(g0_rows[fold]["best_epoch"]) == int(historical[fold]["best_epoch"])
            for fold in range(1, 11)),
        four_metric_exact_folds=sum(
            all(abs(float(g0_rows[fold][metric]) - float(historical[fold][old_metric])) <= 1e-12
                for metric, old_metric in (("acc", "acc"), ("hard_auc", "auc"),
                                          ("sen_asd", "sen_asd"), ("spe_td", "spe_td")))
            for fold in range(1, 11)),
        historical_checkpoint_reused=False,
        historical_checkpoint_used_for_comparison=False,
    )
    g4 = summary["G4_QCAR_FULL"]
    g0 = summary["G0_B0A"]
    comparisons = summary["paired_G4_comparisons"]
    d4 = [row for row in diagnostics if row["config"] == "G4_QCAR_FULL"]
    mean_gain = float(np.mean([row["gain_mean"] for row in d4]))
    nonzero_correction = sum(int(row["n_subjects_nonzero_correction"]) for row in d4) > 0
    conditions = {
        "ACC_and_hard_AUC_above_G0": g4["acc"]["mean"] > g0["acc"]["mean"] and g4["hard_auc"]["mean"] > g0["hard_auc"]["mean"],
        "ACC_gain_at_least_0.23pp": 100 * (g4["acc"]["mean"] - g0["acc"]["mean"]) >= .23,
        "G4_above_G5_on_ACC_and_hard_AUC": comparisons["G5_WRONG_PAIR"]["acc"]["mean_delta"] > 0 and comparisons["G5_WRONG_PAIR"]["hard_auc"]["mean_delta"] > 0,
        "G4_not_weaker_than_G6_on_ACC_and_hard_AUC": comparisons["G6_FC_ONLY"]["acc"]["mean_delta"] >= -1e-12 and comparisons["G6_FC_ONLY"]["hard_auc"]["mean_delta"] >= -1e-12,
        "SEN_ASD_drop_within_0.5pp": 100 * comparisons["G0_B0A"]["sen_asd"]["mean_delta"] >= -SEN_NONINFERIORITY_PP,
        "nonzero_calibration": nonzero_correction,
        "positive_mean_OOF_gain": mean_gain > 0,
    }
    failed = [key for key, value in conditions.items() if not value]
    decision = dict(go=not failed, status="GO" if not failed else "NO-GO",
                    conditions=conditions, failed_conditions=failed,
                    g4_mean_oof_gain=mean_gain,
                    g4_nonzero_correction_count=sum(int(row["n_subjects_nonzero_correction"]) for row in d4))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(status="COMPLETE", completed_folds=80, completed_utc=utc(),
                    workers={config: json.loads((RUNTIME / "state" / f"{config}.json").read_text())
                             for config in CONFIGS},
                    gpu_completed_folds={str(gpu): 40 for gpu in GPU_CONFIGS},
                    decision=decision, g0_historical_reproduction=g0_audit)
    manifest["formal_wall_seconds"] = float(sum(float(torch_load(task_path(config))["seconds"])
                                                for config in CONFIGS))
    manifest["wall_clock_seconds"] = float(max(
        datetime.fromisoformat(manifest["workers"][config]["completed_utc"]).timestamp()
        for config in CONFIGS) - min(
        datetime.fromisoformat(manifest["workers"][config]["started_utc"]).timestamp()
        for config in CONFIGS)) if all("started_utc" in manifest["workers"][config] for config in CONFIGS) else None
    json_write(MANIFEST_PATH, manifest)
    make_report(rows, diagnostics, summary, manifest)
    return manifest


def launch_workers():
    import torch
    lock = (RUNTIME / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest.get("preflight", {}).get("passed") is True, "both-GPU preflight must pass first"
    assert manifest["prompt_sha256"] == sha_file(ROOT / "prompt")
    assert manifest["input_sha256"] == input_hashes()
    manifest.update(status="RUNNING", formal_started_utc=utc())
    json_write(MANIFEST_PATH, manifest)
    procs = []
    for gpu, configs in GPU_CONFIGS.items():
        for config in configs:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       QCAR_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
            proc = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "_worker", config, str(gpu)],
                                    cwd=ROOT, env=env)
            procs.append((config, gpu, proc))
    failures = []
    for config, gpu, proc in procs:
        code = proc.wait()
        if code:
            failures.append((config, gpu, code))
    if failures:
        raise SystemExit(f"worker failures (completed folds remain resumable): {failures}")
    return finalize()


def run_preflight():
    assert MANIFEST_PATH.is_file(), "run prepare first"
    procs = []
    for gpu in (0, 1):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   QCAR_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        proc = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "_preflight", str(gpu)],
                                cwd=ROOT, env=env)
        procs.append((gpu, proc))
    failures = [(gpu, proc.wait()) for gpu, proc in procs]
    if any(code for _, code in failures):
        raise SystemExit(f"preflight failures: {failures}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["preflight"] = dict(passed=True, completed_utc=utc(),
                                 devices=[json.loads((RUNTIME / f"preflight_gpu{gpu}.json").read_text())
                                          for gpu in (0, 1)])
    json_write(MANIFEST_PATH, manifest)
    print("QCAR_PREFLIGHT_BOTH_GPUS_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "run", "_worker", "_preflight"))
    parser.add_argument("config", nargs="?")
    parser.add_argument("gpu", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        run_preflight()
    elif args.command == "_preflight":
        preflight_checks(args.gpu if args.gpu is not None else int(args.config))
    elif args.command == "_worker":
        run_worker(args.config, args.gpu)
    elif args.command == "run":
        launch_workers()


if __name__ == "__main__":
    main()
