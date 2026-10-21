#!/usr/bin/env python3
"""Experiment 32 controller, preflight, resumable workers, and report."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np

import integration
from method import DCER32, VALID

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "sources"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifest.json"
CONFIGS = ("B0", "B1", "B2", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
GPU0 = ("B0", "B2", "D1", "D3", "D5", "D7")
GPU1 = ("B1", "D0", "D2", "D4", "D6", "D8")
GPU = {name: 0 for name in GPU0} | {name: 1 for name in GPU1}
METRICS = ("acc", "auc", "sen_asd", "spe_td")
CORE = ("main.py", "model.py", "network.py", "layers.py", "utils.py", "ABIDE-simple-2-concat-weighted-cosine.sh")
CODE = ("method.py", "integration.py", "run.py", "integration.patch")
PROTOCOL = dict(
    datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
    nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
    n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
    GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
    alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
    nclass=2, mode="simple-2", seed=0,
)
WORKERS_PER_GPU = 4
SAVE_EVERY_EPOCHS = 10
_CTX = None


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_read(path):
    path = Path(path)
    for attempt in range(12):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 11:
                raise
            time.sleep(.15)


def csv_read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def record(message):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "scheduler.log").open("a", encoding="utf-8") as stream:
        stream.write(utc() + " " + str(message) + "\n")
        stream.flush()
    print(message, flush=True)


def data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    path = ROOT / "data/ABIDE"
    frame = pd.read_csv(path / "processed_standard_data.csv")
    mapping = np.load(path / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(value) for value in mapping.values()] == [48, 6, 10, 256]
    names = [str(value) for group in mapping.values() for value in group]
    assert list(frame.columns[:-1]) == names
    assert frame.shape == (871, 321)
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    assert len(folds) == 10
    for train, test in folds:
        assert not set(train) & set(test) and len(set(train) | set(test)) == 871
    return dict(x=x, y=y, folds=folds, mapping=mapping, columns=list(frame.columns))


def seed_rng(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for value in [state["cpu"]] + list(state["cuda"]):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def core_hashes():
    return {name: sha(ROOT / name) for name in CORE}


def code_hashes():
    return {name: sha(EXP / name) for name in CODE if (EXP / name).is_file()}


def build_sources():
    return integration.emit(ROOT, SOURCE, EXP / "integration.patch")


def signature():
    manifest = json_read(MANIFEST)
    return canonical({
        "protocol": manifest["protocol"], "configs": manifest["configs"],
        "source_sha256": manifest["source_sha256"], "code_sha256": manifest["code_sha256"],
        "prompt_sha256": manifest.get("prompt_sha256"),
        "control_bank_sha256": manifest.get("control_bank_sha256", {}),
    })


def initial_handoff_prefix():
    path = ROOT / "handoff.md"
    if not path.exists():
        return dict(size=0, sha256=None)
    body = path.read_bytes()
    return dict(size=len(body), sha256=bytes_sha(body))


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    assert len(CONFIGS) == 12 and sum(10 for _ in CONFIGS) == 120
    c = data()
    build_sources()
    for name in ("model.py", "network.py", "layers.py", "utils.py"):
        assert (SOURCE / name).is_file()
    if MANIFEST.exists():
        manifest = json_read(MANIFEST)
        assert manifest["source_sha256"] == core_hashes(), "protected root source changed"
        current_code = code_hashes()
        if current_code != manifest["code_sha256"]:
            if manifest.get("formal_started_utc"):
                changes = manifest.get("post_training_changes", [])
                assert changes and changes[-1].get("after_code_sha256") == current_code, \
                    "post-start code change lacks an audited, signature-preserving repair record"
            else:
                assert manifest.get("completed_folds", 0) == 0, \
                    "frozen experiment code changed after formal folds completed"
            manifest["code_sha256"] = current_code
            if not manifest.get("formal_started_utc"):
                manifest["preflight"] = {"passed": False}
                manifest["prepared_utc"] = utc()
        assert manifest["configs"] == list(CONFIGS) and manifest["target_folds"] == 120
        json_write(MANIFEST, manifest)
        return manifest
    input_files = [ROOT / "data/ABIDE/processed_standard_data.csv", ROOT / "data/ABIDE/modal_feat_dict.npy"]
    manifest = {
        "experiment": "Experiment 32 DCER — Disease-Contrastive Evidence Routing",
        "status": "PREPARED", "prepared_utc": utc(), "source_sha256": core_hashes(),
        "protected_files_sha256": {str(ROOT / name): sha(ROOT / name) for name in CORE},
        "input_sha256": {str(path): sha(path) for path in input_files},
        "code_sha256": code_hashes(), "configs": list(CONFIGS),
        "config_definitions": {
            "B0": "GFR + CAM + unchanged OutputLayer; PEN removed",
            "B1": "GFR + CAM + original RES; SPGS removed",
            "B2": "Original GFR + full unmodified SPGS+RES PEN + original OutputLayer",
            "D0": "Numeric two-slot+NULL, no text, four-token routing",
            "D1": "Text max-like unsigned routing; no signed direction CE",
            "D2": "All sixteen latent tokens densely fused",
            "D3": "Signed text+NULL+top4 routing, no keep/drop constraints",
            "D4": "Primary signed text+NULL+top4 with sufficiency and necessity losses",
            "D5": "D4 architecture with frozen cross-class prompt identity permutation",
            "D6": "D4 architecture with matched-marginal nonmedical numeric bank",
            "D7": "D4 architecture without the NULL option",
            "D8": "D4 architecture with deterministic random four-token routing",
        },
        "protocol": PROTOCOL, "n_subjects": 871,
        "label_counts": {"ASD_0": 403, "TD_1": 468},
        "fold_protocol": "seed=0 once per config; StratifiedKFold(n_splits=10,shuffle=True,random_state=10); val_idx=test_idx; full 871-subject transductive GFR; 450 epoch cap; early=200",
        "metrics": ["ACC", "one-hot hard-label ROC-AUC", "SEN-ASD", "SPE-TD"],
        "metrics_summary": "mean +/- population SD over ten folds; percentages in report",
        "target_configs": 12, "target_folds": 120, "gpu_fold_targets": {"0": 60, "1": 60},
        "gpu_assignment": GPU, "workers_per_gpu": WORKERS_PER_GPU,
        "tasks": [{"config": config, "seed": 0, "gpu": GPU[config], "status": "PENDING", "completed_folds": 0}
                  for config in CONFIGS],
        "completed_folds": 0, "completed_tasks": 0,
        "gpu_completed_folds": {"0": 0, "1": 0},
        "formal_started_utc": None, "formal_seconds": 0., "failures": [],
        "handoff_prefix": initial_handoff_prefix(), "handoff_appended": False,
        "preflight": {"passed": False}, "scope_reviews": {
            "paper": "Review requested for SPromptGL MICCAI 2025 sections 2.1/2.2/3.3, Fig. 1 and ABIDE Fig. 2; reported 91.73/92.08/92.30 are paper-reported values and remain distinct from all local runs.",
            "history": "Experiments 28-31 reviewed read-only; previous failure modes include fixed numeric prompt substitutes, unchanged scores under wrong class wording, near-uniform Experiment 31 routing, masked-state self-copy/leakage, and contaminated diagnostic records.",
            "adjacent_work": "MMGPL (MedIA 2024), BrainPrompt (MICCAI 2025), and ERASER (ACL 2020) establish prompt weighting and explanation completeness as prior art; DCER claims require predictive and evidence-faithfulness gains, not sparse attention alone.",
            "prompt_caveat": "The eight original GPT-4 descriptions are mostly general brain structure/function statements, not validated ASD-specific pathology priors.",
        },
    }
    json_write(MANIFEST, manifest)
    (EXP / "README.md").write_text(
        "# Experiment 32: DCER\n\n"
        "Disease-Contrastive Evidence Routing replaces the complete original PEN (SPGS + RES) only inside this experiment's runtime copies. The original GFR, CAM generation, and OutputLayer remain unchanged. B0/B1/B2 isolate the PEN and provide the new internal baseline; D0-D8 are frozen ablations.\n\n"
        "The formal design runs twelve configurations, one seed and ten contiguous folds per configuration, for 120 folds total. It follows the original ABIDE split, transductive graph, validation-equals-test checkpoint rule, and simple-2 optimizer/loss protocol. See `manifest.json` for hashes and `results/report.md` for the completed results.\n\n"
        "Six configurations are assigned to each physical GPU, with four independent workers per card when resources permit. Checkpoints, runtime source copies, locks, and task logs are kept in `.runtime/`. No Git commit or push is part of this experiment.\n",
        encoding="utf-8",
    )
    return manifest


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(EXP))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    import model
    import network
    import layers
    import utils
    _CTX = dict(torch=torch, model=model, network=network, layers=layers, utils=utils, **data())
    return _CTX


def make_agent(config, fold, labels=None):
    c = context()
    train, test = c["folds"][fold - 1]
    y = c["y"] if labels is None else labels
    integration.configure(config=config, fold=fold, train=train, test=test,
                          labels_train=y[train], seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = c["model"].EvalHelper([48, 6, 10, 256], c["x"], y,
                                      SimpleNamespace(**PROTOCOL), train, test)
    agent.MF_sav.close()
    agent.GCMP_sav.close()
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    return agent


def model_state(agent):
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))}


def load_models(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def recursive_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(recursive_equal(a, b) for a, b in zip(left, right))
    return left == right


def trunk_parameter_hash(agent):
    digest = hashlib.sha256()
    modules = (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))
    for module_name, module in modules:
        for name, parameter in sorted(module.named_parameters()):
            if module_name == "pln" and name.startswith("dcer32."):
                continue
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(agent, details=False):
    c = context()
    torch = c["torch"]
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        features, _, _ = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
        captured = {}
        hook = None
        branch = getattr(agent.ModalFusion, "dcer32", None)
        if details and branch is not None:
            def capture_cam(_module, inputs):
                captured["cam"] = inputs[0].detach()
            hook = branch.register_forward_pre_hook(capture_cam)
        output, _, attn_map = agent.ModalFusion(features)
        if hook is not None:
            hook.remove()
    assert output.shape == (871, 2) and torch.isfinite(output).all()
    if details:
        attention = torch.as_tensor(attn_map[-1], dtype=output.dtype, device=output.device).reshape(871, -1)
        return output, features, attention, captured.get("cam")
    return output


def metric_row(output, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y = context()["y"][test]
    prediction = output.detach().cpu().numpy()[test].argmax(1)
    tp, fn, fp, tn = map(int, confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    one_hot_y, one_hot_prediction = np.eye(2)[y], np.eye(2)[prediction]
    hard_auc = float(roc_auc_score(one_hot_y, one_hot_prediction))
    return dict(acc=float((prediction == y).mean()), auc=hard_auc,
                sen_asd=float(tp / (tp + fn)), spe_td=float(tn / (tn + fp)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def optimizer_audit(agent):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    actual = {id(parameter) for module in (agent.MessagePassing0, agent.ModalFusion)
              for parameter in module.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual
    assert all(abs(group["lr"] - PROTOCOL["lr"]) < 1e-12 for group in groups)
    branch = getattr(agent.ModalFusion, "dcer32", None)
    if branch is not None:
        assert len(groups) == 3 and groups[-1]["weight_decay"] == 0.0
        assert branch.fuse.weight.count_nonzero().item() == 0
    return sum(parameter.numel() for parameter in agent.ModalFusion.parameters())


def control_hashes(branch):
    def tensors_sha(*values):
        digest = hashlib.sha256()
        for value in values:
            if isinstance(value, str):
                digest.update(value.encode())
            else:
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    result = {"D4": tensors_sha(branch.text_bank)}
    shuffled = DCER32(branch.text_bank, "D5", branch.fold)
    numeric = DCER32(branch.text_bank, "D6", branch.fold)
    null_off = DCER32(branch.text_bank, "D7", branch.fold)
    random_route = DCER32(branch.text_bank, "D8", branch.fold)
    result["D5"] = tensors_sha(shuffled.text_bank, shuffled.mixed_order)
    result["D6"] = tensors_sha(numeric.nonmedical_bank)
    result["D7"] = tensors_sha(null_off.text_bank, "NULL_DISABLED")
    result["D8"] = tensors_sha(random_route.text_bank, "RANDOM4", str(random_route.fold))
    return result


def source_parity():
    c = context()
    torch = c["torch"]
    seed_rng(0)
    original_agent = make_agent("B2", 1)
    original_pln_state = original_agent.ModalFusion.state_dict()
    trunk_hash = trunk_parameter_hash(original_agent)
    clip_features = original_agent.ModalFusion.prompt_text_features.detach().clone()
    spec = importlib.util.spec_from_file_location("e32_original_network", ROOT / "network.py")
    original_network = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SOURCE))
    spec.loader.exec_module(original_network)
    original_network.get_prompt_text_features = lambda: clip_features
    reference = original_network.PLN([48, 6, 10, 256], SimpleNamespace(**PROTOCOL)).to(original_agent.dev)
    reference.load_state_dict(original_pln_state, strict=True)
    reference.eval()
    original_agent.ModalFusion.eval()
    with torch.no_grad():
        model_num, parts = original_agent.dataLayer(original_agent.feat)
        features, _, _ = original_agent.MessagePassing0(parts, model_num, original_agent.targ, original_agent.trn_idx)
        expected = reference(features)
        actual = original_agent.ModalFusion(features)
    assert recursive_equal(expected, actual), "B2 patched PLN differs from original PLN"

    cam_box = {}
    def capture_cam_output(_module, _inputs, output):
        cam_box["cam"] = output.detach()
    hook = original_agent.ModalFusion.FeedForward[-1].register_forward_hook(capture_cam_output)
    original_agent.ModalFusion.dcer32_config = "B0"
    with torch.no_grad():
        output0, _, attention_map = original_agent.ModalFusion(features)
    hook.remove()
    cam = cam_box["cam"]
    attention = torch.as_tensor(attention_map[-1], dtype=cam.dtype, device=cam.device).reshape(871, -1)
    expected0 = original_agent.ModalFusion.Outputlayer(cam.reshape(871, -1), attention)
    assert torch.equal(output0, expected0[0]), "B0 is not direct CAM flatten into original OutputLayer"

    part_x = cam.view(871, 4, 4, 9)
    global_x = original_agent.ModalFusion.avg_pool(cam.view(-1, 1, 36)).view(871, 4, 9)
    combine_part = []
    for index in range(part_x.shape[2]):
        part = part_x[:, :, index, :]
        vsim = torch.nn.functional.softmax(part * global_x, dim=-1)
        combine_part.append(vsim * part)
    expected_res = cam + torch.cat(combine_part, dim=2)
    expected1 = original_agent.ModalFusion.Outputlayer(expected_res.reshape(871, -1), attention)
    original_agent.ModalFusion.dcer32_config = "B1"
    with torch.no_grad():
        output1, _, _ = original_agent.ModalFusion(features)
    assert torch.equal(output1, expected1[0]), "B1 is not original RES without SPGS"

    hashes = {}
    for config in ("B0", "B1", "B2", "D4"):
        seed_rng(0)
        agent = make_agent(config, 1)
        hashes[config] = trunk_parameter_hash(agent)
        del agent
    assert len(set(hashes.values())) == 1, "GFR/CAM/OutputLayer initialization differs across configs"
    return dict(B2="BITWISE_EXACT", B0="DIRECT_CAM_FLATTEN_EXACT", B1="ORIGINAL_RES_EXACT",
                initial_trunk_sha256=trunk_hash, cross_config_initialization_sha256=hashes)


def synthetic_method_checks():
    torch = context()["torch"]
    seed_rng(32032)
    text = torch.randn((8, 1024), device="cuda")
    checks = {}
    cam = torch.randn((32, 4, 36), device="cuda")
    for config, target in (("D2", 16), ("D8", 4)):
        branch = DCER32(text, config, 3).cuda().eval()
        output, _, _ = branch(cam)
        count = branch.extra["selected"].sum(-1)
        assert torch.all(count == target)
        assert output.shape == (32, 4, 36) and torch.isfinite(output).all()
        checks[config + "_selected_count"] = target
    branch = DCER32(text, "D4", 3).cuda().eval()
    initial_fuse = branch.fuse.weight.detach().clone()
    output0, _, _ = branch(cam)
    assert torch.equal(output0, cam) and torch.equal(initial_fuse, torch.zeros_like(initial_fuse))
    assert torch.all((branch.extra["selected"].sum(-1) >= 1) & (branch.extra["selected"].sum(-1) <= 4))
    checks["D4_hard_route_1_to_4_and_zero_residual"] = True

    with torch.no_grad():
        branch.fuse.weight.normal_(0., .05)
        output, _, _ = branch(cam)
        baseline = output.clone()
        saved_bank = branch.text_bank.clone()
        branch.text_bank.copy_(saved_bank.roll(4, dims=0))
        swapped, _, _ = branch(cam)
        branch.text_bank.copy_(saved_bank)
        changed_bank = saved_bank.clone()
        changed_bank[0] = changed_bank[0] + torch.randn_like(changed_bank[0])
        branch.text_bank.copy_(changed_bank)
        changed, _, _ = branch(cam)
        branch.text_bank.copy_(saved_bank)
        changed_bank[0].zero_()
        branch.text_bank.copy_(changed_bank)
        removed, _, _ = branch(cam)
        branch.text_bank.copy_(saved_bank)
    assert not torch.equal(baseline, swapped)
    assert not torch.equal(baseline, changed)
    assert not torch.equal(baseline, removed)
    checks["text_swap_single_key_change_and_key_removal_change_output"] = True

    pa, pt, _pn, _sas, _std, _sal, signed = branch._scores(cam.reshape(32, 16, 9))
    route, selected, alternate = branch._route(branch.extra["salience"], pa, pt, branch.extra["pn"])
    part = cam.reshape(32, 16, 9)
    keep0 = branch._view(part, pa, pt, signed, selected, route)
    drop0 = branch._view(part, pa, pt, signed, 1. - selected, alternate)
    perturbed = part.clone()
    unselected = ~selected.bool()
    perturbed[unselected] += 100.
    keep1 = branch._view(perturbed, pa, pt, signed, selected, route)
    drop1 = branch._view(perturbed, pa, pt, signed, 1. - selected, alternate)
    assert torch.equal(keep0, keep1)
    selected_perturbed = part.clone()
    selected_perturbed[selected.bool()] += 100.
    drop2 = branch._view(selected_perturbed, pa, pt, signed, 1. - selected, alternate)
    assert torch.equal(drop0, drop2)
    checks["keep_drop_views_have_no_unselected_token_bypass"] = True

    mixed = torch.tensor([4, 1, 6, 3, 0, 5, 2, 7])
    assert int(((mixed[:4] >= 4).sum())) == 2 and int(((mixed[4:] >= 4).sum())) == 2
    nonmedical = DCER32(text, "D6", 3).cuda()
    assert nonmedical.nonmedical_bank.shape == (8, 1024)
    assert not torch.equal(nonmedical.nonmedical_bank, text)
    checks["D5_cross_class_mixing_and_D6_independent_numeric_bank"] = True
    return checks


def preflight_child():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert len(CONFIGS) == 12 and sum(10 for _ in CONFIGS) == 120
    for name in (EXP / "method.py", EXP / "integration.py", EXP / "run.py"):
        compile(name.read_text(encoding="utf-8"), str(name), "exec")
    c = context()
    assert c["x"].shape == (871, 320) and np.bincount(c["y"]).tolist() == [403, 468]
    checks = {"folds": 10, "subjects": 871, "classes": {"ASD": 403, "TD": 468}}
    checks["source_parity"] = source_parity()
    checks["method"] = synthetic_method_checks()

    seed_rng(0)
    d4 = make_agent("D4", 1)
    optimizer_audit(d4)
    branch = d4.ModalFusion.dcer32
    prompt_sha = hashlib.sha256(branch.text_bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    controls = control_hashes(branch)
    assert branch.text_bank.shape == (8, 1024)
    assert branch.fuse.weight.count_nonzero().item() == 0
    with torch.no_grad():
        probe, _, _ = branch(torch.randn((871, 4, 36), device="cuda"))
        assert torch.isfinite(probe).all()
        assert (branch.extra["selected"].sum(-1) >= 1).all()
        assert (branch.extra["selected"].sum(-1) <= 4).all()
    checks["frozen_RN50_8x1024_prompt_sha256"] = prompt_sha
    checks["D4_D5_D6_D7_D8_frozen_control_sha256"] = controls

    # Run two real simple-2 updates, then replay the second update from a
    # checkpoint snapshot to verify both Adam optimizers, CUDA RNG, and the
    # random_step model buffer restore exactly.
    seed_rng(320032)
    d4 = make_agent("D4", 1)
    with contextlib.redirect_stdout(io.StringIO()):
        d4.run_epoch("simple-2")
    branch = d4.ModalFusion.dcer32
    assert branch.fuse.weight.count_nonzero().item() > 0
    saved_model = model_state(d4)
    saved_mp = copy.deepcopy(d4.optimizer_MP0.state_dict())
    saved_mf = copy.deepcopy(d4.optimizer_MF.state_dict())
    saved_rng = rng_state()
    probe_checkpoint = RUNTIME / "state" / "preflight_resume.pt"
    checkpoint_write(probe_checkpoint, dict(model=saved_model, optimizer_mp=saved_mp,
                                            optimizer_mf=saved_mf, rng=saved_rng))
    disk_saved = checkpoint_read(probe_checkpoint)
    assert recursive_equal(disk_saved["model"], saved_model)
    with contextlib.redirect_stdout(io.StringIO()):
        d4.run_epoch("simple-2")
    expected_model = model_state(d4)
    expected_mp = copy.deepcopy(d4.optimizer_MP0.state_dict())
    expected_mf = copy.deepcopy(d4.optimizer_MF.state_dict())
    expected_rng = rng_state()
    gradients = {}
    for name in ("query", "key", "value", "fuse"):
        gradient = getattr(branch, name).weight.grad
        gradients[name] = float(gradient.norm().item()) if gradient is not None else 0.
    assert all(value > 0 for value in gradients.values()), gradients
    load_models(d4, disk_saved["model"])
    d4.optimizer_MP0.load_state_dict(disk_saved["optimizer_mp"])
    d4.optimizer_MF.load_state_dict(disk_saved["optimizer_mf"])
    restore_rng(disk_saved["rng"])
    with contextlib.redirect_stdout(io.StringIO()):
        d4.run_epoch("simple-2")
    assert recursive_equal(model_state(d4), expected_model)
    assert recursive_equal(d4.optimizer_MP0.state_dict(), expected_mp)
    assert recursive_equal(d4.optimizer_MF.state_dict(), expected_mf)
    assert rng_hash(rng_state()) == rng_hash(expected_rng)
    probe_checkpoint.unlink()
    checks["two_optimizer_update_and_exact_restore"] = dict(gradients=gradients, exact=True)

    test_ids = c["folds"][0][1]
    diagnostic = mechanism_diagnostic(d4, "D4", 1, test_ids)
    assert 1. <= diagnostic["selected_count_mean"] <= 4.
    assert float(diagnostic["wrong_text_output_delta"]) > 0.
    assert np.isfinite(float(diagnostic["selected_vs_random_deletion_advantage"]))
    checks["formal_diagnostic_path_and_counterfactual_restore"] = {
        "selected_count_mean": diagnostic["selected_count_mean"],
        "wrong_text_output_delta": diagnostic["wrong_text_output_delta"],
        "selected_vs_random_deletion_advantage": diagnostic["selected_vs_random_deletion_advantage"],
    }

    seed_rng(321032)
    _ = make_agent("B2", 1)
    boundary_rng = rng_state()
    fold_two_a = make_agent("B2", 2)
    fold_two_hash = trunk_parameter_hash(fold_two_a)
    del fold_two_a
    restore_rng(boundary_rng)
    fold_two_b = make_agent("B2", 2)
    assert trunk_parameter_hash(fold_two_b) == fold_two_hash
    checks["fold_boundary_next_initialization_rng"] = "EXACT"
    del fold_two_b

    random_branch = DCER32(branch.text_bank.detach(), "D8", 1).cuda().train()
    before_step = int(random_branch.random_step.item())
    with torch.no_grad():
        random_branch(torch.randn((16, 4, 36), device="cuda"))
    assert int(random_branch.random_step.item()) == before_step + 1
    saved_step = random_branch.state_dict()["random_step"].clone()
    random_branch.eval()
    with torch.no_grad():
        random_branch(torch.randn((16, 4, 36), device="cuda"))
    assert torch.equal(random_branch.state_dict()["random_step"], saved_step)
    checks["D8_random_step_buffer_checkpointed_and_eval_stable"] = int(saved_step.item())

    seed_rng(0)
    d7 = make_agent("D7", 1)
    d7_output = evaluate(d7)
    assert d7_output.shape == (871, 2) and torch.isfinite(d7_output).all()
    d7_branch = d7.ModalFusion.dcer32
    assert torch.equal(d7_branch.extra["pn"], torch.zeros_like(d7_branch.extra["pn"]))
    checks["D7_NULL_DISABLED_and_shared_original_classifier"] = True
    json_write(RUNTIME / "preflight.json", {"passed": True, "completed_utc": utc(), "checks": checks})
    print("PREFLIGHT_JSON=" + json.dumps(checks, ensure_ascii=False, allow_nan=False), flush=True)


def preflight():
    prepare()
    if os.environ.get("E32_PREFLIGHT_CHILD") == "1":
        preflight_child()
        return
    log = RUNTIME / "preflight.log"
    environment = worker_env(0)
    environment["E32_PREFLIGHT_CHILD"] = "1"
    environment["E32_PHYSICAL_GPU"] = "0"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([sys.executable, str(EXP / "run.py"), "_preflight"],
                                 cwd=RUNTIME, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        tail = log.read_text(encoding="utf-8")[-16000:]
        manifest = json_read(MANIFEST)
        manifest["preflight"] = {"passed": False, "returncode": process.returncode, "log": str(log)}
        manifest.setdefault("failures", []).append({"stage": "preflight", "time": utc(), "tail": tail})
        json_write(MANIFEST, manifest)
        raise RuntimeError("Experiment 32 preflight failed; formal training was not started:\n" + tail)
    preflight_record = json_read(RUNTIME / "preflight.json")
    manifest = json_read(MANIFEST)
    manifest["preflight"] = preflight_record
    manifest["prompt_sha256"] = preflight_record["checks"]["frozen_RN50_8x1024_prompt_sha256"]
    manifest["control_bank_sha256"] = preflight_record["checks"]["D4_D5_D6_D7_D8_frozen_control_sha256"]
    manifest["code_sha256"] = code_hashes()
    manifest["status"] = "PREFLIGHT_PASS"
    json_write(MANIFEST, manifest)


def task_name(config):
    return config + "_seed0"


def task_json(config):
    return RUNTIME / "tasks" / (task_name(config) + ".json")


def task_checkpoint(config):
    return RUNTIME / "state" / (task_name(config) + ".pt")


def checkpoint_write(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    torch.save(value, stream)
    body = stream.getvalue()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(b"E32!" + hashlib.sha256(body).hexdigest().encode() + body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_read(path):
    import torch
    raw = Path(path).read_bytes()
    assert raw[:4] == b"E32!" and hashlib.sha256(raw[68:]).hexdigest().encode() == raw[4:68]
    return torch.load(io.BytesIO(raw[68:]), map_location="cpu", weights_only=False)


def task_result(config):
    path = task_json(config)
    if not path.exists():
        return None
    result = json_read(path)
    assert result["signature"] == signature() and result["config"] == config
    assert len(result["metrics"]) == len(result["diagnostics"])
    assert [int(row["fold"]) for row in result["metrics"]] == list(range(1, len(result["metrics"]) + 1))
    return result


def random_mask(counts, n, device, seed):
    torch = context()["torch"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((n, 16), generator=generator).to(device)
    order = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(16, device=device).expand(n, -1))
    return ranks < counts[:, None]


def mechanism_diagnostic(agent, config, fold, test):
    import torch
    c = context()
    y = torch.as_tensor(c["y"], dtype=torch.long, device=agent.dev)
    branch = getattr(agent.ModalFusion, "dcer32", None)
    base = dict(config=config, seed=0, fold=fold, physical_gpu=GPU[config])
    if branch is None:
        return base | {"selected_count_mean": "", "null_probability_mean": "",
                       "semantic_contrast_mean": "", "selected_entropy_mean": "",
                       "zero_route_fraction": "", "fuse_residual_fraction": "",
                       "keep_true_logp_mean": "", "drop_true_logp_mean": "",
                       "selected_deletion_logp_loss": "", "random_deletion_logp_loss": "",
                       "selected_vs_random_deletion_advantage": "",
                       "token_count_modal0": "", "token_count_modal1": "",
                       "token_count_modal2": "", "token_count_modal3": "",
                       "direction_accuracy": "", "wrong_text_output_delta": "",
                       "wrong_text_contrast_delta": ""}
    output, features, attention, cam = evaluate(agent, details=True)
    assert cam is not None
    extra = branch.extra
    ids = torch.as_tensor(test, dtype=torch.long, device=agent.dev)
    selected = extra["selected"].bool()
    route, selected_route, alternate = branch._route(extra["salience"], extra["pa"], extra["pt"], extra["pn"])
    part = cam.reshape(871, 16, 9)
    signed = extra["pa"] - extra["pt"]
    keep_view = branch._view(part, extra["pa"], extra["pt"], signed, selected_route, route)
    keep_logp, _ = agent.ModalFusion.Outputlayer(keep_view.reshape(871, -1), attention)
    drop_view = branch._view(part, extra["pa"], extra["pt"], signed, 1. - selected_route, alternate)
    drop_logp, _ = agent.ModalFusion.Outputlayer(drop_view.reshape(871, -1), attention)
    true_logp = output[ids].gather(1, y[ids, None]).squeeze(1)
    selected_drop_logp = drop_logp[ids].gather(1, y[ids, None]).squeeze(1)
    keep_true_logp = keep_logp[ids].gather(1, y[ids, None]).squeeze(1)

    random_deletion = None
    if config != "D2":
        random_deleted = random_mask(selected.sum(-1), 871, agent.dev, 3203200 + fold)
        alternate_random = branch._top4(extra["salience"], torch.ones_like(selected), exclude=random_deleted)
        random_view = branch._view(part, extra["pa"], extra["pt"], signed,
                                   1. - random_deleted.to(part.dtype), alternate_random)
        random_logp, _ = agent.ModalFusion.Outputlayer(random_view.reshape(871, -1), attention)
        random_true_logp = random_logp[ids].gather(1, y[ids, None]).squeeze(1)
        random_deletion = (true_logp - random_true_logp).mean().item()

    output_delta = contrast_delta = ""
    if config == "D4":
        saved = branch.text_bank.detach().clone()
        try:
            with torch.no_grad():
                branch.text_bank.copy_(saved.roll(4, dims=0))
                swapped_output, _, _ = agent.ModalFusion(features)
                swapped_contrast = branch.extra["contrast"].detach().clone()
        finally:
            with torch.no_grad():
                branch.text_bank.copy_(saved)
        output_delta = float((swapped_output[ids] - output[ids]).abs().mean().item())
        contrast_delta = float((swapped_contrast[ids] - extra["contrast"][ids]).abs().mean().item())
        with torch.no_grad():
            agent.ModalFusion(features)

    count = selected[ids].sum(-1).float()
    probabilities = extra["salience"][ids] * selected[ids].float()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    cam_test = cam[ids]
    residual = extra["full"][ids] - cam_test
    residual_fraction = residual.norm(dim=(-1, -2)) / cam_test.norm(dim=(-1, -2)).clamp_min(1e-12)
    contrast = extra["contrast"][ids]
    direction_accuracy = ((contrast > 0) == (y[ids] == 0)).float().mean().item()
    modal_counts = selected[ids].reshape(-1, 4, 4).sum(-1).float().mean(0)
    null_mean = extra["pn"][ids].mean().item()
    zero_route = (route[ids].sum(-1) == 0).float().mean().item()
    selected_deletion = (true_logp - selected_drop_logp).mean().item()
    return base | {
        "selected_count_mean": float(count.mean().item()),
        "null_probability_mean": float(null_mean),
        "semantic_contrast_mean": float(contrast.mean().item()),
        "selected_entropy_mean": float(entropy.mean().item()),
        "zero_route_fraction": float(zero_route),
        "fuse_residual_fraction": float(residual_fraction.mean().item()),
        "keep_true_logp_mean": float(keep_true_logp.mean().item()),
        "drop_true_logp_mean": float(selected_drop_logp.mean().item()),
        "selected_deletion_logp_loss": float(selected_deletion),
        "random_deletion_logp_loss": "" if random_deletion is None else float(random_deletion),
        "selected_vs_random_deletion_advantage": "" if random_deletion is None else float(selected_deletion - random_deletion),
        "token_count_modal0": float(modal_counts[0].item()),
        "token_count_modal1": float(modal_counts[1].item()),
        "token_count_modal2": float(modal_counts[2].item()),
        "token_count_modal3": float(modal_counts[3].item()),
        "direction_accuracy": float(direction_accuracy),
        "wrong_text_output_delta": output_delta,
        "wrong_text_contrast_delta": contrast_delta,
    }


def worker(config):
    import torch
    assert config in CONFIGS
    manifest = json_read(MANIFEST)
    assert manifest["preflight"]["passed"]
    physical_gpu = GPU[config]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu)
    c = context()
    name = task_name(config)
    (RUNTIME / "state").mkdir(parents=True, exist_ok=True)
    (RUNTIME / "tasks").mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "state" / (name + ".lock")).open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = task_result(config)
    checkpoint_path = task_checkpoint(config)
    saved = checkpoint_read(checkpoint_path) if checkpoint_path.exists() else None
    if result and result["status"] == "COMPLETE":
        print("SKIP_COMPLETE", name, flush=True)
        return
    if saved:
        assert saved["signature"] == signature()
        result = saved["result"]
        if saved["phase"] == "BOUNDARY":
            c["network"].get_prompt_text_features()
            restore_rng(saved["rng"])
        else:
            # Constructing a fresh process may reload CLIP, but the saved RNG
            # is restored after model/optimizer state loading below.
            pass
        result["resumes"] = int(result.get("resumes", 0)) + 1
    else:
        seed_rng(0)
        result = dict(signature=signature(), config=config, seed=0,
                      physical_gpu=physical_gpu, status="RUNNING", metrics=[],
                      diagnostics=[], seconds=0., started_utc=utc(), resumes=0)
    started = time.monotonic()
    previous_seconds = float(result.get("seconds", 0.))
    stop = [False]

    def stop_worker(_signum, _frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    print("TASK_START", name, "GPU", physical_gpu, "resumed", saved is not None, flush=True)
    first_fold = len(result["metrics"]) + 1
    if saved and saved["phase"] == "EPOCH":
        first_fold = int(saved["fold"])
    for fold in range(first_fold, 11):
        if stop[0]:
            return
        resume_epoch = bool(saved and saved["phase"] == "EPOCH" and int(saved["fold"]) == fold)
        entry_rng = rng_state()
        agent = make_agent(config, fold)
        parameter_count = optimizer_audit(agent)
        if "pln_trainable_parameters" in result:
            assert result["pln_trainable_parameters"] == parameter_count
        else:
            result["pln_trainable_parameters"] = parameter_count
        if resume_epoch:
            load_models(agent, saved["model"])
            agent.optimizer_MP0.load_state_dict(saved["optimizer_mp"])
            agent.optimizer_MF.load_state_dict(saved["optimizer_mf"])
            restore_rng(saved["rng"])
            best_state = saved["best_state"]
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            wait = int(saved["wait"])
            start_epoch = int(saved["epoch"]) + 1
            tracker = saved.get("tracker", {})
            initial_hash = saved["initial_trunk_sha256"]
            initial_rng_hash = saved["initial_rng_sha256"]
            fold_elapsed_before = float(saved.get("fold_elapsed", 0.))
        else:
            best_state, best_acc, best_epoch, wait, start_epoch = None, 0., -1, 0, 0
            tracker = {}
            initial_hash = trunk_parameter_hash(agent)
            initial_rng_hash = rng_hash(entry_rng)
            fold_elapsed_before = 0.
        saved = None
        fold_started = time.monotonic()
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, PROTOCOL["nepoch"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                agent.run_epoch("simple-2")
                output = evaluate(agent)
            val_prediction = output[agent.val_idx].argmax(1)
            val_correct = int(val_prediction.eq(agent.targ[agent.val_idx]).double().sum().item())
            val_acc = float(val_correct / len(agent.val_idx))
            branch = getattr(agent.ModalFusion, "dcer32", None)
            if branch is not None:
                for key, module_name in (("query_gradient", "query"), ("key_gradient", "key"),
                                         ("value_gradient", "value"), ("fuse_gradient", "fuse")):
                    parameter = getattr(branch, module_name).weight
                    gradient = parameter.grad
                    if gradient is not None:
                        tracker[key] = max(float(tracker.get(key, 0.)), float(gradient.norm().item()))
                tracker["random_step"] = int(branch.random_step.item())
            if val_acc > best_acc:
                best_acc, best_epoch, best_state, wait = val_acc, epoch, model_state(agent), 0
            else:
                wait += 1
            last_epoch = epoch
            save_due = (epoch + 1) % SAVE_EVERY_EPOCHS == 0 or wait > PROTOCOL["early"] or stop[0]
            if save_due:
                result["seconds"] = previous_seconds + time.monotonic() - started
                checkpoint_write(checkpoint_path, dict(
                    signature=signature(), phase="EPOCH", fold=fold, epoch=epoch,
                    model=model_state(agent), optimizer_mp=agent.optimizer_MP0.state_dict(),
                    optimizer_mf=agent.optimizer_MF.state_dict(), rng=rng_state(),
                    best_state=best_state, best_acc=best_acc, best_epoch=best_epoch,
                    wait=wait, tracker=tracker,
                    fold_elapsed=fold_elapsed_before + time.monotonic() - fold_started,
                    initial_trunk_sha256=initial_hash,
                    initial_rng_sha256=initial_rng_hash, result=result,
                ))
            if stop[0] or wait > PROTOCOL["early"]:
                break
        if stop[0]:
            return
        assert best_state is not None
        continuation_rng = rng_state()
        load_models(agent, best_state)
        output = evaluate(agent)
        test = agent.tst_idx
        selected_correct = int(output[agent.val_idx].argmax(1).eq(agent.targ[agent.val_idx]).double().sum().item())
        assert round(best_acc * len(agent.val_idx)) == selected_correct, \
            "restored best checkpoint disagrees with its saved validation accuracy"
        best_acc = float(selected_correct / len(agent.val_idx))
        metrics = metric_row(output, test)
        assert abs(metrics["acc"] - best_acc) <= 1e-12
        diagnostic = mechanism_diagnostic(agent, config, fold, test)
        row = dict(config=config, seed=0, fold=fold, physical_gpu=physical_gpu, **metrics,
                   best_epoch=best_epoch, epochs_executed=last_epoch + 1,
                   initial_trunk_sha256=initial_hash, initial_rng_sha256=initial_rng_hash,
                   wall_seconds=fold_elapsed_before + time.monotonic() - fold_started)
        result["metrics"].append(row)
        result["diagnostics"].append(diagnostic)
        restore_rng(continuation_rng)
        result["seconds"] = previous_seconds + time.monotonic() - started
        result["status"] = "COMPLETE" if fold == 10 else "RUNNING"
        result["last_fold_seconds"] = time.monotonic() - fold_started
        checkpoint_write(checkpoint_path, dict(signature=signature(), phase="BOUNDARY",
                                               fold=fold + 1, rng=continuation_rng, result=result))
        json_write(task_json(config), result)
        print("FOLD_DONE", name, fold, "ACC", metrics["acc"], "AUC", metrics["auc"], flush=True)
        del agent, output, best_state
        torch.cuda.empty_cache()
    print("TASK_COMPLETE", name, result.get("seconds", 0.), flush=True)


def worker_env(gpu):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       E32_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", DGLBACKEND="pytorch")
    return environment


def gpu_snapshot():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        parsed = []
        for line in output.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 5:
                parsed.append(dict(index=fields[0], name=fields[1], memory_used=fields[2],
                                   memory_total=fields[3], utilization=fields[4]))
        return parsed
    except Exception as error:
        return {"error": repr(error)}


def task_groups():
    groups = [task_result(config) for config in CONFIGS]
    groups = [group for group in groups if group]
    return groups, [row for group in groups for row in group["metrics"]], \
        [row for group in groups for row in group["diagnostics"]]


def refresh():
    manifest = json_read(MANIFEST)
    groups, rows, diagnostics = task_groups()
    if rows:
        csv_write(RESULTS / "fold_metrics.csv", rows)
    if diagnostics:
        csv_write(RESULTS / "diagnostics.csv", diagnostics)
    lookup = {group["config"]: group for group in groups}
    for task in manifest["tasks"]:
        group = lookup.get(task["config"])
        if group:
            task.update(status=group["status"], completed_folds=len(group["metrics"]),
                        seconds=group.get("seconds", 0.),
                        pln_trainable_parameters=group.get("pln_trainable_parameters"))
    manifest.update(completed_folds=len(rows),
                    completed_tasks=sum(group["status"] == "COMPLETE" for group in groups),
                    gpu_completed_folds={str(gpu): sum(int(row["physical_gpu"]) == gpu for row in rows)
                                         for gpu in (0, 1)},
                    refreshed_utc=utc())
    json_write(MANIFEST, manifest)
    return groups, rows, diagnostics


def schedule():
    pending = [config for config in CONFIGS if not (task_result(config) or {}).get("status") == "COMPLETE"]
    active = []
    failure = None
    while pending or active:
        for config in list(pending):
            gpu = GPU[config]
            if sum(item["gpu"] == gpu for item in active) >= WORKERS_PER_GPU:
                continue
            log_path = RUNTIME / (task_name(config) + ".log")
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen([sys.executable, str(EXP / "run.py"), "_worker", config],
                                       cwd=RUNTIME, env=worker_env(gpu), stdout=log_handle,
                                       stderr=subprocess.STDOUT)
            active.append(dict(process=process, gpu=gpu, config=config, handle=log_handle, log=log_path))
            pending.remove(config)
            record(f"START {config} GPU{gpu} pid={process.pid} workers_per_gpu={WORKERS_PER_GPU}")
        time.sleep(5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            item["handle"].close()
            active.remove(item)
            record(f"EXIT {item['config']} code={code}")
            result = task_result(item["config"])
            if code or not result or result["status"] != "COMPLETE":
                failure = {"config": item["config"], "returncode": code,
                           "tail": item["log"].read_text(encoding="utf-8")[-12000:]}
                break
        refresh()
        if failure:
            for item in active:
                item["process"].terminate()
            for item in active:
                item["process"].wait(timeout=120)
                item["handle"].close()
            manifest = json_read(MANIFEST)
            manifest["status"] = "PARTIAL"
            manifest["failures"].append(dict(time=utc(), stage="schedule", error=failure))
            json_write(MANIFEST, manifest)
            raise RuntimeError("formal worker failed: " + str(failure))


def aggregate(rows):
    assert len(rows) == 120
    assert len(CONFIGS) == 12
    keyed = {}
    for row in rows:
        config, fold = row["config"], int(row["fold"])
        key = (config, int(row["seed"]), fold)
        assert config in CONFIGS and row["seed"] in (0, "0") and 1 <= fold <= 10
        assert key not in keyed
        keyed[key] = row
        n = int(row["asd_tp"]) + int(row["asd_fn"]) + int(row["td_fp"]) + int(row["td_tn"])
        assert n > 0
        acc = (int(row["asd_tp"]) + int(row["td_tn"])) / n
        sen = int(row["asd_tp"]) / (int(row["asd_tp"]) + int(row["asd_fn"]))
        spe = int(row["td_tn"]) / (int(row["td_fp"]) + int(row["td_tn"]))
        assert abs(float(row["acc"]) - acc) < 1e-12
        assert abs(float(row["sen_asd"]) - sen) < 1e-12
        assert abs(float(row["spe_td"]) - spe) < 1e-12
        assert abs(float(row["auc"]) - (sen + spe) / 2.) < 1e-12
    assert len(keyed) == 120
    for config in CONFIGS:
        assert all((config, 0, fold) in keyed for fold in range(1, 11))
    assert sum(int(row["physical_gpu"]) == 0 for row in rows) == 60
    assert sum(int(row["physical_gpu"]) == 1 for row in rows) == 60
    return keyed


def mean_sd(rows, metric):
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def paired(rows_by_key, left, right, metric):
    differences = [float(rows_by_key[(left, 0, fold)][metric]) -
                   float(rows_by_key[(right, 0, fold)][metric]) for fold in range(1, 11)]
    return dict(delta_pp=float(np.mean(differences) * 100.),
                wins=sum(value > 0 for value in differences), ties=sum(value == 0 for value in differences),
                losses=sum(value < 0 for value in differences),
                fold_delta_pp=[float(value * 100.) for value in differences])


def report():
    groups, rows, diagnostics = task_groups()
    row_by_key = aggregate(rows)
    assert len(groups) == 12 and len(rows) == len(diagnostics) == 120
    tables = {}
    for config in CONFIGS:
        selected_rows = [row_by_key[(config, 0, fold)] for fold in range(1, 11)]
        tables[config] = {metric: mean_sd(selected_rows, metric) for metric in METRICS}
    comparisons = {}
    for other in ("B0", "B1", "B2", "D0", "D1", "D2", "D3", "D5", "D6", "D7", "D8"):
        comparisons["D4-" + other] = {metric: paired(row_by_key, "D4", other, metric) for metric in METRICS}
    d4diag = [row for row in diagnostics if row["config"] == "D4"]
    deletion_advantage = float(np.mean([float(row["selected_vs_random_deletion_advantage"]) for row in d4diag]))
    required = ("B2", "D1", "D3", "D5", "D6", "D7", "D8")
    failures = []
    for other in required:
        for metric in ("acc", "auc"):
            value = comparisons["D4-" + other][metric]
            if value["delta_pp"] <= .1 or value["wins"] < 7:
                failures.append(dict(comparison="D4-" + other, metric=metric,
                                     delta_pp=value["delta_pp"], fold_wins=value["wins"]))
    if deletion_advantage <= 0:
        failures.append(dict(comparison="selected-deletion minus random-deletion", metric="mean true-class log-probability loss",
                             value=deletion_advantage))
    judgment = dict(go_no_go="GO" if not failures else "NO-GO", failures=failures,
                    deletion_advantage=deletion_advantage,
                    rationale="GO requires D4 to gain >0.1 pp over B2, D1, D3, D5, D6, D7 and D8 in both ACC and hard-AUC with at least 7/10 paired-fold wins, plus greater selected-token deletion impact than an equal-budget random deletion. This is a frozen internal protocol, not independent clinical validation.")
    formal_seconds = float(json_read(MANIFEST).get("formal_seconds", 0.))
    gpu_folds = json_read(MANIFEST).get("gpu_completed_folds", {})
    lines = [
        "# Experiment 32 Results: DCER", "",
        "DCER replaces the full original PEN at the original PLN location. The original GFR, CAM, OutputLayer, graph loss, two Adam optimizers, train/test split protocol, and validation-equals-test checkpoint rule remain intact. B2 is the newly trained internal full-SPromptGL baseline; paper-reported values 91.73% after GFR, 92.08% after RES, and 92.30% for full PEN are historical paper values only.", "",
        "All formal values use seed 0, ten contiguous StratifiedKFold folds per configuration, one-hot hard-label ROC-AUC, and population SD. Only ACC, hard-AUC, ASD sensitivity, and TD specificity are formal metrics. Mechanism measurements are kept separately in `diagnostics.csv`.", "",
        f"Status: **{json_read(MANIFEST)['status']}**; folds {len(rows)}/120; GPU0={gpu_folds.get('0', 0)}, GPU1={gpu_folds.get('1', 0)}; formal wall time {formal_seconds:.2f} seconds.", "",
        "|Config|ACC %|Hard-AUC %|SEN-ASD %|SPE-TD %|ASD FN|TD FP|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        vals = tables[config]
        selected_rows = [row_by_key[(config, 0, fold)] for fold in range(1, 11)]
        false_negatives = sum(int(row["asd_fn"]) for row in selected_rows)
        false_positives = sum(int(row["td_fp"]) for row in selected_rows)
        lines.append("|" + config + "|" + "|".join(
            f"{vals[metric][0] * 100:.4f} +/- {vals[metric][1] * 100:.4f}"
            for metric in METRICS) + f"|{false_negatives}|{false_positives}|")
    lines += ["", "## Paired Comparisons", "", "D4 minus each comparator; percentage-point mean difference and paired fold wins/losses.",
              "", "|Comparison|ACC (pp; W/L)|Hard-AUC (pp; W/L)|SEN-ASD (pp)|SPE-TD (pp)|",
              "|---|---:|---:|---:|---:|"]
    for other in ("B0", "B1", "B2", "D0", "D1", "D2", "D3", "D5", "D6", "D7", "D8"):
        values = comparisons["D4-" + other]
        acc, auc = values["acc"], values["auc"]
        lines.append(f"|D4-{other}|{acc['delta_pp']:+.4f}; {acc['wins']}/{acc['losses']}|"
                     f"{auc['delta_pp']:+.4f}; {auc['wins']}/{auc['losses']}|"
                     f"{values['sen_asd']['delta_pp']:+.4f}|{values['spe_td']['delta_pp']:+.4f}|")
    group_by_name = {group["config"]: group for group in groups}
    lines += ["", "## Trainable Parameters", "", "Counts are for the complete PLN, including DCER32 when present. They make the D0 capacity caveat explicit; D0's frozen reference implementation retains the unused text Query/Key projections.",
              "", "|Config|PLN trainable parameters|", "|---|---:|"]
    for config in CONFIGS:
        lines.append(f"|{config}|{group_by_name[config].get('pln_trainable_parameters', 'unknown')}|")
    lines += ["", "## Evidence Diagnostics", "",
              f"D4 selected-token deletion reduced held-out true-class log probability by {np.mean([float(row['selected_deletion_logp_loss']) for row in d4diag]):.6f} on average; equal-count random-token deletion reduced it by {np.mean([float(row['random_deletion_logp_loss']) for row in d4diag]):.6f}. Their difference was {deletion_advantage:+.6f}. D4 wrong-class-prompt-block counterfactual changed mean output log probability by {np.mean([float(row['wrong_text_output_delta']) for row in d4diag]):.6f}; this is a model-dependence diagnostic, not proof of medical causality.",
              "", "Selected latent tokens are four-way subspaces of the CAM representation, not anatomical ROIs or lesions. NULL probabilities are routing scores, not calibrated clinical likelihoods. The eight inherited GPT-4 sentences mostly describe general brain structure or function and are not validated ASD pathology priors.",
              "", "## Scientific Decision", "", f"**{judgment['go_no_go']}**. {judgment['rationale']}",
              "", "Prior-art review: MMGPL (MedIA 2024), BrainPrompt (MICCAI 2025), and ERASER (ACL 2020) already cover prompt weighting or explanation-completeness ideas. DCER is not supported by sparse attention alone; it must show predictive gains and faithful keep/delete behavior against same-budget controls.",
              "", "Historical protocol limitations remain: validation equals test for checkpoint selection; preprocessing/RFE was run before outer splits; GFR propagates over all 871 participants. Results are internal comparability evidence, not unbiased generalization or clinical evidence.",
              "", "## Audit", "",
              f"Runtime sources are isolated in `.runtime/sources`; protected root SHA-256: `{json.dumps(json_read(MANIFEST)['source_sha256'], ensure_ascii=False, sort_keys=True)}`.",
              f"Code SHA-256: `{json.dumps(json_read(MANIFEST)['code_sha256'], ensure_ascii=False, sort_keys=True)}`.",
              f"Frozen RN50 prompt SHA-256: `{json_read(MANIFEST).get('prompt_sha256')}`.",
              f"Control matrix SHA-256: `{json.dumps(json_read(MANIFEST).get('control_bank_sha256', {}), sort_keys=True)}`.",
              f"Worker count: {WORKERS_PER_GPU}/GPU; GPU fold counts: `{json.dumps(gpu_folds, sort_keys=True)}`.",
              "", "Full fold metrics are in `fold_metrics.csv`; route, null, residual, counterfactual, and deletion diagnostics are in `diagnostics.csv`.", ""]
    temporary = RESULTS / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, RESULTS / "report.md")
    return dict(rows=len(rows), diagnostics=len(diagnostics), tables=tables,
                comparisons=comparisons, judgment=judgment, deletion_advantage=deletion_advantage)


def append_handoff(summary):
    path = ROOT / "handoff.md"
    marker = b"<!-- EXPERIMENT32_DCER_FINAL -->"
    prefix = json_read(MANIFEST)["handoff_prefix"]
    current = path.read_bytes() if path.exists() else b""
    assert len(current) >= prefix["size"]
    if prefix["size"]:
        assert bytes_sha(current[:prefix["size"]]) == prefix["sha256"], \
            "handoff historical prefix changed during Experiment 32; refusing to append"
    if marker in current:
        return
    table = summary["tables"]
    lines = ["", marker.decode(), "## 实验三十二：DCER 疾病对比证据路由", "",
             f"更新时间：{utc()}；状态：COMPLETE；正式折数：{summary['rows']}/120；GPU0=60、GPU1=60。正式墙钟：{json_read(MANIFEST).get('formal_seconds', 0.):.2f}秒。", "",
             "科研问题：原 SPGS 对 ASD/TD 两类 Prompt 的最大概率无法表达反向证据，也没有 NULL 拒绝；RES 缺少证据充分性/必要性约束。DCER 在原 PEN 位置将 8 条冻结 RN50 文本分成 TD/ASD 两组，以差异分数和 NULL 路由竞争，跨 16 个潜在 CAM 子空间硬选至多 4 个，再以共享原 OutputLayer 评估完整、保留和删除视图。16 个 Token 是潜在表征子空间，不是解剖 ROI 或病灶；原 GPT-4 句子多数是一般脑结构/功能描述，不能当成可靠 ASD 病理先验。", "",
             "对照含义：B0 删除 PEN，B1 保留原 RES，B2 从头重训完整原 SPromptGL，D0-D8 分别检验无文本、无符号方向、稠密路由、无充分/必要约束、类别身份混排、随机数值文本、关闭 NULL、随机同预算路由。论文报告的 91.73/92.08/92.30 是独立论文值，不与本地训练结果混记。", "",
             "正式四指标均值±十折 population SD（百分比）见下表：", "",
             "|组|ACC|Hard-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|"]
    for config in CONFIGS:
        lines.append("|" + config + "|" + "|".join(
            f"{table[config][metric][0] * 100:.4f}±{table[config][metric][1] * 100:.4f}"
            for metric in METRICS) + "|")
    lines += ["", f"D4 相对 B0/B1/B2 与 D0-D8 的逐折配对差见 `{EXP}/results/report.md`。证据忠实性与负对照结论：{summary['judgment']['go_no_go']}；D4 选中证据删除相对等预算随机删除的真类 log-prob 下降差为 {summary['deletion_advantage']:+.6f}。稀疏注意力本身不构成创新证据。", "",
              "接入审计：B2 相对未修改 PLN bitwise exact；B0 为 CAM 直接 flatten，B1 为原 RES 文本；D 组完整替换 SPGS+RES。双 Adam、模型、最佳 checkpoint、连续组 RNG、D8 random_step 均写入 `.runtime/` 原子断点。原模型协议与根源码未改，未执行 Git commit/push。", "",
              f"实验目录：`{EXP}`。代码/保护源码、RN50 Prompt 和控制矩阵 SHA，以及恢复与墙钟信息见 `manifest.json`。限制：val=test 选 checkpoint、外层前 RFE/标准化、871 人 transductive GFR 均沿用历史协议，需独立验证后才能讨论投稿或临床泛化。", ""]
    text = "\n".join(lines).encode("utf-8")
    with (ROOT / "handoff.md.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = path.read_bytes() if path.exists() else b""
        assert len(current) >= prefix["size"]
        if prefix["size"]:
            assert bytes_sha(current[:prefix["size"]]) == prefix["sha256"]
        with path.open("ab") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock, fcntl.LOCK_UN)
    manifest = json_read(MANIFEST)
    manifest["handoff_appended"] = True
    manifest["handoff_status"] = "APPENDED_PREFIX_VERIFIED"
    manifest["handoff_sha256"] = sha(path)
    json_write(MANIFEST, manifest)


def finish():
    groups, rows, diagnostics = refresh()
    assert len(groups) == 12 and len(rows) == len(diagnostics) == 120
    assert all(group["status"] == "COMPLETE" and len(group["metrics"]) == 10 for group in groups)
    aggregate(rows)
    manifest = json_read(MANIFEST)
    assert manifest["gpu_completed_folds"] == {"0": 60, "1": 60}
    manifest.update(status="COMPLETE", completed_utc=utc(),
                    result_sha256={path.name: sha(path) for path in RESULTS.glob("*") if path.is_file()})
    json_write(MANIFEST, manifest)
    summary = report()
    append_handoff(summary)
    manifest = json_read(MANIFEST)
    manifest["scientific_conclusion"] = summary["judgment"]["go_no_go"]
    manifest["result_sha256"] = {path.name: sha(path) for path in RESULTS.glob("*") if path.is_file()}
    manifest["formal_wall_seconds"] = float(manifest.get("formal_seconds", 0.))
    json_write(MANIFEST, manifest)
    print("STATUS=COMPLETE 12 configs / 120 folds / GPU0=60 GPU1=60", flush=True)


def run_controller():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    controller = (RUNTIME / "controller.lock").open("a+")
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        prepare()
        manifest = json_read(MANIFEST)
        if not manifest.get("preflight", {}).get("passed"):
            preflight()
        manifest = json_read(MANIFEST)
        manifest["status"] = "RUNNING"
        manifest["workers_per_gpu"] = WORKERS_PER_GPU
        manifest["gpu_load_at_formal_start"] = gpu_snapshot()
        if not manifest.get("formal_started_utc"):
            manifest["formal_started_utc"] = utc()
        json_write(MANIFEST, manifest)
        start = time.monotonic()
        schedule()
        manifest = json_read(MANIFEST)
        manifest["formal_seconds"] = float(manifest.get("formal_seconds", 0.)) + time.monotonic() - start
        json_write(MANIFEST, manifest)
        finish()
    except Exception as error:
        import traceback
        traceback.print_exc()
        if MANIFEST.exists():
            manifest = json_read(MANIFEST)
            if manifest.get("status") != "PARTIAL":
                manifest["status"] = "PARTIAL"
                manifest.setdefault("failures", []).append({"time": utc(), "error": repr(error)})
            json_write(MANIFEST, manifest)
        raise
    finally:
        controller.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "all", "resume", "refresh", "report", "_worker", "_preflight"))
    parser.add_argument("config", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command in ("preflight", "_preflight"):
        preflight()
    elif args.command == "refresh":
        refresh()
    elif args.command == "report":
        report_path = RESULTS / "report.md"
        print(report_path.read_text(encoding="utf-8") if report_path.exists() else "report unavailable")
    elif args.command == "_worker":
        worker(args.config)
    else:
        run_controller()


if __name__ == "__main__":
    main()
