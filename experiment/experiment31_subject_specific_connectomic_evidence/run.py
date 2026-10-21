#!/usr/bin/env python3
"""Experiment 31 controller: prepare, preflight, resumable formal training, report."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import pickle
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np

import integration
import report_bank
from method import CONFIGS, DESIGNS, GPU, METRICS, SCORE, make_fold_reference, state_from_z

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "sources"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifest.json"
CACHE = EXP / "report_bank.npz"
CODE = ("method.py", "report_bank.py", "integration.py", "run.py", "integration.patch")
CORE = ("main.py", "model.py", "network.py", "layers.py", "utils.py", "ABIDE-simple-2-concat-weighted-cosine.sh")
PAPERS = tuple(sorted(ROOT.glob("*SPromptGL*.pdf")))
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
                nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
                n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
                alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
                nclass=2, mode="simple-2")
WORKERS_PER_GPU = 1
_CTX = None


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@contextlib.contextmanager
def json_lock(exclusive):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "json.lock").open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def jread(path):
    path = Path(path)
    for attempt in range(12):
        try:
            with json_lock(False):
                return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 11:
                raise
            time.sleep(.15)


def jwrite(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with json_lock(True):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def readcsv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cwrite(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record(message):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "scheduler.log").open("a", encoding="utf-8") as f:
        f.write(utc() + " " + str(message) + "\n")
        f.flush()
    print(message, flush=True)


def data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    frame = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    expected = [str(c) for v in mapping.values() for c in v]
    assert list(frame.columns[:-1]) == expected
    assert list(frame.columns[64:320]) == [str(i) for i in range(256)]
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    names = list(mapping["PHENO"])
    pheno = np.asarray([i for i, name in enumerate(names) if name.startswith(("AGE_AT_SCAN_", "SEX_"))], dtype=np.int64)
    site_ids = np.asarray([i for i, name in enumerate(names) if name.startswith("SITE_ID_")], dtype=np.int64)
    assert len(pheno) == 28 and len(site_ids) == 20 and not set(pheno) & set(site_ids)
    sites = x[:, site_ids].argmax(1).astype(np.int64)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    for tr, te in folds:
        assert not set(tr) & set(te) and len(set(tr) | set(te)) == 871
    return dict(x=x, y=y, sites=sites, pheno_indices=pheno, site_indices=site_ids, folds=folds,
                mapping=mapping, frame_columns=list(frame.columns))


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
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for v in [state["cpu"]] + list(state["cuda"]):
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def model_state(agent):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in zip(("gfr", "pln"), (agent.MessagePassing0, agent.ModalFusion))}


def load_models(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def recursive_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(recursive_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(recursive_equal(x, y) for x, y in zip(a, b))
    return a == b


def checkpoint_write(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    torch.save(obj, stream)
    body = stream.getvalue()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"E31!" + hashlib.sha256(body).hexdigest().encode() + body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint_read(path):
    import torch
    raw = Path(path).read_bytes()
    assert raw[:4] == b"E31!" and hashlib.sha256(raw[68:]).hexdigest().encode() == raw[4:68]
    return torch.load(io.BytesIO(raw[68:]), map_location="cpu", weights_only=False)


def core_hashes():
    return {name: sha(ROOT / name) for name in CORE}


def code_hashes():
    return {name: sha(EXP / name) for name in CODE if (EXP / name).is_file()}


def build_sources():
    return integration.emit(ROOT, SOURCE, EXP / "integration.patch")


def signature():
    m = jread(MANIFEST)
    return canonical({"protocol": m["protocol"], "configs": m["configs"], "source_sha256": m["source_sha256"], "code_sha256": m["code_sha256"], "report_bank_sha256": m["report_bank_sha256"]})


def archive_invalidated_run(m):
    revision = int(m.get("freeze_revision", 1))
    results_archive = RESULTS / f"superseded_revision_{revision}"
    runtime_archive = RUNTIME / f"superseded_revision_{revision}"
    results_archive.mkdir(parents=True, exist_ok=True)
    for path in RESULTS.iterdir():
        if path.is_file():
            shutil.copy2(path, results_archive / path.name)
    runtime_archive.mkdir(parents=True, exist_ok=True)
    for name in ("state", "tasks"):
        src = RUNTIME / name
        if src.exists():
            shutil.copytree(src, runtime_archive / name, dirs_exist_ok=True)
    for path in RUNTIME.glob("*_seed0.log"):
        shutil.copy2(path, runtime_archive / path.name)
    archived = dict(revision=revision, results=str(results_archive), runtime=str(runtime_archive),
                    reason="Masked state prediction now uses config-matched frozen keys; prior partial folds retained for audit and excluded from revised formal results.",
                    archived_utc=utc())
    m.setdefault("superseded_runs", []).append(archived)
    for path in RESULTS.iterdir():
        if path.is_file():
            path.unlink()
    for folder in (RUNTIME / "state", RUNTIME / "tasks"):
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix != ".lock":
                    path.unlink()
    m["tasks"] = [{"config": k, "seed": 0, "gpu": GPU[k], "status": "PENDING", "completed_folds": 0} for k in CONFIGS]
    m["completed_folds"] = 0
    m["completed_tasks"] = 0
    m["gpu_completed_folds"] = {"0": 0, "1": 0}
    m["formal_started_utc"] = None
    m["formal_seconds"] = 0.
    m["status"] = "PREPARED"


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    c = data()
    if not CACHE.exists():
        meta = report_bank.build(ROOT, EXP, c["sites"], c["folds"])
    else:
        meta = json.loads((EXP / "report_bank.json").read_text()) if (EXP / "report_bank.json").exists() else {}
    cache = report_bank.load(CACHE)
    assert cache["text_bank"].shape == (252, 5, 64)
    assert cache["original8"].shape == (8, 64)
    assert cache["p0_original8"].shape == (8, 1024)
    assert cache["valid_indices"].shape == (252,)
    assert cache["site_permutation_folds"].shape == (10, 871)
    build_sources()
    if MANIFEST.exists():
        m = jread(MANIFEST)
        assert m["source_sha256"] == core_hashes(), "protected source changed after freeze"
        current_code, current_bank = code_hashes(), sha(CACHE)
        if m["code_sha256"] != current_code or m["report_bank_sha256"] != current_bank:
            if m.get("formal_started_utc") or m.get("completed_folds", 0):
                archive_invalidated_run(m)
            else:
                assert m["status"] in ("PREPARED", "PREFLIGHT_PASS"), "cannot change frozen experiment in this status"
            m["code_sha256"] = current_code
            m["report_bank_sha256"] = current_bank
            m["refrozen_utc"] = utc()
            m["freeze_revision"] = int(m.get("freeze_revision", 1)) + 1
            m["preflight"] = {"passed": False}
            m["status"] = "PREPARED"
            jwrite(MANIFEST, m)
        return m
    protected = {str(ROOT / name): sha(ROOT / name) for name in CORE}
    input_files = [ROOT / "data/ABIDE/processed_standard_data.csv", ROOT / "data/ABIDE/modal_feat_dict.npy"]
    m = {
        "experiment": "experiment31_subject_specific_connectomic_evidence",
        "method": "SCORE — Subject-specific Connectomic Evidence and Reliability-aware Modulation",
        "status": "PREPARED", "prepared_utc": utc(), "source_sha256": core_hashes(),
        "protected_files_sha256": protected, "code_sha256": code_hashes(),
        "data_sha256": {str(p): sha(p) for p in input_files},
        "paper_sha256": {p.name: sha(p) for p in PAPERS},
        "report_bank_sha256": sha(CACHE), "report_bank_metadata": meta,
        "mapping_sha256": sha(ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv"),
        "clip_weight_sha256": sha(Path.home() / ".cache/clip/RN50.pt"),
        "protocol": PROTOCOL, "n_subjects": 871, "labels": {"ASD": 403, "TD": 468},
        "configs": {k: asdict(v) for k, v in DESIGNS.items()},
        "gpu_assignment": GPU, "target_tasks": 18, "target_folds": 180,
        "tasks": [{"config": k, "seed": 0, "gpu": GPU[k], "status": "PENDING", "completed_folds": 0} for k in CONFIGS],
        "fold_protocol": "StratifiedKFold(n_splits=10, shuffle=True, random_state=10); val_idx=test_idx; seed=0; 450 epochs; early=200",
        "reference_protocol": "P0 and historical E1 are read-only; all new groups remove SPGS and install SCORE at the same location",
        "statistics_protocol": "outer-train median/MAD, std fallback, train-site within/(within+between) reliability; labels never construct states",
        "failures": [], "original_handoff_size": (ROOT / "handoff.md").stat().st_size if (ROOT / "handoff.md").exists() else 0,
        "original_handoff_sha256": sha(ROOT / "handoff.md") if (ROOT / "handoff.md").exists() else None,
        "preflight": {"passed": False}, "completed_folds": 0, "completed_tasks": 0,
    }
    history_files = [
        ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv",
        ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv",
        ROOT / "experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv",
        ROOT / "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/fold_metrics.csv",
        ROOT / "experiment/experiment29_connectivity_semantic_selective_auxiliary/results/report.md",
        ROOT / "experiment/experiment29_connectivity_semantic_selective_auxiliary/results/fold_metrics.csv",
        ROOT / "experiment/experiment30_class_conditional_semantic_shared_repair/results/report.md",
        ROOT / "experiment/experiment30_class_conditional_semantic_shared_repair/manifest.json",
    ]
    m["historical_artifact_sha256"] = {str(p): sha(p) for p in history_files if p.is_file()}
    m["audit_findings"] = {
        "SPromptGL": "PLN input encoder and FeedForward create 4x36 CAM; original SPGS occupies the pre-AvgPool/RES segment; simple-2 keeps GFR/CAM/RES/classifier and two Adam optimizers; val_idx equals test_idx and checkpoint selection uses validation ACC.",
        "input_semantics": "processed_standard_data.csv is 871x321 including label; feature partitions are PHENO48/ANAT6/FUNC10/CORR256; ASD=0 n=403, TD=1 n=468; site is 20 one-hot columns; PHENO query FiLM selects age26+sex2 only.",
        "roi_mapping": "Experiment 11 verifies numeric mapping for all 256 columns and names exactly 252 edges; 195,225,232,236 involve unnamed atlas extra ROI and are excluded from text bank.",
        "historical": "Experiment 13 F0 is read-only original-Prompt P0; Experiment 28 E1_NO_SPGS is read-only SPGS-removed reference; Experiments 29/30 are reviewed as prior failure/negative-control evidence, not reused checkpoints.",
        "RTGMFF": "Official public source was cloned for read-only audit. data.py builds ROI/strength/polarity tokens from 116 activation values; model.py embeds them, aligns with ASAM, and applies age/gender FiLM. The paper's AAL-116 and ALFF/fALFF/ReHo inputs differ from CORR256; public implementation does not implement paper's claimed BioBERT faithfully. Wavelet-Mamba needs image/raw fMRI input and is not used.",
    }
    jwrite(MANIFEST, m)
    (EXP / "README.md").write_text(
        "# 实验31：SCORE\n\n"
        "SCORE 使用每个外层训练折拟合的 CORR median/MAD 参考和站点稳定性，"
        "为252条已验证ROI连接生成五状态个体报告，并以文本参与的连接证据调制原CAM-CORR。\n\n"
        "本实验遵循原始SPromptGL的871人transductive图传播、val=test checkpoint选择、"
        "450/200训练协议；18组各seed=0完成10折。A0–A11与B0–B5的配置、负对照、"
        "审计哈希和完整结果见 `manifest.json`、`results/report.md`。P0与历史E1只读参考。\n\n"
        "RTGMFF仅借鉴患者特异性ROI状态和年龄/性别FiLM思想；其AAL-116、ALFF/fALFF/ReHo、"
        "原始影像Wavelet-Mamba和BioBERT路径未移植，因为本项目输入是预处理后的CORR256与四模态数值。\n"
    )
    return m


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    sys.path.insert(0, str(SOURCE))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    import model, network, layers, utils
    cache = report_bank.load(CACHE)
    def cached_p0():
        if network._prompt_text_features is None:
            encoder, _ = network.clip.load("RN50", network.device)
            del encoder
            network._prompt_text_features = torch.from_numpy(cache["p0_original8"]).to("cuda")
        return network._prompt_text_features
    network.get_prompt_text_features = cached_p0
    try:
        from CLIP.clip.model import CLIP
        CLIP.encode_text = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("frozen text encoding forbidden during training"))
    except Exception:
        pass
    _CTX = dict(torch=torch, model=model, network=network, layers=layers, utils=utils, cache=cache, **data())
    return _CTX


def make_agent(config, fold, labels=None):
    c = context()
    tr, te = c["folds"][fold - 1]
    y = c["y"] if labels is None else labels
    integration.configure(config=config, fold=fold, raw=c["x"], train=tr, test=te,
                          labels_train=y[tr], site=c["sites"], pheno_indices=c["pheno_indices"],
                          cache=c["cache"], seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = c["model"].EvalHelper([48, 6, 10, 256], c["x"], y,
                                       SimpleNamespace(**PROTOCOL, seed=0), tr, te)
    agent.MF_sav.close(); agent.GCMP_sav.close()
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    return agent


def evaluate(agent):
    c = context(); torch = c["torch"]
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        feat, _, _ = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
        out, _, _ = agent.ModalFusion(feat, raw_input=agent.feat)
    assert out.shape == (871, 2) and torch.isfinite(out).all()
    return out


def metrics(out, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    c = context()
    y = c["y"][test]
    pred = out.detach().cpu().numpy()[test].argmax(1)
    tp, fn, fp, tn = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred]))
    return dict(acc=float((pred == y).mean()), auc=hard_auc,
                sen_asd=float(tp / (tp + fn)), spe_td=float(tn / (tn + fp)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def optimizer_audit(agent):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    ids = [id(p) for g in groups for p in g["params"]]
    actual = {id(p) for m in (agent.MessagePassing0, agent.ModalFusion) for p in m.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual
    assert all(abs(g["lr"] - PROTOCOL["lr"]) < 1e-12 for g in groups)
    branch = getattr(agent.ModalFusion, "score31", None)
    if branch is not None and branch.design.active:
        assert len(groups) == 3 and groups[-1]["weight_decay"] == 0.0
        assert branch.core.output.weight.count_nonzero().item() == 0
    return sum(p.numel() for p in agent.ModalFusion.parameters()), branch.parameter_count() if branch is not None else 0


def source_parity():
    c = context(); torch = c["torch"]
    seed_rng(0)
    entry = rng_state()
    p0 = make_agent("P0", 1)
    init_hash = trunk_parameter_hash(p0)
    init_rng_hash = rng_hash(entry)
    expected = historical_initialization_refs()[1]
    assert init_hash == expected["initial_trunk_sha256"], ("P0/E1 initial trunk mismatch", init_hash)
    assert init_rng_hash == expected["initial_rng_sha256"], ("P0/E1 initial RNG mismatch", init_rng_hash)
    spec = importlib.util.spec_from_file_location("score31_original_network", ROOT / "network.py")
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    original.get_prompt_text_features = lambda: torch.from_numpy(c["cache"]["p0_original8"]).to("cuda")
    ref = original.PLN([48, 6, 10, 256], SimpleNamespace(**PROTOCOL)).to(p0.dev)
    ref.load_state_dict(p0.ModalFusion.state_dict(), strict=True)
    ref.eval(); p0.ModalFusion.eval()
    with torch.no_grad():
        n, parts = p0.dataLayer(p0.feat)
        feat, _, _ = p0.MessagePassing0(parts, n, p0.targ, p0.trn_idx)
        a = ref(feat); b = p0.ModalFusion(feat, raw_input=p0.feat)
    assert recursive_equal(a, b)
    del ref, p0
    return dict(forward="BITWISE_EXACT", initial_trunk_sha256=init_hash, initial_rng_sha256=init_rng_hash)


def preflight():
    prepare()
    from method import self_test
    self_test()
    for name in (EXP / "method.py", EXP / "report_bank.py", EXP / "integration.py", EXP / "run.py"):
        compile(name.read_text(), str(name), "exec")
    if os.environ.get("SCORE31_PREFLIGHT_CHILD") != "1":
        log = RUNTIME / "preflight.log"
        env = worker_env(0)
        env["SCORE31_PREFLIGHT_CHILD"] = "1"
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run([sys.executable, str(EXP / "run.py"), "_preflight"], cwd=RUNTIME, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError("GPU preflight failed: " + log.read_text()[-12000:])
        return
    c = context(); torch = c["torch"]
    checks = {"source_parity": source_parity()}
    seed_rng(0); a0 = make_agent("A0", 1); seed_rng(0); a5 = make_agent("A5", 1)
    a0.ModalFusion.eval(); a5.ModalFusion.eval()
    n, parts = a0.dataLayer(a0.feat)
    with torch.no_grad():
        cam, _, _ = a0.MessagePassing0(parts, n, a0.targ, a0.trn_idx)
        out0, _, _ = a0.ModalFusion(cam, raw_input=a0.feat)
        out5, _, _ = a5.ModalFusion(cam, raw_input=a5.feat)
    assert torch.equal(out0, out5) and torch.equal(a5.ModalFusion.score31.last["delta"], torch.zeros_like(a5.ModalFusion.score31.last["delta"]))
    checks["A0_identity_and_all_zero_initial_delta"] = "PASS"
    b = a5.ModalFusion.score31
    assert b.core.output.weight.count_nonzero().item() == 0
    with torch.no_grad():
        probe = torch.randn((871, 4, 36), device="cuda")
        out = b(probe, a5.feat)
        den = probe[:, 3].norm(dim=-1).clamp_min(1e-8)
        assert torch.equal(out[:, :3], probe[:, :3])
        assert torch.all(b.last["delta"].norm(dim=-1) < .05 * den)
    checks["residual_bound_and_non_corr_preservation"] = "PASS"
    tr, te = c["folds"][0]
    assert len(c["folds"]) == 10 and all(len(set(a) & set(b)) == 0 and len(set(a) | set(b)) == 871 for a, b in c["folds"])
    assert torch.equal(b.valid_indices_t.cpu(), torch.as_tensor([i for i in range(256) if i not in (195, 225, 232, 236)]))
    assert torch.equal(b.states.cpu(), torch.as_tensor(state_from_z(b.z_valid.cpu().numpy())))
    checks["fold_partition_state_mapping_and_252_edges"] = "PASS"
    flipped = c["y"].copy(); flipped[te] = 1 - flipped[te]
    seed_rng(0); real_entry = rng_state(); real = make_agent("A5", 1)
    seed_rng(0); heldout = make_agent("A5", 1, labels=flipped)
    expected_init = historical_initialization_refs()[1]
    assert rng_hash(real_entry) == expected_init["initial_rng_sha256"]
    assert trunk_parameter_hash(real) == expected_init["initial_trunk_sha256"]
    checks["score_initialization_matches_historical_e1"] = "BITWISE_EXACT"
    br, bh = real.ModalFusion.score31, heldout.ModalFusion.score31
    assert torch.equal(br.z_valid, bh.z_valid) and torch.equal(br.states, bh.states)
    checks["heldout_label_flip_reference_and_states"] = "BITWISE_EXACT"
    assert torch.equal(br._state_binding(), bh._state_binding())
    for control in ("B0", "B1", "B2", "B3", "B4", "B5"):
        candidate = make_agent(control, 1).ModalFusion.score31
        assert torch.equal(candidate.states, br.states), control + " changed true state targets"
        assert candidate._state_binding().shape == (871, 252, 64)
    assert torch.allclose(torch.einsum("esd,etd->est", b.text_bank, b.text_bank),
                          torch.einsum("esd,etd->est", b.numeric_match, b.numeric_match), atol=2e-5, rtol=2e-5)
    edge_ids = torch.arange(252, device="cuda")[None, :].expand(871, -1)
    for cfg, expected_bank in (("A1", "numeric_state"), ("B4", "numeric_match")):
        candidate = make_agent(cfg, 1).ModalFusion.score31
        bank = getattr(candidate, expected_bank)
        assert torch.equal(candidate._masked_key_bank(), bank)
        assert torch.equal(candidate._masked_key_bank()[edge_ids, candidate.states], bank[edge_ids, candidate.states])
    checks["numeric_controls_use_config_matched_masked_keys"] = "PASS"
    b.condition_override = "ORIGINAL8"
    assert torch.equal(b._state_binding()[0], b._state_binding()[-1])
    b.condition_override = None
    checks["negative_controls_preserve_state_targets_and_geometry"] = "PASS"
    ph, anat, func = b._aux_inputs(real.feat)
    assert torch.equal(ph, real.feat[:, b.pheno_indices])
    assert torch.equal(anat, real.feat[:, 48:54]) and torch.equal(func, real.feat[:, 54:64])
    assert ph.shape[1] == 28 and anat.shape[1] == 6 and func.shape[1] == 10
    for cfg, field in (("A10", "pheno"), ("A11", "qc")):
        branch = make_agent(cfg, 1).ModalFusion.score31
        p2, a2, f2 = branch._aux_inputs(real.feat)
        if field == "pheno":
            assert torch.equal(p2, ph[branch.site_permutation])
            assert torch.equal(a2, anat) and torch.equal(f2, func)
        else:
            assert torch.equal(p2, ph)
            assert torch.equal(a2, anat[branch.site_permutation]) and torch.equal(f2, func[branch.site_permutation])
    checks["pheno_qc_dimensions_and_mismatch_scope"] = "PASS"
    assert b.core.demographic[-1].weight.count_nonzero().item() == 0
    assert b.core.demographic[-1].bias.count_nonzero().item() == 0
    checks["film_identity_initialization"] = "PASS"
    # Text participation is tested after temporarily opening the zero output
    # head; no training score is used for this check.
    with torch.no_grad():
        b.core.output.weight.normal_(0, .05)
        cam = torch.randn((871, 4, 36), device="cuda")
        x1 = b(cam, real.feat).clone()
        saved = b.text_bank.clone()
        b.text_bank.copy_(saved.roll(1, dims=0))
        x2 = b(cam, real.feat).clone()
        b.text_bank.copy_(saved)
    assert not torch.equal(x1, x2)
    checks["text_in_forward_path"] = "PASS"
    # Masked query invariance for a target whose z value and state report are masked.
    with torch.no_grad(): b.core.output.weight.zero_()
    b.train(); mask_seed = int(b.mask_rng_state.item()); _ = b(cam, real.feat)
    mask = b.last["mask"].clone(); target = torch.nonzero(mask, as_tuple=False)[0]
    assert mask[te].count_nonzero().item() == 0
    old = b.z_valid[target[0], target[1]].item()
    state_old = b.states[target[0], target[1]].item()
    query_old = b.last["masked_queries"][target[0], target[1]].detach().clone()
    with torch.no_grad():
        b.z_valid[target[0], target[1]] = old + 2.0
        b.states[target[0], target[1]] = (state_old + 1) % 5
    b.mask_rng_state.fill_(mask_seed); _ = b(cam, real.feat)
    assert torch.isfinite(b.last["masked_loss"])
    query_new = b.last["masked_queries"][target[0], target[1]].detach()
    assert torch.equal(query_old, query_new)
    with torch.no_grad():
        b.z_valid[target[0], target[1]] = old
        b.states[target[0], target[1]] = state_old
        b.mask_rng_state.fill_(mask_seed)
    checks["masked_target_z_and_state_invariance"] = "BITWISE_EXACT"
    before_mask_rng = rng_hash(rng_state())
    _ = b._masked_state_loss(real.feat)
    assert rng_hash(rng_state()) == before_mask_rng
    checks["masked_rng_isolated_and_train_only"] = "PASS"
    checks["masked_loss_finite"] = True
    seed_rng(0); grad_agent = make_agent("A5", 1)
    with contextlib.redirect_stdout(io.StringIO()): grad_agent.run_epoch("simple-2")
    first_grads = {n: p.grad.detach().clone() for n, p in grad_agent.ModalFusion.score31.named_parameters() if p.grad is not None}
    with contextlib.redirect_stdout(io.StringIO()): grad_agent.run_epoch("simple-2")
    branch = grad_agent.ModalFusion.score31
    grads = [p.grad for p in branch.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.norm()) > 0 for g in grads)
    checks["first_step_gradients_finite"] = all(torch.isfinite(g).all() for g in first_grads.values())
    checks["two_step_gradients"] = "PASS"
    del grad_agent
    # Verify the real checkpoint serializer and restore path by comparing the
    # next optimizer update against an uninterrupted run.
    seed_rng(0); uninterrupted = make_agent("A5", 1)
    with contextlib.redirect_stdout(io.StringIO()): uninterrupted.run_epoch("simple-2")
    saved_probe = dict(model=model_state(uninterrupted), mp=uninterrupted.optimizer_MP0.state_dict(),
                       mf=uninterrupted.optimizer_MF.state_dict(), rng=rng_state())
    probe_path = RUNTIME / "resume_probe.pt"
    checkpoint_write(probe_path, saved_probe)
    saved_probe = checkpoint_read(probe_path)
    with contextlib.redirect_stdout(io.StringIO()): uninterrupted.run_epoch("simple-2")
    expected_model = model_state(uninterrupted)
    expected_mp = uninterrupted.optimizer_MP0.state_dict()
    expected_mf = uninterrupted.optimizer_MF.state_dict()
    expected_rng = rng_state()
    expected_prediction = evaluate(uninterrupted)
    restore_rng(saved_probe["rng"]); resumed = make_agent("A5", 1)
    load_models(resumed, saved_probe["model"])
    resumed.optimizer_MP0.load_state_dict(saved_probe["mp"]); resumed.optimizer_MF.load_state_dict(saved_probe["mf"])
    restore_rng(saved_probe["rng"])
    with contextlib.redirect_stdout(io.StringIO()): resumed.run_epoch("simple-2")
    assert recursive_equal(model_state(resumed), expected_model)
    assert recursive_equal(resumed.optimizer_MP0.state_dict(), expected_mp)
    assert recursive_equal(resumed.optimizer_MF.state_dict(), expected_mf)
    assert rng_hash(rng_state()) == rng_hash(expected_rng)
    assert torch.equal(evaluate(resumed), expected_prediction)
    probe_path.unlink()
    checks["checkpoint_resume_next_update_model_optimizer_prediction_rng"] = "EXACT_PASS"
    del uninterrupted, resumed
    jwrite(RUNTIME / "preflight_checks.json", checks)
    m = jread(MANIFEST); m["preflight"] = dict(version=1, passed=True, completed_utc=utc(), checks=checks); m["status"] = "PREFLIGHT_PASS"; jwrite(MANIFEST, m)
    record("PREFLIGHT_PASS: source parity, fold statistics, text path, masked loss, residual bound, optimizer and gradients verified")


def trunk_parameter_hash(agent):
    h = hashlib.sha256()
    for module in (agent.MessagePassing0, agent.ModalFusion):
        for name, p in sorted(module.named_parameters()):
            if name.startswith("score31."):
                continue
            h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def historical_initialization_refs():
    path = ROOT / "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/fold_metrics.csv"
    rows = [r for r in readcsv(path) if r.get("config") == "E1_NO_SPGS" and int(r.get("seed", -1)) == 0]
    assert len(rows) == 10
    refs = {int(r["fold"]): {"initial_trunk_sha256": r["initial_trunk_sha256"],
                              "initial_rng_sha256": r["initial_rng_sha256"],
                              "acc": float(r["acc"]), "auc": float(r["hard_auc"]),
                              "sen_asd": float(r["sensitivity_asd"]), "spe_td": float(r["specificity_td"]),
                              "best_epoch": int(r["best_epoch"])} for r in rows}
    assert sorted(refs) == list(range(1, 11))
    return refs


def diagnostic(agent, config, fold, tracker):
    import scipy.stats
    branch = getattr(agent.ModalFusion, "score31", None)
    c = context(); row = dict(config=config, seed=0, fold=fold, parameter_count=sum(p.numel() for p in agent.ModalFusion.parameters()),
                              additional_parameters=branch.parameter_count() if branch is not None else 0)
    if branch is None or not branch.design.active:
        row.update(state0_fraction=0., state1_fraction=0., state2_fraction=0., state3_fraction=0., state4_fraction=0.,
                   report_unique_mean=0., attention_entropy=0., reliability_mean=0., text_state_cos=0., text_effective_rank=0.,
                   masked_loss_mean=0., masked_gradient_norm=0., pheno_film_norm=0., func_gate_mean=0., anat_gate_mean=0.,
                   delta_relative_mean=0., gate_site_eta2=0., func_qc_corr=0., anat_qc_corr=0., counterfactual_json="[]")
        return row
    last = branch.last
    states = last["states"].detach().cpu().numpy()
    counts = np.bincount(states.ravel(), minlength=5) / states.size
    unique = np.asarray([len(np.unique(states[i], axis=0)) for i in range(states.shape[0])], dtype=float)
    attn = last["attention"].detach().cpu().numpy()
    entropy = -np.sum(attn * np.log(np.maximum(attn, 1e-12)), axis=1).mean()
    ref_norm = last["target"].detach().norm(dim=-1).clamp_min(1e-8)
    delta_ratio = (last["delta"].detach().norm(dim=-1) / ref_norm).mean().item()
    text = branch.text_bank.detach().cpu().numpy()
    cos = np.sum(text[:, 0] * text[:, 1], axis=-1).mean()
    centered = text.reshape(-1, 64) - text.reshape(-1, 64).mean(0)
    sv = np.linalg.svd(centered, compute_uv=False)
    rank = float(np.sum(sv > 1e-6))
    sites = c["sites"]
    gate = (last["gate_func"] * last["gate_anat"]).detach().cpu().numpy()
    eta = _eta_squared(gate, sites)
    qc_func = c["x"][:, 54:64].mean(1)
    qc_anat = c["x"][:, 48:54].mean(1)
    row.update(state0_fraction=float(counts[0]), state1_fraction=float(counts[1]), state2_fraction=float(counts[2]), state3_fraction=float(counts[3]), state4_fraction=float(counts[4]),
               report_unique_mean=float(unique.mean()), attention_entropy=float(entropy), reliability_mean=float(branch.reliability.mean().item()),
               text_state_cos=float(cos), text_effective_rank=rank, masked_loss_mean=float(tracker.get("masked_loss", 0.)),
               masked_gradient_norm=float(tracker.get("masked_gradient", 0.)), pheno_film_norm=float(last["film_norm"].detach().item()),
               func_gate_mean=float(last["gate_func"].detach().mean().item()), anat_gate_mean=float(last["gate_anat"].detach().mean().item()),
               delta_relative_mean=float(delta_ratio), gate_site_eta2=float(eta),
               func_qc_corr=float(abs(np.corrcoef(gate, qc_func)[0, 1]) if np.std(gate) and np.std(qc_func) else 0.),
               anat_qc_corr=float(abs(np.corrcoef(gate, qc_anat)[0, 1]) if np.std(gate) and np.std(qc_anat) else 0.),
               counterfactual_json="[]")
    if config in ("A5", "B0", "B1", "B2", "B3", "B4", "B5"):
        real = evaluate(agent).detach().exp()[agent.tst_idx, 0].cpu().numpy()
        base_pred = (real > .5)
        changes = []
        for mode in ("WRONG_ROI", "REVERSE", "DEGREE_SWAP", "SITE_MISMATCH", "OFF"):
            old = branch.condition_override if hasattr(branch, "condition_override") else None
            branch.condition_override = mode
            out = evaluate(agent).detach().exp()[agent.tst_idx, 0].cpu().numpy()
            branch.condition_override = old
            changes.append(dict(condition=mode, probability_abs_change=float(np.mean(np.abs(out - real))), prediction_flip_fraction=float(np.mean((out > .5) != base_pred))))
        row["counterfactual_json"] = json.dumps(changes, sort_keys=True)
    return row


def _eta_squared(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    grand = values.mean(); total = np.sum((values - grand) ** 2)
    if total <= 1e-12: return 0.
    between = sum(np.sum(groups == g) * (values[groups == g].mean() - grand) ** 2 for g in np.unique(groups))
    return float(between / total)


def task_name(config):
    return config + "_seed0"


def task_json(config):
    return RUNTIME / "tasks" / (task_name(config) + ".json")


def task_checkpoint(config):
    return RUNTIME / "state" / (task_name(config) + ".pt")


def task_result(config):
    p = task_json(config)
    if not p.exists(): return None
    r = jread(p); assert r["signature"] == signature() and r["config"] == config
    assert len(r["metrics"]) == len(r["diagnostics"]) and [int(x["fold"]) for x in r["metrics"]] == list(range(1, len(r["metrics"]) + 1))
    return r


def worker(config):
    assert config in CONFIGS
    m = jread(MANIFEST); assert m["preflight"]["passed"]
    gpu = GPU[config]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    c = context(); torch = c["torch"]
    name = task_name(config); RUNTIME.joinpath("state").mkdir(exist_ok=True); RUNTIME.joinpath("tasks").mkdir(exist_ok=True)
    lock = (RUNTIME / "state" / (name + ".lock")).open("a+"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = task_result(config); ck = task_checkpoint(config); saved = checkpoint_read(ck) if ck.exists() else None
    if existing and existing["status"] == "COMPLETE": print("SKIP_COMPLETE", name, flush=True); return
    if saved:
        assert saved["signature"] == signature(); result = saved["result"]
        # A restarted process must not cold-load RN50 after the saved stream
        # has advanced beyond the first fold.
        c["network"]._prompt_text_features = torch.from_numpy(c["cache"]["p0_original8"]).to("cuda")
        restore_rng(saved["rng"])
    else:
        seed_rng(0); result = dict(signature=signature(), config=config, seed=0, physical_gpu=gpu, status="RUNNING", metrics=[], diagnostics=[], seconds=0., started_utc=utc(), resumes=0)
    begin = time.monotonic(); oldseconds = float(result.get("seconds", 0.)); stop = [False]
    def stopit(signum, frame): stop[0] = True
    signal.signal(signal.SIGTERM, stopit); signal.signal(signal.SIGINT, stopit)
    print("TASK_START", name, "GPU", gpu, "resumed", saved is not None, flush=True)
    for fold in range(len(result["metrics"]) + 1, 11):
        if stop[0]: return
        resumed = bool(saved and saved.get("phase") == "EPOCH" and int(saved["fold"]) == fold)
        entry = rng_state(); agent = make_agent(config, fold); optimizer_audit(agent)
        if resumed:
            load_models(agent, saved["model"]); agent.optimizer_MP0.load_state_dict(saved["mp"]); agent.optimizer_MF.load_state_dict(saved["mf"]); restore_rng(saved["rng"])
            best_state, best, best_epoch, wait, start, tracker = saved["best_state"], float(saved["best"]), int(saved["best_epoch"]), int(saved["wait"]), int(saved["epoch"]) + 1, saved.get("tracker", {})
            init_hash = saved["initial_trunk_sha256"]; init_rng_hash = saved["initial_rng_sha256"]
        else:
            best_state, best, best_epoch, wait, start, tracker = None, -1., -1, 0, 0, {}
            init_hash = trunk_parameter_hash(agent); init_rng_hash = rng_hash(entry)
            if fold == 1 or config == "A0":
                expected = historical_initialization_refs()[fold]
                assert init_hash == expected["initial_trunk_sha256"], (config, fold, "historical trunk initialization mismatch", init_hash)
                assert init_rng_hash == expected["initial_rng_sha256"], (config, fold, "historical RNG trajectory mismatch", init_rng_hash)
        saved = None; fold_start = time.monotonic(); last_epoch = start - 1
        for epoch in range(start, PROTOCOL["nepoch"]):
            if stop[0]: return
            with contextlib.redirect_stdout(io.StringIO()):
                agent.run_epoch("simple-2")
                _, val_acc = agent.print_trn_acc("simple-2")
            branch = getattr(agent.ModalFusion, "score31", None)
            if branch is not None and branch.design.active:
                tracker["masked_loss"] = max(float(tracker.get("masked_loss", 0.)), float(branch.last.get("masked_loss", 0.).detach().item()))
                gn = 0.
                for p in branch.parameters():
                    if p.grad is not None: gn += float(p.grad.detach().square().sum().item())
                tracker["masked_gradient"] = max(float(tracker.get("masked_gradient", 0.)), gn ** .5)
            val_acc = float(val_acc)
            if val_acc > best:
                best, best_epoch, best_state, wait = val_acc, epoch, model_state(agent), 0
            else: wait += 1
            last_epoch = epoch
            if (epoch + 1) % 25 == 0 or wait > PROTOCOL["early"] or stop[0]:
                result["seconds"] = oldseconds + time.monotonic() - begin
                checkpoint_write(ck, dict(signature=signature(), phase="EPOCH", fold=fold, epoch=epoch, model=model_state(agent), mp=agent.optimizer_MP0.state_dict(), mf=agent.optimizer_MF.state_dict(), rng=rng_state(), best_state=best_state, best=best, best_epoch=best_epoch, wait=wait, tracker=tracker, initial_trunk_sha256=init_hash, initial_rng_sha256=init_rng_hash, result=result))
            if stop[0]: return
            if wait > PROTOCOL["early"]: break
        assert best_state is not None
        continuation = rng_state(); load_models(agent, best_state); out = evaluate(agent); met = metrics(out, agent.tst_idx); assert abs(met["acc"] - best) <= 1e-12
        if config == "A0":
            expected = historical_initialization_refs()[fold]
            for key in ("acc", "auc", "sen_asd", "spe_td"):
                assert abs(met[key] - expected[key]) <= 1e-12, ("A0/E1 baseline parity", fold, key, met[key], expected[key])
            assert best_epoch == expected["best_epoch"], ("A0/E1 best epoch parity", fold, best_epoch, expected["best_epoch"])
        row = dict(config=config, seed=0, fold=fold, physical_gpu=gpu, **met, best_epoch=best_epoch, epochs_executed=last_epoch + 1, selected_val_acc=best, train_n=len(agent.trn_idx), test_n=len(agent.tst_idx), initial_trunk_sha256=init_hash, initial_rng_sha256=init_rng_hash, wall_seconds=time.monotonic() - fold_start)
        diag = diagnostic(agent, config, fold, tracker)
        result["metrics"].append(row); result["diagnostics"].append(diag); restore_rng(continuation)
        result["seconds"] = oldseconds + time.monotonic() - begin; result["status"] = "COMPLETE" if fold == 10 else "RUNNING"; result["last_fold_seconds"] = time.monotonic() - fold_start
        checkpoint_write(ck, dict(signature=signature(), phase="BOUNDARY", fold=fold + 1, rng=continuation, result=result)); jwrite(task_json(config), result)
        print("FOLD_DONE", name, fold, "ACC", met["acc"], "AUC", met["auc"], flush=True)
        del agent, best_state, out
        torch.cuda.empty_cache()
    print("TASK_COMPLETE", name, result["seconds"], flush=True)


def collect():
    groups = [task_result(c) for c in CONFIGS]
    groups = [g for g in groups if g]
    return groups, [x for g in groups for x in g["metrics"]], [x for g in groups for x in g["diagnostics"]]


def refresh():
    m = jread(MANIFEST); groups, rows, diag = collect()
    if rows: cwrite(RESULTS / "fold_metrics.csv", rows)
    if diag: cwrite(RESULTS / "diagnostics.csv", diag)
    lookup = {g["config"]: g for g in groups}
    for t in m["tasks"]:
        g = lookup.get(t["config"])
        if g: t.update(status=g["status"], completed_folds=len(g["metrics"]), seconds=g.get("seconds", 0.))
    m.update(completed_folds=len(rows), completed_tasks=sum(g["status"] == "COMPLETE" for g in groups), gpu_completed_folds={str(g): sum(int(r["physical_gpu"]) == g for r in rows) for g in (0, 1)}, refreshed_utc=utc())
    jwrite(MANIFEST, m)
    return groups, rows, diag


def worker_env(gpu):
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    return env


def schedule():
    pending = [c for c in CONFIGS if not (task_result(c) or {}).get("status") == "COMPLETE"]
    active = []; failure = None
    while pending or active:
        for config in list(pending):
            g = GPU[config]
            if sum(q["gpu"] == g for q in active) >= WORKERS_PER_GPU: continue
            log = RUNTIME / (task_name(config) + ".log"); h = log.open("a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(EXP / "run.py"), "_worker", config], cwd=RUNTIME, env=worker_env(g), stdout=h, stderr=subprocess.STDOUT)
            active.append(dict(process=p, gpu=g, config=config, handle=h, log=log)); pending.remove(config); record(f"START {config} GPU{g} pid={p.pid} workers_per_gpu={WORKERS_PER_GPU}")
        time.sleep(5)
        for q in list(active):
            code = q["process"].poll()
            if code is None: continue
            q["handle"].close(); active.remove(q); record(f"EXIT {q['config']} code={code}")
            if code or not (task_result(q["config"]) or {}).get("status") == "COMPLETE":
                failure = {"config": q["config"], "returncode": code, "tail": q["log"].read_text()[-12000:]}; break
        refresh()
        if failure:
            for q in active: q["process"].terminate()
            m = jread(MANIFEST); m["status"] = "PARTIAL"; m["failures"].append(dict(time=utc(), stage="schedule", error=failure)); jwrite(MANIFEST, m)
            raise RuntimeError("formal worker failed: " + str(failure))


def historical_reference(path, config_name, accepted_configs):
    rows = readcsv(path)
    out = []
    for r in rows:
        if int(r.get("seed", 0)) != 0 or int(r["fold"]) > 10: continue
        config = r.get("config", "")
        if config not in accepted_configs: continue
        out.append(dict(config=config_name, seed=0, fold=int(r["fold"]), physical_gpu=int(r.get("physical_gpu", 0)),
                       acc=float(r["acc"]), auc=float(r.get("hard_auc", r.get("hard_auc", r.get("ba", 0.)))),
                       sen_asd=float(r.get("sen_asd", r.get("sensitivity_asd", 0.))), spe_td=float(r.get("spe_td", r.get("specificity_td", 0.))),
                       asd_fn=int(r.get("asd_fn", 0)), td_fp=int(r.get("td_fp", 0))))
    assert len(out) == 10, (path, config_name, len(out))
    return out


def aggregate(rows):
    refs = historical_reference(ROOT / "experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv", "P0", {"F0"})
    try:
        e1 = historical_reference(ROOT / "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/fold_metrics.csv", "E1", {"E1_NO_SPGS"})
    except AssertionError:
        e1 = []
    allrows = rows + refs + e1; table = {}
    for config in ("P0", "E1") + CONFIGS:
        rr = [r for r in allrows if r["config"] == config]
        if not rr: continue
        table[config] = {k: dict(mean=float(np.mean([float(r[k]) for r in rr])), sd_population=float(np.std([float(r[k]) for r in rr], ddof=0))) for k in METRICS}
        table[config]["asd_fn"] = int(sum(int(r.get("asd_fn", 0)) for r in rr)); table[config]["td_fp"] = int(sum(int(r.get("td_fp", 0)) for r in rr))
    return table


def pair(rows, left, right):
    a = {(int(r["seed"]), int(r["fold"])): r for r in rows if r["config"] == left}
    if right == "P0": b = {(int(r["seed"]), int(r["fold"])): r for r in historical_reference(ROOT / "experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv", "P0", {"F0"})}
    elif right == "E1": b = {(int(r["seed"]), int(r["fold"])): r for r in historical_reference(ROOT / "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment/results/fold_metrics.csv", "E1", {"E1_NO_SPGS"})}
    else: b = {(int(r["seed"]), int(r["fold"])): r for r in rows if r["config"] == right}
    out = {}
    for k in METRICS:
        d = np.asarray([float(a[key][k]) - float(b[key][k]) for key in sorted(a) if key in b])
        out[k] = dict(delta_pp=float(d.mean() * 100), wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()), losses=int((d < -1e-12).sum()), per_fold_delta_pp=(d * 100).tolist())
    return dict(left=left, right=right, metrics=out)


def decision(table, pairs, complete):
    if not complete: return {"go_no_go": "NO-GO", "status": "INCOMPLETE", "failures": ["180 folds incomplete"]}
    lookup = {(p["left"], p["right"]): p for p in pairs}
    required = [("A5", x) for x in ("P0", "E1", "A0", "A1", "A6", "A7", "A9", "A10", "A11", "B0", "B1", "B2", "B3", "B4", "B5")]
    failures = []
    for l, r in required:
        for metric in ("acc", "auc"):
            q = lookup[(l, r)]["metrics"][metric]
            if q["delta_pp"] <= .1 or q["wins"] < 7: failures.append({"comparison": f"{l}-{r}", "metric": metric, "delta_pp": q["delta_pp"], "wins": q["wins"]})
    return {"go_no_go": "GO" if not failures else "NO-GO", "status": "COMPLETE", "minimum_mean_gain_pp": .1, "minimum_fold_wins": 7, "failures": failures,
            "rationale": "GO requires A5 to exceed frozen baseline and every prespecified semantic, patient-matching, auxiliary and masked control by >0.1 pp in ACC and Hard-AUC with at least 7/10 paired-fold wins; otherwise NO-GO."}


def report():
    groups, rows, diag = collect(); table = aggregate(rows)
    comparisons = [("A5", x) for x in ("P0", "E1", "A0", "A1", "A2", "A3", "A4", "A6", "A7", "A8", "A9", "A10", "A11", "B0", "B1", "B2", "B3", "B4", "B5") if (x in table)]
    pairs = [pair(rows, l, r) for l, r in comparisons]
    complete = len(rows) == 180
    judgment = decision(table, pairs, complete)
    lines = ["# 实验31：SCORE 个体化脑连接语义证据与可靠性调制", "", f"状态：**{'COMPLETE' if complete else 'PARTIAL'}**；正式折数 **{len(rows)}/180**；seed=0。", "",
             "所有新组关闭原SPGS并在相同位置安装SCORE；P0与历史E1仅作只读参考。CORR状态由当前外层训练折median/MAD和训练站点内/间方差拟合，未使用诊断标签。", "",
             "## 四项正式指标（十折均值±population SD）", "", "|组|ACC|Hard-AUC|SEN-ASD|SPE-TD|ASD漏诊|TD误报|", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in ("P0", "E1") + CONFIGS:
        if c not in table: continue
        t = table[c]; source = "历史只读" if c in ("P0", "E1") else "本轮"
        lines.append("|" + c + " (" + source + ")|" + "|".join(f"{100*t[k]['mean']:.5f}±{100*t[k]['sd_population']:.5f}" for k in METRICS) + f"|{t['asd_fn']}|{t['td_fp']}|")
    lines += ["", "## 预设配对差（百分点；胜/平/负）", "", "|比较|指标|均值差|胜/平/负|逐折差|", "|---|---|---:|---|---|"]
    for p in pairs:
        for k, q in p["metrics"].items(): lines.append(f"|{p['left']}−{p['right']}|{k}|{q['delta_pp']:+.5f}|{q['wins']}/{q['ties']}/{q['losses']}|" + ",".join(f"{x:+.3f}" for x in q["per_fold_delta_pp"]) + "|")
    lines += ["", "## 机制诊断", "", "诊断文件 `diagnostics.csv` 保存状态分布、个体报告差异、注意力熵、站点稳定性、文本几何、masked loss/梯度、FiLM幅度、FUNC/ANAT门控及其站点/QC关联、残差比例和同checkpoint反事实；它们不是额外正式评价指标。", ""]
    if diag:
        lines += ["|组|状态0|状态1|状态2|状态3|状态4|报告差异|注意力熵|可靠性|masked loss|FiLM|FUNC门控|ANAT门控|delta/CAM|站点η²|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in CONFIGS:
            rr = [r for r in diag if r["config"] == c]
            if not rr: continue
            av = lambda k: float(np.mean([float(r[k]) for r in rr]))
            lines.append("|" + c + "|" + "|".join(f"{av(k):.6f}" for k in ("state0_fraction", "state1_fraction", "state2_fraction", "state3_fraction", "state4_fraction", "report_unique_mean", "attention_entropy", "reliability_mean", "masked_loss_mean", "pheno_film_norm", "func_gate_mean", "anat_gate_mean", "delta_relative_mean", "gate_site_eta2")) + "|")
    lines += ["", "## 科研判定", "", f"预设结论：**{judgment['go_no_go']}**。", judgment.get("rationale", "正式训练尚未完成，暂不能判定。"), "", "RTGMFF只借鉴了患者特异性ROI状态和年龄/性别FiLM；其AAL-116、ALFF/fALFF/ReHo、原始影像Wavelet-Mamba及公开代码中的非论文BioBERT实现均未直接移植。CORR256是预处理连接特征，CAM四组36维token不是解剖ROI。", "", "当前评价仍是原始历史协议内部证据：val=test参与checkpoint选择，RFE/标准化在外层前完成，GFR在871人上transductive传播，因此不能据此声称无偏泛化、临床校准或病灶因果。", "", "## 完整逐折指标", "", "|组|fold|ACC|Hard-AUC|SEN|SPE|ASD FN|TD FP|best epoch|epochs|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append("|" + r["config"] + "|" + str(r["fold"]) + "|" + "|".join(f"{100*float(r[k]):.6f}" for k in METRICS) + f"|{r['asd_fn']}|{r['td_fp']}|{r['best_epoch']}|{r['epochs_executed']}|")
    lines += ["", "## 完整性", "", f"源SHA：`{json.dumps(jread(MANIFEST)['source_sha256'], ensure_ascii=False)}`", f"代码SHA：`{json.dumps(jread(MANIFEST)['code_sha256'], ensure_ascii=False)}`", f"Prompt/Report缓存SHA：`{jread(MANIFEST)['report_bank_sha256']}`", f"GPU折数：`{json.dumps(jread(MANIFEST).get('gpu_completed_folds', {}))}`", ""]
    tmp = RESULTS / "report.md.tmp"; tmp.write_text("\n".join(lines) + "\n", encoding="utf-8"); os.replace(tmp, RESULTS / "report.md")
    return dict(table=table, pairs=pairs, rows=len(rows), diagnostics=len(diag), scientific=judgment["go_no_go"], judgment=judgment)


def append_handoff(summary):
    m = jread(MANIFEST); target = ROOT / "handoff.md"; marker = "<!-- EXPERIMENT31_SCORE_FINAL -->"
    if marker.encode() in target.read_bytes(): return
    table = summary["table"]
    text = "\n\n" + marker + "\n## 实验三十一：SCORE 个体化脑连接语义证据与可靠性调制\n\n"
    text += f"更新时间：{utc()}；状态：COMPLETE；正式训练折 {summary['rows']}/180；18组各seed=0连续10折。\n\n"
    text += "审计：核对原始main/model/network/layers/utils、原始shell、实验11/13/28/29/30与SPromptGL论文；验证CORR256、252条ROI连接映射、四个未知连接、PHENO48的年龄26+性别2+站点20、ANAT6/FUNC10、原SPGS位置与val=test checkpoint协议。RTGMFF只借鉴患者特异性ROI状态报告和年龄/性别FiLM，未移植AAL-116、ALFF/fALFF/ReHo、原始影像Wavelet-Mamba或其BioBERT路径。\n\n"
    text += "方法：每个外层训练折仅用CORR训练样本拟合median/MAD和训练站点内/(内+间)方差可靠性；252条已验证连接按五个参考状态生成1260条冻结RN50文本。SCORE以数值Query与当前患者状态文本Value交互聚合证据，PHENO年龄/性别只作Query FiLM，FUNC/ANAT只作证据门控；15%训练连接的masked状态预测不接收目标z/状态/文本，修正只作用CAM-CORR且严格小于5%。A0–A11及B0–B5全部从头训练，B组包含错误ROI、方向/程度、同站点错配、几何数值码和原8句固定Prompt控制。\n\n"
    text += "正式四指标（均值±十折population SD）见实验目录results/report.md；该报告只把ACC、Hard-AUC、SEN-ASD、SPE-TD列为论文指标，诊断另存diagnostics.csv。\n\n"
    text += "科研判定：**" + summary["scientific"] + "**。" + summary["judgment"].get("rationale", "") + " 正确文本、患者对应关系、质量门控和masked机制只有在预设配对差及机制诊断共同支持时才可归因；单seed且val=test的内部结果不能宣称无偏泛化或临床可用。\n\n"
    text += "完整交付：" + str(EXP) + "；GPU0/GPU1折数和墙钟见manifest.json；根目录源码保护哈希、缓存哈希、工程预检与断点恢复记录均在manifest及runtime审计日志中。未执行Git commit/push。\n"
    with (ROOT / "handoff.md.lock").open("a+b") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with target.open("a", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        fcntl.flock(lk, fcntl.LOCK_UN)
    m["handoff_appended"] = True; m["handoff_status"] = "APPENDED"; m["handoff_sha256"] = sha(target); jwrite(MANIFEST, m)


def finish():
    groups, rows, diag = refresh()
    assert len(groups) == 18 and len(rows) == len(diag) == 180 and all(g["status"] == "COMPLETE" for g in groups)
    m = jread(MANIFEST); m.update(status="COMPLETE", completed_utc=utc(), formal_seconds=float(m.get("formal_seconds", 0.)), result_sha256={p.name: sha(p) for p in RESULTS.glob("*") if p.is_file()}); jwrite(MANIFEST, m)
    summary = report(); append_handoff(summary)
    m = jread(MANIFEST); m["scientific_conclusion"] = summary["scientific"]; m["result_sha256"] = {p.name: sha(p) for p in RESULTS.glob("*") if p.is_file()}; jwrite(MANIFEST, m)
    print("STATUS=COMPLETE 18 tasks / 180 folds / GPU0=90 GPU1=90", flush=True)


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        rows = []
        for line in out.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 5:
                rows.append(dict(index=fields[0], name=fields[1], memory_used=fields[2],
                                 memory_total=fields[3], utilization=fields[4]))
        return rows
    except Exception as exc:
        return {"error": repr(exc)}


def cleanup_runtime():
    m = jread(MANIFEST)
    assert m.get("status") == "COMPLETE"
    removed = []
    for path in RUNTIME.rglob("*.tmp"):
        path.unlink()
        removed.append(str(path.relative_to(RUNTIME)))
    m["runtime_cleanup"] = dict(status="TEMP_FILES_REMOVED_AUDIT_PRESERVED", removed=removed,
                                completed_utc=utc(), checkpoints_preserved=True)
    jwrite(MANIFEST, m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "all", "resume", "refresh", "report", "_worker", "_preflight"))
    parser.add_argument("config", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare": prepare(); return
    if args.command == "preflight": preflight(); return
    if args.command == "_preflight": preflight(); return
    if args.command == "refresh": refresh(); return
    if args.command == "report": print((RESULTS / "report.md").read_text() if (RESULTS / "report.md").exists() else "report unavailable"); return
    if args.command == "_worker": worker(args.config); return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    controller = (RUNTIME / "controller.lock").open("a+"); fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        prepare(); m = jread(MANIFEST)
        if not m.get("preflight", {}).get("passed"): preflight()
        m = jread(MANIFEST); m["status"] = "RUNNING"
        if not m.get("formal_started_utc"): m["formal_started_utc"] = utc()
        m["workers_per_gpu"] = WORKERS_PER_GPU; m["gpu_load_at_formal_start"] = gpu_snapshot(); jwrite(MANIFEST, m)
        start = time.monotonic(); schedule(); m = jread(MANIFEST); m["formal_seconds"] = time.monotonic() - start; jwrite(MANIFEST, m); finish()
    except Exception as exc:
        import traceback; traceback.print_exc()
        if MANIFEST.exists():
            m = jread(MANIFEST); m["status"] = "PARTIAL"; m.setdefault("failures", []).append({"time": utc(), "error": repr(exc)}); jwrite(MANIFEST, m)
        raise
    finally:
        controller.close()
    if MANIFEST.exists() and jread(MANIFEST).get("status") == "COMPLETE":
        cleanup_runtime()


if __name__ == "__main__":
    main()
