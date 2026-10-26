#!/usr/bin/env python3
"""HCER-v2 prepare/preflight/all/worker/resume/report; no historical runtime dependency."""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import datetime
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
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = HERE / ".runtime"
CACHE, TASKS, CHECKPOINTS = RT / "cache", RT / "tasks", RT / "checkpoints"
SOURCES = RT / "sources"
RESULTS = HERE / "results"
CFG = json.loads((HERE / "config.json").read_text())
GROUPS = {g["id"]: g for g in CFG["groups"]}
_DATA = _CTX = _BANK = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hbytes(value):
    return hashlib.sha256(value).hexdigest()


def hjson(value):
    return hbytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def json_write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_read(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    if not fields:
        fields = ["group", "seed", "fold"]
    tmp = Path(path).with_name(Path(path).name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def data():
    global _DATA
    if _DATA is None:
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        p = ROOT / "data/ABIDE"
        frame = pd.read_csv(p / "processed_standard_data.csv")
        mapping = np.load(p / "modal_feat_dict.npy", allow_pickle=True).item()
        assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
        assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
        assert list(frame.columns[:-1]) == [str(v) for group in mapping.values() for v in group]
        x, y = frame.iloc[:, :-1].to_numpy(np.float32), frame.iloc[:, -1].to_numpy(np.int64) - 1
        assert x.shape == (871, 320) and np.bincount(y).tolist() == [403, 468]
        assert np.isfinite(x).all()
        folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
        _DATA = dict(x=x, y=y, folds=folds, mapping=mapping, columns=list(frame.columns))
    return _DATA


def prompts():
    obj = read_json(HERE / "prompts.json")
    hypotheses = [h for h in obj["hypotheses"] if h["candidate"]]
    canonical = CFG["data"]["canonical_edges"]
    assert len(hypotheses) == 13 and len(canonical) == len(set(canonical)) == 39
    assert set(canonical) == {e for h in hypotheses for e in h["edges"]}
    assert len(obj["hypotheses"]) == 16 and len({e for h in obj["hypotheses"] for e in h["edges"]}) == 49
    assert not set(canonical) & {195, 225, 232, 236}
    mapping = {int(r["correlation_feature_index"]): r for r in csv_read(ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv")}
    for h in hypotheses:
        for e, name in zip(h["edges"], h["connection_names"]):
            row = mapping[e]
            assert row["connection_anatomical_status"] == "VERIFIED_BOTH_ENDPOINTS"
            assert row["verification_status"] == "VERIFIED_NUMERIC_UNIQUE" and row["unique_match"] == "True"
            assert row["roi_i_name_verified"] == row["roi_j_name_verified"] == "True"
            assert row["connection_name"] == name
    return hypotheses, canonical, mapping


def seed_rng(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    import torch
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])


def rng_hash(value):
    digest = hashlib.sha256(pickle.dumps((value["python"], value["numpy"]), protocol=4))
    for v in [value["cpu"]] + value["cuda"]:
        digest.update(v.cpu().numpy().tobytes())
    return digest.hexdigest()


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def states(agent):
    return {k: {n: p.detach().cpu().clone() for n, p in m.state_dict().items()}
            for k, m in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))}


def set_states(agent, value):
    agent.MessagePassing0.load_state_dict(value["gfr"])
    agent.ModalFusion.load_state_dict(value["pln"])


def trunk_hash(agent):
    h = hashlib.sha256()
    for key, m in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, p in sorted(m.named_parameters()):
            if name.startswith("hcer."): continue
            h.update(key.encode() + b"/" + name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_grads(agent):
    return {key + "/" + name: None if p.grad is None else p.grad.detach().clone()
            for key, m in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))
            for name, p in m.named_parameters() if not name.startswith("hcer.")}


def runtime():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        sys.path.insert(0, str(SOURCES))
        import model, network, utils
        _CTX = dict(torch=torch, model=model, network=network, utils=utils)
    return _CTX


def make_base(fold):
    c = runtime(); d = data(); train, test = d["folds"][fold - 1]
    with contextlib.redirect_stdout(io.StringIO()):
        agent = c["model"].EvalHelper([48, 6, 10, 256], d["x"], d["y"], SimpleNamespace(**CFG["protocol"]), train, test)
    agent.MF_sav.close(); agent.GCMP_sav.close()
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    return agent


def reference(fold):
    from method import build_reference
    path = CACHE / f"reference_{fold:02}.pt"
    if path.exists():
        value = load(path)
        assert value["config_sha"] == sha(HERE / "config.json")
        return value
    d = data(); tr, te = d["folds"][fold - 1]
    value = build_reference(d["x"], d["y"], tr, te, CFG["data"]["canonical_edges"], fold)
    value.update(config_sha=sha(HERE / "config.json"), train=tr, test=te)
    save(path, value)
    return value


def text_cache():
    global _BANK
    if _BANK is not None: return _BANK
    import torch
    path = CACHE / "text_bank.pt"
    if path.exists():
        _BANK = load(path)
        assert _BANK["prompt_sha"] == sha(HERE / "prompts.json")
        return _BANK
    sys.path.insert(0, str(ROOT))
    from CLIP.clip import clip
    hypotheses, canonical, mapping = prompts()
    real = [[h["asd"], h["td"]] for h in hypotheses]
    anatomy = ["Resting-state functional connectivity between " + mapping[e]["connection_name"].replace(" — ", " and ") + "." for e in canonical]
    neutral = []
    for h in hypotheses:
        pair_name = mapping[h["edges"][0]]["connection_name"].replace(" — ", " and ")
        neutral.append(["Measurement record: " + pair_name + ". Resting-state functional coupling is summarized for these atlas parcels and the other mapped connections using the same acquisition and statistical processing rules.",
                        "Measurement reference: " + pair_name + ". Resting-state functional coupling is summarized for these atlas parcels and the other mapped connections with comparable acquisition and statistical processing rules."])
    texts = [t for pair in real for t in pair] + [t for pair in neutral for t in pair] + anatomy
    tokens = clip.tokenize(texts, context_length=77, truncate=False)
    weights = Path.home() / ".cache/clip/RN50.pt"
    assert sha(weights) == CFG["method"]["weights_sha"]
    before = rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model, _ = clip.load("RN50", "cuda")
        model.eval().requires_grad_(False)
        with torch.no_grad():
            encoded = torch.cat([model.encode_text(t.to("cuda")).float().cpu() for t in tokens.split(32)])
            check = model.encode_text(tokens[:32].to("cuda")).float().cpu()
            # RN50 uses half precision on this accelerator. Reproducibility is
            # checked with the frozen batch shape, rather than a different GEMM.
            assert torch.equal(encoded[:32], check), "fixed-batch frozen encoding not reproducible"
    assert rng_hash(before) == rng_hash(rng_state())
    real_bank = encoded[:26].reshape(13, 2, 1024)
    generator = np.random.default_rng(400002)
    perm = generator.permutation(1024)
    signs = torch.from_numpy(generator.choice([-1., 1.], 1024).astype(np.float32))
    anon = real_bank[:, :, perm] * signs
    normed = torch.nn.functional.normalize(real_bank.flatten(0, 1), dim=-1)
    anon_norm = torch.nn.functional.normalize(anon.flatten(0, 1), dim=-1)
    gram_error = float((normed @ normed.T - anon_norm @ anon_norm.T).abs().max())
    assert gram_error < 2e-6
    _BANK = dict(real=real_bank, anonymous=anon, neutral=encoded[26:52].reshape(13, 2, 1024),
                 anatomy=encoded[52:], texts=texts, prompt_sha=sha(HERE / "prompts.json"),
                 token_sha=hbytes(tokens.numpy().tobytes()), embedding_sha=hbytes(encoded.numpy().tobytes()),
                 token_lengths=(tokens != 0).sum(1).tolist(), max_tokens=int((tokens != 0).sum(1).max()),
                 anonymous_gram_max_error=gram_error, weights_sha=sha(weights),
                 clip_script_sha=sha(ROOT / "CLIP/clip/clip.py"), tokenizer_sha=sha(ROOT / "CLIP/clip/simple_tokenizer.py"))
    save(path, _BANK)
    del model
    torch.cuda.empty_cache()
    return _BANK


def attach(agent, group, fold):
    import torch
    from method import HCER
    g = GROUPS[group]; bank = text_cache(); hypotheses, canonical, _ = prompts()
    texts = bank["anonymous" if g["variant"] == "anonymous" else "neutral" if g["variant"] == "neutral" else "real"]
    if g["variant"] == "swapped": texts = texts.flip(1)
    before = rng_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        branch = HCER(texts, bank["anatomy"], canonical, hypotheses, variant=g["variant"], seed=400000 + fold).to(agent.dev)
    assert rng_hash(before) == rng_hash(rng_state()), "new module changed trunk RNG"
    agent.ModalFusion.hcer = branch
    agent.optimizer_MF.add_param_group(dict(params=list(branch.parameters()), lr=.0038, weight_decay=0.))
    ref = reference(fold)
    agent.reference40 = {k: torch.as_tensor(ref[k], device=agent.dev, dtype=torch.float32) for k in ("mu", "sigma", "reliability")}
    agent.ModalFusion.context40 = dict(raw=agent.feat, **agent.reference40)
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    with torch.no_grad():
        forward(agent)
        if g["variant"] != "numeric":
            branch.threshold.copy_(torch.quantile(branch.last["score"][agent.trn_idx].flatten(), .75))
    optimizer_audit(agent)
    return branch


def optimizer_audit(agent):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    ids = [id(p) for pg in groups for p in pg["params"]]
    actual = {id(p) for m in (agent.MessagePassing0, agent.ModalFusion) for p in m.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual
    assert groups[0]["weight_decay"] == groups[1]["weight_decay"] == .11
    if hasattr(agent.ModalFusion, "hcer"): assert len(groups) == 3 and groups[2]["weight_decay"] == 0.


def forward(agent, raw=None, override=None, graph_loss=True):
    raw = agent.feat if raw is None else raw
    count, parts = agent.dataLayer(raw)
    feat, graphs, aloss = agent.MessagePassing0(parts, count, agent.targ, agent.trn_idx)
    loss = None
    if graph_loss:
        u = runtime()["utils"]
        loss = sum(u.GraphConstructLoss(parts[i], graphs[i], 1., .5, 0.) for i in range(4)) / 4
    if hasattr(agent.ModalFusion, "hcer"):
        agent.ModalFusion.context40 = dict(raw=raw, **agent.reference40, **(override or {}))
    logp, hidden, attn = agent.ModalFusion(feat)
    return logp, loss, graphs, feat


def evaluate(agent, **kwargs):
    import torch
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    with torch.no_grad(): return forward(agent, graph_loss=False, **kwargs)


def metric(output, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y = data()["y"][test]; pred = output.detach().cpu().numpy()[test].argmax(1)
    tp, fn, fp, tn = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(acc=float((pred == y).mean()), hard_auc=float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred])),
                spe_td=tn / (tn + fp), sen_asd=tp / (tp + fn), asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def intervene(raw, subject, cols, mu, mask, keep):
    result = raw.clone()
    factor = mask if keep else 1 - mask
    smooth = mu[subject] + (raw[subject, cols] - mu[subject]) * factor
    # Algebraic mu+(x-mu)*1 can round differently from x. Hard values are exact;
    # only the zero-valued straight-through term carries the mask derivative.
    hard = __import__("torch").where(factor.detach() > .5, raw[subject, cols], mu[subject])
    result[subject, cols] = hard + (smooth - smooth.detach())
    return result


def cf_loss(agent, group, probe_ids, diagnostic=False):
    import torch
    import torch.nn.functional as F
    branch = agent.ModalFusion.hcer
    train_ids = set(map(int, agent.trn_idx)); assert set(probe_ids) <= train_ids
    backbone = [p for m in (agent.MessagePassing0, agent.ModalFusion) for name, p in m.named_parameters() if not name.startswith("hcer.")]
    flags = [p.requires_grad for p in backbone]
    original_grads = parameter_grads(agent) if diagnostic else None
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    for p in backbone: p.requires_grad_(False)
    audit = dict(probes=0, unique_ids=[], eligible_probes=0, delete_probes=0, keep_loss=0., delete_loss=0.)
    try:
        # Unperturbed selector is calculated once, with STE despite dropout-free eval.
        branch.train()
        full, _, _, _ = forward(agent, graph_loss=False)
        selected = branch.last["selected"]
        selected.retain_grad()
        hard = branch.last["hard"].detach()
        fixed_delta = branch.last["delta"].detach()
        teacher = full.detach().exp()
        terms = []
        for person in probe_ids:
            audit["probes"] += 1; audit["unique_ids"].append(person)
            if hard[person].sum().item() == 0: continue
            audit["eligible_probes"] += 1
            keep_x = intervene(agent.feat, person, branch.columns, agent.reference40["mu"], selected[person], True)
            context = dict(mask_override=selected.detach(), fixed_delta=fixed_delta)
            keep, _, _, _ = forward(agent, keep_x, context, graph_loss=False)
            loss_keep = F.kl_div(keep[person], teacher[person], reduction="sum")
            term = .10 * loss_keep; audit["keep_loss"] += float(loss_keep.detach())
            if GROUPS[group]["delete"] and int(full[person].argmax()) == int(agent.targ[person]):
                delete_x = intervene(agent.feat, person, branch.columns, agent.reference40["mu"], selected[person], False)
                deleted, _, _, _ = forward(agent, delete_x, context, graph_loss=False)
                label = agent.targ[person]
                loss_delete = F.softplus(.05 - (keep[person, label] - deleted[person, label]))
                term = term + .05 * loss_delete
                audit["delete_probes"] += 1; audit["delete_loss"] += float(loss_delete.detach())
            terms.append(term)
        if terms:
            loss = torch.stack(terms).mean()
            loss.backward()
            assert torch.isfinite(loss)
        audit["selector_mask_gradient"] = 0. if selected.grad is None else float(selected.grad.norm())
        audit["cf_query_gradient"] = 0. if branch.text_projection.weight.grad is None else float(branch.text_projection.weight.grad.norm())
        if diagnostic:
            assert equal(original_grads, parameter_grads(agent)), "CF modified original backbone gradients"
            audit["backbone_gradients_unchanged"] = True
        return audit
    finally:
        for p, flag in zip(backbone, flags): p.requires_grad_(flag)
        agent.MessagePassing0.train(); agent.ModalFusion.train()


def step(agent, group, fold, epoch, probe_order, probe_cursor, diagnostic=False):
    import torch
    import torch.nn.functional as F
    agent.MessagePassing0.train(); agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad(); agent.optimizer_MF.zero_grad()
    output, graph, _, _ = forward(agent)
    cls = F.nll_loss(output[agent.trn_idx], agent.targ[agent.trn_idx], agent.weight)
    total = cls + graph
    assert torch.isfinite(total) and torch.isfinite(output).all()
    total.backward()
    audit = dict(cls=float(cls.detach()), graph=float(graph.detach()), probes=0)
    if group != "B0":
        b = agent.ModalFusion.hcer
        audit["classification_query_gradient"] = 0. if b.text_projection.weight.grad is None else float(b.text_projection.weight.grad.norm())
        audit["classification_fuse_gradient"] = 0. if b.fuse.weight.grad is None else float(b.fuse.weight.grad.norm())
        audit["classification_gate_gradient"] = 0. if b.beta.grad is None else float(b.beta.grad.abs())
        if GROUPS[group]["keep"] and (epoch + 1) % 5 == 0:
            ids = [int(probe_order[(probe_cursor + k) % len(probe_order)]) for k in range(2)]
            audit.update(cf_loss(agent, group, ids, diagnostic=diagnostic))
            probe_cursor += 2
    agent.optimizer_MP0.step(); agent.optimizer_MF.step()
    return audit, probe_cursor


def code_hashes():
    return {name: sha(HERE / name) for name in ("config.json", "prompts.json", "method.py", "integration.py", "integration.patch", "run.py")}


def verify_protected():
    audit = read_json(RT / "source_audit.json")
    for file, digest in audit["read_only_sources"].items():
        assert sha(file) == digest, "protected source/data changed: " + file


def prepare():
    import torch
    import integration
    for path in (SOURCES, CACHE, TASKS, CHECKPOINTS, RT / "logs", RT / "anchors", RESULTS): path.mkdir(parents=True, exist_ok=True)
    verify_protected(); d = data(); hypotheses, edges, mapping = prompts()
    integration.emit(ROOT, SOURCES, HERE / "integration.patch")
    bank = text_cache()
    refs = {}
    for fold in range(1, 11):
        r = reference(fold)
        refs[str(fold)] = {"sha": sha(CACHE / f"reference_{fold:02}.pt"), "oof_td_rms": r["oof_td_rms"],
                           "train_asd_z_rms": r["oof_train_asd_z_rms"], "train_td_z_rms": r["oof_train_td_z_rms"], "test_z_rms": r["test_z_rms"]}
    manifest_path = HERE / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["config_sha"] == sha(HERE / "config.json")
        if manifest.get("formal_started"):
            assert manifest["code_sha"] == code_hashes(), "implementation changed after formal start"
        else:
            manifest["code_sha"] = code_hashes()
    else:
        manifest = dict(experiment=CFG["experiment"], scientific="PREPARED", delivery="PARTIAL_TMP_ONLY",
                        prepared=now(), started_unix=read_json(RT / "storage_probe.json")["time"],
                        stage=str(HERE), formal_target=str(ROOT / "experiment" / HERE.name),
                        config_sha=sha(HERE / "config.json"), code_sha=code_hashes(),
                        source_audit=read_json(RT / "source_audit.json"), storage_probe=read_json(RT / "storage_probe.json"),
                        fold_hash=hbytes(b"".join(np.concatenate([a, b]).astype(np.int64).tobytes() for a, b in d["folds"])),
                        references=refs, canonical_mapping={str(e):mapping[e]["connection_name"] for e in edges},
                        n_subjects=871, classes={"ASD0":403,"TD1":468}, protocol=CFG["protocol"],
                        prompt_sha=sha(HERE / "prompts.json"), text_cache_sha=sha(CACHE / "text_bank.pt"),
                        text_geometry=dict(max_tokens=bank["max_tokens"], anonymous_gram_max_error=bank["anonymous_gram_max_error"]),
                        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                         hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],text=True).strip()),
                        target_folds=100, completed_folds=0, gpu_completed={"0":0,"1":0},
                        anchor="PENDING", preflight="PENDING", groups=CFG["groups"], recoveries=[],
                        handoff_prefix=read_json(RT / "handoff_prefix.json"), handoff="PENDING")
    json_write(manifest_path, manifest)
    for name in ("fold_metrics.csv", "diagnostics.csv"):
        if not (RESULTS / name).exists(): csv_write(RESULTS / name, [])
    if not (RESULTS / "report.md").exists():
        (RESULTS / "report.md").write_text("# Experiment 40 — HCER-v2\n\nStatus: PREPARED; formal folds 0/100. No performance claim.\n")
    print("PREPARED", "tokens_max", bank["max_tokens"], "anonymous_Gram_error", bank["anonymous_gram_max_error"], flush=True)


def preflight():
    import torch
    import torch.nn.functional as F
    from method import HCER, build_reference
    started = time.monotonic(); prepare(); c = runtime(); seed_rng(0)
    a = make_base(1); entry = rng_state(); initial = states(a)
    # Independent exact E32 patch, extracted without importing its unused methods.
    historic = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing/integration.py"
    tree = ast.parse(historic.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_network")
    ns = {"ast":ast}; exec(compile(ast.Module(body=[function], type_ignores=[]), str(historic), "exec"), ns)
    ref_path = RT / "sources" / "network_reference.py"
    ref_path.write_text(ns["patch_network"]((ROOT / "network.py").read_text()))
    spec = importlib.util.spec_from_file_location("network_reference", ref_path)
    reference_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference_module)
    reference_module._prompt_text_features = c["network"]._prompt_text_features
    # Constructor/parity is independent of first-CLIP random consumption, which B0 replay audits.
    state_before = rng_state()
    c["network"]._prompt_text_features = reference_module._prompt_text_features
    b = make_base(1)
    restore_rng(state_before)
    gfr_ref = reference_module.GFR([48,6,10,256], [48,6,10,256], .9, "weighted-cosine").to(a.dev)
    pln_ref = reference_module.PLN([48,6,10,256], SimpleNamespace(**CFG["protocol"])).to(a.dev)
    opt_gfr_ref = torch.optim.Adam(gfr_ref.parameters(), lr=.0038, weight_decay=.11)
    opt_pln_ref = torch.optim.Adam(pln_ref.parameters(), lr=.0038, weight_decay=.11)
    pln_ref.apply(c["utils"].my_weight_init); pln_ref.dcer32_config = "B0"
    assert trunk_hash(b) == trunk_hash(SimpleNamespace(MessagePassing0=gfr_ref, ModalFusion=pln_ref))
    checks = {"independent_E32_constructor_parameters": True}
    rb = SimpleNamespace(**vars(b)); rb.MessagePassing0=gfr_ref; rb.ModalFusion=pln_ref
    rb.optimizer_MP0=opt_gfr_ref; rb.optimizer_MF=opt_pln_ref
    attach(b, "E5", 1)
    b.ModalFusion.context40["gate_disabled"] = True
    for epoch in range(2):
        dropout_rng = rng_state()
        for agent in (rb, b):
            restore_rng(dropout_rng)
            agent.MessagePassing0.train(); agent.ModalFusion.train()
            agent.optimizer_MP0.zero_grad(); agent.optimizer_MF.zero_grad()
            out, graph, graphs, feat = forward(agent, override=dict(gate_disabled=True))
            cls = F.nll_loss(out[agent.trn_idx], agent.targ[agent.trn_idx], agent.weight)
            (cls + graph).backward()
            current = dict(out=out.detach().clone(), feat=feat.detach().clone(), graphs=[q.detach().clone() for q in graphs],
                           cls=cls.detach().clone(), graph=graph.detach().clone(), gradients=parameter_grads(agent))
            agent.optimizer_MP0.step(); agent.optimizer_MF.step()
            current["trunk_sha"] = trunk_hash(agent)
            if agent is rb: expected=current
            else: assert equal(expected,current), "gate0 forward/loss/gradient/Adam step mismatch"
    checks["gate0_E32_forward_loss_graph_gradients_two_Adam_steps_bitwise"] = True
    # Medical data boundary: test labels cannot alter reference; self is absent from its fit.
    d=data(); tr,te=d["folds"][0]; r=reference(1)
    changed=d["y"].copy(); changed[te]=1-changed[te]
    r2=build_reference(d["x"],changed,tr,te,CFG["data"]["canonical_edges"],1,r["inner_assignment"])
    assert np.array_equal(r["mu"],r2["mu"]) and np.array_equal(r["z"],r2["z"])
    person=int(tr[0]); changed=d["y"].copy(); changed[person]=1-changed[person]
    r3=build_reference(d["x"],changed,tr,te,CFG["data"]["canonical_edges"],1,r["inner_assignment"])
    assert np.array_equal(r["mu"][person],r3["mu"][person])
    assert all(person not in r["fitted_ids"][int(r["inner_assignment"][person])] for _ in [0])
    checks["same_crossfit_algorithm_all_train_labels"] = True
    checks["test_label_invariance_reference"] = True
    # Shape/membership, strict masking and real anatomy identities.
    branch=b.ModalFusion.hcer
    normal=evaluate(b)[0]; route={k:v.detach().clone() for k,v in branch.last.items() if isinstance(v,torch.Tensor)}
    assert route["attention"].shape == (871,13,2,39)
    assert torch.allclose(route["attention"].sum(-1), torch.ones(871,13,2,device=b.dev))
    forbidden=route["attention"] * (~branch.membership)[None,:,None,:]
    assert forbidden.eq(0).all()
    assert route["count"].max()<=8 and route["ratio"].max()<=.0500001
    empty=evaluate(b,override=dict(mask_override=torch.zeros(871,39,device=b.dev)))[0]
    assert branch.last["summary"].eq(0).all() and branch.last["delta"].eq(0).all()
    bypass=evaluate(b,override=dict(gate_disabled=True))[0]
    assert torch.equal(empty,bypass), "empty selection has evidence bypass"
    checks["all_summaries_masked_empty_strict_B0A"] = True
    tests=rng_state(); out_before=evaluate(b)[0]; old=b.targ.clone(); b.targ[te]=1-b.targ[te]
    assert torch.equal(evaluate(b)[0],out_before); b.targ.copy_(old)
    checks["heldout_label_invariance_inference"] = True
    branch.text_override=branch.texts.flip(1)
    swapped=evaluate(b)[0]; swap_delta=float((swapped[tr]-normal[tr]).abs().mean())
    assert swap_delta>1e-8 and torch.isfinite(swapped).all()
    branch.text_override=None
    saved_anatomy=branch.anatomy.clone(); branch.anatomy.copy_(saved_anatomy.roll(1,0))
    anatomy_changed=evaluate(b)[0]
    assert float((anatomy_changed[tr]-normal[tr]).abs().max())>1e-8
    branch.anatomy.copy_(saved_anatomy)
    checks["actual_text_swap_and_anatomy_identity_response"] = True
    bank=text_cache(); hypotheses,edges,_=prompts()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        wrong=HCER(bank["real"],bank["anatomy"],edges,hypotheses,"wrong",400001).to(b.dev)
        no_rival=HCER(bank["real"],bank["anatomy"],edges,hypotheses,"no_rival",400001).to(b.dev)
    assert torch.equal(wrong.columns,branch.columns) and not torch.equal(wrong.membership,branch.membership)
    assert sum(p.numel() for p in wrong.parameters())==sum(p.numel() for p in branch.parameters())
    cam=b.ModalFusion.cam40.detach()
    with torch.no_grad():
        branch(cam,b.feat,**b.reference40); ref_ratio=branch.last["ratio"].clone()
        no_rival(cam,b.feat,**b.reference40)
        valid=(branch.last["count"]>0)&(no_rival.last["count"]>0)
        assert valid.any() and (ref_ratio[valid]-no_rival.last["ratio"][valid]).abs().max()<1e-5
    checks["E7_changes_only_association_canonical_tensors_unmoved"] = True
    checks["E5_E9_nonempty_residual_norm_matched"] = True
    # One-target interventions and graph instrumentation.
    out,_,graphs,_=evaluate(b)
    hard=branch.last["hard"].detach().clone(); candidates=[int(i) for i in tr if hard[i].sum()>0]
    assert candidates
    person=candidates[0]
    deleted=intervene(b.feat,person,branch.columns,b.reference40["mu"],hard[person],False)
    diff=deleted!=b.feat; assert diff.sum()==hard[person].sum()
    assert diff[torch.arange(871,device=b.dev)!=person].sum()==0 and diff[:,:64].sum()==0
    assert torch.equal(deleted[person,branch.columns[hard[person].bool()]],b.reference40["mu"][person,hard[person].bool()])
    changed=evaluate(b,raw=deleted,override=dict(mask_override=hard))[0]
    assert (changed[person]-out[person]).abs().max()>0
    b.MessagePassing0.fixed_graph40=None; assert torch.equal(evaluate(b)[0],out)
    b.MessagePassing0.fixed_graph40=[a.detach() for a in graphs]
    assert torch.equal(evaluate(b)[0],out)
    b.MessagePassing0.fixed_graph40=None
    checks["single_target_rawFC_and_graph_freeze_instrumentation"] = True
    # Mechanism-only train updates. Heldout classification performance is not computed.
    order=np.random.default_rng(400005).permutation(tr)
    health=[]
    for e in range(3):
        audit,_=step(b,"E5",1,e,order,0);health.append(audit)
    b.optimizer_MP0.zero_grad();b.optimizer_MF.zero_grad()
    start=time.monotonic()
    audit=cf_loss(b,"E5",candidates[:2],diagnostic=True)
    torch.cuda.synchronize()
    assert audit["selector_mask_gradient"]>0 and audit["cf_query_gradient"]>0, "CF selector/text gradients absent"
    checks["CF_backbone_frozen_raw_mask_gradient_nonzero"] = True
    checks["classification_branch_gradients"] = all(a["classification_fuse_gradient"]>0 and a["classification_query_gradient"]>0 for a in health)
    assert checks["classification_branch_gradients"]
    benchmark=dict(single_CF_two_subject_seconds=time.monotonic()-start,
                   max_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),gradient_audit=audit,
                   text_swap_train_abs_logp=swap_delta,health_updates=health)
    # Exact interrupted execution: capture after two updates; replay third with CF.
    snapshot=dict(model=states(b),mp=copy.deepcopy(b.optimizer_MP0.state_dict()),mf=copy.deepcopy(b.optimizer_MF.state_dict()),rng=rng_state())
    save(RT/"preflight_resume.pt",snapshot)
    step(b,"E5",1,4,order,0)
    expected=dict(model=states(b),mp=copy.deepcopy(b.optimizer_MP0.state_dict()),mf=copy.deepcopy(b.optimizer_MF.state_dict()),rng=rng_state())
    restored=load(RT/"preflight_resume.pt");set_states(b,restored["model"])
    b.optimizer_MP0.load_state_dict(restored["mp"]);b.optimizer_MF.load_state_dict(restored["mf"]);restore_rng(restored["rng"])
    step(b,"E5",1,4,order,0)
    actual=dict(model=states(b),mp=copy.deepcopy(b.optimizer_MP0.state_dict()),mf=copy.deepcopy(b.optimizer_MF.state_dict()),rng=rng_state())
    assert equal(expected,actual), "resume next CF update not bitwise equal"
    checks["checkpoint_next_update_all_states_bitwise"] = True
    verify_protected()
    result=dict(status="PASS",checks=checks,benchmark=benchmark,seconds=time.monotonic()-started,code_sha=code_hashes(),completed=now())
    json_write(RT/"preflight.json",result)
    m=read_json(HERE/"manifest.json");m["preflight"]="PASS";m["code_sha"]=code_hashes();m["preflight_evidence"]=result;json_write(HERE/"manifest.json",m)
    (RT/"preflight_resume.pt").unlink()
    print("PREFLIGHT_PASS",json.dumps(benchmark,ensure_ascii=False),flush=True)


def historical_rows():
    return {int(r["fold"]):r for r in csv_read(ROOT/"experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv") if r["config"]=="B0"}


def compare_anchor(row):
    reference=historical_rows()[int(row["fold"])]
    mismatches=[]
    for field in ("asd_tp","asd_fn","td_fp","td_tn","best_epoch","epochs_executed"):
        if int(row[field]) != int(reference[field]): mismatches.append(field)
    for field in ("initial_trunk_sha256","initial_rng_sha256"):
        if row[field] != reference[field]: mismatches.append(field)
    for left,right in (("acc","acc"),("hard_auc","auc"),("spe_td","spe_td"),("sen_asd","sen_asd")):
        if float(row[left]) != float(reference[right]): mismatches.append(left)
    return mismatches


def fold_diagnostic(agent,group,fold,tracker):
    import torch
    seed=4000000+fold
    d=data();train,test=d["folds"][fold-1]
    if group=="B0": return dict(group=group,seed=0,fold=fold,anchor_exact_match=True)
    branch=agent.ModalFusion.hcer
    output,_,graphs,_=evaluate(agent)
    route={k:v.detach().clone() for k,v in branch.last.items() if isinstance(v,torch.Tensor)}
    hard=route["hard"]; fixed_delta=route["delta"]
    normal_probability=output.exp()
    base=evaluate(agent,override=dict(gate_disabled=True))[0]
    branch.text_override=branch.texts.flip(1)
    swapped=evaluate(agent)[0]
    branch.text_override=None
    neutral=text_cache()["neutral"].to(agent.dev)
    branch.text_override=torch.nn.functional.normalize(neutral,dim=-1)
    neutral_out=evaluate(agent)[0]
    branch.text_override=None
    rng=np.random.default_rng(seed)
    effects={name:[] for name in ("selected_delete","selected_keep","random_delete","topz_delete","frozen_selected_delete","frozen_topz_delete","fixed_residual_selected_delete")}
    flips={name:[] for name in ("selected_delete","selected_keep","random_delete","topz_delete")}
    probability_change={name:[] for name in flips}
    for person in map(int,test):
        count=int(hard[person].sum())
        if not count: continue
        random_mask=torch.zeros(39,device=agent.dev)
        random_mask[torch.as_tensor(rng.choice(39,count,replace=False),device=agent.dev)]=1.
        zmask=torch.zeros(39,device=agent.dev);zmask[route["z"][person].abs().topk(count).indices]=1.
        label=int(d["y"][person]);pred=int(output[person].argmax())
        for name,mask,keep in (("selected_delete",hard[person],False),("selected_keep",hard[person],True),
                               ("random_delete",random_mask,False),("topz_delete",zmask,False)):
            raw=intervene(agent.feat,person,branch.columns,agent.reference40["mu"],mask,keep)
            changed=evaluate(agent,raw=raw,override=dict(mask_override=hard))[0]
            effects[name].append(float(output[person,label]-changed[person,label]))
            flips[name].append(int(changed[person].argmax())!=pred)
            probability_change[name].append(float((normal_probability[person]-changed[person].exp()).abs().mean()))
            if group in ("E1","E5") and name in ("selected_delete","topz_delete"):
                agent.MessagePassing0.fixed_graph40=[a.detach() for a in graphs]
                frozen=evaluate(agent,raw=raw,override=dict(mask_override=hard))[0]
                effects["frozen_"+name].append(float(output[person,label]-frozen[person,label]))
                agent.MessagePassing0.fixed_graph40=None
            if group=="E5" and name=="selected_delete":
                fixed=evaluate(agent,raw=raw,override=dict(mask_override=hard,fixed_delta=fixed_delta))[0]
                effects["fixed_residual_selected_delete"].append(float(output[person,label]-fixed[person,label]))
    ratios=route["ratio"][test].cpu().numpy()
    counts=route["count"][test].cpu().numpy()
    q=route["q"]; rawt=branch.texts
    selectedz=route["z"][test].abs()
    exact_z=0
    for index,person in enumerate(test):
        count=int(counts[index])
        if count:
            top=torch.zeros(39,device=agent.dev);top[selectedz[index].topk(count).indices]=1.
            exact_z+=int(torch.equal(top,hard[person]))
    graph_stats=[]
    for matrix in graphs:
        off=matrix.detach().clone();off.fill_diagonal_(0.)
        graph_stats.append(dict(edges=int((off>0).sum()),density=float((off>0).float().mean()),mean_neighbors=float((off>0).float().sum(1).mean())))
    diag=dict(group=group,seed=0,fold=fold,
              residual_mean=float(ratios.mean()),residual_median=float(np.median(ratios)),residual_p95=float(np.quantile(ratios,.95)),residual_max=float(ratios.max()),
              positive_gate=float(route["positive_gate"]),temperature=float(route["tau"]),
              null_mean=float(route["support"][test,:,2].mean()),selection_mean=float(counts.mean()),empty_fraction=float((counts==0).mean()),
              selected_count_hist=json.dumps(np.bincount(counts.astype(int),minlength=9).tolist()),
              hypothesis_selection_rate=json.dumps((torch.einsum("ne,he->nh",hard[test],branch.membership.float())>0).float().mean(0).cpu().tolist()),
              edge_selection_rate=json.dumps(hard[test].mean(0).cpu().tolist()),
              raw_pair_cos_mean=float((rawt[:,0]*rawt[:,1]).sum(-1).mean()),
              projected_pair_cos_mean=float((q[:,0]*q[:,1]).sum(-1).mean()),query_pair_distance=float((q[:,0]-q[:,1]).norm(dim=-1).mean()),
              swap_abs_logp=float((output[test]-swapped[test]).abs().mean()),swap_class_flip=float((output[test].argmax(1)!=swapped[test].argmax(1)).float().mean()),
              neutral_abs_logp=float((output[test]-neutral_out[test]).abs().mean()),disabled_abs_logp=float((output[test]-base[test]).abs().mean()),
              disabled_class_flip=float((output[test].argmax(1)!=base[test].argmax(1)).float().mean()),
              selection_equals_topz_fraction=exact_z/max(int((counts>0).sum()),1),evaluated_CF_subjects=int((counts>0).sum()),
              probes=int(tracker.get("probes",0)),unique_probe_count=len(tracker.get("probe_unique",[])),
              eligible_probes=int(tracker.get("eligible_probes",0)),delete_probes=int(tracker.get("delete_probes",0)),
              graph_stats=json.dumps(graph_stats),
              canonical_sha=hjson(CFG["data"]["canonical_edges"]),
              parameter_count=sum(p.numel() for p in branch.parameters()),
              effective_query_route=(group!="E1"),
              cf_gradient_policy="fixed unperturbed residual; no backbone CF parameter gradients")
    for name,values in effects.items():diag[name+"_logp_drop"]=float(np.mean(values)) if values else ""
    for name,values in flips.items():
        diag[name+"_flip"]=float(np.mean(values)) if values else ""
        diag[name+"_abs_probability_change"]=float(np.mean(probability_change[name])) if values else ""
    for name in ("classification_query_gradient","classification_fuse_gradient","classification_gate_gradient","selector_mask_gradient","cf_query_gradient"):
        diag[name+"_max"]=float(tracker.get(name+"_max",0.))
    assert np.isfinite(ratios).all() and ratios.max()<=.050001
    return diag


def anchor_path(fold): return RT/"anchors"/f"anchor_fold{fold:02}.pt"


def task_path(group): return TASKS/f"{group}_seed0.json"


def checkpoint_path(group): return CHECKPOINTS/f"{group}_latest.pt"


def checkpoint_best_path(group,fold): return CHECKPOINTS/f"{group}_fold{fold:02}_best.pt"


def worker(group):
    import torch
    manifest=read_json(HERE/"manifest.json")
    assert manifest["preflight"]=="PASS" and manifest["code_sha"]==code_hashes()
    if group!="B0": assert manifest["anchor"]=="ANCHOR_REPLAY_PASS"
    g=GROUPS[group];assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(g["gpu"])
    lock=(TASKS/f"{group}.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime();d=data();signature=hjson(dict(code=code_hashes(),folds=manifest["fold_hash"],cache=manifest["text_cache_sha"]))
    current=read_json(task_path(group)) if task_path(group).exists() else dict(group=group,status="RUNNING",metrics=[],diagnostics=[],started=now(),seconds=0.,resumes=0)
    for row in current["metrics"]:
        assert Path(row["best_checkpoint"]).exists() and sha(row["best_checkpoint"])==row["best_checkpoint_sha256"]
    if current["status"]=="COMPLETE":print("SKIP_COMPLETE",group,flush=True);return
    latest=load(checkpoint_path(group)) if checkpoint_path(group).exists() else None
    if latest:
        assert latest["signature"]==signature
        current=latest["task"];current["resumes"]+=1
    start=time.monotonic();previous_seconds=current["seconds"]
    if latest and latest["phase"]=="BOUNDARY" and group=="B0":
        runtime()["network"].get_prompt_text_features();restore_rng(latest["rng"])
    elif not latest:seed_rng(0)
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    first=int(latest["fold"]) if latest and latest["phase"]=="EPOCH" else len(current["metrics"])+1
    print("TASK_START",group,"GPU",g["gpu"],"fold",first,"resume",bool(latest),flush=True)
    for fold in range(first,11):
        resume=latest is not None and latest["phase"]=="EPOCH" and int(latest["fold"])==fold
        tr,te=d["folds"][fold-1]
        if group=="B0" and resume:
            original_anchor=load(anchor_path(fold))
            if fold==1:runtime()["network"]._prompt_text_features=None
            else:runtime()["network"].get_prompt_text_features()
            restore_rng(original_anchor["entry_rng"])
        if group!="B0":
            anchor=load(anchor_path(fold))
            if fold==1:runtime()["network"]._prompt_text_features=None
            else:runtime()["network"].get_prompt_text_features()
            restore_rng(anchor["entry_rng"])
        entry=rng_state();agent=make_base(fold)
        initial_hash=trunk_hash(agent);entry_hash=rng_hash(entry)
        if group=="B0":
            expected=historical_rows()[fold]
            assert initial_hash==expected["initial_trunk_sha256"],f"ANCHOR_REPLAY_FAIL fold {fold}: initial_trunk_sha256"
            assert entry_hash==expected["initial_rng_sha256"],f"ANCHOR_REPLAY_FAIL fold {fold}: initial_rng_sha256"
            if not resume:
                anchor=dict(fold=fold,entry_rng=entry,post_rng=rng_state(),model=states(agent),
                            mp=copy.deepcopy(agent.optimizer_MP0.state_dict()),mf=copy.deepcopy(agent.optimizer_MF.state_dict()),
                            initial_trunk_sha256=initial_hash,initial_rng_sha256=entry_hash,
                            signature=signature,fold_hash=hbytes(np.concatenate([tr,te]).tobytes()))
                if anchor_path(fold).exists():
                    assert equal(load(anchor_path(fold))["model"],anchor["model"])
                else:save(anchor_path(fold),anchor)
        else:
            assert initial_hash==anchor["initial_trunk_sha256"] and entry_hash==anchor["initial_rng_sha256"]
            assert equal(states(agent),anchor["model"]),"paired initial model state differs"
            assert rng_hash(rng_state())==rng_hash(anchor["post_rng"]),"paired post-construction RNG differs"
            agent.optimizer_MP0.load_state_dict(anchor["mp"]);agent.optimizer_MF.load_state_dict(anchor["mf"])
            attach(agent,group,fold)
            assert trunk_hash(agent)==initial_hash and rng_hash(rng_state())==rng_hash(anchor["post_rng"])
        if resume:
            set_states(agent,latest["model"]);agent.optimizer_MP0.load_state_dict(latest["mp"]);agent.optimizer_MF.load_state_dict(latest["mf"])
            restore_rng(latest["rng"])
            best_state,best_acc,best_epoch,wait=latest["best_state"],latest["best_acc"],latest["best_epoch"],latest["wait"]
            epoch_start=latest["epoch"]+1;tracker=latest["tracker"];cursor=latest["cursor"];elapsed=latest["fold_elapsed"]
            initial_hash=latest["initial_trunk_sha256"];entry_hash=latest["initial_rng_sha256"]
        else:
            best_state,best_acc,best_epoch,wait=None,0.,-1,0
            epoch_start,cursor,elapsed=0,0,0.;tracker={"probes":0,"probe_unique":[],"eligible_probes":0,"delete_probes":0}
        latest=None;fold_started=time.monotonic();last_epoch=epoch_start-1
        order=np.random.default_rng(400004+fold).permutation(tr)
        for epoch in range(epoch_start,450):
            if group=="B0":
                with contextlib.redirect_stdout(io.StringIO()):agent.run_epoch("simple-2")
                audit={"probes":0}
            else:audit,cursor=step(agent,group,fold,epoch,order,cursor)
            for name in ("probes","eligible_probes","delete_probes"):
                tracker[name]=tracker.get(name,0)+int(audit.get(name,0))
            tracker["probe_unique"]=sorted(set(tracker["probe_unique"])|set(audit.get("unique_ids",[])))
            for name,value in audit.items():
                if "gradient" in name:tracker[name+"_max"]=max(float(value),tracker.get(name+"_max",0.))
            output=evaluate(agent)[0]
            correct=int(output[te].argmax(1).eq(agent.targ[te]).double().sum().item())
            acc=correct/len(te)
            if acc>best_acc:best_state,best_acc,best_epoch,wait=states(agent),acc,epoch,0
            else:wait+=1
            last_epoch=epoch
            if (epoch+1)%10==0 or wait>200 or stop[0] or epoch==449:
                current["seconds"]=previous_seconds+time.monotonic()-start
                save(checkpoint_path(group),dict(phase="EPOCH",signature=signature,fold=fold,epoch=epoch,
                     model=states(agent),mp=agent.optimizer_MP0.state_dict(),mf=agent.optimizer_MF.state_dict(),rng=rng_state(),
                     best_state=best_state,best_acc=best_acc,best_epoch=best_epoch,wait=wait,tracker=tracker,cursor=cursor,
                     initial_trunk_sha256=initial_hash,initial_rng_sha256=entry_hash,fold_elapsed=elapsed+time.monotonic()-fold_started,task=current))
            if (epoch+1)%100==0:print("PROGRESS",group,fold,epoch+1,"best",best_epoch,best_acc,flush=True)
            if wait>200 or stop[0]:break
        if stop[0]:print("SAFE_STOP",group,fold,last_epoch,flush=True);return
        assert best_state is not None
        continuation=rng_state();set_states(agent,best_state)
        output=evaluate(agent)[0];metrics=metric(output,te)
        assert metrics["acc"]==best_acc
        best_path=checkpoint_best_path(group,fold)
        save(best_path,dict(model=best_state,group=group,fold=fold,best_epoch=best_epoch,signature=signature,
                            input_sha=manifest["source_audit"]["read_only_sources"][str(ROOT/"data/ABIDE/processed_standard_data.csv")],
                            prompt_sha=manifest["prompt_sha"],clip_sha=manifest["text_cache_sha"],fold_hash=hbytes(np.concatenate([tr,te]).tobytes())))
        verify=load(best_path);assert equal(verify["model"],best_state)
        row=dict(group=group,name=g["name"],seed=0,fold=fold,physical_gpu=g["gpu"],**metrics,
                 best_epoch=best_epoch,epochs_executed=last_epoch+1,initial_trunk_sha256=initial_hash,initial_rng_sha256=entry_hash,
                 anchor_sha=sha(anchor_path(fold)),start_trunk_sha=initial_hash,start_rng_sha=entry_hash,
                 added_branch_seed="" if group=="B0" else 400000+fold,
                 best_checkpoint=str(best_path),best_checkpoint_sha256=sha(best_path),fold_seconds=0.)
        if group=="B0":
            differences=compare_anchor(row)
            assert not differences,f"ANCHOR_REPLAY_FAIL fold {fold}: {differences}"
        diag=fold_diagnostic(agent,group,fold,tracker)
        row["fold_seconds"]=elapsed+time.monotonic()-fold_started
        current["metrics"].append(row);current["diagnostics"].append(diag)
        restore_rng(continuation)
        current["seconds"]=previous_seconds+time.monotonic()-start
        current["status"]="COMPLETE" if fold==10 else "RUNNING"
        save(checkpoint_path(group),dict(phase="BOUNDARY",signature=signature,fold=fold+1,rng=continuation,task=current))
        json_write(task_path(group),current)
        print("FOLD_DONE",group,fold,"ACC",metrics["acc"],"AUC",metrics["hard_auc"],"best",best_epoch,"seconds",round(row["fold_seconds"],2),flush=True)
        del agent,output,best_state;torch.cuda.empty_cache()
    print("TASK_COMPLETE",group,current["seconds"],flush=True)


def refresh():
    rows,diagnostics,tasks=[],[],{}
    for group in GROUPS:
        path=task_path(group)
        if not path.exists():tasks[group]=dict(status="PENDING",folds=0);continue
        task=read_json(path)
        rows.extend(task["metrics"]);diagnostics.extend(task["diagnostics"])
        tasks[group]=dict(status=task["status"],folds=len(task["metrics"]),seconds=task["seconds"],resumes=task["resumes"])
    assert len({(r["group"],int(r["seed"]),int(r["fold"])) for r in rows})==len(rows)
    rows.sort(key=lambda r:(list(GROUPS).index(r["group"]),int(r["fold"])))
    diagnostics.sort(key=lambda r:(list(GROUPS).index(r["group"]),int(r["fold"])))
    csv_write(RESULTS/"fold_metrics.csv",rows);csv_write(RESULTS/"diagnostics.csv",diagnostics)
    m=read_json(HERE/"manifest.json");m["completed_folds"]=len(rows);m["tasks"]=tasks
    m["gpu_completed"]={str(gpu):sum(int(r["physical_gpu"])==gpu for r in rows) for gpu in (0,1)}
    m["refreshed"]=now()
    json_write(HERE/"manifest.json",m)
    return rows,diagnostics,m


def worker_environment(gpu):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID",PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
    return env


def child(command,gpu):
    path=RT/"logs"/(command.replace(" ","_")+".log")
    with path.open("ab") as log:
        process=subprocess.Popen([sys.executable,str(HERE/"run.py")]+command.split(),cwd=HERE,
                                 env=worker_environment(gpu),stdout=log,stderr=subprocess.STDOUT)
    return process


def controller():
    manifest_path=HERE/"manifest.json"
    if not manifest_path.exists():
        p=child("prepare",1);assert p.wait()==0,"prepare failed; see .runtime/logs/prepare.log"
    m=read_json(manifest_path)
    if m["preflight"]!="PASS":
        p=child("preflight",1);assert p.wait()==0,"NO-GO-PREFLIGHT: see preflight.log"
    m=read_json(manifest_path)
    if not m.get("formal_started"):
        m["formal_started"]=now();m["formal_started_unix"]=time.time();m["code_sha"]=code_hashes();json_write(manifest_path,m)
    if m["anchor"]!="ANCHOR_REPLAY_PASS":
        p=child("worker --group B0",0)
        while p.poll() is None:
            time.sleep(15)
            rows,_,_=refresh();print("ANCHOR_PROGRESS",sum(r["group"]=="B0" for r in rows),"/10",flush=True)
        assert p.returncode==0,"ANCHOR_REPLAY_FAIL: see B0 worker log"
        task=read_json(task_path("B0"));assert task["status"]=="COMPLETE" and len(task["metrics"])==10
        differences=[dict(fold=r["fold"],differences=compare_anchor(r)) for r in task["metrics"]]
        assert not any(r["differences"] for r in differences),"ANCHOR_REPLAY_FAIL"
        m=read_json(manifest_path);m["anchor"]="ANCHOR_REPLAY_PASS";m["anchor_audit"]=differences
        m["anchor_sha"]={str(f):sha(anchor_path(f)) for f in range(1,11)};json_write(manifest_path,m)
        print("ANCHOR_REPLAY_PASS 10/10",flush=True)
    pending={gpu:[g for g in GROUPS if g!="B0" and GROUPS[g]["gpu"]==gpu and not(task_path(g).exists() and read_json(task_path(g))["status"]=="COMPLETE")] for gpu in (0,1)}
    active={}
    while any(pending.values()) or active:
        for gpu in (0,1):
            while pending[gpu] and sum(v[1]==gpu for v in active.values())<4:
                group=pending[gpu].pop(0);active[group]=(child("worker --group "+group,gpu),gpu)
                print("START",group,"GPU",gpu,flush=True)
        time.sleep(15)
        for group,(process,gpu) in list(active.items()):
            code=process.poll()
            if code is None:continue
            if code:
                for other,_ in active.values():
                    if other.poll() is None:other.terminate()
                for other,_ in active.values():
                    try:other.wait(timeout=45)
                    except subprocess.TimeoutExpired:pass
                raise RuntimeError(f"FORMAL_WORKER_FAIL {group}: exit {code}; other workers safely stopped")
            print("FINISH",group,flush=True);del active[group]
        rows,diags,m=refresh()
        print("FORMAL_PROGRESS",len(rows),"/100",m["gpu_completed"],flush=True)
    rows,diags,m=refresh()
    assert len(rows)==100 and len(diags)==100 and all(t["folds"]==10 and t["status"]=="COMPLETE" for t in m["tasks"].values())
    assert m["gpu_completed"]=={"0":50,"1":50}
    m["scientific"]="COMPLETE";m["formal_completed"]=now();m["formal_wall_seconds"]=time.time()-m["formal_started_unix"]
    json_write(manifest_path,m)
    report(finalize=True)


def paired(rows,left,right):
    la={int(r["fold"]):r for r in rows if r["group"]==left}
    ra={int(r["fold"]):r for r in rows if r["group"]==right}
    if len(la)!=10 or len(ra)!=10:return None
    result={}
    for key in ("acc","hard_auc","spe_td","sen_asd"):
        diffs=[100*(float(la[f][key])-float(ra[f][key])) for f in range(1,11)]
        result[key]=dict(delta=float(np.mean(diffs)),fold_deltas=diffs,
                         wtl=[sum(x>1e-9 for x in diffs),sum(abs(x)<=1e-9 for x in diffs),sum(x< -1e-9 for x in diffs)])
    result["net_correct"]=sum(int(r["asd_tp"])+int(r["td_tn"]) for r in la.values())-sum(int(r["asd_tp"])+int(r["td_tn"]) for r in ra.values())
    result["FN_change"]=sum(int(r["asd_fn"]) for r in la.values())-sum(int(r["asd_fn"]) for r in ra.values())
    result["FP_change"]=sum(int(r["td_fp"]) for r in la.values())-sum(int(r["td_fp"]) for r in ra.values())
    return result


def safe_handoff(text):
    pending=HERE/"handoff_append_pending.md"
    pending.write_text(text)
    root=ROOT/"handoff.md"
    prefix=root.read_bytes();before=dict(bytes=len(prefix),sha256=hbytes(prefix))
    (RT/"handoff_prefix_backup.md").write_bytes(prefix)
    protection=read_json(RT/"handoff_prefix.json")
    assert hbytes(prefix[:protection["bytes"]])==protection["sha256"],"root historical handoff prefix changed"
    try:
        fd=os.open(root,os.O_WRONLY|os.O_APPEND)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            live=root.read_bytes()
            if text.encode() in live:
                pending.unlink();return dict(status="ALREADY_APPENDED",before=before)
            assert live==prefix,"handoff changed concurrently before append"
            value=text.encode();written=os.write(fd,value)
            assert written==len(value),"short handoff append"
            os.fsync(fd)
        finally:os.close(fd)
        after=root.read_bytes()
        assert hbytes(after[:len(prefix)])==before["sha256"] and after[len(prefix):]==value
        pending.unlink()
        return dict(status="APPENDED_PREFIX_SHA_VERIFIED",before=before,after_sha=hbytes(after))
    except (OSError,AssertionError) as exc:
        after=root.read_bytes()
        return dict(status="FAILED_PENDING",error=repr(exc),before=before,
                    historical_prefix_preserved=hbytes(after[:len(prefix)])==before["sha256"],pending=str(pending))


def synchronize():
    target=ROOT/"experiment"/HERE.name
    try:
        if not target.exists(): target.mkdir()
        # One bounded copy pass; symlink CLIP remains a read-only root link.
        shutil.copytree(HERE,target,dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns("__pycache__","*.lock"))
        files=[p for p in HERE.rglob("*") if p.is_file() and not p.is_symlink() and ".lock" not in p.name and "__pycache__" not in p.parts]
        assert all(sha(p)==sha(target/p.relative_to(HERE)) for p in files)
        return dict(status="SYNCED_SHA_VERIFIED",path=str(target),files=len(files))
    except (OSError,AssertionError) as exc:
        return dict(status="PARTIAL_TMP_ONLY",path=str(HERE),target=str(target),error=repr(exc))


def report(finalize=False):
    rows,diags,m=refresh();aggregates={};pairs={}
    for group in GROUPS:
        rs=[r for r in rows if r["group"]==group]
        if len(rs)!=10:continue
        aggregates[group]={key:dict(mean=100*float(np.mean([float(r[key]) for r in rs])),sd=100*float(np.std([float(r[key]) for r in rs],ddof=0))) for key in ("acc","hard_auc","spe_td","sen_asd")}
    for group in GROUPS:
        if group!="B0":pairs[group+"-B0"]=paired(rows,group,"B0")
        if group!="E5":pairs["E5-"+group]=paired(rows,"E5",group)
    decision=m.get("failure_status","PENDING")
    gates={};e5diag=[r for r in diags if r["group"]=="E5"]
    if len(rows)==100:
        p=pairs["E5-B0"]
        gates["classification"]=p["acc"]["delta"]>.23 and p["hard_auc"]["delta"]>.23 and p["sen_asd"]["delta"]>=-.5
        gates["paired_consistency"]=all(v["wtl"][0]>=5 and sum(v["wtl"][:2])>=7 for v in [p["acc"],p["hard_auc"]])
        gates["semantics"]=all(pairs["E5-"+g][k]["delta"]>0 for g in ("E1","E2","E6","E7","E8") for k in ("acc","hard_auc"))
        for comparator in ("random_delete","topz_delete"):
            diffs=[float(r["selected_delete_logp_drop"])-float(r[comparator+"_logp_drop"]) for r in e5diag if r.get("selected_delete_logp_drop","")!=""]
            gates["fidelity_vs_"+comparator]=len(diffs)==10 and np.mean(diffs)>0 and sum(v>0 for v in diffs)>=7
        decision="PROVISIONAL_GO" if all(gates.values()) else " / ".join(s for s,bad in [("NO-GO-CLASSIFICATION",not gates["classification"] or not gates["paired_consistency"]),("NO-GO-SEMANTICS",not gates["semantics"]),("NO-GO-EVIDENCE-SPECIFICITY",not gates["fidelity_vs_random_delete"] or not gates["fidelity_vs_topz_delete"])] if bad)
    text=["# Experiment 40 — HCER-v2", "",f"科研状态：**{m['scientific']}**；锚点：**{m['anchor']}**；科研判定：**{decision}**。",
          f"正式折数 {len(rows)}/100；GPU0={m['gpu_completed']['0']}，GPU1={m['gpu_completed']['1']}。交付：{m['delivery']}。",
          f"正式墙钟秒：{m.get('formal_wall_seconds','未完成')}；总执行墙钟秒：{time.time()-m['started_unix']:.2f}。",
          "", "## 四项正式指标", "", "十折mean±population SD，单位%；仅完整十折组显示均值。不是全队列micro均值。",
          "", "|组别|说明|ACC|Hard-AUC|SPE-TD|SEN-ASD|ΔACC vs B0|", "|---|---|---:|---:|---:|---:|---:|"]
    for group,g in GROUPS.items():
        a=aggregates.get(group)
        if a:
            vals=[f"{a[k]['mean']:.4f} ± {a[k]['sd']:.4f}" for k in ("acc","hard_auc","spe_td","sen_asd")]
            delta="0.0000" if group=="B0" else f"{pairs[group+'-B0']['acc']['delta']:+.4f}"
        else:vals=["未完成"]*4;delta="—"
        text.append("|"+"|".join([group,g["name"]]+vals+[delta])+"|")
    text += ["", "## B0-A逐折回放", "", "|fold|本次TP/FN/FP/TN|历史TP/FN/FP/TN|best epoch 本次/历史|执行epoch 本次/历史|差异|", "|---|---|---|---|---|---|"]
    history=historical_rows()
    for row in rows:
        if row["group"]!="B0":continue
        hist=history[int(row["fold"])]
        values=lambda r:"/".join(str(r[k]) for k in ("asd_tp","asd_fn","td_fp","td_tn"))
        text.append(f"|{row['fold']}|{values(row)}|{values(hist)}|{row['best_epoch']}/{hist['best_epoch']}|{row['epochs_executed']}/{hist['epochs_executed']}|{compare_anchor(row) or 'EXACT，包括初始参数和入口RNG SHA'}|")
    text += ["", "## 逐fold配对", "", "百分点差；W/T/L来自相同fold，训练集重叠，不是10个独立外部数据集。完整逐fold差保存于manifest配对字段。", "", "|比较|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|净正确数|FN变化|FP变化|", "|---|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    for name,p in pairs.items():
        if p is None:continue
        text.append(f"|{name}|{p['acc']['delta']:+.4f}|{'/'.join(map(str,p['acc']['wtl']))}|{p['hard_auc']['delta']:+.4f}|{'/'.join(map(str,p['hard_auc']['wtl']))}|{p['spe_td']['delta']:+.4f}|{p['sen_asd']['delta']:+.4f}|{p['net_correct']}|{p['FN_change']}|{p['FP_change']}|")
    text += ["", "## 实现与机制", "", "修复矩阵、参考、文本Query/解剖Key、NULL阈值、同幅残差与CF梯度策略见README/config。所有x/z/mu/sigma/反事实列通过同一canonical_edges；E7只改query membership。B0保留未调用PEN构造以复演E32初始化，PEN无前向。", "",f"预检：{m['preflight']}；详细逐值、梯度、恢复与计时证据保存在`.runtime/preflight.json`。"]
    if e5diag:
        for key in ("residual_mean","residual_median","residual_p95","residual_max","null_mean","selection_mean","empty_fraction","swap_abs_logp","swap_class_flip","disabled_abs_logp","disabled_class_flip","selected_delete_logp_drop","selected_keep_logp_drop","random_delete_logp_drop","topz_delete_logp_drop","frozen_selected_delete_logp_drop","frozen_topz_delete_logp_drop","fixed_residual_selected_delete_logp_drop","probes","unique_probe_count","raw_pair_cos_mean","projected_pair_cos_mean"):
            vals=[float(r[key]) for r in e5diag if r.get(key,"")!=""]
            if vals:text.append(f"- E5 {key}: fold平均 {np.mean(vals):.8f}，范围 [{min(vals):.8f}, {max(vals):.8f}]。")
        other=[r for r in diags if r["group"]=="E9"]
        if other:text.append("\nE5/E9实际残差比较："+json.dumps({g:{key:float(np.mean([float(r[key]) for r in ds])) for key in ("residual_mean","residual_median","residual_p95")} for g,ds in [("E5",e5diag),("E9",other)]},ensure_ascii=False))
    text += ["", "## 结论与限制", "", f"预注册筛选条件：`{json.dumps(gates,ensure_ascii=False)}`。", "正确医学文本只有同时胜过数值Top-|z|、Gram匹配匿名、中性解剖、错误配对及类别交换，且selected删边稳定胜过同预算Top-|z|时，才支持继续研究；内部权重变化或梯度非零本身不证明语义有效。", "val=test选checkpoint、全队列预处理/RFE、transductive图、单seed与重叠训练fold均保留；指标只用于历史协议内部比较。FC遮蔽反映模型依赖，可能受分布偏移及图重构影响，不是神经病理因果。", "", "## 来源与交付", "", "[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[ERASER](https://aclanthology.org/2020.acl-main.408/)、[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)、[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)。实际查阅路径与SHA完整记录在manifest/source_audit；未依赖实验39。",f"\n真实暂存路径：`{HERE}`。正式目标：`{ROOT/'experiment'/HERE.name}`。/tmp为临时存储。",f"\n故障：{json.dumps(m.get('failure',{}),ensure_ascii=False)}。交付细节：{json.dumps(m.get('delivery_details',{}),ensure_ascii=False)}；handoff：{json.dumps(m.get('handoff',{}),ensure_ascii=False)}。"]
    report_text="\n".join(text)+"\n"
    (RESULTS/"report.md").write_text(report_text)
    m.update(summary=aggregates,paired=pairs,decision=decision,go_criteria=gates,result_sha={p.name:sha(p) for p in (RESULTS/"fold_metrics.csv",RESULTS/"diagnostics.csv",RESULTS/"report.md")})
    json_write(HERE/"manifest.json",m)
    if finalize:
        verify_protected()
        handoff="\n\n## Experiment 40 — HCER-v2\n\n"+report_text.replace("# Experiment 40 — HCER-v2\n", "",1)
        handoff_result=safe_handoff(handoff)
        m=read_json(HERE/"manifest.json");m["handoff"]=handoff_result;json_write(HERE/"manifest.json",m)
        delivery=synchronize()
        m=read_json(HERE/"manifest.json");m["delivery_details"]=delivery;m["delivery"]=delivery["status"];json_write(HERE/"manifest.json",m)
        report(finalize=False)
    print("REPORT",m["scientific"],m["anchor"],len(rows),decision,m["gpu_completed"],flush=True)
    return m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","preflight","all","resume","worker","report"])
    parser.add_argument("--group",choices=list(GROUPS));args=parser.parse_args()
    try:
        if args.command=="prepare":prepare()
        elif args.command=="preflight":preflight()
        elif args.command in ("all","resume"):controller()
        elif args.command=="worker":worker(args.group)
        else:report()
    except Exception as exc:
        failure=dict(command=args.command,group=args.group,error=repr(exc),traceback=traceback.format_exc(),time=now())
        print(failure["traceback"],flush=True)
        json_write(RT/"failure.json",failure)
        if (HERE/"manifest.json").exists() and args.command!="worker":
            m=read_json(HERE/"manifest.json");m["scientific"]="FAIL";m["failure"]=failure
            status="ANCHOR_REPLAY_FAIL" if "ANCHOR" in str(exc) else "NO-GO-PREFLIGHT" if args.command in ("prepare","preflight") else "FORMAL_FAIL"
            m["failure_status"]=status
            if "ANCHOR" in status:m["anchor"]=status
            json_write(HERE/"manifest.json",m)
        raise


if __name__=="__main__":main()
