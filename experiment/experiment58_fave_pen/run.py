#!/usr/bin/env python3
"""Experiment 58 runner. Historical model/data sources are read-only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/adodas/SPromptGL-main")
BASE_EXP = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
BASE_GROUP = "CADR_WD_0p01_GFR_WD_0p11"
FULL = "E58-FULL"
NUM = "E58-NUM"
RUNTIME = EXP / ".runtime"
RESULTS = EXP / "results"
CKPT_ROOT = RUNTIME / "checkpoints"
LOG_ROOT = RUNTIME / "logs"
TASK_ROOT = RUNTIME / "tasks"
CONFIG_PATH = EXP / "config.json"
PROMPTS_PATH = EXP / "prompts.json"
RN50_PATH = Path("/tmp/e58_models/RN50.pt")
RN50_SHA = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"

sys.path.insert(0, str(BASE_EXP))
import run_cadr_fallback_wd_sweep as W  # noqa: E402
from fave_pen import FAVEPEN, optimizer_for_fave  # noqa: E402

R = W.R
R.ROOT = DATA_ROOT
R.DATA = None
R.FOLDS = None
METRICS = tuple(W.METRICS)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode())


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        R.runtime()["torch"].save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def task_path(group: str, fold: int) -> Path:
    return TASK_ROOT / group / f"{group}_f{fold:02d}.json"


def checkpoint_path(group: str, fold: int, latest: bool = False) -> Path:
    suffix = "latest" if latest else "best"
    return CKPT_ROOT / group / f"{group}_f{fold:02d}.{suffix}.pt"


def install_group(group: str):
    if group not in {FULL, NUM}:
        raise KeyError(group)
    W.RUNTIME = RUNTIME / "group_runs"
    W.RESULTS = RESULTS
    W.install_group(BASE_GROUP)
    state_builder = R.build_state

    def build_state(fold, cache_meta=None):
        state = state_builder(fold, cache_meta)
        state["group"] = group
        assert [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups] == [0.11]
        assert [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups] == [0.11]
        assert [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups] == [0.01, 0.01]
        return state

    R.build_state = build_state
    R.task_path = lambda fold: task_path(group, fold)
    R.best_path = lambda fold: checkpoint_path(group, fold)
    R.latest_path = lambda fold: checkpoint_path(group, fold, latest=True)
    return state_builder


def load_references():
    W.RUNTIME = BASE_EXP / ".runtime/fallback_wd_sweep"
    W.RESULTS = BASE_EXP / "results/fallback_wd_sweep"
    source_manifest, reference_53_2, reference_53_3 = W.load_references()
    assert len(reference_53_3) == 10
    return source_manifest, reference_53_2, reference_53_3


def prompt_payload():
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def encode_prompt_bank():
    import torch

    payload = prompt_payload()
    prompt = payload["hypotheses"][0]
    texts = [
        prompt["full_text"],
        prompt["counterfactual_text"],
        payload["intervention_only_negative_control"]["text"],
        payload["intervention_only_negative_control"]["counterfactual_text"],
    ]
    text_hash = sha_json(texts)
    cache_path = RUNTIME / "prompt_embeddings.pt"
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if saved.get("text_sha256") == text_hash and saved.get("rn50_sha256") == RN50_SHA:
            return saved
    if not RN50_PATH.is_file() or sha_file(RN50_PATH) != RN50_SHA:
        raise RuntimeError("official OpenAI CLIP RN50 weight is missing or SHA256 mismatched")
    sys.path.insert(0, str(EXP / ".runtime/deps"))
    sys.path.insert(0, str(DATA_ROOT / "CLIP"))
    import clip

    model, _ = clip.load(str(RN50_PATH), device="cuda:0", jit=False)
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(texts).to("cuda:0")
        embedding = model.encode_text(tokens).float().cpu()
    del model, tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    saved = {
        "embeddings": embedding,
        "texts": texts,
        "text_sha256": text_hash,
        "rn50_sha256": RN50_SHA,
        "embedding_sha256": sha_bytes(embedding.contiguous().numpy().tobytes()),
        "source_url": "https://huggingface.co/xycheni/clip_RN50/resolve/main/RN50.pt",
        "official_openai_sha256_verified": True,
    }
    atomic_torch_save(cache_path, saved)
    return saved


def numeric_text_embeddings(dim: int):
    import numpy as np
    import torch

    generator = np.random.default_rng(580058)
    value = generator.standard_normal((4, dim)).astype(np.float32)
    value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(value), sha_bytes(value.tobytes())


def get_texts(group: str, prompt_cache):
    if group == FULL:
        return prompt_cache["embeddings"][:2].clone(), prompt_cache["embeddings"][2:4].clone()
    anon, anon_sha = numeric_text_embeddings(int(prompt_cache["embeddings"].shape[1]))
    return anon[:2].clone(), anon[2:4].clone()


def _covariate_indices():
    import numpy as np

    mapping = np.load(DATA_ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    pheno = [str(v) for v in mapping["PHENO"]]
    age = [i for i, name in enumerate(pheno) if name.startswith("AGE_AT_SCAN_")]
    site = [i for i, name in enumerate(pheno) if name.startswith("SITE_ID_")]
    sex = [i for i, name in enumerate(pheno) if name.startswith("SEX_")]
    if (len(age), len(site), len(sex)) != (26, 20, 2):
        raise RuntimeError(f"unexpected PHENO field schema {(len(age), len(site), len(sex))}")
    selected = age[:-1] + site[:-1] + sex[:-1]
    return selected, {"age_bins": age, "sites": site, "sex": sex}


def build_reference_context(state, fold: int):
    """Fit training-TD conditional means and train-TD-only cross-fitted scales."""
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    parts = [part.detach().cpu().numpy().astype(np.float64) for part in state["parts"]]
    pheno, anat, func, corr = parts
    labels = state["y"].detach().cpu().numpy()
    train_idx = state["train"].detach().cpu().numpy().astype(np.int64)
    test_idx = np.asarray(state["test"], dtype=np.int64)
    td_train = train_idx[labels[train_idx] == 1]
    asd_train = train_idx[labels[train_idx] == 0]
    if len(td_train) < 10:
        raise RuntimeError("too few outer-train TD subjects for patient-disjoint reference cross-fitting")
    cov_idx, cov_groups = _covariate_indices()
    cov = pheno[:, cov_idx]

    oof_pred = np.full((len(td_train), corr.shape[1]), np.nan, dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=580000 + fold)
    for inner_train, inner_valid in splitter.split(td_train):
        mdl = Ridge(alpha=5.0)
        mdl.fit(cov[td_train[inner_train]], corr[td_train[inner_train]])
        oof_pred[inner_valid] = mdl.predict(cov[td_train[inner_valid]])
    if not np.isfinite(oof_pred).all():
        raise RuntimeError("incomplete TD reference cross-fitting")
    td_residual = corr[td_train] - oof_pred
    variance = td_residual.var(axis=0, ddof=1)
    global_variance = float(np.median(variance))
    sigma = np.sqrt(0.9 * variance + 0.1 * global_variance)
    sigma = np.maximum(sigma, 0.05)

    final = Ridge(alpha=5.0)
    final.fit(cov[td_train], corr[td_train])
    z = (corr - final.predict(cov)) / sigma[None, :]
    # Training TD rows use out-of-fold residuals, never in-fit residuals.
    td_positions = {int(patient): i for i, patient in enumerate(td_train.tolist())}
    for patient, pos in td_positions.items():
        z[patient] = td_residual[pos] / sigma
    if not np.isfinite(z).all():
        raise FloatingPointError("nonfinite normative z scores")
    fit_scope = np.concatenate((td_train,))
    context = {
        "z": torch.as_tensor(z, dtype=torch.float32, device=state["x"].device),
        "quality": torch.cat((state["parts"][1], state["parts"][2]), dim=1),
        "train_idx": torch.as_tensor(train_idx, dtype=torch.long, device=state["x"].device),
        "test_idx": test_idx,
        "train_td_idx": td_train,
        "train_asd_idx": asd_train,
        "sigma": sigma,
        "oof_td_residual": td_residual,
        "coefficient": final.coef_,
        "intercept": final.intercept_,
        "covariate_indices": cov_idx,
        "covariate_groups": cov_groups,
        "reference_scope": fit_scope.tolist(),
        "z_sha256": sha_bytes(np.asarray(z, dtype=np.float32).tobytes()),
        "sigma_sha256": sha_bytes(np.asarray(sigma, dtype=np.float64).tobytes()),
        "inner_split_sha256": sha_bytes(np.asarray(td_train, dtype=np.int64).tobytes() + np.asarray(list(splitter.split(td_train))[0][0], dtype=np.int64).tobytes()),
        "td_train_n": int(len(td_train)),
        "asd_train_n": int(len(asd_train)),
        "test_n": int(len(test_idx)),
        "z_gt_2p3_train_td": int(np.sum(z[td_train, 0] >= 2.3)),
        "z_lt_neg2p3_train_td": int(np.sum(z[td_train, 0] <= -2.3)),
        "z_gt_2p3_train_asd": int(np.sum(z[asd_train, 0] >= 2.3)),
        "z_lt_neg2p3_train_asd": int(np.sum(z[asd_train, 0] <= -2.3)),
    }
    cache_path = RUNTIME / "reference_cache" / f"fold{fold:02d}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp, z=z.astype(np.float32), sigma=sigma, coefficient=final.coef_,
        intercept=final.intercept_, td_train=td_train, asd_train=asd_train,
        td_residual=td_residual,
    )
    os.replace(tmp, cache_path)
    context["cache_path"] = str(cache_path)
    context["cache_sha256"] = sha_file(cache_path)
    return context


def create_fave(group: str, fold: int, prompt_cache):
    import torch

    texts, wrong_texts = get_texts(group, prompt_cache)
    device = R.runtime()["device"]
    before = R.rng_sha(R.pack_rng())
    module = FAVEPEN(texts, init_seed=580000 + fold).to(device)
    after = R.rng_sha(R.pack_rng())
    if before != after:
        raise RuntimeError("FAVE parameter initialization consumed the main RNG")
    module.eval()
    return module, wrong_texts, before


def _module_param_sha(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode() + b"\0")
        digest.update(str(tuple(parameter.shape)).encode() + b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _grad_norm(parameters):
    import torch

    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.0


def _cam_with_fave(cam, fused, fave, context, active: bool, text_override=None, z_override=None):
    import torch

    n = fused.shape[0]
    if not active:
        logp, hidden, attention = cam(fused)
        zero = fused.new_zeros(())
        return logp, hidden, attention, zero, zero, None
    x, input_attention = cam.InputLayer(fused)
    attention = input_attention.mean(dim=1)
    for encoder, feed_forward in zip(cam.Encoder, cam.FeedForward):
        x, attn_layer = encoder(q=x, k=x, v=x, modal_num=cam.modal_num)
        attention = attn_layer.mean(dim=1)
        x = feed_forward(x)
    if x.ndim != 3 or x.shape[1:] != (4, 36):
        raise RuntimeError(f"CAM position schema changed: {tuple(x.shape)}")
    h_corr, loss_match, loss_verify, info = fave(
        context["z"], context["quality"], x[:, 3, :], context["train_idx"],
        text_override=text_override, z_override=z_override,
    )
    x = torch.cat((x[:, :3, :], h_corr[:, None, :]), dim=1)
    logp, hidden = cam.Outputlayer(x.reshape(n, -1), attention.reshape(n, -1))
    return logp, hidden, [], loss_match, loss_verify, info


def forward_fave_state(state, training: bool, active: bool, text_override=None, z_override=None, fave_enabled=True):
    import torch
    import torch.nn.functional as F

    state["model"].train(training)
    state["cam"].train(training)
    state["fave"].train(training)
    fused, adjs, info = state["model"](
        state["parts"], pair_index=state["train_pairs"], use_cadr=active
    )
    use_fave = bool(active and fave_enabled)
    logp, hidden, attention, match_loss, verify_loss, fave_info = _cam_with_fave(
        state["cam"], fused, state["fave"], state["fave_context"], use_fave,
        text_override=text_override, z_override=z_override,
    )
    cls = R.runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = R._graph_loss(state["parts"], adjs)
    pair = torch.stack([
        F.binary_cross_entropy_with_logits(logits, state["train_same"])
        for logits in info["pair_logits"]
    ]).mean()
    adv, adv_info = R.conditional_adversarial_loss(
        info["pair_reprs"], state["train_same"], state["model"].discriminator,
        state["model"].grl_scale,
    )
    preserve, preserve_gaps = R.relation_preservation_loss(
        info["pair_logits"], state["train_same"], margin=0.02
    )
    legacy_total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    total = legacy_total + 0.05 * match_loss + 0.01 * verify_loss
    return {
        "logp": logp, "hidden": hidden, "attention": attention, "fused": fused,
        "adjs": adjs, "info": info, "cls": cls, "graph": graph,
        "graph_parts": graph_parts, "pair": pair, "adv": adv,
        "adv_info": adv_info, "preserve": preserve, "preserve_gaps": preserve_gaps,
        "match": match_loss, "verify": verify_loss, "legacy_total": legacy_total,
        "total": total, "fave_info": fave_info,
    }


def _tree_equal(left, right):
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def warmup_two_step_parity(prompt_cache, reference_by_fold):
    """Compare two warmup updates with and without the inserted inactive module."""
    import torch

    W.install_group(BASE_GROUP)
    historical = R.build_state(1, R.prepare_fold(1))
    integrated = build_state(FULL, 1, prompt_cache, reference_by_fold)
    if historical["initial_shared_sha"] != integrated["initial_shared_sha"]:
        raise RuntimeError("warmup parity models have different shared initialization")
    rng = historical["construction_rng"]
    rng_sha = R.rng_sha(rng)
    if rng_sha != integrated["construction_rng_sha"]:
        raise RuntimeError("warmup parity models have different construction RNG")
    traces = []
    for step in range(2):
        R.restore_rng(rng)
        for opt in (historical["opt_base"], historical["opt_cam"], historical["opt_added"]):
            opt.zero_grad(set_to_none=True)
        out_a = R.forward_state(historical, training=True, use_cadr=False)
        out_a["total"].backward()
        historical["opt_base"].step(); historical["opt_cam"].step(); historical["opt_added"].step()
        rng_a = R.pack_rng()

        R.restore_rng(rng)
        for opt in (integrated["opt_base"], integrated["opt_cam"], integrated["opt_added"], integrated["opt_fave"]):
            opt.zero_grad(set_to_none=True)
        out_b = forward_fave_state(integrated, training=True, active=False)
        out_b["total"].backward()
        integrated["opt_base"].step(); integrated["opt_cam"].step(); integrated["opt_added"].step()
        rng_b = R.pack_rng()
        if not torch.equal(out_a["logp"], out_b["logp"]) or not torch.equal(out_a["total"], out_b["total"]):
            raise RuntimeError(f"warmup forward/loss parity failed at step {step + 1}")
        for module_a, module_b in ((historical["model"], integrated["model"]), (historical["cam"], integrated["cam"])):
            if not _tree_equal(module_a.state_dict(), module_b.state_dict()):
                raise RuntimeError(f"warmup parameter update parity failed at step {step + 1}")
        for opt_a, opt_b in ((historical["opt_base"], integrated["opt_base"]), (historical["opt_cam"], integrated["opt_cam"]), (historical["opt_added"], integrated["opt_added"])):
            if not _tree_equal(opt_a.state_dict(), opt_b.state_dict()):
                raise RuntimeError(f"warmup optimizer state parity failed at step {step + 1}")
        if R.rng_sha(rng_a) != R.rng_sha(rng_b):
            raise RuntimeError(f"warmup RNG stream parity failed at step {step + 1}")
        traces.append({"step": step + 1, "rng_sha": R.rng_sha(rng_a), "logp_sha": sha_bytes(out_a["logp"].detach().cpu().numpy().tobytes())})
        rng = rng_a
    del historical, integrated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "PASS", "steps": traces, "initial_rng_sha": rng_sha}


def _baseline_audit_and_preflight(prompt_cache):
    import numpy as np
    import torch

    source_manifest, reference_53_2, reference_53_3 = load_references()
    ref_group = BASE_GROUP
    ref_runtime = BASE_EXP / ".runtime/fallback_wd_sweep" / ref_group
    ref_rows = read_rows(BASE_EXP / "results/fallback_wd_sweep/fold_metrics.csv")
    ref_by_fold = {int(row["fold"]): row for row in ref_rows if row.get("group") == ref_group}
    if len(ref_by_fold) != 10:
        raise RuntimeError("the selected historical group does not have ten fold rows")
    ref_audit_rows = read_rows(BASE_EXP / "results/fallback_wd_sweep/checkpoint_audit.csv")
    ref_audit_by_fold = {
        int(row["fold"]): row for row in ref_audit_rows
        if row.get("group") == ref_group
    }
    if len(ref_audit_by_fold) != 10 or any(
        row.get("checkpoint_audit") != "PASS" for row in ref_audit_by_fold.values()
    ):
        raise RuntimeError("the selected historical group lacks ten passing checkpoint audits")
    roi_dir = DATA_ROOT / "experiment/experiment11_roi_mapping_recovery/results"
    mapping_path = roi_dir / "roi_256_named_mapping.csv"
    mapping_sha = sha_file(mapping_path)
    roi_audit = json.loads((roi_dir / "roi_label_audit.json").read_text(encoding="utf-8"))
    if roi_audit["named_feature_map"]["usable_for_roi_semantic_prompt"] != 252:
        raise RuntimeError("ROI map audit does not verify 252 named connections")
    with mapping_path.open(newline="", encoding="utf-8") as stream:
        mapping_rows = list(csv.DictReader(stream))
    feature_zero = mapping_rows[0]
    if int(feature_zero["correlation_feature_index"]) != 0 or feature_zero["connection_name"] != "Left Thalamus — Left Superior Temporal Gyrus, anterior division":
        raise RuntimeError("feature zero ROI map does not match the frozen hypothesis")
    usable = sum(row["usable_for_roi_semantic_prompt"].lower() == "true" for row in mapping_rows)
    if usable != 252 or len(mapping_rows) != 256:
        raise RuntimeError("ROI mapping feature count mismatch")

    folds = []
    for fold in range(1, 11):
        W.install_group(ref_group)
        state = R.build_state(fold, R.prepare_fold(fold))
        reference = reference_53_2[fold]
        W.F.check_initialization(state, reference_53_2)
        row53 = reference_53_3[fold]
        for key in ("initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha", "initial_added_sha"):
            if row53.get(key):
                if state[key] != row53[key]:
                    raise RuntimeError(f"fold {fold}: initialization differs from audited 53.3 reference for {key}")
        selected_row = ref_by_fold[fold]
        selected_audit = ref_audit_by_fold[fold]
        cp = ref_runtime / "checkpoints" / f"{ref_group}_f{fold:02d}.best.pt"
        task = ref_runtime / "tasks" / f"{ref_group}_f{fold:02d}.json"
        checkpoint = R.torch_load(cp)
        checkpoint_sha = sha_file(cp)
        task_obj = json.loads(task.read_text(encoding="utf-8"))
        if task_obj.get("status") != "COMPLETE" or task_obj.get("best_sha256") != checkpoint_sha:
            raise RuntimeError(f"fold {fold}: baseline task/checkpoint SHA mismatch")
        if checkpoint.get("group") != ref_group or checkpoint.get("fold") != fold:
            raise RuntimeError(f"fold {fold}: baseline checkpoint identity mismatch")
        if selected_audit.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"fold {fold}: selected-group audit/checkpoint SHA mismatch")
        if int(selected_row["best_epoch"]) != int(checkpoint["best_epoch"]):
            raise RuntimeError(f"fold {fold}: selected-group metrics/checkpoint epoch mismatch")
        if checkpoint.get("optimizer_weight_decay", {}).get("base") != [0.11] or checkpoint.get("optimizer_weight_decay", {}).get("cam") != [0.11]:
            raise RuntimeError(f"fold {fold}: baseline GFR/CAM weight decay mismatch")
        if checkpoint.get("optimizer_weight_decay", {}).get("added") != [0.01, 0.01]:
            raise RuntimeError(f"fold {fold}: baseline CADR weight decay mismatch")
        context = build_reference_context(state, fold)
        folds.append({
            "fold": fold,
            "split_sha256": state["split_sha"],
            "split_sha": state["split_sha"],
            "anchor_sha256": state["anchor_sha"],
            "initial_shared_sha256": state["initial_shared_sha"],
            "initial_gfr_sha256": state["initial_gfr_sha"],
            "initial_cam_sha256": state["initial_cam_sha"],
            "initial_cadr_added_sha256": state["initial_added_sha"],
            "initial_rng_sha256": state["construction_rng_sha"],
            "patient_pair_sha256": state["cache_meta"]["pair_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"],
            "relation_cache_sha256": state["cache_meta"]["npz_sha256"],
            "reference_checkpoint": str(cp),
            "reference_checkpoint_sha256": checkpoint_sha,
            "reference_best_epoch": int(checkpoint["best_epoch"]),
            "reference_metrics": {name: float(selected_row[name]) for name in METRICS},
            "reference_metrics_source_group": ref_group,
            "reference_metrics_source_csv_sha256": sha_file(BASE_EXP / "results/fallback_wd_sweep/fold_metrics.csv"),
            "reference_checkpoint_audit": selected_audit.get("checkpoint_audit"),
            "reference_config_sha256": sha_file(BASE_EXP / "config_fallback_wd_sweep.json"),
            "reference_source_sha256": sha_file(BASE_EXP / "run_cadr_fallback_wd_sweep.py"),
            "normative_reference": {
                "td_train_n": context["td_train_n"], "asd_train_n": context["asd_train_n"],
                "test_n": context["test_n"], "sigma_feature0": float(context["sigma"][0]),
                "train_td_z_gt_2p3": context["z_gt_2p3_train_td"],
                "train_td_z_lt_neg2p3": context["z_lt_neg2p3_train_td"],
                "train_asd_z_gt_2p3": context["z_gt_2p3_train_asd"],
                "train_asd_z_lt_neg2p3": context["z_lt_neg2p3_train_asd"],
                "cache_sha256": context["cache_sha256"],
                "z_sha256": context["z_sha256"],
                "sigma_sha256": context["sigma_sha256"],
                "covariate_indices": context["covariate_indices"],
            },
        })
        del state, context, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for the frozen CAM implementation")
    install_group(FULL)
    state = R.build_state(1, R.prepare_fold(1))
    prompt_full, prompt_wrong = get_texts(FULL, prompt_cache)
    fave, _, init_rng_sha = create_fave(FULL, 1, prompt_cache)
    state["fave"] = fave
    state["fave_context"] = build_reference_context(state, 1)
    state["opt_fave"] = optimizer_for_fave(fave)

    # Historical direct forwards, including the original 10-epoch warmup.
    state["model"].eval(); state["cam"].eval(); fave.eval()
    with torch.no_grad():
        base_active = R.forward_state(state, training=False, use_cadr=True)
        full_zero = forward_fave_state(state, training=False, active=True)
        base_warmup = R.forward_state(state, training=False, use_cadr=False)
        fave_warmup = forward_fave_state(state, training=False, active=False)
    for key in ("fused", "logp"):
        if not torch.equal(base_active[key], full_zero[key]):
            raise RuntimeError(f"zero-residual active FAVE path failed exact parity for {key}")
        if not torch.equal(base_warmup[key], fave_warmup[key]):
            raise RuntimeError(f"FAVE warmup path failed exact parity for {key}")
    if init_rng_sha != state["construction_rng_sha"]:
        raise RuntimeError("FAVE initialization did not preserve base construction RNG")
    if prompt_full.shape[0] != 2 or not torch.isfinite(prompt_full).all() or torch.any(prompt_full.norm(dim=1) == 0):
        raise RuntimeError("CLIP prompt embedding shape or values are invalid")

    # Test-label isolation: flip only outer-test labels and compare all predictions.
    saved_y = state["y"].clone()
    state["y"][torch.as_tensor(state["test"], device=state["y"].device)] = 1 - state["y"][torch.as_tensor(state["test"], device=state["y"].device)]
    with torch.no_grad():
        permuted = forward_fave_state(state, training=False, active=True)
    if not torch.equal(full_zero["logp"], permuted["logp"]):
        raise RuntimeError("outer-test label permutation changed predictions")
    state["y"].copy_(saved_y)

    # Auxiliary losses must train semantic/evidence parameters; classification
    # must have a real path to the zero-initialized bounded residual coefficient.
    state["model"].train(); state["cam"].train(); fave.train()
    probe = forward_fave_state(state, training=True, active=True)
    g_cls = torch.autograd.grad(probe["cls"], [fave.alpha], retain_graph=True, allow_unused=True)[0]
    g_match = torch.autograd.grad(probe["match"], list(fave.semantic_projection.parameters()) + list(fave.connection_encoder.parameters()), retain_graph=True, allow_unused=True)
    if g_cls is None or not torch.isfinite(g_cls).all() or g_cls.norm() == 0:
        raise RuntimeError("classification loss has no finite nonzero path to FAVE residual strength")
    if not any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in g_match):
        raise RuntimeError("FAVE matching objective has no finite parameter gradient")
    if not torch.isfinite(probe["total"]):
        raise RuntimeError("preflight loss is nonfinite")

    mapping_hash = mapping_sha
    payload = prompt_payload()
    payload["mapping_sha256"] = mapping_hash
    payload["embedding_artifact"] = {
        "path": ".runtime/prompt_embeddings.pt",
        "text_sha256": prompt_cache["text_sha256"],
        "embedding_sha256": prompt_cache["embedding_sha256"],
        "rn50_sha256": prompt_cache["rn50_sha256"],
    }
    atomic_json(PROMPTS_PATH, payload)
    base_config = json.loads((BASE_EXP / "config_fallback_wd_sweep.json").read_text(encoding="utf-8"))
    source_files = [
        DATA_ROOT / "data/ABIDE/processed_standard_data.csv",
        DATA_ROOT / "data/ABIDE/modal_feat_dict.npy",
        mapping_path,
        roi_dir / "roi_label_audit.json",
        DATA_ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_mapping.csv",
        BASE_EXP / "run.py", BASE_EXP / "run_cadr_fallback.py",
        BASE_EXP / "run_cadr_fallback_wd_sweep.py", BASE_EXP / "cadr_gfr.py",
        BASE_EXP / "config.json", BASE_EXP / "config_fallback_wd_sweep.json",
        BASE_EXP / "audit.py",
        BASE_EXP / "results/fallback_wd_sweep/fold_metrics.csv",
        BASE_EXP / "results/fallback_wd_sweep/checkpoint_audit.csv",
        BASE_EXP / ".runtime/reference_53_2_manifest.json",
        DATA_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/index.json",
    ]
    # Same initializer seed yields identical trainable FAVE state in FULL and
    # NUM; the frozen text buffers are the only initialization difference.
    state_full = build_state(FULL, 1, prompt_cache, reference_53_3)
    state_num = build_state(NUM, 1, prompt_cache, reference_53_3)
    if state_full["initial_fave_param_sha"] != state_num["initial_fave_param_sha"]:
        raise RuntimeError("FULL and NUM FAVE trainable parameters do not match at initialization")
    if state_full["construction_rng_sha"] != state_num["construction_rng_sha"] or state_full["fave_init_rng_sha"] != state_num["fave_init_rng_sha"]:
        raise RuntimeError("FULL and NUM initial RNG states do not match")
    if [float(pg["weight_decay"]) for pg in state_full["opt_fave"].param_groups] != [1e-4, 0.0]:
        raise RuntimeError("FAVE AdamW parameter-group weight decay is incorrect")
    if sum(p.numel() for p in state_full["fave"].parameters()) != sum(p.numel() for p in state_num["fave"].parameters()):
        raise RuntimeError("FULL and NUM parameter counts differ")
    del state_full, state_num
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    warmup_parity = warmup_two_step_parity(prompt_cache, reference_53_3)

    baseline = {
        "status": "PASS",
        "base_group": BASE_GROUP,
        "base_config": base_config,
        "reference_source_manifest_status": source_manifest.get("status"),
        "data_sha256": {str(p): sha_file(p) for p in source_files[:2]},
        "roi_source_sha256": {str(p): sha_file(p) for p in source_files[2:5]},
        "historical_source_sha256": {str(p): sha_file(p) for p in source_files[5:] if p.is_file()},
        "clip": {
            "checkpoint_sha256": sha_file(RN50_PATH), "expected_official_sha256": RN50_SHA,
            "embedding_sha256": prompt_cache["embedding_sha256"],
            "text_sha256": prompt_cache["text_sha256"],
        },
        "mounts": {"requested_root": str(DATA_ROOT), "historical_root": "/model/spromptgl", "same_filesystem": False},
        "folds": folds,
        "preflight": {
            "fold1_zero_residual_active_exact_parity": True,
            "fold1_warmup_exact_parity": True,
            "test_label_permutation_invariant": True,
            "fave_init_preserves_rng": True,
            "classification_to_alpha_grad_nonzero": float(g_cls.norm().detach().cpu()),
            "matching_to_semantic_encoder_grad_nonzero": True,
            "fave_total_finite": True,
            "full_num_trainable_init_equal": True,
            "fave_optimizer_wd": [1e-4, 0.0],
            "two_step_warmup_parity": warmup_parity,
        },
        "rois": {
            "named_edges": usable, "total_edges": len(mapping_rows),
            "eligible_feature_indices": [0], "excluded_unknown_indices": [195, 225, 232, 236],
            "feature0_connection": feature_zero["connection_name"],
            "feature0_original_index": int(feature_zero["original_connection_index"]),
            "features_195_225_232_236_excluded_from_prompt_only": True,
        },
    }
    atomic_json(EXP / "baseline_audit.json", baseline)
    preflight_record = {
        "status": "PASS", "fold": 1,
        "baseline_active_zero_residual_exact": True,
        "baseline_warmup_exact": True,
        "test_label_invariant": True,
        "fave_main_rng_preserved": True,
        "fave_classification_alpha_grad_norm": float(g_cls.norm().detach().cpu()),
        "fave_match_parameter_grad_nonzero": True,
        "text_prompt_count": 2,
        "semantic_evidence_feature_count": 1,
        "eligible_feature_index": 0,
        "test_labels_accessed_by_match_loss": False,
    }
    atomic_json(RUNTIME / "preflight.json", preflight_record)
    del state, fave, probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return baseline, reference_53_3


def build_state(group: str, fold: int, prompt_cache, reference_by_fold):
    import torch

    install_group(group)
    cache_meta = R.prepare_fold(fold)
    state = R.build_state(fold, cache_meta)
    baseline_row = reference_by_fold[fold]
    for key, reference_key in (
        ("initial_shared_sha", "initial_shared_sha"),
        ("initial_gfr_sha", "initial_gfr_sha"),
        ("initial_cam_sha", "initial_cam_sha"),
        ("initial_added_sha", "initial_added_sha"),
    ):
        if baseline_row.get(reference_key) and state[key] != baseline_row[reference_key]:
            raise RuntimeError(f"fold={fold} {group}: base initialization mismatch {key}")
    if state["split_sha"] != baseline_row["split_sha"] or state["cache_meta"]["pair_sha"] != baseline_row["pair_sha"]:
        raise RuntimeError(f"fold={fold} {group}: split or patient-pair mismatch")
    fave, wrong_texts, init_rng_sha = create_fave(group, fold, prompt_cache)
    context = build_reference_context(state, fold)
    state["fave"] = fave
    state["fave_context"] = context
    state["fave_wrong_texts"] = wrong_texts.to(R.runtime()["device"])
    state["opt_fave"] = optimizer_for_fave(fave, lr=0.001)
    state["initial_fave_param_sha"] = _module_param_sha(fave)
    state["initial_fave_state_sha"] = sha_json({
        name: value.detach().cpu().numpy().tobytes().hex()
        for name, value in fave.state_dict().items()
    })
    state["fave_init_rng_sha"] = init_rng_sha
    state["prompt_group_sha"] = prompt_cache["embedding_sha256"] if group == FULL else numeric_text_embeddings(prompt_cache["embeddings"].shape[1])[1]
    ids = [id(p) for module in (state["model"], state["cam"], state["fave"]) for p in module.parameters()]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate trainable parameters across GFR/CAM/CADR/FAVE modules")
    return state


def _make_checkpoint(state, group, fold, epoch, epochs_executed, metrics, best_diag, best_state, best_acc):
    prompt_cache = torch_load_prompt_cache()
    return {
        "experiment": "Experiment58 FAVE-PEN",
        "group": group, "fold": fold, "seed": 0,
        "best_epoch": int(epoch), "epochs_executed": int(epochs_executed),
        "best_acc": float(best_acc), "metrics": {name: float(metrics[name]) for name in METRICS},
        "model": best_state["model"], "cam": best_state["cam"], "fave": best_state["fave"],
        "optimizer_base": best_state["optimizer_base"], "optimizer_cam": best_state["optimizer_cam"],
        "optimizer_added": best_state["optimizer_added"], "optimizer_fave": best_state["optimizer_fave"],
        "diagnostics": best_diag,
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_cadr_added_sha": state["initial_added_sha"],
        "initial_fave_parameter_sha": state["initial_fave_param_sha"],
        "initial_fave_state_sha": state["initial_fave_state_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "fave_init_rng_sha": state["fave_init_rng_sha"],
        "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "reference_context_sha": state["fave_context"]["cache_sha256"],
        "initial_context_sha": state["fave_context"]["cache_sha256"],
        "prompt_embedding_sha": prompt_cache["embedding_sha256"],
        "prompt_group_sha": state["prompt_group_sha"],
        "fave_source_sha256": sha_file(EXP / "fave_pen.py"),
        "runner_source_sha256": sha_file(Path(__file__)),
        "optimizer_weight_decay": {
            "gfr": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
            "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "cadr": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
            "fave": [float(pg["weight_decay"]) for pg in state["opt_fave"].param_groups],
        },
        "rng": R.pack_rng(), "rng_sha": R.rng_sha(R.pack_rng()),
        "config_sha256": sha_file(CONFIG_PATH), "prompts_sha256": sha_file(PROMPTS_PATH),
    }


def torch_load_prompt_cache():
    return R.runtime()["torch"].load(RUNTIME / "prompt_embeddings.pt", map_location="cpu", weights_only=False)


def train_one(group: str, fold: int, prompt_cache, reference_by_fold):
    import torch

    state = build_state(group, fold, prompt_cache, reference_by_fold)
    latest_path = checkpoint_path(group, fold, latest=True)
    best_path = checkpoint_path(group, fold)
    task_file = task_path(group, fold)
    task_file.parent.mkdir(parents=True, exist_ok=True)
    if task_file.exists() and best_path.exists():
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(best_path):
            return task
    start_epoch, wait, best_acc, best_epoch = 0, 0, -1.0, -1
    best_state, best_metrics, best_diag = None, None, None
    warmup_best, warmup_epoch = -1.0, -1
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved.get("fold") == fold and saved.get("group") == group and saved.get("config_sha256") == sha_file(CONFIG_PATH) and saved.get("source_sha256") == sha_file(Path(__file__)):
            for key, current in (
                ("initial_shared_sha", state["initial_shared_sha"]),
                ("initial_fave_parameter_sha", state["initial_fave_param_sha"]),
                ("split_sha", state["split_sha"]),
                ("pair_sha", state["cache_meta"]["pair_sha"]),
                ("initial_context_sha", state["fave_context"]["cache_sha256"]),
            ):
                if saved.get(key) != current:
                    raise RuntimeError(f"checkpoint resume identity mismatch: {key}")
            state["model"].load_state_dict(saved["model"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            state["fave"].load_state_dict(saved["fave"], strict=True)
            state["opt_base"].load_state_dict(saved["optimizer_base"])
            state["opt_cam"].load_state_dict(saved["optimizer_cam"])
            state["opt_added"].load_state_dict(saved["optimizer_added"])
            state["opt_fave"].load_state_dict(saved["optimizer_fave"])
            start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
            best_acc, best_epoch = float(saved["best_acc"]), int(saved["best_epoch"])
            best_state, best_metrics, best_diag = saved["best_state"], saved["best_metrics"], saved["best_diag"]
            warmup_best, warmup_epoch = float(saved["warmup_best"]), int(saved["warmup_epoch"])
            R.restore_rng(saved["rng"])

    cfg = R.CFG
    epochs_executed = start_epoch
    started = time.time()
    try:
        for epoch in range(start_epoch, int(cfg["cv"]["max_epochs"])):
            for opt in (state["opt_base"], state["opt_cam"], state["opt_added"], state["opt_fave"]):
                opt.zero_grad(set_to_none=True)
            active = epoch >= int(cfg["warmup"]["epochs"])
            out = forward_fave_state(state, training=True, active=active)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError(f"nonfinite loss at fold={fold}, epoch={epoch}")
            out["total"].backward()
            for module in (state["model"], state["cam"], state["fave"]):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                    raise FloatingPointError(f"nonfinite gradient at fold={fold}, epoch={epoch}")
            grad = {
                "gfr": _grad_norm(state["base_params"]),
                "cam": _grad_norm(state["cam"].parameters()),
                "cadr": _grad_norm(p for _, p in state["added_named"]),
                "fave": _grad_norm(state["fave"].parameters()) if active else 0.0,
            }
            # Preserve the historical shared optimizer call order.
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_added"].step()
            if active:
                state["opt_fave"].step()
            epochs_executed = epoch + 1
            with torch.no_grad():
                evaluated = forward_fave_state(state, training=False, active=active)
                metrics = R.classification_metrics(evaluated["logp"], state["test"], state["y"])
            if not active:
                if metrics["acc"] > warmup_best:
                    warmup_best, warmup_epoch = metrics["acc"], epoch
            elif metrics["acc"] > best_acc:
                best_acc, best_epoch, wait = metrics["acc"], epoch, 0
                best_state = {
                    "model": _cpu_state(state["model"]), "cam": _cpu_state(state["cam"]),
                    "fave": _cpu_state(state["fave"]),
                    "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                    "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                    "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                    "optimizer_fave": copy.deepcopy(state["opt_fave"].state_dict()),
                }
                best_metrics = {name: float(metrics[name]) for name in METRICS}
                best_diag = {
                    "loss_cls": float(out["cls"].detach().cpu()),
                    "loss_graph": float(out["graph"].detach().cpu()),
                    "loss_pair": float(out["pair"].detach().cpu()),
                    "loss_adv": float(out["adv"].detach().cpu()),
                    "loss_preserve": float(out["preserve"].detach().cpu()),
                    "loss_match": float(out["match"].detach().cpu()),
                    "loss_verify": float(out["verify"].detach().cpu()),
                    "adv_accuracy": float(out["adv_info"]["accuracy"].detach().cpu()),
                    "gradient_norms": grad,
                    "fave_enabled": True,
                    "branch_signature": "E1_CADR_GFR_FULL_WITH_FAVE_CORR_POSITION",
                }
                cp = _make_checkpoint(state, group, fold, best_epoch, epochs_executed, metrics, best_diag, best_state, best_acc)
                atomic_torch_save(best_path, cp)
            else:
                wait += 1

            if (epoch + 1) % 10 == 0 or epoch == int(cfg["cv"]["max_epochs"]) - 1 or (active and wait > int(cfg["cv"]["early_stop_wait_gt"])):
                atomic_torch_save(latest_path, {
                    "group": group, "fold": fold, "next_epoch": epoch + 1,
                    "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
                    "best_state": best_state, "best_metrics": best_metrics, "best_diag": best_diag,
                    "warmup_best": warmup_best, "warmup_epoch": warmup_epoch,
                    "model": _cpu_state(state["model"]), "cam": _cpu_state(state["cam"]),
                    "fave": _cpu_state(state["fave"]),
                    "optimizer_base": state["opt_base"].state_dict(),
                    "optimizer_cam": state["opt_cam"].state_dict(),
                    "optimizer_added": state["opt_added"].state_dict(),
                    "optimizer_fave": state["opt_fave"].state_dict(),
                    "rng": R.pack_rng(), "config_sha256": sha_file(CONFIG_PATH),
                    "source_sha256": sha_file(Path(__file__)),
                    "initial_shared_sha": state["initial_shared_sha"],
                    "initial_fave_parameter_sha": state["initial_fave_param_sha"],
                    "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
                    "initial_context_sha": state["fave_context"]["cache_sha256"],
                })
            if active and wait > int(cfg["cv"]["early_stop_wait_gt"]):
                break
            if (epoch + 1) % 25 == 0:
                atomic_json(task_file, {
                    "status": "RUNNING", "group": group, "fold": fold,
                    "epoch": epoch + 1, "best_epoch": best_epoch,
                    "best_acc": best_acc, "rng_sha": R.rng_sha(R.pack_rng()),
                })
        if best_state is None:
            raise RuntimeError("no active FAVE checkpoint selected")
        state["model"].load_state_dict(best_state["model"], strict=True)
        state["cam"].load_state_dict(best_state["cam"], strict=True)
        state["fave"].load_state_dict(best_state["fave"], strict=True)
        with torch.no_grad():
            readback = forward_fave_state(state, training=False, active=True)
            readback_metrics = R.classification_metrics(readback["logp"], state["test"], state["y"])
        for name in METRICS:
            if abs(readback_metrics[name] - best_metrics[name]) > 1e-10:
                raise RuntimeError(f"worker best/readback mismatch fold={fold} metric={name}")
        cp = _make_checkpoint(state, group, fold, best_epoch, epochs_executed, readback_metrics, best_diag, best_state, best_acc)
        atomic_torch_save(best_path, cp)
        task = {
            "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "best_epoch": best_epoch, "epochs_executed": epochs_executed,
            "warmup_best_epoch": warmup_epoch, "warmup_best_acc": warmup_best,
            "metrics": {name: float(readback_metrics[name]) for name in METRICS},
            "best_sha256": sha_file(best_path), "initial_shared_sha": state["initial_shared_sha"],
            "initial_fave_parameter_sha": state["initial_fave_param_sha"],
            "initial_rng_sha": state["construction_rng_sha"], "fave_init_rng_sha": state["fave_init_rng_sha"],
            "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
            "runtime_seconds": time.time() - started,
        }
        atomic_json(task_file, task)
        return task
    except Exception as exc:
        atomic_json(task_file, {"status": "FAILED", "group": group, "fold": fold, "error": repr(exc)})
        raise


def prepare():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    prompt_cache = encode_prompt_bank()
    baseline, reference = _baseline_audit_and_preflight(prompt_cache)
    manifest_path = EXP / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if previous.get("status") != "COMPLETE_AUDITED":
        atomic_json(manifest_path, {
            "experiment": "Experiment 58 FAVE-PEN",
            "status": "PREFLIGHT_PASS_READY_FOR_TRAINING",
            "actual_directory": str(EXP),
            "formal_groups": [FULL, NUM],
            "formal_folds_required": 20,
            "formal_folds_complete": 0,
            "independent_checkpoint_audits": {"required": 20, "passed": 0, "status": "PENDING"},
            "baseline_group": BASE_GROUP,
            "baseline_audit_sha256": sha_file(EXP / "baseline_audit.json"),
            "config_sha256": sha_file(CONFIG_PATH),
            "run_sha256": sha_file(Path(__file__)),
            "fave_module_sha256": sha_file(EXP / "fave_pen.py"),
            "prompts_sha256": sha_file(PROMPTS_PATH),
            "prompt_embedding_sha256": prompt_cache["embedding_sha256"],
            "RN50_checkpoint_sha256": RN50_SHA,
            "historical_reference_folds": len(reference),
            "training_started": False,
            "mounts_equivalent": False,
        })
    return baseline, reference


def run_worker(group, fold):
    prompt_cache = torch_load_prompt_cache()
    baseline = json.loads((EXP / "baseline_audit.json").read_text(encoding="utf-8"))
    ref_by_fold = {int(row["fold"]): row for row in baseline["folds"]}
    return train_one(group, fold, prompt_cache, ref_by_fold)


def schedule(max_workers=5):
    import time as _time

    if not (1 <= max_workers <= 5):
        raise ValueError("Experiment 58 permits at most five workers")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(group, fold) for group in (FULL, NUM) for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        cp, task = checkpoint_path(group, fold), task_path(group, fold)
        if cp.is_file() and task.is_file():
            task_obj = json.loads(task.read_text(encoding="utf-8"))
            if task_obj.get("status") == "COMPLETE" and task_obj.get("best_sha256") == sha_file(cp):
                continue
        pending.append((group, fold))
    active = {}
    failures = []
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            log_path = LOG_ROOT / f"{group}_f{fold:02d}.log"
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
            })
            stream = log_path.open("ab")
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", group, str(fold)],
                stdout=stream, stderr=subprocess.STDOUT, env=env,
            )
            active[process] = (group, fold, stream, time.time())
            print(f"started {group} fold {fold}/10 (active={len(active)}/{max_workers})", flush=True)
        done = []
        for process, (group, fold, stream, launched) in active.items():
            code = process.poll()
            if code is not None:
                stream.close()
                if code == 0:
                    print(f"completed {group} fold {fold}/10", flush=True)
                else:
                    failures.append((group, fold, code, str(LOG_ROOT / f"{group}_f{fold:02d}.log")))
                    print(f"FAILED {group} fold {fold}: exit={code}", flush=True)
                done.append(process)
        for process in done:
            del active[process]
        _time.sleep(2.0)
    if failures:
        raise RuntimeError(f"training fold failures: {failures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("GROUP", "FOLD"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print("preflight PASS; no formal fold trained", flush=True)
    elif args.worker:
        group, fold = args.worker[0], int(args.worker[1])
        task = run_worker(group, fold)
        print(json.dumps(task, ensure_ascii=False), flush=True)
    elif args.train:
        if not args.skip_prepare:
            prepare()
        schedule(args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
