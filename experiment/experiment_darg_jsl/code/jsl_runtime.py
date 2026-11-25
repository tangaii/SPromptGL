"""Standalone DARG + JSL training, evaluation, and audit runtime."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "runtime/python"), str(ROOT / "code/history_base")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(ROOT))
from darg_core import network as darg_network

# The historical DARG core only needs GraphLearn and GCN. Its upstream network
# module also initializes CLIP Prompt text at import time, so supply the exact
# graph classes through this DARG-only compatibility module.
sys.modules["network"] = darg_network

from common import history as h
from common.spectral import fixed_logits, margin
from jsl_model import JSL

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
FOLDS = tuple(range(1, CONFIG["folds"] + 1))
START = CONFIG["start_epoch_completed"]
MAX_EPOCHS = CONFIG["max_epochs"]
WAIT_GT = CONFIG["early_stop_wait_gt"]
CAP = CONFIG["jsl"]["shared_gradient_cap"]
OVERLAP_APPEND_1BASED = {
    2: (681, 779, 784, 260, 288, 375),
    3: (46, 66, 172, 266, 388),
    4: (713, 727, 773, 712),
    8: (383, 430),
}
FORMAL_EXTENDED_SHA = {
    2: ("a8c050a58642ca726afda4802bcf4ab5b7d2dfb24ae6677e41128e364969002d",
        "9b4c58fbb82bdbb2daeee969500f547c27cc81df500e78be30aca75dc1ecbcb1"),
    3: ("bc84174b05a63c07ec123fb1200f49d6154013924f7aa0b504b5cfae72d43f7a",
        "42617329dae04ed3416f881c75e361f01b30ee4ae5462c8a515e5bf02b019898"),
    4: ("788aea140e3c387029900c83155c72c842c74b786cac64d2715a24b34cbec99f",
        "36f1b6adeb1e28e75af424e43820c3142a732ea622a12212d3318f90534e9577"),
    8: ("da33c5220c96cd536ddf50c94c65c92e7c95ddff55ea5b4f050a3a78e6104319",
        "c801eadbc653e3db346ff4dc4f4569efd3d7c87d0efb373a357a64f8e33d6490"),
}


def sha(value):
    if isinstance(value, Path):
        return hashlib.sha256(value.read_bytes()).hexdigest()
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for member in item:
                visit(member)
        elif isinstance(item, bytes):
            digest.update(item)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def balanced(values, labels):
    return 0.5 * (values[labels == 0].mean() + values[labels == 1].mean())


def hard_auc(labels, prediction):
    sensitivity = np.mean(prediction[labels == 0] == 0)
    specificity = np.mean(prediction[labels == 1] == 1)
    return float((sensitivity + specificity) / 2)


def binary_auc(positive, scores):
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positive[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = ((start + 1) + stop) / 2
        start = stop
    positives = int(positive.sum())
    negatives = len(positive) - positives
    if not positives or not negatives:
        raise ValueError("Prob-AUC requires both classes")
    rank_sum = ranks[sorted_positive].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(labels, prediction, probability):
    labels = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    true_asd = labels == 0
    true_td = labels == 1
    tp_asd = int(np.sum(true_asd & (prediction == 0)))
    fp_asd = int(np.sum(true_td & (prediction == 0)))
    fn_asd = int(np.sum(true_asd & (prediction == 1)))
    return {
        "n_test": len(labels), "correct": int((labels == prediction).sum()),
        "acc": float(np.mean(labels == prediction)),
        "hard_auc": hard_auc(labels, prediction),
        "prob_auc": binary_auc(true_asd, probability[:, 0]),
        "sen": float(tp_asd / max(tp_asd + fn_asd, 1)),
        "spe": float(np.mean(prediction[true_td] == 1)),
        "asd_f1": float(2 * tp_asd / max(2 * tp_asd + fp_asd + fn_asd, 1)),
    }


def implementation():
    files = [ROOT / "config.json", ROOT / "README.md", Path(__file__), ROOT / "code/jsl_model.py",
             ROOT / "code/standalone_paths.py", ROOT / "code/history_base/common/history.py",
             ROOT / "code/history_base/common/spectral.py", ROOT / "code/darg_core/__init__.py",
             ROOT / "code/darg_core/network.py", ROOT / "code/rng_forensics.py",
             ROOT / "author_log.py", ROOT / "cli.py", ROOT / "preflight.py"]
    result = {str(path.relative_to(ROOT)): sha(path) for path in files if path.is_file()}
    vendor = ROOT / "code/vendor/repo"
    # The archived E74.4 runner's manifest names an audit script that is not
    # part of the frozen DPSL source closure. Hash the available closure
    # directly so a preflight failure cannot be replaced by a second missing
    # file error while computing the implementation receipt.
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            result[f"vendor/repo/{path.relative_to(vendor).as_posix()}"] = sha(path)
    return dict(sorted(result.items()))


def fold_members(fold):
    state, warmup, identity, optimizers, members = h.warmup_and_membership(fold)
    if fold in OVERLAP_APPEND_1BASED:
        old_train = members["train"]
        add = np.asarray(OVERLAP_APPEND_1BASED[fold], dtype=np.int64) - 1
        expected_test = members["test"]
        labels = state["labels_np"]
        if not np.isin(add, expected_test).all() or np.isin(add, old_train).any():
            raise RuntimeError(f"fold {fold} overlap append order is invalid")
        train = np.concatenate((old_train, add))
        pairs, same = h.runner().load_history().load_e69().load_e61().make_outer_pairs(
            fold, state["supports"], train, labels
        )
        e69 = h.runner().load_history().load_e69()
        e61 = h.runner().load_history().load_e69().load_e61()
        expected_train_sha, expected_pair_sha = FORMAL_EXTENDED_SHA[fold]
        actual_train_sha, actual_pair_sha = e69.index_sha(train), e61.pair_sha(pairs, same)
        if actual_train_sha != expected_train_sha or actual_pair_sha != expected_pair_sha:
            raise RuntimeError(
                f"fold {fold} formal identity mismatch: train {actual_train_sha} != {expected_train_sha}; "
                f"pairs {actual_pair_sha} != {expected_pair_sha}"
            )
        if pairs.shape != (4096, 2) or np.bincount(same, minlength=2).tolist() != [2048, 2048]:
            raise RuntimeError(f"fold {fold} relation pairs are not the archived balanced 4096")
        device = state["y"].device
        excluded = [int(row) for row in train if row in set(expected_test)]
        counts = np.bincount(labels[train], minlength=2)
        state.update(
            train_np=train,
            train=torch.as_tensor(train, dtype=torch.long, device=device),
            weights=torch.from_numpy((len(train) / counts / 2).astype(np.float32)).to(device),
            train_pairs_np=pairs,
            train_same_np=same,
            train_pairs=torch.as_tensor(pairs, dtype=torch.long, device=device),
            train_same=torch.as_tensor(same, dtype=torch.float32, device=device),
            pair_sha=e61.pair_sha(pairs, same),
            effective_train_sha=e69.index_sha(train),
            added_train_rows_1based=[int(row) for row in OVERLAP_APPEND_1BASED[fold]],
            added_train_count=len(add),
            train_test_overlap_1based=[row + 1 for row in excluded],
            pair_rng_unchanged=True,
        )
        state["excluded"] = excluded
        members = h.membership(state)
    expected = [[783, 783, 88, 0], [792, 784, 87, 8], [789, 784, 87, 5], [788, 784, 87, 4],
                [784, 784, 87, 0], [784, 784, 87, 0], [784, 784, 87, 0], [786, 784, 87, 2],
                [784, 784, 87, 0], [784, 784, 87, 0]][fold - 1]
    actual = [len(members[k]) for k in ("train", "legal", "test")] + [len(set(members["train"]) & set(members["test"]))]
    if actual != expected:
        raise RuntimeError(f"fold {fold} membership {actual} != {expected}")
    if fold in FORMAL_EXTENDED_SHA:
        expected_train_sha, expected_pair_sha = FORMAL_EXTENDED_SHA[fold]
        if state["effective_train_sha"] != expected_train_sha or state["pair_sha"] != expected_pair_sha:
            raise RuntimeError(f"fold {fold} final training membership or pair digest differs")
    return state, warmup, identity, optimizers, members


def setup(fold, stage_callback=None):
    state, warmup, identity, optimizers, members = fold_members(fold)
    if stage_callback is not None:
        stage_callback("S4", state=state, members=members, identity=identity, optimizers=optimizers)
    archived = h.spectrum(fold, identity, members["legal"])
    if stage_callback is not None:
        stage_callback("S5", state=state, members=members, identity=identity,
                       optimizers=optimizers, archived=archived)
    before = h.rng_sha()
    capture = {}
    head = state["cam"].Outputlayer

    def inject(_layer, args):
        capture["cam"] = args[0]

    token = head.register_forward_pre_hook(inject)
    try:
        with torch.no_grad():
            h.runner().forward(state, h.BASE, training=False)
    finally:
        token.remove()
    cam = capture["cam"].cpu()
    if not torch.equal(cam, archived["warmup_cam"]):
        raise RuntimeError(f"fold {fold} E61 CAM differs from archived Original O")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CONFIG["seed"])
        module = JSL(archived["spectrum"], state["excluded"]).to(state["x"].device)
    if stage_callback is not None:
        stage_callback("S6", state=state, members=members, identity=identity,
                       optimizers=optimizers, archived=archived, module=module)
    opt = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=0.01)
    if h.rng_sha() != before or warmup["rng_sha"] != before:
        raise RuntimeError(f"fold {fold} JSL initialization changed historical RNG")
    if stage_callback is not None:
        stage_callback("S7", state=state, members=members, identity=identity,
                       optimizers=optimizers, archived=archived, module=module, optimizer=opt)
    return {"fold": fold, "state": state, "warmup": warmup, "identity": identity,
            "optimizers": optimizers, "members": members, "archived": archived,
            "module": module, "optimizer": opt, "provenance": {
                "identity": identity, "optimizers": optimizers, "membership": sha(members),
                "warm_cam": sha(archived["warmup_cam"]), "spectrum": sha(archived["spectrum"]),
                "basis": sha(archived["spectrum"]["U"]), "native_rng": before,
                "legal_index_sha": sha(members["legal"]), "initial_module": sha(module.state_dict()),
            }}


@contextmanager
def dropout_inventory(state):
    calls = []
    module_calls = []
    hooks = []
    original = F.dropout

    for root_name in ("model", "cam"):
        root = state[root_name]
        for name, module in root.named_modules():
            if isinstance(module, torch.nn.Dropout):
                def capture_dropout(_module, args, root_name=root_name, name=name):
                    values = args[0]
                    module_calls.append({"module": f"{root_name}.{name}", "shape": list(values.shape),
                                         "p": float(_module.p), "dtype": str(values.dtype)})
                hooks.append(module.register_forward_pre_hook(capture_dropout))

    def track(inputs, p=0.5, training=True, inplace=False):
        output = original(inputs, p=p, training=training, inplace=inplace)
        if training and 0 < p < 1:
            calls.append({"shape": list(inputs.shape), "p": float(p), "dtype": str(inputs.dtype),
                          "output_sha": sha(output)})
        return output
    F.dropout = track
    try:
        yield {"calls": calls, "modules": module_calls}
    finally:
        F.dropout = original
        for hook in hooks:
            hook.remove()


def original_gfr_alignment(state, output):
    """Reproduce the original GFR ALoss diagnostic from its four base graphs."""
    labels = state["y"].clone()
    labels[labels == 0] = -1
    observed = torch.zeros_like(labels)
    observed[state["train"]] = labels[state["train"]]
    same_label = observed[:, None].eq(observed[None, :])
    identity = torch.eye(len(labels), device=labels.device, dtype=output["fused"].dtype)
    normalize = h.base().runtime()["utils"].normalize_adj
    target = normalize(same_label.to(output["fused"].dtype) + identity)
    values = [torch.linalg.norm(normalize(adjacency + identity) - target)
              for adjacency in output["info"]["original_adjs"]]
    return torch.stack(values).mean()


def forward(ctx, training, intervention="full"):
    state, module = ctx["state"], ctx["module"]
    head = state["cam"].Outputlayer
    capture = {}

    def inject(_layer, args):
        cam = args[0]
        capture["cam"] = cam
        if not training:
            enhanced, details = module(cam, head, intervention)
            capture.update(enhanced=enhanced, details=details)
            return (enhanced, *args[1:])

    def logits(_layer, _args, output):
        capture["logits"] = output
    hooks = [head.register_forward_pre_hook(inject), head.classifier.register_forward_hook(logits)]
    try:
        with torch.set_grad_enabled(training):
            result = h.runner().forward(state, h.BASE, training=training)
    finally:
        for hook in hooks:
            hook.remove()
    capture.update(attention=result.get("attention"), hidden=result.get("hidden"), fused=result.get("fused"))
    if training:
        enhanced, details = module(capture["cam"], head, intervention)
        capture.update(enhanced=enhanced, details=details)
    return result, capture


def shared_params(state):
    return [p for name in h.OPTS for group in state[name].param_groups for p in group["params"]
            if p.requires_grad]


def loss_terms(ctx, capture):
    state, details = ctx["state"], capture["details"]
    rows = torch.as_tensor(ctx["members"]["legal"], device=state["x"].device, dtype=torch.long)
    labels = torch.as_tensor(state["labels_np"][ctx["members"]["legal"]], device=state["x"].device)
    head = state["cam"].Outputlayer
    native = fixed_logits(head, capture["cam"][rows])
    full = fixed_logits(head, capture["enhanced"][rows])
    sign = 1 - 2 * labels
    native_margin = margin(native)
    full_margin = margin(full)
    spectral = fixed_logits(head, capture["cam"][rows] + details["delta_parallel"][rows])
    outside = fixed_logits(head, capture["cam"][rows] + details["delta_perp"][rows])
    ce_full = balanced(F.cross_entropy(full, labels, reduction="none"), labels)
    ce_off = balanced(F.cross_entropy(native, labels, reduction="none"), labels)
    gain = sign * full_margin - sign * native_margin
    spectral_gain = sign * margin(spectral) - sign * native_margin
    outside_gain = sign * margin(outside) - sign * native_margin
    native_true_margin = sign * native_margin
    full_true_margin = sign * full_margin

    def balanced_masked(values, mask):
        class_means = [values[(labels == label) & mask].mean() for label in (0, 1)
                       if ((labels == label) & mask).any()]
        return torch.stack(class_means).mean() if class_means else values.new_zeros(())

    originally_wrong = native_true_margin <= 0
    originally_right = ~originally_wrong
    weights = CONFIG["loss"]
    temperature = weights["margin_temperature"]
    gain_loss = balanced_masked(F.softplus(-gain / temperature), originally_wrong)
    protect = balanced_masked(F.relu(native_true_margin - full_true_margin), originally_right)
    residual = details["ratio"][rows].square().mean()
    jsl = (weights["enhanced_ce"] * ce_full + weights["original_ce"] * ce_off
           + weights["full_off_difference"] * (ce_full - ce_off)
           + weights["margin_gain"] * gain_loss
           + weights["true_margin_protection"] * protect
           + weights["spectral_candidate"] * F.softplus(-spectral_gain / temperature).mean()
           + weights["complement_candidate"] * F.softplus(-outside_gain / temperature).mean()
           + weights["residual_l2"] * residual)
    return jsl, {"ce_full": float(ce_full.detach()), "ce_off": float(ce_off.detach()),
                 "gain": float(gain.mean().detach()), "spectral_gain": float(spectral_gain.mean().detach()),
                 "outside_gain": float(outside_gain.mean().detach()), "residual": float(residual.detach())}


def train_step(ctx):
    state = ctx["state"]
    opts = [state[key] for key in h.OPTS] + [ctx["optimizer"]]
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    with dropout_inventory(state) as dropout:
        output, capture = forward(ctx, True)
        base_loss = output["total"]
        all_shared = shared_params(state)
        module_params = list(ctx["module"].parameters())
        base_grads = torch.autograd.grad(base_loss, all_shared, retain_graph=True, allow_unused=True)
        jsl_loss, diag = loss_terms(ctx, capture)
        parallel_grad, complement_grad = torch.autograd.grad(
            jsl_loss, (capture["details"]["delta_parallel"], capture["details"]["delta_perp"]),
            retain_graph=True, allow_unused=True,
        )
        jsl_grads = torch.autograd.grad(jsl_loss, all_shared + module_params, allow_unused=True)
        shared_extra = jsl_grads[:len(all_shared)]
        module_grads = jsl_grads[len(all_shared):]
        base_norm = torch.sqrt(sum((g.detach().square().sum() for g in base_grads if g is not None), torch.tensor(0., device=state["x"].device)))
        extra_norm = torch.sqrt(sum((g.detach().square().sum() for g in shared_extra if g is not None), torch.tensor(0., device=state["x"].device)))
        scale = torch.minimum(torch.tensor(1., device=base_norm.device), CAP * base_norm / extra_norm.clamp_min(1e-12))
        for p, base, extra in zip(all_shared, base_grads, shared_extra):
            if base is not None or extra is not None:
                p.grad = (base if base is not None else torch.zeros_like(p)) + scale * (extra if extra is not None else 0.)
        for p, grad in zip(module_params, module_grads):
            p.grad = grad
        for opt in opts:
            opt.step()
    if len(dropout["calls"]) != 6:
        raise RuntimeError(f"historical Dropout count changed: {len(dropout['calls'])}")
    group_slices = {}
    offset = 0
    for name, layer in (("spectral_encoder", ctx["module"].token_encoder),
                        ("spectral_weight", ctx["module"].token_score),
                        ("controller", ctx["module"].controller)):
        count = len(list(layer.parameters()))
        grads = module_grads[offset:offset + count]
        group_slices[name] = {
            "parameters": count,
            "with_grad": sum(g is not None for g in grads),
            "nonzero": sum(g is not None and bool(g.detach().count_nonzero()) for g in grads),
            "norm": float(torch.sqrt(sum((g.detach().square().sum() for g in grads if g is not None),
                                         torch.tensor(0., device=state["x"].device))).cpu()),
        }
        offset += count
    diag.update({"jsl_loss": float(jsl_loss.detach()), "darg_loss": float(base_loss.detach()),
                 "shared_extra_fraction": float((scale * extra_norm / base_norm.clamp_min(1e-12)).detach()),
                 "shared_base_norm": float(base_norm.detach()), "shared_extra_norm": float(extra_norm.detach()),
                 "base_gradient_sha": sha(base_grads),
                 "parallel_residual_gradient": float(parallel_grad.detach().norm()) if parallel_grad is not None else 0.0,
                 "complement_residual_gradient": float(complement_grad.detach().norm()) if complement_grad is not None else 0.0,
                 "dropout": dropout, "module_gradients": group_slices,
                 "jsl_grad_nonzero": sum(g is not None and bool(g.detach().count_nonzero()) for g in module_grads),
                 "native_logp_sha": sha(output["logp"]),
                 "native_cam_sha": sha(capture["cam"]),
                 "native_cls": float(output["cls"].detach()),
                 "author_gfr_alignment": float(original_gfr_alignment(state, output).detach())})
    return diag


@torch.no_grad()
def evaluate(ctx, intervention="full"):
    before = h.rng_sha()
    result, capture = forward(ctx, False, intervention)
    test = ctx["members"]["test"]
    all_probability = result["logp"].exp().cpu().numpy()
    probability = all_probability[test]
    prediction = probability.argmax(-1)
    if h.rng_sha() != before:
        raise RuntimeError("evaluation changed RNG")
    return {"logits": capture["logits"][test].cpu().numpy(), "probability": probability,
            "all_probability": all_probability,
            "prediction": prediction, "correct": int((prediction == ctx["state"]["labels_np"][test]).sum())}, capture


def author_epoch_record(ctx, diag, evaluation):
    labels = ctx["state"]["labels_np"]
    train = ctx["members"]["train"]
    train_probability = evaluation["all_probability"][train]
    train_prediction = train_probability.argmax(-1)
    train_labels = labels[train]
    test_labels = labels[ctx["members"]["test"]]
    return {
        "epoch": None,
        "trn_cls": diag["native_cls"],
        "gfr_alignment": diag["author_gfr_alignment"],
        "trn_auc": hard_auc(train_labels, train_prediction),
        "trn_acc": float(np.mean(train_prediction == train_labels)),
        "val_auc": hard_auc(test_labels, evaluation["prediction"]),
        "val_acc": float(np.mean(evaluation["prediction"] == test_labels)),
    }


def bundle(ctx):
    return {"native": h.base_bundle(ctx["state"]), "module": ctx["module"].state_dict(),
            "optimizer": ctx["optimizer"].state_dict(),
            "rng": h.base().pack_rng(), "rng_sha": h.rng_sha()}


def save_checkpoint(path, ctx, epoch, best_epoch, best_correct, wait, evaluation, record=None):
    state = bundle(ctx)
    payload = {"format": "DARG-JSL-v1", "fold": ctx["fold"], "epoch": epoch, "best_epoch": best_epoch,
               "best_correct": best_correct, "wait": wait, "provenance": ctx["provenance"],
               "state": state, "evaluation": evaluation, "journal_record": record,
               "state_sha": sha(state),
               "implementation": implementation()}
    h.save(path, payload)
    return payload


def restore(ctx, payload):
    if payload.get("format") != "DARG-JSL-v1" or payload.get("fold") != ctx["fold"]:
        raise RuntimeError("checkpoint format or fold identity differs")
    if payload.get("provenance") != ctx["provenance"]:
        raise RuntimeError("checkpoint warmup or module provenance differs")
    if payload.get("implementation") != implementation():
        raise RuntimeError("checkpoint implementation SHA differs")
    if payload.get("state_sha") != sha(payload["state"]):
        raise RuntimeError("checkpoint state content SHA differs")
    h.restore_base(ctx["state"], payload["state"]["native"])
    ctx["module"].load_state_dict(payload["state"]["module"], strict=True)
    ctx["optimizer"].load_state_dict(payload["state"]["optimizer"])
    h.base().restore_rng(payload["state"]["rng"])
    if h.rng_sha() != payload["state"]["rng_sha"]:
        raise RuntimeError("checkpoint RNG restore mismatch")


def reconcile_journal(path, latest_epoch, latest_record):
    existing = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    retained = [row for row in existing if int(row["epoch"]) <= latest_epoch]
    if latest_epoch > START:
        current = next((row for row in retained if int(row["epoch"]) == latest_epoch), None)
        if current is None:
            if latest_record is None or int(latest_record.get("epoch", -1)) != latest_epoch:
                raise RuntimeError("Latest checkpoint lacks its committed epoch journal record")
            retained.append(latest_record)
    retained.sort(key=lambda row: int(row["epoch"]))
    if [int(row["epoch"]) for row in retained] != list(range(START + 1, latest_epoch + 1)):
        raise RuntimeError("journal does not cover the committed Latest epoch sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in retained:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_fold(fold, stop_at=None):
    ctx = setup(fold)
    outdir = ROOT / "checkpoints"
    latest = outdir / f"fold{fold:02d}.latest.pt"
    best = outdir / f"fold{fold:02d}.best.pt"
    journal = ROOT / "runtime" / "journal" / f"fold{fold:02d}.jsonl"
    expected_trace = h.hist_trace(fold)
    epoch, best_epoch, best_correct, wait = START, 0, -1, 0
    if latest.is_file():
        payload = torch.load(latest, map_location=ctx["state"]["x"].device, weights_only=False)
        restore(ctx, payload)
        epoch, best_epoch, best_correct, wait = payload["epoch"], payload["best_epoch"], payload["best_correct"], payload["wait"]
        reconcile_journal(journal, epoch, payload.get("journal_record"))
    else:
        reconcile_journal(journal, START, None)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as stream:
        limit = MAX_EPOCHS if stop_at is None else min(stop_at, MAX_EPOCHS)
        while epoch < limit and wait <= WAIT_GT:
            epoch += 1
            epoch_started = time.monotonic()
            rng_before = h.rng_sha()
            diag = train_step(ctx)
            rng_after_train = h.rng_sha()
            evaluation, capture = evaluate(ctx)
            rng_after_eval = h.rng_sha()
            if epoch in expected_trace:
                archived = expected_trace[epoch]
                actual = {"rng_before": rng_before, "rng_after_train": rng_after_train,
                          "rng_after_eval": rng_after_eval}
                if any(actual[key] != archived[key] for key in actual):
                    raise RuntimeError(f"fold {fold} epoch {epoch} shared RNG trajectory differs from DPSL archive")
            labels = ctx["state"]["labels_np"][ctx["members"]["test"]]
            score = metrics(labels, evaluation["prediction"], evaluation["probability"])
            improved = score["correct"] > best_correct
            if improved:
                best_epoch, best_correct, wait = epoch, score["correct"], 0
            else:
                wait += 1
            author_log = author_epoch_record(ctx, diag, evaluation)
            author_log["epoch"] = epoch
            rec = {"epoch": epoch, "correct": score["correct"], "best_epoch": best_epoch, "best_correct": best_correct,
                   "wait": wait, "metrics": score, "diag": diag,
                   "author_log": author_log,
                   "rng": {"rng_before": rng_before, "rng_after_train": rng_after_train,
                           "rng_after_eval": rng_after_eval},
                   "epoch_elapsed_sec": time.monotonic() - epoch_started,
                   "rng_sha": h.rng_sha()}
            if improved:
                save_checkpoint(best, ctx, epoch, best_epoch, best_correct, wait, evaluation, record=rec)
            if improved or epoch % 10 == 0 or wait > WAIT_GT or epoch == MAX_EPOCHS or (epoch == limit and stop_at is not None):
                save_checkpoint(latest, ctx, epoch, best_epoch, best_correct, wait, evaluation, record=rec)
            stream.write(json.dumps(rec, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if epoch == START + 1 or epoch % 25 == 0:
                print(f"JSL fold {fold:02d} epoch {epoch} best {best_correct}@{best_epoch} wait {wait}", flush=True)
    if stop_at is None and not best.is_file():
        raise RuntimeError(f"fold {fold} did not produce Best")
    return {"fold": fold, "status": "TRAINED" if stop_at is None else "PREFLIGHT_STEP", "epoch": epoch,
            "best_epoch": best_epoch, "best_correct": best_correct, "wait": wait,
            "best": str(best), "latest": str(latest)}


def evaluate_all():
    rows, fold_rows = [], []
    author_summaries = []
    for fold in FOLDS:
        ctx = setup(fold)
        payload = torch.load(ROOT / "checkpoints" / f"fold{fold:02d}.best.pt", map_location=ctx["state"]["x"].device, weights_only=False)
        restore(ctx, payload)
        evaluation, capture = evaluate(ctx)
        test = ctx["members"]["test"]
        labels = ctx["state"]["labels_np"][test]
        score = metrics(labels, evaluation["prediction"], evaluation["probability"])
        fold_rows.append({"fold": fold, **score})
        attention_dir = ROOT / "artifacts/attention"
        graph_dir = ROOT / "artifacts/graph"
        attention_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        np.save(attention_dir / f"fold{fold:02d}.npy", capture["attention"], allow_pickle=True)
        np.savez_compressed(graph_dir / f"fold{fold:02d}.npz",
                            fused=capture["hidden"].detach().cpu().numpy(),
                            label=ctx["state"]["labels_np"])
        journal_rows = [json.loads(line) for line in
                        (ROOT / "runtime/journal" / f"fold{fold:02d}.jsonl").read_text(encoding="utf-8").splitlines()
                        if line]
        elapsed = sum(float(row["epoch_elapsed_sec"]) for row in journal_rows)
        author_summaries.append({"fold": fold, "labels": labels.tolist(),
                                 "prediction": evaluation["prediction"].tolist(),
                                 "acc": score["acc"], "hard_auc": score["hard_auc"],
                                 "elapsed_sec": elapsed})
        for i, patient in enumerate(test):
            rows.append({"fold": fold, "test_position": i, "patient_id_1based": int(patient) + 1,
                         "true_label": int(labels[i]), "prediction": int(evaluation["prediction"][i]),
                         "p_asd": float(evaluation["probability"][i, 0]), "p_td": float(evaluation["probability"][i, 1])})
    mean = {k: float(np.mean([r[k] for r in fold_rows])) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")}
    sd = {k: float(np.std([r[k] for r in fold_rows], ddof=0)) for k in mean}
    write_csv(ROOT / "results/fold_metrics.csv", fold_rows + [{"fold": "mean", **mean}, {"fold": "population_sd", **sd}])
    write_csv(ROOT / "results/patient_level.csv", rows)
    from author_log import render_result
    author_log = render_result(author_summaries, ctx["state"]["labels_np"], MAX_EPOCHS)
    return {"folds": fold_rows, "mean": mean, "population_sd": sd, "author_log": author_log}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

