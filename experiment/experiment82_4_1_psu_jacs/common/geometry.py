"""Pretraining geometry report on published checkpoints, without test targets."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
E824 = HERE.parent / "experiment82_4_four_spectral_prompt"
sys.dont_write_bytecode = True
sys.path.insert(0, str(E824))

from common import history as h
import numpy as np
import torch
import run as archived_run


NAMES = ("DARG", "E82.2-G2", "E82.3.2-G2", *(f"E82.4-{g}" for g in archived_run.METHODS))
COLUMNS = ("rho", "cosine_asd", "linear_asd_margin", "actual_asd_margin")


def summarize(values):
    report = {}
    for name, column in zip(COLUMNS, values.T):
        report[name] = {"mean": float(column.mean()), "median": float(np.median(column)),
                        "p10": float(np.quantile(column, .1)), "p90": float(np.quantile(column, .9))}
    return {"n": len(values), **report} if len(values) else {"n": 0}


def distribution(values):
    return summarize(values) if len(values) else {"n": 0}


def restore(fold, name, state, spectrum, text, manifest):
    native = h.runner().load_history()
    if name == "DARG":
        path = h.runner().best_path(h.BASE, fold)
    elif name in ("E82.2-G2", "E82.3.2-G2"):
        directory = h.E822 if name == "E82.2-G2" else h.E8232
        path = directory / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt"
    else:
        group = name.rsplit("-", 1)[1]
        path = archived_run.paths(group, fold)["best"]
    digest = h.sha_file(path)
    h.require(digest == manifest["protected_sha256"][str(path)], f"Geometry Best changed: {name}, {fold}")
    best = h.load_torch(path)
    h.require(best["fold"] == fold, "Geometry checkpoint has wrong fold")
    probability_key = "probability"
    if name == "DARG":
        native.load_bundle(state, best["best_state"])
        module = None
        probability_key = "test_probability"
    elif name == "E82.2-G2":
        native.load_bundle(state, best["current"]["base"])
        module = h.psce().PSCE(spectrum, text, "G2", fold)
        module.load_state_dict(best["current"]["psce"], strict=True)
    elif name == "E82.3.2-G2":
        native.load_bundle(state, best["current"]["base"])
        profile = h.read_json(h.E8232 / "config.json")
        path = h.HISTORY / "experiment82_3_cspe/runtime" / f"class_stats_f{fold:02d}.pt"
        h.require(h.sha_file(path) == manifest["protected_sha256"][str(path)], "Geometry class cache changed")
        stats = h.load_torch(path)
        module = h.da().DASRPE(spectrum, text, stats["stats"], "G2", fold, profile["class_statistics"],
                               profile["ema"]["alpha"])
        module.initialize_gate(profile["patient_gate"])
        module.load_state_dict(best["current"]["cspe"], strict=True)
    else:
        state["model"].load_state_dict(best["state"]["model"], strict=True)
        state["cam"].load_state_dict(best["state"]["cam"], strict=True)
        module = archived_run.METHODS[group](spectrum, text, best["reference"], state["excluded"],
                                              best["config"], best["method"])
        module.load_state_dict(best["state"]["branch"], strict=True)
        best = best["evaluation"]
    if module is not None:
        module.to(state["x"].device).eval()
    return module, best[probability_key], digest


@torch.no_grad()
def capture_cam(state, fold):
    state["model"].eval()
    state["cam"].eval()
    native = h.runner()
    fused, _, _ = native.load_dcr().rudag_forward(
        state["model"], state["parts"], state["train_pairs"], h.BASE, fold=fold,
        permutation_seed_base=native.CFG["method"]["score_permutation_seed_base"],
    )
    values = []

    def save(_head, args):
        values.append(args[0])

    head = state["cam"].Outputlayer
    hook = head.register_forward_pre_hook(save)
    try:
        state["cam"](fused)
    finally:
        hook.remove()
    h.require(len(values) == 1 and values[0].shape == (871, 144), "Native geometry CAM shape differs")
    return values[0]


def diagnose(fold):
    h.verify_device()
    manifest = h.read_json(HERE / "manifest.json")
    state, _, _, _ = h.prepare(fold)
    spectrum = h.load_torch(h.E822 / "runtime" / f"spectrum_f{fold:02d}.pt")["spectrum"]
    basis = spectrum["U"].to(dtype=state["x"].dtype, device=state["x"].device)
    text = h.verify_text()
    head = state["cam"].Outputlayer
    train, test = state["legal_np"], state["test_np"]
    labels = state["labels_np"][train]
    history = {}
    before = h.rng_sha()
    for name in NAMES:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(824100 + fold)
            module, saved, digest = restore(fold, name, state, spectrum, text, manifest)
        cam = capture_cam(state, fold)
        with torch.enable_grad():
            leaf = cam.detach().requires_grad_(True)
            logits = head.classifier(head.mlp_head(leaf))
            gradient, = torch.autograd.grad((logits[:, 0] - logits[:, 1]).sum(), leaf)
        with torch.no_grad():
            if module is None:
                enhanced, delta = cam, torch.zeros_like(cam)
            else:
                flags = {"crossfit": False} if name.startswith("E82.4-") else {}
                enhanced, detail = module(cam, **flags)
                delta = detail["delta"]
            logp, _ = head(enhanced, None)
            probabilities = logp.exp()
            h.require(np.array_equal(probabilities[test].cpu().numpy(), saved), f"Geometry {name} Best predictions differ")
            prediction = probabilities.argmax(-1).cpu().numpy()
            adjusted = head.classifier(head.mlp_head(enhanced))
            actual = (adjusted[:, 0] - adjusted[:, 1]) - (logits[:, 0] - logits[:, 1])
            linear = (gradient * delta).sum(-1)
            cosine = linear / (gradient.norm(dim=-1) * delta.norm(dim=-1) + 1e-8)
            coverage = (gradient @ basis).square().sum(-1) / (gradient.square().sum(-1) + 1e-8)
            values = torch.stack((coverage, cosine, linear, actual), dim=-1).cpu().numpy()
        correct = prediction[train] == labels
        history[name] = {
            "checkpoint_sha": digest, "train": {
                "all": distribution(values[train]), "correct": distribution(values[train][correct]),
                "wrong": distribution(values[train][~correct])},
            "test": {"all": distribution(values[test])},
            "train_rows": [[int(row), bool(good), *map(float, values[row])] for row, good in zip(train, correct)],
            "test_rows": [[int(row), int(prediction[row]), *map(float, values[row])] for row in test],
        }
        h.require(np.isfinite(values).all() and h.rng_sha() == before, "Invalid geometry or changed historical RNG")
        print(f"fold={fold} {name} test coverage={values[test, 0].mean():.6f}", flush=True)
    h.atomic_json(HERE / "runtime/geometry" / f"f{fold:02d}.json", {"status": "PASS", "fold": fold, "history": history})


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "geometry-fold":
        raise SystemExit("usage: geometry.py geometry-fold FOLD")
    diagnose(int(sys.argv[2]))
