"""Expose the archived SCMR computation with TADPOLE class and shape bindings."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from scmr_optimization import HISTORICAL_ROOT, SOURCE_HASHES, require
from source_runtime import archived_definitions, replace_exact, verified_text


@lru_cache(maxsize=1)
def source_config():
    return json.loads(verified_text(HISTORICAL_ROOT / "config.json", SOURCE_HASHES["config.json"]))


def class_pair(logits):
    """Keep binary 0-vs-1; orient the native leading pair by diagnosis in TADPOLE."""
    if logits.shape[-1] == 2:
        return torch.arange(2, device=logits.device).expand(len(logits), 2)
    require(logits.shape[-1] == 3, "SCMR requires the registered binary or TADPOLE class order")
    top_two = logits.detach().argsort(dim=-1, descending=True, stable=True)[:, :2]
    return top_two.sort(dim=-1, descending=True).values


def select_classes(values, pair):
    values = values.expand(len(pair), *values.shape[1:])
    index = pair.reshape(len(pair), *([1] * (values.ndim - 3)), 2, 1)
    return values.gather(-2, index.expand(*values.shape[:-2], 2, values.shape[-1]))


def true_class_margin(logits, labels):
    index = labels.expand(*logits.shape[:-1]).unsqueeze(-1)
    native = logits.gather(-1, index).squeeze(-1)
    competitor = logits.scatter(-1, index, -torch.inf).amax(dim=-1)
    return native - competitor


@lru_cache(maxsize=None)
def scmr_components(subjects, width, *, adapted=True):
    config = source_config()
    spectral_path = HISTORICAL_ROOT / "common/spectral.py"
    spectral_text = verified_text(spectral_path, SOURCE_HASHES["common/spectral.py"])
    if adapted:
        spectral_text = replace_exact(spectral_text, "torch.ones(CFG[\"subjects\"], dtype=torch.bool)",
                                      "torch.ones(SUBJECTS, dtype=torch.bool)")
        spectral_text = replace_exact(spectral_text, "nn.Linear(144, width)", "nn.Linear(CAM_WIDTH, width)")
        spectral_text = replace_exact(spectral_text,
                                      'tuple(cam.shape) == (CFG["subjects"], 144), "Correction input is not CAM144"',
                                      'cam.ndim == 2 and cam.shape[1] == CAM_WIDTH, "Unexpected SCMR CAM width"')
        spectral_text = replace_exact(spectral_text, "allowed = self.allowed[:, None]",
                                      "allowed = (self.allowed if len(cam) == SUBJECTS else "
                                      "self.allowed.new_ones(len(cam)))[:, None]")
        spectral_text = replace_exact(spectral_text, "def head_jacobian(head, cam):",
                                      "def head_jacobian(head, cam, pair):")
        spectral_text = replace_exact(spectral_text, "value = margin(fixed_logits(head, leaf))",
                                      "value = margin(fixed_logits(head, leaf).gather(1, pair))")
    suffix = f"{subjects}_{width}_{int(adapted)}"
    spectrum = archived_definitions(
        spectral_path, ("native_logits", "fixed_logits", "margin", "head_jacobian", "split_jacobian",
                        "l2_bound", "normalized_weights", "MedicalSpectrum"),
        source=spectral_text, CFG=config, EPS=config["spectrum"]["eps"], R=config["residual"],
        P=config["prompt"], require=require, math=math, SUBJECTS=subjects, CAM_WIDTH=width,
        module_name=f"e831_scmr_spectral_{suffix}",
    )

    evidence_path = HISTORICAL_ROOT / "common/evidence.py"
    evidence_text = verified_text(evidence_path, "0c8e2e33d802cd4fd2a1996b3d298fc84297b77e56b7d24045612d7e31eac6ba")
    if adapted:
        evidence_text = replace_exact(evidence_text, "self.mask.reshape(12, -1)",
                                       "self.mask.flatten(0, 1)")
        evidence_text = replace_exact(evidence_text, "summary.reshape(2, 6, 2, 16)",
                                       'summary.reshape(2, *self.ema["mean"].shape)')
        evidence_text = replace_exact(evidence_text, "shape = (groups + 1, groups, 2, 16)",
                                       'shape = (groups + 1, groups, *reference["mean"].shape[1:])')
        evidence_text = replace_exact(evidence_text, "def forward(self, white, training):",
                                       "def forward(self, white, training, pair):")
        evidence_text = replace_exact(evidence_text, "raw, bounded = gaussian(white, mu, var)",
                                       "mu, var = select_classes(mu, pair), select_classes(var, pair)\n"
                                       "        n = select_classes(n[..., None], pair).squeeze(-1)\n"
                                       "        raw, bounded = gaussian(white, mu, var)")
        evidence_text = replace_exact(evidence_text, "def disagreement(self, white, training):",
                                       "def disagreement(self, white, training, pair):")
        evidence_text = replace_exact(evidence_text, "nll = (white[:, None, None, :] - mu).square()",
                                       "mu, variance = select_classes(mu, pair), select_classes(variance, pair)\n"
                                       "        nll = (white[:, None, None, :] - mu).square()")
    evidence = archived_definitions(
        evidence_path, ("stabilize", "gaussian", "ClassEvidence", "AgreementEvidence"),
        source=evidence_text, E=config["evidence"], EPS=config["spectrum"]["eps"],
        require=require, select_classes=select_classes, module_name=f"e831_scmr_evidence_{suffix}",
    )

    g2_path = HISTORICAL_ROOT / "g2.py"
    g2_text = verified_text(g2_path, "b07e3d55690fc0230798b336cdd53aa49ebecc89af1979836a9a5ca51e48ab67")
    if adapted:
        g2_text = replace_exact(g2_text, "class SCMRDASRPE(nn.Module):",
                                 "class SCMRDASRPE(SCMRCheckpointState, nn.Module):")
        g2_text = replace_exact(g2_text, 'raw_llr, llr, reliability = self.statistics(d["white"], training)',
                                 'pair = class_pair(fixed_logits(head, cam.detach()))\n'
                                 '        raw_llr, llr, reliability = self.statistics(d["white"], training, pair)')
        g2_text = replace_exact(g2_text, 'self.statistics.disagreement(d["white"], training)',
                                 'self.statistics.disagreement(d["white"], training, pair)')
        g2_text = replace_exact(g2_text, "head_jacobian(head, cam)", "head_jacobian(head, cam, pair)")
        g2_text = replace_exact(g2_text, "d.update(raw_llr=", "d.update(class_pair=pair, raw_llr=")
    branch = archived_definitions(
        g2_path, ("SCMRDASRPE",), source=g2_text, CFG=config, M=config["margin_calibration"],
        E=config["evidence"], R=config["residual"], P=config["prompt"],
        MedicalSpectrum=spectrum.MedicalSpectrum, AgreementEvidence=evidence.AgreementEvidence,
        head_jacobian=spectrum.head_jacobian, normalized_weights=spectrum.normalized_weights,
        split_jacobian=spectrum.split_jacobian, fixed_logits=spectrum.fixed_logits, class_pair=class_pair,
        module_name=f"e831_scmr_g2_{suffix}", SCMRCheckpointState=SCMRCheckpointState,
    )
    return branch.SCMRDASRPE, spectrum, evidence


def fit_reference(cam, labels, rows, spectrum, fold):
    cfg = source_config()
    evidence_cfg = cfg["evidence"]
    legal = rows.detach().cpu().numpy().copy()
    y = labels.detach().cpu()[rows].long()
    nw = cfg["spectrum"]["eps"]
    tensors = (cam.detach().cpu()[rows].double() - spectrum["center"]) @ spectrum["basis"]
    white = tensors / (spectrum["variance"] + nw).sqrt()
    n_groups = evidence_cfg["inner_folds"]
    seed = evidence_cfg["archived_split_seed_base"] + fold
    inner = StratifiedKFold(n_splits=n_groups, shuffle=True, random_state=seed)
    group = np.zeros(len(rows), dtype=np.int64)
    for bank, (_, held_out) in enumerate(inner.split(np.zeros(len(rows)), y.numpy()), start=1):
        group[held_out] = bank
    bank_index = torch.zeros(len(cam), dtype=torch.long)
    bank_index[rows] = torch.from_numpy(group)
    classes = torch.arange(int(y.max()) + 1)
    mask = ((torch.from_numpy(group)[None, :] != torch.arange(n_groups + 1)[:, None])[:, None, :]
            & (y[None, None, :] == classes[None, :, None]))
    counts = mask.sum(-1)
    ddof = evidence_cfg["ddof"]
    require(bool((counts > ddof).all()), "A legal reference bank lacks a class")
    samples = [white[member] for member in mask.flatten(0, 1)]
    mean = torch.stack([sample.mean(0) for sample in samples]).reshape(*counts.shape, -1)
    variance = torch.stack([sample.var(0, correction=ddof) for sample in samples]).reshape_as(mean)
    _, _, evidence = scmr_components(len(cam), cam.shape[-1])
    _, stable = evidence.stabilize(variance, counts)
    second = variance * ((counts[..., None] - ddof) / counts[..., None].double()) + mean.square()
    nll = (white[:, None, :] - mean[group]).square() / stable[group] + stable[group].log()
    own = nll[torch.arange(len(y)), y]
    competitor = nll.masked_fill(y[:, None, None] == classes[None, :, None], torch.inf).amin(1)
    bound = evidence_cfg["llr_bound"]
    true_llr = bound * ((competitor - own) / (2 * bound)).tanh()
    skill = torch.stack([(true_llr[y == c] / 2).tanh().mean(0) for c in classes]).mean(0)
    return {"legal": legal, "labels": y.numpy().copy(), "group": group, "seed": seed,
            "counts": counts, "mask": mask, "bank": bank_index, "mean": mean, "second": second,
            "stable": stable, "calibration": .5 * (1 + skill), "h": white}


def _historical_objectives(module, head, capture, labels, rows):
    return scmr_objectives()("G2", head, capture, rows, labels[rows], module.spectrum.basis.to(capture["cam"]),
                             module.hard_initial[rows])


class SCMRCheckpointState:
    @property
    def basis(self):
        return self.spectrum.basis

    @property
    def ema_updates(self):
        return self.statistics.ema["updates"]

    historical_objectives = _historical_objectives

    def get_extra_state(self):
        return self.statistics.snapshot()

    def set_extra_state(self, snapshot):
        self.statistics.restore({
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in snapshot.items()
        })

    @torch.no_grad()
    def advance_ema(self, evaluated_cam):
        self.statistics.advance(self.statistics.observation(evaluated_cam))


def make_scmr(stats, prompt_path, nclass, seed):
    original = {"U": stats["basis"], "mu": stats["center"], "lambda": stats["variance"],
                "loglambda_normalized": stats["log_variance"], "nu_normalized": stats["disease_covariance"][0]}
    reference = {**stats["reference"], "bank": stats["bank_index"]}
    require(reference["mean"].shape[1] == nclass, "SCMR classes differ between the head and reference bank")
    cls, _, _ = scmr_components(len(reference["bank"]), stats["basis"].shape[0])
    prompt = Path(prompt_path).read_bytes()
    require(hashlib.sha256(prompt).hexdigest() == "0e6a48bbdfad5e6fcbacfc11122495b334e82bdfd978cee9b6a14b49ddc9d28e",
            "Historical SCMR prompt embedding SHA mismatch")
    text = torch.load(io.BytesIO(prompt), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        module = cls(original, text, [], reference)
    module.register_buffer("hard_initial", stats["hard_initial"].clone())
    return module


@lru_cache(maxsize=1)
def scmr_objectives():
    """Preserve the source loss, including separate bounded candidate gains."""
    config = source_config()
    path = HISTORICAL_ROOT / "common/optimization.py"
    text = verified_text(path, SOURCE_HASHES["common/optimization.py"])
    text = replace_exact(text, "return 0.5 * (x[y == 0].mean() + x[y == 1].mean())",
                         "return torch.stack([x[y == c].mean() for c in y.unique(sorted=True)]).mean()")
    text = replace_exact(text, "sign = 1 - 2 * labels",
                         'pair = d["class_pair"][rows]\n'
                         '    sign = torch.where(labels == pair[:, 0], 1, -1)\n'
                         '    routed = (labels[:, None] == pair).any(dim=1)')
    text = replace_exact(text, "sign * margin(native)", "true_margin(native, labels)")
    text = replace_exact(text, "sign * margin(full)", "true_margin(full, labels)")
    text = replace_exact(text, "sign * margin(candidates)", "true_margin(candidates, labels)")
    text = replace_exact(text, 'balanced(focus * target_loss + cal["reliability_aux_weight"] * reliability_loss, labels)',
                         'balanced(routed * (focus * target_loss + cal["reliability_aux_weight"] * reliability_loss), labels)')
    _, spectral, _ = scmr_components(0, 144)
    return archived_definitions(
        path, ("balanced", "losses"), source=text, CFG=config, B=config["boundary"],
        l2_bound=spectral.l2_bound, native_logits=spectral.native_logits, true_margin=true_class_margin,
    ).losses
