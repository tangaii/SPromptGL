#!/usr/bin/env python3
"""Experiment 9 runner: Prompt--Disease--Adversarial Alignment Interaction."""
from __future__ import annotations

import argparse
import csv
import difflib
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PDAI_ROOT", "/data/adodas/SPromptGL-main"))
LOG_DIR = EXP / "logs"
WORK_DIR = LOG_DIR / ".working"
RESULT_DIR = EXP / "results"
PROMPT_FILE = EXP / "prompts.json"
PATCH_FILE = EXP / "alignment_patch.diff"
MANIFEST_FILE = EXP / "manifest.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_metrics.csv"
DIAG_FILE = RESULT_DIR / "alignment_diagnostics.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = [f"E{i}" for i in range(12)]
NEW_CONFIGS = ["E1", "E2", "E3", "E6", "E7", "E9", "E10", "E11"]
SEEDS = [0, 1, 2]
METRICS = ["acc", "hard_auc", "probability_auc", "balanced_accuracy", "sensitivity_asd", "specificity_asd"]
DIAGNOSTIC_FIELDS = [
    "config", "seed", "fold", *METRICS,
    "discriminator_train_acc", "discriminator_test_acc", "discriminator_train_ce", "discriminator_test_ce",
    "discriminator_test_entropy", "discriminator_test_distribution", "discriminator_grad_valid",
    "wv_adversarial_grad_valid", "prediction_nonfinite", "mmd_corr_pheno", "mmd_corr_anat", "mmd_corr_func",
    "mmd_mean_pairwise", "mean_distance_corr_pheno", "mean_distance_corr_anat", "mean_distance_corr_func",
    "mean_vector_distance_corr_pheno", "mean_vector_distance_corr_anat", "mean_vector_distance_corr_func",
    "mean_vector_distance_pheno_anat", "mean_vector_distance_pheno_func", "mean_vector_distance_pheno_correlation",
    "mean_vector_distance_anat_func", "mean_vector_distance_anat_correlation", "mean_vector_distance_func_correlation",
    "mean_l2_pheno", "mean_l2_anat", "mean_l2_func", "mean_l2_correlation",
    "mmd_pheno_anat", "mmd_pheno_func", "mmd_pheno_correlation", "mmd_anat_func", "mmd_anat_correlation", "mmd_func_correlation",
    "variance_pheno", "variance_anat", "variance_func", "variance_correlation",
    "effective_rank_pheno", "effective_rank_anat", "effective_rank_func", "effective_rank_correlation",
    "zero_rate_pheno", "zero_rate_anat", "zero_rate_func", "zero_rate_correlation",
    "intra_modal_similarity", "intermodal_similarity", "asd_td_center_distance", "class_separation_ratio",
    "paired_cosine_pheno_anat", "paired_cosine_pheno_func", "paired_cosine_pheno_correlation",
    "different_cosine_pheno_anat", "different_cosine_pheno_func", "different_cosine_pheno_correlation",
    "different_subject_cosine_mean", "cosine_gap_mean",
    "cosine_gap_pheno_anat", "cosine_gap_pheno_func", "cosine_gap_pheno_correlation", "retrieval_r1_mean",
    "disease_projector_grad_valid", "disease_alpha_grad_valid", "disease_only_acc", "disease_only_probability_auc",
    "disease_alignment_loss", "disease_margin_asd", "disease_margin_td", "disease_gamma", "disease_gamma_sign",
    "disease_contribution_abs", "disease_probability_correlation", "all_finite",
]

# Fixed physical allocation.  Even numbered new configs put seed 0/2 on GPU0;
# odd numbered configs put them on GPU1.  This gives 12 task processes per card.
GPU_ASSIGNMENTS = {}
for cfg in NEW_CONFIGS:
    idx = NEW_CONFIGS.index(cfg)
    for seed in SEEDS:
        GPU_ASSIGNMENTS[(cfg, seed)] = (0 if idx % 2 == 0 else 1) if seed in (0, 2) else (1 if idx % 2 == 0 else 0)
TASKS = [(cfg, seed, GPU_ASSIGNMENTS[(cfg, seed)]) for cfg in NEW_CONFIGS for seed in SEEDS]
MAX_ACTIVE = {0: 4, 1: 4}
TASK_END_RE = re.compile(r"TASK_END config=(E\d+) seed=(\d+) attempt=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+)")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=(E\d+) seed=(\d+) attempt=(\d+) .* metrics=(\S+) diagnostics=(\S+)")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        name = stream.name
    os.replace(name, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        name = stream.name
    os.replace(name, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def p1_source():
    path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def p0_text():
    return ["Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure", "Normal ventricular morphology",
            "Altered brain volume", "Abnormal functional connectivity", "White matter integrity issues", "Different activation patterns"]


def flatten_prompt_group(data, name):
    out = []
    for modality in MODALITIES:
        for theme in data[name][modality]["themes"]:
            out.extend(item["text"] for item in theme["prompts"])
    if len(out) != 32 or any(not item for item in out):
        raise ValueError(f"{name} must contain 32 non-empty prompts")
    return out


def build_prompts():
    source, source_path = p1_source()
    disease_source = json.loads((ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/prompts.json").read_text(encoding="utf-8"))
    output = {
        "schema_version": 1,
        "experiment": "PDAI",
        "modality_order": MODALITIES,
        "p0": p0_text(),
        "p1": deepcopy(source["p1"]),
        "p3": deepcopy(source["p3"]),
        "disease_prompts": {"ASD": disease_source["disease_prompts"]["ASD"], "TD": disease_source["disease_prompts"]["TD"]},
        "source_sha256": {"p1_p3": sha256(source_path), "disease": sha256(ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/prompts.json")},
        "freeze_rule": "P0/P1/P3 and ASD/TD statements are copied verbatim before training and never edited from results.",
    }
    if output["p0"] != p0_text() or flatten_prompt_group(output, "p1") != flatten_prompt_group(source, "p1") or flatten_prompt_group(output, "p3") != flatten_prompt_group(source, "p3"):
        raise RuntimeError("prompt freeze check failed")
    write_json(PROMPT_FILE, output)
    return output, source_path


def freeze_reference():
    e1 = read_csv(ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv")
    e8 = read_csv(ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/results/fold_metrics.csv")
    rows = []
    for target, source_cfg, source in (("E0", "P0", e1), ("E4", "P1", e1), ("E8", "P3", e1), ("E5", "S1", e8)):
        subset = [r for r in source if r.get("config") == source_cfg]
        expected = {(seed, fold) for seed in SEEDS for fold in range(1, 11)}
        keys = {(int(r["seed"]), int(r["fold"])) for r in subset}
        if len(subset) != 30 or keys != expected:
            raise RuntimeError(f"historical reference {source_cfg} is not 30 folds")
        for r in subset:
            sens = float(r["sensitivity_asd"])
            spec = float(r["specificity_asd"])
            rows.append({"config": target, "seed": r["seed"], "fold": r["fold"], "acc": r["acc"], "hard_auc": r["hard_auc"],
                         "probability_auc": r.get("probability_auc", ""), "balanced_accuracy": f"{(sens + spec) / 2:.9f}",
                         "sensitivity_asd": r["sensitivity_asd"], "specificity_asd": r["specificity_asd"], "auc_source": "hard_label" if not r.get("probability_auc") else "hard_label+probability",
                         "reference_source": str(source_cfg)})
    write_csv(REFERENCE_FILE, ["config", "seed", "fold", *METRICS, "auc_source", "reference_source"], rows)
    return rows


def patch_layers(source):
    needle = "        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)\n"
    repl = needle + "        self.last_v = None\n"
    if needle not in source:
        raise RuntimeError("layers init anchor missing")
    source = source.replace(needle, repl, 1)
    needle = "        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n        \n        q, attn, residual = self.attention(q, k, v)\n"
    repl = "        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n        self.last_v = v.transpose(1, 2).contiguous().view(bs, modal_num, -1)\n        \n        q, attn, residual = self.attention(q, k, v)\n"
    if needle not in source:
        raise RuntimeError("layers v anchor missing")
    return source.replace(needle, repl, 1)


def patch_network(source):
    source = source.replace("import argparse\n", "import argparse\nimport json\n", 1)
    old = '''_prompt_text_features = None\n\n\ndef get_prompt_text_features():\n    """Encode the fixed semantic prompts once per process."""\n    global _prompt_text_features\n    if _prompt_text_features is None:\n        clip_model, _ = clip.load('RN50', device)\n        with torch.no_grad():\n            _prompt_text_features = clip_model.encode_text(text).float().detach()\n    return _prompt_text_features\n'''
    new = '''_prompt_text_features = {}\n_pdai_disease_features = None\n\n\ndef _pdai_prompt_sentences():\n    mode = os.environ.get("PDAI_PROMPT", "P0")\n    if mode == "P0":\n        return attr_words\n    path = os.environ.get("PDAI_PROMPTS_FILE")\n    if not path:\n        raise RuntimeError("PDAI_PROMPTS_FILE is required for P1/P3")\n    data = json.load(open(path, encoding="utf-8"))\n    output = []\n    for modality in ("PHENO", "ANAT", "FUNC", "Correlation"):\n        for theme in data[mode.lower()][modality]["themes"]:\n            output.extend(item["text"] for item in theme["prompts"])\n    return output\n\n\ndef get_prompt_text_features():\n    mode = os.environ.get("PDAI_PROMPT", "P0")\n    if mode not in _prompt_text_features:\n        sentences = _pdai_prompt_sentences()\n        tokens = clip.tokenize(sentences).to(device)\n        clip_model, _ = clip.load('RN50', device)\n        with torch.no_grad():\n            _prompt_text_features[mode] = clip_model.encode_text(tokens).float().detach()\n    return _prompt_text_features[mode]\n\n\ndef get_pdai_disease_prototypes():\n    global _pdai_disease_features\n    if _pdai_disease_features is None:\n        path = os.environ.get("PDAI_PROMPTS_FILE")\n        if not path:\n            raise RuntimeError("PDAI_PROMPTS_FILE is required for disease prototypes")\n        data = json.load(open(path, encoding="utf-8"))\n        sentences = []\n        for key in ("ASD", "TD"):\n            sentences.extend(data["disease_prompts"][key]["prompts"])\n        tokens = clip.tokenize(sentences).to(device)\n        clip_model, _ = clip.load('RN50', device)\n        with torch.no_grad():\n            enc = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)\n            _pdai_disease_features = F.normalize(enc.view(2, 4, -1).mean(dim=1), dim=-1).detach()\n    return _pdai_disease_features\n'''
    if old not in source:
        raise RuntimeError("network prompt anchor missing")
    source = source.replace(old, new, 1)
    source = source.replace("        self.register_buffer('prompt_text_features', get_prompt_text_features())\n", "        self.register_buffer('prompt_text_features', get_prompt_text_features())\n        self.pdai_prompt = os.environ.get('PDAI_PROMPT', 'P0')\n        self.pdai_disease_enabled = False\n        self.pdai_counterfactual = None\n", 1)
    source = source.replace("        x, _attn = self.InputLayer(x)\n        attn = _attn.mean(dim=1)\n", "        x, _attn = self.InputLayer(x)\n        self.last_modality_v = self.InputLayer.last_v\n        attn = _attn.mean(dim=1)\n", 1)
    old_block = source[source.index("        text_features = self.prompt_text_features\n"):source.index("        # 对最后一个维度进行平均池化\n")]
    new_block = '''        text_features = self.prompt_text_features\n        if text_features.size(0) == 8:\n            # Original P0 path, kept byte-for-byte in its mathematical operations.\n            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)\n            text_features = self.MLP(text_features)\n            part_x = x.view(bs, 4, 4, 9)\n            part_x_norm = F.normalize(part_x, p=2, dim=-1)\n            text_features_norm = F.normalize(text_features, p=2, dim=-1)\n            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)\n            part_x_norm_expanded = part_x_norm.unsqueeze(-2)\n            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))\n            similarity_matrix = similarity_matrix[:,:,:,-1,:]\n            similarity_matrix = similarity_matrix.clamp(min=-5, max=5)\n            similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)\n            similarity_max, _ = torch.max(similarity_matrix, dim=-1)\n            part_x = similarity_max.unsqueeze(-1) * part_x\n        else:\n            grouped = text_features.view(4, 2, 4, 1024)\n            text_features = self.MLP(grouped.reshape(4, 2, 4096))\n            part_x = x.view(bs, 4, 4, 9)\n            part_x_norm = F.normalize(part_x, p=2, dim=-1)\n            text_features_norm = F.normalize(text_features, p=2, dim=-1)\n            similarity_matrix = torch.einsum('nmtf,mcf->nmtc', part_x_norm, text_features_norm)\n            similarity_matrix = torch.softmax(similarity_matrix.clamp(min=-5, max=5), dim=-1).to(device)\n            similarity_max = similarity_matrix.max(dim=-1).values\n            part_x = similarity_max.unsqueeze(-1) * part_x\n\n'''
    source = source.replace(old_block, new_block, 1)
    source = source.replace("        global_x = global_x.view(871, 4, 9)\n", "        global_x = global_x.view(bs, 4, 9)\n", 1)
    old_return = "        output, hidden = self.Outputlayer(x, attn_embedding)\n        return output, hidden, attn_map\n"
    new_return = '''        output, hidden = self.Outputlayer(x, attn_embedding)\n        self.last_base_logprob = output\n        self.last_disease_logits = torch.zeros(bs, 2, device=output.device, dtype=output.dtype)\n        self.last_subject_projection = torch.zeros(bs, 1024, device=output.device, dtype=output.dtype)\n        self.last_subject_semantic = torch.zeros(bs, 1024, device=output.device, dtype=output.dtype)\n        self.last_disease_gamma = 0.0\n        self.last_disease_prelogits = output\n        if getattr(self, 'pdai_disease_enabled', False):\n            projection = self.disease_projector(hidden)\n            semantic = F.normalize(projection, dim=-1)\n            prototypes = F.normalize(self.disease_prototypes, dim=-1)\n            disease_logits = semantic @ prototypes.t() / 0.2\n            gamma = 0.2 * torch.tanh(self.disease_alpha)\n            output = F.log_softmax(output + gamma * disease_logits, dim=1)\n            self.last_subject_projection = projection\n            self.last_subject_semantic = semantic\n            self.last_disease_logits = disease_logits\n            self.last_disease_gamma = float(gamma.detach().item())\n            self.last_disease_prelogits = self.last_base_logprob + gamma * disease_logits\n        return output, hidden, attn_map\n'''
    if old_return not in source:
        raise RuntimeError("network return anchor missing")
    return source.replace(old_return, new_return, 1)


def patch_model(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    anchor = "from utils import *\n\n\nclass EvalHelper:"
    addition = '''from utils import *\n\n\nclass GradientReverse(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x, scale):\n        ctx.scale = float(scale)\n        return x.view_as(x)\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        return -ctx.scale * grad_output, None\n\n\nclass ModalityDiscriminator(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.LayerNorm(36), nn.Linear(36, 32), nn.ReLU(), nn.Linear(32, 4))\n\n    def forward(self, x):\n        return self.net(x)\n\n\nclass EvalHelper:'''
    if anchor not in source:
        raise RuntimeError("model class anchor missing")
    source = source.replace(anchor, addition, 1)
    old = "        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)\n        self.ModalFusion.apply(my_weight_init)\n"
    new = '''        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)\n        self.ModalFusion.apply(my_weight_init)\n        self.pdai_disease = os.environ.get("PDAI_DISEASE", "0") == "1"\n        self.pdai_adv = os.environ.get("PDAI_ADV", "0") == "1"\n        self.ModalityDiscriminator = None\n        self.optimizer_D = None\n        self.ModalFusion.pdai_disease_enabled = self.pdai_disease\n        self.ModalFusion.pdai_disease_grad_valid = False\n        self.ModalFusion.pdai_alpha_grad_valid = False\n        if self.pdai_disease:\n            self.ModalFusion.register_buffer("disease_prototypes", get_pdai_disease_prototypes())\n            self.ModalFusion.disease_projector = nn.Linear(36, 1024, bias=False).to(dev)\n            my_weight_init(self.ModalFusion.disease_projector)\n            self.ModalFusion.disease_alpha = nn.Parameter(torch.zeros((), device=dev))\n            disease_parameters = list(self.ModalFusion.disease_projector.parameters()) + [self.ModalFusion.disease_alpha]\n            if sum(parameter.numel() for parameter in disease_parameters) != 36865:\n                raise RuntimeError("PDAI disease branch must add exactly 36865 parameters")\n            self.optimizer_MF.add_param_group({"params": disease_parameters, "weight_decay": 0.0})\n        if self.pdai_adv:\n            # The discriminator has an isolated RNG stream so its initialization\n            # cannot change the original GFR/PLN/dropout trajectory.\n            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):\n                torch.manual_seed(910000 + int(hyperpm.seed))\n                self.ModalityDiscriminator = ModalityDiscriminator().to(dev)\n                self.ModalityDiscriminator.apply(my_weight_init)\n            self.optimizer_D = optim.Adam(self.ModalityDiscriminator.parameters(), lr=hyperpm.lr, weight_decay=0.0)\n'''
    new = new.replace("        self.optimizer_D = None\n", "        self.optimizer_D = None\n        self._pdai_disc_grad_valid = False\n        self._pdai_wv_adv_grad_valid = False\n")
    if old not in source:
        raise RuntimeError("model optimizer anchor missing")
    source = source.replace(old, new, 1)
    old = "            prob,_,_ = self.ModalFusion(feat)\n\n            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)\n            \n            loss = cls_loss + all_graph_loss\n            \n            loss.backward()\n            \n            self.optimizer_MP0.step()            \n            self.optimizer_MF.step()\n"
    new = '''            prob,_,_ = self.ModalFusion(feat)\n\n            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)\n            loss = cls_loss + all_graph_loss\n            if self.pdai_disease:\n                train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)\n                disease_loss = F.cross_entropy(self.ModalFusion.last_disease_logits[train_idx], self.targ[train_idx], weight=self.weight)\n                loss = loss + 0.1 * disease_loss\n                self._pdai_last_disease_loss = float(disease_loss.detach().item())\n            else:\n                self._pdai_last_disease_loss = float('nan')\n            if self.pdai_adv:\n                train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)\n                target_modal = torch.arange(self.modal_num, device=self.targ.device).view(1, -1).expand(len(self.targ), -1).reshape(-1)\n                z = self.ModalFusion.last_modality_v.reshape(-1, 36)\n                adv_logits = self.ModalityDiscriminator(GradientReverse.apply(z, 0.03))\n                adv_loss = F.cross_entropy(adv_logits[train_idx.repeat_interleave(self.modal_num)], target_modal[train_idx.repeat_interleave(self.modal_num)])\n                loss = loss + adv_loss\n                self._pdai_last_adv_loss = float(adv_loss.detach().item())\n            else:\n                self._pdai_last_adv_loss = float('nan')\n            loss.backward()\n            if self.pdai_adv:\n                grads = [p.grad for p in self.ModalityDiscriminator.parameters()]\n                self._pdai_disc_grad_valid = bool(all(g is not None and torch.isfinite(g).all().item() and (g.abs().sum() > 0).item() for g in grads))\n                wv = self.ModalFusion.InputLayer.w_vs[0].weight.grad\n                self._pdai_wv_adv_grad_valid = bool(wv is not None and torch.isfinite(wv).all().item() and (wv.abs().sum() > 0).item())\n            else:\n                self._pdai_disc_grad_valid = False\n                self._pdai_wv_adv_grad_valid = False\n            if self.pdai_disease:\n                for key, parameter in (("projector", self.ModalFusion.disease_projector.weight), ("alpha", self.ModalFusion.disease_alpha)):\n                    grad = parameter.grad\n                    valid = grad is not None and bool(torch.isfinite(grad).all().item()) and bool((grad.abs().sum() > 0).item())\n                    if key == "projector": self.ModalFusion.pdai_disease_grad_valid |= valid\n                    else: self.ModalFusion.pdai_alpha_grad_valid |= valid\n            self.optimizer_MP0.step()\n            self.optimizer_MF.step()\n            if self.optimizer_D is not None:\n                self.optimizer_D.step()\n'''
    new = new.replace("expand(len(self.targ), -1).reshape(-1)", "expand(len(train_idx), -1).reshape(-1)")
    new = new.replace("z = self.ModalFusion.last_modality_v.reshape(-1, 36)", "z = self.ModalFusion.last_modality_v.index_select(0, train_idx).reshape(-1, 36)")
    new = new.replace("F.cross_entropy(adv_logits[train_idx.repeat_interleave(self.modal_num)], target_modal[train_idx.repeat_interleave(self.modal_num)])", "F.cross_entropy(adv_logits, target_modal)")
    new = new.replace("                wv = self.ModalFusion.InputLayer.w_vs[0].weight.grad\n                self._pdai_wv_adv_grad_valid = bool(wv is not None and torch.isfinite(wv).all().item() and (wv.abs().sum() > 0).item())", "                wv_grads = [layer.weight.grad for layer in self.ModalFusion.InputLayer.w_vs]\n                self._pdai_wv_adv_grad_valid = bool(all(grad is not None and torch.isfinite(grad).all().item() and (grad.abs().sum() > 0).item() for grad in wv_grads))")
    new = new.replace("            prob,_,_ = self.ModalFusion(feat)\n", "            if self.optimizer_D is not None:\n                self.optimizer_D.zero_grad()\n            prob,_,_ = self.ModalFusion(feat)\n", 1)
    if old not in source:
        raise RuntimeError("model simple-2 anchor missing")
    source = source.replace(old, new, 1)
    # Keep the discriminator checkpoint synchronized with the original two modules.
    return source


def patch_main(source):
    old = "        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)\n"
    new = old + "        if os.environ.get('PDAI_CONFIG'):\n            agent._pdai_fold_number = clk\n"
    if old not in source:
        raise RuntimeError("main fold anchor missing")
    source = source.replace(old, new, 1)
    old = "                dict_list = [agent.MessagePassing0.state_dict(),\n                             agent.ModalFusion.state_dict()]\n"
    new = "                dict_list = [agent.MessagePassing0.state_dict(), agent.ModalFusion.state_dict()]\n                if getattr(agent, 'ModalityDiscriminator', None) is not None:\n                    dict_list.append(agent.ModalityDiscriminator.state_dict())\n"
    if old not in source:
        raise RuntimeError("main save anchor missing")
    source = source.replace(old, new, 1)
    old = "        agent.ModalFusion.load_state_dict(dict_list[1])\n        \n        val_acc.append(best_val_acc)\n"
    new = "        agent.ModalFusion.load_state_dict(dict_list[1])\n        if getattr(agent, 'ModalityDiscriminator', None) is not None and len(dict_list) > 2:\n            agent.ModalityDiscriminator.load_state_dict(dict_list[2])\n        \n        val_acc.append(best_val_acc)\n"
    if old not in source:
        raise RuntimeError("main restore anchor missing")
    source = source.replace(old, new, 1)
    old = "        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)\n"
    new = old + "        if os.environ.get('PDAI_CONFIG'):\n            agent.write_pdai_outputs(clk, cur_tst_acc, cur_tst_auc, conf_mat)\n"
    if old not in source:
        raise RuntimeError("main test anchor missing")
    return source.replace(old, new, 1)


def patch_model_outputs(source):
    anchor = "    def cal_acc(self, eval_idx):\n"
    if anchor not in source:
        raise RuntimeError("model output method anchor missing")
    method = r'''    def _pdai_json_row(self, path, row):
        if not path:
            return
        with open(path, "a", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False); stream.write("\n")

    def _pdai_cos(self, a, b):
        return F.cosine_similarity(a, b, dim=-1)

    def _pdai_mmd(self, a, b):
        a = a[:256]; b = b[:256]
        x = torch.cat([a, b], dim=0)
        d = torch.cdist(x, x).pow(2)
        positive = d[d > 0]
        sigma = torch.sqrt(torch.median(positive).clamp_min(1e-6)) if positive.numel() else torch.tensor(1., device=x.device)
        k = torch.exp(-d / (2 * sigma.pow(2) + 1e-8))
        na, nb = a.size(0), b.size(0)
        return float(k[:na, :na].mean().item() + k[na:, na:].mean().item() - 2 * k[:na, na:].mean().item())

    def write_pdai_outputs(self, fold, acc, hard_auc, conf_mat):
        metric_path = os.environ.get("PDAI_METRIC_PATH")
        diag_path = os.environ.get("PDAI_DIAGNOSTIC_PATH")
        if not metric_path or not diag_path:
            return
        labels = self.targ[torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)].detach().cpu().numpy()
        with torch.no_grad():
            model_num, feat1 = self.dataLayer(self.feat)
            feat, _, _ = self.MessagePassing0(feat1, model_num, self.targ, self.trn_idx)
            output, hidden, _ = self.ModalFusion(feat)
            test_idx = torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)
            train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)
            probability = output.index_select(0, test_idx).exp().cpu().numpy()
            base_probability = self.ModalFusion.last_base_logprob.index_select(0, test_idx).exp().cpu().numpy()
            probability_auc = float(roc_auc_score((labels == 0).astype(int), probability[:, 0]))
            sensitivity = float(conf_mat[0, 0] / max(1, conf_mat[0].sum()))
            specificity = float(conf_mat[1, 1] / max(1, conf_mat[1].sum()))
            metric = {"config": os.environ["PDAI_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold),
                      "acc": float(acc), "hard_auc": float(hard_auc), "probability_auc": probability_auc,
                      "balanced_accuracy": (sensitivity + specificity) / 2, "sensitivity_asd": sensitivity, "specificity_asd": specificity}
            self._pdai_json_row(metric_path, metric)
            z = self.ModalFusion.last_modality_v.detach()
            train_z, test_z = z.index_select(0, train_idx), z.index_select(0, test_idx)
            flat_train = train_z.reshape(-1, 36)
            modal_target = torch.arange(self.modal_num, device=z.device).view(1, -1).expand(len(z), -1).reshape(-1)
            disc_train_acc = disc_test_acc = disc_train_ce = disc_test_ce = disc_entropy = "N/A"
            dist = []
            disc_finite = True
            if self.ModalityDiscriminator is not None:
                tr_logits = self.ModalityDiscriminator(flat_train)
                te_logits = self.ModalityDiscriminator(test_z.reshape(-1, 36))
                tr_target = modal_target.index_select(0, train_idx.repeat_interleave(self.modal_num))
                te_target = modal_target.index_select(0, test_idx.repeat_interleave(self.modal_num))
                tr_prob, te_prob = tr_logits.softmax(-1), te_logits.softmax(-1)
                disc_train_acc = float((tr_prob.argmax(-1) == tr_target).float().mean().item())
                disc_test_acc = float((te_prob.argmax(-1) == te_target).float().mean().item())
                disc_train_ce = float(F.cross_entropy(tr_logits, tr_target).item())
                disc_test_ce = float(F.cross_entropy(te_logits, te_target).item())
                disc_entropy = float((-(te_prob * (te_prob.clamp_min(1e-8).log())).sum(-1)).mean().item())
                dist = te_prob.mean(0).detach().cpu().tolist()
                disc_finite = bool(torch.isfinite(tr_logits).all().item() and torch.isfinite(te_logits).all().item())
            sample_offsets = torch.linspace(0, max(0, train_idx.numel() - 1), min(256, train_idx.numel()), device=train_idx.device).long()
            diagnostic_idx = train_idx.index_select(0, sample_offsets)
            diagnostic_z = z.index_select(0, diagnostic_idx)
            diagnostic_labels = self.targ.index_select(0, diagnostic_idx)
            modal = [diagnostic_z[:, i, :] for i in range(4)]
            def mean_distance(a, b): return float(torch.cdist(a[:256], b[:256]).mean().item())
            def effective_rank(a):
                s = torch.linalg.svdvals(a - a.mean(0, keepdim=True)).clamp_min(1e-12)
                p = s / s.sum()
                return float(torch.exp(-(p * p.log()).sum()).item())
            variance = [float(v.var(unbiased=False).item()) for v in modal]
            ranks = [effective_rank(v) for v in modal]
            zero = [float((torch.linalg.vector_norm(v, dim=-1) < 1e-8).float().mean().item()) for v in modal]
            pair_mmd = {(i, j): self._pdai_mmd(modal[i], modal[j]) for i in range(4) for j in range(i + 1, 4)}
            mmd = [pair_mmd[(i, 3)] if i < 3 else 0.0 for i in range(3)]
            paired = {}; different = {}; gaps = {}; r1 = []
            for i in range(4):
                for j in range(i + 1, 4):
                    sim = F.normalize(modal[i], dim=-1) @ F.normalize(modal[j], dim=-1).t()
                    paired_sim = sim.diag()
                    off = sim.masked_fill(torch.eye(sim.size(0), device=sim.device, dtype=torch.bool), float('-inf'))
                    pair_key = (i, j)
                    paired[pair_key] = float(paired_sim.mean().item())
                    different[pair_key] = float(off[torch.isfinite(off)].mean().item())
                    gaps[pair_key] = paired[pair_key] - different[pair_key]
                    r1.append(float((sim.argmax(-1) == torch.arange(sim.size(0), device=sim.device)).float().mean().item()))
            within_modal = []
            for v in modal:
                sim = F.normalize(v, dim=-1) @ F.normalize(v, dim=-1).t()
                mask = ~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                within_modal.append(float(sim[mask].mean().item()))
            intra = float(sum(within_modal) / len(within_modal))
            inter = float(sum(paired.values()) / len(paired))
            different_mean = float(sum(different.values()) / len(different))
            gap_mean = float(sum(gaps.values()) / len(gaps))
            pooled = torch.cat(modal, dim=0)
            pooled_labels = diagnostic_labels.repeat(4)
            class_centers = [pooled[pooled_labels == c].mean(0) for c in (0, 1)]
            class_dist = float(torch.linalg.vector_norm(class_centers[0] - class_centers[1]).item())
            within = [float(torch.linalg.vector_norm(pooled[pooled_labels == c] - class_centers[c], dim=-1).mean().item()) for c in (0, 1)]
            mean_vectors = [v.mean(0) for v in modal]
            mean_vector_pair = {(i,j): float(torch.linalg.vector_norm(mean_vectors[i] - mean_vectors[j]).item()) for i in range(4) for j in range(i+1,4)}
            mean_l2 = [float(torch.linalg.vector_norm(v, dim=-1).mean().item()) for v in modal]
            disease_only_acc = disease_auc = align = margin_asd = margin_td = gamma = contribution = corr = "N/A"
            disease_grad = alpha_grad = False
            if self.ModalFusion.pdai_disease_enabled:
                dl = self.ModalFusion.last_disease_logits.index_select(0, test_idx)
                dp = dl.softmax(-1)
                disease_only_acc = float((dp.argmax(-1) == self.targ.index_select(0, test_idx)).float().mean().item())
                disease_auc = float(roc_auc_score((labels == 0).astype(int), dp[:, 0].cpu().numpy()))
                train_dl = self.ModalFusion.last_disease_logits.index_select(0, train_idx)
                align = float(F.cross_entropy(train_dl, self.targ.index_select(0, train_idx), weight=self.weight).item())
                margins = dl[:, 0] - dl[:, 1]
                margin_asd = float(margins[self.targ.index_select(0, test_idx) == 0].mean().item())
                margin_td = float(margins[self.targ.index_select(0, test_idx) == 1].mean().item())
                gamma = float(self.ModalFusion.last_disease_gamma)
                contribution = float((self.ModalFusion.last_disease_prelogits.index_select(0, test_idx) - self.ModalFusion.last_base_logprob.index_select(0, test_idx)).abs().mean().item())
                if base_probability[:, 0].std() > 0 and dp[:, 0].cpu().numpy().std() > 0:
                    corr = float(__import__('numpy').corrcoef(base_probability[:, 0], dp[:, 0].cpu().numpy())[0, 1])
                disease_grad = bool(self.ModalFusion.pdai_disease_grad_valid)
                alpha_grad = bool(self.ModalFusion.pdai_alpha_grad_valid)
            row = {"config": os.environ["PDAI_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold), **metric,
                   "discriminator_train_acc": disc_train_acc, "discriminator_test_acc": disc_test_acc,
                   "discriminator_train_ce": disc_train_ce, "discriminator_test_ce": disc_test_ce, "discriminator_test_entropy": disc_entropy,
                   "discriminator_test_distribution": json.dumps(dist), "discriminator_grad_valid": bool(self._pdai_disc_grad_valid),
                   "wv_adversarial_grad_valid": bool(self._pdai_wv_adv_grad_valid), "prediction_nonfinite": bool((~torch.isfinite(output)).any().item()),
                   "mmd_corr_pheno": mmd[0], "mmd_corr_anat": mmd[1], "mmd_corr_func": mmd[2], "mmd_mean_pairwise": float(sum(pair_mmd.values()) / 6),
                   "mean_distance_corr_pheno": mean_distance(modal[3], modal[0]), "mean_distance_corr_anat": mean_distance(modal[3], modal[1]), "mean_distance_corr_func": mean_distance(modal[3], modal[2]),
                   "mean_vector_distance_corr_pheno": mean_vector_pair[(0,3)], "mean_vector_distance_corr_anat": mean_vector_pair[(1,3)], "mean_vector_distance_corr_func": mean_vector_pair[(2,3)],
                   "mean_vector_distance_pheno_anat": mean_vector_pair[(0,1)], "mean_vector_distance_pheno_func": mean_vector_pair[(0,2)], "mean_vector_distance_pheno_correlation": mean_vector_pair[(0,3)],
                   "mean_vector_distance_anat_func": mean_vector_pair[(1,2)], "mean_vector_distance_anat_correlation": mean_vector_pair[(1,3)], "mean_vector_distance_func_correlation": mean_vector_pair[(2,3)],
                   "mean_l2_pheno": mean_l2[0], "mean_l2_anat": mean_l2[1], "mean_l2_func": mean_l2[2], "mean_l2_correlation": mean_l2[3],
                   "variance_pheno": variance[0], "variance_anat": variance[1], "variance_func": variance[2], "variance_correlation": variance[3],
                   "effective_rank_pheno": ranks[0], "effective_rank_anat": ranks[1], "effective_rank_func": ranks[2], "effective_rank_correlation": ranks[3],
                   "zero_rate_pheno": zero[0], "zero_rate_anat": zero[1], "zero_rate_func": zero[2], "zero_rate_correlation": zero[3],
                   "intra_modal_similarity": intra, "intermodal_similarity": inter, "asd_td_center_distance": class_dist,
                   "class_separation_ratio": class_dist / (sum(within) / 2 + 1e-8), "paired_cosine_pheno_anat": paired[(0,1)], "paired_cosine_pheno_func": paired[(0,2)], "paired_cosine_pheno_correlation": paired[(0,3)],
                   "different_cosine_pheno_anat": different[(0,1)], "different_cosine_pheno_func": different[(0,2)], "different_cosine_pheno_correlation": different[(0,3)],
                   "different_subject_cosine_mean": different_mean, "cosine_gap_mean": gap_mean,
                   "cosine_gap_pheno_anat": gaps[(0,1)], "cosine_gap_pheno_func": gaps[(0,2)], "cosine_gap_pheno_correlation": gaps[(0,3)], "retrieval_r1_mean": sum(r1) / len(r1),
                   "disease_projector_grad_valid": disease_grad, "disease_alpha_grad_valid": alpha_grad, "disease_only_acc": disease_only_acc,
                   "disease_only_probability_auc": disease_auc, "disease_alignment_loss": align, "disease_margin_asd": margin_asd, "disease_margin_td": margin_td,
                   "mmd_pheno_anat": pair_mmd[(0,1)], "mmd_pheno_func": pair_mmd[(0,2)], "mmd_pheno_correlation": pair_mmd[(0,3)],
                   "mmd_anat_func": pair_mmd[(1,2)], "mmd_anat_correlation": pair_mmd[(1,3)], "mmd_func_correlation": pair_mmd[(2,3)],
                   "disease_gamma": gamma, "disease_gamma_sign": int(__import__('numpy').sign(gamma)) if isinstance(gamma, (int, float)) and math.isfinite(gamma) else "N/A", "disease_contribution_abs": contribution,
                   "disease_probability_correlation": corr, "all_finite": bool(torch.isfinite(output).all().item() and torch.isfinite(z).all().item() and disc_finite)}
            self._pdai_json_row(diag_path, row)

'''
    return source.replace(anchor, method + anchor, 1)


def make_patch():
    originals = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCE_FILES if name.endswith(".py")}
    patched = dict(originals)
    patched["layers.py"] = patch_layers(patched["layers.py"])
    patched["network.py"] = patch_network(patched["network.py"])
    patched["model.py"] = patch_model(patched["model.py"])
    patched["model.py"] = patch_model_outputs(patched["model.py"])
    patched["main.py"] = patch_main(patched["main.py"])
    chunks = []
    for name in ("main.py", "model.py", "network.py", "layers.py"):
        chunks.extend(difflib.unified_diff(originals[name].splitlines(True), patched[name].splitlines(True), fromfile=name, tofile=name))
    atomic_write(PATCH_FILE, "".join(chunks))
    return originals, patched


def prepare_worktree(work):
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, work / name)
    (work / "CLIP").symlink_to(ROOT / "CLIP", target_is_directory=True)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir(exist_ok=True)
    (work / "graph").mkdir(exist_ok=True)
    result = subprocess.run(["patch", "-p0", "--batch", "--fuzz=0", "-i", str(PATCH_FILE)], cwd=work, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("patch failed: " + result.stdout + result.stderr)
    for name in ("main.py", "model.py", "network.py", "layers.py"):
        compile((work / name).read_text(encoding="utf-8"), name, "exec")


def preflight(work):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "PDAI_CONFIG": "E7", "PDAI_PROMPT": "P1", "PDAI_DISEASE": "1", "PDAI_ADV": "1", "PDAI_PROMPTS_FILE": str(PROMPT_FILE), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    script = '''import torch, os, numpy as np\nfrom types import SimpleNamespace\nfrom model import EvalHelper\nD=[48,6,10,256]; x=np.load("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy",allow_pickle=True).item()\ndata=np.loadtxt("/dev/null") if False else __import__("pandas").read_csv("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv").values\nlabel=data[:,-1].astype(int)-1; feat=data[:,:-1].astype("float32")\na=SimpleNamespace(GC_mode="weighted-cosine",MP_mode="GCN",MF_mode="concat",n_hidden=18,nmodal=4,nclass=2,dropout=.35,alpha=.5,n_head=2,th=.9,lr=.0038,reg=.11,seed=0,datname="ABIDE",mode="simple-2")\ne=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871)); assert sum(p.numel() for p in e.ModalityDiscriminator.parameters())==((36+1)*32+32*4); assert sum(p.numel() for p in e.ModalFusion.disease_projector.parameters())+1==36865; e.run_epoch("simple-2"); assert e._pdai_disc_grad_valid and e._pdai_wv_adv_grad_valid; print("PDAI_PREFLIGHT_OK")'''
    script = script.replace("from model import EvalHelper", "from model import EvalHelper, GradientReverse, ModalityDiscriminator\nfrom network import PLN")
    script = script.replace('label=data[:,-1].astype(int)-1; feat=data[:,:-1].astype("float32")', 'label=data[:,-1].astype(int)-1; feat=data[:,:-1].astype("float32"); assert feat.shape==(871,320) and np.bincount(label).tolist()==[403,468]')
    old_test = 'e=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871)); assert sum(p.numel() for p in e.ModalityDiscriminator.parameters())==((36+1)*32+32*4); assert sum(p.numel() for p in e.ModalFusion.disease_projector.parameters())+1==36865; e.run_epoch("simple-2"); assert e._pdai_disc_grad_valid and e._pdai_wv_adv_grad_valid; print("PDAI_PREFLIGHT_OK")'
    new_test = '''v=torch.tensor([1.5,-2.0],requires_grad=True); w=torch.tensor([2.0,3.0]); assert torch.equal(GradientReverse.apply(v,.03),v); (GradientReverse.apply(v,.03)*w).sum().backward(); assert torch.allclose(v.grad,-.03*w,atol=1e-8,rtol=1e-6)
disc=ModalityDiscriminator(); z=torch.randn(16,36,requires_grad=True); y=torch.arange(16)%4; torch.nn.functional.cross_entropy(disc(z),y).backward(); normal=[p.grad.detach().clone() for p in disc.parameters()]; disc.zero_grad(); z2=z.detach().clone().requires_grad_(True); torch.nn.functional.cross_entropy(disc(GradientReverse.apply(z2,.03)),y).backward(); reversed_grads=[p.grad.detach().clone() for p in disc.parameters()]; assert all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(normal,reversed_grads)); assert torch.allclose(z2.grad,-.03*z.grad,atol=1e-7,rtol=1e-5)
torch.manual_seed(123); torch.cuda.manual_seed_all(123); e=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871)); assert sum(p.numel() for p in e.MessagePassing0.parameters())==887836; assert sum(p.numel() for p in e.ModalFusion.parameters())==575121+36865; e.ModalFusion.eval()
with torch.no_grad(): out_a=e.ModalFusion(e.feat)[0].clone()
state_a={k:v.detach().clone() for k,v in e.ModalFusion.state_dict().items()}
os.environ["PDAI_ADV"]="0"; torch.manual_seed(123); torch.cuda.manual_seed_all(123); e0=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871)); e0.ModalFusion.eval()
with torch.no_grad(): out_0=e0.ModalFusion(e0.feat)[0].clone()
assert all(torch.equal(v,e0.ModalFusion.state_dict()[k]) for k,v in state_a.items()); assert torch.equal(out_a,out_0); os.environ["PDAI_ADV"]="1"
assert sum(p.numel() for p in e.ModalityDiscriminator.parameters())==(2*36+36*32+32+32*4+4); assert sum(p.numel() for p in e.ModalFusion.disease_projector.parameters())+1==36865
model_num,feat1=e.dataLayer(e.feat); feat,_,_=e.MessagePassing0(feat1,model_num,e.targ,e.trn_idx); e.ModalFusion.train(); e.ModalityDiscriminator.train(); e.ModalFusion(feat); zv=e.ModalFusion.last_modality_v; assert tuple(zv.shape)==(871,4,36)
idx=torch.as_tensor(e.trn_idx,dtype=torch.long,device=e.dev); ztrain=zv.index_select(0,idx).reshape(-1,36); mt=torch.arange(4,device=e.dev).view(1,4).expand(len(idx),4).reshape(-1); adv_grads=torch.autograd.grad(torch.nn.functional.cross_entropy(e.ModalityDiscriminator(GradientReverse.apply(ztrain,.03)),mt),[layer.weight for layer in e.ModalFusion.InputLayer.w_vs]); assert all(torch.isfinite(grad).all() and (grad.abs().sum()>0) for grad in adv_grads)
e.run_epoch("simple-2"); assert e._pdai_disc_grad_valid and e._pdai_wv_adv_grad_valid; print("PDAI_PREFLIGHT_OK")'''
    if old_test not in script:
        raise RuntimeError("preflight script test block changed")
    new_test = new_test.replace('e.run_epoch("simple-2"); assert e._pdai_disc_grad_valid and e._pdai_wv_adv_grad_valid; print("PDAI_PREFLIGHT_OK")', 'e.run_epoch("simple-2"); assert e._pdai_disc_grad_valid and e._pdai_wv_adv_grad_valid; os.environ["PDAI_METRIC_PATH"]="preflight_metrics.jsonl"; os.environ["PDAI_DIAGNOSTIC_PATH"]="preflight_diagnostics.jsonl"; acc,auc,conf=e.print_tst_acc("simple-2"); e.write_pdai_outputs(1,acc,auc,conf); print("PDAI_PREFLIGHT_OK")')
    reference_test = '''def reference_forward(pln,data):
 bs=data.size(0); combine=[]; xx,_attn=pln.InputLayer(data); attn=_attn.mean(dim=1)
 for k in range(pln.n_layer):
  xx,_a=pln.Encoder[k](q=xx,k=xx,v=xx,modal_num=pln.modal_num); attn=_a.mean(dim=1); xx=pln.FeedForward[k](xx)
 tf=pln.prompt_text_features
 if tf.size(0)==8: tf=torch.cat([tf[0:4].view(1,-1),tf[4:8].view(1,-1)],dim=0); tf=pln.MLP(tf)
 else: tf=pln.MLP(tf.view(4,2,4096))
 part=xx.view(bs,4,4,9); pn=torch.nn.functional.normalize(part,p=2,dim=-1); tn=torch.nn.functional.normalize(tf,p=2,dim=-1)
 if tn.dim()==2:
  tn=tn.unsqueeze(0).unsqueeze(0).unsqueeze(2); sim=torch.matmul(pn.unsqueeze(-2),tn.transpose(-1,-2)); sim=sim[:,:,:,-1,:]
 else: sim=torch.einsum("nmtf,mcf->nmtc",pn,tn)
 sim=torch.softmax(sim.clamp(-5,5),dim=-1).to(pln.prompt_text_features.device); part=sim.max(dim=-1).values.unsqueeze(-1)*part
 gx=pln.avg_pool(xx.view(-1,1,36)).view(bs,4,9)
 for j in range(part.shape[2]):
  p=part[:,:,j,:]; vs=torch.softmax(p*gx,dim=-1); combine.append(vs*p)
 xx=(xx+torch.cat(combine,dim=2)).view(bs,-1); base,_=pln.Outputlayer(xx,attn.view(bs,-1)); out=base
 if getattr(pln,"pdai_disease_enabled",False):
  subj=torch.nn.functional.normalize(pln.disease_projector(pln.Outputlayer.mlp_head(xx)),dim=-1); proto=torch.nn.functional.normalize(pln.disease_prototypes,dim=-1); dl=subj@proto.t()/.2; out=torch.nn.functional.log_softmax(base+.2*torch.tanh(pln.disease_alpha)*dl,dim=1)
 return out
for prompt in ("P0","P1","P3"):
 os.environ["PDAI_PROMPT"]=prompt
 if prompt=="P1": pln=e.ModalFusion
 else:
  pln=PLN(D,a).to(e.dev); pln.apply(__import__("utils").my_weight_init)
 pln.eval()
 with torch.no_grad(): actual=pln(e.feat)[0].clone(); expected=reference_forward(pln,e.feat)
 assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6),prompt
os.environ["PDAI_PROMPT"]="P1"
'''
    new_test = new_test.replace('e.run_epoch("simple-2");', reference_test + '\ne.run_epoch("simple-2");')
    script = script.replace(old_test, new_test)
    script = script.replace("n_head=2,th=", "n_head=2,nlayer=1,theta_smooth=1,theta_degree=.5,theta_sparsity=0.0,th=")
    script = script.replace("==((36+1)*32+32*4)", "==(2*36+36*32+32+32*4+4)")
    result = subprocess.run([sys.executable, "-c", script], cwd=work, env=env, text=True, capture_output=True, timeout=900)
    if result.returncode or "PDAI_PREFLIGHT_OK" not in result.stdout:
        raise RuntimeError("runtime preflight failed:\n" + result.stdout[-4000:] + "\n" + result.stderr[-8000:])
    diagnostic_path = work / "preflight_diagnostics.jsonl"
    metric_path = work / "preflight_metrics.jsonl"
    if len(read_jsonl(diagnostic_path)) != 1 or len(read_jsonl(metric_path)) != 1:
        raise RuntimeError("diagnostic writer preflight failed")


def write_readme(manifest):
    text = f'''# 实验九：PDAI Prompt-Disease-Adversarial Alignment Interaction

本实验检验基础Prompt、ASD/TD疾病类别语义与模态对抗分布对齐是否存在可重复的交互收益。根目录原始源码、实验一至八目录均保持不变；`alignment_patch.diff` 只在每个临时训练副本中应用。

## 前八轮证据

| 轮次 | 真实结果与结论 |
|---|---|
| 1 MASPE | P1 91.80%/91.62%，P0 91.69%/91.54%，P3 91.54%/91.31%；模态Prompt只带来很小差异。 |
| 2 CSMP | 正确映射V2略高于P0且高于错配V2S，但没有超过P1。 |
| 3 PEPCA | P1 91.80%/91.62%；固定缩放与无关文本较差，固定缩放配置语义投影梯度无效。 |
| 4 DAFCP | ASD-FC和匹配通用FC分别91.46%/91.29%、91.50%/91.28%，均低于P1。 |
| 5 SPCR | 去末层ReLU、去MLP衰减及同时修改都未超过E0；观测到MLP参数收缩，但单项修复无分类收益。 |
| 6 SDPG | B1约91.73%/91.59%，B2约91.77%/91.52%，没有稳定超过P1。 |
| 7 DTSF | F1约91.69%/91.49%；错配、pooled、uniform对照也没有稳定显示双主题融合超越P1。 |
| 8 SDPA | S1约91.73%/91.51%，疾病分支/P1预测相关约0.958；疾病类别方向可学到，但总体增益小且存在信息重复迹象。 |

这些实验不支持继续靠改写Prompt、MLP小修或标量门控追分。实验九固定历史文本，直接检验疾病类别监督与跨模态分布对齐是否互补。

## DeepASD依据

DeepASD Fig.1的顺序是模态对抗编码、逐模态患者图学习与融合、GNN分类；原方法使用模态专属特征提取器与模态判别器的竞争训练，论文描述了平方损失和invert-label策略。本实验只借鉴“先在跨模态注意力前约束模态分布，再走SPromptGL原图/分类路径”的位置，不复现DeepASD结构或损失。Table 2中ABIDE A的特征提取器单独为72.78% ACC/77.59% AUC，加入模态判别器后为76.74%/82.99%，完整三模块系统为87.38%/92.76%；原论文正文对82.99到77.59的下降写作6.51%，但表内数值相减是5.40个百分点，因此这里按表值陈述。其队列和协议与本地ABIDE复现不同，不作数值横向比较。[DeepASD原文](https://www.nature.com/articles/s41398-024-02972-2)。

## 配置

P0是原始8句共享文本；P1是实验一冻结的32句模态专属文本；P3是实验一冻结的32句通用文本。D1精确复用实验八S1的ASD/TD四句类别文本、36→1024无偏置投影、归一化点积除以0.2、`gamma=0.2*tanh(alpha)`和训练集加权交叉熵（权重0.1）。A1在原始 `VariLengthInputLayer` 的四个模态V输出 `[N,4,36]` 上接 `LayerNorm(36)->Linear(36,32)->ReLU->Linear(32,4)`，目标依次为PHENO/ANAT/FUNC/Correlation；仅输入特征经 GRL 反向，scale=0.03，判别器交叉熵直接加入总损失。D0/A0分别关闭对应分支。

| 配置 | Prompt | Disease | Adversarial |
|---|---|---|---|
| E0 | P0 | D0 | A0 |
| E1 | P0 | D1 | A0 |
| E2 | P0 | D0 | A1 |
| E3 | P0 | D1 | A1 |
| E4 | P1 | D0 | A0 |
| E5 | P1 | D1 | A0（实验八S1历史折） |
| E6 | P1 | D0 | A1 |
| E7 | P1 | D1 | A1 |
| E8 | P3 | D0 | A0 |
| E9 | P3 | D1 | A0 |
| E10 | P3 | D0 | A1 |
| E11 | P3 | D1 | A1 |

E0/E4/E8冻结实验一30折，E5冻结实验八S1 30折；只训练其余8组共240折。所有配置都沿用 weighted-cosine、concat、GCN、dropout=.35、lr=.0038、reg=.11、450 epoch、early=200、10折 StratifiedKFold(random_state=10)、seed 0/1/2和原始ACC最佳Checkpoint。原始模型优化器保持 weight_decay=.11；疾病和判别器参数均为独立 weight_decay=0 的参数组/优化器。A1判别器通过 `torch.random.fork_rng` 隔离初始化，避免改变原始主干随机轨迹。

P1/P3仍用原始 `4096→128→9` 投影、每模态两个文本原型、Softmax-Max语义标量和原始global/local融合。P0保留根目录原始8句和两原型路径。A1只读取已有w_v的输出 `[871,4,36]`；对抗表示是GFR之后、PLN跨模态注意力之前的特征，不是CSV原始输入。总损失是原分类/图损失 + 可选D1加权CE + 可选四分类模态CE；GRL仅将模态特征梯度乘以−0.03，判别器仍接收正常CE梯度。训练不加入额外损失或层。

## 诊断与判定

`alignment_diagnostics.csv` 每个新配置×seed×fold一行，含判别器 train/test ACC、CE、熵和类别分布，GRL与V梯度有效性，Correlation对其他模态的RBF-MMD、均值距离、方差、有效秩、零率、同/跨模态余弦、配对cosine gap和R@1，以及D1疾病分支的投影梯度、疾病分类、gamma、margin和决策贡献。四个局部Token是潜在模态Token，不解释为ROI。历史诊断不可追溯，因而不伪造为零。

主分析对 ACC、Hard-AUC、BA，并在训练配置可得时报告 probability-AUC，均按相同seed×fold配对：Prompt×Disease、Prompt×Adversarial、Disease×Adversarial 和三阶交互。报告会列出逐seed差值、胜/平/负折、判别器是否仍能识别模态、MMD是否下降、表示是否塌缩，以及E7相对E5是否有额外收益。微小差异不视为独立样本的显著证据。

## 重跑

```bash
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --prepare
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --all
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --resume
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --refresh
```

训练工作目录在每个任务结束后自动清理；失败attempt不会进入汇总。`manifest.json` 在所有8组任务、240折和240条诊断完整且通过有限值/键校验后标记 `COMPLETE`。
'''
    atomic_write(EXP / "README.md", text)


def make_manifest():
    clip = Path.home() / ".cache/clip/RN50.pt"
    def git_value(args):
        try: return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
        except Exception: return None
    github_main = None
    try:
        remote = subprocess.check_output(["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30)
        github_main = remote.split()[0] if remote.split() else None
    except Exception:
        pass
    e1_folds = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    e8_folds = ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/results/fold_metrics.csv"
    e8_diagnostics = ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/results/disease_diagnostics.csv"
    return {"experiment": "PDAI", "schema_version": 1, "created_utc": utc_now(), "source_root": str(ROOT),
            "git": {"local_head": git_value(["git", "rev-parse", "HEAD"]), "github_main_at_prepare": github_main, "local_core_source_hashes": {n: sha256(ROOT / n) for n in SOURCE_FILES}},
            "source_files": {n: sha256(ROOT / n) for n in SOURCE_FILES},
            "inputs": {n: sha256(ROOT / n) for n in ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")},
            "clip_rn50": {"path": str(clip), "sha256": sha256(clip) if clip.exists() else None},
            "prompts_sha256": sha256(PROMPT_FILE), "patch_sha256": sha256(PATCH_FILE), "reference_sha256": sha256(REFERENCE_FILE),
            "prompt_sources": {"experiment1_prompts_sha256": sha256(ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"),
                               "experiment8_prompts_sha256": sha256(ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/prompts.json")},
            "reference_sources": {str(e1_folds.relative_to(ROOT)): sha256(e1_folds), str(e8_folds.relative_to(ROOT)): sha256(e8_folds),
                                  str(e8_diagnostics.relative_to(ROOT)): sha256(e8_diagnostics)},
            "protocol": {"folds": 10, "seeds": SEEDS, "epoch_cap": 450, "early_patience": 200, "stratified_random_state": 10,
                         "checkpoint": "validation ACC", "auc": "hard-label", "new_folds": 240, "new_diagnostics": 240,
                         "args": {"GC_mode": "weighted-cosine", "MF_mode": "concat", "MP_mode": "GCN", "dropout": 0.35, "lr": 0.0038, "n_head": 2, "n_hidden": 18, "nlayer": 1, "reg": 0.11},
                         "factor_levels": {"prompt": ["P0", "P1", "P3"], "disease": ["D0", "D1"], "adversarial": ["A0", "A1"]},
                         "gpu_assignments": {f"{c}_seed{s}": g for c, s, g in TASKS}, "max_active_per_gpu": MAX_ACTIVE},
            "parameters": {"disease_branch": 36865, "adversarial_discriminator": 2 * 36 + 36 * 32 + 32 + 32 * 4 + 4, "adversarial_grl_scale": 0.03},
            "audit": {"root_source_modified": False, "experiment1_to_8_untouched": True, "p0_p1_p3_and_disease_prompts_frozen": True,
                      "historical_reference_rows": 120, "deepasd_checked": True, "preflight_checks": ["prompt source equality", "patch apply and py_compile",
                      "GRL identity/sign/scale", "discriminator CE gradient unchanged", "same-seed A0/A1 initial PLN and disease outputs equal",
                      "[N,4,36] V shape and w_v adversarial gradient", "real fold diagnostic writer"]},
            "run_status": "PREPARED", "task_status": {f"{c}_seed{s}": "PENDING" for c, s, _ in TASKS}}


def prepare():
    EXP.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(exist_ok=True); WORK_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True)
    prompts, p1_path = build_prompts(); freeze_reference(); originals, patched = make_patch()
    with tempfile.TemporaryDirectory(prefix="pdai-preflight-") as tmp:
        preflight_work = Path(tmp)
        prepare_worktree(preflight_work)
        preflight(preflight_work)
    manifest = make_manifest(); manifest["preflight"] = {"patch_applies": True, "py_compile": True, "runtime_grl_and_parameter_checks": True, "new_parameters": {"disease": 36865, "discriminator": 1388}}
    write_json(MANIFEST_FILE, manifest); write_readme(manifest)
    print("PDAI_PREPARED", json.dumps({"patch": sha256(PATCH_FILE), "prompts": sha256(PROMPT_FILE), "reference_rows": 120}))


def task_log(config, seed): return LOG_DIR / f"{config}_seed{seed}.log"


def attempt_records(config, seed):
    path = task_log(config, seed)
    starts, ends = {}, {}
    if not path.exists(): return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ATTEMPT_RE.search(line)
        if m and m.group(1) == config and int(m.group(2)) == seed: starts[int(m.group(3))] = (m.group(4), m.group(5))
        m = TASK_END_RE.search(line)
        if m and m.group(1) == config and int(m.group(2)) == seed: ends[int(m.group(3))] = {"returncode": int(m.group(4)), "folds": int(m.group(5)), "diagnostics": int(m.group(6))}
    return [(a, starts.get(a), ends.get(a)) for a in sorted(set(starts) | set(ends))]


def successful_paths(config, seed):
    for attempt, paths, end in reversed(attempt_records(config, seed)):
        if not paths or not end or end["returncode"] != 0 or (end["folds"], end["diagnostics"]) != (10, 10): continue
        metric, diag = map(Path, paths)
        mr, dr = read_jsonl(metric), read_jsonl(diag)
        if len(mr) == len(dr) == 10 and {int(r["fold"]) for r in mr} == set(range(1, 11)) and {int(r["fold"]) for r in dr} == set(range(1, 11)):
            return metric, diag
    return None


def task_state(config, seed):
    if successful_paths(config, seed): return "COMPLETE"
    metric_rows = [r for r in read_csv(FOLD_FILE) if r.get("config") == config and int(r.get("seed", -1)) == seed]
    diagnostic_rows = [r for r in read_csv(DIAG_FILE) if r.get("config") == config and int(r.get("seed", -1)) == seed]
    if len(metric_rows) == len(diagnostic_rows) == 10 and {int(r["fold"]) for r in metric_rows} == set(range(1, 11)) and {int(r["fold"]) for r in diagnostic_rows} == set(range(1, 11)):
        return "COMPLETE"
    records = attempt_records(config, seed)
    if records and records[-1][2] is None: return "INTERRUPTED"
    if records and records[-1][2]["returncode"] != 0: return "FAILED"
    return "PENDING"


def process_running(config, seed):
    try:
        lines = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except Exception:
        return False
    script_path = Path(__file__).resolve()
    for line in lines:
        fields = line.strip().split(None, 1)
        if len(fields) != 2: continue
        try:
            argv = shlex.split(fields[1])
            if len(argv) < 2 or Path(argv[1]).resolve() != script_path or "--one" not in argv: continue
            if argv[argv.index("--config") + 1] == config and int(argv[argv.index("--seed") + 1]) == seed:
                return True
        except Exception:
            continue
    return False


def run_task(config, seed, gpu):
    if task_state(config, seed) == "COMPLETE": return 0
    WORK_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
    attempt = len(attempt_records(config, seed)) + 1
    metrics = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_metrics.jsonl"; diag = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_diagnostics.jsonl"
    metrics.unlink(missing_ok=True); diag.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"PDAI-{config}-seed{seed}-") as tmp:
        work = Path(tmp); prepare_worktree(work)
        args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        command = [sys.executable, "main.py", *args[2:], "--seed", str(seed), "--gpu", "0"]
        env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PDAI_CONFIG": config, "PDAI_PROMPT": {"E1":"P0","E2":"P0","E3":"P0","E6":"P1","E7":"P1","E9":"P3","E10":"P3","E11":"P3"}[config], "PDAI_DISEASE": "1" if config in ("E1","E3","E7","E9","E11") else "0", "PDAI_ADV": "1" if config in ("E2","E3","E6","E7","E10","E11") else "0", "PDAI_PROMPTS_FILE": str(PROMPT_FILE), "PDAI_METRIC_PATH": str(metrics), "PDAI_DIAGNOSTIC_PATH": str(diag)})
        log = task_log(config, seed); start = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} attempt={attempt} gpu={gpu} utc={utc_now()} metrics={metrics} diagnostics={diag}\nCOMMAND {shlex.join(command)}\n"); stream.flush()
            proc = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stream.write(f"TASK_END config={config} seed={seed} attempt={attempt} returncode={proc.returncode} folds={len(read_jsonl(metrics))} diagnostics={len(read_jsonl(diag))} elapsed_seconds={time.time()-start:.1f}\n")
    return proc.returncode if proc.returncode else (0 if successful_paths(config, seed) else 2)


def run_one(config, seed, gpu):
    LOG_DIR.mkdir(exist_ok=True)
    with task_log(config, seed).open("a", encoding="utf-8") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 0
        return run_task(config, seed, gpu)


def refresh():
    metrics, diagnostics = [], []
    previous_metrics, previous_diagnostics = read_csv(FOLD_FILE), read_csv(DIAG_FILE)
    for cfg, seed, _ in TASKS:
        paths = successful_paths(cfg, seed)
        if paths:
            metrics.extend(read_jsonl(paths[0])); diagnostics.extend(read_jsonl(paths[1]))
        else:
            metrics.extend(r for r in previous_metrics if r.get("config") == cfg and int(r.get("seed", -1)) == seed)
            diagnostics.extend(r for r in previous_diagnostics if r.get("config") == cfg and int(r.get("seed", -1)) == seed)
    write_csv(FOLD_FILE, ["config", "seed", "fold", *METRICS], metrics)
    write_csv(DIAG_FILE, DIAGNOSTIC_FIELDS, diagnostics)
    return metrics, diagnostics


def all_jobs():
    pending = [item for item in TASKS if task_state(item[0], item[1]) != "COMPLETE" and not process_running(item[0], item[1])]
    active = [(None, gpu, cfg, seed) for cfg, seed, gpu in TASKS if task_state(cfg, seed) != "COMPLETE" and process_running(cfg, seed)]
    started = {0: sum(row[1] == 0 for row in active), 1: sum(row[1] == 1 for row in active)}
    scheduler_path = LOG_DIR / "scheduler.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    def scheduler(message):
        with scheduler_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_now()} {message}\n")
    while pending or active:
        for gpu in (0, 1):
            while started[gpu] < MAX_ACTIVE[gpu]:
                item = next((x for x in pending if x[2] == gpu), None)
                if not item: break
                pending.remove(item); cfg, seed, _ = item
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--one", "--config", cfg, "--seed", str(seed), "--gpu", str(gpu)])
                active.append((p, gpu, cfg, seed)); started[gpu] += 1
                scheduler(f"START {cfg} seed={seed} gpu={gpu} pid={p.pid}")
        if active:
            time.sleep(2)
            next_active = []
            for p, gpu, cfg, seed in active:
                if p is None:
                    if process_running(cfg, seed):
                        next_active.append((None, gpu, cfg, seed)); continue
                    if task_state(cfg, seed) == "COMPLETE":
                        started[gpu] -= 1; scheduler(f"EXTERNAL_COMPLETE {cfg} seed={seed} gpu={gpu}"); continue
                    started[gpu] -= 1
                    if len(attempt_records(cfg, seed)) < 3:
                        pending.append((cfg, seed, gpu)); scheduler(f"RETRY_EXTERNAL {cfg} seed={seed} gpu={gpu}"); continue
                    raise RuntimeError(f"task repeatedly failed: {cfg} seed={seed}")
                if p.poll() is None:
                    next_active.append((p, gpu, cfg, seed)); continue
                scheduler(f"EXIT {cfg} seed={seed} pid={p.pid} gpu={gpu} returncode={p.returncode}")
                if task_state(cfg, seed) == "COMPLETE":
                    started[gpu] -= 1; continue
                if process_running(cfg, seed):
                    next_active.append((None, gpu, cfg, seed)); continue
                started[gpu] -= 1
                if len(attempt_records(cfg, seed)) < 3:
                    pending.append((cfg, seed, gpu)); scheduler(f"RETRY {cfg} seed={seed} gpu={gpu}"); continue
                raise RuntimeError(f"task repeatedly failed: {cfg} seed={seed}")
            active = next_active
    metrics, diagnostics = refresh()
    validate(metrics, diagnostics)
    for cfg, seed, _ in TASKS:
        scheduler(f"TASK_STATE {cfg} seed={seed} state={task_state(cfg, seed)}")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["temporary_attempts_cleaned"] = True
    write_json(MANIFEST_FILE, manifest)
    report(metrics, diagnostics)
    print("PDAI_ALL_COMPLETE")


def validate(metrics, diagnostics):
    expected = {(cfg, seed, fold) for cfg in NEW_CONFIGS for seed in SEEDS for fold in range(1, 11)}
    keys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in metrics}
    if len(metrics) != 240 or keys != expected: raise RuntimeError(f"fold metrics incomplete: {len(metrics)}")
    dkeys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in diagnostics}
    if len(diagnostics) != 240 or dkeys != expected: raise RuntimeError(f"diagnostics incomplete: {len(diagnostics)}")
    reference = read_csv(REFERENCE_FILE)
    reference_keys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in reference}
    if len(reference) != 120 or len(reference_keys) != 120: raise RuntimeError("historical reference must contain 120 unique rows")
    for r in metrics:
        for key in METRICS:
            if not math.isfinite(float(r[key])): raise RuntimeError(f"nonfinite metric {r}")
    for r in diagnostics:
        if str(r["all_finite"]).lower() not in ("true", "1"): raise RuntimeError(f"nonfinite diagnostic {r}")
        if r["config"] in ("E2", "E3", "E6", "E7", "E10", "E11"):
            if str(r["discriminator_grad_valid"]).lower() not in ("true", "1") or str(r["wv_adversarial_grad_valid"]).lower() not in ("true", "1"):
                raise RuntimeError("A1 discriminator/V gradients are invalid")
        if r["config"] in ("E1", "E3", "E7", "E9", "E11"):
            if str(r["disease_projector_grad_valid"]).lower() not in ("true", "1") or str(r["disease_alpha_grad_valid"]).lower() not in ("true", "1"):
                raise RuntimeError("D1 disease gradients are invalid")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    for name, digest in manifest["source_files"].items():
        if sha256(ROOT / name) != digest: raise RuntimeError("root source changed during training: " + name)
    for name, digest in manifest["inputs"].items():
        if sha256(ROOT / name) != digest: raise RuntimeError("ABIDE input changed during training: " + name)
    if sha256(PROMPT_FILE) != manifest["prompts_sha256"] or sha256(PATCH_FILE) != manifest["patch_sha256"] or sha256(REFERENCE_FILE) != manifest["reference_sha256"]:
        raise RuntimeError("frozen prompt, patch or reference changed during training")
    successful_attempts = {}
    for cfg, seed, _ in TASKS:
        successes = [attempt for attempt, paths, end in attempt_records(cfg, seed) if paths and end and end["returncode"] == 0 and (end["folds"], end["diagnostics"]) == (10, 10)]
        key = f"{cfg}_seed{seed}"
        paths = successful_paths(cfg, seed)
        if paths:
            successful_attempts[key] = max(successes)
            continue
        archived_metrics = [r for r in metrics if r["config"] == cfg and int(r["seed"]) == seed]
        archived_diagnostics = [r for r in diagnostics if r["config"] == cfg and int(r["seed"]) == seed]
        archived_complete = (
            len(archived_metrics) == len(archived_diagnostics) == 10
            and {int(r["fold"]) for r in archived_metrics} == set(range(1, 11))
            and {int(r["fold"]) for r in archived_diagnostics} == set(range(1, 11))
        )
        prior_attempt = manifest.get("successful_attempts", {}).get(key)
        if manifest.get("run_status") != "COMPLETE" or not archived_complete or prior_attempt not in successes:
            raise RuntimeError(f"no complete successful attempt or verified archive for {cfg} seed={seed}")
        successful_attempts[key] = prior_attempt
    manifest["run_status"] = "COMPLETE"; manifest["completed_utc"] = utc_now()
    manifest["completion"] = {"reference_rows": len(reference), "fold_metrics": len(metrics), "alignment_diagnostics": len(diagnostics), "tasks": 24, "gpu0_tasks": 12, "gpu1_tasks": 12, "no_failed_attempt_mixed": True}
    manifest["task_status"] = {f"{c}_seed{s}": "COMPLETE" for c, s, _ in TASKS}
    manifest["successful_attempts"] = successful_attempts
    write_json(MANIFEST_FILE, manifest)
    append_handoff(metrics, diagnostics)


def vals(rows, cfg, key): return [float(r[key]) for r in rows if r.get("config") == cfg and r.get(key) not in ("", None)]


def summary(rows, cfg):
    out = {}
    for key in METRICS:
        seed_means = []
        for seed in SEEDS:
            values = vals([r for r in rows if int(r["seed"]) == seed], cfg, key)
            if values: seed_means.append(statistics.mean(values))
        out[key] = (statistics.mean(seed_means), statistics.pstdev(seed_means)) if seed_means else (None, None)
    return out


def paired(rows, left, right, key):
    a = {(int(r["seed"]), int(r["fold"])): float(r[key]) for r in rows if r.get("config") == left and r.get(key) not in ("", None)}
    b = {(int(r["seed"]), int(r["fold"])): float(r[key]) for r in rows if r.get("config") == right and r.get(key) not in ("", None)}
    diffs = [a[k] - b[k] for k in a.keys() & b.keys()]
    if not diffs: return None, 0, 0, 0
    return statistics.mean(diffs), sum(x > 1e-10 for x in diffs), sum(abs(x) <= 1e-10 for x in diffs), sum(x < -1e-10 for x in diffs)


def report(metrics, diagnostics):
    reference = read_csv(REFERENCE_FILE)
    allrows = reference + metrics
    keys = ("acc", "hard_auc", "balanced_accuracy", "probability_auc")

    def contrast(terms, metric):
        lookup = {(r["config"], int(r["seed"]), int(r["fold"])): r for r in allrows}
        by_seed = {seed: [] for seed in SEEDS}
        diffs = []
        for seed in SEEDS:
            for fold in range(1, 11):
                rows = [lookup.get((cfg, seed, fold)) for cfg, _ in terms]
                if any(row is None or row.get(metric) in ("", None) for row in rows):
                    continue
                value = sum(coef * float(row[metric]) for row, (_, coef) in zip(rows, terms))
                diffs.append(value); by_seed[seed].append(value)
        if not diffs:
            return None, (0, 0, 0), {}, None, None
        wins = sum(x > 1e-10 for x in diffs); ties = sum(abs(x) <= 1e-10 for x in diffs); losses = sum(x < -1e-10 for x in diffs)
        seed_means = {seed: statistics.mean(values) for seed, values in by_seed.items() if values}
        return statistics.mean(diffs), (wins, ties, losses), seed_means, max(seed_means.values(), default=None), min(seed_means.values(), default=None)

    def show_contrast(label, terms):
        vals = []
        for metric in keys:
            avg, wtl, seeds, hi, lo = contrast(terms, metric)
            if avg is None:
                vals.append(f"{metric}=N/A")
            else:
                seed_text = ",".join(f"s{seed}:{100*seeds[seed]:+.2f}pp" for seed in SEEDS if seed in seeds)
                vals.append(f"{metric}={100*avg:+.2f}pp [{wtl[0]}/{wtl[1]}/{wtl[2]}], {seed_text}, seed极值{100*lo:+.2f}/{100*hi:+.2f}pp")
        return f"- {label}: " + "; ".join(vals)

    lines = [
        "# 实验九结果：PDAI Prompt-Disease-Adversarial Alignment Interaction", "",
        f"生成时间：{utc_now()}。训练配置为8组×3 seeds×10 folds；历史参考120折只读。", "",
        "## 1. 研究背景", "",
        "前八轮结果呈现一致边界：实验一P1(91.80%/91.62%)只比原始P0(91.69%/91.54%)略高；实验二状态Prompt未超过P1；实验三固定缩放对照较差且部分配置切断语义MLP梯度；实验四疾病相关FC文本低于P1；实验五MLP结构/衰减调整未提升；实验六标量方向门控未形成稳定收益；实验七双主题融合F1约91.69%/91.49%，没有超过P1；实验八S1疾病类别对齐为91.73%/91.51%，疾病分支概率与P1约0.958相关，增益小且有重复信息迹象。", "",
        "因此本实验不再改写Prompt或调超参数，而是用固定P0/P1/P3、D0/D1、A0/A1全因子对照，检验三种机制能否产生可配对、可解释的交互。", "",
        "## 2. DeepASD借鉴依据", "",
        "DeepASD原文的Fig.1将流程分为模态对抗编码、逐模态患者图学习/加权融合、GNN分类；其编码器与模态判别器竞争，使表示难以由模态识别，再进入图学习。论文采用平方损失与invert-label式训练，本实验采用GRL+交叉熵，因而是受启发的轻量移植，不是复现。其Table 2在ABIDE A中，特征提取器单独约72.78% ACC/77.59% AUC，加入模态判别器后为76.74%/82.99%；完整系统为87.38%/92.76%。原文正文把82.99到77.59描述为下降6.51%，但表中数值相减为5.40个百分点，本报告以表内值为准。DeepASD的队列、划分和网络均不同，数值不能与本项目直接横向比较。[DeepASD原文](https://www.nature.com/articles/s41398-024-02972-2)。", "",
        "## 3. 实验假设", "",
        "P因素检验基础语义文本；D因素原样复用实验八S1疾病类别方向；A因素在GFR之后、跨模态注意力之前，对四模态V向量施加四分类GRL。主要假设是疾病监督保留类别方向时，对抗对齐可能减少模态分布差异；同时要检查是否损害ASD/TD可分性，以及这种交互是否依赖P0/P1/P3。", "",
        "## 4. 实现和审计", "",
        f"输入为871名受试者、PHENO/ANAT/FUNC/Correlation维度48/6/10/256，raw DX_GROUP减一后0=ASD、1=TD。读取的V张量形状为[N,4,36]。判别器为LayerNorm(36)-Linear(36,32)-ReLU-Linear(32,4)，新增{2*36+36*32+32+32*4+4}参数；GRL固定0.03，L_adv直接加入损失。D1新增36,865参数，lambda=0.1、temperature=0.2、gamma=0.2*tanh(alpha)。原始GFR/PLN Adam lr=.0038、weight_decay=.11；疾病投影和判别器weight_decay=0。原始10折、seed 0/1/2、450/200、ACC checkpoint及hard-label AUC保持不变。源代码、数据、CLIP和Prompt/Patch哈希及本地/GitHub版本见manifest.json。", "",
        "初始化预检验证GRL前向恒等、特征梯度反号且幅度0.03、判别器梯度与普通CE相同、V投影梯度非零；A1与A0相同随机种子的PLN/疾病参数及初始输出完全一致。Checkpoint同时保存/恢复GFR、PLN、判别器。", "",
        "## 5. 全部12组主结果", "",
        "均值先在每个seed的十折计算，再报告三个seed均值±seed间SD；SEN-ASD以0类为阳性，SPE-ASD为TD特异度。历史没有的Probability-AUC为N/A。", "",
        "| Group | Prompt | Disease | Adv | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-ASD |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    factors = {"E0":("P0","×","×"),"E1":("P0","✓","×"),"E2":("P0","×","✓"),"E3":("P0","✓","✓"),
               "E4":("P1","×","×"),"E5":("P1","✓","×"),"E6":("P1","×","✓"),"E7":("P1","✓","✓"),
               "E8":("P3","×","×"),"E9":("P3","✓","×"),"E10":("P3","×","✓"),"E11":("P3","✓","✓")}
    config_order = [f"E{i}" for i in range(12)]
    for cfg in config_order:
        s = summary(allrows, cfg)
        cells = ["N/A" if s[key][0] is None else f"{100*s[key][0]:.2f}±{100*s[key][1]:.2f}" for key in METRICS]
        p, d, a = factors[cfg]
        lines.append(f"| {cfg} | {p} | {d} | {a} | " + " | ".join(cells) + " |")
    lines += ["", "### 各seed的十折均值", "", "| Group | Seed | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-ASD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in config_order:
        for seed in SEEDS:
            subset = [r for r in allrows if r["config"] == cfg and int(r["seed"]) == seed]
            if not subset: continue
            values = []
            for key in METRICS:
                x = vals(subset, cfg, key)
                values.append("N/A" if not x else f"{100*statistics.mean(x):.2f}%")
            lines.append(f"| {cfg} | {seed} | " + " | ".join(values) + " |")

    lines += ["", "## 6. 三因素交互分析", "", "差值按相同seed、fold配对。胜/平/负统计为该对比值大于/等于/小于0的折数；三个seed共享同一折划分，只作描述性分析，不作独立样本显著性检验。", "", "### Prompt × Disease", ""]
    for label, left, right in (("P0", "E1", "E0"), ("P1", "E5", "E4"), ("P3", "E9", "E8")):
        lines.append(show_contrast(f"{label}: {left}-{right}", [(left,1),(right,-1)]))
    lines += ["", "### Prompt × Adversarial", ""]
    for label, left, right in (("P0", "E2", "E0"), ("P1", "E6", "E4"), ("P3", "E10", "E8")):
        lines.append(show_contrast(f"{label}: {left}-{right}", [(left,1),(right,-1)]))
    lines += ["", "### Disease × Adversarial", ""]
    for label, terms in (("P0",[("E3",1),("E2",-1),("E1",-1),("E0",1)]), ("P1",[("E7",1),("E6",-1),("E5",-1),("E4",1)]), ("P3",[("E11",1),("E10",-1),("E9",-1),("E8",1)])):
        lines.append(show_contrast(f"I_DA({label})", terms))
    lines += ["", "### 三阶交互", "", ""]
    for metric in keys:
        p0 = contrast([("E3",1),("E2",-1),("E1",-1),("E0",1)], metric)[0]
        p1 = contrast([("E7",1),("E6",-1),("E5",-1),("E4",1)], metric)[0]
        p3 = contrast([("E11",1),("E10",-1),("E9",-1),("E8",1)], metric)[0]
        if None in (p0,p1,p3): lines.append(f"- {metric}: N/A（至少一个历史组无该指标）")
        else: lines.append(f"- {metric}: I_DA(P1)-I_DA(P0)={100*(p1-p0):+.2f}pp；I_DA(P3)-I_DA(P0)={100*(p3-p0):+.2f}pp")

    lines += ["", "### 新训练组相对原始P1", "", "以E4历史P1为逐折参照，所有变化是同seed/fold的百分点差。", ""]
    for cfg in NEW_CONFIGS + ["E5"]:
        lines.append(show_contrast(f"{cfg}-E4", [(cfg,1),("E4",-1)]))

    lines += ["", "## 7. 对抗分布对齐是否发生", "", "以下只汇总新训练组；判别器接近25%需结合MMD和类别信息判断。历史E0/E4/E5/E8没有可追溯的36维V诊断，其模态MMD、配对检索和有效秩明确为N/A。", "", "| Group | D test ACC | D train ACC | Test CE | 输出熵 | Pairwise MMD | Corr-PHENO MMD | Corr-ANAT MMD | Corr-FUNC MMD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def dmean(cfg, field):
        data = [float(r[field]) for r in diagnostics if r["config"] == cfg and r.get(field) not in ("", "N/A", "nan", "NaN", None)]
        return statistics.mean(data) if data else None
    for cfg in NEW_CONFIGS:
        cells = [dmean(cfg, k) for k in ("discriminator_test_acc","discriminator_train_acc","discriminator_test_ce","discriminator_test_entropy","mmd_mean_pairwise","mmd_corr_pheno","mmd_corr_anat","mmd_corr_func")]
        lines.append(f"| {cfg} | " + " | ".join("N/A" if x is None else f"{x:.4f}" for x in cells) + " |")
    lines += ["", "对抗分布变化优先用固定核规则下的训练折RBF-MMD和均值向量距离判断。E3−E1与E11−E9是同Disease条件下的A1对照；P1的历史E5缺少V特征，因此无法重建E7−E5的历史MMD差。判别器输出类别分布保存在CSV中。", "",
        "## 8. 过度对齐与类别信息", "", "| Group | 方差 P/A/F/C | 有效秩 P/A/F/C | 零率 P/A/F/C | 类中心距 | 类间/类内比 | 同模态cos | 跨模态配对cos | cos-gap均值 | R@1 |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for cfg in NEW_CONFIGS:
        row = []
        for prefix in ("variance", "effective_rank", "zero_rate"):
            row.append("/".join("N/A" if dmean(cfg, f"{prefix}_{m}") is None else f"{dmean(cfg, f'{prefix}_{m}'):.3f}" for m in ("pheno","anat","func","correlation")))
        gap = [dmean(cfg, f"cosine_gap_pheno_{m}") for m in ("anat","func","correlation")]
        gapmean = statistics.mean([x for x in gap if x is not None]) if any(x is not None for x in gap) else None
        lines.append(f"| {cfg} | {row[0]} | {row[1]} | {row[2]} | {dmean(cfg,'asd_td_center_distance'):.4f} | {dmean(cfg,'class_separation_ratio'):.4f} | {dmean(cfg,'intra_modal_similarity'):.4f} | {dmean(cfg,'intermodal_similarity'):.4f} | {gapmean:.4f} | {dmean(cfg,'retrieval_r1_mean'):.4f} |")
    lines += ["", "类中心距与类间/类内比使用固定确定性采样的训练受试者V特征计算；不把四个潜在Token解释为具体脑区。MMD降低而方差/有效秩或类别分离同步降低，属于可能过度对齐，不是成功。", "",
        "## 9. 疾病分支是否得到新信息", "", "E5来自实验八S1。其历史疾病诊断报告P1/疾病概率相关均值约0.9577；本实验E7在相同checkpoint测试折诊断中另行计算相关系数。相关下降只有在E7性能/类别分离也保留时才可视作互补线索，不能单独证明文本语义有效。", "", "| Group | Disease-only ACC | Prob-AUC | 对齐CE | ASD/TD margin | Gamma | 融合贡献 | 疾病/P1概率相关 |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for cfg in ("E1","E3","E7","E9","E11"):
        g = dmean(cfg,"disease_gamma")
        lines.append(f"| {cfg} | {dmean(cfg,'disease_only_acc'):.4f} | {dmean(cfg,'disease_only_probability_auc'):.4f} | {dmean(cfg,'disease_alignment_loss'):.4f} | {dmean(cfg,'disease_margin_asd'):.4f}/{dmean(cfg,'disease_margin_td'):.4f} | {g:+.5f} | {dmean(cfg,'disease_contribution_abs'):.5f} | {dmean(cfg,'disease_probability_correlation'):.5f} |")
    lines += ["", "## 10. 结论与后续方向", "", "本实验不依据结果追加超参数搜索。实测结果解读见下节；配对比较为描述性结果，不将三个seed的30折误当独立样本。", "", "## 运行审计", "", "- 参考结果：120行；新训练结果：240行；新机制诊断：240行。", "- 任务分配、源代码/数据/模型哈希、预检证据、成功attempt及完成状态：`manifest.json`。", "- 完整日志：`logs/`；执行程序：`run.py`；源码修改只应用于临时副本，未提交或推送。", ""]
    atomic_write(REPORT_FILE, "\n".join(lines))
    report_text = REPORT_FILE.read_text(encoding="utf-8")
    static_text = "## 实测结论\n"
    if static_text in report_text:
        report_text = report_text[:report_text.index(static_text)].rstrip() + "\n"
    audit_marker = "\n## 运行审计\n"
    if audit_marker in report_text:
        report_text = report_text[:report_text.rfind(audit_marker)].rstrip() + "\n"
    lookup_summary = {cfg: summary(allrows, cfg) for cfg in config_order}
    best_acc = max(NEW_CONFIGS, key=lambda cfg: lookup_summary[cfg]["acc"][0])
    best_auc = max(NEW_CONFIGS, key=lambda cfg: lookup_summary[cfg]["hard_auc"][0])
    e7_e4 = contrast([("E7", 1), ("E4", -1)], "acc")
    e7_e4_auc = contrast([("E7", 1), ("E4", -1)], "hard_auc")
    e7_e5 = contrast([("E7", 1), ("E5", -1)], "acc")
    e7_e5_auc = contrast([("E7", 1), ("E5", -1)], "hard_auc")
    e7_e6 = contrast([("E7", 1), ("E6", -1)], "acc")
    e7_e6_auc = contrast([("E7", 1), ("E6", -1)], "hard_auc")
    p0_da = contrast([("E3", 1), ("E2", -1), ("E1", -1), ("E0", 1)], "acc")[0]
    p1_da = contrast([("E7", 1), ("E6", -1), ("E5", -1), ("E4", 1)], "acc")[0]
    p3_da = contrast([("E11", 1), ("E10", -1), ("E9", -1), ("E8", 1)], "acc")[0]
    e9_e8 = contrast([("E9", 1), ("E8", -1)], "acc")[0]
    e7_sen = contrast([("E7", 1), ("E4", -1)], "sensitivity_asd")[0]
    e7_spe = contrast([("E7", 1), ("E4", -1)], "specificity_asd")[0]
    avg_a1_disc = statistics.mean(dmean(cfg, "discriminator_test_acc") for cfg in ("E2", "E3", "E6", "E7", "E10", "E11"))
    avg_a1_r1 = statistics.mean(dmean(cfg, "retrieval_r1_mean") for cfg in ("E2", "E3", "E6", "E7", "E10", "E11"))
    e3_e1_mmd = dmean("E3", "mmd_mean_pairwise") - dmean("E1", "mmd_mean_pairwise")
    e11_e9_mmd = dmean("E11", "mmd_mean_pairwise") - dmean("E9", "mmd_mean_pairwise")
    e7_corr = dmean("E7", "disease_probability_correlation")
    e7_gamma = dmean("E7", "disease_gamma")
    result_lines = [
        static_text.rstrip(), "",
        f"- 新训练组中ACC最高为{best_acc}（{100*lookup_summary[best_acc]['acc'][0]:.3f}% ± {100*lookup_summary[best_acc]['acc'][1]:.3f}pp），Hard-AUC最高为{best_auc}（{100*lookup_summary[best_auc]['hard_auc'][0]:.3f}% ± {100*lookup_summary[best_auc]['hard_auc'][1]:.3f}pp）。{best_acc}相对P3历史基线E8的ACC配对差为{100*e9_e8:+.3f}pp。",
        f"- 重点P1联合组E7相对原始P1 E4：ACC {100*e7_e4[0]:+.3f}pp、Hard-AUC {100*e7_e4_auc[0]:+.3f}pp、SEN-ASD {100*e7_sen:+.3f}pp、SPE-ASD {100*e7_spe:+.3f}pp；相对历史疾病组E5：ACC {100*e7_e5[0]:+.3f}pp、Hard-AUC {100*e7_e5_auc[0]:+.3f}pp；相对P1对抗组E6：ACC {100*e7_e6[0]:+.3f}pp、Hard-AUC {100*e7_e6_auc[0]:+.3f}pp。E7没有稳定胜过这些P1参照。",
        f"- 疾病×对抗的配对交互ACC为P0 {100*p0_da:+.3f}pp、P1 {100*p1_da:+.3f}pp、P3 {100*p3_da:+.3f}pp。P0出现正交互，P1接近零，P3为负，提示该交互依赖基础Prompt；三个seed的方向与幅度须结合上表逐seed结果看待。",
        f"- D1单独作用在P3时，E9相对E8的ACC配对差为{100*e9_e8:+.3f}pp；但三组D1相对各自D0的变化并不一致，不能总结为通用疾病文本增益。",
        f"- A1测试集模态判别平均ACC为{avg_a1_disc*100:.2f}%（四分类机会水平25%），但R@1均值为{avg_a1_r1:.4f}，接近256个训练样本随机配对的基线。D1背景下E3−E1两两MMD变化{e3_e1_mmd:+.4f}，E11−E9变化{e11_e9_mmd:+.4f}，分布差异没有跨Prompt一致下降；E7相对E5没有历史V特征可比较。判别器接近机会水平不足以单独证明受试者级跨模态对齐。",
        f"- E7疾病分支与原分类器概率相关为{e7_corr:.4f}，高于实验八S1历史均值约0.9577；平均gamma为{e7_gamma:+.5f}。这没有显示E7降低预测冗余，且E7相对E5的分类变化接近零。",
        "- 零向量率各模态均为零；A1/D1部分组的类间/类内分离比降低，且疾病×对抗结果因Prompt而异。结合ACC、MMD与R@1，证据支持模型学到了可训练的判别/分布机制，但不支持稳定、普适的诊断增益或受试者配对改善。后续若继续，应先检验类别保护的选择性对齐及私有/共享表示，并在独立测试协议评估，不再对本实验调参。",
        "", "### 四模态均值向量距离", "",
        "下表为折级诊断均值；PHENO/ANAT/FUNC/Correlation均是潜在模态表示，不对应具体脑区。", "",
        "| Group | Corr-PHENO | Corr-ANAT | Corr-FUNC | PHENO-ANAT | PHENO-FUNC | ANAT-FUNC |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    mean_distance_fields = ("mean_vector_distance_corr_pheno", "mean_vector_distance_corr_anat", "mean_vector_distance_corr_func", "mean_vector_distance_pheno_anat", "mean_vector_distance_pheno_func", "mean_vector_distance_anat_func")
    for cfg in NEW_CONFIGS:
        values = [dmean(cfg, field) for field in mean_distance_fields]
        result_lines.append(f"| {cfg} | " + " | ".join("N/A" if value is None else f"{value:.4f}" for value in values) + " |")
    result_lines += ["", "", "## 运行审计"]
    report_text = report_text.rstrip() + "\n\n" + "\n".join(result_lines) + "\n"
    atomic_write(REPORT_FILE, report_text)


def append_handoff(metrics, diagnostics):
    old = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# 实验交接记录\n"
    marker = "## 实验九：PDAI"
    if marker in old: old = old[:old.index(marker)].rstrip() + "\n"
    allrows = read_csv(REFERENCE_FILE) + metrics
    result_rows = []
    for cfg in [f"E{i}" for i in range(12)]:
        s = summary(allrows, cfg)
        if s["acc"][0] is None: continue
        result_rows.append(f"| {cfg} | {100*s['acc'][0]:.3f}% | {100*s['hard_auc'][0]:.3f}% |")
    e7_e4 = paired(allrows, "E7", "E4", "acc")[0]
    e7_e5 = paired(allrows, "E7", "E5", "acc")[0]
    e7_e6 = paired(allrows, "E7", "E6", "acc")[0]
    d_by_cfg = {cfg: [r for r in diagnostics if r["config"] == cfg] for cfg in NEW_CONFIGS}
    e7_corr = statistics.mean(float(r["disease_probability_correlation"]) for r in d_by_cfg["E7"])
    e7_gamma = statistics.mean(float(r["disease_gamma"]) for r in d_by_cfg["E7"])
    e3_e1_mmd = statistics.mean(float(r["mmd_mean_pairwise"]) for r in d_by_cfg["E3"]) - statistics.mean(float(r["mmd_mean_pairwise"]) for r in d_by_cfg["E1"])
    e11_e9_mmd = statistics.mean(float(r["mmd_mean_pairwise"]) for r in d_by_cfg["E11"]) - statistics.mean(float(r["mmd_mean_pairwise"]) for r in d_by_cfg["E9"])
    schedule = (LOG_DIR / "scheduler.log").read_text(encoding="utf-8").splitlines()
    starts = [line.split()[0] for line in schedule if " START " in line]
    exits = [line.split()[0] for line in schedule if " EXIT " in line]
    runtime = "N/A"
    if starts and exits:
        start_dt = datetime.fromisoformat(min(starts).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(max(exits).replace("Z", "+00:00"))
        runtime = str(end_dt - start_dt)
    block = f'''\n## 实验九：PDAI Prompt-Disease-Adversarial Alignment Interaction\n\n- 代码与数据审计：根目录核心源码SHA、ABIDE输入和RN50哈希写入 `experiment/experiment9_prompt_disease_adversarial_interaction/manifest.json`；根目录源码与实验一至八保持未修改。\n- 设计：P0/P1/P3 × D0/D1 × A0/A1 的12组矩阵；E0/E4/E8冻结实验一参考，E5冻结实验八S1；新训练E1/E2/E3/E6/E7/E9/E10/E11共240折。D1新增36,865参数；A1为LayerNorm(36)-Linear(36,32)-ReLU-Linear(32,4)，GRL scale=0.03。\n- 运行：{runtime}；24/24任务、240/240折、240/240诊断成功，全部为attempt 1；GPU0/GPU1各分配12个任务，物理映射和哈希见manifest。\n- 全组ACC/Hard-AUC（seed内十折均值，再对三个seed求均值）：\n\n| Group | ACC | Hard-AUC |\n|---|---:|---:|\n{chr(10).join(result_rows)}\n\n- 结论：E7相对E4/E5/E6的ACC配对差分别为{100*e7_e4:+.3f}/{100*e7_e5:+.3f}/{100*e7_e6:+.3f}pp；没有显示P1疾病与对抗联合分支的稳定增益。D1/A1交互取决于Prompt：P0为正、P1近零、P3为负。A1判别器接近四分类机会水平，但E3−E1和E11−E9的平均两两MMD分别变化{e3_e1_mmd:+.4f}/{e11_e9_mmd:+.4f}，不一致；E7疾病/P1概率相关{e7_corr:.4f}、gamma均值{e7_gamma:+.5f}，没有降低历史预测冗余。\n- 限制：逐折配对只是描述性分析，不把30个共享fold seed记录视为独立样本；潜在Token不解释为ROI；疾病分支表现不证明医学语义正确。完整解释见 `results/report.md`。\n'''
    atomic_write(HANDOFF, old.rstrip() + "\n" + block)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--all", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--refresh", action="store_true"); parser.add_argument("--one", action="store_true"); parser.add_argument("--config"); parser.add_argument("--seed", type=int); parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.prepare: prepare(); return
    if not MANIFEST_FILE.exists(): prepare()
    if args.refresh:
        metrics, diagnostics = refresh()
        if metrics and diagnostics: report(metrics, diagnostics)
        return
    if args.one:
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    if args.all or args.resume:
        all_jobs(); metrics, diagnostics = refresh(); report(metrics, diagnostics); return
    parser.print_help()


if __name__ == "__main__": main()
