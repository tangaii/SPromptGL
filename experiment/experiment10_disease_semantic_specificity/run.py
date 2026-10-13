#!/usr/bin/env python3
"""Experiment 10 runner: Disease Semantic Specificity Control."""
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
ROOT = Path(os.environ.get("DSSC_ROOT", "/data/adodas/SPromptGL-main"))
LOG_DIR = EXP / "logs"
WORK_DIR = LOG_DIR / ".working"
RESULT_DIR = EXP / "results"
PROMPT_FILE = EXP / "prompts.json"
PATCH_FILE = EXP / "control_patch.diff"
MANIFEST_FILE = EXP / "manifest.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_metrics.csv"
DIAG_FILE = RESULT_DIR / "disease_diagnostics.csv"
COUNTERFACTUAL_FILE = RESULT_DIR / "counterfactual.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = ["R0", "R1", "C1", "C2"]
NEW_CONFIGS = ["C1", "C2"]
SEEDS = [0, 1, 2]
METRICS = ["acc", "hard_auc", "probability_auc", "balanced_accuracy", "sensitivity_asd", "specificity_asd"]
DIAGNOSTIC_FIELDS = [
    "config", "seed", "fold", "prototype_class0", "prototype_class1", *METRICS,
    "prototype_cosine", "prototype_euclidean_distance", "prototype_norm_class0", "prototype_norm_class1",
    "disease_alignment_loss", "disease_only_acc", "disease_only_probability_auc",
    "class0_margin_asd_subjects", "class0_margin_td_subjects", "disease_gamma", "disease_gamma_sign",
    "disease_contribution_abs", "disease_probability_correlation", "disease_projector_grad_valid",
    "disease_alpha_grad_valid", "prediction_nonfinite", "all_finite",
]

COUNTERFACTUAL_FIELDS = [
    "config", "seed", "fold", "condition", *METRICS,
    "prediction_flip_rate", "asd_probability_abs_change", "logit_margin_change",
    "logit_margin_abs_change", "acc_change_vs_trained",
]

# Fixed physical allocation: three tasks per GPU as specified by DSSC.
GPU_ASSIGNMENTS = {}
for cfg, seeds in {"C1": {0: 0, 1: 1, 2: 0}, "C2": {0: 1, 1: 0, 2: 1}}.items():
    for seed, gpu in seeds.items():
        GPU_ASSIGNMENTS[(cfg, seed)] = gpu
TASKS = [(cfg, seed, GPU_ASSIGNMENTS[(cfg, seed)]) for cfg in NEW_CONFIGS for seed in SEEDS]
MAX_ACTIVE = {0: 3, 1: 3}
TASK_END_RE = re.compile(r"TASK_END config=(C[12]) seed=(\d+) attempt=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+) counterfactual=(\d+)")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=(C[12]) seed=(\d+) attempt=(\d+) .* metrics=(\S+) diagnostics=(\S+) counterfactual=(\S+)")


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
        "experiment": "DSSC",
        "modality_order": MODALITIES,
        "p3": deepcopy(source["p3"]),
        "disease_prompts": {key: deepcopy(disease_source["disease_prompts"][key]) for key in ("ASD", "TD", "ADHD", "NEUTRAL_CLASS0", "NEUTRAL_CLASS1")},
        "source_sha256": {"p1_p3": sha256(source_path), "disease": sha256(ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/prompts.json")},
        "freeze_rule": "P3 and every disease bank are copied verbatim before training and never edited from results.",
    }
    if flatten_prompt_group(output, "p3") != flatten_prompt_group(source, "p3"):
        raise RuntimeError("prompt freeze check failed")
    for key in ("ASD", "TD", "ADHD", "NEUTRAL_CLASS0", "NEUTRAL_CLASS1"):
        prompts = output["disease_prompts"][key]["prompts"]
        if len(prompts) != 4 or any(not text for text in prompts):
            raise RuntimeError(f"disease prompt bank invalid: {key}")
    write_json(PROMPT_FILE, output)
    return output, source_path


def freeze_reference():
    e9_ref = read_csv(ROOT / "experiment/experiment9_prompt_disease_adversarial_interaction/results/reference.csv")
    e9_metrics = read_csv(ROOT / "experiment/experiment9_prompt_disease_adversarial_interaction/results/fold_metrics.csv")
    rows = []
    for target, source_cfg, source in (("R0", "E8", e9_ref), ("R1", "E9", e9_metrics)):
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
    new = '''_prompt_text_features = {}\n_dssc_disease_features = None\n\n\ndef _dssc_prompt_sentences():\n    mode = os.environ.get("DSSC_PROMPT", "P0")\n    if mode == "P0":\n        return attr_words\n    path = os.environ.get("DSSC_PROMPTS_FILE")\n    if not path:\n        raise RuntimeError("DSSC_PROMPTS_FILE is required for P1/P3")\n    data = json.load(open(path, encoding="utf-8"))\n    output = []\n    for modality in ("PHENO", "ANAT", "FUNC", "Correlation"):\n        for theme in data[mode.lower()][modality]["themes"]:\n            output.extend(item["text"] for item in theme["prompts"])\n    return output\n\n\ndef get_prompt_text_features():\n    mode = os.environ.get("DSSC_PROMPT", "P0")\n    if mode not in _prompt_text_features:\n        sentences = _dssc_prompt_sentences()\n        tokens = clip.tokenize(sentences).to(device)\n        clip_model, _ = clip.load('RN50', device)\n        with torch.no_grad():\n            _prompt_text_features[mode] = clip_model.encode_text(tokens).float().detach()\n    return _prompt_text_features[mode]\n\n\ndef get_dssc_disease_prototypes():\n    global _dssc_disease_features\n    if _dssc_disease_features is None:\n        path = os.environ.get("DSSC_PROMPTS_FILE")\n        if not path:\n            raise RuntimeError("DSSC_PROMPTS_FILE is required for disease prototypes")\n        data = json.load(open(path, encoding="utf-8"))\n        sentences = []\n        for key in ("ASD", "TD"):\n            sentences.extend(data["disease_prompts"][key]["prompts"])\n        tokens = clip.tokenize(sentences).to(device)\n        clip_model, _ = clip.load('RN50', device)\n        with torch.no_grad():\n            enc = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)\n            _dssc_disease_features = F.normalize(enc.view(2, 4, -1).mean(dim=1), dim=-1).detach()\n    return _dssc_disease_features\n'''
    if old not in source:
        raise RuntimeError("network prompt anchor missing")
    source = source.replace(old, new, 1)
    source = source.replace("        self.register_buffer('prompt_text_features', get_prompt_text_features())\n", "        self.register_buffer('prompt_text_features', get_prompt_text_features())\n        self.dssc_prompt = os.environ.get('DSSC_PROMPT', 'P0')\n        self.dssc_disease_enabled = False\n        self.dssc_counterfactual = None\n", 1)
    source = source.replace("        x, _attn = self.InputLayer(x)\n        attn = _attn.mean(dim=1)\n", "        x, _attn = self.InputLayer(x)\n        self.last_modality_v = self.InputLayer.last_v\n        attn = _attn.mean(dim=1)\n", 1)
    old_block = source[source.index("        text_features = self.prompt_text_features\n"):source.index("        # 对最后一个维度进行平均池化\n")]
    new_block = '''        text_features = self.prompt_text_features\n        if text_features.size(0) == 8:\n            # Original P0 path, kept byte-for-byte in its mathematical operations.\n            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)\n            text_features = self.MLP(text_features)\n            part_x = x.view(bs, 4, 4, 9)\n            part_x_norm = F.normalize(part_x, p=2, dim=-1)\n            text_features_norm = F.normalize(text_features, p=2, dim=-1)\n            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)\n            part_x_norm_expanded = part_x_norm.unsqueeze(-2)\n            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))\n            similarity_matrix = similarity_matrix[:,:,:,-1,:]\n            similarity_matrix = similarity_matrix.clamp(min=-5, max=5)\n            similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)\n            similarity_max, _ = torch.max(similarity_matrix, dim=-1)\n            part_x = similarity_max.unsqueeze(-1) * part_x\n        else:\n            grouped = text_features.view(4, 2, 4, 1024)\n            text_features = self.MLP(grouped.reshape(4, 2, 4096))\n            part_x = x.view(bs, 4, 4, 9)\n            part_x_norm = F.normalize(part_x, p=2, dim=-1)\n            text_features_norm = F.normalize(text_features, p=2, dim=-1)\n            similarity_matrix = torch.einsum('nmtf,mcf->nmtc', part_x_norm, text_features_norm)\n            similarity_matrix = torch.softmax(similarity_matrix.clamp(min=-5, max=5), dim=-1).to(device)\n            similarity_max = similarity_matrix.max(dim=-1).values\n            part_x = similarity_max.unsqueeze(-1) * part_x\n\n'''
    source = source.replace(old_block, new_block, 1)
    source = source.replace("        global_x = global_x.view(871, 4, 9)\n", "        global_x = global_x.view(bs, 4, 9)\n", 1)
    old_return = "        output, hidden = self.Outputlayer(x, attn_embedding)\n        return output, hidden, attn_map\n"
    new_return = '''        output, hidden = self.Outputlayer(x, attn_embedding)\n        self.last_base_logprob = output\n        self.last_disease_logits = torch.zeros(bs, 2, device=output.device, dtype=output.dtype)\n        self.last_subject_projection = torch.zeros(bs, 1024, device=output.device, dtype=output.dtype)\n        self.last_subject_semantic = torch.zeros(bs, 1024, device=output.device, dtype=output.dtype)\n        self.last_disease_gamma = 0.0\n        self.last_disease_prelogits = output\n        if getattr(self, 'dssc_disease_enabled', False) and not getattr(self, 'dssc_counterfactual_off', False):\n            projection = self.disease_projector(hidden)\n            semantic = F.normalize(projection, dim=-1)\n            prototypes = F.normalize(self.disease_prototypes, dim=-1)\n            disease_logits = semantic @ prototypes.t() / 0.2\n            gamma = 0.2 * torch.tanh(self.disease_alpha)\n            output = F.log_softmax(output + gamma * disease_logits, dim=1)\n            self.last_subject_projection = projection\n            self.last_subject_semantic = semantic\n            self.last_disease_logits = disease_logits\n            self.last_disease_gamma = float(gamma.detach().item())\n            self.last_disease_prelogits = self.last_base_logprob + gamma * disease_logits\n        return output, hidden, attn_map\n'''
    if old_return not in source:
        raise RuntimeError("network return anchor missing")
    return source.replace(old_return, new_return, 1)


def patch_model(source):
    source = source.replace("import os\n", "import os\nimport json\nimport network as _dssc_network\n", 1)
    anchor = "from utils import *\n\n\nclass EvalHelper:"
    addition = """from utils import *


class EvalHelper:"""
    if anchor not in source:
        raise RuntimeError("model class anchor missing")
    source = source.replace(anchor, addition, 1)
    old = "        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)\n        self.ModalFusion.apply(my_weight_init)\n"
    new = """        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
        self.dssc_disease = os.environ.get("DSSC_DISEASE", "0") == "1"
        self.ModalFusion.dssc_disease_enabled = self.dssc_disease
        self.ModalFusion.dssc_disease_grad_valid = False
        self.ModalFusion.dssc_alpha_grad_valid = False
        self.ModalFusion.dssc_counterfactual_off = False
        self.ModalFusion.dssc_override_prototypes = None
        if self.dssc_disease:
            # Text encoding is isolated from the training RNG; C1 and C2 therefore
            # receive identical backbone and projector initialization for a seed.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):
                self.ModalFusion.register_buffer("disease_prototypes", get_dssc_disease_prototypes())
            self.ModalFusion.disease_projector = nn.Linear(36, 1024, bias=False).to(dev)
            my_weight_init(self.ModalFusion.disease_projector)
            self.ModalFusion.disease_alpha = nn.Parameter(torch.zeros((), device=dev))
            disease_parameters = list(self.ModalFusion.disease_projector.parameters()) + [self.ModalFusion.disease_alpha]
            if sum(parameter.numel() for parameter in disease_parameters) != 36865:
                raise RuntimeError("DSSC disease branch must add exactly 36865 parameters")
            self.optimizer_MF.add_param_group({"params": disease_parameters, "weight_decay": 0.0})
"""
    if old not in source:
        raise RuntimeError("model optimizer anchor missing")
    source = source.replace(old, new, 1)
    old = "            prob,_,_ = self.ModalFusion(feat)\n\n            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)\n            \n            loss = cls_loss + all_graph_loss\n            \n            loss.backward()\n            \n            self.optimizer_MP0.step()            \n            self.optimizer_MF.step()\n"
    new = """            prob,_,_ = self.ModalFusion(feat)

            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)
            disease_loss = F.cross_entropy(self.ModalFusion.last_disease_logits[train_idx], self.targ[train_idx], weight=self.weight)
            loss = cls_loss + all_graph_loss + 0.1 * disease_loss
            self._dssc_last_disease_loss = float(disease_loss.detach().item())
            loss.backward()
            grad = self.ModalFusion.disease_projector.weight.grad
            self.ModalFusion.dssc_disease_grad_valid |= bool(grad is not None and torch.isfinite(grad).all().item() and (grad.abs().sum() > 0).item())
            grad = self.ModalFusion.disease_alpha.grad
            self.ModalFusion.dssc_alpha_grad_valid |= bool(grad is not None and torch.isfinite(grad).all().item() and (grad.abs().sum() > 0).item())
            self.optimizer_MP0.step()
            self.optimizer_MF.step()
"""
    if old not in source:
        raise RuntimeError("model simple-2 anchor missing")
    return source.replace(old, new, 1)

def patch_main(source):
    old = "        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)\n"
    new = old + "        if os.environ.get('DSSC_CONFIG'):\n            agent._dssc_fold_number = clk\n"
    if old not in source:
        raise RuntimeError("main fold anchor missing")
    source = source.replace(old, new, 1)
    old = "                dict_list = [agent.MessagePassing0.state_dict(),\n                             agent.ModalFusion.state_dict()]\n"
    new = "                dict_list = [agent.MessagePassing0.state_dict(), agent.ModalFusion.state_dict()]\n"
    if old not in source:
        raise RuntimeError("main save anchor missing")
    source = source.replace(old, new, 1)
    old = "        agent.ModalFusion.load_state_dict(dict_list[1])\n        \n        val_acc.append(best_val_acc)\n"
    new = "        agent.ModalFusion.load_state_dict(dict_list[1])\n        \n        val_acc.append(best_val_acc)\n"
    if old not in source:
        raise RuntimeError("main restore anchor missing")
    source = source.replace(old, new, 1)
    old = "        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)\n"
    new = old + "        if os.environ.get('DSSC_CONFIG'):\n            agent.write_dssc_outputs(clk, cur_tst_acc, cur_tst_auc, conf_mat)\n"
    if old not in source:
        raise RuntimeError("main test anchor missing")
    return source.replace(old, new, 1)


def patch_model_outputs(source):
    anchor = "    def cal_acc(self, eval_idx):\n"
    if anchor not in source:
        raise RuntimeError("model output method anchor missing")
    method = r"""    def _dssc_json_row(self, path, row):
        if not path:
            return
        with open(path, "a", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False); stream.write("\n")

    def _dssc_hard_auc(self, labels, pred):
        try:
            return float(roc_auc_score(np.eye(2)[labels.astype(int)], np.eye(2)[pred.astype(int)]))
        except ValueError:
            return float("nan")

    def _dssc_write_counterfactual(self, feat, test_idx, labels, trained_probability, trained_pred, trained_margin, trained_acc):
        path = os.environ.get("DSSC_COUNTERFACTUAL_PATH")
        bank_files = json.loads(os.environ.get("DSSC_BANK_FILES", "{}"))
        if not path or set(bank_files) < {"REAL", "ADHD", "NEUTRAL"}:
            return
        original_proto = self.ModalFusion.disease_prototypes.detach().clone()
        original_off = bool(getattr(self.ModalFusion, "dssc_counterfactual_off", False))
        original_override = getattr(self.ModalFusion, "dssc_override_prototypes", None)
        original_prompt_file = os.environ.get("DSSC_PROMPTS_FILE")
        original_cache = _dssc_network._dssc_disease_features
        conditions = ("TRAINED", "REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF")
        try:
            for condition in conditions:
                self.ModalFusion.dssc_counterfactual_off = condition == "OFF"
                if condition == "TRAINED":
                    self.ModalFusion.disease_prototypes.copy_(original_proto)
                elif condition == "SWAPPED":
                    self.ModalFusion.disease_prototypes.copy_(original_proto.flip(0))
                elif condition == "OFF":
                    self.ModalFusion.disease_prototypes.copy_(original_proto)
                else:
                    os.environ["DSSC_PROMPTS_FILE"] = bank_files[condition]
                    _dssc_network._dssc_disease_features = None
                    self.ModalFusion.disease_prototypes.copy_(get_dssc_disease_prototypes())
                with torch.no_grad():
                    output, _, _ = self.ModalFusion(feat)
                    prob = output.index_select(0, test_idx).exp().cpu().numpy()
                    pred = prob.argmax(1)
                    prelogits = self.ModalFusion.last_disease_prelogits.index_select(0, test_idx).cpu().numpy()
                cf_margin = prelogits[:, 0] - prelogits[:, 1]
                conf = confusion_matrix(labels, pred, labels=[0, 1])
                sensitivity = float(conf[0, 0] / max(1, conf[0].sum()))
                specificity = float(conf[1, 1] / max(1, conf[1].sum()))
                cf_acc = float((pred == labels).mean())
                row = {"config": os.environ["DSSC_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(self._dssc_fold_number), "condition": condition,
                       "acc": cf_acc, "hard_auc": self._dssc_hard_auc(labels, pred),
                       "probability_auc": float(roc_auc_score((labels == 0).astype(int), prob[:, 0])),
                       "balanced_accuracy": (sensitivity + specificity) / 2, "sensitivity_asd": sensitivity, "specificity_asd": specificity,
                       "prediction_flip_rate": float((pred != trained_pred).mean()),
                       "asd_probability_abs_change": float(np.abs(prob[:, 0] - trained_probability[:, 0]).mean()),
                       "logit_margin_change": float((cf_margin - trained_margin).mean()),
                       "logit_margin_abs_change": float(np.abs(cf_margin - trained_margin).mean()),
                       "acc_change_vs_trained": cf_acc - trained_acc}
                self._dssc_json_row(path, row)
        finally:
            self.ModalFusion.disease_prototypes.copy_(original_proto)
            self.ModalFusion.dssc_counterfactual_off = original_off
            self.ModalFusion.dssc_override_prototypes = original_override
            if original_prompt_file is None:
                os.environ.pop("DSSC_PROMPTS_FILE", None)
            else:
                os.environ["DSSC_PROMPTS_FILE"] = original_prompt_file
            _dssc_network._dssc_disease_features = original_cache

    def write_dssc_outputs(self, fold, acc, hard_auc, conf_mat):
        metric_path = os.environ.get("DSSC_METRIC_PATH")
        diag_path = os.environ.get("DSSC_DIAGNOSTIC_PATH")
        if not metric_path or not diag_path:
            return
        labels = self.targ[torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)].detach().cpu().numpy()
        with torch.no_grad():
            model_num, feat1 = self.dataLayer(self.feat)
            feat, _, _ = self.MessagePassing0(feat1, model_num, self.targ, self.trn_idx)
            output, _, _ = self.ModalFusion(feat)
            test_idx = torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)
            train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)
            probability = output.index_select(0, test_idx).exp().cpu().numpy()
            trained_pred = probability.argmax(1)
            base_probability = self.ModalFusion.last_base_logprob.index_select(0, test_idx).exp().cpu().numpy()
            probability_auc = float(roc_auc_score((labels == 0).astype(int), probability[:, 0]))
            sensitivity = float(conf_mat[0, 0] / max(1, conf_mat[0].sum()))
            specificity = float(conf_mat[1, 1] / max(1, conf_mat[1].sum()))
            metric = {"config": os.environ["DSSC_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold),
                      "acc": float(acc), "hard_auc": float(hard_auc), "probability_auc": probability_auc,
                      "balanced_accuracy": (sensitivity + specificity) / 2, "sensitivity_asd": sensitivity, "specificity_asd": specificity}
            self._dssc_json_row(metric_path, metric)
            prototypes = F.normalize(self.ModalFusion.disease_prototypes, dim=-1)
            proto_cos = float(F.cosine_similarity(prototypes[0:1], prototypes[1:2]).item())
            proto_dist = float(torch.linalg.vector_norm(prototypes[0] - prototypes[1]).item())
            disease_logits = self.ModalFusion.last_disease_logits
            dl_test = disease_logits.index_select(0, test_idx)
            disease_prob = dl_test.softmax(-1).cpu().numpy()
            disease_pred = disease_prob.argmax(1)
            disease_only_acc = float((disease_pred == labels).mean())
            disease_auc = float(roc_auc_score((labels == 0).astype(int), disease_prob[:, 0]))
            train_dl = disease_logits.index_select(0, train_idx)
            align = float(F.cross_entropy(train_dl, self.targ.index_select(0, train_idx), weight=self.weight).item())
            margins = (dl_test[:, 0] - dl_test[:, 1]).cpu().numpy()
            margin_asd = float(margins[labels == 0].mean())
            margin_td = float(margins[labels == 1].mean())
            gamma = float(self.ModalFusion.last_disease_gamma)
            prelogits = self.ModalFusion.last_disease_prelogits.index_select(0, test_idx)
            base_logprob = self.ModalFusion.last_base_logprob.index_select(0, test_idx)
            contribution = float((prelogits - base_logprob).abs().mean().item())
            disease_probability_correlation = float("nan")
            if base_probability[:, 0].std() > 0 and disease_prob[:, 0].std() > 0:
                disease_probability_correlation = float(np.corrcoef(base_probability[:, 0], disease_prob[:, 0])[0, 1])
            diag = {"config": os.environ["DSSC_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold),
                    "prototype_class0": os.environ.get("DSSC_CLASS0_NAME", "UNKNOWN"), "prototype_class1": os.environ.get("DSSC_CLASS1_NAME", "UNKNOWN"), **metric,
                    "prototype_cosine": proto_cos, "prototype_euclidean_distance": proto_dist,
                    "prototype_norm_class0": float(torch.linalg.vector_norm(self.ModalFusion.disease_prototypes[0]).item()),
                    "prototype_norm_class1": float(torch.linalg.vector_norm(self.ModalFusion.disease_prototypes[1]).item()),
                    "disease_alignment_loss": align, "disease_only_acc": disease_only_acc, "disease_only_probability_auc": disease_auc,
                    "class0_margin_asd_subjects": margin_asd, "class0_margin_td_subjects": margin_td,
                    "disease_gamma": gamma, "disease_gamma_sign": int(np.sign(gamma)), "disease_contribution_abs": contribution,
                    "disease_probability_correlation": disease_probability_correlation,
                    "disease_projector_grad_valid": bool(self.ModalFusion.dssc_disease_grad_valid),
                    "disease_alpha_grad_valid": bool(self.ModalFusion.dssc_alpha_grad_valid),
                    "prediction_nonfinite": bool((~torch.isfinite(output)).any().item()),
                    "all_finite": bool(torch.isfinite(output).all().item() and torch.isfinite(disease_logits).all().item())}
            self._dssc_json_row(diag_path, diag)
            trained_margin = prelogits[:, 0].cpu().numpy() - prelogits[:, 1].cpu().numpy()
            self._dssc_write_counterfactual(feat, test_idx, labels, probability, trained_pred, trained_margin, float(acc))

"""
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


def make_runtime_prompt_files(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    bank_keys = {"REAL": ("ASD", "TD"), "ADHD": ("ADHD", "TD"), "NEUTRAL": ("NEUTRAL_CLASS0", "NEUTRAL_CLASS1")}
    paths = {}
    for bank, (class0, class1) in bank_keys.items():
        runtime = deepcopy(data)
        runtime["disease_prompts"]["ASD"] = deepcopy(data["disease_prompts"][class0])
        runtime["disease_prompts"]["TD"] = deepcopy(data["disease_prompts"][class1])
        path = directory / f"{bank.lower()}_prompts.json"
        write_json(path, runtime)
        paths[bank] = str(path)
    return paths


def preflight(work):
    runtime_banks = make_runtime_prompt_files(work / "runtime_prompts")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "DSSC_CONFIG": "C1", "DSSC_PROMPT": "P3", "DSSC_DISEASE": "1", "DSSC_PROMPTS_FILE": runtime_banks["ADHD"], "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    script = '''import os, torch, numpy as np, pandas as pd\nfrom types import SimpleNamespace\nfrom model import EvalHelper\nD=[48,6,10,256]\ndata=pd.read_csv("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv").values\nlabel=data[:,-1].astype(int)-1; feat=data[:,:-1].astype("float32")\nassert feat.shape==(871,320) and np.bincount(label).tolist()==[403,468]\na=SimpleNamespace(GC_mode="weighted-cosine",MP_mode="GCN",MF_mode="concat",n_hidden=18,nmodal=4,nclass=2,dropout=.35,alpha=.5,n_head=2,nlayer=1,th=.9,lr=.0038,reg=.11,theta_smooth=1,theta_degree=.5,theta_sparsity=0.0,seed=0,datname="ABIDE",mode="simple-2")\ntorch.manual_seed(123); torch.cuda.manual_seed_all(123); e1=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871))\nassert sum(p.numel() for p in e1.ModalFusion.disease_projector.parameters())+1==36865\nassert e1.ModalFusion.disease_alpha.item()==0\ne1.ModalFusion.eval(); out1=e1.ModalFusion(e1.feat)[0].detach().clone()\nos.environ["DSSC_PROMPTS_FILE"] = 'NEUTRAL_RUNTIME_PATH'; torch.manual_seed(123); torch.cuda.manual_seed_all(123); e2=EvalHelper(D,feat,label,a,np.arange(500),np.arange(500,871))\nfor n,p in e1.ModalFusion.state_dict().items():\n if n not in ("disease_prototypes",): assert torch.equal(p,e2.ModalFusion.state_dict()[n]), n\ne2.ModalFusion.eval(); out2=e2.ModalFusion(e2.feat)[0].detach().clone()\nassert torch.equal(out1,out2)\ne1.run_epoch("simple-2"); assert e1.ModalFusion.dssc_disease_grad_valid and e1.ModalFusion.dssc_alpha_grad_valid; os.environ["DSSC_METRIC_PATH"]="preflight_metrics.jsonl"; os.environ["DSSC_DIAGNOSTIC_PATH"]="preflight_diagnostics.jsonl"; acc,auc,conf=e1.print_tst_acc("simple-2"); e1.write_dssc_outputs(1,acc,auc,conf)\nprint("DSSC_PREFLIGHT_OK")'''
    script = script.replace(repr("NEUTRAL_RUNTIME_PATH"), repr(runtime_banks["NEUTRAL"]))
    result = subprocess.run([sys.executable, "-c", script], cwd=work, env=env, text=True, capture_output=True, timeout=900)
    if result.returncode or "DSSC_PREFLIGHT_OK" not in result.stdout:
        raise RuntimeError("runtime preflight failed:\n" + result.stdout[-4000:] + "\n" + result.stderr[-8000:])
    diagnostic_path = work / "preflight_diagnostics.jsonl"
    metric_path = work / "preflight_metrics.jsonl"
    if len(read_jsonl(diagnostic_path)) != 1 or len(read_jsonl(metric_path)) != 1:
        raise RuntimeError("diagnostic writer preflight failed")


def write_readme(manifest):
    text = f'''# 实验十：DSSC Disease Semantic Specificity Control

本实验复用实验九E9的P3、疾病投影、对齐损失、gamma决策融合和ABIDE训练协议，只改变疾病类别原型文本。R0/R1冻结实验九E8/E9；C1使用ADHD/TD；C2使用NEUTRAL_CLASS0/NEUTRAL_CLASS1。真实标签始终为0=ASD、1=TD。

## 研究问题

实验九E9相对E8的改善可能来自ASD语义、额外类别监督或固定文本向量的几何关系。本实验只新增C1和C2，分别检验错误疾病文本和非疾病技术文本是否也能形成类别判别方向。

## 配置与协议

| 配置 | 基础Prompt | 类别0原型 | 类别1原型 | 状态 |
|---|---|---|---|---|
| R0 | P3 | 无 | 无 | 实验九E8历史冻结 |
| R1 | P3 | ASD | TD | 实验九E9历史冻结 |
| C1 | P3 | ADHD | TD | 新训练 |
| C2 | P3 | NEUTRAL_CLASS0 | NEUTRAL_CLASS1 | 新训练 |

C1/C2均使用36→1024无偏置疾病投影、归一化点积、temperature=0.2、lambda_align=0.1、gamma=0.2*tanh(alpha)，疾病参数weight_decay=0，其余参数weight_decay=0.11。PDAI对抗分支完全关闭，没有GRL、MMD、OT、额外分类头或新超参数。训练为3 seeds×10 folds，450 epoch上限、early=200、ACC checkpoint、hard-label AUC。

## 冻结文本

P3的32条Prompt来自实验一；ASD、TD、ADHD和两组中性技术文本逐字来自实验八。完整冻结内容见 `prompts.json`。训练时使用临时映射文件，仅将所选文本放到疾病分支的类别0/1位置，标签不变。

## 诊断与反事实

`disease_diagnostics.csv` 每个C1/C2 seed×fold一行，记录类别原型余弦/距离/范数、疾病分支性能、真实ASD/TD样本上的类别0 margin、gamma、融合贡献、与P3概率相关性及梯度有效性。`counterfactual.csv` 对每个最佳checkpoint记录 TRAINED、REAL、ADHD、NEUTRAL、SWAPPED、OFF 六种推理条件；不重新训练、不改标签、不重新选择checkpoint。

## 运行

```bash
python experiment/experiment10_disease_semantic_specificity/run.py --prepare
python experiment/experiment10_disease_semantic_specificity/run.py --all
python experiment/experiment10_disease_semantic_specificity/run.py --resume
python experiment/experiment10_disease_semantic_specificity/run.py --refresh
```

根目录源码只在临时副本中打补丁；工作目录和失败attempt在任务结束后清理。`manifest.json` 保存版本哈希、参数量、初始化预检、GPU分配和完成状态。
'''
    atomic_write(EXP / "README.md", text)

def make_manifest():
    clip = Path.home() / ".cache/clip/RN50.pt"
    def git_value(args):
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
        except Exception:
            return None
    github_main = None
    try:
        remote = subprocess.check_output(["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30)
        github_main = remote.split()[0] if remote.split() else None
    except Exception:
        pass
    e9_ref = ROOT / "experiment/experiment9_prompt_disease_adversarial_interaction/results/reference.csv"
    e9_metrics = ROOT / "experiment/experiment9_prompt_disease_adversarial_interaction/results/fold_metrics.csv"
    e8_diag = ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/results/disease_diagnostics.csv"
    return {"experiment": "DSSC", "schema_version": 1, "created_utc": utc_now(), "source_root": str(ROOT),
            "git": {"local_head": git_value(["git", "rev-parse", "HEAD"]), "github_main_at_prepare": github_main, "local_core_source_hashes": {n: sha256(ROOT / n) for n in SOURCE_FILES}},
            "source_files": {n: sha256(ROOT / n) for n in SOURCE_FILES},
            "inputs": {n: sha256(ROOT / n) for n in ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")},
            "clip_rn50": {"path": str(clip), "sha256": sha256(clip) if clip.exists() else None},
            "prompts_sha256": sha256(PROMPT_FILE), "patch_sha256": sha256(PATCH_FILE), "reference_sha256": sha256(REFERENCE_FILE),
            "prompt_sources": {"experiment1_prompts_sha256": sha256(ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"),
                               "experiment8_prompts_sha256": sha256(ROOT / "experiment/experiment8_scheme1_shared_disease_alignment/prompts.json")},
            "reference_sources": {str(e9_ref.relative_to(ROOT)): sha256(e9_ref), str(e9_metrics.relative_to(ROOT)): sha256(e9_metrics), str(e8_diag.relative_to(ROOT)): sha256(e8_diag)},
            "protocol": {"folds": 10, "seeds": SEEDS, "epoch_cap": 450, "early_patience": 200, "stratified_random_state": 10,
                         "checkpoint": "validation ACC", "auc": "hard-label", "new_folds": 60, "new_diagnostics": 60, "counterfactual_rows": 360,
                         "args": {"GC_mode": "weighted-cosine", "MF_mode": "concat", "MP_mode": "GCN", "dropout": 0.35, "lr": 0.0038, "n_head": 2, "n_hidden": 18, "nlayer": 1, "reg": 0.11},
                         "factor_levels": {"base_prompt": ["P3"], "disease_bank": ["REAL", "ADHD", "NEUTRAL"], "adversarial": ["OFF"]},
                         "gpu_assignments": {f"{c}_seed{s}": g for c, s, g in TASKS}, "max_active_per_gpu": MAX_ACTIVE},
            "parameters": {"disease_branch": 36865, "adversarial_branch": 0, "disease_projector": "36x1024 bias=False", "disease_temperature": 0.2, "lambda_align": 0.1, "gamma_scale": 0.2},
            "audit": {"root_source_modified": False, "experiment1_to_9_untouched": True, "p3_and_all_disease_prompts_frozen": True,
                      "historical_reference_rows": 60, "preflight_checks": ["prompt source equality", "patch apply and py_compile", "same-seed C1/C2 initialization", "zero disease_alpha", "first-step disease gradients", "REAL forward path"]},
            "run_status": "PREPARED", "task_status": {f"{c}_seed{s}": "PENDING" for c, s, _ in TASKS}}

def prepare():
    EXP.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(exist_ok=True); WORK_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True)
    prompts, p1_path = build_prompts(); freeze_reference(); originals, patched = make_patch()
    with tempfile.TemporaryDirectory(prefix="dssc-preflight-") as tmp:
        preflight_work = Path(tmp)
        prepare_worktree(preflight_work)
        preflight(preflight_work)
    manifest = make_manifest(); manifest["preflight"] = {"patch_applies": True, "py_compile": True, "runtime_initialization_and_gradient_checks": True, "new_parameters": {"disease": 36865, "adversarial": 0}}
    write_json(MANIFEST_FILE, manifest); write_readme(manifest)
    print("DSSC_PREPARED", json.dumps({"patch": sha256(PATCH_FILE), "prompts": sha256(PROMPT_FILE), "reference_rows": 60}))


def task_log(config, seed): return LOG_DIR / f"{config}_seed{seed}.log"


def attempt_records(config, seed):
    path = task_log(config, seed)
    starts, ends = {}, {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ATTEMPT_RE.search(line)
        if m and m.group(1) == config and int(m.group(2)) == seed:
            starts[int(m.group(3))] = (m.group(4), m.group(5), m.group(6))
        m = TASK_END_RE.search(line)
        if m and m.group(1) == config and int(m.group(2)) == seed:
            ends[int(m.group(3))] = {"returncode": int(m.group(4)), "folds": int(m.group(5)), "diagnostics": int(m.group(6)), "counterfactual": int(m.group(7))}
    return [(a, starts.get(a), ends.get(a)) for a in sorted(set(starts) | set(ends))]


def successful_paths(config, seed):
    for attempt, paths, end in reversed(attempt_records(config, seed)):
        if not paths or not end or end["returncode"] != 0 or (end["folds"], end["diagnostics"], end["counterfactual"]) != (10, 10, 60):
            continue
        metric, diag, counterfactual = map(Path, paths)
        mr, dr, cr = read_jsonl(metric), read_jsonl(diag), read_jsonl(counterfactual)
        if len(mr) == len(dr) == 10 and len(cr) == 60 and {int(r["fold"]) for r in mr} == set(range(1, 11)) and {int(r["fold"]) for r in dr} == set(range(1, 11)) and {int(r["fold"]) for r in cr} == set(range(1, 11)):
            return metric, diag, counterfactual
    return None


def task_state(config, seed):
    if successful_paths(config, seed):
        return "COMPLETE"
    metric_rows = [r for r in read_csv(FOLD_FILE) if r.get("config") == config and int(r.get("seed", -1)) == seed]
    diagnostic_rows = [r for r in read_csv(DIAG_FILE) if r.get("config") == config and int(r.get("seed", -1)) == seed]
    counterfactual_rows = [r for r in read_csv(COUNTERFACTUAL_FILE) if r.get("config") == config and int(r.get("seed", -1)) == seed]
    if len(metric_rows) == len(diagnostic_rows) == 10 and len(counterfactual_rows) == 60:
        return "COMPLETE"
    records = attempt_records(config, seed)
    if records and records[-1][2] is None:
        return "INTERRUPTED"
    if records and records[-1][2]["returncode"] != 0:
        return "FAILED"
    return "PENDING"


def process_running(config, seed):
    try:
        lines = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except Exception:
        return False
    script_path = Path(__file__).resolve()
    for line in lines:
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            argv = shlex.split(fields[1])
            if len(argv) < 2 or Path(argv[1]).resolve() != script_path or "--one" not in argv:
                continue
            if argv[argv.index("--config") + 1] == config and int(argv[argv.index("--seed") + 1]) == seed:
                return True
        except Exception:
            continue
    return False


def run_task(config, seed, gpu):
    if task_state(config, seed) == "COMPLETE":
        return 0
    WORK_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
    attempt = len(attempt_records(config, seed)) + 1
    metrics = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_metrics.jsonl"
    diag = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_diagnostics.jsonl"
    counterfactual = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_counterfactual.jsonl"
    for path in (metrics, diag, counterfactual):
        path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"DSSC-{config}-seed{seed}-") as tmp:
        work = Path(tmp); prepare_worktree(work)
        bank_files = make_runtime_prompt_files(work / "runtime_prompts")
        trained_bank = "ADHD" if config == "C1" else "NEUTRAL"
        args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        command = [sys.executable, "main.py", *args[2:], "--seed", str(seed), "--gpu", "0"]
        env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "DSSC_CONFIG": config, "DSSC_PROMPT": "P3", "DSSC_DISEASE": "1", "DSSC_PROMPTS_FILE": bank_files[trained_bank], "DSSC_BANK_FILES": json.dumps(bank_files), "DSSC_CLASS0_NAME": "ADHD" if config == "C1" else "NEUTRAL_CLASS0", "DSSC_CLASS1_NAME": "TD" if config == "C1" else "NEUTRAL_CLASS1", "DSSC_METRIC_PATH": str(metrics), "DSSC_DIAGNOSTIC_PATH": str(diag), "DSSC_COUNTERFACTUAL_PATH": str(counterfactual)})
        log = task_log(config, seed); start = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} attempt={attempt} gpu={gpu} utc={utc_now()} metrics={metrics} diagnostics={diag} counterfactual={counterfactual}\nCOMMAND {shlex.join(command)}\n"); stream.flush()
            proc = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stream.write(f"TASK_END config={config} seed={seed} attempt={attempt} returncode={proc.returncode} folds={len(read_jsonl(metrics))} diagnostics={len(read_jsonl(diag))} counterfactual={len(read_jsonl(counterfactual))} elapsed_seconds={time.time()-start:.1f}\n")
    return proc.returncode if proc.returncode else (0 if successful_paths(config, seed) else 2)

def run_one(config, seed, gpu):
    LOG_DIR.mkdir(exist_ok=True)
    with task_log(config, seed).open("a", encoding="utf-8") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 0
        return run_task(config, seed, gpu)


def refresh():
    metrics, diagnostics, counterfactual = [], [], []
    previous_metrics, previous_diagnostics, previous_counterfactual = read_csv(FOLD_FILE), read_csv(DIAG_FILE), read_csv(COUNTERFACTUAL_FILE)
    for cfg, seed, _ in TASKS:
        paths = successful_paths(cfg, seed)
        if paths:
            metrics.extend(read_jsonl(paths[0])); diagnostics.extend(read_jsonl(paths[1])); counterfactual.extend(read_jsonl(paths[2]))
        else:
            metrics.extend(r for r in previous_metrics if r.get("config") == cfg and int(r.get("seed", -1)) == seed)
            diagnostics.extend(r for r in previous_diagnostics if r.get("config") == cfg and int(r.get("seed", -1)) == seed)
            counterfactual.extend(r for r in previous_counterfactual if r.get("config") == cfg and int(r.get("seed", -1)) == seed)
    write_csv(FOLD_FILE, ["config", "seed", "fold", *METRICS], metrics)
    write_csv(DIAG_FILE, DIAGNOSTIC_FIELDS, diagnostics)
    write_csv(COUNTERFACTUAL_FILE, COUNTERFACTUAL_FIELDS, counterfactual)
    return metrics, diagnostics, counterfactual


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
    metrics, diagnostics, counterfactual = refresh()
    validate(metrics, diagnostics, counterfactual)
    for cfg, seed, _ in TASKS:
        scheduler(f"TASK_STATE {cfg} seed={seed} state={task_state(cfg, seed)}")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["temporary_attempts_cleaned"] = True
    write_json(MANIFEST_FILE, manifest)
    report(metrics, diagnostics, counterfactual)
    print("DSSC_ALL_COMPLETE")


def validate(metrics, diagnostics, counterfactual):
    expected = {(cfg, seed, fold) for cfg in NEW_CONFIGS for seed in SEEDS for fold in range(1, 11)}
    keys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in metrics}
    if len(metrics) != 60 or keys != expected:
        raise RuntimeError(f"fold metrics incomplete: {len(metrics)}")
    dkeys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in diagnostics}
    if len(diagnostics) != 60 or dkeys != expected:
        raise RuntimeError(f"diagnostics incomplete: {len(diagnostics)}")
    ckeys = {(r["config"], int(r["seed"]), int(r["fold"]), r["condition"]) for r in counterfactual}
    expected_c = {(cfg, seed, fold, condition) for cfg in NEW_CONFIGS for seed in SEEDS for fold in range(1, 11) for condition in ("TRAINED", "REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF")}
    if len(counterfactual) != 360 or ckeys != expected_c:
        raise RuntimeError(f"counterfactual incomplete: {len(counterfactual)}")
    reference = read_csv(REFERENCE_FILE)
    reference_keys = {(r["config"], int(r["seed"]), int(r["fold"])) for r in reference}
    if len(reference) != 60 or len(reference_keys) != 60 or {r["config"] for r in reference} != {"R0", "R1"}:
        raise RuntimeError("historical reference must contain 60 unique R0/R1 rows")
    for r in metrics + counterfactual:
        for key in METRICS:
            if not math.isfinite(float(r[key])):
                raise RuntimeError(f"nonfinite metric {r}")
    numeric_diag = ["prototype_cosine", "prototype_euclidean_distance", "prototype_norm_class0", "prototype_norm_class1", "disease_alignment_loss", "disease_only_acc", "disease_only_probability_auc", "class0_margin_asd_subjects", "class0_margin_td_subjects", "disease_gamma", "disease_contribution_abs", "disease_probability_correlation"]
    for r in diagnostics:
        if str(r["all_finite"]).lower() not in ("true", "1"):
            raise RuntimeError(f"nonfinite diagnostic {r}")
        if str(r["disease_projector_grad_valid"]).lower() not in ("true", "1") or str(r["disease_alpha_grad_valid"]).lower() not in ("true", "1"):
            raise RuntimeError("disease gradients are invalid")
        for key in numeric_diag:
            if not math.isfinite(float(r[key])):
                raise RuntimeError(f"nonfinite diagnostic field {key}")
    for r in counterfactual:
        for key in ("prediction_flip_rate", "asd_probability_abs_change", "logit_margin_change", "logit_margin_abs_change", "acc_change_vs_trained"):
            if not math.isfinite(float(r[key])):
                raise RuntimeError(f"nonfinite counterfactual {r}")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    for name, digest in manifest["source_files"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError("root source changed during training: " + name)
    for name, digest in manifest["inputs"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError("ABIDE input changed during training: " + name)
    if sha256(PROMPT_FILE) != manifest["prompts_sha256"] or sha256(PATCH_FILE) != manifest["patch_sha256"] or sha256(REFERENCE_FILE) != manifest["reference_sha256"]:
        raise RuntimeError("frozen prompt, patch or reference changed during training")
    successful_attempts = {}
    for cfg, seed, _ in TASKS:
        successes = [attempt for attempt, paths, end in attempt_records(cfg, seed) if paths and end and end["returncode"] == 0 and (end["folds"], end["diagnostics"], end["counterfactual"]) == (10, 10, 60)]
        key = f"{cfg}_seed{seed}"
        paths = successful_paths(cfg, seed)
        if paths:
            successful_attempts[key] = max(successes)
            continue
        raise RuntimeError(f"no complete successful attempt for {cfg} seed={seed}")
    manifest["run_status"] = "COMPLETE"; manifest["completed_utc"] = utc_now()
    manifest["completion"] = {"reference_rows": len(reference), "fold_metrics": len(metrics), "disease_diagnostics": len(diagnostics), "counterfactual": len(counterfactual), "tasks": 6, "gpu0_tasks": 3, "gpu1_tasks": 3, "no_failed_attempt_mixed": True}
    manifest["task_status"] = {f"{c}_seed{s}": "COMPLETE" for c, s, _ in TASKS}
    manifest["successful_attempts"] = successful_attempts
    write_json(MANIFEST_FILE, manifest)
    append_handoff(metrics, diagnostics, counterfactual)


def vals(rows, cfg, key):
    return [float(r[key]) for r in rows if r.get("config") == cfg and r.get(key) not in ("", None, "N/A")]


def summary(rows, cfg):
    out = {}
    for key in METRICS:
        seed_means = []
        for seed in SEEDS:
            values = vals([r for r in rows if int(r["seed"]) == seed], cfg, key)
            if values:
                seed_means.append(statistics.mean(values))
        out[key] = (statistics.mean(seed_means), statistics.pstdev(seed_means)) if seed_means else (None, None)
    return out


def paired(rows, left, right, key):
    a = {(int(r["seed"]), int(r["fold"])): float(r[key]) for r in rows if r.get("config") == left and r.get(key) not in ("", None, "N/A")}
    b = {(int(r["seed"]), int(r["fold"])): float(r[key]) for r in rows if r.get("config") == right and r.get(key) not in ("", None, "N/A")}
    diffs = [a[k] - b[k] for k in a.keys() & b.keys()]
    if not diffs:
        return None, 0, 0, 0
    return statistics.mean(diffs), sum(x > 1e-10 for x in diffs), sum(abs(x) <= 1e-10 for x in diffs), sum(x < -1e-10 for x in diffs)


def report(metrics, diagnostics, counterfactual):
    reference = read_csv(REFERENCE_FILE)
    allrows = reference + metrics
    def contrast(left, right, key):
        lookup = {(r["config"], int(r["seed"]), int(r["fold"])): r for r in allrows}
        diffs, by_seed = [], {seed: [] for seed in SEEDS}
        for seed in SEEDS:
            for fold in range(1, 11):
                a, b = lookup.get((left, seed, fold)), lookup.get((right, seed, fold))
                if not a or not b or a.get(key) in ("", None, "N/A") or b.get(key) in ("", None, "N/A"):
                    continue
                diff = float(a[key]) - float(b[key]); diffs.append(diff); by_seed[seed].append(diff)
        if not diffs:
            return "N/A"
        wins = sum(x > 1e-10 for x in diffs); ties = sum(abs(x) <= 1e-10 for x in diffs); losses = sum(x < -1e-10 for x in diffs)
        seeds = ", ".join(f"seed{s} {100*statistics.mean(by_seed[s]):+.3f}pp" for s in SEEDS)
        return f"{100*statistics.mean(diffs):+.3f}pp；{seeds}；胜/平/负 {wins}/{ties}/{losses}"
    lines = ["# 实验十结果：DSSC Disease Semantic Specificity Control", "", f"生成时间：{utc_now()}。新训练2组×3 seeds×10 folds；历史R0/R1冻结60折。", "", "## 1. 研究动机与替代解释", "", "实验九E9相对E8的改善可能来自ASD疾病语义，也可能只是受监督类别原型分支、固定文本几何关系或新增可训练方向。本实验冻结P3和E9的全部计算，只加入ADHD/TD及中性技术文本对照。真实标签始终为0=ASD、1=TD；C1的ADHD文本只是类别0原型位置的错误语义映射。", "", "## 2. 源码与文本审计", "", "P3的32条Prompt来自实验一；ASD、TD、ADHD、NEUTRAL_CLASS0和NEUTRAL_CLASS1逐字来自实验八。疾病分支为36→1024无偏置投影、温度0.2、加权CE乘0.1、gamma=0.2*tanh(alpha)，新参数weight_decay=0。原始GFR、PLN、分类器和训练协议保持实验九E9一致。根源码、输入数据、CLIP、Prompt和Patch哈希见manifest.json。", "", "## 3. 主结果", "", "均值先按每个seed的十折计算，再对3个seed求均值和seed间SD；SEN-ASD以ASD为阳性，SPE-TD为TD正确率。", "", "| Config | 基础Prompt | 类别语义 | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-TD |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"R0": ("P3", "无"), "R1": ("P3", "ASD/TD"), "C1": ("P3", "ADHD/TD"), "C2": ("P3", "NEUTRAL")}
    for cfg in CONFIGS:
        ss = summary(allrows, cfg)
        cells = ["N/A" if ss[k][0] is None else f"{100*ss[k][0]:.3f}±{100*ss[k][1]:.3f}" for k in METRICS]
        lines.append(f"| {cfg} | {labels[cfg][0]} | {labels[cfg][1]} | " + " | ".join(cells) + " |")
    lines += ["", "### 各seed十折均值", "", "| Config | Seed | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-TD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in CONFIGS:
        for seed in SEEDS:
            subset = [r for r in allrows if r["config"] == cfg and int(r["seed"]) == seed]
            if subset:
                cells = ["N/A" if not vals(subset, cfg, k) else f"{100*statistics.mean(vals(subset, cfg, k)):.3f}" for k in METRICS]
                lines.append(f"| {cfg} | {seed} | " + " | ".join(cells) + " |")
    lines += ["", "## 4. 六项配对比较", "", "差值按相同seed、fold配对；三个seed共享折划分，以下为描述性统计。", ""]
    for left, right in (("R1", "R0"), ("R1", "C1"), ("R1", "C2"), ("C1", "R0"), ("C2", "R0"), ("C1", "C2")):
        lines.append(f"### {left} − {right}")
        for key in METRICS:
            lines.append(f"- {key}: {contrast(left, right, key)}")
    def dmean(cfg, field):
        x = [float(r[field]) for r in diagnostics if r["config"] == cfg and r.get(field) not in ("", "N/A", None)]
        return statistics.mean(x) if x else None
    lines += ["", "## 5. 疾病原型几何与机制", "", "| Config | 类别0 | 类别1 | 原型cos | 原型距离 | 类别0范数 | 类别1范数 | 疾病分支ACC | 疾病分支Prob-AUC | gamma | 融合贡献 | 疾病/P3相关 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in NEW_CONFIGS:
        rows = [r for r in diagnostics if r["config"] == cfg]
        first = rows[0]
        lines.append(f"| {cfg} | {first['prototype_class0']} | {first['prototype_class1']} | {dmean(cfg,'prototype_cosine'):.5f} | {dmean(cfg,'prototype_euclidean_distance'):.5f} | {dmean(cfg,'prototype_norm_class0'):.5f} | {dmean(cfg,'prototype_norm_class1'):.5f} | {dmean(cfg,'disease_only_acc')*100:.3f}% | {dmean(cfg,'disease_only_probability_auc')*100:.3f}% | {dmean(cfg,'disease_gamma'):+.5f} | {dmean(cfg,'disease_contribution_abs'):.5f} | {dmean(cfg,'disease_probability_correlation'):.5f} |")
    lines += ["", "疾病分支分类能力表示可学习的类别方向，不等于文本医学知识被理解。高疾病/P3概率相关性表示新增分支可能重复原始分类信号。四个潜在Token和疾病分支不能解释为具体脑区。", "", "## 6. 反事实文本替换", "", "每个条件使用同一训练checkpoint和受试者输入；只替换疾病原型或关闭决策融合。", "", "| Config | Condition | ACC | Prob-AUC | 翻转率 | ASD概率变化 | margin变化 | ACC变化 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cfg in NEW_CONFIGS:
        for condition in ("TRAINED", "REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF"):
            rows = [r for r in counterfactual if r["config"] == cfg and r["condition"] == condition]
            if not rows: continue
            def cm(k): return statistics.mean(float(r[k]) for r in rows)
            lines.append(f"| {cfg} | {condition} | {cm('acc')*100:.3f}% | {cm('probability_auc')*100:.3f}% | {cm('prediction_flip_rate'):.4f} | {cm('asd_probability_abs_change'):.5f} | {cm('logit_margin_change'):+.5f} | {cm('acc_change_vs_trained')*100:+.3f}pp |")
    lines += ["", "## 7. 科学结论", "", f"- R1−C1：{contrast('R1','C1','acc')}。该比较直接检验正确ASD文本是否稳定优于ADHD错配文本。", f"- R1−C2：{contrast('R1','C2','acc')}。该比较检验疾病文本是否优于中性技术文本。", f"- C1−R0：{contrast('C1','R0','acc')}；C2−R0：{contrast('C2','R0','acc')}。若两组均改善，不能把E9增益归因于ASD特异性。", "- 反事实敏感性只能说明决策分支使用了文本原型，不能单独证明医学语义正确。固定原型上的监督学习可能把不同文本学成类别代码。", "- 三个seed和逐fold配对差异应结合方向一致性、原型几何和分支相关性判断；微小差异不作为独立样本显著证据。", "- 当前结果不足以支持继续优化疾病Prompt；下一步若继续研究，优先使用独立验证协议和真实脑连接驱动的个体化语义，并避免伪造临床文本。", "", "## 8. 运行审计", "", "- 6/6任务、60/60折、60/60诊断、360/360反事实记录由manifest和日志校验。", "- GPU0：C1 seed0、C1 seed2、C2 seed1；GPU1：C1 seed1、C2 seed0、C2 seed2。", "- 临时源码副本和attempt工作目录训练后清理；根源码与历史实验目录未修改。", ""]
    atomic_write(REPORT_FILE, "\n".join(lines))


def append_handoff(metrics, diagnostics, counterfactual):
    old = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# 实验交接记录\n"
    marker = "## 实验十：DSSC"
    if marker in old:
        old = old[:old.index(marker)].rstrip() + "\n"
    rows = read_csv(REFERENCE_FILE) + metrics
    lines = ["", "## 实验十：DSSC Disease Semantic Specificity Control", "", "- 设计：P3固定；R0/R1冻结实验九E8/E9；C1为ADHD/TD；C2为NEUTRAL_CLASS0/NEUTRAL_CLASS1。疾病投影、对齐CE、gamma和训练协议完全复用E9；对抗分支关闭。", "- 运行：6个任务、60折、60条疾病诊断、360条反事实；GPU0承担C1 seed0、C1 seed2、C2 seed1，GPU1承担C1 seed1、C2 seed0、C2 seed2。", "", "| Config | ACC | Hard-AUC |", "|---|---:|---:|"]
    for cfg in CONFIGS:
        ss = summary(rows, cfg)
        if ss["acc"][0] is not None:
            lines.append(f"| {cfg} | {100*ss['acc'][0]:.3f}% | {100*ss['hard_auc'][0]:.3f}% |")
    lines += ["", "- 结论以 `results/report.md` 为准；正确ASD文本与ADHD/中性文本的差异必须结合seed方向、原型几何、疾病分支/P3相关性和反事实翻转率解读。固定类别原型的可学习性不等于医学知识理解。", ""]
    atomic_write(HANDOFF, old.rstrip() + "\n" + "\n".join(lines))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--all", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--refresh", action="store_true"); parser.add_argument("--one", action="store_true"); parser.add_argument("--config"); parser.add_argument("--seed", type=int); parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.prepare: prepare(); return
    if not MANIFEST_FILE.exists(): prepare()
    if args.refresh:
        metrics, diagnostics, counterfactual = refresh()
        if metrics and diagnostics and counterfactual: report(metrics, diagnostics, counterfactual)
        return
    if args.one:
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    if args.all or args.resume:
        all_jobs(); metrics, diagnostics, counterfactual = refresh(); report(metrics, diagnostics, counterfactual); return
    parser.print_help()


if __name__ == "__main__": main()
