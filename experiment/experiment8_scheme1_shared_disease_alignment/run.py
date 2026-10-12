#!/usr/bin/env python3
"""Experiment 8 runner: Shared Disease Prototype Alignment (SDPA)."""
import argparse
import csv
import difflib
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SDPA_ROOT", "/data/adodas/SPromptGL-main"))
sys.path.insert(0, str(ROOT))
LOG_DIR = EXP / "logs"
WORK_DIR = LOG_DIR / ".working"
RESULT_DIR = EXP / "results"
PROMPT_FILE = EXP / "prompts.json"
PATCH_FILE = EXP / "disease_patch.diff"
MANIFEST_FILE = EXP / "manifest.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_metrics.csv"
DIAG_FILE = RESULT_DIR / "disease_diagnostics.csv"
CF_FILE = RESULT_DIR / "counterfactual.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = ["S1", "S2", "S3", "S4", "S5"]
SEEDS = [0, 1, 2]
GPU_ASSIGNMENTS = {
    ("S1", 0): 0, ("S1", 2): 0, ("S2", 1): 0, ("S3", 0): 0,
    ("S3", 2): 0, ("S4", 1): 0, ("S5", 0): 0, ("S5", 2): 0,
    ("S1", 1): 1, ("S2", 0): 1, ("S2", 2): 1, ("S3", 1): 1,
    ("S4", 0): 1, ("S4", 2): 1, ("S5", 1): 1,
}
TASKS = [(config, seed, GPU_ASSIGNMENTS[(config, seed)]) for config in CONFIGS for seed in SEEDS]
MAX_ACTIVE = {0: 8, 1: 7}
METRICS = ["acc", "hard_auc", "probability_auc", "balanced_accuracy", "sensitivity_asd", "specificity_asd"]
TASK_END_RE = re.compile(r"TASK_END config=(S[1-5]) seed=(\d+) attempt=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+) counterfactual=(\d+)")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=(S[1-5]) seed=(\d+) attempt=(\d+) .* metrics=(\S+) diagnostics=(\S+) counterfactual=(\S+)")

ASD_PROMPTS = [
    "Autism spectrum disorder is a heterogeneous neurodevelopmental condition involving differences in social communication and restricted or repetitive behaviors.",
    "Autism spectrum disorder has been associated with heterogeneous resting-state functional connectivity patterns rather than one universal abnormal connection.",
    "Studies of autism report variable interactions involving default mode, salience, and executive-control networks across individuals and cohorts.",
    "Autism research reports diverse functional connectivity patterns in sensory, attention, and social-processing networks, with changes varying across people.",
]
TD_PROMPTS = [
    "Typically developing individuals show diverse age-related social communication and behavioral development without an autism spectrum diagnosis.",
    "Typical development involves heterogeneous resting-state functional connectivity patterns rather than one universal reference connection.",
    "Typical development includes age-dependent interactions among default mode, salience, and executive-control networks across individuals.",
    "Typical development includes variable functional connectivity patterns in sensory, attention, and social-processing networks across people.",
]
ADHD_PROMPTS = [
    "Attention-deficit/hyperactivity disorder is a heterogeneous neurodevelopmental condition involving differences in attention, impulsivity, and activity regulation.",
    "Attention-deficit/hyperactivity disorder has been associated with heterogeneous resting-state functional connectivity patterns rather than one universal abnormal connection.",
    "Studies of ADHD report variable interactions involving default mode, salience, and executive-control networks across individuals and cohorts.",
    "ADHD research reports diverse functional connectivity patterns in sensory, attention, and executive-control networks, with changes varying across people.",
]
NEUTRAL_CLASS0 = [
    "A technical MRI record describes structural image contrast and noise estimates.",
    "The imaging record summarizes structural signal quality and image artifacts.",
    "This technical description concerns scanner settings and anatomical image properties.",
    "The record contains structural imaging measurements obtained during preprocessing.",
]
NEUTRAL_CLASS1 = [
    "A technical MRI record describes functional image motion and temporal signal variation.",
    "The imaging record summarizes functional signal quality and movement artifacts.",
    "This technical description concerns fMRI preprocessing and temporal image properties.",
    "The record contains functional imaging measurements obtained during preprocessing.",
]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        temporary = stream.name
    os.replace(temporary, path)


def p1_source():
    path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def p1_sentences(data):
    if data["modality_order"] != MODALITIES:
        raise ValueError("P1 modality order changed")
    sentences = []
    for modality in MODALITIES:
        themes = data["p1"][modality]["themes"]
        if len(themes) != 2 or any(len(theme["prompts"]) != 4 for theme in themes):
            raise ValueError(f"Invalid P1 grouping for {modality}")
        sentences.extend(item["text"] for theme in themes for item in theme["prompts"])
    if len(sentences) != 32 or any(not sentence for sentence in sentences):
        raise ValueError("P1 must contain exactly 32 non-empty prompts")
    return sentences


def build_prompts():
    source, source_path = p1_source()
    output = {
        "schema_version": 1,
        "experiment": "SDPA",
        "p1_source_sha256": sha256(source_path),
        "modality_order": MODALITIES,
        "class_order": {"0": "ASD", "1": "TD"},
        "p1": deepcopy(source["p1"]),
        "disease_prompts": {
            "ASD": {"class_index": 0, "source": "fixed cautious group-level background statements", "prompts": ASD_PROMPTS},
            "TD": {"class_index": 1, "source": "fixed cautious group-level background statements", "prompts": TD_PROMPTS},
            "ADHD": {"class_index": 0, "source": "disease-mismatch control; not an ADHD task", "prompts": ADHD_PROMPTS},
            "NEUTRAL_CLASS0": {"class_index": 0, "source": "non-disease structural MRI technical descriptions", "prompts": NEUTRAL_CLASS0},
            "NEUTRAL_CLASS1": {"class_index": 1, "source": "non-disease functional MRI technical descriptions", "prompts": NEUTRAL_CLASS1},
        },
        "design_notes": {
            "label_mapping": "Raw DX_GROUP 1=ASD and 2=TD; main.py subtracts one, giving model labels 0=ASD and 1=TD.",
            "medical_scope": "Prompts are cautious group-level background statements, not individual diagnostic rules or universal biomarkers.",
            "pheno_scope": "No participant-specific cognition or psychometric values are available; none are fabricated.",
            "freeze_rule": "All text is fixed before training and is never edited from test results.",
        },
    }
    p1_sentences(output)
    write_json(PROMPT_FILE, output)
    return output, source_path


def freeze_reference():
    source = ROOT / "experiment/experiment5_semantic_projection_collapse/results/reference.csv"
    rows = [row for row in csv.DictReader(source.open(encoding="utf-8", newline="")) if row["config"] == "E0"]
    expected = {(seed, fold) for seed in SEEDS for fold in range(1, 11)}
    actual = {(int(row["seed"]), int(row["fold"])) for row in rows}
    if len(rows) != 30 or actual != expected:
        raise ValueError("Experiment 5 E0 reference is not the complete 30-fold P1 reference")
    original = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    p1 = {(int(row["seed"]), int(row["fold"])): row for row in read_csv(original) if row["config"] == "P1"}
    for row in rows:
        other = p1[(int(row["seed"]), int(row["fold"]))]
        if any(row[key] != other[key] for key in ("acc", "hard_auc", "sensitivity_asd", "specificity_asd")):
            raise RuntimeError("Frozen E0/P1 reference mismatch")
    output = []
    for row in rows:
        sensitivity = float(row["sensitivity_asd"])
        specificity = float(row["specificity_asd"])
        output.append({
            "config": "S0", "seed": row["seed"], "fold": row["fold"], "acc": row["acc"], "hard_auc": row["hard_auc"],
            "probability_auc": "", "balanced_accuracy": f"{(sensitivity + specificity) / 2:.6f}",
            "sensitivity_asd": row["sensitivity_asd"], "specificity_asd": row["specificity_asd"],
            "auc_source": "hard_label", "reference_source_sha256": sha256(source),
        })
    write_csv(REFERENCE_FILE, ["config", "seed", "fold", *METRICS, "auc_source", "reference_source_sha256"], output)
    return source


def patch_network(source):
    source = source.replace("import argparse\n", "import argparse\nimport json\n", 1)
    old = "\n".join([
        "def get_prompt_text_features():",
        "    \"\"\"Encode the fixed semantic prompts once per process.\"\"\"",
        "    global _prompt_text_features",
        "    if _prompt_text_features is None:",
        "        clip_model, _ = clip.load('RN50', device)",
        "        with torch.no_grad():",
        "            _prompt_text_features = clip_model.encode_text(text).float().detach()",
        "    return _prompt_text_features",
        "",
    ])
    new = "\n".join([
        "def get_prompt_text_features():",
        "    \"\"\"Encode the frozen Experiment 1 P1 prompts once per process.\"\"\"",
        "    global _prompt_text_features",
        "    if _prompt_text_features is None:",
        "        with open(os.environ['SDPA_PROMPT_FILE'], encoding='utf-8') as stream:",
        "            prompt_data = json.load(stream)",
        "        sentences = [item['text'] for modality in prompt_data['modality_order'] for theme in prompt_data['p1'][modality]['themes'] for item in theme['prompts']]",
        "        if len(sentences) != 32: raise ValueError('SDPA requires 32 ordered P1 prompts')",
        "        clip_model, _ = clip.load('RN50', device)",
        "        with torch.no_grad():",
        "            _prompt_text_features = clip_model.encode_text(clip.tokenize(sentences).to(device)).float().detach()",
        "    return _prompt_text_features",
        "",
        "_sdpa_text_prototypes = None",
        "def get_sdpa_text_prototypes():",
        "    \"\"\"Encode fixed disease/control text groups once per process.\"\"\"",
        "    global _sdpa_text_prototypes",
        "    if _sdpa_text_prototypes is None:",
        "        with open(os.environ['SDPA_PROMPT_FILE'], encoding='utf-8') as stream:",
        "            prompt_data = json.load(stream)",
        "        groups = prompt_data['disease_prompts']",
        "        keys = ['ASD', 'TD', 'ADHD', 'NEUTRAL_CLASS0', 'NEUTRAL_CLASS1']",
        "        sentences = [sentence for key in keys for sentence in groups[key]['prompts']]",
        "        tokens = clip.tokenize(sentences)",
        "        if tuple(tokens.shape) != (20, 77): raise ValueError('Disease/control text token shape mismatch')",
        "        clip_model, _ = clip.load('RN50', device)",
        "        with torch.no_grad():",
        "            encoded = F.normalize(clip_model.encode_text(tokens.to(device)).float(), p=2, dim=-1)",
        "            means = F.normalize(encoded.view(5, 4, 1024).mean(dim=1), p=2, dim=-1)",
        "        _sdpa_text_prototypes = {key: means[index] for index, key in enumerate(keys)}",
        "    return _sdpa_text_prototypes",
        "",
    ])
    if source.count(old) != 1: raise RuntimeError('original CLIP encoder not found')
    source = source.replace(old, new, 1)

    old = "\n".join([
        "        text_features = self.prompt_text_features",
        "        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)",
        "        text_features = self.MLP(text_features)",
        "",
    ])
    new = "\n".join([
        "        text_features = self.prompt_text_features",
        "        if text_features.size(0) == 8:",
        "            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)",
        "            text_features = self.MLP(text_features)",
        "        else:",
        "            text_features = self.MLP(text_features.view(4, 2, 4096))",
        "",
    ])
    if source.count(old) != 1: raise RuntimeError('original P1 grouping not found')
    source = source.replace(old, new, 1)

    old = """        # 扩展 text_features_norm 的维度以便与 part_x_norm 进行广播
        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # 变为 [1, 1, 1, 2, 9]
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)  # 变为 [871, 4, 4, 1, 9]

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))  # 结果为 [871, 4, 4, 1, 2]
"""
    new = """        if text_features_norm.dim() == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            part_x_norm_expanded = part_x_norm.unsqueeze(-2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, text_features_norm).unsqueeze(-2)
"""
    if source.count(old) != 1: raise RuntimeError('original P1 similarity path not found')
    source = source.replace(old, new, 1)

    old = "        self.register_buffer('prompt_text_features', get_prompt_text_features())\n"
    new = "\n".join([
        "        self.register_buffer('prompt_text_features', get_prompt_text_features())",
        "        self.sdpa_config = os.environ.get('SDPA_CONFIG', 'S0')",
        "        self.sdpa_enabled = False",
        "        self.sdpa_counterfactual = None",
        "",
    ])
    if source.count(old) != 1: raise RuntimeError('PLN init anchor not found')
    source = source.replace(old, new, 1)

    old = "\n".join([
        "        output, hidden = self.Outputlayer(x, attn_embedding)",
        "        return output, hidden, attn_map",
        "",
    ])
    new = "\n".join([
        "        output, hidden = self.Outputlayer(x, attn_embedding)",
        "        self.last_base_logprob = output",
        "        self.last_disease_logits = None",
        "        self.last_subject_semantic = None",
        "        self.last_subject_projection = None",
        "        self.last_sdpa_prelogits = output",
        "        self.last_sdpa_output = output",
        "        self.last_sdpa_gamma = 0.0",
        "        if self.sdpa_enabled:",
        "            subject_projection = self.disease_projector(hidden)",
        "            subject_semantic = F.normalize(subject_projection, p=2, dim=-1)",
        "            prototypes = self.disease_prototypes",
        "            condition = self.sdpa_counterfactual",
        "            if condition == 'ADHD': prototypes = self.sdpa_adhd_prototypes",
        "            elif condition == 'NEUTRAL': prototypes = self.sdpa_neutral_prototypes",
        "            elif condition == 'SWAPPED': prototypes = prototypes.flip(0)",
        "            disease_logits = torch.matmul(subject_semantic, prototypes.t()) / 0.2",
        "            gamma = 0.2 * torch.tanh(self.disease_alpha)",
        "            prelogits = output + gamma * disease_logits",
        "            if condition == 'OFF' or self.sdpa_config == 'S4': prelogits = output",
        "            final_output = output if self.sdpa_config == 'S4' else F.log_softmax(prelogits, dim=-1)",
        "            self.last_disease_logits = disease_logits",
        "            self.last_subject_semantic = subject_semantic",
        "            self.last_subject_projection = subject_projection",
        "            self.last_sdpa_prelogits = prelogits",
        "            self.last_sdpa_gamma = float(gamma.detach().item())",
        "            output = final_output",
        "        self.last_sdpa_output = output",
        "        return output, hidden, attn_map",
        "",
    ])
    if source.count(old) != 1: raise RuntimeError('PLN output anchor not found')
    return source.replace(old, new, 1)



def patch_model(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    old = """        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
"""
    new = """        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
        sdpa_config = os.environ.get("SDPA_CONFIG", "S0")
        self.ModalFusion.sdpa_config = sdpa_config
        self.ModalFusion._sdpa_grad_valid = {"projector": False, "alpha": False}
        self.ModalFusion._sdpa_prediction_finite = True
        if sdpa_config in ("S1", "S2", "S3", "S4", "S5"):
            prototype_bank = get_sdpa_text_prototypes()
            if sdpa_config == "S2":
                selected = ("ADHD", "TD")
            elif sdpa_config == "S3":
                selected = ("NEUTRAL_CLASS0", "NEUTRAL_CLASS1")
            else:
                selected = ("ASD", "TD")
            self.ModalFusion.register_buffer("disease_prototypes", torch.stack([prototype_bank[name] for name in selected]))
            self.ModalFusion.register_buffer("sdpa_adhd_prototypes", torch.stack([prototype_bank["ADHD"], prototype_bank["TD"]]))
            self.ModalFusion.register_buffer("sdpa_neutral_prototypes", torch.stack([prototype_bank["NEUTRAL_CLASS0"], prototype_bank["NEUTRAL_CLASS1"]]))
            self.ModalFusion.disease_projector = nn.Linear(36, 1024, bias=False).to(dev)
            my_weight_init(self.ModalFusion.disease_projector)
            self.ModalFusion.disease_alpha = nn.Parameter(torch.zeros((), device=dev))
            self.ModalFusion.sdpa_enabled = True
            disease_parameters = list(self.ModalFusion.disease_projector.parameters()) + [self.ModalFusion.disease_alpha]
            if sum(parameter.numel() for parameter in disease_parameters) != 36865:
                raise RuntimeError("SDPA must add exactly 36865 parameters")
            self.optimizer_MF.add_param_group({"params": disease_parameters, "weight_decay": 0.0})
"""
    if source.count(old) != 1:
        raise RuntimeError("optimizer/init anchor not found")
    source = source.replace(old, new, 1)

    old = """            prob,_,_ = self.ModalFusion(feat)

            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            
            loss = cls_loss + all_graph_loss
            
            loss.backward()
            
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
"""
    new = """            prob,_,_ = self.ModalFusion(feat)

            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            if self.ModalFusion.sdpa_enabled:
                train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)
                align_loss = F.cross_entropy(self.ModalFusion.last_disease_logits[train_idx], self.targ[train_idx], weight=self.weight)
                self.ModalFusion._sdpa_last_alignment_loss = float(align_loss.detach().item())
                lambda_align = 0.0 if os.environ.get("SDPA_CONFIG") == "S5" else 0.1
                loss = cls_loss + all_graph_loss + lambda_align * align_loss
            else:
                loss = cls_loss + all_graph_loss

            loss.backward()
            if self.ModalFusion.sdpa_enabled:
                for key, parameter in (("projector", self.ModalFusion.disease_projector.weight), ("alpha", self.ModalFusion.disease_alpha)):
                    grad = parameter.grad
                    valid = grad is not None and bool(torch.isfinite(grad).all().item()) and bool((grad.abs().sum() > 0).item())
                    self.ModalFusion._sdpa_grad_valid[key] = self.ModalFusion._sdpa_grad_valid[key] or valid

            self.optimizer_MP0.step()
            self.optimizer_MF.step()
"""
    if source.count(old) != 1:
        raise RuntimeError("simple-2 loss anchor not found")
    source = source.replace(old, new, 1)

    old = """        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        
        if mode == 'pre-train':
"""
    new = """        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        if tst and self.ModalFusion.sdpa_enabled:
            self._sdpa_eval_feat = feat.detach()

        if mode == 'pre-train':
"""
    if source.count(old) != 1:
        raise RuntimeError("evaluation feature anchor not found")
    source = source.replace(old, new, 1)

    old = """        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
"""
    new = """        if tst and self.ModalFusion.sdpa_enabled:
            eval_tensor = torch.as_tensor(eval_idx, dtype=torch.long, device=prob.device)
            self._sdpa_test_probability = prob.index_select(0, eval_tensor).detach().exp().cpu().numpy()
            self._sdpa_test_base_probability = self.ModalFusion.last_base_logprob.index_select(0, eval_tensor).detach().exp().cpu().numpy()
            self._sdpa_test_disease_logits = self.ModalFusion.last_disease_logits.index_select(0, eval_tensor).detach().cpu().numpy()
            self._sdpa_test_prelogits = self.ModalFusion.last_sdpa_prelogits.index_select(0, eval_tensor).detach().cpu().numpy()
        self.ModalFusion._sdpa_prediction_finite = bool(torch.isfinite(prob).all().item())
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
"""
    if source.count(old) != 2:
        raise RuntimeError("evaluation output anchor not found")
    source = source.replace(old, new, 1)

    point = "    def cal_acc(self, eval_idx):\n"
    methods = '''    def write_sdpa_fold_metrics(self, fold, acc, hard_auc, conf_mat):
        if not os.environ.get("SDPA_METRIC_PATH") or not self.ModalFusion.sdpa_enabled:
            return
        labels = self.targ[torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)].detach().cpu().numpy()
        probability = self._sdpa_test_probability
        sensitivity = float(conf_mat[0, 0] / max(1, conf_mat[0].sum()))
        specificity = float(conf_mat[1, 1] / max(1, conf_mat[1].sum()))
        row = {
            "config": os.environ["SDPA_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold),
            "acc": float(acc), "hard_auc": float(hard_auc),
            "probability_auc": float(roc_auc_score((labels == 0).astype(int), probability[:, 0])),
            "balanced_accuracy": (sensitivity + specificity) / 2.0,
            "sensitivity_asd": sensitivity, "specificity_asd": specificity,
        }
        with open(os.environ["SDPA_METRIC_PATH"], "a", encoding="utf-8") as stream:
            json.dump(row, stream); stream.write("\\n")

    def write_sdpa_diagnostics(self, fold):
        if not os.environ.get("SDPA_DIAGNOSTIC_PATH") or not self.ModalFusion.sdpa_enabled:
            return
        logits = self.ModalFusion.last_disease_logits
        projected = self.ModalFusion.last_subject_projection
        semantic = self.ModalFusion.last_subject_semantic
        base = self.ModalFusion.last_base_logprob
        fused = self.ModalFusion.last_sdpa_prelogits
        prototypes = self.ModalFusion.disease_prototypes
        train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=self.targ.device)
        test_idx = torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)
        target = self.targ.index_select(0, test_idx)
        with torch.no_grad():
            align = F.cross_entropy(logits.index_select(0, train_idx), self.targ.index_select(0, train_idx), weight=self.weight)
            disease_probability = torch.softmax(logits.index_select(0, test_idx), dim=-1)
            disease_prediction = disease_probability.argmax(dim=1)
            test_margin = (logits[:, 0] - logits[:, 1]).index_select(0, test_idx)
            base_asd_probability = base.index_select(0, test_idx).exp()[:, 0].cpu().numpy()
            disease_asd_probability = disease_probability[:, 0].cpu().numpy()
            correlation = 0.0
            if np.std(base_asd_probability) > 0 and np.std(disease_asd_probability) > 0:
                correlation = float(np.corrcoef(base_asd_probability, disease_asd_probability)[0, 1])
            prediction = self.ModalFusion.last_sdpa_output.index_select(0, test_idx)
            row = {
                "config": os.environ["SDPA_CONFIG"], "seed": int(self.hyperpm.seed), "fold": int(fold),
                "prototype_cosine": float(F.cosine_similarity(prototypes[0], prototypes[1], dim=0).item()),
                "asd_prototype_norm": float(torch.linalg.vector_norm(prototypes[0]).item()),
                "td_prototype_norm": float(torch.linalg.vector_norm(prototypes[1]).item()),
                "prototype_distance": float(torch.linalg.vector_norm(prototypes[0] - prototypes[1]).item()),
                "projected_subject_norm_mean": float(torch.linalg.vector_norm(projected, dim=-1).mean().item()),
                "normalized_subject_norm_mean": float(torch.linalg.vector_norm(semantic, dim=-1).mean().item()),
                "alignment_loss_train": float(align.item()),
                "disease_only_acc": float((disease_prediction == target).float().mean().item()),
                "disease_only_probability_auc": float(roc_auc_score((target == 0).cpu().numpy(), disease_probability[:, 0].cpu().numpy())),
                "disease_logit_asd_mean": float(logits.index_select(0, test_idx)[:, 0].mean().item()),
                "disease_logit_td_mean": float(logits.index_select(0, test_idx)[:, 1].mean().item()),
                "disease_logit_asd_std": float(logits.index_select(0, test_idx)[:, 0].std(unbiased=False).item()),
                "disease_logit_td_std": float(logits.index_select(0, test_idx)[:, 1].std(unbiased=False).item()),
                "asd_margin_mean": float(test_margin[target == 0].mean().item()),
                "td_margin_mean": float(test_margin[target == 1].mean().item()),
                "correct_prototype_match_rate": float((disease_prediction == target).float().mean().item()),
                "gamma": float(self.ModalFusion.last_sdpa_gamma),
                "gamma_abs": abs(float(self.ModalFusion.last_sdpa_gamma)),
                "gamma_sign": int(np.sign(self.ModalFusion.last_sdpa_gamma)),
                "mean_abs_fusion_logit_contribution": float((fused - base).index_select(0, test_idx).abs().mean().item()),
                "std_fusion_logit_contribution": float((fused - base).index_select(0, test_idx).std(unbiased=False).item()),
                "p1_disease_probability_correlation": correlation,
                "projector_grad_valid": bool(self.ModalFusion._sdpa_grad_valid["projector"]),
                "alpha_grad_valid": bool(self.ModalFusion._sdpa_grad_valid["alpha"]),
                "alpha_grad_exempt": os.environ["SDPA_CONFIG"] == "S4",
                "prediction_nonfinite": bool((~torch.isfinite(prediction)).any().item()),
                "nonfinite": bool((~torch.isfinite(logits)).any().item() or (~torch.isfinite(projected)).any().item()
                                  or (~torch.isfinite(semantic)).any().item() or (~torch.isfinite(base)).any().item()),
            }
        with open(os.environ["SDPA_DIAGNOSTIC_PATH"], "a", encoding="utf-8") as stream:
            json.dump(row, stream); stream.write("\\n")

    def run_sdpa_counterfactual(self, fold):
        if os.environ.get("SDPA_CONFIG") != "S1" or not os.environ.get("SDPA_COUNTERFACTUAL_PATH"):
            return
        self.MessagePassing0.eval(); self.ModalFusion.eval()
        feature = self._sdpa_eval_feat
        index = torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.targ.device)
        target = self.targ.index_select(0, index)
        real_probability = self._sdpa_test_probability.copy()
        real_prediction = real_probability.argmax(axis=1)
        labels = target.detach().cpu().numpy()
        real_acc = float((real_prediction == labels).mean())
        rows = []
        for condition in ("REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF"):
            self.ModalFusion.sdpa_counterfactual = None if condition == "REAL" else condition
            with torch.no_grad():
                output, _, _ = self.ModalFusion(feature)
            probability = output.index_select(0, index).detach().exp().cpu().numpy()
            prediction = probability.argmax(axis=1)
            conf = confusion_matrix(labels, prediction, labels=[0, 1])
            sensitivity = float(conf[0, 0] / max(1, conf[0].sum()))
            specificity = float(conf[1, 1] / max(1, conf[1].sum()))
            hard_auc = roc_auc_score(one_hot(target, self.n_class).cpu().numpy(),
                                     one_hot(torch.as_tensor(prediction, device=target.device), self.n_class).cpu().numpy())
            disease_logits = self.ModalFusion.last_disease_logits.index_select(0, index)
            prelogits = self.ModalFusion.last_sdpa_prelogits.index_select(0, index).detach().cpu().numpy()
            real_disease_logits = torch.as_tensor(self._sdpa_test_disease_logits, device=target.device)
            margin = disease_logits[:, 0] - disease_logits[:, 1]
            real_margin = real_disease_logits[:, 0] - real_disease_logits[:, 1]
            rows.append({
                "config": "S1", "seed": int(self.hyperpm.seed), "fold": int(fold), "condition": condition,
                "acc": float((prediction == labels).mean()), "hard_auc": float(hard_auc),
                "probability_auc": float(roc_auc_score((labels == 0).astype(int), probability[:, 0])),
                "balanced_accuracy": (sensitivity + specificity) / 2.0,
                "sensitivity_asd": sensitivity, "specificity_asd": specificity,
                "prediction_flip_rate": float((prediction != real_prediction).mean()),
                "mean_probability_abs_change": float(np.abs(probability - real_probability).mean()),
                "mean_asd_probability_abs_change": float(np.abs(probability[:, 0] - real_probability[:, 0]).mean()),
                "acc_change_vs_real": float((prediction == labels).mean() - real_acc),
                "mean_abs_logit_change": float(np.abs(prelogits - self._sdpa_test_prelogits).mean()),
                "mean_disease_margin_change": float((margin - real_margin).mean().item()),
                "mean_abs_disease_margin_change": float((margin - real_margin).abs().mean().item()),
                "real_output_max_abs_diff": float(np.abs(probability - real_probability).max()) if condition == "REAL" else "",
            })
        self.ModalFusion.sdpa_counterfactual = None
        with open(os.environ["SDPA_COUNTERFACTUAL_PATH"], "a", encoding="utf-8") as stream:
            for row in rows:
                json.dump(row, stream); stream.write("\\n")

'''
    if source.count(point) != 1:
        raise RuntimeError("model method insertion point not found")
    return source.replace(point, methods + point, 1)


def patch_main(source):
    old = """        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)
"""
    new = old + """        if os.environ.get("SDPA_CONFIG"):
            agent._sdpa_fold_number = clk
"""
    if source.count(old) != 1:
        raise RuntimeError("main fold anchor not found")
    source = source.replace(old, new, 1)
    old = """        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
        sens.append(sen(conf_mat, hyperpm.nclass))
"""
    new = """        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
        if os.environ.get("SDPA_CONFIG"):
            agent.write_sdpa_fold_metrics(clk, cur_tst_acc, cur_tst_auc, conf_mat)
            agent.write_sdpa_diagnostics(clk)
            agent.run_sdpa_counterfactual(clk)
        sens.append(sen(conf_mat, hyperpm.nclass))
"""
    if source.count(old) != 1:
        raise RuntimeError("main test metric anchor not found")
    return source.replace(old, new, 1)


def expected_patches():
    originals = {name: (ROOT / name).read_text(encoding="utf-8") for name in ("main.py", "model.py", "network.py")}
    patched = {"main.py": patch_main(originals["main.py"]), "model.py": patch_model(originals["model.py"]),
               "network.py": patch_network(originals["network.py"])}
    chunks = []
    for name in ("main.py", "model.py", "network.py"):
        chunks.extend(difflib.unified_diff(originals[name].splitlines(keepends=True), patched[name].splitlines(keepends=True),
                                           fromfile=f"a/{name}", tofile=f"b/{name}", n=3))
    return patched, "".join(chunks)


def validate_patch_and_tokenization(prompts):
    from CLIP.clip import clip
    import torch
    from torch import nn
    p1 = p1_sentences(prompts)
    if tuple(clip.tokenize(p1).shape) != (32, 77):
        raise RuntimeError("P1 CLIP token shape mismatch")
    groups = prompts["disease_prompts"]
    disease = [sentence for name in ("ASD", "TD", "ADHD", "NEUTRAL_CLASS0", "NEUTRAL_CLASS1") for sentence in groups[name]["prompts"]]
    if tuple(clip.tokenize(disease).shape) != (20, 77):
        raise RuntimeError("disease/control CLIP token shape mismatch")
    patched, patch_text = expected_patches()
    for marker in ("SDPA_CONFIG", "disease_projector", "disease_alpha", "SDPA_COUNTERFACTUAL_PATH"):
        if marker not in patch_text:
            raise RuntimeError(f"missing required patch marker: {marker}")
    for name, body in patched.items():
        compile(body, name, "exec")
    projector = nn.Linear(36, 1024, bias=False)
    alpha = nn.Parameter(torch.zeros(()))
    if projector.weight.numel() + alpha.numel() != 36865 or alpha.item() != 0:
        raise RuntimeError("new parameter count or alpha initialization mismatch")
    return patch_text


def preflight_script():
    return r'''
import os, json
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from model import EvalHelper
from network import get_sdpa_text_prototypes
modal = np.load("data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
table = pd.read_csv("data/ABIDE/processed_standard_data.csv").values
dims = [len(modal[key]) for key in modal]
x = table[:, :-1].astype("float32")
y = table[:, -1].astype("int64") - 1
assert x.shape == (871,320) and dims == [48,6,10,256]
assert dict(zip(*np.unique(y,return_counts=True))) == {0:403,1:468}
train, test = next(StratifiedKFold(10,random_state=10,shuffle=True).split(x,y))
def hp():
    return SimpleNamespace(GC_mode="weighted-cosine", MF_mode="concat", MP_mode="GCN", datname="ABIDE",
        dropout=0.35, lr=0.0038, mode="simple-2", n_head=2, n_hidden=18, nlayer=1, reg=0.11,
        nmodal=4, nclass=2, alpha=0.5, th=0.9, theta_smooth=1.0, theta_degree=0.5,
        theta_sparsity=0.0, seed=0, early=200, nepoch=450)
def make(config):
    os.environ["SDPA_CONFIG"] = config
    torch.manual_seed(2017)
    return EvalHelper(dims,x,y,hp(),train,test)
base = make("S0")
base_state = {k:v.detach().clone() for k,v in base.ModalFusion.state_dict().items()}
base_gcn = {k:v.detach().clone() for k,v in base.MessagePassing0.state_dict().items()}
base_count = sum(p.numel() for p in base.ModalFusion.parameters())
base_gcn_count = sum(p.numel() for p in base.MessagePassing0.parameters())
z = torch.as_tensor(x,device=base.dev)
base.ModalFusion.eval()
with torch.no_grad(): p0, hidden0, _ = base.ModalFusion(z)
assert tuple(hidden0.shape) == (871,36)
real = None
new_state = None
max_diff = 0.
prototype_geometry = {}
bank = get_sdpa_text_prototypes()
for config in ("S1","S2","S3","S4","S5"):
    item = make(config)
    for k,v in base_state.items(): assert torch.equal(v,item.ModalFusion.state_dict()[k]), config+" base "+k
    for k,v in base_gcn.items(): assert torch.equal(v,item.MessagePassing0.state_dict()[k]), config+" GCN "+k
    assert sum(p.numel() for p in item.ModalFusion.parameters()) == base_count+36865
    extra = [item.ModalFusion.disease_projector.weight,item.ModalFusion.disease_alpha]
    if new_state is None: new_state = [p.detach().clone() for p in extra]
    for p,v in zip(extra,new_state): assert torch.equal(p,v), config+" new initialization"
    ids = [id(p) for group in item.optimizer_MF.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(item.ModalFusion.parameters()))
    assert item.optimizer_MF.param_groups[0]["weight_decay"] == .11
    assert item.optimizer_MF.param_groups[1]["weight_decay"] == 0.
    assert all(g["lr"]==.0038 for g in item.optimizer_MF.param_groups)
    assert item.ModalFusion.disease_alpha.item() == 0.
    selected = ("ADHD","TD") if config=="S2" else ("NEUTRAL_CLASS0","NEUTRAL_CLASS1") if config=="S3" else ("ASD","TD")
    for j,k in enumerate(selected): assert torch.equal(item.ModalFusion.disease_prototypes[j],bank[k])
    item.ModalFusion.eval()
    with torch.no_grad():
        out,hid,_ = item.ModalFusion(z)
        assert torch.equal(hid,hidden0)
        diff = (out-p0).abs().max().item(); max_diff = max(max_diff,diff)
        assert diff < 1e-5
        assert tuple(item.ModalFusion.last_disease_logits.shape)==(871,2)
        expected = item.ModalFusion.last_subject_semantic @ item.ModalFusion.disease_prototypes.T / .2
        assert torch.equal(expected,item.ModalFusion.last_disease_logits)
        assert torch.isfinite(out).all() and torch.allclose(out.exp().sum(-1),torch.ones(871,device=out.device),atol=1e-6)
    prot = item.ModalFusion.disease_prototypes
    prototype_geometry[config] = {"cosine":float((prot[0]*prot[1]).sum().item()),
        "norm0":float(prot[0].norm().item()),"norm1":float(prot[1].norm().item()),"distance":float((prot[0]-prot[1]).norm().item())}
    assert torch.isfinite(prot).all() and torch.all(prot.norm(dim=-1)>0)
    os.environ["SDPA_CONFIG"] = config
    for step in range(2): item.run_epoch("simple-2")
    assert item.ModalFusion._sdpa_grad_valid["projector"], config+" projector"
    assert item.ModalFusion._sdpa_grad_valid["alpha"] == (config!="S4"), config+" alpha"
    if config=="S1": real=item
os.environ["SDPA_CONFIG"]="S1"
real.ModalFusion.eval(); real.MessagePassing0.eval()
os.environ["SDPA_METRIC_PATH"] = os.path.abspath("preflight_metrics.jsonl")
os.environ["SDPA_DIAGNOSTIC_PATH"] = os.path.abspath("preflight_diag.jsonl")
os.environ["SDPA_COUNTERFACTUAL_PATH"] = os.path.abspath("preflight_cf.jsonl")
acc,auc,conf = real.print_tst_acc("simple-2")
before={k:v.detach().clone() for k,v in real.ModalFusion.state_dict().items()}
real.write_sdpa_fold_metrics(1,acc,auc,conf)
real.write_sdpa_diagnostics(1)
real.run_sdpa_counterfactual(1)
for k,v in before.items(): assert torch.equal(v,real.ModalFusion.state_dict()[k]), "CF changes parameter "+k
assert real.ModalFusion.sdpa_counterfactual is None
diag=json.loads(open("preflight_diag.jsonl").read())
assert all(k in diag for k in ("disease_logit_asd_std","disease_logit_td_std","projected_subject_norm_mean"))
cf=[json.loads(line) for line in open("preflight_cf.jsonl")]
assert len(cf)==5 and cf[0]["real_output_max_abs_diff"]==0.
assert all("mean_abs_logit_change" in r and "mean_disease_margin_change" in r for r in cf)
assert not diag["nonfinite"] and not diag["prediction_nonfinite"]
print("SDPA_PREFLIGHT_OK "+json.dumps({"base_pln_parameters":base_count,"base_gcn_parameters":base_gcn_count,
    "new_parameters":36865,"initial_output_max_abs_diff":max_diff,
    "prototype_geometry":prototype_geometry,"all_checks_passed":True}))
'''


def preflight_runtime(work):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "SDPA_CONFIG": "S1", "SDPA_PROMPT_FILE": str(PROMPT_FILE)})
    result = subprocess.run([sys.executable, "-c", preflight_script()], cwd=work, env=env,
                            text=True, capture_output=True, timeout=900)
    if result.returncode or "SDPA_PREFLIGHT_OK" not in result.stdout:
        raise RuntimeError("runtime preflight failed:\n" + result.stdout[-4000:] + "\n" + result.stderr[-8000:])
    line = next(line for line in result.stdout.splitlines() if line.startswith("SDPA_PREFLIGHT_OK "))
    return json.loads(line.split(" ", 1)[1])


def github_head():
    try:
        result = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        return result.split()[0] if result.split() else None
    except Exception:
        return None


def make_manifest(reference_source, p1_path):
    clip_path = Path.home() / ".cache" / "clip" / "RN50.pt"
    return {
        "experiment": "SDPA", "schema_version": 1, "created_utc": utc_now(), "source_root": str(ROOT),
        "github_main_commit_at_prepare": github_head(),
        "source_files": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "inputs": {name: sha256(ROOT / name) for name in ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")},
        "clip_rn50": {"path": str(clip_path), "sha256": sha256(clip_path) if clip_path.is_file() else None},
        "prompts_sha256": sha256(PROMPT_FILE), "patch_sha256": sha256(PATCH_FILE),
        "reference_sha256": sha256(REFERENCE_FILE), "reference_source_sha256": sha256(reference_source),
        "p1_source_sha256": sha256(p1_path), "label_mapping": {"raw_1": "ASD/model 0", "raw_2": "TD/model 1"},
        "audit": {
            "p1_shape_path": "[32,1024] -> [4,2,4096] -> [4,2,9] -> original modality Softmax-Max",
            "text_prototypes": "normalize each frozen CLIP text, average four per category, normalize category mean",
            "new_parameters": {"disease_projector": 36864, "disease_alpha": 1, "total": 36865},
            "fusion": "gamma=.2*tanh(alpha); log_softmax(P1_logprob + gamma*disease_logits)",
            "alignment": "weighted CE on training indices only; lambda=.1 for S1-S4, 0 for S5",
            "root_source_modified": False,
        },
        "protocol": {
            "folds": 10, "stratified_random_state": 10, "checkpoint_selection": "original validation ACC",
            "epoch_cap": 450, "early_patience": 200, "auc": "original hard-label AUC",
            "seeds": SEEDS,
            "configs": {"S0": "frozen Experiment5 E0 P1", "S1": "ASD/TD alignment+fusion",
                        "S2": "ADHD/TD mismatch", "S3": "neutral technical text",
                        "S4": "alignment without decision fusion", "S5": "fusion without alignment"},
            "gpu_assignments": {f"{c}_seed{s}": g for c, s, g in TASKS},
        },
        "expected": {"fold_metrics": 150, "disease_diagnostics": 150, "counterfactual": 150},
        "run_status": "PREPARED",
    }


def write_readme():
    atomic_write(EXP / "README.md", """# 实验八（方案一）：SDPA 共享疾病原型对齐

## 假设与代码审计

固定实验一 P1 的 32 条模态文本，在四模态已经融合的 36 维受试者隐藏表示之后引入共享疾病类别原型。核心假设是：显式保留 ASD/TD 的两个语义分数并进行类别对齐，是否能提供原始 P1 尚未利用的判别信息。只运行 S1–S5；无文本、结构或超参数搜索。

审计时 GitHub main 为 0925d5011397b0feb28d04de761b81f163f99914，包含实验七；本地 HEAD 为 0bf4d34。main.py、model.py、network.py、layers.py、utils.py 与原始 Shell 的 SHA256 均与该远端版本相同。完整源文件、ABIDE 输入和 CLIP RN50 权重 SHA256 见 manifest.json。根目录 network.py 仍载入原始八句通用文本；实验八只在临时副本接入已核验的实验一 P1，再追加疾病分支，根源码不改动。

ABIDE 最终表为 871×321：320 特征加 DX_GROUP。模态维度按顺序为 PHENO 48、ANAT 6、FUNC 10、Correlation 256。预处理 notebook 将原始 DX_GROUP 1/2 写入数据；main.py 减一后为 0=ASD（403例）、1=TD/NC（468例）。概率AUC使用 exp(logprob) 的第0列与 labels==0。PHENO包含站点/人口学信息，ANAT/FUNC主要为质量与运动/信号信息，Correlation是压缩连接特征；不能将四个局部Token解释为具体脑区。

P1路径逐计算步骤保留：冻结 RN50 文本 [32,1024] → [4,2,4096] → 原始 4096→128→9 MLP → 模态特异余弦相似度 [871,4,4,2] → Softmax-Max 标量 [871,4,4] → 加权局部特征及原始 global/local 融合 → OutputLayer。分类头返回 log-softmax [871,2] 和 hidden [871,36]。主干 PLN 为575,121参数，GFR为887,836参数；合计1,462,957。所有新配置在此基础上新增36,865参数。

## 前七轮证据

| 实验 | 主要真实结果与结论 |
|---|---|
| 1 模态专属P1 | P1 ACC 91.80%、Hard-AUC 91.62%；P0约91.69%/91.54%，收益很小，P1保留为基线。 |
| 2 对比状态文本 | V2约91.77%/91.60%，低于P1；仅措辞改变没有稳定增益。 |
| 3 有效性诊断 | P1/D1最高均值，但固定缩放及无关文本对照差异很小；原型分散不能直接预测性能。 |
| 4 ASD相关FC文本 | C1约91.46%/91.29%，C2约91.50%/91.28%，均低于P1。 |
| 5 投影退化 | MLP权重显著收缩；去输出ReLU或去MLP衰减的E1/E2/E3均未超过E0/P1。 |
| 6 保留方向标量门控 | B1约91.73%/91.59%，B2约91.77%/91.52%，未显示整体收益。 |
| 7 双主题特征残差 | F1约91.69%/91.49%，F2/F3/F4也未稳定超过P1；反事实敏感性不能单独证明语义价值。 |

这些结果支持停止普通Prompt改写、旧MLP微调、标量方向门控和双主题残差的追分。本实验检验类别级共享语义，不复制实验七模型。历史结果只作描述性证据；当前静态文本与实际压缩特征可能缺乏充分对应。

## 文献核查与借鉴边界

项目中仅发现SPromptGL论文PDF，没有用户提到的三篇上传PDF；以下核查使用公开原始论文，未声称读取不存在的本地附件。

[BrainPrompt MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf) 的3.3节比较受试者表示与疾病类别文本，使用额外疾病交叉熵。本实验借鉴类别语义监督；使用现有CLIP与P1，不复现其Llama、多层Prompt和人群图全套方法。

[SABER ICME 2026](https://arxiv.org/html/2607.01901v1) 的Module III包含图级语义对齐、受控语义注入与疾病监督。本实验借鉴让疾病分数参与决策及限制贡献的思想；其Graph-CLS、患者文本交叉注意力、多尺度超图和对比损失均未加入。

[BrainSCK MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1779_paper.pdf) 将知识注入与疾病知识激活区分为两个阶段，使用HCP人口学和心理测量记录。本项目没有相应个体认知记录，因此没有伪造认知文本。本次不加入ROI知识图谱、个体化文本、Sinkhorn或额外LLM推理；其他候选论文仅作为用户提供的后续方向，未据未核验细节实现组件。

文本审慎性依据还包括[NIMH的ASD核心表型说明](https://www.nimh.nih.gov/health/publications/autism-spectrum-disorder)、[网络特异的混合FC方向研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC5027998/)和[ASD脑网络个体异质性研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9793594/)。这些来源支持群体层面与异质性的措辞，不把静态语句解释为个体生物标志物。所有规定文本保持逐字不变，未根据结果修订。

## 固定模型与对照

| 配置 | 类别0/类别1文本 | 对齐损失权重 | 疾病决策融合 |
|---|---|---:|---|
| S0 | 原始P1，冻结历史30折 | 无 | 无 |
| S1 ASD-Real-Joint | ASD / TD | 0.1 | 有 |
| S2 ADHD-Mismatch-Joint | ADHD / TD | 0.1 | 有 |
| S3 Neutral-Text-Joint | 两组MRI技术文本 | 0.1 | 有 |
| S4 ASD-Align-Only | ASD / TD | 0.1 | 无 |
| S5 ASD-Fusion-Only | ASD / TD | 0 | 有 |

S1–S5仅存在表中的差异。S2保留TD，训练仍是ASD/TD任务；S3仍使用文本及类别监督，不能称为无文本模型。每组四句文本由冻结RN50编码、逐句L2归一化、组内平均并再次归一化。ASD/TD、ADHD/TD、中性对照的初始原型cos分别约0.946976、0.932048、0.968670；单位范数，类别距离约0.325650、0.368651、0.250318。几何差异在机制分析中保留。

计算为：

    z = normalize(disease_projector(subject_hidden))
    disease_logits = z @ disease_prototypes.T / 0.2
    L_align = weighted_CE(disease_logits[train_idx], labels[train_idx])
    gamma = 0.2 * tanh(disease_alpha)
    final_logprob = log_softmax(base_logprob + gamma * disease_logits)
    L_total = original_ClsLoss(final_logprob) + original_graph_loss + lambda * L_align

两个疾病logits完整保留，新增路径不做Max。S4直接返回base_logprob；其alpha注册但不参与决策，零梯度为预期消融。alpha可为负，诊断记录符号，不把负向贡献解释成正确类别语义利用。

原始初始化和ModalFusion.apply(my_weight_init)完成后才注册Linear(36,1024,bias=False)及零alpha；投影采用原有my_weight_init的Xavier规则。新增参数36,864+1=36,865，每个配置完全相同，且优化器无遗漏/重复。原参数Adam lr=.0038、weight_decay=.11，新增参数同lr、weight_decay=0。零alpha初始输出与P1最大绝对差为1.19e-7。最小预检覆盖五配置原有与新增参数初始化、输入/输出形状、参数数量、原型类别对应、归一化、两步真实优化、完整诊断字段和REAL反事实一致性。

## 协议、诊断和运行

沿用原始Shell：weighted-cosine、concat、GCN、dropout .35、lr .0038、simple-2、2 heads、hidden18、1 layer、reg .11；其他默认参数不变。StratifiedKFold(10,random_state=10,shuffle=True)，seed0/1/2，450 epoch上限，patience200。按原程序ACC最优checkpoint选择、类别权重、图损失、transductive传播及Hard-label AUC统计。原程序val_idx=tst_idx，因此数字只作历史一致协议比较，不能表述为独立测试泛化估计。S0缺少逐折概率，Probability-AUC标为N/A。BA直接计算(SEN-ASD+SPE-ASD)/2。

15个进程各自连续十折；GPU0八任务、GPU1七任务同时运行。前一轮因发现两项输出字段遗漏而中断，修正后从各任务第一折完整重跑；旧attempt保留在日志，指标不混入汇总。每任务使用独立文件锁与attempt文件；调度器互斥，完整任务跳过，失败任务完整重跑；临时源码与最佳checkpoint放本机快速磁盘。成功后保留150条折指标、150条机制诊断与150条S1反事实；自动清理临时worker与attempt文件，不保存受试者级预测。

S1反事实共REAL、ADHD、NEUTRAL、SWAPPED、OFF；同一最佳checkpoint和同一输入，仅干预疾病分支。输出ACC、Hard/Probability-AUC、BA、SEN/SPE、预测翻转率、ASD概率变化、融合前logit变化及疾病margin变化。预测敏感性与正确疾病文本收益必须结合S2/S3完整训练对照解释。

    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --prepare
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --all
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --resume
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --refresh

已完成运行不要再次prepare；resume跳过成功任务，refresh从成功attempt或最终CSV重建报告。所有文本独立存于prompts.json，训练不依赖历史实验run.py。最终产物与主结果见results/report.md，版本、哈希、初始化检查与完成状态见manifest.json。本实验不执行git commit或git push。
""")


def prepare_worktree(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, work / name)
    (work / "CLIP").symlink_to(ROOT / "CLIP", target_is_directory=True)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir(exist_ok=True)
    (work / "graph").mkdir(exist_ok=True)
    result = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
                            cwd=work, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("disease_patch.diff did not apply cleanly:\n" + result.stdout + result.stderr)
    patched, _ = expected_patches()
    for name, body in patched.items():
        if (work / name).read_text(encoding="utf-8") != body:
            raise RuntimeError("isolated patch content mismatch: " + name)


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


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def task_log(config, seed):
    return LOG_DIR / f"{config}_seed{seed}.log"


def attempt_count(path):
    if not Path(path).exists():
        return 0
    return len(re.findall(r"^ATTEMPT_START ", Path(path).read_text(encoding="utf-8", errors="replace"), re.M))


def attempt_records(config, seed):
    path = task_log(config, seed)
    if not path.exists():
        return []
    starts = {}
    ends = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = ATTEMPT_RE.search(line)
        if start and start.group(1) == config and int(start.group(2)) == seed:
            starts[int(start.group(3))] = tuple(start.group(i) for i in (4, 5, 6))
        end = TASK_END_RE.search(line)
        if end and end.group(1) == config and int(end.group(2)) == seed:
            ends[int(end.group(3))] = {
                "returncode": int(end.group(4)), "folds": int(end.group(5)),
                "diagnostics": int(end.group(6)), "counterfactual": int(end.group(7)),
            }
    return [(attempt, starts.get(attempt), ends.get(attempt)) for attempt in sorted(set(starts) | set(ends))]


def successful_paths(config, seed):
    expected_cf = 50 if config == "S1" else 0
    for attempt, paths, end in reversed(attempt_records(config, seed)):
        if not paths or not end or end["returncode"] != 0:
            continue
        metric_path, diagnostic_path, counterfactual_path = paths
        metrics = read_jsonl(metric_path)
        diagnostics = read_jsonl(diagnostic_path)
        counterfactual = read_jsonl(counterfactual_path)
        if (end["folds"], end["diagnostics"], end["counterfactual"]) != (10, 10, expected_cf):
            continue
        if len(metrics) != 10 or len(diagnostics) != 10 or len(counterfactual) != expected_cf:
            continue
        if {int(row["fold"]) for row in metrics} != set(range(1, 11)):
            continue
        if {int(row["fold"]) for row in diagnostics} != set(range(1, 11)):
            continue
        if any(row.get("config") != config or int(row["seed"]) != seed for row in metrics + diagnostics + counterfactual):
            continue
        if any(not all(key in row for key in ("disease_logit_asd_std", "disease_logit_td_std")) for row in diagnostics):
            continue
        if config == "S1" and {(int(row["fold"]), row["condition"]) for row in counterfactual} != {
                (fold, condition) for fold in range(1, 11) for condition in ("REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF")}:
            continue
        if any("mean_abs_logit_change" not in row for row in counterfactual):
            continue
        return metric_path, diagnostic_path, counterfactual_path
    return None


def process_running(config, seed):
    try:
        lines = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except Exception:
        return False
    script = Path(__file__).resolve()
    for line in lines:
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            argv = shlex.split(fields[1])
            if len(argv) > 1 and Path(argv[1]).resolve() == script and "--one" in argv:
                if argv[argv.index("--config") + 1] == config and argv[argv.index("--seed") + 1] == str(seed):
                    return True
        except Exception:
            continue
    return False


def task_state(config, seed):
    if successful_paths(config, seed):
        return "COMPLETE"
    end = attempt_records(config, seed)
    if end and end[-1][2] is not None:
        last = end[-1][2]
        expected_cf = 50 if config == "S1" else 0
        if last["returncode"] == 0 and (last["folds"], last["diagnostics"], last["counterfactual"]) == (10, 10, expected_cf):
            fold_rows = [row for row in read_csv(FOLD_FILE) if row.get("config") == config and int(row["seed"]) == seed]
            diagnostic_rows = [row for row in read_csv(DIAG_FILE) if row.get("config") == config and int(row["seed"]) == seed]
            counterfactual_rows = [row for row in read_csv(CF_FILE) if row.get("config") == config and int(row["seed"]) == seed]
            if len(fold_rows) == 10 and len(diagnostic_rows) == 10 and len(counterfactual_rows) == expected_cf:
                return "COMPLETE"
    if process_running(config, seed):
        return "RUNNING"
    records = attempt_records(config, seed)
    if records:
        end = records[-1][2]
        if end is None:
            return "INTERRUPTED"
        if end["returncode"] != 0:
            return "FAILED"
        return "INCOMPLETE"
    return "PENDING"


def latest_rows(config, seed):
    paths = successful_paths(config, seed)
    if not paths:
        if task_state(config, seed) == "COMPLETE":
            return tuple([row for row in read_csv(path) if row.get("config") == config and int(row["seed"]) == seed]
                         for path in (FOLD_FILE, DIAG_FILE, CF_FILE))
        return [], [], []
    return tuple(read_jsonl(path) for path in paths)


def run_task(config, seed, gpu):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if task_state(config, seed) == "COMPLETE":
        print(f"skip {config} seed={seed}", flush=True)
        return 0
    log = task_log(config, seed)
    attempt = attempt_count(log) + 1
    metric_path = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_metrics.jsonl"
    diagnostic_path = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_diagnostics.jsonl"
    counterfactual_path = WORK_DIR / f"{config}_seed{seed}_attempt{attempt}_counterfactual.jsonl"
    for path in (metric_path, diagnostic_path, counterfactual_path):
        path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"SDPA-{config}-seed{seed}-") as temporary:
        work = Path(temporary)
        prepare_worktree(work)
        shell_args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        if shell_args[:2] != ["python", "main.py"]:
            raise RuntimeError("unexpected training shell command")
        command = [sys.executable, "main.py", *shell_args[2:], "--seed", str(seed), "--gpu", "0"]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "SDPA_CONFIG": config,
                    "SDPA_PROMPT_FILE": str(PROMPT_FILE), "SDPA_METRIC_PATH": str(metric_path),
                    "SDPA_DIAGNOSTIC_PATH": str(diagnostic_path)})
        if config == "S1":
            env["SDPA_COUNTERFACTUAL_PATH"] = str(counterfactual_path)
        started = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} attempt={attempt} gpu={gpu} utc={utc_now()} metrics={metric_path} diagnostics={diagnostic_path} counterfactual={counterfactual_path}\n")
            stream.write(f"COMMAND {shlex.join(command)}\n")
            stream.flush()
            process = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            metric_count = len(read_jsonl(metric_path))
            diagnostic_count = len(read_jsonl(diagnostic_path))
            counterfactual_count = len(read_jsonl(counterfactual_path))
            stream.write(f"TASK_END config={config} seed={seed} attempt={attempt} returncode={process.returncode} folds={metric_count} diagnostics={diagnostic_count} counterfactual={counterfactual_count} elapsed_seconds={time.time()-started:.1f}\n")
            stream.flush()
    print(f"{config} seed={seed} attempt={attempt} returncode={process.returncode} status={task_state(config, seed)}", flush=True)
    return process.returncode if process.returncode != 0 else (0 if successful_paths(config, seed) else 2)


def run_one(config, seed, gpu):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with task_log(config, seed).open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"skip running task {config} seed={seed}", flush=True)
            return 0
        return run_task(config, seed, gpu)


def all_rows(kind):
    rows = []
    for config, seed, _ in TASKS:
        metrics, diagnostics, counterfactual = latest_rows(config, seed)
        rows.extend({"metrics": metrics, "diagnostics": diagnostics, "counterfactual": counterfactual}[kind])
    return rows


def write_outputs():
    metrics = all_rows("metrics")
    write_csv(FOLD_FILE, ["config", "seed", "fold", *METRICS], metrics)
    diagnostics = all_rows("diagnostics")
    diagnostic_fields = [
        "config", "seed", "fold", "prototype_cosine", "asd_prototype_norm", "td_prototype_norm",
        "prototype_distance", "projected_subject_norm_mean", "normalized_subject_norm_mean",
        "alignment_loss_train", "disease_only_acc", "disease_only_probability_auc",
        "disease_logit_asd_mean", "disease_logit_asd_std", "disease_logit_td_mean", "disease_logit_td_std", "asd_margin_mean", "td_margin_mean",
        "correct_prototype_match_rate", "gamma", "gamma_abs", "gamma_sign",
        "mean_abs_fusion_logit_contribution", "std_fusion_logit_contribution",
        "p1_disease_probability_correlation", "projector_grad_valid", "alpha_grad_valid",
        "alpha_grad_exempt", "prediction_nonfinite", "nonfinite",
    ]
    write_csv(DIAG_FILE, diagnostic_fields, diagnostics)
    counterfactual = all_rows("counterfactual")
    counterfactual_fields = [
        "config", "seed", "fold", "condition", "acc", "hard_auc", "probability_auc",
        "balanced_accuracy", "sensitivity_asd", "specificity_asd", "prediction_flip_rate",
        "mean_probability_abs_change", "mean_asd_probability_abs_change", "acc_change_vs_real",
        "mean_abs_logit_change", "mean_disease_margin_change", "mean_abs_disease_margin_change", "real_output_max_abs_diff",
    ]
    write_csv(CF_FILE, counterfactual_fields, counterfactual)
    return metrics, diagnostics, counterfactual


def validate_outputs(metrics, diagnostics, counterfactual):
    expected = {(config, seed, fold) for config in CONFIGS for seed in SEEDS for fold in range(1,11)}
    expected_cf = {("S1", seed, fold, condition) for seed in SEEDS for fold in range(1,11)
                   for condition in ("REAL","ADHD","NEUTRAL","SWAPPED","OFF")}
    keys = lambda rows: {(row["config"], int(row["seed"]), int(row["fold"])) for row in rows}
    if len(metrics) != 150 or keys(metrics) != expected:
        raise RuntimeError("fold_metrics must contain exactly 150 unique config×seed×fold rows")
    if len(diagnostics) != 150 or keys(diagnostics) != expected:
        raise RuntimeError("disease_diagnostics must contain exactly 150 unique config×seed×fold rows")
    cf_keys = {(row["config"], int(row["seed"]), int(row["fold"]), row["condition"]) for row in counterfactual}
    if len(counterfactual) != 150 or cf_keys != expected_cf:
        raise RuntimeError("counterfactual must contain exactly 150 unique S1 seed×fold×condition rows")
    is_true = lambda value: str(value).lower() == "true"
    for row in metrics:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0. <= value <= 1.:
                raise RuntimeError(f"invalid metric {row['config']}/{row['seed']}/{row['fold']}/{metric}")
        if abs(float(row["hard_auc"])-float(row["balanced_accuracy"])) > 1e-10:
            raise RuntimeError("hard-label binary AUC does not match balanced accuracy")
    for row in diagnostics:
        if is_true(row["nonfinite"]) or is_true(row["prediction_nonfinite"]):
            raise RuntimeError("nonfinite disease diagnostic/prediction")
        if not is_true(row["projector_grad_valid"]):
            raise RuntimeError("disease projector lacks valid gradient")
        if row["config"] != "S4" and not is_true(row["alpha_grad_valid"]):
            raise RuntimeError("disease alpha lacks valid gradient")
        if row["config"] == "S4" and (float(row["gamma"]) != 0. or float(row["mean_abs_fusion_logit_contribution"]) != 0.):
            raise RuntimeError("S4 unexpectedly has a decision contribution")
        if abs(float(row["gamma"])) > .20000001:
            raise RuntimeError("gamma exceeds fixed bounded scale")
        for key in ("disease_logit_asd_std","disease_logit_td_std","asd_margin_mean","td_margin_mean","prototype_cosine"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError("nonfinite diagnostic " + key)
    s1 = {(int(row["seed"]),int(row["fold"])):row for row in metrics if row["config"]=="S1"}
    for row in counterfactual:
        for key in METRICS + ["prediction_flip_rate","mean_asd_probability_abs_change","mean_abs_logit_change","mean_disease_margin_change"]:
            if not math.isfinite(float(row[key])):
                raise RuntimeError("nonfinite counterfactual metric " + key)
        if row["condition"]=="REAL":
            factual=s1[(int(row["seed"]),int(row["fold"]))]
            if float(row["real_output_max_abs_diff"]) != 0.:
                raise RuntimeError("counterfactual REAL differs from normal output")
            for key in METRICS:
                if abs(float(row[key])-float(factual[key]))>1e-10:
                    raise RuntimeError("counterfactual REAL metric mismatch: "+key)
    manifest = json.loads(MANIFEST_FILE.read_text())
    for name, digest in manifest["source_files"].items():
        if sha256(ROOT/name)!=digest:
            raise RuntimeError("original source changed: "+name)
    for name, digest in manifest["inputs"].items():
        if sha256(ROOT/name)!=digest:
            raise RuntimeError("ABIDE input changed: "+name)
    if sha256(PROMPT_FILE)!=manifest["prompts_sha256"] or sha256(PATCH_FILE)!=manifest["patch_sha256"]:
        raise RuntimeError("frozen prompt/patch changed")
    original,_=p1_source()
    frozen=json.loads(PROMPT_FILE.read_text())
    if p1_sentences(original)!=p1_sentences(frozen):
        raise RuntimeError("P1 prompt mismatch")
    return {"fold_rows_unique":150,"diagnostic_rows_unique":150,"counterfactual_rows_unique":150,
            "source_and_data_unchanged":True,"p1_prompts_exact":True,"all_metrics_finite":True,
            "all_real_counterfactuals_equal":True,"all_required_gradients_valid":True,
            "all_s4_decision_contributions_zero":True}


def numbers(rows, key):
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def seed_summary(rows, config, seed):
    subset = [row for row in rows if row.get("config") == config and int(row["seed"]) == seed]
    result = {"n_folds": len(subset)}
    for metric in METRICS:
        values = numbers(subset, metric)
        result[metric + "_mean"] = statistics.mean(values) if values else None
        result[metric + "_sd"] = statistics.pstdev(values) if values else None
    return result


def aggregate(rows, config):
    summaries = [seed_summary(rows, config, seed) for seed in SEEDS]
    complete = [row for row in summaries if row["n_folds"] == 10]
    result = {}
    for metric in METRICS:
        values = [row[metric + "_mean"] for row in complete if row[metric + "_mean"] is not None]
        result[metric + "_mean"] = statistics.mean(values) if values else None
        result[metric + "_sd"] = statistics.pstdev(values) if len(values) > 1 else None
    return result


def format_metric(value, sd=None):
    if value is None:
        return "N/A"
    return f"{100*value:.2f}%" + (f" (seed SD {100*sd:.2f} pp)" if sd is not None else "")


def paired_counts(rows, left, right, metric):
    digits = 4 if metric in ("acc", "hard_auc") else 6
    first = {(int(row["seed"]), int(row["fold"])): round(float(row[metric]), digits) for row in rows
             if row.get("config") == left and row.get(metric) not in ("", None)}
    second = {(int(row["seed"]), int(row["fold"])): round(float(row[metric]), digits) for row in rows
              if row.get("config") == right and row.get(metric) not in ("", None)}
    diff = [first[key] - second[key] for key in first.keys() & second.keys()]
    epsilon = 10 ** (-digits) / 2
    return sum(value > epsilon for value in diff), sum(abs(value) <= epsilon for value in diff), sum(value < -epsilon for value in diff)


def scientific_analysis(metrics, diagnostics, counterfactual):
    combined = read_csv(REFERENCE_FILE) + metrics
    summaries = {config: aggregate(combined, config) for config in ("S0", *CONFIGS)}
    lines = ["", "## 初始原型与类别方向", "",
        "| 配置 | 原型cos | 类别0范数 | 类别1范数 | 类别间距 | ASD组margin | TD组margin | P1/疾病概率相关 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def mean(rows, key):
        values = numbers(rows, key)
        return statistics.mean(values) if values else math.nan
    for config in CONFIGS:
        subset = [row for row in diagnostics if row["config"] == config]
        lines.append(f"| {config} | {mean(subset,'prototype_cosine'):.6f} | {mean(subset,'asd_prototype_norm'):.6f} | {mean(subset,'td_prototype_norm'):.6f} | {mean(subset,'prototype_distance'):.6f} | {mean(subset,'asd_margin_mean'):+.6f} | {mean(subset,'td_margin_mean'):+.6f} | {mean(subset,'p1_disease_probability_correlation'):.6f} |")
    lines += ["", "S2类别0是ADHD错配候选，S3类别0/1是中性技术候选；诊断CSV使用统一列名记录候选位置，不能把这两组候选称作真实ASD/TD语义。原型几何不同也是对照解释的限制。", "",
        "## gamma分布与疾病分数贡献", "",
        "| 配置 | gamma均值 | SD | min | median | max | 正/零/负折数 | 绝对logit贡献 | ASD logit SD | TD logit SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        subset = [row for row in diagnostics if row["config"] == config]
        gamma = numbers(subset, "gamma")
        if not gamma: continue
        pos = sum(value > 1e-8 for value in gamma)
        neg = sum(value < -1e-8 for value in gamma)
        zero = len(gamma)-pos-neg
        lines.append(f"| {config} | {statistics.mean(gamma):+.6f} | {statistics.pstdev(gamma):.6f} | {min(gamma):+.6f} | {statistics.median(gamma):+.6f} | {max(gamma):+.6f} | {pos}/{zero}/{neg} | {mean(subset,'mean_abs_fusion_logit_contribution'):.6f} | {mean(subset,'disease_logit_asd_std'):.6f} | {mean(subset,'disease_logit_td_std'):.6f} |")
    lines += ["", "S4不使用决策融合，alpha无梯度且gamma保持0是预期结果。疾病分支本身的分类准确率与概率AUC来自训练后受试者投影，较高的分支准确率可能只说明第二个受监督分类器形成；不能据此证明文本医学知识有效。", "",
        "## 反事实logit与margin变化", "",
        "| 条件 | ASD概率绝对变化 | 融合前logit绝对变化 | margin有符号变化 | margin绝对变化 | 翻转率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for condition in ("REAL","ADHD","NEUTRAL","SWAPPED","OFF"):
        subset = [row for row in counterfactual if row["condition"] == condition]
        lines.append(f"| {condition} | {mean(subset,'mean_asd_probability_abs_change'):.6f} | {mean(subset,'mean_abs_logit_change'):.6f} | {mean(subset,'mean_disease_margin_change'):+.6f} | {mean(subset,'mean_abs_disease_margin_change'):.6f} | {mean(subset,'prediction_flip_rate'):.6f} |")
    lines += ["", "logit变化指最终log_softmax之前的P1 logprob+gamma*disease_logits；共同偏移也可能改变该量但不改变类别概率。OFF保持疾病margin计算，只令决策融合为0，所以OFF的疾病margin变化为0。", "",
        "## 科学判断与后续方向", ""]
    if all(summaries[config].get("acc_mean") is not None for config in ("S0", *CONFIGS)):
        for left, right, purpose in (("S1","S0","完整方案是否超过P1"),("S1","S2","正确疾病是否超过错误疾病"),
                                    ("S1","S3","疾病语义是否超过技术文本"),("S1","S4","直接决策融合是否提供收益"),
                                    ("S1","S5","额外类别对齐监督是否提供收益")):
            seed_diff = [seed_summary(combined,left,seed)["acc_mean"]-seed_summary(combined,right,seed)["acc_mean"] for seed in SEEDS]
            direction = "/".join("正" if diff > 1e-6 else "负" if diff < -1e-6 else "零" for diff in seed_diff)
            delta = summaries[left]["acc_mean"]-summaries[right]["acc_mean"]
            lines.append(f"- {purpose}：{left}−{right} ACC {delta*100:+.3f} pp，seed0/1/2方向为{direction}，按实际配对结果作描述性判断。")
        d0 = summaries["S1"]["acc_mean"]-summaries["S0"]["acc_mean"]
        d2 = summaries["S1"]["acc_mean"]-summaries["S2"]["acc_mean"]
        d3 = summaries["S1"]["acc_mean"]-summaries["S3"]["acc_mean"]
        stable = all(seed_summary(combined,"S1",seed)["acc_mean"]>seed_summary(combined,other,seed)["acc_mean"]+1e-6
                     for seed in SEEDS for other in ("S0","S2","S3"))
        if d0 <= 0:
            lines.append("- S1的平均ACC未超过P1，当前方案不能替换P1；即使疾病分支单独分类良好，也须检查其是否只重复原始分类信息。")
        elif stable:
            lines.append("- S1在P1、错配和中性对照上均跨seed胜出，支持在当前协议下继续研究共享疾病语义；仍需独立测试和反事实共同验证。")
        else:
            lines.append("- S1没有同时跨seed稳定超过P1、错配疾病与中性文本，当前证据不足以归因于ASD特定知识；类别监督和额外容量仍是可行解释。")
        real_diag = [row for row in diagnostics if row["config"] == "S1"]
        margin_asd, margin_td = mean(real_diag,"asd_margin_mean"), mean(real_diag,"td_margin_mean")
        positive = sum(float(row["asd_margin_mean"])>0 and float(row["td_margin_mean"])<0 for row in real_diag)
        lines.append(f"- S1真实ASD/TD平均margin为{margin_asd:+.4f}/{margin_td:+.4f}；{positive}/30折同时表现为ASD组正margin与TD组负margin。这说明学会类别方向，不自动证明医学知识贡献。")
        corr = mean(real_diag,"p1_disease_probability_correlation")
        lines.append(f"- S1疾病概率与P1概率平均相关系数{corr:.4f}；相关高且融合增益小支持信息重复的解释，不能据相关性单独认定因果。")
        nonpositive = sum(float(row["gamma"])<=0 for row in real_diag)
        nearzero = sum(float(row["mean_abs_fusion_logit_contribution"])<1e-6 for row in real_diag)
        lines.append(f"- S1有{nonpositive}/30折gamma不为正，{nearzero}/30折平均绝对融合响应小于1e-6；按实测符号和响应描述分支使用情况。")
    lines += ["- 保留本次严格对照结果，不自动追加Prompt、损失或参数搜索。若继续研究，应先在独立验证协议下确认疾病分支互补性，再使用真实受试者脑连接构造可验证的个体化语义；没有认知量表时不伪造个体化临床文本。",
              "- 当前四个局部Token和共享疾病分支均不能定位具体脑区异常；模型预测敏感性、疾病类别监督和医学语义正确性是不同证据。"]
    return "\n".join(lines)


def run_timing():
    path = LOG_DIR / "scheduler.log"
    log = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    starts = re.findall(r"SCHEDULER_START utc=(\S+)", log)
    ends = re.findall(r"SCHEDULER_END utc=(\S+)", log)
    def seconds(start, end):
        return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    end = ends[-1] if ends else utc_now()
    return {"first_training_start_utc": starts[0] if starts else None,
            "successful_run_start_utc": starts[-1] if starts else None,
            "training_end_utc": end,
            "successful_run_seconds": seconds(starts[-1], end) if starts else 0.,
            "including_interrupted_run_seconds": seconds(starts[0], end) if starts else 0.}


def report_text(metrics, diagnostics, counterfactual):
    reference = read_csv(REFERENCE_FILE)
    combined = [{"config": "S0", **row} for row in reference] + metrics
    completed = sum(task_state(config, seed) == "COMPLETE" for config, seed, _ in TASKS)
    lines = [
        "# 实验八（方案一）：SDPA 共享疾病原型对齐", "",
        f"状态：**{'全部完成' if completed == 15 else '未全部完成'}**；S1-S5 任务 {completed}/15；训练折 {len(metrics)}/150；机制诊断 {len(diagnostics)}/150；反事实 {len(counterfactual)}/150。",
        "", "## 主结果", "", "| 配置 | ACC | Hard-AUC | Probability-AUC | BA | SEN-ASD | SPE-ASD |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in ("S0", *CONFIGS):
        summary = aggregate(combined, config)
        lines.append(f"| {config} | {format_metric(summary.get('acc_mean'), summary.get('acc_sd'))} | {format_metric(summary.get('hard_auc_mean'), summary.get('hard_auc_sd'))} | {format_metric(summary.get('probability_auc_mean'), summary.get('probability_auc_sd'))} | {format_metric(summary.get('balanced_accuracy_mean'), summary.get('balanced_accuracy_sd'))} | {format_metric(summary.get('sensitivity_asd_mean'))} | {format_metric(summary.get('specificity_asd_mean'))} |")
    lines += ["", "S0 来自实验五 E0 逐折结果；其 Probability-AUC 未保存，标为 N/A。每个 seed 先计算十折均值，再以三个 seed 均值及 seed 间标准差汇总。SPE-ASD 为以 ASD 为阳性类的特异度，即 TD 样本上的正确率。", "", "## 每个 seed 的十折均值", "", "| 配置 | seed | ACC | Hard-AUC | Probability-AUC | BA | SEN-ASD | SPE-ASD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for config in ("S0", *CONFIGS):
        for seed in SEEDS:
            summary = seed_summary(combined, config, seed)
            lines.append(f"| {config} | {seed} | {format_metric(summary.get('acc_mean'), summary.get('acc_sd'))} | {format_metric(summary.get('hard_auc_mean'), summary.get('hard_auc_sd'))} | {format_metric(summary.get('probability_auc_mean'), summary.get('probability_auc_sd'))} | {format_metric(summary.get('balanced_accuracy_mean'), summary.get('balanced_accuracy_sd'))} | {format_metric(summary.get('sensitivity_asd_mean'))} | {format_metric(summary.get('specificity_asd_mean'))} |")
    lines += ["", "## 配对比较", "", "ACC/Hard-AUC逐折胜负统一按历史四位小数精度比较；其他指标使用六位小数。三个seed共享相同十折划分，只进行描述性配对统计，不将30个结果当成30个独立样本做显著性推断。"]
    for left, right in (("S1", "S0"), ("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S1", "S5"), ("S2", "S0"), ("S3", "S0"), ("S4", "S0"), ("S5", "S0")):
        lines.append(f"### {left} − {right}")
        for metric in METRICS:
            wins, ties, losses = paired_counts(combined, left, right, metric)
            agg_left, agg_right = aggregate(combined, left), aggregate(combined, right)
            overall = "N/A" if agg_left.get(metric+'_mean') is None or agg_right.get(metric+'_mean') is None else f"{(agg_left[metric+'_mean']-agg_right[metric+'_mean'])*100:+.3f} pp"
            per_seed = []
            for seed in SEEDS:
                a, b = seed_summary(combined, left, seed), seed_summary(combined, right, seed)
                if a["n_folds"] == b["n_folds"] == 10 and a[metric+'_mean'] is not None and b[metric+'_mean'] is not None:
                    per_seed.append(f"seed{seed} {(a[metric+'_mean']-b[metric+'_mean'])*100:+.2f} pp")
            lines.append(f"- {metric}: 总体 {overall}；{', '.join(per_seed) or 'N/A'}；逐折胜/平/负 {wins}/{ties}/{losses}。")
    lines += ["", "## 疾病语义机制诊断", "", "每个配置×seed×fold 一行；潜在模态表示不解释为具体 ROI。", "", "| 配置 | 原型cos | 原型距离 | 投影范数 | 对齐损失 | 疾病分支ACC | 疾病分支AUC | gamma | logit贡献 | projector梯度 | alpha梯度 | nonfinite |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        subset = [row for row in diagnostics if row.get("config") == config]
        def mean(key):
            values = numbers(subset, key)
            return statistics.mean(values) if values else math.nan
        projector_valid = sum(str(row.get("projector_grad_valid")).lower() == "true" for row in subset)
        alpha_valid = sum(str(row.get("alpha_grad_valid")).lower() == "true" for row in subset)
        nonfinite = sum(str(row.get("nonfinite")).lower() == "true" for row in subset)
        lines.append(f"| {config} | {mean('prototype_cosine'):.4f} | {mean('prototype_distance'):.4f} | {mean('projected_subject_norm_mean'):.4f} | {mean('alignment_loss_train'):.4f} | {mean('disease_only_acc')*100:.2f}% | {mean('disease_only_probability_auc')*100:.2f}% | {mean('gamma'):.5f} | {mean('mean_abs_fusion_logit_contribution'):.6f} | {projector_valid}/{len(subset)} | {alpha_valid}/{len(subset)} | {nonfinite} |")
    lines += ["", "## S1 反事实文本干预", "", "条件共用同一最佳 checkpoint、受试者输入和原始 P1 路径；只改变辅助疾病文本或关闭疾病融合。", "", "| 条件 | ACC | Hard-AUC | BA | SEN-ASD | SPE-ASD | 翻转率 | 平均概率变化 | ACC变化 | 概率AUC | REAL最大差 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in ("REAL", "ADHD", "NEUTRAL", "SWAPPED", "OFF"):
        subset = [row for row in counterfactual if row.get("condition") == condition]
        def mean_cf(key):
            values = numbers(subset, key)
            return statistics.mean(values) if values else math.nan
        lines.append(f"| {condition} | {mean_cf('acc')*100:.2f}% | {mean_cf('hard_auc')*100:.2f}% | {mean_cf('balanced_accuracy')*100:.2f}% | {mean_cf('sensitivity_asd')*100:.2f}% | {mean_cf('specificity_asd')*100:.2f}% | {mean_cf('prediction_flip_rate'):.4f} | {mean_cf('mean_probability_abs_change'):.6f} | {mean_cf('acc_change_vs_real')*100:+.2f} pp | {mean_cf('probability_auc')*100:.2f}% | {mean_cf('real_output_max_abs_diff') if condition == 'REAL' else '-'} |")
    baseline = aggregate(combined, "S0")
    lines += ["", "## 结果解读", ""]
    for config in CONFIGS:
        current = aggregate(combined, config)
        if current.get("acc_mean") is None:
            lines.append(f"- {config} 相对 S0：训练结果未完整。")
        else:
            lines.append(f"- {config} 相对 S0：ACC {(current['acc_mean']-baseline['acc_mean'])*100:+.2f} pp；Hard-AUC {(current['hard_auc_mean']-baseline['hard_auc_mean'])*100:+.2f} pp。")
    if diagnostics:
        mean_contribution = statistics.mean(float(row["mean_abs_fusion_logit_contribution"]) for row in diagnostics)
        lines.append(f"- 疾病分支平均融合 logit 绝对贡献 {mean_contribution:.6f}；这描述模型响应，不代表医学因果。")
    for condition in ("ADHD", "NEUTRAL", "SWAPPED", "OFF"):
        subset = numbers([row for row in counterfactual if row["condition"] == condition], "prediction_flip_rate")
        if subset:
            lines.append(f"- S1 {condition} 条件的平均翻转率 {statistics.mean(subset):.4f}；预测敏感不能单独证明疾病语义正确。")
    lines += ["", "## 运行与审计", "", "GPU0（8个并行任务）：S1 seed0、S1 seed2、S2 seed1、S3 seed0、S3 seed2、S4 seed1、S5 seed0、S5 seed2。", "GPU1（7个并行任务）：S1 seed1、S2 seed0、S2 seed2、S3 seed1、S4 seed0、S4 seed2、S5 seed1。", f"目录：{EXP}。实验根源码未改动；无 git commit/push。"]
    lines.append(scientific_analysis(metrics, diagnostics, counterfactual))
    timing = run_timing()
    lines += ["", "## 运行时间与版本核验", "", f"有效完整重跑从 {timing['successful_run_start_utc']} 到 {timing['training_end_utc']}，耗时约 {timing['successful_run_seconds']/3600:.2f} 小时；含前次因诊断字段遗漏中断的训练，从 {timing['first_training_start_utc']} 起约 {timing['including_interrupted_run_seconds']/3600:.2f} 小时。", "物理CUDA进程映射已核对：GPU0实际8进程，GPU1实际7进程；不是仅依据环境变量声明。中断attempt没有进入本次CSV。"]
    return "\n".join(lines) + "\n"


def update_manifest(status=None):
    if not MANIFEST_FILE.exists():
        return
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if status is not None:
        manifest["run_status"] = status
    manifest["task_status"] = {f"{config}_seed{seed}": task_state(config, seed) for config, seed, _ in TASKS}
    manifest["completed"] = {"fold_metrics": len(all_rows("metrics")),
                             "disease_diagnostics": len(all_rows("diagnostics")),
                             "counterfactual": len(all_rows("counterfactual"))}
    manifest["progress"] = {
        "current_attempt_fold_rows": len(all_rows("metrics")),
        "current_attempt_diagnostics": len(all_rows("diagnostics")),
        "current_attempt_counterfactuals": len(all_rows("counterfactual")),
        "checked_utc": utc_now(),
    }
    manifest["runner_sha256"] = sha256(EXP / "run.py")
    if status == "COMPLETE":
        manifest.setdefault("finished_utc", utc_now())
        manifest["timing"] = run_timing()
        manifest["successful_attempts"] = {
            f"{config}_seed{seed}": max(attempt for attempt, paths, end in attempt_records(config, seed)
                                        if end and end["returncode"] == 0 and end["folds"] == 10)
            for config, seed, gpu in TASKS}
    write_json(MANIFEST_FILE, manifest)


def update_handoff():
    start, end = "<!-- SDPA EXPERIMENT 8 START -->", "<!-- SDPA EXPERIMENT 8 END -->"
    previous = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
    metrics, diagnostics, counterfactual = read_csv(FOLD_FILE), read_csv(DIAG_FILE), read_csv(CF_FILE)
    complete = len(metrics) == len(diagnostics) == len(counterfactual) == 150
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.exists() else {}
    status = "完成15/15任务、150折、150条诊断和150条反事实" if complete else "最小校验通过，正式训练进行中"
    lines = [start, "", "## 实验八（方案一）：SDPA 共享疾病原型对齐", "",
        f"状态：{status}。目录与完整中文报告：{EXP}/results/report.md。",
        "动机：P1仍是已有基线；普通文本改写、ASD相关FC措辞、MLP去ReLU/去衰减、方向标量门控和双主题特征残差均没有稳定额外收益。实验七F1约91.69%/91.49%低于P1，支持改为检验保留类别方向的共享疾病语义。",
        "设计：S0冻结E0/P1；S1正确ASD/TD对齐与决策融合；S2 ADHD/TD错配；S3中性技术两类；S4仅对齐；S5仅融合。新增Linear(36,1024,bias=False)+alpha共36,865参数，gamma=.2*tanh(alpha)，对齐lambda=.1，仅训练索引有监督；P1原路径与其32条文本保持原样。",
        "论文依据：BrainPrompt疾病类别交叉熵、SABER受控决策语义注入、BrainSCK知识注入/激活区分。核查使用官方公开论文，不假装读取项目里不存在的上传PDF；没有构造不存在的个体认知量表。",
        f"代码版本：GitHub main {manifest.get('github_main_commit_at_prepare','待记录')}；具体源码/CLIP/Prompt/Patch哈希和初始化验证见manifest.json。",
    ]
    for path, digest in manifest.get("inputs",{}).items():
        lines.append(f"数据SHA256：{path} = {digest}。")
    if complete:
        combined = read_csv(REFERENCE_FILE) + metrics
        for config in ("S0",*CONFIGS):
            item = aggregate(combined,config)
            lines.append(f"{config}：ACC {format_metric(item['acc_mean'],item['acc_sd'])}，Hard-AUC {format_metric(item['hard_auc_mean'],item['hard_auc_sd'])}，Probability-AUC {format_metric(item['probability_auc_mean'])}，BA {format_metric(item['balanced_accuracy_mean'])}。")
        real_diag = [row for row in diagnostics if row["config"]=="S1"]
        gamma = numbers(real_diag,"gamma")
        corr = numbers(real_diag,"p1_disease_probability_correlation")
        lines.append(f"S1机制：ASD/TD原型cos=.946976；gamma均值{statistics.mean(gamma):+.6f}，范围{min(gamma):+.6f}到{max(gamma):+.6f}；负gamma折数{sum(x<0 for x in gamma)}；P1/疾病概率相关均值{statistics.mean(corr):.4f}。")
        for condition in ("ADHD","NEUTRAL","SWAPPED","OFF"):
            subset = [row for row in counterfactual if row["condition"]==condition]
            lines.append(f"S1反事实{condition}：翻转率{statistics.mean(numbers(subset,'prediction_flip_rate'))*100:.3f}%，ASD概率绝对变化{statistics.mean(numbers(subset,'mean_asd_probability_abs_change')):.6f}，ACC变化{statistics.mean(numbers(subset,'acc_change_vs_real'))*100:+.3f}pp。")
        lines.append(scientific_analysis(metrics,diagnostics,counterfactual).split("## 科学判断与后续方向",1)[-1].strip())
    lines += ["", "下一步：依真实S1/S2/S3/S4/S5对照判断知识、辅助监督和容量贡献；保持当前P1结果，不自动调Prompt或超参数。若进一步研究，优先独立验证及真实脑连接驱动的个体化语义，不能伪造临床输入。",
        "根目录源码及实验一至七未修改；临时权重仅训练期间保留，完成后清理；未执行git commit或git push。", "", end]
    block = "\n".join(lines)
    pattern = re.compile(re.escape(start)+r".*?"+re.escape(end),re.S)
    updated = pattern.sub(lambda match:block,previous) if pattern.search(previous) else previous.rstrip()+"\n\n"+block+"\n"
    atomic_write(HANDOFF,updated)


def prepare():
    if MANIFEST_FILE.exists() and json.loads(MANIFEST_FILE.read_text()).get("run_status") == "COMPLETE":
        print("Completed experiment is frozen; --prepare skipped. Use --refresh to update its report.", flush=True)
        return
    prompts, p1_path = build_prompts()
    reference_source = freeze_reference()
    patch_text = validate_patch_and_tokenization(prompts)
    atomic_write(PATCH_FILE, patch_text)
    write_readme()
    write_json(MANIFEST_FILE, make_manifest(reference_source, p1_path))
    with tempfile.TemporaryDirectory(prefix="SDPA-preflight-") as temporary:
        work = Path(temporary)
        prepare_worktree(work)
        validation = preflight_runtime(work)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest["validation"] = validation
    manifest["validated_utc"] = utc_now()
    manifest["runner_sha256"] = sha256(EXP / "run.py")
    write_json(MANIFEST_FILE, manifest)
    update_handoff()
    print("SDPA prepare complete: P1=32, frozen S0=30 folds, +36865 parameters; runtime preflight passed", flush=True)


def run_all():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    scheduler = LOG_DIR / "scheduler.log"
    pending = [task for task in TASKS if task_state(task[0], task[1]) != "COMPLETE"]
    if not pending:
        refresh()
        print("All 15 tasks already complete; skipped training", flush=True)
        return
    active = {}
    retries = {}
    with scheduler.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An SDPA scheduler is already running; no duplicate tasks were started")
        stream.write(f"SCHEDULER_START utc={utc_now()} assignments={GPU_ASSIGNMENTS}\n")
        try:
            gpu_status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT).strip()
            stream.write("GPU_STATUS " + gpu_status.replace("\n", " | ") + "\n")
        except Exception as exc:
            stream.write(f"GPU_STATUS unavailable={exc}\n")
        stream.flush()
        while pending or active:
            pending = [task for task in pending if task_state(task[0], task[1]) != "COMPLETE"]
            for gpu in (0, 1):
                while sum(task[2] == gpu for task in active.values()) < MAX_ACTIVE[gpu]:
                    task = next((candidate for candidate in pending if candidate[2] == gpu and not process_running(candidate[0], candidate[1])), None)
                    if task is None:
                        break
                    pending.remove(task)
                    config, seed, assigned = task
                    command = [sys.executable, str(Path(__file__).resolve()), "--one", "--config", config, "--seed", str(seed), "--gpu", str(assigned)]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=stream)
                    active[process] = task
                    stream.write(f"SCHEDULER_START_TASK {config} seed={seed} gpu={assigned}\n")
                    stream.flush()
            finished = []
            for process, task in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                config, seed, gpu = task
                stream.write(f"SCHEDULER_FINISH_TASK {config} seed={seed} gpu={gpu} returncode={code}\n")
                finished.append(process)
                if code != 0 and retries.get((config, seed), 0) < 1:
                    retries[(config, seed)] = retries.get((config, seed), 0) + 1
                    pending.append(task)
                    stream.write(f"SCHEDULER_REQUEUE {config} seed={seed} retry=1\n")
                stream.flush()
            for process in finished:
                del active[process]
            if active or pending:
                time.sleep(5)
        metrics, diagnostics, counterfactual = write_outputs()
        complete = len(metrics) == 150 and len(diagnostics) == 150 and len(counterfactual) == 150 and all(task_state(c, s) == "COMPLETE" for c, s, _ in TASKS)
        if complete:
            checks = validate_outputs(metrics, diagnostics, counterfactual)
            manifest = json.loads(MANIFEST_FILE.read_text())
            manifest["final_validation"] = checks
            write_json(MANIFEST_FILE, manifest)
        update_manifest("COMPLETE" if complete else "INCOMPLETE")
        atomic_write(REPORT_FILE, report_text(metrics, diagnostics, counterfactual))
        update_handoff()
        stream.write(f"SCHEDULER_END utc={utc_now()} complete={complete} metrics={len(metrics)} diagnostics={len(diagnostics)} counterfactual={len(counterfactual)}\n")
        stream.flush()
    if not complete:
        raise RuntimeError(f"SDPA incomplete: {len(metrics)} metrics, {len(diagnostics)} diagnostics, {len(counterfactual)} counterfactual")
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print("SDPA COMPLETE: 150 folds, 150 disease diagnostics, 150 counterfactual rows", flush=True)


def refresh():
    if len(read_csv(FOLD_FILE)) == 150 and len(read_csv(DIAG_FILE)) == 150 and len(read_csv(CF_FILE)) == 150 and not WORK_DIR.exists():
        metrics, diagnostics, counterfactual = read_csv(FOLD_FILE), read_csv(DIAG_FILE), read_csv(CF_FILE)
        checks = validate_outputs(metrics, diagnostics, counterfactual)
        manifest = json.loads(MANIFEST_FILE.read_text())
        manifest["final_validation"] = checks
        write_json(MANIFEST_FILE, manifest)
        update_manifest("COMPLETE")
        atomic_write(REPORT_FILE, report_text(metrics, diagnostics, counterfactual))
        update_handoff()
        print("SDPA report refreshed from completed CSV outputs", flush=True)
        return
    metrics, diagnostics, counterfactual = write_outputs()
    complete = len(metrics) == 150 and len(diagnostics) == 150 and len(counterfactual) == 150
    if complete:
        checks = validate_outputs(metrics, diagnostics, counterfactual)
        manifest = json.loads(MANIFEST_FILE.read_text())
        manifest["final_validation"] = checks
        write_json(MANIFEST_FILE, manifest)
    atomic_write(REPORT_FILE, report_text(metrics, diagnostics, counterfactual))
    update_manifest("COMPLETE" if complete else "INCOMPLETE")
    update_handoff()
    if complete:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    print(f"SDPA refreshed: {len(metrics)} folds, {len(diagnostics)} diagnostics, {len(counterfactual)} counterfactual", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.all or args.resume:
        run_all()
    elif args.refresh:
        refresh()
    elif args.one:
        if args.config is None or args.seed is None or args.gpu is None:
            parser.error("--one requires --config, --seed and --gpu")
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    else:
        parser.error("select --prepare, --all, --resume or --refresh")


if __name__ == "__main__":
    main()

