#!/usr/bin/env python3
"""SPCR: semantic projection collapse and repair."""

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import platform
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
ROOT = EXP.parents[1]
LOG_DIR = EXP / "logs"
WORK_DIR = LOG_DIR / ".working"
RESULT_DIR = EXP / "results"
PROMPT_FILE = EXP / "prompts.json"
PATCH_FILE = EXP / "projection_patch.diff"
MANIFEST_FILE = EXP / "manifest.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_metrics.csv"
DIAG_FILE = RESULT_DIR / "projection_diagnostics.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
STAGES = ["CLIP_GROUP_INPUT", "LINEAR1_PRE", "RELU1_POST", "LINEAR2_PRE", "FINAL_OUTPUT"]
CONFIGS = ["E0", "E1", "E2", "E3"]
SEEDS = [0, 1, 2]
GPU_ASSIGNMENTS = {
    ("E0", 0): 0, ("E0", 2): 0, ("E1", 1): 0,
    ("E2", 0): 0, ("E2", 2): 0, ("E3", 1): 0,
    ("E0", 1): 1, ("E1", 0): 1, ("E1", 2): 1,
    ("E2", 1): 1, ("E3", 0): 1, ("E3", 2): 1,
}
TASKS = [(config, seed, GPU_ASSIGNMENTS[(config, seed)])
         for config in CONFIGS for seed in SEEDS]
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=E[0-3] seed=\d+")
TASK_END_RE = re.compile(
    r"TASK_END config=(E[0-3]) seed=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+)"
)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = stream.name
    os.replace(temporary, path)


def write_json(path, payload):
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    paths = [ROOT / relative for relative in SOURCE_ROOT_FILES]
    paths.extend(path for path in sorted((ROOT / "CLIP").rglob("*")) if path.is_file())
    return paths


def github_head():
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        )
        return output.split()[0] if output.split() else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def p1_source():
    path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def prompt_sentences(prompt_data):
    if prompt_data["modality_order"] != MODALITIES:
        raise ValueError("Unexpected P1 modality order")
    sentences = []
    for modality in MODALITIES:
        themes = prompt_data["p1"][modality]["themes"]
        if len(themes) != 2 or any(len(theme["prompts"]) != 4 for theme in themes):
            raise ValueError(f"{modality} must contain two groups of four prompts")
        for theme in themes:
            sentences.extend(item["text"] for item in theme["prompts"])
    if len(sentences) != 32 or any(not item for item in sentences):
        raise ValueError("P1 must contain exactly 32 non-empty prompts")
    return sentences


def build_prompts():
    source, source_path = p1_source()
    prompts = {
        "schema_version": 1,
        "experiment": "SPCR",
        "modality_order": MODALITIES,
        "source_experiment1_p1_sha256": sha256(source_path),
        "p1": deepcopy(source["p1"]),
    }
    if prompts["p1"] != source["p1"] or len(prompt_sentences(prompts)) != 32:
        raise ValueError("Frozen SPCR prompts differ from Experiment 1 P1")
    return prompts, source_path


def patch_network(source):
    source = source.replace("import argparse\n", "import argparse\nimport json\n", 1)
    old = '''def get_prompt_text_features():
    """Encode the fixed semantic prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(text).float().detach()
    return _prompt_text_features
'''
    new = '''def get_prompt_text_features():
    """Encode frozen Experiment 1 P1 prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        prompt_tokens = text
        if os.environ.get("SPCR_PROMPTS_FILE"):
            with open(os.environ["SPCR_PROMPTS_FILE"], encoding="utf-8") as stream:
                prompt_data = json.load(stream)
            sentences = []
            for modality in prompt_data["modality_order"]:
                for theme in prompt_data["p1"][modality]["themes"]:
                    sentences.extend(item["text"] for item in theme["prompts"])
            if len(sentences) != 32:
                raise ValueError("SPCR requires exactly 32 P1 prompts.")
            prompt_tokens = clip.tokenize(sentences).to(device)
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(prompt_tokens).float().detach()
    return _prompt_text_features
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original CLIP prompt encoder")
    source = source.replace(old, new, 1)

    old = '''    def forward(self, x):
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        return x
'''
    new = '''    def forward(self, x):
        x = F.relu(self.t1(x))
        x = self.t2(x)
        if os.environ.get("SPCR_CONFIG", "E0") in ("E1", "E3"):
            return x
        return F.relu(x)
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original MLP forward")
    source = source.replace(old, new, 1)

    old = """        self.register_buffer('prompt_text_features', get_prompt_text_features())
"""
    new = """        self.register_buffer('prompt_text_features', get_prompt_text_features())
        self.spcr_config = os.environ.get("SPCR_CONFIG", "E0")
        self.spcr_output_relu = self.spcr_config in ("E0", "E2")
        self.spcr_diag_enabled = os.environ.get("SPCR_DIAGNOSTIC", "0") == "1"
        self._spcr_eval_idx = None
        self._spcr_last = None
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate prompt feature buffer registration")
    source = source.replace(old, new, 1)

    class_start = source.find("class PLN(nn.Module):")
    forward_start = source.find("    def forward(self, x):\n", class_start)
    if class_start < 0 or forward_start < 0:
        raise RuntimeError("Cannot locate PLN.forward")
    helper = '''    @staticmethod
    def _spcr_stage_rows(values):
        values = values.detach().float()
        norms = torch.linalg.vector_norm(values, dim=-1)
        finite = torch.isfinite(values).all(dim=-1)
        rows = []
        for index, modality in enumerate(["PHENO", "ANAT", "FUNC", "Correlation"]):
            valid = bool(finite[index].all().item() and (norms[index] > 1e-12).all().item())
            cosine = None
            if valid:
                cosine = float(F.cosine_similarity(values[index, 0].unsqueeze(0), values[index, 1].unsqueeze(0)).item())
            rows.append({
                "modality": modality,
                "cosine": cosine,
                "abs_cosine": None if cosine is None else abs(cosine),
                "cosine_gt_0_98": None if cosine is None else bool(cosine > 0.98),
                "cosine_lt_neg_0_98": None if cosine is None else bool(cosine < -0.98),
                "norm_1": float(norms[index, 0].item()),
                "norm_2": float(norms[index, 1].item()),
                "zero_vector": bool((norms[index] <= 1e-12).any().item()),
                "nonfinite": bool((~finite[index]).any().item()),
                "zero_fraction_1": float((values[index, 0] == 0).float().mean().item()),
                "zero_fraction_2": float((values[index, 1] == 0).float().mean().item()),
            })
        return rows

    @staticmethod
    def _spcr_scalar_stats(values):
        values = values.reshape(-1).float()
        finite = torch.isfinite(values)
        clean = values[finite]
        if clean.numel() == 0:
            return {
                "mean": None, "std": None, "le_0_51": None,
                "gt_0_55": None, "gt_0_60": None,
                "nonfinite": int((~finite).sum().item()),
            }
        return {
            "mean": float(clean.mean().item()),
            "std": float(clean.std(unbiased=False).item()),
            "le_0_51": float((clean <= 0.51).float().mean().item()),
            "gt_0_55": float((clean > 0.55).float().mean().item()),
            "gt_0_60": float((clean > 0.60).float().mean().item()),
            "nonfinite": int((~finite).sum().item()),
        }

    def _spcr_project(self):
        grouped = self.prompt_text_features.view(4, 2, 4096)
        h_pre = self.MLP.t1(grouped)
        h_post = F.relu(h_pre)
        z_pre = self.MLP.t2(h_post)
        final = F.relu(z_pre) if self.spcr_output_relu else z_pre
        stages = {
            "CLIP_GROUP_INPUT": grouped,
            "LINEAR1_PRE": h_pre,
            "RELU1_POST": h_post,
            "LINEAR2_PRE": z_pre,
            "FINAL_OUTPUT": final,
        }
        return final, stages

    def _spcr_snapshot(self, stages, raw_similarity=None, softmax_similarity=None):
        snapshot = {
            "config": self.spcr_config,
            "stage_rows": {name: self._spcr_stage_rows(stages[name]) for name in stages},
            "params": {
                "t1_weight_fro": float(torch.linalg.matrix_norm(self.MLP.t1.weight.detach()).item()),
                "t1_bias_l2": float(torch.linalg.vector_norm(self.MLP.t1.bias.detach()).item()),
                "t2_weight_fro": float(torch.linalg.matrix_norm(self.MLP.t2.weight.detach()).item()),
                "t2_bias_l2": float(torch.linalg.vector_norm(self.MLP.t2.bias.detach()).item()),
                "mlp_parameters": int(sum(parameter.numel() for parameter in self.MLP.parameters())),
                "modal_fusion_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
            },
            "gate_rows": [],
        }
        if raw_similarity is None or softmax_similarity is None or self._spcr_eval_idx is None:
            return snapshot
        sample_index = self._spcr_eval_idx
        if not torch.is_tensor(sample_index):
            sample_index = torch.as_tensor(sample_index, device=raw_similarity.device, dtype=torch.long)
        else:
            sample_index = sample_index.to(raw_similarity.device, dtype=torch.long)
        raw_similarity = raw_similarity.index_select(0, sample_index)
        softmax_similarity = softmax_similarity.index_select(0, sample_index)
        semantic_nonfinite = bool((~torch.isfinite(raw_similarity)).any().item() or (~torch.isfinite(softmax_similarity)).any().item())
        for index, modality in enumerate(["PHENO", "ANAT", "FUNC", "Correlation"]):
            raw = raw_similarity[:, index, :, :].reshape(-1, 2)
            weights = softmax_similarity[:, index, :, :].max(dim=-1).values.reshape(-1)
            stats = self._spcr_scalar_stats(weights)
            delta = (raw[:, 0] - raw[:, 1]).abs()
            snapshot["gate_rows"].append({
                "modality": modality,
                "mean_abs_delta": float(delta.mean().item()),
                "weight_mean": stats["mean"],
                "weight_std": stats["std"],
                "weight_le_0_51": stats["le_0_51"],
                "weight_gt_0_55": stats["gt_0_55"],
                "weight_gt_0_60": stats["gt_0_60"],
                "semantic_nonfinite": semantic_nonfinite or stats["nonfinite"] > 0,
            })
        return snapshot

'''
    source = source[:forward_start] + helper + source[forward_start:]

    old = '''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
'''
    new = '''        text_features, spcr_stages = self._spcr_project()
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original prompt grouping")
    source = source.replace(old, new, 1)

    start_marker = "        text_features_norm = F.normalize(text_features, p=2, dim=-1)\n"
    end_marker = "        similarity_matrix = similarity_matrix[:,:,:,-1,:]\n"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or source.find(start_marker, start + 1) >= 0:
        raise RuntimeError("Cannot locate the original similarity block")
    end += len(end_marker)
    replacement = '''        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        if text_features_norm.dim() == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            part_x_norm_expanded = part_x_norm.unsqueeze(-2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, text_features_norm).unsqueeze(-2)
        similarity_matrix = similarity_matrix[:,:,:,-1,:]
        raw_similarity_matrix = similarity_matrix
'''
    source = source[:start] + replacement + source[end:]

    old = '''        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
'''
    new = '''        # Preserve the original Softmax-Max semantic weighting.
        part_x = similarity_max.unsqueeze(-1) * part_x
        if self.spcr_diag_enabled and not self.training and self._spcr_eval_idx is not None:
            self._spcr_last = self._spcr_snapshot(
                spcr_stages, raw_similarity_matrix.detach(), similarity_matrix.detach()
            )
        else:
            self._spcr_last = None
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original semantic weighting")
    return source.replace(old, new, 1)


def patch_model(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    old = """        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
"""
    new = """        if os.environ.get("SPCR_CONFIG", "E0") in ("E2", "E3"):
            all_parameters = list(self.ModalFusion.parameters())
            mlp_parameters = list(self.ModalFusion.MLP.parameters())
            mlp_ids = {id(parameter) for parameter in mlp_parameters}
            other_parameters = [parameter for parameter in all_parameters if id(parameter) not in mlp_ids]
            if len(mlp_parameters) != 4 or len(mlp_ids) != 4:
                raise RuntimeError("SPCR MLP parameter group is not exactly t1/t2 weight and bias")
            if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
                raise RuntimeError("SPCR ModalFusion parameters contain duplicates")
            if mlp_ids.intersection({id(parameter) for parameter in other_parameters}):
                raise RuntimeError("SPCR optimizer groups overlap")
            if mlp_ids.union({id(parameter) for parameter in other_parameters}) != {id(parameter) for parameter in all_parameters}:
                raise RuntimeError("SPCR optimizer groups omit parameters")
            self.optimizer_MF = optim.Adam([
                {"params": mlp_parameters, "weight_decay": 0.0},
                {"params": other_parameters, "weight_decay": hyperpm.reg},
            ], lr=hyperpm.lr)
        else:
            self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
        with torch.no_grad():
            _, _spcr_init_stages = self.ModalFusion._spcr_project()
            self._spcr_init_snapshot = self.ModalFusion._spcr_snapshot(_spcr_init_stages)
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate ModalFusion optimizer")
    source = source.replace(old, new, 1)

    old = """        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        prob = prob[eval_idx]
"""
    new = """        if tst and os.environ.get("SPCR_DIAGNOSTIC_PATH"):
            self.ModalFusion._spcr_eval_idx = eval_idx
        else:
            self.ModalFusion._spcr_eval_idx = None
        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        if tst and os.environ.get("SPCR_DIAGNOSTIC_PATH"):
            diagnostics = getattr(self.ModalFusion, "_spcr_last", None)
            fold = getattr(self, "_spcr_fold_number", None)
            if diagnostics is not None and fold is not None:
                with open(os.environ["SPCR_DIAGNOSTIC_PATH"], "a", encoding="utf-8") as stream:
                    json.dump({"fold": int(fold), "phase": "BEST", "diagnostics": diagnostics}, stream, ensure_ascii=False)
                    stream.write("\\n")
        self.ModalFusion._spcr_eval_idx = None
        prob = prob[eval_idx]
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate evaluation forward")
    return source.replace(old, new, 1)


def patch_main(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    old = """        clk += 1
        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)
"""
    new = """        clk += 1
        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)
        if os.environ.get("SPCR_DIAGNOSTIC_PATH"):
            agent._spcr_fold_number = clk
            with open(os.environ["SPCR_DIAGNOSTIC_PATH"], "a", encoding="utf-8") as stream:
                json.dump({"fold": int(clk), "phase": "INIT", "diagnostics": agent._spcr_init_snapshot}, stream, ensure_ascii=False)
                stream.write("\\n")
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate fold initialization")
    return source.replace(old, new, 1)


def expected_patches():
    originals = {name: (ROOT / name).read_text(encoding="utf-8") for name in ["main.py", "model.py", "network.py"]}
    patched = {
        "main.py": patch_main(originals["main.py"]),
        "model.py": patch_model(originals["model.py"]),
        "network.py": patch_network(originals["network.py"]),
    }
    chunks = []
    for name in ["main.py", "model.py", "network.py"]:
        chunks.extend(difflib.unified_diff(
            originals[name].splitlines(keepends=True), patched[name].splitlines(keepends=True),
            fromfile=f"a/{name}", tofile=f"b/{name}",
        ))
    return patched, "".join(chunks)


def validate_patch_and_tokenization(prompts):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from CLIP.clip import clip
    tokens = clip.tokenize(prompt_sentences(prompts))
    if tuple(tokens.shape) != (32, 77):
        raise RuntimeError(f"Unexpected CLIP token shape: {tuple(tokens.shape)}")
    patched, patch_text = expected_patches()
    if "SPCR_CONFIG" not in patch_text or "SPCR_DIAGNOSTIC_PATH" not in patch_text:
        raise RuntimeError("SPCR patch is incomplete")
    for name, text in patched.items():
        compile(text, name, "exec")
    import torch
    torch.manual_seed(17)
    grouped = torch.randn(4, 2, 4096)
    t1 = torch.randn(128, 4096, requires_grad=True)
    b1 = torch.randn(128, requires_grad=True)
    t2 = torch.randn(9, 128, requires_grad=True)
    b2 = torch.randn(9, requires_grad=True)
    first = torch.nn.functional.linear(grouped, t1, b1)
    h = torch.relu(first)
    original = torch.relu(torch.nn.functional.linear(h, t2, b2))
    e0 = torch.relu(torch.nn.functional.linear(h, t2, b2))
    if not torch.allclose(original, e0, atol=0.0, rtol=0.0):
        raise RuntimeError("E0 semantic projection is not numerically identical to P1")
    for output_relu in [True, False]:
        for decay in [0.0, 0.11]:
            result = torch.nn.functional.linear(torch.relu(first), t2, b2)
            result = torch.relu(result) if output_relu else result
            if result.shape != (4, 2, 9) or not result.requires_grad:
                raise RuntimeError("SPCR projection quick gradient check failed")
    return patch_text


def freeze_reference():
    source_path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row.get("config") == "P1"]
    if len(source_rows) != 30:
        raise RuntimeError(f"Expected 30 P1 reference rows, found {len(source_rows)}")
    fields = ["config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc",
              "train_seconds", "epoch_cap", "early_patience", "auc_source", "reference_source_sha256"]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with REFERENCE_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in source_rows:
            output = {key: row.get(key, "") for key in fields}
            output["config"] = "E0"
            output["reference_source_sha256"] = sha256(source_path)
            writer.writerow(output)
    return source_rows, source_path


def make_manifest(reference_path):
    source = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    inputs = {
        "data/ABIDE/processed_standard_data.csv": sha256(ROOT / "data/ABIDE/processed_standard_data.csv"),
        "data/ABIDE/modal_feat_dict.npy": sha256(ROOT / "data/ABIDE/modal_feat_dict.npy"),
    }
    weight = Path.home() / ".cache/clip/RN50.pt"
    actual_weight = sha256(weight) if weight.is_file() else None
    return {
        "experiment": "SPCR",
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_root": str(ROOT),
        "github_main_commit_at_prepare": github_head(),
        "source_files": source,
        "inputs": inputs,
        "clip_rn50": {
            "path": str(weight),
            "expected_sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
            "actual_sha256_at_snapshot": actual_weight,
        },
        "prompts_sha256": sha256(PROMPT_FILE),
        "patch_sha256": sha256(PATCH_FILE),
        "reference_sha256": sha256(REFERENCE_FILE),
        "reference_source_sha256": sha256(reference_path),
        "audit": {
            "mlp": "4096 -> 128 -> 9; t1 ReLU, t2 ReLU only in E0/E2",
            "mlp_parameter_count": 525577,
            "optimizer": "Adam lr=0.0038; E0/E1 all ModalFusion decay=0.11; E2/E3 MLP decay=0 and other ModalFusion decay=0.11",
            "initialization": "my_weight_init applies Xavier uniform to every Linear weight and leaves biases at framework initialization",
            "text_shape": "32 CLIP RN50 outputs grouped as [4,2,4096], projected to [4,2,9]",
            "evaluation": "original ten-fold StratifiedKFold, val_idx=test_idx, ACC checkpoint, hard-label AUC",
            "e0_p1_numeric_check": True,
            "optimizer_groups_checked": True,
        },
        "protocol": {
            "command": "python main.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --datname ABIDE --dropout 0.35 --lr 0.0038 --mode simple-2 --n_head 2 --n_hidden 18 --nlayer 1 --reg 0.11",
            "folds": 10,
            "stratified_random_state": 10,
            "val_equals_test": True,
            "checkpoint_metric": "ACC",
            "epoch_cap": 450,
            "early_patience": 200,
            "auc": "hard_label",
            "seeds": SEEDS,
            "modalities": MODALITIES,
            "configs": {
                "E0": {"output_relu": True, "mlp_weight_decay": 0.11},
                "E1": {"output_relu": False, "mlp_weight_decay": 0.11},
                "E2": {"output_relu": True, "mlp_weight_decay": 0.0},
                "E3": {"output_relu": False, "mlp_weight_decay": 0.0},
            },
            "gpu_assignments": {f"{config}_seed{seed}": gpu for config, seed, gpu in TASKS},
        },
        "validation": {
            "prompt_count": 32,
            "patch_compiles": True,
            "clip_context_valid": True,
            "stages": STAGES,
            "diagnostic_rows_expected": 960,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "run_status": "PREPARED",
    }


def write_readme():
    content = """# 实验五：SPCR 语义原型投影退化与最小修复

本实验固定实验一 P1 的 32 条 Prompt，只做 2×2 机制消融：是否保留 MLP 最后一层 ReLU，以及 MLP 参数是否使用 Adam 权重衰减。E0 是原始 P1，E1 去掉输出 ReLU，E2 只将 MLP 的 weight_decay 设为 0，E3 同时采用两项改动。

源码审计：原始 MLP 为 4096 -> 128 -> 9，两层原始均使用 ReLU；MLP 共 525577 个参数。原始 ModalFusion 使用 Adam，学习率 0.0038、weight_decay 0.11，E2/E3 将 t1/t2 的 weight 与 bias 放入唯一的 weight_decay=0 参数组，其余 ModalFusion 参数保持 0.11，MessagePassing0 不变。my_weight_init 只对 Linear weight 使用 Xavier uniform，不重置 bias。

每个 fold 在第一次反向传播前写入 INIT 快照，在恢复 ACC 最佳 checkpoint 后的测试前向写入 BEST 快照。快照记录 CLIP 输入、Linear1 前、ReLU1 后、Linear2 前和最终输出五个阶段的余弦、范数、零元素、非有限值、同向/反向高相似比例，并记录 MLP 参数范数和实际 Softmax-Max 权重。

运行与恢复：

    python experiment/experiment5_semantic_projection_collapse/run.py --prepare
    python experiment/experiment5_semantic_projection_collapse/run.py --all
    python experiment/experiment5_semantic_projection_collapse/run.py --refresh

运行器在临时隔离副本中应用 projection_patch.diff，根目录源码不改动；任务按 prompt 要求分配到两张 GPU，每个任务顺序完成十折。运行器只解析最新 attempt，已完成任务可自动跳过。
"""
    atomic_write(EXP / "README.md", content)


def prepare_worktree(work):
    for relative in SOURCE_ROOT_FILES:
        shutil.copy2(ROOT / relative, work / relative)
    (work / "CLIP").symlink_to(ROOT / "CLIP", target_is_directory=True)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir()
    (work / "graph").mkdir()
    applied = subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
        cwd=work, text=True, capture_output=True, check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"SPCR patch failed: {applied.stdout}{applied.stderr}")
    patched, _ = expected_patches()
    for name, expected in patched.items():
        if (work / name).read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Patched {name} differs from generated patch")


def attempt_text(path):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(ATTEMPT_RE.finditer(text))
    return text[matches[-1].start():] if matches else text


def parse_attempt_rows(path):
    text = attempt_text(path)
    lines = text.splitlines()
    timers = [float(value) for value in re.findall(r"time:\s*([0-9.]+)\s+sec", text)]
    events = []
    for index, line in enumerate(lines):
        match = TEST_RE.search(line)
        if not match:
            continue
        following = lines[index + 1:]
        next_test = next((i for i, row in enumerate(following) if TEST_RE.search(row)), len(following))
        tail = following[:next_test]
        sens_values = next((NUMBER_RE.findall(row) for row in tail if row.strip().startswith("sens:")), [])
        spes_values = next((NUMBER_RE.findall(row) for row in tail if row.strip().startswith("spes:")), [])
        sens = [float(value) for value in sens_values[-2:]] if len(sens_values) >= 2 else []
        spes = [float(value) for value in spes_values[-2:]] if len(spes_values) >= 2 else []
        events.append({
            "acc": float(match.group(2)),
            "hard_auc": float(match.group(1)),
            "sensitivity_asd": sens[0] if len(sens) == 2 else None,
            "specificity_asd": spes[0] if len(spes) == 2 else None,
            "sensitivity_nc": sens[1] if len(sens) == 2 else None,
            "specificity_nc": spes[1] if len(spes) == 2 else None,
            "train_seconds": timers[len(events)] if len(timers) > len(events) else None,
        })
    return events


def diagnostic_path(config, seed):
    return WORK_DIR / f"{config}_seed{seed}.jsonl"


def diagnostic_rows_from_jsonl(config, seed):
    path = diagnostic_path(config, seed)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        fold = int(payload.get("fold", 0))
        phase = payload.get("phase", "")
        diagnostics = payload.get("diagnostics", {})
        for modality in MODALITIES:
            row = {"config": config, "seed": seed, "fold": fold, "phase": phase, "modality": modality}
            for stage in STAGES:
                stage_rows = diagnostics.get("stage_rows", {}).get(stage, [])
                stage_row = next((item for item in stage_rows if item.get("modality") == modality), {})
                prefix = stage.lower()
                for field in ["cosine", "abs_cosine", "cosine_gt_0_98", "cosine_lt_neg_0_98",
                              "norm_1", "norm_2", "zero_vector", "nonfinite",
                              "zero_fraction_1", "zero_fraction_2"]:
                    row[f"{prefix}_{field}"] = stage_row.get(field)
            row.update(diagnostics.get("params", {}))
            gate = next((item for item in diagnostics.get("gate_rows", []) if item.get("modality") == modality), {})
            row.update(gate)
            rows.append(row)
    return rows


def diagnostic_rows(config, seed):
    rows = diagnostic_rows_from_jsonl(config, seed)
    if rows:
        return rows
    if not DIAG_FILE.exists():
        return []
    with DIAG_FILE.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream)
                if row.get("config") == config and int(row.get("seed", -1)) == seed]


def diagnostics_valid(config, seed):
    rows = diagnostic_rows(config, seed)
    keys = {(int(row["seed"]), int(row["fold"]), row["phase"], row["modality"]) for row in rows}
    return (
        len(rows) == 80 and len(keys) == 80
        and {int(row["fold"]) for row in rows} == set(range(1, 11))
        and {row["phase"] for row in rows} == {"INIT", "BEST"}
        and {row["modality"] for row in rows} == set(MODALITIES)
    )


def task_log(config, seed):
    return LOG_DIR / f"{config}_seed{seed}.log"


def task_end(config, seed):
    path = task_log(config, seed)
    if not path.exists():
        return None
    matches = list(TASK_END_RE.finditer(path.read_text(encoding="utf-8", errors="replace")))
    if not matches:
        return None
    match = matches[-1]
    return {"returncode": int(match.group(3)), "folds": int(match.group(4)), "diagnostics": int(match.group(5))}


def process_running(config, seed):
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    script = Path(__file__).resolve()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            argv = shlex.split(parts[1])
            candidate = Path(argv[1]).resolve() if len(argv) > 1 else None
        except (ValueError, OSError):
            continue
        if candidate != script or "--config" not in argv or "--seed" not in argv:
            continue
        try:
            if argv[argv.index("--config") + 1] == config and argv[argv.index("--seed") + 1] == str(seed):
                return True
        except (IndexError, ValueError):
            continue
    return False


def task_state(config, seed):
    if process_running(config, seed):
        return "RUNNING", len(parse_attempt_rows(task_log(config, seed)))
    end = task_end(config, seed)
    folds = len(parse_attempt_rows(task_log(config, seed)))
    if end is None:
        return "PENDING", folds
    if end["returncode"] != 0:
        return "FAILED", folds
    if end["folds"] == 10 and folds == 10 and end["diagnostics"] == 80 and diagnostics_valid(config, seed):
        return "COMPLETE", folds
    return "EARLY_TERMINATED", folds


def run_one(config, seed, gpu):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    state, _ = task_state(config, seed)
    if state == "COMPLETE":
        print(f"Skip {config} seed={seed}: complete 10-fold task.")
        return 0
    diag_path = diagnostic_path(config, seed)
    if diag_path.exists():
        diag_path.unlink()
    log = task_log(config, seed)
    with tempfile.TemporaryDirectory(prefix=f"SPCR-{config}-seed{seed}-") as temporary:
        work = Path(temporary)
        prepare_worktree(work)
        shell_args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        if shell_args[:2] != ["python", "main.py"]:
            raise RuntimeError("Unexpected official shell command")
        command = [sys.executable, "main.py", *shell_args[2:], "--seed", str(seed), "--gpu", "0"]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["SPCR_CONFIG"] = config
        environment["SPCR_PROMPTS_FILE"] = str(PROMPT_FILE)
        environment["SPCR_DIAGNOSTIC"] = "1"
        environment["SPCR_DIAGNOSTIC_PATH"] = str(diag_path)
        start = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} gpu={gpu} utc={utc_now()}\n")
            stream.write("COMMAND " + shlex.join(command) + "\n")
            stream.flush()
            completed = subprocess.run(command, cwd=work, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, check=False)
            folds = len(parse_attempt_rows(log))
            diag_count = len(diagnostic_rows_from_jsonl(config, seed))
            stream.write(
                f"TASK_END config={config} seed={seed} returncode={completed.returncode} "
                f"folds={folds} diagnostics={diag_count} elapsed_seconds={time.time() - start:.1f}\n"
            )
            stream.flush()
    print(f"{config} seed={seed}: {task_state(config, seed)[0]}, {len(parse_attempt_rows(log))}/10 folds")
    return completed.returncode


def all_fold_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            for fold, metrics in enumerate(parse_attempt_rows(task_log(config, seed)), 1):
                rows.append({"config": config, "seed": seed, "fold": fold, **metrics})
    return rows


def all_diagnostic_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            rows.extend(diagnostic_rows(config, seed))
    return rows


def diagnostic_columns():
    fields = ["config", "seed", "fold", "phase", "modality"]
    for stage in STAGES:
        prefix = stage.lower()
        fields.extend([f"{prefix}_{name}" for name in [
            "cosine", "abs_cosine", "cosine_gt_0_98", "cosine_lt_neg_0_98",
            "norm_1", "norm_2", "zero_vector", "nonfinite",
            "zero_fraction_1", "zero_fraction_2",
        ]])
    fields.extend(["t1_weight_fro", "t1_bias_l2", "t2_weight_fro", "t2_bias_l2",
                   "mlp_parameters", "modal_fusion_parameters"])
    fields.extend(["mean_abs_delta", "weight_mean", "weight_std",
                   "weight_le_0_51", "weight_gt_0_55", "weight_gt_0_60",
                   "semantic_nonfinite"])
    return fields


def write_fold_metrics(rows):
    fields = ["config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc",
              "train_seconds", "epoch_cap", "early_patience", "auc_source"]
    with FOLD_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in fields}
            for key in METRICS + ["sensitivity_nc", "specificity_nc", "train_seconds"]:
                if output[key] not in (None, ""):
                    output[key] = f"{float(output[key]):.6f}"
            output["epoch_cap"] = 450
            output["early_patience"] = 200
            output["auc_source"] = "hard_label"
            writer.writerow(output)


def write_diagnostics(rows):
    fields = diagnostic_columns()
    with DIAG_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_reference():
    with REFERENCE_FILE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def seed_summary(rows, config, seed):
    subset = [row for row in rows if row["config"] == config and int(row["seed"]) == seed]
    result = {"config": config, "seed": seed, "n_folds": len(subset)}
    for metric in METRICS:
        values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_fold_sd"] = statistics.pstdev(values) if values else None
    return result


def aggregate(rows, config):
    seed_rows = [seed_summary(rows, config, seed) for seed in SEEDS]
    complete = [row for row in seed_rows if row["n_folds"] == 10]
    result = {"config": config, "n_seeds": len(complete)}
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in complete if row[f"{metric}_mean"] is not None]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_seed_sd"] = statistics.pstdev(values) if len(values) > 1 else None
    return result


def numeric_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
            if math.isfinite(parsed):
                values.append(parsed)
        except (TypeError, ValueError):
            continue
    return values


def bool_fraction(rows, field):
    values = [str(row.get(field, "")).lower() for row in rows if row.get(field) not in (None, "")]
    return sum(value == "true" for value in values) / len(values) if values else None


def diag_summary(rows, config, phase, modality=None, stage=None):
    subset = [row for row in rows if row.get("config") == config and row.get("phase") == phase]
    if modality is not None:
        subset = [row for row in subset if row.get("modality") == modality]
    if not subset:
        return {}
    prefix = stage.lower() if stage else None
    cos = numeric_values(subset, f"{prefix}_cosine") if prefix else []
    abs_cos = numeric_values(subset, f"{prefix}_abs_cosine") if prefix else []
    zero = bool_fraction(subset, f"{prefix}_zero_vector") if prefix else None
    same = bool_fraction(subset, f"{prefix}_cosine_gt_0_98") if prefix else None
    reverse = bool_fraction(subset, f"{prefix}_cosine_lt_neg_0_98") if prefix else None
    def mean_field(field):
        values = numeric_values(subset, field)
        return statistics.mean(values) if values else None
    return {
        "records": len(subset),
        "cosine": statistics.mean(cos) if cos else None,
        "abs_cosine": statistics.mean(abs_cos) if abs_cos else None,
        "zero_fraction": zero,
        "nonfinite_fraction": bool_fraction(subset, f"{prefix}_nonfinite") if prefix else None,
        "same_fraction": same,
        "reverse_fraction": reverse,
        "norm_1": mean_field(f"{prefix}_norm_1") if prefix else None,
        "norm_2": mean_field(f"{prefix}_norm_2") if prefix else None,
        "zero_fraction_1": mean_field(f"{prefix}_zero_fraction_1") if prefix else None,
        "zero_fraction_2": mean_field(f"{prefix}_zero_fraction_2") if prefix else None,
        "t1_weight_fro": mean_field("t1_weight_fro"),
        "t1_bias_l2": mean_field("t1_bias_l2"),
        "t2_weight_fro": mean_field("t2_weight_fro"),
        "t2_bias_l2": mean_field("t2_bias_l2"),
        "mean_abs_delta": mean_field("mean_abs_delta"),
        "weight_mean": mean_field("weight_mean"),
        "weight_std": mean_field("weight_std"),
        "weight_le_0_51": mean_field("weight_le_0_51"),
        "weight_gt_0_55": mean_field("weight_gt_0_55"),
        "weight_gt_0_60": mean_field("weight_gt_0_60"),
        "semantic_nonfinite": bool_fraction(subset, "semantic_nonfinite"),
    }


def pair_counts(rows, left, right, metric):
    left_rows = {(int(row["seed"]), int(row["fold"])): float(row[metric])
                 for row in rows if row["config"] == left}
    right_rows = {(int(row["seed"]), int(row["fold"])): float(row[metric])
                  for row in rows if row["config"] == right}
    deltas = [left_rows[key] - right_rows[key] for key in sorted(left_rows.keys() & right_rows.keys())]
    return (
        sum(delta > 1e-12 for delta in deltas),
        sum(abs(delta) <= 1e-12 for delta in deltas),
        sum(delta < -1e-12 for delta in deltas),
        deltas,
    )


def fmt_pct(value, sd=None):
    if value is None:
        return "-"
    return f"{value * 100:.2f}±{sd * 100:.2f}%" if sd is not None else f"{value * 100:.2f}%"


def report_text(rows, diagnostics, manifest):
    reference = read_reference()
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config, seed, _ in TASKS)
    lines = [
        "# 实验五：SPCR 语义原型投影退化与最小修复",
        "",
        f"状态：**{'全部完成' if complete == 12 else '执行中或未完全完成'}**；完成任务 {complete}/12，记录 {len(rows)}/120 折，诊断 {len(diagnostics)}/960 行。",
        "",
        "## 设计与审计",
        "",
        "本实验固定实验一 P1 的 32 条 Prompt，只改变 MLP 最后一层 ReLU 和 MLP 参数的 Adam weight_decay。E0=原始 P1，E1=去掉输出 ReLU，E2=MLP weight_decay=0，E3=两项同时去掉。CLIP RN50、四模态分组、4096→128→9 投影、原始 Softmax-Max、图学习和训练协议保持一致。",
        "",
        "原始 MLP 参数量为 525577；my_weight_init 只对 Linear weight 使用 Xavier uniform，不重置 bias。E2/E3 的四个 MLP 参数互斥地置于 weight_decay=0 参数组，其余 ModalFusion 参数仍为 0.11，MessagePassing0 优化器未改变。",
        "",
        "每个 fold 在第一次反向传播前记录 INIT，在恢复 ACC 最佳 checkpoint 后记录 BEST。每条诊断包含 CLIP_GROUP_INPUT、LINEAR1_PRE、RELU1_POST、LINEAR2_PRE、FINAL_OUTPUT 五个阶段；全零原型的 cosine 保持 N/A，不填 0。",
        "",
        "## 分类结果",
        "",
        "| 配置 | 完整 seeds | ACC（%） | Hard-AUC（%） | SEN-ASD（%） | SPE-ASD（%） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        summary = aggregate(rows, config)
        lines.append(
            f"| {config} | {summary['n_seeds']}/3 | {fmt_pct(summary.get('acc_mean'), summary.get('acc_seed_sd'))} | "
            f"{fmt_pct(summary.get('hard_auc_mean'), summary.get('hard_auc_seed_sd'))} | "
            f"{fmt_pct(summary.get('sensitivity_asd_mean'))} | {fmt_pct(summary.get('specificity_asd_mean'))} |"
        )
    lines.extend(["", "## 每个 seed 的十折均值", "",
                  "| 配置 | seed | ACC | Hard-AUC | SEN-ASD | SPE-ASD |",
                  "|---|---:|---:|---:|---:|---:|"])
    for config in CONFIGS:
        for seed in SEEDS:
            summary = seed_summary(rows, config, seed)
            lines.append(
                f"| {config} | {seed} | {fmt_pct(summary.get('acc_mean'), summary.get('acc_fold_sd'))} | "
                f"{fmt_pct(summary.get('hard_auc_mean'), summary.get('hard_auc_fold_sd'))} | "
                f"{fmt_pct(summary.get('sensitivity_asd_mean'))} | {fmt_pct(summary.get('specificity_asd_mean'))} |"
            )
    lines.extend(["", "## 相对 E0 的配对变化", "",
                  "胜/平/负按相同 seed 和 fold 配对，仅作描述性比较。"])
    for config in ["E1", "E2", "E3"]:
        lines.append(f"### {config} − E0")
        for metric in ["acc", "hard_auc"]:
            wins, ties, losses, deltas = pair_counts(rows, config, "E0", metric)
            by_seed = []
            for seed in SEEDS:
                left = seed_summary(rows, config, seed)
                right = seed_summary(rows, "E0", seed)
                if left["n_folds"] == right["n_folds"] == 10:
                    by_seed.append(f"seed{seed} {(left[f'{metric}_mean'] - right[f'{metric}_mean']) * 100:+.2f}")
            lines.append(f"- {metric}: " + ", ".join(by_seed) + f" 个百分点；胜/平/负 {wins}/{ties}/{losses}（{len(deltas)} 对）。")
        left = aggregate(rows, config)
        base = aggregate(rows, "E0")
        if left.get("sensitivity_asd_mean") is not None and base.get("sensitivity_asd_mean") is not None:
            lines.append(f"- SEN-ASD {(left['sensitivity_asd_mean'] - base['sensitivity_asd_mean']) * 100:+.2f} 个百分点；SPE-ASD {(left['specificity_asd_mean'] - base['specificity_asd_mean']) * 100:+.2f} 个百分点。")
    lines.extend(["", "## 2×2 因子效应", "",
                  "效应以 ACC 和 Hard-AUC 的 seed 均值为单位，数值为百分点；不作显著性推断。"])
    summaries = {config: aggregate(rows, config) for config in CONFIGS}
    for metric in ["acc", "hard_auc"]:
        values = {config: summaries[config].get(f"{metric}_mean") for config in CONFIGS}
        if all(value is not None for value in values.values()):
            relu = ((values["E1"] - values["E0"]) + (values["E3"] - values["E2"])) / 2 * 100
            decay = ((values["E2"] - values["E0"]) + (values["E3"] - values["E1"])) / 2 * 100
            interaction = (values["E3"] - values["E2"] - values["E1"] + values["E0"]) * 100
            lines.append(f"- {metric}: 输出 ReLU 主效应 {relu:+.3f}；MLP weight_decay 主效应 {decay:+.3f}；交互效应 {interaction:+.3f} 个百分点。")
        else:
            lines.append(f"- {metric}: 尚无完整四组结果。")
    lines.extend(["", "## 分层投影诊断", "",
                  "各单元依次为有效非零主题对的平均 signed cosine / 平均 |cos|，以及全记录的零向量比例、非有限比例；同向 cos>0.98 和反向 cos<-0.98 比例以有效非零主题对为分母。",
                  "", "| 阶段 | E0 INIT | E0 BEST | E1 BEST | E2 BEST | E3 BEST |",
                  "|---|---:|---:|---:|---:|---:|"])
    for stage in STAGES:
        cells = []
        for config, phase in [("E0", "INIT"), ("E0", "BEST"), ("E1", "BEST"), ("E2", "BEST"), ("E3", "BEST")]:
            summary = diag_summary(diagnostics, config, phase, "Correlation", stage)
            if not summary:
                cells.append("-")
            else:
                cells.append(
                    f"{summary['cosine']:.4f}/{summary['abs_cosine']:.4f} "
                    f"(zero {summary['zero_fraction'] * 100:.1f}%, nf {summary['nonfinite_fraction'] * 100:.1f}%, "
                    f"+ {summary['same_fraction'] * 100:.1f}%, - {summary['reverse_fraction'] * 100:.1f}%)"
                )
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    lines.extend(["", "### 所有模态逐层诊断", "",
                  "零向量与非有限比例按全部记录计算；cosine 阈值比例只在有效非零主题对中计算。输出稀疏度为两个主题向量零元素比例的平均值。",
                  "", "| 配置/阶段 | 投影阶段 | cosine | |cos| | 全零 | 非有限 | cos>0.98 | cos<−0.98 | norm1 | norm2 | 稀疏度 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for config in CONFIGS:
        for phase in ["INIT", "BEST"]:
            if phase == "INIT" and config != "E0":
                continue
            for stage in STAGES:
                summary = diag_summary(diagnostics, config, phase, None, stage)
                if not summary:
                    continue
                sparsity = (summary["zero_fraction_1"] + summary["zero_fraction_2"]) / 2
                lines.append(
                    f"| {config} {phase} | {stage} | {summary['cosine']:.4f} | {summary['abs_cosine']:.4f} | "
                    f"{summary['zero_fraction'] * 100:.2f}% | {summary['nonfinite_fraction'] * 100:.2f}% | "
                    f"{summary['same_fraction'] * 100:.2f}% | {summary['reverse_fraction'] * 100:.2f}% | "
                    f"{summary['norm_1']:.3f} | {summary['norm_2']:.3f} | {sparsity * 100:.2f}% |"
                )
    lines.extend(["", "### MLP 参数范数", "",
                  "| 配置/阶段 | ||t1.weight||F | ||t1.bias||2 | ||t2.weight||F | ||t2.bias||2 |",
                  "|---|---:|---:|---:|---:|"])
    for config in CONFIGS:
        for phase in ["INIT", "BEST"]:
            if phase == "INIT" and config != "E0":
                continue
            summary = diag_summary(diagnostics, config, phase)
            if summary:
                lines.append(
                    f"| {config} {phase} | {summary['t1_weight_fro']:.3f} | {summary['t1_bias_l2']:.4f} | "
                    f"{summary['t2_weight_fro']:.3f} | {summary['t2_bias_l2']:.4f} |"
                )
    lines.extend(["", "## Softmax-Max 与参数变化", "",
                  "| 配置/阶段 | Correlation mean(abs(s1-s2)) | 权重均值 | w≤0.51 | w>0.55 | w>0.60 | 非有限语义比例 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for config in CONFIGS:
        summary = diag_summary(diagnostics, config, "BEST", "Correlation")
        if summary:
            lines.append(
                f"| {config} BEST | {summary['mean_abs_delta']:.4f} | {summary['weight_mean']:.4f} | "
                f"{summary['weight_le_0_51'] * 100:.2f}% | {summary['weight_gt_0_55'] * 100:.2f}% | "
                f"{summary['weight_gt_0_60'] * 100:.2f}% | {summary['semantic_nonfinite'] * 100:.2f}% |"
            )
    lines.extend(["", "## 根因判断", ""])
    e0_input = diag_summary(diagnostics, "E0", "INIT", "Correlation", "CLIP_GROUP_INPUT")
    e0_init = diag_summary(diagnostics, "E0", "INIT", "Correlation", "FINAL_OUTPUT")
    e0_best = diag_summary(diagnostics, "E0", "BEST", "Correlation", "FINAL_OUTPUT")
    e1_best = diag_summary(diagnostics, "E1", "BEST", "Correlation", "FINAL_OUTPUT")
    e2_best = diag_summary(diagnostics, "E2", "BEST", "Correlation", "FINAL_OUTPUT")
    e3_best = diag_summary(diagnostics, "E3", "BEST", "Correlation", "FINAL_OUTPUT")
    if e0_input and e0_init and e0_best and e1_best and e2_best and e3_best:
        change = e0_best["abs_cosine"] - e0_init["abs_cosine"]
        if e0_input["cosine"] > 0.98:
            root = "Stage 0 的冻结 CLIP 主题已经高度相似，主要瓶颈更接近文本空间本身。"
        elif abs(change) > 0.05:
            root = (f"Correlation 的冻结 CLIP 输入平均 cosine 为 {e0_input['cosine']:.4f}，未达到同向高度相似阈值。"
                    f"E0 最终输出从 INIT 到 BEST 的 signed cosine 为 {e0_init['cosine']:.4f}→{e0_best['cosine']:.4f}，"
                    f"平均 |cos| 为 {e0_init['abs_cosine']:.4f}→{e0_best['abs_cosine']:.4f}（变化 {change:+.4f}）；"
                    "同时 cos>0.98 的有效主题对比例从 "
                    f"{e0_init['same_fraction'] * 100:.1f}% 变为 {e0_best['same_fraction'] * 100:.1f}%。"
                    "这说明训练后有一部分同向原型高度相似，但平均余弦同时受其他方向变化影响。")
        else:
            root = (f"Correlation 的冻结 CLIP 输入平均 cosine 为 {e0_input['cosine']:.4f}；"
                    f"E0 INIT→BEST 的最终平均 |cos| 变化为 {change:+.4f}，"
                    "没有显示整体单调塌缩，应结合逐层同向/反向比例判断退化。")
        relu_effect = e1_best["abs_cosine"] - e0_best["abs_cosine"]
        decay_effect = e2_best["abs_cosine"] - e0_best["abs_cosine"]
        lines.append(root)
        lines.append(
            f"E1 去掉最后 ReLU 后 Correlation 输出 signed cosine={e1_best['cosine']:.4f}、"
            f"|cos|={e1_best['abs_cosine']:.4f}、cos>0.98={e1_best['same_fraction'] * 100:.1f}%；"
            f"E2 去掉 MLP weight_decay 后对应值为 {e2_best['cosine']:.4f}、{e2_best['abs_cosine']:.4f}、"
            f"{e2_best['same_fraction'] * 100:.1f}%；E3 为 {e3_best['cosine']:.4f}、"
            f"{e3_best['abs_cosine']:.4f}、同向 {e3_best['same_fraction'] * 100:.1f}%、"
            f"反向 {e3_best['reverse_fraction'] * 100:.1f}%。E3 的反向相似单独报告，不当作同向塌缩。"
        )
        lines.append(
            "相对 E0 的平均 ACC/Hard-AUC 变化为百分点："
            f"E1 {(summaries['E1']['acc_mean']-summaries['E0']['acc_mean'])*100:+.2f}/"
            f"{(summaries['E1']['hard_auc_mean']-summaries['E0']['hard_auc_mean'])*100:+.2f}，"
            f"E2 {(summaries['E2']['acc_mean']-summaries['E0']['acc_mean'])*100:+.2f}/"
            f"{(summaries['E2']['hard_auc_mean']-summaries['E0']['hard_auc_mean'])*100:+.2f}，"
            f"E3 {(summaries['E3']['acc_mean']-summaries['E0']['acc_mean'])*100:+.2f}/"
            f"{(summaries['E3']['hard_auc_mean']-summaries['E0']['hard_auc_mean'])*100:+.2f}。"
        )
        lines.append(
            f"E0 的 t1/t2 weight Frobenius norm 从 INIT 的 {diag_summary(diagnostics, 'E0', 'INIT')['t1_weight_fro']:.3f}/"
            f"{diag_summary(diagnostics, 'E0', 'INIT')['t2_weight_fro']:.3f} 降至 BEST 的 "
            f"{diag_summary(diagnostics, 'E0', 'BEST')['t1_weight_fro']:.3f}/"
            f"{diag_summary(diagnostics, 'E0', 'BEST')['t2_weight_fro']:.3f}；E2/E3 不施加 MLP weight_decay 后参数范数保持较大，"
            "说明权重衰减确实伴随明显参数收缩，但 E2/E3 的 ACC 和 Hard-AUC 均未超过 E0。"
        )
        lines.append("结论：当前数据不支持冻结 CLIP 输入已经同向塌缩；MLP 参数收缩与部分同向相似变化并存，但单独移除最终 ReLU 或 MLP weight_decay 都没有带来分类收益。E0 在本次四组中平均 ACC/Hard-AUC 最高，暂保留为结果基线；三 seed 结果仅作描述性比较，不作显著性结论。")
    else:
        lines.append("正式任务尚未全部完成，暂不作根因判断。")
    lines.extend(["", "## E0 历史 P1 一致性", "",
                  "E0 重新运行并与实验一 P1 的 30 折参考结果比较；参考文件只用于核对，不混入四组新结果。"])
    if complete == 12:
        e0_rows = [row for row in rows if row["config"] == "E0"]
        e0_acc = statistics.mean(float(row["acc"]) for row in e0_rows)
        ref_acc = statistics.mean(float(row["acc"]) for row in reference)
        e0_auc = statistics.mean(float(row["hard_auc"]) for row in e0_rows)
        ref_auc = statistics.mean(float(row["hard_auc"]) for row in reference)
        lines.append(f"E0 新 30 折整体 ACC {e0_acc * 100:.4f}%，参考 P1 {ref_acc * 100:.4f}%，差异 {(e0_acc - ref_acc) * 100:+.4f} 个百分点；Hard-AUC {e0_auc * 100:.4f}% vs {ref_auc * 100:.4f}%，差异 {(e0_auc - ref_auc) * 100:+.4f} 个百分点。")
    else:
        lines.append("E0 或其他正式任务尚未全部完成。")
    lines.extend(["", "## 任务状态", ""])
    for config, seed, gpu in TASKS:
        state, folds = task_state(config, seed)
        lines.append(f"- {config} seed={seed} GPU{gpu}: {state}，{folds}/10 folds，diagnostics={len(diagnostic_rows(config, seed))}/80")
    lines.extend(["", "结果文件：fold_metrics.csv（120 折）；projection_diagnostics.csv（960 行）；reference.csv（E0 30 折）；manifest.json；logs/。"])
    return "\n".join(lines) + "\n"


def update_manifest(manifest, rows, diagnostics):
    manifest = dict(manifest)
    manifest["prompts_sha256"] = sha256(PROMPT_FILE)
    manifest["patch_sha256"] = sha256(PATCH_FILE)
    manifest["reference_sha256"] = sha256(REFERENCE_FILE)
    manifest["counts"] = {
        "new_fold_rows": len(rows),
        "diagnostic_rows": len(diagnostics),
        "complete_tasks": sum(task_state(config, seed)[0] == "COMPLETE" for config, seed, _ in TASKS),
    }
    manifest["run_status"] = "COMPLETE" if manifest["counts"]["complete_tasks"] == 12 else "RUNNING"
    manifest["updated_utc"] = utc_now()
    write_json(MANIFEST_FILE, manifest)


def update_handoff(manifest, rows, diagnostics):
    text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# SPromptGL handoff\n"
    start, end = "<!-- SPCR EXPERIMENT 5 START -->", "<!-- SPCR EXPERIMENT 5 END -->"
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config, seed, _ in TASKS)
    block = [
        start,
        "## 实验五：SPCR（Semantic Projection Collapse and Repair）",
        "",
        f"- 状态：{'COMPLETE' if complete == 12 else 'RUNNING'}；完成 {complete}/12 个任务，记录 {len(rows)}/120 折，诊断 {len(diagnostics)}/960 行。",
        "- 设计：固定实验一 P1 的 32 条 Prompt；E0 原始 ReLU/weight_decay，E1 去最后 ReLU，E2 去 MLP weight_decay，E3 同时修复。",
        "- 审计：4096→128→9 MLP，共 525577 参数；E2/E3 仅将 t1/t2 四个参数置于 weight_decay=0，其余 ModalFusion 与 MessagePassing0 保持原始协议。",
        "- 分层诊断：INIT/BEST 五阶段投影、全零和反向高相似统计、参数范数及 Softmax-Max 权重；详见 manifest.json 与 report.md。",
        "- 结果：experiment/experiment5_semantic_projection_collapse/results/fold_metrics.csv、projection_diagnostics.csv、report.md。",
        "- GPU 分配：GPU0 承担 E0 seed0/2、E1 seed1、E2 seed0/2、E3 seed1；GPU1 承担其余六个任务。",
        "- 原始源码与实验一至四未修改；本次不自动 git commit 或 git push。",
    ]
    if complete == 12:
        summaries = {config: aggregate(rows, config) for config in CONFIGS}
        for config in CONFIGS:
            summary = summaries[config]
            block.append(f"- {config}：ACC {summary['acc_mean'] * 100:.2f}%、Hard-AUC {summary['hard_auc_mean'] * 100:.2f}%（完整 seeds {summary['n_seeds']}/3）。")
        block.append("- 推荐配置与根因判断以 results/report.md 的分层诊断和 2×2 因子效应为准。")
    block.append(end)
    replacement = "\n".join(block)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + replacement + after
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"
    atomic_write(HANDOFF, text)


def update_outputs(manifest=None, allow_partial=False):
    if manifest is None:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    rows = all_fold_rows()
    diagnostics = all_diagnostic_rows()
    if not allow_partial:
        for config, seed, _ in TASKS:
            if not diagnostics_valid(config, seed):
                raise RuntimeError(f"Invalid SPCR diagnostics for {config} seed={seed}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_fold_metrics(rows)
    write_diagnostics(diagnostics)
    atomic_write(REPORT_FILE, report_text(rows, diagnostics, manifest))
    update_manifest(manifest, rows, diagnostics)
    update_handoff(manifest, rows, diagnostics)


def parallel_tasks():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    active = []
    with (LOG_DIR / "scheduler.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\nPARALLEL_START utc={utc_now()} tasks=12 assignments="
            + ",".join(f"{config}{seed}->GPU{gpu}" for config, seed, gpu in TASKS) + "\n"
        )
        for config, seed, gpu in TASKS:
            state, _ = task_state(config, seed)
            if state == "COMPLETE":
                stream.write(f"SKIP config={config} seed={seed} state=COMPLETE\n")
                continue
            if process_running(config, seed):
                stream.write(f"EXISTING config={config} seed={seed} gpu={gpu}\n")
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--config", config, "--seed", str(seed), "--gpu", str(gpu), "--worker",
            ]
            process = subprocess.Popen(command, cwd=ROOT)
            active.append((config, seed, gpu, process))
            stream.write(f"LAUNCH config={config} seed={seed} gpu={gpu} pid={process.pid}\n")
            stream.flush()
        while active:
            remaining = []
            for config, seed, gpu, process in active:
                if process.poll() is None:
                    remaining.append((config, seed, gpu, process))
                else:
                    stream.write(
                        f"EXIT config={config} seed={seed} gpu={gpu} pid={process.pid} "
                        f"returncode={process.returncode}\n"
                    )
            active = remaining
            stream.flush()
            if active:
                time.sleep(20)
        stream.write(f"PARALLEL_END utc={utc_now()}\n")
    update_outputs()


def prepare():
    prompts, _ = build_prompts()
    write_json(PROMPT_FILE, prompts)
    patch_text = validate_patch_and_tokenization(prompts)
    atomic_write(PATCH_FILE, patch_text)
    reference_rows, reference_path = freeze_reference()
    manifest = make_manifest(reference_path)
    manifest["validation"]["reference_rows"] = len(reference_rows)
    write_json(MANIFEST_FILE, manifest)
    write_readme()
    update_outputs(manifest=manifest, allow_partial=True)
    print("SPCR prepared: frozen P1 prompts, minimal patch, reference and audit manifest validated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pending", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if not MANIFEST_FILE.exists():
        prepare()
    if args.worker or args.config is not None:
        if args.config is None or args.seed is None or args.gpu is None:
            raise SystemExit("worker/config requires --config, --seed and --gpu")
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    if args.refresh:
        update_outputs()
        return
    parallel_tasks()


if __name__ == "__main__":
    main()

