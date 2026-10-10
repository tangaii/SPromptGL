#!/usr/bin/env python3
"""Experiment 4: disease-aware functional-connectivity Prompting (DAFCP)."""

import argparse
import csv
import difflib
import hashlib
import json
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
RESULT_DIR = EXP / "results"
WORK_DIR = LOG_DIR / ".working"
PROMPT_FILE = EXP / "prompts.json"
PATCH_FILE = EXP / "prompt_patch.diff"
MANIFEST_FILE = EXP / "manifest.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_results.csv"
DIAG_FILE = RESULT_DIR / "diagnostics.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = ["C1", "C2"]
ALL_CONFIGS = ["C0", "C1", "C2"]
SEEDS = [0, 1, 2]
GPU_ASSIGNMENTS = {
    ("C1", 0): 0, ("C1", 2): 0, ("C2", 1): 0,
    ("C1", 1): 1, ("C2", 0): 1, ("C2", 2): 1,
}
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=C[12] seed=\d+")
TASK_END_RE = re.compile(r"TASK_END config=(C[12]) seed=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+)")


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


def git_main_commit():
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        )
        return output.split()[0] if output.split() else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def p1_source():
    source = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    return json.loads(source.read_text(encoding="utf-8")), source


def theme(name, texts):
    return {
        "theme": name,
        "prompts": [
            {"text": text, "feature_basis": "DAFCP fixed Correlation control", "review_status": "fixed"}
            for text in texts
        ],
    }


def prompt_block(theme_a, theme_b):
    return {"themes": [theme_a, theme_b]}


C1_CORRELATION = prompt_block(
    theme("typical_functional_network_organization", [
        "Typical resting-state brain networks coordinate activity across distributed regions.",
        "Functional connectivity describes interactions within and between brain networks.",
        "Typical connectivity patterns include coupling among sensory and cognitive systems.",
        "Resting-state connectivity organization can vary across individuals and ages.",
    ]),
    theme("autism_associated_connectivity_alterations", [
        "Autism-associated resting-state functional connectivity can show heterogeneous differences across brain networks.",
        "Autism-associated connectivity changes have been reported across default-mode and attention-related brain systems.",
        "Autism-associated functional connections may exhibit both higher and lower coupling across brain regions.",
        "Autism-associated connectivity profiles can vary among individuals and across developmental stages.",
    ]),
)
C2_CORRELATION = prompt_block(
    theme("typical_functional_network_organization", [
        "Typical resting-state brain networks coordinate activity across distributed regions.",
        "Functional connectivity describes interactions within and between brain networks.",
        "Typical connectivity patterns include coupling among sensory and cognitive systems.",
        "Resting-state connectivity organization can vary across individuals and ages.",
    ]),
    theme("population_level_connectivity_variations", [
        "Population-level resting-state functional connectivity can show heterogeneous differences across brain networks.",
        "Population-level connectivity changes have been reported across default-mode and attention-related brain systems.",
        "Population-level functional connections may exhibit both higher and lower coupling across brain regions.",
        "Population-level connectivity profiles can vary among individuals and across developmental stages.",
    ]),
)


def flatten_modalities(modalities):
    sentences = []
    for modality in MODALITIES:
        for theme_item in modalities[modality]["themes"]:
            sentences.extend(item["text"] for item in theme_item["prompts"])
    return sentences


def validate_prompts(prompts):
    if prompts["modality_order"] != MODALITIES:
        raise ValueError("Unexpected modality order")
    source, _ = p1_source()
    expected_c0 = flatten_modalities(source["p1"])
    for config in ALL_CONFIGS:
        sentences = flatten_modalities(prompts["configs"][config]["modalities"])
        if len(sentences) != 32:
            raise ValueError(f"{config} must contain exactly 32 prompts")
        if any(not sentence for sentence in sentences):
            raise ValueError(f"{config} contains an empty prompt")
    if flatten_modalities(prompts["configs"]["C0"]["modalities"]) != expected_c0:
        raise ValueError("C0 differs from experiment1 P1")
    c1 = prompts["configs"]["C1"]["modalities"]
    c2 = prompts["configs"]["C2"]["modalities"]
    for modality in MODALITIES[:3]:
        if c1[modality] != c2[modality] or c1[modality] != source["p1"][modality]:
            raise ValueError(f"{modality} is not frozen from P1")
    c1_first = [item["text"] for item in c1["Correlation"]["themes"][0]["prompts"]]
    c2_first = [item["text"] for item in c2["Correlation"]["themes"][0]["prompts"]]
    if c1_first != c2_first:
        raise ValueError("C1/C2 Correlation first four prompts differ")
    c1_last = [item["text"] for item in c1["Correlation"]["themes"][1]["prompts"]]
    c2_last = [item["text"] for item in c2["Correlation"]["themes"][1]["prompts"]]
    for left, right in zip(c1_last, c2_last):
        if left.replace("Autism-associated", "Population-level") != right:
            raise ValueError("C1/C2 Correlation matched substitution failed")
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from CLIP.clip import clip
        for config in ALL_CONFIGS:
            tokens = clip.tokenize(flatten_modalities(prompts["configs"][config]["modalities"]))
            if int(tokens.shape[1]) > 77:
                raise ValueError(f"{config} exceeds CLIP context length")
    except ImportError as exc:
        raise RuntimeError("CLIP tokenizer is required for prompt validation") from exc


def build_prompts():
    source, source_path = p1_source()
    p1 = source["p1"]
    prompts = {
        "schema_version": 1,
        "experiment": "DAFCP",
        "modality_order": MODALITIES,
        "source_experiment1_p1_sha256": sha256(source_path),
        "configs": {
            "C0": {"name": "P1_BASELINE", "source": "experiment1 P1 frozen verbatim", "modalities": deepcopy(p1)},
            "C1": {"name": "ASD_DISEASE_FC", "source": "P1 with fixed ASD-aware Correlation Prompt", "modalities": deepcopy(p1)},
            "C2": {"name": "MATCHED_GENERIC_FC", "source": "P1 with fixed matched generic Correlation Prompt", "modalities": deepcopy(p1)},
        },
        "correlation_design": {
            "C1_theme_B": "Autism-associated connectivity alterations",
            "C2_theme_B": "Population-level connectivity variations",
            "matched_substitution": "Autism-associated -> Population-level",
            "evidence": [
                "Lin et al., 2025, Autism Research, DOI: 10.1002/aur.70128",
                "Nature Mental Health, 2025, DOI: 10.1038/s44220-025-00431-5",
            ],
            "scope_note": "The 256-dimensional Correlation input contains selected edges without retained ROI identities; prompts express candidate network-level semantics, not ROI supervision or an invariant ASD signature.",
        },
    }
    prompts["configs"]["C1"]["modalities"]["Correlation"] = deepcopy(C1_CORRELATION)
    prompts["configs"]["C2"]["modalities"]["Correlation"] = deepcopy(C2_CORRELATION)
    validate_prompts(prompts)
    return prompts


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
    """Encode the frozen DAFCP P1/Correlation prompt set once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        prompt_tokens = text
        config = os.environ.get("DAFCP_CONFIG")
        if config:
            with open(os.environ["DAFCP_PROMPTS_FILE"], encoding="utf-8") as stream:
                prompt_data = json.load(stream)
            sentences = []
            modalities = prompt_data["configs"][config]["modalities"]
            for modality in prompt_data["modality_order"]:
                for theme_item in modalities[modality]["themes"]:
                    sentences.extend(item["text"] for item in theme_item["prompts"])
            if len(sentences) != 32:
                raise ValueError("DAFCP configurations require 32 ordered prompts.")
            prompt_tokens = clip.tokenize(sentences).to(device)
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(prompt_tokens).float().detach()
    return _prompt_text_features
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate original prompt encoder")
    source = source.replace(old, new, 1)
    old = """        self.register_buffer('prompt_text_features', get_prompt_text_features())
"""
    new = """        self.register_buffer('prompt_text_features', get_prompt_text_features())
        self.dafcp_config = os.environ.get("DAFCP_CONFIG", "C1")
        self.dafcp_diag_enabled = os.environ.get("DAFCP_DIAGNOSTIC", "0") == "1"
        self._dafcp_eval_idx = None
        self._dafcp_last = None
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate prompt buffer registration")
    source = source.replace(old, new, 1)
    old = """        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
"""
    new = """        text_features = self.prompt_text_features
        if text_features.size(0) == 8:
            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
            text_features = self.MLP(text_features)
        else:
            text_features = self.MLP(text_features.view(4, 2, 4096))
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate original prompt grouping")
    source = source.replace(old, new, 1)
    start_marker = "        text_features_norm = F.normalize(text_features, p=2, dim=-1)\n"
    end_marker = "        similarity_matrix = similarity_matrix[:,:,:,-1,:]\n"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("Cannot locate similarity block")
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
    old = """        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
"""
    new = """        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # Preserve the original Softmax-Max semantic weighting and fusion.
        part_x = similarity_max.unsqueeze(-1) * part_x
        if self.dafcp_diag_enabled and not self.training and self._dafcp_eval_idx is not None:
            self._dafcp_last = self._dafcp_diagnostics(
                raw_similarity_matrix.detach(), similarity_matrix.detach(), text_features.detach()
            )
        else:
            self._dafcp_last = None
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate semantic weighting block")
    source = source.replace(old, new, 1)
    class_start = source.find("class PLN(nn.Module):")
    forward_anchor = source.find("    def forward(self, x):\n", class_start)
    if class_start < 0 or forward_anchor < 0:
        raise RuntimeError("Cannot locate PLN class")
    helper = '''    @staticmethod
    def _scalar_stats(values):
        values = values.reshape(-1).float()
        finite = torch.isfinite(values)
        clean = values[finite]
        if clean.numel() == 0:
            return {"mean": None, "std": None, "min": None, "max": None,
                    "p5": None, "median": None, "p95": None, "nonfinite": int((~finite).sum().item())}
        q = torch.quantile(clean, torch.tensor([0.05, 0.50, 0.95], device=clean.device))
        return {"mean": float(clean.mean().item()),
                "std": float(clean.std(unbiased=False).item()),
                "min": float(clean.min().item()),
                "max": float(clean.max().item()),
                "p5": float(q[0].item()), "median": float(q[1].item()),
                "p95": float(q[2].item()), "nonfinite": int((~finite).sum().item())}

    def _dafcp_diagnostics(self, raw_similarity, softmax_similarity, text_features):
        with torch.no_grad():
            sample_index = self._dafcp_eval_idx
            if not torch.is_tensor(sample_index):
                sample_index = torch.as_tensor(sample_index, device=raw_similarity.device, dtype=torch.long)
            else:
                sample_index = sample_index.to(raw_similarity.device, dtype=torch.long)
            raw_similarity = raw_similarity.index_select(0, sample_index)
            softmax_similarity = softmax_similarity.index_select(0, sample_index)
            prototypes = text_features if text_features.dim() == 3 else text_features.unsqueeze(0).expand(4, -1, -1)
            norms = torch.linalg.vector_norm(prototypes, dim=-1)
            finite_proto = torch.isfinite(prototypes).all(dim=-1)
            rows = []
            for index, modality in enumerate(["PHENO", "ANAT", "FUNC", "Correlation"]):
                p1, p2 = prototypes[index, 0], prototypes[index, 1]
                valid_proto = bool(finite_proto[index].all().item() and (norms[index] > 1e-12).all().item())
                cosine = None if not valid_proto else float(F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item())
                raw = raw_similarity[:, index, :, :].reshape(-1, 2)
                weights = softmax_similarity[:, index, :, :].max(dim=-1).values.reshape(-1)
                delta = (raw[:, 0] - raw[:, 1]).abs()
                stats = self._scalar_stats(weights)
                rows.append({
                    "modality": modality,
                    "semantic_status": "valid",
                    "prototype_cosine": cosine,
                    "prototype_abs_cosine": None if cosine is None else abs(cosine),
                    "prototype_abs_cosine_gt_0_98": None if cosine is None else abs(cosine) > 0.98,
                    "prototype_norm_1": float(norms[index, 0].item()),
                    "prototype_norm_2": float(norms[index, 1].item()),
                    "prototype_zero_vector": bool((norms[index] <= 1e-12).any().item()),
                    "prototype_nonfinite": bool((~finite_proto[index]).any().item()),
                    "mean_s1": float(raw[:, 0].mean().item()),
                    "mean_s2": float(raw[:, 1].mean().item()),
                    "mean_abs_delta": float(delta.mean().item()),
                    "std_abs_delta": float(delta.std(unbiased=False).item()),
                    "weight_mean": stats["mean"], "weight_std": stats["std"],
                    "weight_min": stats["min"], "weight_max": stats["max"],
                    "weight_p5": stats["p5"], "weight_median": stats["median"], "weight_p95": stats["p95"],
                    "weight_le_0_51": float((weights <= 0.51).float().mean().item()),
                    "weight_gt_0_55": float((weights > 0.55).float().mean().item()),
                    "weight_gt_0_60": float((weights > 0.60).float().mean().item()),
                    "weight_nonfinite": stats["nonfinite"],
                })
            return {"config": self.dafcp_config, "prototype_rows": rows}

'''
    source = source[:forward_anchor] + helper + source[forward_anchor:]
    return source


def patch_model(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    old = """        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        prob = prob[eval_idx]
"""
    new = """        if tst and os.environ.get("DAFCP_DIAGNOSTIC_PATH"):
            self.ModalFusion._dafcp_eval_idx = eval_idx
        else:
            self.ModalFusion._dafcp_eval_idx = None
        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        if tst and os.environ.get("DAFCP_DIAGNOSTIC_PATH"):
            diagnostics = getattr(self.ModalFusion, "_dafcp_last", None)
            fold = getattr(self, "_dafcp_fold_number", None)
            if diagnostics is not None and fold is not None:
                with open(os.environ["DAFCP_DIAGNOSTIC_PATH"], "a", encoding="utf-8") as stream:
                    json.dump({"fold": int(fold), "diagnostics": diagnostics}, stream, ensure_ascii=False)
                    stream.write("\\n")
        self.ModalFusion._dafcp_eval_idx = None
        prob = prob[eval_idx]
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate model evaluation block")
    return source.replace(old, new, 1)


def patch_main(source):
    old = """        val_acc.append(best_val_acc)
        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
"""
    new = """        if os.environ.get("DAFCP_DIAGNOSTIC_PATH"):
            agent._dafcp_fold_number = clk
        val_acc.append(best_val_acc)
        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
"""
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate main test evaluation call")
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


def freeze_reference():
    source_path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    rows = []
    with source_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["config"] == "P1":
                output = {key: row.get(key, "") for key in row}
                output["config"] = "C0"
                output["reference_source_sha256"] = sha256(source_path)
                rows.append(output)
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 P1 reference rows, found {len(rows)}")
    fields = ["config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc", "train_seconds", "epoch_cap", "early_patience", "auc_source", "reference_source_sha256"]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with REFERENCE_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return rows, source_path


def freeze_reference_diagnostics():
    source_path = ROOT / "experiment/experiment3_prompt_effectiveness_diagnostics/results/prompt_diagnostics.csv"
    with source_path.open(newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row["config"] == "D1"]
    result = []
    for seed in SEEDS:
        rows = [row for row in source_rows if int(row["seed"]) == seed]
        if len(rows) != 40:
            raise RuntimeError(f"Expected 40 frozen C0 diagnostic rows for seed {seed}, found {len(rows)}")
        for index, row in enumerate(rows):
            fold = index // 4 + 1
            if row["modality"] != MODALITIES[index % 4]:
                raise RuntimeError("Experiment3 diagnostic modality order is not recoverable")
            output = dict(row)
            output["config"] = "C0"
            output["fold"] = fold
            output["source"] = "experiment3 D1 diagnostic, fold repaired by row order"
            result.append(output)
    keys = {(row["config"], int(row["seed"]), int(row["fold"]), row["modality"]) for row in result}
    if len(result) != 120 or len(keys) != 120:
        raise RuntimeError("Frozen C0 diagnostics do not have unique keys")
    return result, source_path


def make_manifest(prompts, reference_rows, reference_path, reference_diag, reference_diag_path):
    source = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    data = {
        "data/ABIDE/processed_standard_data.csv": sha256(ROOT / "data/ABIDE/processed_standard_data.csv"),
        "data/ABIDE/modal_feat_dict.npy": sha256(ROOT / "data/ABIDE/modal_feat_dict.npy"),
    }
    weight = Path.home() / ".cache/clip/RN50.pt"
    actual_weight = sha256(weight) if weight.is_file() else None
    return {
        "experiment": "DAFCP",
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_root": str(ROOT),
        "github_main_commit_at_prepare": git_main_commit(),
        "source_files": source,
        "inputs": data,
        "clip_rn50": {
            "path": str(weight),
            "expected_sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
            "actual_sha256_at_snapshot": actual_weight,
        },
        "protocol": {
            "command": "python main.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --datname ABIDE --dropout 0.35 --lr 0.0038 --mode simple-2 --n_head 2 --n_hidden 18 --nlayer 1 --reg 0.11",
            "folds": 10, "stratified_random_state": 10, "val_equals_test": True,
            "checkpoint_metric": "ACC", "epoch_cap": 450, "early_patience": 200,
            "auc": "hard_label", "seeds": SEEDS, "modalities": MODALITIES,
        },
        "prompts_sha256": sha256(PROMPT_FILE),
        "patch_sha256": sha256(PATCH_FILE),
        "reference_sha256": sha256(REFERENCE_FILE),
        "reference_source_sha256": sha256(reference_path),
        "reference_diagnostics_source_sha256": sha256(reference_diag_path),
        "reference_diagnostics": reference_diag,
        "validation": {
            "patch_compiles": True, "prompts_valid": True, "clip_context_valid": True,
            "reference_rows": len(reference_rows), "reference_diagnostic_rows": len(reference_diag),
            "diagnostic_fold_repair": "Experiment3 D1 diagnostic rows were all fold=1 because its per-EvalHelper counter reset; rows are frozen here in original 10-fold order.",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "run_status": "PREPARED",
    }


def prepare():
    prompts = build_prompts()
    write_json(PROMPT_FILE, prompts)
    patched, patch_text = expected_patches()
    for name, source in patched.items():
        compile(source, name, "exec")
    atomic_write(PATCH_FILE, patch_text)
    reference_rows, reference_path = freeze_reference()
    reference_diag, reference_diag_path = freeze_reference_diagnostics()
    manifest = make_manifest(prompts, reference_rows, reference_path, reference_diag, reference_diag_path)
    write_json(MANIFEST_FILE, manifest)
    update_outputs(manifest=manifest, allow_partial=True)
    print("DAFCP prepared: prompts, patch, manifest and C0 references validated.")


def attempt_text(path):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = list(ATTEMPT_RE.finditer(text))
    return text[starts[-1].start():] if starts else text


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


def read_diagnostic_rows(config, seed):
    path = diagnostic_path(config, seed)
    if path.exists():
        result = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            fold = int(payload.get("fold", 0))
            for row in payload.get("diagnostics", {}).get("prototype_rows", []):
                result.append({"config": config, "seed": seed, "fold": fold, **row})
        return result
    # Completed runs remove .working; use the durable CSV for resume/refresh.
    if not DIAG_FILE.exists():
        return []
    with DIAG_FILE.open(newline="", encoding="utf-8") as stream:
        return [
            row for row in csv.DictReader(stream)
            if row.get("config") == config and int(row.get("seed", -1)) == int(seed)
        ]


def diagnostics_valid(config, seed):
    rows = read_diagnostic_rows(config, seed)
    keys = {(int(row["seed"]), int(row["fold"]), row["modality"]) for row in rows}
    return (
        len(rows) == 40
        and len(keys) == 40
        and {int(row["fold"]) for row in rows} == set(range(1, 11))
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
    return {
        "returncode": int(match.group(3)),
        "folds": int(match.group(4)),
        "diagnostics": int(match.group(5)),
    }


def process_running(config, seed):
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    own_script = Path(__file__).resolve()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            argv = shlex.split(parts[1])
            script = Path(argv[1]).resolve() if len(argv) > 1 else None
        except (ValueError, OSError):
            continue
        if script != own_script or "--config" not in argv or "--seed" not in argv:
            continue
        try:
            if argv[argv.index("--config") + 1] == config and argv[argv.index("--seed") + 1] == str(seed):
                return True
        except (ValueError, IndexError):
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
    if end["folds"] == 10 and folds == 10 and end["diagnostics"] == 40 and diagnostics_valid(config, seed):
        return "COMPLETE", folds
    return "EARLY_TERMINATED", folds


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
        raise RuntimeError(f"DAFCP patch failed: {applied.stdout}{applied.stderr}")
    patched, _ = expected_patches()
    for name, expected in patched.items():
        if (work / name).read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Patched {name} differs from generated patch")


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
    with tempfile.TemporaryDirectory(prefix=f"SPromptGL-DAFCP-{config}-seed{seed}-") as temporary:
        work = Path(temporary)
        prepare_worktree(work)
        shell_args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        if shell_args[:2] != ["python", "main.py"]:
            raise RuntimeError("Unexpected official shell command")
        command = [sys.executable, "main.py", *shell_args[2:], "--seed", str(seed), "--gpu", "0"]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["DAFCP_CONFIG"] = config
        environment["DAFCP_PROMPTS_FILE"] = str(PROMPT_FILE)
        environment["DAFCP_DIAGNOSTIC"] = "1"
        environment["DAFCP_DIAGNOSTIC_PATH"] = str(diag_path)
        start = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} gpu={gpu} utc={utc_now()}\n")
            stream.write("COMMAND " + shlex.join(command) + "\n")
            stream.flush()
            completed = subprocess.run(
                command, cwd=work, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
            folds = len(parse_attempt_rows(log))
            diag_count = len(read_diagnostic_rows(config, seed))
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


DIAG_COLUMNS = [
    "config", "seed", "fold", "modality", "semantic_status",
    "prototype_cosine", "prototype_abs_cosine", "prototype_abs_cosine_gt_0_98",
    "prototype_norm_1", "prototype_norm_2", "prototype_zero_vector", "prototype_nonfinite",
    "mean_s1", "mean_s2", "mean_abs_delta", "std_abs_delta",
    "weight_mean", "weight_std", "weight_min", "weight_max",
    "weight_p5", "weight_median", "weight_p95",
    "weight_le_0_51", "weight_gt_0_55", "weight_gt_0_60", "weight_nonfinite",
]


def all_diagnostic_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            rows.extend(read_diagnostic_rows(config, seed))
    return rows


def write_fold_results(rows):
    fields = [
        "config", "seed", "fold", *METRICS, "sensitivity_nc",
        "specificity_nc", "train_seconds", "epoch_cap", "early_patience", "auc_source",
    ]
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
    with DIAG_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIAG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in DIAG_COLUMNS})


def reference_rows():
    with REFERENCE_FILE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def seed_summary(rows, config, seed):
    subset = [row for row in rows if row["config"] == config and int(row["seed"]) == int(seed)]
    result = {"config": config, "seed": int(seed), "n_folds": len(subset)}
    for metric in METRICS:
        values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_fold_sd"] = statistics.pstdev(values) if values else None
    return result


def aggregate(rows, config):
    seeds = [seed_summary(rows, config, seed) for seed in SEEDS]
    seeds = [row for row in seeds if row["n_folds"] == 10]
    result = {"config": config, "n_seeds": len(seeds)}
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in seeds if row[f"{metric}_mean"] is not None]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_seed_sd"] = statistics.pstdev(values) if len(values) > 1 else None
    return result


def diag_summary(rows, config, modality="Correlation"):
    subset = [
        row for row in rows
        if row.get("config") == config and row.get("modality") == modality
        and row.get("semantic_status") == "valid"
    ]
    if not subset:
        return {}
    zero = [str(row.get("prototype_zero_vector")).lower() == "true" for row in subset]
    nonzero = [
        row for row, is_zero in zip(subset, zero)
        if not is_zero and row.get("prototype_abs_cosine") not in (None, "")
    ]
    def mean(field, values=subset):
        numbers = [float(row[field]) for row in values if row.get(field) not in (None, "")]
        return statistics.mean(numbers) if numbers else None
    return {
        "records": len(subset),
        "zero_fraction": statistics.mean(zero),
        "nonzero_records": len(nonzero),
        "prototype_abs_cosine_all": mean("prototype_abs_cosine"),
        "prototype_abs_cosine_nonzero": mean("prototype_abs_cosine", nonzero),
        "collapse_nonzero": statistics.mean([
            abs(float(row["prototype_abs_cosine"])) > 0.98 for row in nonzero
        ]) if nonzero else None,
        "mean_abs_delta": mean("mean_abs_delta"),
        "weight_mean": mean("weight_mean"),
        "weight_std": mean("weight_std"),
        "weight_le_0_51": mean("weight_le_0_51"),
        "weight_gt_0_55": mean("weight_gt_0_55"),
        "weight_gt_0_60": mean("weight_gt_0_60"),
    }


def pair_counts(left_rows, right_rows, metric):
    left = {(int(row["seed"]), int(row["fold"])): float(row[metric]) for row in left_rows}
    right = {(int(row["seed"]), int(row["fold"])): float(row[metric]) for row in right_rows}
    deltas = [left[key] - right[key] for key in sorted(left.keys() & right.keys())]
    return (
        sum(delta > 1e-12 for delta in deltas),
        sum(abs(delta) <= 1e-12 for delta in deltas),
        sum(delta < -1e-12 for delta in deltas),
        len(deltas),
    )


def pct(value, digits=2):
    return "-" if value is None else f"{value * 100:.{digits}f}"


def report_text(rows, diagnostics, manifest):
    c0_rows = reference_rows()
    all_metric_rows = {
        "C0": c0_rows,
        "C1": [row for row in rows if row["config"] == "C1"],
        "C2": [row for row in rows if row["config"] == "C2"],
    }
    c0_diag = manifest.get("reference_diagnostics", [])
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS)
    lines = [
        "# 实验四：疾病感知功能连接 Prompt（DAFCP）",
        "",
        f"状态：**{'全部完成' if complete == 6 else '执行中或未完全完成'}**；C1/C2 完成任务 {complete}/6，记录 {len(rows)}/60 个新折。",
        "",
        "## 研究问题与控制变量",
        "",
        "本实验只改变 Correlation 的 8 条 Prompt：C1 的后四条加入谨慎表述的 ASD 相关网络语义，C2 使用句式、长度和术语匹配的 Population-level 对照；PHENO、ANAT、FUNC 的 24 条 P1 Prompt、MLP、原型数量、Softmax-Max、融合、分类器、优化器和训练协议均保持不变。C0 不重新训练，直接冻结实验一 P1 的 30 条逐折结果。",
        "",
        "Correlation 输入是 256 维 RFE 选中特征，未保留 ROI 对应关系，因此 Prompt 只表达候选的网络层面语义，不构成 ROI 定位监督，也不声称所有 ASD 个体具有固定异常。设计依据：Lin et al. (2025, Autism Research, DOI 10.1002/aur.70128) 与 Nature Mental Health (2025, DOI 10.1038/s44220-025-00431-5)。",
        "",
        "源码审计确认：32 条文本按 PHENO→ANAT→FUNC→Correlation 排列为 [4,2,4,1024]，经共享 4096→128→9 MLP 得到 [4,2,9] 原型；局部特征为 [871,4,4,9]，只与所属模态两个原型计算余弦，再按原始 Softmax-Max 和 global/local 融合。",
        "",
        "协议保留 StratifiedKFold(10, random_state=10, shuffle=True)、val_idx == tst_idx、ACC checkpoint、450 epoch、patience 200、原始 Adam/损失/初始化、transductive 图传播和 hard-label AUC；val=test 是原作者 Legacy 评价口径，不是独立测试集泛化估计。",
        "",
        "## 主结果",
        "",
        "| 配置 | 完整 seeds | ACC（seed均值±SD，%） | Hard-AUC（seed均值±SD，%） | SEN-ASD（%） | SPE-ASD（%） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in ALL_CONFIGS:
        summary = aggregate(all_metric_rows[config], config)
        values = []
        for metric in METRICS:
            mean = summary.get(f"{metric}_mean")
            sd = summary.get(f"{metric}_seed_sd")
            values.append(f"{pct(mean)} ± {pct(sd)}" if metric in ("acc", "hard_auc") and mean is not None else pct(mean))
        lines.append(f"| {config} | {summary['n_seeds']}/3 | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    lines.extend([
        "",
        "## 每个 seed 的十折均值",
        "",
        "| 配置 | seed | ACC（均值±折SD，%） | Hard-AUC（均值±折SD，%） | SEN-ASD（均值±折SD，%） | SPE-ASD（均值±折SD，%） |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for config in ALL_CONFIGS:
        for seed in SEEDS:
            summary = seed_summary(all_metric_rows[config], config, seed)
            cells = []
            for metric in METRICS:
                mean = summary.get(f"{metric}_mean")
                sd = summary.get(f"{metric}_fold_sd")
                cells.append("-" if mean is None else f"{mean * 100:.4f} ± {sd * 100:.4f}")
            lines.append(f"| {config} | {seed} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Correlation 训练后原型诊断",
        "",
        "C0 诊断来自实验三 D1 的冻结记录，并按原始诊断输出顺序修复 fold=1 计数 Bug；C1/C2 为本实验每个最终 checkpoint 的真实 10 折诊断。全零原型不计入非零余弦塌缩率。",
        "",
        "| 配置 | 记录数 | 全零比例 | 全部记录 |cos| | 非零记录 |cos| | 非零中 |cos|>0.98 | mean(abs(s1-s2)) | 权重均值 | w≤0.51 | w>0.55 | w>0.60 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    diagnostic_sets = {
        "C0": c0_diag,
        "C1": diagnostics,
        "C2": diagnostics,
    }
    for config in ALL_CONFIGS:
        subset = diagnostic_sets[config]
        summary = diag_summary(subset, config)
        if not summary:
            lines.append(f"| {config} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {config} | {summary['records']} | {summary['zero_fraction'] * 100:.2f}% | "
                f"{summary['prototype_abs_cosine_all']:.4f} | {summary['prototype_abs_cosine_nonzero']:.4f} | "
                f"{summary['collapse_nonzero'] * 100:.2f}% | {summary['mean_abs_delta']:.4f} | "
                f"{summary['weight_mean']:.4f} | {summary['weight_le_0_51'] * 100:.2f}% | "
                f"{summary['weight_gt_0_55'] * 100:.2f}% | {summary['weight_gt_0_60'] * 100:.2f}% |"
            )
    lines.extend([
        "",
        "## 配对比较",
        "",
        "配对使用相同 seed 和 fold；胜/平/负为描述性统计，不作显著性推断。",
    ])
    for left, right in [("C1", "C0"), ("C1", "C2"), ("C2", "C0")]:
        lines.append(f"### {left}−{right}")
        for metric in ["acc", "hard_auc"]:
            deltas = []
            for seed in SEEDS:
                a = seed_summary(all_metric_rows[left], left, seed)
                b = seed_summary(all_metric_rows[right], right, seed)
                if a["n_folds"] == 10 and b["n_folds"] == 10:
                    deltas.append(f"seed{seed} {(a[f'{metric}_mean'] - b[f'{metric}_mean']) * 100:+.2f}")
            wins, ties, losses, count = pair_counts(all_metric_rows[left], all_metric_rows[right], metric)
            lines.append(f"- {metric}: " + ", ".join(deltas) + f" 个百分点；逐折胜/平/负 {wins}/{ties}/{losses}（{count} 对）。")
        a = aggregate(all_metric_rows[left], left)
        b = aggregate(all_metric_rows[right], right)
        if a.get("sensitivity_asd_mean") is None or b.get("sensitivity_asd_mean") is None:
            lines.append("- SEN-ASD/SPE-ASD 变化：正式任务尚未全部完成。")
        else:
            lines.append(
                f"- SEN-ASD 变化：{(a['sensitivity_asd_mean'] - b['sensitivity_asd_mean']) * 100:+.2f} 个百分点；"
                f"SPE-ASD 变化：{(a['specificity_asd_mean'] - b['specificity_asd_mean']) * 100:+.2f} 个百分点。"
            )
    d0 = diag_summary(c0_diag, "C0")
    d1 = diag_summary(diagnostics, "C1")
    d2 = diag_summary(diagnostics, "C2")
    if d0 and d1 and d2:
        lines.extend([
            "",
            "Correlation 权重机制变化：",
            f"- C1 相对 C0 的权重均值变化 {(d1['weight_mean'] - d0['weight_mean']):+.4f}，mean(abs(s1-s2)) 变化 {(d1['mean_abs_delta'] - d0['mean_abs_delta']):+.4f}。",
            f"- C1 相对 C2 的权重均值变化 {(d1['weight_mean'] - d2['weight_mean']):+.4f}，mean(abs(s1-s2)) 变化 {(d1['mean_abs_delta'] - d2['mean_abs_delta']):+.4f}。",
        ])
    c1a = aggregate(all_metric_rows["C1"], "C1")
    c0a = aggregate(all_metric_rows["C0"], "C0")
    c2a = aggregate(all_metric_rows["C2"], "C2")
    c1_acc, c0_acc, c2_acc = c1a.get("acc_mean"), c0a.get("acc_mean"), c2a.get("acc_mean")
    if c1_acc is None or c2_acc is None:
        conclusion = "C1/C2 正式任务尚未全部完成，暂不写入性能结论。"
    elif c1_acc > c0_acc and c1_acc > c2_acc:
        conclusion = "C1 的总体 ACC 高于 C0 和 C2，结果方向支持 ASD 疾病相关 Correlation 语义可能提供额外价值；仍需结合 seed/逐折稳定性和诊断幅度，不能仅凭小幅均值差异确认疾病知识因果有效。"
    elif abs(c1_acc - c2_acc) <= 1e-6 and c1_acc > c0_acc:
        conclusion = "C1 与 C2 的总体 ACC 接近且均高于 C0，支持匹配句式的功能连接文本可能有作用，但没有充分证据将收益专门归因于 ASD 疾病措辞。"
    elif abs(c1_acc - c0_acc) <= 1e-6 and abs(c2_acc - c0_acc) <= 1e-6:
        conclusion = "C1、C2 与 C0 的总体 ACC 接近，当前 PLN 可能没有充分利用新增疾病语义；下一步应优先研究原型投影与语义交互机制。"
    else:
        conclusion = "C1 未超过 C0；本实验不支持在当前 PLN 和原始协议下疾病感知 Correlation Prompt 带来稳定收益，不自动生成新的 Prompt 版本。"
    lines.extend([
        "",
        "## 结论与后续建议",
        "",
        conclusion,
        "",
        "建议先在不使用 val=test 的独立外部测试或嵌套交叉验证上复核 C1/C2 的小幅差异；若继续机制研究，优先分别冻结 CLIP 文本特征与 MLP 投影，或评估共享疾病语义与模态专属语义融合，但不应据本次结果直接改动训练协议。",
        "",
        "## 任务状态",
        "",
    ])
    for config in CONFIGS:
        for seed in SEEDS:
            state, folds = task_state(config, seed)
            lines.append(f"- {config} seed={seed}: {state}, {folds}/10 folds, diagnostics={len(read_diagnostic_rows(config, seed))}/40")
    lines.extend([
        "",
        "结果文件：fold_results.csv（60 新折）；diagnostics.csv（240 新诊断行）；reference.csv（C0 30 折）；manifest.json（审计、哈希及冻结 C0 诊断）；logs/（6 任务及调度日志）。",
    ])
    return "\n".join(lines) + "\n"


def update_manifest(manifest, rows, diagnostics, status=None):
    manifest = dict(manifest)
    manifest["prompts_sha256"] = sha256(PROMPT_FILE)
    manifest["patch_sha256"] = sha256(PATCH_FILE)
    manifest["reference_sha256"] = sha256(REFERENCE_FILE)
    manifest["counts"] = {
        "new_fold_rows": len(rows),
        "new_diagnostic_rows": len(diagnostics),
        "complete_tasks": sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS),
    }
    manifest["run_status"] = status or ("COMPLETE" if manifest["counts"]["complete_tasks"] == 6 else "RUNNING")
    manifest["updated_utc"] = utc_now()
    write_json(MANIFEST_FILE, manifest)
    return manifest


def update_handoff(manifest, rows, diagnostics):
    text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# SPromptGL handoff\n"
    start, end = "<!-- DAFCP EXPERIMENT 4 START -->", "<!-- DAFCP EXPERIMENT 4 END -->"
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS)
    c0_rows = reference_rows()
    summaries = {config: aggregate(c0_rows if config == "C0" else rows, config) for config in ALL_CONFIGS}
    d0 = diag_summary(manifest.get("reference_diagnostics", []), "C0")
    d1 = diag_summary(diagnostics, "C1")
    d2 = diag_summary(diagnostics, "C2")
    block = [
        start,
        "## 实验四：DAFCP（Disease-Aware Functional Connectivity Prompting）",
        "",
        f"- 状态：{('COMPLETE' if complete == 6 else 'RUNNING')}；C1/C2 完成 {complete}/6 个任务，记录 {len(rows)}/60 新折。",
        "- 目标：只替换 Correlation 的 8 条文本，比较 ASD 疾病相关语义与匹配的 Population-level 通用语义；PHENO/ANAT/FUNC 的 P1 文本和 PLN 全部冻结。",
        "- 配置：C0 为实验一 P1 冻结基线；C1 为 ASD_DISEASE_FC；C2 为 MATCHED_GENERIC_FC。",
        "- 依据：Lin et al. 2025 Autism Research DOI 10.1002/aur.70128；Nature Mental Health 2025 DOI 10.1038/s44220-025-00431-5。Prompt 只表达候选网络层面语义，不主张固定 ROI 或普遍 ASD 异常。",
        "- 审计与哈希：见 experiment/experiment4_disease_aware_fc_prompt/manifest.json；原始源码未修改。",
        "- 结果：experiment/experiment4_disease_aware_fc_prompt/results/report.md、fold_results.csv、diagnostics.csv。",
        "",
        "任务状态：",
    ]
    for config in CONFIGS:
        for seed in SEEDS:
            state, folds = task_state(config, seed)
            block.append(f"- {config} seed={seed}: {state}，{folds}/10 folds，diagnostics={len(read_diagnostic_rows(config, seed))}/40")
    for config in ALL_CONFIGS:
        summary = summaries[config]
        if summary.get("acc_mean") is None or summary.get("hard_auc_mean") is None:
            block.append(f"- {config}：尚无完整 3-seed 汇总（完整 seeds {summary['n_seeds']}/3）。")
        else:
            block.append(f"- {config}：ACC {summary['acc_mean'] * 100:.2f}%、Hard-AUC {summary['hard_auc_mean'] * 100:.2f}%（完整 seeds {summary['n_seeds']}/3）。")
    block.extend(["", "Correlation 诊断："])
    for config, summary in [("C0", d0), ("C1", d1), ("C2", d2)]:
        if summary:
            block.append(
                f"- {config}：|cos| {summary['prototype_abs_cosine_nonzero']:.4f}（非零），w 均值 {summary['weight_mean']:.4f}，"
                f"w≤0.51 {summary['weight_le_0_51'] * 100:.2f}%，非零原型塌缩率 {summary['collapse_nonzero'] * 100:.2f}%，"
                f"全零比例 {summary['zero_fraction'] * 100:.2f}%。"
            )
    block.extend([
        "",
        "关键对照结论见 report.md；仅在 C1 同时超过 C0/C2 且 seed/逐折方向稳定时，才支持 ASD 疾病语义的额外价值。",
        "后续建议：使用独立测试或嵌套交叉验证复核；再分别冻结 CLIP 和 MLP 研究共享疾病语义与模态专属语义的融合。",
        end,
    ])
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
        for config in CONFIGS:
            for seed in SEEDS:
                if not diagnostics_valid(config, seed):
                    raise RuntimeError(f"Invalid diagnostics for {config} seed={seed}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_fold_results(rows)
    write_diagnostics(diagnostics)
    atomic_write(REPORT_FILE, report_text(rows, diagnostics, manifest))
    update_manifest(manifest, rows, diagnostics)
    update_handoff(manifest, rows, diagnostics)
    return rows, diagnostics


def parallel_tasks():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(config, seed, GPU_ASSIGNMENTS[(config, seed)]) for config in CONFIGS for seed in SEEDS]
    active = []
    with (LOG_DIR / "scheduler.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\nPARALLEL_START utc={utc_now()} tasks=6 assignments="
            + ",".join(f"{config}{seed}->GPU{gpu}" for config, seed, gpu in tasks)
            + "\n"
        )
        for config, seed, gpu in tasks:
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
    if args.worker:
        if args.config is None or args.seed is None or args.gpu is None:
            raise SystemExit("--worker requires --config, --seed and --gpu")
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    if args.config is not None:
        if args.seed is None or args.gpu is None:
            raise SystemExit("--config requires --seed and --gpu")
        raise SystemExit(run_one(args.config, args.seed, args.gpu))
    if args.refresh:
        update_outputs()
        return
    if args.all or args.pending or not any([args.refresh, args.prepare]):
        parallel_tasks()
        return


if __name__ == "__main__":
    main()




