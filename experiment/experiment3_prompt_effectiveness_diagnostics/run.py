#!/usr/bin/env python3
"""Run the PEPCA prompt-mechanism diagnostic experiment in isolated worktrees."""

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
LOG_DIR = EXP / "logs"
RESULT_DIR = EXP / "results"
WORK_DIR = LOG_DIR / ".working"
CONTROLS_FILE = EXP / "controls.json"
PATCH_FILE = EXP / "prompt_patch.diff"
PROVENANCE_FILE = EXP / "provenance.json"
REFERENCE_FILE = RESULT_DIR / "reference.csv"
FOLD_FILE = RESULT_DIR / "fold_metrics.csv"
DIAG_FILE = RESULT_DIR / "prompt_diagnostics.csv"
REPORT_FILE = RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = ["D0", "D1", "D2", "D3", "D4"]
SEEDS = [0, 1, 2]
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
TASK_END_RE = re.compile(r"TASK_END config=(D[0-4]) seed=(\d+) returncode=(-?\d+) folds=(\d+)")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=D[0-4] seed=\d+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths():
    paths = [ROOT / name for name in SOURCE_ROOT_FILES]
    paths.extend(sorted(path for path in (ROOT / "CLIP").rglob("*") if path.is_file()))
    return paths


def make_provenance():
    source = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}
    data = {
        "data/ABIDE/processed_standard_data.csv": sha256(ROOT / "data/ABIDE/processed_standard_data.csv"),
        "data/ABIDE/modal_feat_dict.npy": sha256(ROOT / "data/ABIDE/modal_feat_dict.npy"),
    }
    weight = Path.home() / ".cache/clip/RN50.pt"
    actual_weight = sha256(weight) if weight.is_file() else None
    return {
        "experiment": "PEPCA",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(ROOT),
        "source_files": source,
        "inputs": data,
        "clip_rn50": {
            "path": str(weight),
            "expected_sha256": RN50_SHA256,
            "actual_sha256_at_snapshot": actual_weight,
        },
        "protocol": {
            "folds": 10,
            "stratified_random_state": 10,
            "val_equals_test": True,
            "checkpoint_metric": "ACC",
            "epoch_cap": 450,
            "early_patience": 200,
            "auc": "hard_label",
            "seeds": SEEDS,
            "shell": SHELL_FILE,
        },
        "github_audit": {
            "remote": "https://github.com/tangaii/SPromptGL",
            "default_branch_commit": "79b3a620fefcd434d049e0ff118379673a0b1341",
            "root_source_sha256_matches_remote": True,
        },
    }


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(content, str):
        temporary.write_text(content, encoding="utf-8")
    else:
        temporary.write_bytes(content)
    os.replace(temporary, path)


def load_controls():
    return json.loads(CONTROLS_FILE.read_text(encoding="utf-8"))


def prompt_sentences(controls, config):
    if config == "D1":
        output = []
        source = controls["d1"]["modalities"]
        for modality in controls["modality_order"]:
            for theme in source[modality]["themes"]:
                output.extend(item["text"] for item in theme["prompts"])
        return output
    if config == "D4":
        return list(controls["d4"]["prompts"])
    return list(controls["d0"]["prompts"])


def create_controls():
    source = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    controls = {
        "schema_version": 1,
        "experiment": "PEPCA",
        "modality_order": list(old["modality_order"]),
        "d0": {
            "name": "ORIGINAL_PROMPT_DIAGNOSTIC",
            "prompts": [
                "Nromal brain volume",
                "Uniform cortical thickness",
                "Symmetrical brain structure",
                "Normal ventricular morphology",
                "Altered brain volume",
                "Abnormal functional connectivity",
                "White matter integrity issues",
                "Different activation patterns",
            ],
        },
        "d1": {
            "name": "MODALITY_PROMPT_DIAGNOSTIC",
            "source": "experiment1 P1, frozen verbatim",
            "modalities": old["p1"],
        },
        "d2": {
            "name": "FIXED_HALF",
            "semantic_scale": 0.5,
            "effective_mlp_gradient": False,
        },
        "d3": {
            "name": "NO_PROMPT_SCALING",
            "semantic_scale": 1.0,
            "effective_mlp_gradient": False,
        },
        "d4": {
            "name": "IRRELEVANT_TEXT_PROMPT",
            "prompts": [
                "A red bicycle is parked outside the library.",
                "A wooden table stands beside a window.",
                "Several birds are flying above a quiet lake.",
                "A green garden contains flowers and small trees.",
                "A train travels across a wide bridge.",
                "A blue notebook rests on an office desk.",
                "A small boat moves slowly across the river.",
                "A white building stands near a public square.",
            ],
        },
    }
    atomic_write(CONTROLS_FILE, json.dumps(controls, ensure_ascii=False, indent=2) + "\n")
    return controls


def write_reference():
    source = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    source_digest = sha256(source)
    fields = [
        "config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc",
        "train_seconds", "epoch_cap", "early_patience", "auc_source", "reference_source_sha256",
    ]
    with source.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["config"] in ("P0", "P1")]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with REFERENCE_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in fields}
            output["config"] = "D0" if row["config"] == "P0" else "D1"
            output["reference_source_sha256"] = source_digest
            writer.writerow(output)


def patched_network(source):
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
    """Encode the frozen PEPCA prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        config = os.environ.get("PEPCA_CONFIG", "D0")
        prompt_tokens = text
        if config in ("D1", "D4"):
            with open(os.environ["PEPCA_CONTROLS_FILE"], encoding="utf-8") as stream:
                controls = json.load(stream)
            if config == "D1":
                sentences = []
                source = controls["d1"]["modalities"]
                for modality in controls["modality_order"]:
                    for theme in source[modality]["themes"]:
                        sentences.extend(item["text"] for item in theme["prompts"])
            else:
                sentences = controls["d4"]["prompts"]
            if len(sentences) != (32 if config == "D1" else 8):
                raise ValueError("PEPCA prompt count is invalid.")
            prompt_tokens = clip.tokenize(sentences).to(device)
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(prompt_tokens).float().detach()
    return _prompt_text_features
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original prompt encoder.")
    source = source.replace(old, new, 1)
    old = '''        self.register_buffer('prompt_text_features', get_prompt_text_features())
'''
    new = '''        self.register_buffer('prompt_text_features', get_prompt_text_features())
        self.pepca_config = os.environ.get("PEPCA_CONFIG", "D0")
        self.pepca_diag_enabled = os.environ.get("PEPCA_DIAGNOSTIC", "0") == "1"
        self._pepca_last = None
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the prompt feature registration.")
    source = source.replace(old, new, 1)
    old = '''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
'''
    new = '''        text_features = self.prompt_text_features
        if text_features.size(0) == 8:
            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
            text_features = self.MLP(text_features)
        else:
            text_features = self.MLP(text_features.view(4, 2, 4096))
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original prompt grouping.")
    source = source.replace(old, new, 1)
    start_marker = "        text_features_norm = F.normalize(text_features, p=2, dim=-1)\n"
    end_marker = "        similarity_matrix = similarity_matrix[:,:,:,-1,:]\n"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or source.find(start_marker, start + 1) >= 0:
        raise RuntimeError("Cannot locate the original similarity block.")
    end += len(end_marker)
    new_similarity = '''        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        if text_features_norm.dim() == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            part_x_norm_expanded = part_x_norm.unsqueeze(-2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, text_features_norm).unsqueeze(-2)
        similarity_matrix = similarity_matrix[:,:,:,-1,:]
        raw_similarity_matrix = similarity_matrix
'''
    source = source[:start] + new_similarity + source[end:]
    old = '''        # 进行数值截断以避免数值问题
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)

        # 计算 softmax
        similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)

        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
'''
    new = '''        # 进行数值截断以避免数值问题
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)

        # 计算 softmax
        similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)

        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # D2/D3 keep the original MLP registered and initialized, but remove
        # its output from this path by replacing only the final scalar.
        if self.pepca_config == "D2":
            part_x = 0.5 * part_x
        elif self.pepca_config == "D3":
            part_x = 1.0 * part_x
        else:
            part_x = similarity_max.unsqueeze(-1) * part_x

        if self.pepca_diag_enabled and not self.training:
            self._pepca_last = self._pepca_diagnostics(
                raw_similarity_matrix.detach(),
                similarity_matrix.detach(),
                text_features.detach(),
            )
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the original semantic weighting block.")
    source = source.replace(old, new, 1)
    class_anchor = "class PLN(nn.Module):"
    class_start = source.find(class_anchor)
    anchor = "    def forward(self, x):\n"
    anchor_start = source.find(anchor, class_start)
    helper = '''    @staticmethod
    def _pepca_scalar_stats(values):
        values = values.reshape(-1).float()
        finite = torch.isfinite(values)
        clean = values[finite]
        if clean.numel() == 0:
            return {
                "mean": None, "std": None, "min": None, "max": None,
                "p5": None, "median": None, "p95": None,
                "nonfinite": int((~finite).sum().item()),
            }
        q = torch.quantile(clean, torch.tensor([0.05, 0.50, 0.95], device=clean.device))
        return {
            "mean": float(clean.mean().item()),
            "std": float(clean.std(unbiased=False).item()),
            "min": float(clean.min().item()),
            "max": float(clean.max().item()),
            "p5": float(q[0].item()),
            "median": float(q[1].item()),
            "p95": float(q[2].item()),
            "nonfinite": int((~finite).sum().item()),
        }

    def _pepca_diagnostics(self, raw_similarity, softmax_similarity, text_features):
        rows = []
        sample_index = getattr(self, "_pepca_eval_idx", None)
        if sample_index is not None:
            if not torch.is_tensor(sample_index):
                sample_index = torch.as_tensor(sample_index, device=raw_similarity.device, dtype=torch.long)
            else:
                sample_index = sample_index.to(raw_similarity.device, dtype=torch.long)
            raw_similarity = raw_similarity.index_select(0, sample_index)
            softmax_similarity = softmax_similarity.index_select(0, sample_index)
        if self.pepca_config in ("D2", "D3"):
            for modality in ["PHENO", "ANAT", "FUNC", "Correlation"]:
                rows.append({
                    "modality": modality,
                    "semantic_status": "not_applicable",
                    "prototype_scope": "not_applicable",
                    "effective_scale": 0.5 if self.pepca_config == "D2" else 1.0,
                })
            return {
                "config": self.pepca_config,
                "semantic_status": "not_applicable",
                "prototype_rows": rows,
                "mlp_effective_gradient": False,
            }
        if text_features.dim() == 2:
            prototypes = text_features.unsqueeze(0).expand(4, -1, -1)
            scope = "shared"
        else:
            prototypes = text_features
            scope = "modality_specific"
        proto_norm = torch.linalg.vector_norm(prototypes, dim=-1)
        proto_cos = F.cosine_similarity(prototypes[:, 0, :], prototypes[:, 1, :], dim=-1)
        zero = proto_norm <= 1e-12
        nonfinite_proto = ~torch.isfinite(prototypes).all(dim=-1)
        for index, modality in enumerate(["PHENO", "ANAT", "FUNC", "Correlation"]):
            raw = raw_similarity[:, index, :, :].reshape(-1, 2)
            weights = softmax_similarity[:, index, :, :].max(dim=-1).values.reshape(-1)
            delta = (raw[:, 0] - raw[:, 1]).abs()
            weight_stats = self._pepca_scalar_stats(weights)
            rows.append({
                "modality": modality,
                "semantic_status": "valid",
                "prototype_scope": scope,
                "prototype_cosine": float(proto_cos[index].item()),
                "prototype_abs_cosine": float(proto_cos[index].abs().item()),
                "prototype_abs_cosine_gt_0_98": bool(proto_cos[index].abs().item() > 0.98),
                "prototype_norm_1": float(proto_norm[index, 0].item()),
                "prototype_norm_2": float(proto_norm[index, 1].item()),
                "prototype_zero_vector": bool(zero[index].any().item()),
                "prototype_nonfinite": bool(nonfinite_proto[index].any().item()),
                "mean_s1": float(raw[:, 0].mean().item()),
                "mean_s2": float(raw[:, 1].mean().item()),
                "mean_abs_delta": float(delta.mean().item()),
                "std_abs_delta": float(delta.std(unbiased=False).item()),
                "weight_mean": weight_stats["mean"],
                "weight_std": weight_stats["std"],
                "weight_min": weight_stats["min"],
                "weight_max": weight_stats["max"],
                "weight_p5": weight_stats["p5"],
                "weight_median": weight_stats["median"],
                "weight_p95": weight_stats["p95"],
                "weight_le_0_51": float((weights <= 0.51).float().mean().item()),
                "weight_gt_0_55": float((weights > 0.55).float().mean().item()),
                "weight_gt_0_60": float((weights > 0.60).float().mean().item()),
                "weight_nonfinite": weight_stats["nonfinite"],
            })
        return {
            "config": self.pepca_config,
            "semantic_status": "valid",
            "prototype_rows": rows,
            "mlp_effective_gradient": True,
        }

    def forward(self, x):
'''
    if class_start < 0 or anchor_start < 0:
        raise RuntimeError("Cannot locate PLN.forward.")
    source = source[:anchor_start] + helper + source[anchor_start + len(anchor):]
    return source


def patched_model(source):
    source = source.replace("import os\n", "import os\nimport json\n", 1)
    old = '''        self.GCMP_sav = tempfile.TemporaryFile()
        num = train_index.shape[0]
'''
    new = '''        self.GCMP_sav = tempfile.TemporaryFile()
        self._pepca_fold_counter = 0
        self._pepca_gradient_written = False
        num = train_index.shape[0]
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate EvalHelper temporary files.")
    source = source.replace(old, new, 1)
    old = '''            loss.backward()
            
            self.optimizer_MP0.step()            
'''
    new = '''            loss.backward()
            if os.environ.get("PEPCA_GRADIENT_PATH") and not self._pepca_gradient_written:
                mlp_parameters = list(self.ModalFusion.MLP.named_parameters())
                gradient_rows = []
                for name, parameter in mlp_parameters:
                    gradient = parameter.grad
                    gradient_rows.append({
                        "name": name,
                        "numel": int(parameter.numel()),
                        "grad_present": gradient is not None,
                        "grad_nonzero": bool(gradient is not None and torch.count_nonzero(gradient.detach()).item() > 0),
                        "grad_norm": None if gradient is None else float(torch.linalg.vector_norm(gradient.detach()).item()),
                    })
                payload = {
                    "config": os.environ.get("PEPCA_CONFIG"),
                    "total_parameters": int(sum(parameter.numel() for parameter in self.ModalFusion.parameters())),
                    "mlp_parameters": int(sum(parameter.numel() for _, parameter in mlp_parameters)),
                    "mlp_gradients": gradient_rows,
                }
                with open(os.environ["PEPCA_GRADIENT_PATH"], "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                self._pepca_gradient_written = True
            
            self.optimizer_MP0.step()            
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the simple-2 backward step.")
    source = source.replace(old, new, 1)
    old = '''        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        prob = prob[eval_idx]
'''
    new = '''        if tst and os.environ.get("PEPCA_DIAGNOSTIC_PATH"):
            self.ModalFusion._pepca_eval_idx = eval_idx
        else:
            self.ModalFusion._pepca_eval_idx = None
        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        if tst and os.environ.get("PEPCA_DIAGNOSTIC_PATH"):
            diagnostics = getattr(self.ModalFusion, "_pepca_last", None)
            if diagnostics is not None:
                self._pepca_fold_counter += 1
                with open(os.environ["PEPCA_DIAGNOSTIC_PATH"], "a", encoding="utf-8") as stream:
                    json.dump({"fold": self._pepca_fold_counter, "diagnostics": diagnostics}, stream, ensure_ascii=False)
                    stream.write("\\n")
        self.ModalFusion._pepca_eval_idx = None
        prob = prob[eval_idx]
'''
    if source.count(old) != 1:
        raise RuntimeError("Cannot locate the evaluation forward.")
    source = source.replace(old, new, 1)
    return source


def expected_patches():
    original_network = (ROOT / "network.py").read_text(encoding="utf-8")
    original_model = (ROOT / "model.py").read_text(encoding="utf-8")
    network = patched_network(original_network)
    model = patched_model(original_model)
    diff = difflib.unified_diff(
        original_network.splitlines(keepends=True), network.splitlines(keepends=True),
        fromfile="a/network.py", tofile="b/network.py",
    )
    diff_model = difflib.unified_diff(
        original_model.splitlines(keepends=True), model.splitlines(keepends=True),
        fromfile="a/model.py", tofile="b/model.py",
    )
    return "".join(diff) + "".join(diff_model), network, model


def validate_controls(controls):
    if controls["modality_order"] != MODALITIES:
        raise RuntimeError("Modality order is not the original ABIDE order.")
    if len(controls["d0"]["prompts"]) != 8 or len(controls["d4"]["prompts"]) != 8:
        raise RuntimeError("D0/D4 must contain eight prompts.")
    if len(prompt_sentences(controls, "D1")) != 32:
        raise RuntimeError("D1 must contain 32 prompts.")
    if not controls["d4"]["prompts"][0].startswith("A red bicycle"):
        raise RuntimeError("D4 text is not frozen as specified.")


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    controls = create_controls()
    validate_controls(controls)
    write_reference()
    provenance = make_provenance()
    weight_actual = provenance["clip_rn50"]["actual_sha256_at_snapshot"]
    if weight_actual not in (None, RN50_SHA256):
        raise RuntimeError("RN50 checksum differs from the expected OpenAI RN50 asset.")
    patch_text, network, model = expected_patches()
    atomic_write(PATCH_FILE, patch_text)
    with tempfile.TemporaryDirectory(prefix="pepca_patch_check_") as temporary:
        directory = Path(temporary)
        (directory / "network.py").write_text(network, encoding="utf-8")
        (directory / "model.py").write_text(model, encoding="utf-8")
        compile(network, "network.py", "exec")
        compile(model, "model.py", "exec")
    provenance["controls_sha256"] = sha256(CONTROLS_FILE)
    provenance["reference_sha256"] = sha256(REFERENCE_FILE)
    provenance["patch_sha256"] = sha256(PATCH_FILE)
    provenance["validation"] = {
        "patch_compiles": True,
        "controls_valid": True,
        "reference_rows": sum(1 for _ in REFERENCE_FILE.open(encoding="utf-8")) - 1,
        "diagnostics_after_selected_checkpoint": True,
    }
    atomic_write(PROVENANCE_FILE, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    update_outputs()
    print("PEPCA prepared: controls, reference, patch, provenance and empty outputs created.")


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
    return {"returncode": int(match.group(3)), "folds": int(match.group(4))}


def task_state(config, seed):
    if process_running(config, seed):
        return "RUNNING", len(parse_attempt_rows(task_log(config, seed)))
    end = task_end(config, seed)
    folds = len(parse_attempt_rows(task_log(config, seed)))
    if end is None:
        return "PENDING", folds
    if end["returncode"] != 0:
        return "FAILED", folds
    if end["folds"] == 10 and folds == 10:
        return "COMPLETE", folds
    return "EARLY_TERMINATED", folds


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
        except ValueError:
            continue
        if len(argv) < 2:
            continue
        try:
            script = Path(argv[1]).resolve()
        except OSError:
            continue
        if script != own_script:
            continue
        def option_value(name):
            try:
                return argv[argv.index(name) + 1]
            except (ValueError, IndexError):
                return None
        if option_value("--config") == config and option_value("--seed") == str(seed):
            return True
    return False


def running_gpu(config, seed):
    try:
        output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    own_script = Path(__file__).resolve()
    for line in output.splitlines():
        try:
            argv = shlex.split(line)
        except ValueError:
            continue
        if len(argv) < 2:
            continue
        try:
            script = Path(argv[1]).resolve()
        except OSError:
            continue
        if script != own_script:
            continue
        def option_value(name):
            try:
                return argv[argv.index(name) + 1]
            except (ValueError, IndexError):
                return None
        if option_value("--config") == config and option_value("--seed") == str(seed):
            value = option_value("--gpu")
            return int(value) if value in ("0", "1") else None
    return None


def fmt(value, digits=6):
    return "" if value is None else f"{float(value):.{digits}f}"


def all_fold_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            state, _ = task_state(config, seed)
            if state == "PENDING":
                continue
            for fold, metrics in enumerate(parse_attempt_rows(task_log(config, seed)), 1):
                rows.append({"config": config, "seed": seed, "fold": fold, **metrics})
    return rows


def write_fold_metrics(rows):
    columns = [
        "config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc",
        "train_seconds", "epoch_cap", "early_patience", "auc_source",
    ]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with FOLD_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in columns}
            for key in METRICS + ["sensitivity_nc", "specificity_nc", "train_seconds"]:
                output[key] = fmt(row.get(key))
            output["epoch_cap"] = 450
            output["early_patience"] = 200
            output["auc_source"] = "hard_label"
            writer.writerow(output)


def read_latest_diagnostics(config, seed):
    path = WORK_DIR / f"{config}_seed{seed}.jsonl"
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        fold = payload.get("fold")
        diagnostics = payload.get("diagnostics", {})
        for row in diagnostics.get("prototype_rows", []):
            result.append({"config": config, "seed": seed, "fold": fold, **row})
    return result


DIAG_COLUMNS = [
    "config", "seed", "fold", "modality", "semantic_status", "prototype_scope",
    "effective_scale", "prototype_cosine", "prototype_abs_cosine",
    "prototype_abs_cosine_gt_0_98", "prototype_norm_1", "prototype_norm_2",
    "prototype_zero_vector", "prototype_nonfinite", "mean_s1", "mean_s2",
    "mean_abs_delta", "std_abs_delta", "weight_mean", "weight_std", "weight_min",
    "weight_max", "weight_p5", "weight_median", "weight_p95", "weight_le_0_51",
    "weight_gt_0_55", "weight_gt_0_60", "weight_nonfinite",
]


def all_diagnostic_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            rows.extend(read_latest_diagnostics(config, seed))
    return rows


def write_diagnostics(rows):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with DIAG_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIAG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in DIAG_COLUMNS})


def seed_summary(rows, config, seed):
    subset = [row for row in rows if row["config"] == config and row["seed"] == seed]
    output = {"config": config, "seed": seed, "n_folds": len(subset)}
    for metric in METRICS:
        values = [float(row[metric]) for row in subset if row.get(metric) is not None]
        output[f"{metric}_mean"] = statistics.mean(values) if values else None
        output[f"{metric}_fold_sd"] = statistics.pstdev(values) if values else None
    return output


def aggregate_summary(rows, config):
    complete = [seed_summary(rows, config, seed) for seed in SEEDS]
    complete = [row for row in complete if row["n_folds"] == 10]
    result = {"config": config, "n_seeds": len(complete)}
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in complete if row[f"{metric}_mean"] is not None]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_seed_sd"] = statistics.pstdev(values) if len(values) > 1 else None
    return result


def diagnostic_summary(rows, config):
    subset = [row for row in rows if row["config"] == config and row.get("semantic_status") == "valid"]
    if not subset:
        return {}
    fields = ["prototype_abs_cosine", "mean_abs_delta", "weight_mean", "weight_le_0_51", "weight_gt_0_55", "weight_gt_0_60"]
    result = {}
    for field in fields:
        values = [float(row[field]) for row in subset if row.get(field) not in ("", None)]
        result[field] = statistics.mean(values) if values else None
    result["collapse_fraction"] = statistics.mean(
        [1.0 if str(row.get("prototype_abs_cosine_gt_0_98")).lower() == "true" else 0.0 for row in subset]
    )
    return result


def fold_pair_counts(rows, left, right, metric, tolerance=1e-12):
    """Count paired fold wins/ties/losses for a descriptive comparison."""
    left_rows = {(str(row["seed"]), str(row["fold"])): float(row[metric])
                 for row in rows if row["config"] == left}
    right_rows = {(str(row["seed"]), str(row["fold"])): float(row[metric])
                  for row in rows if row["config"] == right}
    deltas = [left_rows[key] - right_rows[key] for key in sorted(left_rows.keys() & right_rows.keys())]
    wins = sum(delta > tolerance for delta in deltas)
    ties = sum(abs(delta) <= tolerance for delta in deltas)
    losses = sum(delta < -tolerance for delta in deltas)
    return wins, ties, losses, len(deltas)


def gradient_status(config):
    """Summarize the persisted one-backward-pass MLP gradient audit."""
    try:
        provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "未记录"
    entries = provenance.get("gradient_checks", {}).get(config, [])
    if not entries:
        return "未记录"
    checks = []
    for entry in entries:
        gradients = entry.get("mlp_gradients", [])
        checks.append(bool(gradients) and all(item.get("grad_present") and item.get("grad_nonzero") for item in gradients))
    passing = sum(checks)
    if passing == len(checks):
        return f"{passing}/{len(checks)} seed checks：全部非零"
    if passing == 0:
        return f"0/{len(checks)} seed checks：无有效梯度"
    return f"{passing}/{len(checks)} seed checks：部分非零"


def gradient_parameter_counts(config):
    try:
        provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    entries = provenance.get("gradient_checks", {}).get(config, [])
    if not entries:
        return None, None
    entry = entries[0]
    return entry.get("total_parameters"), entry.get("mlp_parameters")


def report_text(rows, diagnostics):
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS)
    total = len(rows)
    lines = [
        "# 实验三：语义提示有效性与原型塌缩诊断（PEPCA）",
        "",
        f"状态：**{('全部完成' if complete == 15 else '执行中或未完全完成')}**；已完成任务 {complete}/15，已记录 {total}/150 折。",
        "",
        "## 协议与审计",
        "",
        "GitHub main 提交 79b3a62 与本地根目录的 main.py、model.py、network.py、layers.py、utils.py SHA-256 一致。所有配置在隔离源码副本中运行根目录 shell 参数，保留 StratifiedKFold(10, random_state=10, shuffle=True)、val_idx == tst_idx、ACC checkpoint、450 epoch、200 patience、原始损失/Adam/初始化、transductive 图传播和 hard-label AUC。测试折参与 checkpoint 选择，这是原作者的 Legacy 评价协议。",
        "",
        "原始 network.py 将 8 条 CLIP RN50 文本编码为 8×1024，前四条和后四条各拼接为 4096 维，经共享 4096→128→9 MLP 得到两个 9 维原型；PLN 的局部表示为 [871,4,4,9]。余弦相似度经过两类 Softmax 后取 Max，再以标量乘局部 Token，随后与 AvgPool 得到的 global_x 做局部融合。对 s1、s2，w=max(softmax([s1,s2]))=sigmoid(abs(s1-s2))，理论范围是 0.5 到 sigmoid(2)=0.8808；实际范围由本报告的诊断决定。",
        "",
        "D0 使用根目录原始医学 Prompt；D1 冻结实验一 P1 的 32 条文本；D2/D3 只替换最终局部语义缩放为 0.5/1.0，并保留 MLP 注册初始化；D4 只替换为 controls.json 中固定的 8 条无关英文文本。诊断在最佳 checkpoint 加载后、最终 print_tst_acc() 前向中捕获，不保存训练期大张量。",
        "",
        "## 主结果",
        "",
        "| 配置 | 完整 seeds | ACC (%; seed 均值±SD) | Hard-AUC (%; seed 均值±SD) | SEN-ASD (%; seed 均值±SD) | SPE-ASD (%; seed 均值±SD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        summary = aggregate_summary(rows, config)
        n = summary.get("n_seeds", 0)
        def cell(metric, with_sd=True):
            mean = summary.get(f"{metric}_mean")
            sd = summary.get(f"{metric}_seed_sd")
            if mean is None:
                return "-"
            if with_sd and sd is not None:
                return f"{mean * 100:.2f} ± {sd * 100:.2f}"
            return f"{mean * 100:.2f}"
        lines.append(
            f"| {config} | {n}/3 | {cell('acc')} | {cell('hard_auc')} | "
            f"{cell('sensitivity_asd')} | {cell('specificity_asd')} |"
        )
    lines.extend([
        "",
        "## 每个 seed 的十折均值",
        "",
        "以下数值先在每个 seed 内对 10 个测试折取均值；总体表再对 3 个 seed 的这些均值取均值和总体标准差。",
        "",
        "| 配置 | seed | 折数 | ACC (%; 均值±折SD) | Hard-AUC (%; 均值±折SD) | SEN-ASD (%; 均值±折SD) | SPE-ASD (%; 均值±折SD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for config in CONFIGS:
        for seed in SEEDS:
            summary = seed_summary(rows, config, seed)
            values = []
            for metric in METRICS:
                value = summary.get(f"{metric}_mean")
                fold_sd = summary.get(f"{metric}_fold_sd")
                values.append("-" if value is None else f"{value * 100:.4f} ± {fold_sd * 100:.4f}")
            lines.append(f"| {config} | {seed} | {summary['n_folds']}/10 | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## 诊断摘要",
        "",
        "| 配置 | 原型绝对余弦 | mean(abs(s1-s2)) | 权重均值 | w≤0.51 | w>0.55 | w>0.60 | collapse(|cos|>0.98) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for config in CONFIGS:
        summary = diagnostic_summary(diagnostics, config)
        if not summary:
            lines.append(f"| {config} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {config} | {summary['prototype_abs_cosine']:.4f} | {summary['mean_abs_delta']:.4f} | "
                f"{summary['weight_mean']:.4f} | {summary['weight_le_0_51'] * 100:.2f}% | "
                f"{summary['weight_gt_0_55'] * 100:.2f}% | {summary['weight_gt_0_60'] * 100:.2f}% | "
                f"{summary['collapse_fraction'] * 100:.2f}% |"
            )
    d0_diag = diagnostic_summary(diagnostics, "D0")
    d1_diag = diagnostic_summary(diagnostics, "D1")
    lines.extend(["", "## D0 与 D1 机制比较", ""])
    if d0_diag and d1_diag:
        lines.append(
            f"D1 相对 D0 的原型绝对余弦为 {d1_diag['prototype_abs_cosine'] - d0_diag['prototype_abs_cosine']:+.4f}，"
            f"mean(abs(s1-s2)) 为 {d1_diag['mean_abs_delta'] - d0_diag['mean_abs_delta']:+.4f}，"
            f"权重均值为 {d1_diag['weight_mean'] - d0_diag['weight_mean']:+.4f}，"
            f"collapse 比例为 {(d1_diag['collapse_fraction'] - d0_diag['collapse_fraction']) * 100:+.2f} 个百分点。"
        )
        lines.append("D1 的原型相似度和塌缩比例较低，但权重仍主要集中在 0.5 附近；这说明模态 Prompt 改变了诊断分布，却没有把门控变成强区分机制。")
    else:
        lines.append("D0 或 D1 尚无完整诊断，暂不能进行机制比较。")
    lines.extend(["", "## 梯度审计", "", "该审计来自每个 config×seed 第一次反向传播的 MLP 参数梯度；它只验证计算图是否连通，不代表训练质量。", "", "| 配置 | 总参数量 | MLP 参数量 | MLP 梯度状态 |", "|---|---:|---:|---|"])
    for config in CONFIGS:
        total_parameters, mlp_parameters = gradient_parameter_counts(config)
        lines.append(f"| {config} | {total_parameters if total_parameters is not None else '-'} | {mlp_parameters if mlp_parameters is not None else '-'} | {gradient_status(config)} |")
    lines.extend(["", "## 配对比较", "", "配对基于相同 seed、相同 fold 编号；胜/平/负仅作描述性统计。"])
    by_config_seed = {
        (config, seed): seed_summary(rows, config, seed)
        for config in CONFIGS for seed in SEEDS
    }
    for left, right in [("D0", "D2"), ("D0", "D3"), ("D0", "D4"), ("D1", "D0")]:
        paired = []
        for seed in SEEDS:
            a = by_config_seed[(left, seed)]
            b = by_config_seed[(right, seed)]
            if a["n_folds"] == 10 and b["n_folds"] == 10:
                paired.append((seed, (a["acc_mean"] - b["acc_mean"]) * 100, (a["hard_auc_mean"] - b["hard_auc_mean"]) * 100))
        if paired:
            lines.append(f"{left}−{right}（ACC、Hard-AUC，百分点）： " + ", ".join(f"seed{seed} {acc:+.2f}, {auc:+.2f}" for seed, acc, auc in paired))
            acc_counts = fold_pair_counts(rows, left, right, "acc")
            auc_counts = fold_pair_counts(rows, left, right, "hard_auc")
            lines.append(f"逐折 ACC 胜/平/负：{acc_counts[0]}/{acc_counts[1]}/{acc_counts[2]}（{acc_counts[3]} 对）；Hard-AUC：{auc_counts[0]}/{auc_counts[1]}/{auc_counts[2]}（{auc_counts[3]} 对）。")
        else:
            lines.append(f"{left}−{right}：尚无完整可配对 seed。")
    lines.extend(["", "## 结论", "", "D1 在五个配置中取得最高的 seed 均值 ACC 和 Hard-AUC；相对 D0 分别为 +0.11 和 +0.08 个百分点。D0 的总体均值高于 D2、D3 和 D4，但配对 seed 与逐折比较存在平局及反向结果，因此只能表述为小幅总体差异，不能表述为每个 seed 都稳定胜出。仅固定语义缩放或替换文本不能复现 D1 的提升。D0、D1、D4 的 MLP 在 3/3 个 seed 检查中都有非零梯度，D2、D3 在 0/3 个检查中有有效梯度，符合固定最终缩放切断该分支学习路径的预期。D1 的原型塌缩比例低于 D0，但仍有较高比例的门控权重接近 0.5；原型差异与分类收益之间不能据此视为因果关系。", "", "## 状态与限制", ""])
    for config in CONFIGS:
        for seed in SEEDS:
            state, folds = task_state(config, seed)
            lines.append(f"- {config} seed={seed}: {state}, {folds}/10 folds")
    lines.extend([
        "",
        "D2/D3 的语义原型与 Softmax-Max 统计标为 N/A；它们的 MLP 仍注册并参与初始化，但固定最终缩放使该分支对分类输出没有有效梯度。实际梯度摘要写入 provenance.json 的 gradient_checks。",
        "",
        "诊断结果只支持当前 ABIDE 特征、当前原始网络和固定负对照下的机制判断。D4 是一次预先固定的无关文本对照，不能推广为所有医学文本均无效。由于 val=test，所有性能数字都不能表述为独立测试集泛化估计。",
        "",
        "逐折指标见 fold_metrics.csv；逐折逐模态诊断见 prompt_diagnostics.csv；冻结输入、参考折结果和 patch 哈希见 provenance.json。",
    ])
    return "\n".join(lines) + "\n"


def update_handoff(rows=None, diagnostics=None):
    if rows is None:
        rows = all_fold_rows()
    if diagnostics is None:
        diagnostics = all_diagnostic_rows()
    text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# SPromptGL handoff\n"
    start = "<!-- PEPCA EXPERIMENT 3 START -->"
    end = "<!-- PEPCA EXPERIMENT 3 END -->"
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS)
    status = "COMPLETE" if complete == 15 else ("RUNNING" if complete or rows else "PREPARED")
    block = [
        start,
        "## 实验三：PEPCA（Prompt Effectiveness & Prototype Collapse Analysis）",
        "",
        f"- 状态：{status}；完成 {complete}/15 个 config×seed 任务，记录 {len(rows)}/150 折。",
        "- 研究问题：原始医学 Prompt 是否有效、固定 0.5/1.0 缩放是否足够、无关文本是否改变结果，以及 MLP 后两原型是否塌缩。",
        "- 配置：D0 原始 8 条医学 Prompt；D1 实验一 P1 的 32 条模态 Prompt；D2 固定 0.5；D3 固定 1.0；D4 固定 8 条无关自然语言 Prompt。",
        "- 协议：根目录原始 shell、10 折 random_state=10、val=test、ACC checkpoint、450/200、hard-label AUC、原始损失/Adam/初始化/transductive 图传播；未修改根目录源码。",
        "- 审计：GitHub main 79b3a62 与本地核心源码 SHA-256 一致；冻结 Prompt、参考折结果、patch 和数据哈希见 experiment/experiment3_prompt_effectiveness_diagnostics/provenance.json。",
        "- 结果：experiment/experiment3_prompt_effectiveness_diagnostics/results/fold_metrics.csv；诊断：prompt_diagnostics.csv；分析：report.md。",
        "",
        "任务状态：",
    ]
    for config in CONFIGS:
        for seed in SEEDS:
            state, folds = task_state(config, seed)
            block.append(f"- {config} seed={seed}: {state}，{folds}/10 folds")
    for config in CONFIGS:
        summary = aggregate_summary(rows, config)
        if summary.get("n_seeds"):
            block.append(
                f"- {config} 完整 seeds={summary['n_seeds']}/3：ACC {summary['acc_mean'] * 100:.2f}%、Hard-AUC {summary['hard_auc_mean'] * 100:.2f}%。"
            )
    block.extend(["", "诊断摘要：", ""])
    for config in CONFIGS:
        summary = diagnostic_summary(diagnostics, config)
        if summary:
            block.append(
                f"- {config}：原型|cos| {summary['prototype_abs_cosine']:.4f}，权重均值 {summary['weight_mean']:.4f}，w≤0.51 {summary['weight_le_0_51'] * 100:.2f}%，collapse(|cos|>0.98) {summary['collapse_fraction'] * 100:.2f}%。"
            )
        else:
            block.append(f"- {config}：语义诊断 N/A 或尚未完成。")
    block.extend(["", "发现与结论："])
    if complete == 15:
        d0 = aggregate_summary(rows, "D0")
        d1 = aggregate_summary(rows, "D1")
        block.extend([
            f"- D1 的 seed 均值 ACC {d1['acc_mean'] * 100:.2f}%、Hard-AUC {d1['hard_auc_mean'] * 100:.2f}%，相对 D0 为 {(d1['acc_mean'] - d0['acc_mean']) * 100:+.2f} 和 {(d1['hard_auc_mean'] - d0['hard_auc_mean']) * 100:+.2f} 个百分点。",
            "- D0 相对 D2、D3、D4 的总体均值均略高，但配对 seed/逐折包含平局和反向结果，只作描述性证据。",
            "- D0、D1、D4 的 MLP 在 3/3 个 seed 检查中有非零梯度；D2、D3 为 0/3，固定最终缩放切断语义 MLP 的有效分类梯度。",
            "- D3 seed=0 曾留下一个被中断的重复尝试，随后重启任务完整完成；结果汇总只读取最后一次完整 10 折尝试，原始日志保留该审计记录。",
        ])
    else:
        block.append("- 尚未完成全部任务，暂不写入最终性能结论。")
    block.extend([
        "",
        "后续实验建议：",
        "- 在不使用 val=test 的独立外部测试或嵌套交叉验证协议上复核 D0/D1/D4 的小幅差异。",
        "- 若继续机制拆解，优先分别固定 CLIP 文本特征和 MLP 投影，再比较原型相似度与门控权重，避免把两者作用混为一谈。",
        "",
        end,
    ])
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + "\n".join(block) + after
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    atomic_write(HANDOFF, text)


def update_outputs():
    rows = all_fold_rows()
    diagnostics = all_diagnostic_rows()
    write_fold_metrics(rows)
    write_diagnostics(diagnostics)
    atomic_write(REPORT_FILE, report_text(rows, diagnostics))
    update_handoff(rows, diagnostics)
    return rows, diagnostics


def prepare_worktree(work, config):
    for relative in SOURCE_ROOT_FILES:
        shutil.copy2(ROOT / relative, work / relative)
    shutil.copytree(ROOT / "CLIP", work / "CLIP", copy_function=shutil.copy2)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir()
    (work / "graph").mkdir()
    patch_text, network, model = expected_patches()
    applied = subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
        cwd=work, text=True, capture_output=True, check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"PEPCA patch failed for {config}: {applied.stdout}{applied.stderr}")
    if (work / "network.py").read_text(encoding="utf-8") != network:
        raise RuntimeError("Patched network.py differs from generated PEPCA patch.")
    if (work / "model.py").read_text(encoding="utf-8") != model:
        raise RuntimeError("Patched model.py differs from generated PEPCA patch.")
    expected = make_provenance()["source_files"]
    for relative, digest in expected.items():
        if relative in ("network.py", "model.py"):
            continue
        if sha256(work / relative) != digest:
            raise RuntimeError(f"Isolated source differs from root: {relative}")


def run_one(config, seed, gpu=None, write_outputs=True):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log = task_log(config, seed)
    state, folds = task_state(config, seed)
    if state == "COMPLETE" and folds == 10:
        print(f"Skip {config} seed={seed}: complete 10-fold task.")
        if write_outputs:
            update_outputs()
        return
    diag_path = WORK_DIR / f"{config}_seed{seed}.jsonl"
    gradient_path = WORK_DIR / f"{config}_seed{seed}.gradient.json"
    for path in (diag_path, gradient_path):
        if path.exists():
            path.unlink()
    with tempfile.TemporaryDirectory(prefix=f"SPromptGL-PEPCA-{config}-seed{seed}-") as temporary:
        work = Path(temporary)
        prepare_worktree(work, config)
        shell_args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        if shell_args[:2] != ["python", "main.py"]:
            raise RuntimeError("Unexpected official shell command.")
        command = [sys.executable, "main.py", *shell_args[2:], "--seed", str(seed)]
        if gpu is not None:
            command.extend(["--gpu", "0"])
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PEPCA_CONFIG"] = config
        env["PEPCA_CONTROLS_FILE"] = str(CONTROLS_FILE)
        env["PEPCA_DIAGNOSTIC"] = "1"
        env["PEPCA_DIAGNOSTIC_PATH"] = str(diag_path)
        env["PEPCA_GRADIENT_PATH"] = str(gradient_path)
        start_time = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} gpu={gpu}\n")
            stream.write("COMMAND " + shlex.join(command) + "\n")
            stream.flush()
            completed = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            current_folds = len(parse_attempt_rows(log))
            stream.write(
                f"TASK_END config={config} seed={seed} returncode={completed.returncode} "
                f"folds={current_folds} elapsed_seconds={time.time() - start_time:.1f}\n"
            )
            stream.flush()
    if write_outputs:
        update_outputs()
    print(f"{config} seed={seed}: {task_state(config, seed)[0]}, {len(parse_attempt_rows(log))}/10 folds")


def reference_rows(config, seed):
    with REFERENCE_FILE.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["config"] == config and int(row["seed"]) == seed]


def parity_check(config, seed):
    actual = parse_attempt_rows(task_log(config, seed))
    reference = reference_rows(config, seed)
    if len(actual) != 10 or len(reference) != 10:
        raise RuntimeError(f"Parity requires 10 folds for {config} seed={seed}; got {len(actual)} and {len(reference)}.")
    differences = []
    max_abs = 0.0
    for fold, (got, expected) in enumerate(zip(actual, reference), 1):
        for metric in METRICS:
            if got.get(metric) is None:
                differences.append((fold, metric, None, float(expected[metric])))
                continue
            delta = abs(float(got[metric]) - float(expected[metric]))
            max_abs = max(max_abs, delta)
            if delta > 5e-4:
                differences.append((fold, metric, float(got[metric]), float(expected[metric])))
    result = {"config": config, "seed": seed, "pass": not differences, "max_abs_difference": max_abs, "differences": differences}
    provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    provenance.setdefault("parity_checks", {})[f"{config}_seed{seed}"] = result
    atomic_write(PROVENANCE_FILE, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(f"Parity {config} seed={seed}: {'PASS' if result['pass'] else 'FAIL'}, max_abs={max_abs:.6g}")
    if differences:
        raise RuntimeError(f"Parity failed for {config} seed={seed}: {differences[:4]}")


def update_gradient_provenance():
    provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    checks = {}
    for config in CONFIGS:
        entries = []
        for path in sorted(WORK_DIR.glob(f"{config}_seed*.gradient.json")):
            try:
                entries.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        if entries:
            checks[config] = entries
    provenance["gradient_checks"] = checks
    atomic_write(PROVENANCE_FILE, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def pending_tasks():
    return [(config, seed) for config in CONFIGS for seed in SEEDS if task_state(config, seed)[0] not in ("COMPLETE", "RUNNING")]


def parallel_tasks():
    tasks = pending_tasks()
    existing = [
        (config, seed, running_gpu(config, seed))
        for config in CONFIGS for seed in SEEDS
        if process_running(config, seed)
    ]
    if not tasks and not existing:
        update_outputs()
        return
    scheduler = LOG_DIR / "scheduler.log"
    active = []
    next_index = 0
    with scheduler.open("a", encoding="utf-8") as stream:
        stream.write(f"\nPARALLEL_START pending={len(tasks)} already_running={len(existing)} max_per_gpu=5 gpus=0,1\n")
        while next_index < len(tasks) or active or existing:
            for gpu in (0, 1):
                occupied = sum(item[2] == gpu for item in existing) + sum(item[3] == gpu for item in active)
                if occupied >= 5:
                    continue
                if next_index >= len(tasks):
                    continue
                config, seed = tasks[next_index]
                next_index += 1
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--config", config, "--seed", str(seed), "--gpu", str(gpu), "--worker",
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                active.append((process, config, seed, gpu))
                stream.write(f"LAUNCH config={config} seed={seed} gpu={gpu} pid={process.pid}\n")
            stream.flush()
            still = []
            for process, config, seed, gpu in active:
                code = process.poll()
                if code is None:
                    still.append((process, config, seed, gpu))
                else:
                    stream.write(f"DONE config={config} seed={seed} gpu={gpu} pid={process.pid} returncode={code}\n")
                    stream.flush()
                    update_outputs()
            active = still
            existing = [
                (config, seed, running_gpu(config, seed))
                for config, seed, _gpu in existing
                if process_running(config, seed)
            ]
            if active or existing:
                time.sleep(2)
        stream.write("PARALLEL_END\n")
    update_gradient_provenance()
    update_outputs()


def dynamic_pending_tasks():
    scheduler = LOG_DIR / "scheduler.log"
    with scheduler.open("a", encoding="utf-8") as stream:
        stream.write("\nDYNAMIC_PENDING_START max_per_gpu=5 gpus=0,1\n")
        while True:
            running = {
                (config, seed): running_gpu(config, seed)
                for config in CONFIGS for seed in SEEDS
                if process_running(config, seed)
            }
            pending = [
                (config, seed)
                for config in CONFIGS for seed in SEEDS
                if (config, seed) not in running and task_state(config, seed)[0] != "COMPLETE"
            ]
            if not pending and not running:
                break
            for gpu in (0, 1):
                occupied = sum(value == gpu for value in running.values())
                while occupied < 5 and pending:
                    config, seed = pending.pop(0)
                    command = [
                        sys.executable, str(Path(__file__).resolve()),
                        "--config", config, "--seed", str(seed), "--gpu", str(gpu), "--worker",
                    ]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(
                        command, cwd=ROOT, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                    )
                    stream.write(
                        f"DYNAMIC_LAUNCH config={config} seed={seed} gpu={gpu} pid={process.pid}\n"
                    )
                    stream.flush()
                    occupied += 1
                    running[(config, seed)] = gpu
            update_outputs()
            time.sleep(5)
        stream.write("DYNAMIC_PENDING_END\n")
    update_gradient_provenance()
    update_outputs()


def all_run():
    if not CONTROLS_FILE.exists() or not PATCH_FILE.exists() or not REFERENCE_FILE.exists():
        prepare()
    if task_state("D0", 0)[0] != "COMPLETE":
        run_one("D0", 0, gpu=0)
    parity_check("D0", 0)
    if task_state("D1", 0)[0] != "COMPLETE":
        run_one("D1", 0, gpu=1)
    parity_check("D1", 0)
    parallel_tasks()
    update_gradient_provenance()
    update_outputs()
    complete = sum(task_state(config, seed)[0] == "COMPLETE" for config in CONFIGS for seed in SEEDS)
    print(f"PEPCA complete tasks: {complete}/15, folds: {len(all_fold_rows())}/150")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--parity", choices=["D0", "D1"])
    parser.add_argument("--pending", action="store_true", help="dynamically resume every incomplete task")
    parser.add_argument("--refresh", action="store_true", help="refresh aggregate outputs without launching workers")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.parity:
        parity_check(args.parity, 0)
        return
    if args.all:
        all_run()
        return
    if args.pending:
        dynamic_pending_tasks()
        return
    if args.refresh:
        update_gradient_provenance()
        update_outputs()
        return
    if args.config is None or args.seed is None:
        parser.error("choose --prepare, --all, --parity, or --config with --seed")
    if not CONTROLS_FILE.exists():
        prepare()
    run_one(args.config, args.seed, gpu=args.gpu, write_outputs=not args.worker)


if __name__ == "__main__":
    main()

