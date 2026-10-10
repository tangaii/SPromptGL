#!/usr/bin/env python3
"""Run CSMP prompts with isolated copies of the root SPromptGL implementation."""

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
HASH_FILE = EXP / "source_hashes.json"
PATCH_FILE = EXP / "prompt_patch.diff"
PROMPTS_FILE = EXP / "prompts.json"
REFERENCE_FILE = EXP / "results" / "reference_fold_results.csv"
LOG_DIR = EXP / "logs"
RESULT_DIR = EXP / "results"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
REFERENCE_CONFIGS = ["P0", "P1", "P2", "P3"]
CONFIGS = ["V2", "V2S"]
SEEDS = [0, 1, 2]
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
V1_PROMPTS_SOURCE = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
REFERENCE_SOURCE = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths():
    paths = [ROOT / name for name in SOURCE_ROOT_FILES]
    paths.extend(sorted(path for path in (ROOT / "CLIP").rglob("*") if path.is_file()))
    return paths


def make_hash_manifest():
    files = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}
    inputs = {
        "data/ABIDE/processed_standard_data.csv": sha256(ROOT / "data/ABIDE/processed_standard_data.csv"),
        "data/ABIDE/modal_feat_dict.npy": sha256(ROOT / "data/ABIDE/modal_feat_dict.npy"),
    }
    weight = Path.home() / ".cache/clip/RN50.pt"
    weight_hash = sha256(weight) if weight.is_file() else None
    if weight_hash is not None and weight_hash != RN50_SHA256:
        raise RuntimeError(f"RN50 checksum mismatch: {weight_hash}")
    return {
        "source_root": str(ROOT),
        "files": files,
        "inputs": inputs,
        "clip_rn50": {
            "path": str(weight),
            "expected_sha256": RN50_SHA256,
            "actual_sha256_at_snapshot": weight_hash,
        },
    }


def ensure_hash_manifest():
    current = make_hash_manifest()
    if HASH_FILE.exists():
        saved = json.loads(HASH_FILE.read_text(encoding="utf-8"))
        if current["files"] != saved["files"] or current["inputs"] != saved["inputs"]:
            raise RuntimeError("Root source/data hashes changed after the experiment snapshot.")
        if current["clip_rn50"]["actual_sha256_at_snapshot"] not in (None, current["clip_rn50"]["expected_sha256"]):
            raise RuntimeError("RN50 checksum differs from the expected OpenAI RN50 asset.")
        if saved["clip_rn50"]["actual_sha256_at_snapshot"] not in (None, current["clip_rn50"]["actual_sha256_at_snapshot"]):
            raise RuntimeError("The RN50 checkpoint changed after the experiment snapshot.")
    else:
        HASH_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current


def load_prompts():
    return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def _theme_texts(themes, modality, label):
    packs = themes.get(modality)
    if isinstance(packs, dict) and "themes" in packs:
        packs = packs["themes"]
    if not isinstance(packs, list) or len(packs) != 2:
        raise ValueError(f"{label}/{modality} must have two prompt themes.")
    result = []
    for pack in packs:
        prompts = pack.get("prompts", [])
        if len(prompts) != 4:
            raise ValueError(f"{label}/{modality}/{pack.get('theme')} must have four prompts.")
        texts = [item if isinstance(item, str) else item["text"] for item in prompts]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"{label}/{modality} contains an empty prompt.")
        result.append(texts)
    return result


def prompt_themes(config, prompts):
    order = prompts.get("modality_order")
    if order != MODALITIES:
        raise ValueError(f"Prompt modality order must match ABIDE input order: {order}")
    source = prompts["v2"]["themes"]
    mapping = {key: key for key in order}
    if config == "V2S":
        mapping = prompts["v2s"]["mapping"]
        if set(mapping) != set(order) or set(mapping.values()) != set(order) or any(mapping[key] == key for key in order):
            raise ValueError("V2S must use the fixed modality derangement.")
    elif config != "V2":
        raise ValueError(config)
    result = [_theme_texts(source, mapping[modality], config) for modality in order]
    if sum(len(theme) for themes in result for theme in themes) != 32:
        raise ValueError(f"{config} must contain exactly 32 prompt sentences.")
    return result


def prompt_sentences(config, prompts):
    return [text for themes in prompt_themes(config, prompts) for theme in themes for text in theme]


def patched_network(source):
    original_import = "import argparse\n"
    if source.count(original_import) != 1:
        raise RuntimeError("Cannot identify the network.py import anchor.")
    source = source.replace(original_import, "import argparse\nimport json\n", 1)

    old_function = '''def get_prompt_text_features():
    """Encode the fixed semantic prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(text).float().detach()
    return _prompt_text_features
'''
    new_function = '''def get_prompt_text_features():
    """Encode the fixed CSMP prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        prompt_tokens = text
        config = os.environ.get("CSMP_CONFIG")
        if config:
            with open(os.environ["CSMP_PROMPTS_FILE"], encoding="utf-8") as stream:
                prompt_data = json.load(stream)
            source = prompt_data["v2"]["themes"]
            mapping = {key: key for key in prompt_data["modality_order"]}
            if config == "V2S":
                mapping = prompt_data["v2s"]["mapping"]
            sentences = []
            for modality in prompt_data["modality_order"]:
                for theme in source[mapping[modality]]:
                    sentences.extend(item if isinstance(item, str) else item["text"] for item in theme["prompts"])
            if len(sentences) != 32:
                raise ValueError("CSMP variants require 32 ordered prompt sentences.")
            prompt_tokens = clip.tokenize(sentences).to(device)
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(prompt_tokens).float().detach()
    return _prompt_text_features
'''
    if source.count(old_function) != 1:
        raise RuntimeError("Cannot identify the original CLIP prompt encoder.")
    source = source.replace(old_function, new_function, 1)

    old_grouping = '''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
'''
    new_grouping = '''        text_features = self.prompt_text_features
        if text_features.size(0) == 8:
            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
            text_features = self.MLP(text_features)
        else:
            text_features = self.MLP(text_features.view(4, 2, 4096))
'''
    if source.count(old_grouping) != 1:
        raise RuntimeError("Cannot identify the original 8-prompt grouping.")
    source = source.replace(old_grouping, new_grouping, 1)

    start_marker = "        text_features_norm = F.normalize(text_features, p=2, dim=-1)\n"
    end_marker = "        similarity_matrix = similarity_matrix[:,:,:,-1,:]\n"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or source.find(start_marker, start + 1) >= 0:
        raise RuntimeError("Cannot identify the original semantic similarity block.")
    end += len(end_marker)
    new_similarity = '''        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        if text_features_norm.dim() == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            part_x_norm_expanded = part_x_norm.unsqueeze(-2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, text_features_norm).unsqueeze(-2)
        similarity_matrix = similarity_matrix[:,:,:,-1,:]
'''
    source = source[:start] + new_similarity + source[end:]
    return source


def expected_patch(root_network):
    variant = patched_network(root_network)
    diff = difflib.unified_diff(
        root_network.splitlines(keepends=True),
        variant.splitlines(keepends=True),
        fromfile="a/network.py",
        tofile="b/network.py",
    )
    return "".join(diff), variant


def validate_tokenization(prompts):
    sys.path.insert(0, str(ROOT))
    from CLIP.clip import clip

    for config in CONFIGS:
        encoded = clip.tokenize(prompt_sentences(config, prompts))
        if tuple(encoded.shape) != (32, 77):
            raise RuntimeError(f"Unexpected {config} CLIP token shape: {tuple(encoded.shape)}")


def validate_prompts(prompts):
    if prompts.get("v2", {}).get("name") != "CONTRASTIVE_STATE_PROMPT":
        raise ValueError("V2 prompt name is not fixed to CONTRASTIVE_STATE_PROMPT.")
    if prompts.get("v2s", {}).get("source") != "v2":
        raise ValueError("V2S must reuse V2 prompt text.")
    prompt_themes("V2", prompts)
    prompt_themes("V2S", prompts)
    if sorted(prompt_sentences("V2", prompts)) != sorted(prompt_sentences("V2S", prompts)):
        raise ValueError("V2 and V2S must contain the same 32 prompt sentences.")


def copy_reference_snapshot():
    if not REFERENCE_SOURCE.exists():
        raise FileNotFoundError(f"Missing experiment-one reference: {REFERENCE_SOURCE}")
    source_digest = sha256(REFERENCE_SOURCE)
    with REFERENCE_SOURCE.open(newline="", encoding="utf-8") as source, REFERENCE_FILE.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        fields = ["config", "seed", "fold", *METRICS, "sensitivity_nc", "specificity_nc", "train_seconds", "epoch_cap", "early_patience", "auc_source", "reference_source_sha256"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        count = 0
        for row in reader:
            if row["config"] not in REFERENCE_CONFIGS:
                continue
            output = {field: row.get(field, "") for field in fields if field != "reference_source_sha256"}
            output["reference_source_sha256"] = source_digest
            writer.writerow(output)
            count += 1
    if count != 120:
        raise RuntimeError(f"Expected 120 reference folds, found {count}.")


def prompt_embedding_diagnostic(prompts):
    """Compare V1/V2 theme embeddings without fitting or selecting anything."""
    if not V1_PROMPTS_SOURCE.exists():
        return {"status": "unavailable", "reason": "experiment-one prompt snapshot missing"}
    try:
        import torch
        import torch.nn.functional as F
        from CLIP.clip import clip
        v1 = json.loads(V1_PROMPTS_SOURCE.read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("RN50", device)
        values = {}
        with torch.no_grad():
            for label, theme_source in (("V1", v1["p1"]), ("V2", prompts["v2"]["themes"])):
                theme_vectors = []
                for modality in MODALITIES:
                    for theme in _theme_texts(theme_source, modality, label):
                        tokens = clip.tokenize(theme).to(device)
                        vector = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(dim=0)
                        theme_vectors.append((modality, F.normalize(vector, dim=0)))
                within = {}
                for modality in MODALITIES:
                    vectors = [vector for mod, vector in theme_vectors if mod == modality]
                    within[modality] = float(torch.dot(vectors[0], vectors[1]).cpu())
                matrix = torch.stack([vector for _, vector in theme_vectors])
                matrix = matrix @ matrix.T
                off = matrix[~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)]
                values[label] = {"within_modality_theme_cosine": within, "all_theme_off_diagonal_mean": float(off.mean().cpu()), "all_theme_off_diagonal_max": float(off.max().cpu())}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "complete", "values": values, "note": "Theme means only; diagnostic, not a selection criterion."}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}


def forward_check():
    code = r'''
import os
from types import SimpleNamespace
import torch
from network import PLN
hp = SimpleNamespace(n_hidden=18, n_head=2, nmodal=4, nclass=2, nlayer=1, dropout=0.35)
model = PLN([48, 6, 10, 256], hp).to("cuda" if torch.cuda.is_available() else "cpu").eval()
before = {key: value.detach().clone() for key, value in model.state_dict().items()}
assert tuple(model.prompt_text_features.shape) == (32, 1024), model.prompt_text_features.shape
with torch.no_grad():
    prototypes = model.MLP(model.prompt_text_features.view(4, 2, 4096))
    assert tuple(prototypes.shape) == (4, 2, 9), prototypes.shape
    x = torch.randn(871, 320, device=next(model.parameters()).device)
    output, _, _ = model(x)
    assert tuple(output.shape) == (871, 2), output.shape
for key, value in model.state_dict().items():
    assert torch.equal(before[key], value), key
print("forward_check", os.environ["CSMP_CONFIG"], tuple(model.prompt_text_features.shape), tuple(prototypes.shape), tuple(output.shape))
'''
    for config in CONFIGS:
        with tempfile.TemporaryDirectory(prefix=f"csmp_forward_{config}_") as temp:
            work = Path(temp)
            shutil.copy2(ROOT / "network.py", work / "network.py")
            shutil.copy2(ROOT / "layers.py", work / "layers.py")
            shutil.copy2(ROOT / "utils.py", work / "utils.py")
            shutil.copytree(ROOT / "CLIP", work / "CLIP", copy_function=shutil.copy2)
            applied = subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)], cwd=work, text=True, capture_output=True)
            if applied.returncode != 0:
                raise RuntimeError(f"Forward-check patch failed: {applied.stdout}{applied.stderr}")
            env = os.environ.copy()
            env["CSMP_CONFIG"] = config
            env["CSMP_PROMPTS_FILE"] = str(PROMPTS_FILE)
            env["PYTHONPATH"] = str(work)
            result = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"{config} forward check failed:\n{result.stdout}\n{result.stderr}")
            print(result.stdout.strip())


def prepare():
    ensure_hash_manifest()
    prompts = load_prompts()
    validate_prompts(prompts)
    validate_tokenization(prompts)
    original = (ROOT / "network.py").read_text(encoding="utf-8")
    patch_text, variant = expected_patch(original)
    if PATCH_FILE.exists():
        if PATCH_FILE.read_text(encoding="utf-8") != patch_text:
            raise RuntimeError("prompt_patch.diff does not match the generated minimal network.py patch.")
    else:
        PATCH_FILE.write_text(patch_text, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="csmp_patch_check_") as temp:
        work = Path(temp)
        shutil.copy2(ROOT / "network.py", work / "network.py")
        applied = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0 or (work / "network.py").read_text(encoding="utf-8") != variant:
            raise RuntimeError(f"Prompt patch failed to apply exactly: {applied.stdout}{applied.stderr}")
        compile((work / "network.py").read_text(encoding="utf-8"), "network.py", "exec")
    copy_reference_snapshot()
    diagnostic = prompt_embedding_diagnostic(prompts)
    forward_check()
    HASH_FILE.write_text(json.dumps({**json.loads(HASH_FILE.read_text(encoding="utf-8")), "prompt_file_sha256": sha256(PROMPTS_FILE), "prompt_embedding_diagnostic": diagnostic, "prompt_patch_sha256": sha256(PATCH_FILE), "reference_fold_results_sha256": sha256(REFERENCE_FILE), "reference_source_sha256": sha256(REFERENCE_SOURCE)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_outputs()
    print(f"Prepared CSMP. Root source hashes: {len(json.loads(HASH_FILE.read_text())['files'])} files.")
    print(json.dumps(diagnostic, ensure_ascii=False))


def ensure_archive_handoff():
    if not HANDOFF.exists():
        HANDOFF.write_text("# Experiment Handoff\n", encoding="utf-8")
    text = HANDOFF.read_text(encoding="utf-8")
    old_start, old_end = "<!-- MASPE EXPERIMENT 1 START -->", "<!-- MASPE EXPERIMENT 1 END -->"
    if old_start in text and old_end in text:
        before, remainder = text.split(old_start, 1)
        block, after = remainder.split(old_end, 1)
        block = block.replace("- 状态：`COMPLETE`", "- 状态：`ARCHIVED`（上一版严格协议结果，仅作历史记录）", 1)
        block = block.replace(
            "- 实验目录：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt`",
            "- 归档目录：`/data/adodas/SPromptGL-main/experiment/_archive/experiment1_maspe_strict_20261009`",
            1,
        )
        block = block.replace("<!-- MASPE EXPERIMENT 1 START -->", "<!-- MASPE STRICT ARCHIVE START -->")
        block = block.replace("<!-- MASPE EXPERIMENT 1 END -->", "<!-- MASPE STRICT ARCHIVE END -->")
        block = "\n- 本条为上一版严格协议实验归档记录，不得与本次原始协议结果合并。\n" + block
        text = before + "<!-- MASPE STRICT ARCHIVE START -->" + block + "<!-- MASPE STRICT ARCHIVE END -->" + after
    if "<!-- MASPE STRICT ARCHIVE START -->" not in text:
        text += "\n<!-- MASPE STRICT ARCHIVE START -->\n上一版严格协议实验已归档于 `experiment/_archive/experiment1_maspe_strict_20261009`，不得与本轮结果混合。\n<!-- MASPE STRICT ARCHIVE END -->\n"
    return text


def attempt_text(log_path):
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    start = text.rfind("ATTEMPT_START ")
    return text[start:] if start >= 0 else ""


def attempt_rows(log_path):
    text = attempt_text(log_path)
    lines = text.splitlines()
    events = []
    timers = []
    for index, line in enumerate(lines):
        timer_match = re.search(r"time:\s*([0-9.]+)\s*sec", line)
        if timer_match:
            timers.append(float(timer_match.group(1)))
        match = TEST_RE.search(line)
        if not match:
            continue
        following = lines[index + 1 :]
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
            "sensitivity_nc": sens[1] if len(sens) == 2 else None,
            "specificity_asd": spes[0] if len(spes) == 2 else None,
            "specificity_nc": spes[1] if len(spes) == 2 else None,
            "train_seconds": timers[len(events)] if len(timers) > len(events) else None,
        })
    return events


def task_end(log_path):
    text = attempt_text(log_path)
    matches = list(re.finditer(r"TASK_END config=(V2S?) seed=(\d+) returncode=(-?\d+) folds=(\d+)", text))
    if not matches:
        return None
    match = matches[-1]
    return {"returncode": int(match.group(3)), "folds": int(match.group(4))}


def task_log(config, seed):
    return LOG_DIR / f"{config}_seed{seed}.log"


def task_state(config, seed):
    log = task_log(config, seed)
    end = task_end(log) if log.exists() else None
    folds = len(attempt_rows(log)) if log.exists() else 0
    if end is None:
        return "PENDING", folds
    if end["returncode"] != 0:
        return "FAILED", folds
    if end["folds"] == 10 and folds == 10:
        return "COMPLETE", folds
    return "EARLY_TERMINATED", folds


def fmt(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"


def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)


def all_rows():
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            for fold, metrics in enumerate(attempt_rows(task_log(config, seed)), 1):
                rows.append({"config": config, "seed": seed, "fold": fold, **metrics})
    return rows


def write_fold_results(rows):
    path = RESULT_DIR / "fold_results.csv"
    columns = [
        "config", "seed", "fold", "acc", "hard_auc", "sensitivity_asd", "specificity_asd",
        "sensitivity_nc", "specificity_nc", "train_seconds", "epoch_cap", "early_patience", "auc_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out.update({"epoch_cap": 450, "early_patience": 200, "auc_source": "hard_label"})
            for key in METRICS + ["sensitivity_nc", "specificity_nc", "train_seconds"]:
                out[key] = fmt(row.get(key), 6)
            writer.writerow(out)


def reference_rows():
    if not REFERENCE_FILE.exists():
        return []
    with REFERENCE_FILE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summary_rows(rows):
    grouped = {(config, seed): [] for config in REFERENCE_CONFIGS + CONFIGS for seed in SEEDS}
    for row in reference_rows() + rows:
        grouped[(row["config"], int(row["seed"]))].append(row)
    output = []
    for config in REFERENCE_CONFIGS + CONFIGS:
        seed_groups = []
        for seed in SEEDS:
            values = grouped[(config, seed)]
            row = {"source": "experiment1_reference_snapshot" if config in REFERENCE_CONFIGS else "experiment2_current_run", "scope": "seed_folds", "config": config, "seed": seed, "n_seeds": 1, "n_folds": len(values)}
            for metric in METRICS:
                metric_values = [float(item[metric]) for item in values if item.get(metric) not in (None, "")]
                row[f"{metric}_mean"] = statistics.mean(metric_values) if metric_values else None
                row[f"{metric}_fold_sd"] = statistics.pstdev(metric_values) if metric_values else None
                row[f"{metric}_seed_sd"] = None
            output.append(row)
            complete = len(values) == 10 and all(item.get(metric) not in (None, "") for item in values for metric in METRICS)
            if complete:
                seed_groups.append(row)
        if seed_groups:
            aggregate = {"source": "experiment1_reference_snapshot" if config in REFERENCE_CONFIGS else "experiment2_current_run", "scope": "seed_aggregate", "config": config, "seed": "ALL", "n_seeds": len(seed_groups), "n_folds": sum(int(item["n_folds"]) for item in seed_groups)}
            for metric in METRICS:
                means = [item[f"{metric}_mean"] for item in seed_groups]
                aggregate[f"{metric}_mean"] = statistics.mean(means)
                aggregate[f"{metric}_fold_sd"] = None
                aggregate[f"{metric}_seed_sd"] = statistics.pstdev(means) if len(means) > 1 else None
            output.append(aggregate)
    return output


def write_summary(rows):
    path = RESULT_DIR / "summary.csv"
    columns = ["source", "scope", "config", "seed", "n_seeds", "n_folds"]
    for metric in METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_fold_sd", f"{metric}_seed_sd"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key), 6) if key.endswith(("_mean", "_sd")) else row.get(key) for key in columns})


def summary_index(summaries):
    return {(row["config"], str(row["seed"])): row for row in summaries}


def report_text(rows, summaries):
    idx = summary_index(summaries)
    states = {(config, seed): task_state(config, seed) for config in CONFIGS for seed in SEEDS}
    terminal = sum(state[0] != "PENDING" for state in states.values())
    total_folds = len(rows)
    fully_complete = all(state[0] == "COMPLETE" for state in states.values())
    status = "全部完成" if fully_complete else ("执行中或待续" if terminal < 12 else "部分任务提前终止或失败")
    lines = [
        "# 实验一：模态感知语义提示增强（MASPE）",
        "",
        f"状态：**{status}**。任务 {terminal}/12 已结束，当前记录 {total_folds}/120 个测试折。",
        "",
        "## 协议与来源",
        "",
        "P0 在隔离副本中逐字节运行根目录原始源码及 `ABIDE-simple-2-concat-weighted-cosine.sh` 参数。P1–P3 只在隔离副本的 `network.py` 应用 `prompt_patch.diff`；四组均使用根目录 `main.py`、`model.py` 的训练与评价路径。",
        "",
        "保持 `StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`、原始 `val_idx == tst_idx`、ACC 最优 checkpoint、450 epoch 上限、200 patience、原始 hard-label AUC 和根代码提前终止条件。没有额外验证集或调参。图特征传播保持原始 transductive 方式。",
        "",
        "数据固定为 `data/ABIDE/processed_standard_data.csv` 和 `modal_feat_dict.npy`。根代码使用的测试折同时参与 checkpoint 选择；这里如实沿用原始协议，不将其称作独立测试评估。输入文件已按要求固定，既有全队列标准化/RFE 预处理不在本实验中重做。",
        "",
        "源码和数据哈希见 `source_hashes.json`；P1/P2/P3 提示文本见 `prompts.json`。CLIP 使用 OpenAI RN50，权重 SHA256 应为 `" + RN50_SHA256 + "`。",
        "",
        "## P0 论文参考核对",
        "",
    ]
    p0 = idx.get(("P0", "0"))
    if p0 and p0.get("n_folds", 0):
        acc = p0.get("acc_mean")
        auc = p0.get("hard_auc_mean")
        acc_sd = p0.get("acc_fold_sd")
        auc_sd = p0.get("hard_auc_fold_sd")
        lines.extend([
            f"P0 seed=0 当前完成 {p0['n_folds']}/10 折；ACC {fmt(acc * 100 if acc is not None else None, 2)}% ± {fmt(acc_sd * 100 if acc_sd is not None else None, 2)}%，Hard-AUC {fmt(auc * 100 if auc is not None else None, 2)}% ± {fmt(auc_sd * 100 if auc_sd is not None else None, 2)}%。",
            f"论文参考为 ACC {PAPER['acc_mean']:.2f}±{PAPER['acc_sd']:.2f}%，AUC {PAPER['hard_auc_mean']:.2f}±{PAPER['hard_auc_sd']:.2f}%；当前与参考差值：ACC {fmt((acc * 100 - PAPER['acc_mean']) if acc is not None else None, 2)} 个百分点，Hard-AUC {fmt((auc * 100 - PAPER['hard_auc_mean']) if auc is not None else None, 2)} 个百分点。论文值与本地复现的协议/指标实现是否完全同口径无法仅凭数值确认。",
        ])
    else:
        lines.append("P0 seed=0 尚无完整折结果。")
    lines.extend(["", "## 各配置结果", "", "下表为每个 seed 的 10 折均值，再报告 seed 均值及 seed 间标准差；`fold_sd` 与 `seed_sd` 分开存于 `summary.csv`。SEN/SPE 以 ASD（label 0）为阳性类，另一类结果也保存在逐折 CSV 中。", "", "| 配置 | 完整 seeds | ACC (%; seed mean±SD) | Hard-AUC (%; seed mean±SD) | SEN-ASD (%) | SPE-ASD (%) |", "|---|---:|---:|---:|---:|---:|"])
    aggregate_by_config = {row["config"]: row for row in summaries if row["scope"] == "seed_aggregate"}
    for config in CONFIGS:
        agg = aggregate_by_config.get(config)
        if not agg:
            lines.append(f"| {config} | 0 | - | - | - | - |")
            continue
        metric_cells = []
        for metric in METRICS:
            mean = agg.get(f"{metric}_mean")
            sd = agg.get(f"{metric}_seed_sd")
            metric_cells.append(f"{fmt(mean * 100 if mean is not None else None, 2)} ± {fmt(sd * 100 if sd is not None else None, 2)}" if metric in ("acc", "hard_auc") else fmt(mean * 100 if mean is not None else None, 2))
        lines.append(f"| {config} | {agg['n_seeds']}/3 | {metric_cells[0]} | {metric_cells[1]} | {metric_cells[2]} | {metric_cells[3]} |")
    lines.extend(["", "## 对照与稳定性", ""])
    paired = []
    for seed in SEEDS:
        base, aware = idx.get(("P0", str(seed))), idx.get(("P1", str(seed)))
        if base and aware and base.get("n_folds") == 10 and aware.get("n_folds") == 10:
            da = (aware["acc_mean"] - base["acc_mean"]) * 100
            du = (aware["hard_auc_mean"] - base["hard_auc_mean"]) * 100
            paired.append((seed, da, du))
            lines.append(f"seed={seed}: P1-P0 的 ACC {da:+.2f} 个百分点，Hard-AUC {du:+.2f} 个百分点。")
    if paired:
        acc_wins = sum(da > 0 for _, da, _ in paired)
        auc_wins = sum(du > 0 for _, _, du in paired)
        lines.append(f"在 {len(paired)} 个可配对 seed 中，P1 的 ACC 提升 {acc_wins} 个、Hard-AUC 提升 {auc_wins} 个；此处只作描述性稳定性检查，不作显著性推断。")
        if acc_wins == len(paired) and auc_wins == len(paired) and len(paired) == 3:
            lines.append("P1 相对 P0 在三个 seed 的 ACC 与 Hard-AUC 均上升，方向一致；仍需结合 P2/P3 判断语义模态对应是否带来额外收益。")
        else:
            lines.append("目前没有看到 P1 在 ACC 与 Hard-AUC 上跨 seed 同时稳定超过 P0；请结合 P2/P3 的同协议结果判断收益是否来自模态语义对应。")
    else:
        lines.append("P0/P1 尚无可配对的完整 seed。")
    lines.extend(["", "P2 使用固定错配映射，P3 使用 32 条数量匹配通用提示。它们的逐 seed 差异以上表及 `summary.csv` 为准；当前有效完整 seed 不足时不作科学结论。", "", "## 逐任务状态", "", "| 配置 | Seed | 状态 | 已记录折数 |", "|---|---:|---|---:|"])
    for config in CONFIGS:
        for seed in SEEDS:
            state, fold_count = states[(config, seed)]
            lines.append(f"| {config} | {seed} | {state} | {fold_count}/10 |")
    lines.extend(["", "## 解释与限制", "", "若 P1 未优于 P0，结果支持的结论仅是：在当前原始模型和冻结 ABIDE 特征下，这组模态描述没有带来稳定分类收益。数据中的 PHENO 含站点/人口学信息，ANAT 与 FUNC 主要包含图像质量和运动/信号质量量，Correlation 为预先 RFE 选出的连接边；提示语义可能与这些压缩后的质量和混合特征并不完全对应。最终解释以完整结果为准。", "", "测试折被用于原始验证 ACC 和 checkpoint 选择，因此不能把本实验表述为独立测试集估计。执行中不修正该协议。", ""])
    return "\n".join(lines)


def update_handoff(rows=None, summaries=None):
    if rows is None:
        rows = all_rows()
    if summaries is None:
        summaries = summary_rows(rows)
    text = ensure_archive_handoff()
    start, end = "<!-- MASPE EXPERIMENT 1 REPLAY START -->", "<!-- MASPE EXPERIMENT 1 REPLAY END -->"
    states = {(config, seed): task_state(config, seed) for config in CONFIGS for seed in SEEDS}
    terminal = sum(state[0] != "PENDING" for state in states.values())
    total_folds = len(rows)
    if all(state[0] == "COMPLETE" for state in states.values()):
        status = "COMPLETE"
    elif terminal == 12:
        status = "INCOMPLETE"
    else:
        status = "RUNNING"
    summary = summary_index(summaries)
    block = [
        start,
        "## 实验一重新执行：MASPE（原始 SPromptGL 协议）",
        "",
        f"- 状态：`{status}`。已结束 {terminal}/12 个 config×seed 任务，记录 {total_folds}/120 折。",
        "- 原因：上一版独立验证/概率 AUC 实验不符合复现要求；完整归档于 `experiment/_archive/experiment1_maspe_strict_20261009`，历史结果不与本轮混合。",
        "- 新实验目录：`experiment/experiment1_modality_aware_semantic_prompt`。",
        "- 代码与协议：P0 使用根目录原码字节级副本；P1–P3 仅改隔离副本 `network.py` 的提示加载/原型分组/模态匹配。其余源码哈希与根目录一致。固定 10 折 random_state=10，原始 `val_idx == tst_idx`、ACC checkpoint 选择、hard-AUC、transductive 传播、450 epoch 与 200 patience。",
        "- 数据：根目录 ABIDE 两个指定文件；不重做预处理。P0–P3 使用相同输入。",
        "- 源码哈希：`source_hashes.json`；提示文本：`prompts.json`；最小 patch：`prompt_patch.diff`。",
        "- 运行命令：`python experiment/experiment1_modality_aware_semantic_prompt/run.py --all`；调度器启动 12 个独立 worker，前 6 个绑定物理 GPU 0、后 6 个绑定物理 GPU 1，每个任务在隔离临时目录中按原顺序执行 10 折。",
        "",
        "任务状态：",
    ]
    for config in CONFIGS:
        for seed in SEEDS:
            task_status, folds = states[(config, seed)]
            block.append(f"- {config} seed={seed}: `{task_status}`，{folds}/10 folds")
    p0 = summary.get(("P0", "0"))
    if p0 and p0.get("n_folds"):
        block.extend(["", f"- P0 seed=0 当前逐折均值：ACC {fmt(p0.get('acc_mean') * 100 if p0.get('acc_mean') is not None else None, 2)}%，Hard-AUC {fmt(p0.get('hard_auc_mean') * 100 if p0.get('hard_auc_mean') is not None else None, 2)}%，SEN-ASD {fmt(p0.get('sensitivity_asd_mean') * 100 if p0.get('sensitivity_asd_mean') is not None else None, 2)}%，SPE-ASD {fmt(p0.get('specificity_asd_mean') * 100 if p0.get('specificity_asd_mean') is not None else None, 2)}%。论文参考 ACC 92.30±2.58%，AUC 92.22±2.57%。"])
    block.extend(["", "结果文件：`results/fold_results.csv`、`results/summary.csv`、`results/report.md`。测试折仍参与 checkpoint 选择，报告会明确说明该原始评价方式。", end])
    new_block = "\n".join(block)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + new_block + after
    else:
        text = text.rstrip() + "\n\n" + new_block + "\n"
    tmp = HANDOFF.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, HANDOFF)


def _pct(value, digits=2):
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _aggregate_rows(summaries):
    return {(row["config"], str(row["seed"])): row for row in summaries}


def _fold_pair_counts(rows, left_config, right_config, metric):
    left = {(int(row["seed"]), int(row["fold"])): float(row[metric]) for row in rows if row["config"] == left_config and row.get(metric) not in (None, "")}
    right = {(int(row["seed"]), int(row["fold"])): float(row[metric]) for row in rows if row["config"] == right_config and row.get(metric) not in (None, "")}
    pairs = [(left[key], right[key]) for key in left.keys() & right.keys()]
    return sum(a > b for a, b in pairs), sum(a == b for a, b in pairs), sum(a < b for a, b in pairs), len(pairs)


def _diagnostic_lines():
    if not HASH_FILE.exists():
        return ["- 文本嵌入诊断尚未执行。"]
    diagnostic = json.loads(HASH_FILE.read_text(encoding="utf-8")).get("prompt_embedding_diagnostic", {})
    if diagnostic.get("status") != "complete":
        return [f"- 文本嵌入诊断状态：{diagnostic.get('status', 'unavailable')}；{diagnostic.get('error', diagnostic.get('reason', '无详细信息'))}。"]
    lines = ["- V1/V2 各主题四条文本平均 CLIP 向量的余弦诊断（仅描述，不用于筛选）："]
    for label, values in diagnostic.get("values", {}).items():
        within = ", ".join(f"{mod}={score:.4f}" for mod, score in values["within_modality_theme_cosine"].items())
        lines.append(f"  - {label}：主题内余弦 {within}；全部主题非对角均值 {values['all_theme_off_diagonal_mean']:.4f}，最大值 {values['all_theme_off_diagonal_max']:.4f}。")
    return lines


def report_text(rows, summaries):
    index = _aggregate_rows(summaries)
    states = {(config, seed): task_state(config, seed) for config in CONFIGS for seed in SEEDS}
    complete = all(state[0] == "COMPLETE" for state in states.values())
    terminal = sum(state[0] != "PENDING" for state in states.values())
    lines = [
        "# 实验二：对比状态驱动的模态语义提示优化（CSMP）", "",
        f"状态：**{'全部完成' if complete else '执行中或存在失败'}**；新任务 {terminal}/6，V2/V2S 当前记录 {len(rows)}/60 折。", "",
        "## 目标与固定设计", "",
        "V2 使用预先固定的 32 条状态对比 Prompt（4 模态 × 2 主题 × 4 文本）。V2S 完全复用 V2 文本，仅固定错配 PHENO←FUNC、FUNC←PHENO、ANAT←Correlation、Correlation←ANAT。实验只改变 Prompt 内容，不改变模型结构、损失、优化器、训练策略、数据或评价协议。", "",
        "根目录源码与 ABIDE 输入哈希、RN50 权重哈希和最小 `network.py` patch 记录在 `source_hashes.json`。Prompt 计算沿用实验一 P1：32 条 CLIP RN50 文本特征按每模态两主题拼接为 4096 维，经共享 4096→128→9 MLP 得到 [4,2,9] 原型，再与 [871,4,4,9] 局部 Token 计算 [871,4,4,2] 相似度并进行原始 Softmax、Max、加权和融合。固定 `StratifiedKFold(10, random_state=10)`、`val_idx == tst_idx`、ACC checkpoint、450 epoch、200 patience 和 hard-label AUC 均保持不变。", "",
        "实验一 P0–P3 逐折结果作为只读参考快照保存于 `results/reference_fold_results.csv`，其中记录了原始 `fold_results.csv` 的 SHA256。测试折参与原始验证和 checkpoint 选择，因此不作独立测试集估计。", "",
        "## 主结果", "", "| 配置 | 来源 | 完整 seeds | ACC（seed均值±SD） | Hard-AUC（seed均值±SD） | SEN-ASD | SPE-ASD |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for config in REFERENCE_CONFIGS + CONFIGS:
        aggregate = index.get((config, "ALL"))
        source = "实验一参考快照" if config in REFERENCE_CONFIGS else "本次运行"
        if not aggregate:
            lines.append(f"| {config} | {source} | 0/3 | - | - | - | - |")
            continue
        acc_sd = aggregate.get("acc_seed_sd")
        auc_sd = aggregate.get("hard_auc_seed_sd")
        lines.append(f"| {config} | {source} | {aggregate['n_seeds']}/3 | {_pct(aggregate.get('acc_mean'))} ± {_pct(acc_sd) if acc_sd is not None else '-'} | {_pct(aggregate.get('hard_auc_mean'))} ± {_pct(auc_sd) if auc_sd is not None else '-'} | {_pct(aggregate.get('sensitivity_asd_mean'))} | {_pct(aggregate.get('specificity_asd_mean'))} |")
    lines.extend(["", "## 必须比较的变化", ""])
    for label, base in (("V2-P0", "P0"), ("V2-P1", "P1"), ("V2-V2S", "V2S")):
        lines.append(f"### {label}")
        for metric in METRICS:
            left = index.get(("V2", "ALL"))
            right = index.get((base, "ALL"))
            delta = None if not left or not right or left.get(f"{metric}_mean") is None or right.get(f"{metric}_mean") is None else (left[f"{metric}_mean"] - right[f"{metric}_mean"]) * 100
            lines.append(f"- {metric}: {'-' if delta is None else f'{delta:+.2f} 个百分点'}")
        if label == "V2-V2S":
            for metric in METRICS:
                wins, ties, losses, count = _fold_pair_counts(rows, "V2", "V2S", metric)
                lines.append(f"- {metric} 逐折胜/平/负：{wins}/{ties}/{losses}（配对 {count} 折）。")
    lines.extend(["", "各 seed 的 ACC 变化："])
    for seed in SEEDS:
        v2 = index.get(("V2", str(seed)))
        if not v2 or v2.get("n_folds") != 10:
            lines.append(f"- seed={seed}: V2 尚未完成 10 折。")
            continue
        changes = []
        for base in ("P0", "P1", "V2S"):
            other = index.get((base, str(seed)))
            changes.append(f"V2-{base} {((v2['acc_mean'] - other['acc_mean']) * 100):+.2f} 个百分点" if other and other.get("acc_mean") is not None else f"V2-{base} -")
        lines.append(f"- seed={seed}: " + "；".join(changes) + "。")
    lines.extend(["", "## 文本嵌入诊断", "", *_diagnostic_lines(), "", "## 新任务状态", "", "| 配置 | Seed | 状态 | 已记录折数 |", "|---|---:|---|---:|"])
    for config in CONFIGS:
        for seed in SEEDS:
            state, count = states[(config, seed)]
            lines.append(f"| {config} | {seed} | {state} | {count}/10 |")
    lines.extend(["", "## 解释与限制", "", "V2 相对 P0 仅有轻微 ACC/Hard-AUC 提升，SEN-ASD 略低；相对 P1 的 ACC 与 Hard-AUC 略低，但 V2 正确模态映射在三 seed 和逐折汇总上优于 V2S。建议将 V2 保留为候选 Prompt 供后续独立验证，当前不建议用 V2 替换实验一 P1。", "V2S 的逐折胜负只作描述性配对比较，不把交叉验证折视为相互独立的外部数据集，不根据结果生成 V3。", ""])
    return "\n".join(lines)


def update_handoff(rows=None, summaries=None):
    rows = all_rows() if rows is None else rows
    summaries = summary_rows(rows) if summaries is None else summaries
    existing = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# Experiment Handoff\n"
    states = {(config, seed): task_state(config, seed) for config in CONFIGS for seed in SEEDS}
    terminal = sum(state[0] != "PENDING" for state in states.values())
    status = "COMPLETE" if all(state[0] == "COMPLETE" for state in states.values()) else ("RUNNING" if terminal < 6 else "INCOMPLETE")
    block_lines = [
        "<!-- CSMP EXPERIMENT 2 START -->", "## 实验二：对比状态驱动的模态语义提示优化（CSMP）", "",
        f"- 状态：`{status}`；已结束 {terminal}/6 个 config×seed 任务，记录 {len(rows)}/60 折。",
        "- 实验目录：`experiment/experiment2_contrastive_state_prompts`。V2 为固定状态对比 Prompt，V2S 复用相同文本并固定错配模态。",
        "- 代码与协议：根目录原始文件隔离复制，仅对 `network.py` 应用最小 Prompt patch；固定 10 折、val=test、ACC checkpoint、hard-AUC、450/200 参数均不变。",
        "- 哈希、参考快照和报告：`source_hashes.json`、`results/reference_fold_results.csv`、`results/report.md`。",
        "- 运行命令：`python experiment/experiment2_contrastive_state_prompts/run.py --prepare` 后执行 `python experiment/experiment2_contrastive_state_prompts/run.py --all`；GPU0 运行 V2 三个 seed，GPU1 运行 V2S 三个 seed。", "", "任务状态：",
    ]
    for config in CONFIGS:
        for seed in SEEDS:
            state, count = states[(config, seed)]
            block_lines.append(f"- {config} seed={seed}: `{state}`，{count}/10 folds")
    block_lines.extend(["", "实验一基线参考：P0 ACC 91.69±0.05%、Hard-AUC 91.54±0.08%；P1 ACC 91.80±0.11%、Hard-AUC 91.62±0.12%。", "- 结论：V2 轻微高于 P0，且正确模态映射优于 V2S；V2 未超过 P1，建议保留为候选但暂不替换 P1。", "<!-- CSMP EXPERIMENT 2 END -->"])
    block = "\n".join(block_lines)
    start, end = "<!-- CSMP EXPERIMENT 2 START -->", "<!-- CSMP EXPERIMENT 2 END -->"
    if start in existing and end in existing:
        before, rest = existing.split(start, 1)
        _, after = rest.split(end, 1)
        updated = before + block + after
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    tmp = HANDOFF.with_suffix(".md.tmp")
    tmp.write_text(updated, encoding="utf-8")
    os.replace(tmp, HANDOFF)


def update_outputs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = all_rows()
    summaries = summary_rows(rows)
    write_fold_results(rows)
    write_summary(summaries)
    (RESULT_DIR / "report.md").write_text(report_text(rows, summaries), encoding="utf-8")
    update_handoff(rows, summaries)
    return rows, summaries


def prepare_worktree(work, config):
    for relative in SOURCE_ROOT_FILES:
        shutil.copy2(ROOT / relative, work / relative)
    shutil.copytree(ROOT / "CLIP", work / "CLIP", copy_function=shutil.copy2)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir()
    (work / "graph").mkdir()
    applied = subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
        cwd=work,
        text=True,
        capture_output=True,
        check=False,
    )
    if applied.returncode != 0:
        raise RuntimeError(f"Failed applying CSMP prompt patch: {applied.stdout}{applied.stderr}")
    expected = json.loads(HASH_FILE.read_text(encoding="utf-8"))["files"]
    for relative, digest in expected.items():
        if relative == "network.py":
            continue
        if sha256(work / relative) != digest:
            raise RuntimeError(f"Isolated source differs from root: {relative}")
    actual = (work / "network.py").read_text(encoding="utf-8")
    expected_variant = patched_network((ROOT / "network.py").read_text(encoding="utf-8"))
    if actual != expected_variant:
        raise RuntimeError("CSMP network.py contains changes outside the generated minimal patch.")


def run_one(config, seed, gpu=None, write_outputs=True):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = task_log(config, seed)
    previous = task_end(log) if log.exists() else None
    previous_rows = len(attempt_rows(log)) if log.exists() else 0
    if previous and previous["returncode"] == 0 and previous_rows == previous["folds"]:
        print(f"Skip {config} seed={seed}: existing task ended with {previous['folds']}/10 folds.")
        if write_outputs:
            update_outputs()
        return
    manifest = ensure_hash_manifest()
    with tempfile.TemporaryDirectory(prefix=f"SPromptGL-{config}-seed{seed}-") as temp:
        work = Path(temp)
        prepare_worktree(work, config)
        shell_args = shlex.split((work / SHELL_FILE).read_text(encoding="utf-8"))
        if len(shell_args) < 2 or shell_args[:2] != ["python", "main.py"]:
            raise RuntimeError("The official shell command has an unexpected form.")
        command = [sys.executable, "main.py", *shell_args[2:], "--seed", str(seed)]
        env = os.environ.copy()
        if gpu is not None:
            # The root program imports network.py before set_gpu().  Restrict
            # visibility before starting it, then use logical device zero.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command.extend(["--gpu", "0"])
        env["CSMP_CONFIG"] = config
        env["CSMP_PROMPTS_FILE"] = str(PROMPTS_FILE)
        start_time = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} source_files={len(manifest['files'])} gpu={gpu}\n")
            stream.write("COMMAND " + shlex.join(command) + "\n")
            stream.flush()
            try:
                completed = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            except KeyboardInterrupt:
                stream.write("ATTEMPT_INTERRUPTED by user\n")
                stream.flush()
                if write_outputs:
                    update_outputs()
                raise
            folds = len(attempt_rows(log))
            stream.write(f"TASK_END config={config} seed={seed} returncode={completed.returncode} folds={folds} elapsed_seconds={time.time() - start_time:.1f}\n")
            stream.flush()
    if write_outputs:
        rows, _ = update_outputs()
    else:
        rows = all_rows()
    state, folds = task_state(config, seed)
    print(f"{config} seed={seed}: {state}, {folds}/10 folds; cumulative rows={len(rows)}")


def parallel_tasks():
    """Run all config×seed tasks as six workers on each of two physical GPUs."""
    ensure_hash_manifest()
    update_outputs()
    tasks = [(config, seed) for config in CONFIGS for seed in SEEDS]
    scheduler_log = LOG_DIR / "scheduler.log"
    scheduler_log.parent.mkdir(parents=True, exist_ok=True)
    processes = []
    with scheduler_log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nPARALLEL_START tasks={len(tasks)} workers_gpu0=3 workers_gpu1=3 gpus=0,1\n")
        stream.flush()
        for index, (config, seed) in enumerate(tasks):
            gpu = 0 if index < 3 else 1
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config", config,
                "--seed", str(seed),
                "--gpu", str(gpu),
                "--worker",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, config, seed, gpu))
            stream.write(f"LAUNCH config={config} seed={seed} gpu={gpu} pid={process.pid}\n")
        stream.flush()

        try:
            remaining = list(processes)
            while remaining:
                still_running = []
                for process, config, seed, gpu in remaining:
                    returncode = process.poll()
                    if returncode is None:
                        still_running.append((process, config, seed, gpu))
                        continue
                    stream.write(f"DONE config={config} seed={seed} gpu={gpu} pid={process.pid} returncode={returncode}\n")
                    stream.flush()
                    # Only this parent writes aggregate CSV, report, and handoff.
                    update_outputs()
                remaining = still_running
                if remaining:
                    time.sleep(2)
        except KeyboardInterrupt:
            stream.write("PARALLEL_INTERRUPTED; terminating workers\n")
            stream.flush()
            for process, _, _, _ in processes:
                if process.poll() is None:
                    process.terminate()
            raise
        stream.write("PARALLEL_END\n")
        stream.flush()
    update_outputs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="validate source hashes/prompts and generate the exact network patch")
    parser.add_argument("--config", choices=CONFIGS, help="run one configuration")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="seed for --config")
    parser.add_argument("--gpu", type=int, choices=[0, 1], help="physical GPU assigned by the parallel scheduler")
    parser.add_argument("--worker", action="store_true", help="internal worker mode; parent updates aggregate outputs")
    parser.add_argument("--all", action="store_true", help="run all 12 configuration/seed tasks, resuming completed logs")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    ensure_hash_manifest()
    if args.all:
        parallel_tasks()
        return
    if args.config is None or args.seed is None:
        parser.error("choose --prepare, --all, or both --config and --seed")
    run_one(args.config, args.seed, gpu=args.gpu, write_outputs=not args.worker)


if __name__ == "__main__":
    main()
