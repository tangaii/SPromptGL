#!/usr/bin/env python3
"""Run MASPE prompts with isolated copies of the root SPromptGL implementation."""

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
LOG_DIR = EXP / "logs"
RESULT_DIR = EXP / "results"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS = ["P0", "P1", "P2", "P3"]
SEEDS = [0, 1, 2]
PAPER = {"acc_mean": 92.30, "acc_sd": 2.58, "hard_auc_mean": 92.22, "hard_auc_sd": 2.57}
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


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


def prompt_sentences(config, prompts):
    order = prompts["modality_order"]
    if order != MODALITIES:
        raise ValueError(f"Prompt modality order must match ABIDE input order: {order}")
    if config == "P2":
        mapping = prompts["p2"]["mapping"]
        if set(mapping) != set(order) or any(mapping[key] == key for key in order):
            raise ValueError("P2 must use a fixed derangement over the four modalities.")
        source = prompts["p1"]
    elif config == "P1":
        mapping, source = {key: key for key in order}, prompts["p1"]
    elif config == "P3":
        mapping, source = {key: key for key in order}, prompts["p3"]
    else:
        raise ValueError(config)
    result = []
    for modality in order:
        packs = source[mapping[modality]]["themes"]
        if len(packs) != 2:
            raise ValueError(f"{config}/{modality} must have two prompt themes.")
        for pack in packs:
            texts = [row["text"] for row in pack["prompts"]]
            if len(texts) != 4:
                raise ValueError(f"{config}/{modality}/{pack['theme']} must have four prompts.")
            result.extend(texts)
    if len(result) != 32:
        raise ValueError(f"{config} must contain exactly 32 prompt sentences.")
    return result


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
    """Encode the fixed semantic prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        prompt_tokens = text
        config = os.environ.get("MASPE_CONFIG")
        if config:
            with open(os.environ["MASPE_PROMPTS_FILE"], encoding="utf-8") as stream:
                prompt_data = json.load(stream)
            if config == "P2":
                source = prompt_data["p1"]
                mapping = prompt_data["p2"]["mapping"]
            else:
                source = prompt_data[config.lower()]
                mapping = {key: key for key in prompt_data["modality_order"]}
            sentences = []
            for modality in prompt_data["modality_order"]:
                for theme in source[mapping[modality]]["themes"]:
                    sentences.extend(item["text"] for item in theme["prompts"])
            if len(sentences) != 32:
                raise ValueError("MASPE variants require 32 ordered prompt sentences.")
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

    for config in ("P1", "P2", "P3"):
        encoded = clip.tokenize(prompt_sentences(config, prompts))
        if tuple(encoded.shape) != (32, 77):
            raise RuntimeError(f"Unexpected {config} CLIP token shape: {tuple(encoded.shape)}")


def prepare():
    ensure_hash_manifest()
    prompts = load_prompts()
    for config in ("P1", "P2", "P3"):
        prompt_sentences(config, prompts)
    original = (ROOT / "network.py").read_text(encoding="utf-8")
    patch_text, variant = expected_patch(original)
    if PATCH_FILE.exists():
        if PATCH_FILE.read_text(encoding="utf-8") != patch_text:
            raise RuntimeError("prompt_patch.diff does not match the generated minimal network.py patch.")
    else:
        PATCH_FILE.write_text(patch_text, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="maspe_patch_check_") as temp:
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
    validate_tokenization(prompts)
    ensure_archive_handoff()
    update_outputs()
    print(f"Prepared experiment. Root source hashes: {len(json.loads(HASH_FILE.read_text())['files'])} files.")


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
    matches = list(re.finditer(r"TASK_END config=(P[0-3]) seed=(\d+) returncode=(-?\d+) folds=(\d+)", text))
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


def summary_rows(rows):
    grouped = {(config, seed): [] for config in CONFIGS for seed in SEEDS}
    for row in rows:
        grouped[(row["config"], row["seed"])].append(row)
    output = []
    for config in CONFIGS:
        seed_groups = []
        for seed in SEEDS:
            values = grouped[(config, seed)]
            row = {"scope": "seed_folds", "config": config, "seed": seed, "n_seeds": 1, "n_folds": len(values)}
            for metric in METRICS:
                metric_values = [item[metric] for item in values if item.get(metric) is not None]
                row[f"{metric}_mean"] = statistics.mean(metric_values) if metric_values else None
                row[f"{metric}_fold_sd"] = statistics.pstdev(metric_values) if metric_values else None
                row[f"{metric}_seed_sd"] = None
            output.append(row)
            state, fold_count = task_state(config, seed)
            if state == "COMPLETE" and fold_count == 10:
                seed_groups.append(row)
        if seed_groups:
            aggregate = {
                "scope": "seed_aggregate", "config": config, "seed": "ALL",
                "n_seeds": len(seed_groups), "n_folds": sum(int(item["n_folds"]) for item in seed_groups),
            }
            for metric in METRICS:
                means = [item[f"{metric}_mean"] for item in seed_groups]
                aggregate[f"{metric}_mean"] = statistics.mean(means)
                aggregate[f"{metric}_fold_sd"] = None
                aggregate[f"{metric}_seed_sd"] = statistics.pstdev(means) if len(means) > 1 else None
            output.append(aggregate)
    return output


def write_summary(rows):
    path = RESULT_DIR / "summary.csv"
    columns = ["scope", "config", "seed", "n_seeds", "n_folds"]
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
    for relative in SOURCE_ROOT_FILES[:-1]:
        shutil.copy2(ROOT / relative, work / relative)
    shutil.copy2(ROOT / SHELL_FILE, work / SHELL_FILE)
    shutil.copytree(ROOT / "CLIP", work / "CLIP", copy_function=shutil.copy2)
    (work / "data").symlink_to(ROOT / "data", target_is_directory=True)
    (work / "attn").mkdir()
    (work / "graph").mkdir()
    if config != "P0":
        applied = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH_FILE)],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0:
            raise RuntimeError(f"Failed applying prompt patch: {applied.stdout}{applied.stderr}")
    expected = json.loads(HASH_FILE.read_text(encoding="utf-8"))["files"]
    for relative, digest in expected.items():
        if config != "P0" and relative == "network.py":
            continue
        if sha256(work / relative) != digest:
            raise RuntimeError(f"Isolated source differs from root: {relative}")
    if config != "P0":
        actual = (work / "network.py").read_text(encoding="utf-8")
        expected_variant = patched_network((ROOT / "network.py").read_text(encoding="utf-8"))
        if actual != expected_variant:
            raise RuntimeError("P1-P3 network.py contains changes outside the generated minimal patch.")


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
        command = [sys.executable, "main.py", *shell_args[2:]]
        if seed != 0:
            command.extend(["--seed", str(seed)])
        env = os.environ.copy()
        if gpu is not None:
            # The root program imports network.py before set_gpu().  Restrict
            # visibility before starting it, then use logical device zero.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command.extend(["--gpu", "0"])
        env.pop("MASPE_CONFIG", None)
        env.pop("MASPE_PROMPTS_FILE", None)
        if config != "P0":
            env["MASPE_CONFIG"] = config
            env["MASPE_PROMPTS_FILE"] = str(PROMPTS_FILE)
        start_time = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} source_files={len(manifest['files'])}\n")
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
        stream.write(f"\nPARALLEL_START tasks={len(tasks)} workers_per_gpu=6 gpus=0,1\n")
        stream.flush()
        for index, (config, seed) in enumerate(tasks):
            gpu = 0 if index < 6 else 1
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
