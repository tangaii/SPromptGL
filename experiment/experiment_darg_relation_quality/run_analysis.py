"""Frozen DARG disease-aware patient relation quality analysis."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

DPSL = Path("/model/spromptgl/experiment/DPSL")
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(DPSL / "code"))
sys.path.insert(0, str(DPSL / "code/history_base"))
sys.path.insert(0, str(DPSL / "runtime/python"))

import numpy as np
import torch

from dpsl_runtime import OVERLAP_1BASED, checkpoint_source, sha256_file
import real_model_runtime as runtime

DARG_PATH = DPSL / "code/vendor/repo/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py"
CADR_PATH = DPSL / "code/vendor/repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py"
spec = importlib.util.spec_from_file_location("formal_e744_positive_dcr", DARG_PATH)
darg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = darg
spec.loader.exec_module(darg)
darg.CADR_PATH = CADR_PATH

MODALITIES = tuple(darg.MODALITIES)
SEED_BASE = 741001
PERMUTATIONS = 20
GROUP = "E74.4-G2-Positive-DCR"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable) + "\n", encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalized_adjacency(model, adjacency):
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    return model.utils.normalize_adj(adjacency + identity)


def edge_mask(adjacency, labels, clean_test, scope):
    effective = adjacency > 0
    effective.fill_diagonal_(False)
    clean = torch.zeros(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    clean[clean_test] = True
    selected = clean[:, None] | clean[None, :] if scope == "at_least_one_clean_test" else clean[:, None] & clean[None, :]
    valid = effective & selected & torch.triu(torch.ones_like(effective, dtype=torch.bool), diagonal=1)
    cross = labels[:, None] != labels[None, :]
    return valid, valid & cross


def quality(model, original, adjacency, labels, clean_test, scope):
    normalized = normalized_adjacency(model, adjacency)
    valid, cross = edge_mask(adjacency, labels, clean_test, scope)
    weights = normalized[valid]
    denom = float(weights.sum().item()) if weights.numel() else 0.0
    numer = float(normalized[cross].sum().item()) if cross.any() else 0.0
    delta = adjacency - original
    dv, same_delta, cross_delta = delta[valid], delta[valid & ~cross], delta[cross]
    ratio = numer / denom if denom > 0 and np.isfinite(denom) else float("nan")
    return {
        "ratio": ratio, "valid_edges": int(valid.sum()), "same_edges": int((valid & ~cross).sum()), "cross_edges": int(cross.sum()),
        "weight_sum": denom, "cross_weight_sum": numer,
        "same_delta_mean": float(same_delta.mean()) if same_delta.numel() else float("nan"),
        "cross_delta_mean": float(cross_delta.mean()) if cross_delta.numel() else float("nan"),
        "enhanced_edge_ratio": float((dv > 0).float().mean()) if dv.numel() else float("nan"),
        "suppressed_edge_ratio": float((dv < 0).float().mean()) if dv.numel() else float("nan"),
        "relative_change": float(delta.norm() / original.norm().clamp_min(1e-12)),
        "finite": bool(torch.isfinite(adjacency).all() and torch.isfinite(normalized).all()),
        "nonnegative_normalized": bool((normalized >= 0).all()), "negative_adjacency_entries": int((adjacency < 0).sum()),
        "zero_denominator": bool(denom <= 0 or not np.isfinite(denom)),
    }


def fused_quality(model, originals, adjs, fusion_weights, labels, clean_test, scope):
    weights = fusion_weights / fusion_weights.sum().clamp_min(1e-12)
    return quality(model, sum(w * x for w, x in zip(weights, originals)), sum(w * x for w, x in zip(weights, adjs)), labels, clean_test, scope)


def permutation_adjs(info, fold, permutation):
    result = []
    for modality, (original, q, row, col, gate) in enumerate(zip(info["original_adjs"], info["edge_q_graph"], info["edge_rows"], info["edge_cols"], info["gates"])):
        q_used = darg._score_transform(q, modality, "score_permuted", fold, SEED_BASE + permutation * 100003)
        adjacency, _ = darg.unified_adjacency(original, row, col, q_used, gate, modality, intervention="full", fold=fold)
        result.append(adjacency)
    return result


def q_distribution(labels, info, modality):
    row, col, q = info["edge_rows"][modality], info["edge_cols"][modality], info["edge_q"][modality]
    same = labels[row] == labels[col]
    cross = ~same
    return {"candidate_edges": int(q.numel()), "q_same_mean": float(q[same].mean()) if same.any() else float("nan"), "q_cross_mean": float(q[cross].mean()) if cross.any() else float("nan"), "q_same_std": float(q[same].std(unbiased=False)) if same.any() else float("nan"), "q_cross_std": float(q[cross].std(unbiased=False)) if cross.any() else float("nan"), "q_same_edges": int(same.sum()), "q_cross_edges": int(cross.sum())}


def audit_fold(model, info, originals, adjs, neutral):
    checks = {
        "matrices_finite": all(bool(torch.isfinite(x).all()) for x in (*originals, *adjs)),
        "matrices_symmetric": all(float((x - x.T).abs().max()) == 0.0 for x in (*originals, *adjs)),
        "delta_exact": all(torch.equal(a - s, d) for s, a, d in zip(originals, adjs, info["disease_delta_adjs"])),
        "positive_dcr_exact": all(torch.equal(d.clamp_min(0), p) for d, p in zip(info["disease_delta_adjs"], info["direct_delta_adjs"])),
        "neutral_recovers_original": all(torch.equal(a, s) for a, s in zip(neutral["unified_adjs"], neutral["original_adjs"])),
        "normalized_finite": all(bool(torch.isfinite(normalized_adjacency(model, x)).all()) for x in adjs),
        "normalized_nonnegative": all(bool((normalized_adjacency(model, x) >= 0).all()) for x in adjs),
    }
    support_checks = []
    for s, a, rows, cols in zip(originals, adjs, info["edge_rows"], info["edge_cols"]):
        candidate = torch.zeros_like(s, dtype=torch.bool)
        candidate[rows, cols] = True; candidate[cols, rows] = True; candidate.fill_diagonal_(False)
        support_checks.append(bool(((a != s) & ~candidate).sum() == 0 and torch.equal(torch.diag(a), torch.diag(s))))
    checks["changed_only_on_candidate_support"] = all(support_checks)
    return checks


def plot_figure(matrix, labels, fold_rows):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    finite = matrix[np.isfinite(matrix)]
    actual_min, actual_max = float(finite.min()), float(finite.max())
    limit = max(float(np.quantile(np.abs(finite), .99)), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), gridspec_kw={"width_ratios": [1, 1.22]})
    image = axes[0].imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
    boundary = int(np.count_nonzero(labels == 0)) - .5
    axes[0].axhline(boundary, color="black", lw=.65); axes[0].axvline(boundary, color="black", lw=.65)
    axes[0].set_title("(a) Disease Relation Adjustment", fontsize=8); axes[0].set_xlabel("Fold 1 test patients", fontsize=7); axes[0].set_ylabel("Fold 1 test patients", fontsize=7); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].text(.02, 1.02, "ASD", transform=axes[0].transAxes, fontsize=6); axes[0].text(.55, 1.02, "TD", transform=axes[0].transAxes, fontsize=6)
    cbar = fig.colorbar(image, ax=axes[0], fraction=.046, pad=.03); cbar.ax.tick_params(labelsize=5); cbar.set_label("relative adjustment", fontsize=6)
    conditions = ["Original", "DARG", "Score-Permuted"]; labels = ["Original", "DARG", "Permuted"]; values = [[r["ratio"] * 100 for r in fold_rows if r["condition"] == c] for c in conditions]
    for fold in range(10): axes[1].plot(range(2), [values[0][fold], values[1][fold]], color="#b8b8b8", lw=.55, marker="o", ms=2.1)
    axes[1].scatter([0] * 10, values[0], color="#4c78a8", s=10, label="Original Graph"); axes[1].scatter([1] * 10, values[1], color="#1b9e77", s=10, label="DARG Graph")
    axes[1].errorbar(2, np.mean(values[2]), yerr=[[np.mean(values[2]) - np.min(values[2])], [np.max(values[2]) - np.mean(values[2])]], fmt="o", color="#d95f02", ms=3.2, capsize=2, label="Score-Permuted DARG")
    axes[1].set_xticks(range(3), labels, fontsize=6); axes[1].set_ylabel("Cross-Disease Propagation Weight Ratio (%)", fontsize=7); axes[1].set_title("(b) Cross-Disease Edge Weight", fontsize=8); axes[1].grid(axis="y", color="#ddd", lw=.45); axes[1].tick_params(axis="y", labelsize=6); axes[1].legend(frameon=False, fontsize=5.8, loc="upper left")
    fig.tight_layout(pad=.7); FIGURES.mkdir(parents=True, exist_ok=True); fig.savefig(FIGURES / "darg_relation_quality.pdf", bbox_inches="tight"); fig.savefig(FIGURES / "darg_relation_quality.png", dpi=600, bbox_inches="tight"); plt.close(fig)
    return {"actual_min": actual_min, "actual_max": actual_max, "display_limit": limit, "clipped_values": int(np.count_nonzero(np.abs(finite) > limit)), "finite_values": int(finite.size)}


def main():
    started = time.time(); RESULTS.mkdir(parents=True, exist_ok=True)
    sys.modules.setdefault("run", runtime.local_module("history_base/run.py"))
    rows, audits, cache = [], [], {}
    provenance = {"experiment": "DARG Disease-Aware Patient Relation Quality Analysis", "authority": str(ROOT), "model": "E82.4.6.10-G1 / E82.4.6-G2 SCMR-DA-SRPE / E74.4-G2 Positive-DCR", "labels": {"ASD": 0, "TD": 1}, "subjects": 871, "seed": 0, "permutations": PERMUTATIONS, "darg_sha256": sha256_file(DARG_PATH), "cadr_sha256": sha256_file(CADR_PATH), "checkpoint_sources": {}, "folds": {}, "normalization": "formal model.utils.normalize_adj(A + I)"}
    for fold in range(1, 11):
        print(f"Fold {fold:02d}: restoring audited Best", flush=True)
        ctx, engine, saved, _ = runtime.restore(fold, "best"); state = ctx["state"]; out, _ = engine.evaluate(ctx)
        expected = saved["evaluation"]; pred_exact = all(np.array_equal(out[k], expected[k]) for k in ("logits", "probability", "prediction"))
        proof = runtime.local_module("history_base/audit.py"); proof.run = engine; independent, _ = proof.independent_eval(ctx); independent_exact = all(np.array_equal(out[k], independent[k]) for k in ("logits", "probability", "prediction"))
        model = state["model"]
        with torch.no_grad():
            _, _, info = darg.rudag_forward(model, state["parts"], state["train_pairs"], GROUP, fold=fold, collect_diagnostics=True, permutation_seed_base=SEED_BASE)
            _, _, neutral = darg.rudag_forward(model, state["parts"], state["train_pairs"], GROUP, fold=fold, intervention="disease_neutral", collect_diagnostics=True, permutation_seed_base=SEED_BASE)
        originals, adjs = info["original_adjs"], info["unified_adjs"]; labels = torch.as_tensor(state["labels_np"], dtype=torch.long, device=adjs[0].device)
        overlap = np.asarray([x - 1 for x in OVERLAP_1BASED.get(fold, ())], dtype=np.int64); overlap_set = set(overlap.tolist()); clean_test = np.asarray([x for x in state["test_np"] if x not in overlap_set], dtype=np.int64)
        provenance["checkpoint_sources"][str(fold)] = {"path": str(runtime.checkpoint_path(fold, "best")), "source": checkpoint_source(fold, "best")[0], "sha256": sha256_file(runtime.checkpoint_path(fold, "best")), "best_epoch": int(saved["best_epoch"]), "fold": int(saved["fold"]), "seed": int(saved["seed"])}
        provenance["folds"][str(fold)] = {"test_count": len(state["test_np"]), "clean_test_count": len(clean_test), "overlap_1based": overlap.tolist(), "prediction_exact": pred_exact, "independent_evaluator_exact": independent_exact, "test_order_sha256": sha_bytes(np.asarray(state["test_np"], dtype=np.int64).tobytes())}
        audit = audit_fold(model, info, originals, adjs, neutral); audit.update({"fold": fold, "formal_prediction_exact": pred_exact, "independent_evaluator_exact": independent_exact, "checkpoint_sha_verified": True, "labels_used_only_after_graph": True, "clean_test_excludes_overlap": not np.intersect1d(clean_test, overlap).size})
        audits.append(audit); fusion_weights = torch.sigmoid(model.fusion_weight.detach()).reshape(-1)
        cache[fold] = {"model": model, "info": info, "originals": originals, "adjs": adjs, "labels": labels, "clean_test": clean_test, "fusion_weights": fusion_weights}
        for modality, name in enumerate(MODALITIES):
            qstats = q_distribution(labels, info, modality)
            for scope in ("at_least_one_clean_test", "clean_test_test"):
                original_quality = quality(model, originals[modality], originals[modality], labels, clean_test, scope); darg_quality = quality(model, originals[modality], adjs[modality], labels, clean_test, scope); permutations = [quality(model, originals[modality], permutation_adjs(info, fold, p)[modality], labels, clean_test, scope) for p in range(PERMUTATIONS)]
                def make(condition, item, repeat_sd=0.): return {"fold": fold, "modality": name, "scope": scope, "condition": condition, **item, "ratio_percent": item["ratio"] * 100 if np.isfinite(item["ratio"]) else float("nan"), "repeat_sd_ratio": repeat_sd, **qstats}
                rows += [make("Original", original_quality), make("DARG", darg_quality)]
                mean_item = {key: float(np.nanmean([x[key] for x in permutations])) for key in permutations[0]}; rows.append(make("Score-Permuted", mean_item, float(np.nanstd([x["ratio"] for x in permutations]))))
                q = info["edge_q_graph"][modality]; audit[f"{name}_{scope}_permuted_q_multiset"] = all(torch.equal(torch.sort(q).values, torch.sort(darg._score_transform(q, modality, "score_permuted", fold, SEED_BASE + p * 100003)).values) for p in range(PERMUTATIONS))
        for scope in ("at_least_one_clean_test", "clean_test_test"):
            for condition, item in (("Original", fused_quality(model, originals, originals, fusion_weights, labels, clean_test, scope)), ("DARG", fused_quality(model, originals, adjs, fusion_weights, labels, clean_test, scope))): rows.append({"fold": fold, "modality": "FUSED", "scope": scope, "condition": condition, **item, "ratio_percent": item["ratio"] * 100 if np.isfinite(item["ratio"]) else float("nan"), "fusion_weights": ";".join(f"{float(x):.8g}" for x in fusion_weights.cpu())})
        print(f"Fold {fold:02d}: prediction={'PASS' if pred_exact else 'FAIL'}", flush=True)
    write_csv(RESULTS / "graph_quality_by_fold.csv", rows)
    audit_pass = all(all(v is True for v in audit.values() if isinstance(v, bool)) for audit in audits)
    write_json(RESULTS / "audit_status.json", {"status": "PASS" if audit_pass else "FAIL", "folds": audits, "provenance": provenance})
    fold1 = cache[1]; order = np.asarray(sorted(fold1["clean_test"].tolist(), key=lambda x: (int(fold1["labels"][x]), int(x))), dtype=np.int64); base = fold1["originals"][3].detach().cpu().numpy(); delta = (fold1["adjs"][3] - fold1["originals"][3]).detach().cpu().numpy(); matrix = np.full((len(order), len(order)), np.nan, dtype=np.float32); sub_base, sub_delta = base[np.ix_(order, order)], delta[np.ix_(order, order)]; support = (sub_base > 0) & ~np.eye(len(order), dtype=bool); matrix[support] = (sub_delta[support] / sub_base[support]).astype(np.float32); np.savez_compressed(RESULTS / "corr_fold1_heatmap.npz", matrix=matrix, labels=fold1["labels"].detach().cpu().numpy()[order], order=order)
    display = plot_figure(matrix, fold1["labels"].detach().cpu().numpy()[order], [r for r in rows if r["modality"] == "CORR" and r["scope"] == "at_least_one_clean_test"])
    summary = {}
    for name in MODALITIES:
        summary[name] = {}
        for scope in ("at_least_one_clean_test", "clean_test_test"):
            selected = [r for r in rows if r["modality"] == name and r["scope"] == scope]; grouped = {c: [r for r in selected if r["condition"] == c] for c in ("Original", "DARG", "Score-Permuted")}; out = {}
            for condition, group in grouped.items(): out[condition] = {"mean_percent": float(np.nanmean([r["ratio_percent"] for r in group])), "sd_percent": float(np.nanstd([r["ratio_percent"] for r in group])), "values_percent": [r["ratio_percent"] for r in group]}
            delta_values = [a - b for a, b in zip(out["DARG"]["values_percent"], out["Original"]["values_percent"])]
            out.update({"paired_delta_percent": delta_values, "improved_folds": sum(x < 0 for x in delta_values), "unchanged_folds": sum(x == 0 for x in delta_values), "declined_folds": sum(x > 0 for x in delta_values)}); summary[name][scope] = out
    write_json(RESULTS / "summary.json", {"summary": summary, "heatmap_display": display, "runtime_seconds": time.time() - started, "formal_source_sha256": {"DPSL_manifest": sha256_file(DPSL / "manifest.json"), "DPSL_runtime": sha256_file(DPSL / "code/real_model_runtime.py"), "DARG": sha256_file(DARG_PATH), "CADR": sha256_file(CADR_PATH)}})
    write_json(RESULTS / "run_manifest.json", {**provenance, "runtime_seconds": time.time() - started, "outputs": {"csv": str(RESULTS / "graph_quality_by_fold.csv"), "audit": str(RESULTS / "audit_status.json"), "summary": str(RESULTS / "summary.json"), "figure_pdf": str(FIGURES / "darg_relation_quality.pdf"), "figure_png": str(FIGURES / "darg_relation_quality.png"), "heatmap": str(RESULTS / "corr_fold1_heatmap.npz")}})
    print(f"Completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__": main()
