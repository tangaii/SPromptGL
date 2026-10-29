from __future__ import annotations
import csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
RT = HERE / ".runtime"
HIST = RT / "sources" / "historical48_3"
OLD_EXP = Path("/data/adodas/SPromptGL-main/experiment/experiment48_3_apcr_gfr")
ROOT = Path("/data/adodas/SPromptGL-main")
SLICES = ((0, 48), (48, 54), (54, 64), (64, 320))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def historical_run():
    spec = importlib.util.spec_from_file_location("apcr48_3_frozen_run", HIST / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.RT = RT
    module.SRC = RT / "sources"
    module.CTX = None
    return module

def calculate_metrics(logp, y, test):
    p = logp[test].argmax(1).detach().cpu().numpy()
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, p, labels=[0, 1]).ravel())
    onehot = np.eye(2)[target]
    return {
        "acc": float((p == target).mean()),
        "hard_auc": float(roc_auc_score(onehot, np.eye(2)[p])),
        "spe": tn / (tn + fp),
        "sen": tp / (tp + fn),
        "probability_auc": float(roc_auc_score(target, logp[test, 1].detach().cpu().numpy())),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn,
        "flips_vs_full": None,
    }

def relation_residual(apcr, parts, z, mode, wrong, split_mask, generator, modality=None):
    torch = sys.modules["torch"]
    x = torch.cat(parts, dim=1)
    edges, sim = apcr.candidates(z)
    q = apcr.q_edges(z, edges, sim)
    selected, weights = [], []
    for m in range(4):
        order = q[m].topk(16, 1).indices
        idx = edges[m].gather(1, order)
        w = q[m].gather(1, order)
        if mode in ("identity_only", "full_wrong"):
            idx = wrong[idx]
            # Preserve 16 distinct non-self senders after the site/split permutation.
            mapped_pool = wrong[edges[m]]
            rows = []
            for i, row in enumerate(idx):
                row = row.clone()
                used = set(map(int, row.detach().cpu().tolist()))
                valid_pool = mapped_pool[i]
                for k in range(row.numel()):
                    if int(row[k]) == i:
                        replacement = next((int(v) for v in valid_pool.detach().cpu().tolist()
                                            if int(v) != i and int(v) not in used), None)
                        if replacement is None:
                            raise RuntimeError(f"no legal wrong-relation sender for patient {i}")
                        used.remove(int(row[k])); row[k] = replacement; used.add(replacement)
                rows.append(row)
            idx = torch.stack(rows)
            if bool((idx == torch.arange(len(x), device=x.device).unsqueeze(1)).any()):
                raise AssertionError("wrong relation contains self sender")
            if mode == "full_wrong":
                zi = z[m].unsqueeze(1).expand(-1, idx.shape[1], -1)
                zj = z[m][idx]
                cosine = (zi * zj).sum(-1).clamp(-1, 1)
                qsel = torch.sigmoid(apcr.scorer[m](zi, zj, cosine))
                w = qsel / (qsel.sum(1, keepdim=True) + 1e-8)
        elif mode == "random":
            # Randomize within the site/split-constrained mapped candidate pool,
            # then recompute the edge evidence on those new patient identities.
            pool = wrong[edges[m]]
            recv = torch.arange(len(x), device=x.device).unsqueeze(1)
            random_scores = torch.rand(pool.shape, generator=generator, device=x.device)
            random_scores = random_scores.masked_fill(pool == recv, -torch.inf)
            idx = pool.gather(1, random_scores.topk(16, dim=1).indices)
            assert not bool((idx == recv).any())
            zi = z[m].unsqueeze(1).expand_as(z[m][idx]); zj = z[m][idx]
            cosine = (zi * zj).sum(-1).clamp(-1, 1)
            w = torch.sigmoid(apcr.scorer[m](zi, zj, cosine))
        selected.append(idx)
        weights.append(w / (w.sum(1, keepdim=True) + 1e-8))
    deltas, gates = [], []
    for m in range(4):
        msg = ((x[selected[m]] - x[:, None, :]) * weights[m].unsqueeze(-1)).sum(1)
        deltas.append(apcr.msg[m](msg))
        qmean = weights[m].mean(1)
        qstd = weights[m].std(1)
        gates.append(torch.sigmoid(apcr.gate[m](torch.stack([qmean, qstd], 1)).squeeze(-1)))
    keep = [m for m in range(4) if mode != "off" and (modality is None or modality != m)]
    delta = sum((gates[m].unsqueeze(-1) * deltas[m] for m in keep), torch.zeros_like(x))
    return x + apcr.alpha() * delta, selected, deltas, gates, weights

def intervention_forward(run, agent, gid, utils, mode, wrong, split_mask, generator, modality=None):
    torch = run.context()["torch"]
    apcr = agent.APCR
    orig_forward = apcr.forward
    orig_alpha = apcr.alpha
    if mode == "alpha_zero":
        apcr.alpha = lambda: torch.zeros((), device=next(apcr.parameters()).device)
    stats = {}
    def wrapped(parts, group=3, wrong_perm=None, attack_edges=None, feature_only=False):
        original = orig_forward(parts, group, wrong_perm, attack_edges, feature_only)
        fused, z, q, chosen = original
        if mode == "full" and modality is None:
            return original
        new_fused, selected, deltas, gates, weights = relation_residual(apcr, parts, z, mode, wrong, split_mask, generator, modality)
        stats.update({"selected": selected, "deltas": deltas, "gates": gates, "weights": weights})
        return new_fused, z, q, selected
    apcr.forward = wrapped
    try:
        return run.context()["method"].evaluate(agent, gid, utils), stats
    finally:
        apcr.forward = orig_forward
        apcr.alpha = orig_alpha

def main():
    run = historical_run()
    c = run.context()
    torch = c["torch"]
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x, y, folds = run.data()
    old_manifest = json.loads((OLD_EXP / "manifest.json").read_text())
    current_manifest = json.loads((HERE / "manifest.json").read_text())
    expected_sig = old_manifest["code_signature"]
    source_hashes = old_manifest["source_sha256"]
    for rel, digest in source_hashes.items():
        if rel.startswith(".runtime/sources/"):
            snap = RT / "sources" / Path(rel).name
        else:
            snap = HIST / rel
        assert sha(snap) == digest, (str(snap), digest, sha(snap))
    rows = []
    for fold, (tr, te) in enumerate(folds, 1):
        task = json.loads((RT / "frozen_e6_tasks" / f"g6_f{fold:02d}.json").read_text())
        assert task["signature"] == expected_sig
        anchor = run.load(RT / "anchors" / f"fold{fold:02d}.pt")
        model_file = RT / "checkpoints" / "frozen_e6" / f"g6_f{fold:02d}.best.pt"
        assert sha(model_file) == current_manifest["frozen_e6_sha256"][model_file.name]
        bundle = run.load(model_file)
        assert task["best_epoch"] == bundle["best_epoch"]
        assert task["epochs_executed"] == bundle["epochs_executed"]
        a = c["method"].make_agent(torch, c["layers"], c["network"], c["utils"], c["device"], x, y, tr, te, fold, 6, anchor, 100000 + fold)
        c["method"].load_bundle(a, bundle["model"])
        wrong = run.wrong_perm(x, tr, te)
        split_mask = torch.zeros(len(x), dtype=torch.bool, device=c["device"])
        split_mask[torch.as_tensor(tr, device=c["device"])] = True
        generator = torch.Generator(device=c["device"]); generator.manual_seed(480400 + fold)
        modes = [("full", None), ("off", None), ("alpha_zero", None)] + [("single_off", m) for m in range(4)] + [("identity_only", None), ("full_wrong", None), ("random", None)]
        per_fold = {}
        full_pred = full_prob = full_logit = None
        full_correct = None
        full_selected = None
        for mode, modality in modes:
            name = "APCR_OFF" if mode == "off" else ("ALPHA_ZERO" if mode == "alpha_zero" else (f"OFF_{('PHENO','ANAT','FUNC','CORR')[modality]}" if modality is not None else mode.upper()))
            with torch.no_grad():
                out, stats = intervention_forward(run, a, 6, c["utils"], mode, wrong, split_mask, generator, modality)
            met = calculate_metrics(out["logp"], y, te)
            pred = out["logp"][te].argmax(1).detach().cpu().numpy()
            if mode == "full":
                assert hashlib.sha256(pred.tobytes()).hexdigest() == task["prediction_sha256"], fold
                saved = bundle["prediction"]
                assert np.array_equal(pred, saved), fold
                for key in ("acc", "hard_auc", "spe", "sen", "probability_auc"):
                    assert np.isclose(met[key], task[key], atol=1e-12, rtol=0), (fold, key, met[key], task[key])
                full_pred = pred.copy()
                full_prob = out["logp"][te].exp()[:, 1].detach().cpu().numpy().copy()
                full_logit = out["logp"][te].detach().cpu().numpy().copy()
                full_correct = pred == y[te]
                full_selected = [v.detach().clone() for v in a.APCR.last.get("chosen", [])]
            else:
                met["flips_vs_full"] = int(np.count_nonzero(pred != full_pred))
            correct = pred == y[te]
            newly_correct = int((~full_correct & correct).sum()) if mode != "full" else 0
            newly_wrong = int((full_correct & ~correct).sum()) if mode != "full" else 0
            prob = out["logp"][te].exp()[:, 1].detach().cpu().numpy()
            logits = out["logp"][te].detach().cpu().numpy()
            met["delta_acc_vs_full_pp"] = 0. if mode == "full" else 100. * (met["acc"] - per_fold["FULL"]["acc"])
            met["delta_hard_auc_vs_full_pp"] = 0. if mode == "full" else 100. * (met["hard_auc"] - per_fold["FULL"]["hard_auc"])
            met["newly_correct_vs_full"] = newly_correct
            met["newly_wrong_vs_full"] = newly_wrong
            met["flips_vs_full"] = 0 if mode == "full" else int(np.count_nonzero(pred != full_pred))
            met["mean_abs_probability_change"] = 0. if mode == "full" else float(np.mean(np.abs(prob - full_prob)))
            met["mean_abs_log_probability_change"] = 0. if mode == "full" else float(np.mean(np.abs(logits - full_logit)))
            residual = []
            if stats:
                residual = [float((a.APCR.alpha() * g.unsqueeze(-1) * d).norm().detach()) for d, g in zip(stats["deltas"], stats["gates"])]
            elif mode == "full" and modality is None:
                residual = [float((a.APCR.alpha() * g.unsqueeze(-1) * delta).norm().detach()) for delta, g in zip(a.APCR.last["deltas"], a.APCR.last["gates"])]
            effective_residual = float((out["fused"] - out["base"]).norm().detach())
            base_norm = float(out["base"].norm().detach())
            selected = stats.get("selected", a.APCR.last.get("chosen", []))
            if selected:
                sel = torch.stack(selected) if isinstance(selected, (list, tuple)) else selected
                self_edges = int((sel == torch.arange(len(x), device=sel.device).view(1, -1, 1)).sum().item())
                duplicate_rows = 0
                for mat in sel:
                    duplicate_rows += sum(len(set(row.detach().cpu().tolist())) != row.numel() for row in mat)
                changed_edges = 0
                changed_patients = 0
                if mode in ("identity_only", "full_wrong", "random"):
                    base_sel = torch.stack(full_selected)
                    changed = sel != base_sel
                    changed_edges = int(changed.sum().item())
                    changed_patients = int(changed.any(0).any(1).sum().item())
            else:
                self_edges = duplicate_rows = changed_edges = changed_patients = 0
            per_fold[name] = {"group_id": 6, "fold": fold, "intervention": name, **met,
                              "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
                              "residual_norm_by_modality": residual,
                              "alpha": float(a.APCR.alpha().detach()),
                              "effective_apcr_residual_over_base_norm": effective_residual / (base_norm + 1e-8),
                              "selected_self_edges": self_edges, "duplicate_neighbor_rows": duplicate_rows,
                              "changed_edges_vs_original": changed_edges, "affected_receivers": changed_patients}
        # Reference test metrics and task SHA are checked against both task record and bundle payload.
        rows.extend(per_fold.values())
        print("FROZEN_E6_AUDITED", fold, flush=True)
    outp = HERE / "results" / "frozen_e6_interventions.csv"
    keys = list(rows[0])
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (RT / "frozen_e6_audit.json").write_text(json.dumps({"status":"PASS","folds":10,"code_signature":expected_sig,
        "intervention_rows":len(rows),"checkpoint_predictions_reproduced":True,
        "historical_source_sha256":source_hashes,"interventions":sorted({r['intervention'] for r in rows})},indent=2)+'\n')
    print("FROZEN_E6_PASS", len(rows))

if __name__ == "__main__": main()
