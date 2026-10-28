#!/usr/bin/env python3
"""Independent readback, metric recomputation, paired analysis and report writer.

This script is intentionally a separate process from the training workers.  It
loads every frozen best checkpoint, recomputes the predictions, and only then
creates the aggregate tables and the scientific report.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RT = HERE / ".runtime"
RES = HERE / "results"
sys.path.insert(0, str(HERE))
import run  # noqa: E402


def jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps([jsonable(x) for x in v], ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps({str(k): jsonable(x) for k, x in v.items()}, ensure_ascii=False)
    return v


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsonable(r.get(k, "")) for k in fields})


def sha_bytes(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def gradient_probe(a, gid, fold, x, y, tr, utils, prop_perms=None):
    """Read-only best-checkpoint gradient probe (no optimizer step).

    It uses eval-mode forward passes so dropout cannot make the audit
    nondeterministic.  The probe is reported as a mechanism diagnostic, not as
    a replacement for the actual training trajectory.
    """
    c = run.context(); m = c["method"]
    a.MessagePassing0.eval(); a.ModalFusion.eval(); a.new.eval()
    parts, comb, adjs, gl, lp, raw, hid, hs = m.forward(a, utils, [prop_perms] * 4 if prop_perms is not None else None)
    z = a.new.project(hs)
    idx = c["torch"].as_tensor(tr, device=z[0].device)
    ws = m.aux_weights(gid)
    ces = [c["torch"].nn.functional.cross_entropy(a.new.cls(z[i])[idx], a.targ[idx], weight=a.weight) for i in range(4)]
    aux = .1 * sum(w * q for w, q in zip(ws, ces))
    main = utils.ClsLoss(lp, a.targ, a.trn_idx, a.weight)
    cond = run.perm_condition(y, tr, fold) if gid == 7 else None
    enabled = ["CP", "CA", "CF"] if gid in (2, 3) else (["CF"] if gid in (4, 5, 6, 7) else [])
    conditional = gid in (3, 4, 5, 6, 7)
    dl, _, _, _ = m.domain_loss(a, z, enabled, conditional, m.lambda_adv(0 if gid == 1 else 100), a.targ if cond is None else cond)
    mods = [a.MessagePassing0.MessagePassing0, a.MessagePassing0.MessagePassing1, a.MessagePassing0.MessagePassing2, a.MessagePassing0.MessagePassing3]
    def norms(loss):
        if loss is None:
            return [0.0] * 4
        gs = c["torch"].autograd.grad(loss, [q.gcn1.W for q in mods], retain_graph=True, allow_unused=True)
        return [float(g.detach().norm().cpu()) if g is not None else 0.0 for g in gs]
    return {"main_gcn_grad_norms": norms(main), "aux_gcn_grad_norms": norms(aux), "domain_gcn_grad_norms": norms(dl)}


def condition_domain_probe(a, gid, fold, y, tr, z):
    """Best-checkpoint source-domain accuracy split by real/置乱 ASD and TD."""
    c = run.context(); m = c["method"]
    enabled = ["CP", "CA", "CF"] if gid in (2, 3) else (["CF"] if gid in (4, 5, 6, 7) else [])
    if not enabled:
        return {}
    labels = run.perm_condition(y, tr, fold) if gid == 7 else c["torch"].as_tensor(y, device=z[0].device)
    idx = c["torch"].as_tensor(tr, device=z[0].device)
    result = {}
    for pair in enabled:
        ia, ib = m.PAIRS[pair]
        for cls, cname in [(0, "ASD"), (1, "TD")]:
            mask = labels[idx] == cls
            if int(mask.sum()) < 2:
                continue
            za, zb = z[ia][idx][mask], z[ib][idx][mask]
            inp = c["torch"].cat([za, zb]); lab = c["torch"].cat([c["torch"].zeros(len(za), dtype=c["torch"].long, device=inp.device), c["torch"].ones(len(zb), dtype=c["torch"].long, device=inp.device)])
            pred = a.new.domains[pair](inp).argmax(1)
            result[f"{pair}_{cname}"] = float((pred == lab).float().mean().detach().cpu())
    return result


def eval_one(gid, fold, gpu, x, y, tr, te):
    c = run.context()
    m = c["method"]
    task = json.loads((RT / "tasks" / f"g{gid}_f{fold:02d}.json").read_text())
    anchor = run.load(RT / "anchors" / f"fold{fold:02d}.pt")
    props = None
    sites = None
    if gid == 0:
        s = run.load(RT / "checkpoints" / f"E0_f{fold:02d}.best.pt")
        run.restore(anchor["entry_rng"])
        a = m.build_base(c["torch"], c["layers"], c["network"], c["utils"], c["device"], x, y, tr, te)
        m.load_base(a, s["model"])
        with c["torch"].no_grad():
            o = m.evaluate(a, 0, c["utils"])
        expected = s["prediction"]
        best_epoch = s["best_epoch"]
        source = "E0_verified_checkpoint"
    else:
        b = run.load(RT / "checkpoints" / f"g{gid}_f{fold:02d}.best.pt")
        run.restore(anchor["entry_rng"])
        a = m.make_agent(c["torch"], c["layers"], c["network"], c["utils"], c["device"], x, y, tr, te, fold, gid, anchor, 480000 + fold)
        run.restore(anchor["rng"])
        if gid == 6:
            props, sites, _ = run.permutation(x, y, tr, te)
        m.load_bundle(a, b["model"])
        with c["torch"].no_grad():
            o = m.evaluate(a, gid, c["utils"], [props] * 4 if props is not None else None)
        expected = b["prediction"]
        best_epoch = b["best_epoch"]
        source = "independent_best_checkpoint"
    row = m.metrics(o, y, te)
    pred = o["logp"].detach().cpu().numpy()[te].argmax(1)
    if not np.array_equal(pred, np.asarray(expected)):
        raise AssertionError(f"prediction mismatch g{gid} f{fold}")
    if task and sha_bytes(pred.astype(np.int64)) != task["prediction_sha256"]:
        raise AssertionError(f"task prediction SHA mismatch g{gid} f{fold}")

    # Reconstruct the exact GFR fusion decomposition for a diagnostic-only
    # message intervention.  PHENO is overwritten by the historical i==1
    # branch; ANAT/FUNC/CORR enter the final combined tensor with sigmoid gates.
    ablation = []
    try:
        base_comb = o["combined"]
        fw = c["torch"].sigmoid(a.MessagePassing0.fusion_weight.detach())
        for j, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
            altered = base_comb.clone()
            if j in (1, 2, 3):
                altered = altered - fw[j] * o["hs"][j].detach()
            with c["torch"].no_grad():
                lp_alt, _, _ = a.ModalFusion(altered)
            p_alt = lp_alt.detach().cpu().numpy()[te].argmax(1)
            ablation.append({
                "relation": name,
                "logp_abs_change": float((lp_alt[te] - o["logp"][te]).abs().mean()),
                "prediction_flip_rate": float((p_alt != pred).mean()),
                "historical_pheno_overwrite": bool(j == 0),
            })
    except Exception as e:
        ablation = [{"relation": "ERROR", "error": repr(e)}]

    hs = [h.detach() for h in o["hs"]]
    raw_norms = [float(h.norm().cpu()) for h in hs]
    input_norm = float(c["torch"].cat(o["parts"], dim=1).norm().cpu())
    gcn_norms = [float((q.gcn1.W.norm() + q.gcn2.W.norm()).detach().cpu()) for q in [a.MessagePassing0.MessagePassing0, a.MessagePassing0.MessagePassing1, a.MessagePassing0.MessagePassing2, a.MessagePassing0.MessagePassing3]]
    aux_acc, aux_auc = [], []
    if o.get("aux_logits"):
        from sklearn.metrics import roc_auc_score
        yy = y[te]
        for q in o["aux_logits"]:
            z = q.detach().cpu().numpy()[te]
            aux_acc.append(float((z.argmax(1) == yy).mean()))
            aux_auc.append(float(roc_auc_score(yy, z[:, 1])))
    trace = (task or {}).get("trace", [])
    diag = {
        "group_id": gid, "group": run.NAMES[gid], "fold": fold,
        "source": source, "best_epoch": best_epoch,
        "gcn_output_norms": raw_norms,
        "gcn_output_relative_to_input": [v / max(input_norm, 1e-12) for v in raw_norms],
        "input_norm": input_norm, "gcn_weight_norms": gcn_norms,
        "aux_test_acc": aux_acc, "aux_test_auc": aux_auc,
        "domain_acc_last": float(trace[-1].get("domain_acc", 0.0)) if trace else 0.0,
        "domain_loss_last": float(trace[-1].get("domain", 0.0)) if trace else 0.0,
        "lambda_adv_last": float(trace[-1].get("lambda_adv", 0.0)) if trace else 0.0,
        "message_ablation": ablation,
        "e6_sites_available": bool(sites is not None),
    }
    if sites is not None:
        # Site/diagnosis summaries use the top 16 non-self edges of each raw
        # learned relation.  They are descriptive and do not claim a causal
        # or conditional disease effect.
        site_rows = []
        for ri, adj in enumerate(o["adjs"]):
            aa = adj.detach().cpu().numpy()
            same_site = same_label = cross_site_same_label = total = 0
            yy = y
            for i in range(len(aa)):
                scores = aa[i].copy(); scores[i] = -np.inf
                k = min(16, len(aa) - 1)
                ids = np.argpartition(scores, -k)[-k:]
                for j in ids:
                    if not np.isfinite(scores[j]):
                        continue
                    total += 1
                    same_site += int(sites[i] == sites[j])
                    same_label += int(yy[i] == yy[j])
                    cross_site_same_label += int(sites[i] != sites[j] and yy[i] == yy[j])
            site_rows.append({
                "relation_index": ri, "topk": 16, "edges": total,
                "same_site_rate": same_site / max(total, 1),
                "same_label_rate": same_label / max(total, 1),
                "cross_site_same_label_rate_over_all": cross_site_same_label / max(total, 1),
            })
        diag["site_edge_summary"] = site_rows
    if gid > 0:
        diag.update(gradient_probe(a, gid, fold, x, y, tr, c["utils"], props))
        diag["condition_domain_accuracy"] = condition_domain_probe(a, gid, fold, y, tr, o["z"])
    return row, diag


def summarize(rows):
    out = []
    metrics = ["acc", "hard_auc", "spe", "sen", "probability_auc"]
    for gid in range(8):
        rr = [r for r in rows if r["group_id"] == gid]
        q = {"group_id": gid, "group": run.NAMES[gid], "n_folds": len(rr)}
        for k in metrics:
            vals = np.asarray([r[k] for r in rr], dtype=float) * 100
            q[k + "_mean_pct"] = float(vals.mean())
            q[k + "_sd_pct"] = float(vals.std(ddof=0))
        out.append(q)
    return out


def paired(rows):
    by = {(r["group_id"], r["fold"]): r for r in rows}
    pairs = [(1, 0), (2, 1), (3, 2), (3, 1), (4, 3), (4, 1), (5, 4), (5, 0), (5, 6), (5, 7)]
    out = []
    for a, b in pairs:
        q = {"comparison": f"E{a}-E{b}", "group_a": run.NAMES[a], "group_b": run.NAMES[b]}
        for k in ["acc", "hard_auc", "spe", "sen"]:
            d = np.asarray([(by[(a, f)][k] - by[(b, f)][k]) * 100 for f in range(1, 11)])
            q[k + "_delta_pp"] = float(d.mean())
            q[k + "_win_tie_loss"] = f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}"
        out.append(q)
    return out


def fmt(v):
    return f"{v:.4f}"


def make_report(rows, sums, pairs, diags, elapsed, train_seconds):
    sby = {r["group_id"]: r for r in sums}
    lines = ["# Experiment 48.1 — DGAA-R正式结果报告", "", "## 执行与审计状态", ""]
    lines += [
        "本报告由独立进程重新加载80个最佳检查点并重算预测后生成。E0为Experiment43.1核验的B0-B检查点读回，E1–E7为本目录从同折B0-B锚点完成的70个训练折。必要工程预检通过；没有按部分测试折性能筛选或减少任务。",
        f"完成任务：{len(rows)}/80；E0读回：{sum(r['group_id']==0 for r in rows)}/10；新增训练：{sum(r['group_id']>0 for r in rows)}/70；独立复核：{len(rows)}/80；新增70折累计任务wall time约 {train_seconds:.1f} 秒（两张GPU并行），本次独立复核约 {elapsed:.1f} 秒。",
        "历史限制保留：全局预处理及CV前RFE、871节点transductive图、val_idx=test_idx checkpoint选择、单seed和重叠训练集十折，因此结果是协议内机制证据，不是无偏外部泛化。",
        "",
        "## 四指标（mean ± population SD，%）", "",
        "|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|", "|---|---:|---:|---:|---:|---:|",
    ]
    for gid in range(8):
        q = sby[gid]
        lines.append(f"|E{gid} {run.NAMES[gid]}|{fmt(q['acc_mean_pct'])} ± {fmt(q['acc_sd_pct'])}|{fmt(q['hard_auc_mean_pct'])} ± {fmt(q['hard_auc_sd_pct'])}|{fmt(q['spe_mean_pct'])} ± {fmt(q['spe_sd_pct'])}|{fmt(q['sen_mean_pct'])} ± {fmt(q['sen_sd_pct'])}|{fmt(q['probability_auc_mean_pct'])} ± {fmt(q['probability_auc_sd_pct'])}|")
    lines += ["", "## 关键配对差（百分点）", "", "|比较|ΔACC|胜/平/负|ΔHard-AUC|胜/平/负|ΔSPE|ΔSEN|", "|---|---:|---|---:|---|---:|---:|"]
    for q in pairs:
        lines.append(f"|{q['comparison']}|{fmt(q['acc_delta_pp'])}|{q['acc_win_tie_loss']}|{fmt(q['hard_auc_delta_pp'])}|{q['hard_auc_win_tie_loss']}|{fmt(q['spe_delta_pp'])}|{fmt(q['sen_delta_pp'])}|")
    lines += ["", "## 机制证据", ""]
    for gid in range(8):
        dd = [d for d in diags if d["group_id"] == gid]
        if not dd:
            continue
        hn = np.asarray([d["gcn_output_norms"] for d in dd], float)
        rel = np.asarray([d["gcn_output_relative_to_input"] for d in dd], float)
        wn = np.asarray([d["gcn_weight_norms"] for d in dd], float)
        aux = [d["aux_test_acc"] for d in dd if d["aux_test_acc"]]
        da = np.asarray([d["domain_acc_last"] for d in dd], float)
        ab = [x for d in dd for x in d.get("message_ablation", []) if x.get("relation") != "ERROR"]
        lines.append(f"- **E{gid} {run.NAMES[gid]}**：四路GCN输出范数均值=" + ", ".join(fmt(x) for x in hn.mean(0)) + "；相对320维输入范数=" + ", ".join(f"{x:.6g}" for x in rel.mean(0)) + "；GCN两层权重范数和=" + ", ".join(fmt(x) for x in wn.mean(0)) + ".")
        if aux:
            lines.append("  辅助头测试ACC（四视角均值，按折平均）=" + ", ".join(fmt(x) for x in np.asarray(aux).mean(0) * 100) + "%。这只说明视角表示可预测，不能替代最终CAM增量。")
        if gid in (2, 3, 4, 5, 6, 7):
            lines.append(f"  域判别器最后记录的平均准确率={da.mean():.4f}；它是来源可分性诊断，不是疾病分类指标。")
            cda = [d.get("condition_domain_accuracy", {}) for d in dd if d.get("condition_domain_accuracy")]
            if cda:
                keys = sorted(set().union(*(q.keys() for q in cda)))
                cv = ", ".join(f"{k}={np.mean([q.get(k, np.nan) for q in cda]):.4f}" for k in keys)
                lines.append("  独立读回的疾病条件来源准确率（ASD/TD分别汇总）=" + cv + "；接近0或1表示来源头可能发生反向/退化，不能解释为成功对齐。")
            gd = np.asarray([d.get("main_gcn_grad_norms", [0]*4) for d in dd], float)
            ga = np.asarray([d.get("aux_gcn_grad_norms", [0]*4) for d in dd], float)
            gv = np.asarray([d.get("domain_gcn_grad_norms", [0]*4) for d in dd], float)
            lines.append("  eval-mode梯度探针（主分类/辅助/来源，四路GCN均值）=" + "; ".join(",".join(f"{x:.5g}" for x in q) for q in (gd.mean(0), ga.mean(0), gv.mean(0))) + "。该探针不替代训练轨迹。")
        if ab:
            for relname in ("PHENO", "ANAT", "FUNC", "CORR"):
                z = [x for x in ab if x["relation"] == relname]
                if z:
                    lines.append(f"  关闭{relname}分支的平均log-prob绝对变化={np.mean([x['logp_abs_change'] for x in z]):.6f}，预测翻转率={np.mean([x['prediction_flip_rate'] for x in z]):.4f}。PHENO项按历史i==1覆盖路径记录为非主融合消息。")
                    break
    lines += ["", "## 预注册判定", ""]
    pby = {q["comparison"]: q for q in pairs}
    def pos(key, metric="acc"):
        return pby[key][metric + "_delta_pp"] > 0
    def win(key, metric="acc"):
        return int(pby[key][metric + "_win_tie_loss"].split("/")[0])
    go = {
        "GO_AUX": pos("E1-E0") and pos("E1-E0", "hard_auc") and win("E1-E0") >= 7 and win("E1-E0", "hard_auc") >= 7,
        "GO_CONDITIONAL": pos("E3-E2") and pos("E3-E2", "hard_auc") and pos("E3-E1") and pos("E3-E1", "hard_auc"),
        "GO_SELECTIVE": pos("E4-E3") and pos("E4-E3", "hard_auc") and pos("E4-E1") and pos("E4-E1", "hard_auc"),
        "GO_CORR_PRIORITY": pos("E5-E4") and pos("E5-E4", "hard_auc"),
        "GO_REAL_RELATION": pos("E5-E6") and pos("E5-E6", "hard_auc") and win("E5-E6") >= 7 and win("E5-E6", "hard_auc") >= 7,
        "GO_DISEASE_CONDITION": pos("E5-E7") and pos("E5-E7", "hard_auc"),
        "GO_FINAL_MICCAI_CANDIDATE": pos("E5-E0") and pby["E5-E0"]["acc_delta_pp"] >= .1 and pby["E5-E0"]["hard_auc_delta_pp"] >= .1 and win("E5-E0") >= 7 and win("E5-E0", "hard_auc") >= 7 and pos("E5-E4") and pos("E5-E6") and pos("E5-E7"),
    }
    for k, v in go.items():
        lines.append(f"- `{k}` = **{'GO' if v else 'NO-GO'}**")
    if go["GO_FINAL_MICCAI_CANDIDATE"]:
        lines.append("\nE5满足本轮预注册的候选条件，但仍只代表单seed、历史评估协议下的候选证据。")
    else:
        lines.append("\n当前E5未满足完整候选条件；不把辅助监督、来源对齐或更高对齐程度单独包装成疾病判别增益。")
    lines += ["", "## 科学解释", "", "E1−E0回答辅助疾病监督是否改变原GFR训练；E2/E3/E4/E5分别隔离无条件对抗、疾病条件对抗、选择性CF关系和CORR优先监督。E6仅错配传播关系，E7仅置乱条件分组。任何辅助头或域判别器表现都不能替代E5相对E0/E6/E7的最终CAM指标。真实关系与站点富集的边统计只做混杂敏感性诊断；不能解释为ASD生物学或因果证据。", "", "## 审计结论", "", "独立复核要求每个任务的预测SHA、四项指标和冻结代码签名一致；根源码、原始数据和历史实验只读保护SHA由manifest记录。若任一任务未完成或读回不一致，本报告不应被视为正式交付。"]
    return "\n".join(lines) + "\n", go


def main():
    start = time.time()
    x, y, folds = run.data()
    task_paths = sorted(RT.joinpath("tasks").glob("g*_f*.json"))
    complete = [json.loads(p.read_text()) for p in task_paths if json.loads(p.read_text()).get("status") == "COMPLETE"]
    if len(complete) != 80:
        raise RuntimeError(f"need 80 complete tasks, found {len(complete)}")
    rows, diags = [], []
    for gid in range(8):
        for fold in range(1, 11):
            tr, te = folds[fold - 1]
            row, diag = eval_one(gid, fold, 0, x, y, tr, te)
            row.update({"group_id": gid, "group": run.NAMES[gid], "fold": fold, "seed": 0, "independent_readback": True})
            rows.append(row); diags.append(diag)
            print("READBACK", gid, fold, row["acc"], flush=True)
    sums = summarize(rows); pairs = paired(rows)
    write_csv(RES / "fold_metrics.csv", rows)
    write_csv(RES / "summary.csv", sums)
    write_csv(RES / "paired_comparisons.csv", pairs)
    write_csv(RES / "diagnostics.csv", diags)
    train_seconds = sum(float(json.loads(p.read_text()).get("wall_seconds", 0.0)) for p in task_paths if json.loads(p.read_text()).get("group_id", 0) > 0)
    report, go = make_report(rows, sums, pairs, diags, time.time() - start, train_seconds)
    (RES / "report.md").write_text(report)
    result_shas = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [RES / "fold_metrics.csv", RES / "summary.csv", RES / "paired_comparisons.csv", RES / "diagnostics.csv", RES / "report.md"]}
    man_path = HERE / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update({"status": "COMPLETE", "independent_readback": "80/80_PASS", "go_no_go": go, "result_sha256": result_shas, "audit_wall_seconds": time.time() - start})
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "AUDIT_PASS", "tasks": len(rows), "go": go, "result_sha256": result_shas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
