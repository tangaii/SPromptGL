#!/usr/bin/env python3
"""Best-checkpoint mechanism audit only; no optimizer step or formal metric.

Separates pure CF gradients from classification gradients. The online tracker
named cf_query_gradient includes already accumulated classification gradients;
its value is not treated as a pure-CF norm in the scientific interpretation.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
import run as r
import numpy as np
import torch
import torch.nn.functional as F


def norm(parameter):
    return 0. if parameter.grad is None else float(parameter.grad.norm())


class NativeJSON:
    """Reporting-only adapter; the frozen training implementation is unchanged."""
    def __getattr__(self, key):
        return getattr(json, key)

    @staticmethod
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    def dump(self, value, stream, **kwargs):
        return json.dump(value, stream, default=self.default, **kwargs)

    def dumps(self, value, **kwargs):
        return json.dumps(value, default=self.default, **kwargs)


def final_report():
    """Validate completed artifacts and repair only the final JSON serialization.

    No CUDA context, optimizer step, training restart or checkpoint mutation.
    """
    r.json = NativeJSON()
    rows = r.csv_read(r.RESULTS / "fold_metrics.csv")
    diags = r.csv_read(r.RESULTS / "diagnostics.csv")
    assert len(rows) == len(diags) == 100
    assert len({(x["group"], x["seed"], x["fold"]) for x in rows}) == 100
    baseline = {x["fold"]: x for x in rows if x["group"] == "B0"}
    for group in r.GROUPS:
        assert sum(x["group"] == group for x in rows) == 10
    for row in rows:
        assert row["seed"] == "0"
        assert r.sha(row["best_checkpoint"]) == row["best_checkpoint_sha256"]
        assert r.sha(r.anchor_path(int(row["fold"]))) == row["anchor_sha"]
        for key in ("initial_trunk_sha256", "initial_rng_sha256", "start_trunk_sha", "start_rng_sha"):
            assert row[key] == baseline[row["fold"]][key]
        tp, fn, fp, tn = [int(row[k]) for k in ("asd_tp", "asd_fn", "td_fp", "td_tn")]
        expected = [(tp + tn) / (tp + fn + fp + tn),
                    (tp / (tp + fn) + tn / (tn + fp)) / 2,
                    tn / (tn + fp), tp / (tp + fn)]
        for key, value in zip(("acc", "hard_auc", "spe_td", "sen_asd"), expected):
            assert np.isfinite(float(row[key])) and abs(float(row[key]) - value) < 1e-12
    for row in baseline.values():
        assert not r.compare_anchor(row)
    r.verify_protected()
    manifest_path = r.HERE / "manifest.json"
    m = r.read_json(manifest_path)
    assert m["code_sha"] == r.code_hashes()
    assert m["gpu_completed"] == {"0": 50, "1": 50}
    assert all(t["status"] == "COMPLETE" and t["folds"] == 10 for t in m["tasks"].values())
    original_failure = m.get("failure", {})
    if original_failure:
        assert "bool_ is not JSON serializable" in original_failure["error"]
        m["report_recovery"] = dict(original_failure=original_failure,
                                    resolution="report-only NumPy scalar to native JSON adapter in audit.py",
                                    training_code_modified=False, training_restarts=0, resolved=r.now())
        m["failure"] = dict(phase="post-training report", resolved=True,
                            error=original_failure["error"], full_trace=".runtime/failure.json")
        m.pop("failure_status", None)
    m["scientific"] = "COMPLETE"
    m["final_validation"] = dict(unique_folds=100, best_checkpoint_hashes=100,
                                 anchor_and_entry_pairing=100, confusion_recomputed=100,
                                 root_and_history_hashes="UNCHANGED", frozen_training_code="UNCHANGED",
                                 seed_values=[0], completed_groups=10)
    m["postprocessing_sha"] = {"audit.py": r.sha(r.HERE / "audit.py")}
    r.json_write(manifest_path, m)
    r.report(finalize=False)
    m = r.read_json(manifest_path)
    supplement = ["", "## 固定最佳模型的补充审计与科学解释", "",
                  "以下内容只读取已完成的最佳Checkpoint；没有优化器更新、没有重新选epoch、没有新增正式分类指标。audit.py及三个10行mechanism_audit CSV保留复算证据。",
                  "",
                  "### 100折完整性与报告修复", "",
                  "10/10组、100/100唯一(group, seed=0, fold)全部完成，GPU0/GPU1各50折；100个best文件SHA、100折共同初始化/RNG、混淆计数重算和保护文件SHA全部通过。B0十折初始哈希、入口哈希、混淆、指标、best epoch及执行epoch与E32逐折EXACT。未重训实验39或改动任何根模型/历史数据。",
                  "训练结束后的报告将NumPy bool_交给JSON导致一次序列化异常；训练和Checkpoint早已完成。本次只在独立audit.py报告适配器中把NumPy标量转换为Python标量，冻结run.py/method.py/config完全未变，重新训练次数0。原始错误完整保留于.runtime/failure.json。",
                  "",
                  "### 四层结论", "",
                  "1. 实现层：索引、参考、空mask旁路、梯度、两Adam及恢复预检通过。所有受试者均走统一的OOF训练参考算法，outer-test标签不进入参考拟合。",
                  "2. 响应层：医学文本与证据分支有非零log-prob响应，且CF mask梯度存在；这不等于其提供独立医学信息。",
                  "3. 分类层：E5明显低于B0、Top-|z|和Gram匹配匿名控制；不能宣称疾病语义收益。",
                  "4. 证据层：选中删除未稳定超过随机，也明显未超过Top-|z|；没有证明选边优于简单数值偏离。",
                  "",
                  "### 优化稳定性及对照边界", "",
                  "|组|新增登记参数|疾病Query是否参与选边|实际残差均值%|中位数%|p95%|空mask比例|", "|---|---:|---|---:|---:|---:|---:|"]
    for group in list(r.GROUPS)[1:]:
        ds = [x for x in diags if x["group"] == group]
        avg = lambda key: float(np.mean([float(x[key]) for x in ds]))
        supplement.append(f"|{group}|{int(ds[0]['parameter_count'])}|{'否；数值selector' if group == 'E1' else '是'}|{100*avg('residual_mean'):.5f}|{100*avg('residual_median'):.5f}|{100*avg('residual_p95'):.5f}|{avg('empty_fraction'):.6f}|")
    e5 = [x for x in diags if x["group"] == "E5"]
    e9 = [x for x in diags if x["group"] == "E9"]
    relative_dose = [float(a["residual_mean"]) / max(float(b["residual_mean"]), 1e-12) for a, b in zip(e5, e9)]
    supplement += ["",
                   f"E5/E9训练后逐折残差均值比范围{min(relative_dose):.4f}–{max(relative_dose):.4f}，没有E38的数量级剂量差；各自学到的gate仍不同，不能当成完全相同干预强度。预检的同gate逐患者校准相同不表示训练后gate相同。E5相对E9的ΔACC={m['paired']['E5-E9']['acc']['delta']:+.4f}pp、ΔAUC={m['paired']['E5-E9']['hard_auc']['delta']:+.4f}pp，不能宣称竞争优于一般异常注意。",
                   "E1有相同90499个登记参数，但疾病Query不进入selector，部分Query/温度路径不生效；其数值value仍保留共同解剖信息与冻结软方向先验。因此它是同源数值选择控制，不是完全相同有效容量或完全没有任何医学结构的模型。E7的假说关联及其软方向规则一起错配，raw/z/mu/sigma/物理列从不重排，检验的是假说整体配对。E8仅交换两个Query通道，标签和软方向不交换；E9保留数值软先验但取消类别竞争。匿名控制保持疾病bank全Gram矩阵，但相对真实解剖bank的几何按设计改变。",
                   "E3无CF仍有明显折间不稳定：第3/6/9折ACC分别58.6207/66.6667/56.3218%；E5相应为63.2184/65.5172/56.3218%。E5第9折best epoch=1，ASD TP=3/FN=37。由此不能把失败全部归因于删除损失。小前向残差不保证小反向梯度；单位范数、硬选择的STE及共享投影可能带来优化敏感性，这是待另行预注册检验的解释，而非本轮证明的因果。",
                   "",
                   "### 梯度路径分解", "",
                   "在线cf_query_gradient_max在分类backward之后记录，包含累加分类梯度，不能作为纯CF范数。下表用每折固定best模型的同一训练输入独立backward，没有step；CF选择至多2名原预测正确且非空的训练者。比例为各折比值的平均，非两个平均值之比。",
                   "", "|组|分类投影梯度均值|纯CF投影梯度均值|CF/分类投影梯度比均值|纯CF fuse梯度|关闭QC的绝对logp变化|关闭QC类别翻转率|", "|---|---:|---:|---:|---:|---:|---:|"]
    audit_rows = {}
    for group in ("E1", "E5", "E9"):
        rs = r.csv_read(r.RESULTS / f"mechanism_audit_{group}.csv")
        assert len(rs) == 10 and all(float(x["pure_CF_fuse_gradient"]) == 0 for x in rs)
        audit_rows[group] = rs
        avg = lambda key: float(np.mean([float(x[key]) for x in rs]))
        supplement.append(f"|{group}|{avg('classification_query_gradient'):.8g}|{avg('pure_CF_query_gradient'):.8g}|{avg('CF_over_classification_query_gradient'):.5f}|0|{avg('QC_disabled_abs_logp'):.8f}|{avg('QC_disabled_class_flip'):.6f}|")
    supplement += ["", "CF不更新主干或fuse分类残差参数，实际mask路径梯度有限非零；E1的疾病投影CF梯度为0符合其数值selector设计。E5部分折CF投影梯度可大于当时分类投影梯度，但无CF组同样失败，不能据此单独确定失败原因。",
                   "", "### 选边与反事实分解", "",
                   "|机制量|E5十折平均|", "|---|---:|"]
    for key in ("raw_pair_cos_mean", "projected_pair_cos_mean", "query_pair_distance", "null_mean", "selection_mean", "empty_fraction", "selection_equals_topz_fraction", "selected_delete_logp_drop", "random_delete_logp_drop", "topz_delete_logp_drop", "selected_keep_logp_drop", "frozen_selected_delete_logp_drop", "frozen_topz_delete_logp_drop", "fixed_residual_selected_delete_logp_drop", "swap_abs_logp", "swap_class_flip", "disabled_abs_logp", "disabled_class_flip"):
        supplement.append(f"|{key}|{float(np.mean([float(x[key]) for x in e5])):.8f}|")
    for other in ("random_delete", "topz_delete"):
        differences = [float(x["selected_delete_logp_drop"]) - float(x[other+"_logp_drop"]) for x in e5]
        supplement.append(f"\nSelected-delete减{other}：均值{np.mean(differences):+.8f}，逐折胜/平/负={sum(v>0 for v in differences)}/{sum(v==0 for v in differences)}/{sum(v<0 for v in differences)}。")
    supplement += ["",
                   "E5几乎选满8条、空mask比例0，说明可弃权实现可用但训练后没有实际学会广泛弃权；初始化分位数阈值不自动保证训练后NULL校准。选边与Top-|z|并不相同，却没有更好的诊断依赖效应。真实/交换文本改变内部证据和少量logit，类别翻转率很低；不能把非零梯度或文本敏感度替代分类增益。",
                   "固定邻接时选中删除仍弱于Top-|z|，因此该劣势不能仅由图重连解释；固定未干预残差时selected-delete结果也相近。固定图/残差是同模型干预，不是额外训练组。keep只在39候选中保留所选边；其余217条FC及其它模态始终保留，不能把它解释为仅8条FC足以诊断。",
                   "", "### CF覆盖、质量与站点", "",
                   "|组|总probes|每折unique范围|非空probes|允许delete的probes|", "|---|---:|---|---:|---:|"]
    for group in list(r.GROUPS)[1:]:
        ds = [x for x in diags if x["group"] == group]
        vals = [int(x["unique_probe_count"]) for x in ds]
        supplement.append(f"|{group}|{sum(int(x['probes']) for x in ds)}|{min(vals)}–{max(vals)}|{sum(int(x['eligible_probes']) for x in ds)}|{sum(int(x['delete_probes']) for x in ds)}|")
    quality = {k: dict(n=0, total=0.) for k in ("lower_reliability", "higher_reliability")}
    within = []
    sites = {}
    for row in audit_rows["E5"]:
        for key, value in json.loads(row["quality_strata_selected_delete"]).items():
            quality[key]["n"] += value["n"]
            quality[key]["total"] += (value["mean"] or 0.) * value["n"]
        for key, value in json.loads(row["site_strata_selected_delete"]).items():
            sites.setdefault(key, dict(n=0, total=0.))
            sites[key]["n"] += value["n"]; sites[key]["total"] += value["mean"] * value["n"]
        within.append(json.loads(row["within_site_edgeweight_fraction"]))
    quality_summary = {k:dict(n=v["n"],selected_delete_logp_drop=v["total"]/max(v["n"],1)) for k,v in quality.items()}
    site_summary = {k:dict(n=v["n"],selected_delete_logp_drop=v["total"]/max(v["n"],1)) for k,v in sites.items()}
    supplement += ["", "质量层按每fold训练者reliability中位数固定分组；以下是实际人数加权的机制量，不是第五项分类指标，也不用于模型选择：",
                   "", "```json", json.dumps(quality_summary, indent=2), "```",
                   "", f"四图非自环边权中站点内比例的fold平均（PHENO/ANAT/FUNC/CORR）：{np.mean(within,axis=0).tolist()}。各站点样本量及selected-delete效应完整保存在mechanism_audit_E5.csv；小站点和单seed不足以证明无站点混杂。关闭QC没有测试类别翻转，不能宣称可靠性调节已产生疾病判别收益。",
                   "", "### 执行资源与可恢复性", "",
                   f"正式100折墙钟{m['formal_wall_seconds']:.2f}秒（{m['formal_wall_seconds']/60:.2f}分钟），包括anchor与正式最佳Checkpoint机制诊断；B0单进程连续10折耗时{m['tasks']['B0']['seconds']:.2f}秒。额外审计没有训练更新。预检两患者CF耗时0.453857秒、单进程峰值分配513740288字节；8 worker并发时工具观测每卡约3.7GiB分配，未声称该观测是全训练严格峰值。实际卡名为PPU-ZW810E、每卡98304MiB，环境提供CUDA兼容PyTorch接口。",
                   "", "|组|物理GPU|folds|任务墙钟秒（并行，不可相加当总耗时）|恢复次数|", "|---|---:|---:|---:|---:|"]
    for group,g in r.GROUPS.items():
        t=m["tasks"][group]
        supplement.append(f"|{group}|{g['gpu']}|{t['folds']}|{t['seconds']:.3f}|{t['resumes']}|")
    supplement += ["", "所有任务无需故障重启；预检已验证中断后下一步模型/双Adam/RNG逐值复演。最新恢复Checkpoint每10epoch覆盖，完成折精确跳过；必要best、入口anchor、原型/参考缓存及原源码隔离副本保留在.runtime。没有为追分添加seed、修改超参、重写Prompt或挑选成功折。",
                   "", "### 研究判断", "",
                   f"最终：**{m['decision']}**。不建议把本方案作为已成立的MICCAI疾病语义创新。后续若研究单位范数校准的梯度放大、弃权阈值漂移或选择器稳定性，需单独预注册，不能把事后修补并入本轮。当前结果也不能证明特定连接没有生物学意义；它只否定这套实现与冻结协议下的分类/证据主张。"]
    extra = "\n".join(supplement) + "\n"
    report_path = r.RESULTS / "report.md"
    base_report = report_path.read_text()
    report_path.write_text(base_report + extra)
    m = r.read_json(manifest_path)
    m["mechanism_audit"] = dict(files={f"mechanism_audit_{g}.csv":r.sha(r.RESULTS/f"mechanism_audit_{g}.csv") for g in audit_rows},
                                quality_strata=quality_summary, site_strata=site_summary,
                                reporting_gradient_caveat="online CF norm includes accumulated classification; use isolated audit CSV")
    m["result_sha"]["report.md"] = r.sha(report_path)
    r.json_write(manifest_path,m)
    handoff = "\n\n## Experiment 40 — HCER-v2\n\n" + report_path.read_text().replace("# Experiment 40 — HCER-v2\n", "", 1)
    handoff_result = r.safe_handoff(handoff)
    m = r.read_json(manifest_path);m["handoff"] = handoff_result
    r.json_write(manifest_path,m)
    delivery = r.synchronize()
    m = r.read_json(manifest_path);m["delivery_details"]=delivery;m["delivery"]=delivery["status"]
    m["execution_completed"]=r.now();m["total_wall_seconds"]=time.time()-m["started_unix"]
    r.json_write(manifest_path,m)
    r.report(finalize=False)
    report_path.write_text(report_path.read_text()+extra+"\n## 最终交付状态\n\n"+json.dumps(dict(delivery=delivery,handoff=handoff_result),ensure_ascii=False,indent=2)+"\n")
    m=r.read_json(manifest_path);m["result_sha"]["report.md"]=r.sha(report_path)
    r.json_write(manifest_path,m)
    if handoff_result["status"]=="FAILED_PENDING":
        (r.HERE/"handoff_append_pending.md").write_text("\n\n## Experiment 40 — HCER-v2\n\n"+report_path.read_text().replace("# Experiment 40 — HCER-v2\n","",1))
    if delivery["status"]=="SYNCED_SHA_VERIFIED":
        # Only finalize the small metadata changed after the verified bulk copy.
        import shutil
        target=Path(delivery["path"])
        for rel in ("manifest.json","results/report.md","handoff_append_pending.md"):
            src=r.HERE/rel
            if src.exists():
                shutil.copy2(src,target/rel);assert r.sha(src)==r.sha(target/rel)
    print("FINAL",m["scientific"],m["decision"],m["gpu_completed"],m["delivery"],handoff_result["status"],flush=True)
    for group,values in m["summary"].items():
        print(group," / ".join(f"{values[k]['mean']:.4f} +/- {values[k]['sd']:.4f}" for k in ("acc","hard_auc","spe_td","sen_asd")),flush=True)
    return m


def audit_group(group):
    r.runtime();r.seed_rng(0)
    results=[]
    for fold in range(1,11):
        path=r.checkpoint_best_path(group,fold)
        if not path.exists():continue
        saved_hash=r.sha(path)
        agent=r.make_base(fold);branch=r.attach(agent,group,fold)
        r.set_states(agent,r.load(path)["model"])
        tr,te=r.data()["folds"][fold-1]
        logp,_,graphs,_=r.evaluate(agent)
        selected=branch.last["hard"].detach().clone()
        original_ratio=branch.last["ratio"].detach().clone()
        normal_support=branch.last["support"].detach().clone()
        scores=branch.last["score"].detach().clone()
        z=branch.last["z"].detach().clone()
        original_rel=agent.reference40["reliability"]
        agent.reference40["reliability"]=torch.ones_like(original_rel)
        no_qc=r.evaluate(agent)[0]
        agent.reference40["reliability"]=original_rel
        branch.text_override=torch.zeros_like(branch.texts)
        no_text=r.evaluate(agent)[0]
        branch.text_override=None
        agent.optimizer_MP0.zero_grad();agent.optimizer_MF.zero_grad()
        agent.MessagePassing0.train();agent.ModalFusion.train()
        output,_,_,_=r.forward(agent,graph_loss=False)
        mask=branch.last["selected"];mask.retain_grad()
        loss=F.nll_loss(output[tr],agent.targ[tr],agent.weight)
        loss.backward()
        class_mask=0. if mask.grad is None else float(mask.grad.norm())
        class_query=norm(branch.text_projection.weight)
        class_fuse=norm(branch.fuse.weight)
        class_context=norm(branch.cam_context.weight)
        agent.optimizer_MP0.zero_grad();agent.optimizer_MF.zero_grad()
        candidates=[int(p) for p in tr if selected[p].sum()>0 and logp[p].argmax()==agent.targ[p]]
        cf=r.cf_loss(agent,group,candidates[:2],diagnostic=True) if candidates else dict(selector_mask_gradient=0.,cf_query_gradient=0.,probes=0)
        pure_cf_query=norm(branch.text_projection.weight)
        pure_cf_fuse=norm(branch.fuse.weight)
        # CF uses only the input-mask path, so the detached fixed residual does
        # not update its output fusion parameters.
        assert pure_cf_fuse==0.,"counterfactual gradient leaked through residual fusion"
        row=dict(group=group,fold=fold,checkpoint_sha=saved_hash,
                 classification_selector_gradient=class_mask,classification_query_gradient=class_query,
                 classification_fuse_gradient=class_fuse,classification_CAM_context_gradient=class_context,
                 pure_CF_selector_gradient=cf["selector_mask_gradient"],pure_CF_query_gradient=pure_cf_query,
                 pure_CF_fuse_gradient=pure_cf_fuse,
                 CF_over_classification_query_gradient=pure_cf_query/max(class_query,1e-12),
                 CF_over_classification_selector_gradient=cf["selector_mask_gradient"]/max(class_mask,1e-12),
                 QC_disabled_abs_logp=float((no_qc[te]-logp[te]).abs().mean()),
                 QC_disabled_class_flip=float((no_qc[te].argmax(1)!=logp[te].argmax(1)).float().mean()),
                 disease_text_disabled_abs_logp=float((no_text[te]-logp[te]).abs().mean()),
                 disease_text_disabled_class_flip=float((no_text[te].argmax(1)!=logp[te].argmax(1)).float().mean()),
                 residual_ratio_mean=float(original_ratio[te].mean()),
                 selector_absz_correlation=float(np.corrcoef(scores[te].flatten().cpu().numpy(),z[te].abs().flatten().cpu().numpy())[0,1]))
        if group=="E5":
            # Stratified model-dependence checks; no stratum chooses an epoch,
            # a mask, a rule or a parameter. All held-out labels are post hoc.
            agent.MessagePassing0.eval();agent.ModalFusion.eval()
            site=agent.feat[:,26:46].argmax(1)
            site_same=site[:,None]==site[None,:]
            offdiag=~torch.eye(871,dtype=torch.bool,device=agent.dev)
            row["within_site_edgeweight_fraction"]=json.dumps([float((a*site_same*offdiag).sum()/(a*offdiag).sum().clamp_min(1e-8)) for a in graphs])
            med=torch.median(original_rel[tr]);effects={"lower_reliability":[],"higher_reliability":[]};sites={}
            for person in map(int,te):
                if selected[person].sum()==0:continue
                raw=r.intervene(agent.feat,person,branch.columns,agent.reference40["mu"],selected[person],False)
                out=r.evaluate(agent,raw=raw,override=dict(mask_override=selected))[0]
                label=int(agent.targ[person]);effect=float(logp[person,label]-out[person,label])
                key="lower_reliability" if original_rel[person]<med else "higher_reliability"
                effects[key].append(effect);sites.setdefault(str(int(site[person])),[]).append(effect)
            row["quality_strata_selected_delete"]=json.dumps({k:dict(n=len(v),mean=float(np.mean(v)) if v else None) for k,v in effects.items()})
            row["site_strata_selected_delete"]=json.dumps({k:dict(n=len(v),mean=float(np.mean(v))) for k,v in sites.items()})
        assert r.sha(path)==saved_hash,"audit changed a checkpoint"
        results.append(row)
        print("MECHANISM_AUDIT",group,fold,"pure_CF",pure_cf_query,"cls",class_query,flush=True)
        del agent,branch;torch.cuda.empty_cache()
    r.csv_write(r.RESULTS/f"mechanism_audit_{group}.csv",results)
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--group",choices=["E1","E5","E9"])
    actions.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    if args.finalize:final_report()
    else:audit_group(args.group)
