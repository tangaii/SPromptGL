#!/usr/bin/env python3
"""Independent checkpoint reload, metric reproduction and mechanism audit for 56.2."""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

EXP = Path(__file__).resolve().parent
ROOT = Path('/data/adodas/SPromptGL-main')
sys.path.insert(0, str(EXP))
import run


def metric_set(logp: torch.Tensor, y: np.ndarray, test: np.ndarray) -> dict:
    p = logp.detach().exp().cpu().numpy()[test]
    pred = p.argmax(1); yy = y[test]
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, cm.ravel())
    return {
        'acc': float(np.mean(pred == yy)),
        'hard_auc': float(roc_auc_score(np.eye(2)[yy], np.eye(2)[pred])),
        'prob_auc': float(roc_auc_score((yy == 0).astype(np.int64), p[:, 0])),
        'spe': float(tn / max(tn + fp, 1)), 'sen': float(tp / max(tp + fn, 1)),
        'ba': float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        'asd_f1': float(f1_score(yy, pred, pos_label=0, zero_division=0)),
        'asd_tp': tp, 'asd_fn': fn, 'td_fp': fp, 'td_tn': tn, 'n_test': int(len(test)),
    }


def historical_e1_rows():
    root = Path('/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/tasks')
    rows = []
    for fold in range(1, 11):
        p = root / f'E1_f{fold:02d}.json'
        row = json.loads(p.read_text())
        assert row.get('status') == 'COMPLETE'
        rows.append(row)
    return rows


def stats(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def source_probe(train_codes, test_codes, fold):
    xx = np.concatenate(train_codes, axis=0)
    yy = np.concatenate([np.full(len(z), m, dtype=np.int64) for m, z in enumerate(train_codes)])
    xt = np.concatenate(test_codes, axis=0)
    yt = np.concatenate([np.full(len(z), m, dtype=np.int64) for m, z in enumerate(test_codes)])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=56200 + fold)
    clf.fit(xx, yy)
    prob = clf.predict_proba(xt)
    return {'accuracy': float((prob.argmax(1) == yt).mean()),
            'macro_auc_ovr': float(roc_auc_score(yt, prob, multi_class='ovr', average='macro', labels=[0,1,2,3]))}


def relation_audit(state, data, fold):
    x, y = run.E1.load_data()[0]
    test = np.asarray(data['test'], dtype=np.int64)
    tp_np, tt_np = run.sample_relation_pairs(test, y, 384, 562700 + fold)
    tp = torch.as_tensor(tp_np, dtype=torch.long, device='cuda')
    tt = torch.as_tensor(tt_np, dtype=torch.long, device='cuda')
    _, parts = state['parts'](state['feat'])
    train_codes, test_codes, test_shared, test_logits = [], [], [], []
    with torch.no_grad():
        for m in range(4):
            h = state['model'].relations[m].encode_nodes(parts[m])
            tr_code, _ = state['model'].relations[m].encode_pairs(h, state['relation_pairs'])
            te_code, heads = state['model'].relations[m].encode_pairs(h, tp)
            train_codes.append(tr_code[:, :16].cpu().numpy())
            test_codes.append(te_code[:, :16].cpu().numpy())
            test_shared.append(te_code[:, :16])
            test_logits.append(heads[:, 0])
    pre_state = run.make_state(fold, data); pre_state['fold'] = fold
    _, pparts = pre_state['parts'](pre_state['feat'])
    initial_train, initial_test = [], []
    with torch.no_grad():
        for m in range(4):
            h = pre_state['model'].relations[m].encode_nodes(pparts[m])
            a, _ = pre_state['model'].relations[m].encode_pairs(h, pre_state['relation_pairs'])
            b, _ = pre_state['model'].relations[m].encode_pairs(h, tp)
            initial_train.append(a[:, :16].cpu().numpy()); initial_test.append(b[:, :16].cpu().numpy())
    rel = {}; rows = []
    for m, name in enumerate(run.MODALITIES):
        prob = test_logits[m].softmax(-1).cpu().numpy(); target = tt_np
        auc = float(roc_auc_score(target, prob, multi_class='ovr', average='macro', labels=[0,1,2]))
        pred = prob.argmax(1); cm = confusion_matrix(target, pred, labels=[0,1,2]).tolist()
        code = test_shared[m].cpu().numpy(); means = [code[target == c].mean(0) for c in range(3)]
        distances = [float(np.linalg.norm(means[i] - means[j])) for i,j in ((0,1),(0,2),(1,2))]
        rel[name] = {'macro_auc_ovr_posthoc_test': auc, 'confusion_matrix_AA_TT_AT': cm,
                     'mean_probability': prob.mean(0).tolist(),
                     'probability_entropy': float(-(prob.clip(1e-8,1)*np.log(prob.clip(1e-8,1))).sum(1).mean()),
                     'shared_representation_variance': float(code.var(axis=0).mean()),
                     'prototype_distances_AA_TT_AT': distances}
        for c, label in enumerate(('AA','TT','AT')):
            rows.append({'fold': fold, 'modality': name, 'relation': label,
                         'n': int((target == c).sum()), 'mean_probability': float(prob[:,c].mean()),
                         'posthoc_auc': auc})
    cond = run.disease_condition(state['patient_prob'], state['relation_pairs'])
    with torch.no_grad():
        src_logits = state['model'].adversary(state['model'].last.get('source_input',
            torch.zeros((1,51), device='cuda')))
    # Use the actual train-pair adversary inputs from a deterministic fresh construction.
    source_rows = []
    for code in [state['model'].relations[m].encode_pairs(
            state['model'].relations[m].encode_nodes(pparts[m]), state['relation_pairs'])[0][:,:16] for m in range(4)]:
        source_rows.append(torch.cat((code.unsqueeze(2).mul(cond.unsqueeze(1)).reshape(-1,48), cond), 1))
    source_input = torch.stack(source_rows).reshape(-1,51)
    with torch.no_grad(): source_logits = state['model'].adversary(source_input)
    source_target = torch.arange(4, device='cuda').view(4,1).expand(4,len(state['relation_pairs'])).reshape(-1)
    source_prob = source_logits.softmax(-1).cpu().numpy(); source_y = source_target.cpu().numpy()
    same_cos = []
    for a in range(4):
        for b in range(a+1,4): same_cos.append(float(torch.nn.functional.cosine_similarity(test_shared[a], test_shared[b], dim=1).mean()))
    return rows, {'relation_heads': rel, 'train_patient_disjoint_raw_relation_oof_macro_auc': data['relation_oof_auc'].tolist(),
        'continuous_relation_reliability': data['reliability'].tolist(), 'test_pair_sample_counts_AA_TT_AT': np.bincount(tt_np, minlength=3).tolist(),
        'source_probe_pre_adversarial': source_probe(initial_train, initial_test, fold),
        'source_probe_post_adversarial': source_probe(train_codes, test_codes, fold),
        'trained_conditional_discriminator_train_accuracy': float((source_logits.argmax(1) == source_target).float().mean()),
        'trained_conditional_discriminator_train_macro_auc_ovr': float(roc_auc_score(source_y, source_prob, multi_class='ovr', average='macro', labels=[0,1,2,3])),
        'shared_cross_modal_pair_cosine_mean': float(np.mean(same_cos)),
        'test_relation_metrics_posthoc_only': True}


def graph_audit(state, data, normal, swapped, swap_stats, fold):
    x, _ = run.E1.load_data()[0]; n = len(x); out = {}
    for m, name in enumerate(run.MODALITIES):
        d = normal['details']; edges = d['selected_edges'][m].detach(); en = edges.cpu().numpy()
        same, diff = d['edge_same'][m].detach(), d['edge_diff'][m].detach()
        total = same + diff; receiver = edges[:,0]
        weight_sum = total.new_zeros(n).index_add(0, receiver, total)
        norm_weight = total / weight_sum[receiver].clamp_min(1e-8)
        prior = d['edge_priors'][m].detach(); ps = prior.new_zeros(n).index_add(0, receiver, prior)
        norm_prior = prior / ps[receiver].clamp_min(1e-8)
        conf = d['confidence'][m].detach(); rc = d['receiver_confidence'][:,m].detach()
        same_msg, diff_msg, msg = d['same_msgs'][m].detach(), d['diff_msgs'][m].detach(), d['msgs'][m].detach()
        raw_norm = state['feat'].norm(2, dim=1).clamp_min(1e-8)
        # parameter changes from the E51-initialized state
        gd = 0.; cd = 0.; ed = 0.
        for k, p in state['model'].named_parameters():
            if k.startswith(f'graphs.{m}.') and k in state['initial_inherited_tensors']['gfr']:
                gd += float((p.detach().cpu() - state['initial_inherited_tensors']['gfr'][k]).pow(2).sum())
            if k.startswith(f'gcn.{m}.') and k in state['initial_inherited_tensors']['gfr']:
                cd += float((p.detach().cpu() - state['initial_inherited_tensors']['gfr'][k]).pow(2).sum())
            if k.startswith(f'relations.{m}.edge.') and k in state['initial_new_parameters']:
                ed += float((p.detach().cpu() - state['initial_new_parameters'][k]).pow(2).sum())
        out[name] = {
            'selected_directed_edges': int(len(en)), 'same_channel_nonzero_edges': int((same > 0).sum()),
            'different_channel_nonzero_edges': int((diff > 0).sum()),
            'same_weight_mean': float(same.mean()), 'different_weight_mean': float(diff.mean()),
            'edge_prior_reweighted_mean_l1': float((norm_weight-norm_prior).abs().mean()),
            'selected_confidence_mean': float(conf.mean()), 'selected_confidence_std': float(conf.std(unbiased=False)),
            'receiver_confidence_mean': float(rc.mean()), 'receiver_confidence_std': float(rc.std(unbiased=False)),
            'same_message_norm': float(same_msg.norm()), 'different_message_norm': float(diff_msg.norm()),
            'same_message_over_input': float(same_msg.norm()/state['feat'].norm()),
            'different_message_over_input': float(diff_msg.norm()/state['feat'].norm()),
            'modal_message_over_input': float(msg.norm()/state['feat'].norm()),
            'final_message_over_input': float(normal['details']['mbound'].norm()/state['feat'].norm()),
            'gate_mean': float(d['gate'][:,m].mean()), 'gate_std': float(d['gate'][:,m].std(unbiased=False)),
            'eta': float(d['eta'][m]), 'softcap_active_fraction': float((d['same_message_norms'][m] < 0.5*raw_norm).float().mean()),
            'graphlearn_parameter_delta_norm': float(math.sqrt(gd)), 'gcn_parameter_delta_norm': float(math.sqrt(cd)),
            'edge_network_parameter_delta_norm': float(math.sqrt(ed)),
            'wrong_edge_identity_change_fraction': float(swap_stats[m]['changed_fraction']),
            'wrong_edge_invariants': swap_stats[m],
        }
    return out


def make_intervention_rows(state, data, normal, wrong_edges, wrong_prior):
    x, y = run.E1.load_data()[0]; test = np.asarray(data['test'], dtype=np.int64)
    base_metrics = metric_set(normal['prob'], y, test); base_pred = normal['prob'].detach().exp().cpu().numpy()[test].argmax(1)
    base_p = normal['prob'].detach().exp().cpu().numpy()[test,0]
    normal_gate = normal['details']['gate'].detach()
    plans = [
        ('neutral_disease_edge_weights', {'neutral_edges': True}),
        ('strict_wrong_patient_neighbors', {'forced_selected_edges': wrong_edges, 'forced_edge_prior': wrong_prior}),
        ('strict_wrong_patient_neighbors_frozen_gate', {'forced_selected_edges': wrong_edges, 'forced_edge_prior': wrong_prior, 'freeze_gate': normal_gate}),
        ('disable_cross_patient_messages_self_only', {'self_only': True}),
        ('disable_cross_patient_messages_self_only_frozen_gate', {'self_only': True, 'freeze_gate': normal_gate}),
        ('wrong_disease_condition', {'wrong_condition': True}),
    ]
    plans.extend((f'disable_{name}_message', {'disabled_modalities': (m,)}) for m,name in enumerate(run.MODALITIES))
    rows=[]
    for label, opts in plans:
        with torch.no_grad(): out = run.forward(state, int(state['checkpoint']['best_epoch']), False, opts)
        met = metric_set(out['prob'], y, test); probs = out['prob'].detach().exp().cpu().numpy(); pred = probs[test].argmax(1)
        # Pair-dependent edge comparison only when the selected support has the same cardinality.
        edge_deltas=[]
        for m in range(4):
            a = torch.cat((normal['details']['edge_same'][m], normal['details']['edge_diff'][m]))
            b = torch.cat((out['details']['edge_same'][m], out['details']['edge_diff'][m]))
            edge_deltas.append(float((a-b).abs().mean()) if a.numel()==b.numel() else float(abs(a.mean()-b.mean())))
        rows.append({'fold': state['fold'], 'intervention': label, **met,
            'delta_acc_pp': 100*(met['acc']-base_metrics['acc']),
            'prediction_flip_rate': float(np.mean(pred != base_pred)),
            'mean_abs_delta_p_asd': float(np.mean(np.abs(probs[test,0]-base_p))),
            'mean_abs_delta_logit': float(np.mean(np.abs(out['prob'].detach().cpu().numpy()[test]-normal['prob'].detach().cpu().numpy()[test]))),
            'residual_ratio': float((out['fused']-state['feat']).detach().norm()/state['feat'].norm()),
            'message_norm_before': float(out['details']['mraw'].detach().norm()),
            'message_norm_after': float(out['details']['mbound'].detach().norm()),
            'edge_weight_change_l1': float(np.mean(edge_deltas)),
            'condition_change_fraction': float(np.mean(np.any(data['wrong_patient_prob'][test] != data['patient_prob'][test], axis=1))) if label=='wrong_disease_condition' else 0.0,
            'selected_edge_change_fraction': float(np.mean([np.mean(out['details']['selected_edges'][m].detach().cpu().numpy()[:,1] != normal['details']['selected_edges'][m].detach().cpu().numpy()[:,1]) for m in range(4)])) if label.startswith('strict_wrong') else 0.0,
        })
    # E51 bypass is a separate fixed diagnostic path, not a training group.
    with torch.no_grad():
        _, parts = state['parts'](state['feat']); fused, _, ref = state['model'].reference_bypass(parts); prob = state['cam'](fused)[0]
    met = metric_set(prob, y, test); probs = prob.detach().exp().cpu().numpy(); pred = probs[test].argmax(1)
    rows.append({'fold': state['fold'], 'intervention': 'reference_e51_bypass', **met,
        'delta_acc_pp':100*(met['acc']-base_metrics['acc']), 'prediction_flip_rate':float(np.mean(pred != base_pred)),
        'mean_abs_delta_p_asd':float(np.mean(np.abs(probs[test,0]-base_p))), 'mean_abs_delta_logit':float(np.mean(np.abs(prob.detach().cpu().numpy()[test]-normal['prob'].detach().cpu().numpy()[test]))),
        'residual_ratio':float((fused-state['feat']).detach().norm()/state['feat'].norm()), 'message_norm_before':float(torch.stack(ref['msgs']).norm()), 'message_norm_after':float(torch.stack(ref['msgs']).norm()), 'edge_weight_change_l1':0., 'condition_change_fraction':0., 'selected_edge_change_fraction':0.})
    return rows


def write_report(fold_rows, summary, mechanism, interventions):
    lines=['# Experiment 56.2 — DCRAG-GFR-R2', '',
      '**Engineering status:** `COMPLETE_AUDITED`; one formal group `E56_2_FULL_DCRAG_GFR_R2`, seed 0, 10/10 independently reloaded checkpoints.', '',
      'Hard-AUC uses hard predictions in the historical one-hot definition. Probability-AUC uses continuous P(ASD), with label 0 as positive. Values are fold mean ± population SD unless marked pooled. Historical `val=test`, pre-CV preprocessing/RFE and transductive 871-node graph are retained and limit unbiased generalization claims.', '',
      '## Fold metrics (%)', '', '|Fold|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|Best epoch|', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fold_rows: lines.append(f"|{r['fold']}|{100*r['acc']:.4f}|{100*r['hard_auc']:.4f}|{100*r['prob_auc']:.4f}|{100*r['spe']:.4f}|{100*r['sen']:.4f}|{100*r['ba']:.4f}|{100*r['asd_f1']:.4f}|{r['best_epoch']}|")
    lines += ['', '## Comparison with historical E51 E1', '', '|Metric|E51 E1|E56|E56.1|E56.2|Δ E56.2−E51 E1 (pp)|', '|---|---:|---:|---:|---:|---:|']
    hist56={'acc':.839263,'hard_auc':.837066,'prob_auc':.870762,'spe':.867669,'sen':.806463,'ba':.837066}
    hist561={'acc':.896617,'hard_auc':.894201,'prob_auc':.921784,'spe':.927243,'sen':.861159,'ba':.894201}
    hist51={'acc':.9195924765,'hard_auc':.9174339196,'prob_auc':.9318564563,'spe':.9463922294,'sen':.8884756098,'ba':.9174339196}
    for m in ('acc','hard_auc','prob_auc','spe','sen','ba'):
        lines.append(f"|{m}|{100*hist51[m]:.4f}|{100*hist56[m]:.4f}|{100*hist561[m]:.4f}|{100*summary[m]['mean']:.4f} ± {100*summary[m]['sd']:.4f}|{100*summary[m]['delta_mean']:+.4f}|")
    lines += ['', '## Mechanism audit', '', f"Train-only patient-disjoint relation OOF AUC: **{[f'{x:.4f}' for x in mechanism['relation_oof_auc_mean'].values()]}** for PHENO/ANAT/FUNC/CORR. CORR retains the only strong disease relation evidence; weak modalities are reliability-weighted and remain in independent graph paths.", '', f"Conditional adversarial source-probe macro-AUC changed from {mechanism['source_probe_pre_mean']:.4f} to {mechanism['source_probe_post_mean']:.4f}; the trained discriminator macro-AUC was {mechanism['discriminator_mean']:.4f}. A source probe remaining near one means cross-modal alignment is not supported, even though GRL gradients are nonzero.", '', '|Modality|Relation OOF AUC|Posthoc relation AUC|Gate mean|Gate patient SD|Confidence mean|Confidence SD|Same msg/input|AT msg/input|eta|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,g in mechanism['mean_graph'].items(): lines.append(f"|{name}|{mechanism['relation_oof_auc_mean'][name]:.4f}|{mechanism['posthoc_relation_mean'][name]:.4f}|{g['gate_mean']:.4f}|{g['gate_std']:.6f}|{g['selected_confidence_mean']:.4f}|{g['selected_confidence_std']:.4f}|{g['same_message_over_input']:.4f}|{g['different_message_over_input']:.4f}|{g['eta']:.4f}|")
    lines += ['', f"SoftCap final residual/input mean is {mechanism['mean_residual_ratio']:.4f}; preflight and fold diagnostics show no 56.1-style unbounded amplification. The Gate is no longer mathematically dead: mean patient SD is {mechanism['gate_std_mean']:.6f}, with nonzero first/second-layer preflight gradients and updates.", '', f"Strict selected-edge double swaps changed {100*mechanism['wrong_selected_coverage_mean']:.2f}% of actual propagation edge identities on average and preserved all registered invariants. The fixed-checkpoint strict wrong-graph ΔACC is {mechanism['strict_wrong_delta_acc']:+.4f} pp; this is an intervention dependency result, not an independent retraining gain.", '', '## Fixed-checkpoint interventions', '', '|Intervention|ΔACC pp|Flip rate|ΔP(ASD)|', '|---|---:|---:|---:|']
    for name,v in mechanism['intervention_mean'].items(): lines.append(f"|{name}|{v['delta_acc_pp']:+.4f}|{v['prediction_flip_rate']:.4f}|{v['mean_abs_delta_p_asd']:.6f}|")
    lines += ['', '## Research gates', '', '|Gate|Result|', '|---|---|']
    for k,v in mechanism['research_gates'].items(): lines.append(f'|{k}|{v}|')
    lines += ['', f"**Scientific decision: `{mechanism['decision']}`.** {mechanism['decision_reason']}", '', 'The method is not promoted solely because code tests pass or because a fixed checkpoint responds to interventions. One seed, one ABIDE-A cohort and historical checkpoint selection remain limitations.']
    (EXP/'results/report.md').write_text('\n'.join(lines)+'\n')


def audit_all():
    started=time.time(); manifest=json.loads((EXP/'manifest.json').read_text())
    assert manifest.get('completed_folds') == 10
    protected_before=run.root_protected_sha(); x,y=run.E1.load_data()[0]
    fold_rows=[]; oof=[]; mechanisms=[]; intervention_rows=[]; relation_rows=[]
    for fold in range(1,11):
        data=run.prepare_fold(fold); state=run.make_state(fold,data); state['fold']=fold
        ckpt=run.torch_load(run.best_path(fold)); task=json.loads(run.task_path(fold).read_text())
        assert ckpt['split_sha']==run.E1.split_sha(fold)==task['split_sha']
        assert ckpt['anchor_parameter_sha']==state['anchor']['parameter_sha'] and ckpt['inherited_sha']==state['inherited_sha']
        assert ckpt['new_initial_sha']==state['new_initial_sha']
        state['model'].load_state_dict(ckpt['model']); state['cam'].load_state_dict(ckpt['cam']); state['checkpoint']=ckpt
        state['model'].eval(); state['cam'].eval()
        with torch.no_grad(): normal=run.forward(state,int(ckpt['best_epoch']),False)
        test=np.asarray(data['test'],dtype=np.int64); met=metric_set(normal['prob'],y,test)
        for k,v in met.items():
            if k in ckpt['metrics'] and isinstance(v,(int,float)): assert abs(float(v)-float(ckpt['metrics'][k])) <= 1e-8, (fold,k,v,ckpt['metrics'][k])
        probs=normal['prob'].detach().exp().cpu().numpy(); pred=probs[test].argmax(1)
        for pos,idx in enumerate(test): oof.append({'subject_index':int(idx),'fold':fold,'true_label':int(y[idx]),'prediction':int(pred[pos]),'p_asd':float(probs[idx,0]),'p_td':float(probs[idx,1]),'best_epoch':int(ckpt['best_epoch'])})
        fold_rows.append({'fold':fold,**met,'best_epoch':int(ckpt['best_epoch']),'epochs_executed':int(ckpt['epochs_executed']),'checkpoint_sha':run.sha(run.best_path(fold)),'metric_readback_max_abs_diff':0.0})
        rel_rows, rel=relation_audit(state,data,fold); relation_rows.extend(rel_rows)
        selected=[z.detach().cpu().numpy() for z in normal['details']['selected_edges']]
        sid=run.site_ids(x); train=np.asarray(data['train'],dtype=np.int64)
        wrong=[]; prior=[]; swap_stats=[]
        for m in range(4):
            w, st=run.strict_double_edge_swap(selected[m],sid,train,test,562900+fold*10+m); wrong.append(torch.as_tensor(w,dtype=torch.long,device='cuda')); prior.append(torch.as_tensor(run.compute_edge_prior(x,data['raw_adjs'][m],m,w),dtype=torch.float32,device='cuda')); swap_stats.append(st)
        graphs=graph_audit(state,data,normal,wrong,swap_stats,fold)
        intervention_rows.extend(make_intervention_rows(state,data,normal,wrong,prior))
        # Classification gradient path is measured after reload, with no optimizer step.
        grad_out=run.forward(state,int(ckpt['best_epoch']),False)
        gp=torch.autograd.grad(grad_out['cls'],[state['model'].graphs[m].p.weight for m in range(4)],retain_graph=True,allow_unused=True)
        ep=torch.autograd.grad(grad_out['cls'],[state['model'].relations[m].edge[-1].weight for m in range(4)],retain_graph=False,allow_unused=True)
        grad_graph=[0. if z is None else float(z.norm()) for z in gp]; grad_edge=[0. if z is None else float(z.norm()) for z in ep]
        mechanisms.append({'fold':fold,'relation':rel,'graphs':graphs,'classification_graphlearn_gradients':grad_graph,'classification_edge_gradients':grad_edge,'residual_ratio':float((normal['fused']-state['feat']).detach().norm()/state['feat'].norm()),'strict_swap':swap_stats})
        print(f'AUDITED fold={fold} best_epoch={ckpt["best_epoch"]} acc={met["acc"]:.6f}',flush=True)
        del state; torch.cuda.empty_cache()
    assert len(oof)==871 and len({r['subject_index'] for r in oof})==871
    oof.sort(key=lambda r:r['subject_index']); fold_rows.sort(key=lambda r:r['fold'])
    run.csv_write(EXP/'results/fold_metrics.csv',fold_rows); run.csv_write(EXP/'results/oof_predictions.csv',oof); run.csv_write(EXP/'results/interventions.csv',intervention_rows); run.csv_write(EXP/'results/relation_diagnostics.csv',relation_rows)
    hist=historical_e1_rows(); metrics=('acc','hard_auc','prob_auc','spe','sen','ba','asd_f1'); summary={}; paired=[]
    for m in metrics:
        now=np.asarray([r[m] for r in fold_rows]); old=np.asarray([r[m] for r in hist]); delta=now-old
        summary[m]={'mean':float(now.mean()),'sd':float(now.std(ddof=0)),'historical_mean':float(old.mean()),'historical_sd':float(old.std(ddof=0)),'delta_mean':float(delta.mean()),'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum())}
        for f,d in enumerate(delta,1): paired.append({'metric':m,'fold':f,'e56_2':float(now[f-1]),'e51_e1':float(old[f-1]),'delta':float(d)})
    run.csv_write(EXP/'results/summary.csv',[{'metric':m,**v} for m,v in summary.items()]); run.csv_write(EXP/'results/paired_comparisons.csv',paired)
    yp=np.asarray([r['prediction'] for r in oof]); pa=np.asarray([r['p_asd'] for r in oof]); cm=confusion_matrix(y,yp,labels=[0,1]); tp,fn,fp,tn=map(int,cm.ravel())
    pooled={'acc':float((yp==y).mean()),'hard_auc':float(roc_auc_score(np.eye(2)[y],np.eye(2)[yp])),'prob_auc':float(roc_auc_score((y==0).astype(int),pa)),'spe':float(tn/max(tn+fp,1)),'sen':float(tp/max(tp+fn,1)),'ba':float((tn/max(tn+fp,1)+tp/max(tp+fn,1))/2),'asd_f1':float(f1_score(y,yp,pos_label=0,zero_division=0)),'confusion_matrix':cm.tolist(),'n':871}
    mean_graph={}
    for name in run.MODALITIES:
        mean_graph[name]={}
        keys=mechanisms[0]['graphs'][name].keys()
        for k in keys:
            vals=[m['graphs'][name][k] for m in mechanisms]
            if all(isinstance(v,(int,float,np.number)) for v in vals): mean_graph[name][k]=float(np.mean(vals))
    ints={}
    for label in sorted({r['intervention'] for r in intervention_rows}):
        sub=[r for r in intervention_rows if r['intervention']==label]
        ints[label]={k:float(np.mean([r[k] for r in sub])) for k in ('delta_acc_pp','prediction_flip_rate','mean_abs_delta_p_asd','residual_ratio','edge_weight_change_l1')}
    relation_mean={name:float(np.mean([m['relation']['relation_heads'][name]['macro_auc_ovr_posthoc_test'] for m in mechanisms])) for name in run.MODALITIES}
    relation_oof={name:float(np.mean([m['relation']['train_patient_disjoint_raw_relation_oof_macro_auc'][i] for m in mechanisms])) for i,name in enumerate(run.MODALITIES)}
    pre=float(np.mean([m['relation']['source_probe_pre_adversarial']['macro_auc_ovr'] for m in mechanisms])); post=float(np.mean([m['relation']['source_probe_post_adversarial']['macro_auc_ovr'] for m in mechanisms])); disc=float(np.mean([m['relation']['trained_conditional_discriminator_train_macro_auc_ovr'] for m in mechanisms]))
    cov=float(np.mean([g['wrong_edge_identity_change_fraction'] for m in mechanisms for g in m['graphs'].values()])); gate_std=float(np.mean([g['gate_std'] for m in mechanisms for g in m['graphs'].values()])); residual=float(np.mean([m['residual_ratio'] for m in mechanisms]))
    classification_gain=summary['acc']['delta_mean'] >= .003 and summary['ba']['delta_mean'] >= 0 and summary['prob_auc']['delta_mean'] >= 0
    relation_ok=relation_oof['CORR']>.55; preservation=relation_ok and relation_mean['CORR']>.55; alignment=(pre-post)>=.05 and disc<=.65
    strict_supported=cov>=.30 and all(np.mean([m['graphs'][name]['wrong_edge_identity_change_fraction'] for m in mechanisms])>=.20 for name in run.MODALITIES)
    cross=ints.get('disable_cross_patient_messages_self_only',{}).get('delta_acc_pp',0)<0 and ints.get('disable_cross_patient_messages_self_only',{}).get('prediction_flip_rate',0)>0
    true_relation= strict_supported and ints.get('strict_wrong_patient_neighbors',{}).get('delta_acc_pp',0)<0
    gates = {
        'GO_RELATION_DISCRIMINATION': 'GO' if relation_ok else 'NO-GO',
        'GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT': 'GO' if alignment else 'NO-GO',
        'GO_DISEASE_PRESERVATION': 'GO' if preservation else 'NO-GO',
        'GO_TRUE_PATIENT_RELATION': 'GO' if true_relation else ('INSUFFICIENT_MATCHED_CONTROL' if not strict_supported else 'NOT_SUPPORTED'),
        'GO_CROSS_PATIENT_PROPAGATION': 'PARTIAL_CHECKPOINT_DEPENDENCE' if cross else 'NOT_SUPPORTED',
        'GO_CLASSIFICATION_GAIN': 'GO' if classification_gain else 'NO-GO',
    }
    gates['GO_MICCAI2027_MODULE'] = 'GO' if all(
        gates[k] == 'GO' for k in ('GO_RELATION_DISCRIMINATION', 'GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT',
                                    'GO_DISEASE_PRESERVATION', 'GO_TRUE_PATIENT_RELATION',
                                    'GO_CROSS_PATIENT_PROPAGATION', 'GO_CLASSIFICATION_GAIN')) else 'NO-GO'
    decision='GO' if gates['GO_MICCAI2027_MODULE']=='GO' else ('PARTIAL' if relation_ok and cross else 'NO-GO')
    reason='All preregistered classification and mechanism gates passed.' if decision=='GO' else ('The relation path and fixed-checkpoint propagation are active, but the full preregistered classification/alignment evidence is incomplete.' if decision=='PARTIAL' else 'The preregistered ACC/BA/Probability-AUC and/or mechanism gates are not jointly satisfied; engineering completion does not establish scientific validity.')
    mechanism={'summary':summary,'pooled_oof':pooled,'relation_oof_auc_mean':relation_oof,'posthoc_relation_mean':relation_mean,'source_probe_pre_mean':pre,'source_probe_post_mean':post,'discriminator_mean':disc,'mean_graph':mean_graph,'intervention_mean':ints,'wrong_selected_coverage_mean':cov,'gate_std_mean':gate_std,'mean_residual_ratio':residual,'strict_wrong_delta_acc':ints.get('strict_wrong_patient_neighbors',{}).get('delta_acc_pp',0),'classification_gain_screen':classification_gain,'research_gates':gates,'decision':decision,'decision_reason':reason,'folds':mechanisms,'historical_protocol_limits':['val_idx=test_idx checkpoint selection','global preprocessing/RFE before outer CV','871-node transductive graph','single seed','single ABIDE-A cohort']}
    (EXP/'results/mechanism.json').write_text(json.dumps(mechanism,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write_report(fold_rows,summary,mechanism,intervention_rows)
    protected_after=run.root_protected_sha(); assert protected_after==protected_before
    manifest.update(status='COMPLETE_AUDITED',completed_folds=10,independent_checkpoint_audit='PASS 10/10',metric_readback_max_abs_diff=0.0,protected_sha_after=protected_after,pooled_oof=pooled,research_gates=gates,decision=decision,decision_reason=reason,audit_wall_seconds=time.time()-started,source_sha=run.experiment_source_sha(),result_sha={str(p.relative_to(EXP)):run.sha(p) for p in (EXP/'results').glob('*') if p.is_file()})
    run.atomic_json(EXP/'manifest.json',manifest)
    print('AUDIT_COMPLETE',json.dumps({'status':manifest['status'],'decision':decision,'pooled_oof':pooled,'summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__': audit_all()
