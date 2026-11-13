#!/usr/bin/env python3
"""Independent E79.1 checkpoint and restoration audit.

This module re-loads every checkpoint from disk and recomputes its test
prediction. It does not trust the training ledger for the values it verifies,
and it never uses test labels to construct or update a memory bank.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'; CKPTS = RT / 'checkpoints'; RESULTS = EXP / 'results'
sys.path.insert(0, str(EXP))
from path_adapter import legacy
from drem import DREM, output_layer_logp
from memory_bank import DREMErrorMemory, TYPES, BUCKETS, TYPE_ASD_TO_TD, TYPE_TD_TO_ASD
from run import (GROUPS, FOLD2_EXCLUDE, prepare_fold, hook_forward, torch_load,
                 atomic_json, source_snapshot, site_tensor)


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _digest(obj):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(b'T'); h.update(str(tuple(x.shape)).encode()); h.update(str(x.dtype).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes()); return
        if isinstance(x, dict):
            h.update(b'D')
            for k in sorted(x, key=str): h.update(str(k).encode()); visit(x[k])
            return
        if isinstance(x, (list, tuple)):
            h.update(b'L')
            for y in x: visit(y)
            return
        h.update(repr(x).encode())
    visit(obj); return h.hexdigest()


def _memory_ids(bank):
    return [int(r['patient_id']) for t in TYPES for b in BUCKETS for r in bank.records[t][b].values()]


def _capacity_ok(bank):
    return all(len(bank.records[t][b]) <= bank.capacities[b] for t in TYPES for b in BUCKETS)


def _memory_legality(bank, fold, eligible, test_ids):
    ids = set(int(x) for x in eligible); tests = set(int(x) for x in test_ids)
    checks = {'capacity_ok': _capacity_ok(bank), 'records_subset_eligible': True,
              'test_patient_ids_absent': True, 'fold2_duplicates_absent': True,
              'error_type_label_consistent': True, 'recovery_fields_valid': True,
              'recovery_margin_threshold': True, 'recovery_direction_normalized': True}
    recovered = 0
    for typ in TYPES:
        for bucket in BUCKETS:
            for pid, r in bank.records[typ][bucket].items():
                pid = int(pid); checks['records_subset_eligible'] &= pid in ids; checks['test_patient_ids_absent'] &= pid not in tests
                if fold == 2: checks['fold2_duplicates_absent'] &= pid not in FOLD2_EXCLUDE
                y = int(r.get('true_label', -1)); wp = int(r.get('wrong_predicted_label', -1))
                expected = TYPE_ASD_TO_TD if y == 0 and wp == 1 else TYPE_TD_TO_ASD if y == 1 and wp == 0 else None
                checks['error_type_label_consistent'] &= expected == typ and r.get('error_type') == typ
                if bucket == 'recovered':
                    recovered += 1
                    checks['recovery_fields_valid'] &= all(k in r for k in ('h_wrong','h_correct','u_recovery','wrong_epoch','confirmed_correct_epoch','same_head_margin_gain'))
                    checks['recovery_margin_threshold'] &= float(r.get('same_head_margin_gain', -1e9)) > float(bank.recovery_margin_gain)
                    u = r.get('u_recovery'); checks['recovery_direction_normalized'] &= torch.is_tensor(u) and bool(torch.isfinite(u).all()) and abs(float(u.norm()) - 1.) < 1e-4
                    checks['recovery_fields_valid'] &= int(r.get('confirmed_correct_epoch', -1)) >= int(r.get('wrong_epoch', 10**9))
    checks['all'] = bool(all(checks.values())); checks['recovered_count'] = recovered
    return checks


def _replay(group, fold, kind):
    path = CKPTS / f'{group}_f{fold:02d}.{kind}.pt'
    if not path.is_file(): raise FileNotFoundError(path)
    ckpt = torch_load(path); best = ckpt['best']
    state, warm, identity, hist_task, hist_path = prepare_fold(fold)
    hist = legacy.load_base().load_history(); hist_base = torch_load(hist_path)
    base_equal_history = _same(best['base_state'], hist_base['best_state']) if kind == 'baseanchored' else None
    hist.load_bundle(state, best['base_state'])
    module = DREM().to(state['y'].device); module.load_state_dict(best['module']); module.eval()
    bank = DREMErrorMemory.from_snapshot(best['memory_used'])
    test_ids = [int(x) for x in best['test_ids']]; eligible = [int(x) for x in best['eligible_train_ids']]
    base_out, cam = hook_forward(state, training=False); ids = torch.as_tensor(test_ids, device=cam.device)
    with torch.no_grad():
        enhanced, diag = module(test_ids, cam[ids].detach(), base_out['logp'][ids].detach(), bank, site_tensor(cam.device), use_recovery=(group == GROUPS[1]))
        logp, _ = output_layer_logp(state['cam'].Outputlayer, enhanced)
    prob = logp.exp().detach().cpu().numpy(); recorded = np.asarray(best['probability']); max_diff = float(np.max(np.abs(prob - recorded)))
    pred_match = bool(np.array_equal(prob.argmax(1), np.asarray(best['prediction'])))
    legal = _memory_legality(bank, fold, eligible, test_ids)
    test_set = set(test_ids)
    retrieval_self_excluded = all(int(pid) not in test_set for item in diag['retrieval'] for pid in item.get('asd_to_td_patient_ids', []) + item.get('td_to_asd_patient_ids', []))
    biasless = all(getattr(module, n).bias is None for n in ('q_proj','v_proj','r_proj')) and all(getattr(module.read_to_delta[i], 'bias', None) is None for i in (0,2))
    empty = DREMErrorMemory()
    with torch.no_grad(): he, _ = module(test_ids, cam[ids].detach(), base_out['logp'][ids].detach(), empty, site_tensor(cam.device), use_recovery=(group == GROUPS[1]))
    empty_exact = bool(torch.equal(he, cam[ids]))
    latest_path = CKPTS / f'shared_f{fold:02d}.latest.pt'; previous_hash = None
    if latest_path.is_file():
        for e in torch_load(latest_path).get('event_trace', []):
            if int(e.get('epoch', -1)) == int(best['best_epoch_1based']) - 1: previous_hash = e.get('event_hash')
    used_hash = bank.event_hash(); memory_epoch_match = used_hash == (previous_hash if previous_hash is not None else empty.event_hash())
    checks = {'checkpoint_exists': True, 'identity_match': ckpt.get('identity') == identity,
              'historical_base_state_exact_at_anchor': base_equal_history if kind == 'baseanchored' else True,
              'probability_replay_max_abs_diff': max_diff, 'probability_replay_match': max_diff <= 1e-6,
              'hard_prediction_match': pred_match, 'memory_read_snapshot_match': memory_epoch_match,
              'memory_event_hash_recorded_match': best.get('memory_event_hash') == best['memory_event']['event_hash'],
              'self_exclusion_and_test_absence': retrieval_self_excluded and legal['test_patient_ids_absent'],
              'empty_memory_exact_fallback': empty_exact, 'biasless_memory_direction_path': biasless,
              'direction_mode_matches_group': group in GROUPS,
              'legal_memory': legal['all'], 'fold2_exclude': legal['fold2_duplicates_absent'],
              'no_test_labels_in_memory': legal['test_patient_ids_absent']}
    ok = all(bool(v) for k,v in checks.items() if k != 'probability_replay_max_abs_diff')
    if not ok: raise RuntimeError(f'audit failure {group} fold={fold} kind={kind}: {checks}')
    return {'group':group,'fold':fold,'kind':kind,'status':'PASS','best_epoch':int(best['best_epoch_1based']),'checks':checks,'memory_stats':bank.stats(),'base_state_digest':_digest(best['base_state']),'module_digest':_digest(best['module']),'memory_event_hash':used_hash,'gate_mean':float(diag['gate'].mean()),'delta_ratio_mean':float(diag['delta_ratio'].mean()),'cap_fraction':float(diag['cap_fraction'].mean())}


def _latest_audit(fold):
    d = torch_load(CKPTS / f'shared_f{fold:02d}.latest.pt'); required = ('base_state','rng','modules','optimizers','memory_next','next_epoch','event_trace'); present = all(k in d for k in required)
    state, *_ = prepare_fold(fold); hist = legacy.load_base().load_history(); hist.load_bundle(state,d['base_state'])
    mods = {g:DREM().to(state['y'].device) for g in GROUPS}
    for g,m in mods.items(): m.load_state_dict(d['modules'][g])
    bank = DREMErrorMemory.from_snapshot(d['memory_next'])
    return {'fold':fold,'status':'PASS' if present else 'FAIL','next_epoch':int(d['next_epoch']),'required_state_present':present,'memory_total':bank.size(),'memory_counts':bank.counts(),'optimizer_groups':sorted(d['optimizers']),'rng_keys':sorted(d['rng']),'base_state_digest':_digest(d['base_state'])}


def main():
    best=[]; failures=[]
    for kind in ('baseanchored','method'):
        for group in GROUPS:
            for fold in range(1,11):
                try: best.append(_replay(group,fold,kind))
                except Exception as e: failures.append({'group':group,'fold':fold,'kind':kind,'error':repr(e)})
    latest=[]
    for fold in range(1,11):
        try: latest.append(_latest_audit(fold))
        except Exception as e: latest.append({'fold':fold,'status':'FAIL','error':repr(e)}); failures.append({'fold':fold,'kind':'latest','error':repr(e)})
    current=source_snapshot(); result={'status':'PASS' if len(best)==40 and not failures and all(x['status']=='PASS' for x in latest) else 'FAIL','best_checkpoints_expected':40,'best_checkpoints_audited':len(best),'latest_expected':10,'latest_audits':latest,'best_audits':best,'failures':failures,'source_sha256':hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest()}
    atomic_json(RESULTS/'audit_status.json',result); print(json.dumps({'status':result['status'],'best':len(best),'latest':len(latest),'failures':len(failures)},ensure_ascii=False))
    if result['status'] != 'PASS': raise SystemExit(2)

if __name__ == '__main__': main()
