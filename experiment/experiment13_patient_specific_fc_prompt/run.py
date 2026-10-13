#!/usr/bin/env python3
"""Original-protocol SPromptGL: eight controls, three seeds, ten sequential folds."""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import contextlib
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import types
from types import SimpleNamespace
import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OLD = ROOT / 'experiment/experiment12_roi_semantic_alignment_feasibility'
ROI = ROOT / 'experiment/experiment11_roi_mapping_recovery/results'
MANIFEST = EXP / 'manifest.json'
CONFIGS = ('F0', 'F1', 'F2', 'F3', 'P4', 'P5', 'P6', 'P7')
SEEDS = (0, 1, 2)
EXTRA = [195, 225, 232, 236]
GPU = {f'{config}_seed{seed}': (i + seed) % 2
       for i, config in enumerate(CONFIGS) for seed in SEEDS}
CORE = ['main.py', 'model.py', 'network.py', 'layers.py', 'utils.py',
        'ABIDE-simple-2-concat-weighted-cosine.sh']
METRICS = ['acc', 'ba', 'hard_auc', 'probability_auc', 'sensitivity_asd', 'specificity_td']
PROTOCOL = dict(datadir=str(ROOT/'data')+'/', datname='ABIDE', cpu=False, gpu='-1',
                nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
                n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode='weighted-cosine', MP_mode='GCN', MF_mode='concat',
                alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
                nclass=2, mode='simple-2')
STATUS_SEED = 130500
STATE_PHRASES = (
    'below the training-cohort mean by at least one standard deviation',
    'within one standard deviation of the training-cohort mean',
    'above the training-cohort mean by at least one standard deviation')
_CTX = None


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    # Some mounted filesystems briefly hide a replaced directory entry.
    # Retry only that transient I/O failure; malformed records still fail.
    for attempt in range(4):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            if attempt == 3:
                raise
            time.sleep(.05)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def canonical(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def atomic_torch(path, data):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(data, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def history_digest(path):
    h = hashlib.sha256(); n = 0
    for p in sorted(Path(path).rglob('*')):
        if p.is_file():
            h.update((p.relative_to(path).as_posix()+'\0'+sha256(p)+'\n').encode()); n += 1
    return dict(sha256=h.hexdigest(), files=n)


def inputs():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    frame = pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv')
    modals = np.load(ROOT/'data/ABIDE/modal_feat_dict.npy', allow_pickle=True).item()
    assert list(modals) == ['PHENO', 'ANAT', 'FUNC', 'Correlation']
    assert [len(v) for v in modals.values()] == [48, 6, 10, 256]
    assert list(frame.columns[:-1]) == [str(c) for v in modals.values() for c in v]
    assert list(frame.columns[64:320]) == [str(i) for i in range(256)]
    assert frame.shape == (871, 321) and np.isfinite(frame.values).all()
    x = frame.iloc[:, :-1].values
    y = frame.iloc[:, -1].values.astype(int) - 1
    assert set(y) == {0, 1} and np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    assert len(folds[0][0]) == 783 and len(folds[0][1]) == 88
    return x, y, folds


def validate_mapping():
    import pandas as pd
    p = read_json(OLD/'prompts.json')
    a = pd.read_csv(ROI/'roi_256_mapping.csv')
    b = pd.read_csv(ROI/'roi_256_named_mapping.csv')
    names = pd.read_csv(ROI/'roi_111_labels.csv').set_index('roi_index_zero_based')
    assert len(a) == len(b) == 256 and len(names) == 111
    assert int(names.loc[82, 'atlas_integer_value']) == 3455
    assert names.loc[82, 'roi_name'] == 'UNKNOWN'
    assert names.name_verified.sum() == 110
    for col in ('correlation_feature_index', 'original_connection_index',
                'roi_i_zero_based', 'roi_j_zero_based'):
        assert np.array_equal(a[col], b[col])
    assert a.correlation_feature_index.tolist() == list(range(256))
    i, j = np.triu_indices(111, k=1)
    assert np.array_equal(i[a.original_connection_index], a.roi_i_zero_based)
    assert np.array_equal(j[a.original_connection_index], a.roi_j_zero_based)
    valid = b.loc[b.usable_for_roi_semantic_prompt, 'correlation_feature_index'].tolist()
    assert valid == p['valid_feature_indices'] and len(valid) == 252
    assert sorted(set(range(256))-set(valid)) == p['excluded_feature_indices'] == EXTRA
    for position, e in enumerate(valid):
        row = b.iloc[e]
        left = names.loc[int(row.roi_i_zero_based), 'roi_name']
        right = names.loc[int(row.roi_j_zero_based), 'roi_name']
        assert row.connection_name == f'{left} — {right}'
        assert p['REAL'][position] == f'Resting-state functional connectivity between the {left} and the {right}.'
        assert p['ID_ONLY'][position] == f'Resting-state functional connectivity feature number {e}.'
    rng = np.random.default_rng(1200); permutation = rng.permutation(252)
    while np.any(permutation == np.arange(252)): permutation = rng.permutation(252)
    assert permutation.tolist() == p['shuffled_position_permutation']
    assert p['SHUFFLED'] == [p['REAL'][k] for k in permutation]
    return p


def templates(p):
    real, ids = [], []
    for text, e in zip(p['REAL'], p['valid_feature_indices']):
        anatomy = text.removeprefix('Resting-state functional connectivity between the ').removesuffix('.')
        real.append([f'For this participant, resting-state functional connectivity between the {anatomy} is {s}.'
                     for s in STATE_PHRASES])
        ids.append([f'For this participant, functional connectivity feature number {e} is {s}.'
                    for s in STATE_PHRASES])
    assert len(real) == len(ids) == 252
    return real, ids


def patient_states(x, train, test, valid, fold):
    """No label input; population reference is fitted on this outer train only."""
    z = np.asarray(x[:, 64:320][:, valid], dtype=np.float64)
    mu, sigma = z[train].mean(0), z[train].std(0, ddof=0)
    degenerate = sigma < 1e-6
    u = (z-mu)/np.maximum(sigma, 1e-6)
    real = np.ones(z.shape, dtype=np.int64)
    real[u <= -1] = 0; real[u >= 1] = 2; real[:, degenerate] = 1
    shuffled = real.copy()
    rng = np.random.default_rng(STATUS_SEED+fold)
    for e in range(252):
        for indices in (train, test):
            shuffled[indices, e] = real[rng.permutation(indices), e]
            assert np.array_equal(np.bincount(shuffled[indices, e], minlength=3),
                                  np.bincount(real[indices, e], minlength=3))
    d = dict(reference_mu_sha256=canonical(mu.tolist()), reference_sigma_sha256=canonical(sigma.tolist()),
             real_status_sha256=canonical(real.tolist()), shuffled_status_sha256=canonical(shuffled.tolist()),
             status_shuffle_seed=STATUS_SEED+fold, degenerate_connections=int(degenerate.sum()),
             connections_with_multiple_states=int((np.max(real, 0) != np.min(real, 0)).sum()),
             patient_multiple_states_fraction=float((np.max(real, 1) != np.min(real, 1)).mean()),
             p5_status_mismatch_fraction=float((real != shuffled).mean()),
             p7_roi_mismatch_fraction=1., reference_train_only=True, real_value_consistency=True)
    for name, indices in (('all', np.arange(len(x))), ('train', train), ('test', test)):
        for state, label in enumerate(('low', 'mid', 'high')):
            d[name+'_'+label+'_fraction'] = float((real[indices] == state).mean())
    return dict(real=real, shuffled=shuffled, mu=mu, sigma=sigma, degenerate=degenerate, diagnostics=d)


def patched_model():
    source = (ROOT/'model.py').read_text()
    modified = source.replace('from utils import *\n', 'from utils import *\nfrom roi_patient_adapter import AGESA\n', 1)
    anchor = '        self.ModalFusion.apply(my_weight_init)\n'
    assert modified.count(anchor) == 1
    modified = modified.replace(anchor, anchor+'''        self.roi_adapter = None
        if hyperpm.roi_embeddings is not None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(120000 + hyperpm.seed)
                self.roi_adapter = AGESA(hyperpm.roi_embeddings, hyperpm.roi_indices,
                                         hyperpm.roi_status).to(dev)
            assert sum(p.numel() for p in self.roi_adapter.parameters()) == 32865
            self.optimizer_MF.add_param_group({"params": self.roi_adapter.parameters(),
                                               "lr": hyperpm.lr, "weight_decay": 0.0})
''', 1)
    anchor = '            prob,_,_ = self.ModalFusion(feat)\n'
    assert modified.count(anchor) == 1
    modified = modified.replace(anchor, '''            if self.roi_adapter is not None:
                feat, _ = self.roi_adapter(self.feat[:,64:320], feat)
'''+anchor, 1)
    anchor = '        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)\n'
    assert modified.count(anchor) == 1
    modified = modified.replace(anchor, anchor+'''        if self.roi_adapter is not None:
            feat, _ = self.roi_adapter(self.feat[:,64:320], feat)
''', 1)
    compile(modified, 'experiment13_in_memory_model', 'exec')
    patch = ''.join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True),
                                       fromfile='a/model.py', tofile='b/model.py'))
    return modified, patch


def integrity(full=False):
    m = read_json(MANIFEST)
    for name, digest in m['protected_hashes'].items():
        assert sha256(ROOT/name) == digest, 'protected file changed: '+name
    for name, digest in m['implementation_hashes'].items():
        assert sha256(EXP/name) == digest, 'experiment code changed: '+name
    assert sha256(EXP/'cache/patient_text_bank.pt') == m['patient_cache_sha256']
    if full:
        assert sha256(Path.home()/'.cache/clip/RN50.pt') == m['clip_weight_sha256']
        for name, digest in m['protected_history'].items():
            assert history_digest(ROOT/'experiment'/name) == digest, 'history changed: '+name


def signature():
    m = read_json(MANIFEST)
    fields = {k: m[k] for k in ('protected_hashes', 'implementation_hashes', 'protocol',
                              'patient_cache_sha256', 'split_protocol', 'rng_policy')}
    # Audited I/O-only maintenance keeps the original training/resume identity.
    fields['implementation_hashes'] = m.get('resume_implementation_hashes', fields['implementation_hashes'])
    return canonical(fields)


def prepare():
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import urlopen
    sys.path.insert(0, str(ROOT))
    from CLIP.clip import clip
    import torch.nn.functional as F
    for name in ('cache', 'results', 'logs', 'state'): (EXP/name).mkdir(exist_ok=True)
    if MANIFEST.exists(): integrity(full=True)
    p = validate_mapping(); x, y, folds = inputs()
    weight = Path.home()/'.cache/clip/RN50.pt'
    weight_hash = sha256(weight)
    assert weight_hash == clip._MODELS['RN50'].split('/')[-2]
    old = read_json(OLD/'manifest.json')
    assert sha256(OLD/'cache/roi_text_embeddings.pt') == old['text_cache_sha256']
    static = torch.load(OLD/'cache/roi_text_embeddings.pt', map_location='cpu', weights_only=False)
    assert static['clip_weight_sha256'] == weight_hash
    assert static['prompts_file_sha256'] == sha256(OLD/'prompts.json')
    assert static['texts'] == p and static['encoding_protocol'] == 'grouped_batches_p0_8'
    assert torch.equal(static['banks']['SHUFFLED'], static['banks']['REAL'][p['shuffled_position_permutation']])
    real, ids = templates(p)
    bank_signature = canonical(dict(real=real, id_only=ids, source=sha256(OLD/'prompts.json'),
                                    clip=weight_hash, clip_source=sha256(ROOT/'CLIP/clip/model.py')))
    cache_path = EXP/'cache/patient_text_bank.pt'
    saved = torch.load(cache_path, map_location='cpu', weights_only=False) if cache_path.exists() else None
    if saved is None or saved['signature'] != bank_signature:
        sentences = [s for bank in (real, ids) for edge in bank for s in edge]
        lengths = [len(clip._tokenizer.encode(s))+2 for s in sentences]
        assert max(lengths) <= 77, ('patient prompt exceeds CLIP 77-token limit', max(lengths))
        tokens = clip.tokenize(sentences, truncate=False)
        encoder, _ = clip.load('RN50', 'cuda')
        encoder.eval(); encoder.requires_grad_(False)
        before = rng_state()
        with torch.no_grad():
            vectors = torch.cat([encoder.encode_text(t.to('cuda')).float().cpu()
                                 for t in tokens.split(64)], 0)
        after = rng_state()
        assert rng_equal(before, after), 'frozen RN50 encoding unexpectedly changes RNG'
        assert vectors.shape == (1512, 1024) and torch.isfinite(vectors).all()
        saved = dict(signature=bank_signature, clip_weight_sha256=weight_hash,
                     static_cache_sha256=sha256(OLD/'cache/roi_text_embeddings.pt'),
                     template_hash=canonical(dict(real=real, id_only=ids)),
                     source_prompts_sha256=sha256(OLD/'prompts.json'), states=['LOW', 'MID', 'HIGH'],
                     real=F.normalize(vectors[:756].reshape(252, 3, 1024), dim=-1),
                     id_only=F.normalize(vectors[756:].reshape(252, 3, 1024), dim=-1),
                     token_length_max=max(lengths), unique_encoded_texts=1512,
                     encoding_rng_unchanged=True, encoding_calls_during_training=0,
                     encoder='frozen RN50 / eval / no_grad', templates=list(STATE_PHRASES))
        atomic_torch(cache_path, saved)
        del encoder; torch.cuda.empty_cache()
    else:
        print('REUSE verified patient text bank; no text encoding', flush=True)
    _, patch = patched_model(); (EXP/'integration_patch.diff').write_text(patch)
    if MANIFEST.exists():
        m = read_json(MANIFEST)
    else:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(['git', '-c', 'http.version=HTTP/1.1', 'ls-remote', 'https://github.com/tangaii/SPromptGL.git',
                                          'refs/heads/main'], text=True).split()[0]
        def remote_hash(name):
            with urlopen(f'https://raw.githubusercontent.com/tangaii/SPromptGL/{remote}/{name}', timeout=30) as f:
                data = f.read()
            assert data == (ROOT/name).read_bytes(), 'remote/local core discrepancy: '+name
            return name, hashlib.sha256(data).hexdigest()
        with ThreadPoolExecutor(max_workers=6) as pool: verified_remote = dict(pool.map(remote_hash, CORE))
        protected = CORE + ['data/ABIDE/processed_standard_data.csv', 'data/ABIDE/modal_feat_dict.npy']
        protected += ['experiment/experiment11_roi_mapping_recovery/results/'+name for name in
                      ('roi_111_labels.csv', 'roi_256_mapping.csv', 'roi_256_named_mapping.csv')]
        protected += ['experiment/experiment12_roi_semantic_alignment_feasibility/'+name
                      for name in ('prompts.json', 'cache/roi_text_embeddings.pt', 'roi_semantic_adapter.py')]
        protected += ['CLIP/clip/'+name for name in ('clip.py', 'model.py', 'simple_tokenizer.py')]
        m = dict(experiment='Experiment 13 Patient-Specific FC Prompt', git_head=head, remote_main=remote,
                 remote_core_verified=verified_remote, root_sources_equal_remote=True,
                 created_utc=utc(), run_status='PREPARED', clip_weight_sha256=weight_hash,
                 protected_hashes={name: sha256(ROOT/name) for name in protected},
                 protected_history={d.name: history_digest(d) for number in range(1, 13)
                                    for d in (ROOT/'experiment').glob(f'experiment{number}_*') if d.is_dir()},
                 tasks={task: dict(status='PENDING', physical_gpu=gpu, completed_folds=0)
                        for task, gpu in GPU.items()})
    m.update(protocol=PROTOCOL, split_protocol=dict(n_splits=10, shuffle=True, random_state=10,
             val_equals_test=True, inner_validation=False, ddof=0, thresholds=[-1., 1.], std_floor=1e-6),
             rng_policy='F0 original sequential stream; other groups use actual F0 fold-entry RNG, never repeat per-fold seeding',
             adapter_rng_seed='120000 + training seed; fork_rng; identical to Experiment 12',
             status_shuffle_seed='130500 + one-based fold; separate train/test permutations; local numpy Generator',
             configurations=list(CONFIGS), seeds=list(SEEDS), gpu_assignment=GPU,
             static_cache_reused=True, static_clip_encoded_again=False,
             patient_cache_sha256=sha256(cache_path), patient_template_signature=bank_signature,
             patient_text_token_length_max=saved['token_length_max'],
             valid_edges=252, roi_mapping_verified=256, excluded_semantic_edges=EXTRA,
             backbone_parameters=1462957, additional_parameters={c: 0 if c == 'F0' else 32865 for c in CONFIGS},
             state_prompt_phrases=list(STATE_PHRASES), original_checkpoint_initial_acc=0.,
             checkpoint_rule='strict val ACC improvement; equal ACC retained; wait_cnt > 200',
             probability_auc='ASD=label0; output.exp()[:,0]',
             folds=[dict(fold=k, train_n=len(tr), test_n=len(te), val_equals_test=True,
                         train_sha256=canonical(tr.tolist()), test_sha256=canonical(te.tolist()))
                    for k, (tr, te) in enumerate(folds, 1)],
             implementation_hashes={name: sha256(EXP/name) for name in
                                   ('run.py', 'roi_patient_adapter.py', 'integration_patch.diff')},
             prepared_utc=utc())
    atomic_json(MANIFEST, m)
    print('PREPARED: 8 configs / 24 tasks / 240 folds; static cache reused; patient banks 2 x [252,3,1024]', flush=True)


def context():
    global _CTX
    if _CTX is not None: return _CTX
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT))
    import network
    import model as original
    static = torch.load(OLD/'cache/roi_text_embeddings.pt', map_location='cpu', weights_only=False)
    patient = torch.load(EXP/'cache/patient_text_bank.pt', map_location='cpu', weights_only=False)
    assert network.attr_words == static['texts']['P0']
    # Preserve the original lazy RN50 model-construction RNG consumption. The
    # deterministic eight-text forward is replaced by its verified cached output.
    def cached_p0():
        if network._prompt_text_features is None:
            encoder, _ = network.clip.load('RN50', network.device)
            del encoder
            network._prompt_text_features = static['p0_features'].to('cuda')
        return network._prompt_text_features
    network.get_prompt_text_features = cached_p0
    from CLIP.clip.model import CLIP
    def forbidden_encoding(*args, **kwargs):
        raise RuntimeError('CLIP text encoding is forbidden in training/preflight processes')
    CLIP.encode_text = forbidden_encoding
    modified, _ = patched_model()
    module = types.ModuleType('experiment13_isolated_model')
    module.__file__ = str(EXP/'integration_patch.diff')
    exec(compile(modified, 'experiment13_in_memory_model', 'exec'), module.__dict__)
    x, y, folds = inputs()
    p = static['texts']; permutation = p['shuffled_position_permutation']
    banks = {c: e.to('cuda') for c, e in static['banks'].items()}
    banks.update(P4=patient['real'].to('cuda'), P6=patient['id_only'].to('cuda'))
    banks['P5'] = banks['P4']; banks['P7'] = banks['P4'][permutation]
    _CTX = dict(torch=torch, network=network, original=original, patched=module,
                static=static, patient=patient, banks=banks, x=x, y=y, folds=folds,
                valid=static['valid_feature_indices'].to('cuda'), permutation=permutation)
    return _CTX


def seed_rng(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def rng_equal(a, b):
    import torch
    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0]
            and np.array_equal(a['numpy'][1], b['numpy'][1]) and a['numpy'][2:] == b['numpy'][2:]
            and torch.equal(a['cpu'], b['cpu']) and len(a['cuda']) == len(b['cuda'])
            and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def rng_hash(state):
    import pickle
    h = hashlib.sha256(pickle.dumps((state['python'], state['numpy']), protocol=4))
    for t in [state['cpu']]+state['cuda']: h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def modules(agent):
    result = [agent.MessagePassing0, agent.ModalFusion]
    if getattr(agent, 'roi_adapter', None) is not None: result.append(agent.roi_adapter)
    return result


def model_hash(agent, branch=False):
    h = hashlib.sha256()
    selected = [getattr(agent, 'roi_adapter', None)] if branch else modules(agent)[:2]
    for m in selected:
        if m is not None:
            for name, p in sorted(m.named_parameters()):
                h.update(name.encode()+p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def model_state(agent):
    result = {}
    for name, m in zip(('gfr', 'pln', 'adapter'), modules(agent)):
        result[name] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return result


def load_model(agent, state):
    for name, m in zip(('gfr', 'pln', 'adapter'), modules(agent)): m.load_state_dict(state[name])


def selection(config, personal=None):
    c = context()
    if config == 'F0': return None, None
    if config in ('F1', 'F2', 'F3'):
        return c['banks'][dict(F1='REAL', F2='SHUFFLED', F3='ID_ONLY')[config]], None
    state = personal['shuffled' if config == 'P5' else 'real']
    return c['banks'][config], c['torch'].as_tensor(state, dtype=c['torch'].long, device='cuda')


def make_agent(config, seed, fold, personal=None):
    c = context(); train, test = c['folds'][fold-1]
    e, status = selection(config, personal)
    hp = SimpleNamespace(**PROTOCOL, seed=seed, roi_embeddings=e,
                         roi_indices=c['valid'], roi_status=status)
    cls = c['original'].EvalHelper if config == 'F0' else c['patched'].EvalHelper
    with contextlib.redirect_stdout(io.StringIO()):
        agent = cls([48, 6, 10, 256], c['x'], c['y'], hp, train, test)
    if config == 'F0': agent.roi_adapter = None
    assert np.array_equal(agent.val_idx, agent.tst_idx) and np.array_equal(agent.trn_idx, train)
    agent.MF_sav.close(); agent.GCMP_sav.close()
    return agent


def optimizer_audit(agent):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    seen = [id(p) for g in groups for p in g['params']]
    actual = [id(p) for m in modules(agent) for p in m.parameters()]
    assert len(seen) == len(set(seen)) and set(seen) == set(actual)
    assert all(g['lr'] == .0038 for g in groups)
    assert groups[0]['weight_decay'] == groups[1]['weight_decay'] == .11
    expected = 1462957
    if agent.roi_adapter is not None:
        assert groups[2]['weight_decay'] == 0; expected += 32865
    assert sum(p.numel() for m in modules(agent) for p in m.parameters()) == expected
    return expected


def forward(agent, condition=None, personal=None):
    import torch
    for m in modules(agent): m.eval()
    with torch.no_grad():
        n, parts = agent.dataLayer(agent.feat)
        h, adj, prior_loss = agent.MessagePassing0(parts, n, agent.targ, agent.trn_idx)
        e, status = None, None
        if condition is not None and condition != 'OFF':
            e, status = selection(dict(REAL_PERSON='P4', SHUFFLED_PERSON='P5',
                                       ID_ONLY_PERSON='P6', SHUFFLED_ROI='P7')[condition], personal)
        if agent.roi_adapter is None:
            enhanced, delta = h, torch.zeros_like(h[:, 64:320])
        else:
            enhanced, delta = agent.roi_adapter(agent.feat[:, 64:320], h, e, status, off=condition == 'OFF')
        output = agent.ModalFusion(enhanced)[0]
        assert torch.isfinite(output).all() and torch.isfinite(delta).all() and torch.isfinite(h).all()
        assert torch.count_nonzero(delta[:, EXTRA]) == 0
        assert torch.equal(enhanced[:, :64], h[:, :64])
        return dict(output=output, probability=output.exp(), h=h, delta=delta,
                    enhanced=enhanced, prior_loss=prior_loss)


def metrics(output, indices, y=None):
    from sklearn.metrics import roc_auc_score, confusion_matrix
    y = context()['y'][indices] if y is None else np.asarray(y)[indices]
    probability = output.detach().exp().cpu().numpy()[indices]
    pred = probability.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    sen, spe = cm[0, 0]/cm[0].sum(), cm[1, 1]/cm[1].sum()
    hard = roc_auc_score(np.eye(2)[y], np.eye(2)[pred])
    return dict(acc=float((pred == y).mean()), ba=float((sen+spe)/2), hard_auc=float(hard),
                probability_auc=float(roc_auc_score(y == 0, probability[:, 0])),
                sensitivity_asd=float(sen), specificity_td=float(spe),
                asd_tp=int(cm[0, 0]), asd_fn=int(cm[0, 1]), td_fp=int(cm[1, 0]), td_tn=int(cm[1, 1]))


def gradients(agent, all_parameters=False):
    import torch
    result = {}
    parameters = dict(classifier=agent.ModalFusion.Outputlayer.classifier.weight)
    if agent.roi_adapter is not None:
        parameters.update(alpha=agent.roi_adapter.alpha, project_text=agent.roi_adapter.project_text.weight,
                          project_value=agent.roi_adapter.project_value.weight)
    for name, p in parameters.items():
        if p.grad is None: result[name] = 0.
        else:
            assert torch.isfinite(p.grad).all(), name+' nonfinite gradient'
            result[name] = float(p.grad.norm())
    if all_parameters:
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for m in modules(agent) for p in m.parameters())
    return result


def geometry(tensor):
    x = tensor.detach().float().reshape(-1, tensor.shape[-1]).cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(x, axis=-1)
    q = x / np.maximum(norms[:, None], 1e-12)
    cos = np.clip(q @ q.T, -1, 1)[np.triu_indices(len(q), 1)]
    def rank(values):
        singular = np.linalg.svd(values, compute_uv=False)
        if singular.sum() < 1e-10: return 0.
        p = singular / singular.sum(); p = p[p > 0]
        return float(np.exp(-np.sum(p*np.log(p))))
    return dict(projected_text_mean_cos=float(cos.mean()), projected_text_effective_rank=rank(x),
                projected_text_centered_effective_rank=rank(x-x.mean(0)),
                projected_text_zero_vectors=int((norms == 0).sum()))


def preflight():
    import torch
    import importlib.util
    integrity(full=True)
    c = context(); entry_rng = rng_state(); entry_text = c['network']._prompt_text_features
    train, test = c['folds'][0]
    personal = patient_states(c['x'], train, test, c['valid'].cpu().numpy(), 1)
    p = c['static']['texts']; real_text, id_text = templates(p)
    checks = dict(status='RUNNING', input_shape=[871, 320], csv_shape=[871, 321],
                  roi_mapping=256, valid_edges=252, excluded_edges=EXTRA,
                  val_equals_test=True, train_n_fold1=783, test_n_fold1=88,
                  static_cache_reused=True, clip_encoding_calls=0, configs={},
                  patient_status=personal['diagnostics'])
    expected = np.ones((871, 252), dtype=np.int64)
    u = (c['x'][:, 64:][:, p['valid_feature_indices']]-personal['mu'])/np.maximum(personal['sigma'], 1e-6)
    expected[u <= -1] = 0; expected[u >= 1] = 2; expected[:, personal['degenerate']] = 1
    assert np.array_equal(expected, personal['real'])
    assert personal['diagnostics']['connections_with_multiple_states'] > 0
    assert personal['diagnostics']['p5_status_mismatch_fraction'] > 0
    changed_x = c['x'].copy(); changed_x[test, 64:] += 100
    altered = patient_states(changed_x, train, test, p['valid_feature_indices'], 1)
    assert np.array_equal(altered['mu'], personal['mu']) and np.array_equal(altered['sigma'], personal['sigma'])
    assert np.array_equal(altered['real'][train], personal['real'][train])
    assert all(('Left ' not in s and 'Right ' not in s) for edge in id_text for s in edge)
    assert all(STATE_PHRASES[k] in edge[k] for bank in (real_text, id_text) for edge in bank for k in range(3))
    assert len(set(p['shuffled_position_permutation'])) == 252
    assert not any(k == v for k, v in enumerate(p['shuffled_position_permutation']))
    degenerate_x = c['x'].copy(); degenerate_x[train, 64] = 1.
    degenerate = patient_states(degenerate_x, train, test, p['valid_feature_indices'], 1)
    assert degenerate['degenerate'][0] and np.all(degenerate['real'][:, 0] == 1)
    checks['patient_generation_checks'] = dict(value_consistency=True, training_reference_only=True,
                    label_not_an_input=True, id_only_contains_no_anatomy=True,
                    correct_state_text=True, separate_train_test_shuffle=True,
                    p7_states_unchanged=True, p7_roi_derangement=True, degenerate_all_mid=True)
    # Every seed is audited cold, including the original lazy CLIP initialization.
    for seed in SEEDS:
        seed_rng(seed); initial_rng = rng_state()
        c['network']._prompt_text_features = None
        base = make_agent('F0', seed, 1)
        base_hash = model_hash(base); after_init = rng_state()
        reference = forward(base) if seed == 0 else None
        branch_hash = None
        for config in CONFIGS:
            restore_rng(initial_rng); c['network']._prompt_text_features = None
            agent = make_agent(config, seed, 1, personal)
            assert model_hash(agent) == base_hash
            assert rng_equal(rng_state(), after_init), 'adapter changes the original RNG'
            count = optimizer_audit(agent)
            item = dict(initial_trunk_sha256=base_hash, parameters=count,
                        additional_parameters=count-1462957, original_rng_unchanged=True)
            initial_state = model_state(agent)
            if agent.roi_adapter is not None:
                own_hash = model_hash(agent, True)
                if branch_hash is None: branch_hash = own_hash
                assert own_hash == branch_hash and float(agent.roi_adapter.alpha) == 0
                item['initial_adapter_sha256'] = own_hash
            if seed == 0:
                f = forward(agent)
                error = float((f['output']-reference['output']).abs().max())
                assert error <= 1e-6 and torch.equal(f['h'], reference['h'])
                assert torch.count_nonzero(f['delta']) == 0
                item['eval_parity_max_error'] = error
                state = rng_state(); base.ModalFusion.train(); agent.ModalFusion.train()
                with torch.no_grad(): b = base.ModalFusion(reference['h'])[0]
                restore_rng(state)
                with torch.no_grad(): a = agent.ModalFusion(f['enhanced'])[0]
                item['train_parity_max_error'] = float((a-b).abs().max())
                assert item['train_parity_max_error'] <= 1e-6
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    root_acc, root_auc, _, _ = agent._print_acc(test, 'simple-2')
                met = metrics(f['output'], test)
                assert abs(root_acc-met['acc']) < 1e-12 and abs(root_auc-met['hard_auc']) < 1e-12
                item['original_metric_parity'] = True
                if agent.roi_adapter is not None:
                    gs = []
                    for step in range(2):
                        with contextlib.redirect_stdout(io.StringIO()): agent.run_epoch('simple-2')
                        gs.append(gradients(agent, all_parameters=True))
                        if config == 'P4' and step == 0:
                            resume_test = EXP/'state/preflight_optimizer_resume.pt'
                            atomic_torch(resume_test, dict(model=model_state(agent), rng=rng_state(),
                                mp=agent.optimizer_MP0.state_dict(), mf=agent.optimizer_MF.state_dict()))
                    assert gs[0]['alpha'] > 0 and gs[0]['project_text'] == gs[0]['project_value'] == 0
                    assert gs[1]['project_text'] > 0 and gs[1]['project_value'] > 0
                    item['two_step_gradients'] = gs
                    if config == 'P4':
                        expected_hash = (model_hash(agent), model_hash(agent, True))
                        expected_rng = rng_state()
                        saved_resume = torch.load(resume_test, map_location='cpu', weights_only=False)
                        load_model(agent, saved_resume['model'])
                        agent.optimizer_MP0.load_state_dict(saved_resume['mp'])
                        agent.optimizer_MF.load_state_dict(saved_resume['mf']); restore_rng(saved_resume['rng'])
                        with contextlib.redirect_stdout(io.StringIO()): agent.run_epoch('simple-2')
                        assert (model_hash(agent), model_hash(agent, True)) == expected_hash
                        assert rng_equal(rng_state(), expected_rng)
                        item['model_adam_rng_resume_exact'] = True
                        resume_test.unlink()
                    load_model(agent, initial_state)
            checks['configs'][f'{config}_seed{seed}'] = item
            print(f'PREFLIGHT INIT {config} seed={seed} parameters={count} PASS', flush=True)
            del agent
        # A second-fold entry after two original optimization steps verifies that
        # alignment uses a continuing stream, rather than the same seed each fold.
        if seed == 0:
            restore_rng(after_init); initial_state = model_state(base)
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()): base.run_epoch('simple-2')
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()): base.print_trn_acc('simple-2')
            next_entry = rng_state()
            assert not rng_equal(initial_rng, next_entry)
            second = make_agent('F0', seed, 2); second_hash = model_hash(second)
            restore_rng(next_entry)
            second_branch = make_agent('P4', seed, 2, patient_states(c['x'], *c['folds'][1], p['valid_feature_indices'], 2))
            assert model_hash(second_branch) == second_hash
            checks['continuing_fold_initialization'] = True
            del second, second_branch
            load_model(base, initial_state)
        del base
    seed_rng(0); c['network']._prompt_text_features = None
    base = make_agent('F0', 0, 1); reference = forward(base)
    altered_y = base.targ.clone(); altered_y[test] = 1-altered_y[test]
    n, parts = base.dataLayer(base.feat)
    with torch.no_grad(): h, _, prior = base.MessagePassing0(parts, n, altered_y, train)
    assert torch.equal(h, reference['h']) and torch.equal(prior, reference['prior_loss'])
    checks['test_label_gfr_independence'] = True
    assert np.array_equal(patient_states(c['x'], train, test, p['valid_feature_indices'], 1)['real'], personal['real'])
    seed_rng(0); c['network']._prompt_text_features = None
    agent = make_agent('P4', 0, 1, personal); initial_state = model_state(agent)
    with torch.no_grad(): agent.roi_adapter.alpha.fill_(.3)
    adapter = agent.roi_adapter; z = agent.feat[:, 64:]; e = adapter.text_embeddings.clone()
    own = adapter.residual(z); e[0] = e[1]
    text_changed = adapter.residual(z, e)
    target = int(c['valid'][0]); others = [k for k in range(256) if k != target]
    assert float((text_changed[:, target]-own[:, target]).abs().max()) > 0
    assert torch.equal(text_changed[:, others], own[:, others])
    state = adapter.status_index.clone(); state[0, 0] = (state[0, 0]+1) % 3
    state_changed = adapter.residual(z, status_index=state)
    d = (state_changed-own).abs(); assert float(d[0, target]) > 0
    d[0, target] = 0; assert torch.count_nonzero(d) == 0
    real = forward(agent, 'REAL_PERSON', personal); native = forward(agent)
    assert torch.equal(real['output'], native['output'])
    interventions = {}
    for condition in ('SHUFFLED_PERSON', 'ID_ONLY_PERSON', 'SHUFFLED_ROI', 'OFF'):
        f = forward(agent, condition, personal)
        delta_change = float((f['delta']-real['delta']).abs().mean())
        prob_change = float((f['probability']-real['probability']).abs().mean())
        assert delta_change > 0 and prob_change > 0
        interventions[condition] = dict(delta_mean_change=delta_change, probability_mean_change=prob_change)
    checks['semantic_interventions'] = interventions
    load_model(agent, initial_state)
    # The static implementation matches the old adapter, including operation order.
    spec = importlib.util.spec_from_file_location('experiment12_adapter_reference', OLD/'roi_semantic_adapter.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    seed_rng(0); static_agent = make_agent('F1', 0, 1)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        old_adapter = legacy.AGESA(static_agent.roi_adapter.text_embeddings, c['valid']).to('cuda')
    old_adapter.load_state_dict(static_agent.roi_adapter.state_dict())
    with torch.no_grad():
        static_agent.roi_adapter.alpha.fill_(.3); old_adapter.alpha.fill_(.3)
        a = static_agent.roi_adapter.residual(z); b = old_adapter.residual(z)
    assert torch.equal(a, b); checks['experiment12_static_adapter_exact_parity'] = True
    # Check the literal original selection policy, including ties and wait > early.
    best, wait, selected = 0., 0, -1
    for epoch, value in enumerate([.5, .5]+[.4]*200):
        if value > best: best, wait, selected = value, 0, epoch
        else:
            wait += 1
            if wait > PROTOCOL['early']: break
    assert selected == 0 and epoch == 201 and wait == 201
    checks['strict_first_acc_checkpoint_and_wait_gt_200'] = True
    # Full model+Adam+RNG restoration is tested without introducing a pilot fold.
    path = EXP/'state/preflight_restore.pt'
    atomic_torch(path, dict(model=model_state(agent), rng=rng_state(),
                           mp=agent.optimizer_MP0.state_dict(), mf=agent.optimizer_MF.state_dict()))
    before = forward(agent)['output']; saved = torch.load(path, map_location='cpu', weights_only=False)
    load_model(agent, saved['model']); agent.optimizer_MP0.load_state_dict(saved['mp'])
    agent.optimizer_MF.load_state_dict(saved['mf']); restore_rng(saved['rng'])
    assert torch.equal(before, forward(agent)['output']); path.unlink()
    del agent, base, static_agent, old_adapter
    restore_rng(entry_rng); c['network']._prompt_text_features = entry_text
    integrity(full=True)
    checks.update(status='PASS', signature=signature(), completed_utc=utc(),
                  sources_and_experiments_1_to_12_unchanged=True, formal_training_contaminated=False)
    atomic_json(EXP/'results/preflight.json', checks)
    m = read_json(MANIFEST); m.update(preflight_status='PASS', preflight_sha256=sha256(EXP/'results/preflight.json'))
    if m['run_status'] != 'COMPLETE': m['run_status'] = 'PREFLIGHT_PASS'
    atomic_json(MANIFEST, m)
    print('PREFLIGHT PASS: original protocol, 24 initialization checks, P0 parity, patient truth, locality, gradients', flush=True)


def require_preflight():
    integrity()
    p = read_json(EXP/'results/preflight.json')
    assert p['status'] == 'PASS' and p['signature'] == signature()


def publish_anchor(seed, fold, rng, initial_hash):
    import torch
    path = EXP/'state'/f'F0_seed{seed}_initialization.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else dict(signature=signature(), folds={})
    assert saved['signature'] == signature()
    if fold in saved['folds']:
        old = saved['folds'][fold]
        assert old['initial_trunk_sha256'] == initial_hash and rng_equal(old['rng'], rng)
    else:
        saved['folds'][fold] = dict(rng=rng, rng_sha256=rng_hash(rng), initial_trunk_sha256=initial_hash)
        atomic_torch(path, saved)


def wait_anchor(seed, fold, stopping):
    import torch
    path = EXP/'state'/f'F0_seed{seed}_initialization.pt'
    announced = False
    while not stopping[0]:
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['signature'] == signature()
            if fold in saved['folds']: return saved['folds'][fold]
        if not announced:
            print(f'WAIT original F0 RNG seed={seed} fold={fold}', flush=True); announced = True
        time.sleep(2)
    raise InterruptedError('waiting for original RNG reference interrupted')


def fold_diagnostics(agent, config, seed, fold, own, tracker, initial_geometry, personal):
    import torch
    row = dict(config=config, seed=seed, fold=fold, branch_enabled=config != 'F0',
               additional_parameters=0 if config == 'F0' else 32865,
               prediction_finite=True, anomalous_delta_max=0., first_64_max_change=0.)
    if agent.roi_adapter is None: return row
    adapter = agent.roi_adapter; valid = context()['valid']
    with torch.no_grad():
        projected = torch.nn.functional.normalize(adapter.project_text(adapter.text_embeddings), dim=-1)
    row.update(geometry(projected))
    row.update({'initial_'+k: v for k, v in initial_geometry.items()})
    row.update(alpha=float(adapter.alpha.detach()), gate_coefficient=float(.1*adapter.alpha.detach().tanh()),
               delta_valid_mean_abs=float(own['delta'][:, valid].abs().mean()),
               delta_valid_max_abs=float(own['delta'][:, valid].abs().max()),
               delta_nonzero_fraction=float((own['delta'][:, valid].abs() > 1e-8).float().mean()),
               residual_relative_l2=float((own['delta'].norm(dim=-1)/(own['h'][:, 64:].norm(dim=-1)+1e-12)).mean()),
               project_text_gradient_max=tracker['project_text'], project_value_gradient_max=tracker['project_value'],
               alpha_gradient_max=tracker['alpha'], project_text_weight_norm=float(adapter.project_text.weight.detach().norm()),
               project_value_weight_norm=float(adapter.project_value.weight.detach().norm()))
    if config.startswith('P'):
        row.update(personal['diagnostics'])
        selected = personal['shuffled' if config == 'P5' else 'real']
        row.update(selected_state_consistency_fraction=float((selected == personal['real']).mean()),
                   selected_patient_multiple_states_fraction=float((selected.min(1) != selected.max(1)).mean()),
                   selected_roi_mismatch_fraction=1. if config == 'P7' else 0.,
                   within_edge_projected_state_cos_mean=float(torch.stack([
                       (projected[:, 0]*projected[:, 1]).sum(-1),
                       (projected[:, 0]*projected[:, 2]).sum(-1),
                       (projected[:, 1]*projected[:, 2]).sum(-1)]).mean()))
    return row


def counterfactual(agent, seed, fold, own, personal):
    test = context()['folds'][fold-1][1]
    reference = metrics(own['output'], test)
    base_probability = own['probability'][test]
    initial_hash = model_hash(agent, True)
    rows = []
    for condition in ('REAL_PERSON', 'SHUFFLED_PERSON', 'ID_ONLY_PERSON', 'SHUFFLED_ROI', 'OFF'):
        f = forward(agent, condition, personal); met = metrics(f['output'], test)
        rows.append(dict(config='P4', seed=seed, fold=fold, condition=condition, **met,
                   probability_mean_abs_change=float((f['probability'][test, 0]-base_probability[:, 0]).abs().mean()),
                   prediction_flip_fraction=float((f['probability'][test].argmax(1) != base_probability.argmax(1)).float().mean()),
                   ba_change=met['ba']-reference['ba'], probability_auc_change=met['probability_auc']-reference['probability_auc'],
                   acc_change=met['acc']-reference['acc'],
                   delta_mean_abs_change=float((f['delta'][:, context()['valid']]-own['delta'][:, context()['valid']]).abs().mean()),
                   anomalous_delta_max=float(f['delta'][:, EXTRA].abs().max())))
        if condition == 'REAL_PERSON': assert torch_equal(f['output'], own['output'])
    assert model_hash(agent, True) == initial_hash
    assert torch_equal(forward(agent)['output'], own['output'])
    return rows


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def task_result(task):
    path = EXP/'state'/f'{task}.json'
    if not path.exists(): return None
    d = read_json(path)
    assert d['signature'] == signature() and d['task'] == task
    config, seed_text = task.split('_seed'); seed = int(seed_text)
    assert [r['fold'] for r in d['metrics']] == list(range(1, len(d['metrics'])+1))
    assert len(d['metrics']) <= 10 and len(d['diagnostics']) == len(d['metrics'])
    assert len(d['counterfactual']) == (5*len(d['metrics']) if config == 'P4' else 0)
    assert all(r['config'] == config and r['seed'] == seed and all(np.isfinite(r[k]) for k in METRICS)
               for r in d['metrics'])
    assert d['physical_gpu'] == GPU[task]
    if d['status'] == 'COMPLETE': assert len(d['metrics']) == 10
    return d


def worker(config, seed):
    import torch
    import fcntl
    require_preflight(); c = context(); task = f'{config}_seed{seed}'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(GPU[task])
    assert torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
    lock = (EXP/'state'/f'{task}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = task_result(task)
    if previous and previous['status'] == 'COMPLETE': print('SKIP '+task, flush=True); return 0
    path = EXP/'state'/f'{task}_resume.pt'; sig = signature()
    saved = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    if saved:
        assert saved['signature'] == sig and saved['task'] == task
        result = saved['result']
        assert len(result['metrics']) >= (len(previous['metrics']) if previous else 0)
        restore_rng(saved['rng'])
        # RN50's first load already occurred before these saved epochs/folds.
        c['network']._prompt_text_features = c['static']['p0_features'].to('cuda')
        atomic_json(EXP/'state'/f'{task}.json', result)
    else:
        assert previous is None or not previous['metrics'], 'completed folds have no recoverable RNG state'
        seed_rng(seed)
        result = dict(signature=sig, task=task, config=config, seed=seed, status='RUNNING',
                      physical_gpu=GPU[task], metrics=[], diagnostics=[], counterfactual=[],
                      started_utc=utc(), seconds=0.)
    stopping = [False]
    def stop(signum, frame): stopping[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic(); previous_seconds = result['seconds']
    print(json.dumps(dict(task=task, pid=os.getpid(), physical_gpu=GPU[task],
                         logical_gpu=0, device=str(torch.cuda.get_device_properties(0)), resumed=saved is not None)), flush=True)
    for fold in range(len(result['metrics'])+1, 11):
        train, test = c['folds'][fold-1]
        personal = patient_states(c['x'], train, test, c['valid'].cpu().numpy(), fold) if config.startswith('P') else None
        resume_epoch = saved is not None and saved['phase'] == 'EPOCH' and saved['fold'] == fold
        anchor = None
        if not resume_epoch:
            if stopping[0]: return 130
            if config != 'F0':
                anchor = wait_anchor(seed, fold, stopping); restore_rng(anchor['rng'])
            init_rng = rng_state()
        agent = make_agent(config, seed, fold, personal); optimizer_audit(agent)
        tracker = {k: 0. for k in gradients(agent)}
        initial_geometry = geometry(torch.nn.functional.normalize(agent.roi_adapter.project_text(agent.roi_adapter.text_embeddings), dim=-1)) if agent.roi_adapter is not None else {}
        if resume_epoch:
            load_model(agent, saved['model']); agent.optimizer_MP0.load_state_dict(saved['optimizer_mp'])
            agent.optimizer_MF.load_state_dict(saved['optimizer_mf']); restore_rng(saved['rng'])
            best_state, best_acc, best_epoch = saved['best_model'], saved['best_acc'], saved['best_epoch']
            wait, start, tracker = saved['wait'], saved['epoch']+1, saved['tracker']
            initial_hash, branch_hash = saved['initial_hash'], saved['branch_hash']
            initial_geometry = saved['initial_geometry']; init_rng = saved['init_rng']
            print(f'RESUME fold={fold} epoch={start} best={best_epoch}', flush=True)
        else:
            initial_hash, branch_hash = model_hash(agent), model_hash(agent, True)
            if config == 'F0': publish_anchor(seed, fold, init_rng, initial_hash)
            else: assert initial_hash == anchor['initial_trunk_sha256'], 'paired trunk initialization differs'
            best_state, best_acc, best_epoch, wait, start = None, 0., -1, 0, 0
        print(f'FOLD_START fold={fold} train_n={len(train)} test_n={len(test)} init={initial_hash}', flush=True)
        def save_epoch(epoch):
            result['seconds'] = previous_seconds+time.monotonic()-started
            atomic_torch(path, dict(signature=sig, task=task, phase='EPOCH', fold=fold, epoch=epoch,
                         model=model_state(agent), optimizer_mp=agent.optimizer_MP0.state_dict(),
                         optimizer_mf=agent.optimizer_MF.state_dict(), rng=rng_state(),
                         best_model=best_state, best_acc=best_acc, best_epoch=best_epoch, wait=wait,
                         tracker=tracker, initial_hash=initial_hash, branch_hash=branch_hash,
                         initial_geometry=initial_geometry, init_rng=init_rng, result=result))
        last = start-1
        for epoch in range(start, PROTOCOL['nepoch']):
            if wait > PROTOCOL['early']: break
            print(f'fold={fold} epoch={epoch:03d}/450 ', end='', flush=True)
            agent.run_epoch('simple-2')
            for name, norm in gradients(agent).items(): tracker[name] = max(tracker[name], norm)
            with torch.no_grad(): _, acc = agent.print_trn_acc('simple-2')
            improved = acc > best_acc
            if improved:
                best_acc, best_epoch, wait, best_state = acc, epoch, 0, model_state(agent)
            else: wait += 1
            last = epoch
            print(f'SELECTION best_epoch={best_epoch} best_acc={best_acc:.8f} wait={wait}', flush=True)
            if improved or epoch % 10 == 9 or wait > PROTOCOL['early'] or stopping[0]: save_epoch(epoch)
            if stopping[0]: print('INTERRUPTED saved complete epoch', flush=True); return 130
        assert best_state is not None
        # Record the uninterrupted next-fold RNG before any offline diagnostics.
        continuation_rng = rng_state()
        load_model(agent, best_state); own = forward(agent); met = metrics(own['output'], test)
        assert abs(met['acc']-best_acc) <= 1e-12, 'best restored test ACC != selected val ACC'
        assert config == 'F0' or all(tracker[k] > 0 for k in ('alpha', 'project_text', 'project_value'))
        row = dict(config=config, seed=seed, fold=fold, physical_gpu=GPU[task], **met,
                   best_epoch=best_epoch, epochs_executed=last+1, train_n=len(train), test_n=len(test),
                   selected_val_acc=best_acc, val_equals_test=True, initial_trunk_sha256=initial_hash,
                   initial_adapter_sha256=branch_hash if config != 'F0' else '',
                   additional_parameters=0 if config == 'F0' else 32865,
                   reference_train_only=True, reference_mu_sha256=personal['diagnostics']['reference_mu_sha256'] if personal else '')
        result['metrics'].append(row)
        result['diagnostics'].append(fold_diagnostics(agent, config, seed, fold, own, tracker, initial_geometry, personal))
        if config == 'P4': result['counterfactual'].extend(counterfactual(agent, seed, fold, own, personal))
        restore_rng(continuation_rng)
        result['seconds'] = previous_seconds+time.monotonic()-started
        result['status'] = 'COMPLETE' if fold == 10 else 'RUNNING'
        if fold == 10: result['completed_utc'] = utc()
        # State first, metrics second: even a crash between writes cannot retrain
        # a completed fold or lose its continuing RNG/diagnostics.
        atomic_torch(path, dict(signature=sig, task=task, phase='BETWEEN_FOLDS', fold=fold,
                               rng=continuation_rng, result=result))
        atomic_json(EXP/'state'/f'{task}.json', result)
        print('FOLD_COMPLETE '+json.dumps(row), flush=True)
        if fold == 5 and np.mean([r['acc'] for r in result['metrics']]) < .6:
            raise RuntimeError('original fifth-fold mean-ACC < .6 guard triggered; task incomplete')
        del agent, best_state, own; saved = None
        torch.cuda.empty_cache()
        if stopping[0] and fold < 10: return 130
    path.unlink(missing_ok=True)
    print('TASK_COMPLETE '+task+' folds=10', flush=True)
    lock.close()
    return 0


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read_csv(path):
    if not Path(path).exists(): return []
    with Path(path).open(newline='') as f: rows = list(csv.DictReader(f))
    strings = {'config', 'condition', 'initial_trunk_sha256', 'initial_adapter_sha256',
               'reference_mu_sha256', 'reference_sigma_sha256', 'real_status_sha256', 'shuffled_status_sha256'}
    integers = {'seed', 'fold', 'physical_gpu', 'best_epoch', 'epochs_executed', 'train_n', 'test_n',
                'asd_tp', 'asd_fn', 'td_fp', 'td_tn', 'additional_parameters'}
    for r in rows:
        for k, value in r.items():
            if k in strings: continue
            if value == '': r[k] = None
            elif value in ('True', 'False'): r[k] = value == 'True'
            elif k in integers: r[k] = int(value)
            else:
                try: r[k] = float(value)
                except ValueError: pass
    return rows


def active_workers():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: args = (p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if str(EXP/'run.py').encode() in args and b'worker' in args: found.append(int(p.name))
    return found


def paired(rows, a, b, key):
    lookup = {(r['config'], r['seed'], r['fold']): r for r in rows}
    difference = []
    for seed in SEEDS:
        for fold in range(1, 11):
            x, y = lookup.get((a, seed, fold)), lookup.get((b, seed, fold))
            if x and y: difference.append((seed, x[key]-y[key]))
    if not difference: return None
    values = np.array([d for s, d in difference])
    return dict(mean=float(values.mean()), seed=[float(np.mean([d for s, d in difference if s == seed]))
                                                if any(s == seed for s, d in difference) else None for seed in SEEDS],
                win=int((values > 1e-12).sum()), tie=int((abs(values) <= 1e-12).sum()),
                loss=int((values < -1e-12).sum()), n=len(values))


def summary_rows(rows):
    output = []
    for config in CONFIGS:
        seed_rows = []
        for seed in SEEDS:
            own = [r for r in rows if r['config'] == config and r['seed'] == seed]
            if not own: continue
            r = dict(config=config, scope='seed', seed=seed, n_folds=len(own))
            for key in METRICS:
                x = [v[key] for v in own]
                r.update({key+'_mean': float(np.mean(x)), key+'_fold_sd': float(np.std(x)),
                          key+'_fold_min': float(min(x)), key+'_fold_max': float(max(x))})
            output.append(r)
            if len(own) == 10: seed_rows.append(r)
        if len(seed_rows) == 3:
            r = dict(config=config, scope='three_seed', seed='ALL', n_folds=30)
            for key in METRICS:
                means = [d[key+'_mean'] for d in seed_rows]
                r.update({key+'_mean': float(np.mean(means)), key+'_seed_sd': float(np.std(means)),
                          key+'_fold_min': min(d[key+'_fold_min'] for d in seed_rows),
                          key+'_fold_max': max(d[key+'_fold_max'] for d in seed_rows)})
            output.append(r)
    return output


def report():
    integrity()
    m = read_json(MANIFEST); rows, diag, cf = [], [], []
    for task in GPU:
        d = task_result(task)
        if d:
            rows.extend(d['metrics']); diag.extend(d['diagnostics']); cf.extend(d['counterfactual'])
            m['tasks'][task].update(status=d['status'], completed_folds=len(d['metrics']), seconds=d['seconds'])
            if d.get('completed_utc'): m['tasks'][task]['completed_utc'] = d['completed_utc']
    if not rows and m['run_status'] == 'COMPLETE':
        for file, digest in m['data_result_hashes'].items(): assert sha256(EXP/'results'/file) == digest
        rows = read_csv(EXP/'results/fold_metrics.csv'); diag = read_csv(EXP/'results/diagnostics.csv')
        cf = read_csv(EXP/'results/counterfactual.csv')
    rows.sort(key=lambda r: (CONFIGS.index(r['config']), r['seed'], r['fold']))
    diag.sort(key=lambda r: (CONFIGS.index(r['config']), r['seed'], r['fold']))
    assert len({(r['config'], r['seed'], r['fold']) for r in rows}) == len(rows)
    complete = len(rows) == 240 and len(diag) == 240 and len(cf) == 150 and not active_workers()
    if complete:
        integrity(full=True)
        lookup = {(r['config'], r['seed'], r['fold']): r for r in rows}
        for seed in SEEDS:
            for fold in range(1, 11):
                aligned = [lookup[(c, seed, fold)] for c in CONFIGS]
                assert len({r['initial_trunk_sha256'] for r in aligned}) == 1
                assert len({r['initial_adapter_sha256'] for r in aligned if r['config'] != 'F0'}) == 1
                assert all(r['val_equals_test'] and abs(r['selected_val_acc']-r['acc']) < 1e-12 for r in aligned)
        assert [sum(r['physical_gpu'] == g for r in rows) for g in (0, 1)] == [120, 120]
        assert all(r['anomalous_delta_max'] == r['first_64_max_change'] == 0 for r in diag)
    write_csv(EXP/'results/fold_metrics.csv', rows); write_csv(EXP/'results/diagnostics.csv', diag)
    write_csv(EXP/'results/counterfactual.csv', cf)
    summary = summary_rows(rows); write_csv(EXP/'results/summary.csv', summary)
    agg = {r['config']: r for r in summary if r['scope'] == 'three_seed'}
    def stable_gain(a, b):
        return all((p := paired(rows, a, b, key)) and p['n'] == 30 and
                   p['mean'] > 0 and all(d is not None and d > 0 for d in p['seed'])
                   for key in ('ba', 'probability_auc'))
    patient_effect = 'SUPPORTED_DESCRIPTIVELY' if complete and stable_gain('P4', 'P5') else ('NOT_SUPPORTED' if complete else 'PENDING')
    roi_effect = 'SUPPORTED_DESCRIPTIVELY' if complete and all(stable_gain('P4', c) for c in ('P6', 'P7')) else ('NOT_SUPPORTED' if complete else 'PENDING')
    independent = complete and all(stable_gain('P4', c) for c in ('F1', 'P5', 'P6', 'P7'))
    branch_diag = [r for r in diag if r['branch_enabled']]
    concentrated = sum(r['projected_text_mean_cos'] > .99 and r['projected_text_effective_rank'] < 3 for r in branch_diag)
    m.update(completed_training_folds=len(rows), completed_configs=len(agg), diagnostic_rows=len(diag), counterfactual_rows=len(cf),
             gpu_completed_folds={str(g): sum(r['physical_gpu'] == g for r in rows) for g in (0, 1)},
             ENGINEERING_STATUS='PASS' if complete else 'PENDING',
             PATIENT_SEMANTIC_EFFECT=patient_effect, ROI_IDENTITY_EFFECT=roi_effect,
             INDEPENDENT_SEMANTIC_VALUE='SUPPORTED_DESCRIPTIVELY' if independent else ('NOT_SUPPORTED' if complete else 'PENDING'),
             PROJECTION_COLLAPSE=f'COMMON_DIRECTION_CONCENTRATION {concentrated}/{len(branch_diag)}',
             RECOMMEND_GLOBAL_DISEASE_ALIGNMENT='ONLY_AS_NEW_CONTROLLED_STUDY' if independent else 'NO_CURRENT_EVIDENCE',
             seed_sd_ddof=0)
    if complete:
        m.update(run_status='COMPLETE', completed_utc=m.get('completed_utc', utc()),
                 final_sources_and_history_verified=True, initialization_all_30_pairs_verified=True,
                 no_training_processes=True)
        m['wall_seconds'] = (dt.datetime.fromisoformat(m['completed_utc'])-dt.datetime.fromisoformat(m['training_started_utc'])).total_seconds()
    lines = ['# 实验十三：原协议复现与患者个体化功能连接 Prompt', '',
             f'状态：{m["run_status"]}；{len(rows)}/240 folds，{len(agg)}/8配置完成。', '',
             '## 协议、源码与初始化', '',
             f'本地Git HEAD `{m["git_head"]}`；审计远端main `{m["remote_main"]}`。六个核心源码逐字节一致，版本差异为仓库产物更新；未重置或修改根源码。详细设计和代码调用审计见README.md；SHA256见manifest。实验一至十二完整保留。', '',
             '恢复原main.py：871人、48/6/10/256分段、StratifiedKFold(10,shuffle=True,random_state=10)、seed0/1/2；训练为每折除测试折外全部人员，val_idx逐值等于tst_idx。450 epochs，wait_cnt>200才早停，初始最佳ACC=0，ACC严格提高才替换；同ACC保留第一次checkpoint。最终恢复GFR、PLN和适配器，所有已完成折的测试ACC逐值匹配选中的val ACC。原weighted-cosine、GCN、concat、simple-2、.0038/.11/.35/18/2/1及原NLL/图损失保持不变；无独立验证集、额外损失、RFE或超参数搜索。', '',
             'val=test属于指定的原论文复现协议，测试参与模型选择，结果不是无偏独立测试泛化估计。沿用原CSV，上游全局标准化和RFE特征选择有信息共享风险。全部结果只能用于这套既定数据和协议下的配对比较。ASD=label0；Probability-AUC使用output.exp()[:,0]，Hard-AUC复制原one-hot硬预测实现；两者分别列出。', '',
             'F0直接调用未修改EvalHelper。原GFR为受试者图，分支覆盖行为和未纳入最终损失的ALoss保持原样。测试标签仅供原协议评价，NLL及GFR标签先验实际只取train_idx；改变测试标签不改变GFR及其标签先验的预检通过。PLN四个9维局部Token是潜在表示，不是具体脑区。', '',
             '首次RN50加载会消耗CPU初始化随机数。本运行器保留原位置的加载行为，仅使用实验十二的已验证P0编码替代无随机性的文本前向，训练文本编码调用为0。F0保留原始连续十折随机状态。早停长度在不同配置中可能不同；为同时保留原F0轨迹与配对起点，其余组使用F0各折实际入口RNG，不对每折重复设置相同seed。适配器以fork_rng(seed=120000+seed)隔离。预检检查24组初始权重/零残差；正式30个seed×fold配对再次检查主干和七组适配器哈希。', '',
             '## 个体化语义与控制变量', '',
             '有效252条连接严格来自实验十一映射。195/225/232/236保留原数值但从语义库排除，所有残差精确为0。静态F1/F2/F3逐字复用实验十二文本及缓存；静态残差与旧适配器数值完全一致。F0仅原P0；其余七组相同32,865参数：1024→32无偏置文本投影、3→32无偏置数值投影和零初始化共享alpha。delta=.1*tanh(alpha)*z*tanh(normalize(WtE)·normalize(Wv[z,abs(z),sign(z)]))；仅加到传播后的H[:,64:320]，配对z来自原CSV。前64维及原256维输入不改。旧Adam组保持.11，新参数同学习率、wd0。', '',
             '每折只用外层训练人员计算各连接mu/sigma，population std(ddof=0)，阈值固定-1/+1；sigma<1e-6时所有人该边设MID。LOW/MID/HIGH分别为低于训练均值至少一SD、均值一SD以内、高于训练均值至少一SD，均为定量描述，不称为ASD异常。同一连接在不同人之间确实产生不同Prompt；状态选择不使用诊断标签。756个真实ROI状态文本和756个匿名状态文本一次冻结RN50编码；先投影文本库再索引[N,252,32]。严格77-token检查，不截断。', '',
             'P4正确名称+自己状态；P5每条边在训练内部和测试内部独立置换状态，local Generator seed=130500+fold，频率保持、原数值不改；P6匿名特征号+自己状态；P7使用实验十二seed1200无固定点ROI置换，保留实际连接自己的状态。P5破坏患者—状态对应，P7破坏连接—解剖对应，二者分别报告。', '',
             '## 主结果', '',
             '先计算每seed十折均值，再报告三个seed均值±population seed SD（ddof=0）。逐折值和范围见fold_metrics.csv和summary.csv。三个seed共用相同fold，不视为30个独立样本，不做夸大显著性推断。', '',
             '| Config | ACC | BA | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for config in CONFIGS:
        r = agg.get(config)
        cells = [f'{100*r[k+"_mean"]:.3f}±{100*r[k+"_seed_sd"]:.3f}' for k in METRICS] if r else ['待完整30折']*6
        lines.append('| '+config+' | '+' | '.join(cells)+' |')
    lines += ['', '### 每seed的十折均值', '', '| Config | Seed | Folds | ACC | BA | Hard-AUC | Prob-AUC | SEN | SPE |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope'] == 'seed': lines.append('| '+r['config']+f' | {r["seed"]} | {r["n_folds"]} | '+' | '.join(f'{100*r[k+"_mean"]:.3f}' for k in METRICS)+' |')
    lines += ['', '## 八项配对比较', '', '差值均为百分点，按相同seed/fold配对；胜平负容差1e-12，保留完整指标精度。']
    for a, b in (('F1','F0'), ('F1','F2'), ('F1','F3'), ('P4','F1'), ('P4','P5'), ('P4','P6'), ('P4','P7'), ('P4','F0')):
        lines += ['', f'### {a} − {b}', '', '| 指标 | 平均差 pp | seed0/1/2差 pp | 胜/平/负 | 配对折数 |', '|---|---:|---|---|---:|']
        for key in METRICS:
            p = paired(rows, a, b, key)
            if p: lines.append(f'| {key} | {100*p["mean"]:+.3f} | '+ '/'.join('待完成' if d is None else f'{100*d:+.3f}' for d in p['seed'])+f' | {p["win"]}/{p["tie"]}/{p["loss"]} | {p["n"]} |')
    lines += ['', '## 与实验十二单fold的协议差异', '',
              '实验十二为seed0/fold1、665/118/88三分，本次对应783/88/88，val=test；静态文本/数学机制/参数数目不变，同时恢复原代码首次RN50载入的随机轨迹。因此不能将两轮数值变化纯粹归因于测试样本或单个组件；下表仅对应seed0/fold1，不能把旧四个pilot结果混入当前240折。', '',
              '| Config | 旧ACC | 本次ACC | 旧BA | 本次BA | 旧Prob-AUC | 本次Prob-AUC |', '|---|---:|---:|---:|---:|---:|---:|']
    for config in CONFIGS[:4]:
        old = read_json(OLD/'checkpoints'/f'{config}_result.json')['test']
        r = next((v for v in rows if v['config'] == config and v['seed'] == 0 and v['fold'] == 1), None)
        if r: lines.append('| '+config+' | '+' | '.join(f'{100*v:.3f}' for k in ('acc','ba','probability_auc') for v in (old[k],r[k]))+' |')
    lines += ['', '## 最佳checkpoint机制诊断', '',
              '投影几何统计整个252条静态库或756条状态库，不把重复患者索引计为额外独立文本。中心化有效秩用于观察去除共有方向后的差异；高cos和低非中心化秩不自动表示所有语义区别消失。', '',
              '| Config | alpha | 相对L2残差% | delta绝对均值 | 初始/训练后cos | 初始/训练后有效秩 | 训练后中心化秩 | 状态内cos |',
              '|---|---:|---:|---:|---|---|---:|---:|']
    for config in CONFIGS[1:]:
        own = [r for r in branch_diag if r['config'] == config]
        if not own: continue
        mean = lambda k: float(np.mean([r[k] for r in own]))
        state_cos = f'{mean("within_edge_projected_state_cos_mean"):.5f}' if config.startswith('P') else '不适用'
        lines.append(f'| {config} | {mean("alpha"):+.5f} | {100*mean("residual_relative_l2"):.3f} | {mean("delta_valid_mean_abs"):.3e} | {mean("initial_projected_text_mean_cos"):.5f}/{mean("projected_text_mean_cos"):.5f} | {mean("initial_projected_text_effective_rank"):.3f}/{mean("projected_text_effective_rank"):.3f} | {mean("projected_text_centered_effective_rank"):.3f} | {state_cos} |')
    lines += ['', '### 患者状态', '', '| Config | LOW/MID/HIGH总比例 | Train比例 | Test比例 | 连接多样性数 | 自身状态一致率 |', '|---|---|---|---|---:|---:|']
    for config in CONFIGS[4:]:
        own = [r for r in diag if r['config'] == config]
        if not own: continue
        mean = lambda k: float(np.mean([r[k] for r in own]))
        proportions = ['/'.join(f'{mean(prefix+"_"+k+"_fraction"):.4f}' for k in ('low','mid','high')) for prefix in ('all','train','test')]
        lines.append('| '+config+' | '+' | '.join(proportions)+f' | {mean("connections_with_multiple_states"):.1f} | {mean("selected_state_consistency_fraction"):.4f} |')
    lines += ['', '## P4同checkpoint反事实', '',
              '150条记录由30个已选P4 checkpoint各推理五种条件构成。没有重新训练、重置权重或重新选checkpoint；只覆盖辅助分支文本/状态，P0原路径保持。REAL_PERSON与正常输出精确相等，结束后保持真实配置。ASD概率变化及翻转率统计本折相同测试人员，delta变化统计有效连接；不保存患者预测明细。OFF是关闭当前P4分支，不是重新训练的F0。', '',
              '| 条件 | 概率绝对变化 | 类别翻转率 | BA变化 pp | Prob-AUC变化 pp | delta变化 |', '|---|---:|---:|---:|---:|---:|']
    for condition in ('REAL_PERSON','SHUFFLED_PERSON','ID_ONLY_PERSON','SHUFFLED_ROI','OFF'):
        own = [r for r in cf if r['condition'] == condition]
        if own: lines.append('| '+condition+' | '+' | '.join(f'{np.mean([r[k] for r in own])*scale:.6g}' for k,scale in
                              [('probability_mean_abs_change',1),('prediction_flip_fraction',1),('ba_change',100),('probability_auc_change',100),('delta_mean_abs_change',1)])+' |')
    lines += ['', '## 科学结论与研究边界', '',
              f'原协议复现：{"COMPLETE" if complete else "尚未完成"}；个体化模块工程：{m["ENGINEERING_STATUS"]}；患者自身状态匹配：{patient_effect}；ROI解剖身份：{roi_effect}；独立语义诊断价值：{m["INDEPENDENT_SEMANTIC_VALUE"]}。三个判断分别给出，工程parity、非零梯度或推理敏感性均不等同于医学语义有效。', '',
              '稳定收益的描述性规则为BA与Probability-AUC平均差均为正且三个seed均同向；其余情形记为未支持稳定收益，不等于证明总体效应为零。必须同时看P4−F1、P4−P5、P4−P6、P4−P7；只胜静态语义或原F0不足以证明患者真实语义。ACC与SEN/SPE的全部配对变化已列出，不能把类别偏向或阈值变化包装成AUC收益。', '',
              f'共有方向高度集中的折数（平均cos>.99且有效秩<3）为{concentrated}/{len(branch_diag)}。若正确/错配替换影响远小于OFF，且P4没有稳定超过身份/状态对照，更符合已有数值重加权和训练容量作用的解释；仅作为机制线索，不能凭梯度宣布医学成功。', '',
              f'是否进入全局疾病层次对齐：{m["RECOMMEND_GLOBAL_DISEASE_ALIGNMENT"]}。'+('当前存在方向一致线索，可作为新研究起点，仍需正确/错误疾病语义和容量对照、独立评价。' if independent else '当前证据不足以据此叠加全局ASD/TD疾病分支；保留负结果与严格对照，不自动调参。'), '',
              '本实验实现Patient-conditioned ROI-edge Prompts：每人每条连接选择不同的状态描述。没有年龄/性别综合患者Prompt，没有全局ASD/TD分支，不等同于BrainPrompt/SABER完整患者全局语义模块，不能声称发现可靠诊断生物标志物或具体脑区神经病理。', '',
              '## 运行与产物', '',
              f'每卡12个config×seed任务，静态6个、个体化6个，共120折。当前完成GPU0={m["gpu_completed_folds"]["0"]}、GPU1={m["gpu_completed_folds"]["1"]}；正式训练壁钟秒数={m.get("wall_seconds", "执行中")}。独立环境CUDA_VISIBLE_DEVICES、OMP/MKL/OpenBLAS线程2；一个锁定scheduler。失败/中断按原子模型+Adam+全部RNG恢复，已完成折不重训。汇总读取校验的任务记录，不从失败日志拼接指标。', '',
              '最终仅保留代码、必要文本库、manifest、预检、聚合CSV、中文报告和任务日志；临时模型与RNG参考在全部验收后清理。根源码、CSV、实验一至十二未修改，本任务未git commit/push。']
    (EXP/'results/report.md').write_text('\n'.join(lines)+'\n')
    if complete:
        m['data_result_hashes'] = {name: sha256(EXP/'results'/name) for name in
                                  ('fold_metrics.csv','diagnostics.csv','counterfactual.csv')}
    atomic_json(MANIFEST, m)
    if complete:
        for p in (EXP/'state').iterdir():
            if p.is_dir(): shutil.rmtree(p)
            elif p.name != 'scheduler.lock': p.unlink()
        print_final(m, agg, rows)
    else: print(f'REPORT PARTIAL: {len(rows)}/240 folds, no missing values filled with zero', flush=True)
    return m


def print_final(m, agg, rows):
    print('\nExperiment 13 — Patient-Specific FC Prompt\n')
    print('PROTOCOL: ORIGINAL_SPROMPTGL\nVAL_EQUALS_TEST: TRUE\nCONFIGS: 8/8\nSEEDS: 3/3\nFOLDS: 240/240\n')
    print('GPU0_FOLDS: 120\nGPU1_FOLDS: 120\n')
    print('P0_PARITY: PASS\nROI_MAPPING: 256/256, 252 semantic edges\nPERSONAL_PROMPT_VALIDATION: PASS\nCLIP_CACHE_STATUS: VERIFIED_REUSED_STATIC_AND_PATIENT_BANK\nENGINEERING_STATUS: PASS\n')
    for config in CONFIGS:
        r = agg[config]
        print(config+' ACC / BA / Hard-AUC / Prob-AUC: '+' / '.join(f'{100*r[k+"_mean"]:.3f}%' for k in METRICS[:4])+
              f'; SEN={100*r["sensitivity_asd_mean"]:.3f}%; SPE={100*r["specificity_td_mean"]:.3f}%')
    for b in ('F1','P5','P6','P7','F0'):
        print('P4_vs_'+b+': '+' / '.join(f'{k} {100*paired(rows,"P4",b,k)["mean"]:+.3f}pp' for k in METRICS[:4]))
    for key in ('PATIENT_SEMANTIC_EFFECT','ROI_IDENTITY_EFFECT','PROJECTION_COLLAPSE','RECOMMEND_GLOBAL_DISEASE_ALIGNMENT'):
        print(key+': '+m[key])
    print('WALL_SECONDS: '+str(m.get('wall_seconds')))
    print('REPORT: '+str(EXP/'results/report.md'))
    print('GIT_STATUS: no commit/push; root/history preserved; new experiment directory only', flush=True)


def worker_env(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(EXP)+os.pathsep+str(ROOT),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def schedule(workers_per_gpu):
    import fcntl
    require_preflight()
    assert workers_per_gpu > 0
    lock = (EXP/'state/scheduler.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another Experiment 13 scheduler is running')
    m = read_json(MANIFEST)
    if m['run_status'] == 'COMPLETE': report(); lock.close(); return
    m.update(run_status='RUNNING', training_started_utc=m.get('training_started_utc', utc()),
             scheduler_pid=os.getpid(), workers_per_gpu=workers_per_gpu)
    atomic_json(MANIFEST, m)
    stream = (EXP/'logs/scheduler.log').open('a', buffering=1)
    def log(message):
        line = utc()+' '+message; print(line, flush=True); stream.write(line+'\n')
    subprocess.run(['nvidia-smi'], stdout=stream, stderr=subprocess.STDOUT, check=True)
    queues = {}
    for gpu in (0, 1):
        tasks = [task for task, assigned in GPU.items() if assigned == gpu]
        assert len(tasks) == 12 and sum(task.startswith('F') for task in tasks) == 6
        base = [t for t in tasks if t.startswith('F0')]
        static = [t for t in tasks if t.startswith('F') and not t.startswith('F0')]
        personal = [t for t in tasks if t.startswith('P')]
        interleaved = [x for k in range(max(len(static),len(personal)))
                       for group in (personal,static) if k < len(group) for x in [group[k]]]
        queues[gpu] = base+interleaved
    active = {}; attempts = {}; failed = False; stopping = [False]
    def stop(signum, frame): stopping[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    # A resumed scheduler can monitor surviving orphan workers rather than
    # spawning duplicate writers after an abrupt scheduler-only interruption.
    class ExistingProcess:
        def __init__(self, pid): self.pid = pid
        def poll(self): return None if self.pid in active_workers() else 0
        def send_signal(self, sig): os.kill(self.pid, sig)
    for pid in active_workers():
        try:
            args = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        except FileNotFoundError:
            continue
        if not all(flag in args for flag in ('worker', '--config', '--seed')):
            continue
        task = args[args.index('--config')+1]+'_seed'+args[args.index('--seed')+1]
        assert task in GPU and task not in active
        active[task] = (ExistingProcess(pid), None)
        queues[GPU[task]].remove(task)
        log(f'ADOPT {task} pid={pid}')
    def launch(task):
        config, seed = task.split('_seed')
        attempts[task] = attempts.get(task, 0)+1
        output = (EXP/'logs'/f'{task}.log').open('a', buffering=1)
        output.write(f'\nATTEMPT {utc()} config={config} seed={seed} resume-safe\n')
        p = subprocess.Popen([sys.executable, str(EXP/'run.py'), 'worker', '--config', config, '--seed', seed],
                             cwd=ROOT, env=worker_env(GPU[task]), stdout=output, stderr=subprocess.STDOUT)
        active[task] = (p, output)
        m = read_json(MANIFEST); m['tasks'][task].update(status='RUNNING', pid=p.pid, attempt=attempts[task])
        atomic_json(MANIFEST, m); log(f'LAUNCH {task} gpu={GPU[task]} pid={p.pid}')
    try:
        last_refresh = time.monotonic()
        while active or any(queues.values()):
            for task, (p, output) in list(active.items()):
                rc = p.poll()
                if rc is None: continue
                if output: output.close()
                del active[task]
                d = task_result(task)
                if rc == 0 and d and d['status'] == 'COMPLETE': log('SUCCESS '+task)
                elif not stopping[0] and rc != 130 and attempts.get(task, 0) < 2:
                    log(f'RETRY {task} rc={rc} from atomic state'); launch(task)
                else:
                    m = read_json(MANIFEST)
                    m['tasks'][task].update(status='INTERRUPTED' if rc == 130 else 'FAILED', exit_code=rc)
                    atomic_json(MANIFEST, m); failed = True
                    log(f'INCOMPLETE {task} rc={rc}'); stopping[0] = True
            if stopping[0]: break
            for gpu in (0,1):
                count = sum(GPU[task] == gpu for task in active)
                while count < workers_per_gpu and queues[gpu]:
                    task = queues[gpu].pop(0); d = task_result(task)
                    if d and d['status'] == 'COMPLETE': log('SKIP verified '+task); continue
                    launch(task); count += 1
            if time.monotonic()-last_refresh >= 60:
                result = report(); last_refresh = time.monotonic()
                if result['run_status'] == 'COMPLETE':
                    # report may clean task records after the last child exits
                    # but before the next poll; COMPLETE is terminal.
                    for p, output in active.values():
                        assert p.poll() is not None
                        if output: output.close()
                    active.clear()
                    break
            if active: time.sleep(3)
        if stopping[0]:
            log('STOP: workers save completed epochs; finished folds retained')
            for p, _ in active.values():
                try: p.send_signal(signal.SIGTERM)
                except ProcessLookupError: pass
            deadline = time.monotonic()+60
            while active and time.monotonic() < deadline:
                for task,(p, output) in list(active.items()):
                    if p.poll() is not None:
                        if output: output.close()
                        del active[task]
                if active: time.sleep(1)
            if active: raise RuntimeError('workers have not finished saving; do not start duplicate workers')
            m = read_json(MANIFEST); m['run_status'] = 'FAILED' if failed else 'INTERRUPTED'; atomic_json(MANIFEST,m)
        subprocess.run(['nvidia-smi'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = report()
        if not stopping[0] and result['run_status'] != 'COMPLETE': raise RuntimeError('240 folds not complete')
    finally:
        stream.close(); lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','preflight','all','resume','report','worker'])
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--config', choices=CONFIGS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('prepare','preflight'): os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    if args.action == 'prepare': prepare()
    elif args.action == 'preflight': preflight()
    elif args.action in ('all','resume'): schedule(args.workers_per_gpu)
    elif args.action == 'report': report()
    else:
        if args.config is None or args.seed is None: parser.error('worker needs --config and --seed')
        sys.exit(worker(args.config,args.seed))


if __name__ == '__main__': main()
