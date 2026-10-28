from __future__ import annotations
import copy, hashlib, json, os, pickle, random, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(os.environ.get('SPROMPTGL_ROOT', '/data/adodas/SPromptGL-main'))
EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'
SRC = RT / 'sources'
DATA_DIR = ROOT / 'data' / 'ABIDE'
HIST = ROOT / 'experiment' / 'experiment43_2_gfr_incremental_value_audit'
CFG = json.loads((EXP / 'config.json').read_text())
INPUT_DIMS = [48, 6, 10, 256]
CTX = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest_obj(v):
    h = hashlib.sha256()
    def add(x):
        import torch
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda z: repr(z)):
                h.update(repr(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x: add(y)
        else:
            h.update(repr(x).encode())
    add(v)
    return h.hexdigest()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_pack():
    import torch
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'cpu': torch.get_rng_state().cpu(),
        'cuda': [x.cpu() for x in torch.cuda.get_rng_state_all()],
    }


def restore_rng(r):
    import torch
    random.setstate(r['python'])
    np.random.set_state(r['numpy'])
    torch.set_rng_state(r['cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


def rng_sha(r):
    return digest_obj(r)


def get_runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        import sys
        sys.path.insert(0, str(SRC))
        import layers, network, utils
        assert not hasattr(network, 'PLN')
        assert not hasattr(network, 'MLP')
        assert 'CLIP' not in sys.modules
        CTX = {'torch': torch, 'layers': layers, 'network': network, 'utils': utils}
    return CTX


def load_data():
    import pandas as pd
    frame = pd.read_csv(DATA_DIR / 'processed_standard_data.csv')
    mapping = np.load(DATA_DIR / 'modal_feat_dict.npy', allow_pickle=True).item()
    assert list(mapping) == ['PHENO', 'ANAT', 'FUNC', 'Correlation']
    assert [len(x) for x in mapping.values()] == INPUT_DIMS
    assert list(frame.columns[:-1]) == [str(x) for xs in mapping.values() for x in xs]
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    return x, y, mapping


def make_folds(x, y, cv_seed):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=10, shuffle=True, random_state=int(cv_seed)).split(x, y))


def split_sha(train, test):
    return hashlib.sha256(np.asarray(train, dtype=np.int64).tobytes() + np.asarray(test, dtype=np.int64).tobytes()).hexdigest()


def all_split_sha(folds):
    h = hashlib.sha256()
    for tr, te in folds:
        h.update(np.asarray(tr, dtype=np.int64).tobytes())
        h.update(np.asarray(te, dtype=np.int64).tobytes())
    return h.hexdigest()


def cpu_clone(v):
    import torch
    if isinstance(v, torch.Tensor): return v.detach().cpu().clone()
    if isinstance(v, dict): return {k: cpu_clone(x) for k, x in v.items()}
    if isinstance(v, list): return [cpu_clone(x) for x in v]
    if isinstance(v, tuple): return tuple(cpu_clone(x) for x in v)
    return copy.deepcopy(v)


def state_dict_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def module_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def load_state(module, state):
    module.load_state_dict(state)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def read_json(path): return json.loads(Path(path).read_text())


def save_pt(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_pt(path):
    import torch
    return torch.load(path, map_location='cuda', weights_only=False)


def make_pln(torch, layers, dims, hp):
    import torch.nn as nn
    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyperpm = hp
            self.input_data_dims = dims
            self.d_q = self.d_k = self.d_v = hp.n_hidden
            self.n_head = hp.n_head
            self.dropout = hp.dropout
            self.n_layer = hp.nlayer
            self.modal_num = hp.nmodal
            self.n_class = hp.nclass
            self.d_out = self.d_v * self.n_head * self.modal_num
            self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
            self.InputLayer = layers.VariLengthInputLayer(dims, self.d_k, self.d_v, self.n_head, self.dropout)
            self.Encoder = []
            self.FeedForward = []
            for i in range(self.n_layer):
                enc = layers.EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
                self.add_module('encode_%d' % i, enc); self.Encoder.append(enc)
                ff = layers.FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
                self.add_module('feed_%d' % i, ff); self.FeedForward.append(ff)
            self.Outputlayer = layers.OutputLayer(self.d_v * self.n_head * self.modal_num,
                                                   self.d_v * self.n_head, self.n_class,
                                                   self.modal_num, self.dropout)
        def forward(self, x):
            bs = x.size(0); attn_map = []
            x, input_attention = self.InputLayer(x)
            attn = input_attention.mean(dim=1); attn_map.append(attn.detach().cpu().numpy())
            for i in range(self.n_layer):
                x, layer_attention = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
                attn = layer_attention.mean(dim=1); x = self.FeedForward[i](x)
                attn_map.append(attn.detach().cpu().numpy())
            output, hidden = self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))
            return output, hidden, attn_map
    m = PLNNoPEN()
    assert sum(p.numel() for p in m.parameters()) == 49544
    assert not hasattr(m, 'MLP') and not hasattr(m, 'prompt_text_features') and not hasattr(m, 'avg_pool')
    return m


def make_agent(x, y, train_idx, test_idx, hp):
    from collections import Counter
    import torch
    c = get_runtime(); layers, network, utils = c['layers'], c['network'], c['utils']
    class Agent: pass
    a = Agent(); a.dev = torch.device('cuda'); a.hyperpm = hp
    a.feat = torch.from_numpy(x).float().to(a.dev); a.targ = torch.from_numpy(y).long().to(a.dev)
    a.trn_idx = np.asarray(train_idx, dtype=np.int64); a.val_idx = a.tst_idx = np.asarray(test_idx, dtype=np.int64)
    counts = Counter(a.targ[a.trn_idx].cpu().numpy())
    a.weight = torch.from_numpy(len(a.trn_idx) / np.asarray(list(counts.values())) / hp.nclass).float().to(a.dev)
    a.dataLayer = layers.InitialDataLayer(INPUT_DIMS)
    a.MessagePassing0 = network.GFR(INPUT_DIMS, INPUT_DIMS, th=hp.th, mode=hp.GC_mode).to(a.dev)
    a.ModalFusion = make_pln(torch, layers, INPUT_DIMS, hp).to(a.dev)
    a.optimizer_MP0 = torch.optim.Adam(a.MessagePassing0.parameters(), lr=hp.lr, weight_decay=hp.reg)
    a.optimizer_MF = torch.optim.Adam(a.ModalFusion.parameters(), lr=hp.lr, weight_decay=hp.reg)
    a.ModalFusion.apply(utils.my_weight_init)
    assert sum(p.numel() for p in a.MessagePassing0.parameters()) == 887836
    assert sum(p.numel() for p in a.ModalFusion.parameters()) == 49544
    assert all(abs(g['weight_decay'] - hp.reg) < 1e-12 for o in [a.optimizer_MP0, a.optimizer_MF] for g in o.param_groups)
    return a


def forward(a):
    c = get_runtime(); torch, utils = c['torch'], c['utils']
    n, parts = a.dataLayer(a.feat)
    feat, adjs, aloss = a.MessagePassing0(parts, n, a.targ, a.trn_idx)
    gl = sum(utils.GraphConstructLoss(parts[i], adjs[i], a.hyperpm.theta_smooth,
                                      a.hyperpm.theta_degree, a.hyperpm.theta_sparsity) for i in range(n)) / 4
    logp, hidden, attn = a.ModalFusion(feat)
    cl = utils.ClsLoss(logp, a.targ, a.trn_idx, a.weight)
    return logp, cl, gl, aloss, adjs, hidden, attn


def evaluate(a):
    a.MessagePassing0.eval(); a.ModalFusion.eval()
    with get_runtime()['torch'].no_grad(): return forward(a)


def train_step(a):
    torch = get_runtime()['torch']
    a.MessagePassing0.train(); a.ModalFusion.train()
    a.optimizer_MP0.zero_grad(); a.optimizer_MF.zero_grad()
    out, cl, gl, aloss, adjs, hidden, attn = forward(a)
    loss = cl + gl
    assert torch.isfinite(out).all() and torch.isfinite(loss)
    loss.backward()
    for m in [a.MessagePassing0, a.ModalFusion]:
        for p in m.parameters():
            if p.grad is not None: assert torch.isfinite(p.grad).all()
    a.optimizer_MP0.step(); a.optimizer_MF.step()
    return {'cls_loss': float(cl.detach()), 'graph_loss': float(gl.detach()), 'total_loss': float(loss.detach()), 'aloss': float(aloss.detach())}


def metric_row(out, y, test_idx):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    pred = out.detach().cpu().numpy()[test_idx].argmax(1).astype(np.int64)
    targ = y[test_idx].astype(np.int64)
    tp, fn, fp, tn = map(int, confusion_matrix(targ, pred, labels=[0, 1]).ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[targ], np.eye(2)[pred]))
    return {'acc': float((pred == targ).mean()), 'hard_auc': hard_auc,
            'spe': float(tn / (tn + fp)), 'sen': float(tp / (tp + fn)),
            'asd_tp': tp, 'asd_fn': fn, 'td_fp': fp, 'td_tn': tn,
            'prediction': pred}


def model_cpu(a):
    return {'gfr': state_dict_cpu(a.MessagePassing0), 'pln': state_dict_cpu(a.ModalFusion)}


def load_model(a, state):
    a.MessagePassing0.load_state_dict(state['gfr']); a.ModalFusion.load_state_dict(state['pln'])


def digest_model_state(state): return digest_obj(state)


def history_reference():
    import csv
    p = HIST / 'results' / 'fold_metrics.csv'
    if not p.exists(): return {}
    rows = list(csv.DictReader(p.open()))
    return {int(r['fold']): r for r in rows if r.get('group_id') in ('0', 0) or r.get('group') == 'E00_FULL_GFR_B0B_REPLAY'}


def baseline_anchor(fold):
    p = HIST / '.runtime' / 'anchors' / f'fold{fold:02d}.pt'
    return load_pt(p)


def source_signature():
    paths = [EXP / 'config.json', EXP / 'integration.py', EXP / 'run.py', EXP / 'audit.py'] + sorted(SRC.glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)): sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()


def make_hp(): return SimpleNamespace(**CFG['protocol'])
