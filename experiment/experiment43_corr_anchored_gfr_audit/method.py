"""Frozen E43 graph operators, train-only reliability estimation and graph audits."""
from __future__ import annotations
import types
import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ('PHENO', 'ANAT', 'FUNC', 'CORR')


def safe_supervision(scores, train_idx, train_labels):
    """Only unordered, non-self outer-train pairs; never accepts held-out labels."""
    ids = torch.as_tensor(train_idx, device=scores[0].device, dtype=torch.long)
    labels = torch.as_tensor(train_labels, device=ids.device, dtype=torch.long)
    assert labels.shape == ids.shape and len(ids.unique()) == len(ids)
    a, b = torch.triu_indices(len(ids), len(ids), offset=1, device=ids.device)
    target = (labels[a] == labels[b]).float()
    return torch.stack([(((s[ids[a], ids[b]] + 1.) / 2. - target) ** 2).mean() for s in scores]).mean()


def calibrate(score, train_idx):
    ids = torch.as_tensor(train_idx, dtype=torch.long, device=score.device)
    a, b = torch.triu_indices(len(ids), len(ids), 1, device=score.device)
    # CPU NumPy partition is deterministic and avoids a full accelerator sort.
    vals = score[ids[a], ids[b]].detach().cpu().numpy()
    q1, median, q3 = np.quantile(vals, [.25, .50, .75], method='linear')
    iqr = max(float(q3 - q1), 1e-3)
    calibrated = torch.sigmoid((score - float(median)) / iqr)
    return calibrated, {'median': float(median), 'iqr': iqr, 'train_pairs': len(vals)}


def mutual_support(score, k=96):
    assert score.shape[0] > k
    ranked = score.detach().clone()
    ranked.fill_diagonal_(-torch.inf)
    idx = torch.topk(ranked, k=k, dim=1, largest=True, sorted=True).indices
    directed = torch.zeros_like(score, dtype=torch.bool).scatter_(1, idx, True)
    return directed & directed.T


def permute_within_pools(n, train, test, fold, modality):
    g = np.random.default_rng(430000 + 100 * fold + modality)
    result = np.arange(n)
    result[train] = g.permutation(train)
    result[test] = g.permutation(test)
    assert set(result[train]) == set(train) and set(result[test]) == set(test)
    return result


def refined_graphs(scores, train, group, q, perms):
    cc, stats = zip(*(calibrate(s, train) for s in scores))
    masks = [mutual_support(s) for s in scores]
    if group in (3, 5):
        return [c * m for c, m in zip(cc, masks)], list(stats)
    assert group in (6, 7, 8, 9)
    base = masks[3] * cc[3]
    graphs = []
    for m in range(3):
        c = cc[m]
        if group == 8:
            perm = torch.as_tensor(perms[m], device=c.device, dtype=torch.long)
            c = c[perm][:, perm]
        # q=0 is an actual unused auxiliary path, not an artificial zero gradient.
        graphs.append(base if float(q[m]) == 0 else base * torch.exp(.25 * float(q[m]) * (2. * c - 1.)))
    return graphs + [base], list(stats)


def gfr_forward(self, x, model_num, label, index):
    from utils import normalize_adj
    group = self.e43_group
    raw = [getattr(self, f'GraphConstruct{i}')(x[i]) for i in range(model_num)]
    scores = [getattr(self, f'GraphConstruct{i}').last_score43 for i in range(model_num)]
    if group in (3, 5, 6, 7, 8, 9):
        graphs, self.e43_calibration = refined_graphs(scores, index, group, self.e43_q, self.e43_perms)
    else:
        graphs = raw
        self.e43_calibration = []
    # Retain the historical, unused ALoss calculation; it never enters total loss.
    n = len(x[0])
    new_label = label.clone()
    new_label[new_label == 0] = -1
    new_label = torch.tensor([v if i in index else 0 for i, v in enumerate(new_label)])
    a = new_label.unsqueeze(1).repeat(1, n)
    b = new_label.t().repeat(n, 1)
    target = normalize_adj(a.eq_(b).cuda() + torch.eye(n).cuda())
    full = torch.cat(x, dim=1)
    combined = None
    aloss = 0.
    for i, graph in enumerate(graphs):
        fw = torch.sigmoid(self.fusion_weight[i])
        normalized = normalize_adj(graph + torch.eye(n, device=full.device))
        aloss = torch.linalg.norm(normalized - target) + aloss
        value = full
        if group == 4:
            value = torch.zeros_like(full)
            start, stop = self.e43_slices[i]
            value[:, start:stop] = full[:, start:stop]
        msg = getattr(self, f'MessagePassing{i}')(value, normalized)
        if i == 0:
            combined = fw * msg + full
        elif i == 1 and group in (2, 3):
            combined = fw * msg + full
        else:
            combined = fw * msg + combined
    return combined, graphs, aloss / model_num


def install(agent, group, fold, q=None, slices=None):
    g = agent.MessagePassing0
    g.e43_group = group
    g.e43_q = np.asarray([1/3]*3 if group == 7 else [0.]*3 if group == 9 else q if q is not None else [0.]*3)
    g.e43_perms = [permute_within_pools(len(agent.feat), agent.trn_idx, agent.tst_idx, fold, m) for m in range(3)]
    g.e43_slices = slices
    g.e43_messages = {}
    g.e43_off = None
    def hook(i):
        def observe(module, inputs, output):
            g.e43_messages[i] = output.detach()
            return torch.zeros_like(output) if g.e43_off == i else output
        return observe
    for i in range(4):
        getattr(g, f'MessagePassing{i}').register_forward_hook(hook(i))
    if group != 0:
        g.forward = types.MethodType(gfr_forward, g)


def estimate_reliability(x, train, train_labels, fold, slices, pheno_indices):
    """Subject-disjoint inner fitting; arguments contain no outer-test labels."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(train_labels, dtype=np.int64)
    assert labels.shape == (len(train),)
    raw = np.asarray(x[train], np.float64)
    sims = []
    for start, stop in slices:
        vals = raw[:, start:stop]
        norm = vals / np.maximum(np.linalg.norm(vals, axis=1, keepdims=True), 1e-12)
        sims.append(norm @ norm.T)
    age = raw[:, pheno_indices['age']].argmax(1)
    sex = raw[:, pheno_indices['sex']].argmax(1)
    site = raw[:, pheno_indices['site']].argmax(1)
    def pairs(ids):
        a, b = np.triu_indices(len(ids), 1)
        a, b = ids[a], ids[b]
        features = np.column_stack([sims[3][a,b], np.abs(age[a]-age[b]), sex[a]==sex[b], site[a]==site[b], *[s[a,b] for s in sims[:3]]])
        return features, (labels[a] == labels[b]).astype(np.int64)
    records = []
    inner = StratifiedKFold(5, shuffle=True, random_state=43000 + fold)
    for inner_fold, (fit, heldout) in enumerate(inner.split(raw, labels), 1):
        assert not set(fit) & set(heldout)
        xf, yf = pairs(fit)
        xv, yv = pairs(heldout)
        assert set(yf) == set(yv) == {0, 1}
        aucs = []
        for aux in [-1, 0, 1, 2]:
            cols = [0,1,2,3] + ([4+aux] if aux >= 0 else [])
            scaler = StandardScaler().fit(xf[:, cols])
            model = LogisticRegression(C=1., solver='lbfgs', max_iter=500, tol=1e-6, class_weight='balanced', random_state=0)
            model.fit(scaler.transform(xf[:, cols]), yf)
            assert model.n_iter_.max() < 500, 'reliability logistic convergence failure'
            aucs.append(float(roc_auc_score(yv, model.predict_proba(scaler.transform(xv[:, cols]))[:,1])))
        records.append({'inner_fold':inner_fold,'fit_subjects':train[fit].tolist(),'validation_subjects':train[heldout].tolist(),
                        'fit_pairs':len(yf),'validation_pairs':len(yv),'base_auc':aucs[0],
                        'aux_auc':aucs[1:],'delta':[v-aucs[0] for v in aucs[1:]]})
    delta = np.mean([r['delta'] for r in records], axis=0)
    positive = np.maximum(delta, 0.)
    q = positive / positive.sum() if positive.sum() > 0 else positive
    return {'fold':fold,'delta':delta.tolist(),'q':q.tolist(),'inner_records':records,'label_boundary':'outer_train_only_subject_disjoint'}


def graph_audit(adj, train, labels_train, age, sex, site):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    a = adj.detach().cpu().numpy()
    mask = a > 0
    np.fill_diagonal(mask, False)
    degree = mask.sum(1)
    weights = a[mask]
    n = len(a)
    i, j = np.triu_indices(len(train), 1)
    aa, bb = train[i], train[j]
    edge = mask[aa,bb]
    w = a[aa,bb]
    same = labels_train[i] == labels_train[j]
    def corr(v):
        return float(np.corrcoef(w, v.astype(float))[0,1]) if np.std(w)>0 and np.std(v)>0 else 0.
    return {'offdiagonal_directed_edges':int(mask.sum()),'density':float(mask.sum()/(n*(n-1))),
            'degree_mean':float(degree.mean()),'degree_median':float(np.median(degree)),
            'degree_p05':float(np.quantile(degree,.05)),'degree_p95':float(np.quantile(degree,.95)),
            'degree_max':int(degree.max()),'isolated':int((degree==0).sum()),
            'components':int(connected_components(csr_matrix(mask),directed=False,return_labels=False)),
            'edge_weight_mean':float(weights.mean()) if len(weights) else 0.,'edge_weight_p05':float(np.quantile(weights,.05)) if len(weights) else 0.,
            'edge_weight_p95':float(np.quantile(weights,.95)) if len(weights) else 0.,
            'symmetry_max':float(np.abs(a-a.T).max()),'diagonal_mean':float(np.diag(a).mean()),
            'train_same_class_edge_fraction':float(same[edge].mean()) if edge.any() else 0.,
            'train_same_class_density':float(edge[same].mean()),'train_different_class_density':float(edge[~same].mean()),
            'train_weight_same_class_corr':corr(same),'train_weight_age_bin_diff_corr':corr(abs(age[aa]-age[bb])),
            'train_weight_same_sex_corr':corr(sex[aa]==sex[bb]),'train_weight_same_site_corr':corr(site[aa]==site[bb]),
            'within_site_weight_fraction':float(a[mask & (site[:,None]==site[None,:])].sum()/max(weights.sum(),1e-30))}
