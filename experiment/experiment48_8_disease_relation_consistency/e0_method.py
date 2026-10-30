from __future__ import annotations
from types import SimpleNamespace
import math, hashlib, random, contextlib, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DIMS = (48, 6, 10, 256)
SLICES = ((0, 48), (48, 54), (54, 64), (64, 320))
MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")

class HP(SimpleNamespace):
    def __init__(self):
        super().__init__(n_hidden=18, n_head=2, nlayer=1, nmodal=4, nclass=2,
                         dropout=.35, th=.9, GC_mode="Sigmoid-like", MP_mode="GCN",
                         MF_mode="CAM", theta_smooth=1., theta_degree=.5, theta_sparsity=0.)

class PLNNoPEN(nn.Module):
    """The isolated B0-B CAM: original PLN attention/output path, PEN absent."""
    def __init__(self, layers, hp):
        super().__init__()
        self.hyperpm = hp
        self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
        self.InputLayer = layers.VariLengthInputLayer(DIMS, hp.n_hidden, hp.n_hidden, hp.n_head, hp.dropout)
        self.Encoder, self.FeedForward = [], []
        for i in range(hp.nlayer):
            enc = layers.EncodeLayer(hp.n_hidden * hp.n_head, hp.n_hidden, hp.n_hidden, hp.n_head, hp.dropout)
            ff = layers.FeedForwardLayer(hp.n_hidden * hp.n_head, hp.n_hidden * hp.n_head, dropout=hp.dropout)
            self.add_module(f"encode_{i}", enc); self.add_module(f"feed_{i}", ff)
            self.Encoder.append(enc); self.FeedForward.append(ff)
        self.Outputlayer = layers.OutputLayer(hp.n_hidden * hp.n_head * 4, hp.n_hidden * hp.n_head, 2, 4, hp.dropout)

    def forward(self, x):
        bs = x.size(0)
        x, attn = self.InputLayer(x)
        attn = attn.mean(dim=1)
        for i in range(self.hyperpm.nlayer):
            x, _ = self.Encoder[i](q=x, k=x, v=x, modal_num=4)
            x = self.FeedForward[i](x)
        return self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))

class SymmetricRelationScorer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim + 1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, zi, zj, cosine):
        return self.net(torch.cat([(zi - zj).abs(), zi * zj, cosine.unsqueeze(-1)], dim=-1)).squeeze(-1)

class APCR(nn.Module):
    def __init__(self, alpha_init=.01, alpha_max=.2, alpha_trainable=True):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, 32, bias=False) for d in DIMS])
        self.scorer = nn.ModuleList([SymmetricRelationScorer(32) for _ in DIMS])
        self.message = nn.ModuleList([
            nn.Sequential(nn.Linear(320, 64, bias=False), nn.GELU(), nn.Linear(64, 320, bias=False))
            for _ in DIMS
        ])
        self.gate = nn.ModuleList([nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 1)) for _ in DIMS])
        self.alpha_max = float(alpha_max)
        self.zero_alpha = bool(float(alpha_init) == 0.0)
        ratio = max(float(alpha_init), 1e-8) / self.alpha_max
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(ratio / (1. - ratio)), dtype=torch.float32))
        self.alpha_raw.requires_grad_(bool(alpha_trainable))
        self.register_buffer("reliability", torch.full((4,), .5))
        self.last = {}

    def alpha(self):
        if self.zero_alpha:
            return self.alpha_raw.new_zeros(())
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def representations(self, parts):
        # p=2 and dim are explicit. Projection gradients remain connected.
        return [F.normalize(self.proj[m](x), p=2, dim=-1, eps=1e-8) for m, x in enumerate(parts)]

    def candidates(self, raw_scores):
        ids, values = [], []
        for score in raw_scores:
            s = score.clone()
            s.fill_diagonal_(-torch.inf)
            v, ix = torch.topk(s, k=24, dim=1)
            ids.append(ix); values.append(v)
        return ids, values

    def score_candidate_edges(self, z, ids, raw_values):
        qs = []
        for m in range(4):
            zi = z[m].unsqueeze(1).expand(-1, ids[m].shape[1], -1)
            zj = z[m][ids[m]]
            cosine = (zi * zj).sum(-1).clamp(-1., 1.)
            qs.append(torch.sigmoid(self.scorer[m](zi, zj, cosine)))
        return qs

    def union_pair_scores(self, z, candidate_ids, fit_mask):
        """Score all modalities on exactly the same unordered patient pairs."""
        n = z[0].shape[0]; device = z[0].device
        keys_by_view = []
        for m in range(4):
            recv = torch.arange(n, device=device).unsqueeze(1).expand_as(candidate_ids[m])
            send = candidate_ids[m]
            keep = fit_mask[recv] & fit_mask[send] & (recv != send)
            lo = torch.minimum(recv[keep], send[keep]); hi = torch.maximum(recv[keep], send[keep])
            keys_by_view.append(lo * n + hi)
        keys = torch.unique(torch.cat(keys_by_view), sorted=True)
        if keys.numel() == 0:
            empty = z[0].new_empty((0,))
            return torch.empty((0, 2), device=device, dtype=torch.long), [empty] * 4, torch.empty((0, 4), device=device, dtype=torch.bool)
        pair = torch.stack([torch.div(keys, n, rounding_mode="floor"), keys.remainder(n)], dim=1)
        supports = torch.stack([torch.isin(keys, keys_by_view[m]) for m in range(4)], dim=1)
        qs = []
        for m in range(4):
            zi = z[m][pair[:, 0]]; zj = z[m][pair[:, 1]]
            cosine = (zi * zj).sum(-1).clamp(-1., 1.)
            qs.append(torch.sigmoid(self.scorer[m](zi, zj, cosine)))
        return pair, qs, supports

    def sca_loss(self, pair_q, support, reliability):
        # Alignment compares the same patient IDs, only where both views supplied candidates.
        losses, weights = [], []
        for m in range(4):
            for n in range(m + 1, 4):
                qm, qn = pair_q[m], pair_q[n]
                support_pair = support[:, m] & support[:, n]
                confidence = (2. * (qm.detach() - .5).abs()) * (2. * (qn.detach() - .5).abs())
                agreement = (1. - (qm.detach() - qn.detach()).abs()).clamp(0., 1.)
                rho = (reliability[m] * reliability[n]).clamp(.01, 1.)
                w = support_pair.to(qm.dtype) * confidence * agreement * rho
                losses.append(((qm - qn).square() * w).sum())
                weights.append(w.sum())
        den = torch.stack(weights).sum() + 1e-8
        return torch.stack(losses).sum() / den, torch.stack(weights).sum().detach()

    def residual(self, parts, z, ids, raw_values, mode, support_weights=None, attacked=None, wrong_perm=None):
        """Return zero-self-loop residual and audit tensors for an explicit edge set."""
        x = torch.cat(parts, dim=1)
        q_all = self.score_candidate_edges(z, ids, raw_values)
        chosen, edge_weight, deltas, gates = [], [], [], []
        for m in range(4):
            order = q_all[m].topk(16, dim=1).indices
            sel = ids[m].gather(1, order)
            qsel = q_all[m].gather(1, order)
            if mode == "raw":
                w = raw_values[m].gather(1, order).softmax(dim=1)
            else:
                w = qsel / (qsel.sum(dim=1, keepdim=True) + 1e-8)
            if attacked is not None:
                sel = attacked[m]
                zi = z[m].unsqueeze(1).expand(-1, sel.shape[1], -1)
                zj = z[m][sel]
                cosine = (zi * zj).sum(-1).clamp(-1., 1.)
                qsel = torch.sigmoid(self.scorer[m](zi, zj, cosine))
                w = qsel / (qsel.sum(dim=1, keepdim=True) + 1e-8)
            if wrong_perm is not None:
                sel = wrong_perm[sel]
                mapped_pool = wrong_perm[ids[m]]
                recv = torch.arange(len(x), device=x.device).unsqueeze(1)
                used = (mapped_pool.unsqueeze(2) == sel.unsqueeze(1)).any(2)
                valid = (~used) & (mapped_pool != recv)
                invalid = sel == recv
                safe = mapped_pool.gather(1, valid.to(torch.int64).argmax(1, keepdim=True)).expand_as(sel)
                sel = torch.where(invalid, safe, sel)
                zi = z[m].unsqueeze(1).expand(-1, sel.shape[1], -1); zj=z[m][sel]
                cs=(zi*zj).sum(-1).clamp(-1.,1.); qsel=torch.sigmoid(self.scorer[m](zi,zj,cs))
                w=(raw_values[m].new_ones(qsel.shape)/qsel.shape[1]) if mode=="raw" else qsel/(qsel.sum(1,keepdim=True)+1e-8)
            chosen.append(sel); edge_weight.append(w)
            if mode == "self":
                # Capacity-matched control: each output depends on this patient only.
                msg = x
                self_cos = torch.ones(len(x), device=x.device, dtype=x.dtype)
                qself = torch.sigmoid(self.scorer[m](z[m], z[m], self_cos))
                rel_mean = qself; rel_std = torch.zeros_like(qself)
            elif mode == "raw":
                rel_mean = ((raw_values[m] + 1.) * .5).mean(dim=1).clamp(0.,1.)
                rel_std = raw_values[m].std(dim=1).clamp(0.,1.)
                msg = ((x[sel] - x.unsqueeze(1)) * w.unsqueeze(-1)).sum(dim=1)
            else:
                msg = ((x[sel] - x.unsqueeze(1)) * w.unsqueeze(-1)).sum(dim=1)
                rel_mean = q_all[m].mean(dim=1); rel_std = q_all[m].std(dim=1)
            delta = self.message[m](msg)
            rel_conf = (2. * (rel_mean - .5).abs()).clamp(0., 1.)
            quality = torch.full_like(rel_conf, float(self.reliability[m].detach()))
            gate_input = torch.stack([rel_mean, rel_std, quality], dim=1)
            gate = torch.sigmoid(self.gate[m](gate_input).squeeze(-1))
            deltas.append(delta); gates.append(gate)
        residual = sum((g.unsqueeze(-1) * d for g, d in zip(gates, deltas)), torch.zeros_like(x))
        self.last = {"candidate_ids": ids, "chosen": chosen, "q_candidates": q_all,
                     "edge_weights": edge_weight, "deltas": deltas, "gates": gates,
                     "residual": residual, "alpha": self.alpha()}
        return residual

def build_base(torch_mod, layers, network, utils, device, x, y, tr, te):
    hp = HP()
    gfr = network.GFR(DIMS, DIMS, th=.9, mode="Sigmoid-like").to(device)
    cam = PLNNoPEN(layers, hp).to(device)
    # Match the original 43.1 fresh-agent RNG consumption before loading anchors.
    cam.apply(utils.my_weight_init)
    feat = torch_mod.from_numpy(x).float().to(device); targ = torch_mod.from_numpy(y).long().to(device)
    ids = torch_mod.as_tensor(tr, device=device)
    cnt = torch_mod.bincount(targ[ids], minlength=2).float()
    weight = (len(tr) / cnt / 2.).to(device)
    return SimpleNamespace(dev=device, hyperpm=hp, feat=feat, targ=targ, trn_idx=np.asarray(tr),
        tst_idx=np.asarray(te), val_idx=np.asarray(te), weight=weight,
        dataLayer=layers.InitialDataLayer(DIMS), MessagePassing0=gfr, ModalFusion=cam)

def base_bundle(a):
    return {"gfr": {k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},
            "pln": {k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()}}

def load_base(a, b):
    a.MessagePassing0.load_state_dict(b["gfr"]); a.ModalFusion.load_state_dict(b["pln"])

def capture_rng(torch_mod):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch_mod.get_rng_state(), "cuda": torch_mod.cuda.get_rng_state_all()}

def restore_rng(torch_mod, state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch_mod.set_rng_state(state["cpu"].cpu())
    torch_mod.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])

@contextlib.contextmanager
def apcr_rng_scope(a):
    """Run APCR-only work on its own continuous RNG stream.

    The base GFR/CAM stream is restored after the scope, so extra APCR
    candidate work cannot shift the historical CAM dropout stream.
    """
    t = a._torch
    main_state = capture_rng(t)
    restore_rng(t, a.apcr_rng)
    try:
        yield
    finally:
        a.apcr_rng = capture_rng(t)
        restore_rng(t, main_state)

def forked_apcr(torch_mod, utils, fold, alpha_init=.0022, alpha_trainable=True, seed_base=481000):
    base = capture_rng(torch_mod)
    seed = seed_base + fold
    random.seed(seed); np.random.seed(seed); torch_mod.manual_seed(seed); torch_mod.cuda.manual_seed_all(seed)
    apcr = APCR(alpha_init, .2, alpha_trainable).cuda(); apcr.apply(utils.my_weight_init)
    apcr._init_rng = capture_rng(torch_mod)
    # Re-establish the base model's RNG stream before any CAM dropout is sampled.
    restore_rng(torch_mod, base)
    # my_weight_init may touch scalar parameters; exact alpha initialization is contractual.
    with torch_mod.no_grad():
        r = max(float(alpha_init), 1e-8)/.2; apcr.alpha_raw.fill_(math.log(r/(1-r))); apcr.zero_alpha = bool(float(alpha_init)==0.0)
    apcr.alpha_raw.requires_grad_(bool(alpha_trainable))
    assert abs(float(apcr.alpha().detach()) - float(alpha_init)) <= max(1e-10,abs(float(alpha_init))*1e-6)
    return apcr

def make_agent(torch_mod, layers, network, utils, device, x, y, tr, te, fold, gid, anchor, seed):
    # Fresh base construction starts from the archived pre-init RNG boundary.
    run_restore = None
    if "entry_rng" in anchor:
        import random
        random.setstate(anchor["entry_rng"]["python"])
        np.random.set_state(anchor["entry_rng"]["numpy"])
        torch_mod.set_rng_state(anchor["entry_rng"]["cpu"].cpu())
        torch_mod.cuda.set_rng_state_all([v.cpu() for v in anchor["entry_rng"]["cuda"]])
    a = build_base(torch_mod, layers, network, utils, device, x, y, tr, te)
    a.MessagePassing0.load_state_dict(anchor["model"]["gfr"]); a.ModalFusion.load_state_dict(anchor["model"]["pln"])
    a.optimizer_MP0 = torch_mod.optim.Adam(a.MessagePassing0.parameters(), lr=.0038, weight_decay=.11)
    a.optimizer_MF = torch_mod.optim.Adam(a.ModalFusion.parameters(), lr=.0038, weight_decay=.11)
    a.optimizer_MP0.load_state_dict(anchor["mp"]); a.optimizer_MF.load_state_dict(anchor["mf"])
    if "rng" in anchor and "entry_rng" in anchor:
        # Construction must land on the archived post-init RNG boundary before APCR gets a fork.
        current = torch_mod.get_rng_state()
        target = anchor["rng"]["cpu"].cpu()
        if not torch_mod.equal(current, target):
            raise RuntimeError("fresh B0-B initialization did not reproduce archived CPU RNG boundary")
        cuda_now=torch_mod.cuda.get_rng_state_all()
        if len(cuda_now)!=len(anchor["rng"]["cuda"]) or any(not torch_mod.equal(u.cpu(),v.cpu()) for u,v in zip(cuda_now,anchor["rng"]["cuda"])):
            raise RuntimeError("fresh B0-B initialization did not reproduce archived CUDA RNG boundary")
    # APCR initialization uses a forked RNG and cannot perturb B0-B dropout.
    alpha_trainable = bool(gid == 2)
    a.APCR = forked_apcr(torch_mod, utils, fold, alpha_init=.0022, alpha_trainable=alpha_trainable)
    if "rng" in anchor:
        import random
        random.setstate(anchor["rng"]["python"]); np.random.set_state(anchor["rng"]["numpy"])
        torch_mod.set_rng_state(anchor["rng"]["cpu"].cpu())
        torch_mod.cuda.set_rng_state_all([v.cpu() for v in anchor["rng"]["cuda"]])
    a.optimizer_APCR = torch_mod.optim.Adam(a.APCR.parameters(), lr=.00038, weight_decay=.0001)
    a._torch = torch_mod
    a.apcr_rng = copy.deepcopy(a.APCR._init_rng)
    a.fold, a.group, a.alpha_trainable = fold, gid, alpha_trainable
    return a

def base_forward(a, utils):
    n, parts = a.dataLayer(a.feat)
    base, adjs, aloss = a.MessagePassing0(parts, n, a.targ, a.trn_idx)
    graph = sum(utils.GraphConstructLoss(parts[m], adjs[m], 1., .5, 0.) for m in range(4)) / 4
    return parts, base, adjs, graph, aloss

def graph_scores(a):
    """Read live scores from the same four GraphLearn modules used by E51."""
    gfr = a.MessagePassing0
    if hasattr(gfr, "graphs"):
        return [gfr.graphs[m].score431 for m in range(4)]
    return [getattr(gfr, f"GraphConstruct{m}").score431 for m in range(4)]

def output(a, parts, base, adjs, graph, group, wrong_perm=None, attacked=None, include_sca=True):
    apcr = a.APCR
    scores = graph_scores(a)
    z = apcr.representations(parts); candidate_ids, raw_values = apcr.candidates(scores)
    feature = torch.cat(parts, dim=1)
    if group == 1:
        fused = base
        apcr.last = {"residual": torch.zeros_like(feature), "alpha": apcr.alpha(), "deltas": [torch.zeros_like(feature)]*4,
                     "gates": [torch.zeros(feature.shape[0],device=feature.device)]*4,
                     "candidate_ids": candidate_ids, "chosen": candidate_ids}
        q_all = apcr.score_candidate_edges(z, candidate_ids, raw_values)
        pair, pair_q, support = empty_sca(apcr, feature)
    elif group == 2:
        delta = apcr.residual(parts, z, candidate_ids, raw_values, "self")
        fused = base + apcr.alpha() * delta
        pair, pair_q, support = empty_sca(apcr, feature); q_all = apcr.last.get("q_candidates")
    else:
        q_all = apcr.score_candidate_edges(z, candidate_ids, raw_values)
        # Inner alignment pair set is restricted to outer-train fitting patients.
        if include_sca:
            fit_mask = torch.zeros(len(feature), dtype=torch.bool, device=feature.device)
            fit_mask[a.srd_fit_ids] = True
            pair, pair_q, support = apcr.union_pair_scores(z, candidate_ids, fit_mask)
        else:
            pair, pair_q, support = empty_sca(apcr, feature)
        mode = "raw" if group == 3 else "srd"
        if group == 7: mode = "srd"
        if attacked is None and wrong_perm is not None:
            sel=[]
            for m in range(4):
                order = (raw_values[m] if group == 3 else q_all[m]).topk(16,1).indices
                sel.append(candidate_ids[m].gather(1,order))
            attacked=sel
        delta = apcr.residual(parts,z,candidate_ids,raw_values,mode,attacked=attacked,wrong_perm=wrong_perm)
        fused = base + apcr.alpha() * delta
    logp = a.ModalFusion(fused)[0]
    return {"logp":logp,"base":base,"fused":fused,"parts":parts,"adjs":adjs,"graph":graph,
            "z":z,"candidate_ids":candidate_ids,"raw_values":raw_values,"q":q_all,"pair":pair,
            "pair_q":pair_q,"support":support,"residual":a.APCR.last["residual"],
            "deltas":a.APCR.last["deltas"],"gates":a.APCR.last["gates"],"alpha":a.APCR.alpha()}

def empty_sca(apcr, feature):
    return torch.empty((0,2),device=feature.device,dtype=torch.long),[feature.new_empty((0,)) for _ in range(4)],torch.empty((0,4),device=feature.device,dtype=torch.bool)

def partition_srd_patients(tr, labels, seed):
    """Fixed inner patient holdout: fit endpoints never appear in reliability patients."""
    rng=np.random.default_rng(seed); fit=[];cal=[]
    for cls in (0,1):
        ids=np.asarray([i for i in tr if labels[i]==cls],dtype=np.int64);rng.shuffle(ids)
        ncal=max(2,int(round(.2*len(ids))));cal.extend(ids[:ncal].tolist());fit.extend(ids[ncal:].tolist())
    return np.asarray(sorted(fit),dtype=np.int64),np.asarray(sorted(cal),dtype=np.int64)

def sample_pairs(tr_ids, labels, sites, epoch, seed, per_class=1024):
    """Balanced positive/negative pairs; ASD-ASD and TD-TD positives are equal."""
    rng=np.random.default_rng(seed + 1009*epoch)
    ids=np.asarray(tr_ids,dtype=np.int64);lab=np.asarray(labels);site=np.asarray(sites)
    by={c:ids[lab[ids]==c] for c in (0,1)}; pair_groups=[];targets=[];stats={}
    def draw(left,right,count,site_rule=None):
        chosen=set();out=[]
        for _ in range(32):
            size=max(256,4*(count-len(out)))
            aa=left[rng.integers(0,len(left),size=size)];bb=right[rng.integers(0,len(right),size=size)]
            keep=aa!=bb
            if site_rule is not None:
                same=site[aa]==site[bb];keep &= same if site_rule else ~same
            arr=np.stack([np.minimum(aa[keep],bb[keep]),np.maximum(aa[keep],bb[keep])],axis=1)
            if not len(arr):continue
            for v in np.unique(arr,axis=0):
                k=(int(v[0]),int(v[1]))
                if k not in chosen:
                    chosen.add(k);out.append(v)
                    if len(out)>=count:return np.asarray(out,dtype=np.int64)
        return np.asarray(out,dtype=np.int64).reshape(-1,2)
    for c in (0,1):
        p=draw(by[c],by[c],per_class)
        pair_groups.append(p);targets.extend([1]*len(p));stats[f"same_{c}"]=len(p)
    # Each positive class contributes `per_class` pairs; negatives have the same
    # total and are split evenly between same-site hard and cross-site pairs.
    same=draw(by[0],by[1],per_class,True);cross=draw(by[0],by[1],per_class,False)
    n_same=len(same);n_cross=len(cross)
    if n_same:
        pair_groups.append(same);targets.extend([0]*n_same)
    if n_cross:
        pair_groups.append(cross);targets.extend([0]*n_cross)
    stats["hard_negative_same_site"]=n_same;stats["negative_cross_site"]=n_cross
    pairs=np.concatenate(pair_groups,axis=0);ys=np.asarray(targets,dtype=np.float32)
    order=rng.permutation(len(pairs));pairs,ys=pairs[order],ys[order]
    return pairs,ys,stats

def fit_srd(a, z, tr, labels, sites, epoch, true_labels, shuffled_labels=None):
    fit_ids,cal_ids=a.srd_fit_ids,a.srd_cal_ids
    # E9 uses the exact same epoch/fold patient pairs as E6. Only the pair targets
    # change under the patient-level label permutation.
    true_labels=np.asarray(true_labels,dtype=np.int64)
    pairs,ys,stats=sample_pairs(fit_ids,true_labels,sites,epoch,481000+a.fold,1024)
    if shuffled_labels is not None:
        shuffled_labels=np.asarray(shuffled_labels,dtype=np.int64)
        ys=(shuffled_labels[pairs[:,0]]==shuffled_labels[pairs[:,1]]).astype(np.float32)
    pt=torch.as_tensor(pairs,device=a.dev,dtype=torch.long);yt=torch.as_tensor(ys,device=a.dev)
    losses=[]
    for m in range(4):
        zi=z[m][pt[:,0]];zj=z[m][pt[:,1]];cos=(zi*zj).sum(-1).clamp(-1,1)
        logits=a.APCR.scorer[m](zi,zj,cos);losses.append(F.binary_cross_entropy_with_logits(logits,yt))
    loss=torch.stack(losses).mean()
    # Patient-disjoint internal calibration pairs use only the held-out inner training patients.
    cal_pairs,cal_y,calstats=sample_pairs(cal_ids,true_labels,sites,epoch,981000+a.fold,256)
    if shuffled_labels is not None:
        cal_y=(shuffled_labels[cal_pairs[:,0]]==shuffled_labels[cal_pairs[:,1]]).astype(np.float32)
    cp=torch.as_tensor(cal_pairs,device=a.dev,dtype=torch.long);cy=torch.as_tensor(cal_y,device=a.dev)
    auc=[]
    with torch.no_grad():
        for m in range(4):
            zi=z[m][cp[:,0]];zj=z[m][cp[:,1]];cos=(zi*zj).sum(-1).clamp(-1,1)
            pr=torch.sigmoid(a.APCR.scorer[m](zi,zj,cos));auc.append(float(roc_auc_score(cy.cpu(),pr.cpu())))
        rel=torch.as_tensor(np.clip(2*(np.asarray(auc)-.5),.05,1.),device=a.dev,dtype=torch.float32)
        a.APCR.reliability.copy_(rel)
    stats.update({"fit_pairs":len(pairs),"fit_patient_coverage":len(np.unique(pairs)),"cal_pairs":len(cal_pairs),"cal_patient_coverage":len(np.unique(cal_pairs)),"pair_ids_sha256":hashlib.sha256(pairs.astype(np.int64).tobytes()).hexdigest(),"srd_target_flip_rate":float(np.mean(ys != (true_labels[pairs[:,0]]==true_labels[pairs[:,1]]).astype(np.float32))),"srd_label_flip_rate":float(np.mean(true_labels[fit_ids]!=shuffled_labels[fit_ids])) if shuffled_labels is not None else 0.})
    return loss,auc,stats

def candidate_attacks(a, clean_chosen, raw_candidate_ids, tr_fit, seed, step, start_edges=None):
    """Four declared graph-level candidates. Each attacks up to 32 fit patients and <=2 edges/view."""
    rng=np.random.default_rng(seed + 7919*getattr(a,"fold",0) + 101*step)
    targets=np.asarray(tr_fit,dtype=np.int64)
    if len(targets)>32:targets=np.sort(rng.choice(targets,32,replace=False))
    device=raw_candidate_ids[0].device;targets_t=torch.as_tensor(targets,device=device)
    origin = clean_chosen if start_edges is None else start_edges
    candidates=[]
    for rank in range(4):
        graph=[]
        for m in range(4):
            pool=raw_candidate_ids[m].detach().cpu().numpy()
            g=origin[m].detach().cpu().numpy().copy()
            for i in targets:
                selected=set(g[i].tolist());alts=[int(v) for v in pool[i].tolist() if int(v)!=i and int(v) not in selected]
                if not alts:continue
                # Two distinct edges per target/view when legal. Different rank defines a reproducible finite proposal.
                pos1=(rank + m + step) % len(alts);v1=alts[pos1]
                if start_edges is not None:
                    # Step two makes one further replacement; total budget stays <=2 edges/view.
                    g[i,-1]=v1
                    continue
                if len(alts)>1:
                    v2=alts[(pos1+1+(rank%max(1,len(alts)-1)))%len(alts)]
                    if v2==v1:v2=alts[(pos1+1)%len(alts)]
                else:v2=v1
                g[i,-1]=v1
                if v2 != v1:g[i,-2]=v2
            graph.append(torch.as_tensor(g,device=device,dtype=torch.long))
        # Assert global constraints on edited rows.
        for m in range(4):
            rows=graph[m][targets_t]
            assert not bool((rows==targets_t[:,None]).any())
            assert all(len(set(r.tolist()))==len(r) for r in rows)
            baseline=clean_chosen[m][targets_t]
            diff=(rows!=baseline).sum(1)
            assert bool((diff<=2).all())
        candidates.append(graph)
    return candidates,targets

@torch.no_grad()
def maximize_cra(a, parts, base, adjs, graph_loss, group, clean_out, epoch, utils, wrong=None, random_choice=False, start_edges=None):
    candidate_ids=clean_out["candidate_ids"]
    clean_chosen=[]
    for m in range(4):
        order=clean_out["q"][m].topk(16,1).indices;clean_chosen.append(candidate_ids[m].gather(1,order))
    proposals,targets=candidate_attacks(a,clean_chosen,candidate_ids,a.srd_fit_ids,480400+a.fold*10000+epoch,0,start_edges)
    clean_loss=float(utils.ClsLoss(clean_out["logp"],a.targ,a.trn_idx,a.weight))
    candidate_losses=[]
    for attack in proposals:
        o=output(a,parts,base,adjs,graph_loss,group,wrong_perm=wrong,attacked=attack,include_sca=False)
        candidate_losses.append(float(utils.ClsLoss(o["logp"],a.targ,a.trn_idx,a.weight)))
    idx=choose_cra_candidate(candidate_losses,random_choice,(epoch*17+a.fold*13)%len(proposals))
    attack=proposals[idx];selected_loss=candidate_losses[idx]
    if selected_loss <= clean_loss + 1e-8:
        attack=clean_chosen if start_edges is None else start_edges;selected_loss=clean_loss;idx=-1
    return attack,{"cra_clean_ce":clean_loss,"cra_candidate_ce":candidate_losses,"cra_selected_index":idx,
                   "cra_selected_ce":selected_loss,"cra_delta_ce":selected_loss-clean_loss,
                   "cra_targets":len(targets),"cra_target_changed_fraction":float(np.mean([
                       bool((attack[m][torch.as_tensor(targets,device=a.dev)] != clean_chosen[m][torch.as_tensor(targets,device=a.dev)]).any().item()) for m in range(4)]))}

def base_forward(a, utils):
    _,parts=a.dataLayer(a.feat)
    base,adjs,details=a.MessagePassing0(parts)
    gl=sum(utils.GraphConstructLoss(parts[m],adjs[m],1.,.5,0.) for m in range(4))/4
    a.base_details=details
    return parts,base,adjs,gl,base.new_zeros(())

def choose_cra_candidate(losses, random_choice=False, random_index=0):
    """Choose the maximum CE candidate in the declared set; E10 changes only this rule."""
    if len(losses)==0:raise ValueError("CRA candidate set cannot be empty")
    if random_choice:return int(random_index)%len(losses)
    return int(np.argmax(np.asarray([float(v) for v in losses],dtype=np.float64)))

def forward_base_cam(a, base):
    return a.ModalFusion(base)[0]

def train_step(a, group, epoch, utils, sites, srd_labels=None, wrong_perm=None, random_attack=False):
    a.MessagePassing0.train();a.ModalFusion.train();a.APCR.train()
    a.optimizer_MP0.zero_grad(set_to_none=True);a.optimizer_MF.zero_grad(set_to_none=True);a.optimizer_APCR.zero_grad(set_to_none=True)
    parts,base,adjs,graph,aloss=base_forward(a,utils)
    if group==1:
        lp=forward_base_cam(a,base);zero=lp.new_zeros(())
        loss=utils.ClsLoss(lp,a.targ,a.trn_idx,a.weight)+graph
        loss.backward();a.optimizer_MP0.step();a.optimizer_MF.step()
        return {"loss":float(loss.detach()),"main":float(utils.ClsLoss(lp,a.targ,a.trn_idx,a.weight).detach()),"graph":float(graph.detach()),"srd":0.,"sca":0.,"robust":0.,"alpha":float(a.APCR.alpha().detach()),"pair_auc":[.5]*4,"pair_stats":{},"cra":{}}
    z=a.APCR.representations(parts)
    raw_scores=graph_scores(a)
    candidate_ids,raw_values=a.APCR.candidates(raw_scores)
    q_all=a.APCR.score_candidate_edges(z,candidate_ids,raw_values)
    # E2 is an equal-parameter patient-self feature residual. E3 uses original GFR relations.
    if group==2:
        residual=a.APCR.residual(parts,z,candidate_ids,raw_values,"self")
        fused=base+a.APCR.alpha()*residual
        pair,pair_q,support=empty_sca(a,base)
    else:
        mode="raw" if group==3 else "srd"
        fit_mask=torch.zeros(len(base),dtype=torch.bool,device=a.dev);fit_mask[a.srd_fit_ids]=True
        pair,pair_q,support=a.APCR.union_pair_scores(z,candidate_ids,fit_mask)
        train_group=6 if group in (7,8,9,10) else group
        residual=a.APCR.residual(parts,z,candidate_ids,raw_values,mode,wrong_perm=wrong_perm)
        fused=base+a.APCR.alpha()*residual
    clean_lp=a.ModalFusion(fused)[0]
    main=utils.ClsLoss(clean_lp,a.targ,a.trn_idx,a.weight)
    # Everything after the main clean CAM forward is APCR-only work.  Keep its
    # RNG stream separate from the historical base dropout stream.
    _apcr_scope = apcr_rng_scope(a); _apcr_scope.__enter__()
    srd=clean_lp.new_zeros(());srd_auc=[.5]*4;pair_stats={}
    shuffled=group==9
    if group>=4:
        true_labels=np.asarray(a.targ.detach().cpu())
        srd,srd_auc,pair_stats=fit_srd(a,z,a.srd_fit_ids,true_labels,sites,epoch,true_labels,srd_labels if shuffled else None)
    sca=clean_lp.new_zeros(());sca_weight=0.
    sca_enabled=group>=5 and group!=7
    if sca_enabled:
        sca,sw=a.APCR.sca_loss(pair_q,support,a.APCR.reliability);sca_weight=float(sw)
    # Build and maximize a two-step finite candidate set for full CRA groups.
    cra_diag={};robust=clean_lp.new_zeros(());adv_lp=clean_lp
    cra_enabled=group in (6,7,8,9,10)
    if cra_enabled and epoch>=30:
        a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval()
        clean_eval=output(a,parts,base,adjs,graph,group,wrong_perm=wrong_perm,include_sca=False)
        attack1,cra1=maximize_cra(a,parts,base,adjs,graph,group,clean_eval,epoch,utils,wrong_perm,random_choice=(group==10))
        # second maximization starts from the first accepted graph, with a distinct deterministic candidate schedule
        ztmp=z;score=raw_scores;ids2,rv2=candidate_ids,raw_values
        residual2=a.APCR.residual(parts,ztmp,ids2,rv2,"srd",attacked=attack1,wrong_perm=wrong_perm)
        fused2=base+a.APCR.alpha()*residual2;lp2=a.ModalFusion(fused2)[0]
        second_clean={"logp":lp2,"candidate_ids":candidate_ids,"q":q_all}
        attack2,cra2=maximize_cra(a,parts,base,adjs,graph,group,second_clean,epoch+1,utils,wrong_perm,random_choice=(group==10),start_edges=attack1)
        # Evaluate the selected second-step graph with parameters fixed.
        rr=a.APCR.residual(parts,z,candidate_ids,raw_values,"srd",attacked=attack2,wrong_perm=wrong_perm)
        adv_lp=a.ModalFusion(base+a.APCR.alpha()*rr)[0]
        edge_bytes=b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in attack2)
        base_edges=[]
        for m in range(4):
            ord0=q_all[m].topk(16,1).indices;base_edges.append(candidate_ids[m].gather(1,ord0))
        changed=sum(int((attack2[m]!=base_edges[m]).sum().item()) for m in range(4))
        total=sum(int(v.numel()) for v in base_edges)
        cra_diag={"step1":cra1,"step2":cra2,"attack_edge_sha256":hashlib.sha256(edge_bytes).hexdigest(),
                  "attack_changed_edges":changed,"attack_edge_slots":total,"attack_edge_change_rate":changed/max(total,1)}
        a.MessagePassing0.train();a.ModalFusion.train();a.APCR.train()
        robust=utils.ClsLoss(adv_lp,a.targ,a.trn_idx,a.weight)
        idx=torch.as_tensor(a.trn_idx,device=a.dev)
        pclean=clean_eval["logp"][idx].detach().exp().clamp_min(1e-8)
        logadv=F.log_softmax(a.ModalFusion(base+a.APCR.alpha()*rr)[0][idx],dim=1)
        robust=robust+.05*F.kl_div(logadv,pclean,reduction="batchmean")
    _apcr_scope.__exit__(None, None, None)
    # Historical objective is exact for E1. Other groups add only the declared losses.
    cra_weight=0. if not cra_enabled or epoch<30 else .05*min(1.,(epoch-30)/60)
    loss=main+graph+.05*srd+.02*sca+cra_weight*robust
    loss.backward()
    alpha_grad = a.APCR.alpha_raw.grad
    alpha_grad_norm = None if alpha_grad is None else float(alpha_grad.detach().abs())
    if not a.alpha_trainable:
        assert alpha_grad is None
    else:
        assert alpha_grad is not None and torch.isfinite(alpha_grad).all()
    alpha_before_step = float(a.APCR.alpha().detach())
    if getattr(a, "capture_shared_gradients", False):
        a.shared_grads_before_clip = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in a.APCR.named_parameters()
            if name != "alpha_raw" and parameter.grad is not None
        }
    torch.nn.utils.clip_grad_norm_(list(a.APCR.parameters()),5.)
    a.optimizer_MP0.step();a.optimizer_MF.step();a.optimizer_APCR.step()
    return {"loss":float(loss.detach()),"main":float(main.detach()),"graph":float(graph.detach()),"srd":float(srd.detach()),"sca":float(sca.detach()),"sca_weight":sca_weight,"robust":float(robust.detach()),"cra_weight":cra_weight,"pair_auc":srd_auc,"pair_stats":pair_stats,"cra":cra_diag,"alpha_before_step":alpha_before_step,"alpha":float(a.APCR.alpha().detach()),"alpha_grad_abs":alpha_grad_norm,"alpha_trainable":a.alpha_trainable,"message_norms":[float(v.detach().norm()) for v in a.APCR.last.get("deltas",[])],"gate_mean":[float(v.detach().mean()) for v in a.APCR.last.get("gates",[])]}

@torch.no_grad()
def evaluate(a,group,utils,wrong_perm=None):
    a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval()
    parts,base,adjs,graph,aloss=base_forward(a,utils)
    if group==1:
        lp=forward_base_cam(a,base);return {"logp":lp,"base":base,"fused":base,"parts":parts,"adjs":adjs,"residual":torch.zeros_like(base),"deltas":[] if not hasattr(a.APCR,'last') else a.APCR.last.get('deltas',[]),"gates":a.APCR.last.get('gates',[]),"alpha":a.APCR.alpha()}
    z=a.APCR.representations(parts);raw=graph_scores(a);ids,rv=a.APCR.candidates(raw)
    mode="self" if group==2 else ("raw" if group==3 else "srd")
    forced=None
    if wrong_perm is not None:
        q=a.APCR.score_candidate_edges(z,ids,rv)
        forced=[ids[m].gather(1,(rv[m] if group==3 else q[m]).topk(16,1).indices) for m in range(4)]
    residual=a.APCR.residual(parts,z,ids,rv,mode,attacked=forced,wrong_perm=wrong_perm)
    fused=base+a.APCR.alpha()*residual;lp=a.ModalFusion(fused)[0]
    return {"logp":lp,"base":base,"fused":fused,"parts":parts,"adjs":adjs,"residual":residual,"deltas":a.APCR.last['deltas'],"gates":a.APCR.last['gates'],"alpha":a.APCR.alpha(),"chosen":a.APCR.last['chosen']}

def model_bundle(a):
    return {"gfr":{k:v.detach().cpu().clone() for k,v in a.MessagePassing0.state_dict().items()},"pln":{k:v.detach().cpu().clone() for k,v in a.ModalFusion.state_dict().items()},"apcr":{k:v.detach().cpu().clone() for k,v in a.APCR.state_dict().items()}}

def load_bundle(a,bundle):
    a.MessagePassing0.load_state_dict(bundle["gfr"]);a.ModalFusion.load_state_dict(bundle["pln"]);a.APCR.load_state_dict(bundle["apcr"])

def metrics(o,y,te):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    p=o["logp"].detach().cpu().numpy()[te].argmax(1);target=y[te]
    tp,fn,fp,tn=map(int,confusion_matrix(target,p,labels=[0,1]).ravel());one=np.eye(2)[target]
    return {"acc":float((p==target).mean()),"hard_auc":float(roc_auc_score(one,np.eye(2)[p])),"spe":tn/(tn+fp),"sen":tp/(tp+fn),"probability_auc":float(roc_auc_score(target,o["logp"][te,1].detach().cpu().numpy())),"asd_tp":tp,"asd_fn":fn,"td_fp":fp,"td_tn":tn}
