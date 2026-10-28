"""Isolated Experiment 48 implementation.

The original GFR and CAM path is intentionally kept source-compatible with the
immutable Experiment 43.1 B0-B sources.  The DPA modules in this file are new
and are never imported by the project root.
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIMS = [48, 6, 10, 256]
MODALITIES = ["PHENO", "ANAT", "FUNC", "CORR"]
SLICES = [(0, 48), (48, 54), (54, 64), (64, 320)]
PAIR_NAMES = {"CP": (3, 0), "CA": (3, 1), "CF": (3, 2)}


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def grad_reverse(x, coeff):
    return GradientReversal.apply(x, float(coeff))


class PLNNoPEN(nn.Module):
    """Experiment 43.1 PLN-without-PEN, with an optional raw-logit readout.

    Parameter/module names and initialization are kept identical to the
    verified 43.1 implementation so historical B0-B checkpoints can be
    independently read back.
    """

    def __init__(self, torch_mod, layers, dims, hp):
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
        self.InputLayer = layers.VariLengthInputLayer(
            dims, self.d_k, self.d_v, self.n_head, self.dropout
        )
        self.Encoder = []
        self.FeedForward = []
        for index in range(self.n_layer):
            encoder = layers.EncodeLayer(
                self.d_k * self.n_head,
                self.d_k,
                self.d_v,
                self.n_head,
                self.dropout,
            )
            self.add_module("encode_%d" % index, encoder)
            self.Encoder.append(encoder)
            feedforward = layers.FeedForwardLayer(
                self.d_v * self.n_head,
                self.d_v * self.n_head,
                dropout=self.dropout,
            )
            self.add_module("feed_%d" % index, feedforward)
            self.FeedForward.append(feedforward)
        self.Outputlayer = layers.OutputLayer(
            self.d_v * self.n_head * self.modal_num,
            self.d_v * self.n_head,
            self.n_class,
            self.modal_num,
            self.dropout,
        )

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, input_attention = self.InputLayer(x)
        attn = input_attention.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for index in range(self.n_layer):
            x, layer_attention = self.Encoder[index](
                q=x, k=x, v=x, modal_num=self.modal_num
            )
            attn = layer_attention.mean(dim=1)
            x = self.FeedForward[index](x)
            attn_map.append(attn.detach().cpu().numpy())
        hidden = self.Outputlayer.mlp_head(x.view(bs, -1))
        raw = self.Outputlayer.classifier(hidden)
        return F.log_softmax(raw, dim=1), raw, hidden, attn_map


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, z):
        return self.net(z)


class DPAEvidence(nn.Module):
    """Low-capacity, sender-restricted patient relation evidence branch."""

    def __init__(self, fold: int, group_id: int, k: int = 16):
        super().__init__()
        self.fold = int(fold)
        self.group_id = int(group_id)
        self.k = int(k)
        self.modal_embed = nn.ModuleList([nn.Linear(d, 32) for d in INPUT_DIMS])
        self.h_project = nn.Linear(320, 64)
        self.self_project = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        self.msg_project = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        self.align_project = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
        self.gate = nn.Parameter(torch.zeros(4))
        self.graph_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2)
        )
        # A zero initial coefficient is neutral for the B0-B decision while
        # retaining a non-zero derivative for the graph evidence branch.
        self.gamma_raw = nn.Parameter(torch.zeros(1))
        self.domains = nn.ModuleDict(
            {name: DomainDiscriminator() for name in ("CP", "CA", "CF")}
        )

    @property
    def graph_modules(self):
        return [
            self.modal_embed,
            self.h_project,
            self.self_project,
            self.msg_project,
            self.align_project,
            self.graph_head,
        ]

    def graph_parameters(self):
        for module in self.graph_modules:
            for p in module.parameters():
                yield p
        yield self.gate
        yield self.gamma_raw

    def _random_scores(self, n, n_sender, device, epoch, relation):
        # CPU generator gives a reproducible stream independent of CUDA kernel
        # scheduling; the resulting index tensor is moved once to the device.
        g = torch.Generator(device="cpu")
        g.manual_seed(480000000 + self.fold * 100000 + int(epoch) * 100 + relation)
        return torch.rand((n, n_sender), generator=g, dtype=torch.float32).to(device)

    def _relation(self, modal_x, h, train_idx, epoch, relation, random_graph=False):
        n = modal_x.size(0)
        device = modal_x.device
        candidate = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        emb = F.normalize(self.modal_embed[relation](modal_x), dim=-1)
        score = emb @ emb.t()
        candidate_scores = score[:, candidate]
        # Test receivers can read only outer-train senders; train receivers
        # cannot choose themselves.
        train_pos = torch.full((n,), -1, device=device, dtype=torch.long)
        train_pos[candidate] = torch.arange(candidate.numel(), device=device)
        rows = torch.arange(n, device=device)
        self_pos = train_pos[rows]
        valid = self_pos >= 0
        if valid.any():
            candidate_scores[valid, self_pos[valid]] = -torch.inf
        k = min(self.k, candidate.numel() - 1 if candidate.numel() == n else candidate.numel())
        if k < 1:
            raise RuntimeError("outer-train sender set is too small for K=16")
        true_values, true_pos_top = torch.topk(candidate_scores, k=k, dim=1)
        true_sender = candidate[true_pos_top]
        if random_graph:
            rand = self._random_scores(n, candidate.numel(), device, epoch, relation)
            if valid.any():
                rand[valid, self_pos[valid]] = -torch.inf
            rand_pos = torch.topk(rand, k=k, dim=1).indices
            sender = candidate[rand_pos]
            # Keep the true selected weight multiset, but detach it from the
            # relation scores. This matches edge count/weight distribution.
            values = true_values.detach()
        else:
            sender = true_sender.detach()
            values = candidate_scores.gather(1, true_pos_top)
        weights = torch.softmax(values, dim=1)
        sender_h = self.msg_project[relation](h)[sender]
        msg = (weights.unsqueeze(-1) * sender_h).sum(1)
        return msg, {
            "sender": sender.detach(),
            "weights": weights.detach(),
            "true_sender": true_sender.detach(),
            "true_values": true_values.detach(),
            "score": score.detach(),
            "edge_count": int(n * k),
            "weight_mean": float(weights.mean().detach()),
            "weight_sd": float(weights.std(unbiased=False).detach()),
            "self_edge_count": int((sender == rows[:, None]).sum().detach()),
            "same_sender_set": bool(torch.equal(sender, true_sender)),
        }

    def forward(self, parts, base_feat, train_idx, epoch, route="true"):
        # Detaching the inherited GFR output isolates the original B0-B
        # optimization from the new evidence capacity; h_project itself is
        # trainable and therefore receives task/adversarial gradients.
        h = self.h_project(base_feat.detach())
        msgs, meta = [], []
        random_graph = route == "random"
        for i, x in enumerate(parts):
            if route == "self":
                m = self.self_project[i](h)
                info = {"edge_count": 0, "weight_mean": 0.0, "weight_sd": 0.0,
                        "self_edge_count": 0, "same_sender_set": False,
                        "sender": None, "weights": None, "true_sender": None,
                        "true_values": None, "score": None}
            else:
                m, info = self._relation(
                    x, h, train_idx, epoch, i, random_graph=random_graph
                )
            msgs.append(m)
            meta.append(info)
        gates = torch.softmax(self.gate, dim=0)
        fused = sum(gates[i] * msgs[i] for i in range(4))
        graph_logits = self.graph_head(fused)
        z = [self.align_project[i](msgs[i]) for i in range(4)]
        return graph_logits, msgs, z, meta, gates

    def domain_loss(self, z, train_idx, enabled_pairs, lambda_adv):
        if not enabled_pairs or lambda_adv <= 0:
            return None, None, 0.0, 0.0
        idx = torch.as_tensor(train_idx, device=z[0].device, dtype=torch.long)
        losses = []
        accs = []
        for name in enabled_pairs:
            a, b = PAIR_NAMES[name]
            za, zb = z[a][idx], z[b][idx]
            inp = torch.cat([za, zb], dim=0)
            labels = torch.cat(
                [torch.zeros(len(idx), dtype=torch.long, device=inp.device),
                 torch.ones(len(idx), dtype=torch.long, device=inp.device)], dim=0
            )
            logits = self.domains[name](grad_reverse(inp, lambda_adv))
            losses.append(F.cross_entropy(logits, labels))
            accs.append(float(logits.detach().argmax(1).eq(labels).float().mean()))
        loss = torch.stack(losses).mean()
        return loss, losses, float(loss.detach()), float(sum(accs) / len(accs))

    def gamma(self):
        return 0.25 * torch.tanh(self.gamma_raw)[0]


def make_hp():
    return SimpleNamespace(
        n_hidden=18, n_head=2, dropout=0.35, nlayer=1, nmodal=4,
        nclass=2, lr=0.0038, reg=0.11, th=0.9, GC_mode="weighted-cosine",
        MP_mode="GCN", MF_mode="concat", alpha=0.5,
    )


def build_base(torch_mod, layers, network, utils, device, x, y, train_idx, test_idx):
    hp = make_hp()
    data_layer = layers.InitialDataLayer(INPUT_DIMS)
    gfr = network.GFR(INPUT_DIMS, INPUT_DIMS, th=hp.th, mode=hp.GC_mode).to(device)
    pln = PLNNoPEN(torch_mod, layers, INPUT_DIMS, hp).to(device)
    pln.apply(utils.my_weight_init)
    targ = torch_mod.from_numpy(y).long().to(device)
    feat = torch_mod.from_numpy(x).float().to(device)
    counts = torch.bincount(targ[torch.as_tensor(train_idx, device=device)], minlength=2).float()
    weight = (len(train_idx) / counts / 2.0).to(device)
    return SimpleNamespace(
        dev=device, hyperpm=hp, feat=feat, targ=targ, trn_idx=train_idx,
        tst_idx=test_idx, val_idx=test_idx, weight=weight, dataLayer=data_layer,
        MessagePassing0=gfr, ModalFusion=pln,
    )


def base_forward(a, utils):
    nmodal, parts = a.dataLayer(a.feat)
    combined, adjs, aloss = a.MessagePassing0(parts, nmodal, a.targ, a.trn_idx)
    graph = sum(
        utils.GraphConstructLoss(parts[i], adjs[i], 1.0, 0.5, 0.0)
        for i in range(4)
    ) / 4.0
    logp, raw, hidden, attn = a.ModalFusion(combined)
    return parts, combined, adjs, aloss, graph, logp, raw, hidden, attn


def copy_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def load_base_state(a, state):
    a.MessagePassing0.load_state_dict(state["gfr"])
    a.ModalFusion.load_state_dict(state["pln"])


def base_state(a):
    return {"gfr": copy_state(a.MessagePassing0), "pln": copy_state(a.ModalFusion)}


def make_new_agent(torch_mod, layers, network, utils, device, x, y, train_idx, test_idx,
                   fold, group_id, anchor, init_seed):
    # Construct the base with the same parameter layout as B0-B, then restore
    # the immutable anchor and its two Adam states. New modules use a separate
    # deterministic RNG stream and restore the anchor RNG afterwards.
    a = build_base(torch_mod, layers, network, utils, device, x, y, train_idx, test_idx)
    load_base_state(a, anchor["model"])
    a.optimizer_MP0 = torch_mod.optim.Adam(a.MessagePassing0.parameters(), lr=a.hyperpm.lr, weight_decay=a.hyperpm.reg)
    a.optimizer_MF = torch_mod.optim.Adam(a.ModalFusion.parameters(), lr=a.hyperpm.lr, weight_decay=a.hyperpm.reg)
    a.optimizer_MP0.load_state_dict(copy.deepcopy(anchor["mp"]))
    a.optimizer_MF.load_state_dict(copy.deepcopy(anchor["mf"]))
    # Use a forked torch RNG for new-module initialization. The caller's
    # historical B0-B RNG stream is restored byte-for-byte before training.
    cpu_rng = torch_mod.get_rng_state()
    cuda_rng = torch_mod.cuda.get_rng_state_all()
    torch_mod.manual_seed(int(init_seed))
    torch_mod.cuda.manual_seed_all(int(init_seed))
    a.evidence = DPAEvidence(fold, group_id, k=16).to(device)
    torch_mod.set_rng_state(cpu_rng)
    torch_mod.cuda.set_rng_state_all(cuda_rng)
    a.optimizer_new = torch_mod.optim.Adam(a.evidence.graph_parameters(), lr=3.8e-4, weight_decay=0.0)
    a.optimizer_domain = torch_mod.optim.Adam(a.evidence.domains.parameters(), lr=3.8e-4, weight_decay=0.0)
    return a


def lambda_schedule(epoch, lambda_max=0.03):
    if epoch < 45:
        return 0.0
    if epoch < 90:
        return lambda_max * (epoch - 45) / 45.0
    return lambda_max


def set_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.detach().clone()


def add_grads(a, params, grads):
    for p, g in zip(params, grads):
        if g is not None:
            p.grad = g.detach().clone()


def zeros_like_grads(params):
    return [torch.zeros_like(p) for p in params]


def pcgrad_pair(task_grads, adv_grads, params, eps=1e-12):
    device = next((g.device for g in adv_grads if g is not None),
                  next((g.device for g in task_grads if g is not None), torch.device("cpu")))
    tg = [g if g is not None else torch.zeros_like(p) for g, p in zip(task_grads, params)]
    ag = [g if g is not None else torch.zeros_like(p) for g, p in zip(adv_grads, params)]
    dot = sum((t.reshape(-1) @ v.reshape(-1) for t, v in zip(tg, ag)), torch.zeros((), device=ag[0].device))
    tn = torch.sqrt(sum((t.square().sum() for t in tg), torch.zeros((), device=ag[0].device)))
    an = torch.sqrt(sum((v.square().sum() for v in ag), torch.zeros((), device=ag[0].device)))
    if float(dot.detach()) < 0.0:
        denom = tn.square() + eps
        ag = [v - dot / denom * t for t, v in zip(tg, ag)]
        projected = True
    else:
        projected = False
    outn = torch.sqrt(sum((v.square().sum() for v in ag), torch.zeros((), device=ag[0].device)))
    cosine = dot / (tn * an + eps)
    return ag, dict(dot=float(dot.detach()), task_norm=float(tn.detach()), adv_norm=float(an.detach()),
                    projected_norm=float(outn.detach()), cosine=float(cosine.detach()),
                    conflict=int(projected), finite=bool(torch.isfinite(outn)))


def train_step(a, group_id, epoch, utils):
    """One update with exact custom gradient routing for the new modules."""
    t = torch
    a.MessagePassing0.train(); a.ModalFusion.train()
    if group_id > 0:
        a.evidence.train()
    for opt in [a.optimizer_MP0, a.optimizer_MF] + ([a.optimizer_new, a.optimizer_domain] if group_id > 0 else []):
        opt.zero_grad(set_to_none=True)
    parts, combined, adjs, aloss, graph_orig, base_logp, base_raw, hidden, _ = base_forward(a, utils)
    if group_id == 0:
        loss = utils.ClsLoss(base_logp, a.targ, a.trn_idx, a.weight) + graph_orig
        loss.backward()
        a.optimizer_MP0.step(); a.optimizer_MF.step()
        return dict(total=float(loss.detach()), cls=float((loss-graph_orig).detach()), graph=float(graph_orig.detach()),
                    aux=0.0, adv=0.0, domain_acc=0.0, lambda_adv=0.0, conflict=0,
                    conflict_cosine=0.0, task_grad_norm=0.0, adv_grad_norm=0.0,
                    projected_grad_norm=0.0, gamma=0.0, graph_norm=0.0)
    route = "self" if group_id == 1 else ("random" if group_id == 6 else "true")
    enabled = []
    if group_id == 3:
        enabled = ["CP", "CA", "CF"]
    elif group_id in (4, 5, 6):
        enabled = ["CF"]
    elif group_id == 7:
        enabled = ["CA"]
    graph_logits, msgs, z, meta, gates = a.evidence(parts, combined, a.trn_idx, epoch, route=route)
    gamma = a.evidence.gamma()
    final_raw = base_raw + gamma * graph_logits
    final_logp = F.log_softmax(final_raw, dim=1)
    cls_final = utils.ClsLoss(final_logp, a.targ, a.trn_idx, a.weight)
    aux = 0.1 * F.cross_entropy(graph_logits[a.trn_idx], a.targ[a.trn_idx], weight=a.weight)
    task_loss = cls_final + graph_orig + aux
    lam = lambda_schedule(epoch) if enabled else 0.0
    domain_grl, _, domain_value, domain_acc = a.evidence.domain_loss(z, a.trn_idx, enabled, lam)
    base_params = list(a.MessagePassing0.parameters()) + list(a.ModalFusion.parameters())
    graph_params = list(a.evidence.graph_parameters())
    domain_params = list(a.evidence.domains.parameters())
    # Base original parameters receive only task loss, preserving their B0-B
    # gradient path. The evidence module is detached from combined GFR input.
    base_grads = t.autograd.grad(task_loss, base_params, retain_graph=True, allow_unused=True)
    task_grads = t.autograd.grad(task_loss, graph_params, retain_graph=True, allow_unused=True)
    if domain_grl is not None:
        adv_grads = t.autograd.grad(domain_grl, graph_params, retain_graph=True, allow_unused=True)
        dom_grads = t.autograd.grad(domain_grl, domain_params, retain_graph=True, allow_unused=True)
        if group_id == 5:
            adv_used, pc = pcgrad_pair(task_grads, adv_grads, graph_params)
        else:
            adv_used = [g if g is not None else torch.zeros_like(p) for p, g in zip(graph_params, adv_grads)]
            # Record conflict even when no projection is requested.
            _, pc = pcgrad_pair(task_grads, adv_grads, graph_params)
        new_grads = [
            (tg if tg is not None else torch.zeros_like(p)) + ag
            for p, tg, ag in zip(graph_params, task_grads, adv_used)
        ]
        set_grads(domain_params, dom_grads)
    else:
        new_grads = [g if g is not None else torch.zeros_like(p) for p, g in zip(graph_params, task_grads)]
        pc = dict(dot=0.0, task_norm=0.0, adv_norm=0.0, projected_norm=0.0,
                  cosine=0.0, conflict=0, finite=True)
        dom_grads = None
    set_grads(base_params, base_grads)
    set_grads(graph_params, new_grads)
    if not all(torch.isfinite(p.grad).all() for p in base_params + graph_params + (domain_params if domain_grl is not None else []) if p.grad is not None):
        raise FloatingPointError("non-finite DPA gradient")
    a.optimizer_MP0.step(); a.optimizer_MF.step(); a.optimizer_new.step()
    if domain_grl is not None:
        a.optimizer_domain.step()
    msg_norm = float(torch.stack([m.norm() for m in msgs]).mean().detach())
    return dict(total=float(task_loss.detach()), cls=float(cls_final.detach()), graph=float(graph_orig.detach()),
                aux=float(aux.detach()), adv=float(domain_value), domain_acc=float(domain_acc),
                lambda_adv=float(lam), conflict=int(pc["conflict"]), conflict_cosine=float(pc["cosine"]),
                task_grad_norm=float(pc["task_norm"]), adv_grad_norm=float(pc["adv_norm"]),
                projected_grad_norm=float(pc["projected_norm"]), gamma=float(gamma.detach()),
                graph_norm=msg_norm, gate_pheno=float(gates[0].detach()), gate_anat=float(gates[1].detach()),
                gate_func=float(gates[2].detach()), gate_corr=float(gates[3].detach()))


@torch.no_grad()
def evaluate(a, group_id, epoch, utils):
    a.MessagePassing0.eval(); a.ModalFusion.eval()
    if group_id > 0:
        a.evidence.eval()
    parts, combined, adjs, aloss, graph_orig, base_logp, base_raw, hidden, attn = base_forward(a, utils)
    if group_id == 0:
        return dict(parts=parts, combined=combined, adjs=adjs, graph_orig=graph_orig,
                    base_logp=base_logp, raw=base_raw, graph_logits=None, logp=base_logp,
                    msgs=None, z=None, meta=None, gates=None, gamma=0.0)
    route = "self" if group_id == 1 else ("random" if group_id == 6 else "true")
    graph_logits, msgs, z, meta, gates = a.evidence(parts, combined, a.trn_idx, epoch, route=route)
    gamma = a.evidence.gamma()
    raw = base_raw + gamma * graph_logits
    return dict(parts=parts, combined=combined, adjs=adjs, graph_orig=graph_orig,
                base_logp=base_logp, raw=raw, graph_logits=graph_logits,
                logp=F.log_softmax(raw, dim=1), msgs=msgs, z=z, meta=meta,
                gates=gates, gamma=float(gamma.detach()))


def metric_row(output, y, test_idx):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    pred = output["logp"].detach().cpu().numpy()[test_idx].argmax(1)
    target = y[test_idx]
    tp, fn, fp, tn = map(int, confusion_matrix(target, pred, labels=[0, 1]).ravel())
    hard_auc = float(roc_auc_score(torch.eye(2).numpy()[target], torch.eye(2).numpy()[pred]))
    prob = output["logp"].detach().cpu().numpy()[test_idx]
    prob_auc = float(roc_auc_score(target, prob[:, 1]))
    return dict(acc=float((pred == target).mean()), hard_auc=hard_auc,
                spe=float(tn / (tn + fp)), sen=float(tp / (tp + fn)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn,
                probability_auc=prob_auc)

