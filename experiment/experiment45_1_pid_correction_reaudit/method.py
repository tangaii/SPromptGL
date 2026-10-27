"""Corrected training-side PID-DGAA primitives for single-step audits."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def build_augment(fold: int, device="cpu"):
    class Augment(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(320, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64))
            self.aux = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))
            self.domain = nn.ModuleList([
                nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(3)
            ])

    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device).index or 0]):
        torch.manual_seed(45045 + int(fold))
        return Augment().to(device)


def state_sha(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            h.update(str((array.dtype, array.shape)).encode())
            h.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                h.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def pair_batch(z_i, z_j, y_train, conditional: bool):
    if z_i.ndim != 2 or z_j.ndim != 2 or z_i.shape != z_j.shape:
        raise ValueError("relation representations must have matching [N,64] shape")
    if y_train.ndim != 1 or len(y_train) != len(z_i):
        raise ValueError("training labels must align with both relation views")
    relation = torch.cat((z_i, z_j), dim=0)
    source = torch.cat((torch.zeros(len(z_i), dtype=torch.long, device=z_i.device),
                        torch.ones(len(z_j), dtype=torch.long, device=z_j.device)))
    condition = torch.cat((y_train, y_train), dim=0)
    if conditional:
        condition_features = F.one_hot(condition.long(), num_classes=2).to(relation.dtype)
    else:
        condition_features = relation.new_zeros((len(relation), 2))
    domain_input = torch.cat((relation, condition_features), dim=1)
    assert domain_input.shape == (2 * len(y_train), 66)
    assert torch.equal(condition[:len(y_train)], condition[len(y_train):])
    return domain_input, source, condition


def lambda_schedule(values, epoch: int, warmup=(0, 45, 90)):
    _, midpoint, end = warmup
    if epoch < midpoint:
        return [0.0, 0.0, 0.0]
    ramp = min(1.0, (epoch - midpoint) / max(1, end - midpoint))
    return [float(value) * ramp for value in values]


def group_lambdas(group: int, pid_values):
    values = [float(pid_values[p]) for p in ("AF", "AC", "FC")]
    fixed = sum(values) / 3.0
    if group == 2:
        return values, [fixed] * 3, False
    if group == 3:
        return values, [fixed] * 3, True
    if group in (4, 6):
        return values, values, True
    if group == 5:
        return values, [values[1], values[2], values[0]], True
    return values, [0.0] * 3, False


def domain_loss(z_by_view, train_idx, y_train, domain_heads, lambdas, conditional, epoch=90):
    idx = torch.as_tensor(train_idx, dtype=torch.long, device=y_train.device)
    if len(idx) != len(y_train):
        raise ValueError("train index and training label lengths differ")
    scales = lambda_schedule(lambdas, epoch)
    total = y_train.new_zeros((), dtype=torch.float32)
    details = []
    for k, (i, j) in enumerate(((1, 2), (1, 3), (2, 3))):
        zi = z_by_view[i].index_select(0, idx)
        zj = z_by_view[j].index_select(0, idx)
        din, source, condition = pair_batch(zi, zj, y_train, conditional)
        reversed_input = GradientReverse.apply(din[:, :64], scales[k])
        logits = domain_heads[k](torch.cat((reversed_input, din[:, 64:]), dim=1))
        loss = F.cross_entropy(logits, source)
        total = total + loss / 3.0
        details.append({"input": din, "source": source, "condition": condition,
                        "logits": logits, "loss": loss, "scale": scales[k]})
    return total, details


def relation_mapping_checks(pid_values):
    e4 = [float(pid_values[p]) for p in ("AF", "AC", "FC")]
    e5 = [e4[1], e4[2], e4[0]]
    e2 = [sum(e4) / 3.0] * 3
    e3 = list(e2)
    return {
        "E2_E3_equal": e2 == e3,
        "E5_same_multiset": sorted(e5) == sorted(e4),
        "E5_same_sum": abs(sum(e5) - sum(e4)) <= 1e-15,
        "E5_mapping": dict(zip(("AF", "AC", "FC"), e5)),
        "E4_mapping": dict(zip(("AF", "AC", "FC"), e4)),
        "lambda_bar": sum(e4) / 3.0,
    }


def make_pln(torch_module, layers, hp, dims):
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
                encoder = layers.EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
                feed = layers.FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
                self.add_module(f"encode_{i}", encoder)
                self.add_module(f"feed_{i}", feed)
                self.Encoder.append(encoder)
                self.FeedForward.append(feed)
            self.Outputlayer = layers.OutputLayer(self.d_v * self.n_head * self.modal_num,
                                                  self.d_v * self.n_head, self.n_class,
                                                  self.modal_num, self.dropout)

        def forward(self, x):
            batch = x.size(0)
            x, input_attention = self.InputLayer(x)
            attention = input_attention.mean(dim=1)
            for i in range(self.n_layer):
                x, layer_attention = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
                attention = layer_attention.mean(dim=1)
                x = self.FeedForward[i](x)
            return self.Outputlayer(x.view(batch, -1), attention.view(batch, -1))

    return PLNNoPEN()


def make_agent(torch_module, np_module, layers, network, utils, hp, x, y, train, test,
               anchor, fold, group, cfg, device="cuda:0"):
    from collections import Counter

    agent = SimpleNamespace()
    agent.dev = torch_module.device(device)
    agent.hp = hp
    agent.group = int(group)
    agent.fold = int(fold)
    agent.feat = torch_module.as_tensor(x, dtype=torch_module.float32, device=device)
    agent.targ = torch_module.as_tensor(y, dtype=torch_module.long, device=device)
    agent.trn_idx = np_module.asarray(train, dtype=np_module.int64)
    agent.tst_idx = np_module.asarray(test, dtype=np_module.int64)
    agent.val_idx = agent.tst_idx.copy()
    counts = Counter(y[agent.trn_idx])
    class_weights = len(agent.trn_idx) / np_module.asarray([counts[0], counts[1]]) / 2.0
    agent.weight = torch_module.as_tensor(class_weights, dtype=torch_module.float32, device=device)
    agent.dataLayer = layers.InitialDataLayer(list(cfg["dims"]))
    agent.gfr = network.GFR(list(cfg["dims"]), list(cfg["dims"]), th=hp.th, mode=hp.GC_mode).to(device)
    agent.pln = make_pln(torch_module, layers, hp, list(cfg["dims"])).to(device)
    agent.pln.apply(utils.my_weight_init)
    agent.mp = torch_module.optim.Adam(agent.gfr.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.mf = torch_module.optim.Adam(agent.pln.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.augment = build_augment(fold, device=device)
    agent.newopt = torch_module.optim.Adam(agent.augment.parameters(),
                                           lr=cfg["protocol"]["new_lr"],
                                           weight_decay=cfg["protocol"]["new_weight_decay"])
    agent.gfr.load_state_dict(anchor["model"]["gfr"])
    agent.pln.load_state_dict(anchor["model"]["pln"])
    agent.mp.load_state_dict(anchor["mp"])
    agent.mf.load_state_dict(anchor["mf"])
    agent.capture = {}
    agent.hooks = []
    for idx in (1, 2, 3):
        module = getattr(agent.gfr, f"MessagePassing{idx}")
        agent.hooks.append(module.register_forward_hook(
            lambda _module, _inputs, out, idx=idx: agent.capture.__setitem__(idx, out)))
    return agent


def close_agent(agent):
    for hook in getattr(agent, "hooks", []):
        hook.remove()


def forward_train(agent, utils, cfg, pid_values, epoch=90):
    idx = torch.as_tensor(agent.trn_idx, dtype=torch.long, device=agent.dev)
    y_train = agent.targ.index_select(0, idx)
    labels_for_graph = torch.zeros_like(agent.targ)
    labels_for_graph.index_copy_(0, idx, y_train)
    agent.capture.clear()
    n, parts = agent.dataLayer(agent.feat)
    feat, adjs, aloss = agent.gfr(parts, n, labels_for_graph, agent.trn_idx)
    graph_loss = sum(utils.GraphConstructLoss(parts[i], adjs[i], agent.hp.theta_smooth,
                                             agent.hp.theta_degree, agent.hp.theta_sparsity)
                     for i in range(n)) / 4.0
    main_input = agent.feat if agent.group == 6 else feat
    out = agent.pln(main_input)[0]
    cls = utils.ClsLoss(out, agent.targ, agent.trn_idx, agent.weight)
    z = {m: agent.augment.proj(agent.capture[m]) for m in (1, 2, 3)}
    if agent.group == 0:
        aux = torch.zeros((), dtype=cls.dtype, device=agent.dev)
    else:
        aux = sum(F.cross_entropy(agent.augment.aux(z[m].index_select(0, idx)), y_train)
                  for m in (1, 2, 3)) / 3.0
    orig, lambdas, conditional = group_lambdas(agent.group, pid_values)
    domain = torch.zeros((), dtype=cls.dtype, device=agent.dev)
    domain_details = []
    if agent.group in (2, 3, 4, 5, 6):
        domain, domain_details = domain_loss(z, agent.trn_idx, y_train, agent.augment.domain,
                                             lambdas, conditional, epoch=epoch)
    aux_weight = cfg["protocol"]["lambda_aux"] if agent.group > 0 else 0.0
    total = cls + graph_loss + aux_weight * aux + domain
    return total, {"cls": cls, "graph": graph_loss, "aux": aux, "domain": domain,
                   "out": out, "feat": feat, "main_input": main_input, "adjs": adjs, "z": z,
                   "orig_lambdas": orig, "lambda_by_pair": lambdas,
                   "conditional": conditional, "domain_details": domain_details,
                   "aloss_observed_not_optimized": aloss}


def original_b0b_step(agent, utils):
    """The frozen simple-2 objective: weighted classification plus four graph losses."""
    idx = torch.as_tensor(agent.trn_idx, dtype=torch.long, device=agent.dev)
    y_train = agent.targ.index_select(0, idx)
    labels_for_graph = torch.zeros_like(agent.targ)
    labels_for_graph.index_copy_(0, idx, y_train)
    n, parts = agent.dataLayer(agent.feat)
    feat, adjs, _aloss = agent.gfr(parts, n, labels_for_graph, agent.trn_idx)
    graph_loss = sum(utils.GraphConstructLoss(parts[i], adjs[i], agent.hp.theta_smooth,
                                             agent.hp.theta_degree, agent.hp.theta_sparsity)
                     for i in range(n)) / 4.0
    out = agent.pln(feat)[0]
    cls = utils.ClsLoss(out, agent.targ, agent.trn_idx, agent.weight)
    total = cls + graph_loss
    total.backward()
    agent.mp.step()
    agent.mf.step()
    return total.detach(), cls.detach(), graph_loss.detach()


def b0b_forward_only(agent, utils):
    """Frozen B0-B forward and objective without backward or optimizer mutation."""
    idx = torch.as_tensor(agent.trn_idx, dtype=torch.long, device=agent.dev)
    y_train = agent.targ.index_select(0, idx)
    labels_for_graph = torch.zeros_like(agent.targ)
    labels_for_graph.index_copy_(0, idx, y_train)
    n, parts = agent.dataLayer(agent.feat)
    feat, adjs, _aloss = agent.gfr(parts, n, labels_for_graph, agent.trn_idx)
    graph_loss = sum(utils.GraphConstructLoss(parts[i], adjs[i], agent.hp.theta_smooth,
                                             agent.hp.theta_degree, agent.hp.theta_sparsity)
                     for i in range(n)) / 4.0
    out = agent.pln(feat)[0]
    cls = utils.ClsLoss(out, agent.targ, agent.trn_idx, agent.weight)
    return cls + graph_loss, cls, graph_loss, out, feat
