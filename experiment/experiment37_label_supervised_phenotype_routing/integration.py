"""Experiment-local GFR/PLN integration; original repository files stay untouched."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import network
from layers import InitialDataLayer
from network import GFR, PLN, adj_normalization
from utils import GraphConstructLoss, my_weight_init, normalize_adj, ClsLoss
from router import PCPRRouter


def direct_cam_forward(self, x):
    """B0-A PEN bypass: retain PLN construction, run CAM, flatten into OutputLayer."""
    bs = x.size(0)
    attn_map = []
    x, input_attention = self.InputLayer(x)
    attn = input_attention.mean(dim=1)
    attn_map.append(attn.detach().cpu().numpy())
    for i in range(self.n_layer):
        x, encoder_attention = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
        attn = encoder_attention.mean(dim=1)
        x = self.FeedForward[i](x)
        attn_map.append(attn.detach().cpu().numpy())
    x = x.reshape(bs, -1)
    attn_embedding = attn.reshape(bs, -1)
    output, hidden = self.Outputlayer(x, attn_embedding)
    return output, hidden, attn_map


def install_b0a_pln_forward():
    PLN.forward = direct_cam_forward


def row_stochastic(adj):
    n = adj.shape[0]
    value = adj + torch.eye(n, dtype=adj.dtype, device=adj.device)
    return value / value.sum(dim=1, keepdim=True).clamp_min(torch.finfo(value.dtype).eps)


class FairGFR(nn.Module):
    """Shared-GCN fair backbone for four-graph, one-graph and CORR-only arms."""
    def __init__(self, input_dims, th, mode):
        super().__init__()
        initialized = GFR(input_dim=input_dims, output_dim=input_dims, th=th, mode=mode)
        self.th = th
        self.mode = mode
        self.fusion_weight = initialized.fusion_weight
        self.combined_weight = initialized.combined_weight
        self.GraphConstruct0 = initialized.GraphConstruct0
        self.MessagePassing0 = initialized.MessagePassing0
        self.GraphConstruct1 = initialized.GraphConstruct1
        self.GraphConstruct2 = initialized.GraphConstruct2
        self.GraphConstruct3 = initialized.GraphConstruct3
        self.proj_drop = initialized.proj_drop
        self.label_drop = initialized.label_drop
        del initialized
        self.conditioner = None
        self.last_diagnostics = []
        self.last_raw_adjs = []
        self.last_message_norms = []
        self.last_source_weights = None

    def forward(self, x, model_num, label, index, structure="four", directed=False,
                condition=None, condition_enabled=False):
        input_x = torch.cat(x, dim=1)
        constructs = (self.GraphConstruct0, self.GraphConstruct1,
                      self.GraphConstruct2, self.GraphConstruct3)
        raw_adj = [construct(x[m]) for m, construct in enumerate(constructs)]
        fusion = torch.softmax(self.fusion_weight.reshape(-1), dim=0)
        self.last_diagnostics = []
        self.last_raw_adjs = raw_adj
        self.last_message_norms = []
        self.last_source_weights = fusion.detach().cpu().numpy().tolist()

        def propagation(adj, z=None):
            if condition_enabled:
                if self.conditioner is None or z is None:
                    raise RuntimeError("PCPR condition requested but router/input is absent")
                matrix, diagnostic = self.conditioner(adj, z, directed=directed)
                self.last_diagnostics.append(diagnostic)
                return matrix
            return row_stochastic(adj)

        if structure == "four":
            messages = []
            for m, adj in enumerate(raw_adj):
                z = condition[:, m] if condition_enabled else None
                matrix = propagation(adj, z)
                message = self.MessagePassing0(input_x, matrix)
                messages.append(message)
                self.last_message_norms.append(float(message.detach().norm().item()))
            combined = input_x + sum(fusion[m] * messages[m] for m in range(4))
        elif structure == "one":
            fused = sum(fusion[m] * raw_adj[m] for m in range(4))
            matrix = propagation(fused, condition if condition_enabled else None)
            message = self.MessagePassing0(input_x, matrix)
            self.last_message_norms.append(float(message.detach().norm().item()))
            combined = input_x + message
        elif structure == "corr":
            matrix = propagation(raw_adj[3])
            message = self.MessagePassing0(input_x, matrix)
            self.last_message_norms.append(float(message.detach().norm().item()))
            combined = input_x + message
        else:
            raise ValueError(f"unknown propagation structure: {structure}")

        # ALoss is diagnostic only and remains outside the optimized loss.
        aloss = torch.zeros((), dtype=input_x.dtype, device=input_x.device)
        return combined, raw_adj, aloss


class EvalHelper36:
    """The historical simple-2 optimizer/loss protocol with an isolated GFR option."""
    def __init__(self, input_dims, features, labels, hyperpm, train_idx, test_idx,
                 config, structure, directed, condition_enabled, condition_input=None,
                 fold_id=1, relation_mode="none", relation_weight=0.0,
                 train_aux_labels=None, train_sites=None):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_data_dims = list(input_dims)
        self.dev = dev
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.config = config
        self.structure = structure
        self.directed = bool(directed)
        self.condition_enabled = bool(condition_enabled)
        self.relation_mode = str(relation_mode)
        self.relation_weight = float(relation_weight)
        self.feat = torch.from_numpy(features).float().to(dev)
        self.targ = torch.from_numpy(labels).long().to(dev)
        self.trn_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(test_idx, dtype=np.int64)
        self.tst_idx = np.asarray(test_idx, dtype=np.int64)
        assert not np.intersect1d(self.trn_idx, self.tst_idx).size
        if train_aux_labels is None:
            self.aux_labels = None
            self.aux_train_idx = None
        else:
            aux = np.asarray(train_aux_labels, dtype=np.int64)
            if aux.shape != (len(self.trn_idx),) or not set(np.unique(aux)).issubset({0, 1}):
                raise ValueError("auxiliary labels must contain only one value per outer-train participant")
            self.aux_labels = torch.as_tensor(aux, dtype=torch.long, device=dev)
            self.aux_train_idx = torch.as_tensor(self.trn_idx, dtype=torch.long, device=dev)
        self.aux_sites = None if train_sites is None else np.asarray(train_sites).astype(str)
        if self.aux_sites is not None and self.aux_sites.shape != (len(self.trn_idx),):
            raise ValueError("auxiliary site vector must contain outer-train participants only")
        self.relation_audit_done = False
        self.dataLayer = InitialDataLayer(self.input_data_dims)

        if config == "B0_ORIGINAL_B0A":
            self.MessagePassing0 = GFR(input_dim=self.input_data_dims,
                                       output_dim=self.input_data_dims,
                                       th=self.th, mode=self.GC_mode).to(dev)
        else:
            self.MessagePassing0 = FairGFR(self.input_data_dims, self.th, self.GC_mode).to(dev)

        counter = Counter(self.targ[self.trn_idx].detach().cpu().numpy())
        weight = len(self.trn_idx) / np.asarray(list(counter.values())) / self.n_class
        self.weight = torch.from_numpy(weight).float().to(dev)
        self.out_dim = self.d_v * self.n_head
        self.ModalFusion = PLN(self.input_data_dims, hyperpm).to(dev)
        self.optimizer_MP0 = optim.Adam(self.MessagePassing0.parameters(),
                                        lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(),
                                       lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)

        self.condition_input = None if condition_input is None else torch.as_tensor(condition_input, dtype=torch.float32, device=dev)
        if self.condition_enabled:
            if config == "B0_ORIGINAL_B0A":
                raise ValueError("B0 must not have PCPR parameters")
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):
                init_seed = 360000 + int(fold_id)
                torch.manual_seed(init_seed)
                if dev.type == "cuda":
                    torch.cuda.manual_seed_all(init_seed)
                router = PCPRRouter(input_dim=768, key_dim=64, beta_max=0.25,
                                    beta_initial=0.10).to(dev)
            self.MessagePassing0.conditioner = router
            self.optimizer_MP0.add_param_group({
                "params": list(router.parameters()), "lr": hyperpm.lr,
                "weight_decay": 0.0
            })
        assert np.array_equal(self.val_idx, self.tst_idx)

    def _gfr_forward(self, parts):
        if self.config == "B0_ORIGINAL_B0A":
            return self.MessagePassing0(parts, len(parts), self.targ, self.trn_idx)
        return self.MessagePassing0(parts, len(parts), self.targ, self.trn_idx,
                                    structure=self.structure, directed=self.directed,
                                    condition=self.condition_input,
                                    condition_enabled=self.condition_enabled)

    def forward(self):
        model_num, parts = self.dataLayer(self.feat)
        features, raw_adjs, aloss = self._gfr_forward(parts)
        output, hidden, attention = self.ModalFusion(features)
        return output, features, raw_adjs, aloss, attention

    @staticmethod
    def _balanced_logits_loss(logits, same):
        same = same.bool()
        positive, negative = logits[same], logits[~same]
        pos_loss = (F.binary_cross_entropy_with_logits(positive, torch.full_like(positive, 0.9))
                    if positive.numel() else logits.sum() * 0.0)
        neg_loss = (F.binary_cross_entropy_with_logits(negative, torch.full_like(negative, 0.1))
                    if negative.numel() else logits.sum() * 0.0)
        return ((pos_loss + neg_loss) * 0.5, int(positive.numel()), int(negative.numel()),
                {"same_available": bool(positive.numel()), "different_available": bool(negative.numel())})

    def _relation_loss(self, raw_adjs):
        if self.relation_mode == "none":
            return torch.zeros((), dtype=self.feat.dtype, device=self.dev), {"pairs": 0}
        if self.aux_labels is None or self.aux_train_idx is None:
            raise RuntimeError("relation loss has no outer-train-only labels")
        labels = self.aux_labels
        local_n = len(labels)
        upper = torch.triu_indices(local_n, local_n, offset=1, device=self.dev)
        global_i = self.aux_train_idx[upper[0]]
        global_j = self.aux_train_idx[upper[1]]
        same = labels[upper[0]].eq(labels[upper[1]])
        test_tensor = torch.as_tensor(self.tst_idx, dtype=torch.long, device=self.dev)
        assert not bool(torch.isin(global_i, test_tensor).any())
        assert not bool(torch.isin(global_j, test_tensor).any())
        assert int(global_i.numel()) == local_n * (local_n - 1) // 2

        if self.relation_mode == "raw_support":
            modal_losses, counts = [], []
            for adj in raw_adjs:
                sub = adj.index_select(0, self.aux_train_idx).index_select(1, self.aux_train_idx)
                support = (sub > 0)[upper[0], upper[1]]
                values = sub[upper[0], upper[1]][support].clamp(1e-6, 1.0 - 1e-6)
                targets_same = same[support]
                positive, negative = values[targets_same], values[~targets_same]
                pos = (F.binary_cross_entropy(positive, torch.full_like(positive, 0.9))
                       if positive.numel() else values.sum() * 0.0)
                neg = (F.binary_cross_entropy(negative, torch.full_like(negative, 0.1))
                       if negative.numel() else values.sum() * 0.0)
                modal_losses.append((pos + neg) * 0.5)
                counts.append({"same": int(positive.numel()), "different": int(negative.numel()),
                               "same_available": bool(positive.numel()),
                               "different_available": bool(negative.numel()),
                               "empty_support": int(values.numel() == 0)})
            return torch.stack(modal_losses).mean(), {"pairs": int(sum(x["same"] + x["different"] for x in counts)),
                                                      "same_pairs": int(sum(x["same"] for x in counts)),
                                                      "different_pairs": int(sum(x["different"] for x in counts)),
                                                      "per_modality_counts": counts,
                                                      "loss_kind": "raw_graph_supported_train_train_bce"}

        if not self.condition_enabled:
            raise RuntimeError("Q/K relation supervision requires an active PCPR router")
        fair = self.MessagePassing0
        modal_losses, per_modal = [], []
        if self.relation_mode == "ranking":
            same_matrix = labels[:, None].eq(labels[None, :])
            same_matrix.fill_diagonal_(False)
            different_matrix = labels[:, None].ne(labels[None, :])
            valid = same_matrix.any(1) & different_matrix.any(1)
            selected = []
            total_pos = total_neg = 0
            for diag in fair.last_diagnostics:
                scores = diag["pre_tanh_centered_scores"].index_select(0, self.aux_train_idx).index_select(1, self.aux_train_idx)
                hard_pos = scores.masked_fill(~same_matrix, torch.inf).topk(4, dim=1, largest=False).values
                hard_neg = scores.masked_fill(~different_matrix, -torch.inf).topk(4, dim=1, largest=True).values
                if not bool(valid.any()):
                    raise RuntimeError("ranking relation loss has no valid train anchors")
                modal_losses.append(F.softplus(0.5 - (hard_pos[valid] - hard_neg[valid])).mean())
                modal_count = int(valid.sum()) * 4
                total_pos += modal_count
                total_neg += modal_count
                selected.append({"positive": modal_count, "negative": modal_count,
                                 "valid_anchors": int(valid.sum())})
            return torch.stack(modal_losses).mean(), {"pairs": total_pos + total_neg,
                                                      "same_pairs": total_pos,
                                                      "different_pairs": total_neg,
                                                      "per_modality_counts": selected,
                                                      "loss_kind": "hard_positive_negative_softplus_ranking"}

        sites = self.aux_sites
        if self.relation_mode == "site_bce" and sites is None:
            raise RuntimeError("site-balanced relation supervision requires train-only site codes")
        site_pair = None
        if self.relation_mode == "site_bce":
            site_pair = np.asarray(sites)[upper[0].detach().cpu().numpy()] == np.asarray(sites)[upper[1].detach().cpu().numpy()]
        for modality, diag in enumerate(fair.last_diagnostics):
            scores = diag["pre_tanh_centered_scores"].index_select(0, self.aux_train_idx).index_select(1, self.aux_train_idx)
            pair_scores = scores[upper[0], upper[1]]
            pair_same = same
            if self.relation_mode == "support_bce":
                sub_support = diag["edge_support"].index_select(0, self.aux_train_idx).index_select(1, self.aux_train_idx)
                use = sub_support[upper[0], upper[1]]
                pair_scores, pair_same = pair_scores[use], pair_same[use]
                loss, pos_n, neg_n, available = self._balanced_logits_loss(pair_scores, pair_same)
                modal_losses.append(loss)
                per_modal.append({"same": pos_n, "different": neg_n, **available,
                                  "empty_support": int(pos_n + neg_n == 0)})
            elif self.relation_mode == "all_bce":
                loss, pos_n, neg_n, available = self._balanced_logits_loss(pair_scores, pair_same)
                modal_losses.append(loss)
                per_modal.append({"same": pos_n, "different": neg_n, **available})
            elif self.relation_mode == "site_bce":
                site_losses, site_counts, valid_sites, valid_participants = [], [], 0, set()
                for site in sorted(set(sites.tolist())):
                    member = sites == site
                    use = torch.as_tensor(site_pair & member[upper[0].detach().cpu().numpy()],
                                          dtype=torch.bool, device=self.dev)
                    if not bool(use.any()):
                        continue
                    same_site = pair_same[use]
                    if not bool(same_site.any()) or bool(same_site.all()):
                        continue
                    part_loss, pos_n, neg_n, _ = self._balanced_logits_loss(pair_scores[use], same_site)
                    site_losses.append(part_loss)
                    site_counts.append((pos_n, neg_n))
                    valid_sites += 1
                    valid_participants.update(np.flatnonzero(sites == site).tolist())
                if not site_losses:
                    raise RuntimeError("no site contains both valid same- and different-label train pairs")
                modal_losses.append(torch.stack(site_losses).mean())
                per_modal.append({"same": int(sum(x[0] for x in site_counts)),
                                  "different": int(sum(x[1] for x in site_counts)),
                                  "valid_sites": valid_sites,
                                  "valid_train_participants": len(valid_participants)})
            else:
                raise ValueError(f"unknown relation mode {self.relation_mode}")
        meta = {"pairs": int(sum(x.get("same", 0) + x.get("different", 0) for x in per_modal)),
                "same_pairs": int(sum(x.get("same", 0) for x in per_modal)),
                "different_pairs": int(sum(x.get("different", 0) for x in per_modal)),
                "per_modality_counts": per_modal, "loss_kind": self.relation_mode}
        if self.relation_mode == "site_bce":
            meta["site_coverage"] = [x.get("valid_train_participants", 0) / local_n for x in per_modal]
        return torch.stack(modal_losses).mean(), meta

    def run_epoch(self, mode="simple-2"):
        if mode != "simple-2":
            raise ValueError("Experiment 36 preserves the simple-2 protocol")
        self.MessagePassing0.train()
        self.ModalFusion.train()
        self.optimizer_MP0.zero_grad()
        self.optimizer_MF.zero_grad()
        model_num, parts = self.dataLayer(self.feat)
        features, raw_adjs, aloss = self._gfr_forward(parts)
        graph_loss = sum(GraphConstructLoss(parts[i], raw_adjs[i],
                                            self.hyperpm.theta_smooth,
                                            self.hyperpm.theta_degree,
                                            self.hyperpm.theta_sparsity)
                         for i in range(model_num)) / 4
        output, _, _ = self.ModalFusion(features)
        cls_loss = ClsLoss(output, self.targ, self.trn_idx, self.weight)
        relation_loss, relation_meta = self._relation_loss(raw_adjs)
        if not self.relation_audit_done and relation_loss.requires_grad:
            router = getattr(self.MessagePassing0, "conditioner", None)
            if router is not None:
                query_grad, key_grad = torch.autograd.grad(
                    relation_loss, (router.query.weight, router.key.weight),
                    retain_graph=True, allow_unused=True)
                relation_meta["aux_query_grad_norm"] = float(query_grad.norm().detach()) if query_grad is not None else 0.0
                relation_meta["aux_key_grad_norm"] = float(key_grad.norm().detach()) if key_grad is not None else 0.0
            elif self.relation_mode == "raw_support":
                graph_params = [getattr(self.MessagePassing0, f"GraphConstruct{i}").p.weight
                                for i in range(4)]
                graph_gradients = torch.autograd.grad(relation_loss, graph_params,
                                                      retain_graph=True, allow_unused=True)
                relation_meta["aux_graph_p_grad_norms"] = [float(g.norm().detach()) if g is not None else 0.0
                                                             for g in graph_gradients]
            self.relation_audit_done = True
        loss = cls_loss + graph_loss + self.relation_weight * relation_loss
        loss.backward()
        relation_meta["query_total_grad_norm"] = float(self.MessagePassing0.conditioner.query.weight.grad.norm().detach()) if getattr(self.MessagePassing0, "conditioner", None) is not None and self.MessagePassing0.conditioner.query.weight.grad is not None else 0.0
        relation_meta["key_total_grad_norm"] = float(self.MessagePassing0.conditioner.key.weight.grad.norm().detach()) if getattr(self.MessagePassing0, "conditioner", None) is not None and self.MessagePassing0.conditioner.key.weight.grad is not None else 0.0
        self.last_relation_meta = relation_meta
        self.last_relation_loss = float(relation_loss.detach())
        self.optimizer_MP0.step()
        self.optimizer_MF.step()
        return {"loss": float(loss.detach()), "cls_loss": float(cls_loss.detach()),
                "graph_loss": float(graph_loss.detach()), "aloss": float(aloss.detach()),
                "relation_loss": float(relation_loss.detach()), "relation_meta": relation_meta}

    @torch.no_grad()
    def predict(self):
        self.MessagePassing0.eval()
        self.ModalFusion.eval()
        output, _, _, _, _ = self.forward()
        return output


def model_state(agent):
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))}


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def parameter_sha256(agent, baseline_common_only=False):
    import hashlib
    digest = hashlib.sha256()
    modules = (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))
    for module_name, module in modules:
        for name, parameter in sorted(module.named_parameters()):
            if baseline_common_only:
                if module_name == "gfr" and not (name.startswith("GraphConstruct") or
                                                   name.startswith("MessagePassing0") or
                                                   name.startswith("fusion_weight") or
                                                   name.startswith("combined_weight")):
                    continue
                if module_name == "pln" and ("conditioner" in name or name.startswith("MLP.")):
                    continue
            digest.update(module_name.encode() + b"/" + name.encode() + b"\0")
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def historical_trunk_sha256(agent):
    """Use Experiment 32's exact trunk hash recipe for direct E0 comparison."""
    import hashlib
    digest = hashlib.sha256()
    modules = (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion))
    for module_name, module in modules:
        for name, parameter in sorted(module.named_parameters()):
            if module_name == "pln" and name.startswith("dcer32."):
                continue
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_audit(agent):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    modules = (agent.MessagePassing0, agent.ModalFusion)
    expected = {id(parameter) for module in modules for parameter in module.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == expected
    assert all(abs(group["lr"] - agent.hyperpm.lr) < 1e-12 for group in groups)
    if agent.condition_enabled:
        condition_ids = {id(parameter) for parameter in agent.MessagePassing0.conditioner.parameters()}
        condition_groups = [group for group in agent.optimizer_MP0.param_groups
                            if any(id(parameter) in condition_ids for parameter in group["params"])]
        assert len(condition_groups) == 1
        assert condition_groups[0].get("weight_decay") == 0.0
        assert {id(parameter) for parameter in condition_groups[0]["params"]} == condition_ids
        assert agent.optimizer_MP0.param_groups[0].get("weight_decay") == agent.hyperpm.reg
    conditioner = getattr(agent.MessagePassing0, "conditioner", None)
    return {"gfr_parameters": sum(p.numel() for p in agent.MessagePassing0.parameters()),
            "pln_parameters": sum(p.numel() for p in agent.ModalFusion.parameters()),
            "condition_parameters": sum(p.numel() for p in conditioner.parameters()) if conditioner is not None else 0}


def metric_row(output, labels, test_idx):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y = np.asarray(labels, dtype=np.int64)[test_idx]
    pred = output.detach().cpu().numpy()[test_idx].argmax(1)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    asd_tp, asd_fn, td_fp, td_tn = map(int, matrix.ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred]))
    return {"acc": float((pred == y).mean()), "hard_auc": hard_auc,
            "sen_asd": float(asd_tp / (asd_tp + asd_fn)),
            "spe_td": float(td_tn / (td_tn + td_fp)),
            "asd_tp": asd_tp, "asd_fn": asd_fn,
            "td_fp": td_fp, "td_tn": td_tn}
