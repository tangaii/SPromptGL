"""Historical SPromptGL method chain adapted to the locked TADPOLE inputs.

The implementation keeps the original graph, CAM and PEN arithmetic in small
testable modules. E74 graph propagation and E82 SCMR use verified archived
numerical definitions with explicit TADPOLE adaptations.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scmr_runtime import fit_reference, make_scmr, source_config, true_class_margin
from source_runtime import E74_GROUP, darg_sources, e51_legacy_graph


EPS = 1e-8
PEN_PROMPT_COUNT = 8
PROMPT_TEXTS = (
    "Nromal brain volume", "Uniform cortical thickness",
    "Symmetrical brain structure", "Normal ventricular morphology",
    "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
)


def original_linear_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)


def normalize_adj(adj: Tensor) -> Tensor:
    """Original normalization of adjacency matrices with positive self-loops."""
    degree = torch.sum(adj, -1)
    inv = torch.diagflat(torch.pow(degree, -0.5))
    return torch.mm(inv, torch.mm(adj, inv))


def graph_construct_loss(feat: Tensor, adj: Tensor,
                         theta_smooth: float = 1.0,
                         theta_degree: float = 0.5,
                         theta_sparsity: float = 0.0) -> Tensor:
    """Exact GraphConstructLoss from the original utils.py."""
    laplacian = torch.diagflat(torch.sum(adj, -1)) - adj
    ones = torch.ones(adj.size(-1), device=adj.device, dtype=adj.dtype)
    smooth = torch.trace(feat.T @ (laplacian @ feat)) / int(adj.numel())
    degree = (ones.unsqueeze(0) @ torch.log(adj @ ones.unsqueeze(-1) + 1e-5)).squeeze()
    degree = degree / adj.shape[-1]
    sparsity = adj.square().sum() / int(adj.numel())
    return theta_smooth * smooth - theta_degree * degree + theta_sparsity * sparsity


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

    def forward(self, value: Tensor, adj: Tensor) -> Tensor:
        return adj @ (value @ self.W)


class HistoricalGCN(nn.Module):
    """The two GraphConv modules registered by each historical GFR branch."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.gcn1 = GraphConv(input_dim, output_dim)
        self.gcn2 = GraphConv(input_dim, output_dim)

    def forward(self, value: Tensor, adj: Tensor) -> Tensor:
        return self.gcn2(F.relu(self.gcn1(value, adj)), adj)


class HistoricalGraphLearn(nn.Module):
    """Original GraphLearn parameter layout and thresholded cosine graph."""
    def __init__(self, input_dim: int, threshold: float = 0.9):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = float(threshold)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.p(x)
        normalized = F.normalize(projected, dim=-1)
        score = normalized @ normalized.T
        mask = (score > self.th).detach().to(score.dtype)
        # Keep the continuous pre-threshold cosine for the registered
        # cross-repository pair_mean supervision.  The historical graph output
        # remains thresholded and unchanged.
        self.last_score = score
        self.last_mask = mask
        return score * mask


class HistoricalGFR(nn.Module):
    """Original GFR loop with the documented second-modality overwrite."""
    def __init__(self, dims: Sequence[int], threshold: float = 0.9,
                 theta_smooth: float = 1.0, theta_degree: float = 0.5,
                 theta_sparsity: float = 0.0):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.modal_num = len(self.dims)
        self.total_dim = sum(self.dims)
        self.th = float(threshold)
        self.theta_smooth = float(theta_smooth)
        self.theta_degree = float(theta_degree)
        self.theta_sparsity = float(theta_sparsity)
        self.fusion_weight = nn.Parameter(torch.ones(self.modal_num, 1))
        self.combined_weight = nn.Parameter(torch.ones(self.modal_num, 1))
        # The source registers GraphConstruct{i} followed immediately by the
        # two-layer MessagePassing{i}.  Keep plain Python views for the local
        # six-modality loops while registering the modules in that exact order.
        self.graphs = []
        self.gcn = []
        for index, dim in enumerate(self.dims):
            graph = HistoricalGraphLearn(dim, threshold)
            branch = HistoricalGCN(self.total_dim, self.total_dim)
            self.add_module(f"GraphConstruct{index}", graph)
            self.add_module(f"MessagePassing{index}", branch)
            self.graphs.append(graph)
            self.gcn.append(branch)

    def message(self, modality: int, raw: Tensor, adjacency: Tensor) -> Tensor:
        return self.gcn[modality](raw, adjacency)

    def target_adjacency(self, labels: Tensor, train_idx: Tensor) -> Tensor:
        """Historical masked-label target, generalized to NC/MCI/AD.

        Class zero is mapped to -1 as in the source.  Unknown outer-test rows
        retain the source sentinel 0, so this preserves the historical target
        construction without reading their labels.
        """
        masked = labels.detach().clone()
        masked[masked == 0] = -1
        known = torch.zeros_like(masked, dtype=torch.bool)
        known[train_idx] = True
        masked = torch.where(known, masked, torch.zeros_like(masked))
        same = masked[:, None].eq(masked[None, :])
        eye = torch.eye(len(masked), device=labels.device, dtype=torch.bool)
        return normalize_adj(same.float() + eye.float())

    def forward(self, parts: Sequence[Tensor], labels: Tensor, train_idx: Tensor) -> Tuple[Tensor, Dict]:
        raw = torch.cat(tuple(parts), dim=1)
        target = self.target_adjacency(labels, train_idx)
        combined: Optional[Tensor] = None
        adjs: List[Tensor] = []
        normalized: List[Tensor] = []
        messages: List[Tensor] = []
        continuous_scores: List[Tensor] = []
        aloss = raw.new_zeros(())
        for i, part in enumerate(parts):
            adj = self.graphs[i](part)
            continuous_scores.append(self.graphs[i].last_score)
            norm = normalize_adj(adj + torch.eye(len(adj), device=adj.device, dtype=adj.dtype))
            aloss = aloss + torch.linalg.norm(norm - target)
            msg = self.message(i, raw, norm)
            weight = torch.sigmoid(self.fusion_weight[i])
            if i == 0:
                combined = weight * msg + raw
            elif i == 1:
                # This overwrite is present in the historical source and is
                # intentionally retained for the first four modalities.
                combined = weight * msg + raw
            else:
                assert combined is not None
                combined = weight * msg + combined
            adjs.append(adj)
            normalized.append(norm)
            messages.append(msg)
        assert combined is not None
        graph_terms = [graph_construct_loss(part, adj, self.theta_smooth,
                                             self.theta_degree, self.theta_sparsity)
                       for part, adj in zip(parts, adjs)]
        return combined, {
            "adjacencies": adjs,
            "normalized_adjacencies": normalized,
            "messages": messages,
            "continuous_scores": continuous_scores,
            "aloss": aloss / self.modal_num,
            "graph_loss": sum(graph_terms) / self.modal_num,
            "target_adjacency": target,
            "raw": raw,
        }


class HistoricalAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        return torch.matmul(attn, v), attn, v


class VariLengthInputLayer(nn.Module):
    def __init__(self, dims: Sequence[int], d_k: int, d_v: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head, self.dims, self.d_k, self.d_v = int(n_head), tuple(dims), int(d_k), int(d_v)
        self.w_qs, self.w_ks, self.w_vs = [], [], []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module(f"linear_q_{dim}_{i}", self.w_q)
            self.add_module(f"linear_k_{dim}_{i}", self.w_k)
            self.add_module(f"linear_v_{dim}_{i}", self.w_v)
        self.attention = HistoricalAttention(d_k ** 0.5, dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data: Tensor, mask: Optional[Tensor] = None):
        pieces_q, pieces_k, pieces_v = [], [], []
        offset = 0
        for i, dim in enumerate(self.dims):
            part = input_data[:, offset:offset + dim]
            offset += dim
            pieces_q.append(self.w_qs[i](part))
            pieces_k.append(self.w_ks[i](part))
            pieces_v.append(self.w_vs[i](part))
        bs, modal_num = input_data.shape[0], len(self.dims)
        q = torch.stack(pieces_q, dim=1).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        k = torch.stack(pieces_k, dim=1).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        v = torch.stack(pieces_v, dim=1).view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2)
        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q)) + residual
        return self.layer_norm(q), attn


class EncodeLayer(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, n_head: int, dropout: float,
                 attention_dropout: float = 0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = int(n_head), int(d_k), int(d_v)
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = HistoricalAttention(d_k ** 0.5, attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, modal_num: int, mask: Optional[Tensor] = None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2)
        q, attn, _ = self.attention(q, k, v, mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q)) + residual
        return self.layer_norm(q), attn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dropout(self.w_2(F.gelu(self.w_1(x)))) + residual
        return self.layer_norm(x)


class OutputLayer(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.mlp_head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden),
                                      nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(d_hidden, n_classes)

    def raw(self, x: Tensor) -> Tensor:
        return self.classifier(self.mlp_head(x))

    def forward(self, x: Tensor, attn_embedding: Optional[Tensor] = None):
        hidden = self.mlp_head(x)
        return F.log_softmax(self.classifier(hidden), dim=1), hidden


class OriginalPEN(nn.Module):
    """Complete original PEN with dynamic N, modality count and local token width."""
    def __init__(self, hidden: int, prompt_path: str):
        super().__init__()
        prompt = torch.load(prompt_path, map_location="cpu", weights_only=False)
        text = prompt["medical"].float()
        if tuple(text.shape) != (8, 1024):
            raise ValueError(f"unexpected frozen prompt shape {tuple(text.shape)}")
        self.register_buffer("prompt_text_features", text)
        self.MLP = nn.Sequential(nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, hidden), nn.ReLU())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x is [N, M, H*D], and the original local token organization is
        # [N, M, H, D].  No patient count is hard-coded.
        n, modalities, width = x.shape
        hidden = width // 4
        part_x = x.view(n, modalities, 4, hidden)
        text = torch.cat((self.prompt_text_features[:4].reshape(1, -1),
                          self.prompt_text_features[4:].reshape(1, -1)), dim=0)
        text = self.MLP(text)
        part_norm = F.normalize(part_x, p=2, dim=-1)
        text_norm = F.normalize(text, p=2, dim=-1)
        sim = torch.matmul(part_norm.unsqueeze(-2), text_norm.T.view(1, 1, 1, hidden, 2))
        sim = sim[..., -1, :].clamp(-5, 5)
        weights = torch.softmax(sim, dim=-1).max(dim=-1).values
        part_x = weights.unsqueeze(-1) * part_x
        global_x = F.avg_pool1d(x.reshape(n * modalities, 1, width), kernel_size=4, stride=4)
        global_x = global_x.view(n, modalities, hidden)
        combined = []
        for head in range(4):
            part = part_x[:, :, head, :]
            vsim = F.softmax(part * global_x, dim=-1)
            combined.append(vsim * part)
        return x + torch.cat(combined, dim=2), weights


class OriginalPEN12(nn.Module):
    """Three-class extension of the original PEN with four prompts per class.

    The local/global fusion is unchanged.  Only the frozen semantic competition
    expands from the historical two class text rows to three class rows.
    """
    def __init__(self, hidden: int, prompt_path: str):
        super().__init__()
        prompt = torch.load(prompt_path, map_location="cpu", weights_only=False)
        text = prompt["medical"].float()
        if tuple(text.shape) != (12, 1024):
            raise ValueError(f"unexpected frozen MMGPL12 prompt shape {tuple(text.shape)}")
        self.register_buffer("prompt_text_features", text)
        self.MLP = nn.Sequential(nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, hidden), nn.ReLU())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        n, modalities, width = x.shape
        hidden = width // 4
        part_x = x.view(n, modalities, 4, hidden)
        text = self.prompt_text_features.view(3, 4, 1024).reshape(3, -1)
        text = self.MLP(text)
        part_norm = F.normalize(part_x, p=2, dim=-1)
        text_norm = F.normalize(text, p=2, dim=-1)
        sim = torch.matmul(part_norm.unsqueeze(-2), text_norm.T.view(1, 1, 1, hidden, 3))
        sim = sim[..., -1, :].clamp(-5, 5)
        weights = torch.softmax(sim, dim=-1).max(dim=-1).values
        part_x = weights.unsqueeze(-1) * part_x
        global_x = F.avg_pool1d(x.reshape(n * modalities, 1, width), kernel_size=4, stride=4)
        global_x = global_x.view(n, modalities, hidden)
        combined = []
        for head in range(4):
            part = part_x[:, :, head, :]
            vsim = F.softmax(part * global_x, dim=-1)
            combined.append(vsim * part)
        return x + torch.cat(combined, dim=2), weights


class OriginalRES(nn.Module):
    """The parameter-free local/global relation-aware embedding from PEN Eq. 6.

    RES-only uses the same token organization and pooling as the author PEN,
    with semantic guidance removed.  It intentionally adds no trainable
    parameter and therefore cannot be mistaken for a replacement attention
    module.
    """
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        n, modalities, width = x.shape
        hidden = width // 4
        local = x.view(n, modalities, 4, hidden)
        global_x = F.avg_pool1d(x.reshape(n * modalities, 1, width), kernel_size=4, stride=4)
        global_x = global_x.view(n, modalities, hidden)
        combined = []
        for head in range(4):
            part = local[:, :, head, :]
            relation = F.softmax(part * global_x, dim=-1)
            combined.append(relation * part)
        return x + torch.cat(combined, dim=2), global_x


class HistoricalCAM(nn.Module):
    def __init__(self, dims: Sequence[int], hidden: int = 10, heads: int = 4,
                 nlayer: int = 2, nclass: int = 3, dropout: float = 0.0,
                 use_pen: bool = False, prompt_path: Optional[str] = None,
                 encoder_attention_dropout: float = 0.1):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.hidden, self.heads, self.modal_num = int(hidden), int(heads), len(self.dims)
        self.width = self.hidden * self.heads
        self.cam_dim = self.width * self.modal_num
        # Historical PLN/PLNNoPEN registers this parameter even though the
        # published forward path does not consume it.  Keeping it preserves
        # the parameter/optimizer identity and initialization contract.
        self.fusion_weight = nn.Parameter(torch.ones(self.modal_num, 1))
        self.InputLayer = VariLengthInputLayer(self.dims, hidden, hidden, heads, dropout)
        self.Encoder = []
        self.FeedForward = []
        for layer in range(nlayer):
            encoder = EncodeLayer(self.width, hidden, hidden, heads, dropout,
                                  encoder_attention_dropout)
            self.add_module(f"encode_{layer}", encoder)
            self.Encoder.append(encoder)
            feed_forward = FeedForwardLayer(self.width, self.width, dropout)
            self.add_module(f"feed_{layer}", feed_forward)
            self.FeedForward.append(feed_forward)
        self.Outputlayer = OutputLayer(self.cam_dim, self.width, nclass, dropout)
        self.use_pen = bool(use_pen)
        self.use_res_only = False
        if self.use_pen:
            if prompt_path is None:
                raise ValueError("G1 requires frozen prompt embeddings")
            prompt = torch.load(prompt_path, map_location="cpu", weights_only=False)
            prompt_rows = int(prompt["medical"].shape[0])
            if prompt_rows == PEN_PROMPT_COUNT:
                self.PEN = OriginalPEN(hidden, prompt_path)
            elif prompt_rows == 12:
                self.PEN = OriginalPEN12(hidden, prompt_path)
            else:
                raise ValueError(f"unsupported PEN prompt count {prompt_rows}")

    def enable_res_only(self) -> None:
        self.use_res_only = True
        self.RES = OriginalRES()

    def raw_logits_from_cam(self, cam: Tensor) -> Tensor:
        return self.Outputlayer.raw(cam)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        tokens, input_attn = self.InputLayer(x)
        attention_maps = [input_attn]
        for encoder, ff in zip(self.Encoder, self.FeedForward):
            tokens, attn = encoder(tokens, tokens, tokens, self.modal_num)
            tokens = ff(tokens)
            attention_maps.append(attn)
        pen_weights = None
        if self.use_pen:
            tokens, pen_weights = self.PEN(tokens)
        elif self.use_res_only:
            tokens, _ = self.RES(tokens)
        cam = tokens.reshape(tokens.shape[0], -1)
        logits = self.raw_logits_from_cam(cam)
        return F.log_softmax(logits, dim=1), {
            "cam": cam, "tokens": tokens, "input_attention": input_attn,
            "attention_maps": attention_maps, "pen_weights": pen_weights,
            "logits": logits,
        }


class SignedDARG(nn.Module):
    """Historical E51/E53.3 graph core and E74.4 Positive-DCR operations."""

    def __init__(self, dims, *, fold=1, device=None, modalities=None):
        super().__init__()
        self.dims = tuple(dims)
        self.cadr, self.positive = (darg_sources(self.dims) if modalities is None
                                    else darg_sources(self.dims, tuple(modalities)))
        backbone = e51_legacy_graph(self.dims, device=device)
        self.core = self.cadr.CADRGFR(
            backbone, torch.zeros(len(dims)), torch.tensor([[1.0, 0.0]] * len(dims)),
            torch.zeros(len(dims), dtype=torch.bool),
            init_seed=532000 + int(fold),
            grl_scale=0.03, edge_topk=12, positive_threshold=0.55,
            negative_threshold=0.45, negative_scale=0.25, direct_scale=0.02,
        )
        gate_probability = self.core.fusion_weight.new_full((len(dims),), 0.1)
        if "CORR" in self.cadr.MODALITIES:
            gate_probability[self.cadr.MODALITIES.index("CORR")] = 0.99
        self.core.adaptive_gate_logits = nn.Parameter(torch.logit(gate_probability))

    def enter_positive_dcr(self):
        self.core.edge_gate_logits.requires_grad_(False)

    def relation_named_parameters(self):
        return self.core.added_named_parameters()

    def forward(self, parts, pair_index=None, pair_same=None, *, warmup=False, intervention="full"):
        if warmup:
            fused, original, historical = self.core(parts, pair_index, use_cadr=False)
            unified = original
        else:
            fused, original, historical = self.positive.rudag_forward(
                self.core, list(parts), pair_index, E74_GROUP, intervention=intervention
            )
            unified = historical["unified_adjs"]
        graph_terms = [
            self.core.utils.GraphConstructLoss(part, adj, 1.0, 0.5, 0.0)
            for part, adj in zip(parts, original)
        ]
        continuous_scores = [graph.score431 for graph in self.core.graphs]
        aux = {
            **historical, "raw": torch.cat(tuple(parts), dim=1), "adjacencies": original,
            "unified_adjacencies": unified,
            "continuous_scores": continuous_scores,
            "graph_loss": sum(graph_terms) / len(graph_terms),
            "pair_loss": fused.new_zeros(()), "adv_loss": fused.new_zeros(()),
            "preserve_loss": fused.new_zeros(()), "darg_active": not warmup,
        }
        if not warmup:
            aux.update(
                delta_adjacencies=historical["disease_delta_adjs"],
                direct_delta_adjacencies=historical["direct_delta_adjs"],
                bounded_residuals=historical["disease_residuals_bounded"],
            )
        if pair_same is not None:
            aux["pair_loss"] = torch.stack([
                F.binary_cross_entropy_with_logits(logits, pair_same)
                for logits in historical["pair_logits"]
            ]).mean()
            aux["adv_loss"], aux["adv_info"] = self.cadr.conditional_adversarial_loss(
                historical["pair_reprs"], pair_same, self.core.discriminator, self.core.grl_scale
            )
            aux["preserve_loss"], aux["preserve_gaps"] = self.cadr.relation_preservation_loss(
                historical["pair_logits"], pair_same, margin=0.02
            )
        # The source exposes last for diagnostics. The local caller receives the
        # same tensors and need not keep an additional graph between batches.
        self.core.last = {}
        return fused, aux


def _sign_columns(vectors: Tensor) -> Tensor:
    indices = vectors.abs().argmax(dim=0)
    signs = torch.where(vectors[indices, torch.arange(vectors.shape[1], device=vectors.device)] < 0,
                        -1.0, 1.0)
    return vectors * signs


def _canonical_eigh(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    scale = max(float(values.abs().max()), 1e-30)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and float(values[stop] - values[start]) <= 1e-12 * scale:
            stop += 1
        if stop - start > 1:
            projector = vectors[:, start:stop] @ vectors[:, start:stop].T
            chosen = []
            for coordinate in range(len(values)):
                value = projector[:, coordinate].clone()
                for _ in range(2):
                    for basis in chosen:
                        value = value - basis * torch.dot(basis, value)
                norm = value.norm()
                if norm > 1e-9:
                    chosen.append(value / norm)
                if len(chosen) == stop - start:
                    break
            if len(chosen) != stop - start:
                raise RuntimeError("degenerate spectral eigenspace canonicalization failed")
            vectors[:, start:stop] = torch.stack(chosen, dim=1)
        start = stop
    return values, _sign_columns(vectors)


def _normalize_spectral_stat(value: Tensor) -> Tensor:
    centered = value - value.mean()
    return centered / torch.sqrt(centered.square().mean() + EPS)


def fit_spectral_statistics(cam: Tensor, labels: Tensor, train_idx: Tensor,
                            *, fold: int = 1, warm_logits: Optional[Tensor] = None) -> Dict:
    """Build the E82-style spectrum and cross-fitted Gaussian banks from outer-train CAM.

    The original SCMR bank contains an all-legal reference and one leave-inner-fold
    reference for each inner fold.  This implementation retains that structure and
    extends the class-conditioned moments from two classes to NC/MCI/AD.
    """
    train_idx = train_idx.detach().cpu().long()
    rows = cam.detach().cpu()[train_idx].double()
    y = labels.detach().cpu()[train_idx].long()
    spectral_config = source_config()["spectrum"]
    spectral_dim = spectral_config["pca"] + spectral_config["disease_covariance"]
    center = rows.mean(0)
    centered = rows - center
    covariance = centered.T @ centered / max(1, rows.shape[0] - 1)
    classes = torch.arange(int(torch.max(y).item()) + 1, device=y.device)
    binary_mode = int(classes.numel()) == 2
    priors = torch.stack([(y == c).double().mean() for c in classes])
    class_covariances = torch.stack([
        (rows[y == class_id] - rows[y == class_id].mean(0)).T
        @ (rows[y == class_id] - rows[y == class_id].mean(0))
        / max(int((y == class_id).sum().item()) - 1, 1)
        for class_id in classes
    ])
    pooled_covariance = (class_covariances * priors[:, None, None]).sum(0)
    pca_width = spectral_config["pca"]
    disease_width = spectral_dim - pca_width
    covariance_contrasts = class_covariances - pooled_covariance
    if binary_mode:
        # Preserve the exact E82 binary spectrum construction.  This branch is
        # exercised by the historical compatibility preflight and keeps the
        # TADPOLE three-class extension from changing the source algorithm.
        _, global_vectors = _canonical_eigh(covariance)
        global_basis = _sign_columns(global_vectors[:, -pca_width:].flip(1))
        complement = torch.linalg.qr(global_basis, mode="complete").Q[:, pca_width:]
        projector = torch.eye(rows.shape[1], dtype=rows.dtype, device=rows.device)
        projector = projector - global_basis @ global_basis.T
        binary_difference = class_covariances[0] - class_covariances[1]
        projected_difference = projector @ binary_difference @ projector
        fallback = bool(float(projected_difference.norm()) <= EPS)
        if fallback:
            projected_difference = projector @ covariance @ projector
        disease_values, disease_vectors = _canonical_eigh(
            complement.T @ projected_difference @ complement
        )
        sort_values = disease_values if fallback else disease_values.abs()
        disease_order = torch.argsort(sort_values, descending=True, stable=True)[:disease_width]
        disease_basis = _sign_columns(complement @ disease_vectors[:, disease_order])
    else:
        _, global_vectors = _canonical_eigh(covariance)
        global_basis = _sign_columns(global_vectors[:, -pca_width:].flip(1))
        complement = torch.linalg.qr(global_basis, mode="complete").Q[:, pca_width:]
        pair_mass = sum(priors[left] * priors[right]
                        for left in range(len(priors)) for right in range(left + 1, len(priors)))
        projector = torch.eye(rows.shape[1], dtype=rows.dtype, device=rows.device)
        projector = projector - global_basis @ global_basis.T
        contrast_basis = complement.T @ covariance_contrasts @ complement
        disease_operator = sum(
            priors[class_id] * (contrast @ contrast)
            for class_id, contrast in enumerate(contrast_basis)
        ) / pair_mass.clamp_min(EPS)
        disease_values, disease_vectors = _canonical_eigh(disease_operator)
        disease_order = torch.argsort(disease_values, descending=True, stable=True)[:disease_width]
        disease_basis = _sign_columns(complement @ disease_vectors[:, disease_order])
    basis = torch.cat((global_basis, disease_basis), dim=1)
    if basis.shape[1] != spectral_dim:
        raise RuntimeError(f"spectral basis has width {basis.shape[1]}, expected {spectral_dim}")

    variance = torch.diagonal(basis.T @ covariance @ basis).clamp_min(EPS)
    pooled_projection = torch.diagonal(basis.T @ pooled_covariance @ basis)
    contrast_projection = torch.einsum("di,cde,ei->ci", basis, covariance_contrasts, basis)
    if binary_mode:
        disease_covariance = (basis.T @ (class_covariances[0] - class_covariances[1]) @ basis).diagonal().abs()
    else:
        disease_covariance = ((priors[:, None] * contrast_projection.square()).sum(0)
                              / pair_mass.clamp_min(EPS)).sqrt()
    disease_covariance = _normalize_spectral_stat((disease_covariance / (pooled_projection + EPS)).log1p())
    result = {
        "basis": basis, "center": center, "variance": variance,
        "log_variance": _normalize_spectral_stat(variance.log()),
        "disease_covariance": disease_covariance.unsqueeze(0),
        "train_rows": train_idx,
        "source": "outer-train CAM with 8 PCA and 8 weighted pairwise covariance contrast directions",
    }
    reference = fit_reference(cam, labels, train_idx, result, fold)
    result.update(reference=reference, bank_index=reference["bank"], inner_group=reference["group"])
    if warm_logits is not None:
        hard_initial = torch.zeros(len(labels), dtype=warm_logits.dtype)
        selected_logits = warm_logits.detach().cpu()[train_idx]
        temperature = source_config()["boundary"]["error_temperature"]
        hard_initial[train_idx] = (-true_class_margin(selected_logits, y) / temperature).sigmoid()
        result["hard_initial"] = hard_initial
        if selected_logits.shape[1] == 3:
            pairs = ((0, 1), (0, 2), (1, 2))
            pair_errors = torch.zeros((len(labels), len(pairs)), dtype=warm_logits.dtype)
            for pair_index, (mild, severe) in enumerate(pairs):
                in_pair = (y == mild) | (y == severe)
                sign = torch.where(y == severe, 1., -1.)
                signed_margin = sign * (selected_logits[:, severe] - selected_logits[:, mild])
                pair_errors[train_idx[in_pair], pair_index] = (
                    -signed_margin[in_pair] / temperature
                ).sigmoid()
            result["hard_initial_pairs"] = pair_errors
    return result


class FidelityModel(nn.Module):
    def __init__(self, dims: Sequence[int], group: str, nclass: int = 3,
                 prompt_path: Optional[str] = None, spectral_stats: Optional[Dict[str, Tensor]] = None,
                 dpsl_seed: int = 82460201, *, fold: int = 1, device=None,
                 res_only: bool = False, encoder_attention_dropout: float = 0.1):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.group = group
        self.nclass = int(nclass)
        self.use_pen = group == "G1_E28_E0"
        self.use_darg = group in ("G3_E74_4_G2", "G4_E82_4_6_10_G1")
        self.use_dpsl = group == "G4_E82_4_6_10_G1"
        self.baseline_gfr = None if self.use_darg else HistoricalGFR(self.dims).to(device)
        # E53.3 constructs all original GraphLearn modules and GCNs before CAM.
        self.darg = SignedDARG(self.dims, fold=fold, device=device) if self.use_darg else None
        self.cam = HistoricalCAM(self.dims, hidden=10, heads=4, nlayer=2, nclass=nclass,
                                 dropout=0.0, use_pen=self.use_pen, prompt_path=prompt_path,
                                 encoder_attention_dropout=encoder_attention_dropout).to(device)
        self.cam.apply(original_linear_init)
        if res_only:
            if self.use_pen:
                raise ValueError("RES-only cannot be combined with an active PEN")
            self.cam.enable_res_only()
        self.dpsl = make_scmr(spectral_stats, prompt_path, nclass, dpsl_seed).to(device) if self.use_dpsl and spectral_stats is not None else None

    def attach_dpsl(self, stats: Dict[str, Tensor], prompt_path: str, seed: int):
        if not self.use_dpsl:
            raise RuntimeError("DPSL attach requested for a non-G4 group")
        device = next(self.parameters()).device
        self.dpsl = make_scmr(stats, prompt_path, self.nclass, seed).to(device)

    @property
    def gfr(self):
        return self.darg.core if self.use_darg else self.baseline_gfr

    def graph_parameters(self):
        return self.gfr.base_parameters() if self.use_darg else self.gfr.parameters()

    def relation_parameters(self) -> List[nn.Parameter]:
        return [p for _, p in self.darg.relation_named_parameters()] if self.use_darg else []

    def forward(self, x: Tensor, labels: Tensor, train_idx: Tensor,
                pair_index: Optional[Tensor] = None, pair_same: Optional[Tensor] = None,
                dpsl_enabled: bool = True, darg_warmup: bool = False,
                darg_intervention: str = "full", dpsl_intervention: str = "full"):
        parts, start = [], 0
        for dim in self.dims:
            parts.append(x[:, start:start + dim]); start += dim
        if self.darg is None:
            fused, gaux = self.gfr(parts, labels, train_idx)
            gaux["pair_loss"] = x.new_zeros(()); gaux["adv_loss"] = x.new_zeros(()); gaux["preserve_loss"] = x.new_zeros(())
        else:
            fused, gaux = self.darg(parts, pair_index, pair_same, warmup=darg_warmup,
                                     intervention=darg_intervention)
        logp, caux = self.cam(fused)
        native_logits = caux["logits"]
        enhanced = caux["cam"]
        dpsl_aux, dpsl_loss, dpsl_stats, objectives = {}, x.new_zeros(()), {}, {}
        logits = native_logits
        if self.dpsl is not None and dpsl_enabled:
            head = self.cam.Outputlayer
            enhanced, dpsl_aux = self.dpsl(caux["cam"], head, self.training, dpsl_intervention)
            logits = self.cam.raw_logits_from_cam(enhanced)
            logp = F.log_softmax(logits, dim=1)
            if self.training and torch.is_grad_enabled():
                capture = {"cam": caux["cam"], "enhanced": enhanced, "details": dpsl_aux}
                objectives, dpsl_stats = self.dpsl.historical_objectives(head, capture, labels, train_idx)
                controls = source_config()["auxiliary"]
                dpsl_loss = sum(controls[key]["coefficient"] * objectives[key] for key in controls)
            dpsl_aux.update(residual=dpsl_aux["delta"], residual_ratio=dpsl_aux["ratio"],
                            raw_residual=dpsl_aux["raw_parallel"] + dpsl_aux["raw_perp"])
        elif self.dpsl is not None:
            dpsl_aux = {"residual": torch.zeros_like(enhanced),
                        "residual_ratio": enhanced.new_zeros(len(enhanced))}
        return logp, {**gaux, **caux, "native_logits": native_logits, "enhanced_cam": enhanced,
                      "logits": logits, "dpsl": dpsl_aux, "dpsl_loss": dpsl_loss,
                      "dpsl_stats": dpsl_stats, "scmr_objectives": objectives, "pen_present": self.cam.use_pen,
                      "darg_active": self.use_darg and not darg_warmup, "dpsl_active": self.use_dpsl}


def base_parameters(model: FidelityModel) -> List[nn.Parameter]:
    return list(model.graph_parameters()) + list(model.cam.parameters())

