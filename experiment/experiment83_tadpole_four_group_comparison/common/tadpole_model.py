"""E83 isolated SPromptGL adapters for TADPOLE.

The module keeps the original chain graph relation learning -> CAM/PEN -> classifier,
and adds the E74 positive disease-aware residual and the E82 spectral prompt residual
without importing historical ABIDE data, checkpoints or patient lists.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def normalize_adj(adj: Tensor) -> Tensor:
    degree = adj.sum(dim=1).clamp_min(1e-6)
    inv = degree.rsqrt()
    return inv[:, None] * adj * inv[None, :]


def graph_regularizer(feat: Tensor, adj: Tensor) -> Tensor:
    n = float(adj.shape[0] * adj.shape[1])
    degree = adj.sum(dim=1)
    lap = torch.diag(degree) - adj
    smooth = torch.trace(feat.transpose(0, 1) @ lap @ feat) / n
    degree_penalty = torch.log(degree.clamp_min(1e-5)).mean()
    sparsity = adj.square().mean()
    return smooth - 0.5 * degree_penalty + 0.0 * sparsity


class GraphLearn(nn.Module):
    def __init__(self, dim: int, threshold: float = 0.9):
        super().__init__()
        self.project = nn.Linear(dim, dim, bias=False)
        self.threshold = float(threshold)

    def forward(self, x: Tensor) -> Tensor:
        z = F.normalize(self.project(x), dim=-1)
        score = z @ z.transpose(0, 1)
        mask = (score > self.threshold).to(score.dtype)
        return score * mask


class GCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w2 = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.w1, gain=1.414)
        nn.init.xavier_uniform_(self.w2, gain=1.414)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        h = F.relu(adj @ (x @ self.w1))
        return adj @ (h @ self.w2)


class DynamicGFR(nn.Module):
    """Dynamic six-modality GFR with optional E74 positive DARG residual."""
    def __init__(self, dims: Sequence[int], threshold: float = 0.9,
                 darg: bool = False, positive_only: bool = False,
                 direct_scale: float = 0.02):
        super().__init__()
        self.dims = list(map(int, dims))
        self.modal_num = len(self.dims)
        self.total_dim = int(sum(self.dims))
        self.graphs = nn.ModuleList([GraphLearn(d, threshold) for d in self.dims])
        self.messages = nn.ModuleList([GCN(self.total_dim, self.total_dim) for _ in self.dims])
        self.fusion_weight = nn.Parameter(torch.ones(self.modal_num, 1))
        self.combined_weight = nn.Parameter(torch.ones(self.modal_num, 1))
        self.darg = bool(darg)
        self.positive_only = bool(positive_only)
        self.direct_scale = float(direct_scale)
        if self.darg:
            self.relation_proj = nn.ModuleList([nn.Linear(d, 32, bias=False) for d in self.dims])
            self.relation_log_temperature = nn.Parameter(torch.zeros(self.modal_num))
            self.darg_gate = nn.Parameter(torch.zeros(self.modal_num))

    def _label_adjacency(self, labels: Tensor, train_idx: Tensor) -> Tensor:
        n = labels.numel()
        train_mask = torch.zeros(n, dtype=torch.bool, device=labels.device)
        train_mask[train_idx] = True
        same = labels[:, None].eq(labels[None, :])
        known = train_mask[:, None] & train_mask[None, :]
        # Only train labels supervise disease-aware relation learning. Diagonal self
        # loops remain available for every node in the normalized propagation graph.
        return same.to(torch.float32) * known.to(torch.float32)

    def forward(self, modals: Sequence[Tensor], labels: Tensor, train_idx: Tensor,
                pair_data: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> Tuple[Tensor, Dict]:
        full = torch.cat(list(modals), dim=1)
        label_adj = self._label_adjacency(labels, train_idx)
        target_norm = normalize_adj(label_adj + torch.eye(full.shape[0], device=full.device))
        combined = full
        all_adj: List[Tensor] = []
        all_effective: List[Tensor] = []
        graph_loss = full.new_zeros(())
        pair_losses: List[Tensor] = []
        preserve_losses: List[Tensor] = []
        relation_grad_proxy = full.new_zeros(())
        direct_norms: List[Tensor] = []
        start = 0
        for i, (mod, graph, message) in enumerate(zip(modals, self.graphs, self.messages)):
            end = start + self.dims[i]
            base_adj = graph(mod)
            effective = base_adj
            direct_local = torch.zeros_like(mod)
            relation_logits = None
            if self.darg:
                rel = F.normalize(self.relation_proj[i](mod), dim=-1)
                relation_logits = (rel @ rel.transpose(0, 1)) * self.relation_log_temperature[i].exp()
                delta = base_adj * (2.0 * torch.sigmoid(relation_logits) - 1.0)
                eye = torch.eye(delta.shape[0], device=delta.device, dtype=delta.dtype)
                delta = delta * (1.0 - eye)
                if self.positive_only:
                    delta = delta.clamp_min(0.0)
                gate = self.direct_scale * torch.sigmoid(self.darg_gate[i])
                delta = delta * gate
                effective = (base_adj + delta).clamp_min(0.0)
                denom = base_adj.sum(dim=1).clamp_min(1.0)
                # Disease residual is localized to this modality block, matching the
                # E74 local residual construction while keeping six real TADPOLE blocks.
                direct_local = (delta @ mod - delta.sum(dim=1, keepdim=True) * mod) / denom[:, None]
                relation_grad_proxy = relation_grad_proxy + relation_logits.square().mean()
                preserve_losses.append((effective - base_adj).square().mean())
                if pair_data is not None:
                    pi, pj, pt = pair_data
                    pair_losses.append(F.binary_cross_entropy_with_logits(relation_logits[pi, pj], pt))
                direct_norms.append(direct_local.norm(dim=1).mean())
            all_adj.append(base_adj)
            all_effective.append(effective)
            normalized = normalize_adj(effective + torch.eye(full.shape[0], device=full.device))
            msg = message(full, normalized)
            fw = torch.sigmoid(self.fusion_weight[i])
            combined = combined + fw * msg
            if self.darg:
                # Add the bounded local disease residual after message passing.
                combined[:, start:end] = combined[:, start:end] + fw * direct_local
            graph_loss = graph_loss + torch.linalg.norm(normalized - target_norm)
            start = end
        aux = {
            'adjacencies': all_adj,
            'effective_adjacencies': all_effective,
            'graph_loss': graph_loss / float(self.modal_num * full.shape[0]),
            'pair_loss': torch.stack(pair_losses).mean() if pair_losses else full.new_zeros(()),
            'preserve_loss': torch.stack(preserve_losses).mean() if preserve_losses else full.new_zeros(()),
            'relation_grad_proxy': relation_grad_proxy / float(max(self.modal_num, 1)),
            'direct_residual_norm': torch.stack(direct_norms).mean() if direct_norms else full.new_zeros(()),
            'combined': combined,
        }
        return combined, aux


class ModalInput(nn.Module):
    def __init__(self, dims: Sequence[int], hidden: int, heads: int, dropout: float):
        super().__init__()
        self.dims = list(map(int, dims))
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.qs = nn.ModuleList([nn.Linear(d, heads * hidden, bias=False) for d in self.dims])
        self.ks = nn.ModuleList([nn.Linear(d, heads * hidden, bias=False) for d in self.dims])
        self.vs = nn.ModuleList([nn.Linear(d, heads * hidden, bias=False) for d in self.dims])
        self.out = nn.Linear(heads * hidden, heads * hidden)
        self.norm = nn.LayerNorm(heads * hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, modals: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        q = torch.stack([m(x).view(x.shape[0], self.heads, self.hidden) for m, x in zip(self.qs, modals)], dim=2)
        k = torch.stack([m(x).view(x.shape[0], self.heads, self.hidden) for m, x in zip(self.ks, modals)], dim=2)
        v = torch.stack([m(x).view(x.shape[0], self.heads, self.hidden) for m, x in zip(self.vs, modals)], dim=2)
        attn = torch.softmax(torch.einsum('nhmd,nhkd->nhmk', q, k) / math.sqrt(self.hidden), dim=-1)
        out = torch.einsum('nhmk,nhkd->nhmd', attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], len(self.dims), self.heads * self.hidden)
        residual = v.transpose(1, 2).contiguous().view(out.shape[0], len(self.dims), self.heads * self.hidden)
        out = self.norm(residual + self.dropout(self.out(out)))
        return out.view(out.shape[0], len(self.dims), self.heads, self.hidden), attn


class EncodeBlock(nn.Module):
    def __init__(self, width: int, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.heads = heads
        self.q = nn.Linear(width, heads * hidden, bias=False)
        self.k = nn.Linear(width, heads * hidden, bias=False)
        self.v = nn.Linear(width, heads * hidden, bias=False)
        self.out = nn.Linear(heads * hidden, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        n, m, _ = tokens.shape
        q = self.q(tokens).view(n, m, self.heads, self.hidden).transpose(1, 2)
        k = self.k(tokens).view(n, m, self.heads, self.hidden).transpose(1, 2)
        v = self.v(tokens).view(n, m, self.heads, self.hidden).transpose(1, 2)
        attn = torch.softmax(torch.einsum('nhmd,nhkd->nhmk', q, k) / math.sqrt(self.hidden), dim=-1)
        out = torch.einsum('nhmk,nhkd->nhmd', attn, v).transpose(1, 2).contiguous().view(n, m, -1)
        out = self.norm(tokens + self.dropout(self.out(out)))
        out = self.ff_norm(out + self.ff(out))
        return out, attn


class PEN(nn.Module):
    """The original eight-prompt, two-concept PEN generalized to six modalities."""
    def __init__(self, hidden: int, prompt_path: str):
        super().__init__()
        prompt = torch.load(prompt_path, map_location='cpu', weights_only=False)
        text_features = prompt['medical'].float()
        self.register_buffer('prompt_text_features', text_features)
        self.mlp = nn.Sequential(nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, hidden), nn.ReLU())

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        # tokens [N, M, H, D], concept vectors preserve the original first-four vs last-four split.
        concepts = torch.cat((self.prompt_text_features[:4].reshape(1, -1), self.prompt_text_features[4:].reshape(1, -1)), dim=0)
        concepts = F.normalize(self.mlp(concepts), dim=-1)
        norm_tokens = F.normalize(tokens, dim=-1)
        sim = torch.einsum('nmhd,cd->nmhc', norm_tokens, concepts)
        weights = torch.softmax(sim, dim=-1).amax(dim=-1)
        semantic = tokens * weights[..., None]
        return tokens + semantic, weights


class CAMHead(nn.Module):
    def __init__(self, cam_dim: int, nclass: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(cam_dim)
        self.hidden = nn.Linear(cam_dim, 40)
        self.classifier = nn.Linear(40, nclass)
        self.dropout = nn.Dropout(dropout)

    def logits(self, cam: Tensor) -> Tensor:
        return self.classifier(self.dropout(F.gelu(self.hidden(self.norm(cam)))))


class DPSL(nn.Module):
    """TADPOLE three-class adaptation of E82 spectral prompt learning.

    The fixed eight prompt bank, spectral token count, patient-specific prompt
    conditioning, bounded residual and gradient-constrained residual follow the E82
    chain. Binary ASD margin is generalized to true-class versus strongest competing
    class margin.
    """
    def __init__(self, cam_dim: int, nclass: int, prompt_path: str, spectral_dim: int = 16):
        super().__init__()
        self.cam_dim = int(cam_dim)
        self.nclass = int(nclass)
        self.spectral_dim = int(spectral_dim)
        prompt = torch.load(prompt_path, map_location='cpu', weights_only=False)
        self.register_buffer('q_medical', prompt['q_medical'].float())
        self.register_buffer('q_anonymous', prompt['q_anonymous'].float())
        gen = torch.Generator(device='cpu').manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(self.cam_dim, self.spectral_dim, generator=gen), mode='reduced')
        self.register_buffer('basis', q)
        self.prompt_to_cam = nn.Linear(32, cam_dim, bias=False)
        self.spectral_norm = nn.LayerNorm(self.spectral_dim)
        self.controller = nn.Sequential(nn.Linear(cam_dim * 2 + nclass, 128), nn.GELU(), nn.Linear(128, cam_dim))
        self.context_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, cam: Tensor, native_logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        coords = cam @ self.basis
        white = self.spectral_norm(coords)
        spectral_tokens = white.unsqueeze(-1) * self.basis.transpose(0, 1).unsqueeze(0)
        medical_q = F.normalize(self.prompt_to_cam(self.q_medical), dim=-1)
        anonymous_q = F.normalize(self.prompt_to_cam(self.q_anonymous), dim=-1)
        token_norm = F.normalize(spectral_tokens, dim=-1)
        medical_attn = torch.softmax(torch.einsum('nsd,qd->nsq', token_norm, medical_q) / math.sqrt(self.cam_dim), dim=-1)
        anonymous_attn = torch.softmax(torch.einsum('nsd,qd->nsq', token_norm, anonymous_q) / math.sqrt(self.cam_dim), dim=-1)
        medical_context = torch.einsum('nsq,qd->nsd', medical_attn, medical_q).mean(dim=1)
        anonymous_context = torch.einsum('nsq,qd->nsd', anonymous_attn, anonymous_q).mean(dim=1)
        context = (medical_context - anonymous_context) * torch.sigmoid(self.context_gate)
        raw = self.controller(torch.cat((cam, context, native_logits), dim=-1))
        raw_norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cap = 0.08 * cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        amplitude = torch.sigmoid(self.residual_scale)
        residual = raw / raw_norm * cap * amplitude
        enhanced = cam + residual
        aux = {'cam': cam, 'spectral_coordinates': coords, 'spectral_attention': medical_attn,
               'anonymous_attention': anonymous_attn, 'prompt_context': context,
               'medical_context': medical_context, 'anonymous_context': anonymous_context,
               'prompt_direction': F.normalize(medical_context.detach().mean(dim=0) - anonymous_context.detach().mean(dim=0), dim=-1),
               'residual': residual, 'residual_ratio': residual.norm(dim=-1) / cam.norm(dim=-1).clamp_min(1e-8),
               'native_logits': native_logits}
        return enhanced, aux

    def loss(self, aux: Dict[str, Tensor], labels: Tensor, train_idx: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        residual = aux['residual'][train_idx]
        cam = aux['cam']
        native = aux['native_logits']
        true_logits = native[train_idx].gather(1, labels[train_idx, None]).squeeze(1)
        other_logits = native[train_idx].masked_fill(F.one_hot(labels[train_idx], self.nclass).bool(), -1e9).max(dim=1).values
        native_margin = true_logits - other_logits
        residual_penalty = aux['residual_ratio'][train_idx].square().mean()
        direction = aux['prompt_direction'].unsqueeze(0).expand_as(aux['prompt_context'][train_idx])
        semantic_penalty = (1.0 - F.cosine_similarity(aux['prompt_context'][train_idx], direction, dim=-1)).mean()
        # This is an actual first-order CAM gradient constraint. The gradient of the
        # true-class margin is retained in the graph so the DPSL residual is trained to
        # act in a class-discriminative direction.
        margin_grad = torch.autograd.grad(native_margin.sum(), cam, create_graph=True, retain_graph=True, allow_unused=False)[0][train_idx]
        gradient_alignment = (1.0 - F.cosine_similarity(residual, margin_grad, dim=-1)).mean()
        margin_loss = F.softplus(-native_margin).mean()
        total = 0.01 * residual_penalty + 0.001 * semantic_penalty + 0.001 * gradient_alignment + 0.001 * margin_loss
        return total, {'residual_penalty': float(residual_penalty.detach()), 'semantic_penalty': float(semantic_penalty.detach()), 'gradient_constraint': float(gradient_alignment.detach()), 'margin_loss': float(margin_loss.detach())}


class E83Model(nn.Module):
    def __init__(self, dims: Sequence[int], nclass: int, hidden: int, heads: int, nlayer: int,
                 dropout: float, group: str, prompt_path: str, threshold: float = 0.9):
        super().__init__()
        self.dims = list(map(int, dims))
        self.nclass = int(nclass)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.modal_num = len(self.dims)
        self.group = group
        self.use_pen = group in {'G1_E28_E0', 'G3_E74_4_G2', 'G4_E82_4_6_10_G1'}
        self.use_darg = group in {'G3_E74_4_G2', 'G4_E82_4_6_10_G1'}
        self.use_dpsl = group == 'G4_E82_4_6_10_G1'
        self.gfr = DynamicGFR(self.dims, threshold, darg=self.use_darg, positive_only=self.use_darg, direct_scale=0.02)
        self.input_layer = ModalInput(self.dims, hidden, heads, dropout)
        width = hidden * heads
        self.encoders = nn.ModuleList([EncodeBlock(width, hidden, heads, dropout) for _ in range(nlayer)])
        self.pen = PEN(hidden, prompt_path) if self.use_pen else None
        self.cam_dim = self.modal_num * width
        self.head = CAMHead(self.cam_dim, nclass, dropout)
        self.dpsl = DPSL(self.cam_dim, nclass, prompt_path) if self.use_dpsl else None

    def forward(self, x: Tensor, labels: Tensor, train_idx: Tensor,
                pair_data: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> Tuple[Tensor, Dict]:
        modals=[]
        start=0
        for d in self.dims:
            modals.append(x[:,start:start+d]); start += d
        fused, gaux = self.gfr(modals, labels, train_idx, pair_data)
        tokens, attn = self.input_layer([fused[:, sum(self.dims[:i]):sum(self.dims[:i+1])] for i in range(self.modal_num)])
        enc_attn=[]
        flat_tokens=tokens.view(tokens.shape[0], self.modal_num, -1)
        for enc in self.encoders:
            flat_tokens, ea = enc(flat_tokens)
            enc_attn.append(ea)
        tokens=flat_tokens.view(flat_tokens.shape[0], self.modal_num, self.heads, self.hidden)
        pen_weights=None
        if self.pen is not None:
            tokens, pen_weights = self.pen(tokens)
        cam=tokens.reshape(tokens.shape[0],-1)
        native_logits=self.head.logits(cam)
        dpsl_aux={}
        if self.dpsl is not None:
            enhanced, dpsl_aux=self.dpsl(cam,native_logits)
            logits=self.head.logits(enhanced)
            if torch.is_grad_enabled():
                dpsl_loss,dpsl_stats=self.dpsl.loss(dpsl_aux,labels,train_idx)
            else:
                dpsl_loss=cam.new_zeros(())
                dpsl_stats={}
        else:
            enhanced=cam
            logits=native_logits
            dpsl_loss=cam.new_zeros(())
            dpsl_stats={}
        aux={**gaux,'cam':cam,'enhanced_cam':enhanced,'native_logits':native_logits,'logits':logits,
             'attn':attn,'encoder_attn':enc_attn,'pen_weights':pen_weights,'dpsl':dpsl_aux,
             'dpsl_loss':dpsl_loss,'dpsl_stats':dpsl_stats,'pen_present':self.pen is not None,
             'darg_active':self.use_darg,'dpsl_active':self.use_dpsl}
        return F.log_softmax(logits,dim=-1), aux
