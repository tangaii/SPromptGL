"""Frozen HO texts; legacy 2x2 repair and centered semantic ROI graph.

No labels are accepted. Legacy initialization/operators preserve experiment15.
The coupled decoder has no direct FC channel and subtracts its zero reference.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def patient_graph(raw_fc, feature_indices, endpoints, nodes=110):
    z = np.asarray(raw_fc, dtype=np.float32)[:, feature_indices]
    ends = np.asarray(endpoints, dtype=np.int64)
    degree = np.bincount(ends.ravel(), minlength=nodes).astype(np.float64)
    assert len(feature_indices) == 252 and degree.min() >= 1
    both = np.concatenate([ends[:, 0], ends[:, 1]])
    values = np.concatenate([z, z], axis=1).astype(np.float64)
    sums = np.zeros((len(z), nodes), dtype=np.float64)
    abs_sums, sq_sums, maxima = [np.zeros_like(sums) for _ in range(3)]
    np.add.at(sums, (slice(None), both), values)
    np.add.at(abs_sums, (slice(None), both), np.abs(values))
    np.add.at(sq_sums, (slice(None), both), values ** 2)
    np.maximum.at(maxima, (slice(None), both), np.abs(values))
    mean = sums / degree
    sd = np.sqrt(np.maximum(sq_sums / degree - mean ** 2, 0.))
    sd[:, degree == 1] = 0.
    stats = np.stack([mean, abs_sums / degree, sd, maxima,
                      np.broadcast_to(degree / degree.max(), mean.shape)], -1)
    adjacency = np.zeros((len(z), nodes, nodes), dtype=np.float32)
    weights = np.tanh(z)
    adjacency[:, ends[:, 0], ends[:, 1]] = weights
    adjacency[:, ends[:, 1], ends[:, 0]] = weights
    signed = adjacency / degree[None, :, None].astype(np.float32)
    positive, negative = np.maximum(adjacency, 0), np.maximum(-adjacency, 0)
    for a in [positive, negative]:
        a /= np.maximum(a.sum(-1, keepdims=True), 1e-12)
    assert all(np.isfinite(a).all() for a in [stats, signed, positive, negative])
    return dict(stats=torch.from_numpy(stats.astype(np.float32)),
                adjacency=torch.from_numpy(signed),
                positive=torch.from_numpy(positive),negative=torch.from_numpy(negative),
                valid_fc=torch.from_numpy(z.copy()),degree=torch.from_numpy(degree.astype(np.float32)),
                feature_indices=torch.as_tensor(feature_indices, dtype=torch.long),
                endpoints=torch.as_tensor(ends, dtype=torch.long))


class LegacyROIEncoder(nn.Module):
    def __init__(self, fixed, text_features, self_only=False, disable_fc=False):
        super().__init__()
        for key, value in fixed.items():
            self.register_buffer(key, value, persistent=False)
        assert torch.allclose(text_features.norm(dim=-1), torch.ones(110), atol=1e-6, rtol=0)
        self.register_buffer('text_features', text_features.float().detach().clone())
        self.numerical = nn.Sequential(nn.Linear(5, 32), nn.GELU(),nn.Linear(32, 32), nn.GELU())
        self.query = nn.Linear(32, 1024, bias=False)
        self.message = nn.Linear(32, 32, bias=False)
        self.self_transform = nn.Linear(32, 32)
        self.decoder = nn.Sequential(nn.Linear(65, 32), nn.GELU(), nn.Linear(32, 1))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.self_only = bool(self_only)
        self.disable_fc = bool(disable_fc)
        self.text_override = None
        self.self_only_override = None
        self.off = False
        self.last = {}
        assert sum(p.numel() for p in self.parameters()) == 38242

    def similarities(self, h, text_features):
        weight = self.query.weight
        gram = weight.T @ weight
        query_norm = ((h @ gram) * h).sum(-1).clamp_min(1e-24).sqrt()
        numerator = (h * (text_features @ weight)[None]).sum(-1)
        return numerator / query_norm, query_norm

    def forward(self):
        text = self.text_features if self.text_override is None else self.text_override
        h = self.numerical(self.stats)
        score, query_norm = self.similarities(h, text)
        response = 0.5 * torch.tanh(score / 0.2)
        conditioned = h * (1. + response[..., None])
        source = self.message(conditioned)
        self_only = self.self_only if self.self_only_override is None else self.self_only_override
        message = source if self_only else torch.bmm(self.adjacency, source)
        self_path = self.self_transform(conditioned)
        updated = F.gelu(self_path + message)
        i, j = self.endpoints[:, 0], self.endpoints[:, 1]
        left, right = updated[:, i], updated[:, j]
        fc_input = torch.zeros_like(self.valid_fc) if self.disable_fc else self.valid_fc
        edge_context = torch.cat([left + right, (left - right).abs(),fc_input[..., None]], -1)
        raw = self.decoder(edge_context).squeeze(-1)
        beta = 0.3 * self.alpha.tanh()
        valid_delta = beta * raw.tanh()
        if self.off:
            valid_delta = torch.zeros_like(valid_delta)
        delta = valid_delta.new_zeros(len(valid_delta), 256).scatter(
            1,self.feature_indices[None].expand(len(valid_delta),-1),valid_delta)
        self.last = {k:v.detach() for k,v in dict(h=h,score=score,query_norm=query_norm,
            centered_score=score-score.mean(1,keepdim=True),response=response,
            semantic_node=conditioned,message=message,self_path=self_path,
            updated=updated,edge_signal=raw,delta=delta).items()}
        return delta


class CoupledROIEncoder(nn.Module):
    def __init__(self, fixed, text_features, self_only=False):
        super().__init__()
        for key,value in fixed.items():
            self.register_buffer(key,value,persistent=False)
        assert torch.allclose(text_features.norm(dim=-1),torch.ones(110),atol=1e-6,rtol=0)
        self.register_buffer('text_features',text_features.float().detach().clone())
        self.numerical = nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.query = nn.Linear(32,1024,bias=False)
        self.self_transform = nn.Linear(32,32,bias=False)
        self.positive_map = nn.Linear(32,32,bias=False)
        self.negative_map = nn.Linear(32,32,bias=False)
        # An output bias would cancel identically in raw - zero_reference;
        # omit that redundant scalar, equally for B5--B9.
        self.decoder = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1,bias=False))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.self_only = bool(self_only)
        self.text_override = None
        self.self_only_override = None
        self.off = False
        self.last = {}
        assert sum(p.numel() for p in self.parameters()) == 39201

    similarities = LegacyROIEncoder.similarities

    def update_nodes(self,semantic_node):
        self_only = self.self_only if self.self_only_override is None else self.self_only_override
        m_pos = semantic_node if self_only else torch.bmm(self.positive,semantic_node)
        m_neg = semantic_node if self_only else torch.bmm(self.negative,semantic_node)
        self_path = self.self_transform(semantic_node)
        pos_path,neg_path = self.positive_map(m_pos),self.negative_map(m_neg)
        updated = torch.tanh(self_path+pos_path+neg_path)
        return updated,dict(m_pos=m_pos,m_neg=m_neg,self_path=self_path,
                            pos_path=pos_path,neg_path=neg_path,message=pos_path+neg_path)

    def decode_edges(self,updated):
        i,j = self.endpoints[:,0],self.endpoints[:,1]
        left,right = updated[:,i],updated[:,j]
        edge_context = torch.cat([left+right,(left-right).abs()],-1)
        raw = self.decoder(edge_context).squeeze(-1)
        # Use the same GEMM shape for the zero reference. Broadcasting a
        # single zero row introduces device-dependent rounding differences
        # and can violate the exact zero-node/zero-residual invariant.
        zero_reference = self.decoder(torch.zeros_like(edge_context)).squeeze(-1)
        edge_signal = raw-zero_reference
        return edge_signal

    def forward(self):
        text = self.text_features if self.text_override is None else self.text_override
        h = self.numerical(self.stats)
        score,query_norm = self.similarities(h,text)
        centered_score = score-score.mean(dim=1,keepdim=True)
        response = torch.tanh(centered_score/0.2)
        semantic_node = h*response.unsqueeze(-1)
        updated,paths = self.update_nodes(semantic_node)
        edge_signal = self.decode_edges(updated)
        valid_delta = (0.3*self.alpha.tanh())*edge_signal.tanh()
        if self.off:
            valid_delta = torch.zeros_like(valid_delta)
        delta = valid_delta.new_zeros(len(valid_delta),256).scatter(
            1,self.feature_indices[None].expand(len(valid_delta),-1),valid_delta)
        self.last = {k:v.detach() for k,v in dict(h=h,score=score,query_norm=query_norm,
            centered_score=centered_score,response=response,semantic_node=semantic_node,
            updated=updated,edge_signal=edge_signal,delta=delta,**paths).items()}
        return delta
