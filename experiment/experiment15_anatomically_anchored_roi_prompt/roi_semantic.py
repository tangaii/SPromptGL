"""One-layer patient ROI graph; frozen HO text, trainable numerical queries.

No labels are accepted by this module. Fixed patient statistics/adjacency are
computed once, excluded from checkpoints, and reconstructed from hashed input.
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
    adjacency /= degree[None, :, None].astype(np.float32)
    assert np.isfinite(stats).all() and np.isfinite(adjacency).all()
    return dict(stats=torch.from_numpy(stats.astype(np.float32)),
                adjacency=torch.from_numpy(adjacency),
                valid_fc=torch.from_numpy(z.copy()),
                degree=torch.from_numpy(degree.astype(np.float32)),
                feature_indices=torch.as_tensor(feature_indices, dtype=torch.long),
                endpoints=torch.as_tensor(ends, dtype=torch.long))


class ROIEncoder(nn.Module):
    def __init__(self, fixed, text_features, self_only=False):
        super().__init__()
        for key, value in fixed.items():
            self.register_buffer(key, value, persistent=False)
        # The cached bank is already normalized. Re-normalizing changes float32
        # bits and would make REAL and counterfactual banks differ unnecessarily.
        assert torch.allclose(text_features.norm(dim=-1), torch.ones(110), atol=1e-6, rtol=0)
        self.register_buffer('text_features', text_features.float().detach().clone())
        self.numerical = nn.Sequential(nn.Linear(5, 32), nn.GELU(),
                                       nn.Linear(32, 32), nn.GELU())
        self.query = nn.Linear(32, 1024, bias=False)
        self.message = nn.Linear(32, 32, bias=False)
        self.self_transform = nn.Linear(32, 32)
        self.decoder = nn.Sequential(nn.Linear(65, 32), nn.GELU(), nn.Linear(32, 1))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.self_only = bool(self_only)
        self.text_override = None
        self.self_only_override = None
        self.off = False
        self.last = {}
        assert sum(p.numel() for p in self.parameters()) == 38242

    def similarities(self, h, text_features):
        # Exact algebra of normalize(Wq h) dot normalize(e), without an
        # [871,110,1024] query activation. No trainable text projection.
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
        # A7 uses I instead of patient adjacency. The message map remains active.
        message = source if self_only else torch.bmm(self.adjacency, source)
        updated = F.gelu(self.self_transform(conditioned) + message)
        i, j = self.endpoints[:, 0], self.endpoints[:, 1]
        left, right = updated[:, i], updated[:, j]
        edge_context = torch.cat([left + right, (left - right).abs(),
                                  self.valid_fc[..., None]], -1)
        raw = self.decoder(edge_context).squeeze(-1)
        beta = 0.3 * self.alpha.tanh()
        valid_delta = beta * raw.tanh()
        if self.off:
            valid_delta = torch.zeros_like(valid_delta)
        delta = valid_delta.new_zeros(len(valid_delta), 256).scatter(
            1, self.feature_indices[None].expand(len(valid_delta), -1), valid_delta)
        self.last = {k: v.detach() for k, v in dict(
            h=h, score=score, query_norm=query_norm, response=response,
            message=message, updated=updated, raw=raw, delta=delta).items()}
        return delta
