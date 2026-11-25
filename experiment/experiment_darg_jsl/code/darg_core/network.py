"""Graph modules used by the archived DARG core, without Prompt/CLIP code."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode="Sigmoid-like"):
        super().__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th

    def forward(self, x):
        x = self.p(x)
        x_norm = F.normalize(x, dim=-1)
        score = torch.matmul(x_norm, x_norm.T)
        mask = (score > self.th).detach().float()
        return score * mask


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        import layers

        self.gcn1 = layers.GraphConv(input_dim, output_dim)
        self.gcn2 = layers.GraphConv(input_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        return self.gcn2(x, adj)
