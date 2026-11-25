"""Loss, normalization and initialization used in the project's TADPOLE port."""

import torch
from torch import nn
from torch.nn import functional as F


def ClsLoss(output, labels, idx, weight):
    return F.nll_loss(output[idx], labels[idx], weight)


def normalize_adj(adj):
    degree = torch.sum(adj, -1)
    inverse_sqrt = torch.diagflat(torch.pow(degree, -0.5))
    return torch.mm(inverse_sqrt, torch.mm(adj, inverse_sqrt))


def GraphConstructLoss(feat, adj, theta_smooth, theta_degree, theta_sparsity):
    laplacian = torch.diagflat(torch.sum(adj, -1)) - adj
    ones = adj.new_ones(adj.size(-1))
    smoothness = torch.trace(torch.mm(feat.T, torch.mm(laplacian, feat))) / adj.numel()
    degree = torch.mm(ones.unsqueeze(0), torch.log(torch.mm(adj, ones.unsqueeze(-1)) + 1e-5))
    degree = degree.squeeze() / adj.shape[-1]
    sparsity = torch.sum(torch.pow(adj, 2)) / adj.numel()
    return theta_smooth * smoothness - theta_degree * degree + theta_sparsity * sparsity


def my_weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
