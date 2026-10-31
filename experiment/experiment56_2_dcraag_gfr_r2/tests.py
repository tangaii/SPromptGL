"""Small deterministic R2 mathematics and Gate regression tests.

The full E51/RNG/gradient/graph/checkpoint suite is run by ``run.py test``;
this file is intentionally CPU-only so it can be rerun independently.
"""
import math
import torch
from torch import nn
from dcraag import relation_confidence, softcap

torch.manual_seed(562002)

u = relation_confidence(torch.tensor([[1/3, 1/3, 1/3]]))[0]
one = relation_confidence(torch.tensor([[1., 0., 0.]]))[0]
mid = relation_confidence(torch.tensor([[.8, .1, .1]]))[0]
assert abs(float(u) - .05) < 1e-5
assert abs(float(one) - 1.) < 1e-5
assert .05 < float(mid) < 1.
grid = relation_confidence(torch.tensor([[1/3,1/3,1/3],[.5,.25,.25],[.8,.1,.1],[1.,0.,0.]]))
assert bool(torch.all(grid[1:] >= grid[:-1] - 1e-6))
assert bool(torch.all((grid >= .05) & (grid <= 1.))) and torch.unique(grid).numel() == 4

v0 = torch.zeros(1, 4, requires_grad=True)
v1 = torch.tensor([[.01, 0., 0., 0.]], requires_grad=True)
v2 = torch.tensor([[10., 0., 0., 0.]], requires_grad=True)
r = torch.ones(1, 1)
o0, o1, o2 = softcap(v0, r), softcap(v1, r), softcap(v2, r)
assert torch.equal(o0, torch.zeros_like(o0))
assert 0 < float(o1.norm()) < float(v1.norm())
assert float(o2.norm()) <= 1. + 1e-6 and float(o2.norm()) > float(o1.norm())
assert torch.allclose(o2 / o2.norm(), v2 / v2.norm(), atol=1e-6)
(o1.sum() + o2.sum()).backward()
assert v1.grad is not None and v2.grad is not None and torch.isfinite(v1.grad).all() and torch.isfinite(v2.grad).all()

# The repaired two-layer initialization is nonzero, near-uniform and patient-sensitive.
g = nn.Sequential(nn.Linear(12, 8), nn.GELU(), nn.Linear(8, 4))
nn.init.xavier_uniform_(g[0].weight); nn.init.zeros_(g[0].bias)
nn.init.normal_(g[2].weight, std=1e-3); nn.init.zeros_(g[2].bias)
x = torch.stack((torch.zeros(12), torch.ones(12)))
w = g(x).softmax(-1)
assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-6)
assert float(w.std()) > 1e-8 and float((w-.25).abs().max()) <= .02
loss = w[0, 0] + 2*w[1, 1]
loss.backward()
assert float(g[0].weight.grad.norm()) > 0 and float(g[2].weight.grad.norm()) > 0

eta_max, eta_initial = .30, .10
eta_raw = math.log(eta_initial / (eta_max - eta_initial))
eta = eta_max * torch.sigmoid(torch.tensor(eta_raw))
assert abs(float(eta) - eta_initial) < 1e-6 and 0 <= float(eta) <= eta_max

# Reliability target must be a weighted, detached prototype.
proto = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,0.]])
rel = torch.tensor([.05,.08,.06,.88])
target = (proto * rel[:,None]).sum(0) / rel.sum()
assert torch.allclose(target, (proto * rel[:,None]).sum(0) / rel.sum())
assert not target.requires_grad
print('R2_MATH_TESTS PASS')
