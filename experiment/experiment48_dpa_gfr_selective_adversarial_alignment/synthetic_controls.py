#!/usr/bin/env python3
"""Small, deterministic implementation controls separate from ABIDE results."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import method


def main():
    torch.manual_seed(48048)
    n = 48
    y = torch.arange(n) % 2
    signed = y.float() * 2 - 1
    parts = [torch.zeros(n, d) for d in method.INPUT_DIMS]
    for p in parts: p[:, 0] = signed
    base = torch.zeros(n, 320); base[:, 0] = signed
    train = list(range(n))
    e = method.DPAEvidence(fold=1, group_id=2, k=4)
    with torch.no_grad():
        e.h_project.weight.zero_(); e.h_project.bias.zero_(); e.h_project.weight[0, 0] = 1
        for layer in e.modal_embed:
            layer.weight.zero_(); layer.bias.zero_(); layer.weight[0, 0] = 1
        for layer in list(e.msg_project) + list(e.self_project):
            layer.weight.zero_(); layer.bias.zero_(); layer.weight.copy_(torch.eye(64))
    with torch.no_grad():
        _, true_msgs, _, true_meta, _ = e(parts, base, train, 0, route="true")
        _, rand_msgs, _, rand_meta, _ = e(parts, base, train, 0, route="random")
    # The first message coordinate should follow the same-label signal for a
    # true similarity graph, while the random control should be near zero.
    true_corr = float(torch.corrcoef(torch.stack([true_msgs[0][:, 0], signed]))[0, 1])
    rand_corr = float(torch.corrcoef(torch.stack([rand_msgs[0][:, 0], signed]))[0, 1])
    assert true_corr > rand_corr + 0.2, (true_corr, rand_corr)
    assert all(int(m["self_edge_count"]) == 0 for m in true_meta)
    # Artificially separable domain source control.
    d = method.DomainDiscriminator(); opt = torch.optim.Adam(d.parameters(), lr=0.03)
    z0 = torch.randn(n, 64) - 2; z1 = torch.randn(n, 64) + 2
    zz = torch.cat([z0, z1]); dd = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    for _ in range(30):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(d(zz), dd); loss.backward(); opt.step()
    domain_acc = float(d(zz).argmax(1).eq(dd).float().mean())
    assert domain_acc > 0.9
    x = torch.ones(3, requires_grad=True)
    gx = torch.autograd.grad(method.grad_reverse(x, 0.3).sum(), x)[0]
    assert torch.allclose(gx, torch.full_like(gx, -0.3))
    p = torch.nn.Parameter(torch.ones(1))
    projected, pc = method.pcgrad_pair([torch.ones_like(p)], [torch.full_like(p, -1.)], [p])
    assert pc["conflict"] == 1 and abs(float(projected[0])) < 1e-8
    result = {"true_graph_label_corr": true_corr, "random_graph_label_corr": rand_corr,
              "domain_source_accuracy": domain_acc, "grl_sign": "PASS", "pcgrad_projection": "PASS",
              "nonself_true_graph": "PASS", "status": "PASS"}
    (HERE / "results/synthetic_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
