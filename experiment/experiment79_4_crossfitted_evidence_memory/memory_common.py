from __future__ import annotations

import copy
import torch
from torch.nn import functional as F
from torch.func import functional_call


def output_layer_logp(output_layer, h):
    was_training = output_layer.training
    output_layer.eval()
    params = {n: p.detach() for n, p in output_layer.named_parameters()}
    buffers = {n: b.detach() for n, b in output_layer.named_buffers()}
    try:
        out = functional_call(output_layer, (params, buffers), (h, None))
    finally:
        output_layer.train(was_training)
    return out[0] if isinstance(out, (tuple, list)) else out


def detach_memory(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: detach_memory(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_memory(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_memory(v) for v in obj)
    return copy.deepcopy(obj)


def weighted_memory_loss(new_logp, base_logp, labels, ids, class_weights,
                         evidence_logits, evidence_valid, delta, h, coeff):
    idx = torch.as_tensor(ids, dtype=torch.long, device=new_logp.device)
    y = labels[idx].long()
    wrong = base_logp.detach().argmax(1).ne(y)
    sample_w = torch.where(wrong, coeff["error_weight"], coeff["correct_weight"]).to(new_logp)
    cls_w = class_weights.to(new_logp.device, dtype=new_logp.dtype)
    nll = F.nll_loss(new_logp, y, weight=cls_w, reduction="none")
    cls = (sample_w * nll).sum() / (sample_w * cls_w[y]).sum().clamp_min(1e-8)
    true = new_logp.gather(1, y[:, None]).squeeze(1)
    other = new_logp.gather(1, (1-y)[:, None]).squeeze(1)
    wrong_margin = F.softplus(0.2-(true-other))[wrong].mean() if bool(wrong.any()) else new_logp.new_zeros(())
    correct = ~wrong
    preserve = (new_logp[correct]-base_logp.detach()[correct]).square().sum(1).mean() if bool(correct.any()) else new_logp.new_zeros(())
    evidence = F.cross_entropy(evidence_logits[evidence_valid], y[evidence_valid]) if bool(evidence_valid.any()) else new_logp.new_zeros(())
    ratio = delta.norm(dim=1)/h.norm(dim=1).clamp_min(1e-8)
    delta_reg = ratio.square().mean()
    total = cls + .20*wrong_margin + .10*preserve + .10*evidence + .01*delta_reg
    return total, {"cls":float(cls.detach()),"wrong_margin":float(wrong_margin.detach()),
                   "correct_preserve":float(preserve.detach()),"evidence":float(evidence.detach()),
                   "delta_reg":float(delta_reg.detach()),"error_queries":int(wrong.sum()),
                   "evidence_queries":int(evidence_valid.sum()),"delta_ratio_mean":float(ratio.detach().mean())}

