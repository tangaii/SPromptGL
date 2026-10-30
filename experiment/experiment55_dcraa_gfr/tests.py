#!/usr/bin/env python3
"""Focused E1 parity, RNG, adversary, graph and resume preflight checks."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("experiment55_run_tests", EXP / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def assert_close(a, b, tol=1e-6, name="value"):
    if not torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=tol, rtol=tol):
        raise AssertionError(f"{name} mismatch")


def structural():
    RUN.check_input_signatures()
    ctx = RUN.runtime()
    x, y = RUN.load_data()[0]
    assert x.shape == (871, 320) and np.bincount(y).tolist() == [403, 468]
    state_a = RUN.E1.build_state("E1", 1)
    assert state_a["model"].__class__.__name__ == "LegacyFixedGFR"
    assert state_a["cam"].__class__.__name__ == "PLNNoPEN"
    assert not any(m.__class__.__name__ == "PEN" for m in state_a["cam"].modules())
    assert all(abs(float(g["weight_decay"]) - 0.11) < 1e-12 for g in state_a["opt_model"].param_groups)
    assert all(abs(float(g["weight_decay"]) - 0.11) < 1e-12 for g in state_a["opt_cam"].param_groups)
    _, parts = state_a["parts"](state_a["feat"])
    assert [p.shape[1] for p in parts] == [48, 6, 10, 256]
    seen = []
    hooks = [m.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape))) for m in state_a["model"].gcn]
    state_a["model"].eval()
    with torch.no_grad():
        fused, _, details = state_a["model"](parts, None)
    for hook in hooks:
        hook.remove()
    assert seen == [(871, 320)] * 4, seen
    expected = torch.cat(parts, dim=1)
    for m, message in enumerate(details["msgs"]):
        expected = expected + torch.sigmoid(state_a["model"].fusion_weight[m]) * message
    assert torch.equal(fused, expected), "E1 four-message sum changed"
    source_gcn_inputs = [state_a["model"].gcn[m].gcn1.in_features for m in range(4)]
    assert source_gcn_inputs == [320] * 4

    cuda_before = torch.cuda.get_rng_state().clone()
    cpu_before = torch.get_rng_state().clone()
    dcraa, module = RUN.make_dcraa(1)
    assert torch.equal(cuda_before, torch.cuda.get_rng_state()) and torch.equal(cpu_before, torch.get_rng_state())
    pairs = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], device="cuda")
    corr = torch.as_tensor(x[:, 64:320], device="cuda")
    with torch.no_grad():
        z = dcraa.pair_representation(corr, pairs)
    assert torch.equal(z[0], z[1]) and torch.equal(z[2], z[3]), "pair relation score is not symmetric"

    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]], np.float32)
    cond = RUN.disease_conditions(probabilities, np.asarray([[0, 1], [1, 2]], np.int64))
    assert np.allclose(cond.sum(1), 1.0, atol=1e-7)
    assert np.allclose(cond[0], [0.24, 0.14, 0.62])
    assert not torch.as_tensor(cond).requires_grad

    torch.manual_seed(123)
    plain_x = torch.randn(9, 7, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0, 1])
    head = torch.nn.Linear(7, 2, bias=False)
    torch.nn.init.constant_(head.weight, 0.25)
    loss_plain = F.cross_entropy(head(plain_x), labels)
    g_plain_x, g_plain_w = torch.autograd.grad(loss_plain, (plain_x, head.weight))
    reverse_x = plain_x.detach().clone().requires_grad_(True)
    reverse_loss = F.cross_entropy(head(module.grad_reverse(reverse_x, 0.03)), labels)
    g_reverse_x, g_reverse_w = torch.autograd.grad(reverse_loss, (reverse_x, head.weight))
    assert_close(g_reverse_x, -0.03 * g_plain_x, tol=1e-7, name="GRL encoder sign")
    assert_close(g_reverse_w, g_plain_w, tol=1e-7, name="GRL discriminator direction")

    d = torch.randn(5, 256, device="cuda")
    base_corr = torch.randn(5, 256, device="cuda")
    bounded = dcraa.bounded_residual(d, base_corr)
    assert torch.isfinite(bounded).all()
    assert torch.all(bounded.norm(dim=1) <= base_corr.norm(dim=1) + 1e-5)
    base_out = torch.randn(5, 320, device="cuda")
    with torch.no_grad():
        assert torch.equal(dcraa.writeback(base_out, bounded), base_out), "alpha=0 residual must be exactly zero"
        dcraa.alpha.fill_(0.7)
        active_out = dcraa.writeback(base_out, bounded)
    assert torch.isfinite(active_out).all() and not torch.equal(active_out, base_out)
    assert torch.equal(active_out[:, :64], base_out[:, :64]), "non-CORR E1 slices changed"

    del state_a, dcraa
    torch.cuda.empty_cache()
    return {
        "input_layout": "PASS",
        "gfr_full320_four_way_sum": "PASS",
        "original_weight_decay_0_11": "PASS",
        "no_pen": "PASS",
        "symmetric_relation": "PASS",
        "disease_condition_sum": "PASS",
        "grl_sign_and_discriminator_direction": "PASS",
        "bounded_residual_and_corr_slice": "PASS",
        "new_initialization_rng_isolated": "PASS",
    }


def dropout_rng():
    a = RUN.E1.build_state("E1", 1)
    with torch.no_grad():
        first = RUN.E1.forward(a, {}, train_mode=True)["prob"].detach().cpu()
    del a
    torch.cuda.empty_cache()
    b = RUN.E1.build_state("E1", 1)
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    _model, _module = RUN.make_dcraa(1)
    assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state())
    with torch.no_grad():
        second = RUN.E1.forward(b, {}, train_mode=True)["prob"].detach().cpu()
    assert torch.equal(first, second), "E1 dropout output changed after isolated DCRAA initialization"
    return {"same_fold_first_forward_dropout_output": "EXACT", "global_cpu_cuda_rng_restored": True}


def resume_equivalence():
    torch.manual_seed(90210)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.4), torch.nn.Linear(16, 2)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        values = torch.randn(32, 8, device="cuda")
        target = torch.randint(0, 2, (32,), device="cuda")
        loss = F.cross_entropy(net(values), target)
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())
    step(model, optimizer)
    step(model, optimizer)
    checkpoint = {
        "model": RUN.state_cpu(model), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": RUN.rng_pack()
    }
    with tempfile.TemporaryDirectory(dir=RUN.RT) as temp_dir:
        path = Path(temp_dir) / "resume.pt"
        torch.save(checkpoint, path)
        loss_uninterrupted = step(model, optimizer)
        model_uninterrupted = RUN.state_cpu(model)
        resumed = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.4), torch.nn.Linear(16, 2)).cuda()
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.001, weight_decay=0.0001)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        resumed.load_state_dict(saved["model"])
        resumed_opt.load_state_dict(saved["optimizer"])
        RUN.rng_restore(saved["rng"])
        loss_resumed = step(resumed, resumed_opt)
        okay, where = RUN.tensor_state_equal(model_uninterrupted, RUN.state_cpu(resumed), atol=1e-7, rtol=1e-7)
        if abs(loss_uninterrupted - loss_resumed) > 1e-7 or not okay:
            raise AssertionError(f"resume equivalence failed: loss={loss_uninterrupted}/{loss_resumed}, tensor={where}")
    return {"two_steps_save_restore_third_step": "PASS", "third_loss": loss_resumed, "parameters": "PASS"}


def cache_audit():
    import torch

    rows = []
    x, y = RUN.load_data()[0]
    tr, te = RUN.load_data()[1][0]
    original_y = y.copy()
    arrays, meta = RUN.load_fold_cache(1)
    assert np.allclose(arrays["probabilities"][te].sum(axis=1), 1.0)
    assert np.allclose(arrays["train_condition"].sum(axis=1), 1.0)
    assert meta["wrong_disease_condition_changed_fraction"] >= 0.25
    assert meta["wrong_edge_changed_fraction"] > 0.0
    assert meta["wrong_site_source_changed_fraction"] > 0.0
    assert np.bincount(arrays["edge_recv"], minlength=871).max() <= 8
    assert np.array_equal(np.bincount(arrays["edge_send"], minlength=871), np.bincount(arrays["wrong_edge_send"], minlength=871))
    assert meta["candidate_neighbors_max"] <= 8
    assert meta["candidate_same_site_fraction"] + meta["candidate_cross_site_fraction"] == 1.0

    original_arrays = {k: v.copy() for k, v in arrays.items()}
    mutated = original_y.copy()
    mutated[te] = 1 - mutated[te]
    RUN.DATA = (x, mutated)
    rebuilt_meta = RUN.prepare_fold_cache(1, force=True)
    rebuilt, _ = RUN.load_fold_cache(1)
    RUN.DATA = (x, original_y)
    for key in original_arrays:
        if not np.array_equal(original_arrays[key], rebuilt[key]):
            raise AssertionError(f"test-label leakage changes frozen relation cache: {key}")

    cache, meta = RUN.device_cache(1)
    base_state = RUN.E1.build_state("E1", 1)
    with torch.no_grad():
        base_probability = RUN.E1.forward(base_state, {}, train_mode=True)["prob"].detach().cpu()
    del base_state
    torch.cuda.empty_cache()
    state = RUN.new_state("E5", 1)
    out = RUN.forward_group(state, cache, meta, train_mode=True)
    assert torch.equal(base_probability, out["prob"].detach().cpu()), "DCRAA call path changed E1 epoch-zero dropout outputs"
    out["total"].backward()
    rel_grad = RUN.grad_norm_module(state["dcraa"].corr)
    disc_grad = RUN.grad_norm_module(state["dcraa"].site_conditional)
    alpha_grad = float(state["dcraa"].alpha.grad.detach().abs().cpu()) if state["dcraa"].alpha.grad is not None else 0.0
    if rel_grad <= 0.0:
        raise AssertionError("CORR relation encoder has no gradient on a real fold batch")
    if meta["adversary_status"] == "IDENTIFIABLE" and disc_grad <= 0.0:
        raise AssertionError("identified conditional adversary has no discriminator gradient")
    if alpha_grad <= 0.0:
        raise AssertionError("classification does not train alpha at zero initialization")
    if float(out["residual"].detach().norm().cpu()) != 0.0:
        raise AssertionError("alpha=0 initial DCRAA residual is not zero")

    condition_true = cache["train_condition"]
    condition_wrong = cache["wrong_condition"]
    pair_z = state["dcraa"].pair_representation(state["feat"][:, 64:320], cache["train_pairs"])
    logits = state["dcraa"].conditional_site_logits(pair_z)
    loss_true = state["dcraa"].conditional_adv_loss(logits, cache["train_site"], condition_true, state["dcraa_module"].confidence_from_conditions(condition_true))
    loss_wrong = state["dcraa"].conditional_adv_loss(logits, cache["train_site"], condition_wrong, state["dcraa_module"].confidence_from_conditions(condition_wrong))
    assert abs(float((loss_true - loss_wrong).detach().cpu())) > 1e-7, "E6 condition does not change adversary input"
    logits_unconditional = state["dcraa"].unconditional_site_logits(pair_z)
    loss_unconditional = state["dcraa"].unconditional_adv_loss(logits_unconditional, cache["train_site"])
    assert abs(float((loss_true - loss_unconditional).detach().cpu())) > 1e-7, "E3 and E4 adversary objectives are not distinct"
    loss_wrong_site = state["dcraa"].conditional_adv_loss(logits, cache["wrong_site"], condition_true, state["dcraa_module"].confidence_from_conditions(condition_true))
    assert abs(float((loss_true - loss_wrong_site).detach().cpu())) > 1e-7, "E7 site supervision does not alter the adversary target"

    with torch.no_grad():
        state["dcraa"].alpha.fill_(0.05)
    out_active = RUN.forward_group(state, cache, meta, train_mode=False)
    corr_params = list(state["dcraa"].corr.parameters())
    cls_edge_grads = torch.autograd.grad(out_active["cls"], corr_params + [state["dcraa"].edge_scale], allow_unused=True)
    cls_encoder_norm = float(torch.stack([g.detach().float().square().sum() for g in cls_edge_grads[:-1] if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_edge_grads[:-1]) else 0.0
    cls_edge_norm = float(cls_edge_grads[-1].detach().abs().cpu()) if cls_edge_grads[-1] is not None else 0.0
    assert cls_encoder_norm > 0.0 and cls_edge_norm > 0.0, "classification does not reach learned relation/edge weights once alpha opens"

    state["opt_model"].zero_grad(set_to_none=True)
    state["opt_cam"].zero_grad(set_to_none=True)
    state["opt_dcraa"].zero_grad(set_to_none=True)
    state["dcr_group"] = "E2"
    out_no_adv = RUN.forward_group(state, cache, meta, train_mode=True)
    out_no_adv["total"].backward()
    assert all(param.grad is None for param in state["dcraa"].site_conditional.parameters())

    state["opt_model"].zero_grad(set_to_none=True)
    state["opt_cam"].zero_grad(set_to_none=True)
    state["opt_dcraa"].zero_grad(set_to_none=True)
    state["dcr_group"] = "E8"
    out_wrong = RUN.forward_group(state, cache, meta, train_mode=False)
    assert not torch.equal(out["edge_weights"].detach(), out_wrong["edge_weights"].detach())
    assert not torch.equal(out["delta"].detach(), out_wrong["delta"].detach()), "E8 wrong identities did not alter messages"
    state["dcr_group"] = "E9"
    out_none = RUN.forward_group(state, cache, meta, train_mode=False)
    assert torch.equal(out_none["fused"], state["model"].last["fused"])
    assert torch.count_nonzero(out_none["residual"]) == 0

    rows.append({
        "fold": 1,
        "adversary_status": meta["adversary_status"],
        "condition_changed_fraction": meta["wrong_disease_condition_changed_fraction"],
        "site_target_changed_fraction": meta["wrong_site_source_changed_fraction"],
        "wrong_edge_changed_fraction": meta["wrong_edge_changed_fraction"],
        "relation_encoder_gradient": rel_grad,
        "discriminator_gradient": disc_grad,
        "classification_alpha_gradient": alpha_grad,
        "condition_loss_difference": float((loss_true - loss_wrong).detach().cpu()),
        "unconditional_vs_conditional_loss_difference": float((loss_true - loss_unconditional).detach().cpu()),
        "wrong_site_loss_difference": float((loss_true - loss_wrong_site).detach().cpu()),
        "classification_to_relation_encoder_after_alpha_opens": cls_encoder_norm,
        "classification_to_edge_scale_after_alpha_opens": cls_edge_norm,
        "adversary_off_has_zero_discriminator_gradient": True,
        "common_epoch_zero_cam_dropout_output": "EXACT",
        "test_label_mutation": "NO_CHANGE_TO_FROZEN_CACHE",
        "wrong_graph_message": "CHANGED",
        "no_message": "EXACT_ZERO",
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("structural", "cache"), required=True)
    args = parser.parse_args()
    RUN.runtime()
    if args.stage == "structural":
        result = {"structural": structural(), "dropout_rng": dropout_rng(), "resume": resume_equivalence()}
        (RUN.RESULTS / "preflight_structural.json").write_text(json.dumps(result, indent=2))
        print("STRUCTURAL_PREFLIGHT_PASS", flush=True)
    else:
        result = cache_audit()
        (RUN.RESULTS / "preflight_cache.json").write_text(json.dumps(result, indent=2))
        print("CACHE_PREFLIGHT_PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
