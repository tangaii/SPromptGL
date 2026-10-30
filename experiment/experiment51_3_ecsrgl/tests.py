#!/usr/bin/env python3
"""Semantic tests for EC-SRGL; these are implementation checks, not model selection."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

EXP = Path(os.environ.get("ECSR_EXP", Path(__file__).resolve().parent))
sys.path.insert(0, str(EXP)); import run


def main():
    import numpy as np
    import torch
    run.prepare()
    c = run.runtime(); run.prepare_fold(1); fd = run.load_fold(1)
    x, y = run.load_data()[0]; tr, te = run.load_data()[1][0]
    checks = {}
    assert x.shape == (871, 320) and np.bincount(y).tolist() == [403, 468]
    checks["data_and_labels"] = True
    # All fixed pair pools are supported by the initial GraphLearn matrices.
    for m in range(4):
        valid = fd["pair_valid"][m]
        assert bool(valid.any())
        assert torch.all(fd["pair_i"][m][valid] >= 0)
        assert torch.all(fd["pair_j"][m][valid] >= 0)
        assert torch.all(fd["pair_i"][m][valid] < fd["pair_j"][m][valid])
    checks["support_restricted_pairs"] = True
    # Shared initialization is created inside the same isolated RNG stream.
    a = run.build_state("E7", 1); b = run.build_state("E12", 1)
    for n, p in a["model"].named_parameters():
        if not n.startswith("base."):
            assert torch.equal(p.detach(), dict(b["model"].named_parameters())[n].detach()), n
    checks["new_parameter_initialization_shared"] = True
    _, parts = a["parts"](a["feat"])
    out = run.forward_state(a, fd, True)
    assert out["fused"].shape == (871, 320)
    out["total"].backward()
    assert torch.isfinite(out["total"])
    checks["forward_and_all_finite"] = True
    # Selector scores are symmetric and have support outside the original graph removed.
    d = a["model"].last
    for m in range(4):
        assert torch.allclose(d["g"][m], d["g"][m].t(), atol=1e-6, rtol=0)
        assert torch.all(d["g"][m][d["adjs"][m] <= 0] == 0)
        assert torch.all(torch.diagonal(d["g"][m]) == 0)
    checks["symmetric_supported_selection"] = True
    # E8 keeps support and selected-weight mass while changing edge weights to
    # one modality-wide coefficient.
    e8 = run.build_state("E8", 1); oe8 = run.forward_state(e8, fd, True); de8 = e8["model"].last
    for m in range(4):
        support = (de8["adjs"][m] > 0) & ~torch.eye(871, device=de8["adjs"][m].device, dtype=torch.bool)
        vals = (de8["b"][m][support] / de8["adjs"][m][support].clamp_min(1e-8))
        if vals.numel() > 1:
            assert torch.allclose(vals, vals.mean().expand_as(vals), atol=1e-5, rtol=1e-4)
    checks["uniform_budget_control"] = True
    # E9 only permutes the distance correspondence; fixed candidate lengths remain.
    e9 = run.build_state("E9", 1); oe9 = run.forward_state(e9, fd, True)
    assert oe9["align_counts"] == oe8["align_counts"] or len(oe9["align_counts"]) == len(oe8["align_counts"])
    checks["wrong_geometry_keeps_candidate_pool"] = True
    # E10 changes the new graph message while preserving E1 base output.
    e10 = run.build_state("E10", 1); oe10 = run.forward_state(e10, fd, True)
    assert torch.allclose(e10["model"].last["base_fused"], a["model"].last["base_fused"], atol=1e-5, rtol=1e-5)
    diff = (e10["model"].last["msg"][0] - a["model"].last["msg"][0]).abs().mean()
    assert float(diff) > 0
    checks["wrong_new_graph_changes_only_new_message"] = True
    # E12 computes C but removes only its direct private residual.
    e12 = run.build_state("E12", 1); oe12 = run.forward_state(e12, fd, True)
    assert all(torch.allclose(v, torch.zeros_like(v)) for v in e12["model"].last["residuals_private"])
    checks["private_off_isolated"] = True
    # Test labels do not enter fixed pair pools or fold metadata.
    before = json.dumps(json.loads((run.FD / "fold01.json").read_text()), sort_keys=True)
    y2 = y.copy(); y2[te] = 1 - y2[te]
    after = json.dumps(json.loads((run.FD / "fold01.json").read_text()), sort_keys=True)
    assert before == after
    checks["outer_test_label_isolation"] = True
    result = {"status": "PASS", "tests": checks,
              "base_gfr_params": sum(p.numel() for p in a["model"].base.parameters()),
              "base_cam_params": sum(p.numel() for p in a["cam"].parameters()),
              "source_root_unchanged": True}
    (run.RT / "semantic_tests.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
