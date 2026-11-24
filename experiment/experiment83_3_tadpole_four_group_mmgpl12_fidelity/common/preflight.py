from __future__ import annotations

import hashlib
import copy
import importlib.util
import inspect
import io
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from fidelity_model import (FidelityModel, HistoricalGFR, fit_spectral_statistics,
                            graph_construct_loss)
from baseline_parity import baseline_probe, four_modal_graph_probe
from fold_checkpoint import implementation_hashes, write_json
from metrics import acc_auc, one_hot
from source_runtime import BASE, DARG, E74_GROUP, source_graph_utils

DIMS = [145, 148, 3, 36, 24, 9]
NCLASS = 3
PROMPT = ROOT / "sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data():
    import pandas as pd
    frame = pd.read_csv(ROOT / "sources/processed_standard_data.csv")
    x = frame.drop(columns=["LABEL"]).to_numpy(np.float32)
    y = frame["LABEL"].to_numpy(np.int64) - 1
    splits = json.loads((ROOT / "sources/split_indices.json").read_text())['splits']
    return torch.from_numpy(x), torch.from_numpy(y), splits


def verify_locked_data(x, y, splits):
    import pandas as pd
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold

    location = ROOT / "sources"
    manifest = json.loads((location / "data_manifest.json").read_text())
    expected = json.loads((ROOT / "protocol.json").read_text())["dataset"]
    hashes = {}
    for name, definition in manifest["processed_files"].items():
        hashes[name] = sha_file(location / name)
        assert hashes[name] == definition["sha256"] == sha_file(Path(definition["path"]))
    hashes["split_indices.json"] = sha_file(location / "split_indices.json")
    expected_data_sha = expected.get("processed_sha256", expected.get("data_sha256"))
    assert hashes["processed_standard_data.csv"] == expected_data_sha
    assert hashes["split_indices.json"] == expected["split_sha256"] == sha_file(location / "split_indices.json")
    features = pd.read_csv(location / "processed_standard_data.csv", nrows=0)
    blocks = np.load(location / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(blocks) == list(manifest["modalities"]) and [len(v) for v in blocks.values()] == DIMS
    assert list(features.columns) == [c for columns in blocks.values() for c in columns] + ["LABEL"]
    ids = pd.read_csv(location / "sample_ids.csv")
    assert len(ids) == ids["RID"].nunique() == len(y) == 598
    np.testing.assert_array_equal(ids["LABEL"].to_numpy(dtype=np.int64) - 1, y.numpy())
    assert x.shape == (598, 365) and torch.isfinite(x).all()
    expected_counts = [manifest["label_distribution"][name] for name in ("NC", "MCI", "AD")]
    assert np.bincount(y.numpy(), minlength=3).tolist() == expected_counts
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    for saved, (train, test) in zip(splits, cv.split(x, y), strict=True):
        np.testing.assert_array_equal(saved["train_idx"], train)
        np.testing.assert_array_equal(saved["test_idx"], test)
        assert list(Counter(y.numpy()[train])) == [0, 1, 2]
    return {"locked_data_sha256": hashes, "matching_cv_folds": len(splits),
            "label_and_rid_order_verified": True, "feature_order_verified": True}


def _state_digest(module, prefixes):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        if any(name.startswith(prefix) for prefix in prefixes):
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _record(name, fn):
    try:
        detail = fn()
        return {"id": name, "status": "PASS", "detail": detail or {}}
    except Exception as exc:
        return {"id": name, "status": "FAIL", "detail": {
            "error": repr(exc), "traceback": traceback.format_exc(limit=8)}}


def gpu_runtime_probe(x_cpu, y_cpu, train):
    from run_e833 import (anchor_pair_support, build_model, capture_rng_state, configure_worker, historical_batch_loss, make_pairs,
                          optimizers, scmr_optimizer_step, seed_everything, sha_object, warmup_step)

    assert torch.cuda.device_count() == 1
    configure_worker()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything()
    data, labels, rows = x_cpu.to(device), y_cpu.to(device), train.to(device)
    group = "G4_E82_4_6_10_G1"
    model = build_model(group, device=device, fold=1)
    support = anchor_pair_support(model, data)
    pair_np, targets = make_pairs(train.numpy(), y_cpu.numpy(), 1, support)
    pair, same = torch.from_numpy(pair_np).to(device), torch.from_numpy(targets).to(device)
    opts, _ = optimizers(model, group)
    assert torch.isfinite(warmup_step(model, opts, data, labels, rows, pair, same))
    model.darg.enter_positive_dcr()
    model.eval()
    with torch.no_grad():
        _, warm = model(data, labels, rows, pair, same, dpsl_enabled=False)
    stats = fit_spectral_statistics(warm["cam"], labels, rows, warm_logits=warm["native_logits"])
    initial_rng = sha_object(capture_rng_state())
    from run_e833 import PROMPT12
    model.attach_dpsl(stats, PROMPT12, 82460201)
    assert initial_rng == sha_object(capture_rng_state())
    opts["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=.0005, weight_decay=.01)
    model.train()
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    _, before = model(data, labels, rows, pair, same)
    _, total = historical_batch_loss(before, labels, rows, True)
    assert torch.isfinite(total)
    budget = scmr_optimizer_step(model, opts, total, before["scmr_objectives"])
    model.eval()
    with torch.no_grad():
        best_eval, details = model(data, labels, rows, pair, same)
        assert float(details["dpsl"]["ratio"].max()) > 0
        assert float(details["dpsl"]["ratio"].max()) <= .08
        assert torch.isfinite(best_eval).all()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    model.dpsl.advance_ema(details["cam"])
    assert model.dpsl.ema_updates == 1
    checkpoint.seek(0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    assert model.dpsl.ema_updates == 0
    with torch.no_grad():
        after, _ = model(data, labels, rows, pair, same)
    torch.testing.assert_close(best_eval, after, rtol=0, atol=0)
    return {"device": torch.cuda.get_device_name(0),
            "free_mib": torch.cuda.mem_get_info(device)[0] / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "scmr_step_and_eval_seconds": seconds, "gpu_best_ema_replay_exact": True,
            "gpu_update_shared_cap": budget["shared_cap"]}


def run():
    torch.set_num_threads(1)
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    x, y, splits = load_data()
    train = torch.tensor(splits[0]["train_idx"], dtype=torch.long)
    test = torch.tensor(splits[0]["test_idx"], dtype=torch.long)
    checks = []

    def make(group, stats=None):
        prompt_path = str(ROOT / "inputs/prompts/mmGPL12_scmr.pt") if group == "G4_E82_4_6_10_G1" else str(PROMPT)
        return FidelityModel(DIMS, group, NCLASS, prompt_path, stats)

    g1 = make("G1_E28_E0")
    g2 = make("G2_E43_1_E00")
    g3 = make("G3_E74_4_G2")
    g4base = make("G4_E82_4_6_10_G1")

    def g1_probe():
        g1.eval()
        x_small, y_small, rows = x[:16], y[:16], torch.arange(10)
        logp, info = g1(x_small, y_small, rows)
        parts, offset = [], 0
        for width in DIMS:
            parts.append(x_small[:, offset:offset + width]); offset += width
        fused, gaux = g1.gfr(parts, y_small, rows)
        manual = gaux["raw"]
        for modality, message in enumerate(gaux["messages"]):
            value = torch.sigmoid(g1.gfr.fusion_weight[modality]) * message
            manual = value + gaux["raw"] if modality in (0, 1) else value + manual
        assert torch.allclose(fused, manual, atol=1e-6, rtol=1e-6)
        tokens, _ = g1.cam.InputLayer(fused)
        for encoder, feed_forward in zip(g1.cam.Encoder, g1.cam.FeedForward):
            tokens, _ = encoder(tokens, tokens, tokens, g1.cam.modal_num)
            tokens = feed_forward(tokens)
        pen = g1.cam.PEN
        n, modalities, width = tokens.shape
        hidden = width // 4
        local = tokens.view(n, modalities, 4, hidden)
        text = torch.cat((pen.prompt_text_features[:4].reshape(1, -1),
                          pen.prompt_text_features[4:].reshape(1, -1)), dim=0)
        text = pen.MLP(text)
        local_norm = F.normalize(local, p=2, dim=-1)
        text_norm = F.normalize(text, p=2, dim=-1)
        similarity = torch.matmul(local_norm.unsqueeze(-2), text_norm.T.view(1, 1, 1, hidden, 2))
        similarity = similarity[..., -1, :].clamp(-5, 5)
        weights = torch.softmax(similarity, dim=-1).amax(dim=-1)
        local = weights.unsqueeze(-1) * local
        pooled = F.avg_pool1d(tokens.reshape(n * modalities, 1, width), kernel_size=4, stride=4)
        pooled = pooled.view(n, modalities, hidden)
        pieces = []
        for head in range(4):
            part = local[:, :, head, :]
            pieces.append(F.softmax(part * pooled, dim=-1) * part)
        pen_reference = tokens + torch.cat(pieces, dim=2)
        assert torch.allclose(info["tokens"], pen_reference, atol=1e-6, rtol=1e-6)
        assert tuple(info["tokens"].shape) == (16, 6, 40)
        assert tuple(info["pen_weights"].shape) == (16, 6, 4)
        parameter_names = [name for name, _ in g1.cam.named_parameters()]
        assert parameter_names.index("Outputlayer.mlp_head.1.weight") < parameter_names.index("PEN.MLP.0.weight")
        linear_weights = []
        for module in g1.cam.modules():
            if isinstance(module, torch.nn.Linear):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                assert float(module.weight.detach().abs().max()) <= limit + 1e-7
                linear_weights.append(module.weight)
        loss = logp[rows].mean() + info["graph_loss"]
        selected = [g1.gfr.graphs[0].p.weight, g1.cam.InputLayer.w_qs[0].weight, g1.cam.PEN.MLP[0].weight]
        gradients = torch.autograd.grad(loss, selected, allow_unused=True)
        assert all(value is not None and torch.isfinite(value).all() and value.norm() > 0 for value in gradients)
        parity = baseline_probe("G1_E28_E0", x_small, y_small, DIMS, PROMPT)
        return {"cam_dim": int(info["cam"].shape[1]), "pen": bool(info["pen_present"]), "archive_parity": parity,
                "layers": [type(g1.cam.InputLayer).__name__, type(g1.cam.Encoder[0]).__name__,
                           type(g1.cam.FeedForward[0]).__name__, type(g1.cam.Outputlayer).__name__],
                "gfr_fusion_max_abs": float((fused - manual).abs().max()),
                "pen_reference_max_abs": float((info["tokens"] - pen_reference).abs().max()),
                "historical_registration_order": True,
                "xavier_linear_count": len(linear_weights),
                "gradient_norms": [float(value.norm()) for value in gradients],
                "token_shape": list(info["tokens"].shape)}
    checks.append(_record("TEST_01_G1_original_gfr_cam_pen_graph", g1_probe))
    def g2_probe():
        assert not g2.cam.use_pen and not hasattr(g2.cam, "PEN")
        assert all("PEN" not in name for name, _ in g2.named_parameters())
        return {"pen": False, "pen_parameters": 0,
                "archive_parity": baseline_probe("G2_E43_1_E00", x[:16], y[:16], DIMS, PROMPT)}
    checks.append(_record("TEST_02_G2_physical_no_pen", g2_probe))
    checks.append(_record("TEST_03_G3_physical_no_pen", lambda: {
        "pen": g3.cam.use_pen, "pen_parameters": sum(p.numel() for n, p in g3.named_parameters() if "PEN" in n)
    } if not g3.cam.use_pen and not any("PEN" in n for n, _ in g3.named_parameters()) else (_ for _ in ()).throw(AssertionError("PEN present"))))
    checks.append(_record("TEST_04_G4_physical_no_pen", lambda: {
        "pen": g4base.cam.use_pen, "pen_parameters": sum(p.numel() for n, p in g4base.named_parameters() if "PEN" in n)
    } if not g4base.cam.use_pen and not any("PEN" in n for n, _ in g4base.named_parameters()) else (_ for _ in ()).throw(AssertionError("PEN present"))))

    out1, aux1 = g1(x[:16], y[:16], torch.arange(10))
    def aloss_probe():
        from run_e833 import historical_batch_loss, weighted_nll

        rows = torch.arange(10)
        _, formal = historical_batch_loss(aux1, y[:16], rows, use_darg=False)
        expected = weighted_nll(out1, y[:16], rows) + aux1["graph_loss"]
        torch.testing.assert_close(formal, expected, rtol=0, atol=0)
        _, different_aloss = historical_batch_loss({**aux1, "aloss": aux1["aloss"] * 10}, y[:16], rows, use_darg=False)
        torch.testing.assert_close(formal, different_aloss, rtol=0, atol=0)
        parameter = g1.gfr.graphs[0].p.weight
        grad_formal, = torch.autograd.grad(formal, parameter, retain_graph=True, allow_unused=False)
        grad_expected, = torch.autograd.grad(expected, parameter, retain_graph=True, allow_unused=False)
        grad_with_aloss, = torch.autograd.grad(formal + aux1["aloss"], parameter, retain_graph=True, allow_unused=False)
        aloss_grad = grad_with_aloss - grad_formal
        torch.testing.assert_close(grad_formal, grad_expected, rtol=0, atol=0)
        assert aloss_grad.norm() > 0
        return {"formal_loss": float(formal.detach()), "aloss": float(aux1["aloss"].detach()),
                "formal_gradient_norm": float(grad_formal.norm()), "aloss_gradient_norm": float(aloss_grad.norm()),
                "source_contract": "ALoss is returned for observation and excluded from cls+graph"}
    checks.append(_record("TEST_05_aloss_observation_only", aloss_probe))
    def graph_probe():
        feat = torch.randn(5, 4)
        adj = (torch.rand(5, 5) + torch.eye(5)).requires_grad_()
        theta_smooth, theta_degree, theta_sparsity = 1.3, 0.7, 0.2
        source = BASE / "utils.py"
        historical = source_graph_utils(source).GraphConstructLoss
        expected = historical(feat, adj, theta_smooth, theta_degree, theta_sparsity)
        observed = graph_construct_loss(feat, adj, theta_smooth, theta_degree, theta_sparsity)
        grad_expected, = torch.autograd.grad(expected, adj)
        grad_observed, = torch.autograd.grad(observed, adj)
        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close(grad_observed, grad_expected)
        return {"max_abs": float((observed - expected).abs().detach()),
                "gradient_max_abs": float((grad_observed - grad_expected).abs().max()),
                "historical_source_sha256": sha_file(source)}
    checks.append(_record("TEST_06_graph_construct_loss_exact", graph_probe))
    def overwrite_probe():
        source = inspect.getsource(HistoricalGFR.forward)
        assert "elif i == 1" in source
        parts = [x[:12] for x in torch.split(x[:12], DIMS, dim=1)]
        fused, aux = g1.gfr(parts, y[:12], torch.arange(8))
        raw = aux["raw"]
        manual = raw
        for i, message in enumerate(aux["messages"]):
            value = torch.sigmoid(g1.gfr.fusion_weight[i]) * message
            manual = value + raw if i in (0, 1) else value + manual
        assert torch.allclose(fused, manual, atol=1e-6, rtol=1e-6)
        assert torch.allclose(aux["messages"][1] * torch.sigmoid(g1.gfr.fusion_weight[1]) + raw,
                              fused - sum((torch.sigmoid(g1.gfr.fusion_weight[j]) * aux["messages"][j]
                                           for j in range(2, 6)), torch.zeros_like(raw)), atol=1e-6, rtol=1e-6)
        return {"source_has_overwrite": True, "six_modality_extension": True,
                **four_modal_graph_probe(x[:12], y[:12], DIMS[:4], PROMPT),
                "fused_reference_max_abs": float((fused - manual).abs().max()),
                "message_count": len(aux["messages"])}
    checks.append(_record("TEST_07_gfr_second_modality_historical_overwrite", overwrite_probe))

    def signed_probe():
        base = torch.tensor([[1., .8, .4, .3], [.8, 1., .6, .3], [.4, .6, 1., .3], [.3, .3, .3, 1.]])
        row, col = torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3])
        gate = .01 * g3.gfr.adaptive_gate_logits[0].sigmoid()
        q = torch.tensor([.8, .2, .5])
        unified, info = g3.darg.positive.unified_adjacency(base, row, col, q, gate, 0)
        expected_delta = base[row, col] * .8 * gate * torch.tensor([.6, -.6, 0.])
        expected = base.clone()
        expected[row, col] += expected_delta
        expected[col, row] += expected_delta
        torch.testing.assert_close(unified, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(info["delta_values"], expected_delta, rtol=0, atol=1e-7)
        assert unified[0, 1] > base[0, 1] and unified[0, 2] < base[0, 2]
        assert unified[0, 3] == base[0, 3]
        return {"positive_delta": float(info["delta_values"][0].detach()),
                "negative_delta": float(info["delta_values"][1].detach()),
                "uncertain_delta": float(info["delta_values"][2].detach()),
                "source_sha256": sha_file(DARG / "soft_positive_dcr.py"),
                "unified_reference_max_abs": float((unified - expected).abs().detach().max())}
    checks.append(_record("TEST_08_darg_signed_unified_graph", signed_probe))

    class FixedGraph(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", value)

        def forward(self, features):
            assert self.value.shape == (len(features), len(features))
            return self.value.to(features)

    @torch.no_grad()
    def constructed_darg():
        model = copy.deepcopy(g3.darg).eval()
        size = 12
        adjacency = torch.full((size, size), .5)
        adjacency.fill_diagonal_(.1)
        model.core.graphs = torch.nn.ModuleList([FixedGraph(adjacency) for _ in DIMS])
        for scorer, bias in zip(model.core.scorers, (5., -5., 0., 5., -5., 0.)):
            scorer.net[-1].weight.zero_()
            scorer.net[-1].bias.fill_(bias)
        parts = list(x[:size].split(DIMS, dim=1))
        return model, parts

    @torch.no_grad()
    def positive_direct_probe():
        darg, parts = constructed_darg()
        _, aux = darg(parts)
        raw = torch.cat(parts, dim=1)
        count_positive = count_negative = 0
        for modality, (base, signed, direct, residual, bounded, message) in enumerate(zip(
                aux["adjacencies"], aux["delta_adjacencies"], aux["direct_delta_adjacencies"],
                aux["disease_residuals_raw"], aux["bounded_residuals"], aux["direct_messages"])):
            count_positive += int(torch.count_nonzero(signed > 0))
            count_negative += int(torch.count_nonzero(signed < 0))
            torch.testing.assert_close(direct, signed.clamp_min(0), rtol=0, atol=0)
            degrees = (base.sum(dim=1) - base.diag()).clamp_min(1).unsqueeze(1)
            expected_residual = (direct @ raw - direct.sum(dim=1, keepdim=True) * raw) / degrees
            torch.testing.assert_close(residual, expected_residual)
            start = sum(DIMS[:modality])
            stop = start + DIMS[modality]
            block = raw[:, start:stop]
            rms = block.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
            expected_bounded = torch.zeros_like(raw)
            expected_bounded[:, start:stop] = rms * torch.tanh(expected_residual[:, start:stop] / (rms + 1e-8))
            torch.testing.assert_close(bounded, expected_bounded)
            torch.testing.assert_close(message, .02 * aux["gates"][modality] * expected_bounded)
        assert count_positive > 0 and count_negative > 0
        assert torch.count_nonzero(torch.cat(aux["direct_messages"], dim=1)) > 0
        return {"positive_signed_entries": count_positive, "negative_signed_entries": count_negative,
                "signed_graph_and_bounded_local_residual_verified": True}
    checks.append(_record("TEST_09_darg_direct_positive_only", positive_direct_probe))

    @torch.no_grad()
    def direct_off_probe():
        darg, parts = constructed_darg()
        fused_on, on = darg(parts)
        fused_off, off = darg(parts, intervention="direct_off")
        fused_neutral, neutral = darg(parts, intervention="disease_neutral")
        for a, b in zip(on["unified_adjacencies"], off["unified_adjacencies"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(on["messages"], off["messages"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for original, actual in zip(on["adjacencies"], neutral["unified_adjacencies"]):
            torch.testing.assert_close(original, actual, rtol=0, atol=0)
        direct_on = torch.cat(on["direct_messages"], dim=1)
        direct_off = torch.cat(off["direct_messages"], dim=1)
        direct_response = (fused_on - fused_off).norm()
        graph_response = (fused_off - fused_neutral).norm()
        assert direct_on.norm() > 0 and direct_off.norm() == 0
        assert direct_response > 0 and graph_response > 0
        return {"unified_graph_max_abs": 0., "gcn_message_max_abs": 0.,
                "direct_output_change": float(direct_response), "graph_only_output_change": float(graph_response)}
    checks.append(_record("TEST_10_direct_off_keeps_unified_graph_gcn", direct_off_probe))

    @torch.no_grad()
    def relation_probe():
        darg, parts = constructed_darg()
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        same = torch.tensor([1., 1., 0., 0.])
        fused, result = darg(parts, pairs, same)
        source = Path(inspect.getsourcefile(darg.positive.rudag_forward)).resolve()
        assert source == DARG / "soft_positive_dcr.py"
        reference, _, _ = darg.positive.rudag_forward(darg.core, parts, pairs, E74_GROUP)
        torch.testing.assert_close(fused, reference, rtol=0, atol=0)
        expected_sum = torch.cat(parts, dim=1)
        for modality, original in enumerate(result["adjacencies"]):
            row, col = result["edge_rows"][modality], result["edge_cols"][modality]
            selected = torch.zeros_like(original, dtype=torch.bool)
            selected[row, col] = True
            selected[col, row] = True
            support = original > 0
            support.fill_diagonal_(False)
            assert row.numel() > 0 and int(selected.sum(1).max()) <= 12
            assert torch.equal(selected, selected.T) and not bool((selected & ~support).any())
            embedded = darg.core.encoders[modality](parts[modality])
            left, right = embedded[pairs[:, 0]], embedded[pairs[:, 1]]
            expected_repr = torch.cat(((left - right).abs(), left * right,
                                       F.cosine_similarity(left, right, dim=-1, eps=1e-8).unsqueeze(-1)), dim=1)
            torch.testing.assert_close(result["pair_reprs"][modality], expected_repr)
            torch.testing.assert_close(result["edge_q"][modality], result["edge_logits"][modality].sigmoid())
            torch.testing.assert_close(result["edge_q"][modality], result["edge_q_graph"][modality], rtol=0, atol=0)
            torch.testing.assert_close(result["gates"][modality], .01 * darg.core.adaptive_gate_logits[modality].sigmoid())
            expected_sum = expected_sum + darg.core.fusion_weight[modality].sigmoid() * (
                result["messages"][modality] + result["direct_messages"][modality])
        torch.testing.assert_close(fused, expected_sum, rtol=0, atol=0)
        expected_pair = torch.stack([F.binary_cross_entropy_with_logits(logits, same)
                                     for logits in result["pair_logits"]]).mean()
        torch.testing.assert_close(result["pair_loss"], expected_pair, rtol=0, atol=0)
        assert torch.isfinite(result["adv_loss"]) and torch.isfinite(result["preserve_loss"])
        return {"source_sha256": sha_file(source), "relation_repr_dim": 65,
                "candidate_rule": "non-CORR original-support mutual top-12",
                "relation_loss_terms_present": 3, "historical_source_call_matches_forward": True}
    checks.append(_record("TEST_11_darg_relation_encoder_candidate_gate", relation_probe))

    # Warmup-like CAM statistics for all spectral checks.
    g4base.eval()
    with torch.no_grad():
        _, warm_aux = g4base(x, y, train)
    stats = fit_spectral_statistics(warm_aux["cam"], y, train, warm_logits=warm_aux["native_logits"])
    g4 = make("G4_E82_4_6_10_G1", stats).eval()
    with torch.no_grad():
        _, g4aux = g4(x[:16], y[:16], torch.arange(10), dpsl_enabled=True)
    def spectrum_probe():
        basis = stats["basis"]
        assert tuple(basis.shape) == (240, 16)
        assert torch.allclose(basis.T @ basis, torch.eye(16, dtype=basis.dtype), atol=1e-5, rtol=1e-5)
        assert torch.equal(stats["train_rows"].long(), train)
        generator = torch.Generator(device="cpu").manual_seed(0)
        binary_cam = torch.randn(120, 144, generator=generator, dtype=torch.float64)
        binary_labels = torch.arange(120, dtype=torch.long).remainder(2)
        binary_rows = torch.arange(len(binary_labels))
        adapted_binary = fit_spectral_statistics(binary_cam, binary_labels, binary_rows)
        source_path = ROOT / "sources/E82_dependencies/psce.py"
        spec = importlib.util.spec_from_file_location("e82_psce_spectrum_reference", source_path)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        historical = source.build_spectra(binary_cam, binary_labels)
        basis_error = (adapted_binary["basis"].double() - historical["U"]).abs().max()
        center_error = (adapted_binary["center"].double() - historical["mu"]).abs().max()
        variance_error = (adapted_binary["variance"].double() - historical["lambda"]).abs().max()
        disease_error = (adapted_binary["disease_covariance"][0].double()
                         - historical["nu_normalized"]).abs().max()
        assert basis_error < 1e-8 and center_error < 1e-8
        assert variance_error < 1e-8 and disease_error < 1e-8
        return {"basis_shape": list(basis.shape), "source": stats["source"],
                "train_rows": int(len(stats["train_rows"])), "binary_limit_source_sha256": sha_file(source_path),
                "binary_parity_max_abs": {"basis": float(basis_error), "center": float(center_error),
                                          "variance": float(variance_error), "disease_stat": float(disease_error)},
                "binary_historical_limit_pass": True, "random_basis": False}
    checks.append(_record("TEST_12_spectral_basis_train_statistics_not_random_qr", spectrum_probe))
    def branch_probe():
        from fidelity_model import OutputLayer
        from scmr_optimization import HISTORICAL_ROOT, load_scmr_optimization
        from scmr_runtime import make_scmr, scmr_components, source_config

        paths = {"g2.py": type(g4.dpsl), "common/spectral.py": type(g4.dpsl.spectrum),
                 "common/evidence.py": type(g4.dpsl.statistics)}
        for name, component in paths.items():
            assert Path(inspect.getsourcefile(component)).resolve() == HISTORICAL_ROOT / name
        assert g4.dpsl.spectrum.suppression_logit is None
        n = source_config()["subjects"]
        cam = torch.randn(n, 144, generator=torch.Generator().manual_seed(0))
        labels = torch.arange(n).remainder(2)
        rows = torch.arange(n - 80)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            head = OutputLayer(144, 36, 2, 0.).eval()
        test_stats = fit_spectral_statistics(cam, labels, rows, warm_logits=head.raw(cam).detach())
        original = {"U": test_stats["basis"], "mu": test_stats["center"], "lambda": test_stats["variance"],
                    "loglambda_normalized": test_stats["log_variance"],
                    "nu_normalized": test_stats["disease_covariance"][0]}
        canonical, _, evidence = scmr_components(n, 144, adapted=False)
        text = torch.load(PROMPT, map_location="cpu", weights_only=True)
        ref = test_stats["reference"]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            archived = canonical(original, text, [], ref)
        _, oof = evidence.gaussian(ref["h"], ref["mean"][ref["group"]], ref["stable"][ref["group"]])
        skill = .5 * (oof[labels[rows] == 0].div(2).tanh().mean(0)
                      + (-oof[labels[rows] == 1]).div(2).tanh().mean(0))
        torch.testing.assert_close(ref["calibration"], .5 * (1 + skill), rtol=0, atol=0)
        adapted = make_scmr(test_stats, str(PROMPT), 2, 0)
        interventions = ("full", "anonymous_query", "residual_off", "spectral_only", "complement_only",
                         "calibrated_evidence_off", "evidence_conflict_off")
        names = ("white", "tokens", "attention", "coverage", "raw_llr", "llr", "reliability",
                 "disagreement_axis", "jacobian", "gate", "shift", "delta", "ratio")
        with torch.no_grad():
            for model in (archived, adapted):
                model.calibrator[-1].bias[0] = .125
            for training in (False, True):
                for intervention in interventions:
                    expected, historical = archived(cam, head, training, intervention)
                    actual, details = adapted(cam, head, training, intervention)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for name in names:
                        torch.testing.assert_close(details[name], historical[name], rtol=0, atol=0)
                    assert (details["delta"].norm() > 0) == (intervention != "residual_off")
        expected, historical = archived(cam, head, True)
        actual, details = adapted(cam, head, True)
        reference_capture = {"cam": cam, "enhanced": expected, "details": historical}
        capture = {"cam": cam, "enhanced": actual, "details": details}
        opt = load_scmr_optimization()
        expected_losses, _ = opt.losses("G2", head, reference_capture, rows, labels[rows], original["U"].float(),
                                       test_stats["hard_initial"][rows])
        actual_losses, _ = adapted.historical_objectives(head, capture, labels, rows)
        for name, value in expected_losses.items():
            torch.testing.assert_close(value, actual_losses[name], rtol=0, atol=0)
        archived.statistics.advance(archived.statistics.observation(cam + .1))
        adapted.advance_ema(cam + .1)
        for name, value in archived.statistics.snapshot().items():
            torch.testing.assert_close(adapted.statistics.snapshot()[name], value, rtol=0, atol=0)
        np.testing.assert_array_equal(g4.dpsl.statistics.counts[0].numpy(), np.bincount(y[train], minlength=3))
        return {"binary_tensor_parity": "exact", "binary_training_objective_parity": "exact",
                "binary_ema_parity": "exact", "binary_interventions": list(interventions),
                "runtime_definitions": {name: sha_file(HISTORICAL_ROOT / name) for name in paths},
                "three_class_bank_width": int(g4.dpsl.statistics.counts.shape[1])}
    checks.append(_record("TEST_13_scmr_branch_not_pcsr", branch_probe))

    def three_class_prompt_route_probe():
        from scmr_runtime import class_pair

        native = torch.tensor([[3., 2., 1.], [1., 3., 2.], [2., 1., 3.]])
        pair = class_pair(native)
        low = torch.minimum(pair[:, 0], pair[:, 1])
        high = torch.maximum(pair[:, 0], pair[:, 1])
        attention = torch.zeros(3, 12, 4)
        for row in range(3):
            for group in range(3):
                attention[row, 4 * group:4 * group + 4] = 0.1 * (group + 1)
        groups = torch.stack((1 - (1 - attention[:, :4]).prod(1),
                              1 - (1 - attention[:, 4:8]).prod(1),
                              1 - (1 - attention[:, 8:12]).prod(1)), 1)
        normal = groups.gather(1, low[:, None, None].expand(-1, 1, groups.shape[-1])).squeeze(1)
        abnormal = groups.gather(1, high[:, None, None].expand(-1, 1, groups.shape[-1])).squeeze(1)
        expected_pairs = [[0, 1], [1, 2], [0, 2]]
        assert pair.tolist() == [[1, 0], [2, 1], [2, 0]]
        assert torch.all(abnormal > normal)
        assert torch.equal(torch.stack((low, high), 1), torch.tensor(expected_pairs))
        assert torch.all((abnormal - normal) > 0)
        return {"native_top2_descending": pair.tolist(), "routed_low_high": expected_pairs,
                "prompt_groups": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                "signed_margin_direction": "higher disease stage minus lower disease stage"}
    checks.append(_record("TEST_13B_three_class_prompt_route_direction", three_class_prompt_route_probe))
    def spectral_shape_probe():
        token_shape = tuple(g4aux["dpsl"]["tokens"].shape)
        assert tuple(g4aux["cam"].shape[1:]) == (240,)
        assert tuple(stats["basis"].shape) == (240, 16)
        assert token_shape[1:] == (16, 32)
        return {"cam_dim": int(g4aux["cam"].shape[1]), "basis": list(stats["basis"].shape),
                "token_shape": list(token_shape), "observed_spectral_tokens": int(token_shape[1])}
    checks.append(_record("TEST_14_spectral_16_tokens_cam240", spectral_shape_probe))
    def residual_probe():
        probe = copy.deepcopy(g4)
        probe.train()
        optimizer = torch.optim.AdamW(probe.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
        optimizer.zero_grad(set_to_none=True)
        step_logp, step_aux = probe(x, y, train, dpsl_enabled=True)
        step_loss = step_aux["dpsl_loss"]
        assert torch.isfinite(step_loss)
        step_loss.backward()
        assert any(parameter.grad is not None and parameter.grad.norm() > 0
                   for parameter in probe.dpsl.parameters())
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            _, updated = probe(x[:16], y[:16], torch.arange(10), dpsl_enabled=True)
        residual = updated["dpsl"]["residual"]
        ratio = updated["dpsl"]["residual_ratio"]
        cap = .08 * updated["cam"].norm(dim=-1).clamp_min(1e-8)
        assert torch.all(residual.norm(dim=-1) <= cap + 1e-5)
        assert torch.allclose(ratio, residual.norm(dim=-1) / updated["cam"].norm(dim=-1).clamp_min(1e-8), atol=1e-6)
        assert torch.count_nonzero(residual) > 0
        assert torch.count_nonzero(updated["logits"] - updated["native_logits"]) > 0
        return {"max_ratio": float(ratio.max()), "cap": .08, "nonzero_rows": int(torch.count_nonzero(ratio > 0)),
                "preflight_optimizer_step_loss": float(step_loss.detach()),
                "output_change_norm": float((updated["enhanced_cam"] - updated["cam"]).norm()),
                "classification_response_norm": float((updated["logits"] - updated["native_logits"]).norm())}
    checks.append(_record("TEST_15_dpsl_residual_8_percent_cap", residual_probe))

    def gradient_budget_probe():
        import run_e833
        from scmr_optimization import HISTORICAL_ROOT, load_scmr_optimization

        selected = torch.cat([train[y[train] == label][:4] for label in range(NCLASS)])
        x_small, y_small = x[selected], y[selected]
        rows = torch.arange(len(selected))
        pair_index = torch.tensor([[0, 1], [4, 5], [8, 9], [0, 4], [0, 8], [4, 8]])
        pair_same = (y_small[pair_index[:, 0]] == y_small[pair_index[:, 1]]).float()
        local_stats = dict(stats, bank_index=stats["bank_index"][selected],
                           hard_initial=stats["hard_initial"][selected])
        model = make("G4_E82_4_6_10_G1", local_stats).train()
        reference = make("G3_E74_4_G2").train()
        reference.load_state_dict({key: value for key, value in model.state_dict().items()
                                   if not key.startswith("dpsl.")})
        model.darg.enter_positive_dcr()
        reference.darg.enter_positive_dcr()
        opts, _ = run_e833.optimizers(model, run_e833.GROUPS[3])
        reference_opts, _ = run_e833.optimizers(reference, run_e833.GROUPS[2])
        historical_api = load_scmr_optimization()
        expected_source = HISTORICAL_ROOT / "common/optimization.py"
        assert Path(inspect.getsourcefile(historical_api.apply_budget)).resolve() == expected_source
        measure = inspect.unwrap(historical_api.optimizer_measurements)
        assert Path(inspect.getsourcefile(measure)).resolve() == expected_source

        def forward_objectives():
            _, aux = model(x_small, y_small, rows, pair_index, pair_same, dpsl_enabled=True)
            _, base = run_e833.historical_batch_loss(aux, y_small, rows, model.use_darg)
            return aux, base, aux["scmr_objectives"]

        rng_before_forward = torch.get_rng_state()
        aux, base, objectives = forward_objectives()
        rng_after_forward = torch.get_rng_state()
        assert torch.equal(aux["native_logits"], aux["logits"])
        initial = run_e833.scmr_optimizer_step(model, opts, base, objectives)
        assert torch.equal(torch.get_rng_state(), rng_after_forward)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng_before_forward)
            _, reference_aux = reference(x_small, y_small, rows, pair_index, pair_same, dpsl_enabled=False)
            assert torch.equal(torch.get_rng_state(), rng_after_forward)
        _, reference_loss = run_e833.historical_batch_loss(reference_aux, y_small, rows, reference.use_darg)
        reference_loss.backward()
        run_e833.step_optimizers(reference_opts)
        assert initial["shared_cap"] == 0.001
        shared = ("gfr", "darg", "cam", "head")
        assert all(initial["families"][name]["raw"] == initial["families"][name]["total_applied"] == 0
                   for name in shared)
        assert _state_digest(model, ("gfr.", "darg.", "cam.")) == _state_digest(reference, ("gfr.", "darg.", "cam."))
        assert all(run_e833.sha_object(opts[name].state_dict()) == run_e833.sha_object(other.state_dict())
                   for name, other in reference_opts.items())

        run_e833.zero_optimizers(opts)
        aux, base, objectives = forward_objectives()
        assert torch.count_nonzero(aux["logits"] - aux["native_logits"])
        families = run_e833.scmr_parameter_families(model, opts)
        shared_parameters = tuple(param for name in shared for param in families[name])
        active = tuple(param for param in shared_parameters if param.requires_grad)

        def shared_gradients(loss):
            gradients = iter(torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True))
            return tuple(next(gradients) if param.requires_grad else None for param in shared_parameters)

        base_grads = shared_gradients(base)
        difference_grads = shared_gradients(objectives["difference"])

        def flatten(params, values):
            return torch.cat([(p.new_zeros(p.numel()) if v is None else v.detach().flatten()).double()
                              for p, v in zip(params, values)])

        vectors = {}
        offset = 0
        for name in shared:
            params = families[name]
            stop = offset + len(params)
            vectors[name] = (flatten(params, base_grads[offset:stop]),
                             flatten(params, difference_grads[offset:stop]), flatten(params, params))
            offset = stop
        budget = run_e833.scmr_optimizer_step(model, opts, base, objectives)
        assert set(budget["families"]) == {*shared, "prompt", "control"}
        observed_active = []
        for name, (b, q, previous) in vectors.items():
            params = families[name]
            written = flatten(params, (p.grad for p in params))
            extra = written - b
            update = flatten(params, params) - previous
            norm = b.norm()
            anchor = b / norm.clamp_min(1e-12)
            tangent = q - (q @ anchor) * anchor
            record = budget["families"][name]
            assert torch.isfinite(extra).all() and torch.isfinite(update).all()
            torch.testing.assert_close(norm.item(), record["historical"], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(q.norm().item(), record["raw"], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(tangent.norm().item(), record["projected"], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(extra.norm().item(), record["total_applied"], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(update.norm().item(), record["update_norm"], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close((update @ extra).item(), record["update_dot_extra"], rtol=1e-10, atol=1e-12)
            assert extra.norm() <= 0.001 * norm
            assert (extra @ anchor).abs() <= record["dot_bound"] + 1e-12
            if extra.norm() > 0:
                observed_active.append(name)
                assert update.norm() > 0
                assert (extra @ tangent) > 0
        assert observed_active, budget
        return {"historical_optimization_sha256": sha_file(expected_source),
                "zero_residual_shared_parameters_and_optimizer_state_exact": True,
                "nonzero_differential_families": observed_active, **budget}
    checks.append(_record("TEST_16_dpsl_shared_gradient_budget", gradient_budget_probe))

    def zero_dpsl_probe():
        torch.manual_seed(9); a = make("G3_E74_4_G2"); torch.manual_seed(9); b = make("G4_E82_4_6_10_G1", stats)
        b.gfr.load_state_dict(a.gfr.state_dict()); b.cam.load_state_dict(a.cam.state_dict());
        b.darg.load_state_dict(a.darg.state_dict(), strict=False)
        a.eval(); b.eval()
        with torch.no_grad():
            la, _ = a(x[:16], y[:16], torch.arange(10), dpsl_enabled=False)
            lb, _ = b(x[:16], y[:16], torch.arange(10), dpsl_enabled=False)
        assert torch.equal(la, lb)
        return {"max_abs": 0.0}
    checks.append(_record("TEST_17_dpsl_off_restores_darg_output", zero_dpsl_probe))
    def shared_rng_probe():
        import run_e833

        selected = []
        y_np = y.numpy()
        train_np = train.numpy()
        for class_id in range(NCLASS):
            class_rows = train_np[y_np[train_np] == class_id][:4]
            assert len(class_rows) == 4
            selected.extend(class_rows.tolist())
        selected_t = torch.tensor(selected, dtype=torch.long)
        x_small, y_small = x[selected_t], y[selected_t]
        rows = torch.arange(len(selected))
        pair_index = torch.tensor([[0, 1], [4, 5], [8, 9], [0, 4], [0, 8], [4, 8]])
        pair_same = (y_small[pair_index[:, 0]] == y_small[pair_index[:, 1]]).float()

        def construct(group):
            run_e833.seed_everything()
            before = run_e833.capture_rng_state()
            model = run_e833.build_model(group, device=x.device, fold=1)
            run_e833.anchor_pair_support(model, x_small)
            opts, config = run_e833.optimizers(model, group)
            after = run_e833.capture_rng_state()
            loss = run_e833.warmup_step(model, opts, x_small, y_small, rows, pair_index, pair_same)
            assert torch.isfinite(loss)
            assert len(opts["darg"].state) > 0
            assert model.gfr.adaptive_gate_logits not in opts["darg"].state
            return model, opts, config, before, after, run_e833.capture_rng_state()

        a, opts_a, cfg_a, before_a, after_a, final_a = construct("G3_E74_4_G2")
        b, opts_b, cfg_b, before_b, after_b, final_b = construct("G4_E82_4_6_10_G1")
        shared_prefixes = ("gfr.", "cam.", "darg.")
        assert _state_digest(a, shared_prefixes) == _state_digest(b, shared_prefixes)
        assert run_e833.sha_object(opts_a["base"].state_dict()) == run_e833.sha_object(opts_b["base"].state_dict())
        assert run_e833.sha_object(opts_a["cam"].state_dict()) == run_e833.sha_object(opts_b["cam"].state_dict())
        assert run_e833.sha_object(opts_a["darg"].state_dict()) == run_e833.sha_object(opts_b["darg"].state_dict())
        assert cfg_a == cfg_b
        assert run_e833.sha_object(before_a) == run_e833.sha_object(before_b)
        assert run_e833.sha_object(after_a) == run_e833.sha_object(after_b)
        assert run_e833.sha_object(final_a) == run_e833.sha_object(final_b)
        return {"shared_parameter_digest": _state_digest(a, shared_prefixes),
                "rng_before_equal": True, "rng_after_initialization_equal": True,
                "rng_after_update_equal": True, "darg_included": True,
                "one_step_model_and_optimizer_parity": True,
                "optimizer_contract": cfg_a,
                "warmup_source_includes_darg_and_historical_losses": True}
    checks.append(_record("TEST_18_g3_g4_shared_initial_state_rng", shared_rng_probe))
    def metrics_probe():
        truth = np.array([0, 0, 1, 1, 2, 2, 2])
        prediction = np.array([0, 1, 0, 1, 2, 2, 1])
        metrics = acc_auc(truth, prediction, 3)
        auc = roc_auc_score(np.eye(3)[truth], np.eye(3)[prediction], average="macro", multi_class="ovr")
        assert abs(metrics["ACC"] - 4 / 7) <= 1e-12 and abs(metrics["AUC"] - auc) <= 1e-12
        return metrics
    checks.append(_record("TEST_19_three_class_acc_hard_auc", metrics_probe))
    def scope_probe():
        from run_e833 import anchor_pair_support, build_model, make_pairs
        source_manifest = json.loads((ROOT / "sources/source_manifest.json").read_text())
        manifest_paths = [entry["path"] for entry in source_manifest.get("files", [])]
        assert len(y) == 598 and all(not set(s["train_idx"]) & set(s["test_idx"]) for s in splits)
        pair_counts = []
        for split in splits:
            train_idx = np.asarray(split["train_idx"], dtype=np.int64)
            fold = int(split["fold"])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = build_model("G3_E74_4_G2", device=x.device, fold=fold)
                support = anchor_pair_support(model, x)
            pairs, pair_same = make_pairs(train_idx, y.numpy(), fold, support)
            assert set(pairs.reshape(-1)).issubset(set(train_idx.tolist()))
            assert len(pairs) == 4096 and int(pair_same.sum()) == 2048 and int((pair_same == 0).sum()) == 2048
            assert np.array_equal(pair_same.astype(bool), y.numpy()[pairs[:, 0]] == y.numpy()[pairs[:, 1]])
            pair_counts.append([len(pairs), int(pair_same.sum()), int((pair_same == 0).sum())])
        forbidden = [p for p in manifest_paths if "checkpoint" in p.lower() or "patient_level" in p.lower()]
        assert not forbidden
        return {"n_samples": int(len(y)), "no_abide_model_or_prediction_artifacts": True,
                **verify_locked_data(x, y, splits),
                "train_only_relation_scope": True, "all_outer_test_disjoint": True,
                "pair_counts": pair_counts, "forbidden_manifest_paths": forbidden,
                "tadpole_rids": int(len(set(json.loads((ROOT / "sources/split_indices.json").read_text())["rid_order"]))) if "rid_order" in json.loads((ROOT / "sources/split_indices.json").read_text()) else 598}
    checks.append(_record("TEST_20_no_abide_overlap_or_test_label_training", scope_probe))

    target_probe = _record("GPU_runtime_and_checkpoint", lambda: gpu_runtime_probe(x, y, train))
    passed = all(item["status"] == "PASS" for item in checks) and target_probe["status"] == "PASS"
    query_path = ROOT / "inputs/prompts/mmGPL12_scmr.pt"
    query = torch.load(query_path, map_location="cpu", weights_only=True)
    adapter = {
        "query_shape": list(query["q_medical"].shape),
        "anonymous_query_shape": list(query["q_anonymous"].shape),
        "route_groups": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        "query_file_sha256": sha_file(query_path),
        "finite": bool(torch.isfinite(query["q_medical"]).all() and torch.isfinite(query["q_anonymous"]).all()),
    }
    result = {
        "status": "PASS" if passed else "FAIL", "checks": checks,
        "data_sha256": sha_file(ROOT / "sources/processed_standard_data.csv"),
        "split_sha256": sha_file(ROOT / "sources/split_indices.json"),
        "seed": 0, "device": str(x.device),
        "target_gpu_probe": target_probe, "implementation_sha256": implementation_hashes(),
        "method_fidelity_audit": "PASS" if passed else "FAIL",
        "formal_training_allowed": passed,
        "mmgpl12_adapter": adapter,
    }
    write_json(ROOT / "audits/preflight.json", result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())

