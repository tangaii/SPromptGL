"""Compare baseline initialization, forward tensors and gradients against the archive."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from fidelity_model import FidelityModel, HistoricalGFR
from source_runtime import BASE, ROOT, archived_definitions, replace_exact, source_graph_utils


def reference_baselines(prompt_bank):
    layer_path = BASE / "layers.py"
    layer_text = replace_exact(layer_path.read_text(), ".cuda()", ".to(input_data.device)", count=3)
    layers = archived_definitions(
        layer_path, ("GraphConv", "Attention", "VariLengthInputLayer", "EncodeLayer", "FeedForwardLayer", "OutputLayer"),
        source=layer_text,
    )
    utils_path = BASE / "utils.py"
    utils = source_graph_utils(utils_path)
    initialize = archived_definitions(utils_path, ("my_weight_init",)).my_weight_init
    net_path = BASE / "network.py"
    net_text = replace_exact(net_path.read_text(), ".cuda()", ".to(label.device)", count=2)
    net_text = replace_exact(net_text, "self.proj_drop = nn.Dropout(proj_drop)",
                             "for i in range(4, len(input_dim)):\n"
                             "            self.add_module(f'GraphConstruct{i}', GraphLearn(input_dim[i], self.th, self.mode))\n"
                             "            self.add_module(f'MessagePassing{i}', GCN(input_sum, input_sum))\n"
                             "        self.proj_drop = nn.Dropout(proj_drop)")
    net_text = replace_exact(net_text, "            all_adj.append(adj)",
                             "            elif i >= 4:\n"
                             "                adj = getattr(self, f'GraphConstruct{i}')(x[i])\n"
                             "                norm = normalize_adj(adj + torch.eye(adj.size(0)).to(f_w.device))\n"
                             "                ALoss = torch.linalg.norm(norm - A).to(f_w.device) + ALoss\n"
                             "                feat = getattr(self, f'MessagePassing{i}')(torch.cat(x, dim=1), norm)\n"
                             "                combined_feat = f_w * feat + combined_feat\n"
                             "            all_adj.append(adj)")
    for old, new in (("output_dim = 9", "output_dim = self.d_v"),
                     ("x.view(871, 4, 4, 9)", "x.view(bs, self.modal_num, self.n_head, self.d_v)"),
                     ("x.view(-1, 1, 36)", "x.view(-1, 1, self.d_v * self.n_head)"),
                     ("global_x.view(871, 4, 9)", "global_x.view(bs, self.modal_num, self.d_v)")):
        net_text = replace_exact(net_text, old, new)
    net = archived_definitions(
        net_path, ("GFR", "GraphLearn", "GCN", "PLN", "MLP"), source=net_text,
        **vars(layers), normalize_adj=utils.normalize_adj, device=prompt_bank.device,
        get_prompt_text_features=lambda: prompt_bank,
    )
    e43_path = ROOT / "sources/E43_1_E00_original/run.py"
    e43 = archived_definitions(e43_path, ("make_pln_without_pen",)).make_pln_without_pen
    return net, layers, utils, initialize, e43


def compare_parameters(actual, expected, keys):
    for name, parameter in expected.items():
        torch.testing.assert_close(actual[keys.get(name, name)], parameter, rtol=0, atol=0)
    assert len(actual) == len(expected)


def baseline_probe(group, x, labels, dims, prompt_path):
    use_pen = group == "G1_E28_E0"
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)["medical"].float().to(x.device)
    source, layers, utils, initialize, e43 = reference_baselines(prompt)
    hp = SimpleNamespace(n_hidden=10, n_head=4, nlayer=2, nmodal=len(dims), nclass=3, dropout=0.)
    devices = [x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        reference_gfr = source.GFR(dims, dims, .9).to(x.device)
        reference_cam = (source.PLN(dims, hp) if use_pen else e43(torch, layers, dims, hp)).to(x.device)
        reference_cam.apply(initialize)
        reference_rng = torch.get_rng_state()
        torch.manual_seed(0)
        model = FidelityModel(dims, group, 3, str(prompt_path), device=x.device)
        assert torch.equal(torch.get_rng_state(), reference_rng)

    # The fixed embedding is a buffer. E83.1 groups the two original MLP
    # layers under the physically removable PEN submodule.
    keys = {"prompt_text_features": "PEN.prompt_text_features"}
    for rel, index in (("t1", 0), ("t2", 2)):
        keys.update({f"MLP.{rel}.{p}": f"PEN.MLP.{index}.{p}" for p in ("weight", "bias")})
    compare_parameters(model.gfr.state_dict(), reference_gfr.state_dict(), {})
    compare_parameters(model.cam.state_dict(), reference_cam.state_dict(), keys)
    rows = torch.arange(len(x) - 3, device=x.device)
    reference_parts = list(x.split(dims, dim=1))
    trace = []
    handles = [getattr(reference_gfr, f"MessagePassing{i}").register_forward_hook(
        lambda _module, args, output: trace.append((args[1], output))) for i in range(len(dims))]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        fused, graphs, aloss = reference_gfr(reference_parts, len(dims), labels, rows)
        expected, _, attention = reference_cam(fused)
        after = torch.get_rng_state()
        torch.manual_seed(0)
        actual, result = model(x, labels, rows)
        assert torch.equal(torch.get_rng_state(), after)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(result["aloss"], aloss, rtol=0, atol=0)
    for i, (normalized, message) in enumerate(trace):
        torch.testing.assert_close(result["adjacencies"][i], graphs[i], rtol=0, atol=0)
        torch.testing.assert_close(result["normalized_adjacencies"][i], normalized, rtol=0, atol=0)
        torch.testing.assert_close(result["messages"][i], message, rtol=0, atol=0)
    for local, archived in zip(result["attention_maps"], attention):
        torch.testing.assert_close(local.mean(1).detach().cpu(), torch.from_numpy(archived), rtol=0, atol=0)
    expected_graph = sum(utils.GraphConstructLoss(part, graph, 1., .5, 0.)
                         for part, graph in zip(reference_parts, graphs)) / len(dims)
    weights = torch.ones(3, device=x.device)
    expected_loss = utils.ClsLoss(expected, labels, rows, weights) + expected_graph
    actual_loss = utils.ClsLoss(actual, labels, rows, weights) + result["graph_loss"]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    gradient_count, max_error = 0, 0.
    for local, archived, mapping in ((model.gfr, reference_gfr, {}), (model.cam, reference_cam, keys)):
        local_params = dict(local.named_parameters())
        for name, expected_param in archived.named_parameters():
            grad = local_params[mapping.get(name, name)].grad
            if expected_param.grad is None:
                assert grad is None
                continue
            torch.testing.assert_close(grad, expected_param.grad, rtol=1e-5, atol=1e-7)
            gradient_count += 1
            max_error = max(max_error, float((grad - expected_param.grad).abs().max()))
    return {"forward_max_abs": 0., "gradient_max_abs": max_error, "compared_gradients": gradient_count,
            "modalities": len(dims), "historical_attention_dropout_rng_equal": True}


def four_modal_graph_probe(x, labels, dims, prompt_path):
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)["medical"]
    source, _, _, _, _ = reference_baselines(prompt)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        historical = source.GFR(dims, dims, .9)
        torch.random.default_generator.manual_seed(0)
        current = HistoricalGFR(dims)
    parts = list(x[:, :sum(dims)].split(dims, dim=1))
    rows = torch.arange(len(x) - 3)
    expected, _, _ = historical(parts, len(dims), labels, rows)
    actual, _ = current(parts, labels, rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"historical_four_modal_output_max_abs": 0.}
