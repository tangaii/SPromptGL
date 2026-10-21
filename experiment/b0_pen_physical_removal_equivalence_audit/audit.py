#!/usr/bin/env python3
"""Compare Experiment 32 B0 with a PLN that never constructs the PEN."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
AUDIT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = AUDIT_DIR / "results.json"
SEED = 0
FOLD = 1
ATOL = 1e-6
RTOL = 1e-5
INPUT_DIMS = [48, 6, 10, 256]
PROTOCOL = dict(
    datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
    nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
    n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
    GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
    alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
    nclass=2, mode="simple-2", seed=SEED,
)
ROOT_CORE = (
    "main.py", "model.py", "network.py", "layers.py", "utils.py",
    "ABIDE-simple-2-concat-weighted-cosine.sh",
)
EXP_PROTECTED = (
    "README.md", "method.py", "integration.py", "integration.patch", "run.py",
    "manifest.json", "results/report.md", "results/fold_metrics.csv",
    "results/diagnostics.csv",
)
DATA_FILES = (
    "data/ABIDE/processed_standard_data.csv",
    "data/ABIDE/modal_feat_dict.npy",
)
CLIP_REPORT = ROOT / "experiment/clip_encoder_equivalence_audit/report.md"
CLIP_AUDIT = ROOT / "experiment/clip_encoder_equivalence_audit/audit.json"
RN50_PATH = Path.home() / ".cache/clip/RN50.pt"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def protected_hashes() -> dict[str, str]:
    paths = ([ROOT / name for name in ROOT_CORE]
             + [EXP / name for name in EXP_PROTECTED]
             + [ROOT / name for name in DATA_FILES]
             + [CLIP_REPORT, CLIP_AUDIT])
    missing = [str(path) for path in paths if not path.is_file()]
    assert not missing, "protected inputs missing: " + ", ".join(missing)
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def rng_seed(torch, seed: int) -> None:
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_capture(torch) -> dict:
    import numpy as np
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def rng_restore(torch, state: dict) -> None:
    random.setstate(state["python"])
    import numpy as np
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_digest(state: dict) -> dict[str, str]:
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    cpu = sha_bytes(state["cpu"].cpu().numpy().tobytes())
    cuda = [sha_bytes(value.cpu().numpy().tobytes()) for value in state["cuda"]]
    digest.update(bytes.fromhex(cpu))
    for value in cuda:
        digest.update(bytes.fromhex(value))
    return {"all_sha256": digest.hexdigest(), "cpu_sha256": cpu, "cuda_sha256": cuda}


def tensor_sha(value) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_snapshot(module) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def named_state_snapshot(agent) -> dict:
    values = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            values[prefix + "." + name] = parameter.detach().cpu().clone()
    return values


def state_digest(values: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def copy_and_validate_shared(torch, agent_a, agent_b) -> dict:
    params_a = named_state_snapshot(agent_a)
    params_b = named_state_snapshot(agent_b)
    removed = sorted(set(params_a) - set(params_b))
    added = sorted(set(params_b) - set(params_a))
    assert not added, "B0-B has unexpected parameters: " + repr(added)
    assert removed and all(name.startswith("pln.MLP.") for name in removed), removed
    assert set(params_b) == set(params_a) - set(removed)
    destinations = {
        **{"gfr." + name: parameter for name, parameter in agent_b.MessagePassing0.named_parameters()},
        **{"pln." + name: parameter for name, parameter in agent_b.ModalFusion.named_parameters()},
    }
    copied = []
    with torch.no_grad():
        for name, source in params_a.items():
            if name not in params_b:
                continue
            target = destinations[name]
            assert tuple(source.shape) == tuple(target.shape), name
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            copied.append(name)
    checked = named_state_snapshot(agent_b)
    mismatched = [name for name in copied if not torch.equal(params_a[name], checked[name])]
    assert not mismatched, "parameter copy failed: " + repr(mismatched)

    buffers_a = {
        "pln." + name: value.detach().cpu().clone()
        for name, value in agent_a.ModalFusion.named_buffers()
    }
    buffers_b = {
        "pln." + name: value.detach().cpu().clone()
        for name, value in agent_b.ModalFusion.named_buffers()
    }
    assert set(buffers_a) - set(buffers_b) == {"pln.prompt_text_features"}
    assert set(buffers_b) <= set(buffers_a)
    for name, value in buffers_b.items():
        assert torch.equal(value, buffers_a[name]), "common buffer mismatch: " + name
    return {"copied_parameter_names": copied, "removed_parameter_names": removed,
            "common_buffer_names": sorted(buffers_b),
            "removed_buffer_names": sorted(set(buffers_a) - set(buffers_b))}


def count_parameters(module) -> dict:
    values = list(module.named_parameters())
    return {"tensor_count": len(values), "element_count": sum(p.numel() for _, p in values),
            "names": [name for name, _ in values]}


def optimizer_summary(optimizer, module) -> dict:
    named_ids = {id(parameter): name for name, parameter in module.named_parameters()}
    group_summaries = []
    names = []
    for group in optimizer.param_groups:
        names.extend(named_ids.get(id(parameter), "<unmatched>") for parameter in group["params"])
        group_summaries.append({key: value for key, value in group.items() if key != "params"})
    assert "<unmatched>" not in names
    return {"parameter_tensor_count": len(names),
            "parameter_element_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
            "parameter_names": names, "groups": group_summaries}


def clone_tree(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    return value


def leaf_tensor(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return None


def compare_tree(left, right) -> dict:
    import torch
    leaves = []
    structural = []

    def walk(a, b, path):
        ta, tb = leaf_tensor(a), leaf_tensor(b)
        if ta is not None or tb is not None:
            if ta is None or tb is None or ta.shape != tb.shape or ta.dtype != tb.dtype:
                structural.append(path)
                return
            equal = torch.equal(ta, tb)
            if ta.is_floating_point() or ta.is_complex():
                error = float((ta - tb).abs().max().item()) if ta.numel() else 0.0
                close = bool(torch.allclose(ta, tb, atol=ATOL, rtol=RTOL))
            else:
                error = 0.0 if equal else 1.0
                close = equal
            leaves.append({"name": path, "exact": equal, "allclose": close,
                           "max_abs_error": error, "shape": list(ta.shape), "dtype": str(ta.dtype)})
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                structural.append(path + ".keys")
            for key in sorted(set(a) & set(b), key=str):
                walk(a[key], b[key], path + "." + str(key))
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                structural.append(path + ".length")
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + f"[{index}]")
            return
        if a != b:
            structural.append(path)

    walk(left, right, "value")
    return {
        "exact": not structural and all(item["exact"] for item in leaves),
        "allclose": not structural and all(item["allclose"] for item in leaves),
        "max_abs_error": max((item["max_abs_error"] for item in leaves), default=0.0),
        "tensor_count": len(leaves), "structural_differences": structural,
        "differing_tensors": [item["name"] for item in leaves if not item["exact"]],
        "tensors": leaves,
    }


def compare_parameter_maps(left: dict, right: dict) -> dict:
    assert set(left) == set(right), "parameter map keys differ"
    result = compare_tree(left, right)
    result["differing_parameter_names"] = [item["name"].removeprefix("value.")
                                          for item in result["tensors"] if not item["exact"]]
    return result


def status_for(*comparisons) -> str:
    if all(item["exact"] for item in comparisons):
        return "EXACT"
    if all(item["allclose"] for item in comparisons):
        return "NUMERICAL"
    return "FAIL"


def build_pln_no_pen(torch, layers, input_data_dims, hyperpm):
    """Register the original CAM and classifier without constructing PEN modules."""
    import torch.nn as nn

    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyperpm = hyperpm
            self.input_data_dims = input_data_dims
            self.d_q = hyperpm.n_hidden
            self.d_k = hyperpm.n_hidden
            self.d_v = hyperpm.n_hidden
            self.n_head = hyperpm.n_head
            self.dropout = hyperpm.dropout
            self.n_layer = hyperpm.nlayer
            self.modal_num = hyperpm.nmodal
            self.n_class = hyperpm.nclass
            self.d_out = self.d_v * self.n_head * self.modal_num
            self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
            self.InputLayer = layers.VariLengthInputLayer(
                self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout
            )
            self.Encoder = []
            self.FeedForward = []
            for index in range(self.n_layer):
                encoder = layers.EncodeLayer(
                    self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout
                )
                self.add_module("encode_%d" % index, encoder)
                self.Encoder.append(encoder)
                feedforward = layers.FeedForwardLayer(
                    self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout
                )
                self.add_module("feed_%d" % index, feedforward)
                self.FeedForward.append(feedforward)
            d_in = self.d_v * self.n_head * self.modal_num
            self.Outputlayer = layers.OutputLayer(
                d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout
            )
            self.device = "cuda"
            self.dcer32_config = "B0"

        def forward(self, x):
            batch_size = x.size(0)
            attn_map = []
            x, input_attention = self.InputLayer(x)
            attn = input_attention.mean(dim=1)
            attn_map.append(attn.detach().cpu().numpy())
            for index in range(self.n_layer):
                x, layer_attention = self.Encoder[index](q=x, k=x, v=x, modal_num=self.modal_num)
                attn = layer_attention.mean(dim=1)
                x = self.FeedForward[index](x)
                attn_map.append(attn.detach().cpu().numpy())
            x = x.view(batch_size, -1)
            attn_embedding = attn.view(batch_size, -1)
            output, hidden = self.Outputlayer(x, attn_embedding)
            return output, hidden, attn_map

    return PLNNoPEN()


def build_b_agent(torch, np, layers, network, utils, hyperpm, x, y, train, test):
    from torch import optim

    class Agent:
        pass

    agent = Agent()
    device = torch.device("cuda")
    agent.dev = device
    agent.hyperpm = hyperpm
    agent.th = hyperpm.th
    agent.GC_mode = hyperpm.GC_mode
    agent.MP_mode = hyperpm.MP_mode
    agent.MF_mode = hyperpm.MF_mode
    agent.d_v = hyperpm.n_hidden
    agent.modal_num = hyperpm.nmodal
    agent.n_class = hyperpm.nclass
    agent.dropout = hyperpm.dropout
    agent.alpha = hyperpm.alpha
    agent.n_head = hyperpm.n_head
    agent.feat = torch.from_numpy(x).float().to(device)
    agent.targ = torch.from_numpy(y).long().to(device)
    agent.input_data_dims = INPUT_DIMS
    agent.trn_idx = train
    agent.tst_idx = np.asarray(test)
    agent.val_idx = np.asarray(test)
    trn_label = agent.targ[train].cpu().numpy()
    counter = Counter(trn_label)
    weight = len(trn_label) / np.asarray(list(counter.values())) / agent.n_class
    agent.weight = torch.from_numpy(weight).float().to(device)
    agent.best_acc = 0
    agent.best_acc_2 = 0
    agent.dataLayer = layers.InitialDataLayer(INPUT_DIMS)
    agent.MessagePassing0 = network.GFR(
        input_dim=INPUT_DIMS, output_dim=INPUT_DIMS, th=agent.th, mode=agent.GC_mode
    ).to(device)
    agent.ModalFusion = build_pln_no_pen(torch, layers, INPUT_DIMS, hyperpm).to(device)
    agent.optimizer_MP0 = optim.Adam(agent.MessagePassing0.parameters(), lr=hyperpm.lr,
                                    weight_decay=hyperpm.reg)
    agent.optimizer_MF = optim.Adam(agent.ModalFusion.parameters(), lr=hyperpm.lr,
                                   weight_decay=hyperpm.reg)
    agent.ModalFusion.apply(utils.my_weight_init)
    return agent


def build_forward(agent, torch, utils, captures=True):
    captured = {}
    handles = []
    if captures:
        modules = (("input", agent.ModalFusion.InputLayer),
                   ("encoder0", agent.ModalFusion.encode_0),
                   ("feedforward0", agent.ModalFusion.feed_0))
        for name, module in modules:
            def hook(_module, _args, output, key=name):
                captured[key] = clone_tree(output)
            handles.append(module.register_forward_hook(hook))
    try:
        model_num, parts = agent.dataLayer(agent.feat)
        features, adjs, aloss = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
        logp, hidden, attn_map = agent.ModalFusion(features)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "gfr_features": features,
        "gfr_adjacencies": adjs,
        "gfr_alignment_loss": aloss,
        "cam_stages": captured,
        "logp": logp,
        "probability": logp.exp(),
        "prediction": logp.argmax(dim=1),
        "hidden": hidden,
        "attention_maps": attn_map,
    }


def forward_snapshot(agent, torch, utils, train: bool):
    agent.MessagePassing0.train(train)
    agent.ModalFusion.train(train)
    with torch.no_grad():
        return clone_tree(build_forward(agent, torch, utils))


def simple2_backward(agent, torch, utils):
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad()
    agent.optimizer_MF.zero_grad()
    model_num, parts = agent.dataLayer(agent.feat)
    features, adjs, alignment_loss = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
    all_graph_loss = 0
    for index in range(model_num):
        graph_loss = utils.GraphConstructLoss(
            parts[index], adjs[index], agent.hyperpm.theta_smooth,
            agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity,
        )
        all_graph_loss = all_graph_loss + graph_loss
    all_graph_loss = all_graph_loss / 4
    logp, _, _ = agent.ModalFusion(features)
    cls_loss = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    total_loss = cls_loss + all_graph_loss
    total_loss.backward()
    gradients = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            gradients[prefix + "." + name] = (
                None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            )
    return {
        "cls_loss": float(cls_loss.detach().item()),
        "all_graph_loss": float(all_graph_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "alignment_loss_excluded_from_total": float(alignment_loss.detach().item()),
        "gradients": gradients,
    }


def gradient_comparison(left: dict, right: dict) -> dict:
    import torch
    assert set(left) == set(right), "gradient parameter names differ"
    both_none = []
    mismatches = []
    tensor_comparisons = {}
    for name in sorted(left):
        a, b = left[name], right[name]
        if a is None or b is None:
            if a is None and b is None:
                both_none.append(name)
            else:
                mismatches.append(name)
            continue
        comparison = compare_tree(a, b)
        tensor_comparisons[name] = comparison
        if not comparison["allclose"]:
            mismatches.append(name)
    present = list(tensor_comparisons.values())
    return {
        "exact": not mismatches and all(item["exact"] for item in present),
        "allclose": not mismatches and all(item["allclose"] for item in present),
        "max_abs_error": max((item["max_abs_error"] for item in present), default=0.0),
        "parameter_count": len(left), "gradient_tensor_count": len(present),
        "both_grad_none_names": both_none, "mismatched_names": mismatches,
        "per_parameter": tensor_comparisons,
    }


def shared_gradient_maps(gradients_a: dict, gradients_b: dict) -> tuple[dict, dict]:
    assert set(gradients_b) <= set(gradients_a)
    return ({name: gradients_a[name] for name in gradients_b}, gradients_b)


def optimizer_state_by_name(agent, optimizer, module, prefix):
    names = {id(parameter): prefix + "." + name for name, parameter in module.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        name = names[id(parameter)]
        state[name] = {key: clone_tree(value) for key, value in values.items()}
    return state


def all_optimizer_states(agent) -> dict:
    return {
        "gfr": optimizer_state_by_name(agent, agent.optimizer_MP0, agent.MessagePassing0, "gfr"),
        "pln": optimizer_state_by_name(agent, agent.optimizer_MF, agent.ModalFusion, "pln"),
    }


def train_step(agent, torch, utils):
    report = simple2_backward(agent, torch, utils)
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    report["parameters"] = named_state_snapshot(agent)
    report["optimizer_state"] = all_optimizer_states(agent)
    return report


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "0") in ("0", "0,", "0 "), \
        "run with CUDA_VISIBLE_DEVICES=0 to use only physical GPU 0"
    assert RN50_PATH.is_file(), "local RN50 checkpoint missing; audit must not download it"
    protected_before = protected_hashes()
    handoff_before = sha_file(ROOT / "handoff.md")
    temp_root = tempfile.mkdtemp(prefix="b0_pen_equivalence_")
    try:
        sys.path.insert(0, str(EXP))
        import integration
        integration.emit(ROOT, Path(temp_root) / "sources")
        sys.path.insert(0, str(Path(temp_root) / "sources"))

        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        torch.use_deterministic_algorithms(True)
        import network
        import layers
        from sklearn.model_selection import StratifiedKFold

        calls = {"clip_load": 0, "get_prompt_text_features": 0, "mlp_init": 0, "avg_pool_init": 0}
        original_clip_load = network.clip.load
        original_get_features = network.get_prompt_text_features
        original_mlp_init = network.MLP.__init__
        original_pool_init = torch.nn.AvgPool1d.__init__

        def tracked_clip_load(name, device, *args, **kwargs):
            calls["clip_load"] += 1
            assert RN50_PATH.is_file(), "refuse remote CLIP checkpoint download"
            return original_clip_load(name, device, *args, **kwargs)

        def tracked_get_features(*args, **kwargs):
            calls["get_prompt_text_features"] += 1
            return original_get_features(*args, **kwargs)

        def tracked_mlp_init(self, *args, **kwargs):
            calls["mlp_init"] += 1
            return original_mlp_init(self, *args, **kwargs)

        def tracked_pool_init(self, *args, **kwargs):
            calls["avg_pool_init"] += 1
            return original_pool_init(self, *args, **kwargs)

        network.clip.load = tracked_clip_load
        network.get_prompt_text_features = tracked_get_features
        network.MLP.__init__ = tracked_mlp_init
        torch.nn.AvgPool1d.__init__ = tracked_pool_init

        import model
        import utils

        data_dir = ROOT / "data/ABIDE"
        frame = pd.read_csv(data_dir / "processed_standard_data.csv")
        mapping = np.load(data_dir / "modal_feat_dict.npy", allow_pickle=True).item()
        assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
        assert [len(value) for value in mapping.values()] == INPUT_DIMS
        names = [str(value) for group in mapping.values() for value in group]
        assert list(frame.columns[:-1]) == names and frame.shape == (871, 321)
        x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
        y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
        assert x.shape == (871, 320) and np.isfinite(x).all()
        assert np.bincount(y).tolist() == [403, 468]
        folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
        train, test = folds[FOLD - 1]
        hyperpm = SimpleNamespace(**PROTOCOL)

        # Independent same-seed natural initializations isolate constructor RNG effects.
        rng_seed(torch, SEED)
        rng_before_a = rng_capture(torch)
        calls_a_before = calls.copy()
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            integration.configure(config="B0", fold=FOLD, train=train, test=test,
                                  labels_train=y[train], seed=SEED)
            agent_a = model.EvalHelper(INPUT_DIMS, x, y, hyperpm, train, test)
        agent_a.MF_sav.close()
        agent_a.GCMP_sav.close()
        rng_after_a = rng_capture(torch)
        calls_a = {key: calls[key] - calls_a_before[key] for key in calls}
        params_a_natural = named_state_snapshot(agent_a)
        assert hasattr(agent_a.ModalFusion, "MLP")
        assert hasattr(agent_a.ModalFusion, "prompt_text_features")
        assert hasattr(agent_a.ModalFusion, "avg_pool")
        assert agent_a.ModalFusion.dcer32_config == "B0"

        rng_seed(torch, SEED)
        rng_before_b = rng_capture(torch)
        calls_b_before = calls.copy()
        agent_b = build_b_agent(torch, np, layers, network, utils, hyperpm, x, y, train, test)
        rng_after_b = rng_capture(torch)
        calls_b = {key: calls[key] - calls_b_before[key] for key in calls}
        params_b_natural = named_state_snapshot(agent_b)
        assert not hasattr(agent_b.ModalFusion, "MLP")
        assert not hasattr(agent_b.ModalFusion, "prompt_text_features")
        assert not hasattr(agent_b.ModalFusion, "avg_pool")
        assert agent_b.ModalFusion.dcer32_config == "B0"
        assert calls_b == {key: 0 for key in calls_b}, calls_b

        native_comparison = compare_parameter_maps(
            {key: params_a_natural[key] for key in params_b_natural}, params_b_natural
        )
        match_details = copy_and_validate_shared(torch, agent_a, agent_b)
        matched_a = named_state_snapshot(agent_a)
        matched_b = named_state_snapshot(agent_b)
        matched_common_a = {name: matched_a[name] for name in matched_b}
        matched_comparison = compare_parameter_maps(matched_common_a, matched_b)
        assert matched_comparison["exact"]

        pln_a_params = count_parameters(agent_a.ModalFusion)
        pln_b_params = count_parameters(agent_b.ModalFusion)
        mlp_params = count_parameters(agent_a.ModalFusion.MLP)
        assert pln_a_params["element_count"] == 575121
        assert mlp_params["element_count"] == 525577
        assert pln_b_params["element_count"] == 49544
        optim_a_pln = optimizer_summary(agent_a.optimizer_MF, agent_a.ModalFusion)
        optim_b_pln = optimizer_summary(agent_b.optimizer_MF, agent_b.ModalFusion)
        optim_a_gfr = optimizer_summary(agent_a.optimizer_MP0, agent_a.MessagePassing0)
        optim_b_gfr = optimizer_summary(agent_b.optimizer_MP0, agent_b.MessagePassing0)
        assert optim_a_pln["parameter_element_count"] == pln_a_params["element_count"]
        assert optim_b_pln["parameter_element_count"] == pln_b_params["element_count"]
        assert optim_a_gfr["groups"] == optim_b_gfr["groups"]
        assert optim_a_pln["groups"] == optim_b_pln["groups"]
        assert len({id(p) for g in agent_a.optimizer_MP0.param_groups for p in g["params"]}
                   & {id(p) for g in agent_a.optimizer_MF.param_groups for p in g["params"]}) == 0

        # Negative controls operate on temporary parameter perturbations and restore exactly.
        rng_seed(torch, SEED)
        baseline_a = forward_snapshot(agent_a, torch, utils, train=False)
        baseline_b = forward_snapshot(agent_b, torch, utils, train=False)
        assert compare_tree(baseline_a, baseline_b)["exact"]
        classifier_weight = agent_b.ModalFusion.Outputlayer.classifier.weight
        saved_classifier_value = classifier_weight[0, 0].detach().clone()
        with torch.no_grad():
            classifier_weight[0, 0].add_(5.0)
        changed_b = forward_snapshot(agent_b, torch, utils, train=False)
        n1 = compare_tree(baseline_a["logp"], changed_b["logp"])
        with torch.no_grad():
            classifier_weight[0, 0].copy_(saved_classifier_value)
        assert not n1["exact"] and n1["max_abs_error"] > 0
        n1["detected_output_difference"] = True
        n1["perturbation"] = "B0-B Outputlayer.classifier.weight[0,0] += 5"

        rng_seed(torch, SEED)
        before_mlp_perturb = forward_snapshot(agent_a, torch, utils, train=False)
        mlp_parameter = agent_a.ModalFusion.MLP.t1.weight
        saved_mlp_value = mlp_parameter[0, 0].detach().clone()
        with torch.no_grad():
            mlp_parameter[0, 0].add_(5.0)
        after_mlp_perturb = forward_snapshot(agent_a, torch, utils, train=False)
        n2 = compare_tree(before_mlp_perturb["logp"], after_mlp_perturb["logp"])
        with torch.no_grad():
            mlp_parameter[0, 0].copy_(saved_mlp_value)
        assert n2["exact"]
        n2["unchanged_output"] = True
        n2["perturbation"] = "B0-A ModalFusion.MLP.t1.weight[0,0] += 5"

        forward_tests = {}
        for mode, is_train in (("eval", False), ("train", True)):
            rng_seed(torch, SEED)
            start_rng = rng_capture(torch)
            snapshot_a = forward_snapshot(agent_a, torch, utils, train=is_train)
            after_a = rng_capture(torch)
            rng_restore(torch, start_rng)
            snapshot_b = forward_snapshot(agent_b, torch, utils, train=is_train)
            after_b = rng_capture(torch)
            cmp = compare_tree(snapshot_a, snapshot_b)
            cmp["rng_after_a"] = rng_digest(after_a)
            cmp["rng_after_b"] = rng_digest(after_b)
            cmp["matched_rng_consumption"] = rng_digest(after_a) == rng_digest(after_b)
            cmp["gfr_max_abs_error"] = compare_tree(snapshot_a["gfr_features"],
                                                    snapshot_b["gfr_features"])["max_abs_error"]
            cmp["cam_max_abs_error"] = compare_tree(snapshot_a["cam_stages"],
                                                    snapshot_b["cam_stages"])["max_abs_error"]
            cmp["classifier_max_abs_error"] = compare_tree(snapshot_a["logp"],
                                                            snapshot_b["logp"])["max_abs_error"]
            cmp["predictions_equal"] = bool(torch.equal(snapshot_a["prediction"], snapshot_b["prediction"]))
            forward_tests[mode] = cmp
            assert cmp["allclose"], f"forward {mode} mismatch: {cmp['differing_tensors'][:8]}"
            assert cmp["matched_rng_consumption"], f"different RNG consumption in {mode}"

        # Backward comparison uses the exact simple-2 objective and shared RNG masks.
        rng_seed(torch, SEED)
        backward_rng_start = rng_capture(torch)
        backward_a = simple2_backward(agent_a, torch, utils)
        backward_rng_a = rng_capture(torch)
        rng_restore(torch, backward_rng_start)
        backward_b = simple2_backward(agent_b, torch, utils)
        backward_rng_b = rng_capture(torch)
        loss_cmp = compare_tree(
            {key: backward_a[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
            {key: backward_b[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
        )
        shared_grad_a, shared_grad_b = shared_gradient_maps(
            backward_a["gradients"], backward_b["gradients"]
        )
        grad_cmp = gradient_comparison(shared_grad_a, shared_grad_b)
        assert all(parameter.grad is None for parameter in agent_a.ModalFusion.MLP.parameters())
        mlp_grad_none = [name for name, parameter in agent_a.ModalFusion.MLP.named_parameters()
                         if parameter.grad is None]
        assert len(mlp_grad_none) == len(agent_a.ModalFusion.MLP.state_dict())
        grad_none_a = [name for name, parameter in named_module_parameters(agent_a)
                       if parameter.grad is None]
        grad_none_b = [name for name, parameter in named_module_parameters(agent_b)
                       if parameter.grad is None]
        assert grad_cmp["allclose"], "gradient mismatch: " + repr(grad_cmp["mismatched_names"][:8])
        assert rng_digest(backward_rng_a) == rng_digest(backward_rng_b)

        effective_pln_a = parameter_effectiveness(agent_a.ModalFusion)
        effective_pln_b = parameter_effectiveness(agent_b.ModalFusion)
        effective_gfr_a = parameter_effectiveness(agent_a.MessagePassing0)
        effective_gfr_b = parameter_effectiveness(agent_b.MessagePassing0)
        assert effective_pln_a["gradient_parameter_names"] == effective_pln_b["gradient_parameter_names"]
        assert effective_gfr_a["gradient_parameter_names"] == effective_gfr_b["gradient_parameter_names"]

        mlp_before_updates = {name: tensor_sha(parameter)
                              for name, parameter in agent_a.ModalFusion.MLP.named_parameters()}
        adam_updates = []
        for step in range(1, 4):
            rng_seed(torch, SEED + step)
            step_start_rng = rng_capture(torch)
            update_a = train_step(agent_a, torch, utils)
            step_rng_a = rng_capture(torch)
            rng_restore(torch, step_start_rng)
            update_b = train_step(agent_b, torch, utils)
            step_rng_b = rng_capture(torch)
            shared_params_a = {name: update_a["parameters"][name] for name in update_b["parameters"]}
            param_cmp = compare_parameter_maps(shared_params_a, update_b["parameters"])
            shared_step_grad_a, shared_step_grad_b = shared_gradient_maps(
                update_a["gradients"], update_b["gradients"]
            )
            step_grad_cmp = gradient_comparison(shared_step_grad_a, shared_step_grad_b)
            loss_step_cmp = compare_tree(
                {key: update_a[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
                {key: update_b[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
            )
            adam_state_cmp = compare_tree(update_a["optimizer_state"], update_b["optimizer_state"])
            assert all(parameter.grad is None for parameter in agent_a.ModalFusion.MLP.parameters())
            mlp_state_now = {name: tensor_sha(parameter)
                             for name, parameter in agent_a.ModalFusion.MLP.named_parameters()}
            assert mlp_state_now == mlp_before_updates
            mlp_in_optimizer_state = any(
                parameter in agent_a.optimizer_MF.state
                for parameter in agent_a.ModalFusion.MLP.parameters()
            )
            assert not mlp_in_optimizer_state
            assert rng_digest(step_rng_a) == rng_digest(step_rng_b), f"step {step} RNG drift"
            assert param_cmp["allclose"], f"step {step} parameter mismatch"
            assert step_grad_cmp["allclose"], f"step {step} gradient mismatch"
            assert loss_step_cmp["allclose"], f"step {step} loss mismatch"
            assert adam_state_cmp["allclose"], f"step {step} Adam mismatch"
            adam_updates.append({
                "step": step,
                "loss_values": {
                    "B0_A": {key: update_a[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
                    "B0_B": {key: update_b[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
                },
                "losses": loss_step_cmp,
                "max_parameter_abs_error": param_cmp["max_abs_error"],
                "parameters_exact": param_cmp["exact"],
                "max_gradient_abs_error": step_grad_cmp["max_abs_error"],
                "gradients_exact": step_grad_cmp["exact"],
                "max_adam_state_abs_error": adam_state_cmp["max_abs_error"],
                "adam_state_exact": adam_state_cmp["exact"],
                "mlp_has_grad": any(p.grad is not None for p in agent_a.ModalFusion.MLP.parameters()),
                "mlp_adam_state_exists": mlp_in_optimizer_state,
                "rng_after_a": rng_digest(step_rng_a),
                "rng_after_b": rng_digest(step_rng_b),
                "rng_matched": rng_digest(step_rng_a) == rng_digest(step_rng_b),
                "parameter_differences": param_cmp["differing_parameter_names"],
                "gradient_differences": step_grad_cmp["mismatched_names"],
                "adam_state_differences": adam_state_cmp["differing_tensors"],
            })

        forward_status = status_for(*(forward_tests[mode] for mode in forward_tests))
        backward_status = status_for(loss_cmp, grad_cmp)
        adam_status = status_for(*[
            {"exact": step["parameters_exact"] and step["gradients_exact"] and step["adam_state_exact"],
             "allclose": step["max_parameter_abs_error"] <= ATOL
                         and step["max_gradient_abs_error"] <= ATOL
                         and step["max_adam_state_abs_error"] <= ATOL}
            for step in adam_updates
        ])
        mlp_optimizer_names = [name for name in optim_a_pln["parameter_names"] if name.startswith("MLP.")]
        assert len(mlp_optimizer_names) == 4
        assert not any(parameter.grad is not None for parameter in agent_a.ModalFusion.MLP.parameters())
        mlp_states = [parameter in agent_a.optimizer_MF.state
                      for parameter in agent_a.ModalFusion.MLP.parameters()]
        assert not any(mlp_states)

        results = {
            "audit": {
                "title": "B0 Physical PEN Removal Equivalence Audit",
                "date": date.today().isoformat(), "seed": SEED, "fold": FOLD,
                "fold_protocol": "StratifiedKFold(n_splits=10, shuffle=True, random_state=10), fold 1 only",
                "training_run": False, "formal_epochs": 0, "optimizer_update_steps": 3,
                "experiment33_created": False, "physical_gpu": 0,
                "forward_atol": ATOL, "forward_rtol": RTOL,
                "train_protocol": "simple-2: ClsLoss + mean(GraphConstructLoss over 4 modalities); two default Adam optimizers",
                "hyperparameters": PROTOCOL,
                "clip_audit_report_sha256": sha_file(CLIP_REPORT),
                "clip_cache_verified_a_equals_c": True,
                "clip_checkpoint": str(RN50_PATH), "clip_checkpoint_sha256": sha_file(RN50_PATH),
            },
            "environment": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "cuda_available": torch.cuda.is_available(),
                "visible_cuda_device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "protected_sha256_before": protected_before,
            "protected_sha256_after": protected_hashes(),
            "protected_files_unchanged": protected_before == protected_hashes(),
            "handoff_before_sha256": handoff_before,
            "data": {
                "shape": list(x.shape), "label_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
                "modality_dims": INPUT_DIMS, "train_indices_count": int(len(train)),
                "test_indices_count": int(len(test)), "train_indices_sha256": sha_bytes(train.tobytes()),
                "test_indices_sha256": sha_bytes(test.tobytes()),
            },
            "construction": {
                "B0_A": {"implementation": "Experiment 32 runtime EvalHelper + integration.install(config=B0)",
                         "calls_during_construction": calls_a,
                         "rng_before": rng_digest(rng_before_a), "rng_after": rng_digest(rng_after_a),
                         "attributes": {"MLP": hasattr(agent_a.ModalFusion, "MLP"),
                                        "prompt_text_features": hasattr(agent_a.ModalFusion, "prompt_text_features"),
                                        "avg_pool": hasattr(agent_a.ModalFusion, "avg_pool")}},
                "B0_B": {"implementation": "audit-only PLN with original InputLayer/Encoder/FeedForward/OutputLayer, B0 forward", 
                         "calls_during_construction": calls_b,
                         "rng_before": rng_digest(rng_before_b), "rng_after": rng_digest(rng_after_b),
                         "attributes": {"MLP": hasattr(agent_b.ModalFusion, "MLP"),
                                        "prompt_text_features": hasattr(agent_b.ModalFusion, "prompt_text_features"),
                                        "avg_pool": hasattr(agent_b.ModalFusion, "avg_pool")}},
                "same_seed_initial_rng": rng_digest(rng_before_a) == rng_digest(rng_before_b),
                "post_construction_rng_equal": rng_digest(rng_after_a) == rng_digest(rng_after_b),
                "natural_common_parameter_comparison": native_comparison,
                "natural_common_parameter_state_sha256": {
                    "B0_A": state_digest({k: v for k, v in params_a_natural.items() if k in params_b_natural}),
                    "B0_B": state_digest(params_b_natural),
                },
                "matching": match_details,
                "matched_common_parameters_exact": matched_comparison["exact"],
                "matched_parameter_state_sha256": {
                    "B0_A": state_digest(matched_common_a), "B0_B": state_digest(matched_b),
                },
                "named_parameter_match": matched_comparison,
            },
            "parameters": {
                "B0_A_PLN_registered": pln_a_params,
                "B0_B_PLN_registered": pln_b_params,
                "B0_A_MLP": mlp_params,
                "B0_A_MLP_parameter_names_in_optimizer": mlp_optimizer_names,
                "B0_A_MLP_grad_none_names": mlp_grad_none,
                "B0_A_MLP_hash_before_updates": mlp_before_updates,
                "B0_A_MLP_hash_after_updates": {
                    name: tensor_sha(parameter) for name, parameter in agent_a.ModalFusion.MLP.named_parameters()
                },
                "B0_A_MLP_optimizer_state_exists_after_updates": mlp_states,
                "B0_A_optimizer_MF": optim_a_pln,
                "B0_B_optimizer_MF": optim_b_pln,
                "B0_A_optimizer_MP0": optim_a_gfr,
                "B0_B_optimizer_MP0": optim_b_gfr,
                "B0_A_PLN_effectiveness_after_backward": effective_pln_a,
                "B0_B_PLN_effectiveness_after_backward": effective_pln_b,
                "B0_A_GFR_effectiveness_after_backward": effective_gfr_a,
                "B0_B_GFR_effectiveness_after_backward": effective_gfr_b,
                "B0_A_all_grad_none_names": grad_none_a,
                "B0_B_all_grad_none_names": grad_none_b,
            },
            "forward": {"status": forward_status, "tests": forward_tests},
            "backward": {
                "status": backward_status, "loss_comparison": loss_cmp,
                "loss_values": {
                    "B0_A": {key: backward_a[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
                    "B0_B": {key: backward_b[key] for key in ("cls_loss", "all_graph_loss", "total_loss")},
                },
                "gradient_comparison": grad_cmp,
                "rng_after_a": rng_digest(backward_rng_a), "rng_after_b": rng_digest(backward_rng_b),
                "rng_matched": rng_digest(backward_rng_a) == rng_digest(backward_rng_b),
            },
            "adam": {"status": adam_status, "steps": adam_updates,
                     "mlp_grad_none_all_steps": True,
                     "mlp_parameter_unchanged": mlp_before_updates == {
                         name: tensor_sha(parameter) for name, parameter in agent_a.ModalFusion.MLP.named_parameters()
                     },
                     "mlp_state_absent_all_steps": not any(mlp_states)},
            "negative_controls": {"N1_effective_output_weight_perturbation_detected": n1,
                                  "N2_unused_MLP_perturbation_does_not_change_output": n2},
            "checks": {
                "PEN_physically_removed": not hasattr(agent_b.ModalFusion, "MLP")
                    and not hasattr(agent_b.ModalFusion, "prompt_text_features")
                    and not hasattr(agent_b.ModalFusion, "avg_pool")
                    and calls_b == {key: 0 for key in calls_b},
                "B0_A_has_original_PEN_modules": hasattr(agent_a.ModalFusion, "MLP")
                    and hasattr(agent_a.ModalFusion, "prompt_text_features")
                    and hasattr(agent_a.ModalFusion, "avg_pool"),
                "forward": forward_status, "backward": backward_status, "adam": adam_status,
                "protected_files_unchanged": protected_before == protected_hashes(),
                "negative_controls": bool(n1["detected_output_difference"] and n2["unchanged_output"]),
            },
            "scientific_decision": {
                "B0_history_valid": "YES" if forward_status != "FAIL" and backward_status != "FAIL"
                    and adam_status != "FAIL" else "UNCERTAIN",
                "retain_experiment32_B0_ACC_92_0742": True,
                "physical_B0_B_suitable_as_future_structure_base": forward_status != "FAIL"
                    and backward_status != "FAIL" and adam_status != "FAIL",
                "recommend_B0_vs_B2_same_fold_20_fold_retest": False,
                "qualification": "Equivalence under matched shared initialization does not establish a new full-run score or B0/B2 performance superiority.",
            },
        }
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
        assert results["protected_files_unchanged"], "a protected source/data/Experiment32 file changed"
        assert all(results["checks"].values()), "one or more audit checks failed: " + repr(results["checks"])
        assert handoff_before == sha_file(ROOT / "handoff.md"), "handoff changed while the audit was running"
        print(json.dumps({
            "results": str(RESULTS_PATH), "forward": forward_status,
            "backward": backward_status, "adam": adam_status,
            "B0_A_parameters": pln_a_params["element_count"],
            "B0_B_parameters": pln_b_params["element_count"],
            "B0_A_effective_parameters": effective_pln_a["gradient_element_count"],
            "B0_B_effective_parameters": effective_pln_b["gradient_element_count"],
            "natural_shared_params_equal": native_comparison["exact"],
            "max_forward_error": max(item["max_abs_error"] for item in forward_tests.values()),
            "max_gradient_error": grad_cmp["max_abs_error"],
            "adam_max_parameter_error": max(item["max_parameter_abs_error"] for item in adam_updates),
            "adam_max_state_error": max(item["max_adam_state_abs_error"] for item in adam_updates),
            "protected_files_unchanged": results["protected_files_unchanged"],
        }, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def named_module_parameters(agent):
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            yield prefix + "." + name, parameter


def parameter_effectiveness(module) -> dict:
    values = list(module.named_parameters())
    active = [(name, parameter) for name, parameter in values if parameter.grad is not None]
    inactive = [name for name, parameter in values if parameter.grad is None]
    return {"registered_tensor_count": len(values),
            "registered_element_count": sum(parameter.numel() for _, parameter in values),
            "gradient_tensor_count": len(active),
            "gradient_element_count": sum(parameter.numel() for _, parameter in active),
            "gradient_parameter_names": [name for name, _ in active],
            "grad_none_parameter_names": inactive}


if __name__ == "__main__":
    main()
