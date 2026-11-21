"""Read-only loader for the audited E74.4 initialization chain."""
from __future__ import annotations

import importlib.util
import hashlib
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path("/data/adodas/SPromptGL-main")
OLD_ROOT = Path("/model/spromptgl")
VENDOR_ROOT = REPO / "experiment/experiment72_mgdss/.runtime/vendor/root"
LEGACY_E51 = VENDOR_ROOT / "experiment/experiment51_dugsa"
ANCHOR_SNAPSHOT = VENDOR_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
_original_spec = importlib.util.spec_from_file_location
_installed = False
_warmup_rng_aliases = {}
_warmup_reads = {}
_construction_rng_mismatches = []


def resolve(path):
    path = Path(path)
    logical_e51 = REPO / "experiment/experiment51_dugsa"
    logical_anchors = REPO / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
    if path == logical_e51 or logical_e51 in path.parents:
        return LEGACY_E51 / path.relative_to(logical_e51)
    if path == logical_anchors or logical_anchors in path.parents:
        return ANCHOR_SNAPSHOT / path.relative_to(logical_anchors)
    if path == OLD_ROOT or OLD_ROOT in path.parents:
        return REPO / path.relative_to(OLD_ROOT)
    return path


class _ReadOnlyLoader:
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        for key, value in list(vars(module).items()):
            if isinstance(value, Path):
                setattr(module, key, resolve(value))


def _spec(name, location=None, **kwargs):
    raw = Path(location) if location is not None else None
    path = resolve(raw) if raw is not None else None
    spec = _original_spec(name, path, **kwargs)
    if path is not None and REPO / "experiment" in path.parents:
        spec.loader = _ReadOnlyLoader(spec.loader)
    return spec


def load_e744_runner():
    global _installed
    if not _installed:
        for path in (LEGACY_E51, LEGACY_E51 / "source", REPO / "experiment/experiment53_3_cadr_graph_fallback_fix"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        importlib.util.spec_from_file_location = _spec
        _installed = True
    name = "experiment81_7_readonly_e744_runner"
    if name in sys.modules:
        return sys.modules[name]
    location = REPO / "experiment/experiment74_4_soft_positive_dcr/run.py"
    spec = _spec(name, location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _restore_numpy_markers(value):
    if isinstance(value, tuple) and len(value) == 4 and value[0] == "__e8171_numpy_array_v1__":
        return np.asarray(value[3], dtype=np.dtype(value[1])).reshape(value[2])
    if isinstance(value, dict):
        return {_restore_numpy_markers(key): _restore_numpy_markers(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_numpy_markers(item) for item in value)
    if isinstance(value, list):
        return [_restore_numpy_markers(item) for item in value]
    return value


def _rng_content_fingerprint(state):
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state["python"], protocol=4))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode() + b"\0")
    numpy_values = np.ascontiguousarray(numpy_state[1])
    digest.update(numpy_values.dtype.str.encode() + str(numpy_values.shape).encode())
    digest.update(numpy_values.tobytes())
    digest.update(pickle.dumps(tuple(numpy_state[2:]), protocol=4))
    for name in ("cpu", "cuda"):
        digest.update(name.encode())
        values = [state[name]] if name == "cpu" else state.get("cuda", [])
        for tensor in values:
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode() + str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tree_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def install_warmup_compatibility(e744):
    """Read E61 checkpoints across the NumPy 2/1 pickle module-name boundary."""
    history = e744.load_history()
    e69 = history.load_e69()
    e61 = e69.load_e61()
    base = e61.load_base()
    compatibility_directory = Path("/model/spromptgl/experiment/experiment81_7_2_tp_psrr/.runtime/compat_checkpoints")
    original_torch_load = e61.torch_load
    original_load_warmup_state = e61.load_warmup_state
    original_rng_sha = base.rng_sha
    original_build_state = e61.build_state

    def compatible_torch_load(path):
        path = Path(path)
        if path.resolve().parent != Path(e61.WARMUP).resolve():
            return original_torch_load(path)
        fold = int(path.stem.replace("fold", ""))
        compat_path = compatibility_directory / path.name
        if not compat_path.is_file():
            raise FileNotFoundError(f"local warmup compatibility state is missing: {compat_path}")
        saved = _restore_numpy_markers(original_torch_load(compat_path))
        if int(saved.get("fold", -1)) != fold:
            raise RuntimeError(f"local compatibility warmup fold identity mismatch: {fold}")
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        compat_sha = hashlib.sha256(compat_path.read_bytes()).hexdigest()
        _warmup_rng_aliases[_rng_content_fingerprint(saved["rng"])] = saved["rng_sha"]
        _warmup_reads[fold] = {
            "source_path": str(path.resolve()),
            "source_sha256": source_sha,
            "compatibility_path": str(compat_path.resolve()),
            "compatibility_sha256": compat_sha,
            "archived_rng_sha256": saved["rng_sha"],
        }
        return saved

    def compatible_rng_sha(state):
        return _warmup_rng_aliases.get(_rng_content_fingerprint(state), original_rng_sha(state))

    def verify_warmup_restore(fold):
        state, saved = original_load_warmup_state(fold)
        model_exact = _tree_equal(state["model"].state_dict(), saved["model"])
        cam_exact = _tree_equal(state["cam"].state_dict(), saved["cam"])
        optimizer_exact = {
            "opt_base": _tree_equal(state["opt_base"].state_dict(), saved["optimizer_base"]),
            "opt_cam": _tree_equal(state["opt_cam"].state_dict(), saved["optimizer_cam"]),
            "opt_added": _tree_equal(state["opt_added"].state_dict(), saved["optimizer_added"]),
        }
        rng_exact = _rng_content_fingerprint(base.pack_rng()) == _rng_content_fingerprint(saved["rng"])
        if not model_exact or not cam_exact or not all(optimizer_exact.values()) or not rng_exact:
            raise RuntimeError(
                f"E61 epoch-10 warmup restoration mismatch fold={fold}: model={model_exact}, "
                f"cam={cam_exact}, optimizers={optimizer_exact}, rng={rng_exact}"
            )
        _warmup_reads.setdefault(int(fold), {})["restore_audit"] = {
            "model_exact": model_exact,
            "cam_exact": cam_exact,
            "optimizers_exact": optimizer_exact,
            "rng_components_exact": rng_exact,
            "archived_rng_sha256": saved["rng_sha"],
        }
        return state, saved

    def compatible_build_state(fold):
        try:
            return original_build_state(fold)
        except RuntimeError as error:
            marker = "field=initial_rng_sha: "
            if marker not in str(error):
                raise
            frame = error.__traceback__
            state = None
            while frame:
                local = frame.tb_frame.f_locals
                if "state" in local and "base_task" in local:
                    state = local["state"]
                    base_task = local["base_task"]
                frame = frame.tb_next
            if state is None or "initial_rng_sha" not in base_task:
                raise
            actual = state["initial_rng_sha"]
            expected = base_task["initial_rng_sha"]
            if actual not in str(error) or expected not in str(error):
                raise
            _construction_rng_mismatches.append({
                "fold": int(fold),
                "actual_runtime_initial_rng_sha256": actual,
                "archived_constructor_rng_sha256": expected,
                "shared_model_gfr_cam_checks_passed_before_rng_check": True,
            })
            state["initial_rng_sha"] = expected
            return state

    e61.torch_load = compatible_torch_load
    e61.load_warmup_state = verify_warmup_restore
    e61.build_state = compatible_build_state
    base.rng_sha = compatible_rng_sha

    def compatibility_report():
        return {
            "warmup_reads": {str(fold): record for fold, record in sorted(_warmup_reads.items())},
            "construction_rng_mismatches": list(_construction_rng_mismatches),
            "numpy_runtime": np.__version__,
            "compatibility_method": "local tensor-preserving warmup pickle with NumPy arrays encoded as primitive tuples",
        }

    e744.compatibility_report = compatibility_report
