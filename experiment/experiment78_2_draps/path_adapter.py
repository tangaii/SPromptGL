"""SHA-checked path relocation for the read-only E74.4 dependency chain."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

REPO = Path("/data/adodas/SPromptGL-main")
HERE = Path(__file__).resolve().parent
BASE = REPO / "experiment/experiment74_4_soft_positive_dcr"
E773 = REPO / "experiment/experiment77_3_ds_cbpm"
OLD = Path("/model/spromptgl")
VENDOR = REPO / "experiment/experiment72_mgdss/.runtime/vendor/root"
LEGACY = VENDOR / "experiment/experiment51_dugsa"
ANCHORS = VENDOR / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
_original_spec = importlib.util.spec_from_file_location
_installed = False


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    if path == OLD or OLD in path.parents:
        path = REPO / path.relative_to(OLD)
    for logical, actual in (
        (REPO / "experiment/experiment51_dugsa", LEGACY),
        (REPO / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors", ANCHORS),
    ):
        if path == logical or logical in path.parents:
            return actual / path.relative_to(logical)
    return path


def verify_relocation():
    protected = json.loads((E773 / "results/protected_after.json").read_text())
    checked = {}
    for logical, expected in protected.items():
        actual_path = resolve(logical)
        actual = sha_file(actual_path)
        if actual != expected:
            raise RuntimeError(f"historical SHA mismatch: {logical} -> {actual_path}: {actual} != {expected}")
        checked[str(actual_path)] = actual
    frozen = json.loads((E773 / "manifest.json").read_text())["frozen_source_map"]
    for logical, expected in frozen.items():
        actual_path = resolve(logical)
        actual = sha_file(actual_path)
        if actual != expected:
            raise RuntimeError(f"frozen-source SHA mismatch: {logical} -> {actual_path}")
        checked[str(actual_path)] = actual
    anchor_manifest = json.loads((REPO / "experiment/experiment53_3_cadr_graph_fallback_fix/manifest.json").read_text())
    for logical, expected in anchor_manifest["anchor_sha256"].items():
        actual_path = ANCHORS / Path(logical).name
        if sha_file(actual_path) != expected:
            raise RuntimeError(f"anchor SHA mismatch: {actual_path}")
        checked[str(actual_path)] = expected
    expected_local = {
        "dugsa.py": "ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12",
        "source/layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
        "source/network.py": "5542f52d598e2f4d32aaad0c46b80c214fd562f74110a1d0ffcc619b9aefb3d1",
        "source/utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67",
    }
    for relative, expected in expected_local.items():
        actual_path = LEGACY / relative
        actual = sha_file(actual_path)
        if actual != expected:
            raise RuntimeError(f"legacy implementation SHA mismatch: {actual_path}: {actual}")
        checked[str(actual_path)] = actual
    return checked


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
    path = resolve(location) if location is not None else None
    spec = _original_spec(name, path, **kwargs)
    if path is not None and REPO / "experiment" in path.parents and HERE not in path.parents:
        spec.loader = _ReadOnlyLoader(spec.loader)
    return spec


def load_base_runner():
    global _installed
    if not _installed:
        verify_relocation()
        importlib.util.spec_from_file_location = _spec
        for path in (LEGACY, LEGACY / "source", REPO / "experiment/experiment53_3_cadr_graph_fallback_fix"):
            sys.path.insert(0, str(path))
        _installed = True
    name = "experiment78_2_readonly_e744_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = _spec(name, BASE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
