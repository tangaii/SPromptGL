"""Read-only path routing for the historical E61-to-E74.4 loader chain."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path("/data/adodas/SPromptGL-main")
OLD_ROOT = Path("/model/spromptgl")
E81_ROOT = Path("/model/spromptgl/experiment/experiment81_prcl")
VENDOR_ROOT = REPO / "experiment/experiment72_mgdss/.runtime/vendor/root"
LEGACY_E51 = VENDOR_ROOT / "experiment/experiment51_dugsa"
ANCHOR_SNAPSHOT = VENDOR_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
_original_spec = importlib.util.spec_from_file_location
_installed = False


def resolve(path):
    path = Path(path)
    logical_e51 = REPO / "experiment/experiment51_dugsa"
    logical_anchors = REPO / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
    if path == logical_e51 or logical_e51 in path.parents:
        return LEGACY_E51 / path.relative_to(logical_e51)
    if path == logical_anchors or logical_anchors in path.parents:
        return ANCHOR_SNAPSHOT / path.relative_to(logical_anchors)
    if path == OLD_ROOT or OLD_ROOT in path.parents:
        path = REPO / path.relative_to(OLD_ROOT)
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
    is_e81 = raw is not None and (raw == E81_ROOT or E81_ROOT in raw.parents)
    path = raw if is_e81 else (resolve(raw) if raw is not None else None)
    spec = _original_spec(name, path, **kwargs)
    if path is not None and not is_e81 and REPO / "experiment" in path.parents:
        spec.loader = _ReadOnlyLoader(spec.loader)
    return spec


def load_base_runner():
    global _installed
    if not _installed:
        for path in (LEGACY_E51, LEGACY_E51 / "source", REPO / "experiment/experiment53_3_cadr_graph_fallback_fix"):
            if str(path) not in __import__("sys").path:
                __import__("sys").path.insert(0, str(path))
        importlib.util.spec_from_file_location = _spec
        _installed = True
    name = "experiment81_readonly_e744_runner"
    import sys
    if name in sys.modules:
        return sys.modules[name]
    base_path = REPO / "experiment/experiment74_4_soft_positive_dcr/run.py"
    spec = _spec(name, base_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

