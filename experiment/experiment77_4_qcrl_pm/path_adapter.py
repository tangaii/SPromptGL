"""SHA-checked read-only relocation of historical modules; no source rewriting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

REPO = Path('/data/adodas/SPromptGL-main')
HERE = Path(__file__).resolve().parent
BASE = REPO / 'experiment/experiment74_4_soft_positive_dcr'
E773 = REPO / 'experiment/experiment77_3_ds_cbpm'
OLD = Path('/model/spromptgl')
VENDOR = REPO / 'experiment/experiment72_mgdss/.runtime/vendor/root'
LEGACY = VENDOR / 'experiment/experiment51_dugsa'
ANCHORS = VENDOR / 'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors'
_original_spec = importlib.util.spec_from_file_location
_installed = False

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def resolve(path):
    p = Path(path)
    if p == OLD or OLD in p.parents:
        p = REPO / p.relative_to(OLD)
    for logical, actual in ((REPO/'experiment/experiment51_dugsa', LEGACY),
                            (REPO/'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors', ANCHORS)):
        if p == logical or logical in p.parents:
            return actual / p.relative_to(logical)
    return p

def verify_relocation():
    expected = json.loads((E773/'results/protected_after.json').read_text())
    checked = {}
    for old, checksum in expected.items():
        p = resolve(old)
        actual = sha_file(p)
        if actual != checksum:
            raise RuntimeError(f'Historical SHA mismatch: {old} -> {p}: {actual} != {checksum}')
        checked[str(p)] = actual
    for old, checksum in json.loads((E773/'manifest.json').read_text())['frozen_source_map'].items():
        p = resolve(old)
        assert sha_file(p) == checksum, str(p)
        checked[str(p)] = checksum
    reference = json.loads((REPO/'experiment/experiment53_3_cadr_graph_fallback_fix/manifest.json').read_text())
    for old, checksum in reference['anchor_sha256'].items():
        p = ANCHORS / Path(old).name
        assert sha_file(p) == checksum, str(p)
        checked[str(p)] = checksum
    for name, checksum in {
        'dugsa.py': 'ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12',
        'source/layers.py': '4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a',
        'source/network.py': '5542f52d598e2f4d32aaad0c46b80c214fd562f74110a1d0ffcc619b9aefb3d1',
        'source/utils.py': '95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67',
    }.items():
        p = LEGACY / name
        assert sha_file(p) == checksum, str(p)
        checked[str(p)] = checksum
    return checked

class _ReadOnlyLoader:
    def __init__(self, loader):
        self.loader = loader
    def create_module(self, spec):
        return self.loader.create_module(spec)
    def exec_module(self, module):
        self.loader.exec_module(module)
        # Adapt only path-valued globals after executing the exact historical
        # bytes. Config/source hash dictionaries retain their original keys.
        for k, v in list(vars(module).items()):
            if isinstance(v, Path):
                setattr(module, k, resolve(v))

def _spec(name, location=None, **kwargs):
    p = resolve(location) if location is not None else None
    spec = _original_spec(name, p, **kwargs)
    if p is not None and REPO/'experiment' in p.parents and HERE not in p.parents:
        spec.loader = _ReadOnlyLoader(spec.loader)
    return spec

def load_base_runner():
    global _installed
    if not _installed:
        verify_relocation()
        importlib.util.spec_from_file_location = _spec
        for p in (LEGACY, LEGACY/'source', REPO/'experiment/experiment53_3_cadr_graph_fallback_fix'):
            sys.path.insert(0, str(p))
        _installed = True
    name = 'qcrl_readonly_e744'
    if name in sys.modules:
        return sys.modules[name]
    spec = _spec(name, BASE/'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
