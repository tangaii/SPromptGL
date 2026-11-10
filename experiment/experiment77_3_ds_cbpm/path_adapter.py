"""Read-only logical-path adapter; never creates links or edits historical files."""
from pathlib import Path
import hashlib
import importlib.util
import sys

REPO = Path('/data/adodas/SPromptGL-main')
BASE = REPO / 'experiment/experiment74_4_soft_positive_dcr'
OLD_BASE = Path('/model/spromptgl/experiment/experiment74_4_soft_positive_dcr')
SNAPSHOT = Path('/model/spromptgl/experiment/experiment72_mgdss/.runtime/vendor/root')
LEGACY = SNAPSHOT / 'experiment/experiment51_dugsa'
ANCHORS = SNAPSHOT / 'experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors'


def resolve_history(path):
    path = Path(path)
    if path == OLD_BASE or OLD_BASE in path.parents:
        return BASE / path.relative_to(OLD_BASE)
    return path


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_base_runner():
    # __file__ is the real relocated source. E74.4's own EXP consequently resolves
    # its config, model and tasks correctly, without patching its Python text.
    spec = importlib.util.spec_from_file_location('e773_readonly_e744', BASE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The original locations are broken existing links. These are byte-identical
    # archived artifacts, not reconstructed parameters. Only path globals change.
    import json
    expected_sources = {
        'dugsa.py': 'ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12',
        'source/layers.py': '4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a',
        'source/network.py': '5542f52d598e2f4d32aaad0c46b80c214fd562f74110a1d0ffcc619b9aefb3d1',
        'source/utils.py': '95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67',
    }
    for name, digest in expected_sources.items():
        assert sha_file(LEGACY / name) == digest, name
    reference = json.loads(Path('/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix/manifest.json').read_text())
    for name, digest in reference['anchor_sha256'].items():
        assert sha_file(ANCHORS / Path(name).name) == digest, name
    e61 = module.load_history().load_e69().load_e61()
    r = e61.load_base()
    r.LEGACY_EXP, r.LEGACY_SOURCE, r.ANCHORS = LEGACY, LEGACY / 'source', ANCHORS
    e61.ANCHORS = ANCHORS
    for p in (LEGACY, LEGACY / 'source'):
        sys.path.insert(0, str(p))
    return module
