"""Read-only compatibility adapter for the audited E74.4-G2/E61 loader.

This file intentionally imports the locked E76.6 compatibility layer.  The
experiment code never mutates the historical modules or their files.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys

E766 = Path('/model/spromptgl/experiment/experiment76_6_bcspc')
LOCKS = {
    'run.py': '7f6a99eb61134c3820dae74636314abab987cb84c7eb14be469189b6bf4edf94',
    'bcspc.py': 'fadce3e5c94f6bd21525fa57616d04a60bbfc90ccdd5bae0414fb2610a7e2804',
    'audit.py': '919ee84da8abe467228bade11a344f142227dee9e51950b209b59316bb1a6e73',
    'config.json': '397ae6d4dfe820e89aa1e1766f160ae0263d36557f008087bf2b9dd8586d873e',
}

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

for name, expected in LOCKS.items():
    actual = hashlib.sha256((E766 / name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'E76.6 source identity mismatch: {name}: {actual}')

legacy_module = load_file('bcspc', E766 / 'bcspc.py')
legacy = load_file('e766_readonly_for_e79', E766 / 'run.py')
