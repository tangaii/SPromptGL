"""Load the sealed SCMR optimization code with TADPOLE-independent dependencies."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


HISTORICAL_ROOT = (Path(__file__).resolve().parents[1] / "sources"
                   / "E82_4_6_G2_SCMR_DA_SRPE" / "code" / "history_base")
SOURCE_HASHES = {
    "config.json": "f6b56fa862c4e3ba2aae9e34863057f3c3968583fa2b9563cf947f2268b1f495",
    "common/spectral.py": "176d9ba7923beed39d583bfd9831ac99ac2e4e5204d1da7cef0bb727567a2e13",
    "common/optimization.py": "6071142cbbabc21dfe31e1a5cb561b214e0430d4b2e75827b3d7b773157591f0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def load_scmr_optimization():
    for relative_path, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((HISTORICAL_ROOT / relative_path).read_bytes()).hexdigest()
        require(actual == expected, f"Historical SCMR source hash mismatch for {relative_path}")

    # The numerical modules need only configuration and an assertion helper from
    # history.py. Supply that interface without invoking its ABIDE archive loader.
    package = ModuleType("_e831_scmr_source")
    package.__path__ = [str(HISTORICAL_ROOT / "common")]
    history = ModuleType(f"{package.__name__}.history")
    history.CFG = json.loads((HISTORICAL_ROOT / "config.json").read_text())
    history.require = require
    sys.modules[package.__name__] = package
    sys.modules[history.__name__] = history
    return importlib.import_module(f"{package.__name__}.optimization")
