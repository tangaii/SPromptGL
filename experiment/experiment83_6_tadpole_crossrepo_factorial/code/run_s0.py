"""Start the registered sp positive control behind the E83.6 training gate."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e836


def main() -> None:
    run_e836.require_training_gate()
    os.environ.setdefault("E836_SP_RUN_ROOT", str(ROOT / "groups/S0/sp"))
    sys.path.insert(0, str(CODE / "sp_runner"))
    import run as sp_run
    sp_run.run(SimpleNamespace(variant="mmgpl_class_graph_sup", epochs=None, folds_to_run=None))


if __name__ == "__main__":
    main()
