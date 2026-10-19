"""Per-process runtime, set BEFORE constructing each fold model.
No new optimizer or parameters. GFR's existing GraphLearn3 remains trainable.
"""
from e25_gate import GraphCalibrator
_ACTIVE=None

def configure(config,fold,x,train,site):
    global _ACTIVE
    _ACTIVE=GraphCalibrator(config,fold,x,train,site)

def calibrate(adj,parts,train_index):
    assert _ACTIVE is not None,'Call e25_runtime.configure first'
    return _ACTIVE(adj)

def set_enabled(enabled):
    assert _ACTIVE is not None
    _ACTIVE.enabled=bool(enabled)

def capture_stats(enabled):
    assert _ACTIVE is not None
    _ACTIVE.capture_stats=bool(enabled)

def stats():
    return {} if _ACTIVE is None else {**_ACTIVE.last,**_ACTIVE.data['metadata']}

