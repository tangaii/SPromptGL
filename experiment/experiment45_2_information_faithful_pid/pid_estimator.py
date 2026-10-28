"""Discrete two-source BROJA PID with an auditable convex coupling solve."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import xlogy


def _mi(joint: np.ndarray) -> float:
    p = np.asarray(joint, dtype=np.float64)
    total = float(p.sum())
    if total <= 0 or not np.isfinite(p).all() or np.min(p) < -1e-12:
        raise ValueError("invalid probability table")
    p = np.maximum(p / total, 0.0)
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    den = px @ py
    ratio = np.divide(p, den, out=np.ones_like(p), where=den > 0)
    return float(np.sum(xlogy(p, ratio)) / np.log(2.0))


def information_terms(p: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    i1 = _mi(p.sum(axis=1))
    i2 = _mi(p.sum(axis=0))
    ijoint = _mi(p.transpose(2, 0, 1).reshape(p.shape[2], -1))
    return i1, i2, ijoint


def _constraints(p: np.ndarray, tol: float = 1e-12):
    n1, n2, ny = p.shape
    p1y = p.sum(axis=1)
    p2y = p.sum(axis=0)
    py = p.sum(axis=(0, 1))
    cells = [(i, j, y) for i in range(n1) for j in range(n2) for y in range(ny)
             if p1y[i, y] > tol and p2y[j, y] > tol]
    if not cells:
        raise ValueError("empty feasible support")
    lookup = {cell: k for k, cell in enumerate(cells)}
    rows, targets = [], []
    for i in range(n1):
        for y in range(ny):
            row = np.zeros(len(cells), dtype=np.float64)
            for j in range(n2):
                if (i, j, y) in lookup:
                    row[lookup[(i, j, y)]] = 1.0
            if row.any():
                rows.append(row)
                targets.append(p1y[i, y])
    for j in range(n2):
        for y in range(ny):
            row = np.zeros(len(cells), dtype=np.float64)
            for i in range(n1):
                if (i, j, y) in lookup:
                    row[lookup[(i, j, y)]] = 1.0
            if row.any():
                rows.append(row)
                targets.append(p2y[j, y])
    full_a = np.asarray(rows, dtype=np.float64)
    full_b = np.asarray(targets, dtype=np.float64)
    _, r, piv = qr(full_a.T, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    cutoff = max(full_a.shape) * np.finfo(float).eps * (diag.max() if diag.size else 1.0)
    rank = int(np.sum(diag > cutoff))
    keep = np.sort(piv[:rank])
    return cells, full_a[keep], full_b[keep], p1y, p2y, py, rank


def _unpack(x: np.ndarray, cells, shape) -> np.ndarray:
    q = np.zeros(shape, dtype=np.float64)
    ijy = np.asarray(cells, dtype=np.int64)
    q[tuple(ijy.T)] = x
    return q


def _objective_factory(cells, shape, py):
    n1, n2, ny = shape
    ijy = np.asarray(cells, dtype=np.int64)
    log2 = np.log(2.0)

    def objective(x):
        q = _unpack(x, cells, shape)
        qxy = q.sum(axis=2)
        den = qxy[:, :, None] * py[None, None, :]
        ratio = np.divide(q, den, out=np.ones_like(q), where=den > 0)
        value = float(np.sum(xlogy(q, ratio)) / log2)
        grad = np.log(np.maximum(q[tuple(ijy.T)], 1e-300) /
                      np.maximum(qxy[ijy[:, 0], ijy[:, 1]] * py[ijy[:, 2]], 1e-300)) / log2
        return value, grad

    return objective


def _interior_start(p, cells):
    py = p.sum(axis=(0, 1))
    p1y = p.sum(axis=1)
    p2y = p.sum(axis=0)
    return np.asarray([p1y[i, y] * p2y[j, y] / py[y] for i, j, y in cells])


def _minimum_joint_mi(p: np.ndarray, tol: float = 1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    cells, a, b, p1y, p2y, py, rank = _constraints(p, tol)
    shape = p.shape
    objective = _objective_factory(cells, shape, py)
    x0 = np.asarray([p[i, j, y] for i, j, y in cells], dtype=np.float64)
    result = minimize(objective, x0, method="SLSQP", jac=True,
                      bounds=Bounds(np.zeros(len(cells)), np.ones(len(cells))),
                      constraints=[{"type": "eq", "fun": lambda x: a @ x - b,
                                    "jac": lambda x: a}],
                      options={"ftol": 1e-12, "maxiter": 2000, "disp": False})
    if not np.isfinite(result.x).all():
        raise FloatingPointError("SLSQP returned non-finite coupling")
    if float(result.x.min()) < -1e-8 or float(result.x.max()) > 1.0 + 1e-8:
        raise FloatingPointError("SLSQP returned coupling outside probability bounds")
    x = np.where(result.x < 0.0, 0.0, result.x)
    q = _unpack(x, cells, shape)
    marginal_error = max(float(np.max(np.abs(q.sum(axis=1) - p1y))),
                         float(np.max(np.abs(q.sum(axis=0) - p2y))),
                         float(np.max(np.abs(q.sum(axis=(0, 1)) - py))),
                         abs(float(q.sum()) - 1.0))
    recomputed = information_terms(q)[2]
    objective_delta = abs(float(result.fun) - recomputed)
    meta = {
        "success": bool(result.success),
        "status": str(result.message),
        "iterations": int(result.nit),
        "objective_returned_bit": float(result.fun),
        "objective_recomputed_bit": recomputed,
        "objective_delta_bit": objective_delta,
        "marginal_error": marginal_error,
        "support_cells": len(cells),
        "constraint_rank": rank,
    }
    return q, meta


def estimate_distribution_pid(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3 or np.min(p) < -1e-12 or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("expected a finite nonnegative three-dimensional probability table")
    p = np.maximum(p, 0.0)
    p /= p.sum()
    q, solver = _minimum_joint_mi(p)
    i1, i2, ijoint = information_terms(p)
    iq = information_terms(q)[2]
    redundancy = i1 + i2 - iq
    unique1 = iq - i2
    unique2 = iq - i1
    synergy = ijoint - iq
    parts = np.asarray([redundancy, unique1, unique2, synergy])
    identity = max(abs(float(parts.sum() - ijoint)),
                   abs(float(redundancy + unique1 - i1)),
                   abs(float(redundancy + unique2 - i2)))
    minimum_component = float(parts.min())
    py = p.sum(axis=(0, 1))
    hy = float(-np.sum(xlogy(py, py)) / np.log(2.0))
    bound_residual = max(0.0, float(redundancy - min(i1, i2)), float(ijoint - hy),
                         float(i1 - hy), float(i2 - hy))
    valid = (solver["success"] and solver["marginal_error"] <= 1e-7 and
             solver["objective_delta_bit"] <= 1e-8 and identity <= 1e-8 and
             minimum_component >= -1e-7 and bound_residual <= 1e-7 and np.isfinite(parts).all())
    return {
        "R": float(redundancy), "U1": float(unique1), "U2": float(unique2),
        "S": float(synergy), "I1": float(i1), "I2": float(i2),
        "Ijoint": float(ijoint), "Ijoint_min": float(iq),
        "identity_residual": float(identity), "minimum_component": minimum_component,
        "H_Y": hy, "information_bound_residual": bound_residual,
        "redundancy_fraction": float(redundancy / ijoint) if ijoint > 1e-12 else 0.0,
        "solver": solver, "valid": bool(valid), "distribution": p, "q_min": q,
    }


def estimate_pid(x1, x2, y, k1=None, k2=None, ny=None):
    x1 = np.asarray(x1, dtype=np.int64).reshape(-1)
    x2 = np.asarray(x2, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(x1) == 0 or not (len(x1) == len(x2) == len(y)):
        raise ValueError("PID arrays must be nonempty and have equal length")
    if min(x1.min(), x2.min(), y.min()) < 0:
        raise ValueError("discrete state labels must be nonnegative")
    k1 = int(k1 if k1 is not None else x1.max() + 1)
    k2 = int(k2 if k2 is not None else x2.max() + 1)
    ny = int(ny if ny is not None else y.max() + 1)
    if x1.max() >= k1 or x2.max() >= k2 or y.max() >= ny:
        raise ValueError("state label outside declared cardinality")
    counts = np.zeros((k1, k2, ny), dtype=np.float64)
    np.add.at(counts, (x1, x2, y), 1.0)
    return estimate_distribution_pid(counts / counts.sum())


def independent_trust_constr(p: np.ndarray):
    """Independent Hessian-based reference solve used only by the audit."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    cells, a, b, p1y, p2y, py, rank = _constraints(p)
    shape = p.shape
    objective = _objective_factory(cells, shape, py)
    ijy = np.asarray(cells, dtype=np.int64)

    def hessian(x):
        q = _unpack(x, cells, shape)
        qxy = q.sum(axis=2)
        qcell = np.maximum(q[tuple(ijy.T)], 1e-12)
        h = np.diag(1.0 / qcell)
        groups = {}
        for row, (i, j, y) in enumerate(cells):
            groups.setdefault((i, j), []).append(row)
        for (i, j), rows in groups.items():
            h[np.ix_(rows, rows)] -= 1.0 / max(float(qxy[i, j]), 1e-12)
        return h / np.log(2.0)

    start = np.maximum(_interior_start(p, cells), 1e-10)
    result = minimize(objective, start, method="trust-constr", jac=True, hess=hessian,
                      constraints=[LinearConstraint(a, b, b)],
                      bounds=Bounds(np.full(len(cells), 1e-12), np.ones(len(cells))),
                      options={"gtol": 1e-10, "xtol": 1e-12, "barrier_tol": 1e-12,
                               "maxiter": 2000, "verbose": 0})
    # trust-constr can leave tiny negative feasibility noise near the lower
    # bound.  Clip only numerical noise before recomputing information; the
    # SLSQP production solve above remains the primary estimate.
    q = np.maximum(_unpack(result.x, cells, shape), 0.0)
    if q.sum() > 0:
        q = q / q.sum()
    err = max(float(np.max(np.abs(q.sum(axis=1) - p1y))),
              float(np.max(np.abs(q.sum(axis=0) - p2y))),
              float(np.max(np.abs(q.sum(axis=(0, 1)) - py))))
    return q, {"success": bool(result.success), "status": str(result.message),
               "objective_bit": information_terms(q)[2], "marginal_error": err,
               "iterations": int(result.nit), "support_cells": len(cells),
               "constraint_rank": rank}
