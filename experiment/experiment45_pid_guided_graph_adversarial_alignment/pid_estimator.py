"""Auditable discrete two-source I_min PID with the label marginal preserved."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import xlogy


def _mi(joint):
    p = np.asarray(joint, dtype=np.float64)
    p = p / p.sum()
    a = p.sum(axis=1, keepdims=True)
    b = p.sum(axis=0, keepdims=True)
    den = a @ b
    return float(np.sum(xlogy(p, p / np.maximum(den, 1e-300))) / np.log(2.0))


def information_terms(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    ix1y = _mi(p.sum(axis=1))
    ix2y = _mi(p.sum(axis=0))
    iyx1x2 = _mi(p.transpose(2, 0, 1).reshape(p.shape[2], -1))
    return ix1y, ix2y, iyx1x2


def _minimum_joint_mi(p, tol=1e-12):
    """Minimize I_Q((X1,X2);Y) preserving both source-target marginals.

    This is the small discrete convex program used by the upstream PID example.
    The upstream source divides its shared X1-X2 marginal equally across Y;
    that is only correct for a uniform target. Here the fixed empirical P(Y)
    is included, which is required for the imbalanced ABIDE folds.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    n1, n2, ny = p.shape
    p1y = p.sum(axis=1)
    p2y = p.sum(axis=0)
    py = p.sum(axis=(0, 1))
    cells = [(i, j, y) for i in range(n1) for j in range(n2) for y in range(ny)
             if p1y[i, y] > tol and p2y[j, y] > tol]
    if not cells:
        raise ValueError('empty feasible support')
    lookup = {cell: q for q, cell in enumerate(cells)}
    rows, targets = [], []
    for i in range(n1):
        for y in range(ny):
            row = np.zeros(len(cells))
            for j in range(n2):
                if (i, j, y) in lookup:
                    row[lookup[(i, j, y)]] = 1.0
            if row.any():
                rows.append(row); targets.append(p1y[i, y])
    for j in range(n2):
        for y in range(ny):
            row = np.zeros(len(cells))
            for i in range(n1):
                if (i, j, y) in lookup:
                    row[lookup[(i, j, y)]] = 1.0
            if row.any():
                rows.append(row); targets.append(p2y[j, y])
    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(targets, dtype=np.float64)
    # Drop linearly dependent equalities while preserving the feasible target.
    _, r, piv = qr(a.T, mode='economic', pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int(np.sum(diag > max(a.shape) * np.finfo(float).eps * (diag.max() if diag.size else 1.0)))
    keep = np.sort(piv[:rank])
    a, b = a[keep], b[keep]
    x0 = np.asarray([p[i, j, y] for i, j, y in cells], dtype=np.float64)

    def objective(x):
        q = np.zeros((n1, n2, ny), dtype=np.float64)
        q[tuple(np.asarray(cells).T)] = x
        qxy = q.sum(axis=2)
        denom = qxy[:, :, None] * py[None, None, :]
        ratio = np.maximum(q, 1e-300) / np.maximum(denom, 1e-300)
        val = np.sum(xlogy(q, ratio)) / np.log(2.0)
        grad = np.asarray([np.log(max(q[i, j, y], 1e-300) /
                                       max(qxy[i, j] * py[y], 1e-300)) / np.log(2.0)
                           for i, j, y in cells])
        return float(val), grad

    result = minimize(lambda x: objective(x), x0, method='SLSQP', jac=True,
                      bounds=[(0.0, 1.0)] * len(cells),
                      constraints=[{'type': 'eq', 'fun': lambda x: a @ x - b,
                                    'jac': lambda x: a}],
                      options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False})
    q = np.zeros((n1, n2, ny), dtype=np.float64)
    for value, cell in zip(result.x, cells):
        q[cell] = max(0.0, value)
    marginal_error = max(float(np.max(np.abs(q.sum(axis=1) - p1y))),
                         float(np.max(np.abs(q.sum(axis=0) - p2y))),
                         abs(float(q.sum()) - 1.0))
    mi = _mi(q.transpose(2, 0, 1).reshape(ny, -1))
    return q, {'success': bool(result.success), 'status': str(result.message),
               'iterations': int(result.nit), 'objective_bit': mi,
               'marginal_error': marginal_error, 'support_cells': len(cells)}


def estimate_pid(x1, x2, y, k1=None, k2=None):
    x1 = np.asarray(x1, dtype=np.int64).reshape(-1)
    x2 = np.asarray(x2, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if not (len(x1) == len(x2) == len(y)) or len(y) == 0:
        raise ValueError('PID arrays must be nonempty and have equal length')
    k1 = int(k1 if k1 is not None else x1.max() + 1)
    k2 = int(k2 if k2 is not None else x2.max() + 1)
    ny = int(y.max() + 1)
    counts = np.zeros((k1, k2, ny), dtype=np.float64)
    np.add.at(counts, (x1, x2, y), 1.0)
    p = counts / counts.sum()
    q, solver = _minimum_joint_mi(p)
    i1, i2, ijoint = information_terms(p)
    iq = information_terms(q)[2]
    redundancy = i1 + i2 - iq
    unique1 = iq - i2
    unique2 = iq - i1
    synergy = ijoint - iq
    parts = np.asarray([redundancy, unique1, unique2, synergy], dtype=np.float64)
    residual = float(parts.sum() - ijoint)
    identity = max(abs(residual), abs(redundancy + unique1 - i1),
                   abs(redundancy + unique2 - i2))
    minimum_component = float(parts.min())
    valid = (solver['success'] and solver['marginal_error'] <= 1e-7 and
             identity <= 1e-8 and minimum_component >= -1e-7 and
             np.isfinite(parts).all())
    return {
        'R': float(redundancy), 'U1': float(unique1), 'U2': float(unique2),
        'S': float(synergy), 'I1': float(i1), 'I2': float(i2),
        'Ijoint': float(ijoint), 'Ijoint_min': float(iq),
        'identity_residual': identity, 'minimum_component': minimum_component,
        'redundancy_fraction': float(redundancy / ijoint) if ijoint > 1e-12 else 0.0,
        'solver': solver, 'valid': bool(valid), 'distribution': p, 'q_min': q,
    }


def synthetic_audit(seed=45000, n=4096):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    cases = {
        'redundant': (a, a.copy(), a.copy(), 'R'),
        'unique_x1': (a, b, a.copy(), 'U1'),
        'xor_synergy': (a, b, np.bitwise_xor(a, b), 'S'),
        'independent': (a, b, rng.integers(0, 2, n), 'zero'),
    }
    out = {}
    for name, (x1, x2, y, expected) in cases.items():
        z = estimate_pid(x1, x2, y, 2, 2)
        if not z['valid']:
            raise AssertionError(f'{name}: invalid PID {z}')
        if expected == 'R' and not (z['R'] > .8 and abs(z['U1']) < .08 and abs(z['U2']) < .08 and abs(z['S']) < .08):
            raise AssertionError(f'{name}: not redundancy dominated {z}')
        if expected == 'U1' and not (z['U1'] > .8 and abs(z['R']) < .08 and abs(z['U2']) < .08 and abs(z['S']) < .08):
            raise AssertionError(f'{name}: not U1 dominated {z}')
        if expected == 'S' and not (z['S'] > .8 and abs(z['R']) < .08 and abs(z['U1']) < .08 and abs(z['U2']) < .08):
            raise AssertionError(f'{name}: not synergy dominated {z}')
        if expected == 'zero' and max(z[k] for k in ['R', 'U1', 'U2', 'S']) > .08:
            raise AssertionError(f'{name}: independent variables show signal {z}')
        out[name] = {k: z[k] for k in ['R', 'U1', 'U2', 'S', 'I1', 'I2', 'Ijoint',
                                         'identity_residual', 'minimum_component', 'solver']}
    return out
