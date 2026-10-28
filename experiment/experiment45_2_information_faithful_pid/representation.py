"""Frozen, nested, low-capacity representations for Experiment 45.2."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def fit_score_model(x, y):
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
    scaler = StandardScaler().fit(x)
    model.fit(scaler.transform(x), y)
    return scaler, model


def score_probability(scaler, model, x):
    # State 0 is ASD. Return P(ASD) independent of estimator class ordering.
    positive_col = int(np.flatnonzero(model.classes_ == 0)[0])
    return model.predict_proba(scaler.transform(x))[:, positive_col]


def oof_score(x, y, seed=0, n_splits=5):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = np.full(len(y), np.nan, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for train, test in cv.split(x, y):
        scaler, model = fit_score_model(x[train], y[train])
        out[test] = score_probability(scaler, model, x[test])
    if not np.isfinite(out).all():
        raise AssertionError("OOF disease score has missing or nonfinite predictions")
    return out


def quantile_states(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.quantile(scores, np.arange(1, int(k)) / int(k), method="linear")
    states = np.searchsorted(edges, scores, side="right").astype(np.int64)
    return states, edges.tolist()


def _canonical_kmeans(x_train, k, seed, structural=False):
    scaler = StandardScaler().fit(x_train)
    z_train = scaler.transform(x_train)
    if structural:
        pca = PCA(n_components=1, svd_solver="full", random_state=int(seed)).fit(z_train)
        component = pca.components_[0].copy()
        anchor = int(np.argmax(np.abs(component)))
        sign = 1.0 if component[anchor] >= 0 else -1.0
        pca.components_[0] *= sign
        z_fit = pca.transform(z_train)
    else:
        pca = None
        z_fit = z_train
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed), algorithm="lloyd").fit(z_fit)
    # Canonicalize arbitrary KMeans labels using ordered centroid coordinates.
    cent = km.cluster_centers_
    order = sorted(range(int(k)), key=lambda j: tuple(np.round(cent[j].tolist(), 12)))
    remap = np.empty(int(k), dtype=np.int64)
    for new, old in enumerate(order):
        remap[old] = new
    return scaler, pca, km, remap


def _transform_kmeans(state, x):
    scaler, pca, km, remap = state
    z = scaler.transform(x)
    if pca is not None:
        z = pca.transform(z)
    return remap[km.predict(z)].astype(np.int64)


def fit_unsupervised_codes(x_train, x_eval, k, seed, structural=False):
    state = _canonical_kmeans(np.asarray(x_train, dtype=np.float64), k, seed, structural)
    return _transform_kmeans(state, x_train), _transform_kmeans(state, x_eval), state


def make_pid_codes(x, y, method, k, seed):
    """Encode one outer training fold; B/C scores are cross-fitted within it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if method == "A_UNSUP_KMEANS":
        train_codes, _, state = fit_unsupervised_codes(x, x, k, seed, structural=False)
        return train_codes, {"actual_k": int(np.unique(train_codes).size), "edges": [], "structure_k": None}
    if method not in ("B_TASK_AWARE_SCORE", "C_TASK_STRUCTURE_HYBRID"):
        raise ValueError(method)
    scores = oof_score(x, y, seed=seed, n_splits=5)
    task_k = int(k) if method == "B_TASK_AWARE_SCORE" else 2
    task, edges = quantile_states(scores, task_k)
    if method == "B_TASK_AWARE_SCORE":
        return task, {"actual_k": int(np.unique(task).size), "edges": edges, "structure_k": None}
    structure_k = int(k) // 2
    struct, _, _ = fit_unsupervised_codes(x, x, structure_k, seed + 1009, structural=True)
    codes = task * structure_k + struct
    return codes, {"actual_k": int(np.unique(codes).size), "edges": edges, "structure_k": structure_k}


def _train_codes_for_diagnostic(x_train, y_train, method, k, seed):
    """Training-side diagnostic codes. Supervised score codes are themselves OOF."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    if method == "A_UNSUP_KMEANS":
        codes, _, _ = fit_unsupervised_codes(x_train, x_train, k, seed, structural=False)
        return codes, None
    # Four folds here are the inner-inner cross-fit. Every training row's score
    # comes from a model that did not see that row's label.
    scores = oof_score(x_train, y_train, seed=seed, n_splits=4)
    task_k = int(k) if method == "B_TASK_AWARE_SCORE" else 2
    task, edges = quantile_states(scores, task_k)
    state = {"edges": edges, "task_k": task_k}
    if method == "B_TASK_AWARE_SCORE":
        return task, state
    struct_k = int(k) // 2
    struct, _, struct_state = fit_unsupervised_codes(x_train, x_train, struct_k, seed + 1009, structural=True)
    state.update({"structure_k": struct_k, "structure_state": struct_state})
    return task * struct_k + struct, state


def _eval_codes(x_train, y_train, x_eval, method, k, seed, state):
    if method == "A_UNSUP_KMEANS":
        _, codes, _ = fit_unsupervised_codes(x_train, x_eval, k, seed, structural=False)
        return codes
    scaler, model = fit_score_model(x_train, y_train)
    score = score_probability(scaler, model, x_eval)
    task = np.searchsorted(np.asarray(state["edges"]), score, side="right").astype(np.int64)
    if method == "B_TASK_AWARE_SCORE":
        return task
    struct_k = int(state["structure_k"])
    struct = _transform_kmeans(state["structure_state"], x_eval)
    return task * struct_k + struct


def nested_codes_oof(x, y, method, k, seed, n_splits=5):
    """Nested OOF discrete codes for evaluating an encoded predictive probe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = np.full(len(y), -1, dtype=np.int64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for part, (train, test) in enumerate(cv.split(x, y)):
        train_codes, state = _train_codes_for_diagnostic(x[train], y[train], method, k, seed + 31 * part + 1)
        test_codes = _eval_codes(x[train], y[train], x[test], method, k, seed + 31 * part + 2, state)
        out[test] = test_codes
    if (out < 0).any():
        raise AssertionError("nested OOF codes incomplete")
    return out


def nested_pair_probabilities(x1, x2, y, method, k, seed, n_splits=5):
    """Return nested OOF ASD probabilities from one or two raw modalities."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = None if x2 is None else np.asarray(x2, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = np.full(len(y), np.nan, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for part, (train, test) in enumerate(cv.split(x1, y)):
        c1_train, state1 = _train_codes_for_diagnostic(x1[train], y[train], method, k, seed + 53 * part + 3)
        c1_test = _eval_codes(x1[train], y[train], x1[test], method, k, seed + 53 * part + 4, state1)
        if x2 is not None:
            c2_train, state2 = _train_codes_for_diagnostic(x2[train], y[train], method, k, seed + 53 * part + 5)
            c2_test = _eval_codes(x2[train], y[train], x2[test], method, k, seed + 53 * part + 6, state2)
            xt = np.column_stack((c1_train, c2_train))
            xv = np.column_stack((c1_test, c2_test))
        else:
            xt = c1_train.reshape(-1, 1)
            xv = c1_test.reshape(-1, 1)
        # One-hot categories before the fixed L2 logistic diagnostic model.
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        xt = enc.fit_transform(xt)
        xv = enc.transform(xv)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(xt, y[train])
        out[test] = score_probability_for_encoded(model, xv)
    if not np.isfinite(out).all():
        raise AssertionError("nested compressed probe did not produce all probabilities")
    return out


def nested_codes_for_modalities(xs, y, method, k, seed, n_splits=5):
    """Cache nested OOF codes once per modality so all six pairs reuse them."""
    y = np.asarray(y, dtype=np.int64)
    outputs = [np.full(len(y), -1, dtype=np.int64) for _ in xs]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for part, (train, test) in enumerate(cv.split(xs[0], y)):
        for modal, x in enumerate(xs):
            c_train, state = _train_codes_for_diagnostic(x[train], y[train], method, k,
                                                          seed + 53 * part + 17 * modal + 3)
            c_test = _eval_codes(x[train], y[train], x[test], method, k,
                                 seed + 53 * part + 17 * modal + 4, state)
            outputs[modal][test] = c_test
    if any((c < 0).any() for c in outputs):
        raise AssertionError("nested cached modality codes incomplete")
    return outputs


def encoded_probe_from_codes(codes, y, pair, seed=0, n_splits=5):
    """Cross-fit a low-capacity classifier on already nested, cached codes."""
    from sklearn.preprocessing import OneHotEncoder
    y = np.asarray(y, dtype=np.int64)
    pair = tuple(int(v) for v in pair)
    matrix = np.column_stack([codes[i] for i in pair])
    out = np.full(len(y), np.nan, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for train, test in cv.split(matrix, y):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        xt = enc.fit_transform(matrix[train])
        xv = enc.transform(matrix[test])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(xt, y[train])
        out[test] = score_probability_for_encoded(model, xv)
    if not np.isfinite(out).all():
        raise AssertionError("encoded OOF probe has incomplete probabilities")
    return out


def nested_all_probe_probabilities(xs, y, method, k, seed, pairs, n_splits=5):
    """One nested procedure, reusing each modality's fitted encoder per fold."""
    from sklearn.preprocessing import OneHotEncoder
    y = np.asarray(y, dtype=np.int64)
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    pairs = [tuple(int(v) for v in pair) for pair in pairs]
    single = [np.full(len(y), np.nan, dtype=np.float64) for _ in xs]
    joint = {pair: np.full(len(y), np.nan, dtype=np.float64) for pair in pairs}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for part, (train, test) in enumerate(cv.split(xs[0], y)):
        train_codes, test_codes = [], []
        for modal, x in enumerate(xs):
            ctrain, state = _train_codes_for_diagnostic(x[train], y[train], method, k,
                                                         seed + 53 * part + 17 * modal + 3)
            ctest = _eval_codes(x[train], y[train], x[test], method, k,
                                seed + 53 * part + 17 * modal + 4, state)
            train_codes.append(ctrain)
            test_codes.append(ctest)
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            xt = enc.fit_transform(ctrain.reshape(-1, 1))
            xv = enc.transform(ctest.reshape(-1, 1))
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
            model.fit(xt, y[train])
            single[modal][test] = score_probability_for_encoded(model, xv)
        for pair in pairs:
            xt_raw = np.column_stack([train_codes[i] for i in pair])
            xv_raw = np.column_stack([test_codes[i] for i in pair])
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            xt = enc.fit_transform(xt_raw)
            xv = enc.transform(xv_raw)
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
            model.fit(xt, y[train])
            joint[pair][test] = score_probability_for_encoded(model, xv)
    if any(not np.isfinite(p).all() for p in single) or any(not np.isfinite(p).all() for p in joint.values()):
        raise AssertionError("nested OOF encoded probes incomplete")
    return single, joint


def nested_many_probe_probabilities(xs, y, specs, seed, pairs, n_splits=5):
    """Nested probes for all frozen codes, reusing each modality's score models."""
    from sklearn.preprocessing import OneHotEncoder
    y = np.asarray(y, dtype=np.int64)
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    pairs = [tuple(int(v) for v in pair) for pair in pairs]
    outputs = {spec: ([np.full(len(y), np.nan) for _ in xs],
                      {pair: np.full(len(y), np.nan) for pair in pairs}) for spec in specs}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for part, (train, test) in enumerate(cv.split(xs[0], y)):
        train_codes = {spec: [] for spec in specs}
        test_codes = {spec: [] for spec in specs}
        for modal, x in enumerate(xs):
            xt, xv = x[train], x[test]
            score_train = oof_score(xt, y[train], seed + 43 * part + 7 * modal + 1, n_splits=4)
            scaler_score, model_score = fit_score_model(xt, y[train])
            score_test = score_probability(scaler_score, model_score, xv)
            state_by_spec = {}
            for spec in specs:
                method, k = spec
                k = int(k)
                if method == "A_UNSUP_KMEANS":
                    ctrain, _, state = fit_unsupervised_codes(xt, xt, k, seed + 43 * part + 7 * modal + 2)
                    ctest = _transform_kmeans(state, xv)
                else:
                    task_k = k if method == "B_TASK_AWARE_SCORE" else 2
                    ctrain, edges = quantile_states(score_train, task_k)
                    ctest = np.searchsorted(np.asarray(edges), score_test, side="right").astype(np.int64)
                    if method == "C_TASK_STRUCTURE_HYBRID":
                        struct_k = k // 2
                        if struct_k not in state_by_spec:
                            _, _, state_by_spec[struct_k] = fit_unsupervised_codes(
                                xt, xt, struct_k, seed + 43 * part + 7 * modal + 3, structural=True)
                        struct_state = state_by_spec[struct_k]
                        struct_train = _transform_kmeans(struct_state, xt)
                        struct_test = _transform_kmeans(struct_state, xv)
                        ctrain = ctrain * struct_k + struct_train
                        ctest = ctest * struct_k + struct_test
                train_codes[spec].append(ctrain)
                test_codes[spec].append(ctest)
        for spec in specs:
            single, joint = outputs[spec]
            for modal in range(len(xs)):
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                a = enc.fit_transform(train_codes[spec][modal].reshape(-1, 1))
                b = enc.transform(test_codes[spec][modal].reshape(-1, 1))
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
                model.fit(a, y[train])
                single[modal][test] = score_probability_for_encoded(model, b)
            for pair in pairs:
                a_raw = np.column_stack([train_codes[spec][i] for i in pair])
                b_raw = np.column_stack([test_codes[spec][i] for i in pair])
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                a, b = enc.fit_transform(a_raw), enc.transform(b_raw)
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
                model.fit(a, y[train])
                joint[pair][test] = score_probability_for_encoded(model, b)
    for spec, (single, joint) in outputs.items():
        if any(not np.isfinite(p).all() for p in single) or any(not np.isfinite(p).all() for p in joint.values()):
            raise AssertionError(f"nested encoded probes incomplete: {spec}")
    return outputs


def score_probability_for_encoded(model, x):
    positive_col = int(np.flatnonzero(model.classes_ == 0)[0])
    return model.predict_proba(x)[:, positive_col]


def state_diagnostics(codes, declared_k):
    codes = np.asarray(codes, dtype=np.int64)
    counts = np.bincount(codes, minlength=int(declared_k))
    p = counts[counts > 0] / max(int(counts.sum()), 1)
    entropy = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    return {"declared_k": int(declared_k), "occupied_k": int(np.count_nonzero(counts)),
            "occupancy": counts.tolist(), "minimum_occupied_count": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
            "entropy_bit": entropy}
