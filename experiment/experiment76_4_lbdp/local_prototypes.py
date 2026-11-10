"""Patient-level local boundary prototype evidence for E76.4."""
from __future__ import annotations

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def role_masks(data, ids, base_roles):
    """Return E76.2 reliable/hard roles plus correct boundary reference pools."""
    ids = np.asarray(ids, dtype=int)
    y, p, pred, stability = data["y"], data["prob"], data["pred"], data["stability"]
    r_asd = ids[(y[ids] == 0) & (pred[ids] == 0) & (p[ids, 0] >= .70) & (stability[ids] >= .90)]
    r_td = ids[(y[ids] == 1) & (pred[ids] == 1) & (p[ids, 1] >= .70) & (stability[ids] >= .90)]
    h_asd = ids[(y[ids] == 0) & (pred[ids] == 1) & (p[ids, 1] >= .60) & (stability[ids] >= .90)]
    h_td = ids[(y[ids] == 1) & (pred[ids] == 0) & (p[ids, 0] >= .60) & (stability[ids] >= .90)]
    b_asd = ids[(y[ids] == 0) & (pred[ids] == 0) & (p[ids, 0] >= .50) & (p[ids, 0] < .70)]
    b_td = ids[(y[ids] == 1) & (pred[ids] == 1) & (p[ids, 1] >= .50) & (p[ids, 1] < .70)]
    hard = [h_asd if len(h_asd) >= 3 else np.array([], dtype=int),
            h_td if len(h_td) >= 3 else np.array([], dtype=int)]
    if base_roles is not None:
        reference = base_roles(data, ids)
        ours = [r_asd, r_td, h_asd, h_td]
        for actual, mask in zip(ours, reference):
            if not np.array_equal(actual, np.flatnonzero(mask)):
                raise RuntimeError("local prototype role masks differ from audited E76.2 definitions")
    return {"reliable": [r_asd, r_td], "hard": hard, "hard_raw": [h_asd, h_td],
            "boundary": [b_asd, b_td],
            "hard_candidate_counts": [int(len(h_asd)), int(len(h_td))],
            "hard_available": [bool(len(hard[0])), bool(len(hard[1]))]}


def _nearest(q, candidates, hn):
    if len(candidates) == 0:
        return None, None
    sims = hn[candidates] @ hn[q]
    j = int(np.argmax(sims))
    return int(candidates[j]), float(sims[j])


def build_local_memory(data, support, h_global, sites, base_roles, shuffle_seed=None):
    """Build role-balanced prototypes and one local triple per true hard case."""
    support = np.asarray(support, dtype=int)
    h = unit(h_global)
    roles = role_masks(data, support, base_roles)
    reliable, hard, boundary = roles["reliable"], roles["hard"], roles["boundary"]
    centers, center_valid = [], []
    for members in reliable:
        centers.append(unit(h[members].mean(0)) if len(members) else np.zeros(h.shape[1]))
        center_valid.append(bool(len(members)))

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    records_by_class = [[], []]
    fallbacks = [0, 0]
    for cls in (0, 1):
        hard_ids = hard[cls]
        same_pool = boundary[cls] if len(boundary[cls]) else reliable[cls]
        if len(boundary[cls]) == 0:
            fallbacks[cls] = int(len(hard_ids))
        other_pool = reliable[1 - cls]
        if len(hard_ids) < 3 or len(same_pool) == 0 or len(other_pool) == 0:
            continue
        triples = []
        for q in hard_ids:
            same_id, same_sim = _nearest(int(q), same_pool[same_pool != q], h)
            other_id, other_sim = _nearest(int(q), other_pool[other_pool != q], h)
            if same_id is None or other_id is None:
                continue
            triples.append({"hard_id": int(q), "same_id": same_id, "other_id": other_id,
                            "same_similarity": same_sim, "other_similarity": other_sim,
                            "hard_margin": float(data["prob"][q, cls] - data["prob"][q, 1-cls]),
                            "same_site": bool(sites[q] == sites[same_id]),
                            "other_site": bool(sites[q] == sites[other_id]),
                            "fallback": bool(len(boundary[cls]) == 0)})
        if rng is not None and len(triples) > 1:
            same_refs = np.asarray([r["same_id"] for r in triples], dtype=int)
            other_refs = np.asarray([r["other_id"] for r in triples], dtype=int)
            perm_same = rng.permutation(same_refs)
            perm_other = rng.permutation(other_refs)
            for j, record in enumerate(triples):
                record["same_id"] = int(perm_same[j])
                record["other_id"] = int(perm_other[j])
                record["same_similarity"] = float(h[record["hard_id"]] @ h[record["same_id"]])
                record["other_similarity"] = float(h[record["hard_id"]] @ h[record["other_id"]])
                record["same_site"] = bool(sites[record["hard_id"]] == sites[record["same_id"]])
                record["other_site"] = bool(sites[record["hard_id"]] == sites[record["other_id"]])
                record["shuffled"] = True
            changed = sum((a["same_id"] != b) or (a["other_id"] != c)
                          for a, b, c in zip(triples, same_refs, other_refs))
            for record in triples:
                record["permutation_changed_n"] = int(changed)
                record["permutation_total_n"] = int(len(triples))
        else:
            for record in triples:
                record["shuffled"] = False
                record["permutation_changed_n"] = 0
                record["permutation_total_n"] = int(len(triples))
        records_by_class[cls] = triples

    return {"centers": np.asarray(centers), "center_valid": np.asarray(center_valid, dtype=bool),
            "reliable_ids": reliable, "hard_ids": hard, "boundary_ids": boundary,
            "hard_raw_counts": roles["hard_candidate_counts"], "hard_available": roles["hard_available"],
            "records": records_by_class, "fallback_counts": fallbacks,
            "support_ids": support.copy(), "shuffle_seed": shuffle_seed}


def _softmax(values, tau):
    values = np.asarray(values, dtype=float) / float(tau)
    values = values - np.max(values)
    w = np.exp(values)
    return w / max(w.sum(), 1e-12)


def local_evidence(query_h, memory, h_global, tau=.10, top_k=3):
    """Fixed 12 slots: 6 cosine evidence values and 6 validity masks."""
    h = unit(h_global)
    q = unit(np.asarray(query_h, dtype=float))
    n = len(q)
    values = np.zeros((n, 6), dtype=float)
    masks = np.r_[memory["center_valid"], memory["hard_available"],
                  [bool(len(memory["records"][0]) >= 3), bool(len(memory["records"][1]) >= 3)]].astype(float)
    if np.any(memory["center_valid"]):
        for c in (0, 1):
            if memory["center_valid"][c]:
                values[:, c] = q @ memory["centers"][c]
    for c in (0, 1):
        hard_ids = memory["hard_ids"][c]
        records = memory["records"][c]
        if len(hard_ids) < 3 or len(records) < 3:
            masks[2 + c] = 0.0
            masks[4 + c] = 0.0
            continue
        record_by_id = {r["hard_id"]: r for r in records}
        usable_ids = np.asarray([i for i in hard_ids if int(i) in record_by_id], dtype=int)
        if len(usable_ids) < 3:
            masks[2 + c] = 0.0
            masks[4 + c] = 0.0
            continue
        for i, query in enumerate(q):
            sim = h[usable_ids] @ query
            take = np.argsort(sim)[-min(top_k, len(sim)):][::-1]
            selected = usable_ids[take]
            weights = _softmax(sim[take], tau)
            values[i, 2 + c] = float(weights @ sim[take])
            dirs = []
            for hard_id in selected:
                rec = record_by_id[int(hard_id)]
                dirs.append(h[rec["same_id"]] - h[rec["other_id"]])
            direction = unit(np.sum(np.asarray(dirs) * weights[:, None], axis=0))
            values[i, 4 + c] = float(query @ direction)
    values *= masks[None, :]
    return np.c_[values, np.broadcast_to(masks, (n, 6))]


def local_diagnostics(data, support, memory, h_global, sites, tau=.10, seed=764000):
    """Mechanism counts, site matching, and bootstrap pair-direction stability."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, name in ((0, "ASD_to_TD"), (1, "TD_to_ASD")):
        hard_ids = memory["hard_ids"][c]
        records = memory["records"][c]
        directions = []
        for r in records:
            directions.append(unit(h_global[r["same_id"]] - h_global[r["other_id"]]))
        boot = []
        if len(directions) >= 2:
            full = unit(np.mean(directions, axis=0))
            for _ in range(200):
                ix = rng.integers(0, len(directions), size=len(directions))
                boot.append(float(unit(np.mean(np.asarray(directions)[ix], axis=0)) @ full))
        rows.append({"direction": name, "raw_hard_count": memory["hard_raw_counts"][c],
                     "usable_hard_count": len(hard_ids), "local_pair_count": len(records),
                     "available": len(records) >= 3, "fallback_count": memory["fallback_counts"][c],
                     "same_site_fraction": float(np.mean([r["same_site"] for r in records])) if records else "",
                     "other_site_fraction": float(np.mean([r["other_site"] for r in records])) if records else "",
                     "mean_hard_same_similarity": float(np.mean([r["same_similarity"] for r in records])) if records else "",
                     "mean_hard_other_similarity": float(np.mean([r["other_similarity"] for r in records])) if records else "",
                     "bootstrap_direction_cosine_mean": float(np.mean(boot)) if boot else "",
                     "bootstrap_replicates": len(boot),
                     "pair_permutation_changed_fraction": float(np.mean([r.get("permutation_changed_n", 0) / max(1, r.get("permutation_total_n", 1)) for r in records])) if records else ""})
    return rows
