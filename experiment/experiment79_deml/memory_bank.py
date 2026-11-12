"""Dynamic error-event memory used by Experiment 79.

The bank is deliberately a small Python-side state machine.  Every tensor is
detached before it is stored, and snapshots are immutable copies for an epoch.
No test patient is ever passed to this class by the runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import deque

import torch


TYPE_ASD_TO_TD = "ASD_to_TD"
TYPE_TD_TO_ASD = "TD_to_ASD"
TYPES = (TYPE_ASD_TO_TD, TYPE_TD_TO_ASD)


def _clone_tensor(x):
    return x.detach().cpu().float().clone()


def _record_copy(rec):
    out = {}
    for key, value in rec.items():
        if torch.is_tensor(value):
            out[key] = _clone_tensor(value)
        elif isinstance(value, deque):
            out[key] = deque(value, maxlen=value.maxlen)
        else:
            out[key] = copy.deepcopy(value)
    return out


class ErrorMemoryBank:
    """Two deterministic, capacity-limited stores of legal training errors."""

    def __init__(self, capacity_per_type=64, top_k=8, site_cap=2,
                 remove_after_correct_streak=10):
        self.capacity_per_type = int(capacity_per_type)
        self.top_k = int(top_k)
        self.site_cap = int(site_cap)
        self.remove_after_correct_streak = int(remove_after_correct_streak)
        self.records = {TYPE_ASD_TO_TD: {}, TYPE_TD_TO_ASD: {}}
        self.epoch = 0
        self.history = []

    @staticmethod
    def _type(true_label, wrong_label):
        if int(true_label) == 0 and int(wrong_label) == 1:
            return TYPE_ASD_TO_TD
        if int(true_label) == 1 and int(wrong_label) == 0:
            return TYPE_TD_TO_ASD
        raise ValueError(f"error event must flip ASD/TD, got {true_label}->{wrong_label}")

    @staticmethod
    def _priority(rec, epoch):
        margin = max(float(rec.get("wrong_margin", 0.0)), 0.0)
        h = 1.0 - math.exp(-margin)
        recent = list(rec.get("recent_errors", ()))
        f = float(sum(recent[-10:])) / 10.0
        age = max(0.0, float(epoch - int(rec.get("last_error_epoch", epoch))))
        r = math.exp(-age / 10.0)
        return 0.5 * h + 0.3 * f + 0.2 * r

    @classmethod
    def _sort_key(cls, rec, epoch):
        # Python's ascending sort on this tuple gives deterministic descending
        # priority, then newer error, then smaller patient ID.
        return (-float(cls._priority(rec, epoch)),
                -int(rec.get("last_error_epoch", 0)),
                int(rec["patient_id"]))

    def snapshot(self):
        return {
            "capacity_per_type": self.capacity_per_type,
            "top_k": self.top_k,
            "site_cap": self.site_cap,
            "remove_after_correct_streak": self.remove_after_correct_streak,
            "epoch": self.epoch,
            "records": {
                typ: {int(pid): _record_copy(rec) for pid, rec in bucket.items()}
                for typ, bucket in self.records.items()
            },
        }

    @classmethod
    def from_snapshot(cls, snap):
        bank = cls(snap.get("capacity_per_type", 64), snap.get("top_k", 8),
                   snap.get("site_cap", 2), snap.get("remove_after_correct_streak", 10))
        bank.epoch = int(snap.get("epoch", 0))
        for typ in TYPES:
            bank.records[typ] = {
                int(pid): _record_copy(rec)
                for pid, rec in snap.get("records", {}).get(typ, {}).items()
            }
        return bank

    def _flat_records(self):
        return [rec for typ in TYPES for rec in self.records[typ].values()]

    def size(self, typ=None):
        if typ is None:
            return sum(len(x) for x in self.records.values())
        return len(self.records[typ])

    def _new_record(self, pid, true_label, wrong_label, site, epoch, feature,
                    error_probability, wrong_margin, cross_entropy, direction):
        typ = self._type(true_label, wrong_label)
        return {
            "patient_id": int(pid),
            "true_label": int(true_label),
            "wrong_predicted_label": int(wrong_label),
            "error_type": typ,
            "cam144_error_feature": _clone_tensor(feature),
            "cam144_retrieval_feature": _clone_tensor(feature),
            "error_probability": float(error_probability),
            "wrong_margin": float(max(0.0, wrong_margin)),
            "cross_entropy_at_error": float(cross_entropy),
            "first_error_epoch": int(epoch),
            "last_error_epoch": int(epoch),
            "total_error_count": 1,
            "recent_10_epoch_error_count": 1,
            "consecutive_error_count": 1,
            "correct_streak": 0,
            "recent_errors": deque([1], maxlen=10),
            "correction_direction_144": _clone_tensor(direction),
            "site_id": int(site),
            "priority_score": 0.0,
        }

    def _remove_pid(self, pid):
        removed = 0
        for typ in TYPES:
            if int(pid) in self.records[typ]:
                del self.records[typ][int(pid)]
                removed += 1
        return removed

    def _event_hash_data(self):
        rows = []
        for typ in TYPES:
            for pid in sorted(self.records[typ]):
                rec = self.records[typ][pid]
                row = {
                    key: (value.detach().cpu().numpy().round(7).tolist()
                          if torch.is_tensor(value) else list(value)
                          if isinstance(value, deque) else value)
                    for key, value in rec.items()
                    if key not in ("cam144_error_feature", "cam144_retrieval_feature",
                                   "correction_direction_144")
                }
                row["feature_sha"] = hashlib.sha256(
                    rec["cam144_retrieval_feature"].numpy().tobytes()).hexdigest()
                row["direction_sha"] = hashlib.sha256(
                    rec["correction_direction_144"].numpy().tobytes()).hexdigest()
                rows.append((typ, int(pid), row))
        return rows

    def event_hash(self):
        payload = json.dumps(self._event_hash_data(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def update_from_base_eval(self, eligible_ids, labels, cam144, logp, sites,
                              epoch, correction_directions=None):
        """Update from *original* BaseEval errors and return an audit record.

        `eligible_ids` is supplied by the outer-fold runner and has already
        removed the test set and the two Fold-2 duplicate rows.
        """
        labels_cpu = labels.detach().cpu().long()
        logp_cpu = logp.detach().cpu().float()
        feat_cpu = cam144.detach().cpu().float()
        sites_cpu = sites.detach().cpu().long()
        directions = correction_directions or {}
        writes = updates = evictions = removals = errors = 0
        old_sizes = {typ: len(self.records[typ]) for typ in TYPES}
        eligible_set = {int(x) for x in eligible_ids}
        pred = logp_cpu.argmax(dim=1)

        # Correct/error state is updated once per patient, so repeated epochs
        # update a single event rather than creating duplicate entries.
        for pid in sorted(eligible_set):
            y = int(labels_cpu[pid].item())
            p = int(pred[pid].item())
            existing = None
            for typ in TYPES:
                if pid in self.records[typ]:
                    existing = self.records[typ][pid]
                    break
            if p != y:
                errors += 1
                typ = self._type(y, p)
                wrong_margin = max(0.0, float((logp_cpu[pid, p] - logp_cpu[pid, y]).item()))
                ce = float((-logp_cpu[pid, y]).item())
                prob_wrong = float(logp_cpu[pid, p].exp().item())
                direction = directions.get(pid, torch.zeros(144))
                if existing is not None and existing.get("error_type") == typ:
                    old = existing["cam144_retrieval_feature"]
                    existing["cam144_error_feature"] = _clone_tensor(feat_cpu[pid])
                    existing["cam144_retrieval_feature"] = 0.9 * old + 0.1 * _clone_tensor(feat_cpu[pid])
                    existing["correction_direction_144"] = _clone_tensor(direction)
                    existing["error_probability"] = prob_wrong
                    existing["wrong_margin"] = wrong_margin
                    existing["cross_entropy_at_error"] = ce
                    existing["last_error_epoch"] = int(epoch)
                    existing["total_error_count"] = int(existing["total_error_count"]) + 1
                    q = existing["recent_errors"]
                    q.append(1)
                    existing["recent_10_epoch_error_count"] = int(sum(q))
                    existing["consecutive_error_count"] = int(existing["consecutive_error_count"]) + 1
                    existing["correct_streak"] = 0
                    updates += 1
                else:
                    if existing is not None:
                        self._remove_pid(pid)
                    rec = self._new_record(pid, y, p, int(sites_cpu[pid].item()), epoch,
                                           feat_cpu[pid], prob_wrong, wrong_margin, ce, direction)
                    self.records[typ][pid] = rec
                    writes += 1
            elif existing is not None:
                q = existing["recent_errors"]
                q.append(0)
                existing["recent_10_epoch_error_count"] = int(sum(q))
                existing["consecutive_error_count"] = 0
                existing["correct_streak"] = int(existing["correct_streak"]) + 1
                if existing["correct_streak"] >= self.remove_after_correct_streak:
                    self._remove_pid(pid)
                    removals += 1

        # Priority is recomputed after the current epoch's event updates.
        for typ in TYPES:
            for rec in self.records[typ].values():
                rec["priority_score"] = float(self._priority(rec, epoch))
            bucket = self.records[typ]
            if len(bucket) > self.capacity_per_type:
                candidates = sorted(bucket.values(), key=lambda r: self._sort_key(r, epoch))
                keep = {int(r["patient_id"]) for r in candidates[:self.capacity_per_type]}
                for pid in list(bucket):
                    if int(pid) not in keep:
                        del bucket[pid]
                        evictions += 1
        self.epoch = int(epoch)
        event = {
            "epoch": int(epoch), "eligible_count": len(eligible_set),
            "base_error_count": int(errors), "writes": int(writes),
            "updates": int(updates), "evictions": int(evictions),
            "correct_streak_removals": int(removals),
            "size_asd_to_td": self.size(TYPE_ASD_TO_TD),
            "size_td_to_asd": self.size(TYPE_TD_TO_ASD),
            "event_hash": self.event_hash(),
        }
        self.history.append(event)
        return event

    def retrieval(self, query_ids, query_h, sites, device, use_direction=True,
                  error_type_swap=False, feature_permuted=False,
                  recency_off=False, direction_permuted=False,
                  query_repr=None, key_projection=None):
        """Return per-query memory tensors for the differentiable module.

        Candidate filtering (including self-exclusion) happens before Top-k.
        The site cap is a deterministic selection rule, with a deterministic
        fallback when fewer than k candidates survive.
        """
        qids = [int(x) for x in query_ids]
        q = query_h
        out = []
        for row, pid in enumerate(qids):
            row_types = []
            for typ in TYPES:
                actual_type = (TYPE_TD_TO_ASD if typ == TYPE_ASD_TO_TD else TYPE_ASD_TO_TD) if error_type_swap else typ
                candidates = [r for r in self.records[actual_type].values() if int(r["patient_id"]) != pid]
                # similarity and priority are computed on CPU for deterministic
                # candidate ordering, then tensors move to the query device.
                if not candidates:
                    row_types.append(None)
                    continue
                features = torch.stack([r["cam144_retrieval_feature"] for r in candidates])
                directions = torch.stack([r["correction_direction_144"] for r in candidates])
                sites_c = torch.tensor([int(r["site_id"]) for r in candidates], dtype=torch.long)
                priorities = torch.tensor([0.0 if recency_off else float(r["priority_score"]) for r in candidates])
                if feature_permuted:
                    perm = torch.arange(len(candidates) - 1, -1, -1)
                    features = features[perm]
                    directions = directions[perm]
                if direction_permuted and len(candidates) > 1:
                    directions = directions.roll(1, 0)
                query = (query_repr[row] if query_repr is not None else q[row]).detach().cpu().float()
                # retrieval ranking uses detached query/key values, while the
                # selected values remain differentiable only through module
                # projections downstream.
                if key_projection is None:
                    key_raw = features
                else:
                    with torch.no_grad():
                        key_raw = key_projection(features.to(device)).detach().cpu().float()
                keys = torch.nn.functional.normalize(key_raw, dim=-1)
                sim = torch.nn.functional.normalize(query, dim=-1) @ keys.T
                order = sorted(range(len(candidates)),
                               key=lambda idx: (-float(sim[idx]), -float(priorities[idx]), int(candidates[idx]["patient_id"])))
                chosen = []
                used_sites = {}
                for idx in order:
                    site = int(sites_c[idx])
                    if used_sites.get(site, 0) >= self.site_cap:
                        continue
                    chosen.append(idx); used_sites[site] = used_sites.get(site, 0) + 1
                    if len(chosen) >= self.top_k:
                        break
                if len(chosen) < min(self.top_k, len(order)):
                    for idx in order:
                        if idx in chosen: continue
                        chosen.append(idx)
                        if len(chosen) >= self.top_k: break
                chosen_t = torch.tensor(chosen, dtype=torch.long)
                row_types.append({
                    "features": features[chosen_t].to(device),
                    "directions": directions[chosen_t].to(device),
                    "sites": sites_c[chosen_t].to(device),
                    "priority": priorities[chosen_t].to(device),
                    "patient_ids": [int(candidates[idx]["patient_id"]) for idx in chosen],
                    "error_type": typ,
                })
            out.append(row_types)
        return out

    def stats(self):
        return {
            "size_asd_to_td": self.size(TYPE_ASD_TO_TD),
            "size_td_to_asd": self.size(TYPE_TD_TO_ASD),
            "size_total": self.size(),
            "capacity_asd_to_td": self.capacity_per_type,
            "capacity_td_to_asd": self.capacity_per_type,
            "last_event": self.history[-1] if self.history else None,
        }
