"""Fold-private, deterministic support memories and persistent confusion relations."""
from __future__ import annotations
import copy
import numpy as np
import torch

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def cluster(x, k, iterations):
    x = normalize(x)
    k = min(k, len(x))
    chosen = [int(np.square(x-x.mean(0)).sum(1).argmax())]
    for _ in range(1, k):
        dist = np.maximum(0, np.square(x[:, None]-x[chosen][None]).sum(-1)).min(1)
        dist[chosen] = -1
        chosen.append(int(dist.argmax()))
    centers, previous = x[chosen].copy(), None
    for _ in range(iterations):
        assignment = np.square(x[:, None]-centers[None]).sum(-1).argmin(1)
        for j in range(k):
            if np.any(assignment == j):
                centers[j] = x[assignment == j].mean(0)
        if previous is not None and np.array_equal(previous, assignment):
            break
        previous = assignment.copy()
    return assignment

def identity_split(allowed, labels, fold):
    """Independent local RNG, class stratification; outer folds remain untouched."""
    g = np.random.default_rng(774300+fold)
    split = np.full(len(labels), -1, dtype=np.int64)
    for cls in (0, 1):
        ids = np.sort(allowed[labels[allowed] == cls]).copy()
        g.shuffle(ids)
        split[ids] = np.arange(len(ids)) % 3
    return split

class Memory:
    def __init__(self, cfg, allowed, labels, split, fold):
        self.cfg, self.fold = cfg, fold
        self.allowed = np.asarray(allowed, dtype=np.int64).copy()
        self.labels = np.full(len(labels), -1, dtype=np.int64)
        self.labels[self.allowed] = np.asarray(labels)[self.allowed]
        self.split = np.asarray(split).copy()
        n = len(labels)
        self.history = np.full((3, n), -1, dtype=np.int64)
        self.observations = 0
        self.error_count = np.zeros(n, dtype=np.int64)
        self.stable_frequency = np.zeros(n, dtype=np.int64)
        self.streak = np.zeros(n, dtype=np.int64)
        self.last_stable = np.full(n, -1, dtype=np.int64)
        self.first_error = np.full(n, -1, dtype=np.int64)
        self.last_error = np.full(n, -1, dtype=np.int64)
        self.pairs = {}  # fixed patient identities, retained across support rotations
        self.pool = {0: [], 1: []}  # 0=FP (true TD), 1=FN (true ASD)
        self.epoch = 0
        self.stats = {}
        self.eval_bank = None

    def observe_predictions(self, prediction, epoch):
        # Predictions are collected without comparing Query labels. Error
        # identities below are evaluated only for the current Support.
        self.history[self.observations % 3].fill(-1)
        self.history[self.observations % 3, self.allowed] = prediction[self.allowed]
        self.observations += 1
        self.epoch = epoch

    def update_relations(self, h, raw, support, active):
        cfg, epoch, y = self.cfg, self.epoch, self.labels
        support = np.asarray(support)
        assert np.isin(support, self.allowed).all()
        valid = self.history[:, support] >= 0
        votes = ((self.history[:, support] != y[support][None]) & valid).sum(0)
        correct_votes = ((self.history[:, support] == y[support][None]) & valid).sum(0)
        pred = self.history[(self.observations-1) % 3]
        wrong = pred[support] != y[support]
        stable = (votes >= cfg['min_votes']) & (valid.sum(0) == cfg['window'])
        reliable = (correct_votes >= cfg['min_votes']) & ~wrong & (valid.sum(0) == cfg['window'])
        self.error_count[support] += wrong
        self.streak[support] = np.where(wrong, self.streak[support]+1, 0)
        self.first_error[support[(self.first_error[support] < 0) & wrong]] = epoch
        self.last_error[support[wrong]] = epoch
        self.stable_frequency[support] += stable
        self.last_stable[support[stable]] = epoch
        stats = dict(stable_fp=int(np.sum(stable & (y[support] == 1))),
                     stable_fn=int(np.sum(stable & (y[support] == 0))),
                     currently_wrong_support=int(wrong.sum()),
                     stable_correct_support=int(reliable.sum()), new_pairs=0, updated_pairs=0,
                     expired_pairs=0, retained_pairs=0)
        if not active:
            self.stats = stats
            return stats
        # Query identity never enters candidate eligibility or pair updates.
        support_set = set(support.tolist())
        vote_lookup = dict(zip(support.tolist(), votes.tolist()))
        for key in list(self.pairs):
            if epoch-self.pairs[key]['last_epoch'] > cfg['expiry']:
                del self.pairs[key]
                stats['expired_pairs'] += 1
        for direction, cls in ((0, 1), (1, 0)):
            candidates = support[(y[support] == cls) & (self.last_stable[support] >= 0) &
                                 (epoch-self.last_stable[support] <= cfg['expiry'])]
            ordered = sorted(candidates.tolist(), key=lambda i: (
                -vote_lookup[i], -min(int(self.stable_frequency[i]), 10),
                -int(self.last_stable[i]), i))
            # The stored pool uses unique identities and has a fixed capacity.
            retained = [i for i in self.pool[direction] if epoch-self.last_stable[i] <= cfg['expiry']]
            merged = sorted(set(retained+ordered), key=lambda i: (
                -min(int(self.stable_frequency[i]), 10), -int(self.last_stable[i]), i))
            self.pool[direction] = merged[:cfg['capacity_per_direction']]
        pools = set(self.pool[0]+self.pool[1])
        for key in list(self.pairs):
            if self.pairs[key]['hard_id'] not in pools:
                del self.pairs[key]
        hc, rc = normalize(h), normalize(raw)
        # Retained legal records count toward per-patient and opposite reuse caps.
        reuse, hard_count = {}, {}
        for rec in self.pairs.values():
            reuse[rec['opposite_id']] = reuse.get(rec['opposite_id'], 0)+1
            hard_count[rec['hard_id']] = hard_count.get(rec['hard_id'], 0)+1
        reliable_ids = support[reliable]
        for direction in (0, 1):
            for i in self.pool[direction]:
                if i not in support_set:
                    continue
                candidates = reliable_ids[y[reliable_ids] != y[i]]
                if not len(candidates):
                    continue
                sim = cfg['pair_cam_weight']*(hc[candidates] @ hc[i]) + (1-cfg['pair_cam_weight'])*(rc[candidates] @ rc[i])
                for j in np.argsort(-sim, kind='stable'):
                    t = int(candidates[j])
                    pair = (i, t) if y[i] == 0 else (t, i)
                    key = f'{self.fold}:{pair[0]}:{pair[1]}'
                    if key in self.pairs:
                        rec = self.pairs[key]
                        # Only one update per epoch and always while both are Support.
                        if rec['last_epoch'] != epoch:
                            ema = cfg['pair_ema']
                            rec['cam'] = ema*rec['cam']+(1-ema)*h[list(pair)]
                            rec['raw'] = ema*rec['raw']+(1-ema)*raw[list(pair)]
                            rec['frequency'] += 1
                            rec['last_epoch'] = epoch
                            rec['quality'] = float((1+sim[j])/2)
                            rec['reliability'] = float(.5+.5*min(int(self.stable_frequency[i]), 10)/10)
                            rec['delta_cam'] = rec['cam'][0]-rec['cam'][1]
                            rec['delta_raw'] = rec['raw'][0]-rec['raw'][1]
                            stats['updated_pairs'] += 1
                        continue
                    if hard_count.get(i, 0) >= cfg['max_partners'] or reuse.get(t, 0) >= cfg['max_opposite_reuse']:
                        continue
                    assert pair[0] != pair[1] and y[pair[0]] == 0 and y[pair[1]] == 1
                    self.pairs[key] = dict(pair_id=key, ids=pair, hard_id=i, opposite_id=t,
                        source=direction, first_epoch=epoch, last_epoch=epoch, frequency=1,
                        cam=h[list(pair)].copy(), raw=raw[list(pair)].copy(),
                        delta_cam=h[pair[0]]-h[pair[1]], delta_raw=raw[pair[0]]-raw[pair[1]],
                        quality=float((1+sim[j])/2), similarity=float(sim[j]),
                        reliability=float(.5+.5*min(int(self.stable_frequency[i]), 10)/10))
                    reuse[t] = reuse.get(t, 0)+1
                    hard_count[i] = hard_count.get(i, 0)+1
                    stats['new_pairs'] += 1
        stats.update(retained_pairs=len(self.pairs),pool_fp=len(self.pool[0]),pool_fn=len(self.pool[1]))
        self.stats = stats
        return stats

    def bank(self, h, raw, support, device):
        """Four CAM banks plus ONE independent global CORR bank."""
        support = np.asarray(support, dtype=np.int64)
        assert np.isin(support, self.allowed).all()
        k = self.cfg['k']
        cam = np.zeros((4, 2, k, 36), dtype=np.float32)
        counts = np.zeros((4, 2, k), dtype=np.float32)
        raw_centers = np.zeros((2, k, 256), dtype=np.float32)
        raw_counts = np.zeros((2, k), dtype=np.float32)
        members, raw_members = [], []
        for m in range(4):
            for cls in (0, 1):
                ids = support[self.labels[support] == cls]
                if not len(ids):
                    members.extend([[] for _ in range(k)])
                    continue
                a = cluster(h[ids, 36*m:36*(m+1)], k, self.cfg['cluster_iterations'])
                for j in range(k):
                    group = ids[a == j]
                    members.append(group.tolist())
                    counts[m, cls, j] = len(group)
                    if len(group):
                        cam[m, cls, j] = h[group, 36*m:36*(m+1)].mean(0)
        for cls in (0, 1):
            ids = support[self.labels[support] == cls]
            if not len(ids):
                raw_members.extend([[] for _ in range(k)])
                continue
            a = cluster(raw[ids], k, self.cfg['cluster_iterations'])
            for j in range(k):
                group = ids[a == j]
                raw_members.append(group.tolist())
                raw_counts[cls, j] = len(group)
                if len(group):
                    raw_centers[cls, j] = raw[group].mean(0)
        ss = set(support.tolist())
        records = [r for _, r in sorted(self.pairs.items()) if set(r['ids']) <= ss and
                   self.epoch-r['last_epoch'] <= self.cfg['expiry']]
        n = len(records)
        def t(x, dtype=torch.float32):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        out = dict(cam=t(cam), counts=t(counts), raw=t(raw_centers), raw_counts=t(raw_counts),
                   support=t(support, torch.long), members=members, raw_members=raw_members,
                   pair_ids=t([r['ids'] for r in records], torch.long).reshape(n, 2),
                   pair_cam=t([r['cam'] for r in records]).reshape(n, 2, 144),
                   pair_raw=t([r['raw'] for r in records]).reshape(n, 2, 256),
                   delta_cam=t([r['delta_cam'] for r in records]).reshape(n, 144),
                   delta_raw=t([r['delta_raw'] for r in records]).reshape(n, 256),
                   pair_reliability=t([r['quality']*r['reliability']*(1-(self.epoch-r['last_epoch'])/(self.cfg['expiry']+1)) for r in records]),
                   pair_keys=[r['pair_id'] for r in records],
                   pair_frequency=t([r['frequency'] for r in records]), epoch=self.epoch)
        return out

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))

def cpu_bank(bank):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in bank.items()}

def device_bank(bank, device):
    return {k:v.to(device) if torch.is_tensor(v) else copy.deepcopy(v) for k,v in bank.items()}

def audit_bank(bank, query, test, labels):
    support = set(bank['support'].cpu().tolist())
    q, tst = set(np.asarray(query).tolist()), set(np.asarray(test).tolist())
    assert not (support & q) and not (support & tst)
    for members in bank['members']+bank['raw_members']:
        assert set(members) <= support
    p = bank['pair_ids'].cpu().numpy()
    assert np.isin(p, list(support)).all()
    if len(p):
        assert (np.asarray(labels)[p[:, 0]] == 0).all() and (np.asarray(labels)[p[:, 1]] == 1).all()
        assert len({tuple(v) for v in p.tolist()}) == len(p)
    assert torch.equal(bank['delta_raw'],bank['pair_raw'][:,0]-bank['pair_raw'][:,1])
    assert torch.equal(bank['delta_cam'],bank['pair_cam'][:,0]-bank['pair_cam'][:,1])
    return dict(query_count=len(q),support_count=len(support),pairs=len(p),identity_isolation=True,
                test_exclusion=True,pair_orientation=True,prototype_members=True)
