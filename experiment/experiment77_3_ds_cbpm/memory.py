"""Deterministic, fold-private dual-view centers and real FP/FN patient pairs."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def clusters(x, k, iterations):
    """Farthest-first seeds and Lloyd on normalized CAM slots; ID tie order."""
    x = F.normalize(x, dim=-1)
    k = min(k, len(x))
    first = int((x - x.mean(0)).square().sum(-1).argmax())
    selected = [first]
    for _ in range(1, k):
        distance = torch.cdist(x, x[selected]).square().min(1).values
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    centers = x[selected].clone()
    assignment = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for _ in range(iterations):
        new_assignment = torch.cdist(x, centers).argmin(1)
        for j in range(k):
            selected_j = new_assignment == j
            if selected_j.any():
                centers[j] = x[selected_j].mean(0)
        if torch.equal(assignment, new_assignment):
            assignment = new_assignment
            break
        assignment = new_assignment
    return assignment


class Memory(nn.Module):
    def __init__(self, cfg, n=871):
        super().__init__()
        self.cfg = cfg
        k, cap = cfg['k'], cfg['capacity_per_direction']
        p = 2 * cap * cfg['max_partners']
        def buf(name, shape, dtype=torch.float32, value=0):
            self.register_buffer(name, torch.full(shape, value, dtype=dtype))
        buf('cam_cache', (n, 144)); buf('raw_cache', (n, 256))
        buf('cam_sum', (4, 2, k, 144)); buf('raw_sum', (4, 2, k, 256))
        buf('counts', (4, 2, k)); buf('assignment', (4, n), torch.long, -1)
        buf('allowed', (n,), torch.bool); buf('labels', (n,), torch.long, -1)
        buf('base_history', (3, n), torch.long, -1)
        buf('final_history', (3, n), torch.long, -1)
        for name in ('errors', 'error_streak', 'stable_frequency', 'first_error', 'last_error', 'last_stable'):
            buf(name, (n,), torch.long, -1 if name.startswith(('first', 'last')) else 0)
        buf('stable', (n,), torch.bool); buf('reliable', (n,), torch.bool)
        buf('pool_ids', (2, cap), torch.long, -1)
        buf('pair_ids', (p, 2), torch.long, -1)  # columns always ASD,TD
        buf('hard_ids', (p,), torch.long, -1); buf('direction', (p,), torch.long, -1)  # 0 FP,1 FN
        buf('pair_key', (p, 144)); buf('pair_cam', (p, 2, 144))
        buf('pair_raw', (p, 2, 256)); buf('pair_difference', (p, 256))
        buf('pair_similarity', (p,)); buf('pair_reliability', (p,))
        buf('pair_votes', (p,)); buf('epoch', (), torch.long, 0)
        buf('pair_count', (), torch.long, 0)
        buf('history_count', (), torch.long, 0)

    @torch.no_grad()
    def update_class(self, h, raw, y, allowed):
        self.allowed.zero_(); self.allowed[allowed] = True
        self.labels.fill_(-1); self.labels[allowed] = y[allowed]
        # Never cache test-node features as members, even in the full-graph path.
        self.cam_cache.zero_(); self.raw_cache.zero_()
        self.cam_cache[allowed] = h[allowed]; self.raw_cache[allowed] = raw[allowed]
        self.cam_sum.zero_(); self.raw_sum.zero_(); self.counts.zero_(); self.assignment.fill_(-1)
        k = self.cfg['k']
        for m in range(4):
            for cls in (0, 1):
                ids = allowed[y[allowed] == cls]
                if not len(ids):
                    continue
                a = clusters(h[ids, m*36:(m+1)*36], k, self.cfg['cluster_iterations'])
                self.assignment[m, ids] = cls*k + a
                for j in range(k):
                    members = ids[a == j]
                    self.counts[m, cls, j] = len(members)
                    self.cam_sum[m, cls, j] = h[members].sum(0)
                    self.raw_sum[m, cls, j] = raw[members].sum(0)

    def leave_one_out(self, query_ids, m, cls):
        n, k = len(query_ids), self.cfg['k']
        counts = self.counts[m, cls].unsqueeze(0).expand(n, k)
        member = self.assignment[m, query_ids, None] == (cls*k + torch.arange(k, device=query_ids.device))
        number = counts - member.to(counts.dtype)
        cam = (self.cam_sum[m, cls].unsqueeze(0) - member[..., None]*self.cam_cache[query_ids, None]) / number.clamp_min(1)[..., None]
        raw = (self.raw_sum[m, cls].unsqueeze(0) - member[..., None]*self.raw_cache[query_ids, None]) / number.clamp_min(1)[..., None]
        return cam, raw, number, number > 0

    @torch.no_grad()
    def update_errors_and_pairs(self, base_logp, final_logp, h, raw, y, allowed, epoch, make_pairs):
        index = int(self.history_count) % 3
        bp, fp = base_logp.argmax(-1), final_logp.argmax(-1)
        self.base_history[index].fill_(-1); self.final_history[index].fill_(-1)
        self.base_history[index, allowed] = bp[allowed]
        self.final_history[index, allowed] = fp[allowed]
        self.history_count += 1; self.epoch.fill_(epoch)
        wrong = torch.zeros_like(self.allowed); wrong[allowed] = bp[allowed] != y[allowed]
        self.errors += wrong.long()
        self.error_streak.copy_(torch.where(wrong, self.error_streak + 1, torch.zeros_like(self.error_streak)))
        self.first_error[(self.first_error < 0) & wrong] = epoch
        self.last_error[wrong] = epoch
        valid = self.base_history >= 0
        votes = ((self.base_history != self.labels.unsqueeze(0)) & valid).sum(0)
        correct = ((self.base_history == self.labels.unsqueeze(0)) & valid).sum(0)
        self.stable.copy_((votes >= 2) & self.allowed & (valid.sum(0) == 3))
        self.reliable.copy_((correct >= 2) & self.allowed & ~wrong & (valid.sum(0) == 3))
        self.stable_frequency += self.stable.long()
        self.last_stable[self.stable] = epoch
        if not make_pairs:
            return self.summary(bp, fp, y)
        self.pool_ids.fill_(-1)
        for direction, cls in ((0, 1), (1, 0)):
            eligible = allowed[(y[allowed] == cls) & (self.last_stable[allowed] >= 0) &
                               (epoch - self.last_stable[allowed] <= self.cfg['expiry'])]
            # Bounded per-subject priority, no repeated admissions as extra records.
            ordered = sorted(eligible.tolist(), key=lambda i: (-int(votes[i]),
                -min(int(self.stable_frequency[i]), 10), -int(self.last_stable[i]), i))
            chosen = ordered[:self.cfg['capacity_per_direction']]
            if chosen:
                self.pool_ids[direction, :len(chosen)] = torch.tensor(chosen, device=h.device)
        for name in ('pair_ids', 'hard_ids', 'direction'):
            getattr(self, name).fill_(-1)
        for name in ('pair_key', 'pair_cam', 'pair_raw', 'pair_difference', 'pair_similarity', 'pair_reliability', 'pair_votes'):
            getattr(self, name).zero_()
        hc, rc = F.normalize(h, dim=-1), F.normalize(raw, dim=-1)
        counter, reuse, unique_pairs = 0, {}, set()
        for direction in (0, 1):
            for i in self.pool_ids[direction].tolist():
                if i < 0:
                    continue
                candidates = allowed[self.reliable[allowed] & (y[allowed] != y[i])]
                if not len(candidates):
                    continue
                similarity = .75*(hc[candidates] @ hc[i]) + .25*(rc[candidates] @ rc[i])
                order = torch.argsort(similarity, descending=True, stable=True).tolist()
                admitted = 0
                for j in order:
                    other = int(candidates[j]); pair = (i, other) if int(y[i]) == 0 else (other, i)
                    if pair in unique_pairs or reuse.get(other, 0) >= self.cfg['max_opposite_reuse']:
                        continue
                    unique_pairs.add(pair); reuse[other] = reuse.get(other, 0) + 1
                    ids = torch.tensor(pair, device=h.device)
                    self.pair_ids[counter] = ids; self.hard_ids[counter] = i; self.direction[counter] = direction
                    self.pair_cam[counter] = h[ids]; self.pair_key[counter] = h[ids].mean(0)
                    self.pair_raw[counter] = raw[ids]; self.pair_difference[counter] = raw[pair[0]] - raw[pair[1]]
                    self.pair_similarity[counter] = similarity[j]
                    age = epoch - int(self.last_stable[i])
                    frequency = min(int(self.stable_frequency[i]), 10) / 10
                    self.pair_reliability[counter] = (.5+.5*frequency) * (1-age/(self.cfg['expiry']+1))
                    self.pair_votes[counter] = votes[i]
                    counter += 1; admitted += 1
                    if admitted == self.cfg['max_partners']:
                        break
        self.pair_count.fill_(counter)
        return self.summary(bp, fp, y)

    def summary(self, base_pred=None, final_pred=None, y=None):
        p = int(self.pair_count)
        result = dict(pair_count=p, unique_pairs=len(set(map(tuple, self.pair_ids[:p].cpu().tolist()))),
                      pool_fp=int((self.pool_ids[0] >= 0).sum()), pool_fn=int((self.pool_ids[1] >= 0).sum()),
                      stable_fp=int((self.stable & (self.labels == 1)).sum()), stable_fn=int((self.stable & (self.labels == 0)).sum()),
                      paired_fp=int((self.direction[:p] == 0).sum()), paired_fn=int((self.direction[:p] == 1).sum()),
                      paired_hard_patients=int(self.hard_ids[:p].unique().numel()),
                      match_similarity_mean=float(self.pair_similarity[:p].mean()) if p else 0.,
                      diff_norm_mean=float(self.pair_difference[:p].norm(dim=-1).mean()) if p else 0.)
        if base_pred is not None:
            result.update(stable_base_errors_final_correct=int((self.stable & (final_pred == y)).sum()),
                          stable_base_errors_final_wrong=int((self.stable & (final_pred != y)).sum()),
                          current_base_errors=int(((base_pred != y) & self.allowed).sum()))
        return result
