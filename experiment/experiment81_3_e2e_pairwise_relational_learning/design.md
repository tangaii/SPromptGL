# Experiment 81.3 design

E81.3 has exactly two formal groups and twenty tasks.  Every task restores its
matching E61 epoch-10 warmup through the audited E74.4 loader chain.  GFR,
CAM, DARG and the original OutputLayer remain trainable; the relation branch
is trainable through a fourth AdamW optimizer.

The original DARG forward is run once.  A pre-hook captures CAM144 and a
post-hook records the original OutputLayer output while returning it unchanged
to the historical path.  The historical output therefore supplies the
original DARG classification loss and all graph/pair/adversarial/preservation
losses.  The relation branch receives `z_rel = z.detach()+0.1*(z-z.detach())`
and `logp_anchor = logp_base.detach()`.  Training uses
`L_DARG + L_increment + 0.001*L_corr`, where the increment is the weighted
historical `ClsLoss` difference between corrected and anchor log-probabilities,
and the regularizer is the squared margin correction on legal training rows.

G1 is a 15-neighbor unary cross-local attention control.  G2 uses the same
query, candidate count, width, null candidate, readout, gate, head and loss,
but constructs key/value inputs from explicit receiver/sender products and
signed differences.  Both groups use patient-conditioned local readout before
the final relation summary and a learned gate from 32 relation dimensions plus
four detached/non-random statistics.  The correction head is exactly zero
initialized, the final gate bias is -1, and initialization uses a private CPU
RNG seeded by fold.

Fold 2 keeps the historical overlap rows 377 and 407 in DARG training.  The
new delta margin is forced to zero for their training forward, and the
additional correction regularizer excludes those rows.  No labels or test
correctness are used to decide test-time correction.

Preflight covers base identity, exact zero-correction/base-off parity, eta=0,
eta=0.1 and eta=1 gradient routing, initialization fairness, tensor shapes,
null-only identity, one-pass OutputLayer behavior and a two-step interruption
replay.  Formal workers refuse to start unless the preflight SHA is frozen.
