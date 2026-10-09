"""SPromptGL Experiment 5 model wrapper.

The copied repository architecture is kept intact.  This wrapper adds only
protocol-aware indices, checkpoint evaluation and the pre-prompt alignment
term required by the registered experiment.
"""

import copy
from collections import Counter

import numpy as np
import torch
import torch.optim as optim

from alignment import compute_alignment_loss, compute_alignment_metrics
from metrics import evaluate_prob
from network import GFR, PLN
from utils import ClsLoss, GraphConstructLoss, my_weight_init


class EvalHelper:
    def __init__(
        self,
        input_data_dims,
        feat,
        label,
        hyperpm,
        train_index,
        val_index,
        test_index,
    ):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_data_dims = list(input_data_dims)
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.protocol = hyperpm.protocol
        self.align_mode = hyperpm.align_mode
        self.align_lambda = float(hyperpm.align_lambda)
        self.align_temperature = float(hyperpm.align_temperature)
        self.align_warmup = int(hyperpm.align_warmup)
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th

        self.feat = torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(self.dev)
        self.targ = torch.from_numpy(np.asarray(label, dtype=np.int64)).to(self.dev)
        self.trn_idx_np = np.asarray(train_index, dtype=np.int64)
        self.val_idx_np = np.asarray(val_index, dtype=np.int64)
        self.tst_idx_np = np.asarray(test_index, dtype=np.int64)
        self.trn_idx = torch.as_tensor(self.trn_idx_np, dtype=torch.long, device=self.dev)
        self.val_idx = torch.as_tensor(self.val_idx_np, dtype=torch.long, device=self.dev)
        self.tst_idx = torch.as_tensor(self.tst_idx_np, dtype=torch.long, device=self.dev)

        # Preserve the two class-weight protocols exactly.
        train_y = self.targ[self.trn_idx].detach().cpu().numpy()
        if self.protocol == "strict":
            counts = np.bincount(train_y, minlength=self.n_class).astype(np.float64)
            weight = len(train_y) / (self.n_class * np.maximum(counts, 1.0))
        else:
            counter = Counter(train_y.tolist())
            weight = len(train_y) / np.array(list(counter.values()), dtype=np.float64) / self.n_class
        self.weight = torch.tensor(weight, dtype=torch.float32, device=self.dev)

        # Keep original simple-2 initialization order.
        from layers import InitialDataLayer
        self.dataLayer = InitialDataLayer(self.input_data_dims)
        self.MessagePassing0 = GFR(
            input_dim=self.input_data_dims,
            output_dim=self.input_data_dims,
            th=self.th,
            mode=self.GC_mode,
        ).to(self.dev)
        self.out_dim = self.d_v * self.n_head
        if self.MF_mode != "concat":
            raise ValueError("Experiment 5 only supports original concat fusion")
        self.ModalFusion = PLN(self.input_data_dims, hyperpm).to(self.dev)
        self.optimizer_MP0 = optim.Adam(
            self.MessagePassing0.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MF = optim.Adam(
            self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.ModalFusion.apply(my_weight_init)

        self.base_init_sha256 = None
        self.last_losses = {
            "cls": 0.0, "graph": 0.0, "ALoss": 0.0,
            "align": 0.0, "align_nce": 0.0, "align_cmmd": 0.0,
            "lambda_eff": 0.0,
        }

    def set_base_fingerprint(self, fingerprint_fn):
        self.base_init_sha256 = fingerprint_fn(self.MessagePassing0, self.ModalFusion)

    def _lambda_eff(self, epoch):
        if self.align_mode == "none":
            return 0.0
        scale = min(1.0, float(epoch + 1) / float(self.align_warmup))
        return self.align_lambda * scale

    def run_epoch(self, epoch):
        self.MessagePassing0.train()
        self.ModalFusion.train()
        self.optimizer_MP0.zero_grad()
        self.optimizer_MF.zero_grad()

        model_num, feat1 = self.dataLayer(self.feat)
        feat, adj, a_loss = self.MessagePassing0(
            feat1, model_num, self.targ, self.trn_idx_np
        )

        all_graph_loss = 0.0
        for i in range(model_num):
            graph_loss = GraphConstructLoss(
                feat1[i], adj[i], self.hyperpm.theta_smooth,
                self.hyperpm.theta_degree, self.hyperpm.theta_sparsity,
            )
            all_graph_loss = all_graph_loss + graph_loss
        # Deliberately preserve the original ABIDE simple-2 /4 expression.
        all_graph_loss = all_graph_loss / 4

        prob, _, _, pre_tokens, post_tokens = self.ModalFusion(
            feat, return_modal_tokens=True
        )
        if tuple(pre_tokens.shape) != (871, 4, 36):
            raise RuntimeError("Unexpected pre token shape: {}".format(tuple(pre_tokens.shape)))
        if tuple(post_tokens.shape) != (871, 4, 36):
            raise RuntimeError("Unexpected post token shape: {}".format(tuple(post_tokens.shape)))
        cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)

        align_loss = pre_tokens.sum() * 0.0
        align_parts = {"nce": 0.0, "cmmd": 0.0, "total": 0.0}
        if self.align_mode == "hybrid":
            train_tokens = pre_tokens[self.trn_idx]
            train_labels = self.targ[self.trn_idx]
            align_loss, align_parts = compute_alignment_loss(
                train_tokens, train_labels,
                mode="hybrid", temperature=self.align_temperature,
            )
        elif self.align_mode != "none":
            raise ValueError("Unknown alignment mode: {}".format(self.align_mode))

        lambda_eff = self._lambda_eff(epoch)
        # ALoss is recorded only.  It is intentionally absent from this loss.
        total_loss = cls_loss + all_graph_loss + lambda_eff * align_loss
        total_loss.backward()
        self.optimizer_MP0.step()
        self.optimizer_MF.step()

        self.last_losses = {
            "cls": float(cls_loss.detach()),
            "graph": float(all_graph_loss.detach()),
            "ALoss": float(a_loss.detach()),
            "align": float(align_loss.detach()),
            "align_nce": float(align_parts["nce"]),
            "align_cmmd": float(align_parts["cmmd"]),
            "lambda_eff": float(lambda_eff),
        }
        return self.last_losses

    @torch.no_grad()
    def forward_all(self, return_modal_tokens=True):
        self.MessagePassing0.eval()
        self.ModalFusion.eval()
        model_num, feat1 = self.dataLayer(self.feat)
        feat, adj, a_loss = self.MessagePassing0(
            feat1, model_num, self.targ, self.trn_idx_np
        )
        output = self.ModalFusion(feat, return_modal_tokens=return_modal_tokens)
        if return_modal_tokens:
            log_prob, hidden, attn, pre_tokens, post_tokens = output
        else:
            log_prob, hidden, attn = output
            pre_tokens, post_tokens = None, None
        return {
            "log_prob": log_prob,
            "hidden": hidden,
            "attn": attn,
            "pre_tokens": pre_tokens,
            "post_tokens": post_tokens,
            "ALoss": a_loss,
            "adj": adj,
            "feat": feat,
        }

    def state_dicts(self):
        return {
            "MessagePassing0": copy.deepcopy(self.MessagePassing0.state_dict()),
            "ModalFusion": copy.deepcopy(self.ModalFusion.state_dict()),
        }

    def load_state_dicts(self, states):
        self.MessagePassing0.load_state_dict(states["MessagePassing0"])
        self.ModalFusion.load_state_dict(states["ModalFusion"])

    @torch.no_grad()
    def validation_metrics(self):
        result = self.forward_all(return_modal_tokens=False)
        return evaluate_prob(result["log_prob"], self.targ, self.val_idx)

    @torch.no_grad()
    def selected_test_result(self):
        result = self.forward_all(return_modal_tokens=True)
        metrics = evaluate_prob(result["log_prob"], self.targ, self.tst_idx)
        test_indices = self.tst_idx
        test_labels = self.targ[test_indices]
        pre_metrics = compute_alignment_metrics(
            result["pre_tokens"][test_indices], test_labels
        )
        post_metrics = compute_alignment_metrics(
            result["post_tokens"][test_indices], test_labels
        )
        delta = result["post_tokens"][test_indices] - result["pre_tokens"][test_indices]
        pre_norm = torch.linalg.vector_norm(result["pre_tokens"][test_indices], dim=-1)
        return {
            "metrics": metrics,
            "log_prob": result["log_prob"].detach().cpu().numpy(),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "prompt_delta_mean_abs": float(delta.abs().mean()),
            "prompt_delta_relative_norm": float(
                (torch.linalg.vector_norm(delta, dim=-1) / (pre_norm + 1e-8)).mean()
            ),
            "last_losses": dict(self.last_losses),
        }

