from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim

from network import GFR, PLN
from layers import InitialDataLayer
from utils import GraphConstructLoss, my_weight_init


def make_initial_trainable_state(input_data_dims, hyperpm, prompt_features, prompt_config, seed):
    """Create a device-independent canonical initialization shared by P0-P3."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    gfr = GFR(
        input_dim=list(input_data_dims),
        output_dim=list(input_data_dims),
        th=hyperpm.th,
        mode=hyperpm.GC_mode,
    )
    pln = PLN(
        list(input_data_dims),
        hyperpm,
        prompt_features.detach().cpu(),
        prompt_config=prompt_config,
    )
    pln.apply(my_weight_init)
    state = {}
    for prefix, module in (("gfr", gfr), ("pln", pln)):
        for name, parameter in module.named_parameters():
            state[f"{prefix}.{name}"] = parameter.detach().cpu().clone()
    return state


def parameter_state_sha256(state: dict[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EvalHelper:
    """Isolated full-batch transductive trainer for one split and prompt condition."""

    def __init__(self, input_data_dims, feat, label, hyperpm, train_index, prompt_features, prompt_config, device):
        self.dev = torch.device(device)
        self.hyperpm = hyperpm
        self.input_data_dims = list(input_data_dims)
        self.modal_num = len(self.input_data_dims)
        self.n_class = 2
        self.GC_mode = hyperpm.GC_mode
        self.MF_mode = hyperpm.MF_mode
        self.th = hyperpm.th
        self.feat = torch.as_tensor(feat, dtype=torch.float32, device=self.dev)
        self.targ = torch.as_tensor(label, dtype=torch.long, device=self.dev)
        self.trn_idx = torch.as_tensor(train_index, dtype=torch.long, device=self.dev)
        counts = torch.bincount(self.targ[self.trn_idx], minlength=self.n_class).float()
        if torch.any(counts == 0):
            raise ValueError("training split is missing a class")
        self.weight = counts.sum() / (self.n_class * counts)

        self.dataLayer = InitialDataLayer(self.input_data_dims)
        self.MessagePassing0 = GFR(
            input_dim=self.input_data_dims,
            output_dim=self.input_data_dims,
            th=self.th,
            mode=self.GC_mode,
        ).to(self.dev)
        self.ModalFusion = PLN(
            self.input_data_dims,
            hyperpm,
            prompt_features,
            prompt_config=prompt_config,
        ).to(self.dev)
        self.ModalFusion.apply(my_weight_init)
        self.optimizer_MP0 = optim.Adam(
            self.MessagePassing0.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )
        self.optimizer_MF = optim.Adam(
            self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg
        )

    def named_trainable_parameters(self):
        result = OrderedDict()
        for prefix, module in (("gfr", self.MessagePassing0), ("pln", self.ModalFusion)):
            for name, parameter in module.named_parameters():
                result[f"{prefix}.{name}"] = parameter
        return result

    def copy_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        current = self.named_trainable_parameters()
        if current.keys() != state.keys():
            raise ValueError("trainable parameter names differ from the canonical initialization")
        with torch.no_grad():
            for name, parameter in current.items():
                value = state[name]
                if parameter.shape != value.shape:
                    raise ValueError(f"shape mismatch for {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.named_trainable_parameters().items()}

    def parameter_sha256(self) -> str:
        return parameter_state_sha256(self.trainable_state())

    def train_step(self) -> dict[str, float]:
        self.MessagePassing0.train()
        self.ModalFusion.train()
        self.optimizer_MP0.zero_grad(set_to_none=True)
        self.optimizer_MF.zero_grad(set_to_none=True)

        model_num, modal_features = self.dataLayer(self.feat)
        fused_features, adjacency_list, label_graph_loss = self.MessagePassing0(
            modal_features, model_num, self.targ, self.trn_idx
        )
        graph_loss = torch.zeros((), device=self.dev)
        for i in range(model_num):
            graph_loss = graph_loss + GraphConstructLoss(
                modal_features[i],
                adjacency_list[i],
                self.hyperpm.theta_smooth,
                self.hyperpm.theta_degree,
                self.hyperpm.theta_sparsity,
            )
        graph_loss = graph_loss / model_num
        log_probability, _, _ = self.ModalFusion(fused_features)
        classification_loss = F.nll_loss(
            log_probability[self.trn_idx], self.targ[self.trn_idx], self.weight
        )
        loss = classification_loss + graph_loss
        loss.backward()
        self.optimizer_MP0.step()
        self.optimizer_MF.step()
        return {
            "loss": float(loss.detach().cpu()),
            "classification_loss": float(classification_loss.detach().cpu()),
            "graph_loss": float(graph_loss.detach().cpu()),
            "label_graph_loss_diagnostic_only": float(label_graph_loss.detach().cpu()),
        }

    @torch.no_grad()
    def predict_all(self, capture_prompt_diagnostics: bool = False):
        self.MessagePassing0.eval()
        self.ModalFusion.eval()
        self.ModalFusion.capture_prompt_diagnostics = capture_prompt_diagnostics
        model_num, modal_features = self.dataLayer(self.feat)
        fused_features, _, _ = self.MessagePassing0(
            modal_features, model_num, self.targ, self.trn_idx
        )
        log_probability, hidden, attention = self.ModalFusion(fused_features)
        probability = log_probability.exp()
        diagnostics = self.ModalFusion.last_prompt_diagnostics
        self.ModalFusion.capture_prompt_diagnostics = False
        return probability.detach().cpu().numpy(), hidden.detach(), attention, diagnostics

