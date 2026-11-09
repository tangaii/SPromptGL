from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from dual_prompt import decode_patient_prompts


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def make_nonlinguistic_category_tokens(patient_cache: dict, seed: int = 725006) -> tuple[torch.Tensor, dict]:
    """Map each PHENO category to a shared fixed random vector using a private CPU RNG."""
    embeddings = patient_cache["embeddings"]
    metadata = patient_cache["metadata"]
    if tuple(embeddings.shape) != (871, 3, 1024):
        raise ValueError("E72.4 patient cache must be [871,3,1024]")
    selections = metadata.get("selections")
    if not isinstance(selections, list) or len(selections) != 871:
        raise ValueError("E72.4 patient cache lacks row-aligned PHENO category selections")

    fields = ("age", "sex", "site")
    keys_by_field = []
    for field in fields:
        keys = sorted({row[field]["column"] for row in selections if row.get(field) is not None})
        if not keys:
            raise ValueError(f"no observed category found for {field}")
        keys_by_field.append(keys)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    token_tables = [torch.randn((len(keys), 1024), generator=generator, dtype=torch.float32) for keys in keys_by_field]
    for table in token_tables:
        table = F.layer_norm(table, (1024,))
        if torch.unique(table, dim=0).shape[0] != table.shape[0]:
            raise AssertionError("random category token collision")

    key_maps = [{key: index for index, key in enumerate(keys)} for keys in keys_by_field]
    rows = []
    for row in selections:
        row_tokens = []
        for field_index, field in enumerate(fields):
            selected = row.get(field)
            if selected is None:
                raise ValueError(f"row has missing {field}; G6 requires a category identity")
            key = selected.get("column")
            if key not in key_maps[field_index]:
                raise ValueError(f"unmapped {field} category {key!r}")
            row_tokens.append(token_tables[field_index][key_maps[field_index][key]])
        rows.append(torch.stack(row_tokens))
    result = torch.stack(rows).contiguous()
    if tuple(result.shape) != (871, 3, 1024):
        raise AssertionError("nonlinguistic category token row alignment failed")
    info = {
        "source_cache_sha256": metadata.get("cache_sha256"),
        "seed": int(seed),
        "generator": "private CPU torch.Generator",
        "category_counts": {field: len(keys) for field, keys in zip(fields, keys_by_field)},
        "category_keys_sha256": hashlib.sha256(json.dumps(keys_by_field, separators=(",", ":")).encode()).hexdigest(),
        "row_alignment": "same processed_standard_data.csv PHENO row order as E72.4 patient text cache",
        "disease_labels_read": False,
        "trainable": False,
        "embeddings_sha256": tensor_sha(result),
    }
    return result, info


class SafePatientPrompt(nn.Module):
    """Patient semantic attention with explicit null routing and adaptive bounded magnitude."""

    def __init__(self, condition_dim: int = 10):
        super().__init__()
        self.feature_norm = nn.LayerNorm(320, elementwise_affine=False)
        self.text_norm = nn.LayerNorm(1024, elementwise_affine=False)
        self.query = nn.Linear(320, 64, bias=False)
        self.key = nn.Linear(1024, 64, bias=False)
        self.value = nn.Linear(1024, 320, bias=False)
        self.null_key = nn.Parameter(torch.empty(64))
        self.direction = nn.Linear(320, 320, bias=False)
        self.gate_hidden = nn.Linear(640, 64)
        self.gate_output = nn.Linear(64, 1)
        self.condition_projection = nn.Linear(320, condition_dim, bias=False)
        self.condition_query = nn.Linear(condition_dim, 64, bias=False)
        self.condition_lambda_logit = nn.Parameter(torch.tensor(-1.0986122886681098))
        self.residual_cap = 0.02
        self.attention_scale = 5.0

        for layer in (self.query, self.key, self.value, self.direction, self.condition_projection, self.condition_query):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.null_key, mean=0.0, std=1.0)
        nn.init.xavier_uniform_(self.gate_hidden.weight)
        nn.init.zeros_(self.gate_hidden.bias)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.constant_(self.gate_output.bias, -2.1972245773362196)

    def forward(
        self,
        fused: torch.Tensor,
        patient_tokens: torch.Tensor,
        disease_prototypes: torch.Tensor | None = None,
        use_disease_condition: bool = False,
        residual_enabled: bool = True,
        condition_override: str | None = None,
        condition_permutation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if tuple(fused.shape) != (871, 320):
            raise ValueError(f"expected DARG [871,320], got {tuple(fused.shape)}")
        if tuple(patient_tokens.shape) != (871, 3, 1024):
            raise ValueError(f"expected patient tokens [871,3,1024], got {tuple(patient_tokens.shape)}")

        normalized_fused = self.feature_norm(fused)
        q_base = F.normalize(self.query(normalized_fused), dim=-1, eps=1e-12)
        condition_strength = None
        condition_direction = None
        lambda_value = fused.new_zeros(())
        query = q_base
        if use_disease_condition:
            if disease_prototypes is None or tuple(disease_prototypes.shape) != (2, self.condition_projection.out_features):
                raise ValueError("disease-conditioned query requires frozen [ASD,TD] prototypes")
            prototypes = F.normalize(disease_prototypes.to(device=fused.device, dtype=fused.dtype), dim=-1, eps=1e-12)
            difference = F.normalize(prototypes[0] - prototypes[1], dim=-1, eps=1e-12)
            if float(difference.norm().detach().cpu()) < 0.99:
                raise AssertionError("disease prototype difference direction is degenerate")
            condition_direction = difference
            projected = F.normalize(self.condition_projection(fused), dim=-1, eps=1e-12)
            condition_strength = torch.tanh((projected * difference).sum(-1))
            if condition_override == "zero":
                condition_strength = torch.zeros_like(condition_strength)
            elif condition_override == "reverse":
                condition_strength = -condition_strength
            elif condition_override == "permute":
                if condition_permutation is None or condition_permutation.shape != (871,):
                    raise ValueError("permuted condition requires an 871-row permutation")
                condition_strength = condition_strength[condition_permutation]
            elif condition_override not in (None, "off"):
                raise ValueError(f"unknown condition intervention {condition_override}")
            disease_query = F.normalize(self.condition_query(difference), dim=-1, eps=1e-12)
            lambda_value = 0.2 * torch.sigmoid(self.condition_lambda_logit)
            if condition_override in ("off", "zero"):
                lambda_value = fused.new_zeros(())
                query = q_base
            else:
                query = F.normalize(q_base + lambda_value * condition_strength[:, None] * disease_query[None, :], dim=-1, eps=1e-12)

        text = self.text_norm(patient_tokens)
        keys = F.normalize(self.key(text), dim=-1, eps=1e-12)
        null_key = F.normalize(self.null_key, dim=-1, eps=1e-12).view(1, 1, -1).expand(fused.shape[0], 1, -1)
        keys = torch.cat((keys, null_key), dim=1)
        values = self.value(text)
        null_value = torch.zeros((fused.shape[0], 1, 320), device=fused.device, dtype=fused.dtype)
        values = torch.cat((values, null_value), dim=1)
        attention_logits = self.attention_scale * torch.einsum("nd,nkd->nk", query, keys)
        attention = torch.softmax(attention_logits, dim=-1)
        retrieved = torch.einsum("nk,nkd->nd", attention, values)

        gate_input = torch.cat((normalized_fused, self.feature_norm(retrieved)), dim=-1)
        gate = torch.sigmoid(self.gate_output(F.gelu(self.gate_hidden(gate_input))))
        direction = F.normalize(self.direction(retrieved), dim=-1, eps=1e-12)
        rho = self.residual_cap * gate.squeeze(-1) * (1.0 - attention[:, 3])
        input_norm = fused.norm(dim=-1)
        delta = rho[:, None] * input_norm[:, None] * direction
        if not residual_enabled:
            delta = torch.zeros_like(delta)
        output = fused + delta
        return {
            "fused": output,
            "delta": delta,
            "relative_residual": delta.norm(dim=-1) / input_norm.clamp_min(1e-12),
            "gate": gate,
            "rho": rho,
            "direction": direction,
            "attention": attention,
            "attention_entropy": -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1),
            "null_attention": attention[:, 3],
            "retrieved": retrieved,
            "condition_strength": condition_strength,
            "condition_direction": condition_direction,
            "condition_lambda": lambda_value,
        }
