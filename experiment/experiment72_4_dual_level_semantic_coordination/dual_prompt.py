from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


AGE_PREFIX = "AGE_AT_SCAN_"
SITE_PREFIX = "SITE_ID_"
SEX_PREFIX = "SEX_"
MISSING_TEXT = (
    "The participant's age interval at the time of the brain scan is unavailable.",
    "The participant's recorded sex is unavailable.",
    "The participant's brain imaging acquisition site is unavailable.",
)
PROMPT_TEMPLATES = (
    "The participant's age at the time of the brain scan falls within [AGE_INTERVAL] years.",
    "The participant's recorded sex is [SEX].",
    "The participant's brain imaging data were acquired at [SITE_NAME].",
)
AGE_PATTERN = re.compile(r"^AGE_AT_SCAN_\[([^,]+),\s*([^\)]+)\)$")


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def decode_patient_prompts(csv_path: Path, mapping_path: Path) -> dict:
    mapping = np.load(mapping_path, allow_pickle=True).item()
    pheno = list(mapping["PHENO"])
    feature_frame = pd.read_csv(csv_path, usecols=pheno).reindex(columns=pheno)
    groups = {
        "age": [i for i, name in enumerate(pheno) if name.startswith(AGE_PREFIX)],
        "site": [i for i, name in enumerate(pheno) if name.startswith(SITE_PREFIX)],
        "sex": [i for i, name in enumerate(pheno) if name.startswith(SEX_PREFIX)],
    }
    names = {key: [pheno[i] for i in indices] for key, indices in groups.items()}
    if tuple(map(len, (groups["age"], groups["site"], groups["sex"]))) != (26, 20, 2):
        raise ValueError("PHENO48 field map must contain 26 age, 20 site, and 2 sex categories")
    values = feature_frame.to_numpy(dtype=np.float64)
    texts: list[list[str]] = []
    masks = np.zeros((len(feature_frame), 3), dtype=np.bool_)
    selections: list[dict] = []
    for row_index, row in enumerate(values):
        row_texts = []
        row_selection = {}
        for field_index, field in enumerate(("age", "sex", "site")):
            column_indices = groups[field]
            selected = np.flatnonzero(np.isclose(row[column_indices], 1.0, atol=0.0, rtol=0.0))
            if selected.size == 0:
                row_texts.append(MISSING_TEXT[field_index])
                row_selection[field] = None
                continue
            if selected.size != 1 or not np.isin(row[column_indices], (0.0, 1.0)).all():
                raise ValueError(f"row {row_index} has invalid {field} one-hot data")
            category = names[field][int(selected[0])]
            masks[row_index, field_index] = True
            if field == "age":
                match = AGE_PATTERN.fullmatch(category)
                if match is None:
                    raise ValueError(f"cannot recover age interval from {category!r}")
                lower, upper = match.groups()
                value = f"{lower} to under {upper}"
                text = PROMPT_TEMPLATES[0].replace("[AGE_INTERVAL]", value)
                row_selection[field] = {"column": category, "lower_inclusive": lower, "upper_exclusive": upper}
            elif field == "sex":
                value = category.removeprefix(SEX_PREFIX)
                text = PROMPT_TEMPLATES[1].replace("[SEX]", value)
                row_selection[field] = {"column": category, "recorded_category": value}
            else:
                value = category.removeprefix(SITE_PREFIX)
                text = PROMPT_TEMPLATES[2].replace("[SITE_NAME]", value)
                row_selection[field] = {"column": category, "site_id": value}
            row_texts.append(text)
        texts.append(row_texts)
        selections.append(row_selection)
    return {
        "texts": texts,
        "field_validity": masks,
        "selections": selections,
        "feature_sha256": hashlib.sha256(feature_frame.to_csv(index=False).encode("utf-8")).hexdigest(),
        "row_count": int(len(feature_frame)),
        "mapping_sha256": hashlib.sha256(Path(mapping_path).read_bytes()).hexdigest(),
    }


class PatientSemanticPrompt(nn.Module):
    def __init__(self, condition_dim: int = 10, cap: float = 0.1, output_init_scale: float = 0.01):
        super().__init__()
        self.query = nn.Linear(320, 64, bias=False)
        self.key = nn.Linear(1024, 64, bias=False)
        self.value = nn.Linear(1024, 320, bias=False)
        self.output = nn.Linear(320, 320, bias=False)
        self.gate = nn.Linear(640, 320, bias=True)
        self.condition_projection = nn.Linear(320, condition_dim, bias=False)
        self.condition_query = nn.Linear(condition_dim, 64, bias=False)
        self.condition_lambda_logit = nn.Parameter(torch.tensor(-1.3862943611198906))
        self.cap = float(cap)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight)
        with torch.no_grad():
            self.output.weight.mul_(float(output_init_scale))
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)
        nn.init.xavier_uniform_(self.condition_projection.weight)
        nn.init.xavier_uniform_(self.condition_query.weight)

    def forward(
        self,
        fused: torch.Tensor,
        patient_tokens: torch.Tensor,
        disease_prototypes: torch.Tensor | None = None,
        use_disease_condition: bool = False,
        residual_enabled: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if fused.ndim != 2 or tuple(fused.shape) != (871, 320):
            raise ValueError(f"expected DARG [871,320], got {tuple(fused.shape)}")
        if patient_tokens.shape != (871, 3, 1024):
            raise ValueError(f"expected patient tokens [871,3,1024], got {tuple(patient_tokens.shape)}")
        query = self.query(fused)
        condition_probabilities = None
        condition = None
        lambda_value = fused.new_zeros(())
        if use_disease_condition:
            if disease_prototypes is None or disease_prototypes.shape != (2, 10):
                raise ValueError("G8 requires both frozen ASD/TD prototypes with shape [2,10]")
            proto = F.normalize(disease_prototypes.to(device=fused.device, dtype=fused.dtype), dim=-1)
            patient_cond = F.normalize(self.condition_projection(fused), dim=-1)
            condition_probabilities = torch.softmax((patient_cond @ proto.T) / 1.0, dim=-1)
            condition = condition_probabilities @ proto
            lambda_value = 0.5 * torch.sigmoid(self.condition_lambda_logit)
            query = query + lambda_value * self.condition_query(condition)
        keys = self.key(patient_tokens)
        values = self.value(patient_tokens)
        attention = torch.softmax(torch.einsum("nd,nkd->nk", query, keys) / (64.0 ** 0.5), dim=-1)
        retrieved = torch.einsum("nk,nkd->nd", attention, values)
        gate = torch.sigmoid(self.gate(torch.cat((fused, retrieved), dim=-1)))
        raw_residual = gate * self.output(retrieved)
        input_norm = fused.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        raw_norm = raw_residual.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.cap * input_norm / raw_norm.clamp_min(1e-12), max=1.0)
        delta = raw_residual * scale if residual_enabled else torch.zeros_like(raw_residual)
        output = fused + delta
        return {
            "fused": output,
            "delta": delta,
            "relative_residual": delta.norm(dim=-1) / input_norm.squeeze(-1),
            "gate": gate,
            "attention": attention,
            "attention_entropy": -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1),
            "retrieved": retrieved,
            "condition_probabilities": condition_probabilities,
            "condition": condition,
            "condition_lambda": lambda_value,
        }
