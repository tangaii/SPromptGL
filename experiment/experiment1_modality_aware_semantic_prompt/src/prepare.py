from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import (
    CACHE_DIR,
    DATA_DIR,
    EXP_DIR,
    MODALITIES,
    PROMPT_DIR,
    PROMPT_FILES,
    atomic_json,
    load_or_create_splits,
    get_prompt_features,
    load_data,
    load_prompt_texts,
    prompt_digest,
    sha256_bytes,
    sha256_file,
)


def canonical_mapping_hash(mapping: dict) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def prompt_records(config: str) -> list[dict]:
    if config == "P0":
        raw = json.loads(PROMPT_FILES[config].read_text(encoding="utf-8"))
        return [
            {
                "modality": "shared",
                "theme": theme["theme"],
                **prompt,
            }
            for theme in raw["themes"]
            for prompt in theme["prompts"]
        ]
    if config in ("P1", "P3"):
        raw = json.loads(PROMPT_FILES[config].read_text(encoding="utf-8"))
        return [
            {
                "modality": modality,
                "theme": theme["theme"],
                **prompt,
            }
            for modality in MODALITIES
            for theme in raw["modalities"][modality]["themes"]
            for prompt in theme["prompts"]
        ]
    if config == "P2":
        mapping = json.loads(PROMPT_FILES["P2"].read_text(encoding="utf-8"))["mapping"]
        p1 = prompt_records("P1")
        by_modality = {modality: [r for r in p1 if r["modality"] == modality] for modality in MODALITIES}
        return [
            {
                **record,
                "modality": target_modality,
                "source_modality": source_modality,
                "theme": record["theme"],
            }
            for target_modality in MODALITIES
            for source_modality in [mapping[target_modality]]
            for record in by_modality[source_modality]
        ]
    raise ValueError(config)


def main() -> None:
    src_dir = str(EXP_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from CLIP.clip import clip

    mapping_path = PROMPT_FILES["P2"]
    mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping_data["sha256"] = canonical_mapping_hash(mapping_data["mapping"])
    atomic_json(mapping_path, mapping_data)

    audit = {
        "clip_model": "RN50",
        "context_length": 77,
        "embedding_dim": 1024,
        "semantic_mlp_input_dim": 4096,
        "semantic_mlp_hidden_dim": 128,
        "semantic_mlp_output_dim": 9,
        "configs": {},
        "prompt_records": {},
        "checks": {},
        "prompt_file_sha256": {},
        "data_sha256": {
            "processed_standard_data.csv": sha256_file(DATA_DIR / "processed_standard_data.csv"),
            "modal_feat_dict.npy": sha256_file(DATA_DIR / "modal_feat_dict.npy"),
        },
    }
    for config in ("P0", "P1", "P2", "P3"):
        texts = load_prompt_texts(config)
        if len(texts) != 4 or any(len(modality) != 2 for modality in texts):
            raise AssertionError(f"{config}: expected [4 modalities, 2 themes]")
        if any(len(theme) != 4 for modality in texts for theme in modality):
            raise AssertionError(f"{config}: every theme must contain exactly four prompts")
        flat = (
            [text for theme in texts[0] for text in theme]
            if config == "P0"
            else [text for modality in texts for theme in modality for text in theme]
        )
        if any(not text.strip() for text in flat):
            raise AssertionError(f"{config}: empty prompt")
        tokenized = clip.tokenize(flat)
        lengths = (tokenized != 0).sum(dim=1).tolist()
        if max(lengths) > 77:
            raise AssertionError(f"{config}: CLIP context length exceeded")
        expected_prompts = 8 if config == "P0" else 32
        if len(flat) != expected_prompts:
            raise AssertionError(f"{config}: expected {expected_prompts} input text slots, got {len(flat)}")
        if config != "P0" and len(set(flat)) != expected_prompts:
            raise AssertionError(f"{config}: expected 32 distinct prompt texts")
        audit["configs"][config] = {
            "input_text_slots": len(flat),
            "unique_texts": len(set(flat)),
            "duplicate_slots": len(flat) - len(set(flat)),
            "max_token_length_including_sot_eot": max(lengths),
            "min_token_length_including_sot_eot": min(lengths),
            "modalities": {
                modality: {
                    "theme_count": len(texts[i]),
                    "prompt_count": sum(len(theme) for theme in texts[i]),
                    "theme_names": [
                        ("original_group_1", "original_group_2")[j] if config == "P0" else f"theme_{j + 1}"
                        for j in range(len(texts[i]))
                    ],
                }
                for i, modality in enumerate(MODALITIES)
            },
            "prompt_sha256": prompt_digest(config),
            "token_lengths": lengths,
        }
        features = get_prompt_features(config, torch.device("cpu"))
        if tuple(features.shape) != (4, 2, 4, 1024):
            raise AssertionError(f"{config}: unexpected cached feature shape {tuple(features.shape)}")
        del features
        audit["prompt_file_sha256"][PROMPT_FILES[config].name] = sha256_file(PROMPT_FILES[config])
        records = prompt_records(config)
        expected_records = 8 if config == "P0" else 32
        if len(records) != expected_records:
            raise AssertionError(f"{config}: malformed prompt audit records")
        audit["prompt_records"][config] = records

    p2_map = mapping_data["mapping"]
    if set(p2_map) != set(MODALITIES) or set(p2_map.values()) != set(MODALITIES):
        raise AssertionError("P2 must map all modalities bijectively")
    if any(p2_map[modality] == modality for modality in MODALITIES):
        raise AssertionError("P2 mapping must misassign every modality")
    if p2_map != {"PHENO": "FUNC", "ANAT": "Correlation", "FUNC": "PHENO", "Correlation": "ANAT"}:
        raise AssertionError("P2 mapping differs from its preregistered cycle")

    x, y, dims, feature_dict = load_data()
    if dims != [48, 6, 10, 256] or x.shape != (871, 320):
        raise AssertionError(f"data contract mismatch dims={dims}, x={x.shape}")
    if int((y == 0).sum()) != 403 or int((y == 1).sum()) != 468:
        raise AssertionError("label mapping/count mismatch")
    load_or_create_splits(y)
    audit["checks"] = {
        "prompt_structure": "PASS",
        "p2_fixed_derangement": "PASS",
        "token_limit_77": "PASS",
        "empty_prompts": "PASS_NONE",
        "data_dimensions": dims,
        "csv_rows": int(x.shape[0]),
        "csv_predictor_columns": int(x.shape[1]),
        "label_mapping": {"DX_GROUP_1_ASD": 0, "DX_GROUP_2_NC": 1},
        "label_counts": {"ASD_0": int((y == 0).sum()), "NC_1": int((y == 1).sum())},
        "modal_feature_lists_match_csv_order": True,
    }
    audit["environment"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    atomic_json(PROMPT_DIR / "prompt_audit.json", audit)
    print(json.dumps(audit["checks"], indent=2))
    print(f"prompt audit: {PROMPT_DIR / 'prompt_audit.json'}")
    print(f"CLIP caches: {len(list(CACHE_DIR.glob('P*_*.pt')))}")


if __name__ == "__main__":
    main()
