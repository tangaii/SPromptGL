from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
KB_PATH = HERE / "knowledge_base.json"
KB = json.loads(KB_PATH.read_text(encoding="utf-8"))
RN50_PATH = Path(CONFIG["text_encoder"]["weights_path"])
CACHE_PATH = HERE / ".runtime" / "text_embeddings.pt"
REPO_ROOT = HERE.parents[1]
TOKENIZER_NORMALIZATION = "ftfy"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def metadata_prompts(mapping: dict[str, list[str]], x: np.ndarray) -> tuple[list[str], np.ndarray]:
    pheno = mapping["PHENO"]
    columns = {
        "age": [i for i, value in enumerate(pheno) if str(value).startswith("AGE_AT_SCAN_")],
        "site": [i for i, value in enumerate(pheno) if str(value).startswith("SITE_ID_")],
        "sex": [i for i, value in enumerate(pheno) if str(value).startswith("SEX_")],
    }
    if [len(columns[name]) for name in ("age", "site", "sex")] != [26, 20, 2]:
        raise AssertionError("PHENO category layout differs from the audited 26/20/2 map")
    labels, ids = [], []
    templates = {
        "age": lambda value: f"Age category {value.removeprefix('AGE_AT_SCAN_')} years.",
        "site": lambda value: f"Imaging acquisition site category {value.removeprefix('SITE_ID_')}.",
        "sex": lambda value: f"Recorded sex category {value.removeprefix('SEX_')}.",
    }
    for name in ("age", "site", "sex"):
        values = [str(pheno[index]) for index in columns[name]]
        ids.append(np.argmax(x[:, columns[name]], axis=1).astype(np.int64))
        labels.extend(templates[name](value) for value in values)
    return labels, np.stack(ids, axis=1)


def all_texts(mapping: dict[str, list[str]], x: np.ndarray) -> tuple[dict[str, list[str]], np.ndarray]:
    medical = [
        item["medical_description"]
        for cls in ("ASD", "TD")
        for item in KB["classes"][cls]
    ]
    class_only = KB["class_only_templates"]["ASD"] + KB["class_only_templates"]["TD"]
    brainprompt = CONFIG["brainprompt"]["texts_in_published_order_TC_then_ASD"]
    brainprompt = [brainprompt[1], brainprompt[0]]
    original = CONFIG["original_spromptgl_pen"]["texts_in_source_order"]
    patient, patient_ids = metadata_prompts(mapping, x)
    return {
        "medical": medical,
        "class_only": class_only,
        "brainprompt": brainprompt,
        "original": original,
        "patient_categories": patient,
    }, patient_ids


def encode_texts(texts: list[str], device: str = "cuda:0") -> torch.Tensor:
    if not RN50_PATH.is_file() or sha_file(RN50_PATH) != CONFIG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("the verified OpenAI CLIP RN50 checkpoint is unavailable")
    global TOKENIZER_NORMALIZATION
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "ftfy" or any(not text.isascii() for text in texts):
            raise
        ftfy_compat = types.ModuleType("ftfy")
        ftfy_compat.fix_text = lambda text: text
        sys.modules["ftfy"] = ftfy_compat
        TOKENIZER_NORMALIZATION = "ascii_identity_equivalent_to_ftfy"

    from CLIP.clip import clip

    model, _ = clip.load(str(RN50_PATH), device=device, jit=False)
    model.eval()
    tokens = clip.tokenize(texts)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 128):
            encoded = model.encode_text(tokens[start:start + 128].to(device)).float().cpu()
            chunks.append(encoded)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def prepare_text_cache(R) -> dict[str, Any]:
    if sha_file(RN50_PATH) != CONFIG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("CLIP checkpoint hash mismatch")
    expected = {
        "knowledge_sha256": sha_file(KB_PATH),
        "rn50_sha256": sha_file(RN50_PATH),
        "config_sha256": sha_file(HERE / "config.json"),
        "text_features_sha256": sha_file(HERE / "text_features.py"),
    }
    if CACHE_PATH.is_file():
        cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if cached.get("source_hashes") == expected:
            return cached

    x, _ = R.load_data()[0]
    mapping = R.runtime()["np"].load(R.ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    text_map, patient_ids = all_texts(mapping, x)
    embeddings = {
        name: encode_texts(texts)
        for name, texts in text_map.items()
    }
    result = {
        "source_hashes": expected,
        "text_map": text_map,
        "embeddings": embeddings,
        "patient_category_ids": torch.from_numpy(patient_ids),
        "patient_count": int(x.shape[0]),
        "text_dim": int(embeddings["medical"].shape[-1]),
        "tokenizer_normalization": TOKENIZER_NORMALIZATION,
    }
    atomic_torch_save(CACHE_PATH, result)
    return result


def group_text_banks(cache: dict[str, Any], group: str) -> dict[str, torch.Tensor]:
    e = cache["embeddings"]
    medical = e["medical"].reshape(2, 6, -1)
    if group == "E60_DSS_CLASS_ONLY":
        dss = e["class_only"].reshape(2, 6, -1)
    elif group == "E60_DSS_WRONG_SEMANTICS":
        dss = medical.flip(0)
    elif group == "E60_DSS_SINGLE_PROTOTYPE":
        centroids = torch.nn.functional.normalize(medical.mean(dim=1, keepdim=True), dim=-1)
        dss = centroids.expand(-1, 6, -1).contiguous()
    elif group == "E60_DSS_RANDOM_ANCHORS":
        dss = geometry_matched_random(medical, seed=60010)
    elif group == "E60_DSS_LEARNED_PROTOTYPES":
        slots = torch.zeros(12, medical.shape[-1])
        slots[:, :12] = torch.eye(12)
        dss = slots.reshape(2, 6, -1)
    else:
        dss = medical
    brainprompt = e["brainprompt"]
    centroids = torch.nn.functional.normalize(medical.mean(dim=1), dim=-1)
    if group == "E60_BRAINPROMPT_MEDICAL":
        brainprompt = centroids
    patient_categories = e["patient_categories"]
    category_ids = cache["patient_category_ids"].long()
    patient_text = torch.stack([
        patient_categories[0:26][category_ids[:, 0]],
        patient_categories[26:46][category_ids[:, 1]],
        patient_categories[46:48][category_ids[:, 2]],
    ], dim=1)
    return {
        "medical": medical,
        "dss": dss,
        "brainprompt": brainprompt,
        "disease_centroids": centroids,
        "patient_text": patient_text,
        "original": e["original"],
    }


def geometry_matched_random(bank: torch.Tensor, seed: int) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(bank.reshape(-1, bank.shape[-1]), dim=-1)
    gram = normalized @ normalized.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    root = eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_matrix = torch.randn(bank.shape[-1], normalized.shape[0], generator=generator)
    orthogonal, _ = torch.linalg.qr(random_matrix, mode="reduced")
    rotated = root @ orthogonal.T
    return rotated.reshape_as(bank)
