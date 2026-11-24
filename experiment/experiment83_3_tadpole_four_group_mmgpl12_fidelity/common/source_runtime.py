"""Bind TADPOLE adapters to the archived historical arithmetic."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "sources/history_base"
DARG = ROOT / "sources/history_darg"
SCMR = ROOT / "sources/E82_4_6_G2_SCMR_DA_SRPE/code/history_base"
E74_GROUP = "E74.4-G2-Positive-DCR"
TADPOLE_MODALITIES = (
    "UCSFFSX", "UCBERKELEYAV45", "UPENNBIOMK9",
    "RISK_FACTOR", "COGNITIVE_TEST", "ROI_AVERAGE",
)
ABIDE_MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
ABIDE_DIMS = (48, 6, 10, 256)


def verified_text(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"Historical source SHA mismatch for {path}")
    return raw.decode("utf-8")


def replace_exact(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f"Unexpected archived source definition: {before!r}")
    return source.replace(before, after)


def archived_definitions(path, names, *, source=None, module_name=None, **bindings):
    """Execute selected definitions without the historical experiment entry points."""
    source = path.read_text(encoding="utf-8") if source is None else source
    tree = ast.parse(source, filename=str(path))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in tree.body} != set(names):
        raise RuntimeError(f"Missing archived definitions in {path}")
    namespace = ModuleType(module_name or f"e831_archived_{path.stem}")
    globals_ = vars(namespace)
    globals_.update({
        "__file__": str(path),
        "torch": torch, "nn": nn, "F": F, "np": np,
        "Dict": Dict, "List": List, "Optional": Optional,
    })
    globals_.update(bindings)
    exec(compile(tree, str(path), "exec"), globals_)
    if module_name is not None:
        sys.modules[module_name] = namespace
    return SimpleNamespace(**{name: globals_[name] for name in names})


def source_graph_utils(path):
    return archived_definitions(path, ("normalize_adj", "GraphConstructLoss", "ClsLoss"),
                                source=replace_exact(
                                    path.read_text(encoding="utf-8"),
                                    "dev = torch.device('cuda' if use_cuda else 'cpu')",
                                    "dev = adj.device",
                                ))


@lru_cache(maxsize=1)
def e51_graph_classes():
    e51 = DARG / "e51"
    manifest = json.loads(verified_text(
        e51 / "manifest.json", "4db1f5c52836dd573447f750c4bca1d61cce89f3a2c2683e6a44abaf738d66bb"
    ))
    sources = {name: verified_text(e51 / "source" / name, digest)
               for name, digest in manifest["source_sha"].items()}
    conv = archived_definitions(e51 / "source/layers.py", ("GraphConv",),
                                source=sources["layers.py"])
    network = archived_definitions(e51 / "source/network.py", ("GraphLearn", "GCN"),
                                   source=sources["network.py"], GraphConv=conv.GraphConv)
    utilities = source_graph_utils(e51 / "source/utils.py")
    return network, utilities, verified_text(e51 / "dugsa.py", manifest["code_sha"]["dugsa.py"])


def e51_legacy_graph(dims, device=None):
    network, utilities, text = e51_graph_classes()
    path = DARG / "e51/dugsa.py"
    text = replace_exact(text, "self.fusion_weight = nn.Parameter(torch.ones(4, 1))",
                         "self.fusion_weight = nn.Parameter(torch.ones(len(DIMS), 1))")
    factory = archived_definitions(path, ("LegacyFixedGFR",), source=text, DIMS=dims)
    return factory.LegacyFixedGFR(network.GraphLearn, network.GCN, utilities, th=0.9).to(device)


@lru_cache(maxsize=None)
def darg_sources(dims, modalities=TADPOLE_MODALITIES):
    if len(dims) != len(modalities):
        raise ValueError("Graph source adaptation needs one name per modality")
    corr = modalities.index("CORR") if "CORR" in modalities else None
    cadr_path = DARG / "cadr_gfr.py"
    cadr_text = verified_text(cadr_path, "f5032d96c9ac68afda47a253f794ddfe41373fabebbbbe27057f1425f988815d")
    # Only modality cardinality and CORR identity are changed in CADR.
    cadr_text = replace_exact(cadr_text, "if modality == 3 else grad_reverse",
                              "if modality == CORR_INDEX else grad_reverse")
    cadr_text = replace_exact(cadr_text, "ConditionalModalityDiscriminator(65, 32, 4)",
                              "ConditionalModalityDiscriminator(65, 32, len(DIMS))")
    cadr_text = replace_exact(cadr_text, "torch.zeros(4)", "torch.zeros(len(DIMS))")
    cadr_text = replace_exact(cadr_text, "torch.zeros(4, device=raw.device)",
                              "torch.zeros(len(DIMS), device=raw.device)")
    cadr = archived_definitions(
        cadr_path, ("_GradientReverse", "grad_reverse", "pair_features", "RelationEncoder",
                    "RelationScorer", "ConditionalModalityDiscriminator", "conditional_adversarial_loss",
                    "relation_preservation_loss", "_mutual_topk", "_symmetric_adjacency",
                    "_bounded_block_delta", "_weighted_neighbor_difference", "CADRGFR"),
        source=cadr_text, DIMS=dims, MODALITIES=modalities, CORR_INDEX=corr, nullcontext=nullcontext,
    )
    cadr.MODALITIES = modalities
    positive_path = DARG / "soft_positive_dcr.py"
    positive_text = verified_text(
        positive_path, "3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718"
    )
    positive_text = replace_exact(positive_text, "modality == 3", "modality == CORR_INDEX", count=7)
    positive_text = replace_exact(positive_text, "range(4)", "range(len(DIMS))", count=2)
    positive_text = replace_exact(positive_text, '["RUDAG_UNIFIED_GRAPH"] * 4',
                                  '["RUDAG_UNIFIED_GRAPH"] * len(DIMS)')
    offsets = np.cumsum((0, *dims))
    positive = archived_definitions(
        positive_path, ("_pair_score_chunks", "_original_mutual_topk", "_symmetric_delta",
                        "_score_transform", "_edge_delta", "_bounded_disease_residual",
                        "direct_delta_for_group", "localize_residual", "unified_adjacency",
                        "rudag_forward"),
        source=positive_text, _cadr_module=lambda: cadr,
        MODALITIES=modalities, DIMS=dims, CORR_INDEX=corr,
        GATE_SCALES=tuple(1.0 if name == "CORR" else 0.01 for name in modalities),
        BLOCK_SLICES=tuple(zip(offsets[:-1], offsets[1:])),
        REWEIGHT_STRENGTH=0.8, CHUNK_SIZE=16384,
        GROUP_SPECS={E74_GROUP: {"direct_scale": 0.02, "positive_only": True, "corr_temperature": 1.0}},
    )
    return cadr, positive


@lru_cache(maxsize=1)
def relation_pair_sources():
    e53 = DARG / "e53_3_run.py"
    e61 = DARG / "e61_run.py"
    e53_text = verified_text(e53, "574fb23745ccbd49f45250cae8b678d05137d092187e9e959e474e185d02be3c")
    e61_text = verified_text(e61, "f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334")
    e53_helpers = archived_definitions(e53, ("_pair_key", "sample_balanced_pairs"), source=e53_text)
    return archived_definitions(e61, ("make_outer_pairs",), source=e61_text, load_base=lambda: e53_helpers)
