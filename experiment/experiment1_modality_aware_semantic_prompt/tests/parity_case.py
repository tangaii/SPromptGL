from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=("reference", "experiment"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
exp_dir = Path(__file__).resolve().parents[1]
project_root = exp_dir.parents[1]

if args.kind == "reference":
    source_dir = exp_dir / "src" / "reference"
else:
    source_dir = exp_dir / "src"
sys.path.insert(0, str(source_dir))
sys.path.insert(1, str(project_root))

from layers import InitialDataLayer
from utils import my_weight_init
import network as model_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7421)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7421)

class Hyper:
    n_hidden = 18
    n_head = 2
    nlayer = 1
    dropout = 0.35
    nmodal = 4
    nclass = 2

dims = [48, 6, 10, 256]
if args.kind == "reference":
    original_text = model_network.get_prompt_text_features().detach().cpu()
    torch.manual_seed(7421)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7421)
    model = model_network.PLN(dims, Hyper()).to(device)
    model.apply(my_weight_init)
    original_text = original_text.to(device)
    prototypes = model.MLP(torch.cat([original_text[0:4].reshape(1, -1), original_text[4:8].reshape(1, -1)], dim=0))
else:
    from common import get_prompt_features
    prompt_features = get_prompt_features("P0", device)
    torch.manual_seed(7421)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7421)
    model = model_network.PLN(dims, Hyper(), prompt_features, prompt_config="P0").to(device)
    model.apply(my_weight_init)
    prototypes = model.semantic_prototypes()[0]

model.eval()
captured = {}
def capture_output_layer(module, inputs):
    captured["pre_output"] = inputs[0].detach().cpu()
model.Outputlayer.register_forward_pre_hook(capture_output_layer)

generator = torch.Generator(device="cpu").manual_seed(88442)
x = torch.randn((871, sum(dims)), generator=generator, dtype=torch.float32).to(device)
with torch.no_grad():
    prediction, hidden, attn = model(x)

trainable = {
    name: parameter.detach().cpu().contiguous().clone()
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
}
torch.save({
    "kind": args.kind,
    "trainable": trainable,
    "prototypes": prototypes.detach().cpu(),
    "pre_output": captured["pre_output"],
    "log_probability": prediction.detach().cpu(),
    "hidden": hidden.detach().cpu(),
    "trainable_count": sum(value.numel() for value in trainable.values()),
    "prompt_features_shape": list(model.prompt_text_features.shape),
}, args.output)
print(json.dumps({
    "kind": args.kind,
    "trainable_count": sum(value.numel() for value in trainable.values()),
    "prompt_features_shape": list(model.prompt_text_features.shape),
    "log_probability_shape": list(prediction.shape),
}))
