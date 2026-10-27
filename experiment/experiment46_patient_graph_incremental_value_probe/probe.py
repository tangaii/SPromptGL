"""The only trainable model in Experiment 46: a small classification probe."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
class Probe(nn.Module):
 def __init__(self,d):
  super().__init__(); self.input_dim=d; self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,36),nn.GELU(),nn.Dropout(.35),nn.Linear(36,2))
 def forward(self,x): return F.log_softmax(self.net(x),dim=1)
