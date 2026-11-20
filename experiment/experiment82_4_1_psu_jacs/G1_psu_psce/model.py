"""Prompt-conditioned signed actions supervised by real native-head utility."""
from __future__ import annotations

import math

from common.spectral import DIM, SpectralPrompt


class PsuPsce(SpectralPrompt):
    def __init__(self, spectrum, text, _reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg, method, descriptor_width=2)
        self.method = method

    def forward(self, cam, *, head, training, query=None, semantic_off=False, variant=None):
        d = self.tokens(cam, query, semantic_off)
        white = d["white"].tanh()
        logits = self.signed_logits(d, (white, d["importance"] - 1))
        action = (d["importance"] * (logits + self.method["spectral_skip_scale"] * white)).tanh()
        if variant == "unsigned":
            action = action.abs()
        return self.finish(cam, action / math.sqrt(DIM), {**d, "action": action})
