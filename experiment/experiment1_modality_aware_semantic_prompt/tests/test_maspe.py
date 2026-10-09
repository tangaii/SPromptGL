from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXP_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from common import MODALITIES, PROMPT_FILES, load_data, load_prompt_texts, prompt_digest
from metrics import classification_metrics
from model import make_initial_trainable_state, parameter_state_sha256
from network import build_label_target, PLN


class Hyper:
    n_hidden = 18
    n_head = 2
    nlayer = 1
    dropout = 0.35
    nmodal = 4
    nclass = 2
    GC_mode = "weighted-cosine"
    th = 0.9


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y, dims, _ = load_data()
        cls.x, cls.y, cls.dims = x, y, dims
        cls.prompt_features = {}
        for config in ("P0", "P1", "P2", "P3"):
            path = EXP_DIR / "cache"
            matches = sorted(path.glob(f"{config}_{prompt_digest(config)}_*.pt"))
            if not matches:
                raise RuntimeError("run python src/prepare.py before the test suite")
            cls.prompt_features[config] = torch.load(matches[0], map_location="cpu", weights_only=True)

    def test_prompt_counts_mapping_and_data(self):
        expected = {
            "PHENO": 48,
            "ANAT": 6,
            "FUNC": 10,
            "Correlation": 256,
        }
        self.assertEqual(self.dims, list(expected.values()))
        self.assertEqual(self.x.shape, (871, 320))
        self.assertEqual((int((self.y == 0).sum()), int((self.y == 1).sum())), (403, 468))
        for config in ("P0", "P1", "P2", "P3"):
            texts = load_prompt_texts(config)
            self.assertEqual(len(texts), 4)
            self.assertTrue(all(len(modality) == 2 for modality in texts))
            self.assertTrue(all(len(theme) == 4 for modality in texts for theme in modality))
            self.assertEqual(tuple(self.prompt_features[config].shape), (4, 2, 4, 1024))
            self.assertTrue(torch.isfinite(self.prompt_features[config]).all())

    def test_p2_is_fixed_bijective_derangement(self):
        p1 = load_prompt_texts("P1")
        p2 = load_prompt_texts("P2")
        mapping = json.loads(PROMPT_FILES["P2"].read_text(encoding="utf-8"))["mapping"]
        for target_index, target_name in enumerate(MODALITIES):
            self.assertNotEqual(mapping[target_name], target_name)
            source_index = MODALITIES.index(mapping[target_name])
            self.assertEqual(p2[target_index], p1[source_index])
        self.assertEqual(set(mapping.values()), set(MODALITIES))

    def test_trainable_parameter_counts_and_seeded_init_match(self):
        states = {
            config: make_initial_trainable_state(
                self.dims,
                Hyper(),
                self.prompt_features[config],
                config,
                seed=113,
            )
            for config in ("P0", "P1", "P2", "P3")
        }
        hashes = {config: parameter_state_sha256(state) for config, state in states.items()}
        self.assertEqual(len(set(hashes.values())), 1, hashes)
        counts = {config: sum(value.numel() for value in state.values()) for config, state in states.items()}
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_label_target_ignores_validation_and_test_labels(self):
        train = np.arange(600)
        altered = torch.as_tensor(self.y.copy(), dtype=torch.long)
        hidden = np.arange(600, len(self.y))
        altered[torch.as_tensor(hidden)] = 1 - altered[torch.as_tensor(hidden)]
        reference = build_label_target(torch.as_tensor(self.y, dtype=torch.long), train, torch.device("cpu"))
        changed = build_label_target(altered, train, torch.device("cpu"))
        self.assertTrue(torch.equal(reference, changed))

    def test_probability_auc_uses_asd_probability(self):
        y = np.array([0, 0, 1, 1])
        probability = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        metrics = classification_metrics(y, probability)
        self.assertEqual(metrics["probability_auc_asd_positive"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_p0_forward_parity_against_root_source(self):
        with tempfile.TemporaryDirectory(prefix="maspe-parity-") as tmp:
            artifacts = {}
            for kind in ("reference", "experiment"):
                path = Path(tmp) / f"{kind}.pt"
                subprocess.run(
                    [sys.executable, str(EXP_DIR / "tests" / "parity_case.py"), "--kind", kind, "--output", str(path)],
                    cwd=EXP_DIR.parents[1],
                    check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "2"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                artifacts[kind] = torch.load(path, map_location="cpu", weights_only=True)
            ref, exp = artifacts["reference"], artifacts["experiment"]
            self.assertEqual(ref["trainable_count"], exp["trainable_count"])
            self.assertEqual(ref["trainable"].keys(), exp["trainable"].keys())
            param_error = max(float((ref["trainable"][key] - exp["trainable"][key]).abs().max()) for key in ref["trainable"])
            prototype_error = float((ref["prototypes"] - exp["prototypes"]).abs().max())
            prompt_feature_error = float((ref["pre_output"] - exp["pre_output"]).abs().max())
            output_error = float((ref["log_probability"] - exp["log_probability"]).abs().max())
            hidden_error = float((ref["hidden"] - exp["hidden"]).abs().max())
            report = {
                "trainable_parameter_count_reference": ref["trainable_count"],
                "trainable_parameter_count_experiment": exp["trainable_count"],
                "trainable_parameter_max_abs_error": param_error,
                "semantic_prototype_max_abs_error": prototype_error,
                "prompt_enhanced_feature_max_abs_error": prompt_feature_error,
                "log_probability_max_abs_error": output_error,
                "hidden_max_abs_error": hidden_error,
                "tolerance": 1e-6,
            }
            (EXP_DIR / "results" / "p0_parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            self.assertLessEqual(param_error, 1e-6, report)
            self.assertLessEqual(prototype_error, 1e-6, report)
            self.assertLessEqual(prompt_feature_error, 1e-6, report)
            self.assertLessEqual(output_error, 1e-6, report)
            self.assertLessEqual(hidden_error, 1e-6, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
