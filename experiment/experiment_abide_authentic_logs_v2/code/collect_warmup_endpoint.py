#!/usr/bin/env python3
"""Evaluate the saved, verified E61 epoch-10 states without training updates."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = V2 / "code/replay_e61_warmup.py"
OUT = V2 / "Warmup"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    spec = importlib.util.spec_from_file_location("e61_replay_module_for_endpoint", REPLAY_SCRIPT)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    runtime = replay.load_runtime()
    base = runtime.load_base()
    tools = base.runtime()
    torch, np = tools["torch"], tools["np"]
    torch.cuda.set_device(0)
    rows = []
    for fold in range(1, 11):
        checkpoint = runtime.torch_load(OUT / "checkpoints" / f"fold{fold:02d}.pt")
        if checkpoint.get("completed_epoch") != 10:
            raise RuntimeError(f"Warmup fold {fold} replay checkpoint does not end at epoch 10")
        state = runtime.build_state(fold)
        state["model"].load_state_dict(checkpoint["model"], strict=True)
        state["cam"].load_state_dict(checkpoint["cam"], strict=True)
        base.restore_rng(checkpoint["rng"])
        rng_before = runtime.rng_sha(base.pack_rng())
        with torch.no_grad():
            evaluated = base.forward_state(state, training=False, use_cadr=False)
            metrics = base.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after = runtime.rng_sha(base.pack_rng())
        if rng_before != rng_after or rng_before != checkpoint["rng_sha"]:
            raise RuntimeError(f"Warmup endpoint evaluation changed RNG in fold {fold}")
        prediction = metrics["pred"].astype(np.int64)
        probability = metrics["prob"]
        labels = state["y"].detach().cpu().numpy().astype(np.int64)
        path = OUT / "results" / f"fold{fold:02d}.endpoint.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, labels=labels, train_indices=state["train_np"],
                                test_indices=state["test_np"], prediction=prediction,
                                probability=probability, test_logp=evaluated["logp"][state["test"]].detach().cpu().numpy())
            stream.flush()
            os.fsync(stream.fileno())
        rows.append({
            "fold": fold, "status": "PASS", "endpoint_epoch": 10,
            "test_metrics": {key: float(metrics[key]) for key in
                             ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")},
            "rng_sha_before_and_after": rng_before,
            "endpoint_sha256": replay.sha_file(path), "endpoint_bytes": path.stat().st_size,
            "prediction_sha256": replay.structured_sha(prediction, np, torch),
            "probability_sha256": replay.structured_sha(probability, np, torch),
            "train_members_1based": [int(index) + 1 for index in state["train_np"]],
            "test_members_1based": [int(index) + 1 for index in state["test_np"]],
            "posthoc_evaluation_rng_unchanged": True,
        })
        print(f"Warmup endpoint fold {fold:02d} deterministic evaluation PASS", flush=True)
    atomic_json(OUT / "results/endpoint_evaluation.json", {"status": "PASS", "folds": rows})


if __name__ == "__main__":
    main()
