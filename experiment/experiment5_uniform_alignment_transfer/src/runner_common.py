import csv
import os

import numpy as np


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write: {}".format(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fold_row(
    protocol, config, seed, fold, outer_train, train_index, val_index,
    test_index, best_epoch, best_val_acc, result, selection_type,
    base_init_sha256, split_sha256, epochs_run, runtime_sec,
):
    metrics = result["metrics"]
    pre = result["pre_metrics"]
    post = result["post_metrics"]
    losses = result["last_losses"]
    return {
        "protocol": protocol,
        "config": config,
        "seed": int(seed),
        "fold": int(fold),
        "n_outer_train": int(len(outer_train)),
        "n_train": int(len(train_index)),
        "n_val": int(len(val_index)),
        "n_test": int(len(test_index)),
        "val_equals_test": bool(np.array_equal(val_index, test_index)),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "ACC": metrics["ACC"],
        "BA": metrics["BA"],
        "probability_ASD_AUC": metrics["probability_ASD_AUC"],
        "legacy_hard_AUC": metrics["legacy_hard_AUC"],
        "ASD_F1": metrics["ASD_F1"],
        "ASD_sensitivity": metrics["ASD_sensitivity"],
        "NC_specificity": metrics["NC_specificity"],
        "pre_pos_cos": pre["pos_cos"],
        "pre_neg_cos": pre["neg_cos"],
        "pre_cos_gap": pre["cos_gap"],
        "pre_R1": pre["retrieval_r1"],
        "pre_CMMD": pre["cmmd"],
        "post_pos_cos": post["pos_cos"],
        "post_neg_cos": post["neg_cos"],
        "post_cos_gap": post["cos_gap"],
        "post_R1": post["retrieval_r1"],
        "post_CMMD": post["cmmd"],
        "prompt_delta_mean_abs": result["prompt_delta_mean_abs"],
        "prompt_delta_relative_norm": result["prompt_delta_relative_norm"],
        "final_cls_loss": losses["cls"],
        "final_graph_loss": losses["graph"],
        "final_ALoss": losses["ALoss"],
        "final_align_loss": losses["align"],
        "final_align_nce": losses["align_nce"],
        "final_align_cmmd": losses["align_cmmd"],
        "final_lambda_eff": losses["lambda_eff"],
        "base_init_sha256": base_init_sha256,
        "split_sha256": split_sha256,
        "epochs_run": int(epochs_run),
        "runtime_sec": float(runtime_sec),
        "selection_type": selection_type,
    }


def oof_rows(protocol, config, seed, fold, test_index, original_labels,
             result, selected_epoch, selection_type):
    log_prob = np.asarray(result["log_prob"], dtype=np.float64)
    probabilities = np.exp(log_prob)
    pred_internal = np.argmax(log_prob, axis=1)
    rows = []
    for subject_index in np.asarray(test_index, dtype=np.int64):
        internal = int(original_labels[subject_index] - 1)
        rows.append({
            "protocol": protocol,
            "config": config,
            "seed": int(seed),
            "fold": int(fold),
            "subject_index": int(subject_index),
            "true_original_label": int(original_labels[subject_index]),
            "true_internal_label": internal,
            "true_asd": int(internal == 0),
            "p_asd": float(probabilities[subject_index, 0]),
            "pred_internal": int(pred_internal[subject_index]),
            "pred_asd": int(pred_internal[subject_index] == 0),
            "selected_epoch": int(selected_epoch),
            "selection_type": selection_type,
        })
    return rows

