"""Train one SPromptGL TADPOLE variant and save fold-level evidence."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    CLASS_NAMES, DEFAULTS, MMGPL_CLASS_PROMPTS, MMGPL_CONCEPTS, MMGPL_PAPER, PROVENANCE,
    PUBLIC_PROMPTS, TADPOLE_PROMPTS, VARIANTS,
)
from prepare import sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'src'))
from network import CLIP_WEIGHTS, GFR, PLN, prompt_features
from utils import ClsLoss, GraphConstructLoss, my_weight_init

CLASSES = np.arange(len(CLASS_NAMES))


def read_json(path):
    return json.loads(path.read_text())


def initialize_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(config, device):
    data_dir = ROOT / 'data/TADPOLE' if config.data == 'previous' else HERE / 'data'
    csv_path = data_dir / 'processed_standard_data.csv'
    dict_path = data_dir / 'modal_feat_dict.npy'
    frame = pd.read_csv(csv_path)
    modal_features = np.load(dict_path, allow_pickle=True).item()
    columns = [feature for names in modal_features.values() for feature in names]
    if columns != frame.columns[:-1].tolist() or frame.columns[-1] != 'LABEL':
        raise ValueError('The feature matrix and modality dictionary disagree')
    labels = frame.LABEL.to_numpy(dtype=np.int64) - 1
    if len(frame) != 598 or np.bincount(labels).tolist() != [209, 315, 74]:
        raise ValueError('Unexpected TADPOLE cohort')
    features = torch.from_numpy(frame[columns].to_numpy()).float().to(device)
    dims = [len(names) for names in modal_features.values()]
    info = {
        'paths': {str(path): sha256(path) for path in (csv_path, dict_path)},
        'modalities': list(modal_features), 'dimensions': dims,
        'samples': len(labels), 'label_counts': dict(zip(CLASS_NAMES, np.bincount(labels).tolist())),
        'row_metadata': str(HERE / 'data/samples.csv'),
    }
    return features, labels, dims, info


def all_folds(labels, config):
    splitter = StratifiedKFold(n_splits=config.folds, random_state=config.split_seed, shuffle=True)
    return list(splitter.split(labels, labels))


def experiment_manifest(variant, config, source_prompts, device, data_info):
    sources = [HERE / name for name in ('run.py', 'config.py', 'prepare.py')]
    sources += sorted(
        path for path in (HERE / 'src').rglob('*')
        if path.is_file() and path.suffix in {'.py', '.gz'}
    )
    packages = ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'torchvision']
    return {
        'variant': variant,
        'hyperparameters': vars(config),
        'provenance': PROVENANCE,
        'environment': {
            'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in packages},
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cuda_build': torch.version.cuda,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'torch_cpu_threads': torch.get_num_threads(),
        },
        'inputs': data_info,
        'mmgpl_prompt_reference': {
            'path': str(ROOT / MMGPL_PAPER),
            'sha256': sha256(ROOT / MMGPL_PAPER),
            'figure': 'Figure 6',
        },
        'prompts': {name: list(group) for name, group in source_prompts.items()},
        'clip_weights': {'path': str(CLIP_WEIGHTS), 'sha256': sha256(CLIP_WEIGHTS)},
        'source_sha256': {str(path.relative_to(HERE)): sha256(path) for path in sources},
        'metric_definitions': {
            'acc': 'argmax accuracy',
            'auc_hard': 'macro one-vs-rest ROC AUC from one-hot argmax predictions',
            'auc_probability': 'macro one-vs-rest ROC AUC from softmax probabilities',
            'aggregate': 'unweighted mean and population standard deviation over outer folds',
        },
    }


def save_or_check(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing configuration or inputs differ at {path}')
    else:
        write_json(path, value)


def evaluate(labels, predictions, probabilities):
    targets = np.eye(len(CLASSES))[labels]
    decisions = np.eye(len(CLASSES))[predictions]
    confusion = confusion_matrix(labels, predictions, labels=CLASSES)
    true_positive = confusion.diagonal()
    support = confusion.sum(axis=1)
    negative_support = confusion.sum() - support
    false_positive = confusion.sum(axis=0) - true_positive
    return {
        'acc': float(np.mean(predictions == labels)),
        'auc_hard': float(roc_auc_score(targets, decisions)),
        'auc_probability': float(roc_auc_score(targets, probabilities)),
        'sensitivity': dict(zip(CLASS_NAMES, (true_positive / support).tolist())),
        'specificity': dict(zip(CLASS_NAMES, (1 - false_positive / negative_support).tolist())),
        'confusion': confusion.tolist(),
    }


def graph_supervision(scores, train_index, targets, mode):
    squared_errors = []
    for score in scores:
        train_score = score[train_index[:, None], train_index[None, :]]
        difference = train_score - targets
        squared_errors.append(difference.square().sum())
    squared_frobenius = torch.stack(squared_errors).mean()
    if mode == 'pair_mean':
        return squared_frobenius / targets.numel()
    return squared_frobenius


def relation_audit(gfr, adjacency, cosine, train_index, targets):
    weights = gfr.fusion_weight
    weights = weights.softmax(dim=0) if gfr.fusion == 'softmax_sum' else weights.sigmoid()
    relations = []
    for index, (graph, score) in enumerate(zip(adjacency, cosine)):
        train_graph = graph[train_index[:, None], train_index[None, :]]
        train_score = score[train_index[:, None], train_index[None, :]]
        positives = targets.bool()
        density = (torch.count_nonzero(graph) - torch.count_nonzero(graph.diagonal())).item()
        relations.append({
            'index': index,
            'weight': weights[index].item(),
            'used_for_fusion': gfr.fusion != 'legacy' or index != 0,
            'off_diagonal_edge_density': density / (len(graph) * (len(graph) - 1)),
            'train_pair_mse': (train_score - targets).square().mean().item(),
            'mean_train_cosine_same_label': train_score[positives].mean().item(),
            'mean_train_cosine_different_label': train_score[~positives].mean().item(),
            'train_positive_edge_rate': (train_graph[positives] > 0).float().mean().item(),
            'train_negative_edge_rate': (train_graph[~positives] > 0).float().mean().item(),
        })
    return relations


def train_fold(fold, config, prompts, features, y, dims, labels, train, test, output):
    start = time.monotonic()
    device = features.device
    train_tensor = torch.as_tensor(train, device=device)
    train_labels = y[train_tensor]
    target_edges = train_labels[:, None].eq(train_labels[None, :]).float()
    modalities = list(features.split(dims, dim=1))
    # The original GFR uses a contiguous concatenation of the modality slices.
    raw = torch.cat(modalities, dim=1)
    eye = torch.eye(len(features), device=device, dtype=features.dtype)
    counts = Counter(labels[train])
    weights = len(train) / np.array(list(counts.values())) / len(CLASSES)
    class_weights = torch.from_numpy(weights).float().to(device)
    gfr = GFR(dims, config.th, config.fusion, config.alpha).to(device)
    pln = PLN(dims, config, prompts, device).to(device)
    optimizer_gfr = torch.optim.Adam(gfr.parameters(), lr=config.lr, weight_decay=config.reg)
    optimizer_pln = torch.optim.Adam(pln.parameters(), lr=config.lr, weight_decay=config.reg)
    pln.apply(my_weight_init)
    history = []
    best_correct = -1
    best_epoch = None
    model_file = output / 'best.pt'

    for epoch in range(config.epochs):
        gfr.train()
        pln.train()
        optimizer_gfr.zero_grad()
        optimizer_pln.zero_grad()
        fused, graphs, scores = gfr(raw, modalities, eye)
        log_probs = pln(fused)
        classification = ClsLoss(log_probs, y, train_tensor, class_weights)
        structure = sum(
            GraphConstructLoss(x, adj, config.theta_smooth, config.theta_degree, config.theta_sparsity)
            for x, adj in zip(modalities, graphs)
        ) / len(graphs)
        supervision = classification.new_zeros(())
        if config.graph_supervision != 'none':
            supervision = graph_supervision(scores, train_tensor, target_edges, config.graph_supervision)
        loss = config.eta_classification * classification + config.lambda_graph * (structure + supervision)
        loss.backward()
        optimizer_gfr.step()
        optimizer_pln.step()

        gfr.eval()
        pln.eval()
        with torch.no_grad():
            fused, _, _ = gfr(raw, modalities, eye)
            post_update = pln(fused)
            predictions = post_update.argmax(1).cpu().numpy()
        correct = int(np.count_nonzero(predictions[test] == labels[test]))
        loss_values = torch.stack((classification, structure, supervision, loss)).detach().cpu().tolist()
        if not np.isfinite(loss_values).all():
            raise FloatingPointError(f'Non-finite loss at fold {fold} epoch {epoch}')
        trn_acc = float(np.mean(predictions[train] == labels[train]))
        test_acc = correct / len(test)
        history.append([epoch, *loss_values, trn_acc, test_acc])
        if correct > best_correct:
            best_correct = correct
            best_epoch = epoch
            torch.save({
                'fold': fold, 'epoch': epoch, 'gfr': gfr.state_dict(), 'pln': pln.state_dict(),
            }, model_file)

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f'fold {fold:02d} epoch {epoch + 1:03d}/{config.epochs} '
                  f'loss={loss_values[-1]:.5f} test_acc={test_acc:.4f}', flush=True)

    checkpoint = torch.load(model_file, weights_only=True, map_location=device)
    gfr.load_state_dict(checkpoint['gfr'])
    pln.load_state_dict(checkpoint['pln'])
    with torch.no_grad():
        fused, adjacency, scores = gfr(raw, modalities, eye)
        log_probs = pln(fused)
        predictions = log_probs.argmax(1).cpu().numpy()[test]
        probabilities = log_probs.exp().cpu().numpy()[test]
        relations = relation_audit(gfr, adjacency, scores, train_tensor, target_edges)
    metrics = evaluate(labels[test], predictions, probabilities)
    if metrics['acc'] != best_correct / len(test):
        raise RuntimeError(f'Checkpoint accuracy changed during reload of fold {fold}')
    metrics.update({
        'fold': fold,
        'best_epoch_zero_based': best_epoch,
        'train_size': len(train),
        'test_size': len(test),
        'class_weight_counter_order': list(map(int, counts)),
        'class_weights_in_loss_index_order': weights.tolist(),
        'seconds': time.monotonic() - start,
        'graph_relations': relations,
    })
    pd.DataFrame(history, columns=[
        'epoch_zero_based', 'classification_loss', 'graph_structure_loss', 'graph_supervision_loss',
        'total_loss', 'train_acc', 'outer_test_acc',
    ]).to_csv(output / 'history.csv', index=False)
    frame = pd.DataFrame({
        'fold': fold, 'row_index': test, 'label': labels[test], 'prediction': predictions,
    })
    for index, name in enumerate(CLASS_NAMES):
        frame[f'p_{name}'] = probabilities[:, index]
    frame.to_csv(output / 'predictions.csv', index=False, float_format='%.9g')
    torch.save({
        'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
    }, output / 'next_fold_rng.pt')
    write_json(output / 'metrics.json', metrics)
    return metrics


def aggregate(output, results):
    overall = {}
    for metric in ('acc', 'auc_hard', 'auc_probability'):
        values = [result[metric] for result in results]
        overall[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    for metric in ('sensitivity', 'specificity'):
        overall[metric] = {}
        for name in CLASS_NAMES:
            values = [result[metric][name] for result in results]
            overall[metric][name] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    predictions = pd.concat(
        [pd.read_csv(output / f'fold_{result["fold"]:02d}' / 'predictions.csv') for result in results],
        ignore_index=True,
    ).sort_values('row_index')
    predictions.to_csv(output / 'out_of_fold_predictions.csv', index=False, float_format='%.9g')
    write_json(output / 'summary.json', {'overall': overall, 'folds': results})


def run(args):
    epochs = args.epochs
    variant = args.variant
    config = SimpleNamespace(**(DEFAULTS | VARIANTS[variant]))
    if epochs is not None:
        config.epochs = epochs
    folds_to_run = args.folds_to_run or config.folds
    if not 1 <= folds_to_run <= config.folds or config.epochs <= 0:
        raise ValueError('Use positive epoch and fold counts within the configured split')
    full = config.epochs == DEFAULTS['epochs'] and folds_to_run == DEFAULTS['folds']
    output = HERE / ('runs' if full else 'smoke') / variant
    config.executed_folds = folds_to_run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_rng(config.seed)
    features, labels, dims, info = load_data(config, device)
    y = torch.from_numpy(labels).to(device)
    prompt_sets = {
        'public': PUBLIC_PROMPTS,
        'tadpole': TADPOLE_PROMPTS,
        'mmgpl_concepts': MMGPL_CONCEPTS,
        'mmgpl_class_prompts': MMGPL_CLASS_PROMPTS,
    }
    prompts = prompt_sets[config.prompts]
    output.mkdir(parents=True, exist_ok=True)
    manifest = experiment_manifest(variant, config, prompts, device, info)
    config_path = output / 'config.json'
    original_runner = HERE / 'references/run_as_executed.py'
    if full and original_runner.is_file() and (output / 'summary.json').is_file():
        # A completed run keeps the exact source hash recorded during training.
        recorded = read_json(config_path)['source_sha256']['run.py']
        if recorded == sha256(original_runner) and all(
            (output / f'fold_{fold:02d}' / 'metrics.json').is_file()
            for fold in range(1, folds_to_run + 1)
        ):
            manifest['source_sha256']['run.py'] = recorded
    save_or_check(config_path, manifest)
    folds = all_folds(labels, config)
    save_or_check(output / 'splits.json', [
        {'fold': index, 'train': train.tolist(), 'test': test.tolist()}
        for index, (train, test) in enumerate(folds, start=1)
    ])

    results = []
    initialized = False
    for fold, (train, test) in enumerate(folds[:folds_to_run], start=1):
        fold_output = output / f'fold_{fold:02d}'
        if fold_output.is_dir():
            metrics = read_json(fold_output / 'metrics.json')
            print(f'{variant} fold {fold:02d} already complete', flush=True)
        else:
            if results and not initialized:
                # Restore the text cache before restoring training RNG from the last completed fold.
                texts = [text for group in prompts.values() for text in group]
                prompt_features(texts, features.device, 'frozen_pretrained')
                prior = output / f'fold_{fold - 1:02d}' / 'next_fold_rng.pt'
                state = torch.load(prior, weights_only=True, map_location='cpu')
                torch.set_rng_state(state['cpu'])
                torch.cuda.set_rng_state_all(state['cuda'])
            staging = output / f'.fold_{fold:02d}.working'
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            metrics = train_fold(fold, config, prompts, features, y, dims, labels, train, test, staging)
            staging.rename(fold_output)
            initialized = True
            print(f'{variant} fold {fold:02d} complete  epoch={metrics["best_epoch_zero_based"]} '
                  f'acc={metrics["acc"]:.4f} hard_auc={metrics["auc_hard"]:.4f} '
                  f'prob_auc={metrics["auc_probability"]:.4f} time={metrics["seconds"]:.1f}s', flush=True)
        results.append(metrics)
    aggregate(output, results)
    print(f'{variant} complete  {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=tuple(VARIANTS))
    parser.add_argument('--epochs', type=int, help='Smoke test only, unless the full 300 epochs are requested')
    parser.add_argument('--folds-to-run', type=int, help='Run the first N folds for a smoke test')
    run(parser.parse_args())
