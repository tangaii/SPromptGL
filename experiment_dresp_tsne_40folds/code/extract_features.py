from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = Path('/data/adodas/SPromptGL-main')
V2 = Path('/model/spromptgl/experiment/experiment_abide_authentic_logs_v2')
DPSL = Path('/model/spromptgl/experiment/DPSL')
JSL = Path('/model/spromptgl/experiment/experiment_darg_jsl')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def capture_e28(fold):
    import numpy as np, torch
    sys.path.insert(0, str(EXP / 'code'))
    import run as e28
    e28.context()
    payload = e28.checkpoint_read(EXP / 'checkpoints/E28_E0_seed0' / f'fold{fold:02d}.best.pt')
    model = e28.make_agent('E0_P0', 0, fold)
    e28.load_models(model, payload['model'])
    captured = {}
    def capture_hook(_m, args):
        captured['feature'] = args[0].detach().clone()
    hook = model.ModalFusion.Outputlayer.register_forward_pre_hook(capture_hook)
    out = e28.evaluate(model)
    hook.remove()
    test = np.asarray(model.tst_idx)
    pred = out.detach().cpu().numpy().argmax(1)[test]
    return captured['feature'].cpu().numpy(), {'prediction': pred, 'test': test}, e28.context()['y'], np.arange(871)


def capture_jsl(fold):
    sys.path[:0] = [str(JSL / 'code'), str(JSL / 'runtime/python'), str(JSL / 'code/history_base')]
    import numpy as np, torch
    import jsl_runtime as rt
    ctx = rt.setup(fold)
    payload = torch.load(JSL / 'checkpoints' / f'fold{fold:02d}.best.pt', map_location=ctx['state']['x'].device, weights_only=False)
    rt.restore(ctx, payload)
    result, capture = rt.forward(ctx, False)
    evaluation = rt.evaluate(ctx)[0]
    return capture['enhanced'].detach().cpu().numpy(), evaluation, ctx['state']['labels_np'], np.arange(871), capture['cam'].detach().cpu().numpy()


def capture_dpsl(fold):
    sys.path[:0] = [str(DPSL / 'code'), str(DPSL / 'runtime/python'), str(DPSL / 'code/history_base')]
    import numpy as np, torch
    import dpsl_runtime as rt
    import real_model_runtime as real
    engine = real.fold_engine(fold)
    ctx = real.setup(fold, engine)
    payload = rt.load_checkpoint(fold, 'best')
    engine.restore(ctx, payload)
    evaluation, capture = engine.evaluate(ctx)
    return capture['enhanced'].detach().cpu().numpy(), evaluation, ctx['state']['labels_np'], np.arange(871), capture['cam'].detach().cpu().numpy()


def capture_darg(fold):
    spec = importlib.util.spec_from_file_location('extract_darg', V2 / 'code/train_darg_e744.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    mod = helper.load_runner()
    import numpy as np, torch
    mod.RT = V2 / 'DARG/runtime'
    mod.TASKS, mod.CKPTS, mod.LOGS = mod.RT / 'tasks', mod.RT / 'checkpoints', mod.RT / 'logs'
    hist = helper.install_history_path_adapter(mod, torch, np)
    hist.load_e69().load_e61().load_base()
    torch.cuda.set_device(0)
    dcr_source = V2 / 'inputs/source_repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py'
    mod.load_dcr().CADR_PATH = dcr_source
    state, _, identity = mod.prepare(fold)
    group = helper.GROUP
    state.update(fold=fold, group=group, rudag_identity=identity)
    path = V2 / 'DARG/runtime/checkpoints' / f'{group}_f{fold:02d}.best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    hist.load_bundle(state, payload['best_state'])
    captured = {}
    def capture_hook(_m, args):
        captured['feature'] = args[0].detach().clone()
    hook = state['cam'].Outputlayer.register_forward_pre_hook(capture_hook)
    out = mod.forward(state, group, training=False, collect=True)
    hook.remove()
    labels = state['labels_np']
    test = state['test_np']
    predicted = out['logp'].detach().cpu().numpy().argmax(-1)[test]
    return captured['feature'].cpu().numpy(), {'prediction': predicted, 'test': test}, labels, np.arange(871)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['SPromptGL', 'DARG', 'DARG_JSL', 'DReSP'])
    parser.add_argument('--fold', required=True, type=int, choices=range(1, 11))
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    functions = {'SPromptGL': capture_e28, 'DARG': capture_darg, 'DARG_JSL': capture_jsl, 'DReSP': capture_dpsl}
    result = functions[args.model](args.fold)
    features, evaluation, labels, patient_ids = result[:4]
    cam = result[4] if len(result) > 4 else features
    import numpy as np
    if features.shape != (871, 144) or not np.isfinite(features).all():
        raise ValueError(f'{args.model} fold {args.fold}: invalid patient feature array {features.shape}')
    if labels.shape != (871,) or not np.isin(labels, [0, 1]).all():
        raise ValueError('invalid ABIDE labels')
    pred = np.asarray(evaluation.get('prediction', []))
    test_idx = np.asarray(evaluation.get('test', np.arange(len(pred))), dtype=np.int64)
    np.savez_compressed(EXP / 'results' / f'{args.model}_fold{args.fold:02d}_features.npz',
                        features=features.astype(np.float32), cam=cam.astype(np.float32), labels=labels.astype(np.int64), patient_ids=patient_ids, test_indices=test_idx, predictions=pred)
    print(json.dumps({'model': args.model, 'fold': args.fold, 'shape': list(features.shape),
                      'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
