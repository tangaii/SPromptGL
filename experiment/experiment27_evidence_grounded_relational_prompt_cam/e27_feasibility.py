"""Read-only, CPU Stage-0 audit. Never imports the CUDA model or fits teachers.

Clinical sidecars are joined for availability audit ONLY, not added to inputs.
No association/permutation/OOF test is run unless the semantic Gate 0 passes.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from e27_contract import (ANAT, FUNC, DIMENSIONS, ROLES, P0_TEXT, RELATIONS,
                          HISTORY, SOURCES, decide_gate0)

CORE = ('main.py', 'model.py', 'network.py', 'layers.py', 'utils.py',
        'ABIDE-simple-2-concat-weighted-cosine.sh')
DATA = ('data/ABIDE/processed_standard_data.csv', 'data/ABIDE/modal_feat_dict.npy',
        'data/ABIDE/data_processing.ipynb', 'data/ABIDE/phenotyic_processing.ipynb',
        'dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def history_dirs(root):
    found = {}
    for k in range(1, 27):
        candidates = [p for b in (root / 'experiment', Path('/tmp'))
                      for p in sorted(b.glob(f'experiment{k}_*'))
                      if (p / 'results/report.md').is_file()]
        if candidates:
            found[k] = candidates[0]
    for k in HISTORY:
        assert k in found, f'Missing mandatory history {k}'
    return found


def protected_snapshot(root):
    """Protect all discovered experiment 1–26 files, including small checkpoints."""
    files = {root / n for n in (*CORE, *DATA)}
    files.update(root.glob('*.pdf'))
    if (root / 'config.py').exists():
        files.add(root / 'config.py')
    files.add(root / 'prompt')
    for d in history_dirs(root).values():
        files.update(p for p in d.rglob('*') if p.is_file())
    return {str(p): sha(p) for p in sorted(files)}


def source_audit(root):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
    remote = subprocess.check_output(['git', 'ls-remote', 'https://github.com/tangaii/SPromptGL.git',
                                      'refs/heads/main'], text=True).split()[0]
    def compare(name):
        url = f'https://raw.githubusercontent.com/tangaii/SPromptGL/{remote}/{name}'
        body = urllib.request.urlopen(url, timeout=40).read()
        local_sha = sha(root / name)
        remote_sha = hashlib.sha256(body).hexdigest()
        assert local_sha == remote_sha, f'Core source mismatch: {name}'
        return name, {'local_sha256': local_sha, 'remote_sha256': remote_sha,
                      'equal': True, 'remote_url': url}
    with ThreadPoolExecutor(max_workers=4) as pool:
        comparisons = dict(pool.map(compare, CORE))
    text = {n: (root / n).read_text() for n in CORE}
    for n in CORE:
        if n.endswith('.py'):
            ast.parse(text[n])
    tree = ast.parse(text['network.py'])
    prompts = next(ast.literal_eval(x.value) for x in tree.body
                   if isinstance(x, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'attr_words' for t in x.targets))
    assert tuple(prompts) == P0_TEXT
    checks = {
        'prompt_list': ('network.py', 'attr_words = ['),
        'clip_cached': ('network.py', 'def get_prompt_text_features():'),
        'clip_rn50': ('network.py', "clip.load('RN50', device)"),
        'mlp_4096_128_9': ('network.py', 'self.MLP = MLP(input_dim = 4096,hidden_dim = 128, output_dim = 9)'),
        'first_cam': ('network.py', 'x, _attn = self.InputLayer(x)'),
        'encoder': ('network.py', 'x, _attn = self.Encoder[i]'),
        'latent_tokens': ('network.py', 'part_x = x.view(871, 4, 4, 9)'),
        'spgs_softmax': ('network.py', 'similarity_matrix = torch.softmax(similarity_matrix'),
        'spgs_max': ('network.py', 'similarity_max, _ = torch.max(similarity_matrix'),
        'spgs_weight': ('network.py', 'part_x = similarity_max.unsqueeze(-1) * part_x'),
        'res': ('network.py', 'x = x + torch.cat(combine_part,dim=2)'),
        'gfr_pheno': ('network.py', 'combined_feat = f_w*feat0+torch.cat(x,dim=1)'),
        'gfr_anat_overwrite': ('network.py', 'combined_feat = f_w*feat1+torch.cat(x,dim=1)'),
        'attention_min': ('layers.py', 'attn = attn/abs(attn.min())'),
        'attention_normalize_softmax': ('layers.py', 'attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))'),
        'input_layer': ('layers.py', 'class VariLengthInputLayer'),
        'encode_layer': ('layers.py', 'class EncodeLayer'),
        'test_as_val': ('model.py', 'self.val_idx = np.array(test_index)'),
        'test_index': ('model.py', 'self.tst_idx = np.array(test_index)'),
        'adam_mp': ('model.py', 'self.optimizer_MP0 = optim.Adam'),
        'adam_mf': ('model.py', 'self.optimizer_MF = optim.Adam'),
        'aloss_computed': ('model.py', 'feat, adj, ALoss = self.MessagePassing0'),
        'four_graph_losses': ('model.py', 'all_graph_loss = all_graph_loss/4'),
        'simple2_loss_no_aloss': ('model.py', 'loss = cls_loss + all_graph_loss'),
        'cls_loss': ('utils.py', 'return F.nll_loss(output[idx], labels[idx], weight)'),
        'graph_loss': ('utils.py', 'def GraphConstructLoss('),
        'historical_cv': ('main.py', 'skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)'),
        'preprocessed_input': ('main.py', "pd.read_csv(path + 'processed_standard_data.csv')"),
        'checkpoint_strict': ('main.py', 'if cur_val_acc > best_val_acc:'),
        'early_strict': ('main.py', 'if wait_cnt > hyperpm.early:'),
    }
    anchors = {}
    for key, (name, needle) in checks.items():
        lines = [(i + 1, l.strip()) for i, l in enumerate(text[name].splitlines()) if needle in l]
        assert len(lines) == 1, (key, lines)
        anchors[key] = {'file': name, 'line': lines[0][0], 'text': lines[0][1]}
    # ALoss is not a dependency of the simple-2 assigned total loss.
    assert anchors['simple2_loss_no_aloss']['text'] == 'loss = cls_loss + all_graph_loss'
    assert text['network.py'].count('adj = self.GraphConstruct') == 4
    notebooks = {}
    for name in DATA[2:4]:
        nb = json.loads((root / name).read_text())
        cells = {str(i): ''.join(c['source']) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'}
        selected = {i: s for i, s in cells.items() if any(x in s for x in (
            'RFE(', 'skf', 'feature_selection(features', 'StandardScaler', 'pd.get_dummies',
            'pd.cut', 'pd.concat', "feat_list", 'anat =', 'anat_list', 'func_list'))}
        notebooks[name] = {'sha256': sha(root / name), 'relevant_code_cells': selected}
    return {'local_head': head, 'remote_main_head': remote, 'core_comparisons': comparisons,
            'config_py': {'exists': (root / 'config.py').exists(),
                          'note': '当前根目录无config.py；参数定义在main.py及历史shell，未创建替代文件。'},
            'initial_git_status': status, 'anchors': anchors, 'notebooks': notebooks,
            'original_eight_prompts': prompts, 'p0_prompt_sha256': digest(prompts)}


def data_audit(root):
    table = pd.read_csv(root / DATA[0])
    mapping = np.load(root / DATA[1], allow_pickle=True).item()
    assert list(mapping) == list(DIMENSIONS)
    assert [len(v) for v in mapping.values()] == list(DIMENSIONS.values())
    assert table.shape == (871, 321) and np.isfinite(table.to_numpy(float)).all()
    expected = [str(c) for cols in mapping.values() for c in cols]
    assert table.columns[:-1].tolist() == expected
    assert list(mapping['ANAT']) == list(ANAT) and list(mapping['FUNC']) == list(FUNC)
    assert list(mapping['Correlation']) == list(map(str, range(256)))
    ph = list(mapping['PHENO'])
    assert sum(c.startswith('AGE_AT_SCAN_') for c in ph) == 26
    assert sum(c.startswith('SITE_ID_') for c in ph) == 20
    assert sum(c.startswith('SEX_') for c in ph) == 2
    assert table.label.value_counts().to_dict() == {2.: 468, 1.: 403}
    for a, b in ((0, 26), (26, 46), (46, 48)):
        assert set(np.unique(table.iloc[:, a:b].to_numpy())) <= {0., 1.}
        assert (table.iloc[:, a:b].sum(1) == 1).all()
    clinical_input = [c for c in expected if re.search(r'ADOS|ADAS|ADI_|SRS|MMSE|MOCA|^FIQ$', c.upper())]
    assert not clinical_input
    groups = {}
    start = 0
    for name, cols in mapping.items():
        groups[name] = {'columns': [str(x) for x in cols], 'dimensions': len(cols),
                        'slice': [start, start + len(cols)], 'role': ROLES[name]}
        start += len(cols)
    qc_definitions = {
        'anat_cnr': '灰/白质对比噪声比；不是灰白质体积',
        'anat_efc': '图像强度熵焦点指标；不是神经系统熵或厚度',
        'anat_fber': '头部前景/背景能量比', 'anat_fwhm': '强度图像空间平滑度FWHM，原始体素单位；不是皮层厚度',
        'anat_qi1': '背景伪影体素比例', 'anat_snr': '灰质信号/背景噪声比；不是脑容量',
        'func_efc': '平均功能图像强度熵焦点指标', 'func_fber': '平均功能图像前景/背景能量比',
        'func_fwhm': '平均功能图像空间平滑度', 'func_dvars': '标准化时间差分变动幅度',
        'func_outlier': '时间序列每帧异常体素比例的均值', 'func_quality': '每帧对中位图像的秩相关距离',
        'func_mean_fd': '平均头动帧间位移，原始毫米', 'func_num_fd': '超过头动阈值的体积/帧数',
        'func_perc_fd': '超过头动阈值的帧比例/百分比',
        'func_gsr': 'ghost-to-signal ratio；不是global signal regression开关',
    }
    return table, {'shape': [871, 320], 'labels': {'0': 'ASD', '1': 'TD'},
                   'class_counts': {'ASD': 403, 'TD': 468}, 'field_order_exact': True,
                   'groups': groups, 'clinical_columns_in_fixed_input': clinical_input,
                   'genuine_disease_measurement_groups': ['Correlation'],
                   'qc_definitions': qc_definitions,
                   'qc_units_in_model': '16列都已由历史全队列StandardScaler转为无量纲z分数',
                   'correlation_units': '逐患者LedoitWolf相关上三角的Fisher-z；本CSV中没有后续统一z标准化',
                   'preprocessing': '旧5折最后697人拟合RFE、CV前16列QC标准化、按行concat；本轮均不重算',
                   'data_sha256': {n: sha(root / n) for n in DATA},
                   'patient_reordering_in_experiment27': False}


def identity_and_clinical(root, table):
    folder = root / 'experiment/experiment11_roi_mapping_recovery/results'
    rowmap = pd.read_csv(folder / 'subject_row_mapping.csv')
    raw = pd.read_csv(root / DATA[4], low_memory=False)
    old = json.loads((folder / 'data_audit.json').read_text())
    assert old['inputs']['csv_sha256'] == sha(root / DATA[0])
    assert old['inputs']['phenotype_sha256'] == sha(root / DATA[4])
    assert len(rowmap) == 871 and rowmap.csv_row_zero_based.tolist() == list(range(871))
    assert rowmap.subject_SUB_ID.nunique() == 871 and raw.SUB_ID.is_unique
    selected = raw[raw.SUB_ID.isin(rowmap.subject_SUB_ID)].copy()
    assert len(selected) == 871
    age = selected.AGE_AT_SCAN
    bins = [float(age.min())]
    while bins[-1] < age.max():
        bins.append(bins[-1] + 2)
    reconstruct = selected.copy()
    reconstruct['AGE_AT_SCAN'] = pd.cut(age, bins, right=False)
    for key in ('AGE_AT_SCAN', 'SITE_ID', 'SEX'):
        reconstruct = pd.get_dummies(reconstruct, columns=[key])
    qc = list(ANAT + FUNC)
    # Reproduces historical preprocessing ONLY to verify identity, never fits a model.
    reconstruct[qc] = StandardScaler().fit_transform(reconstruct[qc])
    cols = table.columns[:64].tolist()
    vectors = reconstruct[cols].to_numpy(float)
    index = {np.round(v, 8).tobytes(): i for i, v in enumerate(vectors)}
    assert len(index) == 871
    matched = np.array([index[np.round(v, 8).tobytes()] for v in table.iloc[:, :64].to_numpy(float)])
    assert len(np.unique(matched)) == 871
    matched_ids = selected.SUB_ID.to_numpy()[matched]
    assert np.array_equal(matched_ids, rowmap.subject_SUB_ID.to_numpy())
    error = np.abs(vectors[matched] - table.iloc[:, :64].to_numpy(float))
    assert error.max() < 1e-10
    cohort = selected.iloc[matched].reset_index(drop=True)
    assert np.array_equal(cohort.DX_GROUP.to_numpy(), table.label.to_numpy())
    named = pd.read_csv(folder / 'roi_256_named_mapping.csv')
    atlas = pd.read_csv(folder / 'roi_111_labels.csv')
    assert len(named) == 256 and len(atlas) == 111
    assert (named.verification_status == 'VERIFIED_NUMERIC_UNIQUE').all()
    excluded = named.loc[~named.usable_for_roi_semantic_prompt.astype(bool), 'correlation_feature_index'].tolist()
    assert excluded == [195, 225, 232, 236]
    assert named.correlation_feature_index.tolist() == list(range(256))
    tri = np.triu_indices(111, 1)
    assert np.array_equal(named.roi_i_zero_based, tri[0][named.original_connection_index.to_numpy(int)])
    assert np.array_equal(named.roi_j_zero_based, tri[1][named.original_connection_index.to_numpy(int)])
    old21 = root / 'experiment/experiment21_evidence_constrained_adaptive_alignment/results/clinical_audit.json'
    ca = json.loads(old21.read_text())
    fields = [c for c in cohort if c.startswith(('ADOS_', 'ADI_', 'SRS_')) or c in ('FIQ', 'VIQ', 'PIQ', 'ADI_RRB_TOTAL_C')]
    metadata = {'ADOS_MODULE', 'ADOS_RSRCH_RELIABLE', 'ADI_R_RSRCH_RELIABLE', 'SRS_VERSION'}
    coverage = []
    for field in fields:
        value = pd.to_numeric(cohort[field], errors='coerce')
        rows = {}
        for dx, diagnosis in ((1, 'ASD'), (2, 'TD')):
            mask = cohort.DX_GROUP == dx
            observed = value.notna() & (value >= 0)
            rows[diagnosis] = {'total': int(mask.sum()),
                               'nonnegative_observed': int((mask & observed).sum()),
                               'missing': int((mask & value.isna()).sum()),
                               'negative_codes': int((mask & (value < 0)).sum()),
                               'sites_with_nonnegative_values': int(cohort.loc[mask & observed, 'SITE_ID'].nunique())}
        coverage.append({'field': field, 'is_metadata_not_score': field in metadata,
                         'coverage': rows, 'included_in_model': False,
                         'clinical_eligibility_established': False})
    return {'status': 'ROW_IDENTITY_REVERIFIED_871', 'matched_subjects': 871,
            'fingerprint': '48 PHENO + 16 QC, rounded8 unique bijection; independent of diagnosis labels',
            'fingerprint_max_absolute_error': float(error.max()),
            'fingerprint_rmse': float(np.sqrt(np.mean(error ** 2))),
            'row_mapping_sha256': sha(folder / 'subject_row_mapping.csv'),
            'identity_digest': hashlib.sha256(matched_ids.astype('<i8').tobytes()).hexdigest(),
            'labels_only_secondary_integrity_check': True,
            'roi_mapping': {'features': 256, 'numeric_verified': 256, 'named_usable': 252,
                            'excluded': excluded, 'roi_count': 111,
                            'reconstructed_all_fc_again': False,
                            'prior_full_fc_verification_max_error': old['connectivity_reconstruction']['max_absolute_error'],
                            'current_atlas_and_index_checks_pass': True},
            'historical_contradiction': {
                'experiment11': '871人PHENO/QC与FC指纹已经建立身份链',
                'experiment21': ca['status'], 'experiment21_verified_rows': ca['verified_processed_rows'],
                'cause': '实验21只搜索subject_IDs.txt，未消费实验11已存在的subject_row_mapping.csv与Notebook指纹证据',
                'current_resolution': '本轮复用旧映射并独立重建871x64，身份可核验；不沿用0/871结论',
                'historical_files_edited': False},
            'sidecar_scope': '当前871人按唯一身份链接后的可用性审计，不是1112全源表的覆盖率',
            'sidecar_coverage': coverage,
            'coverage_caveat': '只统计非负观测/缺失/负值代码，不等于量表合法范围、版本、研究可靠度及跨站点可比性验证；缺失不可作为诊断捷径',
            'clinical_source_added_to_input': False,
            'diagnosis_based_coverage_is_descriptive_only': True,
            'missing_input_families': [s for s in ('ADAS', 'ADOS', 'ADI-R', 'SRS', 'FIQ', 'MRI morphometry', 'DTI')],
            'raw_adas_present': any(c.startswith('ADAS') for c in cohort)}


def references(root):
    paths = {
        'P0': (root / 'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv', 'F0', 'ba'),
        'E9': (root / 'experiment/experiment9_prompt_disease_adversarial_interaction/results/fold_metrics.csv', 'E9', 'balanced_accuracy'),
    }
    expected = {'P0': (.9169105, .9153935, .933352), 'E9': (.9188262, .9169151, .9382272)}
    result = {}
    for label, (path, config, ba_name) in paths.items():
        df = pd.read_csv(path)
        sub = df[df.config == config].copy()
        assert len(sub) == 30 and set(zip(sub.seed, sub.fold)) == {(s, f) for s in range(3) for f in range(1, 11)}
        assert sub[['acc', ba_name, 'probability_auc']].notna().all().all()
        byseed = sub.groupby('seed')[['acc', ba_name, 'probability_auc']].mean()
        means = byseed.mean().to_numpy()
        assert np.max(np.abs(means - np.array(expected[label]))) < 5e-7
        result[label] = {'file': str(path), 'sha256': sha(path), 'config': config,
                         'folds': 30, 'historical_only': True,
                         'mean': dict(zip(('acc', 'ba', 'probability_auc'), map(float, means))),
                         'population_sd': dict(zip(('acc', 'ba', 'probability_auc'), map(float, byseed.std(ddof=0)))),
                         'seed_means': {str(k): dict(zip(('acc', 'ba', 'probability_auc'), map(float, v))) for k, v in byseed.iterrows()}}
    return result


def ledger(input_contract):
    result = []
    for proposal in RELATIONS:
        r = copy.deepcopy(proposal)
        pair = r['modality_pair']
        r['actual_modalities'] = {p: input_contract['groups'][p] for p in pair}
        r['two_disease_measurement_endpoints'] = all(ROLES[p] == 'DISEASE_RELEVANT_FC_MEASUREMENT' for p in pair) and pair[0] != pair[1]
        r['measurable_in_fixed_871_inputs'] = all(r['actual_endpoint_fields'])
        # No unsupported clinical direction is promoted into a verified fact.
        r['independently_testable_prespecified_relation'] = False
        r['current_implementable'] = False
        r['action'] = 'REJECT_NEW_CORRECTION'
        r['original_cam_preserved'] = True
        result.append(r)
    return result


def collect(root):
    sources = source_audit(root)
    table, inputs = data_audit(root)
    clinical = identity_and_clinical(root, table)
    refs = references(root)
    rows = ledger(inputs)
    decision = decide_gate0(rows)
    assert decision['status'] == 'STOP_UNGROUNDED_INPUTS', 'Changed input contract: audit before any G1 implementation'
    histories = {}
    directories = history_dirs(root)
    for k, summary in HISTORY.items():
        d = directories[k]
        report = d / 'results/report.md'
        report.read_text()  # Audit the complete report; preserve its source hash.
        code = {}
        for p in sorted(d.glob('*.py')):
            source = p.read_text()
            ast.parse(source)
            code[p.name] = {'sha256': sha(p), 'lines': len(source.splitlines())}
        histories[str(k)] = {'directory': str(d), 'report_sha256': sha(report),
                             'source_files_reviewed': code, 'conclusion': summary}
    return {'status': decision['status'], 'stage': 'GROUNDING_ONLY', 'science': 'NOT_EVALUABLE',
            'repo': sources, 'input_contract': inputs, 'identity_and_clinical': clinical,
            'grounding_ledger': rows, 'gate0': decision,
            'gate1': {'status': 'NOT_RUN_G0_FAILED', 'fold_statistics': None,
                      'permutation_nulls': None, 'oof_gain': None,
                      'reason': '没有两项真实疾病测量关系，不用QC/年龄相关性替代G1。'},
            'stage2': {'status': 'NOT_STARTED_G0_FAILED', 'completed_folds': 0,
                       'gpu_folds': {'0': 0, '1': 0}, 'workers_launched': 0,
                       'model_sources_created': False, 'new_model_metrics': None},
            'references': refs, 'history_review': histories,
            'papers_and_official_documents': SOURCES,
            'label_boundary': {'gate0_accepts_labels': False,
                               'labels_used_for_descriptive_counts_and_reference_validation_only': True,
                               'no_association_teacher_gate_or_model_fitted': True}}
