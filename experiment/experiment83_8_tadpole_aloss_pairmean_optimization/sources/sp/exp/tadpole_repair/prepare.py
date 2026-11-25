"""Deterministic NC/MCI/AD preprocessing derived from MMGL's TADPOLE notebook."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = Path('/model/spromptgl/Tadpole/tadpole_challenge')
MMGL_NOTEBOOK = (
    'https://github.com/SsGood/MMGL/blob/c13b05c190f85e614ccb3ace60bb6b152a1f4a1b/'
    'data/TADPOLE/Data_preprocessing_TADPOLE%20%282%29.ipynb'
)
MODALITIES = ('UCSFFSX', 'UCBERKELEYAV45', 'UPENNBIOMK9', 'RISK_FACTOR',
              'COGNITIVE_TEST', 'ROI_AVERAGE')
VISITS = ['bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48']
CSF_REPLACEMENTS = {
    'ABETA_UPENNBIOMK9_04_19_17': {'<200': 200},
    'TAU_UPENNBIOMK9_04_19_17': {'<80': 80, '>1300': 1300},
    'PTAU_UPENNBIOMK9_04_19_17': {'<8': 8, '>120': 120},
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_visits(raw_csv):
    data = pd.read_csv(raw_csv, low_memory=False)
    columns = data.columns[~data.columns.str.endswith('_bl')]
    columns = columns[~columns.str.contains('UCSFFSL', case=False)]
    data = data[columns]
    data = data.where(data.ne(' ')).dropna(axis=1, how='all')
    data = pd.concat([data.loc[data.DXCHANGE.eq(label)] for label in (1, 2, 3)])
    return data.loc[data.VISCODE.isin(VISITS)].rename(columns={'DXCHANGE': 'LABEL'}).copy()


def encode_discrete(data):
    for name, replacements in CSF_REPLACEMENTS.items():
        data[name] = pd.to_numeric(data[name].replace(replacements), errors='raise')
    for name in ('AGE', 'PTEDUCAT'):
        values = data[name]
        edge = values.min()
        bins = [edge]
        while edge < values.max():
            edge += 2
            bins.append(edge)
        data[name] = pd.cut(values, bins, right=False)
    data['PTGENDER'] = data.PTGENDER.map({'Male': 1, 'Female': 0})
    return pd.get_dummies(data, columns=['AGE', 'PTMARRY', 'PTEDUCAT', 'APOE4'])


def select_features(data, dictionary):
    dictionary = dictionary.loc[~dictionary.FLDNAME.str.contains(
        r'RID|SID|UID|PTID|VISCODE|DATE|FLDSTRENG|VERSION|STATUS|NAME|VISIT', na=False)]
    candidates = {
        name: dictionary.loc[dictionary.TBLNAME.eq(name), 'FLDNAME']
        for name in MODALITIES[:2]
    }
    qc = ('OVERALLQC', 'TEMPQC', 'FRONTQC', 'PARQC', 'INSULAQC', 'OCCQC',
          'BGQC', 'CWMQC', 'VENTQC')
    mri_qc_columns = {f'{name}_UCSFFSX_11_02_15_UCSFFSX51_08_01_16' for name in qc}
    candidates['UCSFFSX'] = set(candidates['UCSFFSX']).difference(mri_qc_columns)
    candidates['UPENNBIOMK9'] = CSF_REPLACEMENTS.keys()
    for name, pattern in {
        'RISK_FACTOR': r'AGE_|PTMARRY|PTEDUCAT|APOE4|PTGENDER',
        'COGNITIVE_TEST': r'CDRSB|ADAS|MMSE|RAVLT|FAQ|MOCA|Ecog',
        'ROI_AVERAGE': r'^(?:FDG|AV45|Ventricles|Hippocampus|WholeBrain|Entorhinal|Fusiform|MidTemp|ICV)$',
    }.items():
        candidates[name] = data.columns[data.columns.str.contains(pattern)]

    selected = {}
    details = {}
    for name in MODALITIES:
        features = sorted(set(candidates[name]).intersection(data.columns))
        complete = data[features + ['LABEL']].dropna()
        if not features or complete.empty:
            raise ValueError(f'No eligible records or features for {name}')
        keep = features
        if len(features) > 150:
            # The notebook fits RFE to the unscaled complete records.
            selector = RFE(RidgeClassifier(), n_features_to_select=150, step=200)
            selector.fit(complete[features], complete.LABEL)
            keep = np.asarray(features)[selector.support_].tolist()
        selected[name] = keep
        details[name] = {'candidates': len(features), 'complete_visits_for_rfe': len(complete),
                         'selected_before_missing_filter': len(keep)}
    return selected, details


def prepare(raw, output, reference):
    raw_csv = raw / 'TADPOLE_D1_D2.csv'
    dict_csv = raw / 'TADPOLE_D1_D2_Dict.csv'
    data = encode_discrete(load_visits(raw_csv))
    n_visits = len(data)
    with threadpool_limits(limits=1, user_api='blas'):
        selected, selection_details = select_features(data, pd.read_csv(dict_csv))
    required = [column for name in MODALITIES[2:] for column in selected[name]]
    data = data.dropna(subset=required)
    n_complete = len(data)
    missing_limit = int(n_complete * 0.15)
    available = set(data.columns[data.isna().sum().lt(missing_limit)])
    selected = {name: [column for column in columns if column in available]
                for name, columns in selected.items()}
    columns = [column for features in selected.values() for column in features]
    if len(columns) != len(set(columns)):
        raise ValueError('Feature appears in more than one modality')

    # Preserve notebook row order, including the NC/MCI/AD concatenation.
    subjects = data.drop_duplicates(subset='RID', keep='last')
    scaler = StandardScaler()
    standardized = pd.DataFrame(scaler.fit_transform(subjects[columns]),
                                index=subjects.index, columns=columns)
    standardized = standardized.fillna(standardized.mean())
    processed = standardized.assign(LABEL=subjects.LABEL.astype(float))
    samples = subjects[['RID', 'VISCODE', 'LABEL']].astype({'RID': int, 'LABEL': int})
    samples.insert(0, 'raw_row_index', samples.index)
    samples.insert(0, 'row_index', range(len(samples)))
    dims = [len(values) for values in selected.values()]
    counts = {str(key): int(count) for key, count in samples.LABEL.value_counts().sort_index().items()}
    if counts != {'1': 209, '2': 315, '3': 74} or not np.isfinite(processed.to_numpy()).all():
        raise ValueError(f'Unexpected TADPOLE cohort: {processed.shape}, class counts {counts}')

    reference_csv = reference / 'processed_standard_data.csv'
    reference_dict = reference / 'modal_feat_dict.npy'
    original = pd.read_csv(reference_csv)
    reference_modal = np.load(reference_dict, allow_pickle=True).item()
    same_features = {name: set(selected[name]) == set(reference_modal[name]) for name in MODALITIES}
    common = [name for name in processed if name in original]
    same_label_order = np.array_equal(original.LABEL, processed.LABEL)
    max_difference = float(np.abs(processed[common].to_numpy() - original[common].to_numpy()).max())

    output.mkdir(parents=True, exist_ok=False)
    processed.to_csv(output / 'processed_standard_data.csv', index=False)
    samples.to_csv(output / 'samples.csv', index=False)
    np.save(output / 'modal_feat_dict.npy', selected)
    write_json(output / 'features.json', selected)
    manifest = {
        'preprocessing_source': MMGL_NOTEBOOK,
        'raw_sources': {str(path): sha256(path) for path in (raw_csv, dict_csv)},
        'output_hashes': {path.name: sha256(path) for path in sorted(output.iterdir())},
        'shape': list(processed.shape), 'modalities': list(selected), 'dimensions': dims,
        'class_counts': counts, 'visits_by_code': samples.VISCODE.value_counts().sort_index().to_dict(),
        'visit_filter': VISITS, 'eligible_visits': n_visits, 'complete_visits': n_complete,
        'visit_selection': 'keep last qualifying row per RID in label-concatenated source order',
        'missing_value_rule': f'missing visit count < {missing_limit}',
        'feature_order': 'fixed modality order and lexical column order within each modality',
        'feature_selection': selection_details,
        'fit_scope': 'full cohort RFE, standardization and mean imputation per MMGL',
        'reference_comparison': {
            'csv_sha256': sha256(reference_csv), 'dictionary_sha256': sha256(reference_dict),
            'same_feature_sets_by_modality': same_features, 'same_label_order': same_label_order,
            'common_columns': len(common), 'max_abs_difference_after_alignment': max_difference,
        },
    }
    write_json(output / 'preprocessing.json', manifest)
    print(f'Wrote {len(subjects)} subjects, modality dimensions {dims}, class counts {counts}')
    print(f'Same feature sets as previous input: {all(same_features.values())}')
    print(f'Same label order: {same_label_order}; common-column max abs difference: {max_difference:g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--reference', type=Path, default=ROOT / 'data' / 'TADPOLE')
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    prepare(args.raw, args.output, args.reference)
