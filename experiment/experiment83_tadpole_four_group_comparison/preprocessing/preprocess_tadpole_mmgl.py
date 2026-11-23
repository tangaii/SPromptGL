import os
import pickle
import numpy as np
import pandas as pd
# from emvert_functional_code import emvert_ml
from scipy.sparse import csgraph
import csv

import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

input_dict_df = pd.read_csv('/model/spromptgl/Tadpole/tadpole_challenge/TADPOLE_D1_D2_Dict.csv')
input_df = pd.read_csv('/model/spromptgl/Tadpole/tadpole_challenge/TADPOLE_D1_D2.csv',low_memory=False)
PMCI_selection = False

feature_list = input_df.columns
# remove _bl features
feature_list = feature_list[~feature_list.str.contains('_bl$')]
input_df = input_df[feature_list]

#remove longitudinal MRI features
feature_list = feature_list[~feature_list.str.upper().str.contains('UCSFFSL')]
input_df = input_df[feature_list]
print('remove longitudinal feature', input_df.shape)

#replace " " to nan
cond = input_df != " "
input_df = input_df.where(cond, np.nan)
#########cond = input_df != "-4"
#########input_df = input_df.where(cond, np.nan)

#delete columns with all Nan
feat_nan = []
for feat in feature_list:
    if input_df[feat].isnull().all() == True:
        feat_nan.append(feat)
input_df.drop(feat_nan,axis=1, inplace=True)
print('delete columns with all Nan', input_df.shape)

#select subjects whose state is NL or MCI or AD. Notice: The selected subjects are state-stable
NL_df = input_df[input_df.DXCHANGE == 1]
sMCI_df = input_df[input_df.DXCHANGE == 2]
AD_df = input_df[input_df.DXCHANGE == 3]
pMCI_df = input_df[input_df.DXCHANGE == 5]

if PMCI_selection == False:
    data = pd.concat([NL_df, sMCI_df, AD_df])
else:
    data = pd.concat([NL_df, sMCI_df, AD_df, pMCI_df])

#select samples from first visit(bl) to 48th month(m48)
index = data.VISCODE.str.contains('^bl$|^m06$|^m12$|^m18$|^m24$|^m30$|^m36$|^m42$|^m48$')
data = data[index]
print('only select stable samples from bl to m48', data.shape)

#extract baseline of samples(bl) and the last version of samples
data_baseline = data[data.VISCODE == 'bl']
#data_last = data.drop_duplicates(subset=['RID'], keep='last')
print('data_baseline', data_baseline.shape)
assert data_baseline.shape == data_baseline.drop_duplicates(subset=['RID']).shape
print('data shape', data.shape)

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

def modal_extraction(feat_dict):
    """
    TODO: 
    1.Delete invaild information like date and visit code.
    2.Separate different modals and corresponding features according to the feature dictionary.
    
    Parameters:
    feat_dict: feature dictionary.
    
    Return:
    modal_feat: modal-specific features dict.
    """
    feat_dict = feat_dict[~feat_dict['FLDNAME'].str.contains('RID|SID|UID|PTID|VISCODE|DATE|FLDSTRENG|FLDSTRENG|VERSION|STATUS|NAME|VISIT')]
    feat_modal = feat_dict['TBLNAME']
    modal = feat_modal.drop_duplicates(keep = 'last')
    modal = modal[modal != " "]
    modal = list(modal)
    detached_feat = []
    for i in modal:
        temp = list(feat_dict[feat_modal == i]['FLDNAME'])
        detached_feat.append(temp)
    modal_feat = dict(zip(modal, detached_feat))
    return modal_feat

def discrete_feature_processing(data, feat_dict, remove_longitudinal_mri = True, PMCI_selection = False):
    # Discrete feature processing
    # UCSFFSX = MRI
    m_d = modal_extraction(feat_dict)
    delete = set(['OVERALLQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'TEMPQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'FRONTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'PARQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'INSULAQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'OCCQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'BGQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'CWMQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'VENTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'])
    m_d['UCSFFSX'] = list(set(m_d['UCSFFSX']) - delete)
    '''
    data = pd.get_dummies(data, columns = ['OVERALLQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'TEMPQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'FRONTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'PARQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'INSULAQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'OCCQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'BGQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'CWMQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'VENTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'])
    '''
    #temp = list(data.columns[data.columns.str.contains('Pass|Fail|Partial|Hippocampus Only')])
    #m_d['UCSFFSX'].extend(temp)
    #m_d['One_hot'] = temp

    # UCBERKELEYAV45 = AV45PET
    pass

    #DTIROI = DTI
    pass

    #UPENNBIOMK9 = CSF
    m_d['UPENNBIOMK9'] = ['ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']
    data.loc[data['ABETA_UPENNBIOMK9_04_19_17'] == '<200', 'ABETA_UPENNBIOMK9_04_19_17'] = 200
    data.loc[data['TAU_UPENNBIOMK9_04_19_17'] == '<80', 'TAU_UPENNBIOMK9_04_19_17'] = 80
    data.loc[data['TAU_UPENNBIOMK9_04_19_17'] == '>1300', 'TAU_UPENNBIOMK9_04_19_17'] = 1300
    data.loc[data['PTAU_UPENNBIOMK9_04_19_17'] == '<8', 'PTAU_UPENNBIOMK9_04_19_17'] = 8
    data.loc[data['PTAU_UPENNBIOMK9_04_19_17'] == '>120', 'PTAU_UPENNBIOMK9_04_19_17'] = 120

    #Meta_feature
    #AGE
    min_age, max_age = data.AGE.min(), data.AGE.max()
    step, bins, block = 2, [min_age], min_age
    while block < max_age:
        block += 2
        bins.append(block)
    data.loc[:,'AGE'] = pd.cut(data.AGE, bins, right = False)
    data = pd.get_dummies(data, columns=['AGE'])
    #GENDER
    data.loc[data.PTGENDER == 'Male', 'PTGENDER'] = 1
    data.loc[data.PTGENDER == 'Female', 'PTGENDER'] = 0
    #MARRY
    data = pd.get_dummies(data, columns = ['PTMARRY'])

    #PTEDUCAT
    min_educat, max_educat = data.PTEDUCAT.min(), data.PTEDUCAT.max()
    step, bins, block = 2, [min_educat], min_educat
    while block < max_educat:
        block += 2
        bins.append(block)
    data.loc[:,'PTEDUCAT'] = pd.cut(data.PTEDUCAT, bins, right = False)
    data = pd.get_dummies(data, columns=['PTEDUCAT'])
    #APOE
    data = pd.get_dummies(data, columns=['APOE4'])
    
    RISK_FACTOR = list(data.columns[data.columns.str.contains('AGE_|PTMARRY|PTEDUCAT|APOE4|PTGENDER')])
    # for normal
    if PMCI_selection == False:
        COGNITIVE_TEST = list(data.columns[data.columns.str.contains('CDRSB|ADAS|MMSE|RAVLT|FAQ|MOCA|Ecog')])
        ROI_AVERAGE = list(data.columns[data.columns.str.contains('^FDG$|^AV45$|^Ventricles$|^Hippocampus$|^WholeBrain$|^Entorhinal$|^Fusiform$|^MidTemp$|^ICV$')])
        
    # for pMCI
    else:
        COGNITIVE_TEST = list(data.columns[data.columns.str.contains('CDRSB|ADAS|MMSE|RAVLT|FAQ')])
        ROI_AVERAGE = list(data.columns[data.columns.str.contains('^Ventricles$|^Hippocampus$|^WholeBrain$|^Entorhinal$|^Fusiform$|^MidTemp$|^ICV$')])
    print(ROI_AVERAGE)
    m_d['RISK_FACTOR'] = RISK_FACTOR
    m_d['COGNITIVE_TEST'] = COGNITIVE_TEST
    m_d['ROI_AVERAGE'] = ROI_AVERAGE
    m_d.pop('ADNIMERGE')
    if remove_longitudinal_mri:
        m_d.pop('UCSFFSL')
    
    data.rename(columns={"DXCHANGE": "LABEL"}, inplace = True)
    return data, m_d

def sample_selection(data, based_feat):
    """
    TODO: 
    1.Select samples whose given features are non-null.
    
    Parameters:
    data       : Data to be processed.
    based_feat : given features list.
    
    Return:
    data_non_null: Return the selected samples.
    """
    data_x = data[based_feat]
    temp = ~data_x.isnull()
    index = temp.all(axis='columns')
    data_non_null = data[index]
    
    return data_non_null

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def feature_selection(data_x, feat_dict, modal, num_features_selected):
    """
    TODO: 
    1.Extract features with given modal from data.
    2.Remove the invaild information from the given modal features and remove subjects with Nan given modal feature.
    
    Parameters:
    data_x        : data with all features.
    feat_dict     : feature dictionary
    modal         : the modal.
    
    Return:
    processed modal features.
    """
    set_modal_feat = set(feat_dict[modal])
    #print('The number of specific modal features:', len(set_modal_feat))
    set_data_feat = set(data_x.columns)
    #print('The number of data features:', len(set_data_feat))
    feat_list = list(set_data_feat.intersection(set_modal_feat))
    feat_list.append('LABEL')
    print('The number of features of intersection between modal and data:', len(feat_list))
    
    #Obtain feature and label from the given data
    data_modal = data_x[feat_list]
    data = sample_selection(data_modal, feat_list)
    feat = data[data.columns.drop('LABEL')]
    label = data['LABEL']
    #mask = input_df == "-4"
    
    if len(feat) != 0:
    #feature selection by RFE
        print('The number of samples applicable to the feature selection condition:', len(feat))
        scaler = preprocessing.StandardScaler()
        input_feat = scaler.fit_transform(feat)
        temp = input_feat.copy()
        temp = feat.copy()
        estimator = RidgeClassifier()
        selector = RFE(estimator, n_features_to_select=num_features_selected, step=200, verbose=1)
        selector = selector.fit(temp, label)
        selected_feat = list(np.array(feat.columns)[selector.support_])
        return selected_feat
    else:
        print('There is no sample has all the features of Modal '+ modal)
        return []

data_processed, m_d = discrete_feature_processing(data, input_dict_df, PMCI_selection = PMCI_selection)
keys = list(m_d.keys())
selected_feat = {}
for i in keys:
    print('feature selection for modal: '+ i)
    temp = feature_selection(data_processed, m_d, i, 150)
    selected_feat[i] = temp
    print('--------------------------------------------------------------')

inital_data = data.copy()
feat_sam_sel = selected_feat['COGNITIVE_TEST'] + selected_feat['ROI_AVERAGE'] + selected_feat['RISK_FACTOR'] + selected_feat['UPENNBIOMK9']
data = sample_selection(data_processed, feat_sam_sel)
sample_num, feat_num = data.shape
nan_counts = data.apply(lambda x: x.isnull().value_counts())
nan_counts[nan_counts.isnull()] = 0
temp = nan_counts.iloc[[1]] < int(sample_num * 0.15)
avail_feat_set = set(nan_counts.columns[temp.values[0]])
for i in selected_feat:
    if selected_feat[i] != None:
        selected_feat[i] = list(set(selected_feat[i]).intersection(avail_feat_set))

selected_feat.pop('BAIPETNMRC')
selected_feat.pop('UCBERKELEYAV1451')
selected_feat.pop('DTIROI')
#for pMCI
if PMCI_selection == True:
    selected_feat.pop('UCBERKELEYAV45')

feat_all = []
for i in list(selected_feat.values()):
    feat_all += i
feat_all += ['LABEL']
feat_all += ['RID']
#feat_all += ['VISCODE']

final_data = data[feat_all]
#final_data.to_csv("processed_data_test.csv", index = False)
#np.save('modal_feat_dict_test.npy',selected_feat)

data = final_data
data_ = data.drop_duplicates(subset=['RID'], keep='last')

feat_all.remove('RID')
feat_all.remove('LABEL')

feat_list = feat_all
scaler = preprocessing.StandardScaler()
standard_data = scaler.fit_transform(data_[feat_list])
data_[feat_list] = standard_data

data = data_.drop('RID',axis=1)

# Mean Imputation
data = data.astype(dtype='float')
nan_index = data.isnull().sum()
index = nan_index.index
for i in index:
    if nan_index[i] != 0:
        data[i].fillna(data[i].mean(), inplace=True)

if PMCI_selection == False:
    np.save('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/modal_feat_dict.npy',selected_feat)
    data.to_csv('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/processed_standard_data.csv', index=False)
    sample_ids = data_[['RID','LABEL']].copy()
    sample_ids.to_csv('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/sample_ids.csv', index=False)
else:
    data.LABEL[data.LABEL == 5.0] = 4
    print('the number of samples with label==4: ', data.LABEL[data.LABEL == 4].shape)
    print('the number of samples with label==3: ', data.LABEL[data.LABEL == 3].shape)
    print('the number of samples with label==2: ', data.LABEL[data.LABEL == 2].shape)
    print('the number of samples with label==1: ', data.LABEL[data.LABEL == 1].shape)
    
    NL_index = list(np.random.choice(list(data[data.LABEL == 1].index), 211, replace = False))
    sMCI_index = list(np.random.choice(list(data.LABEL[data.LABEL == 2].index), 275, replace = False))
    AD_index = list(np.random.choice(list(data.LABEL[data.LABEL == 3].index), 72, replace = False))
    pMCI_index = list(data[data.LABEL == 4].index)
    index = NL_index + sMCI_index + AD_index + pMCI_index
    data = data.loc[index]
    
    np.save('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/modal_feat_dict.npy',selected_feat)
    data.to_csv('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/processed_standard_data.csv', index=False)
    sample_ids = data_[['RID','LABEL']].copy()
    sample_ids.to_csv('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison/preprocessing/sample_ids.csv', index=False)
    
print('modalities: ', selected_feat.keys())
print('preprocessed data shape: ', data.shape)