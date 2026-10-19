"""Experiment 27 frozen Stage-0 contract. No label-derived grounding rules."""
from __future__ import annotations

NAME = 'EGRP-CAM'
DIRECTORY = 'experiment27_evidence_grounded_relational_prompt_cam'
SEEDS = (0, 1, 2)
FOLDS = tuple(range(1, 11))
DIMENSIONS = {'PHENO': 48, 'ANAT': 6, 'FUNC': 10, 'Correlation': 256}
ANAT = ('anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr')
FUNC = ('func_efc', 'func_fber', 'func_fwhm', 'func_dvars', 'func_outlier',
        'func_quality', 'func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_gsr')
ROLES = {'PHENO': 'DEMOGRAPHIC_CONTEXT', 'ANAT': 'ACQUISITION_QC',
         'FUNC': 'ACQUISITION_QC', 'Correlation': 'DISEASE_RELEVANT_FC_MEASUREMENT'}
P0_TEXT = ('Nromal brain volume', 'Uniform cortical thickness',
           'Symmetrical brain structure', 'Normal ventricular morphology',
           'Altered brain volume', 'Abnormal functional connectivity',
           'White matter integrity issues', 'Different activation patterns')
GATE0 = {'minimum_independent_grounded_relations': 2,
         'requires_two_distinct_disease_or_clinical_measurement_groups': True,
         'qc_and_demographics_do_not_count': True,
         'sidecar_new_inputs_allowed': False,
         'requires_independently_prespecified_testable_relation': True}
GATE1 = {'minimum_relations': 2, 'minimum_consistent_outer_train_folds': 7,
         'outer_folds': 10, 'permutations': (99, 199), 'holm_alpha': .05,
         'oof_folds': 5, 'minimum_average_oof_improvement_pp': .30,
         'site_age_sex_adjustment_required': True}
STAGE2 = {'gamma': .20, 'configs': 6, 'seeds': SEEDS, 'folds': FOLDS,
          'new_fold_target': 180, 'gpu_fold_targets': {'0': 90, '1': 90},
          'max_workers_per_gpu': 2, 'cpu_threads_per_worker': 2,
          'requires_gate0_and_gate1': True}
CONFIGS = {
    'R0': {'name': 'NUMERIC_CONTEXT', 'mechanism': '同预算数值关联CAM偏置'},
    'R1': {'name': 'MEDICAL_UNVERIFIED', 'mechanism': '有依据的候选关系不做患者拒绝'},
    'R2': {'name': 'EGRP_CORE', 'mechanism': '医学关系+训练内实测核验及三路动作'},
    'R3': {'name': 'NEUTRAL_TEXT', 'mechanism': '同患者验证及预算的中性文字'},
    'R4': {'name': 'WRONG_RELATION', 'mechanism': '同预算预定错关系映射'},
    'R5': {'name': 'ISOMETRIC_TEXT', 'mechanism': '同Gram固定旋转关系文字'},
}
HISTORY = {
    11: 'FC/身份数值恢复可验证；潜在Token没有因此获得ROI身份。',
    17: 'PHENO Shared弱，类原型趋同，真实文本未稳定胜阴性控制。',
    18: '判别保护/Private可读出信息不等于最终收益；两教师同时正确不是增益真值。',
    19: '中性C5优于真实C4，自由Key可以吸收固定查询方向。',
    20: '固定医学概念局部证据、连接线图及E9组合均未验证语义必要性。',
    21: '语义、自适应、图及E9协同未稳定成立；临床审计未复用实验11身份链。',
    22: '符号与概念幅度反事实响应极弱，正交残差没有提供E9互补。',
    23: '增益排序弱；完整训练全部新组低于P0，同预算常量优于增益路由。',
    24: '48折pilot按预注册规则停止，91.980%只来自前两折。',
    25: '质量相似不是医学可靠度，真实患者QC对应未稳定胜置乱。',
    26: '有符号读出进入决策，但真实文本不稳定胜无关/同Gram随机；FC-only有竞争力。',
}
SOURCES = {
    'SPromptGL': {'url': 'https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf',
                  'scope': '本地论文Fig1、§2.1/2.2、Eq1–7、Table2与Fig2/3；概念示意不是字段证据。'},
    'MMGPL': {'url': 'https://arxiv.org/pdf/2312.14574v2',
              'scope': '作者全文Fig1、§III-B/C/D Eq5–14、ABIDE1029、TableI、Fig2；非本871人协议。'},
    'QA': {'url': 'https://preprocessed-connectomes-project.org/abide/quality_assessment.html',
           'scope': 'ANAT6/FUNC10官方质量字段定义，不是形态/临床症状。'},
    'ABIDE_DOWNLOAD': {'url': 'https://preprocessed-connectomes-project.org/abide/download.html',
                       'scope': '表型/FILE_ID/SUB_ID及预处理输出来源。'},
    'LEGEND': {'url': 'https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf',
               'scope': '临床量表、版本及研究可靠度字段；非负值计数不代表跨站点临床可比。'},
    'CSD': {'url': 'https://papers.miccai.org/miccai-2026/1073-Paper1452.html',
            'scope': '官方摘要/作者答辩：已有类别语义共享/特异对齐；未声称已阅读全文。'},
    'OTCHA': {'url': 'https://github.com/labhai/OTCHA',
              'scope': '官方README/摘要：OT部分匹配、dustbin、置信度消息传递已有。'},
    'ALIGN': {'url': 'https://ojs.aaai.org/index.php/AAAI/article/view/39248',
              'scope': '官方摘要：对齐收益依赖模态共享/特有信息；不支持强制统一对齐。'},
    'DEVELOPMENT': {'url': 'https://doi.org/10.1126/science.1194144',
                    'scope': '发育/FC背景，不能作为ASD病理关系证明。'},
    'MOTION': {'url': 'https://doi.org/10.1016/j.neuroimage.2011.10.018',
               'scope': '头动与功能连接伪相关背景，采集混杂不是疾病模态。'},
}


def _candidate(hid, title, english, groups, intended, units, actual, kind, rule, reason, refs):
    """Candidate PROPOSALS are not asserted to be validated medical laws."""
    return dict(hypothesis_id=hid, text_zh=title, text_en=english,
                modality_pair=list(groups), intended_endpoints=list(intended),
                intended_units=list(units), actual_endpoint_fields=list(actual),
                category=kind, prespecified_relationship=rule,
                observable_without_diagnosis_label=True,
                sources=refs, source_support='候选/字段审计依据；不构成已验证的有符号ASD关系',
                confounders=['SITE', 'AGE', 'SEX', '扫描QC'], rejection_reason=reason)


# Frozen from the user's legacy-prompt audit, before any association or CV fitting.
# Missing endpoints/directions remain missing: they must never be invented from QC.
RELATIONS = (
    _candidate('H01', '脑体积与同患者FC的条件关系',
               'Regional brain volume and functional connectivity require independently measured, circuit-specific endpoints.',
               ('ANAT', 'Correlation'), ('regional_volume', 'FC_edge'), ('mm^3', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '非单调/区域特异；未有预注册有符号关系',
               'ANAT6无脑体积，CNR/SNR等不能替代体积。', ['SPromptGL', 'QA']),
    _candidate('H02', '皮层厚度与同患者FC的条件关系',
               'Cortical thickness and functional connectivity require matched anatomical measurements.',
               ('ANAT', 'Correlation'), ('cortical_thickness', 'FC_edge'), ('mm', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '非单调/区域和年龄依赖；未固定方向',
               'ANAT6无皮层厚度；anat_fwhm是平滑度，不是厚度。', ['SPromptGL', 'QA']),
    _candidate('H03', '白质微结构与功能连接的条件关系',
               'White-matter microstructure and functional connectivity require separate diffusion and functional measurements.',
               ('ANAT', 'Correlation'), ('diffusion_FA_or_MD', 'FC_edge'), ('FA无量纲或MD mm^2/s', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '非单调/束与连接特异；未固定方向',
               '无DTI/FA/MD；信噪比不是白质完整性。', ['SPromptGL', 'QA']),
    _candidate('H04', '患者症状量表与FC的条件关系',
               'Patient symptom scores and functional connectivity require verified, version-comparable clinical measurements.',
               ('PHENO', 'Correlation'), ('ADOS_ADI_R_or_SRS_score', 'FC_edge'), ('量表分，版本特异', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '量表/连接特异的非单调关系，未外部固定方向',
               'PHENO48没有症状量表；原始旁路量表不能自动加入固定320维。', ['LEGEND', 'QA']),
    _candidate('H05', '脑结构侧化与FC的条件关系',
               'Structural asymmetry and functional connectivity need independently observed bilateral morphometry.',
               ('ANAT', 'Correlation'), ('bilateral_morphometry_asymmetry', 'FC_edge'), ('mm^3/mm或侧化比', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '非单调/左右脑区特异；未固定方向',
               '没有双侧形态端点；HO端点标签不提供形态数值。', ['SPromptGL', 'QA']),
    _candidate('H06', '脑室形态与FC的条件关系',
               'Ventricular morphology and functional connectivity need separate quantitative morphology endpoints.',
               ('ANAT', 'Correlation'), ('ventricle_volume_or_shape', 'FC_edge'), ('mm^3或形状描述量', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '非单调/年龄与结构特异；未固定方向',
               'ANAT6没有脑室体积或形状。', ['SPromptGL', 'QA']),
    _candidate('H07', '激活模式与功能连接的条件关系',
               'Activation patterns and functional connectivity require independent activation measurements.',
               ('FUNC', 'Correlation'), ('task_activation_or_spatial_activation_map', 'FC_edge'), ('任务效应/激活统计量', 'Fisher-z'),
               ([], ['0..255']), 'DISEASE_CANDIDATE', '任务/区域条件的非单调关系；未固定方向',
               'FUNC10为QA统计，无任务激活；FC不等于激活图。', ['SPromptGL', 'QA']),
    _candidate('H08', '年龄/性别与FC发育变化',
               'Age and sex may condition developmental functional-connectivity patterns, without establishing ASD pathology.',
               ('PHENO', 'Correlation'), ('age_sex', 'FC_edge'), ('两岁年龄分箱/性别类别', 'Fisher-z'),
               (['AGE_AT_SCAN_*', 'SEX_*'], ['0..255']), 'DEVELOPMENTAL_NON_DISEASE',
               '发育条件化非单调/多变量关系；不预设统一边方向',
               '人口学上下文不满足两种疾病/临床测量端点。', ['DEVELOPMENT', 'QA']),
    _candidate('H09', '头动/采集质量与FC伪相关',
               'Head motion and quality measures may affect measured FC, which is an acquisition relationship rather than a disease correspondence.',
               ('FUNC', 'Correlation'), ('motion_QA', 'FC_edge'), ('原始FD毫米/帧计数，输入已z标准化', 'Fisher-z'),
               (['func_mean_fd', 'func_dvars', 'func_num_fd', 'func_perc_fd'], ['0..255']),
               'ACQUISITION_NON_DISEASE', '边距离/处理依赖的伪相关，不设统一病理单调方向',
               '即使可测量，也属于采集混杂；不能充当第二疾病模态。', ['MOTION', 'QA']),
    _candidate('H10', '结构与功能扫描的QA共变',
               'Anatomical and functional image-quality metrics can share acquisition confounds without being disease measurements.',
               ('ANAT', 'FUNC'), ('anatomical_QA', 'functional_QA'), ('原始QA各异，当前输入z分数', '原始QA各异，当前输入z分数'),
               (list(ANAT), list(FUNC)), 'ACQUISITION_NON_DISEASE', '扫描仪/运动共同条件关系，非预定临床方向',
               '两端都是QA，无法验证跨疾病测量关系。', ['QA']),
)


def decide_gate0(ledger):
    """No labels, logits, similarity scores, or clinical sidecar values accepted."""
    qualifying = [r for r in ledger if r['current_implementable']
                  and r['two_disease_measurement_endpoints']
                  and r['independently_testable_prespecified_relation']]
    passed = len({r['hypothesis_id'] for r in qualifying}) >= GATE0['minimum_independent_grounded_relations']
    return {'pass': passed, 'qualifying_count': len(qualifying),
            'required_count': 2, 'qualifying_ids': [r['hypothesis_id'] for r in qualifying],
            'status': 'GROUNDING_PASS' if passed else 'STOP_UNGROUNDED_INPUTS',
            'stage': 'GROUNDING_ONLY', 'science': 'NOT_EVALUABLE',
            'label_or_clip_similarity_used': False}
