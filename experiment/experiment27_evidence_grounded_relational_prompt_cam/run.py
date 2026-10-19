"""Experiment 27: audited, resumable scientific stop at semantic Gate 0.

Run: python run.py all | prepare | preflight | resume | refresh | report
Current fixed ABIDE inputs fail Gate 0; training code is deliberately not created.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '2')
import argparse
import fcntl
import hashlib
import json
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from e27_contract import (NAME, DIRECTORY, CONFIGS, GATE0, GATE1, STAGE2,
                          P0_TEXT, SOURCES, decide_gate0)
from e27_feasibility import (sha, digest, collect, protected_snapshot)

ROOT = Path(os.environ.get('SPROMPTGL_ROOT', '/data/adodas/SPromptGL-main')).resolve()
EXP = Path(__file__).resolve().parent
MANIFEST = EXP / 'manifest.json'
FEASIBILITY = EXP / 'results/feasibility.json'
REPORT = EXP / 'results/report.md'
EXPECTED_PROMPT_SHA = 'ba674d08c28362d0c3e4d6d1927ee66520c6e0b3fa024dd31d2899682265a963'
TERMINAL = 'STOP_UNGROUNDED_INPUTS'
CODE_FILES = ('e27_contract.py', 'e27_feasibility.py', 'run.py')
MARKER = '<!-- EGRP27-GROUNDING-STOP:' + EXPECTED_PROMPT_SHA + ' -->'


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + '\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    return {n: sha(EXP / n) for n in CODE_FILES}


def papers():
    try:
        import pymupdf as pdf
    except ImportError:
        import fitz as pdf
    local = next(ROOT.glob('*MICCAI SPromptGL*.pdf'))
    with pdf.open(local) as document:
        local_text = '\n'.join(page.get_text() for page in document)
        assert all(t in local_text for t in ('Graph-based Feature Representation',
                                             'Prompt-guided Learning Network', '871', '92.30'))
        local_pages = len(document)
    # In-memory read; no downloaded PDF/image duplicates are left in the experiment.
    body = urllib.request.urlopen(SOURCES['MMGPL']['url'], timeout=45).read()
    with pdf.open(stream=body, filetype='pdf') as document:
        text = '\n'.join(page.get_text() for page in document)
        assert '1029' in text and '0.7239' in text and 'Graph prompt learning' in text
        mmgpl_pages = len(document)
    return {'SPromptGL': {'file': str(local), 'sha256': sha(local), 'pages': local_pages,
                          'review': 'Fig1; §2.1/2.2 Eq1–7; §3.1/3.2; Table2; Fig2/3',
                          'paper_abide_acc_percent': 92.30,
                          'paper_shell_hyperparameters_differ': True},
            'MMGPL': {'url': SOURCES['MMGPL']['url'], 'sha256': hashlib.sha256(body).hexdigest(),
                      'pages': mmgpl_pages, 'version': 'arXiv2312.14574v2 author full text',
                      'review': 'Fig1; §III-B tokenizer; §III-C Eq7–10; §III-D Eq11–14; TableI; Fig2',
                      'abide_subjects': 1029, 'cv': '5fold x 3seed',
                      'abide_acc_percent': 72.39,
                      'brain_patch_inputs_not_current_qc_groups': True},
            'other_primary_sources': SOURCES,
            'review_boundary': 'CSD/OTCHA/AAAI为官方摘要/README范围；不冒称未读取的全文。'}


def append_handoff(m, audit):
    target = ROOT / 'handoff.md'
    pending = EXP / 'handoff_append_pending.md'
    section = f'''\n\n## Experiment 27 — EGRP-CAM | UTC {utc()}
{MARKER}

- STATUS: STOP_UNGROUNDED_INPUTS；STAGE: GROUNDING_ONLY；SCIENCE: NOT_EVALUABLE。Gate 0完整审计结束；这是预注册科学可识别性判停，不是训练失败，也不是模型性能阴性试验。
- REPO: 本地HEAD `{audit['repo']['local_head']}`；远端main `{audit['repo']['remote_main_head']}`；六核心文件SHA逐字相同。config.py在当前根目录不存在，参数来自main.py与历史shell。源/数据/论文/历史文件的完整SHA及代码锚点见实验manifest与feasibility.json。
- DATA SHA256: `{audit['input_contract']['data_sha256']['data/ABIDE/processed_standard_data.csv']}`；input 871×320，403 ASD/468 TD，0=ASD；PHENO48=年龄26独热+站点20独热+性别2，ANAT6/FUNC10全为QA，CORR256是RFE后的LedoitWolf相关Fisher-z。质量指标不能称脑体积、皮层厚度、白质完整性或激活图，CAM后的4×4×9是潜在Token，不是ROI。
- 论文/源码差异：MMGPL已有疾病概念Token加权、概念响应Token图及GCN Prompt；SPromptGL是患者GFR→CAM→SPGS/RES。当前Attention还做全局min缩放和行L2归一化；若未来有真实关系，偏置须在归一化后/softmax前，只接首层CAM。本轮未创建补丁。原i=1覆盖及ALoss不加入simple-2反传如实记录，未修复。
- Gate0: 10条候选逐项登记，符合两种不同疾病/临床测量端点且方向可独立检验的关系为0，要求至少2。体积/厚度/白质/侧化/脑室/激活端点缺失；症状量表不在固定输入；年龄/性别—FC属于发育，头动/QC—FC属于采集关系，均不计入疾病grounding。REJECT只意味着新增修正为0，不删除原CAM。
- 身份链复核纠正：利用实验11旧映射重建871×64的PHENO/QC指纹，871人唯一匹配，最大误差{audit['identity_and_clinical']['fingerprint_max_absolute_error']:.15g}；没有使用标签选择匹配。实验21仅按subject_IDs.txt搜索得NOT_JOINABLE，未消费实验11已有指纹映射，故其0/871结论不能代表当前完整证据。历史文件不改。
- 旁路量表只作可用性审计：ADOS_TOTAL非负观测ASD282/403、TD19/468；ADI_R社会分276/403与0/468；SRS_RAW_TOTAL为163/403与137/468。非负观测不等于临床有效/版本可比；原始量表未加入320维，也未用其缺失模式监督。引入它们属于另一个输入实验，需新基线和独立诊断/量表泄漏审计。
- Gate1: NOT_RUN_G0_FAILED，未计算关联、置乱p值或OOF增益；不可把未运行写成效应0。Stage2实际0/180fold，GPU0=0、GPU1=0，未启动GPU、CLIP、训练worker或模型副本；R0–R5仅为冻结的条件性计划。
- 历史P0实验13 F0与E9实验9均核实完整30折，ACC/BA/ProbAUC分别91.69105/91.53935/93.33520%和91.88262/91.69151/93.82272%，仅作来源核验，无新模型分数。没有使用实验24两折pilot作为完整基线。
- 历史教训：17/18年龄性别选择对齐无稳定收益；19自由Key与中性控制否定简单文本归因；20–22医学概念图及E9组合未胜严格控制；23增益路由失败；24按pilot门槛止损；25质量相似无可靠增益；26方向敏感不等于真实文案必要。此次不重做这些模块。
- 协议及偏差：未来条件性训练仍应披露test=val选模、外层CV前RFE/全队列QC标准化、全871 transductive及此前26轮探索；本轮未训练，不能把既往指标写成本轮独立外部泛化。医学关系方法未可评估，不能声称MICCAI2027创新成立。
- 实验文件：`{EXP}`；报告`results/report.md`，数据/候选账本`results/feasibility.json`，状态`manifest.json`；保留三份CPU审计源码，共6个文件。根源码、数据、历史1–26文件保护SHA保持；不自动git add/commit/push。
- 后续最小建议：先在已有身份链上独立核验量表版本/覆盖/可靠度与诊断依赖，或获取同患者定量MRI/DTI/PET等真实互补测量；预先固定至少两项可检验关系，再在训练内选模与独立站点/外队列上另立协议。不能继续把这四组QA/人口学上下文更名为疾病模态追分。
'''
    try:
        # Main coordinator only. Lock the original handoff inode while appending.
        # Existing content is never rewritten or truncated.
        with target.open('a+b', buffering=0) as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            before = f.read()
            marker = MARKER.encode()
            if marker in before:
                assert before.count(marker) == 1
                fcntl.flock(f, fcntl.LOCK_UN)
                if pending.exists():
                    pending.unlink()
                previous = m.get('handoff', {})
                if previous.get('status') == 'UPDATED_FSYNC_VERIFIED':
                    return previous
                return {'status': 'ALREADY_APPENDED_VERIFIED', 'path': str(target),
                        'updated_utc': utc(), 'unique_marker': MARKER,
                        'note': 'Existing unique section verified; no duplicate append.'}
            original_sha = hashlib.sha256(before).hexdigest()
            payload = section.encode()
            f.seek(0, os.SEEK_END)
            view = memoryview(payload)
            while view:
                count = f.write(view)
                assert count and count > 0
                view = view[count:]
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            after = f.read()
            assert after[:len(before)] == before and after[len(before):] == payload
            assert after.count(marker) == 1
            fcntl.flock(f, fcntl.LOCK_UN)
        if pending.exists():
            pending.unlink()
        return {'status': 'UPDATED_FSYNC_VERIFIED', 'path': str(target), 'updated_utc': utc(),
                'before_bytes': len(before), 'before_sha256': original_sha,
                'after_bytes': len(after), 'after_sha256': hashlib.sha256(after).hexdigest(),
                'old_prefix_preserved': True, 'unique_marker': MARKER,
                'section_start_line': before.count(b'\n') + 3}
    except OSError as error:
        atomic_bytes(pending, section.encode())
        return {'status': 'ROOT_HANDOFF_PENDING', 'path': str(target),
                'pending_file': str(pending), 'error': str(error), 'updated_utc': utc()}


def render_report(m, a):
    inputs, identity = a['input_contract'], a['identity_and_clinical']
    lines = ['# 实验二十七：EGRP-CAM 医学关系可识别性审计', '',
             '`STATUS=STOP_UNGROUNDED_INPUTS` · `STAGE=GROUNDING_ONLY` · `SCIENCE=NOT_EVALUABLE`', '',
             '按更新prompt的预注册Gate 0结束：可支撑的独立跨疾病测量关系为0，要求至少2。'
             '没有训练失败；也没有产生可供判断模型性能优劣的新试验。不能把输入前提未成立解释为所有关系路由方法普遍无效。', '',
             '## 仓库、论文与源码核对', '',
             f"本地HEAD `{a['repo']['local_head']}`；远端main `{a['repo']['remote_main_head']}`。"
             '提交不同，但六核心文件逐字节一致；config.py实际不存在，参数在main.py和历史shell中。原工作区既有修改保留。', '',
             'MMGPL在概念响应上加权Token并构建Token图，经GCN形成Transformer图Prompt；已有相关机制，不能声称首次疾病概念建图。'
             '作者版本ABIDE使用1029对MRI/FC、BiomedCLIP及5折×3seed，72.39%不能与当前871人历史分数横向比较。'
             f"[原文Fig1、§III-B/C/D、Eq7–14、TableI/Fig2]({SOURCES['MMGPL']['url']})。", '',
             'SPromptGL将患者多关系GFR表示送入CAM，随后SPGS对局部Token加权、RES融合。论文的示意病变概念不能证明本地QA列测得病变。'
             '论文监督图损失与当前ALoss不反传实现有差异，论文ABIDE超参与历史shell也不同；本轮审计记录差异，未更改协议。'
             f"[原文Fig1、§2、Eq1–7和ABIDE消融]({SOURCES['SPromptGL']['url']})。", '',
             '|本地源码|SHA256|远端相等|', '|---|---|---|']
    for name, v in a['repo']['core_comparisons'].items():
        lines.append(f"|{name}|{v['local_sha256']}|是|")
    lines += ['', '原始Attention先计算QK，再除以全局abs(min)，L2 normalize后才softmax/dropout。'
              '此全局缩放和归一化不是标准缩放点积Attention。若未来通过前提，新增偏置只应加在归一化之后、softmax之前的InputLayer.attention；'
              'Encoder路径不动。本轮未创建e27_relation.py或e27_patch.py，更未运行网络。', '',
              '原GFR i=1确实覆盖i=0的combined_feat；ALoss计算并打印，但simple-2最终loss只有原分类loss与四图正则均值。'
              '两个Adam、val_idx=test_idx、严格ACC提升选首个最佳Checkpoint、wait>early均由真实源码核实。', '',
              '### 逐行源码锚点', '', '|机制|文件:行|实际锚点|', '|---|---|---|']
    for name, v in a['repo']['anchors'].items():
        text = v['text'].replace('|', '\\|')
        lines.append(f"|{name}|{v['file']}:{v['line']}|`{text}`|")
    lines += ['', '## 真实输入和变换', '',
              '当前数据871×320，403 ASD/468 TD。CSV列序与modal_feat_dict逐列一致；年龄/站点/性别独热各行严格一项为1；全表有限、无缺失。', '',
              '|组|维数/内容|真实角色|', '|---|---|---|',
              '|PHENO|48=年龄26个两岁分箱+站点20独热+性别2独热|人口学与采集上下文，非临床症状量表|',
              '|ANAT|6个anat_*字段|结构扫描QA，不是体积/厚度/白质结构|',
              '|FUNC|10个func_*字段|功能扫描QA，不是激活图或独立功能影像流|',
              '|CORR|256个RFE连接列|收缩协方差相关的Fisher-z功能连接测量|', '',
              '16列QC采用历史全队列StandardScaler；PHENO独热未被数值标准化。CORR按HO111的6105条上三角候选连接计算相关、Fisher-z后选择256条，'
              '当前CSV无后续统一z标准化。Notebook中的RFE由旧5折最后一折697人拟合、随后在当前10折外使用；未保存原selector，实验11用数值指纹恢复索引。'
              '本轮没有重算RFE、变换训练输入或重排患者。', '',
              '|QA字段|官方含义/解释边界|', '|---|---|']
    for field, description in inputs['qc_definitions'].items():
        lines.append(f'|{field}|{description}|')
    lines += ['', f"以上字段分类由[ABIDE PCP官方QA定义]({SOURCES['QA']['url']})核对；当前输入单位均为历史z分数，不能由字段名ANAT/FUNC推断测得脑病变。", '',
              '## 患者身份与旁路临床量表复核', '',
              f"复用实验11的subject_row_mapping并独立重建871×64的PHENO/QC：871人唯一双射，最大绝对误差{identity['fingerprint_max_absolute_error']:.15g}，"
              f"RMSE={identity['fingerprint_rmse']:.15g}。匹配不读诊断标签，标签仅在匹配完成后作一致性核查。"
              '当前256条连接映射编号/三角索引均通过，252条有可信名称，4条[195,225,232,236]不适合语义命名。'
              '沿用同CSV签名下实验11既有FC全量数值验证，未为本轮再重建整套FC。', '',
              '**历史审计矛盾的具体来源：** 实验11建立了指纹身份链；实验21只在有限目录查找subject_IDs.txt，未读取实验11映射，故给出NOT_JOINABLE。'
              '本轮已验证现存映射，不能再把“0/871可链接”当作当前事实；同时，链接成功不等于量表已进入当前输入、量表版本可比或诊断泄漏已排除。历史文件保持不变。', '',
              '以下是当前871人旁路表的可用性，只计非负观测，另记缺失/负值代码；不是合法量表范围、施测可靠度或跨版本有效性验收。'
              '覆盖明显依赖诊断与站点，不能用缺失模式替代疾病标签。', '',
              '|旁路字段|ASD非负观测/403|TD非负观测/468|ASD缺失/负代码|TD缺失/负代码|已加入本轮输入|',
              '|---|---:|---:|---:|---:|---|']
    selected = {'ADOS_TOTAL', 'ADI_R_SOCIAL_TOTAL_A', 'ADI_RRB_TOTAL_C', 'SRS_RAW_TOTAL', 'SRS_COGNITION', 'FIQ'}
    for row in identity['sidecar_coverage']:
        if row['field'] in selected:
            x, y = row['coverage']['ASD'], row['coverage']['TD']
            lines.append(f"|{row['field']}|{x['nonnegative_observed']}|{y['nonnegative_observed']}|{x['missing']}/{x['negative_codes']}|{y['missing']}/{y['negative_codes']}|否|")
    lines += ['', f"量表定义及版本/研究可靠度应继续参考[ABIDE官方字典]({SOURCES['LEGEND']['url']})。"
              '原表有部分ADOS/ADI-R/SRS/IQ，不含ADAS；这些字段不在320维。引入它们或MRI形态会改变输入任务，必须另立基线，不能套用旧P0证明本方法增益。', '',
              '## Gate 0逐条候选账本', '',
              '候选在e27_contract.py中冻结，没有做标签驱动的相关性搜索。完整中英文、两个模态、拟用/真实字段、单位、单调/非单调定义、混杂、文献支持范围见feasibility.json的grounding_ledger。'
              '尚无外部固定的有符号关系就记录“未固定”，不从文案生成医学定律。', '',
              '|ID|候选关系|组对|类别|不满足依据|', '|---|---|---|---|---|']
    for row in a['grounding_ledger']:
        lines.append(f"|{row['hypothesis_id']}|{row['text_zh']}|{'↔'.join(row['modality_pair'])}|{row['category']}|{row['rejection_reason']}|")
    lines += ['', '**Gate 0：STOP，0项符合≥2项独立关系要求。** 两个真实疾病/临床测量端点尚不存在；'
              '年龄/性别—FC即使有关联也是发育关系，头动/QC—FC是采集关系，ANAT-QA与FUNC-QA共变也不能改叫病理对应。'
              'CLIP相似度、共同ASD/TD标签或潜在Token相似度都无法补足缺失的测量端点与独立临床命题。', '',
              '### 原始八句SPGS与输入映射', '',
              '|原句（原拼写冻结）|能否由当前测量支撑为跨疾病模态关系|', '|---|---|']
    notes = ('无脑体积端点', '无皮层厚度端点', '无双侧形态端点', '无脑室形态端点',
             '无脑体积端点', '只对应FC测量自身，不能证明异常或第二疾病模态',
             '无DTI/白质完整性端点', '无激活图；FC不是激活模式')
    lines += [f'|{text}|{note}|' for text, note in zip(P0_TEXT, notes)]
    lines += ['', '原Prompt列表从根network.py的AST直接提取并逐字对照，拼写Nromal保留；本轮不加载CLIP、不改Prompt。', '',
              '## Gate 1与Stage 2的真实执行边界', '',
              '`Gate1=NOT_RUN_G0_FAILED`：未计算outer-train关联、同站点置乱零分布、Holm p值或5fold OOF增益。'
              '不填0、不制造训练内量化成绩；没有用QA相关性越过疾病语义前提。', '',
              '`Stage2=NOT_STARTED_G0_FAILED`：实际0/180fold、0/18任务、GPU0=0/GPU1=0，无模型副本/新关系模块/Checkpoint。'
              'GPU双Adam/RNG/偏置parity不适用；没有声称GPU工程预检通过。', '',
              '|条件性配置|冻结问题|本轮状态|', '|---|---|---|']
    for key, value in CONFIGS.items():
        lines.append(f"|{key} {value['name']}|{value['mechanism']}|Gate0未通过，未创建/训练|")
    lines += ['', '### 历史完整30折只读来源核查', '',
              '|历史参照|ACC %|BA %|ASD Prob-AUC %|来源|', '|---|---:|---:|---:|---|']
    for key, value in a['references'].items():
        v = value['mean']
        lines.append(f"|{key}|{100*v['acc']:.5f}|{100*v['ba']:.5f}|{100*v['probability_auc']:.5f}|{value['file']}（{value['config']}）|")
    lines += ['', '两份CSV均包含seed0/1/2×fold1–10唯一30行；先每seed10折均值再三seed均值，与冻结参考一致。'
              '没有重训P0/E9，没有R2新成绩或配对差；实验24两fold的91.980%不作为完整基线。', '',
              '## 历史经验与创新边界', '', '|实验|当前采用的教训|', '|---|---|']
    for key, value in a['history_review'].items():
        lines.append(f"|{key}|{value['conclusion']}|")
    lines += ['', f"[CSD]({SOURCES['CSD']['url']})已有类别语义共享/特异对齐；"
              f"[OTCHA官方仓库]({SOURCES['OTCHA']['url']})已有置信度部分匹配/dustbin；"
              f"[AAAI To Align or Not to Align]({SOURCES['ALIGN']['url']})已有非强制统一对齐的论证。"
              '本轮仅按官方摘要/README记录这些边界，未假称已读全部正文。新增门控、拒绝、OT或对比损失本身不构成创新。', '',
              '原历史协议包含测试参与Checkpoint选择、外层CV前QC标准化/监督RFE、871人transductive传播，且同队列已被多轮探索。'
              '即使未来内部结果阳性，也不能称无泄漏外部泛化或患者因果获益；本轮更没有可评估的训练性能或已成立的MICCAI2027创新。', '',
              '## 工程、签名、重跑与最小下一步', '',
              f"自动审计开始UTC {m['started_utc']}，结束UTC {m.get('finished_utc', '进行中')}；"
              f"自动审计/取证计时{m.get('audit_wall_seconds', 0):.2f}秒。之前的人工式论文/源码审阅未单独计时；没有GPU训练耗时。", '',
              f"保护清单包含{len(m['protected_files_sha256'])}个根源/数据/历史文件；结束SHA对照结果：{m.get('protection', {}).get('status', '待核验')}。"
              '没有git add/commit/push/pull/reset/checkout。审计完成后再次运行all/resume仅核验签名和唯一handoff标记，不重算或创建训练目录。', '',
              f"根handoff状态：{m.get('handoff', {}).get('status', 'PENDING')}；"
              f"更新UTC：{m.get('handoff', {}).get('updated_utc', 'NA')}；旧前缀保留={m.get('handoff', {}).get('old_prefix_preserved', 'NA')}。", '',
              '永久文件共6个：e27_contract.py、e27_feasibility.py、run.py、manifest.json、results/feasibility.json、results/report.md。'
              '不创建空白fold_metrics/diagnostics，不导出患者级量表或预测，不生成图片/README/训练源码或权重。', '',
              '代码与主要输入SHA256（报告自身签名由manifest登记）：', '', '|文件|SHA256|', '|---|---|']
    for name, h in m['code_sha256'].items():
        lines.append(f'|{name}|{h}|')
    for name, h in inputs['data_sha256'].items():
        lines.append(f'|{name}|{h}|')
    for name, ref in a['references'].items():
        lines.append(f"|历史{name} CSV|{ref['sha256']}|")
    for name, ref in m['paper_review'].items():
        if isinstance(ref, dict) and 'sha256' in ref:
            lines.append(f"|{name} PDF|{ref['sha256']}|")
    lines += ['', '**下一步的最小前提**：在已有可核验身份链上先审计真实量表版本、可靠度与不平衡缺失，或补充同患者定量MRI/DTI/PET等真实测量。'
              '先外部预定至少两项不同疾病/临床测量的可检验关系，再按训练内选模、独立站点/外队列重新预注册；'
              '症状量表若参与原ASD诊断，须避免把该诊断过程本身当预测输入。保留本次严格的不可评估结果，停止用QC改名或更换Prompt追分。', '']
    atomic_bytes(REPORT, '\n'.join(lines).encode())


def verify(m):
    assert m['status'] == TERMINAL and m['science'] == 'NOT_EVALUABLE'
    assert code_hashes() == m['code_sha256'], 'Audit implementation changed'
    current = protected_snapshot(ROOT)
    assert current == m['protected_files_sha256'], 'Protected root/data/history changed'
    assert sha(FEASIBILITY) == m['artifacts_sha256']['results/feasibility.json']
    assert sha(REPORT) == m['artifacts_sha256']['results/report.md']
    a = read_json(FEASIBILITY)
    assert decide_gate0(a['grounding_ledger']) == a['gate0']
    assert a['stage2']['completed_folds'] == 0 and a['stage2']['workers_launched'] == 0
    assert not any((EXP / n).exists() for n in ('e27_relation.py', 'e27_patch.py', '.work', '_runtime_sources', '_state', 'results/fold_metrics.csv', 'results/diagnostics.csv'))
    if m['handoff']['status'] in ('UPDATED_FSYNC_VERIFIED', 'ALREADY_APPENDED_VERIFIED'):
        content = (ROOT / 'handoff.md').read_bytes()
        assert content.count(MARKER.encode()) == 1
        if 'before_bytes' in m['handoff']:
            assert hashlib.sha256(content[:m['handoff']['before_bytes']]).hexdigest() == m['handoff']['before_sha256']
    return a


def execute():
    assert EXP.name == DIRECTORY and ROOT == Path('/data/adodas/SPromptGL-main')
    assert sha(ROOT / 'prompt') == EXPECTED_PROMPT_SHA, 'The requested prompt changed; read it before continuing'
    if MANIFEST.exists():
        m = read_json(MANIFEST)
        if m.get('status') == TERMINAL and 'artifacts_sha256' in m:
            a = verify(m)
            if m['handoff']['status'] == 'ROOT_HANDOFF_PENDING':
                m['handoff'] = append_handoff(m, a)
                render_report(m, a)
                m['artifacts_sha256']['results/report.md'] = sha(REPORT)
                if m['handoff']['status'] != 'ROOT_HANDOFF_PENDING':
                    m['permanent_files'] = [n for n in m['permanent_files']
                                            if n != 'handoff_append_pending.md']
                write_json(MANIFEST, m)
            print('VERIFIED_EXISTING_SCIENTIFIC_STOP; no repeated handoff or training')
            return m
        assert code_hashes() == m['code_sha256'], 'Partial audit code signature differs'
        assert protected_snapshot(ROOT) == m['protected_files_sha256']
    else:
        m = {'experiment': NAME, 'status': 'AUDITING', 'started_utc': utc(),
             'execution_root': str(EXP), 'prompt_sha256': EXPECTED_PROMPT_SHA,
             'code_sha256': code_hashes(), 'protected_files_sha256': protected_snapshot(ROOT),
             'gate0_contract': GATE0, 'gate1_contract': GATE1, 'stage2_plan': STAGE2,
             'configs': CONFIGS, 'tasks_launched': [], 'completed_folds': 0,
             'gpu_folds': {'0': 0, '1': 0}, 'gpu_or_clip_loaded': False}
        write_json(MANIFEST, m)
    start = time.monotonic()
    m['paper_review'] = papers()
    write_json(MANIFEST, m)
    a = collect(ROOT)
    assert a['gate0']['qualifying_count'] == 0
    write_json(FEASIBILITY, a)
    current = protected_snapshot(ROOT)
    assert current == m['protected_files_sha256'], 'Read-only audit protection failed'
    m.update(status=TERMINAL, stage='GROUNDING_ONLY', science='NOT_EVALUABLE',
             gate0=a['gate0'], gate1=a['gate1'], stage2=a['stage2'],
             finished_utc=utc(), audit_wall_seconds=time.monotonic() - start,
             protection={'status': 'UNCHANGED_SHA256', 'file_count': len(current),
                         'aggregate_sha256': digest(current)},
             source_local_head=a['repo']['local_head'], source_remote_head=a['repo']['remote_main_head'],
             data_sha256=a['input_contract']['data_sha256'],
             history_reference_sha256={k: v['sha256'] for k, v in a['references'].items()})
    write_json(MANIFEST, m)
    m['handoff'] = append_handoff(m, a)
    render_report(m, a)
    m['artifacts_sha256'] = {'results/feasibility.json': sha(FEASIBILITY), 'results/report.md': sha(REPORT)}
    m['permanent_files'] = [*CODE_FILES, 'manifest.json', 'results/feasibility.json', 'results/report.md']
    if m['handoff']['status'] == 'ROOT_HANDOFF_PENDING':
        m['permanent_files'].append('handoff_append_pending.md')
    write_json(MANIFEST, m)
    verify(m)
    print(json.dumps({'status': m['status'], 'stage': m['stage'], 'science': m['science'],
                      'gate0': m['gate0'], 'completed_folds': 0,
                      'handoff': m['handoff'], 'audit_wall_seconds': m['audit_wall_seconds']},
                     ensure_ascii=False, indent=2))
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'all', 'resume', 'preflight', 'refresh', 'report'))
    args = parser.parse_args()
    if args.command == 'report':
        m = read_json(MANIFEST)
        verify(m)
        print(REPORT.read_text())
    else:
        m = execute()
        if args.command == 'preflight':
            print('AUDIT_ONLY_VERIFIED; GPU preflight NOT_APPLICABLE because Gate0 failed')


if __name__ == '__main__':
    main()
