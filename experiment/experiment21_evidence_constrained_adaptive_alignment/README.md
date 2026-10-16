# SABER Experiment 21：三线验证与有界自适应疾病证据对齐

一个实验包含A历史机制离线复核、B患者临床身份可用性审计、C固定20组正式训练。A/B失败不改变C的预注册组别。

实验20只读输入位于 `/tmp/experiment20_grounded_concept_graph_alignment`；静态语义缓存 `/tmp/experiment20_grounded_concept_graph_alignment/semantic_cache.npz`，SHA256 `4f0f746ca4b65a638e8dbc0d937ad0f38145c9492d8f80688efa35efa910e96d`。未复制数据/CLIP权重或旧runner，静态模块字节与Exp20原文件相同。

唯一新增自适应参数eta_logit：eta=0.35sigmoid(theta)，初始0.10；att=softmax(prior/0.5 + eta*tanh(abs(d)-1))，mask外0。d只用outer-train全体FC估计无标签均值/标准差。0.02KL仅训练索引。
节点为252条双端HO核验FC连接；固定共享ROI支持集。图权重为患者数值gate乘exp(0.7*(概念attention profile余弦-0.5))，单位self-loop仅一次，按destination行归一化。图不改变物理脑FC拓扑。
CORR残差=0.1tanh(alpha)||base_CORR||normalize(readout)，alpha零初始化。仅插入Encoder+FeedForward后、SPGS前；原P0/P3/E9损失、GFR、分类器、两个Adam保持。新参数WD=0，原WD=.11。
Static local/graph新增19429/20517参数，adaptive各多1个；E9另36865。静态200000+seed，自适应210000+seed置于fork_rng。

|组|基座|模块|prior|图|历史对应|
|---|---|---|---|---|---|
|U00|P0|NONE|None|False|T00|
|U01|P3+D|NONE|None|False|T02|
|U02|P0|STATIC|REAL|False|T05|
|U03|P0|STATIC|REAL|True|T11|
|U04|P0|STATIC|NUMERIC|True|T10|
|U05|P0|STATIC|NUMERIC|False|T04|
|U06|P0|ADAPTIVE|REAL|False|None|
|U07|P0|ADAPTIVE|NUMERIC|False|None|
|U08|P0|ADAPTIVE|NEUTRAL|False|None|
|U09|P0|ADAPTIVE|WRONG_ANATOMY|False|None|
|U10|P0|ADAPTIVE|ISOMETRIC|False|None|
|U11|P0|ADAPTIVE|REAL|True|None|
|U12|P0|ADAPTIVE|NUMERIC|True|None|
|U13|P0|ADAPTIVE|NEUTRAL|True|None|
|U14|P3+D|ADAPTIVE|REAL|False|None|
|U15|P3+D|ADAPTIVE|NUMERIC|False|None|
|U16|P3+D|ADAPTIVE|REAL|True|None|
|U17|P3+D|ADAPTIVE|NUMERIC|True|None|
|U18|P3+D|ADAPTIVE|NEUTRAL|False|None|
|U19|P3+D|ADAPTIVE|NEUTRAL|True|None|

按正式配置表处理U02=T05/U03=T11/U04=T10/U05=T04，表外文字的U2--U5 shorthand不用于重排。正式U00-U05共180fold（其中120静态fold）逐值验证历史六指标、epoch、主干初始化与入口RNG；全部通过后才解锁U06-U19。
历史协议：871人，10fold固定random_state10，seed0/1/2连续十折不重播种，450epoch/early200、Adam .0038、dropout.35。val=test且test ACC选Checkpoint、CV前全队列RFE/标准化、全871人transductive传播，会产生乐观偏差，不称无泄漏独立泛化。
每卡30任务300fold、最多4worker，每worker2CPU线程。动态资源压力仅降低并发。完整模型/两Adam/原E9/缓冲区/所有RNG每25epoch及fold边界原子恢复；同签名结果去重；完成后删除临时权重。

反事实四核心组×30fold×10条件=1200行，local图条件120行NA。固定best checkpoint，只改被指定分支因素，恢复原参数、prior、mask、FC输入及REAL预测逐值一致。所有响应为非加性效应，不能相加为贡献率。

B包只输出汇总。没有原subject_IDs和871x64逐行复现，不称患者身份匹配，不引入任何量表、缺失指示或临床损失。
MMGPL已有概念加权与概念语义图；MGDR已有多图分解；BrainPrompt已有ROI/subject/disease多层提示。候选创新需正确医学语义的独立必要性与患者有界适应实证支持。

根HEAD `1167dd392837a02eb4f83481ae49f02607a9dae5`；GitHub审计 `f11bf7966fcf5756c96f0b074e162827056d8f3d`；核心源码逐字节相同。代码/文本/数据/模型权重/先验hash见manifest。根核心与历史实验只读，根handoff仅追加。禁止自动commit/push。

```bash
cd /data/adodas/SPromptGL-main
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py audit
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py prepare
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py preflight
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py all
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py resume
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py refresh
python /data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment/run.py report
```

重建顺序：复制根核心→fuzz0应用实验9原alignment_patch.diff→fuzz0应用本实验integration_patch.diff。补丁相对实验9基线，严格保留原Prompt/疾病类别原型数学与加载RNG。

精确恢复实现：仅 U06–U19 的进程在导入 PyTorch 前设 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并启用 `torch.use_deterministic_algorithms(True)`，固定稀疏 CUDA `index_add` 的累加顺序。U00–U05 保留历史计算设置，用180正式fold的完整指标/epoch/RNG/初始化hash检验可比性。Adam恢复时深复制保存状态，避免CPU step计数器与恢复点共享存储。

解释边界：严格复用EXP20静态模块和用户提供的自适应模块，因此U06−U02不只是增加eta；attention从`prior+0.5log(gate)`变为`prior/0.5+eta*tanh(abs(d)-1)`，并加入train-only KL。图版还从固定signed-prior profile与原dense归一化，改为attention profile、指数因子及只计一次的unit self-loop。因此adaptive−static检验整体机制改动，不能单独估计eta的训练因果贡献；ADAPT_OFF只测固定已训练模型的eta推理敏感性。没有增加新组或改动任何冻结公式。

从头复现与原位续跑：完成后的原目录执行 `resume` 会核验并跳过600折；manifest保存原服务器绝对路径和HEAD。跨机器/新checkout应保留该审计产物，将五个源码文件复制到新目录，不携带旧manifest/results/logs，再以新的ROOT与EXP20路径执行audit→prepare→preflight→all。新manifest将重新记录实际HEAD、绝对路径及SHA；六核心源码仍必须与f11bf79审计版本一致。依赖ROOT中的实验9/11/19/20冻结文件、原数据、项目CLIP代码和相同SHA的RN50.pt；安装环境见最终manifest。临时source由ROOT→实验9 patch→本实验patch重建，审计者可据源码、双patch及SHA完整复核。

```bash
# 仅示范新目录重跑；会另跑600折。请先保留完整原始结果。
python - <<'COPY_SOURCE'
from pathlib import Path
import shutil
src=Path('/data/adodas/SPromptGL-main/experiment/experiment21_evidence_constrained_adaptive_alignment')
dst=Path('/tmp/spromptgl21-fresh-replay')
dst.mkdir(exist_ok=False)
for name in ['run.py','adaptive_evidence.py','static_evidence.py','offline_diagnostics.py','clinical_audit.py']:
    shutil.copyfile(src/name,dst/name)
COPY_SOURCE
export SP21_ROOT=/data/adodas/SPromptGL-main
export SP21_EXP20=/tmp/experiment20_grounded_concept_graph_alignment
python /tmp/spromptgl21-fresh-replay/run.py audit
python /tmp/spromptgl21-fresh-replay/run.py prepare
python /tmp/spromptgl21-fresh-replay/run.py preflight
python /tmp/spromptgl21-fresh-replay/run.py all
```

最终状态COMPLETE：20组/60任务/600折、600诊断、1200反事实（120 NA），GPU各300折，180历史parity通过。正式调度112.335分钟。ENGINEERING_PASS；医学语义、自适应、语义图及E9协同门槛均FAIL。U01/E9和U14的ACC并列91.88262%，E9的BA/Prob-AUC最高；U14相对E9的Prob-AUC下降0.40043pp。完整三seed、六指标、全部配对W/T/L及科学止损见 [中文报告](results/report.md)。
