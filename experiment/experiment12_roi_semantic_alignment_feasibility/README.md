# Experiment 12 — Anatomy-Grounded ROI Semantic Alignment Feasibility

本实验只运行 F0/OFF、F1/REAL、F2/SHUFFLED、F3/ID_ONLY 四组，seed0、原外层十折协议第一折；CONSTANT仅离线诊断。采用原P0八句、原GFR与PLN/SPGS，新增连接级AGESA残差，不改潜在token的解释，不开展超参数搜索或自动30折扩展。

原始320维列顺序和256条连接索引严格保留；端点名称以实验十一111标签表关联生成，避免继承的UNKNOWN字段陷阱。252条真实解剖连接英文模板没有疾病方向或个体标签。特征195/225/232/236涉及额外ROI82/atlas3455，仍保留数值，语义delta恒为0。

AGESA使用原始CSV Fisher-z值Z与冻结RN50连接向量，共享Linear(1024,32,bias=False)及Linear(3,32,bias=False)。phi(z)=[z,abs(z),sign(z)]，归一化后余弦s控制r=z*tanh(s)，delta=.1*tanh(alpha)*r；alpha零初始化，新增32,865参数。delta只注入GFR输出的最后256维，原前64维保持逐值一致。根源码完整保留，仅在本目录临时model.py应用integration_patch.diff；原网络类从根目录只读导入。

原两个Adam的lr=.0038/wd=.11不变，新组同lr/wd0；epoch上限450、early wait>200，完整fullbatch871。内部验证集为外层train的15%分层划分，random_state0；内部训练标签仅用于优化/GFR标签先验。验证ACC严格增大选checkpoint，连续ASD概率AUC和BA同步保存；外层测试只在选定后评估。历史使用测试折选择checkpoint，故不能直接与其91.8%比较。

冻结CLIP文本单次编码缓存；新投影在原my_weight_init之后、fork_rng内建立。Preflight检验原始P0与增强模型初始eval/train模式parity、主干和新参数一致性、两次实际优化梯度、单连接文本/数值扰动的局部性、异常mask、checkpoint恢复及历史文件哈希不变。

```bash
cd /data/adodas/SPromptGL-main
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiment/experiment12_roi_semantic_alignment_feasibility/run.py prepare
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiment/experiment12_roi_semantic_alignment_feasibility/run.py preflight
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiment/experiment12_roi_semantic_alignment_feasibility/run.py all
# 中断后：
python experiment/experiment12_roi_semantic_alignment_feasibility/run.py resume
# 只刷新完成任务与已训练模型诊断，不重训：
CUDA_VISIBLE_DEVICES=0 python experiment/experiment12_roi_semantic_alignment_feasibility/run.py refresh
```

GPU0运行F0/F2，GPU1运行F1/F3，每卡两个独立进程。保存完整epoch、模型/Adam/RNG及最佳验证状态供resume，完成任务按签名和checkpoint哈希跳过。失败没有默认零分，失败尝试日志保留、汇总只接受验证成功任务。完成后只保留各配置最佳checkpoint，清理临时接入副本和last状态；可重建临时源码再refresh。

主要成果：results/code_audit.md、edge_semantic_manifest.csv、text_embedding_diagnostics.json、preflight.json、pilot_metrics.csv、alignment_diagnostics.csv、counterfactual.csv、report.md。manifest记录版本、受保护文件和历史目录哈希、文本/缓存/补丁哈希、划分和任务状态。反事实只改变新分支输入或OFF，不重新训练，预测只在内存中汇总，不保存公开个体ID表。

工程PASS不等于医学语义SUPPORTED。一个外层fold只能给出探索性趋势；固定数据的原预处理/RFE也限制独立泛化解释。本任务不执行Git提交或推送；以后发布应排除checkpoints、二进制缓存和任何个体数据。
