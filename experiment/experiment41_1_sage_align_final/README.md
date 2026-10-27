# Experiment 41.1 — SAGE-Align Final

最终单seed0、10组×10折研究。当前状态以 `manifest.json` 为准，最终中文结果见 `results/report.md`。原始源码、数据和历史实验均只读；仅本目录及根handoff幂等追加允许写入。

`design_audit.md`记录论文/逐函数/历史审计、医学来源访问范围与冻结规则。`prompts.json`及`ontology.json`定义实际可计算的解剖适用范围；自然语言未经过CLIP/LLM编码器，验证的是编译后的结构医学先验。

入口命令均从本目录外任意工作目录运行，环境固定 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。

1. `python run.py prepare`：编译先验与训练TD OOF参考。
2. `CUDA_VISIBLE_DEVICES=0 python run.py preflight`：科学/工程预检；失败不进入正式训练。
3. `python run.py freeze`：冻结设计、Prompt、代码和哈希。
4. `CUDA_VISIBLE_DEVICES=0 python run.py e0`：原双Adam、跨折连续seed0基线，逐折精确核对E32。
5. `python run.py dispatch`：GPU0 E2/E4/E6/E8；GPU1 E1/E3/E5/E7/E9；已完成有效折跳过，中断从latest恢复。
6. `CUDA_VISIBLE_DEVICES=0 python run.py report`：100折独立校验、固定E9反事实、中文报告。
7. `python run.py finalize`：幂等追加handoff更正段并验证历史前缀不变。

每折best保存模型/Adam/RNG；latest每25epoch原子覆盖。基线anchor另含下一折RNG。文件载荷SHA、冻结签名和训练轨迹用于恢复与计分核对。

`.runtime/rejected_protocol_audit_v1`为曾被误称有效的旧100行结果：E5 ROI/FC脱节、E2/E6重复、QC与恢复不足，已全部排除。`.runtime/e0_rejected`为更早的10折RNG哈希格式核验失败记录。不得混入修正后的100折。纠错前结果已观察，不宣称盲法预注册。

沿用val=test、预CV RFE/预处理与871人transductive图，仅支持历史协议内比较。最终未通过分类和医学语义必要性，则永久停止当前路线，不提出41.2，不自动启动其他实验或提交/推送Git。
