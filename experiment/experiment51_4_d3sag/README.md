# Experiment 51.4 — D³-SAG

D³-SAG（Disease-Discriminative Disentanglement with Selective Alignment Gate）以Experiment 51 E1的四路聚合GFR+CAM为唯一工程基座。新增分支只读取CORR与去除站点字段的PHENO，并将候选表示分为S（Shared Diagnostic）、C（Complementary Diagnostic）、I（Incidental Consensus）和N（Modality-specific Nuisance）。S的关系几何对齐强度由outer-train内5折交叉拟合的疾病增量、bootstrap单侧90%下界和站点条件CKA共同决定；C保留互补交互；I/N仅进入有限阶非诊断约束。所有新增写回为无bias、有界残差；路由为零时严格回到E1。

论文与边界审计：SPromptGL的官方论文和代码定义了四张患者关系图、GCN上下文和CAM融合（[SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)，[official code](https://github.com/wanxixi11/SPromptGL)）。MGDR强调多模态公共/私有表征共同用于疾病预测（[MGDR](https://papers.miccai.org/miccai-2024/514-Paper2494.html)）。MM-GTUNets展示影像与非影像模态作用不对称，不能把共享平均自动当作疾病冗余（[MM-GTUNets code](https://github.com/NZWANG/MM-GTUNets)）。SMGD提供Shared/Complementary/Incidental/Nuisance组织和关系几何思想，但其图是患者内部ROI图，不是SPromptGL患者图（[SMGD](https://proceedings.mlr.press/v306/shi26ag.html)）。AAAI 2026的结论是对齐收益取决于模态冗余，应在共享与特异信号之间取平衡（[To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/view/39248)）。

历史审计记录：原GFR四路均传播完整320维并存在PHENO覆盖；ALoss在simple-2不参与实际总loss；Experiment47.3、49、50–51.3均未建立稳定的真实跨患者关系增量；51.3的E7低于E1且统一权重更高；本轮不把这些负结果改写为方法成功。未发现可核实的Experiment52，记录为`EXPERIMENT52_NOT_FOUND`。

## Reproduction

`CUDA_VISIBLE_DEVICES=0 python run.py prepare` → semantic tests → `D3SAG_WORKERS=5 python run.py controller` → `python run.py aggregate`。`.runtime`包含训练恢复状态和最佳checkpoint，不属于紧凑交付目录。历史协议保留`val=test`、transductive图和CV前处理，因此结果只用于内部可比性。
