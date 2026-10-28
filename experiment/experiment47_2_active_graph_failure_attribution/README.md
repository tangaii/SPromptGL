# Experiment 47.2 — Active Patient Graph Failure Attribution

本实验是 Experiment47/47.1 的问题归因延续，不是新 GFR 模块。所有运行文件位于
`/tmp/SPromptGL-E47/experiment47_2_active_graph_failure_attribution`；项目根目录模型、数据和历史实验未修改。

## 设计冻结

四模态输入为 PHENO48、ANAT6、FUNC10、CORR256；沿用原始阈值0.9、两层GCN、PHENO
在 `GFR.forward` 的 `i==1` 覆盖行为、seed0、10折和历史 `val=test` 选模协议。
Stage A 使用 B0-B、T1（有效三路 GCN W 去 weight decay）和 Experiment43.4 CAM-only
最佳 checkpoint 做只读诊断。Stage B 只使用 `rho=0.25` 的训练折中心化尺度控制，不搜索
比例。D0是动态图；D1是真实 B0-B 固定图；D2在保持对称性、每节点度数、站点对边数、
自环、非零权重多重集的前提下做站点类别内双边交换；D3只保留隐式 self-loop。

## 历史事实—证据—本轮问题

|历史事实|原始证据|本轮仍需回答|
|---|---|---|
|B0-B消息极弱|Experiment43.2/47.1最佳 checkpoint与逐层审计|消息被激活后，真实关系是否有增量？|
|去有效GCN weight decay使消息变大但ACC降至约71%|47.1 T1十折与训练轨迹|失败是尺度优化还是关系质量？|
|CAM-only约91.73% ACC|Experiment43.4物理删除GFR checkpoint|跨患者关系能否超过自身表征？|
|固定Top-16均值不是原GFR图|Experiment46|本轮直接使用原始阈值图，而非Top-16替代|
|反事实教师在44/44.1未形成可靠监督|44/44.1报告|本轮不复用该教师，不把它当作关系价值证据|

## 代码路径

`integration.py` 只调用 `.runtime/sources` 的不可变原始模块；`graph_controls.py` 构造固定图
负控制；`scale_audit.py` 完成十折 Stage A；`train.py` 是 D0–D3 的共同训练入口；
`audit.py` 计算训练侧 Gate 和固定 checkpoint 干预。所有四组使用相同 anchor、优化预算、
有效 GCN 无 weight decay、rho及随机流；D1–D3只改变固定邻接关系。

## 解释边界

尺度控制改善训练 NLL 只能支持优化归因，不能自动证明患者图有疾病信息。D1 相对 D2/D3
的配对训练才是关系价值检验；固定 checkpoint 的图交换和消息置零属于模型依赖诊断，
不是反事实临床因果。结果仍受全队列预处理/RFE、transductive 871人图和 `val=test` 选模
影响，不能作为无偏临床泛化证明。
