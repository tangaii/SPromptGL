# 实验十一：Correlation特征到HO ROI连接索引恢复

**实验状态：COMPLETE**。按256维特征和受试者映射的全量数值验证判定。

## 核心结果

- ROI文件、唯一SUB_ID：871/871；每个时间序列有111个ROI，长度78–316个时间点。
- 完整上三角向量：6105维；871行CSV受试者映射：871/871。
- 找回并通过误差≤1e-6及唯一性检查：256/256维；256条连接无重复。
- 最大绝对误差：3.997e-15；整体RMSE：7.189e-17；全量连接中最佳候选与次佳候选的最小最大误差间隔：2.996e-01。
- PHENO/QC独立匹配误差最大值：7.105e-15；标签不匹配：0行。

## 源码和数值处理审计

`data_processing.ipynb`使用`np.loadtxt`读取`.1D`、`ConnectivityMeasure(kind='correlation')`计算矩阵、`np.triu_indices_from(matrix, 1)`去掉对角线并按上三角顺序展开，然后用`np.arctanh`做Fisher-z。111个ROI得到111×110/2=6105个候选连接。Notebook输出保存了871个原始SUB_ID的读取顺序；这些ID和当前文件集合一一对应。

原Notebook内核为Python 3.6.10，未记录Nilearn或scikit-learn版本。Nilearn 0.7.1源码中的该API对时间序列去均值、按总体标准差标准化（极小标准差用1）、默认使用LedoitWolf协方差，再转换为相关矩阵。本次按该实现重建；对全部CSV值的逐元素误差约为浮点舍入量级。因而实际目标是默认收缩协方差转相关后的Fisher-z，而不是未经收缩的普通Pearson系数。
参考：[Nilearn 0.7.1连接矩阵源码](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/connectome/connectivity_matrices.py)，[Nilearn 0.7.1标准化源码](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/signal.py)。

Notebook使用`RidgeClassifier`、RFE目标256维、`step=100`；5折StratifiedKFold循环后以最后一折697名训练者拟合。原Notebook没有保存selector索引。本次直接用完整CSV数值恢复索引，没有重跑RFE。Correlation在Fisher-z后无额外标准化。

## 受试者对齐

从Notebook输出恢复的871人顺序与表型文件中筛出的顺序相同。CSV的48个PHENO one-hot列、16个ANAT/FUNC标准化质量值和label均与原表型按SUB_ID重建后逐行一致；Correlation列又经256维连接指纹交叉验证。没有发现多模态行错位。

## 前10维映射

`original_connection_index`按`np.triu_indices(111, k=1)`编号，从0开始。

|特征|完整索引|ROI i (0-based)|ROI j (0-based)|名称|最大误差|
|---:|---:|---:|---:|---|---:|
|0|30|0|31|UNKNOWN / UNKNOWN|2.220e-16|
|1|39|0|40|UNKNOWN / UNKNOWN|2.220e-16|
|2|59|0|60|UNKNOWN / UNKNOWN|2.220e-16|
|3|61|0|62|UNKNOWN / UNKNOWN|4.441e-16|
|4|90|0|91|UNKNOWN / UNKNOWN|2.220e-16|
|5|126|1|18|UNKNOWN / UNKNOWN|2.220e-16|
|6|144|1|36|UNKNOWN / UNKNOWN|2.220e-16|
|7|165|1|57|UNKNOWN / UNKNOWN|3.331e-16|
|8|263|2|47|UNKNOWN / UNKNOWN|4.441e-16|
|9|293|2|77|UNKNOWN / UNKNOWN|1.665e-16|

本地未找到这批CPAC输出配套的HO标签文件和列序说明，因此名称统一标为UNKNOWN，不按另一版atlas猜填。ABIDE PCP公开记录曾指出HO输出第83个变量存在额外ROI问题；列编号本身仍由数值连接严格验证，名称需取得原处理版本的标签/顺序文件确认。

## 三名受试者数值示例

以下对比的是前三个特征的Fisher-z原始重建值与CSV值。

|CSV行|SUB_ID|特征0 重建 / CSV|特征1 重建 / CSV|特征2 重建 / CSV|
|---:|---:|---:|---:|---:|
|0|50003|0.366033121331165 / 0.366033121331165|0.397059342555883 / 0.397059342555883|0.51667523238699 / 0.51667523238699|
|435|50728|0.0651369799971996 / 0.0651369799971995|0.341988912463431 / 0.341988912463431|-0.0113098857456253 / -0.0113098857456252|
|870|51607|0.256702449550838 / 0.256702449550838|0.70108604117067 / 0.70108604117067|0.599090881403635 / 0.599090881403635|

## 验证结论

- 先比较完整871人排序后的连接指纹，搜索不依赖CSV行序；然后以PHENO/QC指纹匹配CSV行到SUB_ID，并逐值验证全部871人。
- 256维均在1e-6容差内通过；每维在6105个候选中只有一个连接通过容差，次佳候选间隔最小值为2.996e-01。随机种子0抽查20人，最大误差3.775e-15；全871人核验已覆盖该抽查。
- 时间序列均为有限数；存在常数ROI列，按原Nilearn标准化算法处理，未删除、裁剪或填补。ROI标签文件缺失不影响编号映射。

结论：**COMPLETE**。可以开始以已恢复ROI编号为依据的个体化功能连接Prompt实验；如果提示词需要脑区名称，需要先取得与这些111列完全配套的HO标签和列顺序清单。

## 输出文件

`recover_roi_mapping.py`、`README.md`、`logs/recovery.log`、`results/roi_256_mapping.csv`、`results/subject_row_mapping.csv`、`results/feature_validation.csv`、`results/data_audit.json`。
