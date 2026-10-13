# Experiment 11：ABIDE PCP Harvard–Oxford ROI 标签与列序恢复

生成时间（UTC）：2026-10-13T07:51:02.218121+00:00

## 状态

- `MIGRATION_STATUS`: **COMPLETE**。迁移 8/8 个源文件通过 SHA256 校验；源目录在完整校验后按任务说明移除。
- `ROI_LABEL_STATUS`: **COMPLETE**。871/871 个 ROI 时间序列文件的标签头完全相同，111 个标签与原始 ABIDE PCP 掩膜的 111 个非背景值集合完全一致。
- 已恢复 110 个标准 ROI 的解剖名称；额外编码 3455 保留为 `UNKNOWN` / `EXTRA_ROI`。没有修改原始 256 维映射或受试者数据。

## 文件迁移审计

源目录：`/tmp/spromptgl-work11/experiment11_roi_mapping_recovery`；目标：`/data/adodas/SPromptGL-main/experiment/experiment11_roi_mapping_recovery`。目标文件系统探测已完成 fsync、关闭、重新读取并通过。详细的逐文件 SHA256 记录见 [migration.log](../logs/migration.log)。

## 图谱来源与映射方法

ABIDE PCP 官方 `abide_rois.txt` 将 HO 图谱资源指向 `ho_mask_pad.nii.gz`；本地原始掩膜 SHA256 为 `e2ef01f7394a5d0dde7c8cd57f8cc72079b2115b4180d6466aafb0831e08f8c1`，形状为 `(65, 77, 63)`，分辨率为 3 mm。另记录了官方 pipeline 配置及 FSL Harvard–Oxford XML/最大概率标签图的来源、SHA256 与获取时间，见 [atlas_source_manifest.json](../reference/atlas_source_manifest.json)。

列序没有按整数排序推断：逐一读取与 871 行受试者映射对应的全部 `.1D` 文件头。所有文件都给出相同的 111 个 `#<atlas_integer_value>`，其顺序与官方 ABIDE 掩膜的全部非零标签集合一一对应。因而 `rois_ho.1D` 第 k 列到掩膜标签整数的对应关系有直接文件头证据。

皮层标签按 FSL cortical XML 的 48 个索引/名称对应原始 FSL 标签值；ABIDE 掩膜对每个皮层区域按 x=0 分成两侧，输出编码形如 `100 × FSL 标签值 + 半球后缀`，后缀 1/2 由掩膜质心的 MNI x 符号独立核验。皮层下代码不按数值顺序填充：将 FSL XML 坐标变换到原始 FSL4.0.2 图像空间，以各标签体素质心最近邻和第二名距离差匹配，然后验证 ABIDE 掩膜该代码的质心也唯一最近于同一原图标签。14 个匹配均有大于 2 mm 的最近邻间隔。FSL5.0.11 XML 仅用于采用规范的当前名称文本；两版本索引数量相同，差异仅为拼写修正。

ABIDE PCP 说明 HO 概率图阈值、左右半球拆分和子皮层排除规则；相关历史 issue 报告了 110 个标准 ROI 外多出的 `3455`。当前数据头和掩膜确认它在第 83 列，原掩膜中仅有 2 个体素。将原 FSL4 图像按最近邻重采样到掩膜网格后，这两个点的皮层图标签均为背景，子皮层图标签均落在右侧杏仁核位置；由于整数值 3455 不属于标准标签代码，本报告仍将其记为额外未知编码，不将它命名为杏仁核。

## ROI 名称覆盖

| 分类 | 数量 | 处理 |
|---|---:|---|
| 名称、标签、列序均验证 | 110 | 96 个皮层 + 14 个皮层下 |
| 仅可能名称 | 0 | 无间接猜测名称 |
| 其他未知 | 0 | 唯一未命名项单独列为 EXTRA_ROI |
| EXTRA_ROI / UNKNOWN | 1 | atlas 值 3455；第 83 列；2 个体素 |

### 前 20 个 ROI

| 列(0-based) | 列(1-based) | 掩膜整数值 | 名称 | 类别 | 体素数 | 置信级别 |
|---:|---:|---:|---|---|---:|---|
| 0 | 1 | 10 | Left Thalamus | subcortical | 668 | VERIFIED |
| 1 | 2 | 11 | Left Caudate | subcortical | 420 | VERIFIED |
| 2 | 3 | 12 | Left Putamen | subcortical | 449 | VERIFIED |
| 3 | 4 | 13 | Left Pallidum | subcortical | 137 | VERIFIED |
| 4 | 5 | 17 | Left Hippocampus | subcortical | 380 | VERIFIED |
| 5 | 6 | 18 | Left Amygdala | subcortical | 185 | VERIFIED |
| 6 | 7 | 26 | Left Accumbens | subcortical | 59 | VERIFIED |
| 7 | 8 | 49 | Right Thalamus | subcortical | 612 | VERIFIED |
| 8 | 9 | 50 | Right Caudate | subcortical | 404 | VERIFIED |
| 9 | 10 | 51 | Right Putamen | subcortical | 434 | VERIFIED |
| 10 | 11 | 52 | Right Pallidum | subcortical | 135 | VERIFIED |
| 11 | 12 | 53 | Right Hippocampus | subcortical | 349 | VERIFIED |
| 12 | 13 | 54 | Right Amygdala | subcortical | 169 | VERIFIED |
| 13 | 14 | 58 | Right Accumbens | subcortical | 45 | VERIFIED |
| 14 | 15 | 101 | Right Frontal Pole | cortical | 3605 | VERIFIED |
| 15 | 16 | 102 | Left Frontal Pole | cortical | 3035 | VERIFIED |
| 16 | 17 | 201 | Right Insular Cortex | cortical | 508 | VERIFIED |
| 17 | 18 | 202 | Left Insular Cortex | cortical | 482 | VERIFIED |
| 18 | 19 | 301 | Right Superior Frontal Gyrus | cortical | 1149 | VERIFIED |
| 19 | 20 | 302 | Left Superior Frontal Gyrus | cortical | 1130 | VERIFIED |

### 第 83 个 ROI

0-based 索引 `82` / 1-based 列 83 对应掩膜值 3455。体素 IJK 为 `[[25, 44, 16], [25, 44, 17]]`，MNI 世界坐标为 `[[21.0, 0.0, -23.0], [21.0, 0.0, -20.0]]`，共 `2` 个体素。两个点在重采样 FSL 子皮层标签图中均落在 Right Amygdala 区域，但标签整数 3455 本身不是标准 FSL/ABIDE ROI 编码，故不覆盖成正式脑区名、不删除，也不用于 ROI 语义 Prompt。涉及它的 Correlation 特征索引：`[195, 225, 232, 236]`。

## 256 维 Correlation 连接命名

旧映射逐字节保持不变（SHA256 `53c927ebf8a594108942146992b14bc0945f0271b713a669ff00ce37e5f8788d`）；数值核验仍为 max absolute error `3.997e-15`、global RMSE `7.189e-17`。新命名表保留旧表全部字段，仅追加标签与证据字段。

| 覆盖情况 | 连接数 |
|---|---:|
| 两端脑区名和列序均可信 | 252 |
| 一端可信、一端为额外未知 ROI | 4 |
| 两端均未知 | 0 |
| 涉及额外 ROI 3455 | 4 |
| 可安全使用 ROI 名称级语义描述 | 252 |

下表列出前 20 个 Correlation 特征；`UNKNOWN (atlas 3455)` 保持明确未知，不以位置或 ASD 结果代填名称。

| 特征索引 | 连接(0-based ROI 列) | 解剖连接 | 数值最大误差 | 状态 |
|---:|---|---|---:|---|
| 0 | 0–31 | Left Thalamus — Left Superior Temporal Gyrus, anterior division | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 1 | 0–40 | Left Thalamus — Right Inferior Temporal Gyrus, anterior division | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 2 | 0–60 | Left Thalamus — Right Intracalcarine Cortex | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 3 | 0–62 | Left Thalamus — Right Frontal Medial Cortex | 4.441e-16 | VERIFIED_BOTH_ENDPOINTS |
| 4 | 0–91 | Left Thalamus — Right Temporal Occipital Fusiform Cortex | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 5 | 1–18 | Left Caudate — Right Superior Frontal Gyrus | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 6 | 1–36 | Left Caudate — Right Middle Temporal Gyrus, posterior division | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 7 | 1–57 | Left Caudate — Left Lateral Occipital Cortex, superior division | 3.331e-16 | VERIFIED_BOTH_ENDPOINTS |
| 8 | 2–47 | Left Putamen — Left Postcentral Gyrus | 4.441e-16 | VERIFIED_BOTH_ENDPOINTS |
| 9 | 2–77 | Left Putamen — Left Cuneal Cortex | 1.665e-16 | VERIFIED_BOTH_ENDPOINTS |
| 10 | 2–98 | Left Putamen — Left Central Opercular Cortex | 4.441e-16 | VERIFIED_BOTH_ENDPOINTS |
| 11 | 3–5 | Left Pallidum — Left Amygdala | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 12 | 3–50 | Left Pallidum — Right Supramarginal Gyrus, anterior division | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 13 | 3–55 | Left Pallidum — Left Angular Gyrus | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 14 | 3–64 | Left Pallidum — Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex) | 4.441e-16 | VERIFIED_BOTH_ENDPOINTS |
| 15 | 3–89 | Left Pallidum — Right Temporal Fusiform Cortex, posterior division | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 16 | 3–100 | Left Pallidum — Left Parietal Operculum Cortex | 3.331e-16 | VERIFIED_BOTH_ENDPOINTS |
| 17 | 3–108 | Left Pallidum — Left Supracalcarine Cortex | 1.110e-16 | VERIFIED_BOTH_ENDPOINTS |
| 18 | 4–8 | Left Hippocampus — Right Caudate | 2.220e-16 | VERIFIED_BOTH_ENDPOINTS |
| 19 | 4–34 | Left Hippocampus — Right Middle Temporal Gyrus, anterior division | 4.441e-16 | VERIFIED_BOTH_ENDPOINTS |

## 医学语义审计与限制

- 110 个标准 ROI 可安全使用其 Harvard–Oxford 解剖名称；252 条连接的两端均有可信名称，可用于准确描述连接两端。4 条涉及 3455 的连接不得生成正式 ROI 名称 Prompt；可作为单独敏感性对照或保持为未知端点。
- 可引用官方/可信神经科学资料补充通用解剖功能描述，但不能把正常解剖名称改写成 ASD 特异性异常。RFE 入选、相关系数方向或本次名字恢复都不构成 ASD 因果证据。
- 项目中未发现与这些 ROI 时间序列配套的受试者功能 NIfTI；因此没有重提取时间序列的独立数值校验。现有 `feature_validation.csv` 的 256 维数值映射未改动，整体误差证据继续有效。
- 当前可将 252 个已命名 ROI 连接的受试者相关系数写成忠实的定量描述；仅凭静态相关系数尚不能安全推断个体临床功能状态或疾病含义，仍需外部功能/临床验证。

## 结论

`ROI_LABEL_STATUS=COMPLETE`：110 个标准 ROI 名称与列序已通过官方掩膜、871 份 `.1D` 头、FSL 标签/空间中心交叉核验；第 83 列的 3455 已明确作为未知额外编码标出。后续可以开始构造真实 ROI 名称级的通用解剖语义，但疾病特异性和患者功能解释仍需单独验证。

### 主要来源

- [ABIDE PCP ROI / pipeline 说明](https://preprocessed-connectomes-project.org/abide/Pipelines.html)
- [ABIDE PCP 官方图谱配置](https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/abide_rois.txt)
- [ABIDE PCP 第 83 个 ROI issue #9](https://github.com/preprocessed-connectomes-project/abide/issues/9)
- [FSL XML 图谱标签及坐标定义](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/atlas_xml.html)
- [FSL data_atlases FSL 4.0.2](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-4_0_2)
- [FSL data_atlases FSL 5.0.11](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-5_0_11)
