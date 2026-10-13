# Experiment 11 — HO ROI 连接索引与解剖标签恢复

## 数值连接映射

本实验使用 CPU，不重训 SPromptGL，也不修改模型代码或原始数据。数值恢复脚本从已保存的 Notebook 执行输出恢复 871 名受试者顺序，重建 111×111 功能连接矩阵，并将 CSV 的 256 个 Correlation 列与 6105 个上三角连接逐列核对。

初始数值恢复阶段的结果与边界：

- 256/256 个特征获得唯一连接映射；最大绝对误差约 `3.997e-15`，整体 RMSE 约 `7.189e-17`。
- 初版审计尚未找到 ROI 名称/列序清单，因而当时把名称记为 `UNKNOWN`。本次后续审计通过全部 `.1D` 头、ABIDE 原始掩膜和 FSL 标签资料补齐了标准 ROI 名称；详细证据见下文及 `results/roi_label_report.md`。
- 原数值映射 `results/roi_256_mapping.csv` 保持不变。逐受试者映射表和详细数据审计含个体级信息，仅保留在本地，不随公开实验包发布。

原数值恢复重跑命令：

```bash
python experiment/experiment11_roi_mapping_recovery/recover_roi_mapping.py
```

该脚本在具备本地原始数据时会生成逐受试者映射表和详细数据审计；这两份个体级输出不包含在 GitHub 发布包中。ROI 标签脚本所需的 NIfTI 图谱也因二进制分发限制未上传，来源和 SHA256 记录在 `reference/atlas_source_manifest.json`。

原脚本依赖 NumPy、pandas、SciPy、scikit-learn 和 threadpoolctl。当前服务器没有安装 Nilearn；脚本依据 Nilearn 0.7.1 公开源码复现其 `ConnectivityMeasure(kind='correlation')` 流程，并以 871×256 全部数值验证。原 Notebook 记录 Python 3.6.10 内核，但没有记录 Nilearn/scikit-learn 版本。

## 文件安全迁移

8 个原始实验文件已从 `/tmp/spromptgl-work11/experiment11_roi_mapping_recovery` 迁移到本目录。迁移时 8/8 个目标文件的 SHA256 与源文件一致，无冲突；文件系统写入、fsync、关闭和重新读取探测通过。校验完成后才移除了本次对应的临时源目录。逐文件记录见 `logs/migration.log`。本次更新后的 README 保存在本路径；原迁移 README 的逐字节副本保存在 `reference/README_migrated_original.md`，其 SHA256 与源文件一致。

## ROI 名称与列序恢复

- 逐一读取与受试者表对应的 871 个 `rois_ho.1D` 文件头；871 个有序标签头完全相同，并与 ABIDE PCP 官方 `ho_mask_pad.nii.gz` 的 111 个非背景值集合一致。
- 96 个皮层 ROI 和 14 个皮层下 ROI 的名称、标签和列序均已核验。第 83 列（0-based 82）是额外标签 3455，仅有 2 个体素；保持 `UNKNOWN` / `EXTRA_ROI`，不删除也不冒充标准脑区。
- 256 条命名连接中，252 条的两端名称都可信；4 条涉及标签 3455，保持未知端点且不用于 ROI 名称级 Prompt。
- 新结果见 `results/roi_111_labels.csv`、`results/roi_256_named_mapping.csv`、`results/roi_label_audit.json` 和 `results/roi_label_report.md`。官方来源及 SHA256 见 `reference/atlas_source_manifest.json`。

标签恢复脚本需要 NumPy 和 Nibabel。可在独立虚拟环境中运行，避免修改全局 Python：

```bash
python -m venv --system-site-packages /tmp/roi-label-venv
/tmp/roi-label-venv/bin/python -m pip install --index-url https://pypi.org/simple --no-deps nibabel
/tmp/roi-label-venv/bin/python experiment/experiment11_roi_mapping_recovery/recover_roi_labels.py
```

`recover_roi_labels.py` 只生成标签清单、命名连接表、审计与报告；不会写入旧版 `roi_256_mapping.csv` 或 `data_audit.json`。

## 研究解释边界

ABIDE PCP 官方 `abide_rois.txt` 指向原始 HO 掩膜。FSL 4.0.2 标签图提供与 ABIDE 掩膜代码对应的历史整数码；FSL 5.0.11 XML 提供规范名称文本。原 pipeline 配置只记录 FSL 5.0 路径，没有具体补丁版本。项目中没有找到配套的受试者功能 NIfTI，因此本次不能重新提取时间序列做独立数值验证；列序的直接证据来自全部 `.1D` 文件头和原始掩膜标签集合的完全匹配。

252 条连接可以用于忠实描述已核验的解剖 ROI 对及其相关系数；名称本身不证明 ASD 特异性关系，相关系数也不足以推断患者临床功能状态或因果意义。涉及 3455 的 4 条连接应保持异常端点标记。

参考：[ABIDE PCP Pipeline / HO 图谱说明](https://preprocessed-connectomes-project.org/abide/Pipelines.html)、[ABIDE PCP 官方图谱配置](https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/abide_rois.txt)、[ABIDE PCP issue #9](https://github.com/preprocessed-connectomes-project/abide/issues/9)、[FSL XML 标签格式](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/atlas_xml.html)、[FSL 4.0.2 atlas](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-4_0_2)、[FSL 5.0.11 atlas](https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/tree/fsl-5_0_11)。
