# Experiment 11 — HO ROI连接索引恢复

本实验使用CPU，不重训SPromptGL，也不修改原始数据、模型代码或训练配置。脚本从`data_processing.ipynb`保存的执行输出恢复871名受试者顺序，重建全部111×111功能连接矩阵，并将CSV的256个Correlation列与6105个完整上三角连接逐列核对。

## 运行

在项目根目录运行：

```bash
python experiment/experiment11_roi_mapping_recovery/recover_roi_mapping.py
```

如果需要指定输入或输出目录，可使用`--root`和`--output-dir`。脚本在中断后复用签名匹配的连接缓存；完成后会清理临时缓存。结果文件已存在且内容相同则保留；若内容不同，脚本停止而不覆盖。

依赖NumPy、pandas、SciPy、scikit-learn和threadpoolctl。当前服务器没有安装Nilearn；脚本依据Nilearn 0.7.1公开源码复现`ConnectivityMeasure(kind='correlation')`的时间序列标准化、默认Ledoit–Wolf协方差与cov_to_corr流程，并以完整871×256数值进行验证。原Notebook记录Python 3.6.10内核，但未记录Nilearn/scikit-learn版本。

## 结果

最终状态和限制见`results/report.md`。映射表中的ROI编号按`np.triu_indices(111, k=1)`生成。项目数据中没有发现与这批CPAC输出配套的ROI名称及列序清单，脑区名称因此标为`UNKNOWN`。

源码依据：[Nilearn 0.7.1 connectivity_matrices.py](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/connectome/connectivity_matrices.py)、[signal.py](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/signal.py)。HO第83个输出变量的历史说明见[ABIDE PCP issue #9](https://github.com/preprocessed-connectomes-project/abide/issues/9)。
