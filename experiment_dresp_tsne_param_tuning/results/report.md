# DReSP ABIDE t-SNE 参数优化审计报告

状态：`COMPLETE_AUDITED_TSNE_PARAMETER_TUNING`

## 原始特征与身份核验
冻结了历史实验中的40个真实NPZ输入。每组为871名患者、144维特征，包含features、labels、patient_ids、test_indices、predictions和cam。患者顺序为0至870，标签为ASD=0和TD=1，特征均有限，未重新提取模型特征。预测与历史审计结果保持一致。

## 8组参数扫描
第一阶段使用Fold 7、8、9的四个模型，共96次t-SNE。每次执行StandardScaler、PCA50和无标签参与的t-SNE，保存KL divergence、trustworthiness k=15/30、二维Silhouette、坐标范围、非有限值数量、运行时间和输入身份SHA。

参数均值如下。
```text
       kl_divergence           trustworthiness_k15           trustworthiness_k30           silhouette_2d           runtime_sec           
                mean       std                mean       std                mean       std          mean       std        mean        std
config                                                                                                                                   
P0          0.894728  0.066215            0.958522  0.008782            0.937086  0.011243      0.454895  0.072895   38.616586   3.728739
P1          0.962351  0.047282            0.959191  0.008445            0.933188  0.010109      0.414883  0.075198   46.936902   3.039932
P2          0.932472  0.053453            0.958480  0.009726            0.934475  0.012098      0.439312  0.104575   50.540072   3.178595
P3          0.815173  0.067220            0.958915  0.008596            0.939835  0.010708      0.449957  0.092965   51.572728   5.130995
P4          0.757299  0.064506            0.956188  0.009370            0.938524  0.010884      0.442835  0.110424   38.379005   8.729655
P5          0.698842  0.067879            0.954669  0.009586            0.939579  0.012041      0.448057  0.085708   39.056422  10.884203
P6          0.814563  0.066414            0.958281  0.009082            0.940361  0.010804      0.457935  0.083276   45.428044   7.980427
P7          0.755973  0.065332            0.956818  0.010089            0.939596  0.013640      0.450929  0.096697   32.038979   7.401417
```

P0重建在当前scikit-learn 1.3.2下与历史坐标不逐点相等。当前版本使用n_iter兼容旧接口。差异已记录为软件环境差异，没有将新坐标伪称历史坐标复现。

## 最终统一参数
选择P3。perplexity=60，early_exaggeration=12，learning_rate=auto，max_iter=3000，random_state=0，init=pca，metric=euclidean。P3的邻域保持指标在候选中较均衡，KL低于P0，且没有像部分高perplexity或高exaggeration设置那样明显提高二维类别Silhouette。所有模型和折统一使用该设置。

## 正式结果
使用P3对四模型十折重新执行40次t-SNE，生成40张独立PNG、10张四模型对比PNG和10张PDF。参数预览图为24张。坐标保存于results/tsne_coordinates.csv。

## 科研解释与局限
碎片化小岛可能来自真实特征中的多簇结构、站点或患者异质性，也可能被perplexity和优化迭代改变。提高perplexity通常扩大邻域范围并使局部小簇合并，early_exaggeration主要改变早期簇间排斥和布局尺度。P3改善了部分折的连续性，但不能把二维紧凑或类别Silhouette变化解释为分类性能提升。不同Fold的二维绝对坐标不可直接比较。P0坐标受软件版本影响，正式结果以本轮冻结环境和明确参数为准。

## 文件
`inputs/manifest.json`冻结输入SHA。`results/parameter_comparison.csv`保存96组扫描。`results/selected_parameters.json`保存统一参数。`results/tsne_coordinates.csv`保存40组正式坐标。图片位于`results/figures/individual`、`comparison`和`parameter_previews`。
