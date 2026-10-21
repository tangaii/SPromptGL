# 实验31：SCORE

SCORE 使用每个外层训练折拟合的 CORR median/MAD 参考和站点稳定性，为252条已验证ROI连接生成五状态个体报告，并以文本参与的连接证据调制原CAM-CORR。

本实验遵循原始SPromptGL的871人transductive图传播、val=test checkpoint选择、450/200训练协议；18组各seed=0完成10折。A0–A11与B0–B5的配置、负对照、审计哈希和完整结果见 `manifest.json`、`results/report.md`。P0与历史E1只读参考。

RTGMFF仅借鉴患者特异性ROI状态和年龄/性别FiLM思想；其AAL-116、ALFF/fALFF/ReHo、原始影像Wavelet-Mamba和BioBERT路径未移植，因为本项目输入是预处理后的CORR256与四模态数值。
