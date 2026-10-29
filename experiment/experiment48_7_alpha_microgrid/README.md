# Experiment 48.7 — APCR小残差系数系统消融

冻结48.5的B0-B引擎与48.6.1的E51 LegacyFixedGFR引擎，扫描六档α并比较Fixed/Learnable。B沿用48.5逐epoch随机计划，G沿用48.6.1连续主RNG与APCR独立RNG；所有组调用training_group=6，SRD/SCA/CRA和损失权重不变。

/data路径OSSFS返回Input/output error，本交付暂存于/tmp/SPromptGL-E48.7/experiment48_7_alpha_microgrid；.runtime包含审计来源、任务及独立读回结果。
