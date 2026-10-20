# 实验28补充E10

只新增E10_COARSE_FULL三个seed各连续十折。E4原模型/辅助结构/anchor语义损失不变，唯一差异为逐值复用E2粗粒度教师。历史E0–E9不重训、不修改。

执行：`python run.py prepare`、`python run.py preflight`、`python run.py all`；中断执行`python run.py resume`。最终报告`results/report.md`。

训练：GPU0 seed0/2，GPU1 seed1；原450epoch、wait>200、test=val严格ACC选模、两Adam及跨折连续RNG不变。新增28517参数，wd=0。原始源码仅复制到本目录.runtime/sources，完成后清理。

缓存为原实验28 prompt_banks.npz字节副本，绝不重新编码文本。e28_prompts.py和e28_integration.py为历史逐字节副本，方法差异在manifest记录。

历史协议存在测试参与选模、外层前RFE/标准化和全队列图传播的乐观偏差。2×2交互仅作内部描述，不证明医学因果协同。

正式训练已完成30/30折。根handoff追加的fsync仍受Errno5阻塞，补充章节已保存为handoff_addendum.md；确认根I/O修复后可执行`python run.py handoff`，不会重训。训练冻结代码SHA和完成后仅报告/I/O函数完善的diff均在manifest中。
