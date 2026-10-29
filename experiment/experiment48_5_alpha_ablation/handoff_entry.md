<!-- EXPERIMENT48_5_ALPHA_ABLATION -->
## Experiment 48.5 — APCR残差系数消融

- **执行**：E1固定α与E2可学习α均完成10折训练，共20/20；单张98GB GPU，5个worker，独立最佳checkpoint读回20/20通过。48.4实际目录为`/model/spromptgl/experiment/experiment48_4_verified_apcr`，与`/data/adodas/SPromptGL-main`下同名路径不同；父实验源码快照、十折锚点和静态结果已只读复制。两组均调用48.4完整E6 `group=6`训练路径。
- **唯一变量**：两组α初始化0.0022、上限0.2、原Sigmoid参数化、同一APCR Adam和其他全部模型/损失/训练配置。E1仅将`alpha_raw.requires_grad=False`；E2为True。
- **E1固定α（mean±population SD，%）**：ACC 91.8430±3.1094，Hard-AUC 91.6977±3.0720，SPE 93.5661±4.2242，SEN 89.8293±3.4050，Probability-AUC 93.4698±3.5083。
- **E2可学习α**：ACC 91.4995±2.8784，Hard-AUC 91.2806±2.8831，SPE 94.2137±3.2156，SEN 88.3476±3.4922，Probability-AUC 92.7895±2.7829。
- **α审计**：E1全部epoch和best α恒为0.0022。E2所有fold均有有限非零α梯度；best α均值0.002392、SD 0.000072，范围0.002327–0.002594，未触及0.2上限。GFR/CAM/APCR初始SHA、初始前向/主loss、group=6单步共享梯度和损失在两组内匹配。
- **主要配对差**：E2−E1 ACC −0.3435pp（2/3/5胜/平/负），Hard-AUC −0.4171pp（2/1/7），SPE +0.6475pp，SEN −1.4817pp。E1相对B0-B ACC +0.1136pp、Hard-AUC +0.0931pp；E2相对B0-B ACC −0.2299pp、Hard-AUC −0.3240pp。E1/E2 ACC分别比48.3 E6低0.1162/0.4598pp，比48.4 E6高0.9182/0.5747pp。
- **APCR关闭（best checkpoint只设α=0，不关闭原GFR）**：E1 ACC均值下降0.4598pp、6个预测翻转；E2下降0.1149pp、1个翻转。逐折值和消息范数在`results/apcr_off_interventions.csv`。
- **结论**：本轮固定低α优于可学习α，但增益小；固定组ACC仅略高于B0-B且未超过48.3 E6。可学习α稳定上升到约0.0024，没有表现出收益。单seed、CV前全局RFE、transductive患者图和历史`val=test` checkpoint选择限制泛化解释，不建议据此宣称普遍提升。
- **产物/交付**：当前实验完整位于`/tmp/experiment48_5_alpha_ablation`，代码签名`9cbed58eab66fe4e79dbe272c1b6bd5d93c06967c83e8992644ce48b8bc1565f`，20/20独立读回、根源码和数据保护SHA未变。请求目标`/data/adodas/SPromptGL-main/experiment/experiment48_5_alpha_ablation`因OSSFS目录创建持续返回`Input/output error`，暂不能声称同步；handoff亦为`HANDOFF_PENDING`。
