# Experiment 81.6.2 独立审计报告

任务状态：30/30；独立审计：PASS。

六指标均为百分比，显示10折均值±population SD；Hard-AUC等于历史定义的hard-prediction BA。配对区间为描述性均值±1.96×population SD/√n，不作显著性检验。

|模型|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|SA-PRCL-G1-Protected|92.0742±2.6000|91.9230±2.5481|93.2339±2.7999|90.0671±3.3693|93.7789±4.4992|91.3480±2.6928|
|Scrambled-SA-PRCL-G2-Protected|92.0742±2.6000|91.9230±2.5481|93.2288±2.8174|90.0671±3.3693|93.7789±4.4992|91.3480±2.6928|
|SA-PRCL-G3-Joint|91.8443±2.6004|91.6411±2.5939|93.3385±2.8969|89.0732±3.9177|94.2091±4.1186|91.0063±2.8194|
|DARG|92.1891±2.7648|92.0294±2.7255|93.4946±2.9195|90.0671±3.8862|93.9917±4.7320|91.4590±2.9013|
|E81.6.1-G1|91.8456±2.5480|91.5990±2.5788|93.8184±2.8718|88.3415±4.0168|94.8566±3.2307|90.9253±2.8120|

## 去Fold2结果

|模型|指标|9折均值±population SD|
|---|---|---:|
|SA-PRCL-G1-Protected|ACC|92.0875±2.7403|
|SA-PRCL-G1-Protected|Hard-AUC/BA|91.9299±2.6858|
|SA-PRCL-G1-Protected|Prob-AUC|93.3246±2.9374|
|SA-PRCL-G1-Protected|SEN-ASD|90.0474±3.5510|
|SA-PRCL-G1-Protected|SPE-TD|93.8123±4.7414|
|SA-PRCL-G1-Protected|ASD-F1|91.3469±2.8385|
|Scrambled-SA-PRCL-G2-Protected|ACC|92.0875±2.7403|
|Scrambled-SA-PRCL-G2-Protected|Hard-AUC/BA|91.9299±2.6858|
|Scrambled-SA-PRCL-G2-Protected|Prob-AUC|93.3130±2.9578|
|Scrambled-SA-PRCL-G2-Protected|SEN-ASD|90.0474±3.5510|
|Scrambled-SA-PRCL-G2-Protected|SPE-TD|93.8123±4.7414|
|Scrambled-SA-PRCL-G2-Protected|ASD-F1|91.3469±2.8385|
|SA-PRCL-G3-Joint|ACC|91.9598±2.7166|
|SA-PRCL-G3-Joint|Hard-AUC/BA|91.7522±2.7115|
|SA-PRCL-G3-Joint|Prob-AUC|93.4467±3.0344|
|SA-PRCL-G3-Joint|SEN-ASD|89.2141±4.1055|
|SA-PRCL-G3-Joint|SPE-TD|94.2903±4.3338|
|SA-PRCL-G3-Joint|ASD-F1|91.1181±2.9508|
|DARG|ACC|92.2153±2.9131|
|DARG|Hard-AUC/BA|92.0481±2.8723|
|DARG|Prob-AUC|93.6142±3.0541|
|DARG|SEN-ASD|90.0474±4.0959|
|DARG|SPE-TD|94.0487±4.9847|
|DARG|ASD-F1|91.4703±3.0580|
|E81.6.1-G1|ACC|91.9613±2.6608|
|E81.6.1-G1|Hard-AUC/BA|91.7054±2.6974|
|E81.6.1-G1|Prob-AUC|93.9858±2.9805|
|E81.6.1-G1|SEN-ASD|88.4011±4.2299|
|E81.6.1-G1|SPE-TD|95.0098±3.3709|
|E81.6.1-G1|ASD-F1|91.0282±2.9462|

## 配对比较

|比较|样本|指标|平均差值|描述性区间|W/T/L|
|---|---|---|---:|---:|---:|
|G1_vs_DARG|all10|ACC|-0.1149|[-0.3287, +0.0988]|0/9/1|
|G1_vs_DARG|all10|Hard-AUC/BA|-0.1064|[-0.3108, +0.0980]|1/7/2|
|G1_vs_DARG|all10|Prob-AUC|-0.2607|[-0.5141, -0.0072]|2/3/5|
|G1_vs_DARG|all10|SEN-ASD|+0.0000|[-0.6930, +0.6930]|1/8/1|
|G1_vs_DARG|all10|SPE-TD|-0.2128|[-0.9229, +0.4974]|1/7/2|
|G1_vs_DARG|all10|ASD-F1|-0.1110|[-0.3463, +0.1242]|1/7/2|
|G1_vs_DARG|exclude_fold2|ACC|-0.1277|[-0.3637, +0.1083]|0/8/1|
|G1_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.1182|[-0.3440, +0.1076]|1/6/2|
|G1_vs_DARG|exclude_fold2|Prob-AUC|-0.2896|[-0.5648, -0.0145]|2/2/5|
|G1_vs_DARG|exclude_fold2|SEN-ASD|+0.0000|[-0.7700, +0.7700]|1/7/1|
|G1_vs_DARG|exclude_fold2|SPE-TD|-0.2364|[-1.0240, +0.5511]|1/6/2|
|G1_vs_DARG|exclude_fold2|ASD-F1|-0.1234|[-0.3835, +0.1368]|1/6/2|
|G1_vs_E81.6.1-G1|all10|ACC|+0.2286|[-0.3918, +0.8489]|5/2/3|
|G1_vs_E81.6.1-G1|all10|Hard-AUC/BA|+0.3240|[-0.2921, +0.9400]|6/1/3|
|G1_vs_E81.6.1-G1|all10|Prob-AUC|-0.5845|[-0.9719, -0.1970]|2/0/8|
|G1_vs_E81.6.1-G1|all10|SEN-ASD|+1.7256|[+0.1953, +3.2559]|4/6/0|
|G1_vs_E81.6.1-G1|all10|SPE-TD|-1.0777|[-2.5714, +0.4160]|2/3/5|
|G1_vs_E81.6.1-G1|all10|ASD-F1|+0.4227|[-0.2778, +1.1231]|6/1/3|
|G1_vs_E81.6.1-G1|exclude_fold2|ACC|+0.1263|[-0.5298, +0.7823]|4/2/3|
|G1_vs_E81.6.1-G1|exclude_fold2|Hard-AUC/BA|+0.2244|[-0.4285, +0.8774]|5/1/3|
|G1_vs_E81.6.1-G1|exclude_fold2|Prob-AUC|-0.6612|[-1.0615, -0.2609]|1/0/8|
|G1_vs_E81.6.1-G1|exclude_fold2|SEN-ASD|+1.6463|[-0.0461, +3.3388]|3/6/0|
|G1_vs_E81.6.1-G1|exclude_fold2|SPE-TD|-1.1975|[-2.8386, +0.4437]|2/2/5|
|G1_vs_E81.6.1-G1|exclude_fold2|ASD-F1|+0.3187|[-0.4293, +1.0668]|5/1/3|
|G1_vs_G2|all10|ACC|+0.0000|[+0.0000, +0.0000]|0/10/0|
|G1_vs_G2|all10|Hard-AUC/BA|+0.0000|[+0.0000, +0.0000]|0/10/0|
|G1_vs_G2|all10|Prob-AUC|+0.0052|[-0.0291, +0.0395]|4/2/4|
|G1_vs_G2|all10|SEN-ASD|+0.0000|[+0.0000, +0.0000]|0/10/0|
|G1_vs_G2|all10|SPE-TD|+0.0000|[+0.0000, +0.0000]|0/10/0|
|G1_vs_G2|all10|ASD-F1|+0.0000|[+0.0000, +0.0000]|0/10/0|
|G1_vs_G2|exclude_fold2|ACC|+0.0000|[+0.0000, +0.0000]|0/9/0|
|G1_vs_G2|exclude_fold2|Hard-AUC/BA|+0.0000|[+0.0000, +0.0000]|0/9/0|
|G1_vs_G2|exclude_fold2|Prob-AUC|+0.0116|[-0.0241, +0.0473]|4/2/3|
|G1_vs_G2|exclude_fold2|SEN-ASD|+0.0000|[+0.0000, +0.0000]|0/9/0|
|G1_vs_G2|exclude_fold2|SPE-TD|+0.0000|[+0.0000, +0.0000]|0/9/0|
|G1_vs_G2|exclude_fold2|ASD-F1|+0.0000|[+0.0000, +0.0000]|0/9/0|
|G1_vs_G3|all10|ACC|+0.2299|[-0.3032, +0.7630]|4/4/2|
|G1_vs_G3|all10|Hard-AUC/BA|+0.2819|[-0.2803, +0.8441]|4/3/3|
|G1_vs_G3|all10|Prob-AUC|-0.1045|[-0.3969, +0.1878]|6/0/4|
|G1_vs_G3|all10|SEN-ASD|+0.9939|[-0.2429, +2.2307]|4/5/1|
|G1_vs_G3|all10|SPE-TD|-0.4302|[-1.2253, +0.3649]|1/6/3|
|G1_vs_G3|all10|ASD-F1|+0.3417|[-0.2766, +0.9600]|4/3/3|
|G1_vs_G3|exclude_fold2|ACC|+0.1277|[-0.4258, +0.6812]|3/4/2|
|G1_vs_G3|exclude_fold2|Hard-AUC/BA|+0.1777|[-0.4087, +0.7641]|3/3/3|
|G1_vs_G3|exclude_fold2|Prob-AUC|-0.1220|[-0.4449, +0.2008]|5/0/4|
|G1_vs_G3|exclude_fold2|SEN-ASD|+0.8333|[-0.5003, +2.1669]|3/5/1|
|G1_vs_G3|exclude_fold2|SPE-TD|-0.4780|[-1.3559, +0.4000]|1/5/3|
|G1_vs_G3|exclude_fold2|ASD-F1|+0.2288|[-0.4173, +0.8749]|3/3/3|

## 同epoch净纠错

A为历史DARG Best，B为当前SA Best epoch关闭新增残差，C为完整SA-PRCL。纠正/误伤按测试患者汇总；Fold2历史重叠仍在。

|组|折集|阶段|纠正|误伤|净纠错|
|---|---|---|---:|---:|---:|
|SA-PRCL-G1-Protected|all10|A_to_B|2|5|-3|
|SA-PRCL-G1-Protected|all10|B_to_C|2|0|2|
|SA-PRCL-G1-Protected|all10|A_to_C|2|3|-1|
|SA-PRCL-G1-Protected|exclude_fold2|A_to_B|2|5|-3|
|SA-PRCL-G1-Protected|exclude_fold2|B_to_C|2|0|2|
|SA-PRCL-G1-Protected|exclude_fold2|A_to_C|2|3|-1|
|Scrambled-SA-PRCL-G2-Protected|all10|A_to_B|2|5|-3|
|Scrambled-SA-PRCL-G2-Protected|all10|B_to_C|2|0|2|
|Scrambled-SA-PRCL-G2-Protected|all10|A_to_C|2|3|-1|
|Scrambled-SA-PRCL-G2-Protected|exclude_fold2|A_to_B|2|5|-3|
|Scrambled-SA-PRCL-G2-Protected|exclude_fold2|B_to_C|2|0|2|
|Scrambled-SA-PRCL-G2-Protected|exclude_fold2|A_to_C|2|3|-1|
|SA-PRCL-G3-Joint|all10|A_to_B|3|9|-6|
|SA-PRCL-G3-Joint|all10|B_to_C|3|0|3|
|SA-PRCL-G3-Joint|all10|A_to_C|3|6|-3|
|SA-PRCL-G3-Joint|exclude_fold2|A_to_B|3|8|-5|
|SA-PRCL-G3-Joint|exclude_fold2|B_to_C|3|0|3|
|SA-PRCL-G3-Joint|exclude_fold2|A_to_C|3|5|-2|

## 审计与机制判定

- Best六指标独立复算：30/30。
- Latest完整恢复：30/30；E61逐折起点：10/10。
- 同折三组初始化身份完全一致：10/10；新增参数、DARG参数/优化器、warmup RNG、公共RNG、划分及训练关系对均逐项核对。
- G1/G2主干共有epoch参数、优化器与公共RNG不一致数：0。
- Prompt-off/Null-only、SVD Gram、RES、残差约束及Fold2保护：PASS。
- 科研判定：`NO_GO_SA_PRCL_BASELINE;SEMANTIC_GROUPING_INCREMENT_NOT_ESTABLISHED`。

G2使用原始医学文本，仅固定分组索引被打乱；Prompt-off会关闭整个新增分支，不能单独证明医学语义有效。Fold2历史训练/测试重叠、test=val、预CV预处理/RFE及transductive图均保留。G1/G3的目标函数和梯度路由不同。当前结果仍是单seed、历史协议下的候选证据，不代表独立临床泛化。

固定语义缓存SHA256：`7bfd1d6e11bb9983e9896e682b5a8c40bf606c919dfd38bae368436955f83aff`；正式源码SHA256：`fcf34aac1d93cb75ce922c273b957e70b3cfc36f96542f159b3268f4c2906213`。
