# Experiment 75.4 — DPUSA

## Execution and audit

- Status: `COMPLETE_AUDITED`; frozen E74.4-G2 checkpoint replay PASS 10/10; no DARG/CAM retraining.
- Directory: `/data/adodas/SPromptGL-main/experiment/experiment75_4_prompt_utility_audit`; preferred-path fallback: `None`.
- Device: PPU-ZW810E frozen feature extraction; CPU lightweight probes (one process); frozen forward 10 folds × one pass; CPU probes 3799.0s.
- RN50 SHA: `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`; prompts encoded once; anonymous Gram max error 1.78e-15.
- Engineering retries/failures before the final run: 9; each is itemized in `manifest.json`, and failed partial results were excluded.
- E74.4 source audit found a documented audit-only script repair after training. All non-audit protected source hashes match the historical manifest; reconstructed preflight source digest matches the training source digest.

## Method audit

- SPromptGL's original PLN/PEN computes two class text representations from the eight source phrases, applies class-softmax-derived maximum weights to the `view(871,4,4,9)` CAM latent slots, then adds the original `AvgPool1d(kernel_size=4,stride=4)` residual before the 144-D OutputLayer. The four-dimensional partition is a learned feature layout, not 16 labeled anatomical ROIs.
- E74.4-G2 uses its audited PLNNoPEN CAM-to-OutputLayer path. This audit captures X320, DARG F320 and H144 at that frozen path without changing model parameters or prediction.
- E75.1 and E75.2 were inspected as read-only antecedents. E75.1's G2 ACC was 91.8443% (0.3448 pp below E74.4-G2), residual/H was about 0.8789%, and confidence neutralization produced almost no hard-prediction change; its anonymous G3 reached 92.0742%, so real-text advantage was not established. E75.2 G2 ACC was 91.4995% (0.6897 pp below E74.4-G2), and module-off caused zero hard flips. Neither is used as the base here.
- BrainPrompt's two ABIDE label prompts are retained in official TC then ASD order and TC is mapped to TD for analysis. The inspected `compute_label_loss()` normalizes fused graph and label representations, uses cosine similarities as logits and applies cross-entropy; this audit does not reproduce the published multi-level prompting network or its language encoder. Candidate vectors use the verified SPromptGL/E60 CLIP RN50 cache.
- P4-P6 are preregistered candidate strings, not clinician-validated prompts. The PHENO block includes demographic/site/sex-coded variables; ANAT/FUNC fields are quality-related features in this prepared dataset. P5 cannot be localized to named networks because CORR is stored as 256 positional indices with no verified ROI/network edge map.
- Frozen prompts and token similarities are deterministic transforms of existing H144. They cannot create patient information; only their restricted-learning utility is assessed.

## Prompt probes

Rates are mean ± population SD across folds, in percent; log-loss and Brier are unitless. B1 is fixed random projection; weak B1 performance is not evidence that CLIP semantics are absent because CAM and CLIP coordinates are not pretrained to align. B2 trains only a shared 9→128 linear token map and linear head on outer-training features, with C selected by inner five-fold CV.

|Prompt / control|Setting|Method|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|Log-loss|Brier|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BASE|frozen|S0-Base|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|0.3107 ± 0.0745|0.0833 ± 0.0224|
|NO_TEXT|incremental|M0-Calibrated-Base|90.9248 ± 3.5846|90.8125 ± 3.5251|93.4946 ± 2.9195|89.3354 ± 3.3306|92.2895 ± 4.8463|90.1591 ± 3.7250|0.3468 ± 0.1183|0.0830 ± 0.0242|
|NO_TEXT|incremental|M1-B1-Local-NoText|89.8916 ± 2.8628|89.8527 ± 2.7600|93.3874 ± 2.6403|89.3354 ± 2.2143|90.3700 ± 4.5290|89.1575 ± 2.9104|0.3649 ± 0.1276|0.0866 ± 0.0228|
|NO_TEXT|incremental|M1-B2-Trainable-NoText|89.7766 ± 3.2783|89.6524 ± 3.1974|92.9596 ± 2.7201|88.0976 ± 3.4629|91.2072 ± 5.1760|88.9169 ± 3.3238|0.3695 ± 0.1121|0.0881 ± 0.0234|
|P1-SPromptGL-8|B1-fixed|B1-Prompt|89.5520 ± 2.7903|89.5164 ± 2.7738|92.9901 ± 2.1953|89.0976 ± 3.3146|89.9352 ± 3.7994|88.7773 ± 2.8017|0.3926 ± 0.0989|0.0920 ± 0.0209|
|P1-SPromptGL-8|B1-fixed|M2-B1-BasePlusPrompt|89.7806 ± 2.3855|89.7494 ± 2.3839|93.2663 ± 2.2543|89.3415 ± 3.1147|90.1573 ± 3.1012|89.0092 ± 2.4818|0.3870 ± 0.1113|0.0907 ± 0.0218|
|P1-SPromptGL-8|B2-trained-alignment|B2-Prompt|88.4013 ± 3.4346|88.2965 ± 3.5085|92.9107 ± 2.9091|86.8659 ± 5.4291|89.7271 ± 4.0810|87.3638 ± 3.7791|0.3861 ± 0.1329|0.0946 ± 0.0270|
|P1-SPromptGL-8|B2-trained-alignment|M2-B2-BasePlusPrompt|89.4357 ± 3.0805|89.3044 ± 3.0683|93.0695 ± 2.7550|87.6098 ± 3.6372|90.9991 ± 3.8850|88.4969 ± 3.2124|0.3804 ± 0.1289|0.0879 ± 0.0256|
|P2-BrainPrompt-2|B1-fixed|B1-Prompt|89.5494 ± 1.7523|89.5420 ± 1.7468|93.0463 ± 2.4517|89.3476 ± 3.0920|89.7364 ± 3.1544|88.7838 ± 1.8353|0.3821 ± 0.1117|0.0911 ± 0.0205|
|P2-BrainPrompt-2|B1-fixed|M2-B1-BasePlusPrompt|89.6630 ± 2.4763|89.6220 ± 2.4289|93.2506 ± 2.6478|89.0915 ± 2.7377|90.1526 ± 3.7779|88.8882 ± 2.5221|0.3806 ± 0.1237|0.0882 ± 0.0219|
|P2-BrainPrompt-2|B2-trained-alignment|B2-Prompt|89.3208 ± 3.5677|89.1506 ± 3.4955|92.8063 ± 2.4096|86.8720 ± 3.7741|91.4292 ± 5.1083|88.3287 ± 3.7031|0.3675 ± 0.1093|0.0901 ± 0.0246|
|P2-BrainPrompt-2|B2-trained-alignment|M2-B2-BasePlusPrompt|90.3527 ± 2.4848|90.2331 ± 2.4781|92.9716 ± 2.9753|88.5976 ± 2.9401|91.8686 ± 3.0195|89.4843 ± 2.6631|0.3724 ± 0.1271|0.0882 ± 0.0249|
|P3-Class-Definition-8|B1-fixed|B1-Prompt|88.8636 ± 2.9983|88.7353 ± 2.9588|92.5883 ± 2.3520|87.1098 ± 3.4320|90.3608 ± 4.1508|87.8939 ± 3.1277|0.4129 ± 0.1372|0.0936 ± 0.0235|
|P3-Class-Definition-8|B1-fixed|M2-B1-BasePlusPrompt|89.7766 ± 3.3971|89.6858 ± 3.3743|92.8737 ± 2.6244|88.5854 ± 3.7294|90.7863 ± 4.3490|88.9409 ± 3.5738|0.3999 ± 0.1436|0.0891 ± 0.0263|
|P3-Class-Definition-8|B2-trained-alignment|B2-Prompt|88.8584 ± 3.2634|88.7182 ± 3.3107|92.8686 ± 2.7827|86.8537 ± 4.5965|90.5828 ± 3.6396|87.8130 ± 3.6228|0.3857 ± 0.1195|0.0922 ± 0.0251|
|P3-Class-Definition-8|B2-trained-alignment|M2-B2-BasePlusPrompt|88.7448 ± 3.2521|88.6475 ± 3.2570|92.8455 ± 2.3776|87.3598 ± 3.8777|89.9352 ± 3.7837|87.7951 ± 3.4355|0.3816 ± 0.1074|0.0923 ± 0.0240|
|P4-Modality-Aware-8|B1-fixed|B1-Prompt|89.5481 ± 3.6225|89.4366 ± 3.4983|92.9318 ± 2.7542|88.0915 ± 2.8927|90.7817 ± 5.6394|88.7195 ± 3.6364|0.3862 ± 0.1384|0.0916 ± 0.0284|
|P4-Modality-Aware-8|B1-fixed|M2-B1-BasePlusPrompt|89.4357 ± 2.8580|89.3511 ± 2.7690|93.0655 ± 2.8939|88.3415 ± 2.7175|90.3608 ± 4.4660|88.6060 ± 2.8705|0.3884 ± 0.1394|0.0898 ± 0.0267|
|P4-Modality-Aware-8|B2-trained-alignment|B2-Prompt|89.5481 ± 3.3180|89.4419 ± 3.2896|92.8290 ± 2.7378|88.0976 ± 3.6390|90.7863 ± 4.3490|88.6694 ± 3.4848|0.3736 ± 0.1159|0.0917 ± 0.0266|
|P4-Modality-Aware-8|B2-trained-alignment|M2-B2-BasePlusPrompt|89.7766 ± 3.1967|89.6221 ± 3.1831|92.7417 ± 2.6505|87.5976 ± 3.6782|91.6466 ± 3.9245|88.8210 ± 3.4211|0.3956 ± 0.1279|0.0902 ± 0.0252|
|P5-Functional-Network-8|B1-fixed|B1-Prompt|88.5123 ± 3.8565|88.4480 ± 3.7687|93.1447 ± 2.7768|87.6037 ± 3.8209|89.2923 ± 5.6970|87.6544 ± 3.8957|0.3546 ± 0.0921|0.0921 ± 0.0242|
|P5-Functional-Network-8|B1-fixed|M2-B1-BasePlusPrompt|89.3169 ± 2.9661|89.2082 ± 2.9104|93.1552 ± 2.7601|87.8476 ± 3.3886|90.5689 ± 4.5874|88.4249 ± 3.0499|0.3676 ± 0.1039|0.0901 ± 0.0235|
|P5-Functional-Network-8|B2-trained-alignment|B2-Prompt|89.2045 ± 3.0146|89.0963 ± 2.9940|92.5801 ± 2.8038|87.6098 ± 3.6372|90.5828 ± 3.9954|88.2706 ± 3.1885|0.4017 ± 0.1211|0.0946 ± 0.0267|
|P5-Functional-Network-8|B2-trained-alignment|M2-B2-BasePlusPrompt|89.0909 ± 2.7389|89.0024 ± 2.7441|92.6969 ± 2.6080|87.8476 ± 3.4064|90.1573 ± 3.2378|88.1757 ± 2.9201|0.3822 ± 0.1145|0.0907 ± 0.0230|
|P6-Heterogeneity-Aware-8|B1-fixed|B1-Prompt|89.8929 ± 3.0839|89.7603 ± 3.0271|93.1439 ± 2.4802|88.0915 ± 3.8075|91.4292 ± 4.8269|89.0045 ± 3.1951|0.3718 ± 0.1029|0.0886 ± 0.0259|
|P6-Heterogeneity-Aware-8|B1-fixed|M2-B1-BasePlusPrompt|90.2377 ± 2.7883|90.1346 ± 2.7851|93.1914 ± 2.5112|88.8354 ± 3.5522|91.4339 ± 3.6332|89.3964 ± 2.9606|0.3614 ± 0.0996|0.0867 ± 0.0246|
|P6-Heterogeneity-Aware-8|B2-trained-alignment|B2-Prompt|88.9760 ± 3.5743|88.8789 ± 3.5943|92.6917 ± 2.5774|87.6098 ± 4.4135|90.1480 ± 4.0362|88.0410 ± 3.7922|0.3804 ± 0.1051|0.0942 ± 0.0245|
|P6-Heterogeneity-Aware-8|B2-trained-alignment|M2-B2-BasePlusPrompt|89.2019 ± 3.2694|89.1011 ± 3.2446|92.6828 ± 2.4850|87.8415 ± 3.4243|90.3608 ± 4.1556|88.3026 ± 3.4391|0.3923 ± 0.1179|0.0909 ± 0.0241|
|C3-Unrelated-English-8|B1-fixed|B1-Prompt|88.2837 ± 5.0910|88.2112 ± 5.0257|91.8265 ± 4.2654|87.3476 ± 5.0329|89.0749 ± 6.6374|87.4171 ± 5.3217|0.4166 ± 0.1783|0.0956 ± 0.0376|
|C3-Unrelated-English-8|B1-fixed|M2-B1-BasePlusPrompt|89.4344 ± 3.6340|89.3495 ± 3.5427|92.4234 ± 2.9620|88.3476 ± 3.4836|90.3515 ± 5.5211|88.6272 ± 3.6673|0.3968 ± 0.1398|0.0896 ± 0.0270|
|C3-Unrelated-English-8|B2-trained-alignment|B2-Prompt|89.3208 ± 3.0486|89.2057 ± 2.9980|92.6085 ± 2.8076|87.6159 ± 3.4262|90.7956 ± 4.2026|88.3970 ± 3.2065|0.3809 ± 0.1195|0.0926 ± 0.0262|
|C3-Unrelated-English-8|B2-trained-alignment|M2-B2-BasePlusPrompt|89.0922 ± 3.3037|88.9799 ± 3.3100|92.9947 ± 2.3466|87.6037 ± 3.9887|90.3562 ± 3.9632|88.1596 ± 3.4500|0.3831 ± 0.1193|0.0895 ± 0.0250|
|P1-SPromptGL-Anonymous-Gram-Matched|B1-fixed|B1-Prompt|87.7077 ± 4.3123|87.6296 ± 4.2756|92.6197 ± 2.5731|86.6098 ± 4.2963|88.6494 ± 5.2167|86.7509 ± 4.5361|0.4194 ± 0.1509|0.0961 ± 0.0305|
|P1-SPromptGL-Anonymous-Gram-Matched|B1-fixed|M2-B1-BasePlusPrompt|89.4318 ± 2.5204|89.3216 ± 2.5021|93.0757 ± 2.4014|87.8476 ± 2.7916|90.7956 ± 3.2339|88.5135 ± 2.6488|0.3840 ± 0.1242|0.0868 ± 0.0220|
|P1-SPromptGL-Anonymous-Gram-Matched|B2-trained-alignment|B2-Prompt|89.4331 ± 3.7430|89.2983 ± 3.7576|93.0388 ± 2.8411|87.5976 ± 4.7201|90.9991 ± 4.4340|88.4798 ± 4.0030|0.3633 ± 0.1108|0.0894 ± 0.0264|
|P1-SPromptGL-Anonymous-Gram-Matched|B2-trained-alignment|M2-B2-BasePlusPrompt|89.5481 ± 3.0698|89.4279 ± 3.0408|92.8280 ± 2.6268|87.8476 ± 3.5851|91.0083 ± 4.1831|88.6336 ± 3.2497|0.3897 ± 0.1161|0.0901 ± 0.0236|
|P5-Functional-Network-Anonymous-Gram-Matched|B1-fixed|B1-Prompt|89.8916 ± 3.4870|89.8325 ± 3.3896|93.1132 ± 2.5996|89.0915 ± 2.7267|90.5735 ± 5.0398|89.1487 ± 3.5488|0.3818 ± 0.1370|0.0889 ± 0.0260|
|P5-Functional-Network-Anonymous-Gram-Matched|B1-fixed|M2-B1-BasePlusPrompt|90.1215 ± 3.1439|90.0608 ± 3.0733|93.1663 ± 2.5741|89.3354 ± 2.7097|90.7863 ± 4.4563|89.3776 ± 3.2034|0.3850 ± 0.1399|0.0879 ± 0.0245|
|P5-Functional-Network-Anonymous-Gram-Matched|B2-trained-alignment|B2-Prompt|89.0870 ± 3.1898|88.9915 ± 3.2375|92.6147 ± 2.7381|87.6037 ± 4.4203|90.3793 ± 3.3523|88.1207 ± 3.5170|0.3956 ± 0.1239|0.0926 ± 0.0255|
|P5-Functional-Network-Anonymous-Gram-Matched|B2-trained-alignment|M2-B2-BasePlusPrompt|89.4344 ± 3.1258|89.3611 ± 3.1047|92.9105 ± 2.4014|88.3476 ± 3.5008|90.3747 ± 3.9818|88.5757 ± 3.3234|0.3872 ± 0.1184|0.0905 ± 0.0237|
|RND-Unit-75451|B1-fixed|B1-Prompt|89.2072 ± 3.4193|89.1475 ± 3.4025|93.0476 ± 2.3011|88.3598 ± 4.0861|89.9352 ± 4.3363|88.3648 ± 3.5995|0.3929 ± 0.1426|0.0917 ± 0.0275|
|RND-Unit-75451|B1-fixed|M2-B1-BasePlusPrompt|89.7780 ± 3.3174|89.7284 ± 3.2813|93.1845 ± 2.4310|89.0915 ± 3.5006|90.3654 ± 4.2385|89.0007 ± 3.4843|0.3799 ± 0.1396|0.0882 ± 0.0254|
|RND-Unit-75451|B2-trained-alignment|B2-Prompt|89.2032 ± 2.7888|89.0692 ± 2.7608|92.6674 ± 2.6400|87.3476 ± 3.4206|90.7909 ± 4.0030|88.2380 ± 2.9565|0.3814 ± 0.1073|0.0903 ± 0.0239|
|RND-Unit-75451|B2-trained-alignment|M2-B2-BasePlusPrompt|89.3195 ± 3.2199|89.1871 ± 3.2086|92.7536 ± 2.5967|87.3659 ± 3.8548|91.0083 ± 3.9558|88.3514 ± 3.4372|0.3835 ± 0.1224|0.0902 ± 0.0257|
|RND-Unit-75452|B1-fixed|B1-Prompt|89.4305 ± 2.5202|89.3953 ± 2.4941|93.3303 ± 2.2288|88.8415 ± 3.1691|89.9491 ± 3.7175|88.6241 ± 2.6904|0.3748 ± 0.1055|0.0886 ± 0.0180|
|RND-Unit-75452|B1-fixed|M2-B1-BasePlusPrompt|89.7753 ± 2.6046|89.7300 ± 2.5871|93.4632 ± 2.6144|89.0854 ± 2.9805|90.3747 ± 3.4976|88.9810 ± 2.7913|0.3705 ± 0.1180|0.0868 ± 0.0192|
|RND-Unit-75452|B2-trained-alignment|B2-Prompt|88.9747 ± 2.8809|88.8486 ± 2.8119|92.8145 ± 2.5032|87.1098 ± 2.8441|90.5874 ± 4.2002|88.0064 ± 2.9877|0.3844 ± 0.1218|0.0939 ± 0.0256|
|RND-Unit-75452|B2-trained-alignment|M2-B2-BasePlusPrompt|88.7474 ± 2.7145|88.6414 ± 2.6607|92.4510 ± 2.4173|87.3476 ± 3.2328|89.9352 ± 4.2353|87.8107 ± 2.7900|0.3971 ± 0.1109|0.0934 ± 0.0237|
|RND-Unit-75453|B1-fixed|B1-Prompt|89.5520 ± 2.6939|89.4621 ± 2.7110|92.8643 ± 2.4345|88.3415 ± 3.3459|90.5828 ± 2.9529|88.6728 ± 2.8955|0.3821 ± 0.1078|0.0904 ± 0.0218|
|RND-Unit-75453|B1-fixed|M2-B1-BasePlusPrompt|89.4357 ± 2.8580|89.3860 ± 2.8221|92.9812 ± 2.7439|88.8415 ± 2.9755|89.9306 ± 3.9227|88.6474 ± 2.9035|0.3912 ± 0.1270|0.0902 ± 0.0235|
|RND-Unit-75453|B2-trained-alignment|B2-Prompt|89.2045 ± 3.6127|89.0465 ± 3.6086|92.3717 ± 2.7001|86.8720 ± 4.5203|91.2211 ± 4.3562|88.1782 ± 3.9116|0.3928 ± 0.1173|0.0926 ± 0.0278|
|RND-Unit-75453|B2-trained-alignment|M2-B2-BasePlusPrompt|89.3195 ± 2.7315|89.2368 ± 2.7067|92.7741 ± 2.7502|88.1037 ± 3.2490|90.3700 ± 3.6426|88.4381 ± 2.8546|0.3987 ± 0.1261|0.0931 ± 0.0241|

## Incremental utility

The primary contrasts are M2−M0 (calibrated base score) and M2−M1 (same-capacity no-text local signal). Positive values mean improvement for rate metrics and reduction for losses. Sign-flip tests use the ten fold-level paired differences; folds share training data, so p-values and intervals are descriptive, not independent-sample inference. Holm correction is applied within the registered comparison families.

|Prompt|Setting|Contrast|Log-loss improvement|Brier improvement|Prob-AUC improvement (pp)|ACC improvement (pp)|W/T/L Prob-AUC|Holm p, Prob-AUC|
|---|---|---|---:|---:|---:|---:|---|---:|
|P1-SPromptGL-8|B1|M2_vs_M0|-0.04026 [-0.06189,-0.01968]|-0.00769|-0.2283|-1.1442|2/0/8|1.0000|
|P1-SPromptGL-8|B1|M2_vs_M1|-0.02213 [-0.04679,0.00874]|-0.00408|-0.1212|-0.1110|2/0/8|1.0000|
|P1-SPromptGL-8|B2|M2_vs_M0|-0.03362 [-0.06457,-0.00347]|-0.00494|-0.4251|-1.4890|3/0/7|1.0000|
|P1-SPromptGL-8|B2|M2_vs_M1|-0.01088 [-0.03781,0.01496]|0.00015|0.1100|-0.3409|6/0/4|1.0000|
|P2-BrainPrompt-2|B1|M2_vs_M0|-0.03380 [-0.05388,-0.01411]|-0.00522|-0.2440|-1.2618|3/0/7|1.0000|
|P2-BrainPrompt-2|B1|M2_vs_M1|-0.01567 [-0.03639,0.00392]|-0.00160|-0.1369|-0.2286|3/0/7|1.0000|
|P2-BrainPrompt-2|B2|M2_vs_M0|-0.02561 [-0.04497,-0.00553]|-0.00525|-0.5230|-0.5721|1/0/9|0.8906|
|P2-BrainPrompt-2|B2|M2_vs_M1|-0.00286 [-0.02807,0.01996]|-0.00017|0.0120|0.5760|6/0/4|1.0000|
|P3-Class-Definition-8|B1|M2_vs_M0|-0.05315 [-0.08033,-0.03035]|-0.00613|-0.6209|-1.1481|2/0/8|1.0000|
|P3-Class-Definition-8|B1|M2_vs_M1|-0.03502 [-0.07018,-0.00465]|-0.00251|-0.5137|-0.1149|3/0/7|1.0000|
|P3-Class-Definition-8|B2|M2_vs_M0|-0.03486 [-0.06279,-0.00745]|-0.00931|-0.6491|-2.1800|2/0/8|1.0000|
|P3-Class-Definition-8|B2|M2_vs_M1|-0.01211 [-0.03078,0.00719]|-0.00422|-0.1141|-1.0319|4/0/6|1.0000|
|P4-Modality-Aware-8|B1|M2_vs_M0|-0.04160 [-0.07482,-0.01560]|-0.00680|-0.4291|-1.4890|2/0/8|1.0000|
|P4-Modality-Aware-8|B1|M2_vs_M1|-0.02346 [-0.06021,0.00785]|-0.00318|-0.3219|-0.4559|4/0/6|1.0000|
|P4-Modality-Aware-8|B2|M2_vs_M0|-0.04879 [-0.08377,-0.01198]|-0.00724|-0.7529|-1.1481|1/0/9|1.0000|
|P4-Modality-Aware-8|B2|M2_vs_M1|-0.02604 [-0.05214,-0.00263]|-0.00215|-0.2179|0.0000|3/0/7|1.0000|
|P5-Functional-Network-8|B1|M2_vs_M0|-0.02084 [-0.04654,0.01086]|-0.00709|-0.3394|-1.6079|3/0/7|1.0000|
|P5-Functional-Network-8|B1|M2_vs_M1|-0.00270 [-0.03415,0.04132]|-0.00348|-0.2322|-0.5747|4/0/6|1.0000|
|P5-Functional-Network-8|B2|M2_vs_M0|-0.03539 [-0.05855,-0.01352]|-0.00770|-0.7977|-1.8339|2/0/8|1.0000|
|P5-Functional-Network-8|B2|M2_vs_M1|-0.01265 [-0.03571,0.01018]|-0.00262|-0.2627|-0.6857|5/1/4|1.0000|
|P6-Heterogeneity-Aware-8|B1|M2_vs_M0|-0.01458 [-0.03893,0.01667]|-0.00368|-0.3032|-0.6870|3/0/7|1.0000|
|P6-Heterogeneity-Aware-8|B1|M2_vs_M1|0.00355 [-0.03157,0.04956]|-0.00006|-0.1961|0.3461|4/0/6|1.0000|
|P6-Heterogeneity-Aware-8|B2|M2_vs_M0|-0.04551 [-0.07622,-0.01593]|-0.00788|-0.8118|-1.7228|1/0/9|1.0000|
|P6-Heterogeneity-Aware-8|B2|M2_vs_M1|-0.02276 [-0.04659,-0.00275]|-0.00279|-0.2767|-0.5747|3/0/7|1.0000|
|C3-Unrelated-English-8|B1|M2_vs_M0|-0.04997 [-0.07895,-0.02713]|-0.00664|-1.0712|-1.4903|1/1/8|0.4609|
|C3-Unrelated-English-8|B1|M2_vs_M1|-0.03183 [-0.06287,-0.00740]|-0.00302|-0.9640|-0.4572|2/0/8|0.7930|
|C3-Unrelated-English-8|B2|M2_vs_M0|-0.03634 [-0.05807,-0.01458]|-0.00651|-0.4999|-1.8325|2/0/8|1.0000|
|C3-Unrelated-English-8|B2|M2_vs_M1|-0.01359 [-0.03958,0.01053]|-0.00142|0.0351|-0.6844|7/0/3|1.0000|
|P1-SPromptGL-Anonymous-Gram-Matched|B1|M2_vs_M0|-0.03720 [-0.05440,-0.01758]|-0.00382|-0.4189|-1.4929|3/0/7|1.0000|
|P1-SPromptGL-Anonymous-Gram-Matched|B1|M2_vs_M1|-0.01906 [-0.03212,-0.00576]|-0.00020|-0.3117|-0.4598|3/0/7|1.0000|
|P1-SPromptGL-Anonymous-Gram-Matched|B2|M2_vs_M0|-0.04287 [-0.06859,-0.01776]|-0.00712|-0.6666|-1.3767|2/0/8|1.0000|
|P1-SPromptGL-Anonymous-Gram-Matched|B2|M2_vs_M1|-0.02012 [-0.04066,-0.00138]|-0.00204|-0.1315|-0.2286|4/0/6|1.0000|
|P5-Functional-Network-Anonymous-Gram-Matched|B1|M2_vs_M0|-0.03823 [-0.05924,-0.01849]|-0.00493|-0.3283|-0.8033|2/1/7|1.0000|
|P5-Functional-Network-Anonymous-Gram-Matched|B1|M2_vs_M1|-0.02009 [-0.03717,-0.00503]|-0.00131|-0.2212|0.2299|3/0/7|1.0000|
|P5-Functional-Network-Anonymous-Gram-Matched|B2|M2_vs_M0|-0.04044 [-0.06682,-0.01300]|-0.00749|-0.5841|-1.4903|2/0/8|1.0000|
|P5-Functional-Network-Anonymous-Gram-Matched|B2|M2_vs_M1|-0.01769 [-0.03752,-0.00042]|-0.00240|-0.0491|-0.3422|5/0/5|1.0000|
|RND-Unit-75451|B1|M2_vs_M0|-0.03307 [-0.05445,-0.01391]|-0.00521|-0.3101|-1.1468|4/0/6|1.0000|
|RND-Unit-75451|B1|M2_vs_M1|-0.01494 [-0.04009,0.00606]|-0.00159|-0.2029|-0.1136|5/0/5|1.0000|
|RND-Unit-75451|B2|M2_vs_M0|-0.03671 [-0.06844,-0.00689]|-0.00723|-0.7410|-1.6053|3/1/6|1.0000|
|RND-Unit-75451|B2|M2_vs_M1|-0.01397 [-0.03320,0.00448]|-0.00214|-0.2060|-0.4572|4/1/5|1.0000|
|RND-Unit-75452|B1|M2_vs_M0|-0.02370 [-0.05106,0.00686]|-0.00384|-0.0314|-1.1494|4/1/5|1.0000|
|RND-Unit-75452|B1|M2_vs_M1|-0.00557 [-0.03055,0.01920]|-0.00022|0.0758|-0.1162|6/0/4|1.0000|
|RND-Unit-75452|B2|M2_vs_M0|-0.05027 [-0.07852,-0.02316]|-0.01042|-1.0436|-2.1774|1/0/9|0.3516|
|RND-Unit-75452|B2|M2_vs_M1|-0.02753 [-0.04858,-0.00899]|-0.00533|-0.5086|-1.0293|2/0/8|1.0000|
|RND-Unit-75453|B1|M2_vs_M0|-0.04438 [-0.07215,-0.02119]|-0.00722|-0.5134|-1.4890|2/0/8|1.0000|
|RND-Unit-75453|B1|M2_vs_M1|-0.02624 [-0.06064,0.00699]|-0.00360|-0.4062|-0.4559|5/0/5|1.0000|
|RND-Unit-75453|B2|M2_vs_M0|-0.05193 [-0.07834,-0.02765]|-0.01014|-0.7205|-1.6053|1/0/9|1.0000|
|RND-Unit-75453|B2|M2_vs_M1|-0.02919 [-0.05064,-0.00986]|-0.00506|-0.1855|-0.4572|4/0/6|1.0000|

B2 paired utility after excluding Fold 2 (nine-fold descriptive; no refit):

|Prompt|Contrast|Log-loss improvement|Prob-AUC improvement (pp)|W/T/L Log-loss|W/T/L Prob-AUC|
|---|---|---:|---:|---|---|
|P1-SPromptGL-8|M2_vs_M0|-0.02799|-0.4075|4/0/5|3/0/6|
|P1-SPromptGL-8|M2_vs_M1|-0.01116|0.0927|4/0/5|5/0/4|
|P2-BrainPrompt-2|M2_vs_M0|-0.02212|-0.4927|1/0/8|1/0/8|
|P2-BrainPrompt-2|M2_vs_M1|-0.00529|0.0075|6/0/3|5/0/4|
|P3-Class-Definition-8|M2_vs_M0|-0.02740|-0.5563|3/0/6|2/0/7|
|P3-Class-Definition-8|M2_vs_M1|-0.01057|-0.0560|5/0/4|4/0/5|
|P4-Modality-Aware-8|M2_vs_M0|-0.04157|-0.6716|2/0/7|1/0/8|
|P4-Modality-Aware-8|M2_vs_M1|-0.02474|-0.1714|2/0/7|3/0/6|
|P5-Functional-Network-8|M2_vs_M0|-0.03512|-0.7037|1/0/8|2/0/7|
|P5-Functional-Network-8|M2_vs_M1|-0.01829|-0.2035|3/0/6|5/1/3|
|P6-Heterogeneity-Aware-8|M2_vs_M0|-0.04230|-0.8254|1/0/8|1/0/8|
|P6-Heterogeneity-Aware-8|M2_vs_M1|-0.02547|-0.3251|2/0/7|2/0/7|

## Negative controls

C1 fixed-head rows preserve the correctly fitted diagnostic and reverse the ASD/TD score direction. The refit rows train the same capacity with swapped class semantics, so the learner can learn the reversal; this is not a strict semantic-destruction test.

|Wrong-class bank|Setting|Method|ACC|Prob-AUC|
|---|---|---|---:|---:|
|P1-SPromptGL-WrongClass|B1-fixed|B1-Prompt|89.5520 ± 2.7903|92.9901 ± 2.1953|
|P1-SPromptGL-WrongClass|B1-fixed|M2-B1-BasePlusPrompt|89.7806 ± 2.3855|93.2663 ± 2.2543|
|P1-SPromptGL-WrongClass|B2-trained-alignment|B2-Prompt|89.3182 ± 2.9204|93.2828 ± 2.8243|
|P1-SPromptGL-WrongClass|B2-trained-alignment|M2-B2-BasePlusPrompt|89.5468 ± 2.5017|92.9282 ± 2.4958|
|P1-SPromptGL-WrongClass|C1-B2-fixed-head|C1-B2-WrongClass-FixedHead|26.1925 ± 15.1121|13.3526 ± 12.7791|
|P1-SPromptGL-WrongClass|C1-fixed-head|C1-WrongClass-FixedHead|27.8971 ± 13.8846|6.8344 ± 2.0801|
|P1-SPromptGL-WrongClass|C1-refit-head|C1-WrongClass-RefitHead|89.5520 ± 2.7903|92.9901 ± 2.1953|
|P4-Modality-Aware-WrongClass|B1-fixed|B1-Prompt|89.5481 ± 3.6225|92.9318 ± 2.7542|
|P4-Modality-Aware-WrongClass|B1-fixed|M2-B1-BasePlusPrompt|89.4357 ± 2.8580|93.0655 ± 2.8939|
|P4-Modality-Aware-WrongClass|B2-trained-alignment|B2-Prompt|89.0896 ± 3.1875|92.7270 ± 3.0931|
|P4-Modality-Aware-WrongClass|B2-trained-alignment|M2-B2-BasePlusPrompt|88.9760 ± 3.2652|93.3073 ± 2.4385|
|P4-Modality-Aware-WrongClass|C1-B2-fixed-head|C1-B2-WrongClass-FixedHead|25.0483 ± 13.8312|16.4742 ± 16.8847|
|P4-Modality-Aware-WrongClass|C1-fixed-head|C1-WrongClass-FixedHead|22.8383 ± 11.9265|11.6054 ± 13.0659|
|P4-Modality-Aware-WrongClass|C1-refit-head|C1-WrongClass-RefitHead|89.5481 ± 3.6225|92.9318 ± 2.7542|
|P5-Functional-Network-WrongClass|B1-fixed|B1-Prompt|88.5123 ± 3.8565|93.1447 ± 2.7768|
|P5-Functional-Network-WrongClass|B1-fixed|M2-B1-BasePlusPrompt|89.3169 ± 2.9661|93.1552 ± 2.7601|
|P5-Functional-Network-WrongClass|B2-trained-alignment|B2-Prompt|89.5468 ± 3.3188|92.6838 ± 2.6302|
|P5-Functional-Network-WrongClass|B2-trained-alignment|M2-B2-BasePlusPrompt|89.7793 ± 3.0672|92.7541 ± 2.5447|
|P5-Functional-Network-WrongClass|C1-B2-fixed-head|C1-B2-WrongClass-FixedHead|31.7150 ± 18.3968|12.2443 ± 12.8789|
|P5-Functional-Network-WrongClass|C1-fixed-head|C1-WrongClass-FixedHead|18.6129 ± 12.9788|6.8553 ± 2.7768|
|P5-Functional-Network-WrongClass|C1-refit-head|C1-WrongClass-RefitHead|88.5123 ± 3.8565|93.1447 ± 2.7768|

## Fold 2 exposed-row sensitivity

Descriptive predictions after removing only 1-based rows 377 (ASD) and 407 (TD) from the Fold-2 test metrics; no model or probe was refit for this calculation.

|Prompt/control|Method|ACC|Prob-AUC|Log-loss|N test after removal|
|---|---|---:|---:|---:|---:|
|BASE|S0-Base|91.7647|92.4444|0.33637|85|
|NO_TEXT|M0-Calibrated-Base|91.7647|92.4444|0.37556|85|
|NO_TEXT|M1-B2-Trainable-NoText|85.8824|91.5000|0.45872|85|
|P1-SPromptGL-8|M2-B2-BasePlusPrompt|84.7059|91.8889|0.46416|85|
|P2-BrainPrompt-2|M2-B2-BasePlusPrompt|88.2353|91.6111|0.43764|85|
|P3-Class-Definition-8|M2-B2-BasePlusPrompt|84.7059|90.9444|0.47731|85|
|P4-Modality-Aware-8|M2-B2-BasePlusPrompt|87.0588|91.0000|0.49291|85|
|P5-Functional-Network-8|M2-B2-BasePlusPrompt|87.0588|90.6667|0.41753|85|
|P6-Heterogeneity-Aware-8|M2-B2-BasePlusPrompt|87.0588|91.8333|0.45124|85|

## Token diagnostics

S1-Token, S2-CAM and S3-DARG probes are in `prompt_comparison.csv`; individual prompt-token similarity and train-bootstrap top-4 stability are in `token_diagnostics.csv`. The four output-position bins below reflect learned CAM feature positions and cannot be assigned anatomical or canonical functional-network names.

|Prompt|Slot origin|Mean supervised importance|Mean prompt contrast|Bootstrap top-4 frequency|
|---|---|---:|---:|---:|
|C3-Unrelated-English-8|PHENO-origin|4.016|0.024993|0.2963|
|C3-Unrelated-English-8|ANAT-origin|6.2078|0.02371|0.1739|
|C3-Unrelated-English-8|FUNC-origin|6.8933|0.025535|0.2225|
|C3-Unrelated-English-8|CORR-origin|3.0027|0.024256|0.3074|
|P1-SPromptGL-8|PHENO-origin|4.016|0.027531|0.2253|
|P1-SPromptGL-8|ANAT-origin|6.2078|0.02878|0.2484|
|P1-SPromptGL-8|FUNC-origin|6.8933|0.028819|0.2538|
|P1-SPromptGL-8|CORR-origin|3.0027|0.029151|0.2726|
|P1-SPromptGL-Anonymous-Gram-Matched|PHENO-origin|4.016|0.031164|0.2353|
|P1-SPromptGL-Anonymous-Gram-Matched|ANAT-origin|6.2078|0.030376|0.2964|
|P1-SPromptGL-Anonymous-Gram-Matched|FUNC-origin|6.8933|0.029026|0.1483|
|P1-SPromptGL-Anonymous-Gram-Matched|CORR-origin|3.0027|0.027415|0.3201|
|P2-BrainPrompt-2|PHENO-origin|4.016|0.015458|0.3862|
|P2-BrainPrompt-2|ANAT-origin|6.2078|0.014157|0.2242|
|P2-BrainPrompt-2|FUNC-origin|6.8933|0.013616|0.2174|
|P2-BrainPrompt-2|CORR-origin|3.0027|0.012368|0.1721|
|P3-Class-Definition-8|PHENO-origin|4.016|0.020357|0.2444|
|P3-Class-Definition-8|ANAT-origin|6.2078|0.02082|0.2515|
|P3-Class-Definition-8|FUNC-origin|6.8933|0.022129|0.1851|
|P3-Class-Definition-8|CORR-origin|3.0027|0.022497|0.3190|
|P4-Modality-Aware-8|PHENO-origin|4.016|0.024405|0.1421|
|P4-Modality-Aware-8|ANAT-origin|6.2078|0.028472|0.2834|
|P4-Modality-Aware-8|FUNC-origin|6.8933|0.02646|0.2485|
|P4-Modality-Aware-8|CORR-origin|3.0027|0.030708|0.3260|
|P5-Functional-Network-8|PHENO-origin|4.016|0.020334|0.1516|
|P5-Functional-Network-8|ANAT-origin|6.2078|0.026232|0.3236|
|P5-Functional-Network-8|FUNC-origin|6.8933|0.024364|0.2256|
|P5-Functional-Network-8|CORR-origin|3.0027|0.025533|0.2991|
|P5-Functional-Network-Anonymous-Gram-Matched|PHENO-origin|4.016|0.021039|0.2888|
|P5-Functional-Network-Anonymous-Gram-Matched|ANAT-origin|6.2078|0.022186|0.2275|
|P5-Functional-Network-Anonymous-Gram-Matched|FUNC-origin|6.8933|0.022185|0.2335|
|P5-Functional-Network-Anonymous-Gram-Matched|CORR-origin|3.0027|0.022398|0.2502|
|P6-Heterogeneity-Aware-8|PHENO-origin|4.016|0.020979|0.1889|
|P6-Heterogeneity-Aware-8|ANAT-origin|6.2078|0.022693|0.2135|
|P6-Heterogeneity-Aware-8|FUNC-origin|6.8933|0.024276|0.2849|
|P6-Heterogeneity-Aware-8|CORR-origin|3.0027|0.026966|0.3128|
|RND-Unit-75451|PHENO-origin|4.016|0.039078|0.1279|
|RND-Unit-75451|ANAT-origin|6.2078|0.047872|0.2821|
|RND-Unit-75451|FUNC-origin|6.8933|0.046189|0.2505|
|RND-Unit-75451|CORR-origin|3.0027|0.043025|0.3395|
|RND-Unit-75452|PHENO-origin|4.016|0.045754|0.3355|
|RND-Unit-75452|ANAT-origin|6.2078|0.045534|0.2152|
|RND-Unit-75452|FUNC-origin|6.8933|0.045564|0.2276|
|RND-Unit-75452|CORR-origin|3.0027|0.044634|0.2216|
|RND-Unit-75453|PHENO-origin|4.016|0.054222|0.3018|
|RND-Unit-75453|ANAT-origin|6.2078|0.04994|0.2774|
|RND-Unit-75453|FUNC-origin|6.8933|0.052231|0.2016|
|RND-Unit-75453|CORR-origin|3.0027|0.046428|0.2193|

The 16 standalone token probes report outer-test fold-mean performance below. Coefficient rank/top-four frequency is a diagnostic from separately fitted training-only logistic probes, not an explanation or anatomical assignment.

|Token|CAM output position|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|Mean coefficient rank|Supervised Top-4 fold frequency|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|PHENO-output-slot|78.41|78.14|84.55|74.90|81.37|76.16|9.50|0.00|
|1|PHENO-output-slot|77.26|77.03|82.98|73.95|80.12|74.81|11.00|0.10|
|2|PHENO-output-slot|79.10|78.88|84.54|75.95|81.82|76.78|11.20|0.00|
|3|PHENO-output-slot|77.85|77.67|84.15|75.40|79.94|75.75|10.90|0.00|
|4|ANAT-output-slot|84.61|84.44|89.81|82.14|86.73|83.13|6.10|0.30|
|5|ANAT-output-slot|83.81|83.59|88.50|80.64|86.53|81.92|6.40|0.30|
|6|ANAT-output-slot|86.45|86.19|91.36|82.66|89.72|84.94|4.60|0.40|
|7|ANAT-output-slot|83.23|83.04|87.99|80.63|85.45|81.57|5.60|0.30|
|8|FUNC-output-slot|81.74|81.55|88.68|79.14|83.96|79.83|3.30|0.70|
|9|FUNC-output-slot|76.12|75.73|83.14|70.89|80.56|72.21|5.50|0.50|
|10|FUNC-output-slot|79.46|79.06|86.34|73.95|84.18|76.54|4.00|0.80|
|11|FUNC-output-slot|75.65|75.28|82.10|70.42|80.13|72.00|5.20|0.60|
|12|CORR-output-slot|89.09|89.05|93.60|88.59|89.51|88.27|14.30|0.00|
|13|CORR-output-slot|88.63|88.63|93.72|88.60|88.65|87.84|12.20|0.00|
|14|CORR-output-slot|89.32|89.24|93.23|88.34|90.15|88.52|14.00|0.00|
|15|CORR-output-slot|88.74|88.68|93.19|87.85|89.51|87.93|12.20|0.00|

Prompt text geometry is measured in the actual frozen 1024-D CLIP RN50 space before any B1/B2 projection. `prompt_semantic_geometry.csv` includes each sentence norm, within-class cosine and ASD-TD cosine; random unit-vector controls are reported separately.

|Prompt/control|N|Mean norm|Within ASD cosine|Within TD cosine|Between-class cosine|
|---|---:|---:|---:|---:|---:|
|C3-Unrelated-English-8|8|11.79496|0.6111055184365551|0.7127071835564625|0.6754727449872409|
|P1-SPromptGL-8|8|12.56893|0.8049269050016212|0.7601535930042967|0.758950663809137|
|P2-BrainPrompt-2|2|10.59343|NA|NA|0.9847825654411677|
|P3-Class-Definition-8|8|12.96043|0.8833923251852379|0.8375854652200386|0.8477585333745579|
|P4-Modality-Aware-8|8|11.77524|0.8508631547314888|0.7621044743680035|0.7662276139456323|
|P5-Functional-Network-8|8|11.39689|0.8939136128022828|0.9063339408537355|0.8506249751332444|
|P6-Heterogeneity-Aware-8|8|11.88488|0.9044899589776754|0.8344731155074219|0.8411334007002133|
|random_unit_75451|8|1.00000|0.004616696195858039|-0.005783472301210846|0.007070435216180086|
|random_unit_75452|8|1.00000|0.0004269063526463176|0.009492714647727545|0.0036053519655490957|
|random_unit_75453|8|1.00000|0.006017400634890272|-0.0071017574700465895|-0.020762312096997356|
Every individual text vector norm, source class, within-class cosine and opposite-class cosine is stored as a `prompt_item` row in `prompt_semantic_geometry.csv`.

P4/P5 prompt similarity by source-slot index is in `prompt_slot_alignment.csv`. P4 is only an exploratory position correspondence: model tokens have cross-modal CAM interactions. P5 cannot establish DMN, attention, sensorimotor or frontoparietal localization without an audited edge-to-network mapping. The ABIDE functional-connectivity literature reports substantial site and sample variability, so the candidate network sentences are not treated as universal participant-level findings.

## Historical protocol and limitations

- E74.4-G2 checkpoint selection used `val=test`; preprocessing/feature selection occurred before outer CV; graphs are 871-node transductive. Historical Fold 2 contains 1-based rows 377 (ASD) and 407 (TD) in both train and test. Both rows were removed from all new Fold-2 probe training and inner-CV tuning. The fixed historical base prediction still reflects the original selection/training protocol.
- Ten-fold and nine-fold metrics are descriptive paired summaries on frozen checkpoints. Removing Fold 2 or its two exposed rows does not create a new leakage-free evaluation.
- P2 has one prompt per class versus four per class for most banks; class means are used for compatibility, but prompt-count and wording differences remain.
- B2 can learn a discriminative direction from labels using any fixed prompt vectors; positive B2 results require comparison to random/anonymous controls and M1, and do not by themselves establish medical semantic value.
- The functional-connectivity prompt literature motivates candidate text only. The current 256 CORR positions lack verified named-network mapping; no network-localization claim is made.

## Decision

- Scientific classification: **NO_DETECTABLE_PROMPT_BENEFIT**.
- Rationale: The real Prompt banks did not show a stable incremental advantage over calibrated-base and no-text controls under the registered probes.
- This is a frozen-model diagnostic, not an end-to-end validation of a second innovation module. Any utility is restricted learner use of existing CAM/DARG information plus a semantic prior; no information-theoretic increase in patient data is implied.

## Sources

- [SPromptGL MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf); [original repository](https://github.com/wanxixi11/SPromptGL).
- [BrainPrompt MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf); [official ABIDE label prompt file](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt).
- Candidate network wording is grounded as group-level literature context, not a per-subject assertion: [multisite ABIDE functional-connectivity classification](https://pubmed.ncbi.nlm.nih.gov/24093016/), [frontoparietal/sensorimotor connectivity in ASD](https://pubmed.ncbi.nlm.nih.gov/40388040/), and [default-mode findings plus site/cohort reproducibility limits](https://pmc.ncbi.nlm.nih.gov/articles/PMC6591952/), [site and cohort variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC5628079/). These papers do not create a mapping from this experiment's 256 CORR indices to named networks.
