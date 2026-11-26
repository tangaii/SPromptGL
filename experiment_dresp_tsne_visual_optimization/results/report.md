# DReSP ABIDE t-SNE Visual Optimization

Status: `COMPLETE_AUDITED_TSNE_VISUAL_OPTIMIZATION`

## Parameter selection
B5 was selected as the visual-readability candidate after inspection of folds 4 and 10 and the Stage 2 folds 7 and 8. It uses perplexity=100, early_exaggeration=16, learning_rate=200, Euclidean metric, PCA initialization, random_state=0, and 3000 iterations. The choice is exploratory and favors broad class-region readability while maintaining trustworthiness. It is not a model-performance optimization.

## Metrics, mean ± population SD
| Model | 2D silhouette | Raw-144 silhouette | Trust@15 | Trust@30 | 2D between/within | Raw-144 between/within |
|---|---:|---:|---:|---:|---:|---:|
| SPromptGL | 0.3939 ± 0.0913 | 0.3645 ± 0.1094 | 0.9531 ± 0.0125 | 0.9382 ± 0.0152 | 1.9904 ± 0.5690 | 1.8190 ± 0.5761 |
| DARG | 0.4539 ± 0.1277 | 0.3867 ± 0.0934 | 0.9563 ± 0.0076 | 0.9395 ± 0.0099 | 2.4627 ± 0.7681 | 1.9126 ± 0.4614 |
| DARG_JSL | 0.4371 ± 0.1033 | 0.3359 ± 0.1271 | 0.9573 ± 0.0105 | 0.9417 ± 0.0123 | 2.3113 ± 0.6506 | 1.6908 ± 0.6648 |
| DReSP | 0.4569 ± 0.1544 | 0.3422 ± 0.1149 | 0.9617 ± 0.0097 | 0.9478 ± 0.0124 | 2.5160 ± 0.9771 | 1.6918 ± 0.5519 |

## Interpretation
Higher perplexity can merge local islands and emphasize broader structure, while a larger early exaggeration can expand early cluster spacing. Learning rate and metric also alter the embedding, but no single setting improves every model and fold. B5 has a higher DReSP 2D silhouette than B0 and B1 in the Stage 2 folds, with some reduction in DReSP trustworthiness relative to B1. Euclidean and cosine were both explored. Stage 2 cosine B15 retained strong trustworthiness but did not provide a consistent visual advantage over Euclidean. DARG improves some class separation relative to SPromptGL. JSL changes the embedding and its visual separation is variable. DReSP retains its real representation geometry, and its displayed 2D structure depends on the common t-SNE setting.

The previous-round P3 used perplexity=60 and learning_rate=auto, which left the learning-rate and distance-metric effects untested. P3 can remain visually diffuse because t-SNE preserves local neighborhoods rather than enforcing two compact class clusters. The current scan tested numeric learning rates and cosine distance.

The final 2D silhouette was part of exploratory parameter selection, so it is descriptive and not independent evidence of generalization. KL values are convergence diagnostics and are not ranked across different perplexities. The raw-144 silhouette and distances describe the frozen feature geometry. Trustworthiness describes neighborhood preservation. None of these measures equals ACC or AUC.

## Cross-version comparison
V0 used perplexity 40, exaggeration 12, auto learning rate, and Euclidean distance. V1 selected P3 with perplexity 60 and 3000 iterations. V2 uses B5 and an explicit numeric learning rate. V2 can show larger, more continuous class regions in some folds, but it does not remove fragmentation consistently. The held-out folds still contain dispersed and overlapping regions. The coordinate spaces are independently fitted and their absolute positions are not comparable.

## Paper recommendation
Fold 8 is the first visual review candidate, followed by Fold 7 and Fold 4. This ordering prioritizes readability across all four panels and neighborhood preservation, not the largest apparent DReSP advantage. Review all ten folds before final paper selection.

Suggested caption: “t-SNE visualization of 871 ABIDE participants using frozen 144-dimensional CAM features for SPromptGL and DARG and enhanced CAM features for DARG+JSL and DReSP. Each panel shows true ASD and TD labels. All models and folds use the same t-SNE parameters: perplexity 100, early exaggeration 16, learning rate 200, Euclidean metric, PCA initialization, and random seed 0. The plots illustrate representation geometry and do not establish differences in classification performance.”

## Audit and outputs
All input hashes match the frozen manifest. Formal output contains 40 coordinate arrays, 40 individual PNGs, and 10 comparison PNGs. Every image was verified readable. No PDF, SVG, or EPS was generated.
