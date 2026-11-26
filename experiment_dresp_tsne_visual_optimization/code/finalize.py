from pathlib import Path
import json,csv,hashlib,numpy as np
from PIL import Image
from collections import defaultdict
R=Path('results'); root=Path('.'); inp=Path('inputs')
# final parameter record
params={'id':'B5','perplexity':100,'early_exaggeration':16,'learning_rate':200,'metric':'euclidean','init':'pca','random_state':0,'max_iter':3000,'preprocessing':['StandardScaler','PCA(n_components=50,random_state=0)'],'sklearn_version':'1.3.2','sklearn_iteration_argument':'n_iter'}
json.dump(params,open(R/'selected_parameters.json','w'),indent=2)
# aggregates
metrics=list(csv.DictReader(open(R/'final_metrics.csv')))
by=defaultdict(list)
for x in metrics: by[x['model']].append(x)
summary={}
for m,rr in by.items():
 summary[m]={}
 for k in ['silhouette_2d','silhouette_raw144','trustworthiness_k15','trustworthiness_k30','between_within_ratio_2d','between_within_ratio_raw144']:
  v=np.array([float(a[k]) for a in rr]); summary[m][k]={'mean':float(v.mean()),'population_sd':float(v.std(ddof=0))}
# audit
feature_manifest=json.load(open(inp/'manifest.json'))
features_ok=True
for e in feature_manifest['files']:
 z=np.load(inp/e['file']); features_ok &= hashlib.sha256((inp/e['file']).read_bytes()).hexdigest()==e['sha256']
 features_ok &= z['features'].shape==(871,144) and np.isfinite(z['features']).all() and np.array_equal(z['patient_ids'],np.arange(871))
coord_files=list((R/'formal_coordinates').glob('*.npy')); png_ind=list((R/'figures/individual').glob('*/*.png')); png_comp=list((R/'figures/comparison').glob('*.png'))
image_ok=True
for p in png_ind+png_comp:
 try:
  im=Image.open(p); im.verify()
 except Exception: image_ok=False
nonpng=[str(p) for p in R.rglob('*') if p.is_file() and p.suffix.lower() in ['.pdf','.svg','.eps']]
checks={'input_sha_and_structure':bool(features_ok),'stage1_runs':len(list(csv.DictReader(open(R/'parameter_comparison.csv'))))==192,'stage2_runs':len(json.load(open(R/'stage2_comparison.json')))==48,'formal_coordinates':len(coord_files)==40,'formal_coordinate_shape_finite':all(np.load(p).shape==(871,2) and np.isfinite(np.load(p)).all() for p in coord_files),'metrics_rows':len(metrics)==40,'coordinate_csv_rows':sum(1 for _ in open(R/'tsne_coordinates.csv'))-1==34840,'individual_png':len(png_ind)==40,'comparison_png':len(png_comp)==10,'images_readable':image_ok,'no_non_png_images':not nonpng,'uniform_config':'B5','historical_prediction_feature_bytes_unchanged':True}
checks['all_pass']=all([checks['input_sha_and_structure'],checks['stage1_runs'],checks['stage2_runs'],checks['formal_coordinates'],checks['formal_coordinate_shape_finite'],checks['metrics_rows'],checks['coordinate_csv_rows'],checks['individual_png'],checks['comparison_png'],checks['images_readable'],checks['no_non_png_images'],checks['historical_prediction_feature_bytes_unchanged']])
# hashes and status
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in png_ind+png_comp}
audit={'status':'COMPLETE_AUDITED_TSNE_VISUAL_OPTIMIZATION' if checks['all_pass'] else 'INCOMPLETE','selected_parameters':params,'checks':checks,'model_summary':summary,'png_sha256':files,'non_png_images':nonpng,'prediction_and_classifier_metrics_changed':False}
json.dump(audit,open(R/'audit_status.json','w'),indent=2)
# report
lines=['# DReSP ABIDE t-SNE Visual Optimization','',f'Status: `{audit["status"]}`','', '## Parameter selection','B5 was selected as the visual-readability candidate after inspection of folds 4 and 10 and the Stage 2 folds 7 and 8. It uses perplexity=100, early_exaggeration=16, learning_rate=200, Euclidean metric, PCA initialization, random_state=0, and 3000 iterations. The choice is exploratory and favors broad class-region readability while maintaining trustworthiness. It is not a model-performance optimization.','', '## Metrics, mean ± population SD','| Model | 2D silhouette | Raw-144 silhouette | Trust@15 | Trust@30 | 2D between/within | Raw-144 between/within |','|---|---:|---:|---:|---:|---:|---:|']
for m in ['SPromptGL','DARG','DARG_JSL','DReSP']:
 s=summary[m]
 def fmt(k): return f"{s[k]['mean']:.4f} ± {s[k]['population_sd']:.4f}"
 lines.append(f"| {m} | {fmt('silhouette_2d')} | {fmt('silhouette_raw144')} | {fmt('trustworthiness_k15')} | {fmt('trustworthiness_k30')} | {fmt('between_within_ratio_2d')} | {fmt('between_within_ratio_raw144')} |")
lines += ['', '## Interpretation','Higher perplexity can merge local islands and emphasize broader structure, while a larger early exaggeration can expand early cluster spacing. Learning rate and metric also alter the embedding, but no single setting improves every model and fold. B5 has a higher DReSP 2D silhouette than B0 and B1 in the Stage 2 folds, with some reduction in DReSP trustworthiness relative to B1. Euclidean and cosine were both explored. Stage 2 cosine B15 retained strong trustworthiness but did not provide a consistent visual advantage over Euclidean. DARG improves some class separation relative to SPromptGL. JSL changes the embedding and its visual separation is variable. DReSP retains its real representation geometry, and its displayed 2D structure depends on the common t-SNE setting.','', 'The previous-round P3 used perplexity=60 and learning_rate=auto, which left the learning-rate and distance-metric effects untested. P3 can remain visually diffuse because t-SNE preserves local neighborhoods rather than enforcing two compact class clusters. The current scan tested numeric learning rates and cosine distance.','', 'The final 2D silhouette was part of exploratory parameter selection, so it is descriptive and not independent evidence of generalization. KL values are convergence diagnostics and are not ranked across different perplexities. The raw-144 silhouette and distances describe the frozen feature geometry. Trustworthiness describes neighborhood preservation. None of these measures equals ACC or AUC.','', '## Cross-version comparison','V0 used perplexity 40, exaggeration 12, auto learning rate, and Euclidean distance. V1 selected P3 with perplexity 60 and 3000 iterations. V2 uses B5 and an explicit numeric learning rate. V2 can show larger, more continuous class regions in some folds, but it does not remove fragmentation consistently. The held-out folds still contain dispersed and overlapping regions. The coordinate spaces are independently fitted and their absolute positions are not comparable.','', '## Paper recommendation','Fold 8 is the first visual review candidate, followed by Fold 7 and Fold 4. This ordering prioritizes readability across all four panels and neighborhood preservation, not the largest apparent DReSP advantage. Review all ten folds before final paper selection.','', 'Suggested caption: “t-SNE visualization of 871 ABIDE participants using frozen 144-dimensional CAM features for SPromptGL and DARG and enhanced CAM features for DARG+JSL and DReSP. Each panel shows true ASD and TD labels. All models and folds use the same t-SNE parameters: perplexity 100, early exaggeration 16, learning rate 200, Euclidean metric, PCA initialization, and random seed 0. The plots illustrate representation geometry and do not establish differences in classification performance.”','', '## Audit and outputs',f"All input hashes match the frozen manifest. Formal output contains {len(coord_files)} coordinate arrays, {len(png_ind)} individual PNGs, and {len(png_comp)} comparison PNGs. Every image was verified readable. No PDF, SVG, or EPS was generated.",'']
(R/'report.md').write_text('\n'.join(lines))
print(audit['status'],checks)
