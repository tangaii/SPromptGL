from pathlib import Path
import json,hashlib,numpy as np,csv
from datetime import datetime
R=Path('results'); ROOT=Path('.')
run=json.load(open(R/'stage4_run_manifest.json'))
IN=Path('inputs'); man={e['file']:e['sha256'] for e in json.load(open(IN/'manifest.json'))['files']}
checks={
 'inputs_sha_match': all(hashlib.sha256((IN/f).read_bytes()).hexdigest()==s for f,s in man.items()),
 'formal_coords_40': len(list((R/'formal_coordinates').glob('*.npy')))==40,
 'run_manifest_40': len(run['runs'])==40,
 'uniform_params': run['config']=='B5',
 'png_individual_40': len(list((R/'figures/individual').glob('*/*.png')))==40,
 'png_comparison_10': len(list((R/'figures/comparison').glob('*.png')))==10,
 'no_pdf': not list(R.rglob('*.pdf')) and not list(R.rglob('*.svg')) and not list(R.rglob('*.eps')),
 'metrics_rows_40': sum(1 for _ in open(R/'final_metrics.csv'))-1==40,
 'coords_csv_rows_34840': sum(1 for _ in open(R/'tsne_coordinates.csv'))-1==34840,
}
checks['all_pass']=all(checks.values())
json.dump({'timestamp':datetime.now().isoformat(),'config':run['config'],'param':run['param'],'checks':checks,'status':'COMPLETE_AUDITED_TSNE_VISUAL_OPTIMIZATION' if checks['all_pass'] else 'INCOMPLETE'},open(R/'audit_status.json','w'),indent=2)
print(checks)
