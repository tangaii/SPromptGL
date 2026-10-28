from pathlib import Path
import hashlib,json,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parent; RT=ROOT/'.runtime'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
files={}
for p in sorted((RT/'sources').glob('*')):
 if p.is_file():files[f'sources/{p.name}']=sha(p)
for p in sorted((RT/'anchors').glob('fold*.pt')):files[f'anchors/{p.name}']=sha(p)
for p in sorted((RT/'checkpoints').glob('*.best.pt')):files[f'checkpoints/{p.name}']=sha(p)
for p in sorted((RT/'fixed_graphs').glob('fold*.pt')):files[f'fixed_graphs/{p.name}']=sha(p)
protected={}
for n in ['main.py','model.py','network.py','layers.py','utils.py']:
 p=Path('/data/adodas/SPromptGL-main')/n
 if p.exists():protected[n]=sha(p)
datafiles={}
for p in [Path('/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv'),Path('/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy')]:
 if p.exists():datafiles[str(p)]=sha(p)
obj={'experiment':'47.2','execution_root':str(ROOT),'files':files,'protected_root_hash_after_run':protected,'input_hashes':datafiles,'results':{p.name:sha(p) for p in sorted((ROOT/'results').glob('*')) if p.is_file()}}
(ROOT/'manifest.json').write_text(json.dumps(obj,indent=2))
print(json.dumps({'runtime_files':len(files),'result_files':len(obj['results'])},indent=2))
