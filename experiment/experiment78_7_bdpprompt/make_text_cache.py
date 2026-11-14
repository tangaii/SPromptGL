#!/usr/bin/env python3
"""Build the fixed 10-prompt RN50 cache from the SHA-checked historical bank."""
import hashlib, json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
SOURCE=Path('/model/spromptgl/experiment/experiment78_dcpss/runtime/text_embeddings.pt')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.loads((HERE/'config.json').read_text())
 src=torch.load(SOURCE,map_location='cpu',weights_only=False)
 if src['text_strings'][:10] != [
  'Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure','Normal ventricular morphology','Altered brain volume','Abnormal functional connectivity','White matter integrity issues','Different activation patterns',
  'A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.',
  'A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.']:
  raise RuntimeError('historical RN50 cache has unexpected medical/BrainPrompt order')
 # Source cache: TD four [0:4], ASD four [4:8], BrainPrompt TD [8], ASD [9].
 med_idx=[4,5,6,7,9,0,1,2,3,8]
 med=src['texts'][med_idx].float()
 anon_idx=[10,11,12,13,10,14,15,16,17,14]
 anon=src['texts'][anon_idx].float()
 med_text=cfg['prompts']['medical_text']; anon_text=cfg['prompts']['anonymous_text']
 if len(med_text)!=10 or len(anon_text)!=10: raise RuntimeError('prompt count')
 original=src['text_strings']
 if any(anon_text[i]!=original[anon_idx[i]] for i in range(10)): raise RuntimeError('anonymous cache/string mismatch')
 all_emb=torch.cat([med,anon],0)
 payload={'medical_embeddings':med,'anonymous_embeddings':anon,'embeddings':all_emb,
  'medical_text':med_text,'anonymous_text':anon_text,
  'medical_classes':torch.tensor(cfg['prompts']['medical_classes'],dtype=torch.long),
  'anonymous_classes':torch.tensor(cfg['prompts']['anonymous_classes'],dtype=torch.long),
  'source_cache_path':str(SOURCE),'source_cache_sha256':sha(SOURCE),
  'source_text_strings':original,'reused_anonymous_slots':[0,4],
  'text_dim':int(all_emb.shape[-1]),'encoding':'frozen OpenAI CLIP RN50; historical SHA-checked cache'}
 out=HERE/'runtime/text_embeddings.pt'; out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out)
 print(json.dumps({'path':str(out),'sha256':sha(out),'shape':list(all_emb.shape),'source_sha256':sha(SOURCE)},indent=2))
if __name__=='__main__': main()
