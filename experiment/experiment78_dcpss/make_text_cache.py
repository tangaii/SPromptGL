import torch,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/data/adodas/SPromptGL-main')
from CLIP.clip import clip
cfg=json.loads(Path('/model/spromptgl/experiment/experiment78_dcpss/config.json').read_text())
ps=cfg['prompts']; texts=ps['spromptgl_text']+ps['brainprompt_text']+ps['anonymous_text']
model,_=clip.load('RN50','cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad(): emb=model.encode_text(clip.tokenize(texts).to(next(model.parameters()).device)).float().cpu()
out={'texts':emb,'spromptgl':emb[:8],'brainprompt':emb[8:10],'anonymous':emb[10:18],'text_strings':texts}
p=Path('/model/spromptgl/experiment/experiment78_dcpss/runtime/text_embeddings.pt');torch.save(out,p)
print(p,emb.shape,hashlib.sha256(p.read_bytes()).hexdigest())
