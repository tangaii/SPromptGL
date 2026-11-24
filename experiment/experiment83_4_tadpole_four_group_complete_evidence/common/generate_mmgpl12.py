import sys, json, hashlib, pathlib, torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources'))
from CLIP.clip import clip
texts=[
'Normal brain volume','Normal ventricle morphology','Symmetrical brain structure','Uniform cortical thickness',
'Mild hippocampal atrophy','Mild brain volume reduction','Mild ventricular enlargement','Mild local cortical thinning',
'Pronounced hippocampal atrophy','Marked local cortical thinning','Significant brain volume reduction','Significant ventricular enlargement']
weights=pathlib.Path('/root/.cache/clip/RN50.pt')
sha=hashlib.sha256(weights.read_bytes()).hexdigest()
assert sha=='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
model,_=clip.load(str(weights),device='cpu',jit=False); model.eval()
with torch.no_grad():
  tokens=clip.tokenize(texts,truncate=False)
  emb=model.encode_text(tokens).float()
assert emb.shape==(12,1024) and torch.isfinite(emb).all()
out=ROOT/'inputs/prompts/mmGPL12_rn50.pt'
torch.save({'medical':emb,'texts':texts,'groups':{'NC':[0,1,2,3],'MCI':[4,5,6,7],'AD':[8,9,10,11]},'clip_weights_sha256':sha,'encoder':'OpenAI CLIP RN50','tokenizer':'CLIP SimpleTokenizer','source_url':'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt'},out)
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), emb.shape)
# repeat determinism
with torch.no_grad(): emb2=model.encode_text(clip.tokenize(texts,truncate=False)).float()
assert torch.equal(emb,emb2)
print('repeat_exact=True')
