"""Deterministic E83.3 bridge from MMGPL-12 RN50 text to historical SCMR queries."""
from pathlib import Path
import hashlib, json, torch
from torch.nn import functional as F

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(old_prompt, new_prompt, out):
    old=torch.load(old_prompt,map_location='cpu',weights_only=False)
    new=torch.load(new_prompt,map_location='cpu',weights_only=False)
    old_m=F.normalize(old['medical'].float(),dim=-1); old_a=F.normalize(old['anonymous'].float(),dim=-1)
    # Orthogonal Procrustes maps historical medical space to historical anonymous space.
    u,_,vh=torch.linalg.svd(old_m.T @ old_a, full_matrices=False)
    R=u @ vh
    new_m=F.normalize(new['medical'].float(),dim=-1)
    new_a=F.normalize(new_m @ R,dim=-1)
    P=old['P_text'].float()
    q_m=F.normalize(new_m,dim=-1) @ P
    q_a=F.normalize(new_a,dim=-1) @ P
    payload={'medical':new['medical'].float(),'anonymous':new_a,'P_text':P,'q_medical':q_m,'q_anonymous':q_a,
             'texts':new['texts'],'groups':new['groups'],'query_count':12,'query_width':32,
             'method':'q_medical=normalize(MM GPL12 RN50) @ historical P_text; anonymous=normalize(medical @ Procrustes_R), R from historical 8-row medical->anonymous orthogonal Procrustes','historical_prompt_sha':sha(old_prompt),'new_prompt_sha':sha(new_prompt),'P_text_sha':hashlib.sha256(P.numpy().tobytes()).hexdigest(),'procrustes_sha':hashlib.sha256(R.numpy().tobytes()).hexdigest()}
    torch.save(payload,out); return payload
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    p=build(root/'sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt',root/'inputs/prompts/mmGPL12_rn50.pt',root/'inputs/prompts/mmGPL12_scmr.pt')
    print(p['query_count'],p['q_medical'].shape,p['q_anonymous'].shape,sha(root/'inputs/prompts/mmGPL12_scmr.pt'))
