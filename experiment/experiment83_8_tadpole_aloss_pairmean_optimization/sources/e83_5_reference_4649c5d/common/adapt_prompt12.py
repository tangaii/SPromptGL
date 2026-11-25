"""Deterministic Gram-preserving MMGPL-12 bridge for SCMR queries."""
from pathlib import Path
import hashlib, json, torch
from torch.nn import functional as F

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(old_prompt, new_prompt, out):
    old=torch.load(old_prompt,map_location='cpu',weights_only=False)
    new=torch.load(new_prompt,map_location='cpu',weights_only=False)
    new_m=F.normalize(new['medical'].float(),dim=-1)
    # The Cholesky rows are a deterministic factor of the frozen RN50 Gram
    # matrix.  Zero padding embeds the exact 12-query geometry in width 32.
    features64 = new_m.double()
    gram = features64 @ features64.T
    gram = (gram + gram.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues.min()) <= 0:
        raise RuntimeError(f"MMGPL12 Gram matrix is not positive definite: {eigenvalues.tolist()}")
    chol = torch.linalg.cholesky(gram)
    q_m64 = F.pad(chol, (0, 32 - chol.shape[1]))
    q_m = q_m64.float()
    # Anonymous is retained as a diagnostic route.  It uses the same fixed
    # Gram-preserving embedding and is never selected using labels or results.
    q_a = q_m.clone()
    payload={'medical':new['medical'].float(),'anonymous':new['medical'].float(),'P_text':old['P_text'].float(),
             'q_medical':q_m,'q_anonymous':q_a,
             'texts':new['texts'],'groups':new['groups'],'query_count':12,'query_width':32,
             'method':'q_medical=pad32(cholesky(normalize(RN50) @ normalize(RN50).T)); float64 Gram, deterministic lower Cholesky, zero-pad to 32; q_anonymous diagnostic copy','historical_prompt_sha':sha(old_prompt),'new_prompt_sha':sha(new_prompt),'P_text_sha':hashlib.sha256(old['P_text'].float().numpy().tobytes()).hexdigest(),'gram_sha':hashlib.sha256(gram.numpy().tobytes()).hexdigest(),'gram_symmetry_max_abs':float((gram-gram.T).abs().max()),'gram_min_eigenvalue':float(eigenvalues.min()),'gram_rank':int(torch.linalg.matrix_rank(gram)),'gram_reconstruction_max_abs':float((q_m64@q_m64.T-gram).abs().max())}
    torch.save(payload,out); return payload
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    p=build(root/'sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt',root/'inputs/prompts/mmGPL12_rn50.pt',root/'inputs/prompts/mmGPL12_scmr.pt')
    print(p['query_count'],p['q_medical'].shape,p['q_anonymous'].shape,sha(root/'inputs/prompts/mmGPL12_scmr.pt'))
