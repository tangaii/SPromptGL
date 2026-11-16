#!/usr/bin/env python3
"""Encode the frozen E78.8 texts with verified RN50 and build shared anchors."""
from __future__ import annotations
import hashlib, json, sys, types
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
REPO=Path('/data/adodas/SPromptGL-main')

def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _encode(texts, weights):
    if sha(weights)!='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762':
        raise RuntimeError('CLIP RN50 checkpoint SHA mismatch')
    if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name!='ftfy' or any(not text.isascii() for text in texts): raise
        compat=types.ModuleType('ftfy'); compat.fix_text=lambda text:text; sys.modules['ftfy']=compat
    from CLIP.clip import clip
    tokens=clip.tokenize(texts,truncate=False)
    lengths=(tokens!=0).sum(-1)
    if bool((lengths>77).any()): raise RuntimeError('a prompt was truncated by CLIP RN50 tokenizer')
    model,_=clip.load(str(weights),device='cpu',jit=False)
    model.eval()
    with torch.no_grad(): encoded=model.encode_text(tokens).float().cpu()
    del model
    return encoded,lengths

def main():
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    p=cfg['prompts']; enc=cfg['text_encoder']; weights=Path(enc['weights_path'])
    if not weights.is_file(): raise FileNotFoundError(weights)
    med_text=p['medical_text']; anon_text=p['anonymous_text']
    if len(med_text)!=10 or len(anon_text)!=10: raise RuntimeError('exactly ten texts per bank are required')
    if len(set(med_text))!=10 or len(set(anon_text))!=10: raise RuntimeError('duplicate prompts in a text bank')
    if any(not text.isascii() for text in med_text+anon_text): raise RuntimeError('unexpected non-ASCII prompt text')
    if any(any(term in text.lower() for term in ('brain','autism','asd','td','connectiv','disease','cortical','white matter','ventric')) for text in anon_text):
        raise RuntimeError('anonymous control contains disease or neuroanatomical vocabulary')
    all_text=med_text+anon_text
    encoded,lengths=_encode(all_text,weights)
    if tuple(encoded.shape)!=(20,1024): raise RuntimeError(f'RN50 output shape {tuple(encoded.shape)}')
    med=F.normalize(encoded[:10].double(),dim=-1).float()
    anon=F.normalize(encoded[10:].double(),dim=-1).float()
    raw=torch.cat([med,anon],0).double()
    U,S,Vh=torch.linalg.svd(raw,full_matrices=False)
    tol=float(max(raw.shape)*torch.finfo(raw.dtype).eps*float(S.max()))
    rank=int((S>tol).sum().item())
    if rank>20 or rank<1: raise RuntimeError(f'invalid row rank {rank}')
    B=Vh[:rank].T.contiguous()
    coords=raw@B
    anchors=torch.zeros((20,32),dtype=torch.float64); anchors[:,:rank]=coords
    gram_raw=raw@raw.T; gram_anchor=anchors@anchors.T
    err=float((gram_raw-gram_anchor).abs().max())
    if err>2e-10: raise RuntimeError(f'anchor Gram error {err}')
    anchors=anchors.float(); raw=raw.float(); B=B.float()
    code_root=Path(enc['clip_source_root'])
    source_files={str(weights):sha(weights),str(code_root/'clip.py'):sha(code_root/'clip.py'),str(code_root/'simple_tokenizer.py'):sha(code_root/'simple_tokenizer.py'),str(code_root/'bpe_simple_vocab_16e6.txt.gz'):sha(code_root/'bpe_simple_vocab_16e6.txt.gz')}
    payload={
      'medical_embeddings_raw':med, 'anonymous_embeddings_raw':anon,
      'embeddings_raw':raw, 'medical_anchors':anchors[:10], 'anonymous_anchors':anchors[10:],
      'anchors':anchors, 'row_space_basis':B, 'singular_values':S.float(), 'effective_rank':rank,
      'gram_max_abs_error':err, 'norms_raw':raw.norm(dim=-1), 'norms_anchor':anchors.norm(dim=-1),
      'medical_text':med_text, 'anonymous_text':anon_text,
      'medical_classes':torch.tensor(cfg['prompts']['medical_classes'],dtype=torch.long),
      'anonymous_classes':torch.tensor(cfg['prompts']['anonymous_classes'],dtype=torch.long),
      'token_lengths':lengths,
      'text_dim':1024, 'anchor_dim':32, 'encoding':'all 20 fixed prompts directly encoded once by frozen OpenAI CLIP RN50; common CPU float64 row-space SVD; zero padded',
      'text_encoder_name':enc['name'],'text_encoder_weights_path':str(weights),'text_encoder_weights_sha256':sha(weights),
      'text_encoder_source_sha256':source_files,'tokenizer_normalization':'ftfy' if 'ftfy' in sys.modules else 'ftfy ASCII identity fallback',
      'prompt_strings_sha256':hashlib.sha256(json.dumps(all_text,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
      'source_note':'All 20 vectors were re-encoded from the exact frozen RN50 weights. No prior text embedding bank was reused.'
    }
    out=HERE/'runtime/text_embeddings.pt'; out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out)
    print(json.dumps({'path':str(out),'sha256':sha(out),'shape':list(anchors.shape),'rank':rank,'gram_max_abs_error':err,'medical_text':med_text,'anonymous_text':anon_text,'token_lengths':lengths.tolist(),'rn50_sha256':sha(weights)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
