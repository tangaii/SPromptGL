#!/usr/bin/env python3
from pathlib import Path
import fcntl,hashlib,json,os

HERE=Path(__file__).resolve().parent
TARGET=Path('/data/adodas/SPromptGL-main/handoff.md')
MARK='<!-- EXPERIMENT48_5_ALPHA_ABLATION -->'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    entry=(HERE/'handoff_entry.md').read_bytes()
    manifest=json.loads((HERE/'manifest.json').read_text())
    expected=manifest.get('handoff_before_sha256')
    if not expected:raise RuntimeError('missing pre-append handoff SHA')
    try:
        with TARGET.open('r+b') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0);before=f.read()
            if sha(before)!=expected:raise RuntimeError('handoff changed after snapshot')
            if before.count(MARK.encode()):raise RuntimeError('marker already exists')
            f.seek(0,os.SEEK_END);f.write(b'\n\n'+entry);f.flush();os.fsync(f.fileno())
            f.seek(0);after=f.read()
            if not after.startswith(before) or after.count(MARK.encode())!=1:raise RuntimeError('post-append verification failed')
        manifest['handoff_status']='APPENDED_AND_VERIFIED';manifest['handoff_after_sha256']=sha(TARGET.read_bytes())
    except Exception as exc:
        pending=HERE/'.runtime/handoff_pending.md';pending.write_bytes(entry)
        manifest['handoff_status']='HANDOFF_PENDING';manifest['handoff_error']=str(exc);manifest['handoff_pending_path']=str(pending)
    tmp=HERE/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,HERE/'manifest.json')
    print(json.dumps({k:manifest.get(k) for k in ('handoff_status','handoff_error','handoff_pending_path','handoff_after_sha256')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
