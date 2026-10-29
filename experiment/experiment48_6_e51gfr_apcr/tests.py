#!/usr/bin/env python3
"""Fast interface and alpha checks; full E51 parity is run by run.py preflight."""
import json
import run

def main():
    c=run.context();t=c["torch"];m=c["method"]
    for trainable in (False,True):
        ap=m.forked_apcr(t,c["utils"],1,alpha_init=.0022,alpha_trainable=trainable,seed_base=481000)
        assert abs(float(ap.alpha().detach())-.0022)<1e-7
        assert ap.alpha_raw.requires_grad is trainable
    x,y,folds,mapping=run.data()
    assert x.shape==(871,320) and y.shape==(871,) and [len(mapping[k]) for k in ("PHENO","ANAT","FUNC","Correlation")]==[48,6,10,256]
    out=run.preflight()
    assert out["status"]=="PASS" and out["folds_passed"]==10 and out["srd_sca_cra_active"]
    print(json.dumps({"status":"PASS","alpha_init":0.0022,"legacy_fixed_gfr_parity_folds":10,"details":out},indent=2))

if __name__=="__main__":main()
