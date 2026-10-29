import copy,hashlib,json,pickle
import run

def digest(s):
 h=hashlib.sha256();h.update(pickle.dumps(s['python'],4));h.update(pickle.dumps(s['numpy'],4));h.update(s['cpu'].cpu().numpy().tobytes());
 for x in s['cuda']:h.update(x.cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 c=run.context();m=c['method'];t=c['torch'];x,y,folds,mapping=run.data();tr,te=folds[0]
 a=run.initialize_agent(1,1,x,y,tr,te,mapping)
 main0=run.rng_state();ap0=copy.deepcopy(a.apcr_rng)
 with m.apcr_rng_scope(a):
  _=t.rand((37,17),device='cuda')
 main1=run.rng_state();ap1=a.apcr_rng
 out={'base_rng_unchanged_by_apcr_init':digest(main0)==digest(run.rng_state()),'apcr_scope_preserves_main_rng':digest(main0)==digest(main1),'apcr_scope_advances_independent_rng':digest(ap0)!=digest(ap1),'main_rng_sha':digest(main1),'apcr_rng_before_sha':digest(ap0),'apcr_rng_after_sha':digest(ap1)}
 out['status']='PASS' if all(out[k] for k in ('base_rng_unchanged_by_apcr_init','apcr_scope_preserves_main_rng','apcr_scope_advances_independent_rng')) else 'FAIL'
 (run.RT/'rng_isolation_tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
