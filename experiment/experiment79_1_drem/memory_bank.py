"""DREM dynamic error/recovery memory state machine.

All records and state live inside one outer fold.  Stored tensors are detached
CPU snapshots.  The bank never receives test labels or test features.
"""
from __future__ import annotations
import copy, hashlib, json, math
from collections import deque
import torch

TYPE_ASD_TO_TD = "ASD_to_TD"
TYPE_TD_TO_ASD = "TD_to_ASD"
TYPES = (TYPE_ASD_TO_TD, TYPE_TD_TO_ASD)
BUCKETS = ("stable", "recent", "recovered")


def clone(x):
    return x.detach().cpu().float().clone()

def clone_obj(x):
    if torch.is_tensor(x): return clone(x)
    if isinstance(x, deque): return deque(x, maxlen=x.maxlen)
    return copy.deepcopy(x)

def typ_of(y, p):
    y, p = int(y), int(p)
    if y == 0 and p == 1: return TYPE_ASD_TO_TD
    if y == 1 and p == 0: return TYPE_TD_TO_ASD
    raise ValueError((y,p))

class DREMErrorMemory:
    def __init__(self, capacity_per_type=64, stable_capacity=16, recent_capacity=16,
                 recovered_capacity=32, top_k=8, site_cap=2,
                 stable_min_errors_last5=3, recovery_min_errors_last5=2,
                 recovery_correct_streak=3, ema_old=.9, ema_new=.1,
                 recovery_margin_gain=.05):
        self.capacity_per_type=int(capacity_per_type); self.capacities={"stable":int(stable_capacity),"recent":int(recent_capacity),"recovered":int(recovered_capacity)}
        assert sum(self.capacities.values()) == self.capacity_per_type
        self.top_k=int(top_k); self.site_cap=int(site_cap)
        self.stable_min=int(stable_min_errors_last5); self.recovery_min=int(recovery_min_errors_last5)
        self.recovery_streak=int(recovery_correct_streak); self.ema_old=float(ema_old); self.ema_new=float(ema_new); self.recovery_margin_gain=float(recovery_margin_gain)
        self.records={t:{b:{} for b in BUCKETS} for t in TYPES}
        self.states={}
        self.epoch=0
        self.counters={"error_observations":0,"writes":0,"updates":0,"moves":0,"evictions":0,"recovery_candidates":0,"recovery_confirmed":0,"recovery_rejected":0,"recovery_invalidated":0}
        self.history=[]

    def _record_copy(self, r): return {k:clone_obj(v) for k,v in r.items()}
    def snapshot(self):
        return {"capacity_per_type":self.capacity_per_type,"capacities":self.capacities,"top_k":self.top_k,"site_cap":self.site_cap,
          "stable_min":self.stable_min,"recovery_min":self.recovery_min,"recovery_streak":self.recovery_streak,
          "ema_old":self.ema_old,"ema_new":self.ema_new,"recovery_margin_gain":self.recovery_margin_gain,"epoch":self.epoch,
          "records":{t:{b:{int(pid):self._record_copy(r) for pid,r in self.records[t][b].items()} for b in BUCKETS} for t in TYPES},
          "states":{int(pid):self._record_copy(s) for pid,s in self.states.items()},"counters":copy.deepcopy(self.counters),"history":copy.deepcopy(self.history[-20:])}
    @classmethod
    def from_snapshot(cls,s):
        b=cls(s.get("capacity_per_type",64),s.get("capacities",{}).get("stable",16),s.get("capacities",{}).get("recent",16),s.get("capacities",{}).get("recovered",32),s.get("top_k",8),s.get("site_cap",2),s.get("stable_min",3),s.get("recovery_min",2),s.get("recovery_streak",3),s.get("ema_old",.9),s.get("ema_new",.1),s.get("recovery_margin_gain",.05))
        b.epoch=int(s.get("epoch",0)); b.counters=copy.deepcopy(s.get("counters",b.counters)); b.history=copy.deepcopy(s.get("history",[]))
        for t in TYPES:
            for bucket in BUCKETS:
                b.records[t][bucket]={int(pid):b._record_copy(r) for pid,r in s.get("records",{}).get(t,{}).get(bucket,{}).items()}
        b.states={int(pid):b._record_copy(st) for pid,st in s.get("states",{}).items()}
        return b
    def size(self,t=None,bucket=None):
        if t is None: return sum(len(self.records[x][b]) for x in TYPES for b in BUCKETS)
        if bucket is None: return sum(len(self.records[t][b]) for b in BUCKETS)
        return len(self.records[t][bucket])
    def counts(self):
        return {f"{t}:{b}":len(self.records[t][b]) for t in TYPES for b in BUCKETS}
    def _find(self,pid):
        for t in TYPES:
            for b in BUCKETS:
                if int(pid) in self.records[t][b]: return t,b,self.records[t][b][int(pid)]
        return None,None,None
    def _remove_pid(self,pid):
        n=0
        for t in TYPES:
            for b in BUCKETS:
                if int(pid) in self.records[t][b]: del self.records[t][b][int(pid)]; n+=1
        return n
    def _priority(self,r,b,epoch):
        margin=max(float(r.get("wrong_margin",0)),0)
        sev=1-math.exp(-margin)
        age=max(0,float(epoch-int(r.get("last_error_epoch",epoch))))
        rec=math.exp(-age/10)
        freq=min(1,float(r.get("error_count",0))/5)
        q=float(r.get("recovery_stability",0))
        if b=="stable": return .55*freq+.30*sev+.15*rec
        if b=="recent": return .60*rec+.30*sev+.10*freq
        return .50*q+.30*rec+.20*sev
    def _sort(self,r,b,epoch):
        return (-self._priority(r,b,epoch),-int(r.get("last_error_epoch",0)),int(r["patient_id"]))
    def _insert(self,t,b,r):
        self._remove_pid(r["patient_id"]); r["bucket"]=b; self.records[t][b][int(r["patient_id"])]=r
    def _trim(self,epoch):
        ev=0
        for t in TYPES:
            for b in BUCKETS:
                for r in self.records[t][b].values(): r["priority_score"]=float(self._priority(r,b,epoch))
                cap=self.capacities[b]; vals=sorted(self.records[t][b].values(),key=lambda r:self._sort(r,b,epoch))
                keep={int(r["patient_id"]) for r in vals[:cap]}
                for pid in list(self.records[t][b]):
                    if int(pid) not in keep: del self.records[t][b][pid]; ev+=1
        self.counters["evictions"]+=ev
        return ev
    def _state_default(self,pid,y,site):
        return {"patient_id":int(pid),"true_label":int(y),"site_id":int(site),"active_type":None,"wrong_predicted_label":None,
          "wrong_epoch":None,"last_error_epoch":None,"error_count":0,"recent_errors":deque([],maxlen=5),"correct_streak":0,
          "h_wrong":None,"h_retrieval":None,"mu_wrong":None,"wrong_margin":0.,"wrong_probability":0.,"cross_entropy":0.,
          "recovery_attempted":False,"recovered_active":False,"last_recovery":None}
    def _error_record(self,st,bucket):
        return {"patient_id":st["patient_id"],"true_label":st["true_label"],"wrong_predicted_label":st["wrong_predicted_label"],"error_type":st["active_type"],"bucket":bucket,
          "h_wrong":clone(st["h_wrong"]),"h_retrieval":clone(st["h_retrieval"]),"wrong_probability":float(st["wrong_probability"]),"wrong_margin":float(st["wrong_margin"]),"cross_entropy":float(st["cross_entropy"]),
          "wrong_epoch":int(st["wrong_epoch"]),"last_error_epoch":int(st["last_error_epoch"]),"error_count":int(st["error_count"]),"recent_errors":deque(st["recent_errors"],maxlen=5),"site_id":int(st["site_id"]),"priority_score":0.0}
    def _recovered_record(self,st,q):
        return {"patient_id":st["patient_id"],"true_label":st["true_label"],"wrong_predicted_label":st["wrong_predicted_label"],"error_type":st["active_type"],"bucket":"recovered",
          "h_wrong":clone(st["h_wrong"]),"h_retrieval":clone(st["h_wrong"]),"h_correct":clone(q["h_correct"]),"u_recovery":clone(q["u_recovery"]),
          "wrong_epoch":int(st["wrong_epoch"]),"confirmed_correct_epoch":int(q["confirmed_correct_epoch"]),"class_center_wrong":clone(st["mu_wrong"]),"class_center_correct":clone(q["mu_correct"]),
          "raw_recovery_change":clone(q["raw_change"]),"drift_corrected_recovery_change":clone(q["d_recovery"]),"same_head_margin_gain":float(q["margin_gain"]),"recovery_stability":float(q.get("recovery_stability",1.0)),
          "wrong_margin":float(st["wrong_margin"]),"last_error_epoch":int(st["last_error_epoch"]),"error_count":int(st["error_count"]),"site_id":int(st["site_id"]),"priority_score":0.0}
    def update_from_base_eval(self,eligible_ids,labels,cam144,logp,sites,epoch,class_centers,recovery_quality=None):
        labels=labels.detach().cpu().long(); cam=cam144.detach().cpu().float(); lp=logp.detach().cpu().float(); sites=sites.detach().cpu().long(); pred=lp.argmax(1); eligible=sorted(int(x) for x in eligible_ids)
        recovery_quality=recovery_quality or {}; writes=updates=moves=errors=confirmed=rejected=invalidated=0
        for pid in eligible:
            y=int(labels[pid]); p=int(pred[pid]); site=int(sites[pid]); st=self.states.get(pid)
            if st is None: st=self._state_default(pid,y,site); self.states[pid]=st
            # append this epoch's error state before checking recovery thresholds
            st["recent_errors"].append(1 if p!=y else 0)
            if p!=y:
                errors+=1; self.counters["error_observations"]+=1; typ=typ_of(y,p)
                if st["recovered_active"] or (st["active_type"] is not None and st["correct_streak"]>0):
                    if st["recovered_active"]: invalidated+=1; self.counters["recovery_invalidated"]+=1
                    self._remove_pid(pid); st["recovered_active"]=False; st["recovery_attempted"]=False; st["correct_streak"]=0; st["active_type"]=None
                if st["active_type"] != typ or st["wrong_epoch"] is None:
                    self._remove_pid(pid); st["active_type"]=typ; st["wrong_predicted_label"]=p; st["wrong_epoch"]=int(epoch); st["last_error_epoch"]=int(epoch); st["error_count"]=1; st["h_wrong"]=clone(cam[pid]); st["h_retrieval"]=clone(cam[pid]); st["mu_wrong"]=clone(class_centers[pid]); st["correct_streak"]=0; st["recovery_attempted"]=False; writes+=1
                else:
                    st["error_count"]+=1; st["last_error_epoch"]=int(epoch); st["h_retrieval"]=self.ema_old*st["h_retrieval"]+self.ema_new*clone(cam[pid]); st["correct_streak"]=0; updates+=1
                st["wrong_predicted_label"]=p; st["wrong_margin"]=max(0.,float((lp[pid,p]-lp[pid,y]).item())); st["wrong_probability"]=float(lp[pid,p].exp()); st["cross_entropy"]=float(-lp[pid,y]); st["true_label"]=y; st["site_id"]=site
                bucket="stable" if sum(st["recent_errors"])>=self.stable_min else "recent"
                rec=self._error_record(st,bucket); old_t,old_b,_=self._find(pid)
                if old_b!=bucket: moves+=1
                self._insert(typ,bucket,rec)
            else:
                if st["active_type"] is not None and not st["recovered_active"]:
                    st["correct_streak"]+=1
                    if st["correct_streak"]>=self.recovery_streak and sum(st["recent_errors"])>=self.recovery_min and not st["recovery_attempted"]:
                        self.counters["recovery_candidates"]+=1
                        q=recovery_quality.get(pid,{"accepted":False,"reason":"missing"})
                        st["recovery_attempted"]=True
                        if q.get("accepted",False):
                            rec=self._recovered_record(st,q); self._insert(st["active_type"],"recovered",rec); st["recovered_active"]=True; confirmed+=1; self.counters["recovery_confirmed"]+=1
                        else:
                            rejected+=1; self.counters["recovery_rejected"]+=1
                    elif st["active_type"] is not None:
                        t,b,_=self._find(pid)
                        if t is not None: self.records[t][b][pid]["correct_streak"]=int(st["correct_streak"])
        evict=self._trim(epoch); self.epoch=int(epoch)
        self.counters["writes"]+=writes; self.counters["updates"]+=updates; self.counters["moves"]+=moves
        event={"epoch":int(epoch),"eligible_count":len(eligible),"base_error_count":errors,"writes":writes,"updates":updates,"moves":moves,"evictions":evict,"recovery_candidates":self.counters["recovery_candidates"],"recovery_confirmed":self.counters["recovery_confirmed"],"recovery_rejected":self.counters["recovery_rejected"],"recovery_invalidated":invalidated,"counts":self.counts(),"event_hash":self.event_hash()}
        self.history.append(event); return event
    def event_hash(self):
        rows=[]
        for t in TYPES:
            for b in BUCKETS:
                for pid in sorted(self.records[t][b]):
                    r=self.records[t][b][pid]; row={k:v for k,v in r.items() if k not in ("h_wrong","h_retrieval","h_correct","u_recovery","raw_recovery_change","drift_corrected_recovery_change","class_center_wrong","class_center_correct")}
                    row["h_wrong_sha"]=hashlib.sha256(r["h_wrong"].numpy().tobytes()).hexdigest();
                    if "u_recovery" in r: row["u_sha"]=hashlib.sha256(r["u_recovery"].numpy().tobytes()).hexdigest()
                    rows.append((t,b,int(pid),row))
        payload=json.dumps({"rows":rows,"counters":self.counters,"epoch":self.epoch},sort_keys=True,default=str,separators=(",",":"))
        return hashlib.sha256(payload.encode()).hexdigest()
    def _all_candidates(self,t):
        return [(b,r) for b in BUCKETS for r in self.records[t][b].values()]
    def retrieval(self,query_ids,query_repr,sites,device,key_fn,*,type_swap=False,recency_off=False):
        out=[]; qids=[int(x) for x in query_ids]; sites_cpu=sites.detach().cpu().long()
        for row,pid in enumerate(qids):
            rowout=[]
            for requested in TYPES:
                actual=(TYPE_TD_TO_ASD if requested==TYPE_ASD_TO_TD else TYPE_ASD_TO_TD) if type_swap else requested
                cand=[(b,r) for b,r in self._all_candidates(actual) if int(r["patient_id"])!=pid]
                if not cand: rowout.append(None); continue
                feats=torch.stack([r["h_wrong"] for b,r in cand]); q=query_repr[row].detach().cpu().float()
                with torch.no_grad(): keys=torch.nn.functional.normalize(key_fn(feats.to(device)).detach().cpu().float(),dim=-1)
                sim=torch.nn.functional.normalize(q,dim=-1)@keys.T
                pri=torch.tensor([0.0 if recency_off else float(r.get("priority_score",0.)) for b,r in cand])
                order=sorted(range(len(cand)),key=lambda i:(-float(sim[i]),-float(pri[i]),int(cand[i][1]["patient_id"])))
                chosen=[]; site_count={}
                for i in order:
                    site=int(cand[i][1]["site_id"])
                    if site_count.get(site,0)>=self.site_cap: continue
                    chosen.append(i); site_count[site]=site_count.get(site,0)+1
                    if len(chosen)>=self.top_k: break
                if len(chosen)<min(self.top_k,len(order)):
                    for i in order:
                        if i not in chosen: chosen.append(i)
                        if len(chosen)>=self.top_k: break
                # bucket centers are computed before selection; key features remain unmodified.
                bybucket={}
                for b,r in cand:
                    bybucket.setdefault(b,[]).append(r["h_wrong"])
                centers={b:torch.stack(v).mean(0) for b,v in bybucket.items()}
                rowout.append({"requested_type":requested,"actual_type":actual,"patient_ids":[int(cand[i][1]["patient_id"]) for i in chosen],"buckets":[cand[i][0] for i in chosen],"key_features":feats[torch.tensor(chosen)].to(device),"value_features":feats[torch.tensor(chosen)].to(device),"directions":torch.stack([cand[i][1].get("u_recovery",torch.zeros(144)) for i in chosen]).to(device),"centers":{b:c.to(device) for b,c in centers.items()},"bucket_features":{b:torch.stack(v).to(device) for b,v in bybucket.items()},"priority":pri[torch.tensor(chosen)].to(device),"similarities":sim[torch.tensor(chosen)].to(device)})
            out.append(rowout)
        return out
    def stats(self):
        return {"counts":self.counts(),"size_total":self.size(),"counters":copy.deepcopy(self.counters),"epoch":self.epoch}

    def pending_recovery_candidates(self, eligible_ids, labels, pred):
        out=[]
        for pid in sorted(int(x) for x in eligible_ids):
            st=self.states.get(pid)
            if st is None or st.get('active_type') is None or st.get('recovered_active') or st.get('recovery_attempted'): continue
            if int(pred[pid]) == int(labels[pid]) and int(st.get('correct_streak',0)) == self.recovery_streak-1 and sum(st.get('recent_errors',())) >= self.recovery_min:
                out.append(pid)
        return out
