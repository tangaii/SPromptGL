"""Independent paired metrics and handoff publisher for Experiment 82.4.6.4."""
from __future__ import annotations
import csv,fcntl,hashlib,io,os
from pathlib import Path
import numpy as np
from core import CONFIG,EXPOSED,ADDED,FOLDS,ROOT,RESULTS,MANIFEST,formats,f,h
from independent_464 import METRICS
DARG="E74.4-G2 DARG"; BASE="E82.4.6-G2"; P2="E82.4.6.2"; P3="E82.4.6.3"; G1="E82.4.6.4-G1"; G2="E82.4.6.4-G2"

def metric(x):
 try:
  if "probability" in x:
   return formats.score(x["probability"],x["prediction"],x["truth"])
  return formats.score(x["full"]["probability"],x["full"]["prediction"],x["truth"])
 except RuntimeError as exc:
  if "Both ASD and TD" not in str(exc):
   raise
  return {key: float("nan") for key in METRICS}
def mk(rows,pre):
 y=np.array([int(r["true_label"]) for r in rows]); ids=np.array([int(r["patient_id_1based"]) for r in rows])
 p0=np.array([float(r[pre]) for r in rows]); p=np.stack([p0,1-p0],1); pred=np.array([int(r[pre.replace("p_asd","prediction")]) for r in rows])
 z=np.log(np.clip(p,1e-15,1)); item={"full":{"probability":p,"prediction":pred,"logits":z},"off":{"probability":p,"prediction":pred,"logits":z},"anonymous":{"probability":p,"prediction":pred,"logits":z},"truth":y,"subjects":ids,"best_epoch":0,"best_sha":"archived","best_path":"archived"}
 item["residual"]={k:np.zeros((len(y),144)) for k in ("delta","delta_parallel","delta_perp")}; return item

def historical():
 p2=list(csv.DictReader((Path("/model/spromptgl/experiment/experiment82_4_6_2_g2_full_path_extension/results/patient_level.csv")).open(newline="")))
 p3=list(csv.DictReader((Path(CONFIG["prior_darg_only_directory"])/"results/patient_level.csv").open(newline="")))
 out={DARG:{},BASE:{},P2:{},P3:{}}
 for fold in range(1,11):
  a=[r for r in p2 if int(r["fold"])==fold]; b=[r for r in p3 if int(r["fold"])==fold]
  out[DARG][fold]=mk(a,"p_asd_darg"); out[BASE][fold]=mk(a,"p_asd_original_g2"); out[P2][fold]=mk(a,"p_asd_current_full"); out[P3][fold]=mk(b,"p_asd_full")
 return out

def pair(a,b,sel,name,fold,scope,start="full",finish="full"):
 y=a["truth"][sel]; old=a[start]["prediction"][sel]; new=b[finish]["prediction"][sel]; wrong=old!=y; right=~wrong; fix=wrong&(new==y); harm=right&(new!=y)
 gm=(1-2*y)*(b[finish]["logits"][sel,0]-b[finish]["logits"][sel,1]-a[start]["logits"][sel,0]+a[start]["logits"][sel,1])
 x=metric({"probability":a[start]["probability"][sel],"prediction":old,"truth":y}); z=metric({"probability":b[finish]["probability"][sel],"prediction":new,"truth":y})
 return {"comparison":name,"fold":fold,"scope":scope,"n":int(sel.sum()),"corrected":int(fix.sum()),"harmed":int(harm.sum()),"net":int(fix.sum()-harm.sum()),"asd_corrected":int((fix&(y==0)).sum()),"asd_harmed":int((harm&(y==0)).sum()),"td_corrected":int((fix&(y==1)).sum()),"td_harmed":int((harm&(y==1)).sum()),"wrong_positive_margin_gain":int((wrong&(gm>0)).sum()),"wrong_negative_margin_gain":int((wrong&(gm<0)).sum()),"right_positive_margin_gain":int((right&(gm>0)).sum()),"right_negative_margin_gain":int((right&(gm<0)).sum()),"wrong_margin_mean_change":float(gm[wrong].mean()) if wrong.any() else "","right_margin_mean_change":float(gm[right].mean()) if right.any() else "",**{k+"_before":x[k] for k in METRICS},**{k+"_after":z[k] for k in METRICS},**{"delta_"+k+"_pp":100*(z[k]-x[k]) for k in METRICS}}

def masks(fold,ids):
 e=np.isin(ids,np.array([i+1 for i in EXPOSED.get(fold,())])); n=np.isin(ids,np.array([i+1 for i in ADDED.get(fold,())]))
 return {"all":np.ones(len(ids),bool),"new":n,"historical":e&~n,"never":~e}

def csvsave(name,rows):
 s=io.StringIO(newline=""); w=csv.DictWriter(s,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows);formats.commit(RESULTS/name,s.getvalue().encode())

def jsonable(value):
 if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
 if hasattr(value,"tolist"): return jsonable(value.tolist())
 if hasattr(value,"item"): return value.item()
 return value

def publish(manifest,hist,prev,current,checks):
 data=historical(); data[P2]=hist;data[P3]=prev
 for legacy in (P2,P3):
  for fold in data[legacy]:
   if "anonymous" not in data[legacy][fold]: data[legacy][fold]["anonymous"]=data[legacy][fold]["full"]
 archived_for_current=historical()[P2]
 data[G1]=dict(archived_for_current);data[G2]=dict(archived_for_current)
 for fold in FOLDS: data[G1][fold]=current[("G1",fold)];data[G2][fold]=current[("G2",fold)]
 methods=[(DARG,DARG),(BASE,BASE),(P2,P2),(P3,P3),(G1,G1),(G2,G2)]
 scores=[]
 for label,key in methods:
  for fold in range(1,11):
   item=data[key][fold]; scores.append({"method":label,"fold":fold,"scope":"all","best_epoch":item["best_epoch"],"best_path":item["best_path"],"best_sha":item.get("best_sha",""),**metric(item)})
 pairs=[]
 pair_specs=[(P3,G1,"E82.4.6.3 → G1","full","full"),(P2,G2,"E82.4.6.2 → G2","full","full"),(G1,G2,"G1 → G2","full","full"),(G1,G1,"G1 Residual-off → Full","off","full"),(G2,G2,"G2 Residual-off → Full","off","full"),(G1,G1,"G1 Anonymous → Medical","anonymous","full"),(G2,G2,"G2 Anonymous → Medical","anonymous","full")]
 for fold in FOLDS:
  ids=data[P2][fold]["subjects"]
  for scope,sel in masks(fold,ids).items():
   for a,b,name,st,fi in pair_specs: pairs.append(pair(data[a][fold],data[b][fold],sel,name,fold,scope,st,fi))
 for a,b,name,st,fi in pair_specs:
  for scope in ("new","historical","never","all"):
   parts=[]; ys=[]
   for fold in FOLDS:
    ids=data[P2][fold]["subjects"]; sel=masks(fold,ids)[scope]; ys.append(data[a][fold]["truth"][sel]); parts.append((data[a][fold],data[b][fold],sel))
   y=np.concatenate(ys)
   def pooled(which,phase):
    def source(part): return part[0] if which == 0 else part[1]
    return {"truth":y,"probability":np.concatenate([source(part)[phase]["probability"][part[2]] for part in parts]),"prediction":np.concatenate([source(part)[phase]["prediction"][part[2]] for part in parts]),"logits":np.concatenate([source(part)[phase]["logits"][part[2]] for part in parts])}
   aa={"truth":y,st:pooled(0,st)};bb={"truth":y,fi:pooled(1,fi)}
   pairs.append(pair(aa,bb,np.ones(len(y),bool),name,"pooled",scope,st,fi))
 patients=[]
 for fold in range(1,11):
  ids=data[P2][fold]["subjects"]; e=set(i+1 for i in EXPOSED.get(fold,())); n=set(i+1 for i in ADDED.get(fold,()))
  for j,pid0 in enumerate(ids):
   pid=int(pid0); y=int(data[P2][fold]["truth"][j]); r={"fold":fold,"patient_id_1based":pid,"patient_index_0based":pid-1,"true_label":y,"cohort":"new" if pid in n else "historical_overlap" if pid in e else "never_added","in_current_training":pid in e,"g1_residual_masked":pid in e,"g2_residual_allowed":pid in e}
   for label,key,phase in (("e82_4_6_2",P2,"full"),("e82_4_6_3",P3,"full"),("g1_off",G1,"off"),("g1",G1,"full"),("g1_anonymous",G1,"anonymous"),("g2_off",G2,"off"),("g2",G2,"full"),("g2_anonymous",G2,"anonymous")):
    it=data[key][fold]; r["prediction_"+label]=int(it[phase]["prediction"][j]);r["p_asd_"+label]=float(it[phase]["probability"][j,0]);r["true_margin_"+label]=float((1-2*y)*(it[phase]["logits"][j,0]-it[phase]["logits"][j,1]))
   for label,key in (("g1",G1),("g2",G2)):
    for comp in ("delta","delta_parallel","delta_perp"): r[label+"_"+comp+"_l2"]=float(np.linalg.norm(data[key][fold]["residual"][comp][j]))
   r["g1_direct_corrected"]=r["prediction_g1_off"]!=y and r["prediction_g1"]==y;r["g1_direct_harmed"]=r["prediction_g1_off"]==y and r["prediction_g1"]!=y;r["g2_direct_corrected"]=r["prediction_g2_off"]!=y and r["prediction_g2"]==y;r["g2_direct_harmed"]=r["prediction_g2_off"]==y and r["prediction_g2"]!=y;patients.append(r)
 ten={}
 for label in [x[0] for x in methods]:
  rr=[x for x in scores if x["method"]==label];ten[label]={k:{"mean":float(np.mean([x[k] for x in rr])),"population_sd":float(np.std([x[k] for x in rr]))} for k in METRICS}
 eight={label:{k:{"mean":float(np.mean([x[k] for x in scores if x["method"]==label and x["fold"] not in FOLDS])),"population_sd":float(np.std([x[k] for x in scores if x["method"]==label and x["fold"] not in FOLDS]))} for k in METRICS} for label in ten}
 def ptxt(x): return f"{x['comparison']} {x['scope']}: n={x['n']}, 纠正={x['corrected']}, 误伤={x['harmed']}, 净={x['net']:+d}, ΔACC={x['delta_acc_pp']:+.4f}pp, ΔBA={x['delta_hard_auc_pp']:+.4f}pp"
 keypairs=[x for x in pairs if x["scope"] in ("new","historical","never","all")]
 report=["# Experiment 82.4.6.4 — Cumulative Overlap DARG-only versus Full Spectral-Prompt Supervision","COMPLETE_AUDITED","本轮比较相同扩充 DARG 成员下累计重叠患者只进入 DARG，或同时进入 SCMR 谱/Prompt 监督与 Gaussian 统计的影响。结果属于有意训练—测试标签重叠诊断，不代表独立泛化。","## 十折指标（均值 ± population SD，单位 %）",formats.metrics_table(ten),"## 排除两折的八折",formats.metrics_table(eight),"## 两折逐任务指标",formats.table(("方法","Fold","ACC","BA","Prob-AUC","SEN","SPE","ASD-F1"),[(x["method"],x["fold"],*(f"{100*x[k]:.4f}" for k in METRICS)) for x in scores if x["fold"] in FOLDS]),"## 患者级配对",*([ptxt(x) for x in keypairs]),"## 机制","G1 屏蔽累计重叠患者的 SCMR 残差。G2 允许累计重叠患者进入新增监督、Gaussian 统计和残差。固定权重 Residual-off → Full 以及匿名 Query → 医学 Query 均已重新计算。","## 审计","四个正式任务 Best、Latest 和下一步训练/评估均独立重载。逐 epoch RNG、历史 Dropout、关系对、成员索引、Gaussian OOF、EMA、残差限幅和受保护文件 SHA 已核验。","## 限制","单 seed、val=test 选模、CV 前 RFE/预处理、871 节点转导图和测试标签暴露限制解释。原始论文与历史 PLNNoPEN 的区别按 E82.4.6-G2 报告记录。","[SPromptGL 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)"]
 csvsave("fold_metrics.csv",scores);csvsave("paired_comparisons.csv",pairs);csvsave("patient_level.csv",patients);h.write_json(RESULTS/"membership_audit.json",jsonable(checks));formats.commit(RESULTS/"report.md","\n\n".join(report).encode())
 hp=Path("/model/spromptgl/experiment/handoff.md"); block=("\\n\\n## Experiment 82.4.6.4\\nCOMPLETE_AUDITED · Cumulative Overlap DARG-only versus Full Spectral-Prompt Supervision\\n权威目录 "+str(ROOT)+"。G1 和 G2 各完成 Fold 2、Fold 3，四个正式任务通过独立审计。十折指标和新增 7 人、历史 10 人、未加入 157 人的逐患者比较已写入结果文件。G1 屏蔽累计重叠残差，G2 允许累计重叠进入 SCMR。该结果属于训练—测试标签暴露诊断，不能解释为独立泛化或创新模块 GO。\\n").encode()
 with hp.open("a+b") as s:
  fcntl.flock(s,fcntl.LOCK_EX);s.seek(0);old=s.read();prefix=old
  if b"## Experiment 82.4.6.4\n" not in old:s.write(block);s.flush();os.fsync(s.fileno())
  read=hp.read_bytes();h.require(read[:len(prefix)]==prefix and read[len(prefix):len(prefix)+len(block)]==block,"Handoff readback failed")
 hand={"path":str(hp),"prefix_sha256":hashlib.sha256(prefix).hexdigest(),"prefix_bytes":len(prefix),"append_sha256":hashlib.sha256(block).hexdigest(),"append_bytes":len(block),"file_sha256_at_readback":hashlib.sha256(read).hexdigest(),"historical_prefix_exact":True,"new_section_exact":True}
 safe_checks=jsonable(checks)
 audit={"status":"COMPLETE_AUDITED","experiment":CONFIG["experiment"],"best_reloaded":4,"latest_reloaded":4,"latest_next_exact":4,"tasks":safe_checks,"test_label_exposure":True,"eligible_for_independent_generalization_or_innovation_go":False,"handoff":hand}
 h.write_json(RESULTS/"audit_status.json",audit);manifest.update(status="COMPLETE_AUDITED",ten_fold=ten,unchanged_eight=eight,audit=safe_checks,handoff=hand,result_files={str(x.relative_to(ROOT)):h.sha_file(x) for x in RESULTS.iterdir() if x.is_file()});h.write_json(MANIFEST,manifest);print("COMPLETE_AUDITED | four tasks | verified handoff",flush=True)
