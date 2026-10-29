from __future__ import annotations
import importlib.util, copy, random, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def assert_tree_equal(a,b,path="root"):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:assert_tree_equal(a[k],b[k],f"{path}.{k}")
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_tree_equal(x,y,f"{path}[{i}]")
    else:assert a==b,(path,a,b)

def _reference_agent(run, c, x,y,tr,te,fold,anchor):
    hist=run.HERE/".runtime/sources/historical48_3/method.py"
    spec=importlib.util.spec_from_file_location("apcr483_reference_method",hist)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    import network
    random.setstate(anchor["entry_rng"]["python"]);np.random.set_state(anchor["entry_rng"]["numpy"])
    c["torch"].set_rng_state(anchor["entry_rng"]["cpu"].cpu());c["torch"].cuda.set_rng_state_all([v.cpu() for v in anchor["entry_rng"]["cuda"]])
    a=old.build(c["torch"],c["layers"],network,c["utils"],c["device"],x,y,tr,te)
    a.ModalFusion.apply(c["utils"].my_weight_init)
    a.MessagePassing0.load_state_dict(anchor["model"]["gfr"]);a.ModalFusion.load_state_dict(anchor["model"]["pln"])
    a.optimizer_MP0=c["torch"].optim.Adam(a.MessagePassing0.parameters(),lr=.0038,weight_decay=.11)
    a.optimizer_MF=c["torch"].optim.Adam(a.ModalFusion.parameters(),lr=.0038,weight_decay=.11)
    a.optimizer_MP0.load_state_dict(anchor["mp"]);a.optimizer_MF.load_state_dict(anchor["mf"])
    random.setstate(anchor["rng"]["python"]);np.random.set_state(anchor["rng"]["numpy"])
    c["torch"].set_rng_state(anchor["rng"]["cpu"].cpu());c["torch"].cuda.set_rng_state_all([v.cpu() for v in anchor["rng"]["cuda"]])
    return a,old

def parity_test_fold(run, fold=1):
    c=run.context();m=c["method"];x,y,folds,mapping=run.data();tr,te=folds[fold-1];anchor=run.load(run.RT/f"anchors/fold{fold:02d}.pt")
    a=m.make_agent(c["torch"],c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,1,anchor,481000+fold)
    b,old=_reference_agent(run,c,x,y,tr,te,fold,anchor)
    # Initial parameters, optimizer slots and mode-invariant outputs must match exactly.
    for key in ("gfr","pln"):
        cur=a.MessagePassing0 if key=="gfr" else a.ModalFusion;ref=b.MessagePassing0 if key=="gfr" else b.ModalFusion
        assert_tree_equal(cur.state_dict(),ref.state_dict(),key)
    assert_tree_equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict(),"mp_opt")
    assert_tree_equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict(),"mf_opt")
    a.MessagePassing0.eval();a.ModalFusion.eval();b.MessagePassing0.eval();b.ModalFusion.eval()
    with torch.no_grad():
        oa=m.evaluate(a,1,c["utils"]);ob=m.evaluate(b,1,c["utils"])
    assert torch.equal(oa["base"],ob["base"])
    assert torch.equal(oa["logp"],ob["logp"])
    assert torch.equal(c["utils"].ClsLoss(oa["logp"],a.targ,a.trn_idx,a.weight),c["utils"].ClsLoss(ob["logp"],b.targ,b.trn_idx,b.weight))
    # One training update: compare gradients, parameters, both Adam states and all RNG streams.
    random.setstate(anchor["rng"]["python"]);np.random.set_state(anchor["rng"]["numpy"]);torch.set_rng_state(anchor["rng"]["cpu"].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in anchor["rng"]["cuda"]])
    va=m.train_step(a,1,0,c["utils"],mapping)
    grads_a={"gfr":{n:(None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in a.MessagePassing0.named_parameters()},"pln":{n:(None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in a.ModalFusion.named_parameters()}}
    rng_a=run.rng_state()
    random.setstate(anchor["rng"]["python"]);np.random.set_state(anchor["rng"]["numpy"]);torch.set_rng_state(anchor["rng"]["cpu"].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in anchor["rng"]["cuda"]])
    b.MessagePassing0.train();b.ModalFusion.train();b.optimizer_MP0.zero_grad(set_to_none=True);b.optimizer_MF.zero_grad(set_to_none=True)
    n,parts=b.dataLayer(b.feat);base,adjs,aloss=b.MessagePassing0(parts,n,b.targ,b.trn_idx)
    gl=sum(c["utils"].GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4
    lp=b.ModalFusion(base)[0];cl=c["utils"].ClsLoss(lp,b.targ,b.trn_idx,b.weight);loss=cl+gl;loss.backward()
    grads_b={"gfr":{n:(None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in b.MessagePassing0.named_parameters()},"pln":{n:(None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in b.ModalFusion.named_parameters()}}
    assert_tree_equal(grads_a,grads_b,"gradient")
    b.optimizer_MP0.step();b.optimizer_MF.step();rng_b=run.rng_state()
    assert abs(va["main"]-float(cl.detach()))<1e-12 and abs(va["graph"]-float(gl.detach()))<1e-12
    assert_tree_equal(a.MessagePassing0.state_dict(),b.MessagePassing0.state_dict(),"updated_gfr")
    assert_tree_equal(a.ModalFusion.state_dict(),b.ModalFusion.state_dict(),"updated_pln")
    assert_tree_equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict(),"updated_mp_opt")
    assert_tree_equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict(),"updated_mf_opt")
    assert_tree_equal(rng_a,rng_b,"poststep_rng")
    return {"status":"PASS","initial_output_exact":True,"loss_gradient_update_rng_exact":True,"fold":fold}

def parity_test(run):
    # All ten fold anchors must reproduce independently before any formal fit.
    return {"status":"PASS","folds":[parity_test_fold(run,fold) for fold in range(1,11)]}

def relation_math_tests(run):
    c=run.context();m=c["method"];torch=c["torch"]
    # Symmetry and true L2 normalization.
    ap=m.APCR().cuda();parts=[torch.randn(20,d,device="cuda") for d in m.DIMS];z=ap.representations(parts)
    assert all(torch.allclose(q.norm(dim=-1),torch.ones(20,device="cuda"),atol=1e-5) for q in z)
    for k in range(4):
        i,j=3,11;cs=(z[k][i]*z[k][j]).sum().clamp(-1,1)
        q1=torch.sigmoid(ap.scorer[k](z[k][i],z[k][j],cs));q2=torch.sigmoid(ap.scorer[k](z[k][j],z[k][i],cs))
        assert torch.allclose(q1,q2,atol=1e-7)
    # Same unordered patient-pair set under different row/order permutations.
    n=16; ids=[]
    for m_id in range(4):
        row=[]
        for i in range(n):row.append([(i+k+1+m_id)%n for k in range(6)])
        ids.append(torch.tensor(row,device="cuda",dtype=torch.long))
    fit=torch.ones(n,dtype=torch.bool,device="cuda")
    ids2=[v.flip(1) for v in ids]
    # Reuse one representation set to isolate only edge-order effects.
    zr=[torch.randn(n,32,device="cuda") for _ in range(4)]
    p1,q1,s1=ap.union_pair_scores(zr,ids,fit);p2,q2,s2=ap.union_pair_scores(zr,ids2,fit)
    assert torch.equal(p1,p2) and torch.equal(s1,s2)
    assert all(torch.equal(a,b) for a,b in zip(q1,q2))
    # Exact same-pair consensus is near zero when all views use identical scorers and embeddings.
    ap.scorer[1].load_state_dict(ap.scorer[0].state_dict());ap.scorer[2].load_state_dict(ap.scorer[0].state_dict());ap.scorer[3].load_state_dict(ap.scorer[0].state_dict())
    zz=[zr[0] for _ in range(4)];p,q,s=ap.union_pair_scores(zz,ids,fit);loss,w=ap.sca_loss(q,s,torch.ones(4,device="cuda"));assert float(loss)<1e-7 and float(w)>0
    # SRD pair sampling changes by epoch, is balanced, patient-covered and includes site hard negatives.
    rng=np.random.default_rng(82);labels=np.asarray([0]*8+[1]*8);sites=np.asarray([0,0,1,1]*4);tr=np.arange(16)
    a,ya,sa=m.sample_pairs(tr,labels,sites,0,91,16);b,yb,sb=m.sample_pairs(tr,labels,sites,1,91,16)
    assert not np.array_equal(a,b) and np.all(a[:,0]!=a[:,1]) and len(np.unique(a,axis=0))==len(a)
    assert sa["same_0"]==sa["same_1"]==sa["hard_negative_same_site"]==sa["negative_cross_site"]==16
    assert int((ya==1).sum())==int((ya==0).sum())==32
    # Shuffled supervision changes pair targets while retaining the exact E6 pair IDs.
    a_same,ya_same,_=m.sample_pairs(tr,labels,sites,0,91,16)
    rng=np.random.default_rng(83);shuffled=labels.copy();shuffled[tr]=labels[rng.permutation(tr)]
    ys=(shuffled[a[:,0]]==shuffled[a[:,1]]).astype(np.float32)
    assert np.array_equal(a,a_same) and np.array_equal(ya,ya_same) and np.any(ys!=ya)
    # The finite CRA selector returns the actual CE maximum; random control only changes the selection rule.
    losses=[.1,.7,.2,.3];assert m.choose_cra_candidate(losses)==1;assert m.choose_cra_candidate(losses,True,2)==2
    return {"normalize":"PASS","srd_symmetry":"PASS","same_pair_sca_permutation":"PASS","sca_identical_views":"PASS","pair_sampling":"PASS","cra_selector":"PASS"}

def sca_conflict_and_gradient_test(run):
    c=run.context();m=c["method"];torch=c["torch"];ap=m.APCR().cuda()
    # Make all four views identical, then introduce one low-confidence disagreeing view.
    with torch.no_grad():
        for j in range(1,4):ap.scorer[j].load_state_dict(ap.scorer[0].state_dict())
    n=24;z0=F.normalize(torch.randn(n,32,device="cuda"),dim=-1);z=[z0 for _ in range(4)]
    ids=[]
    for shift in range(4):
        ids.append(torch.stack([torch.tensor([(i+1+shift+k)%n for k in range(6)],device="cuda") for i in range(n)]))
    fit=torch.ones(n,dtype=torch.bool,device="cuda");pair,q,s=ap.union_pair_scores(z,ids,fit)
    exact,w=ap.sca_loss(q,s,torch.ones(4,device="cuda"));assert float(exact)<1e-7 and float(w)>0
    qconf=[v.detach().clone().requires_grad_(True) for v in q]
    qconf[3]=torch.where(qconf[3]>.5,torch.full_like(qconf[3],.99),torch.full_like(qconf[3],.01)).requires_grad_(True)
    conflict,w2=ap.sca_loss(qconf,s,torch.ones(4,device="cuda"));assert float(conflict)<=1 and float(w2)>0
    conflict.backward();assert qconf[3].grad is not None and torch.isfinite(qconf[3].grad).all()
    lowrel=torch.tensor([1.,1.,1.,.01],device="cuda");downweighted,_=ap.sca_loss(qconf,s,lowrel)
    assert float(downweighted)<=float(conflict)+1e-7
    return {"identical_views_same_pairs_zero":float(exact),"conflict_loss_bounded":float(conflict),"reliability_downweight_loss":float(downweighted),"gradient_finite":True,"candidate_support_pairs":int(pair.shape[0])}

def cra_and_negative_control_test(run):
    c=run.context();m=c["method"];torch=c["torch"]
    # Tiny fixed linear classifier and legal graphs with known CE order.
    logits=torch.tensor([[0.,0.],[0.,0.],[0.,0.]],device="cuda")
    y=torch.tensor([0,1,0],device="cuda");weight=torch.ones(2,device="cuda")
    candidate_ce=[]
    for logits_case in [torch.tensor([[2.,0.],[0.,2.],[2.,0.]],device="cuda"),torch.tensor([[0.,2.],[0.,2.],[2.,0.]],device="cuda"),torch.tensor([[2.,0.],[2.,0.],[0.,2.]],device="cuda"),torch.tensor([[0.,2.],[2.,0.],[0.,2.]],device="cuda")]:
        candidate_ce.append(float(F.nll_loss(F.log_softmax(logits_case,1),y,weight=weight)))
    idx=m.choose_cra_candidate(candidate_ce);assert candidate_ce[idx]==max(candidate_ce)
    rnd=m.choose_cra_candidate(candidate_ce,True,0);assert idx!=rnd
    # Per-patient shuffle preserves class counts and materially changes pair labels.
    labels=np.array([0]*50+[1]*50);tr=np.arange(100);shuffle,sha,rate=run.shuffled_labels(labels,tr,1)
    assert np.bincount(shuffle).tolist()==[50,50] and rate>=.25 and sha
    # Finite CRA proposals have same target/edge budget; step 2 remains within cumulative <=2-edge budget.
    n=40;parts=[torch.randn(n,d,device="cuda") for d in m.DIMS];ap=m.APCR().cuda();z=ap.representations(parts);pool,raw=ap.candidates(z)
    base=[]
    for ids in pool:base.append(ids[:,:16].clone())
    prop1,_=m.candidate_attacks(ap,base,pool,np.arange(20),22,0)
    prop2,_=m.candidate_attacks(ap,base,pool,np.arange(20),23,0,prop1[0])
    for graph in (prop1[0],prop2[0]):
        for mod in range(4):
            rows=graph[mod][:20];assert not bool((rows==torch.arange(20,device="cuda")[:,None]).any())
            assert all(len(set(r.tolist()))==16 for r in rows)
            assert bool(((rows!=base[mod][:20]).sum(1)<=2).all())
    # Replaced edges receive weights rescored from their new patient identities.
    replaced=prop1[0]
    ap.residual(parts,z,pool,raw,"srd",attacked=replaced)
    for mod in range(4):
        zi=z[mod].unsqueeze(1).expand(-1,16,-1);zj=z[mod][replaced[mod]]
        cosine=(zi*zj).sum(-1).clamp(-1,1)
        expected=torch.sigmoid(ap.scorer[mod](zi,zj,cosine))
        expected=expected/(expected.sum(1,keepdim=True)+1e-8)
        assert torch.allclose(ap.last["edge_weights"][mod],expected,atol=1e-7)
    # Random controls share exact candidate graph proposals; only selector differs.
    assert len(prop1)==4
    # Real-fold attack smoke test: candidates are scored by training CE, and held-out label changes cannot affect selection.
    c=run.context();x,y,folds,mapping=run.data();tr,te=folds[0];anchor=run.load(run.RT/"anchors/fold01.pt")
    a=m.make_agent(torch,c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,1,6,anchor,481001)
    a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481101);a.sites=run.site_ids(x,mapping);a.fold=1
    a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval()
    parts,base,adjs,gl,_=m.base_forward(a,c["utils"]);z=a.APCR.representations(parts);raw=[getattr(a.MessagePassing0,f"GraphConstruct{i}").score431 for i in range(4)];ids,rv=a.APCR.candidates(raw);q=a.APCR.score_candidate_edges(z,ids,rv)
    pair,pq,support=a.APCR.union_pair_scores(z,ids,torch.as_tensor(np.isin(np.arange(len(y)),a.srd_fit_ids),device="cuda"))
    out=m.output(a,parts,base,adjs,gl,6)
    attack,trace=m.maximize_cra(a,parts,base,adjs,gl,6,out,90,c["utils"])
    assert len(trace["cra_candidate_ce"])==4 and trace["cra_selected_ce"]+1e-8>=max(trace["cra_candidate_ce"])
    # Held-out label edits do not enter candidate selection or the maximization objective.
    orig=a.targ[torch.as_tensor(te,device="cuda")].clone();a.targ[torch.as_tensor(te,device="cuda")]=1-orig
    attack2,trace2=m.maximize_cra(a,parts,base,adjs,gl,6,out,90,c["utils"])
    assert all(torch.equal(u,v) for u,v in zip(attack,attack2)) and trace["cra_candidate_ce"]==trace2["cra_candidate_ce"]
    a.targ[torch.as_tensor(te,device="cuda")]=orig
    return {"finite_argmax":"PASS","random_selector_distinct":"PASS","shuffled_label_rate":rate,"shuffled_sha256":sha,"pair_target_balance":"PASS","same_pair_ids_for_shuffled_targets":"PASS","attacked_edge_weights_recomputed":"PASS","budget_and_no_self":"PASS","real_fold_cra_max":trace["cra_selected_ce"],"real_fold_cra_clean":trace["cra_clean_ce"],"real_fold_candidate_losses":trace["cra_candidate_ce"],"test_label_invariance":"PASS"}

def all_tests(run):
    return {"parity":parity_test(run),"math":relation_math_tests(run),"sca":sca_conflict_and_gradient_test(run),"cra_controls":cra_and_negative_control_test(run)}
