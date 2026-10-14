#!/usr/bin/env python3
"""Independent, frozen 10-group DS3A-v3 experiment and continuous-fold recovery."""
import argparse
import contextlib
import csv
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
if "preflight" in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from align import (CONFIGURATIONS, DESIGNS, LAMBDA_DISC, LAMBDA_PAIR,
                   LAMBDA_PRIVATE, RESIDUAL_LIMIT, SelectiveAlignment, weighted_pair_distance)
from evidence import TEACHER_SETTINGS, build_outer, load_evidence, pack_evidence

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
STATE = Path("/tmp") / ("spromptgl18-state-" + hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
SOURCES = STATE / "sources"
MANIFEST, RESULTS, LOGS = EXP / "manifest.json", EXP / "results", EXP / "logs"
CORE = ["main.py", "model.py", "network.py", "layers.py", "utils.py", "ABIDE-simple-2-concat-weighted-cosine.sh"]
CONFIGS = list(DESIGNS)
TASKS = [(c, s, (i * 3 + s) % 2) for i, c in enumerate(CONFIGS) for s in range(3)]
METRICS = ["acc", "ba", "hard_auc", "probability_auc", "sensitivity_asd", "specificity_td"]
CONDITIONS = ["REAL", "SHARED_OFF", "PRIVATE_OFF", "RESIDUAL_OFF", "PHENO_SHUFFLED"]
PAIRS = [("V2", "V1"), ("V3", "V2"), ("V3", "V8"), ("V5", "V4"),
         ("V5", "V6"), ("V5", "V9"), ("V7", "V5"), ("V5", "V0"), ("V7", "V0"),
         ("V5", "V3")] + [(c, "V0") for c in ("V1", "V2", "V3", "V4", "V6", "V8", "V9")]
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
                nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
                n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat", alpha=.5,
                theta_smooth=1., theta_degree=.5, theta_sparsity=0., nclass=2, mode="simple-2")
AUDIT = {
    "source": "GFR四图传播拼接320维；i=0先赋值，i=1覆盖前次combined_feat，i=2/3累加。保留这一实际顺序。输出已混合。PLN InputLayer→Encoder→FeedForward生成871×4×36潜在特征；注入在最后FeedForward后、原SPGS前。P0原8句CLIP RN50→2×4096→4096/128/9 ReLU MLP→cosine/softmax/max标量；随后原global平均池化、局部channel-softmax和RES加法、log-softmax分类器完整保留。",
    "training": "simple-2为加权NLL+四原输入图正则均值；ALoss计算但不反传。两个原Adam，PLN原my_weight_init执行后fork_rng创建辅助模块。450epoch，wait>200停止，严格ACC提升保存首次最佳；val=test，871人transductive，全队列既有标准化/RFE保留。不同组执行epoch不同后连续折随机流自然分叉，不能假称后续折初始化相同。",
    "paper": "已实际阅读本地SPromptGL PDF Fig.1、Section2.1–2.2、Eq1–7和消融。Fig.2 ABIDE从CAM86.48到GFR91.73、RES92.08、SPGS92.30；主要增量来自GFR。论文Eq1的类别图项与源码未反传ALoss、论文图加权求和与源码覆盖行为、论文实现超参与本地shell有差异；本实验复现历史源码协议，不修补这些差异。",
    "data": "PHENO48=26年龄分箱+20站点+2性别；辅助分支仅28年龄/性别，主干仍含20站点。ANAT6/FUNC10是QC；func_gsr指Ghost-to-Signal Ratio。CORR256为既有FC/RFE；不重算。当前无ADOS/ADI-R/SRS真实症状；CAM位置和局部Token不能解释为ROI。",
    "history": "实验13患者FC文本未稳定超过自身状态/身份错配；14层级投影趋同且无稳定净收益；15文本内部响应未传递至最终预测、强衰减伴随消息接近零；16修复了文本敏感性但B5较P0 ACC下降0.841pp、残差约32.33%；17D5较P0 ACC/BA下降0.4210/0.3987pp，FC/PHENO类别原型cos0.9997/0.9986，Shared probes约60.14/49.36%BA，文本反事实零翻转。不能把距离降低、低Shared/Private cosine或非零梯度当作临床信息有效。E9真实30折六指标作为历史描述性对照。",
    "novelty": "Shared/Private、辅助类别监督、蒸馏、加权对齐和残差融合均为已有组件。候选研究点为outer-train跨模态联合正确性代理驱动的可拒绝对齐，且同权重预算、错配权重和同类错配患者分别验证选择与对应。r仅控制训练损失，不进入推理融合；不是推理期患者Gate，也不是真实医学可靠度或实际对齐收益标签。"
}
LITERATURE = {
    "MGDR": dict(url="https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf", read="官方可解析全文方法及消融；shell下载返回空文本二页包装，未把包装当全文", conclusion="模态关系图、SVD共性表示、私有表示对比约束及融合已有先例；本轮不使用SVD或私有对比损失。"),
    "SPromptGL": dict(url="https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf", read="根目录11页原论文全文及Fig.1图像", conclusion="原GFR/CAM/SPGS/RES作为固定主路径，保留实际源码与论文差异。"),
    "Class-Aware Semantic Decoupling": dict(url="https://papers.miccai.org/miccai-2026/1073-Paper1452.html", read="官方摘要；PDF入口不可可靠解析，不声称已读全文", conclusion="CSSM/SDL/CSSF在类别语义层实现共享/特异、对齐和融合；本轮无类别文本查询或缺失模态任务。"),
    "To Align or Not to Align": dict(url="https://ojs.aaai.org/index.php/AAAI/article/download/39248/43209", preprint="https://arxiv.org/abs/2511.12121", read="官方AAAI26论文入口与14页arXiv全文", conclusion="显式对齐的价值随模态冗余和特异信息变化；本轮冻结系数，用同强度常量控制区分选择与整体强度。")
}
_CTX = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save_checkpoint(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    atomic(path, b"SP18" + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP18" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
    return torch.load(io.BytesIO(data[68:]), map_location="cpu", weights_only=False)


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda"]])


def rng_hash(saved):
    h = hashlib.sha256(pickle.dumps((saved["python"], saved["numpy"]), protocol=4))
    for v in [saved["cpu"]] + saved["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    table = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    assert table.columns[:-1].tolist() == [str(v) for fields in mapping.values() for v in fields]
    values = table.values
    assert values.shape == (871, 321) and np.isfinite(values).all()
    x, y = values[:, :-1], values[:, -1].astype(int) - 1
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468
    pheno = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith(("AGE_AT_SCAN_", "SEX_"))]
    sites = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith("SITE_ID_")]
    assert len(pheno) == 28 and len(sites) == 20 and not set(pheno).intersection(sites)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    audit = dict(shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
                 modality_columns={k: list(map(str, v)) for k, v in mapping.items()},
                 pheno_age_sex_indices=pheno, site_indices=sites)
    return x, y, folds, audit


def patched_sources():
    network = (ROOT / "network.py").read_text()
    model = (ROOT / "model.py").read_text()
    old = "    def forward(self, x):\n        bs = x.size(0)"
    assert network.count(old) == 1
    network = network.replace(old, "    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)")
    old = "        text_features = self.prompt_text_features\n"
    assert network.count(old) == 1
    network = network.replace(old, "        if hasattr(self, 'alignment'):\n            x = self.alignment(x, raw_modalities)\n" + old)
    model = model.replace("from network import *\n", "from network import *\nimport alignment_runtime as _alignment_runtime\n", 1)
    old = "        self.ModalFusion.apply(my_weight_init)"
    assert model.count(old) == 1
    model = model.replace(old, old + "\n        _alignment_runtime.install_branch(self)")
    model = model.replace("prob,_,_ = self.ModalFusion(feat)", "prob,_,_ = self.ModalFusion(feat, self._alignment_raw)")
    model = model.replace("prob, fusion_feat, attn = self.ModalFusion(feat)", "prob, fusion_feat, attn = self.ModalFusion(feat, self._alignment_raw)")
    model = model.replace("prob, _, attn = self.ModalFusion(feat)", "prob, _, attn = self.ModalFusion(feat, self._alignment_raw)")
    old = "            loss = cls_loss + all_graph_loss"
    assert model.count(old) == 1
    model = model.replace(old, old + "\n            self._alignment_original_losses = (cls_loss.detach(), all_graph_loss.detach())\n            if hasattr(self.ModalFusion, 'alignment'):\n                loss = loss + self.ModalFusion.alignment.losses(self.targ)")
    return {"network.py": network, "model.py": model}


def make_sources():
    expected = patched_sources()
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name in CORE:
        content = expected.get(name, (ROOT / name).read_text())
        path = SOURCES / name
        if path.exists():
            assert path.read_text() == content, ("temporary source differs", name)
        else:
            atomic(path, content)
    link = SOURCES / "CLIP"
    if not link.exists():
        link.symlink_to(ROOT / "CLIP", target_is_directory=True)


def historical_folders():
    return sorted(p for p in (ROOT / "experiment").iterdir() if p.is_dir() and
                  p.name.startswith("experiment") and p.name.split("_", 1)[0][10:].isdigit() and
                  1 <= int(p.name.split("_", 1)[0][10:]) <= 17)


def prepare():
    import numpy as np
    if MANIFEST.exists():
        integrity(full=True)
        if read_json(MANIFEST)["run_status"] != "COMPLETE":
            make_sources()
        print("PREPARE_VERIFIED", flush=True)
        return read_json(MANIFEST)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    x, y, folds, audit = data()
    protected = {str(p.relative_to(ROOT)): sha(p) for folder in historical_folders()
                 for p in sorted(folder.rglob("*")) if p.is_file()}
    sources = {str(ROOT / name): sha(ROOT / name) for name in CORE + [
        "data/ABIDE/modal_feat_dict.npy", "data/ABIDE/processed_standard_data.csv",
        "data/ABIDE/data_processing.ipynb", "CLIP/clip/clip.py", "CLIP/clip/model.py", "CLIP/clip/simple_tokenizer.py"]}
    patch = "".join("".join(difflib.unified_diff((ROOT / n).read_text().splitlines(True), text.splitlines(True),
                      fromfile="a/" + n, tofile="b/" + n)) for n, text in patched_sources().items())
    atomic(EXP / "integration_patch.diff", patch)
    # No raw-data copy: just 10 small OOF evidence arrays, masks and permutations.
    evidence_folds, feasibility, warnings_found = {}, [], {}
    for fold, (train, test) in enumerate(folds, 1):
        ev, rows, notices = build_outer(x, y, train, test, fold, audit["pheno_age_sex_indices"], audit["site_indices"])
        evidence_folds[fold] = ev
        feasibility.extend(rows)
        if notices:
            warnings_found[str(fold)] = notices
        print(f"OOF_EVIDENCE_COMPLETE fold={fold}/10 r_mean={ev['reliability'][train].mean():.6f} r_sd={ev['reliability'][train].std():.6f}", flush=True)
    atomic(EXP / "evidence.npz", pack_evidence(evidence_folds))
    save_csv(RESULTS / "feasibility.csv", feasibility)
    registered = sum(p.numel() for p in SelectiveAlignment("V5", audit["pheno_age_sex_indices"], evidence_folds[1]).parameters())
    m = dict(experiment="DS3A-v3", schema_version=1, run_status="PREPARED", prepared_utc=utc(),
             root=str(ROOT), experiment_directory=str(EXP), temporary_state=str(STATE),
             protocol=PROTOCOL, configurations=CONFIGURATIONS, audit=AUDIT, literature_audit=LITERATURE,
             data_audit=audit, source_data_sha256=sources, protected_history_sha256=protected,
             additional_parameters={c: 0 if c == "V0" else registered for c in CONFIGS},
             fixed_design=dict(shared_dim=32, private_dim=32, fusion=[128, 64, 72], lambda_disc=LAMBDA_DISC,
                               lambda_pair=LAMBDA_PAIR, lambda_private=LAMBDA_PRIVATE,
                               new_weight_decay=0., residual_relative_l2_limit=RESIDUAL_LIMIT,
                               auxiliary_init_seed="180000+task seed; fork_rng after my_weight_init",
                               weighting_mean_denominator="number of outer-train subjects; NOT sum(weights)",
                               reliability_fusion_input=False),
             evidence_protocol=dict(teacher=TEACHER_SETTINGS, inner_splits=5, inner_random_state="18000+outer_fold",
                                    reliability_derangement_seed="18600+outer_fold", class_pair_seed="18900+outer_fold",
                                    counterfactual_seed="18800+outer_fold", reliability_target="joint OOF correctness, train only",
                                    test_q="NaN/masked", reliability_fit="outer train evidence only", warnings=warnings_found,
                                    constant_prior_folds=[f for f, ev in evidence_folds.items() if bool(ev["constant_prior"])]),
             evidence_sha256=sha(EXP / "evidence.npz"), feasibility_sha256=sha(RESULTS / "feasibility.csv"),
             patch_sha256=sha(EXP / "integration_patch.diff"),
             code_sha256={n: sha(EXP / n) for n in ["run.py", "align.py", "evidence.py"]},
             clip_weight_sha256=sha(Path.home() / ".cache/clip/RN50.pt"),
             paper_sha256=sha(ROOT / "2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf"),
             split_sha256=canonical([dict(train=tr.tolist(), test=te.tolist()) for tr, te in folds]),
             instruction_prompt_sha256=sha(ROOT / "prompt"),
             git_head_at_prepare=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             handoff_prefix_sha256=sha(ROOT / "handoff.md"), handoff_prefix_bytes=(ROOT / "handoff.md").stat().st_size,
             preflight=None, completed_folds=0, completed_tasks=0, workers_per_gpu=4,
             tasks={f"{c}_seed{s}": dict(config=c, seed=s, physical_gpu=g, status="PENDING", folds=0) for c, s, g in TASKS})
    save_json(MANIFEST, m)
    make_sources()
    write_readme(m)
    return m


def integrity(full=False):
    m = read_json(MANIFEST)
    for p, h in m["source_data_sha256"].items():
        assert sha(p) == h, ("source/data changed", p)
    for n, h in m["code_sha256"].items():
        assert sha(EXP / n) == h, ("frozen code changed", n)
    for name, field in [("evidence.npz", "evidence_sha256"), ("integration_patch.diff", "patch_sha256")]:
        assert sha(EXP / name) == m[field]
    if full:
        current = {str(p.relative_to(ROOT)) for folder in historical_folders() for p in folder.rglob("*") if p.is_file()}
        assert current == set(m["protected_history_sha256"]), "historical file set changed"
        for p, h in m["protected_history_sha256"].items():
            assert sha(ROOT / p) == h, ("history changed", p)
        assert sha(Path.home() / ".cache/clip/RN50.pt") == m["clip_weight_sha256"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == m["git_head_at_prepare"]
    return m


def signature():
    m = read_json(MANIFEST)
    return canonical({k: m[k] for k in ("source_data_sha256", "code_sha256", "evidence_sha256", "patch_sha256",
                                       "protocol", "configurations", "fixed_design", "split_sha256")})


def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        make_sources()
        sys.path.insert(0, str(SOURCES))
        sys.modules["alignment_runtime"] = sys.modules[__name__]
        import network
        import model
        x, y, folds, audit = data()
        _CTX = dict(torch=torch, network=network, model=model, x=x, y=y, folds=folds, audit=audit,
                    evidence={i: load_evidence(EXP / "evidence.npz", i) for i in range(1, 11)})
    return _CTX


def install_branch(a):
    c = context()
    a._alignment_raw = a.dataLayer(a.feat)[1]
    if c["current_config"] == "V0":
        return
    with c["torch"].random.fork_rng(devices=[0]):
        c["torch"].manual_seed(180000 + int(a.hyperpm.seed))
        a.ModalFusion.alignment = SelectiveAlignment(
            c["current_config"], c["audit"]["pheno_age_sex_indices"], c["evidence"][c["current_fold"]]).cuda()
    a.optimizer_MF.add_param_group(dict(params=a.ModalFusion.alignment.parameters(), lr=.0038, weight_decay=0.))


def agent_for(config, seed, fold, original=False):
    c = context()
    c.update(current_config=config, current_fold=fold)
    cls = c["model"].EvalHelper
    if original:
        module = types.ModuleType("alignment_original_network")
        sys.modules[module.__name__] = module
        exec(compile((ROOT / "network.py").read_text(), "original_network.py", "exec"), module.__dict__)
        original_model = types.ModuleType("alignment_original_model")
        source = (ROOT / "model.py").read_text().replace("from network import *", "from alignment_original_network import *")
        exec(compile(source, "original_model.py", "exec"), original_model.__dict__)
        cls = original_model.EvalHelper
    train, test = c["folds"][fold - 1]
    with contextlib.redirect_stdout(io.StringIO()):
        a = cls([48, 6, 10, 256], c["x"], c["y"], SimpleNamespace(**PROTOCOL, seed=seed), train, test)
    a.MF_sav.close()
    a.GCMP_sav.close()
    assert (a.val_idx == a.tst_idx).all()
    return a


def branch(a):
    return getattr(a.ModalFusion, "alignment", None)


def model_state(a):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]}


def load_state(a, saved):
    a.MessagePassing0.load_state_dict(saved["gfr"])
    a.ModalFusion.load_state_dict(saved["pln"])


def weight_hash(a, auxiliary=False, legacy13=False):
    h = hashlib.sha256()
    for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]:
        for key, value in sorted(module.named_parameters()):
            if key.startswith("alignment.") != auxiliary:
                continue
            tag = key if legacy13 else name + "." + key
            h.update(tag.encode() + value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a, config):
    groups = a.optimizer_MP0.param_groups + a.optimizer_MF.param_groups
    all_params = [p for module in [a.MessagePassing0, a.ModalFusion] for p in module.parameters()]
    supplied = [p for group in groups for p in group["params"]]
    assert len(supplied) == len(set(map(id, supplied)))
    assert set(map(id, supplied)) == set(map(id, all_params))
    assert all(g["lr"] == .0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]["weight_decay"] == .11
    assert a.optimizer_MF.param_groups[0]["weight_decay"] == .11
    total = sum(p.numel() for p in all_params)
    assert total == 1462957 + read_json(MANIFEST)["additional_parameters"][config]
    if config != "V0":
        assert len(a.optimizer_MF.param_groups) == 2
        assert a.optimizer_MF.param_groups[1]["weight_decay"] == 0.
    return total


def forward(a):
    import torch
    a.MessagePassing0.eval()
    a.ModalFusion.eval()
    def capture(module, inputs, output):
        a._raw_logits = output.detach().clone()
    hook = a.ModalFusion.Outputlayer.classifier.register_forward_hook(capture)
    try:
        with torch.no_grad():
            n, parts = a.dataLayer(a.feat)
            hidden, _, _ = a.MessagePassing0(parts, n, a.targ, a.trn_idx)
            out, _, _ = a.ModalFusion(hidden, a._alignment_raw) if branch(a) else a.ModalFusion(hidden)
            assert torch.isfinite(out).all()
        return out.clone()
    finally:
        hook.remove()


def metrics(out, indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    target = context()["y"][indices]
    probability = out.exp().detach().cpu().numpy()[indices]
    pred = probability.argmax(1)
    cm = confusion_matrix(target, pred, labels=[0, 1])
    sen, spe = float(cm[0, 0] / cm[0].sum()), float(cm[1, 1] / cm[1].sum())
    return dict(acc=float((pred == target).mean()), ba=(sen + spe) / 2,
                hard_auc=float(roc_auc_score(np.eye(2)[target], np.eye(2)[pred])),
                probability_auc=float(roc_auc_score(target == 0, probability[:, 0])),
                sensitivity_asd=sen, specificity_td=spe)


def parameter_norms(a):
    import torch
    b = branch(a)
    if b is None:
        return {}
    return {k: float(torch.stack([p.detach().square().sum() for p in pp]).sum().sqrt())
            for k, pp in b.component_parameters().items()}


def gradients(a):
    import torch
    b = branch(a)
    parts = {} if b is None else b.component_parameters()
    parts.update(original_gfr=list(a.MessagePassing0.parameters()),
                 original_pln=[p for k, p in a.ModalFusion.named_parameters() if not k.startswith("alignment.")])
    result = {}
    for k, pp in parts.items():
        gg = [p.grad.detach() for p in pp if p.grad is not None]
        result[k] = float(torch.stack([g.square().sum() for g in gg]).sum().sqrt()) if gg else 0.
    assert all(math.isfinite(v) for v in result.values())
    return result


def objectives(a):
    """Best-checkpoint, eval-mode loss geometry; never an optimizer step."""
    c = context()
    a.MessagePassing0.eval()
    a.ModalFusion.eval()
    n, parts = a.dataLayer(a.feat)
    hidden, adj, _ = a.MessagePassing0(parts, n, a.targ, a.trn_idx)
    out, _, _ = a.ModalFusion(hidden, a._alignment_raw) if branch(a) else a.ModalFusion(hidden)
    cls = c["model"].ClsLoss(out, a.targ, a.trn_idx, a.weight)
    graph = sum(c["model"].GraphConstructLoss(parts[i], adj[i], 1., .5, 0.) for i in range(n)) / 4
    losses = {"classification": cls, "original_graph": graph}
    b = branch(a)
    if b:
        b.losses(a.targ)
        if b.design.disc:
            losses["discriminative"] = LAMBDA_DISC * b.last["disc_loss"]
        if b.design.weighting != "none":
            losses["pair_alignment"] = LAMBDA_PAIR * b.last["pair_weighted_loss"]
        if b.design.private:
            losses["private_distillation"] = LAMBDA_PRIVATE * b.last["private_loss"]
    return losses


def loss_geometry(a):
    import torch
    values = objectives(a)
    params = [p for module in [a.MessagePassing0, a.ModalFusion] for p in module.parameters()]
    result = {}
    for name, loss in values.items():
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        norm = float(torch.stack([g.detach().square().sum() for g in grads if g is not None]).sum().sqrt())
        assert math.isfinite(norm) and math.isfinite(float(loss.detach()))
        result["loss_" + name] = float(loss.detach())
        result["gradient_l2_" + name] = norm
    return result


def snapshot(a):
    b = branch(a)
    return {k: v.detach().clone() for k, v in b.last.items() if hasattr(v, "detach")} if b else {}


def counterfactual(a, seed, fold, real):
    import torch
    b = branch(a)
    assert b is not None and b.condition == "REAL"
    baseline, real_logits = snapshot(a), a._raw_logits.clone()
    before = (weight_hash(a), weight_hash(a, True), rng_hash(rng()))
    original_metrics = metrics(real, a.tst_idx)
    rows = []
    try:
        for condition in CONDITIONS:
            b.condition = condition
            altered = forward(a)
            now = snapshot(a)
            assert torch.equal(now["base"], baseline["base"])
            if condition == "REAL":
                assert torch.equal(altered, real)
            values = metrics(altered, a.tst_idx)
            indices = a.tst_idx
            relative = now["delta"][indices].norm(dim=-1) / (now["base"][indices].norm(dim=-1) + 1e-8)
            rows.append(dict(config="V5", seed=seed, fold=fold, condition=condition, **values,
                             **{"delta_" + k: values[k] - original_metrics[k] for k in METRICS},
                             prediction_flip_rate=float(altered[indices].argmax(1).ne(real[indices].argmax(1)).double().mean()),
                             mean_abs_asd_probability_change=float((altered.exp()[indices, 0] - real.exp()[indices, 0]).abs().mean()),
                             mean_abs_logit_change=float((a._raw_logits[indices] - real_logits[indices]).abs().mean()),
                             delta_mean_abs_change=float((now["delta"][indices] - baseline["delta"][indices]).abs().mean()),
                             fusion_input_mean_abs_change=float((now["fusion_input"][indices] - baseline["fusion_input"][indices]).abs().mean()),
                             residual_relative_mean=float(relative[:, [0, 3]].mean()),
                             same_checkpoint=True, base_tokens_exact=True, parameters_unchanged=True,
                             test_labels_used_for_shuffle=False))
    finally:
        b.condition = "REAL"
        restored = forward(a)
    assert torch.equal(restored, real) and torch.equal(a._raw_logits, real_logits)
    assert (weight_hash(a), weight_hash(a, True), rng_hash(rng())) == before
    return rows


def assert_historical_fold(row):
    if row["config"] != "V0":
        return
    for number, config in [(13, "F0"), (14, "E00"), (15, "A0"), (16, "B0"), (17, "D0")]:
        folder = next((ROOT / "experiment").glob(f"experiment{number}_*"))
        ref = next(r for r in read_csv(folder / "results/fold_metrics.csv")
                   if r["config"] == config and int(r["seed"]) == int(row["seed"]) and int(r["fold"]) == int(row["fold"]))
        for key in METRICS:
            reference_key = "balanced_accuracy" if number == 14 and key == "ba" else key
            assert abs(float(row[key]) - float(ref[reference_key])) < 1e-12, ("P0 parity metrics", number, row["seed"], row["fold"], key)
        for key in ("best_epoch", "epochs_executed"):
            assert int(row[key]) == int(ref[key]), ("P0 parity epoch", number, key)
        initial_key = "initial_trunk_legacy13_sha256" if number == 13 else "initial_trunk_sha256"
        assert row[initial_key] == ref["initial_trunk_sha256"], ("P0 initialization", number)
        rng_key = "fold_entry_rng_sha256" if number == 14 else "entry_rng_sha256"
        if rng_key in ref:
            assert row["entry_rng_sha256"] == ref[rng_key], ("P0 RNG parity", number)


def assert_equal_tree(a, b):
    import numpy as np
    import torch
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for v, w in zip(a, b):
            assert_equal_tree(v, w)
    else:
        assert a == b


def preflight():
    import numpy as np
    import torch
    m = integrity(full=True)
    if m.get("preflight", {}).get("status") == "PASS" if m.get("preflight") else False:
        print("PREFLIGHT_ALREADY_PASS", flush=True)
        return
    c = context()
    checks = dict(status="RUNNING", started_utc=utc(), initializations={})
    for seed in range(3):
        seed_rng(seed)
        entry = rng()
        native = agent_for("V0", seed, 1, original=True)
        native_init, native_rng, native_out = weight_hash(native), rng(), forward(native)
        common_branch = None
        for config in CONFIGS:
            restore_rng(entry)
            c["network"]._prompt_text_features = None
            a = agent_for(config, seed, 1)
            assert weight_hash(a) == native_init and rng_hash(rng()) == rng_hash(native_rng)
            assert torch.equal(forward(a), native_out)
            assert torch.equal(a.ModalFusion.prompt_text_features, native.ModalFusion.prompt_text_features)
            count = optimizer_audit(a, config)
            b = branch(a)
            if b:
                assert b.alpha.count_nonzero() == 0 and b.last["delta"].count_nonzero() == 0
                current_hash = weight_hash(a, True)
                if common_branch is None:
                    common_branch = current_hash
                assert current_hash == common_branch
                assert all(b.last[k].shape == (871, 32) for k in ("shared_corr", "private_corr", "shared_pheno", "private_pheno"))
                assert b.last["fusion_input"].shape == (871, 128)
            checks["initializations"][f"{config}_seed{seed}"] = dict(total_parameters=count,
                initial_trunk_sha256=native_init, initial_branch_sha256=common_branch or "",
                output_exact=True, trunk_parameters_exact=True, cpu_cuda_rng_exact=True)
            del a
        restore_rng(native_rng)
        oracles = []
        for _ in range(2):
            native.run_epoch("simple-2")
            buffer = io.BytesIO()
            torch.save([native.optimizer_MP0.state_dict(), native.optimizer_MF.state_dict()], buffer)
            oracles.append((model_state(native), rng_hash(rng()), forward(native), buffer.getvalue()))
        restore_rng(entry)
        c["network"]._prompt_text_features = None
        a = agent_for("V0", seed, 1)
        for index in range(2):
            a.run_epoch("simple-2")
            oracle = oracles[index]
            assert_equal_tree(model_state(a), oracle[0])
            assert rng_hash(rng()) == oracle[1] and torch.equal(forward(a), oracle[2])
            optimizers = torch.load(io.BytesIO(oracle[3]), map_location="cpu", weights_only=False)
            assert_equal_tree(a.optimizer_MP0.state_dict(), optimizers[0])
            assert_equal_tree(a.optimizer_MF.state_dict(), optimizers[1])
        del native, a, oracles
    checks["p0_forward_two_adam_updates_rng_exact"] = True
    # Identical teacher evidence is reused across all seeds and all four controls.
    controls = {}
    for fold in range(1, 11):
        ev = c["evidence"][fold]
        tr, te = c["folds"][fold - 1]
        assert np.array_equal(ev["train_indices"], tr) and np.array_equal(ev["test_indices"], te)
        assert np.isnan(ev["joint_correct"][te]).all()
        expected_q = ((np.where(ev["teacher_fc_probability"][tr] >= .5, 0, 1) == c["y"][tr]) &
                      (np.where(ev["teacher_pheno_probability"][tr] >= .5, 0, 1) == c["y"][tr]))
        assert np.array_equal(expected_q.astype(float), ev["joint_correct"][tr])
        assert np.all(ev["inner_fold"][tr] >= 0) and np.all(ev["inner_fold"][te] == -1)
        models = {config: SelectiveAlignment(config, c["audit"]["pheno_age_sex_indices"], ev) for config in ["V4", "V5", "V6", "V9"]}
        w4, w5, w6, w9 = [models[k].weights for k in ["V4", "V5", "V6", "V9"]]
        assert (w4 - w4[0]).abs().max() == 0
        assert abs(float(w4.sum() - w5.sum())) < 1e-12
        assert torch.equal(w5.sort().values, w6.sort().values) and torch.equal(w5, w9)
        paired = models["V9"].paired_indices.numpy()
        assert np.all(paired != tr) and np.array_equal(c["y"][paired], c["y"][tr])
        assert set(paired) == set(tr) and not np.intersect1d(paired, te).size
        controls[str(fold)] = dict(v4_v5_sum_abs_error=float((w4.sum() - w5.sum()).abs()),
                                   v5_v6_sorted_weights_exact=True, v5_v9_weights_exact=True,
                                   v9_same_class_train_derangement=True, test_q_missing=True,
                                   evidence_sha256=m["evidence_sha256"])
    checks["frozen_control_evidence"] = controls
    # Rebuild just one fold with poisoned held-out labels to audit the full evidence boundary.
    poisoned = c["y"].copy()
    train, test = c["folds"][0]
    poisoned[test] = 1 - poisoned[test]
    ev, _, _ = build_outer(c["x"], poisoned, train, test, 1, c["audit"]["pheno_age_sex_indices"], c["audit"]["site_indices"])
    for key in ("teacher_fc_probability", "teacher_pheno_probability", "reliability", "class_pair_permutation"):
        assert np.array_equal(ev[key], c["evidence"][1][key])
    checks["outer_test_labels_cannot_change_evidence"] = True
    generator = torch.Generator().manual_seed(1801)
    fc, ph = torch.randn(13, 32, generator=generator), torch.randn(13, 32, generator=generator)
    _, unweighted = weighted_pair_distance(fc, ph, torch.ones(13, dtype=torch.float64))
    _, weak = weighted_pair_distance(fc, ph, torch.full((13,), .03, dtype=torch.float64))
    assert abs(float(weak / unweighted) - .03) < 1e-12
    checks["weight_003_loss_ratio"] = float(weak / unweighted)
    # Gradients include expected zero first-step fusion-connected encoders in V1.
    checks["two_step_component_gradients"] = {}
    for config in CONFIGS[1:]:
        seed_rng(0)
        a = agent_for(config, 0, 1)
        samples = []
        for _ in range(2):
            a.run_epoch("simple-2")
            samples.append(gradients(a))
        b = branch(a)
        for key in b.effective_parameters()["active_components"]:
            assert max(v[key] for v in samples) > 0, ("disconnected active parameter", config, key)
        for key in b.effective_parameters()["inactive_components"]:
            assert all(v[key] == 0 for v in samples), ("disabled head unexpectedly trained", config, key)
        checks["two_step_component_gradients"][config] = samples
        geometry = loss_geometry(a)
        for key, value in geometry.items():
            assert math.isfinite(value)
        # Direct auxiliary loss gradients, independent of original CE.
        for name, component in [("discriminative", "shared_corr"), ("pair_alignment", "shared_pheno"), ("private_distillation", "private_corr")]:
            if "gradient_l2_" + name in geometry:
                losses = objectives(a)
                gg = torch.autograd.grad(losses[name], b.component_parameters()[component], allow_unused=True)
                assert any(g is not None and bool(g.abs().sum() > 0) for g in gg)
        del a
    checks["finite_loss_gradients_and_disabled_head_audit"] = True
    # Check CPU/CUDA/Python/NumPy plus BOTH Adam states after an interruption.
    for config in ["V0", "V5", "V7"]:
        seed_rng(1)
        a = agent_for(config, 1, 1)
        for _ in range(2):
            a.run_epoch("simple-2")
        path = STATE / "recovery-preflight.pt"
        save_checkpoint(path, dict(model=model_state(a), mp=a.optimizer_MP0.state_dict(), mf=a.optimizer_MF.state_dict(), rng=rng()))
        a.run_epoch("simple-2")
        expected, expected_rng, expected_output = model_state(a), rng_hash(rng()), forward(a)
        saved = checkpoint(path)
        load_state(a, saved["model"])
        a.optimizer_MP0.load_state_dict(saved["mp"])
        a.optimizer_MF.load_state_dict(saved["mf"])
        restore_rng(saved["rng"])
        a.run_epoch("simple-2")
        assert_equal_tree(model_state(a), expected)
        assert rng_hash(rng()) == expected_rng and torch.equal(forward(a), expected_output)
        path.unlink()
    checks["full_checkpoint_third_step_exact"] = True
    seed_rng(2)
    a = agent_for("V5", 2, 1)
    b = branch(a)
    with torch.no_grad():
        b.alpha[:] = torch.tensor([100., -100.], device=b.alpha.device)
    real = forward(a)
    ll = snapshot(a)
    relative = ll["delta"].norm(dim=-1) / (ll["base"].norm(dim=-1) + 1e-8)
    assert float(relative.max()) <= .100001
    assert torch.count_nonzero(ll["delta"][:, [1, 2]]) == 0
    altered_target = a.targ.clone()
    altered_target[a.tst_idx] = 1 - altered_target[a.tst_idx]
    assert torch.equal(b.losses(a.targ), b.losses(altered_target))
    original_target = a.targ
    a.targ = altered_target
    assert torch.equal(forward(a), real)
    a.targ = original_target
    raw = [v.clone() for v in a._alignment_raw]
    raw[0][:, c["audit"]["site_indices"]] *= -5
    assert torch.equal(b(ll["base"], raw), ll["enhanced"])
    frozen_weights = b.weights.clone()
    b.weights.mul_(0.)
    assert torch.equal(forward(a), real), "reliability entered inference fusion"
    b.weights.copy_(frozen_weights)
    cf = counterfactual(a, 2, 1, real)
    assert all(r["mean_abs_asd_probability_change"] > 0 for r in cf if r["condition"] != "REAL")
    checks.update(residual_bound_verified=True, auxiliary_site_excluded=True,
                  auxiliary_loss_outer_train_only=True, predictions_test_label_invariant=True,
                  reliability_not_used_in_inference=True, counterfactual_same_checkpoint_base_exact=True,
                  counterfactual_probe={r["condition"]:r["mean_abs_asd_probability_change"] for r in cf})
    # Writers are exercised before any formal result exists.
    diagnostic = diagnostics(a, "V5", 2, 1, real, gradients(a), parameter_norms(a))
    assert all(math.isfinite(v) for v in diagnostic.values() if isinstance(v, float))
    checks["diagnostic_writer_finite"] = True
    checkdir = STATE / "patch-check"
    checkdir.mkdir(exist_ok=True)
    for name in ["model.py", "network.py"]:
        shutil.copy2(ROOT / name, checkdir / name)
    subprocess.run(["git", "apply", "--check", str(EXP / "integration_patch.diff")], cwd=checkdir, check=True)
    shutil.rmtree(checkdir)
    integrity(full=True)
    checks.update(status="PASS", completed_utc=utc(), root_and_history_unchanged=True, patch_applies=True)
    m = read_json(MANIFEST)
    m["preflight"] = checks
    save_json(MANIFEST, m)
    print("PREFLIGHT_PASS", flush=True)


def representation_diagnostics(ll, train, test, labels):
    import numpy as np
    from evidence import fit_teacher, probability_asd
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    row = {}
    for scope in ("shared", "private"):
        for modality in ("corr", "pheno"):
            key = scope + "_" + modality
            values = ll[key]
            train_values = values[train]
            centered = train_values - train_values.mean(0)
            singular = np.linalg.svd(centered, compute_uv=False)
            proportion = singular / max(singular.sum(), 1e-12)
            row[key + "_variance"] = float(train_values.var(0).mean())
            row[key + "_effective_rank"] = float(np.exp(-(proportion * np.log(np.maximum(proportion, 1e-12))).sum()))
            row[key + "_norm_mean"] = float(np.linalg.norm(train_values, axis=1).mean())
            teacher, notices = fit_teacher(train_values, labels[train])
            probability = probability_asd(teacher, values[test])
            row[key + "_probe_ba"] = float(balanced_accuracy_score(labels[test], np.where(probability >= .5, 0, 1)))
            row[key + "_probe_probability_auc"] = float(roc_auc_score(labels[test] == 0, probability))
            row[key + "_probe_convergence_notices"] = len(notices)
            for partition, indices in [("train", train), ("test", test)]:
                representations = values[indices]
                unit = representations / np.maximum(np.linalg.norm(representations, axis=1, keepdims=True), 1e-12)
                prototypes = np.stack([representations[labels[indices] == category].mean(0) for category in (0, 1)])
                prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
                row[key + "_class_prototype_cosine_" + partition] = float(np.dot(prototypes[0], prototypes[1]))
                distances = np.maximum(2 - 2 * (unit @ unit.T), 0.)
                same = labels[indices, None] == labels[indices][None, :]
                opposite = ~same
                np.fill_diagonal(same, False)
                row[key + "_same_class_distance_" + partition] = float(distances[same].mean())
                row[key + "_different_class_distance_" + partition] = float(distances[opposite].mean())
                row[key + "_between_within_ratio_" + partition] = float(distances[opposite].mean() / max(distances[same].mean(), 1e-12))
    for modality in ("corr", "pheno"):
        shared, private = ll["shared_" + modality][train], ll["private_" + modality][train]
        cos = (shared * private).sum(-1) / np.maximum(np.linalg.norm(shared, axis=1) * np.linalg.norm(private, axis=1), 1e-12)
        row["shared_private_cosine_" + modality] = float(cos.mean())
        row["shared_private_abs_cosine_" + modality] = float(np.abs(cos).mean())
    for partition, indices in [("train", train), ("test", test)]:
        sc, sp = ll["shared_corr"][indices], ll["shared_pheno"][indices]
        uc = sc / np.maximum(np.linalg.norm(sc, axis=1, keepdims=True), 1e-12)
        up = sp / np.maximum(np.linalg.norm(sp, axis=1, keepdims=True), 1e-12)
        row["shared_cross_modal_distance_" + partition] = float(np.square(uc - up).sum(-1).mean())
    return row


def diagnostics(a, config, seed, fold, real, tracker, initial_norms):
    import numpy as np
    row = dict(config=config, seed=seed, fold=fold, train_n=len(a.trn_idx), test_n=len(a.tst_idx),
               branch_enabled=branch(a) is not None, prediction_nonfinite=False,
               sampled_gradient_max=json.dumps(tracker, sort_keys=True),
               **loss_geometry(a))
    b = branch(a)
    ev = context()["evidence"][fold]
    reliability, target = ev["reliability"][a.trn_idx], ev["joint_correct"][a.trn_idx]
    row.update(reliability_mean=float(reliability.mean()), reliability_sd=float(reliability.std()),
               reliability_p05=float(np.quantile(reliability, .05)), reliability_p50=float(np.median(reliability)),
               reliability_p95=float(np.quantile(reliability, .95)), reliability_gt_half_fraction=float((reliability > .5).mean()),
               reliability_kish_effective_fraction=float(reliability.sum() ** 2 / max(len(reliability) * (reliability ** 2).sum(), 1e-12)),
               joint_correct_fraction=float(target.mean()),
               reliability_q_correlation=float(np.corrcoef(reliability, target)[0, 1]) if reliability.std() > 1e-12 and target.std() > 1e-12 else 0.,
               reliability_q_auc=float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(target, reliability)) if target.std() else .5,
               reliability_evidence_sha256=read_json(MANIFEST)["evidence_sha256"],
               reliability_q_fit_scope="outer train; in-sample reliability calibration on OOF teacher evidence")
    if b is None:
        row.update(residual_relative_pheno_mean=0., residual_relative_corr_mean=0., residual_relative_max=0.,
                   effective_alignment_weight_sum=0., effective_alignment_weight_mean=0.,
                   registered_auxiliary_parameters=0, structurally_active_auxiliary_parameters=0,
                   disc_enabled=False, pair_enabled=False, private_distillation_enabled=False)
        forward(a)
        return row
    row.update(disc_raw_loss=float(b.last["disc_loss"].detach()), pair_raw_loss=float(b.last["pair_raw_loss"].detach()),
               pair_weighted_loss=float(b.last["pair_weighted_loss"].detach()), private_raw_loss=float(b.last["private_loss"].detach()),
               disc_enabled=b.design.disc, pair_enabled=b.design.weighting != "none", private_distillation_enabled=b.design.private,
               effective_alignment_weight_sum=float(b.weights.sum()), effective_alignment_weight_mean=float(b.weights.mean()),
               effective_alignment_weight_sd=float(b.weights.std(unbiased=False)))
    assert __import__("torch").equal(forward(a), real)
    ll = {k: v.cpu().numpy() for k, v in snapshot(a).items()}
    assert all(np.isfinite(v).all() for v in ll.values())
    row.update(representation_diagnostics(ll, a.trn_idx, a.tst_idx, context()["y"]))
    relative = np.linalg.norm(ll["delta"], axis=-1) / (np.linalg.norm(ll["base"], axis=-1) + 1e-8)
    assert relative.max() <= .100001 and np.count_nonzero(ll["delta"][:, [1, 2]]) == 0
    for index, name in enumerate(("pheno", "anat", "func", "corr")):
        row["residual_relative_" + name + "_mean"] = float(relative[:, index].mean())
        row["residual_relative_" + name + "_max"] = float(relative[:, index].max())
        row["base_norm_" + name + "_mean"] = float(np.linalg.norm(ll["base"][:, index], axis=-1).mean())
        row["delta_norm_" + name + "_mean"] = float(np.linalg.norm(ll["delta"][:, index], axis=-1).mean())
    row.update(residual_relative_max=float(relative.max()),
               residual_near_zero_selected_fraction=float((relative[:, [0, 3]] < 1e-5).mean()),
               alpha_pheno=float(b.alpha[0].detach()), alpha_corr=float(b.alpha[1].detach()),
               bound_verified=True, pheno_site_excluded=True)
    capacity = b.effective_parameters()
    norms = parameter_norms(a)
    row.update(registered_auxiliary_parameters=capacity["registered"],
               structurally_active_auxiliary_parameters=capacity["structurally_active"],
               structurally_active_components=json.dumps(capacity["active_components"]),
               disabled_head_components=json.dumps(capacity["inactive_components"]),
               sampled_nonzero_components=json.dumps([k for k in norms if tracker.get(k, 0.) > 0]))
    for key, norm in norms.items():
        row["parameter_norm_" + key] = norm
        row["parameter_norm_ratio_" + key] = norm / max(initial_norms.get(key, 0.), 1e-8)
        row["sampled_gradient_max_" + key] = tracker.get(key, 0.)
    for name, module in [("original_gfr", a.MessagePassing0), ("original_pln", a.ModalFusion)]:
        row["parameter_norm_" + name] = math.sqrt(sum(float(p.detach().square().sum()) for k, p in module.named_parameters() if not k.startswith("alignment.")))
        row["sampled_gradient_max_" + name] = tracker.get(name, 0.)
    for loss in ("discriminative", "pair_alignment", "private_distillation"):
        row.setdefault("loss_" + loss, 0.)
        row.setdefault("gradient_l2_" + loss, 0.)
    return row


def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def worker(config, seed):
    import torch
    m = integrity()
    assert m["preflight"]["status"] == "PASS"
    task = f"{config}_seed{seed}"
    gpu = int(os.environ["DS3AV3_PHYSICAL_GPU"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(gpu) and m["tasks"][task]["physical_gpu"] == gpu
    handle = acquire_lock(STATE / (task + ".lock"))
    result_path, checkpoint_path = STATE / (task + ".json"), STATE / (task + ".pt")
    sig = signature()
    if result_path.exists():
        old = read_json(result_path)
        if old["status"] == "COMPLETE":
            assert old["signature"] == sig and len(old["metrics"]) == 10
            print("SKIP_COMPLETE " + task, flush=True)
            return
    c = context()
    saved = checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    if saved:
        assert saved["signature"] == sig
        result = saved["result"]
        # Recreate the lazy CLIP cache BEFORE restoring the continuous RNG stream.
        c["network"].get_prompt_text_features()
        restore_rng(saved["rng"])
        start_fold = int(saved["fold"]) + (1 if saved["phase"] == "BETWEEN_FOLDS" else 0)
        if saved["phase"] == "BETWEEN_FOLDS":
            save_json(result_path, result)
        print(f"RESUME {task} phase={saved['phase']} fold={start_fold}", flush=True)
    else:
        seed_rng(seed)
        start_fold = 1
        result = dict(signature=sig, task=task, config=config, seed=seed, physical_gpu=gpu, status="RUNNING",
                      metrics=[], diagnostics=[], counterfactual=[], started_utc=utc(), seconds=0.)
    print(f"WORKER {task} GPU{gpu} logical=cuda:0 device={torch.cuda.get_device_name(0)}", flush=True)
    for fold in range(start_fold, 11):
        entry = rng()
        a = agent_for(config, seed, fold)
        optimizer_audit(a, config)
        initial, legacy13, branch_initial = weight_hash(a), weight_hash(a, legacy13=True), weight_hash(a, True)
        initial_norms = parameter_norms(a)
        tracker = {k: 0. for k in gradients(a)}
        best_acc, best_epoch, best_model, wait, start, prior_seconds = 0., -1, None, 0, 0, 0.
        if saved and saved["phase"] == "EPOCH" and int(saved["fold"]) == fold:
            load_state(a, saved["model"])
            a.optimizer_MP0.load_state_dict(saved["mp"])
            a.optimizer_MF.load_state_dict(saved["mf"])
            restore_rng(saved["rng"])
            best_acc, best_epoch, best_model, wait, start = (saved[k] for k in ["best_acc", "best_epoch", "best_model", "wait", "next_epoch"])
            tracker, entry, initial, legacy13, branch_initial = (saved[k] for k in ["tracker", "entry_rng", "initial", "legacy13", "branch_initial"])
            initial_norms, prior_seconds = saved["initial_norms"], saved["fold_seconds"]
        saved = None
        started, last = time.monotonic(), start - 1
        print(f"FOLD_START {fold} initial={initial} entry_rng={rng_hash(entry)}", flush=True)
        for epoch in (range(start, 450) if wait <= 200 else []):
            print(f"EPOCH {fold}:{epoch} ", end="")
            a.run_epoch("simple-2")
            if epoch in (0, 1) or (epoch + 1) % 25 == 0:
                for key, value in gradients(a).items():
                    tracker[key] = max(tracker[key], value)
            out = forward(a)
            validation = float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            print(f"val-acc={validation:.12f}", flush=True)
            if validation > best_acc:
                best_acc, best_epoch, best_model, wait = validation, epoch, model_state(a), 0
            else:
                wait += 1
            last = epoch
            stop = wait > 200 or epoch == 449
            if (epoch + 1) % 25 == 0 or stop:
                if branch(a):
                    ll = branch(a).last
                    assert float((ll["delta"].norm(dim=-1) / (ll["base"].norm(dim=-1) + 1e-8)).max()) <= .100001
                save_checkpoint(checkpoint_path, dict(signature=sig, phase="EPOCH", fold=fold, next_epoch=epoch + 1,
                    model=model_state(a), mp=a.optimizer_MP0.state_dict(), mf=a.optimizer_MF.state_dict(), rng=rng(),
                    best_acc=best_acc, best_epoch=best_epoch, best_model=best_model, wait=wait, tracker=tracker,
                    entry_rng=entry, initial=initial, legacy13=legacy13, branch_initial=branch_initial,
                    initial_norms=initial_norms, fold_seconds=prior_seconds + time.monotonic() - started, result=result))
            if stop:
                break
        assert best_model is not None
        continuation = rng()
        load_state(a, best_model)
        out = forward(a)
        values = metrics(out, a.tst_idx)
        assert abs(values["acc"] - best_acc) < 1e-12 and abs(values["ba"] - values["hard_auc"]) < 1e-12
        row = dict(config=config, seed=seed, fold=fold, physical_gpu=gpu, **values, best_epoch=best_epoch,
                   epochs_executed=last + 1, selected_val_acc=best_acc, train_n=len(a.trn_idx), test_n=len(a.tst_idx),
                   val_equals_test=True, initial_trunk_sha256=initial, initial_trunk_legacy13_sha256=legacy13,
                   initial_branch_sha256=branch_initial if config != "V0" else "", entry_rng_sha256=rng_hash(entry),
                   additional_parameters=m["additional_parameters"][config])
        assert_historical_fold(row)
        diagnostic = diagnostics(a, config, seed, fold, out, tracker, initial_norms)
        diagnostic.update(best_epoch=best_epoch, epochs_executed=last + 1)
        cf = counterfactual(a, seed, fold, out) if config == "V5" else []
        row["seconds"] = prior_seconds + time.monotonic() - started
        result["metrics"].append(row)
        result["diagnostics"].append(diagnostic)
        result["counterfactual"].extend(cf)
        restore_rng(continuation)
        result.update(seconds=sum(r["seconds"] for r in result["metrics"]), status="COMPLETE" if fold == 10 else "RUNNING", updated_utc=utc())
        save_checkpoint(checkpoint_path, dict(signature=sig, phase="BETWEEN_FOLDS", fold=fold, rng=continuation, result=result))
        save_json(result_path, result)
        print("FOLD_COMPLETE " + json.dumps(row, ensure_ascii=False), flush=True)
        del a, best_model
        if fold == 10:
            checkpoint_path.unlink(missing_ok=True)
    print("TASK_COMPLETE " + task, flush=True)
    handle.close()


def live_workers():
    import psutil
    found = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if str(EXP / "run.py") in args and "--worker" in args:
            i = args.index("--worker")
            found[f"{args[i + 1]}_seed{args[i + 2]}"] = process.info["pid"]
    return found


def task_results():
    results = []
    for config, seed, gpu in TASKS:
        path = STATE / f"{config}_seed{seed}.json"
        if not path.exists():
            continue
        result = read_json(path)
        assert result["signature"] == signature() and result["physical_gpu"] == gpu
        folds = [r["fold"] for r in result["metrics"]]
        assert folds == list(range(1, len(folds) + 1)) and len(result["diagnostics"]) == len(folds)
        assert len(result["counterfactual"]) == (5 * len(folds) if config == "V5" else 0)
        if result["status"] == "COMPLETE":
            assert len(folds) == 10
        results.append(result)
    return results


def refresh():
    m = read_json(MANIFEST)
    if m["run_status"] == "COMPLETE" and not STATE.exists():
        return m
    completed = task_results()
    metrics_rows = [r for t in completed for r in t["metrics"]]
    ds = [r for t in completed for r in t["diagnostics"]]
    cf = [r for t in completed for r in t["counterfactual"]]
    assert len(set((r["config"], r["seed"], r["fold"]) for r in metrics_rows)) == len(metrics_rows)
    for task in completed:
        m["tasks"][task["task"]].update(status=task["status"], folds=len(task["metrics"]), seconds=task["seconds"], updated_utc=task["updated_utc"])
    m.update(completed_folds=len(metrics_rows), completed_tasks=sum(t["status"] == "COMPLETE" for t in completed),
             diagnostic_rows=len(ds), counterfactual_rows=len(cf), updated_utc=utc())
    for name, rows in [("fold_metrics", metrics_rows), ("diagnostics", ds), ("counterfactual", cf)]:
        save_csv(RESULTS / (name + ".csv"), rows)
    save_json(MANIFEST, m)
    return m


def worker_env(gpu):
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), DS3AV3_PHYSICAL_GPU=str(gpu),
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")


def resources():
    import psutil
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
    cards = []
    for line in raw.strip().splitlines():
        index, name, total, used, utilization = [v.strip() for v in line.split(",")]
        cards.append(dict(index=int(index), name=name, total_mib=int(total.split()[0]), used_mib=int(used.split()[0]), utilization=int(utilization.split()[0])))
    pids = list(live_workers().values())
    memory, cpu_seconds = 0, 0.
    for pid in pids:
        try:
            p = psutil.Process(pid)
            memory += p.memory_info().rss
            cpu_seconds += sum(p.cpu_times()[:2])
        except psutil.NoSuchProcess:
            pass
    return dict(utc=utc(), gpu=cards, worker_rss_sum_gib=memory / 2**30, worker_cpu_seconds=cpu_seconds,
                host_available_gib=psutil.virtual_memory().available / 2**30, system_cpu_percent=psutil.cpu_percent(),
                active_workers=len(pids), gpu_memory_scope="whole device including unrelated jobs")


def schedule(workers_per_gpu=4):
    assert 1 <= workers_per_gpu <= 4
    if MANIFEST.exists() and read_json(MANIFEST)["run_status"] == "COMPLETE":
        validate_complete()
        print("ALREADY_COMPLETE 30 tasks / 300 folds", flush=True)
        return
    prepare()
    handle = acquire_lock(STATE / "scheduler.lock")
    m = read_json(MANIFEST)
    if not m["preflight"] or m["preflight"]["status"] != "PASS":
        with open(LOGS / "preflight.log", "a") as output:
            subprocess.run([sys.executable, str(EXP / "run.py"), "preflight"], cwd=EXP, env=worker_env(0), stdout=output, stderr=subprocess.STDOUT, check=True)
    m = read_json(MANIFEST)
    assert m["preflight"]["status"] == "PASS"
    m.update(run_status="RUNNING", workers_per_gpu=workers_per_gpu)
    m.setdefault("started_unix", time.time())
    m.setdefault("started_utc", utc())
    m["launch_resources"] = resources()
    save_json(MANIFEST, m)
    failures = dict(m.get("failures_and_restarts", {}))
    active = {task:dict(pid=pid, process=None, gpu=m["tasks"][task]["physical_gpu"]) for task, pid in live_workers().items()}
    limits = {0:workers_per_gpu, 1:workers_per_gpu}
    last_refresh, last_resource = 0., 0.
    print("SCHEDULER_START " + utc(), flush=True)
    while True:
        for task, info in list(active.items()):
            proc = info["process"]
            exit_code = proc.poll() if proc else (None if Path("/proc", str(info["pid"])).exists() else 0)
            if exit_code is None:
                continue
            if "log" in info:
                info["log"].close()
            del active[task]
            if exit_code != 0:
                failures[task] = failures.get(task, 0) + 1
                m = read_json(MANIFEST)
                m["failures_and_restarts"] = failures
                save_json(MANIFEST, m)
                print(f"RESTART {task} exit={exit_code} from atomic checkpoint", flush=True)
                if failures[task] >= 3:
                    raise RuntimeError("Repeated task failure: " + task)
        done = {t["task"] for t in task_results() if t["status"] == "COMPLETE"}
        for gpu in (0, 1):
            for config, seed, assigned in TASKS:
                task = f"{config}_seed{seed}"
                if assigned != gpu or task in done or task in active:
                    continue
                if sum(i["gpu"] == gpu for i in active.values()) >= limits[gpu]:
                    break
                output = open(LOGS / (task + ".log"), "a")
                output.write("\nATTEMPT_START " + utc() + "\n")
                output.flush()
                proc = subprocess.Popen([sys.executable, str(EXP / "run.py"), "--worker", config, str(seed)],
                                        cwd=EXP, env=worker_env(gpu), stdout=output, stderr=subprocess.STDOUT)
                active[task] = dict(pid=proc.pid, process=proc, gpu=gpu, log=output)
                print(f"LAUNCH {task} GPU{gpu} pid={proc.pid}", flush=True)
        if time.monotonic() - last_refresh > 20 or not active:
            m = refresh()
            if time.monotonic() - last_resource >= 60:
                sample = resources()
                peaks = m.setdefault("runtime_resource_peaks", dict(worker_rss_sum_gib=0., gpu0_mib=0, gpu1_mib=0))
                peaks["worker_rss_sum_gib"] = max(peaks["worker_rss_sum_gib"], sample["worker_rss_sum_gib"])
                for card in sample["gpu"]:
                    gpu = card["index"]
                    peaks[f"gpu{gpu}_mib"] = max(peaks[f"gpu{gpu}_mib"], card["used_mib"])
                    limit = 1 if card["used_mib"] > 85000 or sample["host_available_gib"] < 20 else workers_per_gpu
                    if limits[gpu] != limit:
                        m.setdefault("resource_throttling", []).append(dict(utc=utc(), gpu=gpu, new_limit=limit))
                        limits[gpu] = limit
                save_json(MANIFEST, m)
                print("RESOURCE " + json.dumps(sample), flush=True)
                last_resource = time.monotonic()
            print(f"PROGRESS tasks={len(done)}/30 folds={m['completed_folds']}/300 active={len(active)}", flush=True)
            last_refresh = time.monotonic()
        if len(done) == 30 and not active:
            break
        time.sleep(2)
    m = refresh()
    m.update(run_status="TRAINING_COMPLETE", training_completed_utc=utc(), training_wall_seconds=time.time() - m["started_unix"], failures_and_restarts=failures)
    save_json(MANIFEST, m)
    validate_complete()
    report()
    append_handoff()
    assert not live_workers()
    integrity(full=True)
    handle.close()
    shutil.rmtree(STATE)
    m = read_json(MANIFEST)
    m.update(run_status="COMPLETE", completed_utc=utc(), temporary_state_cleaned=True, no_training_processes=True,
             handoff_sha256=sha(ROOT / "handoff.md"), report_sha256=sha(RESULTS / "report.md"),
             results_sha256={n: sha(RESULTS / (n + ".csv")) for n in ["feasibility", "fold_metrics", "diagnostics", "counterfactual"]})
    save_json(MANIFEST, m)
    print("COMPLETE 30/30 tasks 300/300 folds 300 diagnostics 150 counterfactual; temporary state cleaned", flush=True)


def validate_complete():
    m = integrity(full=True)
    rows = read_csv(RESULTS / "fold_metrics.csv")
    diagnostics_rows = read_csv(RESULTS / "diagnostics.csv")
    cf = read_csv(RESULTS / "counterfactual.csv")
    keys = {(c, str(s), str(f)) for c in CONFIGS for s in range(3) for f in range(1, 11)}
    assert len(rows) == len(diagnostics_rows) == 300
    for records in (rows, diagnostics_rows):
        assert {(r["config"], r["seed"], r["fold"]) for r in records} == keys
    assert len(cf) == 150 and {(r["seed"], r["fold"], r["condition"]) for r in cf} == {
        (str(s), str(f), cond) for s in range(3) for f in range(1, 11) for cond in CONDITIONS}
    assert [sum(int(r["physical_gpu"]) == g for r in rows) for g in (0, 1)] == [150, 150]
    assert all(t["status"] == "COMPLETE" and t["folds"] == 10 for t in m["tasks"].values())
    assert m["preflight"]["status"] == "PASS"
    for row in rows:
        assert_historical_fold(row)
        assert all(math.isfinite(float(row[key])) for key in METRICS)
        assert abs(float(row["ba"]) - float(row["hard_auc"])) < 1e-12
        assert abs(float(row["acc"]) - float(row["selected_val_acc"])) < 1e-12
    for row in diagnostics_rows:
        assert float(row["residual_relative_max"]) <= .100001
        if row["config"] != "V0":
            assert float(row["residual_relative_anat_max"]) == 0 and float(row["residual_relative_func_max"]) == 0
        for key, value in row.items():
            if value and (key.startswith(("loss_", "gradient_l2_", "sampled_gradient_max_", "parameter_norm_"))):
                assert math.isfinite(float(value)), ("nonfinite diagnosis", key)
    for fold in range(1, 11):
        for seed in range(3):
            groups = {r["config"]:r for r in diagnostics_rows if int(r["fold"]) == fold and int(r["seed"]) == seed}
            assert abs(float(groups["V4"]["effective_alignment_weight_sum"]) - float(groups["V5"]["effective_alignment_weight_sum"])) < 1e-12
            for config in ("V6", "V9"):
                assert abs(float(groups[config]["effective_alignment_weight_sum"]) - float(groups["V5"]["effective_alignment_weight_sum"])) < 1e-12
    assert all(r["same_checkpoint"] == r["base_tokens_exact"] == r["parameters_unchanged"] == "True" for r in cf)
    assert all(float(r["mean_abs_asd_probability_change"]) == 0 and float(r["prediction_flip_rate"]) == 0 for r in cf if r["condition"] == "REAL")
    assert len(read_csv(RESULTS / "feasibility.csv")) == 60
    parity = {number:30 for number in (13, 14, 15, 16, 17)}
    init_matches = {c:sum(r["initial_trunk_sha256"] == next(v["initial_trunk_sha256"] for v in rows if v["config"] == "V0" and
                      v["seed"] == r["seed"] and v["fold"] == r["fold"]) for r in rows if r["config"] == c) for c in CONFIGS}
    m["final_validation"] = dict(status="PASS", groups=10, tasks=30, folds=300, diagnostics=300, counterfactual=150,
                                 gpu_folds=[150, 150], unique_results=True, historical_p0_parity=parity,
                                 checkpoint_rng_recovery=True, frozen_evidence_control_audit=True,
                                 root_history_hashes_unchanged=True, original_init_matches_v0=init_matches)
    save_json(MANIFEST, m)
    return m


def write_readme(m):
    lines = ["# 实验十八：DS³A-v3 判别性保持与可靠度驱动的选择性对齐", "",
             "固定10组×3seeds×连续10fold，主模型保留全部48/6/10/256维。无新增疾病Prompt；不做超参数搜索。", "",
             "## 实际源码、论文与历史审计", ""]
    for key in ("source", "training", "paper", "data", "history", "novelty"):
        lines += [AUDIT[key], ""]
    lines += ["## 文献核对与创新边界", ""]
    for name, entry in LITERATURE.items():
        lines += [f'- [{name}]({entry["url"]})：{entry["conclusion"]} 阅读范围：{entry["read"]}。']
    lines += ["", "MICCAI2026 PDF的shell响应为无法抽取正文的二页包装；这里只依据官方摘要比较其CSSM/SDL/CSSF，不借用未读全文的实验数据。", "",
              "## OOF证据与标签边界", "",
              "每个outer fold只构建一次：inner StratifiedKFold(5,shuffle=True,random_state=18000+fold)，StandardScaler及LogisticRegression(C=1,lbfgs,max_iter=1000)都只拟合inner train。FC/年龄性别/二者拼接/SITE/FC+SITE/FC+年龄性别+SITE六组仅记录outer-train OOF诊断。", "",
              "ASD取class0概率。证据=[pFC,pPH,2|pFC−0.5|,2|pPH−0.5|,|pFC−pPH|,预测类别相同]；q为两教师OOF均正确的0/1，仅outer train存在；outer test q=NaN且mask。标准化+相同LR仅在outer train拟合r=P(q=1|证据)。q单类退化为训练先验，manifest记录。outer test教师只用outer train拟合再预测，不读test标签。", "",
              "r为联合正确性代理，训练期冻结；未校准成医学可靠度，亦非真实对齐收益。可靠度拟合在训练OOF证据上完成，对r与q的相关/AUC是该拟合数据上的描述，不是独立可靠度验证。SITE仅作诊断，辅助模型和r估计器均不读取SITE列；原主干仍保留它。", "",
              "evidence.npz仅含10份小型OOF概率、6维证据、q/r、mask、inner fold、固定置乱与可靠度参数，不复制原始输入。含训练q的研究产物不得作为测试监督使用。", "",
              "## 固定数学", "",
              "四个Linear→GELU→LayerNorm编码器：FC256、PHENO28各有Shared32/Private32。四表示拼接128→Linear64→GELU→Linear72，生成PHENO位置0/CORR位置3的36维方向。delta=0.1*tanh(alpha)*detach(||CAM token||)*direction/sqrt(sum(direction²)+1e−8)，alpha两个零初始化标量；ANAT/FUNC残差恒0，每人每位置相对L2≤10%。注入最后FeedForward之后、原SPGS之前，原GFR/CAM/SPGS/RES/OutputLayer不改。", "",
              "Ltotal=Loriginal+0.1*enabled_disc*CE(FC Shared→Linear32/2,ytrain)+0.1*enabled_pair*mean_train(stopgrad(w)*||norm(sFC)−norm(sPH_pair)||²)+0.05*enabled_private*BCEWithLogits(FC Private→Linear32/1,FC OOF ASD软概率)。没有正交、对抗、MMD或其他辅助损失。mean分母为训练人数，绝不除权重和。", "",
              "r不进入融合输入或残差尺度。本轮检验训练期选择性约束；推理期间患者输入仍决定四编码表示和残差方向，可靠度不作为推理Gate。", "",
              "## 十组冻结矩阵", "", "|组|判别监督|对齐权重|FC教师Private蒸馏|对齐PHENO对象|新增注册参数|", "|---|---|---|---|---|---:|"]
    for config, d in DESIGNS.items():
        lines.append(f'|{config} {d.name}|{d.disc}|{d.weighting}|{d.private}|{"同类固定无自身置乱" if d.mismatched_pair else "本人"}|{m["additional_parameters"][config]}|')
    lines += ["", "V4与V5均值、总权重在数学上相同；64位浮点核对容差1e−12且记录实际误差。V6通过train索引无固定点置乱，保留r的数值集合/均值/方差；重复r值可能数值相同，不虚称所有不同患者权重数值必然不同。V9在train每类内无固定点置乱，权重及类别组成不变，仅训练对齐对象变化；分类及推理使用真实输入。", "",
              "V1–V9注册31597新增参数。disc_head66/private_head33只在各自损失启用时获得梯度；有效结构参数V1=31498，V2–V6/V9=31564，V7=31597，V8=31498。注册相同不等于实际参与相同，diagnostics同时记录结构及实测梯度。", "",
              "## 初始化、训练及恢复", "",
              "原参数先完成构造及my_weight_init；fork_rng隔离seed=180000+task seed初始化新参数，恢复原CPU/CUDA随机流。每config×seed仅播种一次，10折原顺序连续，禁止逐折重新播种。P0原8句和RN50缓存保持原函数，正式每worker首次编码一次，其首次CLIP构造所消费的CPU随机流完整保留。", "",
              "两个原Adam lr0.0038/WD0.11；新参数加入PLN Adam独立组lr0.0038/WD0。固定450epoch、wait>200停止，按val=test首次严格最高ACC checkpoint。Hard-AUC=BA，正式连续AUC用exp(log_softmax)第0类。原全局标准化和RFE、transductive图、test选模都有评价局限，不能称独立无偏泛化。", "",
              "调度器与worker锁；每25epoch及fold边界原子保存校验和、模型、双Adam、fold/epoch、best、wait、CPU/CUDA/Python/NumPy RNG、已提交结果及源码/证据签名。epoch中断先重建模型与CLIP缓存，再恢复optimizer/RNG；fold边界先加载CLIP再恢复连续随机流。已完成任务校验跳过，唯一结果来自一致checkpoint，不混未完成尝试。", "",
              "两卡各15任务/150fold，每卡最多4worker、每worker2CPU线程。资源压力仅调后续并发，不改模型、fold或epoch。临时来源副本仅六根文件与最小patch，CLIP只读链接；完成后删除临时源码/checkpoint。GPU总占用日志包含其他项目，不能当作本实验独占显存。", "",
              "```bash", "python experiment/experiment18_discriminability_preserving_selective_alignment/run.py prepare",
              "python experiment/experiment18_discriminability_preserving_selective_alignment/run.py preflight",
              "python experiment/experiment18_discriminability_preserving_selective_alignment/run.py all",
              "python experiment/experiment18_discriminability_preserving_selective_alignment/run.py resume",
              "python experiment/experiment18_discriminability_preserving_selective_alignment/run.py report", "```", "",
              "all自动完成缺失prepare/preflight；成功步骤验证跳过。COMPLETE后all/resume只验收，不自动重跑。完整另行重训需独立备份本目录结果并重置manifest/临时状态，不能把新结果混入这次冻结实验。", "",
              "## 诊断与反事实", "",
              "每fold一条诊断：最佳checkpoint eval模式损失及各已启用项梯度范数（不会更新optimizer）、训练期每25epoch和头两步采样component最大梯度、Shared/Private原型/方差/秩/距离/probe、r/q分布、alpha/参数范数及残差。线性probe固定StandardScaler+LR(C1)在outer train拟合、checkpoint确定后在test诊断；实验17用Ridge，跨轮probe数字不直接当同口径差值。原型使用未加r权重的类均值再L2归一化；17原型按Align加权，两者定义不同。V3/V8、V5/V4等本轮同口径比较更重要。", "",
              "V5同checkpoint REAL/SHARED_OFF/PRIVATE_OFF/RESIDUAL_OFF/PHENO_SHUFFLED，共150行；置乱仅新分支年龄/性别，train/test各自固定循环置乱，不用test标签；主干输入、参数、注入前CAM Token不变，最后恢复REAL逐值验证。记录概率、真实logits、翻转、六指标及残差变化。反事实证明训练后依赖，不替代重训消融；r只在损失，故无推理交换r反事实。", "",
              "成功判据保守描述：BA和Prob-AUC在三个seed均正向且ACC没有反向退化；逐seed ACC/SEN完整列出。三seed共享数据划分，30折不是30独立数据集，不做独立样本显著性推断。冻结核心比较V2−V1/V3−V2/V3−V8/V5−V4/V5−V6/V5−V9/V7−V5/V5或V7−V0。", "",
              "实验1–17和根源码哈希保护，不自动Git commit/push/pull/reset/checkout。只有全部300fold、300诊断、150反事实、P0 parity及恢复验收通过，才追加handoff并标记COMPLETE。"]
    atomic(EXP / "README.md", "\n".join(lines) + "\n")


def summary(rows):
    import numpy as np
    result = {}
    configs = sorted(set(r["config"] for r in rows), key=lambda v: (v[0], int(v[1:])))
    for config in configs:
        result[config] = {}
        for key in METRICS:
            seeds = [float(np.mean([float(r[key]) for r in rows if r["config"] == config and int(r["seed"]) == seed])) for seed in range(3)]
            result[config][key] = dict(mean=float(np.mean(seeds)), sd=float(np.std(seeds)), seeds=seeds)
    return result


def paired(rows, left, right, key):
    import numpy as np
    lookup = {(r["config"], int(r["seed"]), int(r["fold"])):float(r[key]) for r in rows}
    differences = np.array([[lookup[left, seed, fold] - lookup[right, seed, fold] for fold in range(1, 11)] for seed in range(3)])
    return dict(mean=float(differences.mean()), seeds=differences.mean(1).tolist(),
                wins=int((differences > 1e-12).sum()), ties=int((np.abs(differences) <= 1e-12).sum()), losses=int((differences < -1e-12).sum()))


def mean_field(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.


def stable_support(rows, left, right):
    major = all(all(value > 1e-12 for value in paired(rows, left, right, key)["seeds"]) for key in ["ba", "probability_auc"])
    # Conservatively require every seed's ACC to be non-decreasing, avoiding a post-hoc threshold.
    acc_ok = all(value >= -1e-12 for value in paired(rows, left, right, "acc")["seeds"])
    return major and acc_ok


def history_e9():
    folder = next((ROOT / "experiment").glob("experiment9_*"))
    return [dict(r, ba=r["balanced_accuracy"], specificity_td=r["specificity_asd"])
            for r in read_csv(folder / "results/fold_metrics.csv") if r["config"] == "E9"]


def scientific_judgment(rows):
    comparisons = [("discriminative", "V2", "V1"), ("uniform_alignment", "V3", "V2"),
                   ("discriminative_under_alignment", "V3", "V8"), ("selection_over_same_strength", "V5", "V4"),
                   ("correct_reliability", "V5", "V6"), ("patient_correspondence", "V5", "V9"),
                   ("private_distillation", "V7", "V5"), ("v5_over_p0", "V5", "V0"), ("v7_over_p0", "V7", "V0")]
    result = {name:stable_support(rows, left, right) for name, left, right in comparisons}
    with_e9 = rows + history_e9()
    result.update(v5_over_e9=stable_support(with_e9, "V5", "E9"), v7_over_e9=stable_support(with_e9, "V7", "E9"))
    result["first_module_supported_in_historical_protocol"] = all(result[k] for k in (
        "selection_over_same_strength", "correct_reliability", "patient_correspondence", "v5_over_p0"))
    result["miccai2027_full_scientific_evidence"] = False
    result["success_rule"] = "three seed BA/Prob-AUC all positive AND three seed ACC non-decreasing; conservative descriptive, not significance"
    return result


def report():
    m = validate_complete()
    rows, ds, cf = [read_csv(RESULTS / (n + ".csv")) for n in ("fold_metrics", "diagnostics", "counterfactual")]
    feasibility = read_csv(RESULTS / "feasibility.csv")
    agg = summary(rows)
    judgment = scientific_judgment(rows)
    lines = ["# 实验十八：DS³A-v3 判别性保持与可靠度驱动的选择性多模态对齐", "",
             f'训练完成：{m["training_completed_utc"]}；30/30任务、300/300真实fold、300条诊断、150条V5反事实。训练墙钟{m["training_wall_seconds"] / 60:.2f}分钟。', "",
             "## 1. 审计、研究动机与历史失败", ""]
    for key in ("source", "training", "paper", "history", "data"):
        lines += [AUDIT[key], ""]
    lines += ["实验十七的类别原型级对齐既不保证患者一一对应，也不保护Shared的判别信息；其learned Gate易整体缩小，不能从低对齐距离或患者变化推断选择有效。DS³A-v3改为训练OOF固定代理、患者级距离、FC Shared类别监督和严格同权重预算控制。零初始化≤10%残差保留，但取消新疾病语义、QC神经Gate、Shared/Private余弦正则及r条件融合。Shared/Private独立编码器不自动构成信息解耦。", "",
              "## 2. 固定结构与精确控制变量", "",
              "四个32维编码器读取FC256/年龄性别28；四表示拼接128→64→72，两36维方向以0.1*tanh(alpha)*detach(CAM范数)归一化残差注入PHENO/CORR位置，随后完整执行原SPGS/RES/分类器。r不送入融合，无推理期可靠度Gate。", "",
              "Ltotal=Loriginal+0.1*I_disc*Ldisc+0.1*I_pair*mean_train(w_i*d_i)+0.05*I_private*Lprivate。d_i=||normalize(sFC_i)−normalize(sPH_pair(i))||²；分母仅训练人数。Private软BCE目标是FC教师OOF ASD概率，准确名称是FC教师诊断信息蒸馏。", "",
              "|组|新增注册参数|判别监督|权重|蒸馏|配对|", "|---|---:|---|---|---|---|"]
    for config, design in DESIGNS.items():
        lines.append(f'|{config} {design.name}|{m["additional_parameters"][config]}|{design.disc}|{design.weighting}|{design.private}|{"train同类无自身置乱" if design.mismatched_pair else "本人"}|')
    lines += ["", "V4/V5只改样本权重分配，V5/V6保留权重数值集合，V5/V9只改训练对齐PHENO对象，V3/V8只改判别监督，V5/V7只改蒸馏。新参数init在原my_weight_init后fork_rng，原两个Adam及正则保持；原参数1462957。关闭头保留注册但无有效梯度，后文列实际结构容量。", "",
              "## 3. 训练折内可对齐性与站点诊断", "",
              "各项均为10套高度重叠outer-train内部五折OOF，不是outer-test正式指标，不作独立数据集统计推断。inner随机种子18000+outer_fold，Scaler+LR(C1/lbfgs/1000)逐inner train拟合。测试标签未进入教师/可靠度拟合。", "",
              "|输入|OOF ACC%|OOF BA%|OOF Prob-AUC%|预测ASD比例%|", "|---|---:|---:|---:|---:|"]
    feature_means = {}
    for name in ("FC", "PHENO", "FC_PHENO", "SITE", "FC_SITE", "FC_PHENO_SITE"):
        selected = [r for r in feasibility if r["feature_set"] == name]
        feature_means[name] = {k:mean_field(selected, k) for k in ["oof_acc", "oof_ba", "oof_probability_auc", "predicted_asd_fraction"]}
        lines.append(f'|{name}|' + "|".join(f'{100 * feature_means[name][k]:.4f}' for k in feature_means[name]) + "|")
    lines += ["", "增量(pp)：FC+PHENO−FC " + " / ".join(f'{k} {100 * (feature_means["FC_PHENO"][k] - feature_means["FC"][k]):+.4f}' for k in ["oof_acc", "oof_ba", "oof_probability_auc"]) + "；控制SITE后 FC+PHENO+SITE−FC+SITE " + " / ".join(f'{k} {100 * (feature_means["FC_PHENO_SITE"][k] - feature_means["FC_SITE"][k]):+.4f}' for k in ["oof_acc", "oof_ba", "oof_probability_auc"]) + "。", ""]
    improvements = [float(a["oof_probability_auc"]) - float(next(r["oof_probability_auc"] for r in feasibility if r["feature_set"] == "FC" and r["fold"] == a["fold"])) for a in feasibility if a["feature_set"] == "FC_PHENO"]
    site_improvements = [float(a["oof_probability_auc"]) - float(next(r["oof_probability_auc"] for r in feasibility if r["feature_set"] == "FC_SITE" and r["fold"] == a["fold"])) for a in feasibility if a["feature_set"] == "FC_PHENO_SITE"]
    lines += [f'FC+PHENO连续AUC增量为正的outer-train划分为{sum(v > 0 for v in improvements)}/10；加入SITE后增量为正{sum(v > 0 for v in site_improvements)}/10。PHENO单模态BA={100 * feature_means["PHENO"]["oof_ba"]:.4f}%，AUC={100 * feature_means["PHENO"]["oof_probability_auc"]:.4f}%；FC/PHENO类别一致率={100 * mean_field(feasibility, "fc_pheno_agreement"):.4f}%，ASD概率相关={mean_field(feasibility, "fc_pheno_probability_correlation"):.6f}。', "",
              "年龄/性别可包含采样与人口学关联，却不是症状量表；若其OOF增量弱，联合正确性选择可能主要挑选FC容易病例或性别/年龄常见群体，无法凭重加权创造原输入不存在的疾病信息。SITE增量说明应关注混杂，不能断定因果站点影响；辅助路径不直接读取SITE也不能宣称消除了站点相关性。全部十组按预注册继续训练，未由这些诊断选择超参。", "",
              "## 4. OOF可靠度及冻结证据", "",
              "q=两单模态OOF均正确；六证据的Scaler+LR仅outer train拟合r。train概率为严格OOF，test概率来自全outer train拟合教师；test q缺失。r训练拟合本身未再进行cross-fitting，r−q关系是拟合数据上的诊断，不能称独立校准质量或医学可靠度。", "",
              "|fold|q正例%|r均值|r SD|r p05/50/95|r>0.5%|Kish有效占比%|r−q相关|r预测q训练AUC|", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    reliability_rows = [r for r in ds if r["config"] == "V5" and r["seed"] == "0"]
    for r in reliability_rows:
        lines.append(f'|{r["fold"]}|{100 * float(r["joint_correct_fraction"]):.4f}|{float(r["reliability_mean"]):.6f}|{float(r["reliability_sd"]):.6f}|' + "/".join(f'{float(r[k]):.6f}' for k in ["reliability_p05", "reliability_p50", "reliability_p95"]) + f'|{100 * float(r["reliability_gt_half_fraction"]):.4f}|{100 * float(r["reliability_kish_effective_fraction"]):.4f}|{float(r["reliability_q_correlation"]):.6f}|{float(r["reliability_q_auc"]):.6f}|')
    max_error = max(v["v4_v5_sum_abs_error"] for v in m["preflight"]["frozen_control_evidence"].values())
    lines += ["", f'冻结证据SHA256：`{m["evidence_sha256"]}`；q单类别常数退化fold={m["evidence_protocol"]["constant_prior_folds"]}。V4/V5总量64位误差最大{max_error:.3g}（预设容差1e−12），V5/V6排序后的权重逐值相同，V5/V9权重逐值相同且train同类置乱无固定点。r存在患者差异不证明选择有益，必须结合V4/V6/V9对照。', "",
              "## 5. 全部正式六指标", "",
              "先各seed独立取十折均值，再计算3seed mean±population SD，单位%。ASD=0，连续AUC用exp(log_softmax)第0类；Hard-AUC按原硬one-hot口径，在此二分类等于BA。", "",
              "|组|ACC|BA|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        lines.append(f'|{config} {DESIGNS[config].name}|' + "|".join(f'{100 * agg[config][k]["mean"]:.4f}±{100 * agg[config][k]["sd"]:.4f}' for k in METRICS) + "|")
    lines += ["", "三项最佳：" + "；".join(f'{key}={max(CONFIGS, key=lambda config: agg[config][key]["mean"])}' for key in ["acc", "ba", "probability_auc"]) + "。仅报告排序，不据此选新设计。", "",
              "## 6. 每seed十折均值", "", "|组|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        for seed in range(3):
            lines.append(f'|{config}|{seed}|' + "|".join(f'{100 * agg[config][k]["seeds"][seed]:.5f}' for k in METRICS) + "|")
    lines += ["", "## 7. 预设配对比较", "",
              "同seed/fold差，单位pp；W/T/L为逐折胜/平/负，1e−12仅用于浮点平局识别。三seed共用10套划分且后续折因epoch不同RNG自然分叉，不把30折视为30独立数据集，不宣称统计显著。", "",
              "|比较|指标|均值差pp|seed0/1/2差pp|W/T/L|", "|---|---|---:|---|---|"]
    for left, right in PAIRS:
        for key in METRICS:
            p = paired(rows, left, right, key)
            lines.append(f'|{left}−{right}|{key}|{100 * p["mean"]:+.5f}|' + "/".join(f'{100 * v:+.5f}' for v in p["seeds"]) + f'|{p["wins"]}/{p["ties"]}/{p["losses"]}|')
    lines += ["", "## 8. 判别性、Shared/Private信息与坍塌", "",
              "最佳checkpoint之后记录；类别原型为训练未加权类均值再归一化，cos趋近1可与信息弱化并存，但单个cos不充分证明所有信息消失。probe固定Scaler+LR训练折拟合、测试折诊断，不参与选模；实验17用Ridge且原型按Align加权，因此历史probe/原型数值不是完全相同口径。当前V3/V8及V2/V1比较才是主要保护证据。", "",
              "|组|FC/PH Shared类原型cos(train)|FC Shared BA/AUC%|FC Private BA/AUC%|PH Shared BA/AUC%|PH Private BA/AUC%|FC/PH Shared-Private abs cos|", "|---|---|---|---|---|---|---|"]
    grouped = {c:[r for r in ds if r["config"] == c] for c in CONFIGS}
    for config in CONFIGS[1:]:
        selected = grouped[config]
        line = f'|{config}|' + "/".join(f'{mean_field(selected, k):.6f}' for k in ["shared_corr_class_prototype_cosine_train", "shared_pheno_class_prototype_cosine_train"]) + "|"
        for key in ("shared_corr", "private_corr", "shared_pheno", "private_pheno"):
            line += "/".join(f'{100 * mean_field(selected, key + suffix):.4f}' for suffix in ["_probe_ba", "_probe_probability_auc"]) + "|"
        lines.append(line + "/".join(f'{mean_field(selected, "shared_private_abs_cosine_" + key):.6f}' for key in ["corr", "pheno"]) + "|")
    for left, right in [("V2", "V1"), ("V3", "V8"), ("V7", "V5")]:
        lines += ["", f'{left}−{right}机制均值差：FC Shared类原型cos {mean_field(grouped[left], "shared_corr_class_prototype_cosine_train") - mean_field(grouped[right], "shared_corr_class_prototype_cosine_train"):+.6f}；FC Shared probe BA {100 * (mean_field(grouped[left], "shared_corr_probe_ba") - mean_field(grouped[right], "shared_corr_probe_ba")):+.4f}pp；FC Private probe BA {100 * (mean_field(grouped[left], "private_corr_probe_ba") - mean_field(grouped[right], "private_corr_probe_ba")):+.4f}pp。']
    lines += ["", "|组|FC Shared类cos>0.99折数|PH Shared类cos>0.99折数|跨模态Shared平方距离|FC Shared同/异类距离|FC Shared/Private秩|PH Shared/Private秩|", "|---|---:|---:|---:|---|---|---|"]
    for config in CONFIGS[1:]:
        selected = grouped[config]
        lines.append(f'|{config}|{sum(float(r["shared_corr_class_prototype_cosine_train"]) > .99 for r in selected)}/30|{sum(float(r["shared_pheno_class_prototype_cosine_train"]) > .99 for r in selected)}/30|{mean_field(selected, "shared_cross_modal_distance_train"):.6f}|' + "/".join(f'{mean_field(selected, k):.6f}' for k in ["shared_corr_same_class_distance_train", "shared_corr_different_class_distance_train"]) + "|" + "/".join(f'{mean_field(selected, k):.4f}' for k in ["shared_corr_effective_rank", "private_corr_effective_rank"]) + "|" + "/".join(f'{mean_field(selected, k):.4f}' for k in ["shared_pheno_effective_rank", "private_pheno_effective_rank"]) + "|")
    lines += ["", "Shared/Private低cos只说明方向不同；高Private probe只说明该表示可线性读出诊断关联，不证明融合净收益。V7必须同时结合V7−V5正式性能和Private probe增量解释，不能以蒸馏BCE下降代替保护成功。", "",
              "## 9. 对齐强度、残差与梯度", "",
              "loss及其梯度是最佳checkpoint eval模式的固定几何，未执行optimizer更新；训练期component梯度每25epoch及前两步采样，记录最大值而非每epoch重复统计。", "",
              "|组|原分类loss|原图loss|disc原始|pair原始/加权|Private原始|权重均值/SD|PH/FC残差%|最大单Token残差%|", "|---|---:|---:|---:|---|---:|---|---|---:|"]
    for config in CONFIGS:
        selected = grouped[config]
        lines.append(f'|{config}|{mean_field(selected, "loss_classification"):.6f}|{mean_field(selected, "loss_original_graph"):.6f}|{mean_field(selected, "disc_raw_loss"):.6f}|' + "/".join(f'{mean_field(selected, k):.6f}' for k in ["pair_raw_loss", "pair_weighted_loss"]) + f'|{mean_field(selected, "private_raw_loss"):.6f}|' + "/".join(f'{mean_field(selected, k):.6f}' for k in ["effective_alignment_weight_mean", "effective_alignment_weight_sd"]) + "|" + "/".join(f'{100 * mean_field(selected, k):.4f}' for k in ["residual_relative_pheno_mean", "residual_relative_corr_mean"]) + f'|{100 * max(float(r["residual_relative_max"]) for r in selected):.4f}|')
    lines += ["", "|组|分类梯度|原图梯度|0.1disc梯度|0.1pair梯度|0.05private梯度|有效新增结构参数|近零目标Token%|alpha PH/FC均值|", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for config in CONFIGS:
        selected = grouped[config]
        lines.append(f'|{config}|' + "|".join(f'{mean_field(selected, "gradient_l2_" + name):.6f}' for name in ["classification", "original_graph", "discriminative", "pair_alignment", "private_distillation"]) + f'|{mean_field(selected, "structurally_active_auxiliary_parameters"):.0f}|{100 * mean_field(selected, "residual_near_zero_selected_fraction"):.4f}|' + "/".join(f'{mean_field(selected, "alpha_" + name):+.6f}' for name in ["pheno", "corr"]) + "|")
    lines += ["", "新增alpha有符号；其初始范数为0，norm-ratio仅数值记账，不解释为权重收缩率。ANAT/FUNC新残差严格0，注入PHENO/CORR位置已经混入CAM上下文，其大小不能作为纯模态因果贡献或真实ROI证据。全部预测/loss/梯度有限。训练期实测component梯度与头启用状态吻合，disabled_head_components不计有效容量。参数范数与方差的完整数字见单条fold诊断。", "",
              "## 10. V5同checkpoint五条件反事实", "",
              "不重训、不重新选epoch、主干参数及注入前Token逐值不变；置乱仅新增年龄/性别输入且不用测试标签，结束恢复REAL逐值一致。r不进入推理，未做推理交换r。", "",
              "|条件|ΔACC pp|ΔBA pp|ΔProb-AUC pp|翻转%|ASD概率绝对变化|logit绝对变化|残差绝对变化|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        selected = [r for r in cf if r["condition"] == condition]
        lines.append(f'|{condition}|' + "|".join(f'{mean_field(selected, key) * scale:.8f}' for key, scale in [
            ("delta_acc", 100), ("delta_ba", 100), ("delta_probability_auc", 100), ("prediction_flip_rate", 100),
            ("mean_abs_asd_probability_change", 1), ("mean_abs_logit_change", 1), ("delta_mean_abs_change", 1)]) + "|")
    lines += ["", "这些只检验训练后分支依赖；OFF下降可能因为主干/分类器共同适应，不能代替V0重训净增益。Shared/Private输入清零带来分布变化，PHENO错配敏感也不自动说明真实对应提高算法性能。患者对应是否有净价值主要看V5−V9重训对照。", "",
              "## 11. 预设科学判断", "",
              "BA和Prob-AUC三个seed均正向，且采用更保守的三个seed ACC均不下降条件才记稳定支持；这是描述性规则，无统计显著性含义。所有SEN/SPE差已列，提升整体ACC不能掩盖ASD漏检。", ""]
    scientific_pairs = [("A 判别性保护", "V2", "V1", "discriminative"), ("B 统一对齐增量", "V3", "V2", "uniform_alignment"),
                        ("C 对齐时判别保护", "V3", "V8", "discriminative_under_alignment"),
                        ("D 同强度患者选择", "V5", "V4", "selection_over_same_strength"),
                        ("E 正确可靠度分配", "V5", "V6", "correct_reliability"),
                        ("F 真实患者配对", "V5", "V9", "patient_correspondence"),
                        ("G Private教师蒸馏", "V7", "V5", "private_distillation"),
                        ("H1 V5超过P0", "V5", "V0", "v5_over_p0"), ("H2 V7超过P0", "V7", "V0", "v7_over_p0")]
    for name, left, right, key in scientific_pairs:
        lines += [f'- {name}（{left}−{right}）：稳定支持={judgment[key]}；ACC/BA/Prob-AUC均值差 ' + "/".join(f'{100 * paired(rows, left, right, metric)["mean"]:+.5f}pp' for metric in ["acc", "ba", "probability_auc"]) + "。"]
    lines += ["", "如果V5仅超过V3却不能超过V4，应优先考虑整体弱化对齐而非患者选择；若V5未稳定超过V6/V9，本轮分别不能支持正确可靠度或真实患者对应的独立价值。这并不证明总体效应为零，但不足以支持预设机制。", "",
              "## 12. 与P0及历史E9的比较", ""]
    e9 = history_e9()
    old = summary(e9)["E9"]
    lines += ["历史E9真实30折均值：" + "/".join(f'{k} {100 * old[k]["mean"]:.5f}%' for k in METRICS) + "。其P3及随机轨迹不同，只作同数据划分上的历史描述性对照。", "",
              "|组−E9|指标|均值差pp|seed0/1/2差pp|W/T/L|", "|---|---|---:|---|---|"]
    for config in CONFIGS:
        for key in METRICS:
            p = paired(rows + e9, config, "E9", key)
            lines.append(f'|{config}−E9|{key}|{100 * p["mean"]:+.5f}|' + "/".join(f'{100 * v:+.5f}' for v in p["seeds"]) + f'|{p["wins"]}/{p["ties"]}/{p["losses"]}|')
    lines += ["", "## 13. 运行、恢复与完整验收", "",
              f'GPU0/GPU1分别15任务、150fold；每卡最多4worker、每worker2CPU线程；正式墙钟{m["training_wall_seconds"] / 60:.2f}分钟。任务如下：', ""]
    for gpu in (0, 1):
        lines += [f'GPU{gpu}：' + ", ".join(f'{c}_seed{s}' for c, s, g in TASKS if g == gpu) + "。", ""]
    lines += [f'资源峰值：{json.dumps(m.get("runtime_resource_peaks", {}), ensure_ascii=False)}。GPU峰值是整卡含其他项目；worker RSS为本实验进程驻留内存加总。失败/恢复：{json.dumps(m.get("failures_and_restarts", {}))}；调度限流：{json.dumps(m.get("resource_throttling", []))}。', "",
              "P0逐折六指标及best/executed epoch与实验13F0/14E00/15A0/16B0/17D0全部30折一致；历史对应初始化序列化、14–17RNG也核对。原前向、原两步Adam及RNG、新零alpha初始化、完整第三步恢复、权重0.03比例、训练标签边界和所有五反事实均预检通过。", "",
              "不同组与V0相同初始主干hash的fold数：" + json.dumps(m["final_validation"]["original_init_matches_v0"]) + "。由于组间epoch差异导致下一fold随机流分叉，不能把所有fold解释为严格相同初始化配对。", "",
              "完整源、数据、CLIP、证据及旧实验保护hash见manifest；当前300唯一折、300诊断、150反事实、两卡各150折和同强度冻结证据验收通过。完成控制器随后安全追加handoff、清理本实验临时源码/权重，最后标记COMPLETE。未自动commit/push。", "",
              "## 14. 文献边界、MICCAI2027与下一阶段", "", AUDIT["novelty"], ""]
    for name, entry in LITERATURE.items():
        lines += [f'- [{name}]({entry["url"]})：{entry["conclusion"]} 阅读范围：{entry["read"]}。']
    lines += ["", f'本历史协议内第一模块支持标准（患者选择+正确权重+真实对应+P0净收益同时成立）={judgment["first_module_supported_in_historical_protocol"]}。完整MICCAI2027科学证据要求仍为未达到：当前test选checkpoint、全队列标准化/历史RFE、transductive评估及无独立站点/队列验证限制外部泛化结论；组件组合本身也不是独立创新。', "",
              "若以上机制未获支持，接受负结果、保留P0，不事后改Prompt或搜索损失系数。若局部指标改善而核心配对不稳定，同样不包装成患者级选择有效。下一步最值得研究的问题是：具有真实疾病症状信息的临床表型，是否在训练内选模和独立站点评估中提供超过FC的增量，并能预测哪些患者确实从跨模态对齐获益。当前年龄/性别联合正确性不等于该收益目标。"]
    atomic(RESULTS / "report.md", "\n".join(lines) + "\n")
    m["scientific_judgment"] = judgment
    m["aggregated_metrics"] = agg
    save_json(MANIFEST, m)
    print("REPORT_WRITTEN " + str(RESULTS / "report.md"), flush=True)


def append_handoff():
    m = read_json(MANIFEST)
    assert m["final_validation"]["status"] == "PASS"
    path = ROOT / "handoff.md"
    marker = "<!-- experiment18_discriminability_preserving_selective_alignment -->"
    with open(ROOT / "handoff.md.lock", "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        content = path.read_bytes()
        prefix_size = m["handoff_prefix_bytes"]
        assert hashlib.sha256(content[:prefix_size]).hexdigest() == m["handoff_prefix_sha256"]
        if marker.encode() not in content:
            block = "\n\n" + marker + "\n\n## 实验十八：DS³A-v3正式完成\n\n"
            block += f'本地源码版本HEAD `{m["git_head_at_prepare"]}`；实验目录 `{EXP}`；完整源/数据/证据签名 `{MANIFEST}`。30任务、300真实fold、300诊断、150反事实已验收；双GPU各150fold。\n\n'
            block += "研究动机、数学控制、OOF可靠度、六指标和种子稳定性、判别保护、同强度/错配权重/错配患者比较、Private蒸馏、反事实、负结果及MICCAI2027判断如下，旧内容完整保留。\n\n"
            block += (RESULTS / "report.md").read_text()
            with open(path, "ab") as output:
                output.write(block.encode())
                output.flush()
                os.fsync(output.fileno())
    print("HANDOFF_APPENDED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=["prepare", "preflight", "all", "resume", "report", "refresh"])
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "SEED"))
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
    elif args.action == "prepare":
        prepare()
    elif args.action == "preflight":
        prepare()
        preflight()
    elif args.action in ("all", "resume"):
        schedule(args.workers_per_gpu)
    elif args.action == "report":
        report()
    elif args.action == "refresh":
        print(json.dumps(refresh(), ensure_ascii=False)[:1600])
    else:
        parser.error("action or --worker is required")


if __name__ == "__main__":
    main()


