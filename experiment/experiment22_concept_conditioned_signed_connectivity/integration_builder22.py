"""Experiment22 exact surgical integration with Experiment9 accepted sources.

Input: dictionary CORE filename -> Experiment9-patched source. Output: final
source dictionary. Runner writes unified diff against ROOT originals and applies
it to disposable per-worker workspaces, never modifying ROOT.
"""
import ast


def patched_sources22(src):
    src=dict(src)
    model,network=src['model.py'],src['network.py']
    # Existing Exp9 write_pdai_outputs is an unused side-effect/report writer;
    # the independent runner owns metrics. Preserve all model computations.
    lines=model.splitlines(keepends=True)
    found=0
    for node in reversed(ast.parse(model).body):
        if isinstance(node,ast.ClassDef) and node.name=='EvalHelper':
            for fn in reversed(node.body):
                if isinstance(fn,ast.FunctionDef) and fn.name=='write_pdai_outputs':
                    del lines[fn.lineno-1:fn.end_lineno]
                    found+=1
    assert found==1,'Experiment9 reporter anchor changed'
    model=''.join(lines)
    old='    def forward(self, x):\n        bs = x.size(0)'
    assert network.count(old)==1
    network=network.replace(old,'    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)')
    anchor='        text_features = self.prompt_text_features\n'
    assert network.count(anchor)==1
    network=network.replace(anchor,
        "        if hasattr(self, 'evidence22'):\n"
        "            x = self.evidence22(x, raw_modalities)\n"+anchor)
    entry='    def run_epoch(self, mode, iteration_MF=10, iteration_GC=10, end = \'\'):'
    assert model.count(entry)==1
    model=model.replace(entry,
        '        from e22_runtime import install_evidence22\n'
        '        install_evidence22(self)\n\n'+entry)
    lines=model.splitlines(keepends=True)
    tree=ast.parse(model)
    found_call=0;found_loss=0;found_eval=0
    for node in tree.body:
        if not isinstance(node,ast.ClassDef) or node.name!='EvalHelper':continue
        for fn in node.body:
            if not isinstance(fn,ast.FunctionDef):continue
            if fn.name=='run_epoch':
                lo=next(i for i in range(fn.lineno-1,fn.end_lineno) if "elif mode == 'simple-2':" in lines[i])
                hi=next(i for i in range(lo+1,fn.end_lineno) if "elif mode == 'normal':" in lines[i])
                for i in range(lo,hi):
                    if 'self.ModalFusion(feat)' in lines[i]:
                        lines[i]=lines[i].replace('self.ModalFusion(feat)', 'self.ModalFusion(feat, self._e22_raw)')
                        found_call+=1
                    if lines[i].strip()=='loss = cls_loss + all_graph_loss':
                        # All new groups share same bounded adaptation loss.
                        # Existing E9 auxiliary disease CE is added by the Exp9 patch.
                        lines[i]+=("            branch = getattr(self.ModalFusion, 'evidence22', None)\n"
                                   "            if branch is not None:\n"
                                   "                loss = loss + branch.auxiliary_loss()\n")
                        found_loss+=1
            elif fn.name=='_print_acc':
                for i in range(fn.lineno-1,fn.end_lineno):
                    if 'self.ModalFusion(feat)' in lines[i]:
                        lines[i]=lines[i].replace('self.ModalFusion(feat)', 'self.ModalFusion(feat, self._e22_raw)')
                        found_eval+=1
    assert (found_call,found_loss,found_eval)==(1,1,2),(found_call,found_loss,found_eval)
    model=''.join(lines)
    ast.parse(model);ast.parse(network)
    assert model.count('install_evidence22(self)')==1
    assert model.count('loss = loss + branch.auxiliary_loss()')==1
    assert network.count('self.evidence22(x, raw_modalities)')==1
    src['model.py'],src['network.py']=model,network
    return src

