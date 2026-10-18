"""Minimal exact patch for Experiment23 runner's disposable root source copies.

Do not apply to repository root. Build once in experiment local _runtime_sources.
"""
from pathlib import Path
import ast


def patch_model23(original: str) -> str:
    # Require exact baseline anchors; no fuzzy find and replace.
    assert original.count('        self.ModalFusion.apply(my_weight_init)') == 1
    assert original.count("        elif mode == 'simple-2':") == 1
    assert original.count('            prob,_,_ = self.ModalFusion(feat)') == 1
    assert original.count('        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)') == 1
    # This is a *copy* of the original SPromptGL model.py.
    original=original.replace('from network import *\n',
                              'from network import *\nimport e23_runtime\n',1)
    original=original.replace('        self.ModalFusion.apply(my_weight_init)',
                              '        self.ModalFusion.apply(my_weight_init)\n'
                              '        e23_runtime.install_branch23(self)',1)
    # In exact simple-2 training only. All other historical paths untouched.
    old='            prob,_,_ = self.ModalFusion(feat)'
    new=("            if hasattr(self.ModalFusion, 'bridge23'):\n"
         "                feat = self.ModalFusion.bridge23(feat)\n"
         '            prob,_,_ = self.ModalFusion(feat)')
    original=original.replace(old,new,1)
    # In actual _print_acc used by train validation and test, one injection.
    old='        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)'
    new=(old+'\n'+
         "        if hasattr(self.ModalFusion, 'bridge23'):\n"
         '            feat = self.ModalFusion.bridge23(feat)')
    original=original.replace(old,new,1)
    ast.parse(original)
    assert original.count("feat = self.ModalFusion.bridge23(feat)")==2
    return original


def emit_sources(root: Path, output: Path) -> None:
    """Generate exact-copy local sources; SHA checks live in the runner."""
    root=Path(root); output=Path(output)
    assert output.resolve()!=root.resolve()
    output.mkdir(parents=True,exist_ok=True)
    names=['model.py','network.py','layers.py','utils.py']
    for n in names:
        text=(root/n).read_text()
        if n=='model.py':text=patch_model23(text)
        dest=output/n
        dest.write_text(text)
    print('Experiment23 disposable sources generated:',output)


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:raise SystemExit('Usage: python e23_patch.py ROOT OUT')
    emit_sources(Path(sys.argv[1]),Path(sys.argv[2]))

