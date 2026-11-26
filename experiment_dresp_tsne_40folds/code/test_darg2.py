import sys, importlib.util, json
from pathlib import Path
V2=Path('/model/spromptgl/experiment/experiment_abide_authentic_logs_v2');DPSL=Path('/model/spromptgl/experiment/DPSL');DARG=Path('/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr')
spec=importlib.util.spec_from_file_location('td',V2/'code/train_darg_e744.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
mod=m.load_runner();mod.RT=V2/'DARG/runtime';mod.TASKS=mod.RT/'tasks';mod.CKPTS=mod.RT/'checkpoints';mod.LOGS=mod.RT/'logs';
np=__import__('numpy');torch=__import__('torch');hist=m.install_history_path_adapter(mod,torch,np);base=hist.load_e69().load_e61().load_base(); torch.cuda.set_device(0);dcr=mod.load_dcr();dcr.CADR_PATH=V2/'inputs/source_repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py'
state,warm,ident=mod.prepare(1);state.update(fold=1,group=m.GROUP,rudag_identity=ident);p=torch.load(mod.CKPTS/f'{m.GROUP}_f01.best.pt',map_location='cuda',weights_only=False);hist.load_bundle(state,p['best_state']);print('DEBUG',state['cam'].__class__,state['cam'].__class__.__module__,mod.load_dcr().__file__)
out=mod.forward(state,m.GROUP,training=False,collect=True);print(out.keys(),out['fused'].shape);q=state['cam'](out['fused']);print([x.shape if hasattr(x,'shape') else type(x) for x in q]);print('info',out['info'].keys())
