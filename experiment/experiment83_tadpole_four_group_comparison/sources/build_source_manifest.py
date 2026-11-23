import hashlib,json,subprocess
from pathlib import Path
E=Path('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison')

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def files(rel):
 d=E/'sources'/rel
 return {str(p.relative_to(E/'sources')):sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
def git_hash(path,commit='ccca6484de8e0c4f448c314a4863e7b56f46208c'):
 try:return subprocess.check_output(['git','rev-parse',commit],cwd='/data/adodas/SPromptGL-main',text=True).strip()
 except:return commit
m={
 'experiment':'Experiment 83',
 'generated_utc':'2026-11-23T16:00:00Z',
 'repository':'https://github.com/tangaii/SPromptGL',
 'groups':{
  'G1_E28_E0':{
   'selected_method':'E0_P0 original SPromptGL baseline from E28, with original PEN/GFR/CAM chain',
   'original_directory':'experiment/experiment28_imaging_anchored_finegrained_semantic_alignment',
   'branch':'local historical checkout, manifest source_remote_head 8217b36bfe7f51d878b793c070774129d11cad73',
   'commit':'1167dd392837a02eb4f83481ae49f02607a9dae5',
   'core_files':['run.py','e28_method.py','e28_integration.py','e28_prompts.py'],
   'copied_hashes':files('E28_E0_original'),
   'adaptation':'Dynamic six-modality graph and token dimensions, TADPOLE three-class classifier, fixed historical eight semantic prompts, no ABIDE checkpoint or fold reuse. E0_P0 selected explicitly.'
  },
  'G2_E43_1_E00':{
   'selected_method':'E00_B0B_REPLAY / physical B0-B baseline with PEN removed',
   'original_directory':'experiment/experiment43_1_b0b_gfr_mechanism_retest',
   'branch':'local git history commit ccca6484de8e0c4f448c314a4863e7b56f46208c',
   'commit':'ccca6484de8e0c4f448c314a4863e7b56f46208c',
   'core_files':['run.py','method.py','integration.py','audit.py','config.json','manifest.json'],
   'copied_hashes':files('E43_1_E00_original'),
   'adaptation':'Physical no-PEN path retained. Dynamic six-modality dimensions, TADPOLE three-class labels and common split protocol.'
  },
  'G3_E74_4_G2':{
   'selected_method':'E74.4-G2-Positive-DCR, treated as requested DARG lineage',
   'original_directory':'experiment/experiment74_4_soft_positive_dcr',
   'branch':'upload/experiment74-4-soft-positive-dcr',
   'commit':'711d87f4ccc9e11124bce62751fcd3dbb97fdbdf',
   'upstream_branches':{'E74.1':'upload/experiment74-1-rudag d1522a23682b62e41d899d95c9721a68d3828456','E74.2':'upload/experiment74-2-dcr-rudag 6555308455814af5b6b4c82ceece73eb2333be15','E74.3':'upload/experiment74-3-local-dcr e05275cb4b18dcd01679682b9df02f3cbc13b713'},
   'core_files':['run.py','soft_positive_dcr.py','config.json','manifest.json','audit.py'],
   'copied_hashes':files('E74_4_G2_original'),
   'adaptation':'Positive-only disease-aware relation residual and graph fusion retained. ABIDE four-block slices and overlap IDs replaced by six TADPOLE modality blocks and train-only three-class same/different relation supervision.'
  },
  'G4_E82_4_6_10_G1':{
   'selected_method':'E82.4.6.10-G1 incremental wrapper traced to E82.4.6 PCSR-PSCE / SCMR-DA-SRPE DPSL chain',
   'original_directory':'experiment82_4_6_10_fold8_dual_asd',
   'branch':'codex/experiment82-4-6-10-fold8-dual-asd-upload',
   'commit':'1ef9a17fb50585aad9b2f878018ded16646a19a7',
   'upstream_branch':'codex/experiment82-4-6-pcsr-scmr-upload',
   'upstream_commit':'f1e0697d06eeff160cd93920a419dcf4dbec1a47',
   'core_files':['core.py','sources.py','execute.py','audit_tasks.py'],
   'dependency_files':['g1.py','g2.py','common/spectral.py','common/evidence.py','common/history.py','common/optimization.py','runtime/prompts.pt'],
   'copied_hashes':files('E82_4_6_10_G1_original')|files('E82_dependencies'),
   'adaptation':'DARG to CAM to DPSL to classifier order retained. Binary ASD margin and ABIDE overlap additions are not copied. The eight historical fixed prompts and prompt embeddings are copied from the E82 runtime artifact, while spectral basis width and classifier margins are generalized to TADPOLE CAM width 240 and three classes.'
  }
 },
 'common_base_sources':{
  'files':files('base_original'),
  'original_core_commit':'1167dd392837a02eb4f83481ae49f02607a9dae5',
  'notes':'Original local SPromptGL core files copied for audit only. E83 executes isolated adapters in common/.'
 },
 'provenance_rules':[
  'No historical ABIDE prediction, checkpoint, fold, patient ID or overlap list is used.',
  'All copied source files are hashed in this manifest.',
  'E83 model implementation is isolated under common/ and does not modify historical experiment directories.'
 ]
}
(E/'sources/source_manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False))
print('wrote',E/'sources/source_manifest.json')
