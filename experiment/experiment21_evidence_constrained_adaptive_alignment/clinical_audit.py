"""ABIDE clinical availability audit with strict, fail-closed subject alignment.
No patient-level identifying output; no guesses from diagnosis, age, sex or site.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Representative candidate families. Final columns must come from original data dictionary.
DOMAINS = {
    'social': ['ADOS_G_SOCIAL','ADOS_2_SOCAFFECT','ADI_R_SOCIAL_TOTAL_A','SRS_SOCIAL_COGNITION_T','SRS_COMMUNICATION_T'],
    'repetitive': ['ADOS_G_STEREO_BEHAV','ADOS_2_RRB','ADI_R_RRB_TOTAL_C','SRS_MANNERISMS_T'],
    'executive_attention': ['SRS_COGNITION_T','BRIEF_GEC_T','BRIEF_WORKING_T','CBCL_6-18_ATTENTION_T'],
    'sensory': ['ADI_R_C4_UNUSUAL_SENSORY_INTERESTS'],
}


def audit(root: Path, phenotype_csv: Path, subject_ids_txt: Path, output_json: Path):
    import json
    root,phenotype_csv,subject_ids_txt=map(Path,(root,phenotype_csv,subject_ids_txt))
    summary={'status':'NOT_JOINABLE','subject_id_source':str(subject_ids_txt),'phenotype_source':str(phenotype_csv)}
    if not phenotype_csv.is_file() or not subject_ids_txt.is_file():
        summary['reason']='Exact phenotype CSV or subject_IDs.txt unavailable'
    else:
        ph=pd.read_csv(phenotype_csv,low_memory=False)
        if 'SUB_ID' not in ph or 'DX_GROUP' not in ph or 'SITE_ID' not in ph:
            summary['reason']='Missing required columns'
        else:
            ids=np.loadtxt(subject_ids_txt,dtype=str).reshape(-1)
            norm=lambda values:pd.to_numeric(pd.Series(values),errors='coerce')
            ids_num=norm(ids)
            ph_num=norm(ph['SUB_ID'])
            if len(ids)!=871 or ids_num.isna().any() or not ids_num.is_unique:
                summary['reason']='Subject IDs not exactly 871 unique integers'
            else:
                ph=ph.assign(__id=ph_num)
                cohort=ph[ph['__id'].isin(ids_num)].copy()
                if len(cohort)!=871 or cohort['__id'].duplicated().any() or not set(ids_num)==set(cohort['__id']):
                    summary['reason']='Phenotypic cohort cannot be matched bijectively to subject IDs'
                else:
                    # CRITICAL: Notebook uses df[SUB_ID.isin(ids)], preserving phenotype-CSV order.
                    # Subject IDs alone do NOT prove that this equals processed_standard_data row order.
                    summary['status']='IDS_MATCH_ONLY'
                    summary['reason']='Must additionally reproduce and numerically verify original 64 PHENO/QC columns row-by-row before joining.'
                    summary['n_subjects']=len(cohort)
                    cols=sorted({s for vals in DOMAINS.values() for s in vals if s in cohort.columns})
                    summary['candidate_columns_present']=cols
                    missing=['','-9999','-999','nan','NaN','None','n/a','N/A']
                    stats=[]
                    for field in cols:
                        v=pd.to_numeric(cohort[field].replace(missing,np.nan),errors='coerce')
                        for dx in sorted(cohort['DX_GROUP'].dropna().unique(),key=str):
                            rows=cohort['DX_GROUP']==dx
                            n=int(rows.sum());nonmissing=int(v[rows].notna().sum())
                            sites=int(cohort.loc[rows & v.notna(),'SITE_ID'].nunique())
                            stats.append(dict(field=field,dx_group=str(dx),subjects=n,nonmissing=nonmissing,
                                              fraction=nonmissing/max(n,1),sites_with_data=sites))
                    summary['coverage']=stats
                    summary['clinical_training_allowed']=False
                    summary['audit_only']=True
                    summary['pending']='Reproduce exact notebook PHENO/QC transformations and verify 871x64 row-aligned equality.'
    output_json=Path(output_json);output_json.parent.mkdir(parents=True,exist_ok=True)
    output_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phenotype',required=True)
    p.add_argument('--subject-ids',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(audit(a.root,a.phenotype,a.subject_ids,a.output))


def full_audit(root, out):
    """Bounded source search; preserve notebook row order; export aggregates only."""
    import json,hashlib,os
    from sklearn.preprocessing import StandardScaler
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    phenotype=root/'dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
    found=[];searched=[]
    for base in [root/'data',root/'dataset',Path('/DATA/wanxixi/GATE-main/ABIDE_pcp')]:
        searched.append(str(base))
        if not base.exists():continue
        for parent,dirs,files in os.walk(base):
            if len(Path(parent).relative_to(base).parts)>=4:dirs[:]=[]
            if 'subject_IDs.txt' in files:found.append(Path(parent)/'subject_IDs.txt')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    s=dict(status='NOT_JOINABLE',clinical_alignment='CLINICAL_ALIGNMENT_NOT_ESTABLISHED',clinical_training_allowed=False,
        phenotype_source=str(phenotype),phenotype_sha256=digest(phenotype) if phenotype.exists() else None,
        phenotype_origin='Local existing file; official distribution https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv',
        bounded_search=searched,subject_id_files=[dict(path=str(p),sha256=digest(p)) for p in found],
        row_identity_verified=False,verified_processed_rows=0,prefix_comparison='NOT_PERFORMED_NO_VERIFIABLE_IDS',
        dictionary_url='https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf',
        missing_policy='Blank/NaN are missing. Scores outside field-specific documented nonnegative range (including -9999) are counted INVALID separately, not generically all negative fields or invented -999 codes.',
        source_coverage_scope='UNMATCHED_1112_SOURCE_NOT_871_COHORT',symptom_supervision='UNUSABLE_FOR_CURRENT_COHORT')
    if not phenotype.exists():
        s['reason']='Phenotype unavailable';(out/'clinical_audit.json').write_text(json.dumps(s,ensure_ascii=False,indent=2));return s
    ph=pd.read_csv(phenotype,low_memory=False);s.update(source_rows=len(ph),source_unique_sub_ids=int(ph.SUB_ID.nunique()))
    cohort=None
    anat=['anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr']
    func=['func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality','func_mean_fd','func_num_fd','func_perc_fd','func_gsr']
    prefix_tests=[]
    for p in found:
        ids=pd.to_numeric(pd.Series(np.loadtxt(p,dtype=str).reshape(-1)),errors='coerce')
        if len(ids)!=871 or ids.isna().any() or not ids.is_unique:continue
        candidate=ph[ph.SUB_ID.isin(ids)].copy()
        if len(candidate)!=871 or not candidate.SUB_ID.is_unique:continue
        d=candidate.copy();minimum,maximum=d.AGE_AT_SCAN.min(),d.AGE_AT_SCAN.max();bins=[minimum];block=minimum
        while block<maximum:block+=2;bins.append(block)
        d['AGE_AT_SCAN']=pd.cut(d.AGE_AT_SCAN,bins,right=False)
        for key in ['AGE_AT_SCAN','SITE_ID','SEX']:d=pd.get_dummies(d,columns=[key])
        cols=list(d.columns[d.columns.str.contains('^SITE|^AGE_AT_SCAN|^SEX')])+anat+func
        d[anat+func]=StandardScaler().fit_transform(d[anat+func])
        target=pd.read_csv(root/'data/ABIDE/processed_standard_data.csv').iloc[:,:64]
        same_columns=cols==target.columns.tolist();same_shape=d[cols].shape==target.shape
        error=float(np.max(np.abs(d[cols].to_numpy(float)-target.to_numpy(float)))) if same_shape else None
        passed=same_columns and same_shape and np.allclose(d[cols].to_numpy(float),target.to_numpy(float),rtol=1e-7,atol=1e-7)
        prefix_tests.append(dict(subject_source=str(p),same_columns=same_columns,same_shape=same_shape,max_error=error,passed=bool(passed),roundoff_atol=1e-7))
        if passed:cohort=candidate;s.update(status='ROW_IDENTITY_VERIFIED',row_identity_verified=True,verified_processed_rows=871,prefix_comparison='871x64_MATCH_NOTEBOOK_ORDER',source_coverage_scope='VERIFIED_871_COHORT')
    s['prefix_tests']=prefix_tests
    s['reason']='No original subject_IDs.txt found in bounded source directories; cannot link 871 processed rows to source clinical records' if not found else 'No subject file passed independent 871x64 notebook reconstruction'
    population=cohort if cohort is not None else ph
    # Data dictionary defines these score families as nonnegative; other negative metadata is not masked.
    score_columns=[c for c in ph if c.startswith(('ADOS','ADI_','ADI_RRB','SRS_','SCQ_','BRIEF_','CBCL_'))]
    metadata={'ADOS_MODULE','ADOS_RSRCH_RELIABLE','ADI_R_RSRCH_RELIABLE','SRS_VERSION'}
    missing_families=[prefix for prefix in ('BRIEF','CBCL','SENSORY') if not any(prefix in c for c in ph)]
    s['absent_candidate_families']=missing_families;s['candidate_fields']=score_columns
    age=pd.cut(population.AGE_AT_SCAN,[0,12,18,30,50,100],right=False).astype(str)
    rows=[];scale_summary=[]
    for col in score_columns:
        v=pd.to_numeric(population[col],errors='coerce');lo=1 if col in ('ADOS_MODULE','ADOS_GOTHAM_SEVERITY','SRS_VERSION') else 0
        valid=v.notna() & (v>=lo);invalid=v.notna() & ~valid
        per_class=[]
        for dx in sorted(population.DX_GROUP.unique()):
            mask=population.DX_GROUP==dx
            for site in ['ALL']+sorted(population.SITE_ID.unique()):
                sm=mask if site=='ALL' else mask & (population.SITE_ID==site)
                for band in ['ALL']+(sorted(age.unique()) if site=='ALL' else []):
                    m=sm if band=='ALL' else sm & (age==band)
                    n=int(m.sum())
                    if n==0:continue
                    rows.append(dict(scope=s['source_coverage_scope'],field=col,dx_group=int(dx),diagnosis='ASD' if dx==1 else 'TD',site=site,age_band=band,
                        subjects=n,nonmissing=int((m&valid).sum()),missing=int((m&v.isna()).sum()),invalid=int((m&invalid).sum()),
                        nonmissing_fraction=float((m&valid).sum()/n),missing_or_invalid_fraction=float((m&~valid).sum()/n),
                        sites_with_valid=int(population.loc[m&valid,'SITE_ID'].nunique()),
                        score_direction='higher=greater symptom load' if col not in metadata else 'metadata, not symptom severity',
                        cohort_join_verified=cohort is not None))
            per_class.append(dict(dx=int(dx),subjects=int(mask.sum()),valid=int((mask&valid).sum()),coverage=float((mask&valid).sum()/mask.sum()),sites=int(population.loc[mask&valid,'SITE_ID'].nunique())))
        numeric_symmetric=col not in metadata and all(x['valid']>=50 and x['coverage']>=.4 and x['sites']>=2 for x in per_class)
        asd=next(x for x in per_class if x['dx']==1)
        candidate_asd=col not in metadata and asd['valid']>=120 and asd['sites']>=3
        # Instrument version / administration differences still require independent comparability, even with adequate N.
        scale_summary.append(dict(field=col,source_counts=per_class,numeric_symmetric_coverage=numeric_symmetric,
            asd_only_numeric_candidate=candidate_asd,classification='UNUSABLE' if cohort is None else 'ASD_ONLY_EXPLORATORY' if candidate_asd else 'UNUSABLE',
            comparability='NOT_ESTABLISHED across versions/sites; ADOS and ADI-R are ASD diagnostic instruments'))
    s['coverage_summary']=scale_summary;s['cohort_coverage_available']=cohort is not None
    pd.DataFrame(rows).to_csv(out/'clinical_coverage.csv',index=False)
    (out/'clinical_audit.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return s
