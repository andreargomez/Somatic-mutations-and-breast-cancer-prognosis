import pandas as pd
import os

os.makedirs("data", exist_ok=True)

mutdf = pd.read_csv("data/brcamut.csv")

clinicaldf = pd.read_csv("data/brcac.csv")

oncogenes = {
    'PIK3CA', 'ERBB2', 'FGFR1', 'CCND1', 'MYC',
    'KRAS', 'HRAS', 'NRAS', 'AKT1', 'EGFR',
    'FGFR2', 'IGF1R', 'MDM2', 'CDK4', 'CDK6'
}
tsuppressors = {
    'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'RB1',
    'CDH1', 'MAP3K1', 'GATA3', 'KMT2C', 'ARID1A',
    'TBX3', 'RUNX1', 'CBFB', 'MLL3', 'SF3B1',
    'NCOR1', 'NF1', 'CASP8', 'HLA-A', 'CDKN2A'
}
cgenes = oncogenes | tsuppressors

mutdf['is_cancer_gene'] = mutdf['Hugo_Symbol'].isin(cgenes).astype(int)
mutdf['is_oncogene'] = mutdf['Hugo_Symbol'].isin(oncogenes).astype(int)
mutdf['is_tumor_suppressor'] = mutdf['Hugo_Symbol'].isin(tsuppressors).astype(int)

impact = {
    'Nonsense_Mutation': 3, 'Frame_Shift_Del': 3, 'Frame_Shift_Ins': 3,
    'Splice_Site': 3, 'Translation_Start_Site': 3, 'Nonstop_Mutation': 2,
    'Missense_Mutation': 2, 'In_Frame_Del': 2, 'In_Frame_Ins': 2,
    'Silent': 0, 'Intron': 0, "3'UTR": 0, "5'UTR": 0,
    "3'Flank": 0, "5'Flank": 0, 'RNA': 1, 'IGR': 0,
}
mutdf['impact_score'] = mutdf['Variant_Classification'].map(impact).fillna(1)
mutdf['is_snp'] = (mutdf['Variant_Type'] == 'SNP').astype(int)
mutdf['is_indel'] = mutdf['Variant_Type'].isin(['INS', 'DEL']).astype(int)
mutdf['is_missense'] = (mutdf['Variant_Classification'] == 'Missense_Mutation').astype(int)
mutdf['is_truncating'] = mutdf['Variant_Classification'].isin([
    'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
    'Splice_Site', 'Translation_Start_Site'
]).astype(int)

pathways = {
    'PI3K_AKT': {'PIK3CA','PIK3R1','AKT1','AKT2','PTEN','PIK3CB','AKT3','PIK3CD','MTOR'},
    'DNA_REPAIR': {'BRCA1','BRCA2','ATM','CHEK2','PALB2','RAD51','BARD1','ATR','CHEK1'},
    'CELL_CYCLE': {'RB1','CDKN2A','CDK4','CDK6','CCND1','CCNE1','CDK2','CDKN1B','CDKN1A'},
    'RTK_RAS': {'ERBB2','EGFR','KRAS','NRAS','HRAS','FGFR1','FGFR2','IGF1R','MET'},
    'HORMONE_SIGNALING': {'ESR1','PGR','FOXA1','GATA3','TBX3','NCOR1','NCOR2','ESR2'}
}
for pname, gset in pathways.items():
    mutdf[f'pathway_{pname}'] = mutdf['Hugo_Symbol'].isin(gset).astype(int)

mutdf['patient_id'] = mutdf['Tumor_Sample_Barcode'].str[:12]

patdf = mutdf.groupby('patient_id').agg(
    total_mutations=('Hugo_Symbol', 'count'),
    mean_impact_score=('impact_score', 'mean'),
    max_impact_score=('impact_score', 'max'),
    n_high_impact=('impact_score', lambda x: (x >= 3).sum()),
    n_cancer_gene_muts=('is_cancer_gene', 'sum'),
    n_oncogene_muts=('is_oncogene', 'sum'),
    n_tsg_muts=('is_tumor_suppressor', 'sum'),
    n_missense=('is_missense', 'sum'),
    n_truncating=('is_truncating', 'sum'),
    n_snp=('is_snp', 'sum'),
    n_indel=('is_indel', 'sum'),
    mean_vaf=('VAF', 'mean'),
    max_vaf=('VAF', 'max'),
    n_pi3k_muts=('pathway_PI3K_AKT', 'sum'),
    n_dna_repair_muts=('pathway_DNA_REPAIR', 'sum'),
    n_cell_cycle_muts=('pathway_CELL_CYCLE', 'sum'),
    n_rtk_ras_muts=('pathway_RTK_RAS', 'sum'),
    n_hormone_muts=('pathway_HORMONE_SIGNALING', 'sum'),
    has_TP53=('Hugo_Symbol', lambda x: int('TP53' in x.values)),
    has_PIK3CA=('Hugo_Symbol', lambda x: int('PIK3CA' in x.values)),
    has_CDH1=('Hugo_Symbol', lambda x: int('CDH1' in x.values)),
    has_BRCA1=('Hugo_Symbol', lambda x: int('BRCA1' in x.values)),
    has_BRCA2=('Hugo_Symbol', lambda x: int('BRCA2' in x.values)),
    has_GATA3=('Hugo_Symbol', lambda x: int('GATA3' in x.values)),
    has_PTEN=('Hugo_Symbol', lambda x: int('PTEN' in x.values)),
    has_MAP3K1=('Hugo_Symbol', lambda x: int('MAP3K1' in x.values)),
).reset_index()

mergeddf = patdf.merge(clinicaldf, on='patient_id', how='inner')
mergeddf.to_csv("data/brcaa.csv", index=False)
print("done")