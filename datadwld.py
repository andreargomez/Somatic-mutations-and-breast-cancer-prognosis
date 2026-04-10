import requests
import json
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

url = "https://api.gdc.cancer.gov/cases"

# breast cancer patients only
filt = {
    "op": "=",
    "content": {
        "field": "project.project_id",
        "value": "TCGA-BRCA"
    }
}

# for survival analysis
fields = [
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.age_at_diagnosis",
    "diagnoses.tumor_stage",
    "diagnoses.primary_diagnosis",
]

params = {
    "filters": json.dumps(filt),
    "fields": ",".join(fields),
    "format": "JSON",
    "size": 2000
}

response = requests.get(url, params=params, timeout=30)
cases = response.json()["data"]["hits"]
print(f"{len(cases)} patients")

# put data into a table
rows = []
for case in cases:
    row = {}
    row['patient_id'] = case.get('submitter_id', None)

    demo = case.get('demographic', {})
    row['vital_status'] = demo.get('vital_status', None)
    row['days_to_death'] = demo.get('days_to_death', None)

    diag = case.get('diagnoses', [{}])
    d = diag[0] if diag else {}
    row['days_to_last_follow_up'] = d.get('days_to_last_follow_up', None)
    row['age_at_diagnosis'] = d.get('age_at_diagnosis', None)
    row['tumor_stage'] = d.get('tumor_stage', None)

    rows.append(row)

clinicaldf = pd.DataFrame(rows)

# 1825 days = 5 years
clinicaldf['days_to_death'] = pd.to_numeric(clinicaldf['days_to_death'], errors='coerce')
clinicaldf['days_to_last_follow_up'] = pd.to_numeric(clinicaldf['days_to_last_follow_up'], errors='coerce')

def survival(row):
    if pd.notnull(row['days_to_death']):
        days = row['days_to_death']
    elif pd.notnull(row['days_to_last_follow_up']):
        days = row['days_to_last_follow_up']
    else:
        return None
    return 1 if days >= 1825 else 0

clinicaldf['survival_5yr'] = clinicaldf.apply(survival, axis=1)

# save both files
clinicaldf.to_csv("data/brcac.csv", index=False)


# mutation data
from io import StringIO, BytesIO
import gzip
import time

muturl = "https://api.gdc.cancer.gov/files"

mutfilt = {
    "op": "and",
    "content": [
        {
            "op": "=",
            "content": {
                "field": "cases.project.project_id",
                "value": "TCGA-BRCA"
            }
        },
        {
            "op": "=",
            "content": {
                "field": "data_type",
                "value": "Masked Somatic Mutation"
            }
        },
        {
            "op": "=",
            "content": {
                "field": "access",
                "value": "open"
            }
        }
    ]
}

mutparams = {
    "filters": json.dumps(mutfilt),
    "fields": "file_id,file_name",
    "format": "JSON",
    "size": 1000
}

mutresp = requests.get(muturl, params=mutparams, timeout=30)
mutfiles = mutresp.json()["data"]["hits"]

mutrows = []
for f in mutfiles:
    fid = f["file_id"]
    try:
        r = requests.post(
            "https://api.gdc.cancer.gov/data",
            json={"ids": [fid]},
            headers={"Content-Type": "application/json"},
            timeout=120,
            stream=True
        )
        raw = b""
        for chunk in r.iter_content(chunk_size=8192):
            raw += chunk
        try:
            txt = gzip.decompress(raw).decode("utf-8")
        except:
            txt = raw.decode("utf-8")
        lines = [l for l in txt.split("\n") if not l.startswith("#") and l.strip()]
        tmp = pd.read_csv(StringIO("\n".join(lines)), sep="\t", low_memory=False)
        mutrows.append(tmp)
        time.sleep(1)
    except Exception as e:
        continue

mutdf = pd.concat(mutrows, ignore_index=True)

# keep useful columns only
cols = [
    'Hugo_Symbol', 'Variant_Classification', 'Variant_Type',
    'Tumor_Sample_Barcode', 'HGVSp_Short',
    't_depth', 't_alt_count'
]
cols = [c for c in cols if c in mutdf.columns]
mutdf = mutdf[cols].copy()

# calculate VAF
if 't_alt_count' in mutdf.columns and 't_depth' in mutdf.columns:
    mutdf['VAF'] = pd.to_numeric(mutdf['t_alt_count'], errors='coerce') / pd.to_numeric(mutdf['t_depth'], errors='coerce')

mutdf.to_csv("data/brcamut.csv", index=False)
print("done")