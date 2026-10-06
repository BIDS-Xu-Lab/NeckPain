import duckdb
import pandas as pd
import numpy as np

path = "path/to/outcome_data"

visit = "Neckpain_visit_filter.csv"
opioid = "Opioid_drug.csv"
procedure = "proc_flags.parquet"

cluster = "outputs/kmeans/k_4/data_output.tsv"

df_cluster = pd.read_csv(cluster, sep="\t")
df_cluster

df = duckdb.sql(f"""
SELECT person_id, drug_exposure_start_datetime, drug_source_value_name, drug_generic_source_value_name, drug_class_source_value
FROM "{path}/{drug_exposure}"
LIMIT 10
""").df()

df

df = duckdb.sql(f"""
SELECT person_id, visit_start_datetime, visit_source_value, admitted_from_source_value, discharge_to_source_value
FROM "{path}/{visit_occurrence}"
LIMIT 10
""").df()

df

df = duckdb.sql(f"""
SELECT person_id, procedure_datetime, procedure_source_value, status_source_value
FROM "{path}/{procedure}"
LIMIT 10
""").df()

df

df_visit = duckdb.sql(f"""
SELECT DISTINCT
    person_id,
    visit_start_datetime,
    OMOP_concept_name
FROM "{path}/{visit}"
""").df()

print(f"* loaded {df_visit.person_id.nunique()}patients/{len(df_visit)}visits from {visit}!")
df_visit.head().T

df_visit['OMOP_concept_name'] = df_visit['OMOP_concept_name'].replace({
    'Emergency department': 'Emergency Room Visit'
})

df_visit['visit_date'] = df_visit['visit_start_datetime'].dt.date

df_visit = (
    df_visit.sort_values('visit_start_datetime')
      .drop_duplicates(subset=['person_id', 'visit_date', 'OMOP_concept_name'])
      .drop(columns='visit_date')
)

df_visit['emergency'] = df_visit['OMOP_concept_name'].isin([
    'Emergency Room Visit',
    'Emergency department',
    'Emergency Room and Inpatient Visit'
]).astype(int)

df_visit['inpatient'] = df_visit['OMOP_concept_name'].isin([
    'Inpatient Visit',
    'Emergency Room and Inpatient Visit',
    'Other acute inpatient hospital'
]).astype(int)

df_visit['outpatient'] = df_visit['OMOP_concept_name'].isin([
    'Office Visit',
    'Outpatient Visit',
    'Telehealth'
]).astype(int)

print(f"* finalized {df_visit.person_id.nunique()}patients/{len(df_visit)}visits!")
df_visit.head().T

df_visit_cluster = df_visit.merge(
    df_cluster[['person_id', 'cluster_label']],
    on='person_id',
    how='inner'
)

visit_dist = df_visit_cluster.groupby('cluster_label').agg(
    total_visits=('person_id', 'size'),
    total_patients=('person_id', 'nunique'),
    emergency=('emergency', 'sum'),
    inpatient=('inpatient', 'sum'),
    outpatient=('outpatient', 'sum'),
).reset_index()

visit_dist['visits_per_patient'] = (
    visit_dist['total_visits'] / visit_dist['total_patients']
).round(2)

visit_dist['emergency_visits_per_patient'] = (
    visit_dist['emergency'] / visit_dist['total_patients']
).round(2)

visit_dist['inpatient_visits_per_patient'] = (
    visit_dist['inpatient'] / visit_dist['total_patients']
).round(2)

visit_dist['outpatient_visits_per_patient'] = (
    visit_dist['outpatient'] / visit_dist['total_patients']
).round(2)

for col in ['emergency', 'inpatient', 'outpatient']:
    visit_dist[f'{col}_pct'] = (
        visit_dist[col] / visit_dist['total_visits'] * 100
    ).round(2).astype(str) + '%'

total = {
    'cluster_label': 'Total',
    'total_visits': visit_dist['total_visits'].sum(),
    'total_patients': visit_dist['total_patients'].sum(),
    'emergency': visit_dist['emergency'].sum(),
    'inpatient': visit_dist['inpatient'].sum(),
    'outpatient': visit_dist['outpatient'].sum(),
}

total['visits_per_patient'] = round(total['total_visits'] / total['total_patients'], 2)
total['emergency_visits_per_patient'] = round(total['emergency'] / total['total_patients'], 2)
total['inpatient_visits_per_patient'] = round(total['inpatient'] / total['total_patients'], 2)
total['outpatient_visits_per_patient'] = round(total['outpatient'] / total['total_patients'], 2)

for col in ['emergency', 'inpatient', 'outpatient']:
    total[f'{col}_pct'] = f"{total[col] / total['total_visits'] * 100:.2f}%"

visit_dist = pd.concat([visit_dist, pd.DataFrame([total])], ignore_index=True)

visit_dist

df_opioid = duckdb.sql(f"""
SELECT
    person_id,
    visit_occurrence_id,
    drug_exposure_start_datetime,
    --drug_exposure_id,
    --drug_concept_id,
    --drug_type_concept_id,
    --drug_source_value_code,
    drug_source_value_name,
    --drug_generic_source_value_code,
    drug_generic_source_value_name,
    --drug_source_concept_id,
    drug_class_source_value
FROM "{path}/{opioid}"
--LIMIT 5
""").df()

print(f"* loaded {df_opioid.person_id.nunique()}patients/{len(df_opioid)}records from {opioid}!")
df_opioid.head().T

df_opioid_flag = (
    df_opioid[['person_id']]
    .drop_duplicates()
    .assign(opioid=1)
)

df_cluster_opioid = df_cluster.merge(df_opioid_flag, on='person_id', how='left')
df_cluster_opioid['opioid'] = df_cluster_opioid['opioid'].fillna(0).astype(int)

opioid_dist = df_cluster_opioid.groupby('cluster_label')['opioid'].agg(
    total_patients='count',
    opioid_users='sum',
).reset_index()

opioid_dist['opioid_pct'] = ((opioid_dist['opioid_users'] / opioid_dist['total_patients']).round(4)).apply(lambda x: f"{x*100:.2f}%")

opioid_dist.loc['Total'] = [
    'Total',
    opioid_dist['total_patients'].sum(),
    opioid_dist['opioid_users'].sum(),
    f"{opioid_dist['opioid_users'].sum() / opioid_dist['total_patients'].sum() * 100:.2f}%"
]

opioid_dist

fp = f"{path}/{procedure}"
df = pd.read_parquet(fp)
print(df.shape)
df.head()

df_proc = df_cluster[["person_id", "cluster_label"]].merge(
    df,
    on="person_id",
    how="inner"
)

df_proc["grp_imaging"] = (
    (df_proc["grp_ct"] > 0) |
    (df_proc["grp_mri"] > 0) |
    (df_proc["grp_xray"] > 0)
).astype(int)

df_proc["grp_intervention"] = (
    (df_proc["grp_injection"] > 0) |
    (df_proc["grp_surgery"] > 0)
).astype(int)

df_proc["grp_imaging_n"] = (
    df_proc["grp_ct_n"] +
    df_proc["grp_mri_n"] +
    df_proc["grp_xray_n"]
)

df_proc["grp_intervention_n"] = (
    df_proc["grp_injection_n"] +
    df_proc["grp_surgery_n"]
)

group_cols = [
    "grp_imaging", "grp_imaging_n",
    "grp_ct", "grp_ct_n",
    "grp_mri", "grp_mri_n",
    "grp_xray", "grp_xray_n",
    "grp_intervention", "grp_intervention_n",
    "grp_injection", "grp_injection_n",
    "grp_surgery", "grp_surgery_n",

]

df_proc_all = df_proc.copy()
df_proc_all["cluster_label"] = "Overall"
df_combined = pd.concat([df_proc, df_proc_all], ignore_index=True)

group_summary = (
    df_combined
    .groupby("cluster_label")[group_cols]
    .agg(["mean", "sum"])
)

flag_cols = ["grp_imaging", "grp_ct", "grp_mri", "grp_xray",
             "grp_intervention", "grp_injection", "grp_surgery"]
for flag in flag_cols:
    cnt = f"{flag}_n"
    group_summary[(cnt, "avg_among_users")] = (
        group_summary[(cnt, "sum")] / group_summary[(flag, "sum")]
    )

group_summary.T.to_csv("./outputs/procedures_group.tsv", sep="\t")
group_summary.T

df_proc = df_cluster[["person_id", "cluster_label"]].merge(
    df,
    on="person_id",
    how="inner"
)

proc_cols = [
    c for c in df_proc.columns
    if c.startswith(("proc_", "neck_proc_"))
]

proc_rate = (
    df_proc
    .groupby("cluster_label")[proc_cols]
    .mean()
)

proc_rate.T.to_csv("./outputs/procedures.tsv", sep="\t")
proc_rate.T

overall_rate = df_proc[proc_cols].mean()

keep_proc = overall_rate[
    overall_rate > 0.01
].index

len(keep_proc)

import seaborn as sns
import matplotlib.pyplot as plt

heatmap_df = proc_rate[keep_proc]

plt.figure(
    figsize=(5, max(4, len(keep_proc) * 0.25))
)

sns.heatmap(
    heatmap_df.T,
    cmap="Reds"
)

plt.xlabel("Cluster")
plt.ylabel("Procedure")
plt.title("Procedure prevalence across clusters")

plt.tight_layout()
plt.show()
