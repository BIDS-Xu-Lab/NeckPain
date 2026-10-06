import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

path = "path/to/outcome_data"
opioid_file = "Opioid_drug.csv"
procedure_file = "proc_flags.parquet"
cluster = "outputs/kmeans/k_4/data_output.tsv"

OUT = "outputs/sensitivity"
import os; os.makedirs(OUT, exist_ok=True)

cluster_col = "cluster_label"
LABEL_TO_MS = {0: "Cluster 3", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 4"}
MS_ORDER = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]

df_cluster = pd.read_csv(cluster, sep="\t")[["person_id", cluster_col]]
df_cluster["ms_cluster"] = df_cluster[cluster_col].map(LABEL_TO_MS)

df_op = pd.read_csv(f"{path}/{opioid_file}", usecols=["person_id"])
op_flag = df_op[["person_id"]].drop_duplicates().assign(opioid=1)

df_proc = pd.read_parquet(f"{path}/{procedure_file}")
surg = df_proc[["person_id", "grp_surgery"]].copy()
surg["surgery"] = (surg["grp_surgery"] > 0).astype(int)

df = (df_cluster
      .merge(op_flag, on="person_id", how="left")
      .merge(surg[["person_id", "surgery"]], on="person_id", how="left"))
df["opioid"] = df["opioid"].fillna(0).astype(int)
df["surgery"] = df["surgery"].fillna(0).astype(int)
print(f"* {len(df):,} patients; opioid={df.opioid.mean()*100:.2f}%, surgery={df.surgery.mean()*100:.2f}%")

rows = []
for ms in MS_ORDER:
    sub = df[df["ms_cluster"] == ms]
    ct = pd.crosstab(sub["opioid"], sub["surgery"]).reindex(index=[1, 0], columns=[1, 0], fill_value=0)

    op_surg_y = int(ct.loc[1, 1])
    op_surg_n = int(ct.loc[1, 0])
    noop_surg_y = int(ct.loc[0, 1])
    noop_surg_n = int(ct.loc[0, 0])
    n_surg = op_surg_y + noop_surg_y
    n_nosurg = op_surg_n + noop_surg_n
    rows.append({
        "Cluster": ms,
        "N": len(sub),
        "opioid_rate_overall_%": round(sub["opioid"].mean() * 100, 2),
        "opioid_rate_in_SURGICAL_%": round(op_surg_y / n_surg * 100, 2) if n_surg else np.nan,
        "opioid_rate_in_NONsurgical_%": round(op_surg_n / n_nosurg * 100, 2) if n_nosurg else np.nan,
        "n_surgical": n_surg,
        "opioid+surgery+": op_surg_y, "opioid+surgery-": op_surg_n,
        "opioid-surgery+": noop_surg_y, "opioid-surgery-": noop_surg_n,
    })
tab1 = pd.DataFrame(rows)
tab1.to_csv(f"{OUT}/opioid_by_surgery_2x2.csv", index=False)
print("\n===== Part 1: opioid x surgery per cluster =====")
print(tab1.to_string(index=False))

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    adj = np.empty_like(q); adj[o] = q; return adj

df_ns = df[df["surgery"] == 0]
had = [int(df_ns[(df_ns.ms_cluster == ms)]["opioid"].sum()) for ms in MS_ORDER]
tot = [int((df_ns.ms_cluster == ms).sum()) for ms in MS_ORDER]
non = [t - h for t, h in zip(tot, had)]
chi2, p_excl, dof, _ = chi2_contingency([had, non])

sensA = pd.DataFrame({
    "Cluster": MS_ORDER,
    "opioid_%_all": [round(df[df.ms_cluster == ms]["opioid"].mean() * 100, 2) for ms in MS_ORDER],
    "opioid_%_excl_surgical": [round(h / t * 100, 2) for h, t in zip(had, tot)],
    "n_excl_surgical": tot,
})
print("\n===== Part 2: Sensitivity A — opioid % excluding surgical patients =====")
print(sensA.to_string(index=False))
print(f"2x4 chi-square (non-surgical only): chi2={chi2:,.1f}, dof={dof}, p={p_excl:.3e}")
sensA.to_csv(f"{OUT}/opioid_sensitivity_exclude_surgery.csv", index=False)

try:
    import statsmodels.formula.api as smf
    dd = df.copy()
    dd["cluster"] = pd.Categorical(dd["ms_cluster"], categories=MS_ORDER, ordered=False)

    dd["cluster"] = dd["cluster"].cat.reorder_categories(
        ["Cluster 4", "Cluster 1", "Cluster 2", "Cluster 3"])
    m = smf.logit("opioid ~ C(cluster) + surgery", data=dd).fit(disp=False)
    res = pd.DataFrame({
        "term": m.params.index,
        "OR": np.exp(m.params.values).round(3),
        "CI_low": np.exp(m.conf_int()[0].values).round(3),
        "CI_high": np.exp(m.conf_int()[1].values).round(3),
        "p": m.pvalues.values,
    })
    print("\n===== Part 3: Sensitivity B — logistic opioid ~ cluster + surgery (ref = Cluster 4) =====")
    print(res.to_string(index=False))
    res.to_csv(f"{OUT}/opioid_logit_adjusted_for_surgery.csv", index=False)
except ImportError:
    print("\n[Part 3 skipped] install statsmodels to run the adjusted logistic model.")

print(f"\n✅ outputs written to {OUT}/")
