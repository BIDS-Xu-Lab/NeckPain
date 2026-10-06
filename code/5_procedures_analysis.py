import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

path = "path/to/outcome_data"
procedure = "proc_flags.parquet"
cluster = "outputs/kmeans/k_4/data_output.tsv"

OUT = "outputs/procedures"
os.makedirs(OUT, exist_ok=True)

cluster_col = "cluster_label"

LABEL_TO_MS = {0: "Cluster 3", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 4"}
MS_ORDER = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]
MS_TO_LABEL = {v: k for k, v in LABEL_TO_MS.items()}

CCOL = {"Cluster 1": "#E07B39", "Cluster 2": "#2E86AB",
        "Cluster 3": "#C73E3A", "Cluster 4": "#5BA672"}

df_cluster = pd.read_csv(cluster, sep="\t")
df = pd.read_parquet(f"{path}/{procedure}")
df_proc = df_cluster[["person_id", cluster_col]].merge(df, on="person_id", how="inner")

df_proc["grp_imaging"] = (
    (df_proc["grp_ct"] > 0) | (df_proc["grp_mri"] > 0) | (df_proc["grp_xray"] > 0)
).astype(int)
df_proc["grp_intervention"] = (
    (df_proc["grp_injection"] > 0) | (df_proc["grp_surgery"] > 0)
).astype(int)
df_proc["grp_imaging_n"] = df_proc["grp_ct_n"] + df_proc["grp_mri_n"] + df_proc["grp_xray_n"]
df_proc["grp_intervention_n"] = df_proc["grp_injection_n"] + df_proc["grp_surgery_n"]

for f in ["grp_ct", "grp_mri", "grp_xray", "grp_injection", "grp_surgery"]:
    df_proc[f] = (df_proc[f] > 0).astype(int)

df_proc["ms_cluster"] = df_proc[cluster_col].map(LABEL_TO_MS)

PROCS = [
    ("grp_imaging",      "grp_imaging_n",      "Any Imaging (CT, MRI or X-ray)"),
    ("grp_ct",           "grp_ct_n",           "CT scan"),
    ("grp_mri",          "grp_mri_n",          "MRI"),
    ("grp_xray",         "grp_xray_n",         "X-ray"),
    ("grp_intervention", "grp_intervention_n", "Any Intervention (Injection or Surgery)"),
    ("grp_injection",    "grp_injection_n",    "Injection"),
    ("grp_surgery",      "grp_surgery_n",      "Surgery"),
]

N = {ms: int((df_proc["ms_cluster"] == ms).sum()) for ms in MS_ORDER}
N_ALL = len(df_proc)
print(f"df_proc: {N_ALL} patients")
for ms in MS_ORDER:
    print(f"  {ms} (label={MS_TO_LABEL[ms]}): N={N[ms]:,}")

def metrics(sub):
    n = len(sub)
    out = {}
    for flag, nfeat, label in PROCS:
        ppl = int(sub[flag].sum())
        cpt = int(sub[nfeat].sum())
        out[label] = {
            "pct": ppl / n * 100 if n else np.nan,
            "avg": cpt / ppl if ppl else np.nan,
            "ppl": ppl, "cpt": cpt,
        }
    return out

per_ms = {ms: metrics(df_proc[df_proc["ms_cluster"] == ms]) for ms in MS_ORDER}
overall = metrics(df_proc)

metric_rows = [
    ("Percent of patients (%)",           "pct", "{:.2f}%"),
    ("Avg # CPTs for those patients",     "avg", "{:.2f}"),
    ("Number of people who had it",       "ppl", "{:,}"),
    ("Number of CPTs",                    "cpt", "{:,}"),
]

rows = []
for _, _, label in PROCS:
    for mname, mkey, fmt in metric_rows:
        row = {"Procedure": label, "Metric": mname}
        for ms in MS_ORDER:
            row[ms] = fmt.format(per_ms[ms][label][mkey])
        row["All patients"] = fmt.format(overall[label][mkey])
        rows.append(row)

table1 = pd.DataFrame(rows)
table1.to_csv(f"{OUT}/procedure_summary_table.csv", index=False)
print("\n===== Task 1: full table (manuscript naming) =====")
print(table1.to_string(index=False))

def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    adj = np.empty_like(q); adj[order] = q
    return adj

def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

tests_rows = []
for flag, _, label in PROCS:
    tab = pd.crosstab(df_proc[flag], df_proc["ms_cluster"])[MS_ORDER]
    chi2, p, dof, _ = chi2_contingency(tab)
    rec = {"Procedure": label, "Chi2": round(chi2, 2), "dof": dof, "p_value": p}
    for ms in MS_ORDER:
        rec[f"{ms} %"] = round(per_ms[ms][label]["pct"], 2)
    tests_rows.append(rec)

tests = pd.DataFrame(tests_rows)
tests["p_adj_BH"] = bh_fdr(tests["p_value"].values)
tests["Sig"] = tests["p_adj_BH"].apply(star)
tests.to_csv(f"{OUT}/procedure_cluster_tests.csv", index=False)
print("\n===== Task 2: chi-square across clusters (BH-FDR) =====")
print(tests.to_string(index=False))

pct_mat = pd.DataFrame(
    {ms: [per_ms[ms][label]["pct"] for _, _, label in PROCS] for ms in MS_ORDER},
    index=[label for _, _, label in PROCS],
)
short_labels = ["Any Imaging", "CT", "MRI", "X-ray",
                "Any Intervention", "Injection", "Surgery"]

p_by_label = dict(zip(tests["Procedure"], tests["p_adj_BH"]))
stars_by_label = dict(zip(tests["Procedure"], tests["Sig"]))
padj = np.array([p_by_label[label] for _, _, label in PROCS])
star_arr = [stars_by_label[label] for _, _, label in PROCS]

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(pct_mat.values, cmap="Reds", aspect="auto",
               vmin=0, vmax=max(pct_mat.values.max(), 1))
ax.set_xticks(range(len(MS_ORDER)))
ax.set_xticklabels(MS_ORDER, rotation=0, fontsize=18)
ax.set_yticks(range(len(short_labels)))
ax.set_yticklabels(short_labels, fontsize=18)

for i in range(pct_mat.shape[0]):
    for j in range(pct_mat.shape[1]):
        v = pct_mat.values[i, j]
        color = "white" if v > pct_mat.values.max() * 0.55 else "#222"
        ax.text(j, i, f"{v:.2f}%", ha="center", va="center",
                fontsize=18, color=color)

n_cols = pct_mat.shape[1]
for i, (s, p) in enumerate(zip(star_arr, padj)):
    p_str = "p < 1e-300" if p == 0 else f"p = {p:.2g}"

    ax.text(n_cols - 0.35, i, s, va="center", ha="left",
            fontsize=24, weight="bold", color="#C73E3A",
            transform=ax.transData, clip_on=False)

    ax.text(n_cols + 0.15, i, p_str, va="center", ha="left",
            fontsize=18, color="#333",
            transform=ax.transData, clip_on=False)

ax.text(n_cols - 0.35, -0.55, "Chi-square, BH-FDR",
        va="bottom", ha="left", fontsize=17,
        color="#666", style="italic",
        transform=ax.transData, clip_on=False)

ax.set_title("% of patients with each procedure (by cluster)",
             weight="bold", fontsize=22, pad=18, loc="left")

cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.28)
cbar.set_label("% of patients", fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig(f"{OUT}/procedure_heatmap.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/procedure_heatmap.pdf", bbox_inches="tight")
plt.show()
print(f"\n✅ heatmap saved to {OUT}/procedure_heatmap.png (+ .pdf)")

imaging_full = ["Any Imaging (CT, MRI or X-ray)", "CT scan", "MRI", "X-ray"]
inter_full = ["Any Intervention (Injection or Surgery)", "Injection", "Surgery"]
imaging_labels = ["Any Imaging", "CT", "MRI", "X-ray"]
inter_labels = ["Any Intervention", "Injection", "Surgery"]

img_stars = [stars_by_label[l] for l in imaging_full]
int_stars = [stars_by_label[l] for l in inter_full]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(18, 8.5),
    gridspec_kw={"width_ratios": [4, 3], "wspace": 0.18},
)
w = 0.2

x = np.arange(4)
for i, ms in enumerate(MS_ORDER):
    vals = pct_mat.loc[imaging_full, ms].values
    bars = ax1.bar(x + (i - 1.5) * w, vals, w, label=ms, color=CCOL[ms],
                   edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.6,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=11)

ymax1 = pct_mat.loc[imaging_full].values.max()
for j, s in enumerate(img_stars):
    if s != "ns":
        ax1.text(j, ymax1 * 1.05, s, ha="center", va="bottom",
                 fontsize=24, color="#333", weight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(imaging_labels, fontsize=20)
ax1.tick_params(axis="y", labelsize=16)
ax1.set_ylabel("% of patients", fontsize=22)
ax1.set_title("(a) Imaging procedures", weight="bold", fontsize=24,
              pad=12, loc="left")
ax1.set_ylim(0, ymax1 * 1.18)
for s in ["top", "right"]: ax1.spines[s].set_visible(False)

xi = np.arange(3)
for i, ms in enumerate(MS_ORDER):
    vals = pct_mat.loc[inter_full, ms].values
    bars = ax2.bar(xi + (i - 1.5) * w, vals, w, label=ms, color=CCOL[ms],
                   edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.04,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=11)

ymax2 = pct_mat.loc[inter_full].values.max()
for j, s in enumerate(int_stars):
    if s != "ns":
        ax2.text(j, ymax2 * 1.05, s, ha="center", va="bottom",
                 fontsize=24, color="#333", weight="bold")
ax2.set_xticks(xi); ax2.set_xticklabels(inter_labels, fontsize=20)
ax2.tick_params(axis="y", labelsize=16)
ax2.set_ylabel("% of patients", fontsize=22)
ax2.set_title("(b) Intervention procedures  ⚠ y-axis scaled to <5%",
              weight="bold", fontsize=24, pad=12, loc="left")
ax2.set_ylim(0, ymax2 * 1.18)
for s in ["top", "right"]: ax2.spines[s].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=CCOL[ms]) for ms in MS_ORDER]
fig.legend(handles, MS_ORDER, loc="upper center", ncol=4,
           frameon=False, fontsize=20, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Procedure prevalence across clusters",
             weight="bold", fontsize=26, y=1.06)

plt.savefig(f"{OUT}/procedure_bars.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/procedure_bars.pdf", bbox_inches="tight")
plt.show()
print(f"✅ grouped bar (two panels) saved to {OUT}/procedure_bars.png (+ .pdf)")
