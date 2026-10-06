import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency, fisher_exact

fp_clustered = "outputs/kmeans/k_4/data_output.tsv"
df = pd.read_csv(fp_clustered, sep="\t")

id_col = "person_id"
cluster_col = "cluster_label"

df = df.dropna(subset=[cluster_col])

numerical_cols = ['age', 'chronic_pain_conditions', 'FCI_Score']
categorical_cols = ['ethnicity_source_value', 'race_source_value', 'gender_source_value', 'CCI_Score']

binary_cols = []
for col in df.columns:
    if col in [id_col, cluster_col] + numerical_cols + categorical_cols:
        continue
    n_uniq = df[col].dropna().nunique()
    if n_uniq <= 2 and n_uniq > 0:
        binary_cols.append(col)

print("Continuous variables:", numerical_cols)
print("Multiclass variables:", categorical_cols)
print("Binary variables:", binary_cols)

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    p_adj = np.empty_like(q)
    p_adj[order] = q
    return p_adj

def star_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

rows = []

for col in numerical_cols:
    if col not in df.columns:
        continue
    groups = [df.loc[df[cluster_col] == k, col].dropna()
              for k in sorted(df[cluster_col].unique())]

    if any(len(g) == 0 for g in groups) or len(groups) < 2:
        continue
    stat, p = kruskal(*groups)
    rows.append({
        "Variable": col,
        "Type": "Continuous",
        "Test": "Kruskal-Wallis",
        "Statistic": stat,
        "p_value": p
    })

for col in binary_cols:
    if col not in df.columns:
        continue
    tab = pd.crosstab(df[col], df[cluster_col])
    if tab.shape[0] < 2:
        continue

    use_fisher = (tab.values < 5).any() and tab.shape == (2, 2)
    if use_fisher:
        stat, p = fisher_exact(tab.values)
        test_name = "Fisher exact (2x2)"
    else:
        chi2, p, dof, exp = chi2_contingency(tab)
        stat = chi2
        test_name = "Chi-square"

    rows.append({
        "Variable": col,
        "Type": "Binary",
        "Test": test_name,
        "Statistic": stat,
        "p_value": p
    })

for col in categorical_cols:
    if col not in df.columns:
        continue
    tab = pd.crosstab(df[col], df[cluster_col])
    if tab.shape[0] < 2:
        continue
    chi2, p, dof, exp = chi2_contingency(tab)
    rows.append({
        "Variable": col,
        "Type": "Multiclass",
        "Test": "Chi-square",
        "Statistic": chi2,
        "p_value": p
    })

results = pd.DataFrame(rows)

results = results.sort_values("p_value").reset_index(drop=True)

results["p_adj_bh"] = bh_fdr(results["p_value"].values)
results["Star"] = results["p_adj_bh"].apply(star_from_p)

n_total = len(results)
n_sig = sum(results["p_adj_bh"] < 0.05)
print("\n===== Overall (BH-FDR) =====")
print(f"Total variables tested: {n_total}")
print(f"Significant variables (FDR q<0.05): {n_sig}")
print("============================\n")

out_csv = "outputs/kmeans/k_4/cluster_diff_tests.csv"
results.to_csv(out_csv, index=False)
print(f"Saved CSV: {out_csv}")
