import pandas as pd

fp = "data/data_input.tsv"
df_input = pd.read_csv(fp, sep="\t")
df_input.head().T

def apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    for col, dtype in schema.items():
        if col in df:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce') if dtype == 'datetime' else df[col].astype(dtype)
            except Exception as e:
                print(f"[Warning] Column '{col}': {e}")
    return df

schema = {
    col: (
        'string' if col == 'person_id' else
        'int64' if col == 'age' or col == 'chronic_pain_conditions' or col == 'FCI_Score' else
        'category' if col.endswith('_source_value') or col == 'CCI_Score' else
        'bool'
    )
    for col in df_input.columns.tolist()
}

df_input = apply_schema(df_input, schema)
for col, dtype in df_input.dtypes.items():
    print(f"{col}: {dtype}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def find_best_k_and_model(X, k_min=2, k_max=10, max_sample_size=5000, selected_k=None):
    inertias, silhouettes, models = [], [], []
    k_range = range(k_min, k_max + 1)
    sample_size = min(max_sample_size, len(X))

    if sample_size < len(X):
        print(f"⚡ Silhouette score computed on {sample_size} samples out of {len(X)} total.")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
        models.append(kmeans)

    best_idx = np.argmax(silhouettes) if selected_k is None else k_range.index(selected_k)
    print(f"⚡ Best k selected: {k_range[best_idx]}!")
    return k_range, inertias, silhouettes, models[best_idx], k_range[best_idx]

def describe_any_group(df, group_col=None, numerical_cols=None, categorical_cols=None, bool_cols=None):
    """
    Summarize dataframe either:
    - As a whole (group_col=None)
    - Or grouped by group_col
    """
    groups = {'All': df} if group_col is None else {g: d for g, d in df.groupby(group_col)}
    summary = {}

    for group_name, group_df in groups.items():
        info = {}
        info['size'] = len(group_df)
        if numerical_cols:
            info['numerical'] = group_df[numerical_cols].agg(['mean', 'std']).T
        if categorical_cols:
            info['categorical'] = {col: (group_df[col].value_counts(normalize=True) * 100).round(2) for col in categorical_cols}
        if bool_cols:
            info['bool_positive_rate'] = (group_df[bool_cols].mean() * 100).round(2).sort_values(ascending=False)
        summary[group_name] = info

    return summary

def full_clustering_pipeline(df_input, selected_k = None, id_col='person_id',
                            numerical_cols=['age', 'chronic_pain_conditions', 'FCI_Score'],
                            categorical_cols=['ethnicity_source_value', 'race_source_value', 'gender_source_value', 'CCI_Score'],
                            k_min=2, k_max=10, max_sample_size=5000):
    df_X = df_input.drop(columns=[id_col], errors='ignore').copy()
    bool_cols = [col for col in df_X.columns if df_X[col].dtype == 'bool' and col not in (numerical_cols or []) + (categorical_cols or [])]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
        ('bool', 'passthrough', bool_cols)
    ])

    X_transformed = preprocessor.fit_transform(df_X)
    k_range, inertias, silhouettes, best_model, best_k = find_best_k_and_model(X_transformed, k_min, k_max, max_sample_size, selected_k)

    df_input['cluster_label'] = best_model.predict(X_transformed)

    full_summary = describe_any_group(df_input, numerical_cols=numerical_cols, categorical_cols=categorical_cols, bool_cols=bool_cols)
    cluster_summary = describe_any_group(df_input, group_col='cluster_label', numerical_cols=numerical_cols, categorical_cols=categorical_cols, bool_cols=bool_cols)

    return df_input, full_summary, cluster_summary, X_transformed, df_input['cluster_label'].values, k_range, inertias, silhouettes

def plot_and_save_pca_clusters(X_transformed, cluster_labels, save_fp='outputs/pca_clusters.png'):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_transformed)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=0.1, alpha=0.8)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_fp, dpi=300)
    plt.close()
    print(f"✅ PCA cluster plot saved to {save_fp}")

def plot_and_save_elbow_silhouette(k_range, inertias, silhouettes, save_fp='outputs/elbow_silhouette.png'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(k_range, inertias, marker='o')
    ax[0].set_title('Elbow Method')
    ax[0].set_xlabel('Number of clusters (k)')
    ax[0].set_ylabel('Inertia')
    ax[0].grid(True)

    ax[1].plot(k_range, silhouettes, marker='o')
    ax[1].set_title('Silhouette Score')
    ax[1].set_xlabel('Number of clusters (k)')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_fp, dpi=300)
    plt.close()
    print(f"✅ Elbow and Silhouette plots saved to {save_fp}")

def save_cluster_summary_to_markdown(cluster_summary, output_fp='outputs/cluster_summary.md'):
    """
    Save cluster summary into a markdown file:
    (1) Combined comparison table (clusters + All Data)
    (2) Detailed description per cluster
    """
    numerical_features, categorical_features, bool_features = set(), set(), set()

    for info in cluster_summary.values():
        if 'numerical' in info:
            numerical_features.update(info['numerical'].index)
        if 'categorical' in info:
            for cat_feature, cats in info['categorical'].items():
                categorical_features.update(f"{cat_feature}={cat}" for cat in cats.index)
        if 'bool_positive_rate' in info:
            bool_features.update(info['bool_positive_rate'].index)

    all_features = sorted(numerical_features) + sorted(categorical_features) + sorted(bool_features)
    cluster_ids = sorted(cid for cid in cluster_summary.keys() if cid != 'All')

    feature_table = {}

    for feature in all_features:
        feature_table[feature] = {}
        for cluster_id, info in cluster_summary.items():
            value = ''
            if 'numerical' in info and feature in info['numerical'].index:
                row = info['numerical'].loc[feature]
                value = f"{row['mean']:.2f} ± {row['std']:.2f}"
            elif 'bool_positive_rate' in info and feature in info['bool_positive_rate'].index:
                value = f"{info['bool_positive_rate'][feature]:.2f}%"
            else:
                for cat_feature, cats in (info.get('categorical') or {}).items():
                    if feature.startswith(f"{cat_feature}="):
                        cat_value = feature.split('=')[1]
                        if cat_value in cats.index:
                            value = f"{cats[cat_value]:.2f}%"
            feature_table[feature][cluster_id] = value

    with open(output_fp, 'w') as f:

        f.write("# Cluster Sizes\n\n")
        f.write("| Cluster | Size |\n")
        f.write("|:---|---:|\n")
        for cid in sorted(cluster_summary.keys(), key=lambda x: (x == 'All', x)):
            size = cluster_summary[cid].get('size', 'N/A')
            f.write(f"| {cid} | {size} |\n")
        f.write("\n\n---\n\n")

        f.write("# Cluster Comparison Table (Including All Data)\n\n")
        f.write("| Feature | " + " | ".join(f"Cluster {cid}" for cid in cluster_ids) + " | All Data |\n")
        f.write("|:---|" + " ---|" * (len(cluster_ids) + 1) + "\n")

        for feature in all_features:
            row = [feature_table[feature].get(cid, '') for cid in cluster_ids]
            row.append(feature_table[feature].get('All', ''))
            f.write(f"| {feature} | " + " | ".join(row) + " |\n")

        f.write("\n\n---\n\n")

        for cluster_id in cluster_ids + ['All']:
            info = cluster_summary[cluster_id]
            f.write(f"# Cluster {cluster_id}\n\n")

            if 'numerical' in info:
                f.write("## Numerical Features (Mean ± Std)\n\n| Feature | Mean | Std |\n|:---|:---|:---|\n")
                for feature, row in info['numerical'].iterrows():
                    f.write(f"| {feature} | {row['mean']:.2f} | {row['std']:.2f} |\n")
                f.write("\n")

            if 'categorical' in info:
                f.write("## Categorical Feature Distribution (%)\n\n")
                for cat_feature, value_counts in info['categorical'].items():
                    f.write(f"### {cat_feature}\n\n| Category | Percentage |\n|:---|:---|\n")
                    for category, pct in value_counts.items():
                        f.write(f"| {category} | {pct:.2f}% |\n")
                    f.write("\n")

            if 'bool_positive_rate' in info:
                f.write("## Boolean Features Positive Rate (%)\n\n| Feature | Positive Rate |\n|:---|:---|\n")
                for feature, rate in info['bool_positive_rate'].items():
                    f.write(f"| {feature} | {rate:.2f}% |\n")
                f.write("\n")

            f.write("\n\n---\n\n")

    print(f"✅ Cluster summary (with All Data) saved to {output_fp}")

for k in range(4,5):
    dir_outputs = f'outputs/kmeans/k_{k}' if k else f'outputs'
    os.makedirs(dir_outputs, exist_ok=True)

    df_clustered, full_summary, cluster_summary, X_transformed, labels, k_range, inertias, silhouettes = full_clustering_pipeline(df_input, selected_k=k, max_sample_size=100000)

    merged_summary = cluster_summary.copy()
    merged_summary['All'] = full_summary['All']

    save_cluster_summary_to_markdown(merged_summary, output_fp=f'{dir_outputs}/cluster_summary.md')

    plot_and_save_pca_clusters(X_transformed, labels, save_fp=f'{dir_outputs}/pca_clusters.png')
    plot_and_save_elbow_silhouette(k_range, inertias, silhouettes, save_fp=f'{dir_outputs}/elbow_silhouette.png')

    df_clustered.to_csv(f"{dir_outputs}/data_output.tsv", sep="\t", index=False)
    print("✅ df_clustered saved")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm

def random_seed_stability(X, k, n_runs=20):
    labels_list = []

    for seed in tqdm(range(n_runs)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X)
        labels_list.append(labels)

    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari_scores.append(
                adjusted_rand_score(labels_list[i], labels_list[j])
            )

    return np.mean(ari_scores)

def bootstrap_stability(X, k, n_boot=20, sample_frac=0.8):

    full_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    scores = []
    n = X.shape[0]
    m = int(n * sample_frac)

    for b in tqdm(range(n_boot)):
        idx = np.random.choice(n, m, replace=True)
        Xb = X[idx]

        boot_labels = KMeans(n_clusters=k, random_state=b, n_init=20).fit_predict(Xb)

        scores.append(adjusted_rand_score(full_labels[idx], boot_labels))

    return float(np.mean(scores))

def clusterwise_jaccard_bootstrap(X, k, n_boot=20, sample_frac=0.8):
    n = X.shape[0]
    m = int(n * sample_frac)

    full_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    per_cluster_scores = {c: [] for c in range(k)}

    for b in tqdm(range(n_boot)):
        idx = np.random.choice(n, m, replace=True)
        Xb = X[idx]

        boot_labels = KMeans(n_clusters=k, random_state=b, n_init=20).fit_predict(Xb)

        overlap = np.zeros((k, k), dtype=int)
        for c in range(k):
            c_mask = (full_labels[idx] == c)
            for d in range(k):
                overlap[c, d] = np.sum(c_mask & (boot_labels == d))

        row_ind, col_ind = linear_sum_assignment(-overlap)
        mapping = {boot_c: full_c for full_c, boot_c in zip(row_ind, col_ind)}

        for boot_c, full_c in mapping.items():
            A = (full_labels[idx] == full_c)
            B = (boot_labels == boot_c)
            inter = np.sum(A & B)
            union = np.sum(A | B)
            jacc = inter / union if union > 0 else 0.0
            per_cluster_scores[full_c].append(jacc)

    mean_jaccard = {c: float(np.mean(per_cluster_scores[c])) if len(per_cluster_scores[c]) else float("nan")
                    for c in range(k)}

    return mean_jaccard, per_cluster_scores

k = 4

df_clustered, full_summary, cluster_summary, X_transformed, labels, k_range, inertias, silhouettes = full_clustering_pipeline(df_input, selected_k=k, k_min=k, k_max=k)

mean_ari = random_seed_stability(X_transformed, k=k, n_runs=20)
print("Random seed stability (mean ARI):", round(mean_ari, 4))

boot_ari = bootstrap_stability(X_transformed, k=k, n_boot=20, sample_frac=0.8)
print("Bootstrap stability (mean ARI):", round(boot_ari, 4))

mean_jacc, all_jacc = clusterwise_jaccard_bootstrap(X_transformed, k=k, n_boot=20, sample_frac=0.8)
print("Cluster-wise Jaccard (mean over bootstraps):")
for c in sorted(mean_jacc):
    print(c, round(mean_jacc[c], 4))
