"""Cluster profiling: radar charts and feature histograms."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

CLUSTER_COLORS = ["#e8000b", "#1ac938", "#023eff"]


def plot_radar(customer_data_cleaned: pd.DataFrame, plots_dir: str = "plots"):
    """Save radar charts for each cluster centroid."""
    os.makedirs(plots_dir, exist_ok=True)

    df_c = customer_data_cleaned.set_index("CustomerID")
    scaler = StandardScaler()
    feat_cols = [c for c in df_c.columns if c != "cluster"]
    df_std = pd.DataFrame(
        scaler.fit_transform(df_c[feat_cols].astype(float)),
        columns=feat_cols, index=df_c.index
    )
    df_std["cluster"] = df_c["cluster"]
    centroids = df_std.groupby("cluster").mean()

    labels = np.array(centroids.columns)
    n_vars  = len(labels)
    angles  = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    labels  = np.concatenate((labels, [labels[0]]))
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(centroids), figsize=(20, 8),
                             subplot_kw=dict(polar=True))
    if len(centroids) == 1:
        axes = [axes]

    for i, (ax, color) in enumerate(zip(axes, CLUSTER_COLORS[:len(centroids)])):
        data = centroids.loc[i].tolist() + [centroids.loc[i].tolist()[0]]
        ax.fill(angles, data, color=color, alpha=0.4)
        ax.plot(angles, data, color=color, linewidth=2)
        ax.set_title(f"Cluster {i}", size=18, color=color, y=1.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1], fontsize=7)

    plt.suptitle("Cluster Radar Charts", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/radar_charts.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {plots_dir}/radar_charts.png")


def plot_histograms(customer_data_cleaned: pd.DataFrame, plots_dir: str = "plots"):
    """Save per-feature histograms segmented by cluster."""
    os.makedirs(plots_dir, exist_ok=True)

    features = [c for c in customer_data_cleaned.columns
                if c not in ("CustomerID", "cluster")]
    cluster_ids = sorted(customer_data_cleaned["cluster"].unique())
    colors = CLUSTER_COLORS[:len(cluster_ids)]

    n_rows = len(features)
    n_cols = len(cluster_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))

    for i, feat in enumerate(features):
        for j, cid in enumerate(cluster_ids):
            data = customer_data_cleaned[customer_data_cleaned["cluster"] == cid][feat]
            axes[i, j].hist(data.astype(float), bins=20, color=colors[j],
                            edgecolor="w", alpha=0.7)
            axes[i, j].set_title(f"Cluster {cid} — {feat}", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/histograms.png", dpi=80)
    plt.close()
    print(f"      Saved: {plots_dir}/histograms.png")
