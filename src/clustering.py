"""K-Means clustering: optimal-k search + final model."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter


def find_optimal_k(
    customer_data_pca: pd.DataFrame,
    k_range: tuple = (2, 15),
    sil_range: tuple = (3, 12),
    plots_dir: str = "plots",
):
    """Run Elbow + Silhouette analysis and save plots (manual version)."""

    os.makedirs(plots_dir, exist_ok=True)

    df = customer_data_pca.drop(columns=["cluster"], errors="ignore")

    # 🔹 Elbow Method
    k_values = range(k_range[0], k_range[1])
    inertia = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df)
        inertia.append(km.inertia_)

    sns.set(style="darkgrid")
    plt.figure()
    plt.plot(list(k_values), inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.savefig(f"{plots_dir}/elbow.png")
    plt.close()

    # 🔹 Silhouette Scores
    sil_values = range(sil_range[0], sil_range[1])
    scores = []

    for k in sil_values:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(df)
        score = silhouette_score(df, labels)
        scores.append(score)

    plt.figure()
    plt.plot(list(sil_values), scores, marker="o")
    plt.title("Silhouette Scores")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.savefig(f"{plots_dir}/silhouette.png")
    plt.close()

    print("      Saved elbow.png and silhouette.png")


def run_kmeans(
    customer_data_pca: pd.DataFrame,
    customer_data_cleaned: pd.DataFrame,
    n_clusters: int = 3,
) -> tuple:
    """
    Fit KMeans and attach 'cluster' column
    """

    pca_feats = customer_data_pca.drop(columns=["cluster"], errors="ignore")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pca_feats)

    # Remap labels (largest cluster = 0)
    freq = Counter(labels)
    sorted_labels = [lbl for lbl, _ in freq.most_common()]
    label_map = {old: new for new, old in enumerate(sorted_labels)}
    new_labels = np.array([label_map[l] for l in labels])

    customer_data_cleaned = customer_data_cleaned.copy()
    customer_data_cleaned["cluster"] = new_labels

    customer_data_pca = pca_feats.copy()
    customer_data_pca["cluster"] = new_labels

    print(f"      Cluster distribution: { {i: int((new_labels==i).sum()) for i in range(n_clusters)} }")

    return customer_data_cleaned, customer_data_pca