"""PCA dimensionality reduction."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA


def apply_pca(
    customer_data_scaled: pd.DataFrame,
    n_components: int = 6,
    plots_dir: str = "plots",
    skip_plots: bool = False,
) -> tuple:
    """
    Fit PCA and return the transformed DataFrame (PC1…PCn) with CustomerID as index.

    Returns
    -------
    customer_data_pca : pd.DataFrame
    pca_model         : fitted PCA object
    """
    os.makedirs(plots_dir, exist_ok=True)

    if not skip_plots:
        _plot_explained_variance(customer_data_scaled, n_components, plots_dir)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(customer_data_scaled)
    cols = [f"PC{i+1}" for i in range(n_components)]
    customer_data_pca = pd.DataFrame(transformed, columns=cols, index=customer_data_scaled.index)

    return customer_data_pca, pca


def _plot_explained_variance(data: pd.DataFrame, optimal_k: int, plots_dir: str):
    pca_full = PCA().fit(data)
    ev   = pca_full.explained_variance_ratio_
    cev  = np.cumsum(ev)

    sns.set(rc={"axes.facecolor": "#fcf0dc"}, style="darkgrid")
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(range(1, len(ev)+1), ev, color="#fcc36d", alpha=0.8, label="Explained Variance")
    ax.plot(range(1, len(cev)+1), cev, marker="o", linestyle="--", color="#ff6200",
            linewidth=2, label="Cumulative Explained Variance")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k = {optimal_k}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Cumulative Variance vs. Number of Components")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pca_variance.png", dpi=100)
    plt.close()
