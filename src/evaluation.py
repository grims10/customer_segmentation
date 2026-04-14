"""Cluster quality evaluation metrics."""
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def evaluate_clusters(customer_data_pca: pd.DataFrame):
    """Print Silhouette, Calinski-Harabasz, and Davies-Bouldin scores."""
    X        = customer_data_pca.drop(columns=["cluster"])
    clusters = customer_data_pca["cluster"]

    sil = silhouette_score(X, clusters)
    cal = calinski_harabasz_score(X, clusters)
    dav = davies_bouldin_score(X, clusters)

    table = [
        ["Observations",           len(X)],
        ["Silhouette Score",        f"{sil:.4f}"],
        ["Calinski-Harabasz Score", f"{cal:.2f}"],
        ["Davies-Bouldin Score",    f"{dav:.4f}"],
    ]
    print("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))
