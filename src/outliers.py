"""Outlier detection using Isolation Forest."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest


def remove_outliers(
    customer_data: pd.DataFrame,
    contamination: float = 0.05,
    plots_dir: str = "plots",
    skip_plots: bool = False,
) -> tuple:
    """
    Fit IsolationForest on customer features, separate outliers.

    Returns
    -------
    customer_data_cleaned : pd.DataFrame
    outliers_data         : pd.DataFrame
    """
    os.makedirs(plots_dir, exist_ok=True)

    model = IsolationForest(contamination=contamination, random_state=0)
    customer_data = customer_data.copy()
    customer_data = customer_data.fillna(0)
    customer_data["Outlier_Scores"] = model.fit_predict(
        customer_data.iloc[:, 1:].to_numpy()
    )
    customer_data["Is_Outlier"] = (customer_data["Outlier_Scores"] == -1).astype(int)

    if not skip_plots:
        _plot_outlier_pct(customer_data, plots_dir)

    outliers_data = customer_data[customer_data["Is_Outlier"] == 1].copy()
    cleaned       = customer_data[customer_data["Is_Outlier"] == 0].copy()
    cleaned.drop(columns=["Outlier_Scores", "Is_Outlier"], inplace=True)
    cleaned.reset_index(drop=True, inplace=True)

    return cleaned, outliers_data


def _plot_outlier_pct(customer_data: pd.DataFrame, plots_dir: str):
    pct = customer_data["Is_Outlier"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    pct.plot(kind="barh", color="#ff6200", ax=ax)
    for idx, val in enumerate(pct):
        ax.text(val, idx, f"{val:.2f}%", fontsize=13)
    ax.set_title("Percentage of Inliers and Outliers")
    ax.set_xlabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/outliers.png", dpi=100)
    plt.close()
