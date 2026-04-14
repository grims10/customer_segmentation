"""Data cleaning & transformation steps."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def clean_data(df: pd.DataFrame, plots_dir: str = "plots", skip_plots: bool = False) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop missing CustomerID / Description
      2. Remove duplicate rows
      3. Tag cancelled transactions
      4. Remove anomalous StockCodes
      5. Remove service-related descriptions; uppercase descriptions
      6. Remove zero-UnitPrice records
      7. Reset index
    """
    os.makedirs(plots_dir, exist_ok=True)

    # ── 1. Missing values ────────────────────────────────────────
    if not skip_plots:
        _plot_missing(df, plots_dir)

    df = df.dropna(subset=["CustomerID", "Description"])

    # ── 2. Duplicates ────────────────────────────────────────────
    n_dups = df.duplicated().sum()
    print(f"      Removed {n_dups:,} duplicate rows")
    df.drop_duplicates(inplace=True)

    # ── 3. Cancelled transactions ─────────────────────────────────
    df["Transaction_Status"] = np.where(
        df["InvoiceNo"].astype(str).str.startswith("C"), "Cancelled", "Completed"
    )

    # ── 4. Anomalous StockCodes ───────────────────────────────────
    unique_sc = df["StockCode"].unique()
    anomalous = [c for c in unique_sc if sum(ch.isdigit() for ch in str(c)) in (0, 1)]
    df = df[~df["StockCode"].isin(anomalous)]

    # ── 5. Description cleanup ────────────────────────────────────
    service_descs = ["Next Day Carriage", "High Resolution Image"]
    df = df[~df["Description"].isin(service_descs)]
    df["Description"] = df["Description"].str.upper()

    # ── 6. Zero unit price ────────────────────────────────────────
    df = df[df["UnitPrice"] > 0]

    # ── 7. Reset index ────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)
    return df


def _plot_missing(df: pd.DataFrame, plots_dir: str):
    missing_data = df.isnull().sum()
    missing_pct = (missing_data[missing_data > 0] / df.shape[0]) * 100
    missing_pct.sort_values(ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.barh(missing_pct.index, missing_pct, color="#ff6200")
    for i, (val, name) in enumerate(zip(missing_pct, missing_pct.index)):
        ax.text(val + 0.5, i, f"{val:.2f}%", va="center", fontweight="bold", fontsize=14)
    ax.set_xlim([0, 40])
    plt.title("Percentage of Missing Values", fontweight="bold", fontsize=18)
    plt.xlabel("Percentages (%)")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/missing_values.png", dpi=100)
    plt.close()
