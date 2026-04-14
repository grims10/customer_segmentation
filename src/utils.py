"""Utility helpers."""
import os
import pandas as pd


def save_output(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"      Saved: {path}  ({len(df):,} rows)")
