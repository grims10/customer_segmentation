"""Data loading utilities."""
import pandas as pd
import os


def load_data(path: str) -> pd.DataFrame:
    """Load the Online Retail CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://archive.ics.uci.edu/dataset/352/online+retail\n"
            "and place it as data/data.csv"
        )
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df
