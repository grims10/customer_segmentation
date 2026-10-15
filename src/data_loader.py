"""Data loading utilities."""
import pandas as pd
import os

REQUIRED_COLUMNS = {
    "CustomerID", "Description", "InvoiceNo", "StockCode", "Quantity",
    "InvoiceDate", "UnitPrice", "Country",
}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimum retail schema and return the unchanged frame."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("Dataset has no rows.")
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load the Online Retail CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://archive.ics.uci.edu/dataset/352/online+retail\n"
            "and place it as data/data.csv"
        )
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return validate_data(df)
