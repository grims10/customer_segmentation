"""Feature scaling (StandardScaler, excluding binary/categorical columns)."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(customer_data_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all features except CustomerID, Is_UK, Day_Of_Week.
    Returns a scaled copy with CustomerID set as index.
    """
    exclude = ["CustomerID", "Is_UK", "Day_Of_Week"]
    to_scale = [c for c in customer_data_cleaned.columns if c not in exclude]

    scaler = StandardScaler()
    scaled = customer_data_cleaned.copy()
    scaled[to_scale] = scaler.fit_transform(scaled[to_scale].astype(float))

    # Set CustomerID as index for downstream PCA / clustering
    scaled.set_index("CustomerID", inplace=True)
    return scaled
