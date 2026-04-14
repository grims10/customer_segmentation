"""Customer-centric feature engineering (RFM + behavioural + geographic + cancellation + seasonality)."""
import numpy as np
import pandas as pd
from scipy.stats import linregress


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a customer-level DataFrame with all engineered features."""

    # Ensure datetime
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)
    df["InvoiceDay"]  = df["InvoiceDate"].dt.date

    # ── Recency ──────────────────────────────────────────────────
    customer_data = df.groupby("CustomerID")["InvoiceDay"].max().reset_index()
    most_recent   = pd.to_datetime(df["InvoiceDay"].max())
    customer_data["InvoiceDay"] = pd.to_datetime(customer_data["InvoiceDay"])
    customer_data["Days_Since_Last_Purchase"] = (most_recent - customer_data["InvoiceDay"]).dt.days
    customer_data.drop(columns=["InvoiceDay"], inplace=True)

    # ── Frequency ────────────────────────────────────────────────
    total_tx = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    total_tx.rename(columns={"InvoiceNo": "Total_Transactions"}, inplace=True)

    total_qty = df.groupby("CustomerID")["Quantity"].sum().reset_index()
    total_qty.rename(columns={"Quantity": "Total_Products_Purchased"}, inplace=True)

    customer_data = customer_data.merge(total_tx,  on="CustomerID")
    customer_data = customer_data.merge(total_qty, on="CustomerID")

    # ── Monetary ─────────────────────────────────────────────────
    df["Total_Spend"] = df["UnitPrice"] * df["Quantity"]
    total_spend = df.groupby("CustomerID")["Total_Spend"].sum().reset_index()

    avg_tx_val = total_spend.merge(total_tx, on="CustomerID")
    avg_tx_val["Average_Transaction_Value"] = (
        avg_tx_val["Total_Spend"] / avg_tx_val["Total_Transactions"]
    )

    customer_data = customer_data.merge(total_spend, on="CustomerID")
    customer_data = customer_data.merge(
        avg_tx_val[["CustomerID", "Average_Transaction_Value"]], on="CustomerID"
    )

    # ── Product diversity ─────────────────────────────────────────
    unique_prods = df.groupby("CustomerID")["StockCode"].nunique().reset_index()
    unique_prods.rename(columns={"StockCode": "Unique_Products_Purchased"}, inplace=True)
    customer_data = customer_data.merge(unique_prods, on="CustomerID")

    # ── Behavioural ───────────────────────────────────────────────
    df["Day_Of_Week"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"]        = df["InvoiceDate"].dt.hour

    # Average days between purchases
    days_btw = (
        df.groupby("CustomerID")["InvoiceDay"]
        .apply(lambda x: pd.to_datetime(x).drop_duplicates().sort_values().diff().dt.days.dropna())
    )
    avg_days_btw = days_btw.groupby("CustomerID").mean().reset_index()
    avg_days_btw.rename(columns={"InvoiceDay": "Average_Days_Between_Purchases"}, inplace=True)

    fav_day = (
        df.groupby(["CustomerID", "Day_Of_Week"]).size().reset_index(name="Count")
    )
    fav_day = fav_day.loc[fav_day.groupby("CustomerID")["Count"].idxmax()][
        ["CustomerID", "Day_Of_Week"]
    ]

    fav_hour = (
        df.groupby(["CustomerID", "Hour"]).size().reset_index(name="Count")
    )
    fav_hour = fav_hour.loc[fav_hour.groupby("CustomerID")["Count"].idxmax()][
        ["CustomerID", "Hour"]
    ]

    customer_data = customer_data.merge(avg_days_btw, on="CustomerID")
    customer_data = customer_data.merge(fav_day,      on="CustomerID")
    customer_data = customer_data.merge(fav_hour,     on="CustomerID")

    # ── Geographic ────────────────────────────────────────────────
    cust_country = df.groupby(["CustomerID", "Country"]).size().reset_index(name="N")
    cust_main_country = (
        cust_country.sort_values("N", ascending=False).drop_duplicates("CustomerID")
    )
    cust_main_country["Is_UK"] = cust_main_country["Country"].apply(
        lambda x: 1 if x == "United Kingdom" else 0
    )
    customer_data = customer_data.merge(
        cust_main_country[["CustomerID", "Is_UK"]], on="CustomerID", how="left"
    )

    # ── Cancellation ──────────────────────────────────────────────
    cancelled = df[df["Transaction_Status"] == "Cancelled"]
    cancel_freq = (
        cancelled.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    )
    cancel_freq.rename(columns={"InvoiceNo": "Cancellation_Frequency"}, inplace=True)

    customer_data = customer_data.merge(cancel_freq, on="CustomerID", how="left")
    customer_data["Cancellation_Frequency"].fillna(0, inplace=True)
    customer_data["Cancellation_Rate"] = (
        customer_data["Cancellation_Frequency"] / customer_data["Total_Transactions"]
    )

    # ── Seasonality & trends ──────────────────────────────────────
    df["Year"]  = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    monthly = df.groupby(["CustomerID", "Year", "Month"])["Total_Spend"].sum().reset_index()

    seasonal = monthly.groupby("CustomerID")["Total_Spend"].agg(["mean", "std"]).reset_index()
    seasonal.rename(
        columns={"mean": "Monthly_Spending_Mean", "std": "Monthly_Spending_Std"}, inplace=True
    )
    seasonal["Monthly_Spending_Std"].fillna(0, inplace=True)

    def calc_trend(series):
        if len(series) > 1:
            x = np.arange(len(series))
            slope, *_ = linregress(x, series)
            return slope
        return 0.0

    trends = monthly.groupby("CustomerID")["Total_Spend"].apply(calc_trend).reset_index()
    trends.rename(columns={"Total_Spend": "Spending_Trend"}, inplace=True)

    customer_data = customer_data.merge(seasonal, on="CustomerID")
    customer_data = customer_data.merge(trends,   on="CustomerID")

    # ── Final dtypes ──────────────────────────────────────────────
    customer_data["CustomerID"] = customer_data["CustomerID"].astype(str)
    customer_data = customer_data.convert_dtypes()

    return customer_data
