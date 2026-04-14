"""Cluster-based product recommendation system."""
import pandas as pd


def build_recommendations(
    df: pd.DataFrame,
    customer_data_cleaned: pd.DataFrame,
    outliers_data: pd.DataFrame,
    top_n: int = 10,
    rec_n: int = 3,
) -> pd.DataFrame:
    """
    For each customer, recommend `rec_n` top-selling products from their cluster
    that they have NOT yet purchased.

    Returns a DataFrame with columns:
        CustomerID, cluster,
        Rec1_StockCode, Rec1_Description,
        Rec2_StockCode, Rec2_Description,
        Rec3_StockCode, Rec3_Description,
        + all original customer features
    """
    # Remove outlier customers from transaction data
    outlier_ids = outliers_data["CustomerID"].astype(str).unique()
    df_filt = df[~df["CustomerID"].astype(str).isin(outlier_ids)].copy()

    # Ensure consistent CustomerID type
    customer_data_cleaned = customer_data_cleaned.copy()
    customer_data_cleaned["CustomerID"] = customer_data_cleaned["CustomerID"].astype(str)
    df_filt["CustomerID"] = df_filt["CustomerID"].astype(str)

    # Merge to get cluster info per transaction
    merged = df_filt.merge(
        customer_data_cleaned[["CustomerID", "cluster"]], on="CustomerID", how="inner"
    )

    # Top-N best-selling products per cluster
    best = (
        merged.groupby(["cluster", "StockCode", "Description"])["Quantity"]
        .sum().reset_index()
        .sort_values(["cluster", "Quantity"], ascending=[True, False])
    )
    top_products = best.groupby("cluster").head(top_n)

    # Products each customer already purchased
    cust_purchases = (
        merged.groupby(["CustomerID", "cluster", "StockCode"])["Quantity"]
        .sum().reset_index()
    )

    recommendations = []
    for cluster_id in top_products["cluster"].unique():
        cluster_top = top_products[top_products["cluster"] == cluster_id]
        customers = customer_data_cleaned[
            customer_data_cleaned["cluster"] == cluster_id
        ]["CustomerID"]

        for cust_id in customers:
            already_bought = cust_purchases[
                (cust_purchases["CustomerID"] == cust_id) &
                (cust_purchases["cluster"] == cluster_id)
            ]["StockCode"].tolist()

            not_bought = cluster_top[
                ~cluster_top["StockCode"].isin(already_bought)
            ].head(rec_n)

            row = [cust_id, cluster_id]
            for _, prod in not_bought.iterrows():
                row += [prod["StockCode"], prod["Description"]]
            # Pad if fewer than rec_n recommendations
            while len(row) < 2 + rec_n * 2:
                row += [None, None]
            recommendations.append(row)

    rec_cols = ["CustomerID", "cluster"]
    for r in range(1, rec_n + 1):
        rec_cols += [f"Rec{r}_StockCode", f"Rec{r}_Description"]

    rec_df = pd.DataFrame(recommendations, columns=rec_cols)
    result = customer_data_cleaned.merge(rec_df, on=["CustomerID", "cluster"], how="right")

    print(f"      Recommendations generated for {len(result):,} customers")
    return result
