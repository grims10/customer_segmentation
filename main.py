"""
Customer Segmentation & Recommendation System
==============================================
End-to-end pipeline: data cleaning → feature engineering →
outlier removal → scaling → PCA → K-Means → recommendations.

Usage:
    python main.py
    python main.py --data data/data.csv --n_clusters 3 --skip_plots
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from src.data_loader   import load_data
from src.preprocessing import clean_data
from src.feature_eng   import build_customer_features
from src.outliers      import remove_outliers
from src.scaling       import scale_features
from src.pca           import apply_pca
from src.clustering    import find_optimal_k, run_kmeans
from src.evaluation    import evaluate_clusters
from src.profiling     import plot_radar, plot_histograms
from src.recommender   import build_recommendations
from src.utils         import save_output


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Segmentation & Recommendation System")
    parser.add_argument("--data",       default="data/data.csv", help="Path to raw CSV file")
    parser.add_argument("--n_clusters", type=int, default=3,     help="Number of K-Means clusters (default: 3)")
    parser.add_argument("--n_pca",      type=int, default=6,     help="Number of PCA components (default: 6)")
    parser.add_argument("--skip_plots", action="store_true",     help="Skip generating plots (faster run)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  Customer Segmentation & Recommendation System")
    print("="*60)

    print("\n[1/8] Loading data...")
    df = load_data(args.data)
    print(f"      Rows loaded: {len(df):,}")

    print("\n[2/8] Cleaning data...")
    df = clean_data(df, plots_dir="plots", skip_plots=args.skip_plots)
    print(f"      Rows after cleaning: {len(df):,}")

    print("\n[3/8] Engineering customer-level features...")
    customer_data = build_customer_features(df)
    print(f"      Customers: {len(customer_data):,}  |  Features: {customer_data.shape[1]-1}")

    print("\n[4/8] Detecting & removing outliers (IsolationForest)...")
    customer_data_cleaned, outliers_data = remove_outliers(
        customer_data, contamination=0.05,
        plots_dir="plots", skip_plots=args.skip_plots
    )
    print(f"      Cleaned: {len(customer_data_cleaned):,}  |  Outliers: {len(outliers_data):,}")

    print("\n[5/8] Scaling features (StandardScaler)...")
    customer_data_scaled = scale_features(customer_data_cleaned)

    print(f"\n[6/8] Dimensionality reduction (PCA → {args.n_pca} components)...")
    customer_data_pca, pca_model = apply_pca(
        customer_data_scaled, n_components=args.n_pca,
        plots_dir="plots", skip_plots=args.skip_plots
    )

    print(f"\n[7/8] K-Means clustering (k={args.n_clusters})...")
    if not args.skip_plots:
        print("      Running Elbow & Silhouette analysis (~1-2 min)...")
        find_optimal_k(customer_data_pca, plots_dir="plots")

    customer_data_cleaned, customer_data_pca = run_kmeans(
        customer_data_pca, customer_data_cleaned, n_clusters=args.n_clusters
    )

    print("\n[8/8] Evaluating & profiling clusters...")
    evaluate_clusters(customer_data_pca)

    if not args.skip_plots:
        plot_radar(customer_data_cleaned, plots_dir="plots")
        plot_histograms(customer_data_cleaned, plots_dir="plots")

    print("      Building product recommendations...")
    recommendations_df = build_recommendations(df, customer_data_cleaned, outliers_data)

    save_output(customer_data_cleaned, "outputs/customer_segments.csv")
    save_output(recommendations_df,    "outputs/recommendations.csv")
    save_output(customer_data_pca,     "outputs/customer_pca.csv")

    print("\n" + "="*60)
    print("  Done! Results saved to outputs/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
