"""Customer segmentation pipeline: cleaning → features → clustering → recommendations."""

import argparse
from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_eng import build_customer_features
from src.outliers import remove_outliers
from src.scaling import scale_features
from src.pca import apply_pca
from src.clustering import find_optimal_k, run_kmeans
from src.evaluation import evaluate_clusters
from src.profiling import plot_radar, plot_histograms
from src.recommender import build_recommendations
from src.utils import save_output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Customer Segmentation & Recommendation System")
    parser.add_argument("--data", default="data/data.csv", help="Path to input retail CSV")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of customer clusters")
    parser.add_argument("--n_pca", type=int, default=6, help="Maximum PCA components")
    parser.add_argument("--skip_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--output_dir", default=".", help="Directory for plots and CSV results")
    return parser.parse_args(argv)


def run_pipeline(args):
    output_root = Path(args.output_dir)
    plots_dir = output_root / "plots"
    outputs_dir = output_root / "outputs"

    print("\nCustomer Segmentation & Recommendation System")
    df = load_data(args.data)
    df = clean_data(df, plots_dir=str(plots_dir), skip_plots=args.skip_plots)
    customer_data = build_customer_features(df)
    if len(customer_data) < 4:
        raise ValueError("At least 4 customers with usable transactions are required.")

    customer_data_cleaned, outliers_data = remove_outliers(
        customer_data, plots_dir=str(plots_dir), skip_plots=args.skip_plots
    )
    if len(customer_data_cleaned) < 3:
        raise ValueError("Too few non-outlier customers remain to form meaningful clusters.")

    customer_data_scaled = scale_features(customer_data_cleaned)
    n_components = min(args.n_pca, customer_data_scaled.shape[1], len(customer_data_scaled) - 1)
    if n_components < 1:
        raise ValueError("Not enough customer features to run PCA.")
    customer_data_pca, _ = apply_pca(
        customer_data_scaled, n_components=n_components,
        plots_dir=str(plots_dir), skip_plots=args.skip_plots,
    )

    n_clusters = args.n_clusters
    if n_clusters < 2 or n_clusters >= len(customer_data_cleaned):
        raise ValueError("n_clusters must be at least 2 and smaller than the number of inlier customers.")
    if not args.skip_plots and len(customer_data_pca) >= 4:
        find_optimal_k(customer_data_pca, plots_dir=str(plots_dir))

    customer_data_cleaned, customer_data_pca = run_kmeans(
        customer_data_pca, customer_data_cleaned, n_clusters=n_clusters
    )
    metrics = evaluate_clusters(customer_data_pca)
    if not args.skip_plots:
        plot_radar(customer_data_cleaned, plots_dir=str(plots_dir))
        plot_histograms(customer_data_cleaned, plots_dir=str(plots_dir))

    recommendations = build_recommendations(df, customer_data_cleaned, outliers_data)
    save_output(customer_data_cleaned, str(outputs_dir / "customer_segments.csv"))
    save_output(recommendations, str(outputs_dir / "recommendations.csv"))
    save_output(customer_data_pca, str(outputs_dir / "customer_pca.csv"))
    print("\nDone. Results saved under", output_root.resolve())
    return {"segments": customer_data_cleaned, "recommendations": recommendations,
            "metrics": metrics, "plots_dir": plots_dir, "outputs_dir": outputs_dir}


def main(argv=None):
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
