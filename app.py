from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from main import main
from src.data_loader import REQUIRED_COLUMNS

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation & Recommendation System")
st.caption("Upload retail transaction data to explore customer groups. This unsupervised demo does not establish that recommendations improve business outcomes.")

uploaded_file = st.file_uploader("Upload a retail transaction CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        except Exception:
            st.error("Could not read this CSV. Check its encoding and file format.")
            st.stop()
    except Exception:
        st.error("Could not read this CSV. Check its file format.")
        st.stop()

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        st.error("Required columns are missing: " + ", ".join(missing))
        st.stop()
    if df.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

    clusters = st.number_input("Number of customer clusters", min_value=2, max_value=12, value=3)
    if st.button("Run segmentation", type="primary"):
        with tempfile.TemporaryDirectory(prefix="customer-segmentation-") as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.csv"
            df.to_csv(input_path, index=False)
            try:
                result = main(["--data", str(input_path), "--n_clusters", str(int(clusters)),
                               "--output_dir", str(root)])
                st.success("Analysis completed.")
                metrics = result["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Customers clustered", metrics["observations"])
                c2.metric("Silhouette score", f'{metrics["silhouette_score"]:.3f}')
                c3.metric("Davies–Bouldin score", f'{metrics["davies_bouldin_score"]:.3f}')

                plots_dir = result["plots_dir"]
                for filename, caption in [
                    ("elbow.png", "Elbow analysis"),
                    ("silhouette.png", "Silhouette scores across cluster counts"),
                    ("pca_variance.png", "PCA explained variance"),
                    ("radar_charts.png", "Cluster feature profiles"),
                    ("histograms.png", "Feature distributions by cluster"),
                    ("outliers.png", "Outlier detection summary"),
                ]:
                    image_path = plots_dir / filename
                    if image_path.exists():
                        st.image(str(image_path), caption=caption, use_container_width=True)

                st.subheader("Customer segments")
                st.dataframe(result["segments"], use_container_width=True)
                st.subheader("Product recommendations")
                st.dataframe(result["recommendations"], use_container_width=True)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The analysis failed. Verify the input values and try again.")
