# Customer Segmentation & Recommendation System

A Python and Streamlit demo that groups customers from retail transaction data. It engineers customer-level behavior features, removes outliers, scales features, applies PCA and K-Means, reports cluster metrics, and creates within-cluster product recommendations.

## Input data

The CSV must include these columns: `CustomerID`, `Description`, `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, and `Country`. The sample file is a small synthetic dataset for checking the interface and pipeline; it is not real customer data and is not suitable for business conclusions. For a more meaningful analysis, use the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

## Run locally

Requirements: Python 3.10 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV in the dashboard. The dashboard validates the required columns, displays the input preview, and reports the selected cluster count with silhouette and Davies–Bouldin metrics. Use `data/sample_retail.csv` to try the project.

For command-line use:

```bash
python main.py --data data/sample_retail.csv --n_clusters 3 --output_dir run
```

Results are saved under `run/outputs/` and plots under `run/plots/`. Use `--skip_plots` for a faster run without charts.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Interpretation and limitations

Silhouette and Davies–Bouldin scores describe geometric separation in the feature space; they do not prove that segments are useful or that recommendations increase sales. The recommendation list is based on products bought by other customers in the same cluster, not an evaluated recommendation model. Review feature choices, data quality, and business context before drawing conclusions. The sample dataset is synthetic. No model is trained to predict future behavior.
