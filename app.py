import streamlit as st
import pandas as pd
import os
import sys

from main import main  # ✅ direct import instead of subprocess

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation & Recommendation System")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # 🔹 Handle encoding safely
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        df = pd.read_csv(uploaded_file, encoding='latin-1')

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # 🔹 Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data.csv", index=False)

    if st.button("🚀 Run Model"):
        st.info("Running pipeline... please wait ⏳")

        # Ensure outputs folder exists
        os.makedirs("outputs", exist_ok=True)

        # 🔥 Run pipeline directly
        try:
            sys.argv = ["main.py", "--skip_plots"]  # simulate CLI args
            main()
            st.success("✅ Model executed successfully!")
        except Exception as e:
            st.error(f"Error running model: {e}")

        # 🔹 Show outputs
        if os.path.exists("outputs/customer_segments.csv"):
            seg = pd.read_csv("outputs/customer_segments.csv")
            st.subheader("📌 Customer Segments")
            st.dataframe(seg.head())

            st.subheader("📊 Cluster Distribution")
            st.bar_chart(seg["cluster"].value_counts())
        else:
            st.warning("Customer segments file not found.")

        if os.path.exists("outputs/recommendations.csv"):
            rec = pd.read_csv("outputs/recommendations.csv")
            st.subheader("⭐ Recommendations")
            st.dataframe(rec.head())
        else:
            st.warning("Recommendations file not found.")
