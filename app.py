import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation & Recommendation System")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Save file to correct location
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data.csv", index=False)

    if st.button("🚀 Run Model"):
        st.info("Running pipeline... please wait ⏳")

        # Run your main script
        subprocess.run(["python", "main.py", "--skip_plots"])

        st.success("✅ Model executed successfully!")

        # Show outputs
        st.subheader("📌 Customer Segments")
        seg = pd.read_csv("outputs/customer_segments.csv")
        st.dataframe(seg.head())

        st.subheader("⭐ Recommendations")
        rec = pd.read_csv("outputs/recommendations.csv")
        st.dataframe(rec.head())

        st.subheader("📊 Cluster Distribution")
        st.bar_chart(seg["cluster"].value_counts())