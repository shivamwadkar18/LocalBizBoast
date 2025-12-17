import streamlit as st
from data_utils import load_sales_csv
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="LocalBizBoost",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1 style="color:#2E8B57;">🌟 LocalBizBoost — AI Business Intelligence</h1>
    <p style="font-size:18px;">
        Upload your sales data to unlock AI-powered analytics, forecasting, insights & recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize storage
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.markdown("## 📂 DATA UPLOAD")
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
use_sample = st.sidebar.button("📊 Use Sample Data")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
st.sidebar.markdown("Use the left pages to explore features.")
st.sidebar.markdown("---")

# -----------------------------
# Load Data
# -----------------------------
if use_sample:
    st.session_state.df = load_sales_csv("sample_data/sample_sales.csv")
    st.success("✅ Sample dataset loaded successfully.")

elif uploaded:
    try:
        st.session_state.df = load_sales_csv(uploaded)
        st.success("✅ File processed successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")

# -----------------------------
# Show Preview
# -----------------------------
if st.session_state.df is None:
    st.info("👈 Upload or load sample data to begin.")
else:
    st.markdown("### 📊 Preview of Uploaded Data")
    st.dataframe(st.session_state.df.head(150), use_container_width=True)

    st.success("Data loaded! Navigate pages from sidebar.")
