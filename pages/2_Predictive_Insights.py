import streamlit as st
import altair as alt
import pandas as pd
from ml_engine import (
    daily_sales_series,
    weekly_sales_series,
    compute_growth,
    detect_anomalies,
    inventory_alerts,
    lstm_forecast
)
from ai_advisor import get_advice  # Optional AI integration

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.title("📊 Predictive Insights Dashboard")
st.write("Get AI-driven predictions, detect anomalies, analyze seasonality, and forecast inventory risks.")

# -----------------------------
# DATA VALIDATION
# -----------------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload your dataset on the **Home** page first.")
    st.stop()

df = st.session_state.df

# Ensure date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -----------------------------
# 📈 1. Growth & Trend Overview
# -----------------------------
st.subheader("📈 Growth & Trend Overview")

use_weekly = st.toggle("Aggregate weekly data for smoother trends", value=True)

if use_weekly:
    series = weekly_sales_series(df)
    freq_label = "Weekly"
else:
    series = daily_sales_series(df)
    freq_label = "Daily"

growth = compute_growth(series)
st.metric(f"{freq_label} Growth Rate", f"{growth:.2f}%" if growth != 0 else "Stable")

trend_df = series.reset_index().rename(columns={"date": "Date", 0: "Revenue"})
trend_df.columns = ["Date", "Revenue"]

trend_chart = (
    alt.Chart(trend_df)
    .mark_line(point=True)
    .encode(
        x="Date:T",
        y="Revenue:Q",
        tooltip=["Date:T", "Revenue:Q"]
    )
    .interactive()
)

st.altair_chart(trend_chart, use_container_width=True)

# -----------------------------
# 📅 2. Seasonality Patterns
# -----------------------------
st.subheader("📅 Seasonality Patterns")

df["month"] = df["date"].dt.month_name()
df["weekday"] = df["date"].dt.day_name()

monthly = df.groupby("month")["quantity"].sum().reindex([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])

weekly = df.groupby("weekday")["quantity"].sum().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])

col1, col2 = st.columns(2)
with col1:
    st.write("**📆 Monthly Sales Distribution**")
    st.bar_chart(monthly)
with col2:
    st.write("**🗓️ Weekday Sales Distribution**")
    st.bar_chart(weekly)

# -----------------------------
# 📉 3. Anomaly Detection
# -----------------------------
st.subheader("📉 Demand Anomaly Detection")

anomalies = detect_anomalies(series)

if anomalies.empty:
    st.success("✅ No unusual sales activity detected — demand looks stable.")
else:
    st.warning(f"⚠️ {len(anomalies)} anomalies detected — sudden sales spikes or drops found.")
    anomaly_chart = (
        alt.Chart(anomalies)
        .mark_circle(size=90, color="red")
        .encode(x="Date:T", y="Revenue:Q", tooltip=["Date:T", "Revenue:Q"])
    )
    st.altair_chart((trend_chart + anomaly_chart).interactive(), use_container_width=True)
    st.dataframe(anomalies)

# -----------------------------
# ⚠️ 4. Inventory Risk Overview
# -----------------------------
st.subheader("⚠️ Inventory Risk Forecast")

lead_time = st.number_input("Lead Time (days)", value=3, min_value=1)
threshold = st.number_input("Threshold (days left)", value=5, min_value=1)

alerts = inventory_alerts(df, lead_time_days=lead_time, threshold_days=threshold)

if alerts.empty:
    st.success("✅ Inventory levels look healthy — no immediate restocking needed.")
else:
    st.warning(f"⚠️ {len(alerts)} products may run out soon:")
    st.dataframe(alerts)

# -----------------------------
# 🧠 5. AI-Generated Business Recommendations
# -----------------------------
st.subheader("🧠 AI-Generated Recommendations")

prompt = f"""
Analyze this business data and provide 4 actionable insights:

- Sales growth: {growth:.2f}% {freq_label.lower()} change.
- Number of anomalies detected: {len(anomalies)}.
- Products at risk of stockout: {len(alerts)}.
- Describe short-term opportunities and risks.
Provide the response as bullet points.
"""

try:
    with st.spinner("🤖 AI analyzing your business performance..."):
        advice = get_advice(prompt)
    st.info(advice)
except Exception:
    st.warning("⚠️ AI Recommendation engine is currently unavailable. Check your API key.")

st.success("✅ Predictive insights generated successfully.")
