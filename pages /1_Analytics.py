import streamlit as st
from ml_engine import compute_kpis, top_products, inventory_alerts, daily_sales_series, lstm_forecast
import altair as alt
import io
import pandas as pd
import numpy as np

st.title("📊 Business Analytics Dashboard")

# Check data
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please upload data on the Home page first.")
    st.stop()

df = st.session_state.df

# ---------------------------
# ✅ KPIs
# ---------------------------
st.subheader("📈 Key Business Metrics")

kpis = compute_kpis(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{kpis['total_revenue']:,.2f}")
col2.metric("Total Quantity", f"{kpis['total_quantity']}")
col3.metric("Total Cost", f"₹{kpis['total_cost']:,.2f}")
col4.metric("Profit", f"₹{kpis['profit']:,.2f}")

# ---------------------------
# ✅ Smart Business Insights
# ---------------------------
st.subheader("🧠 Smart Business Insights")

insights = []
series = daily_sales_series(df)

# Insight: Trend
if series.iloc[-1] > series.mean():
    insights.append("📈 Recent sales are above average — business trend is positive.")
else:
    insights.append("📉 Recent sales are below average — consider marketing or discounts.")

# Insight: Top product
tp = top_products(df, n=10)
if not tp.empty:
    best = tp.iloc[0]
    insights.append(f"🏆 Top Product: **{best['product']}** with revenue ₹{best['revenue']:.0f}")

# Insight: Inventory
alerts = inventory_alerts(df)
if alerts.empty:
    insights.append("✅ Inventory looks healthy — no urgent restock needed.")
else:
    insights.append(f"⚠️ {len(alerts)} products may run out soon — check restocking.")

for note in insights:
    st.write(note)

# ---------------------------
# ✅ Daily Sales Trend + Anomaly Detection
# ---------------------------
st.subheader("📅 Daily Revenue Trend")

daily_df = series.reset_index().rename(columns={'date':'Date', 0:'Revenue'})
daily_df.columns = ["Date", "Revenue"]

# Base line chart
line_chart = alt.Chart(daily_df).mark_line(point=True).encode(
    x='Date:T',
    y='Revenue:Q',
    tooltip=['Date:T', 'Revenue:Q']
)

# ✅ Anomaly detection using Z-Score
daily_df['zscore'] = (daily_df['Revenue'] - daily_df['Revenue'].mean()) / daily_df['Revenue'].std()
anomalies = daily_df[(daily_df['zscore'] > 2) | (daily_df['zscore'] < -2)]

if not anomalies.empty:
    anomaly_chart = alt.Chart(anomalies).mark_circle(size=90, color='red').encode(
        x='Date:T', y='Revenue:Q', tooltip=['Date:T', 'Revenue:Q']
    )
    st.altair_chart((line_chart + anomaly_chart).interactive(), use_container_width=True)
    st.warning("🚨 Unusual sales spikes or drops detected:")
    st.table(anomalies[['Date', 'Revenue']])
else:
    st.altair_chart(line_chart.interactive(), use_container_width=True)
    st.success("✅ No major anomalies detected. Sales look normal.")

# ---------------------------
# ✅ Sales Seasonality Heatmap
# ---------------------------
st.subheader("🌡️ Sales Seasonality Heatmap")

df['date'] = pd.to_datetime(df['date'])
df['Month'] = df['date'].dt.month_name()
df['Day'] = df['date'].dt.day_name()

heatmap_df = df.groupby(['Month', 'Day'])['quantity'].sum().reset_index()

heatmap_chart = alt.Chart(heatmap_df).mark_rect().encode(
    x=alt.X('Day:N', sort=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']),
    y=alt.Y('Month:N', sort=['January','February','March','April','May','June','July','August','September','October','November','December']),
    color=alt.Color('quantity:Q', scale=alt.Scale(scheme='reds')),
    tooltip=['Month', 'Day', 'quantity']
)

st.altair_chart(heatmap_chart, use_container_width=True)

# ---------------------------
# ✅ Weekly & Monthly Trends
# ---------------------------
st.subheader("📆 Weekly & Monthly Sales Trends")

weekly = df.groupby(df['date'].dt.to_period('W'))['quantity'].sum().reset_index()
weekly['date'] = weekly['date'].dt.start_time

monthly = df.groupby(df['date'].dt.to_period('M'))['quantity'].sum().reset_index()
monthly['date'] = monthly['date'].dt.to_timestamp()

tab1, tab2 = st.tabs(["📅 Weekly Sales", "📅 Monthly Sales"])

with tab1:
    st.altair_chart(
        alt.Chart(weekly).mark_line(point=True).encode(
            x='date:T', y='quantity:Q', tooltip=['date:T','quantity:Q']
        ).interactive(),
        use_container_width=True
    )

with tab2:
    st.altair_chart(
        alt.Chart(monthly).mark_line(point=True).encode(
            x='date:T', y='quantity:Q', tooltip=['date:T','quantity:Q']
        ).interactive(),
        use_container_width=True
    )

# ---------------------------
# ✅ Top Products
# ---------------------------
st.subheader("🏆 Top Products by Revenue")
st.table(tp)

st.altair_chart(
    alt.Chart(tp).mark_bar().encode(
        x='revenue:Q',
        y=alt.Y('product:N', sort='-x'),
        tooltip=['product', 'quantity', 'revenue']
    ),
    use_container_width=True
)

# ---------------------------
# ✅ Inventory Alerts
# ---------------------------
st.subheader("⚠️ Inventory Alerts")

lead = st.number_input("Lead time (days)", value=3, min_value=1)
threshold = st.number_input("Threshold (days left)", value=5, min_value=1)

alerts = inventory_alerts(df, lead_time_days=lead, threshold_days=threshold)

if alerts.empty:
    st.success("✅ No low-stock alerts — inventory looks healthy!")
else:
    st.warning("⚠️ Low-stock items detected:")
    st.table(alerts)

# ---------------------------
# ✅ Download Cleaned CSV
# ---------------------------
st.subheader("📥 Download Cleaned Data")

buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)

st.download_button(
    label="📎 Download Cleaned CSV",
    data=buf,
    file_name="cleaned_sales.csv",
    mime="text/csv"
)


