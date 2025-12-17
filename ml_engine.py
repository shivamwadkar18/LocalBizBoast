import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------
# ✅ KPI Calculation
# ---------------------------------------------------
def compute_kpis(df):
    df = df.copy()
    df['revenue'] = df['quantity'] * df['price']
    df['cost_total'] = df['quantity'] * df.get('cost', 0.0)

    total_revenue = df['revenue'].sum()
    total_qty = df['quantity'].sum()
    total_cost = df['cost_total'].sum()
    profit = total_revenue - total_cost

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    daily_sales = df.groupby('date')['revenue'].sum()
    avg_daily_sales = daily_sales.mean() if not daily_sales.empty else 0.0

    return {
        'total_revenue': float(total_revenue),
        'total_quantity': int(total_qty),
        'total_cost': float(total_cost),
        'profit': float(profit),
        'avg_daily_sales': float(avg_daily_sales)
    }


# ---------------------------------------------------
# ✅ Top Products
# ---------------------------------------------------
def top_products(df, n=5):
    tmp = df.groupby('product', as_index=False).agg({'quantity': 'sum', 'price': 'mean'})
    tmp['revenue'] = tmp['quantity'] * tmp['price']
    tmp = tmp.sort_values('revenue', ascending=False).head(n)
    return tmp[['product', 'quantity', 'revenue']]


# ---------------------------------------------------
# ✅ Daily and Weekly Sales Series
# ---------------------------------------------------
def daily_sales_series(df):
    df2 = df.copy()
    df2['revenue'] = df2['quantity'] * df2['price']
    series = df2.groupby('date')['revenue'].sum().sort_index()
    idx = pd.date_range(series.index.min(), series.index.max(), freq='D')
    series = series.reindex(idx, fill_value=0)
    series.index.name = 'date'
    return series


def weekly_sales_series(df):
    """Aggregate sales weekly to smooth noisy data."""
    df2 = df.copy()
    df2['revenue'] = df2['quantity'] * df2['price']
    df2 = df2.set_index('date').resample('W').sum()
    df2.index.name = 'date'
    return df2['revenue']


# ---------------------------------------------------
# ✅ Trend & Growth Analysis
# ---------------------------------------------------
def compute_growth(series, period=1):
    """Compute simple percentage growth between last two points."""
    if len(series) < period + 1:
        return 0.0
    try:
        growth = ((series.iloc[-1] - series.iloc[-1 - period]) / series.iloc[-1 - period]) * 100
        return round(growth, 2)
    except Exception:
        return 0.0


def moving_average(series, window=7):
    """Smooth series to highlight trend."""
    return series.rolling(window=window, min_periods=1).mean()


def detect_anomalies(series, threshold=2.5):
    """Detect spikes or drops using z-score."""
    if len(series) < 5:
        return pd.DataFrame()
    z = (series - series.mean()) / series.std()
    anomalies = series[(z > threshold) | (z < -threshold)]
    return anomalies.reset_index().rename(columns={'index': 'Date', 0: 'Revenue'})


# ---------------------------------------------------
# ✅ LSTM Model Definition
# ---------------------------------------------------
class LSTMForecastModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------------------------------------------------
# ✅ Adaptive LSTM Forecast (Auto Daily/Weekly)
# ---------------------------------------------------
def lstm_forecast(series, steps=7, freq='auto'):
    """LSTM-based time series forecasting with auto smoothing."""
    if series is None or len(series) < 20:
        idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=steps, freq='D')
        return pd.Series([series.mean()] * steps, index=idx)

    # Auto smooth if noisy
    if freq == 'auto' and len(series) > 60 and series.std() / series.mean() > 0.5:
        series = series.resample('W').mean()
        freq = 'W'
    else:
        freq = 'D' if freq == 'auto' else freq

    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    window = min(14, len(series) // 3)
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    X_tensor = torch.tensor(np.array(X).reshape(-1, window, 1), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

    model = LSTMForecastModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(300):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

    last_seq = data[-window:]
    preds_scaled = []
    for _ in range(steps):
        inp = torch.tensor(last_seq.reshape(1, window, 1), dtype=torch.float32)
        pred = model(inp).item()
        preds_scaled.append(pred)
        last_seq = np.append(last_seq[1:], pred)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    preds = pd.Series(preds).rolling(window=3, min_periods=1).mean().values

    idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=steps, freq=freq)
    return pd.Series(preds, index=idx)


# ---------------------------------------------------
# ✅ Inventory Alerts
# ---------------------------------------------------
def inventory_alerts(df, lead_time_days=3, threshold_days=5):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    totals = df.groupby('product').agg({'quantity': 'sum'})
    period_days = max(1, (df['date'].max() - df['date'].min()).days + 1)
    totals['avg_daily'] = totals['quantity'] / period_days

    if 'stock' in df.columns:
        latest_stock = df.drop_duplicates(subset=['product'], keep='last').set_index('product')['stock']
    else:
        latest_stock = pd.Series(0, index=totals.index)

    totals['stock'] = latest_stock.reindex(totals.index).fillna(0)
    totals['days_left_after_lead'] = (
        totals['stock'] - (totals['avg_daily'] * lead_time_days)
    ) / totals['avg_daily'].replace(0, np.nan)

    alerts = totals[totals['days_left_after_lead'] < threshold_days].reset_index()
    alerts = alerts[['product', 'stock', 'avg_daily', 'days_left_after_lead']]
    alerts.rename(columns={'days_left_after_lead': 'days_left'}, inplace=True)
    return alerts
