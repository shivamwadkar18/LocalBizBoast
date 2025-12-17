# data_utils.py
import pandas as pd
import io

def load_sales_csv(path_or_buffer):
    """
    Load CSV/XLSX safely and standardize columns.
    Handles:
    ✅ Parsing dates
    ✅ Auto-renaming price/cost/stock/customer variations
    ✅ Cleaning numeric values
    ✅ Filling missing cost & stock
    """

    # Auto detect CSV vs XLSX
    try:
        if hasattr(path_or_buffer, "name") and path_or_buffer.name.endswith(".xlsx"):
            df = pd.read_excel(path_or_buffer)
        else:
            df = pd.read_csv(path_or_buffer, encoding="utf-8", engine="python")
    except Exception:
        df = pd.read_csv(path_or_buffer, encoding="latin1", engine="python")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # ✅ Smart auto-renaming
    column_map = {
        'price per unit': 'price',
        'unit price': 'price',
        'rate': 'price',
        'cost per unit': 'cost',
        'product name': 'product',
        'item': 'product',
        'product category': 'category',
        'total amount': 'total_amount',
        'transaction id': 'transaction_id',
        'customer id': 'customer'
    }

    df.rename(columns=column_map, inplace=True)

    # ✅ Required columns
    required = ['date', 'product', 'quantity', 'price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Dataset must contain: {required}"
        )

    # ✅ Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        raise ValueError("Some dates could not be parsed. Format should be YYYY-MM-DD.")

    # ✅ Clean numeric columns
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)

    # ✅ Optional cost
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0.0)
    else:
        df['cost'] = 0.0

    # ✅ Optional stock
    if 'stock' in df.columns:
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
    else:
        df['stock'] = 0

    # ✅ Clean product names
    df['product'] = df['product'].astype(str).str.strip()

    return df
