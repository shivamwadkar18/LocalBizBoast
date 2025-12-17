# ai_advisor.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Load Groq Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Add it to .env")

# Init Client
client = Groq(api_key=api_key)


def generate_business_context(df, compute_kpis, top_products):
    kpis = compute_kpis(df)
    tp = top_products(df, n=5)

    days = max(1, (df['date'].max() - df['date'].min()).days + 1)
    avg_daily_sales = kpis['total_revenue'] / days

    summary = (
        f"Business Summary:\n"
        f"- Total Revenue: ₹{kpis['total_revenue']:,.0f}\n"
        f"- Profit: ₹{kpis['profit']:,.0f}\n"
        f"- Average Daily Sales: ₹{avg_daily_sales:,.0f}\n"
        f"- Total Quantity Sold: {kpis['total_quantity']}\n"
        f"- Top Products: {', '.join(tp['product'].tolist())}\n"
        f"- Total Orders: {len(df)}"
    )
    return summary


def get_advice(prompt):
    """
    Uses llama-3.1-8b-instant (supported + available).
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        # ✅ Correct content access
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI error: {str(e)}"
