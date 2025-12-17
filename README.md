#LocalBizBoost


📈 LocalBizBoost — AI-Driven Business Analytics Platform

LocalBizBoost is an AI-powered analytics platform built to help local shopkeepers and small businesses analyze sales data, forecast demand, and receive AI-generated business insights — without requiring technical expertise.

🚀 Key Highlights

📊 Automated Data Processing
Handles CSV/XLSX uploads with smart cleaning, validation, and normalization.

📈 Business KPIs & Insights
Revenue, profit, average daily sales, top products, and sales trends.

🔮 Sales Forecasting
LSTM-based time-series forecasting using PyTorch with adaptive smoothing.

📦 Inventory Alerts
Identifies low-stock products based on sales velocity and lead-time logic.

🤖 AI Business Advisor
Uses an LLM to generate human-readable summaries and actionable recommendations.

🖥️ Interactive Dashboard
Built with Streamlit for non-technical users.

🛠️ Tech Stack

Python | Pandas | NumPy | Scikit-learn | PyTorch (LSTM) | Streamlit | LLM APIs | Matplotlib

📂 Project Structure
Home.py        → Streamlit UI  
data_utils.py → Data preprocessing  
ml_engine.py  → KPIs, forecasting, inventory logic  
ai_advisor.py → LLM-based business insights  

▶️ Run Locally
pip install -r requirements.txt
streamlit run Home.py

🎯 What This Project Demonstrates

End-to-end AI + data analytics pipeline

Practical time-series forecasting

LLM integration for business insights

Real-world, business-focused problem solving
