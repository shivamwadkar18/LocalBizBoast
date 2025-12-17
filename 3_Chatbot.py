import streamlit as st
from ai_advisor import generate_business_context, get_advice
from ml_engine import compute_kpis, top_products

st.title("🤖 AI Business Chatbot")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please upload data in the Home page first.")
    st.stop()

df = st.session_state.df

if "ai_context" not in st.session_state:
    st.session_state.ai_context = generate_business_context(df, compute_kpis, top_products)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your AI Business Advisor..."):
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_advice(f"{st.session_state.ai_context}\n\nUser Question: {prompt}")
        st.session_state.messages.append({"role":"assistant","content":response})
        st.markdown(response)
