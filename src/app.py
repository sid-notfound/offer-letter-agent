# app.py
import streamlit as st
from agent_graph import app  # this is your LangGraph app

st.title("📄 Offer Letter Generator")

name = st.text_input("Enter Employee Name")

if st.button("Generate Offer Letter"):
    if name:
        with st.spinner("Generating..."):
            try:
                result = app.invoke({"name": name})
                st.success("Offer Letter Generated!")
                st.code(result["letter"])
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a name.")
