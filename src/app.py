import streamlit as st
from agent_graph import app  # LangGraph app

st.title("📄 Offer Letter Generator")

name = st.text_input("Enter Employee Name")

if st.button("Generate Offer Letter"):
    clean_name = name.strip()

    if clean_name:
        with st.spinner("Generating..."):
            try:
                # Debug: print to Streamlit log
                st.text(f"🛠 Invoking with: {clean_name}")
                result = app.invoke({"name": clean_name})
                st.success("✅ Offer Letter Generated!")
                st.code(result["letter"])
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a name.")
