import streamlit as st


with st.sidebar:
    st.title("RAG WITH STREAMLIT AND GROQ")
    
    pdf_file = st.file_uploader("uppload your file", type=["pdf"])

    load_button = st.button(label="let's go!", type="primary")
    clear_button = st.button(label="clear chat", type="secondary")



