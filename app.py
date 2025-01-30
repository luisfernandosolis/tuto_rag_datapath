import streamlit as st

from dotenv import load_dotenv

load_dotenv()

## langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def vectordb_from_file(pdf_file):

    path_pdf_file = pdf_file.name

    ##read the file
    pdf_loader = PyPDFLoader(path_pdf_file)
    documents = pdf_loader.load()

    ## chunking file
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    ## embedding chunks
    embedding_model = OpenAIEmbeddings()

    ## indexing in vdb
    vector_db = FAISS.from_documents(chunks,embedding_model)

    return vector_db



with st.sidebar:
    st.title("RAG WITH STREAMLIT AND GROQ")
    
    pdf_file = st.file_uploader("uppload your file", type=["pdf"])

    load_button = st.button(label="let's go!", type="primary")
    clear_button = st.button(label="clear chat", type="secondary")
    
    if load_button:
        vector_db = vectordb_from_file(pdf_file)

        if vector_db:
            print("vdb creado!")


## show the chat
chat_container = st.container()


# the prompt input
input_container = st.container()


with input_container:
    with st.form(key="my_form",clear_on_submit=True):
        query = st.text_area("write a prompt!", key="input", height=80)
        submit_button = st.form_submit_button(label="Submit")

        if query and submit_button:
            print(query)






