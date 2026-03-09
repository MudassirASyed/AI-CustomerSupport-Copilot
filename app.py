import streamlit as st
from ingest import load_and_split_docs
from rag_pipeline import create_vector_store, load_vector_store, answer_question
import os

st.title("AI Customer Support Copilot")

uploaded_file = st.file_uploader("Upload support document (PDF)", type=["pdf"])

if uploaded_file:

    file_path = os.path.join("documents", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded")

    chunks = load_and_split_docs(file_path)

    vectorstore = create_vector_store(chunks)

    st.success("Knowledge base created")

question = st.text_input("Ask a support question")

if question:

    vectorstore = load_vector_store()

    answer = answer_question(vectorstore, question)

    st.write("### Answer")
    st.write(answer)