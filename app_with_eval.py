import streamlit as st
from rag_pipeline import load_vector_store, answer_question
import os

# Page config
st.set_page_config(
    page_title="AI Customer Support Copilot",
    page_icon="🤖",
    layout="wide"
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Support Assistant", "📊 Evaluation Harness", "📝 Test Cases"])

with tab1:
    st.title("🤖 AI Customer Support Copilot")
    
    # Your original RAG interface code
    from ingest import load_and_split_docs
    from rag_pipeline import create_vector_store
    
    uploaded_file = st.file_uploader("Upload support document (PDF)", type=["pdf"])
    
    if uploaded_file:
        os.makedirs("documents", exist_ok=True)
        file_path = os.path.join("documents", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Document uploaded")
        
        chunks = load_and_split_docs(file_path)
        vectorstore = create_vector_store(chunks)
        st.success("Knowledge base created")
    
    question = st.text_input("Ask a support question")
    
    if question:
        try:
            vectorstore = load_vector_store()
            answer = answer_question(vectorstore, question)
            st.write("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}. Please upload a PDF document first.")

with tab2:
    from evaluation_harness import create_evaluation_dashboard
    create_evaluation_dashboard()

with tab3:
    from evaluation_harness import show_test_cases_creator
    show_test_cases_creator()