from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_docs(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,    
        chunk_overlap=300     
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")  # Debug: See how many chunks
    return chunks
   
