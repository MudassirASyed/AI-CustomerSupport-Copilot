from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def answer_question(vectorstore, question):

    # retrieve relevant docs
    docs = vectorstore.similarity_search(question, k=5)  

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the provided context.

CONTEXT (multiple sections from CV):
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Search the context for SPECIFIC dates, roles, and durations
2. If information is spread across multiple sections, COMBINE it
3. If you find the information, provide it with SPECIFIC details
4. Only say "no information" if you've thoroughly checked all context
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )

    vectorstore.persist()

    return vectorstore


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )

    return vectorstore