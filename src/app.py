from main import run_ingestion,rag

from rag_retriever import RAGRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
# 4️⃣ Entry point
GOOGLE_API_KEY =  os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",temperature=0.8, api_key = GOOGLE_API_KEY)

        # --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 Symptom Checker", page_icon="🧠", layout="centered")

    # --- Title and Description ---
st.title("🧠 AI-Powered Symptom Checker")
st.write("Ask any health-related question and get AI-generated answers powered by RAG (Retrieval-Augmented Generation).")

query = st.text_input("💬 Enter your question (e.g., What is Diabetes?)")
    
    # --- Button to Trigger Search ---
if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving information..."):
            vectorstore, embedding_manager = run_ingestion()
            rag_retriever = RAGRetriever(vectorstore, embedding_manager)
            result = rag(query, rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)

        # --- Display Results ---
        st.subheader("🩺 Answer:")
        st.write(result["answer"])

        st.caption(f"Confidence: {result['confidence']:.2f}")

        # --- Display Sources ---
        st.markdown("### 📚 Sources:")
        for src in result["sources"]:
            st.markdown(f"- `{src['sources']}` (score: {src['score']:.3f})")

        # --- Expandable Context ---
        with st.expander("🔍 View Retrieved Context"):
            st.text(result["context"])




    

