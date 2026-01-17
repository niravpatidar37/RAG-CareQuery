import streamlit as st
import os
from dotenv import load_dotenv
from main import ingest_knowledge_base, get_rag_chain, rag

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(page_title="🩺 Symptom Checker (Advanced RAG)", page_icon="🧠", layout="wide")

# --- Sidebar: Admin / Ingestion ---
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Refresh Knowledge Base"):
        with st.spinner("Ingesting data from S3, Chunking & Embedding..."):
            try:
                ingest_knowledge_base()
                st.success("Knowledge Base Returned & Updated!")
                # Clear cache to force reload of retriever
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

# --- Main App ---
st.title("🧠 AI-Powered Symptom Checker")
st.caption("Powered by Parallel Hybrid Search & Semantic Chunking")

st.markdown("""
Ask any health-related question. The system searches the knowledge base using both **keyword matching** and **semantic understanding** in parallel for maximum speed.
""")

# --- Initialize RAG Chain (Cached) ---
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_chain():
    return get_rag_chain()

try:
    retriever, llm = load_chain()
except Exception as e:
    st.error(f"Failed to load RAG Chain: {e}")
    st.stop()

# --- User Input ---
query = st.text_input("💬 Enter your question (e.g., 'What are the symptoms of Diabetes?')")

if st.button("🔍 Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        # We handle retrieval separately to show the spinner and get sources
        with st.spinner("Searching Knowledge Base..."):
            try:
                # 1. Retrieve
                results = retriever.retrieve(query, top_k=5, score_threshold=0.0)
                
                if not results:
                    st.warning("I couldn't find any relevant information in the knowledge base.")
                else:
                    # 2. Context Construction (repeated from main.rag for streaming)
                    context = "\n\n".join([f"Source: {doc['metadata'].get('source', 'Unknown')}\nContent: {doc['content']}" for doc in results])
                    
                    # 3. Stream Answer
                    st.subheader("🩺 Answer")
                    
                    prompt = f"""You are an expert healthcare assistant. Use the following context to answer the user's question accurately.
If the answer is not in the context, say so. Do not hallucinate.

Context:
{context}

Question: {query}

Answer:"""
                    
                    # Define a generator for streaming
                    def stream_generator():
                        for chunk in llm.stream(prompt):
                            yield chunk.content

                    # Write stream
                    response_text = st.write_stream(stream_generator)
                    
                    # Calculate confidence proxy
                    confidence = results[0]['similarity_score'] if results else 0.0
                    st.info(f"Confidence Score: {confidence:.4f}")

                    # --- Display Sources in Expanders ---
                    st.subheader("📚 Sources")
                    col1, col2 = st.columns([1, 1])
                    for i, src in enumerate(results):
                        target_col = col1 if i % 2 == 0 else col2
                        with target_col:
                            with st.expander(f"Source {i+1} (Score: {src.get('similarity_score', 0):.2f})"):
                                st.markdown(f"**File:** `{src['metadata'].get('source', 'Unknown')}`")
                                st.text(src['content'][:500] + "...")

            except Exception as e:
                st.error(f"An error occurred: {e}")
