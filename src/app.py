import streamlit as st
import os
from dotenv import load_dotenv
from main import ingest_knowledge_base, get_rag_chain, rag  # rag imported for potential internal use

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="🩺 Symptom Checker (Advanced RAG)",
    page_icon="🧠",
    layout="wide",
)

# --- Sidebar: Admin / Ingestion ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Sync your knowledge base and configure the AI model.")
    
    # Model Selection
    st.subheader("🤖 Model Configuration")
    model_options = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-2.0-flash-exp"
    ]
    
    # Check if model selection changed to auto-clear cache
    if "current_model" not in st.session_state:
        st.session_state.current_model = model_options[0]

    selected_model = st.selectbox("Select AI Model", model_options, index=model_options.index(st.session_state.current_model))

    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.cache_resource.clear()
        st.rerun()

    # Retrieval Configuration
    st.subheader("🔍 Search Settings")
    top_k = st.slider("Top K (Documents to retrieve)", min_value=1, max_value=10, value=5)
    score_threshold = st.slider("Score Threshold (RRF)", min_value=0.0, max_value=0.1, value=0.03, step=0.005)
    st.caption("Adjust threshold to filter out low-confidence results.")

    if st.button("🔄 Refresh Knowledge Base"):
        with st.spinner("Ingesting data from S3, Chunking & Embedding..."):
            try:
                ingest_knowledge_base()
                st.success("Knowledge Base Returned & Updated!")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    if st.button("🗑️ Reset App Cache"):
        st.cache_resource.clear()
        st.success("App cache cleared!")
        st.rerun()

# --- Main App ---
st.title("🧠 AI-Powered Symptom Checker")
st.caption("Powered by Parallel Hybrid Search & Semantic Chunking")

st.markdown("""
Ask any health-related question. The system searches the knowledge base using both **keyword matching** and **semantic understanding** in parallel for maximum speed.
""")

# --- Initialize RAG Chain (Cached) ---
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_chain(model_name):
    # Pass model name to get_rag_chain if we want to customize, 
    # but for now we'll just override the llm here or update get_rag_chain
    retriever, llm = get_rag_chain()
    # Override model if different from default
    llm.model = model_name
    return retriever, llm

try:
    retriever, llm = load_chain(selected_model)
    with st.sidebar:
        st.divider()
        st.info(f"Model: {llm.model}")
except Exception as e:
    st.error(f"Failed to load RAG Chain: {e}")
    st.stop()

# --- User Input ---
query = st.text_input("💬 Enter your question (e.g., 'What are the symptoms of Diabetes?')")

if st.button("🔍 Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching Knowledge Base..."):
            try:
                # 1. Retrieve
                results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)

                if not results:
                    st.warning("I couldn't find any relevant information in the knowledge base.")
                else:
                    # 2. Build context from results
                    context = "\n\n".join(
                        [
                            f"Source: {doc['metadata'].get('source', 'Unknown')}\nContent: {doc['content']}"
                            for doc in results
                        ]
                    )

                    # 3. Stream Answer
                    st.subheader("🩺 Answer")

                    prompt = f"""You are a senior healthcare assistant with expertise in medical diagnostics and internal medicine. 

### INSTRUCTIONS:
1. **Analyze the Context**: Carefully review the provided medical documentation excerpts.
2. **Be Precise**: Answer the question using ONLY the provided context. 
3. **Handle Uncertainty**: If the context doesn't contain the answer, state: "I'm sorry, my current medical database doesn't contain information about that specific query."
4. **Structure**: Use bullet points for symptoms or lists.
5. **Tone**: Professional, empathetic, and factual.
6. **Grounding**: Always start with "Based on the available medical records..."

### CONTEXT:
{context}

### USER QUESTION: 
{query}

### FINAL RESPONSE GUIDELINES:
- No hallucinations.

Answer:"""

                    def stream_generator():
                        for chunk in llm.stream(prompt):
                            yield chunk.content

                    response_text = st.write_stream(stream_generator)

                    # Confidence proxy
                    confidence = results[0].get("similarity_score", 0.0) if results else 0.0
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
