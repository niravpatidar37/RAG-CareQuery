from data_ingested import data_ingestion, split_documents
from vectors import VectorStore
from embedding import EmbeddingManager
from rag_retriever import RAGRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_knowledge_base():
    """
    Trigger the full ingestion pipeline:
    S3 Download -> Semantic Chunking -> Embedding -> Vector Store Upsert
    """
    print("Starting data ingestion pipeline...")
    
    # 1. Load documents
    documents = data_ingestion()
    if not documents:
        print("No documents found to ingest.")
        return None, None

    # 2. Split (Semantic Chunking)
    chunks = split_documents(documents)

    # 3. Embedding Manager
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    
    # Generate embeddings (batching handled by LangChain/Gemini client usually, 
    # but we pass straight to vector store in some designs. 
    # Here we generate explicitly to pass to VectorStore wrapper)
    embeddings = embedding_manager.generate_embedding(texts)

    # 4. Store
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)
    
    # 5. Invalidate BM25 cache to force rebuild on next retrieval
    cache_path = "cache/bm25_index.pkl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("BM25 cache invalidated.")

    print("✅ Knowledge Base updated successfully!")
    return vectorstore, embedding_manager

def get_rag_chain():
    """
    Initialize the RAG components for RETRIEVAL only.
    Does NOT search or ingest.
    """
    print("Initializing RAG Chain...")
    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore() # Connects to existing persistence
    
    # Retriever (Hybrid)
    retriever = RAGRetriever(vectorstore, embedding_manager)
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, 
        max_retries=3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    return retriever, llm

def rag(query, retriever, llm, top_k=5, min_score=0.0):
    """
    Execute RAG search and generation
    """
    # 1. Retrieve
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    
    if not results:
        return {
            'answer': "I couldn't find any relevant information in the knowledge base.",
            'sources': [],
            'confidence': 0.0,
            'context': ''
        }

    # 2. Context Construction
    context = "\n\n".join([f"Source: {doc['metadata'].get('source', 'Unknown')}\nContent: {doc['content']}" for doc in results])
    
    # 3. Generation
    prompt = f"""You are a senior healthcare assistant with expertise in medical diagnostics. 

### INSTRUCTIONS:
1. **Analyze the Context**: Use ONLY the clinical information provided below.
2. **Handle Uncertainty**: If the context doesn't contain the answer, state that you don't have enough data.
3. **Structure**: Use bullet points for symptoms or lists for readability.
4. **Grounding**: Start with "Based on the provided medical context..."

### CLINICAL CONTEXT:
{context}

### USER QUESTION: 
{query}

Answer:"""
    
    response = llm.invoke(prompt)
    
    # Calculate confidence proxy (avg score of top result)
    confidence = results[0]['similarity_score'] if results else 0.0

    return {
        'answer': response.content,
        'sources': results, # Return full result objects
        'confidence': confidence,
        'context': context
    }
