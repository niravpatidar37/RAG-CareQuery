from main import ingest_knowledge_base, get_rag_chain, rag
import os
from dotenv import load_dotenv

# Force load .env
load_dotenv()

def test_pipeline():
    print("--- 1. Testing Ingestion ---")
    try:
        # Run ingestion (this requires S3 access, assuming credentials are in .env)
        # If S3 fails, we rely on local files if they exist.
        ingest_knowledge_base()
        print("Ingestion Passed.")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
        return

    print("\n--- 2. Testing Retrieval ---")
    try:
        retriever, llm = get_rag_chain()
        query = "What are the symptoms of Diabetes?"
        
        result = rag(query, retriever, llm)
        print("Query:", query)
        print("Confidence:", result['confidence'])
        print("Answer Snippet:", result['answer'][:100])
        print("Sources:", len(result['sources']))
        print("Retrieval Passed.")
    except Exception as e:
        print(f"Retrieval Failed: {e}")

if __name__ == "__main__":
    test_pipeline()
