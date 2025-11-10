from vectors import VectorStore
from embedding import EmbeddingManager
from typing import List, Dict, Any, Tuple

class RAGRetriever:
    """Handles query-based retrieval from the vector Store"""
    def __init__(self,vector_store:VectorStore,embedding_manager:EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self,query:str,top_k: int = 5, score_threshold:float = 0.0)-> List[dict[str,Any ]]:
        """
            Retrieve relevant documents for a query

            Args:
                query: The search query
                top_k: Number of top results  to return
                score_threshold: Minimum similarity score threshold
            
            Returns:
                List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, score threshold: {score_threshold}")

        # generate query embedding
        query_embedding = self.embedding_manager.generate_embedding([query])[0]     # [query] -> convert full string into list

        # search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings= [query_embedding.tolist()],
                n_results= top_k
            )

            # Process results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i , (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score = 1-distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'distance':distance,
                            'similarity_score':similarity_score,
                            'distance': distance,
                            'rank':i+1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# vectorstore = VectorStore()
# embedding_manager = EmbeddingManager()
# rag_retriever = RAGRetriever(vectorstore,embedding_manager)

