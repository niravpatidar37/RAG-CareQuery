from vectors import VectorStore
from embedding import EmbeddingManager
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
import concurrent.futures
import pickle
import os

class RAGRetriever:
    """Handles query-based retrieval combining Vector Search + BM25 (Hybrid Search) with Latency Optimizations"""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, cache_dir: str = "cache/"):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.cache_dir = cache_dir
        self.bm25_cache_path = os.path.join(cache_dir, "bm25_index.pkl")
        
        # Initialize BM25 Index
        self.bm25 = None
        self.doc_registry = {}
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Loads BM25 index from cache if available, otherwise builds it and saves to cache."""
        print("Initializing BM25 Hybrid Search Index...")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check if cache exists
        if os.path.exists(self.bm25_cache_path):
            try:
                print("Loading BM25 Index from cache...")
                with open(self.bm25_cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.doc_registry = data['doc_registry']
                print(f"BM25 Index loaded from cache with {len(self.doc_registry)} documents.")
                return
            except Exception as e:
                print(f"Failed to load BM25 cache: {e}. Rebuilding...")

        # Rebuild if cache missing or failed
        try:
            result = self.vector_store.collection.get()
            documents = result['documents']
            metadatas = result['metadatas']
            ids = result['ids']
            
            if not documents:
                print("Warning: No documents found in VectorStore. BM25 not initialized.")
                return

            tokenized_corpus = [doc.split(" ") for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            for idx, (doc_id, doc_content, meta) in enumerate(zip(ids, documents, metadatas)):
                self.doc_registry[idx] = {
                    'id': doc_id,
                    'content': doc_content,
                    'metadata': meta
                }
            
            # Save to cache
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump({'bm25': self.bm25, 'doc_registry': self.doc_registry}, f)
            print(f"BM25 Index built and cached with {len(documents)} documents.")
            
        except Exception as e:
            print(f"Error initializing BM25: {e}")

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Internal method for vector search to be run in parallel"""
        vector_results = []
        try:
            query_embedding = self.embedding_manager.generate_embedding([query])[0]
            if hasattr(query_embedding, 'tolist'):
                q_emb = query_embedding.tolist()
            else:
                q_emb = query_embedding

            v_res = self.vector_store.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k
            )
            
            if v_res['ids'] and v_res['ids'][0]:
                for i, doc_id in enumerate(v_res['ids'][0]):
                    vector_results.append({
                        'id': doc_id,
                        'content': v_res['documents'][0][i],
                        'metadata': v_res['metadatas'][0][i],
                        'score': v_res['distances'][0][i]
                    })
        except Exception as e:
            print(f"Vector search failed: {e}")
        return vector_results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Internal method for BM25 search to be run in parallel"""
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n = np.argsort(doc_scores)[::-1][:top_k]
            
            for idx in top_n:
                if doc_scores[idx] > 0:
                    doc_data = self.doc_registry.get(idx)
                    if doc_data:
                        bm25_results.append({
                            'id': doc_data['id'],
                            'content': doc_data['content'],
                            'metadata': doc_data['metadata'],
                            'score': doc_scores[idx]
                        })
        return bm25_results

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using Parallel Hybrid Search (Vector + BM25) with RRF fusion.
        """
        print(f"Retrieving documents for query: '{query}' (Parallel Mode)")
        
        # Execute searches in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_vector = executor.submit(self._vector_search, query, top_k)
            future_bm25 = executor.submit(self._bm25_search, query, top_k)
            
            vector_results = future_vector.result()
            bm25_results = future_bm25.result()

        # Reciprocal Rank Fusion (RRF)
        k = 60
        fused_scores = {}
        
        def process_results(results):
            for rank, item in enumerate(results):
                doc_id = item['id']
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        'score': 0,
                        'content': item['content'],
                        'metadata': item['metadata']
                    }
                fused_scores[doc_id]['score'] += 1 / (k + rank + 1)

        process_results(vector_results)
        process_results(bm25_results)

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x]['score'], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids[:top_k]:
            item = fused_scores[doc_id]
            final_results.append({
                'id': doc_id,
                'content': item['content'],
                'metadata': item['metadata'],
                'similarity_score': item['score']
            })

        print(f"Retrieved {len(final_results)} documents after Optimized Hybrid Fusion")
        return final_results
