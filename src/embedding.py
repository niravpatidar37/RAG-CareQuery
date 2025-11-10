import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os


class EmbeddingManager:
    """Handles document embedding generation using sentenceTransformer"""
    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args: model_name: Huggingface model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentenceTransformer model"""
        try:
            print(f"loading Embedding Model:{self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded Successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"error loading model{self.model_name}: {e}")

    def generate_embedding(self,texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        Args:
            texts: List of text strings to embed
        
        returns:
            Numpy array of embeddings with shape(len(texts),embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embedding for {len(texts)} texts...")
        embedding = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings with shape:{embedding.shape}")
        return embedding

# Initialize the embedding Manager
# embedding_manager = EmbeddingManager()
# embedding_manager
