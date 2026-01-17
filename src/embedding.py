import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    """Handles document embedding generation using Google Gemini Embeddings"""
    def __init__(self, model_name: str = "models/text-embedding-004"):
        """
        Initialize the embedding manager
        
        Args: model_name: Gemini model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Google Gemini Embedding model"""
        try:
            print(f"Loading Embedding Model: {self.model_name}")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
                
            self.model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key
            )
            print("Model loaded Successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embedding(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embeddings (list of floats)
        """

        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embedding for {len(texts)} texts...")
        # invoke handles batching internally usually, or we can use embed_documents
        embeddings = self.model.embed_documents(texts)
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
