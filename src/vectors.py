from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
class VectorStore:
    """Manages document in a ChromaDB vector store"""
    def __init__(self,collection_name: str = 'Csv_data',persist_directory: str = "vector_store/"):
        """
        Initialize the vector store

            Args: 
                collection_name: Name of the chromadb collection
                persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB Client and collection"""
        try:
            # Create ChromaDB client
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            # Get or create Collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description":"Csv Data embedding for RAG"}
            )
            print(f"vector store initiallized. Collection:{self.collection_name}")
            print(f"Existing documents in collection:{self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store:{e}")
            raise

    def add_documents(self,documents: List[Any], embeddings:np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of Langchain documents
            embeddings: corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match of embeddings")
        print(f"adding {len(documents)} documents to vector store...")

        ids =[]
        metadatas = []
        document_text = []
        embedding_list = []

        for i , (doc,embeddings) in enumerate(zip(documents,embeddings)):
            # Create unique ids
            doc_id = uuid.uuid4().hex[:8]
            ids.append(doc_id)

            # Prepare Metadata
            metadata = dict(doc.metadata)
            metadata['doc_index']= i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)

            # Embeddings
            embedding_list.append(embeddings.tolist())
        
        try:
            self.collection.add(
                ids = ids,
                embeddings = embedding_list,
                documents=document_text,
                metadatas=metadatas
            )
            print(f"succesfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection:{self.collection.count()}")
        except Exception as e:
            print(f"error adding documents to vector store {e}")
            raise

# vectorstore = VectorStore()
# vectorstore
