import chromadb
from chromadb.config import Settings
import uuid
import hashlib
from typing import List, Dict, Any
import os
import numpy as np

class VectorStore:
    """Manages document in a ChromaDB vector store"""
    def __init__(self, collection_name: str = 'Csv_data', persist_directory: str = "vector_store/"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB Client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create Collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Csv Data embedding for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.
        Uses deterministic IDs based on content hash to prevent duplicates.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        document_text = []
        
        # We need embeddings as list of lists, usually they come that way from new Manager
        # If they are numpy array, convert them.
        if hasattr(embeddings, 'tolist'):
           embedding_list = embeddings.tolist()
        else:
           embedding_list = embeddings

        for i, (doc, emb) in enumerate(zip(documents, embedding_list)):
            # Create deterministic ID based on content
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            ids.append(content_hash)

            # Prepare Metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            # Ensure metadata values are strings, ints, floats, or bools (Chroma requirement)
            for k, v in metadata.items():
                if v is None:
                    metadata[k] = ""
            metadatas.append(metadata)

            document_text.append(doc.page_content)

        try:
            # upsert handles updates or inserts
            self.collection.upsert(
                ids=ids,
                embeddings=embedding_list,
                documents=document_text,
                metadatas=metadatas
            )
            print(f"Successfully added/updated {len(documents)} documents in vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
