import pickle
import numpy as np
from typing import Optional, Dict, Any, Tuple
from enum import Enum, auto

# Stable, non-conflicting imports
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreType(Enum):
    FAISS = auto()
    CHROMA = auto()

class VectorStoreConfig:
    def __init__(self):
        # Common config
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # FAISS-specific
        self.FAISS_INDEX_NAME = "my_faiss_index"
        self.FAISS_PKL_NAME = "my_faiss_index.pkl"
        
        # Chroma-specific
        self.CHROMA_PERSIST_DIR = "chroma_db"
        self.CHROMA_COLLECTION_NAME = "my_chroma_collection"

class VectorStoreManager:
    def __init__(self, config: VectorStoreConfig, store_type: VectorStoreType = VectorStoreType.FAISS):
        self.config = config
        self.store_type = store_type
    
    def process_documents(self, data_path: str):
        """Load and split documents"""
        try:
            loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Document processing failed: {str(e)}")
    
    def create_vector_store(self, documents, embeddings: Embeddings) -> Tuple[Any, Dict]:
        """Create vector store based on selected type"""
        if self.store_type == VectorStoreType.FAISS:
            return self._create_faiss_store(documents, embeddings)
        return self._create_chroma_store(documents, embeddings)
    
    def _create_faiss_store(self, documents, embeddings: Embeddings) -> Tuple[Any, Dict]:
        """Create FAISS index"""
        try:
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(self.config.FAISS_INDEX_NAME)
            
            metadata = {
                "store_type": "FAISS",
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "document_count": len(documents),
                "embedding_model": str(embeddings.model)
            }
            
            with open(self.config.FAISS_PKL_NAME, "wb") as f:
                pickle.dump(metadata, f)
            
            return db, metadata
        except Exception as e:
            raise RuntimeError(f"FAISS store creation failed: {str(e)}")

    def _create_chroma_store(self, documents, embeddings: Embeddings) -> Tuple[Any, Dict]:
        """Create ChromaDB index"""
        try:
            db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.config.CHROMA_PERSIST_DIR,
                collection_name=self.config.CHROMA_COLLECTION_NAME
            )
            
            metadata = {
                "store_type": "Chroma",
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "document_count": len(documents),
                "embedding_model": str(embeddings.model),
                "collection_name": self.config.CHROMA_COLLECTION_NAME
            }
            
            return db, metadata
        except Exception as e:
            raise RuntimeError(f"Chroma store creation failed: {str(e)}")
    
    def load_vector_store(self, embeddings: Embeddings) -> Tuple[Optional[Any], Optional[Dict]]:
        """Load existing vector store"""
        if self.store_type == VectorStoreType.FAISS:
            return self._load_faiss_store(embeddings)
        return self._load_chroma_store(embeddings)
    
    def _load_faiss_store(self, embeddings: Embeddings) -> Tuple[Optional[Any], Optional[Dict]]:
        """Load FAISS index"""
        try:
            db = FAISS.load_local(
                self.config.FAISS_INDEX_NAME,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            try:
                with open(self.config.FAISS_PKL_NAME, "rb") as f:
                    metadata = pickle.load(f)
                return db, metadata
            except FileNotFoundError:
                print("Warning: Metadata file not found")
                return db, None
        except Exception as e:
            print(f"FAISS load error: {str(e)}")
            return None, None
    
    def _load_chroma_store(self, embeddings: Embeddings) -> Tuple[Optional[Any], Optional[Dict]]:
        """Load ChromaDB index"""
        try:
            db = Chroma(
                persist_directory=self.config.CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=self.config.CHROMA_COLLECTION_NAME
            )
            
            metadata = {
                "store_type": "Chroma",
                "collection_name": self.config.CHROMA_COLLECTION_NAME,
                "embedding_model": str(embeddings.model)
            }
            
            return db, metadata
        except Exception as e:
            print(f"Chroma load error: {str(e)}")
            return None, None

if __name__ == "__main__":
    # Example usage
    config = VectorStoreConfig()
    manager = VectorStoreManager(config, VectorStoreType.CHROMA)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    DATA_PATH = "./data"
    
    db, metadata = manager.load_vector_store(embeddings)
    
    if db is None:
        print(f"Creating new {manager.store_type.name} vector store...")
        try:
            documents = manager.process_documents(DATA_PATH)
            db, metadata = manager.create_vector_store(documents, embeddings)
            print(f"Created index with {len(documents)} documents")
        except Exception as e:
            print(f"Creation failed: {str(e)}")
            exit(1)
    else:
        print(f"Loaded existing index")
        if metadata:
            print(f"Metadata: {metadata}")
    
    # Example search
    query = "What are Splan products?"
    results = db.similarity_search(query, k=3)
    print(f"\nResults for '{query}':")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...")
