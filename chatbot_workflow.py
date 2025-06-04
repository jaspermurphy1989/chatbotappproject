import pickle
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum, auto
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os

class VectorStoreType(Enum):
    FAISS = auto()
    CHROMA = auto()

class VectorStoreConfig:
    def __init__(self):
        # Common config
        self.CHUNK_SIZE = 500
        self.CHUNK_OVERLAP = 100
        
        # FAISS-specific
        self.FAISS_INDEX_NAME = "faiss_vectorstore"
        self.FAISS_PKL_NAME = "faiss_metadata.pkl"
        
        # Chroma-specific
        self.CHROMA_PERSIST_DIR = "chroma_vectorstore"
        self.CHROMA_COLLECTION_NAME = "splan_docs"

class VectorStoreManager:
    def __init__(self, config: VectorStoreConfig, store_type: VectorStoreType = VectorStoreType.FAISS):
        self.config = config
        self.store_type = store_type
    
    def process_json_documents(self, json_path: str) -> List[Document]:
        """Load and process JSON documents"""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            documents = []
            for entry in data:
                content = entry.get("text", "")
                if content:
                    metadata = {k: v for k, v in entry.items() if k != "text"}
                    documents.append(Document(page_content=content, metadata=metadata))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"Document processing failed: {str(e)}")
    
    def create_vector_store(self, documents: List[Document], embeddings: Embeddings) -> Tuple[Any, Dict]:
        """Create vector store based on selected type"""
        if self.store_type == VectorStoreType.FAISS:
            return self._create_faiss_store(documents, embeddings)
        return self._create_chroma_store(documents, embeddings)
    
    def _create_faiss_store(self, documents: List[Document], embeddings: Embeddings) -> Tuple[Any, Dict]:
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

    def _create_chroma_store(self, documents: List[Document], embeddings: Embeddings) -> Tuple[Any, Dict]:
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
            if not os.path.exists(f"{self.config.FAISS_INDEX_NAME}/index.faiss"):
                return None, None
                
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
            if not os.path.exists(self.config.CHROMA_PERSIST_DIR):
                return None, None
                
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

def test_vector_stores():
    """Test function to demonstrate usage"""
    config = VectorStoreConfig()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    json_path = "data_splan_com_visitor-management-piam-blogs_part_1.json"
    
    # Test FAISS
    print("\nTesting FAISS vector store...")
    faiss_manager = VectorStoreManager(config, VectorStoreType.FAISS)
    faiss_db, faiss_metadata = faiss_manager.load_vector_store(embeddings)
    
    if faiss_db is None:
        print("Creating new FAISS store...")
        documents = faiss_manager.process_json_documents(json_path)
        faiss_db, faiss_metadata = faiss_manager.create_vector_store(documents, embeddings)
    else:
        print("Loaded existing FAISS store")
    
    # Test Chroma
    print("\nTesting Chroma vector store...")
    chroma_manager = VectorStoreManager(config, VectorStoreType.CHROMA)
    chroma_db, chroma_metadata = chroma_manager.load_vector_store(embeddings)
    
    if chroma_db is None:
        print("Creating new Chroma store...")
        documents = chroma_manager.process_json_documents(json_path)
        chroma_db, chroma_metadata = chroma_manager.create_vector_store(documents, embeddings)
    else:
        print("Loaded existing Chroma store")
    
    # Test search
    query = "What is visitor management?"
    print(f"\nSearching for: '{query}'")
    
    if faiss_db:
        print("\nFAISS Results:")
        results = faiss_db.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:200] + "...")
    
    if chroma_db:
        print("\nChroma Results:")
        results = chroma_db.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:200] + "...")

if __name__ == "__main__":
    test_vector_stores()
