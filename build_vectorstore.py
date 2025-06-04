import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
from typing import List

def load_json_data(file_path: str) -> List[dict]:
    """Load JSON data from file"""
    with open(file_path, "r") as f:
        return json.load(f)

def create_documents_from_json(data: List[dict]) -> List[Document]:
    """Create LangChain Documents from JSON data"""
    docs = []
    for entry in data:
        content = entry.get("text", "")
        if content:
            metadata = {k: v for k, v in entry.items() if k != "text"}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def create_faiss_vectorstore(documents: List[Document], embeddings: OpenAIEmbeddings, save_path: str = "faiss_vectorstore"):
    """Create and save FAISS vector store"""
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(save_path)
    print(f"✅ FAISS vectorstore created and saved to {save_path}")
    return db

def create_chroma_vectorstore(documents: List[Document], embeddings: OpenAIEmbeddings, persist_directory: str = "chroma_vectorstore"):
    """Create and save Chroma vector store"""
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"✅ Chroma vectorstore created and saved to {persist_directory}")
    return db

def main():
    # Load and process data
    data = load_json_data("data_splan_com_visitor-management-piam-blogs_part_1.json")
    documents = create_documents_from_json(data)
    
    # Configure text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create both vector stores
    create_faiss_vectorstore(chunks, embeddings)
    create_chroma_vectorstore(chunks, embeddings)

if __name__ == "__main__":
    main()
