from langgraph.graph import StateGraph
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# Initialize components
def initialize_components():
    embeddings = OpenAIEmbeddings()
    
    # Load vectorstore with error handling
    try:
        vectorstore = FAISS.load_local("vectorstore", embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorstore: {str(e)}")
    
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Configure RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Optional: include source documents
    )
    return rag_chain

# Initialize components once
rag_chain = initialize_components()

# RAG processing function
def call_rag_chain(state):
    try:
        result = rag_chain.invoke({"query": state["question"]})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])  # Include sources if available
        }
    except Exception as e:
        return {"error": str(e)}

# Build the graph
graph = StateGraph()
graph.add_node("rag", RunnableLambda(call_rag_chain))
graph.set_entry_point("rag")
graph.set_finish_point("rag")

# Compile the app
app = graph.compile()
