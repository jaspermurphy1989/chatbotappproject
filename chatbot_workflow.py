from langgraph.graph import StateGraph
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # New import path
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda  # Moved to langchain_core

# Rest of your code remains the same...

vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def call_rag_chain(state):
    query = state["question"]
    result = rag_chain.run(query)
    return {"answer": result}

graph = StateGraph()
graph.add_node("rag", RunnableLambda(call_rag_chain))
graph.set_entry_point("rag")
graph.set_finish_point("rag")

app = graph.compile()
