import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

with open("data_splan_com_visitor-management-piam-blogs_part_1.json", "r") as f:
    data = json.load(f)

docs = []
for entry in data:
    content = entry.get("text", "")
    if content:
        docs.append(Document(page_content=content))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print("✅ Vectorstore created")
