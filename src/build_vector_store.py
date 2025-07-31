import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from parse_documents import parse_all_documents

DB_DIR = "vectorstore/chroma_db"

def build_vector_store():
    print("📚 Parsing documents...")
    chunks = parse_all_documents()
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "section": chunk["section"], "page": chunk["page"]} for chunk in chunks]

    print("🔍 Loading HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Building Chroma vector store...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_DIR
    )

    vectorstore.persist()
    print(f"✅ Chroma vector DB saved to {DB_DIR}")


if __name__ == "__main__":
    build_vector_store()
