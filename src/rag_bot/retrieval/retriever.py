import time
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from langchain_ollama import ChatOllama

# Setup: paths and models
CHROMA_DB_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL = "phi3"  # or "deepseek-coder", "mistral", etc.

# Load embedding model and ChromaDB, all at module level for speed
embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection("infra_docs")

# Instantiate ChatOllama globally; no URL needed if Ollama is default localhost:11434
llm = ChatOllama(model=OLLAMA_MODEL)

def retrieve_context(query, top_k=5):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0], results['metadatas'][0]

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an expert IT infrastructure assistant. "
        "Given the following documentation snippets, answer the user's question as clearly and concisely as possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def query_ollama(prompt):
    response = llm.invoke(prompt)
    return response.content

def answer_question(question, top_k=2):
    t0 = time.time()
    context_chunks, metadatas = retrieve_context(question, top_k=top_k)
    print("[Timing] Retrieval:", time.time() - t0, "sec")
    t1 = time.time()
    prompt = build_prompt(context_chunks, question)
    print("[Timing] Build prompt:", time.time() - t1, "sec")
    t2 = time.time()
    answer = query_ollama(prompt)
    print("[Timing] LLM generation:", time.time() - t2, "sec")
    sources = [
        f"{m['vendor']} - {m['document']} (page: {m.get('page', 'n/a')}, chunk: {m['chunk']})"
        for m in metadatas
    ]
    print("[Timing] TOTAL:", time.time() - t0, "sec")
    return answer, sources


if __name__ == "__main__":
    import sys
    question = input("Enter your IT infra question: ") if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    answer, sources = answer_question(question)
    print("\n[ANSWER]\n", answer)
    print("\n[SOURCES]")
    for s in sources:
        print(s)
