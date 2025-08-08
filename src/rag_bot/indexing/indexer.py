# Manages embedding model and ChromaDB setup

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path

CHROMA_DB_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chroma_db"

def get_embedding_model(model_name="phi3"):
    return SentenceTransformer(model_name)

def get_chromadb_collection(collection_name="vendor"):
    chroma_client = chromadb.Client(Settings(
        persist_directory=str(CHROMA_DB_DIR)
    ))
    return chroma_client.get_or_create_collection(collection_name)
