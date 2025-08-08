# Handles LLM prompt → response

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"  # or "mistral", etc.

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

def query_llm(prompt, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]
