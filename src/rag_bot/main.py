from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from rag_bot.retrieval.retriever import answer_question

app = FastAPI(title="RAG IT Infrastructure Assistant")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        # Get answer and print sources in the terminal
        # context_chunks, metadatas = answer_question(
        #     request.question, top_k=request.top_k
        # )
        # answer = context_chunks  # The function returns answer as first value

        # # Build readable sources
        # sources = [
        #     f"{meta['vendor']} - {meta['document']} (page: {meta.get('page', 'n/a')}, chunk: {meta['chunk']})"
        #     for meta in metadatas
        # ]
        # return QueryResponse(answer=answer, sources=sources)
        answer, sources = answer_question(request.question, top_k=request.top_k)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_bot.main:app", host="0.0.0.0", port=8000, reload=True)
