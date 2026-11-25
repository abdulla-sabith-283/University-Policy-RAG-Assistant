from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import RAGPipeline


app = FastAPI(title="Northbridge Policy RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    relevant: bool


_rag_pipeline = None


def get_rag():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    rag = get_rag()
    if request.use_hybrid:
        result = rag.answer_hybrid(request.question)
    else:
        result = rag.answer(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        relevant=bool(result["sources"]),
    )


@app.get("/")
def root():
    return {"service": "Northbridge Policy RAG", "docs": "/docs"}