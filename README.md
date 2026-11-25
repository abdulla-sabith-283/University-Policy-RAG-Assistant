# University Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) system for answering questions about university policies. Built with Streamlit, FastAPI, ChromaDB, sentence-transformers, and OpenRouter.

## Features

- **📚 Streamlit Chat UI** — Conversational interface for asking policy questions
- **🔍 Hybrid Retrieval** — Vector search (ChromaDB + all-MiniLM-L6-v2) + BM25 keyword search with Reciprocal Rank Fusion
- **🔄 Cross-Encoder Reranking** — Re-ranks top candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` for better relevance
- **🚀 FastAPI REST API** — Decoupled backend with `/query` and `/health` endpoints
- **🛡️ Security Guardrails** — Prompt-injection detection and relevance gating to prevent hallucinations
- **📊 Evaluation Suite** — 22 test cases (12 answerable, 10 unanswerable) with source coverage metrics

## How It Works

```
User Question
    │
    ▼
Security Check (prompt injection detection)
    │
    ▼
Hybrid Retrieval (ChromaDB vector + BM25 keyword, RRF fusion)
    │
    ▼
Cross-Encoder Reranking (top-10 → top-5)
    │
    ▼
Relevance Gate (abstain if no relevant evidence)
    │
    ▼
LLM Generation (Llama-3.3-70b via OpenRouter, grounded in retrieved context)
    │
    ▼
Answer + Citations ([EVIDENCE 1], [EVIDENCE 2], etc.)
```

## Project Structure

```
rag-ieee/
├── app.py                 # Streamlit chat UI
├── api.py                 # FastAPI REST API
├── evaluate.py            # Evaluation suite (22 test cases)
├── main.py                # Entry point stub
├── pyproject.toml         # Dependencies
├── README.md
├── .env                   # API keys (NOT committed)
├── .gitignore
├── data/
│   └── documents/         # 8 university policy PDFs
├── chroma_db/             # Persistent vector store (NOT committed)
├── .venv/                 # Virtual environment (NOT committed)
└── src/
    ├── __init__.py
    ├── rag.py             # RAGPipeline (orchestration)
    ├── retriever.py       # Chroma + BM25 hybrid retrieval
    ├── reranker.py        # Cross-encoder reranking
    ├── generator.py       # OpenRouter LLM generation
    ├── embeddings.py      # sentence-transformers embeddings
    ├── ingestion.py       # PDF text extraction
    ├── chunking.py        # Section-aware chunking
    ├── vectorstore.py     # ChromaDB index creation
    ├── security.py        # (placeholder)
    └── pipeline.py        # (placeholder)
```

## Installation

```powershell
# Clone the repo
git clone https://github.com/abdulla-sabith-283/University-Policy-RAG-Assistant.git
cd University-Policy-RAG-Assistant

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r pyproject.toml

# Set up environment variables
Copy .env.example .env   # Add your OPENROUTER_API_KEY

# Build the vector database (first time only)
python src/vectorstore.py
```

## Usage

### Streamlit UI
```powershell
streamlit run app.py
```
Open `http://localhost:8501`

### FastAPI API
```powershell
uvicorn api:app --reload
```
- API docs: `http://localhost:8000/docs`
- Health: `GET http://localhost:8000/health`
- Query: `POST http://localhost:8000/query` with `{"question": "...", "use_hybrid": true}`

### Headless
```powershell
python -c "from src.rag import RAGPipeline; r=RAGPipeline().answer_hybrid('What is the attendance requirement?'); print(r['answer'])"
```

## Evaluation

Run the test suite:
```powershell
python evaluate.py
```

Results on 22 test cases:
- Answerable accuracy: 92% (11/12)
- Abstention accuracy: 100% (5/5 unanswerable correctly rejected)
- Overall gate accuracy: 94%

## Tech Stack

- **Framework**: Python 3.11+
- **UI**: Streamlit 1.62
- **API**: FastAPI + Uvicorn
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Reranking**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **Keyword Search**: rank-bm25
- **LLM**: Llama-3.3-70b-instruct via OpenRouter
- **PDF Processing**: PyMuPDF

## Resume Highlights

- Implemented hybrid retrieval (vector + BM25 with Reciprocal Rank Fusion)
- Built cross-encoder reranking pipeline for improved relevance
- Exposed RAG logic as a decoupled FastAPI REST API
- Designed evaluation suite with relevance gating and source coverage metrics
- Implemented prompt-injection security guardrails

## License

MIT
