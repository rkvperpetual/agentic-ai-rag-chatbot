# Agentic AI eBook RAG Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-blueviolet.svg)](https://www.pinecone.io/)

A production-grade, strictly grounded Retrieval-Augmented Generation (RAG) chatbot in Python. It indexes and answers user questions strictly based on the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) using **LangGraph**, **Pinecone**, local sentence embeddings (**sentence-transformers/all-MiniLM-L6-v2**), and **Groq (Llama-3.1-8B-Instant)**.

---

## Architecture Overview

```
                      User Question
                            │
                            ▼
                   ┌──────────────────┐
                   │  FastAPI /chat   │
                   └────────┬─────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ LangGraph RAG Workflow    │
              │                           │
              │  ┌─────────────────────┐  │
              │  │      retrieve       │  │ ◄── MiniLM Embeddings & Pinecone
              │  └──────────┬──────────┘  │     (or local vector cache fallback)
              │             │             │
              │      [Check Confidence]   │
              │       score < 0.30 ?      │
              │      /              \     │
              │    YES              NO    │
              │    /                  \   │
              │   ▼                    ▼  │
              │ ┌──────────┐    ┌─────────┴┐
              │ │ fallback │    │ generate │ ◄── Groq Llama 3.1 (temp=0)
              │ └────┬─────┘    └────┬─────┘     with strict grounding prompt
              │      │               │    │
              │      ▼               ▼    │
              │     END             END   │
              └─────────────┬─────────────┘
                            │
                            ▼
               JSON Response with Answer,
             Source Chunks & Confidence
```

### Ingestion Flow
1. **Document Loading:** `Ebook-Agentic-AI.pdf` (60 pages) is loaded page-by-page using `pypdf`, preserving 1-based page metadata.
2. **Text Chunking:** Text is split using `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)` producing 137 context chunks.
3. **Embeddings:** Each chunk is converted to a 384-dimensional dense vector using the local `sentence-transformers/all-MiniLM-L6-v2` model (fast, free, running on CPU).
4. **Pinecone Indexing:** Vectors are uploaded in batches to a Pinecone serverless index with cosine metric. A local cache is also generated for seamless offline testing.

### Retrieval & Guardrailed Generation Flow
1. **Embedding Query:** The user's query is converted to a 384-dim vector using the same embedding model.
2. **Similarity Search:** Top 4 matching chunks are retrieved by cosine similarity.
3. **Confidence Scoring:** `confidence` is calculated as the maximum cosine similarity among retrieved chunks (`round(max_score, 4)`).
4. **Conditional Routing (LangGraph Edge):**
   - If `confidence < 0.30` or chunks are missing: routes to `fallback` node.
   - If `confidence >= 0.30`: routes to `generate` node.
5. **Grounded Synthesis:** The `generate` node sends only the retrieved chunks to the LLM (Groq `llama-3.1-8b-instant`, `temperature=0`) with strict instructions to answer only using context or return the fallback phrase.

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Required runtime |
| **Vector DB** | **Pinecone** (Serverless) | Explicitly specified; fast serverless vector search |
| **Embeddings** | `all-MiniLM-L6-v2` (384 dims) | Runs locally on CPU, high quality, zero cost, no rate limits |
| **Orchestrator** | **LangGraph** | Explicitly required state-graph with conditional routing |
| **LLM** | **Groq** (`llama-3.1-8b-instant`) | Ultra-fast inference, high accuracy, free tier |
| **API & UI** | **FastAPI** + Swagger Docs + Web UI | Instant `/docs` OpenAPI interface + interactive dark-mode web chat |

---

## Project Structure

```
agentic-ai-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration and environment management
│   ├── vector_store.py     # Pinecone client & local similarity engine
│   ├── ingest.py           # PDF ingestion, chunking, and embedding pipeline
│   ├── graph.py            # LangGraph workflow (retrieve -> route -> generate/fallback)
│   └── main.py             # FastAPI app with /chat, /health, /docs, and web UI
├── data/
│   └── local_vector_cache.json  # Pre-embedded offline cache for instant grading
├── sample_queries.md       # 6 evaluated test queries with real outputs & scores
├── Ebook-Agentic-AI.pdf    # Source knowledge base eBook (60 pages)
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Ignored files (.env, __pycache__, etc.)
├── plan.md                 # Project implementation plan
└── README.md               # Documentation and setup guide
```

---

## Setup & Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Pinecone API Key (from https://www.pinecone.io)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=agentic-ai-ebook
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Groq API Key (from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

*(Note: If you run without external API keys, the chatbot will still operate using the embedded vector cache and provide grounded page-referenced extractions).*

### 4. Run PDF Ingestion

Ingest the eBook into Pinecone and create the vector index:

```bash
python -m app.ingest
```

**Expected output:**
```
============================================================
 Ingestion Summary:
 - Non-empty Pages: 59
 - Total Chunks: 137
 - Embeddings Dimension: 384
 - Local Cache: Ready
 - Pinecone Upload: SUCCESS
============================================================
```

### 5. Launch the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The application is now running at:
- **Interactive Web Chat UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger OpenAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Usage

### `POST /chat`

Submit a question about the Agentic AI eBook.

#### Request (curl)
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Agentic AI?"}'
```

#### Response (JSON)
```json
{
  "answer": "Agentic AI refers to systems capable of autonomous decision-making, goal pursuit, and taking independent actions to achieve specified outcomes...",
  "retrieved_chunks": [
    {
      "text": "Agentic AI\nAn Executive's Guide to In-depth\nUnderstanding of Agentic AI",
      "page": 3,
      "score": 0.8168
    },
    {
      "text": "In this section, we will define what Agentic AI is and, more importantly, what it's not...",
      "page": 7,
      "score": 0.7812
    }
  ],
  "confidence": 0.8168
}
```

### `GET /health`

Inspect system readiness and configuration status:

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

---

## Sample Queries & Strict Grounding

Six comprehensive queries are detailed in [`sample_queries.md`](sample_queries.md):

1. **Core Concept:** *"What is Agentic AI?"* (Confidence: **0.8168**) -> Returns definition from Pages 3, 7, 18.
2. **Comparison:** *"How is Agentic AI different from traditional AI or generative AI?"* (Confidence: **0.6663**) -> Returns reactive vs. proactive comparison.
3. **Architecture:** *"What are the key components of an AI agent?"* (Confidence: **0.7473**) -> Returns Core Pillars from Page 19.
4. **Enterprise Applications:** *"What are some use cases of Agentic AI in enterprises?"* (Confidence: **0.7798**) -> Returns McKinsey data and manufacturer case study.
5. **Multi-Agent Systems:** *"How do Multi-Agent Systems work?"* (Confidence: **0.7832**) -> Returns MAS orchestration from Pages 30, 41.
6. **Out-of-Scope (Strict Grounding):** *"Who won the FIFA World Cup in 2018?"* (Confidence: **0.1096**) -> **Safely refused:** `"I couldn't find this in the Agentic AI eBook."`

### How Grounding & Hallucination Prevention Work
1. **Cosine Similarity Gating:** Out-of-scope questions naturally yield low similarity against eBook embeddings. If `confidence < 0.30`, LangGraph conditionally short-circuits directly to `fallback`.
2. **Zero-Shot Prompt Constraint:** The system prompt explicitly instructs the LLM:
   > "You are an assistant that answers ONLY using the context below... If the answer is not in the context, reply exactly: 'I couldn't find this in the Agentic AI eBook.' Do not use outside knowledge."
3. **Deterministic Output (`temperature=0`):** LLM sampling creativity is set to 0.

---

## Limitations & Future Work

- **Confidence Definition:** The confidence score reflects vector semantic similarity between the question and the top retrieved chunk. It measures context relevance rather than an absolute mathematical guarantee of factual completeness.
- **Conversational Memory:** The current implementation processes single-turn independent questions. Future improvements could integrate LangGraph checkpointers (`MemorySaver`) for multi-turn dialogue.
- **Complex Table / Figure Extraction:** Tabular layouts and diagrams in the PDF are converted via text extraction. Integrating layout-aware OCR (such as `pdfplumber` or `unstructured`) would enhance table parsing.
- **Reranking:** Adding a Cohere or BGE cross-encoder reranker between `retrieve` and `generate` would further optimize ranking precision.

---

## Submission Checklist

- [x] Repository structured and self-contained
- [x] Ingestion pipeline with chunking and embeddings (`app/ingest.py`)
- [x] LangGraph state graph with retrieval, conditional routing, and fallback (`app/graph.py`)
- [x] FastAPI `/chat` and `/health` endpoints returning answer, chunks, and confidence (`app/main.py`)
- [x] Embedded interactive dark-mode web chat UI served on `/`
- [x] Sample queries with real confidence scores and outputs (`sample_queries.md`)
- [x] Architecture explanation and ASCII diagram
- [x] `.env.example`, `.gitignore`, and `requirements.txt` included
