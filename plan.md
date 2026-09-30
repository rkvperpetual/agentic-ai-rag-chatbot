# Plan: RAG Chatbot on the Agentic AI eBook (Fresher Edition)

**Goal:** Build a small, clean, working RAG chatbot. Not fancy. Correct, readable, and something I can explain line by line in an interview.

**Deadline:** 48 hours from receiving the mail.

---

## 0. Do These First (15 minutes)

- [ ] Reply to the mail confirming I accept the assignment, and attach my CV (template at the bottom).
- [ ] Note the time I received the mail, so I know my real deadline.
- [ ] Create a public GitHub repo: `agentic-ai-rag-chatbot`.

---

## 1. Tech Stack (kept simple and free)

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Required |
| PDF loading | `pypdf` via LangChain `PyPDFLoader` | Easiest option |
| Chunking | `RecursiveCharacterTextSplitter` | Standard, easy to explain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dims) | Free, no API key, fast on CPU |
| Vector DB | Pinecone (free serverless tier) | Explicitly named in the task |
| Orchestration | LangGraph | Required |
| LLM | Groq (`llama-3.1-8b-instant`), free tier. OpenAI or Gemini also work | Only one API key needed for the LLM |
| API | FastAPI + Swagger UI (`/docs`) | Swagger gives a free test UI, so **no frontend needed** |

**Decision:** FastAPI only. Skip Streamlit unless I finish early. The task says API *or* UI.

---

## 2. Project Structure

```
agentic-ai-rag-chatbot/
├── app/
│   ├── config.py        # loads .env, constants
│   ├── ingest.py        # PDF -> chunks -> embeddings -> Pinecone (run once)
│   ├── graph.py         # LangGraph RAG pipeline
│   └── main.py          # FastAPI app
├── sample_queries.md    # 5-6 questions + real outputs
├── requirements.txt
├── .env.example         # keys with empty values (NEVER commit .env)
├── .gitignore           # .env, venv/, __pycache__/, *.pdf
└── README.md
```

---

## 3. Step-by-Step Build

### Step 1: Setup (30 min)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install fastapi uvicorn python-dotenv pypdf \
    langchain-community langchain-text-splitters langchain-huggingface \
    sentence-transformers pinecone langgraph langchain-groq
pip freeze > requirements.txt
```

Accounts and keys:
- Pinecone: https://www.pinecone.io (free) → API key
- Groq: https://console.groq.com (free) → API key

`.env.example`:
```
PINECONE_API_KEY=
PINECONE_INDEX=agentic-ai-ebook
GROQ_API_KEY=
```

### Step 2: Ingestion, `ingest.py` (2-3 hours)

Flow: **download PDF → extract text → chunk → embed → upsert to Pinecone**

Things to do:
1. Download the PDF with `requests`, or place it locally.
2. `PyPDFLoader(path).load()` gives one document per page (page number stays in metadata).
3. Split with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)`.
4. Embed with `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`.
5. Create the Pinecone index if it doesn't exist: `dimension=384`, `metric="cosine"`, serverless.
6. Upsert in batches (e.g. 50) with metadata: `text`, `page`, `chunk_id`.

Skeleton:
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX, dimension=384, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX)

vectors = []
for i, chunk in enumerate(chunks):
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedder.embed_query(chunk.page_content),
        "metadata": {"text": chunk.page_content, "page": chunk.metadata.get("page", -1)},
    })
index.upsert(vectors=vectors)
```

**Check:** Print the number of chunks and look at the Pinecone dashboard. Vector count should match.

### Step 3: LangGraph pipeline, `graph.py` (3-4 hours)

Keep the graph small: **retrieve → (check score) → generate**, with a fallback for out-of-scope questions.

```
        ┌──────────┐
START → │ retrieve │ ──(top score too low)──► fallback ──► END
        └────┬─────┘
             │ (good score)
             ▼
        ┌──────────┐
        │ generate │ ──► END
        └──────────┘
```

State:
```python
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    chunks: List[dict]     # [{"text":..., "page":..., "score":...}]
    answer: str
    confidence: float
```

Nodes:
- **retrieve**: embed the question, `index.query(vector=..., top_k=4, include_metadata=True)`, store text, page and score in state. Set `confidence` = top match's cosine score (or the average of the top 3, and say which in the README).
- **route** (conditional edge): if `confidence < 0.30` (tune this after testing), go to `fallback`, else `generate`.
- **generate**: send the chunks plus the question to the LLM with a strict prompt.
- **fallback**: answer `"I couldn't find this in the Agentic AI eBook."` (No LLM call.)

Strict grounding prompt (the most important part of the task):
```
You are a assistant that answers ONLY using the context below,
which is taken from the "Agentic AI" eBook.
- If the answer is not in the context, reply exactly:
  "I couldn't find this in the Agentic AI eBook."
- Do not use outside knowledge.
- Keep the answer short and clear.

Context:
{context}

Question: {question}
Answer:
```
Set `temperature=0` on the LLM so it doesn't get creative.

Wire it up:
```python
from langgraph.graph import StateGraph, END

g = StateGraph(RAGState)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.add_node("fallback", fallback)
g.set_entry_point("retrieve")
g.add_conditional_edges("retrieve", route, {"generate": "generate", "fallback": "fallback"})
g.add_edge("generate", END)
g.add_edge("fallback", END)
rag_app = g.compile()
```

**Check:** Test from a plain Python script before touching FastAPI:
`rag_app.invoke({"question": "What is agentic AI?"})`

### Step 4: FastAPI, `main.py` (1 hour)

One endpoint is enough:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Agentic AI eBook RAG Chatbot")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    result = rag_app.invoke({"question": req.question})
    return {
        "answer": result["answer"],
        "retrieved_chunks": result["chunks"],   # text, page, score
        "confidence": result["confidence"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run: `uvicorn app.main:app --reload`, then open `http://127.0.0.1:8000/docs` and test with the built-in Swagger UI.

Response shape:
```json
{
  "answer": "...",
  "retrieved_chunks": [{"text": "...", "page": 3, "score": 0.71}],
  "confidence": 0.71
}
```

### Step 5: Test with 5-6 Sample Queries (1 hour)

Run these (adjust after skimming the PDF), and paste real request and response into `sample_queries.md`:

1. What is Agentic AI?
2. How is Agentic AI different from traditional AI or generative AI?
3. What are the key components of an AI agent?
4. What are some use cases of Agentic AI in enterprises?
5. What challenges or risks does the eBook mention?
6. **Out-of-scope test:** "Who won the FIFA World Cup in 2018?" (should trigger the fallback)

Bonus: the out-of-scope test proves the "strictly grounded" requirement. Highlight it in the README.

### Step 6: README, Architecture, Cleanup (2 hours)

README sections:
1. **Project overview** (2-3 lines)
2. **Architecture** (short paragraph plus the ASCII diagram)
3. **Tech stack**
4. **Setup instructions** (clone → venv → install → `.env` → `python -m app.ingest` → `uvicorn ...`)
5. **API usage** (curl example and sample JSON response)
6. **Sample queries** (link to `sample_queries.md`)
7. **How grounding and confidence work**
8. **Limitations / future improvements**

Architecture explanation (adapt this):
> The PDF is loaded page by page, split into ~800-character overlapping chunks, and converted to 384-dimensional vectors using a local MiniLM embedding model. Vectors and their text and page metadata are stored in a Pinecone index. At query time, a LangGraph workflow embeds the user's question, retrieves the top 4 chunks by cosine similarity, and routes on the top score. If it is too low, the bot politely declines. Otherwise an LLM (Llama 3.1 via Groq, temperature 0) generates an answer using only the retrieved context. FastAPI returns the answer, the retrieved chunks, and the confidence score.

Limitations to be honest about:
- Confidence is a retrieval similarity score, not a guarantee of factual correctness.
- No chat memory (each question is independent).
- Tables and images in the PDF are not parsed.

Future improvements: reranking, conversation history, Streamlit UI, evaluation set.

---

## 4. 48-Hour Schedule

| When | What |
|---|---|
| Hour 0 | Reply to the mail with the CV, create the repo |
| Day 1, morning | Setup, accounts, API keys, Step 2 (ingestion) |
| Day 1, afternoon | Step 3 (LangGraph), test in a script |
| Day 1, evening | Step 4 (FastAPI) |
| Day 2, morning | Step 5 (test queries, tune chunk size and threshold) |
| Day 2, afternoon | Step 6 (README, cleanup) |
| Day 2, evening | Fresh-clone test, submit the form, **before** the deadline |

Aim to submit with a few hours of buffer.

---

## 5. Rules That Keep Me Safe

- I write the code myself. I can read docs and Stack Overflow, and I should understand every line. No AI coding platforms or auto-generated repos, since they can ask me to explain it.
- Never commit `.env` or API keys. If a key leaks, rotate it immediately.
- Use meaningful commits (`add ingestion script`, `add langgraph pipeline`), not one giant commit.
- Keep code short and commented. No over-engineering: no Docker, no agents-with-tools, no custom UI.
- Do a **fresh-clone test**: clone the repo into a new folder and follow my own README. If it fails, fix it.

---

## 6. Final Submission Checklist

- [ ] Repo is **public**
- [ ] README has setup steps that work from a fresh clone
- [ ] `.env.example` present, `.env` NOT committed
- [ ] `requirements.txt` present
- [ ] `/chat` returns answer, retrieved chunks, and confidence
- [ ] `sample_queries.md` has 5-6 real queries, including one out-of-scope
- [ ] Architecture explanation in the README
- [ ] Submitted the link through the Google Form (https://forms.gle/X8cnSCVSRToz1Svg6), **not by email**
- [ ] CV sent on the mail thread

---

## 7. Interview Prep (they will likely ask)

1. What is RAG and why use it instead of just an LLM?
2. Why chunk the text? Why overlap? Why 800 / 150?
3. What are embeddings? Why cosine similarity?
4. What does LangGraph give you over a simple function chain? (State, nodes, edges, conditional routing.)
5. How did you stop the LLM from hallucinating? (Prompt, temperature 0, score threshold, fallback.)
6. What does your confidence score really mean, and what are its limits?
7. What would you improve next? (Reranker, memory, evaluation, better PDF parsing.)

---

## 8. Reply Mail Template (acceptance)

**Subject:** Re: AI Engineer – Interview Task – Acceptance

Hi [Recruiter Name],

Thank you for the opportunity. I'm confirming that I have accepted the AI Engineer interview task and will submit my GitHub repository through the Google Form within the 48-hour deadline. My CV is attached for your reference.

Best regards,
[Your Name]
[Phone] | [Email] | [GitHub / LinkedIn]
