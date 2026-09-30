import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD,
)
from app.graph import rag_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="A strictly grounded RAG chatbot powered by LangGraph, Pinecone, HuggingFace embeddings, and Groq Llama 3.1.",
    version="1.0.0",
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The question about the Agentic AI eBook to ask the chatbot.",
        json_schema_extra={"example": "What is Agentic AI?"},
    )


class ChunkMetadata(BaseModel):
    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[ChunkMetadata]
    confidence: float


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify service and environment readiness."""
    return {
        "status": "healthy",
        "pinecone_configured": bool(PINECONE_API_KEY),
        "pinecone_index": PINECONE_INDEX,
        "groq_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }


@app.post("/chat", response_model=ChatResponse, tags=["RAG Chat"])
def chat(request: ChatRequest):
    """
    Main RAG Chatbot endpoint.
    Retrieves context from the eBook, routes through LangGraph, and returns the grounded answer,
    retrieved chunks with page numbers, and the retrieval confidence score.
    """
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_app.invoke({"question": query})
        return {
            "answer": result.get("answer", ""),
            "retrieved_chunks": result.get("chunks", []),
            "confidence": result.get("confidence", 0.0),
        }
    except Exception as e:
        logger.error(f"Error handling /chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def web_ui():
    """Clean, modern light-mode UI for chatting with the Agentic AI eBook."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic AI eBook - RAG Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --border-focus: #3b82f6;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-sub: #475569;
      --user-bubble: #2563eb;
      --badge-bg: #eff6ff;
      --badge-text: #1d4ed8;
      --badge-border: #bfdbfe;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--border);
      padding: 0.9rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .brand-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .badge {
      font-size: 0.72rem;
      background: var(--badge-bg);
      color: var(--badge-text);
      border: 1px solid var(--badge-border);
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-weight: 600;
    }
    .nav-links {
      display: flex;
      gap: 1rem;
    }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      transition: color 0.15s;
    }
    .nav-links a:hover {
      color: var(--primary);
    }
    main {
      flex: 1;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
      padding: 1.5rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .info-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .info-card h1 {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text-main);
    }
    .info-card p {
      font-size: 0.875rem;
      color: var(--text-sub);
      line-height: 1.5;
    }
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.4rem;
    }
    .suggestion-chip {
      background: #f1f5f9;
      border: 1px solid var(--border);
      padding: 0.35rem 0.7rem;
      border-radius: 9999px;
      font-size: 0.76rem;
      color: var(--text-sub);
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }
    .suggestion-chip:hover {
      background: #e2e8f0;
      border-color: #cbd5e1;
      color: var(--text-main);
    }
    .chat-container {
      flex: 1;
      min-height: 420px;
      max-height: 580px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: 0.5rem 0.25rem;
    }
    .message {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      max-width: 82%;
    }
    .message.user {
      align-self: flex-end;
    }
    .message.assistant {
      align-self: flex-start;
      max-width: 92%;
    }
    .bubble {
      padding: 0.85rem 1.15rem;
      border-radius: 0.85rem;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .user .bubble {
      background: var(--user-bubble);
      color: #ffffff;
      border-bottom-right-radius: 0.2rem;
    }
    .assistant .bubble {
      background: #ffffff;
      border: 1px solid var(--border);
      border-bottom-left-radius: 0.2rem;
      color: var(--text-main);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .meta-bar {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.45rem;
    }
    .confidence-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.72rem;
    }
    .confidence-high {
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }
    .confidence-mid {
      background: #fffbeb;
      color: #b45309;
      border: 1px solid #fde68a;
    }
    .confidence-low {
      background: #fef2f2;
      color: #b91c1c;
      border: 1px solid #fecaca;
    }
    .chunks-details {
      margin-top: 0.65rem;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      overflow: hidden;
      background: #f8fafc;
    }
    .chunks-toggle {
      padding: 0.45rem 0.75rem;
      font-size: 0.75rem;
      cursor: pointer;
      color: var(--text-sub);
      display: flex;
      justify-content: space-between;
      user-select: none;
      background: #f1f5f9;
      font-weight: 500;
    }
    .chunks-toggle:hover {
      background: #e2e8f0;
    }
    .chunks-content {
      display: none;
      padding: 0.65rem;
      border-top: 1px solid var(--border);
      font-size: 0.8rem;
      color: var(--text-sub);
      max-height: 220px;
      overflow-y: auto;
    }
    .chunk-item {
      padding: 0.6rem 0.75rem;
      margin-bottom: 0.5rem;
      background: #ffffff;
      border: 1px solid var(--border);
      border-left: 3px solid var(--primary);
      border-radius: 0.35rem;
    }
    .chunk-item:last-child {
      margin-bottom: 0;
    }
    .chunk-header {
      font-weight: 600;
      font-size: 0.72rem;
      color: var(--primary);
      margin-bottom: 0.25rem;
      display: flex;
      justify-content: space-between;
    }
    .input-box {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.5rem 0.65rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    }
    .input-box:focus-within {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
    }
    .input-box input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-main);
      font-size: 0.92rem;
      padding: 0.45rem 0.5rem;
    }
    .input-box input::placeholder {
      color: #94a3b8;
    }
    .send-btn {
      background: var(--primary);
      color: #ffffff;
      border: none;
      outline: none;
      padding: 0.55rem 1.15rem;
      border-radius: 0.5rem;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background 0.15s;
    }
    .send-btn:hover {
      background: var(--primary-hover);
    }
    .send-btn:disabled {
      background: #94a3b8;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"></path>
        <path d="M6 6h10"></path>
        <path d="M6 10h10"></path>
      </svg>
      <span class="brand-title">Agentic AI Assistant</span>
      <span class="badge">Strictly Grounded</span>
    </div>
    <div class="nav-links">
      <a href="/docs" target="_blank">API Docs (/docs)</a>
      <a href="/health" target="_blank">Health</a>
    </div>
  </header>

  <main>
    <div class="info-card">
      <h1>Knowledge Base: Agentic AI eBook</h1>
      <p>Ask any question about the eBook (definitions, multi-agent systems, pillars, or enterprise use cases). Responses are grounded with retrieval confidence scores and cited page numbers.</p>
      <div class="suggestions">
        <span class="suggestion-chip" onclick="ask(this.innerText)">What is Agentic AI?</span>
        <span class="suggestion-chip" onclick="ask(this.innerText)">How is Agentic AI different from traditional AI?</span>
        <span class="suggestion-chip" onclick="ask(this.innerText)">What are the key components of an AI agent?</span>
        <span class="suggestion-chip" onclick="ask(this.innerText)">How do Multi-Agent Systems work?</span>
        <span class="suggestion-chip" onclick="ask(this.innerText)">Who won the FIFA World Cup in 2018?</span>
      </div>
    </div>

    <div class="chat-container" id="chatArea">
      <div class="message assistant">
        <div class="bubble">
          Hello! I can answer questions strictly based on the <strong>Agentic AI eBook</strong>. Feel free to type a query or choose a suggestion above.
        </div>
      </div>
    </div>

    <form class="input-box" id="chatForm" onsubmit="handleSubmit(event)">
      <input type="text" id="userInput" placeholder="Ask a question about the eBook..." autocomplete="off" />
      <button type="submit" class="send-btn" id="sendBtn">Send</button>
    </form>
  </main>

  <script>
    const chatArea = document.getElementById('chatArea');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    function ask(text) {
      userInput.value = text;
      handleSubmit(new Event('submit'));
    }

    function toggleChunks(id) {
      const el = document.getElementById(id);
      el.style.display = el.style.display === 'block' ? 'none' : 'block';
    }

    async function handleSubmit(e) {
      e.preventDefault();
      const question = userInput.value.trim();
      if (!question) return;

      appendMessage('user', question);
      userInput.value = '';
      sendBtn.disabled = true;
      sendBtn.innerText = 'Searching...';

      const loadingId = 'loading-' + Date.now();
      appendLoading(loadingId);

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        removeLoading(loadingId);

        if (res.ok) {
          appendAssistantResponse(data);
        } else {
          appendMessage('assistant', 'Error: ' + (data.detail || 'Could not retrieve answer.'));
        }
      } catch (err) {
        removeLoading(loadingId);
        appendMessage('assistant', 'Network error connecting to API: ' + err.message);
      } finally {
        sendBtn.disabled = false;
        sendBtn.innerText = 'Send';
      }
    }

    function appendMessage(sender, text) {
      const msg = document.createElement('div');
      msg.className = `message ${sender}`;
      msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
      chatArea.appendChild(msg);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function appendLoading(id) {
      const msg = document.createElement('div');
      msg.className = 'message assistant';
      msg.id = id;
      msg.innerHTML = '<div class="bubble" style="color: var(--text-muted);"><em>Searching eBook knowledge base...</em></div>';
      chatArea.appendChild(msg);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function removeLoading(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
    }

    function appendAssistantResponse(data) {
      const msg = document.createElement('div');
      msg.className = 'message assistant';

      const conf = data.confidence || 0.0;
      let pillClass = 'confidence-low';
      if (conf >= 0.5) pillClass = 'confidence-high';
      else if (conf >= 0.3) pillClass = 'confidence-mid';

      const chunksId = 'chunks-' + Date.now();
      let chunksHtml = '';
      if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        const chunkItems = data.retrieved_chunks.map((c, i) => `
          <div class="chunk-item">
            <div class="chunk-header">
              <span>Source Chunk #${i+1} &bull; Page ${c.page}</span>
              <span>Similarity: ${(c.score * 100).toFixed(1)}%</span>
            </div>
            <div>${escapeHtml(c.text)}</div>
          </div>
        `).join('');

        chunksHtml = `
          <div class="chunks-details">
            <div class="chunks-toggle" onclick="toggleChunks('${chunksId}')">
              <span>&#128196; View ${data.retrieved_chunks.length} Retrieved Context Chunks</span>
              <span>&#9662;</span>
            </div>
            <div class="chunks-content" id="${chunksId}">
              ${chunkItems}
            </div>
          </div>
        `;
      }

      msg.innerHTML = `
        <div class="bubble">
          ${escapeHtml(data.answer)}
          <div class="meta-bar">
            <span>Confidence:</span>
            <span class="confidence-pill ${pillClass}">&#9679; ${(conf * 100).toFixed(1)}%</span>
            <span>&bull; Chunks: ${data.retrieved_chunks ? data.retrieved_chunks.length : 0}</span>
          </div>
          ${chunksHtml}
        </div>
      `;
      chatArea.appendChild(msg);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.innerText = text;
      return div.innerHTML;
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
