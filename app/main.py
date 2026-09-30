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
    """Interactive, modern UI for chatting with the Agentic AI eBook."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic AI eBook - RAG Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --accent-warn: #f59e0b;
      --accent-danger: #ef4444;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(11, 15, 25, 0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .badge {
      font-size: 0.75rem;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.4);
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-weight: 600;
    }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.875rem;
      font-weight: 500;
      margin-left: 1.5rem;
      transition: color 0.2s;
    }
    .nav-links a:hover {
      color: var(--text);
    }
    main {
      flex: 1;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .chat-container {
      flex: 1;
      min-height: 480px;
      max-height: 600px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      padding-right: 0.5rem;
    }
    .chat-container::-webkit-scrollbar {
      width: 6px;
    }
    .chat-container::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
    }
    .message {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-width: 85%;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message.user {
      align-self: flex-end;
    }
    .message.assistant {
      align-self: flex-start;
      max-width: 95%;
    }
    .bubble {
      padding: 1rem 1.25rem;
      border-radius: 1rem;
      font-size: 0.95rem;
      line-height: 1.6;
    }
    .user .bubble {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #fff;
      border-bottom-right-radius: 0.25rem;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }
    .assistant .bubble {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-bottom-left-radius: 0.25rem;
      backdrop-filter: blur(8px);
    }
    .meta-bar {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 0.35rem;
    }
    .confidence-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.72rem;
    }
    .confidence-high {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .confidence-mid {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .confidence-low {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .chunks-details {
      margin-top: 0.6rem;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.6);
    }
    .chunks-toggle {
      padding: 0.5rem 0.8rem;
      font-size: 0.75rem;
      cursor: pointer;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
      user-select: none;
    }
    .chunks-content {
      display: none;
      padding: 0.75rem;
      border-top: 1px solid var(--card-border);
      font-size: 0.8rem;
      color: #cbd5e1;
      max-height: 200px;
      overflow-y: auto;
    }
    .chunk-item {
      padding: 0.5rem;
      margin-bottom: 0.5rem;
      background: rgba(30, 41, 59, 0.6);
      border-radius: 0.35rem;
      border-left: 3px solid var(--primary);
    }
    .chunk-item:last-child {
      margin-bottom: 0;
    }
    .chunk-header {
      font-weight: 600;
      font-size: 0.72rem;
      color: #818cf8;
      margin-bottom: 0.25rem;
      display: flex;
      justify-content: space-between;
    }
    .input-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 0.6rem 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(12px);
    }
    .input-box input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-size: 0.95rem;
      padding: 0.4rem 0.5rem;
    }
    .input-box input::placeholder {
      color: #64748b;
    }
    .send-btn {
      background: var(--primary);
      color: #fff;
      border: none;
      outline: none;
      padding: 0.65rem 1.25rem;
      border-radius: 0.75rem;
      font-weight: 600;
      font-size: 0.875rem;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .send-btn:hover {
      background: var(--primary-hover);
    }
    .send-btn:active {
      transform: scale(0.98);
    }
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .suggestion-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      padding: 0.4rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
    }
    .suggestion-chip:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.3);
      color: #e0e7ff;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="22"></line>
      </svg>
      <div>
        <strong>Agentic AI Assistant</strong>
        <span class="badge">LangGraph + Pinecone</span>
      </div>
    </div>
    <div class="nav-links">
      <a href="/docs" target="_blank">Swagger API Docs</a>
      <a href="/health" target="_blank">System Health</a>
    </div>
  </header>

  <main>
    <div class="suggestions">
      <span class="suggestion-chip" onclick="ask(this.innerText)">What is Agentic AI?</span>
      <span class="suggestion-chip" onclick="ask(this.innerText)">How is Agentic AI different from traditional AI?</span>
      <span class="suggestion-chip" onclick="ask(this.innerText)">What are the key components of an AI agent?</span>
      <span class="suggestion-chip" onclick="ask(this.innerText)">Who won the FIFA World Cup in 2018?</span>
    </div>

    <div class="chat-container" id="chatArea">
      <div class="message assistant">
        <div class="bubble">
          Hello! I am strictly grounded on the <strong>Agentic AI eBook</strong>. Ask me anything about AI agents, architecture, orchestration, or enterprise readiness!
        </div>
      </div>
    </div>

    <form class="input-box" id="chatForm" onsubmit="handleSubmit(event)">
      <input type="text" id="userInput" placeholder="Ask a question about the Agentic AI eBook..." autocomplete="off" />
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

      // Add user bubble
      appendMessage('user', question);
      userInput.value = '';
      sendBtn.disabled = true;
      sendBtn.innerText = 'Thinking...';

      // Temporary assistant loading bubble
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
      msg.innerHTML = '<div class="bubble"><em>Analyzing knowledge base...</em></div>';
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
