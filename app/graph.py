import logging
import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from app.vector_store import query_vector_store

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I couldn't find this in the Agentic AI eBook."

STRICT_RAG_PROMPT_TEMPLATE = """You are an assistant that answers ONLY using the context below, which is taken from the "Agentic AI" eBook.
- If the answer is not in the context, reply exactly:
  "{fallback_message}"
- Do not use outside knowledge.
- Keep the answer short and clear.

Context:
{context}

Question: {question}
Answer:"""


class RAGState(TypedDict):
    question: str
    chunks: List[Dict[str, Any]]
    answer: str
    confidence: float


def get_llm():
    """Initializes LLM instance (Groq Llama 3.1 default)."""
    # 1. Groq (Primary choice as per project spec)
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_key,
            temperature=0,
            max_tokens=600,
        )

    # 2. OpenAI fallback if key exists
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
        except Exception as e:
            logger.warning(f"Could not load ChatOpenAI: {e}")

    # 3. Google Gemini fallback if key exists
    google_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key, temperature=0)
        except Exception as e:
            logger.warning(f"Could not load ChatGoogleGenerativeAI: {e}")

    return None


def retrieve(state: RAGState) -> Dict[str, Any]:
    """Retrieves relevant chunks from vector store and calculates confidence score."""
    question = state.get("question", "").strip()
    logger.info(f"Retrieving chunks for question: '{question}'")

    chunks = query_vector_store(question, top_k=TOP_K)

    # Calculate confidence as the highest cosine similarity score among top matches
    if chunks and len(chunks) > 0:
        top_score = max(c.get("score", 0.0) for c in chunks)
        confidence = round(float(top_score), 4)
    else:
        confidence = 0.0

    logger.info(f"Retrieved {len(chunks)} chunks. Top confidence score: {confidence}")
    return {
        "chunks": chunks,
        "confidence": confidence,
    }


def route(state: RAGState) -> str:
    """
    Conditional routing edge:
    If confidence is below threshold, routes to fallback to prevent hallucinations.
    Otherwise, routes to generate node.
    """
    confidence = state.get("confidence", 0.0)
    chunks = state.get("chunks", [])

    if not chunks or confidence < SIMILARITY_THRESHOLD:
        logger.info(f"Routing to 'fallback' (Confidence {confidence} < Threshold {SIMILARITY_THRESHOLD})")
        return "fallback"

    logger.info(f"Routing to 'generate' (Confidence {confidence} >= Threshold {SIMILARITY_THRESHOLD})")
    return "generate"


def generate(state: RAGState) -> Dict[str, Any]:
    """Generates an answer using retrieved chunks and LLM with strict grounding prompt."""
    question = state["question"]
    chunks = state.get("chunks", [])

    # Format context with explicit page citations
    context_parts = []
    for c in chunks:
        page_num = c.get("page", "Unknown")
        text = c.get("text", "")
        context_parts.append(f"[Page {page_num}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = STRICT_RAG_PROMPT_TEMPLATE.format(
        fallback_message=FALLBACK_MESSAGE,
        context=context,
        question=question,
    )

    llm = get_llm()

    if llm is not None:
        try:
            response = llm.invoke(prompt)
            answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return {"answer": answer_text}
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            # Fall back to extractive synthesis if LLM API call fails
            return {
                "answer": f"According to the Agentic AI eBook (retrieved with confidence {state.get('confidence', 0)}):\n\n"
                          f"{chunks[0].get('text', '')}"
            }
    else:
        # Graceful fallback when no external LLM API key has been added to .env yet
        top_chunk = chunks[0] if chunks else {}
        top_text = top_chunk.get("text", "").strip()
        top_page = top_chunk.get("page", 1)
        return {
            "answer": f"[Grounded Context from Page {top_page}]: {top_text}\n\n"
                      f"(Note: To enable natural language synthesis with Llama-3.1-8B, add your free GROQ_API_KEY to .env)"
        }


def fallback(state: RAGState) -> Dict[str, Any]:
    """Fallback node triggered when question is out-of-scope or relevance is low."""
    logger.info("Executing fallback node.")
    return {"answer": FALLBACK_MESSAGE}


# Build LangGraph StateGraph
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("fallback", fallback)

# Entry point
workflow.set_entry_point("retrieve")

# Conditional edge based on retrieval score
workflow.add_conditional_edges(
    "retrieve",
    route,
    {
        "generate": "generate",
        "fallback": "fallback",
    },
)

# Terminal edges
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# Compile LangGraph application
rag_app = workflow.compile()
