import pytest
from fastapi.testclient import TestClient

from app.config import SIMILARITY_THRESHOLD
from app.graph import rag_app, route, RAGState
from app.main import app
from app.vector_store import cosine_similarity, query_vector_store


@pytest.fixture
def client():
    return TestClient(app)


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pinecone_configured" in data
    assert "groq_model" in data


def test_langgraph_route_logic():
    # Test high confidence routing to generate
    state_high: RAGState = {
        "question": "What is Agentic AI?",
        "chunks": [{"text": "Sample text", "page": 1, "score": 0.85}],
        "answer": "",
        "confidence": 0.85,
    }
    assert route(state_high) == "generate"

    # Test low confidence routing to fallback
    state_low: RAGState = {
        "question": "Random out of scope query?",
        "chunks": [{"text": "Irrelevant", "page": 1, "score": 0.12}],
        "answer": "",
        "confidence": 0.12,
    }
    assert route(state_low) == "fallback"

    # Test empty chunks routing to fallback
    state_empty: RAGState = {
        "question": "Empty test",
        "chunks": [],
        "answer": "",
        "confidence": 0.0,
    }
    assert route(state_empty) == "fallback"


def test_rag_pipeline_in_scope():
    result = rag_app.invoke({"question": "What is Agentic AI?"})
    assert "answer" in result
    assert "chunks" in result
    assert "confidence" in result
    assert result["confidence"] >= SIMILARITY_THRESHOLD
    assert len(result["chunks"]) > 0


def test_rag_pipeline_out_of_scope():
    result = rag_app.invoke({"question": "Who won the FIFA World Cup in 2018?"})
    assert result["answer"] == "I couldn't find this in the Agentic AI eBook."
    assert result["confidence"] < SIMILARITY_THRESHOLD


def test_fastapi_chat_endpoint(client):
    res = client.post("/chat", json={"question": "What is Agentic AI?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert "confidence" in data
    assert isinstance(data["retrieved_chunks"], list)
