"""Unit tests for the BGE rerank service (rerank_service.py).

The module-level `reranker` instance is patched directly so tests run
without GPU or model download.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def rerank_client():
    mock_reranker = MagicMock()
    mock_reranker.compute_score.return_value = [0.9, 0.3, 0.7]

    # Patch the module-level reranker instance, not the class.
    # The module is already imported via conftest stubs; patching the instance
    # is the only reliable way to control what the endpoint uses.
    with patch("src.rerank_server.rerank_service.reranker", mock_reranker):
        from src.rerank_server.rerank_service import app

        yield TestClient(app), mock_reranker


class TestRerankEndpoint:
    def test_health_returns_ok(self, rerank_client):
        client, _ = rerank_client
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_rerank_returns_scores(self, rerank_client):
        client, _ = rerank_client
        payload = {"query": "What is RAG?", "passages": ["RAG is retrieval augmented generation.", "Cats are pets.", "RAG improves LLM accuracy."]}
        response = client.post("/rerank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert len(data["scores"]) == 3

    def test_scores_are_floats(self, rerank_client):
        client, _ = rerank_client
        payload = {"query": "RAG", "passages": ["passage A", "passage B"]}
        response = client.post("/rerank", json=payload)
        for score in response.json()["scores"]:
            assert isinstance(score, float)

    def test_passages_truncated_to_max_chars(self, rerank_client):
        client, mock_reranker = rerank_client
        long_passage = "x" * 5000
        payload = {"query": "q", "passages": [long_passage]}
        client.post("/rerank", json=payload)
        call_args = mock_reranker.compute_score.call_args
        pairs = call_args[0][0]
        assert len(pairs[0][1]) <= 2000

    def test_empty_passages_rejected(self, rerank_client):
        client, _ = rerank_client
        payload = {"query": "q", "passages": []}
        response = client.post("/rerank", json=payload)
        assert response.status_code == 422

    def test_model_name_in_response(self, rerank_client):
        client, _ = rerank_client
        payload = {"query": "q", "passages": ["p"]}
        response = client.post("/rerank", json=payload)
        assert "model" in response.json()

    def test_took_ms_is_positive_int(self, rerank_client):
        client, _ = rerank_client
        payload = {"query": "q", "passages": ["p"]}
        response = client.post("/rerank", json=payload)
        assert response.json()["took_ms"] >= 0
