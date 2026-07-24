"""Unit tests for the chunking API (app_chunks.py).

Module-level instances (chunker, summarizer_llm) are created with MagicMock
stubs via conftest.py; this fixture configures their return values for each test.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_chunk(text="Sample chunk text"):
    chunk = MagicMock()
    chunk.text = text
    return chunk


@pytest.fixture()
def chunking_client():
    mock_chunks = [_make_mock_chunk("chunk 1"), _make_mock_chunk("chunk 2")]

    mock_chunker = MagicMock()
    mock_conversion = MagicMock()
    mock_conversion.document.export_to_markdown.return_value = "Mocked document markdown"
    mock_chunker.converter.convert.return_value = mock_conversion
    mock_chunker.chunk_from_result.return_value = mock_chunks

    with (
        # Replace the module-level `chunker` instance entirely.
        patch("src.ingestor_server.ingestor_service.app_chunks.chunker", mock_chunker),
        patch(
            "src.ingestor_server.ingestor_service.app_chunks.generate_real_summary",
            new=AsyncMock(return_value="Mocked summary"),
        ),
        patch(
            "src.ingestor_server.ingestor_service.app_chunks.chunks_to_dicts",
            return_value=[{"text": "chunk 1"}, {"text": "chunk 2"}],
        ),
    ):
        from src.ingestor_server.ingestor_service.app_chunks import app

        yield TestClient(app)


def _pdf_upload(filename="test.pdf", content=b"%PDF-1.4 fake content"):
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


class TestChunkingEndpoint:
    def test_post_chunks_returns_200(self, chunking_client):
        response = chunking_client.post(
            "/chunks",
            files=[_pdf_upload()],
        )
        assert response.status_code == 200

    def test_response_structure(self, chunking_client):
        response = chunking_client.post("/chunks", files=[_pdf_upload()])
        data = response.json()
        assert "chunk_count" in data
        assert "chunks" in data
        assert "filename" in data
        assert "strategy_used" in data

    def test_default_strategy_is_hybrid(self, chunking_client):
        response = chunking_client.post("/chunks", files=[_pdf_upload()])
        assert response.json()["strategy_used"] == "hybrid"

    def test_recursive_strategy_accepted(self, chunking_client):
        response = chunking_client.post(
            "/chunks?strategy=recursive", files=[_pdf_upload()]
        )
        assert response.status_code == 200
        assert response.json()["strategy_used"] == "recursive"

    def test_missing_filename_rejected(self, chunking_client):
        response = chunking_client.post(
            "/chunks",
            files=[("file", ("", io.BytesIO(b"content"), "application/pdf"))],
        )
        assert response.status_code in (400, 422)

    def test_chunk_count_matches_chunks_list(self, chunking_client):
        response = chunking_client.post("/chunks", files=[_pdf_upload()])
        data = response.json()
        assert data["chunk_count"] == len(data["chunks"])

    def test_summary_present_in_response(self, chunking_client):
        response = chunking_client.post("/chunks", files=[_pdf_upload()])
        data = response.json()
        assert "summary_generated" in data
        assert isinstance(data["summary_generated"], str)
