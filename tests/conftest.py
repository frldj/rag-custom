import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy packages unavailable in the CI test environment (no GPU, no Docker).
# setdefault: only stubs if the real package is not already installed.
# ---------------------------------------------------------------------------
_STUB_PACKAGES = [
    # ML / GPU
    "torch",
    "torch.backends",
    # Reranker
    "FlagEmbedding",
    # Docling (document parsing)
    "docling",
    "docling.datamodel",
    "docling.datamodel.base_models",
    "docling.datamodel.pipeline_options",
    "docling.datamodel.accelerator_options",
    "docling.document_converter",
    "docling.backend",
    "docling.backend.docling_parse_v4_backend",
    "docling.chunking",
    "docling_ocr_onnxtr",
    "onnxtr",
    "onnxtr.models",
    "docling_core",
    "docling_core.transforms",
    "docling_core.transforms.chunker",
    "docling_core.transforms.chunker.hierarchical_chunker",
    "docling_core.transforms.serializer",
    "docling_core.transforms.serializer.markdown",
    # LangChain (only the submodules used in the services)
    "langchain_ollama",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    # HuggingFace Hub (used in docling_chunker)
    "huggingface_hub",
    # Text splitter (used in docling_chunker)
    "langchain_text_splitters",
]
for _pkg in _STUB_PACKAGES:
    sys.modules.setdefault(_pkg, MagicMock())

# ---------------------------------------------------------------------------
# Make src/ importable without installing the package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Minimal env vars so modules that read os.getenv at import time don't crash
# ---------------------------------------------------------------------------
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
os.environ.setdefault("EMBEDDING_DIMENSION", "768")
os.environ.setdefault("MILVUS_COLLECTION", "test_collection")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("OLLAMA_EMB_MODEL", "qwen3-embedding:0.6b")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
os.environ.setdefault("RERANK_MODEL", "BAAI/bge-reranker-base")
os.environ.setdefault("USE_FP16", "false")
os.environ.setdefault("MAX_PASSAGES", "256")
os.environ.setdefault("MAX_TEXT_CHARS", "2000")
os.environ.setdefault("RERANK_URL", "http://localhost:8282/rerank")
os.environ.setdefault("TOP_K_FINAL", "5")
os.environ.setdefault("TOP_K_RECALL", "60")
os.environ.setdefault("BUILD_CONTEXT_MAX_CHARS", "6000")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6380")
os.environ.setdefault("MAX_EMB_CHARS", "8000")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")

# ---------------------------------------------------------------------------
# Pre-import service modules so `patch("src.x.y.attr", ...)` can always
# resolve the dotted path (mock.patch resolves via getattr on the parent
# package, which only works once the submodule is in sys.modules).
# ---------------------------------------------------------------------------
import src.rerank_server.rerank_service
import src.ingestor_server.ingestor_service.app_chunks
