import os
import sys
from pathlib import Path

# Make src/ importable without installing the package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Minimal env so modules that read os.getenv at import time don't crash
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
