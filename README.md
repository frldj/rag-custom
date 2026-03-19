# RAG pipeline (Docling chunking + Milvus hybrid + Ollama + BGE rerank)

Ce repo contient plusieurs services FastAPI + scripts CLI pour :
1) extraire et chunker des documents (Docling + chunking structure/semantic),
2) ingérer dans Milvus (hybride BM25 + dense),
3) répondre via un service RAG (Milvus hybrid + rerank BGE + génération LLM),
4) exposer un service de rerank BGE via HTTP.

## Architecture (ports par défaut)

- **Ollama** : `http://localhost:11434`
- **Milvus** : `http://localhost:19530`
- **Rerank service** (`rerank_service.py`) : `http://localhost:8001`
- **Chunking API** (`app_vchunking.py`) : `http://localhost:8002`
- **VDB service** (`app_vdb_milvus.py`) : `http://localhost:8003`
- **RAG service** (`rag_service.py`) : `http://localhost:8004`

---

## Prérequis

- Python 3.10+ recommandé
- Ollama installé et lancé
- Milvus lancé (standalone recommandé)
- Modèle Docling disponible 
- (optionnel) LibreOffice si ingestion de `.docx` via conversion PDF (commande `soffice`)

---

## Installation modèle docling

il faut télécharger via docling-tools les modèles de traitement :

```bash
docling-tools models download
```

## Installation de Milvus
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

## Installation (Python)

Créer un venv et installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows

pip install -U pip
pip install -r requirements.txt

```

## LibreOffice
Si Mac:
```bash
brew install --cask libreoffice
```

Si Linux:
```bash
sudo apt update
sudo apt install -y libreoffice
```

Test:
```bash
soffice --version
```

## Modèle Ollama

Installer les modèles utilisés :

```bash
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5vl:3b
```

## Démarrer les services 

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
uvicorn rerank_service:app --host 0.0.0.0 --port 8001
uvicorn app_chunks:app --host 0.0.0.0 --port 8002
uvicorn app_vdb_milvus:app --host 0.0.0.0 --port 8003
uvicorn rag_service:app --host 0.0.0.0 --port 8004
```


## Ingestion (PDF/Words -> chunks -> upsert)

Script CLI : ingestion_kb.py

Exemple :

```bash
python ingestion_kb.py ./docs \
  --chunking-url http://localhost:8002 \
  --vdb-url http://localhost:8003 \
  --collection rag_minist_int_hybrid_v2 \
  --mode insert \
  --pdf-strategy structure \
  --chunk-size-tokens 1000 \
  --min-chunk-chars 200
```