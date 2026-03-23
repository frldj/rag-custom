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
- Compte huggingface accessible
- Docker disponible
- (optionnel) LibreOffice si ingestion de `.docx` via conversion PDF (commande `soffice`)

---
## Docker

Le projet peut être lancé via Docker Compose pour démarrer les différents services backend du pipeline RAG.

### Services exposés

- **Ollama** : `http://localhost:11434`
- **Milvus** : `http://localhost:19530`
- **Rerank service** (`rerank_service.py`) : `http://localhost:8001`
- **Chunking API** (`app_vchunking.py`) : `http://localhost:8002`
- **VDB service** (`app_vdb_milvus.py`) : `http://localhost:8003`
- **RAG service** (`rag_service.py`) : `http://localhost:8004`

---

### Structure Docker

Les fichiers Docker sont regroupés dans le dossier `deploy/compose/`.

#### `Dockerfile`

Le conteneur est basé sur `python:3.10-slim` et :

- définit `/app` comme répertoire de travail
- installe les dépendances système nécessaires
- copie `requirements.txt` et le fichier `.env`
- installe les dépendances Python
- copie le code source depuis `src/`
- configure `PYTHONPATH=/app/src`
- configure le cache Hugging Face dans `/app/.cache`

#### `docker-compose.yml`

Le `docker-compose.yml` démarre 4 services applicatifs :

- `rerank-server`
- `ingestor-server`
- `vdb-server`
- `rag-server`

Il s’appuie également sur des services externes/accessibles :

- **Ollama** sur l’hôte : `host.docker.internal:11434`
- **Milvus** sur le réseau Docker `milvus-net`

---
### Structure API

### Pré-requis

Avant de lancer le projet, vérifier que :

1. **Ollama** est démarré sur la machine hôte
2. **Milvus** est déjà lancé et accessible sur le réseau Docker `milvus-net`
3. le fichier **`.env`** existe à la racine du projet
4. le réseau Docker **`milvus`** existe

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

Script CLI : src/ingestor_server/ingestor.py

Exemple :

```bash
python ingestor.py ./docs \
  --strategy hybrid \
  --max-chars 1200 \
```