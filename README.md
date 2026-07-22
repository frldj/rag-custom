# RAG pipeline (Docling chunking + Milvus hybrid + Ollama + BGE rerank)

Ce repo contient plusieurs services FastAPI + scripts CLI pour :
1) extraire et chunker des documents (Docling + chunking structure/semantic),
2) ingérer dans Milvus (hybride BM25 + dense),
3) répondre via un service RAG (Milvus hybrid + rerank BGE + génération LLM),
4) exposer un service de rerank BGE via HTTP.

Ce repo contient également plusieurs fichiers docker compose :
1) exposer un service d'embedding via TEI.
2) exposer un service langfuse en local.
3) lancement de milvus en local.


## Architecture

Ports :
- **Ollama** : `http://localhost:11434`
- **Milvus** : `http://localhost:19530`
- **Embedding TEI** : `http://localhost:8083`
- **Rerank service** (`src/rerank_server/rerank_service.py`) : `http://localhost:8001`
- **Chunking API** (`src/ingestor_server/ingestor_service/app_chunks.py`) : `http://localhost:8002`
- **VDB service** (`src/ingestor_server/vdb_service/app_vdb_milvus.py`) : `http://localhost:8003`
- **RAG service** (`src/rag_server/rag_service_telemetry.py`) : `http://localhost:8004`

Fichiers Docker Compose :
- **Milvus** : `deploy/compose/docker-compose-milvus.yml`
- **Embedding TEI** : `deploy/compose/docker-compose-embedding.yaml`
- **Ingestor** : `deploy/compose/docker-compose-ingestor.yaml`
- **RAG server** : `deploy/compose/docker-compose-rag-server.yaml`
- **Langfuse** : `https://github.com/langfuse/langfuse.git`
- **Monitoring** : `deploy/compose/monitoring/docker-compose.yaml`

---

## Prérequis

- Python 3.10+ recommandé
- Ollama installé et lancé
- Docker disponible
- Compte Hugging Face accessible
- (optionnel) LibreOffice si ingestion de `.docx` via conversion PDF (commande `soffice`)

---

## Déploiement Docker

Le projet se déploie via plusieurs Docker Compose indépendants à lancer **dans l'ordre suivant**.

### 1. Milvus (base vectorielle)

```bash
cd deploy/compose
docker compose -f docker-compose-milvus.yml up -d
```

### 2. Embedding TEI

Télécharger le modèle d'embedding avant le premier lancement :

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-base', local_dir='~/hf_models/multilingual-e5-base')"
```

Lancer le service depuis `deploy/compose/` :

```bash
docker compose -f docker-compose-embedding.yaml up -d
```

### 3. Langfuse (observabilité)

Cloner Langfuse **en dehors** du projet (ex. dans le home), puis le démarrer :

```bash
cd ~
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

### 4. Services ingestor (chunk-service + vdb-server)

Depuis `deploy/compose/` :

```bash
docker compose -f docker-compose-ingestor.yaml up --build
```

### 5. RAG server (rerank-server + rag-server)

Depuis `deploy/compose/` :

```bash
docker compose -f docker-compose-rag-server.yaml up --build
```

### 6. Monitoring — optionnel (Prometheus + Grafana + Zipkin + OTEL)

Depuis `deploy/compose/` :

```bash
docker compose -f monitoring/docker-compose.yaml up -d
```

---

## Modèles Ollama

Installer les modèles utilisés :

```bash
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5vl:3b
```

---

## Installation modèle Docling

Télécharger les modèles de traitement via docling-tools :

```bash
docling-tools models download
```

---

## LibreOffice (optionnel — ingestion `.docx`)

Si Mac :
```bash
brew install --cask libreoffice
```

Si Linux :
```bash
sudo apt update && sudo apt install -y libreoffice
```

Test :
```bash
soffice --version
```

---

## Serveur GLiNER (anonymisation PII dans Langfuse)

GLiNER masque les données personnelles (email, téléphone, nom, adresse) dans les traces Langfuse avant envoi. Il doit tourner **sur la machine hôte** sur le port `1235`.

> **Optionnel** — si GLiNER est arrêté, l'anonymisation est ignorée silencieusement et les traces sont envoyées à Langfuse avec les textes bruts.

Depuis `src/rag_server/` :

```bash
python -m gliner_server.server
```

Vérifier que le serveur répond :

```bash
curl http://localhost:1235/v1/extract
```

En Docker, le rag-server contacte GLiNER via `host.docker.internal:1235` (configuré automatiquement via la variable `GLINER_SERVER_ENDPOINT` dans `docker-compose-rag-server.yaml`).

---

## Redis (cache)

```bash
docker run -d --name redis-rag -p 6380:6379 redis
```

Surveiller la consommation mémoire :
```bash
docker exec -it redis-rag redis-cli info memory
```

---

## Installation Python (développement local sans Docker)

Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -U pip
pip install -r requirements.txt
```

Ou via Poetry :
```bash
poetry install
```

Si installation via Poetry, lancer les microservices avec `poetry run` :
```bash
poetry run python ingestor.py
```

---

## Démarrer les services localement (sans Docker)

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
uvicorn rerank_server.rerank_service:app --host 0.0.0.0 --port 8001
uvicorn ingestor_server.ingestor_service.app_chunks:app --host 0.0.0.0 --port 8002
uvicorn ingestor_server.vdb_service.app_vdb_milvus:app --host 0.0.0.0 --port 8003
uvicorn rag_server.rag_service_telemetry:app --host 0.0.0.0 --port 8004
```

---

## Ingestion (PDF/Word → chunks → upsert)

Script CLI : `src/ingestor_server/ingestor_service/ingestor.py`

Stratégie hybrid :
```bash
python src/ingestor_server/ingestor_service/ingestor.py ./src/ingestor_server/ingestor_service/pdfs \
  --strategy hybrid \
  --max-chars 1200
```

Stratégie recursive :
```bash
python src/ingestor_server/ingestor_service/ingestor.py ./src/ingestor_server/ingestor_service/pdfs \
  --strategy recursive \
  --batch-size 16
```
