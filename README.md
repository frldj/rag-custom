# RAG pipeline (Docling chunking + Milvus hybrid + Ollama + BGE rerank)

Ce repo contient plusieurs services FastAPI + scripts CLI pour :
1) extraire et chunker des documents (Docling + chunking structure/semantic),
2) ingérer dans Milvus (hybride BM25 + dense),
3) répondre via un service RAG (Milvus hybrid + rerank BGE + génération LLM),
4) exposer un service de rerank BGE via HTTP.

Ce repo contient également plusieurs liaison docker
1) exposer un service d'embedding via TEI.
2) exposer un service langfuse en local
3) lancement de milvus en local.

## Architecture 

Ports : 
- **Ollama** : `http://localhost:11434`
- **Milvus** : `http://localhost:19530`
- **Rerank service** (`src/rerank_server/rerank_service.py`) : `http://localhost:8001`
- **Chunking API** (`src/ingestor_server/ingestor_service/app_chunks.py`) : `http://localhost:8002`
- **VDB service** (`src/ingestor_server/vdb_service/app_vdb_milvus.py`) : `http://localhost:8003`
- **RAG service** (`src/rag_server/rag_service_langfuse.py`) : `http://localhost:8004`

Dockerfile : 
- **Milvus** : `docker-compose.yml` (projet racine)
- **Embedding TEI** : `deploy/compose/docker-compose-embedding.yaml`
- **Langfuse** : `https://github.com/langfuse/langfuse.git`

Module :
- **src/evaluation** : évaluation offline du rag avant déploiement
- **src/finetuning** : finetuning custom du modèle d'embedding

---

## Prérequis

- Python 3.11 recommandé
- Ollama installé et lancé
- Milvus lancé (standalone recommandé)
- Modèle Docling disponible 
- Compte huggingface accessible
- Docker disponible
- Langfuse self-hosted
- (optionnel) LibreOffice si ingestion de `.docx` via conversion PDF (commande `soffice`)

---
## Docker

Le projet peut être lancé via Docker Compose pour démarrer les différents services backend du pipeline RAG.

### Services exposés

- **Ollama** : `http://localhost:11434`
- **Milvus** : `http://localhost:19530`
- **Rerank service** (`src/rerank_server/rerank_service.py`) : `http://localhost:8001`
- **Chunking API** (`src/ingestor_server/ingestor_service/app_chunks.py.py`) : `http://localhost:8002`
- **VDB service** (`src/ingestor_server/vdb_service/app_vdb_milvus.py`) : `http://localhost:8003`
- **RAG service** (`src/rag_server/rag_service_langfuse.py`) : `http://localhost:8004`

---

### Structure Docker

Les fichiers Docker sont regroupés dans le dossier `deploy/compose/`.

#### `Dockerfile`

Le conteneur est basé sur `python:3.11-slim` et :

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
- **Embedding TEI**

#### Lancement

---
### Structure API/Microservices

### Pré-requis

Avant de lancer le projet, vérifier que :

1. **Ollama** est démarré sur la machine hôte
2. **Milvus** est déjà lancé et accessible sur le réseau Docker `milvus-net`
3. le fichier **`.env`** existe à la racine du projet
4. le réseau Docker **`milvus`** existe
5. Langfuse est démarré en self hosted
6. L'embedding TEI est lancé

## Installation de langfuse

  ```bash
  # Get a copy of the latest Langfuse repository
  git clone https://github.com/langfuse/langfuse.git
  cd langfuse

  # Run the langfuse docker compose
  docker compose up
  ```


## Installation embedding TEI

```bash
# Crée un dossier pour le modèle (par exemple dans ton home)
mkdir -p ~/hf_models
cd ~/hf_models

# Clone le modèle (ton Mac utilisera ton authentification système)
git clone https://huggingface.co/intfloat/multilingual-e5-base
```

Allez dans deploy/compose et lancez : 

```bash
docker compose -f docker-compose-embedding.yaml up --build 
```


## Installation modèle docling

il faut télécharger via docling-tools les modèles de traitement :

```bash
docling-tools models download
```

## Installation de Milvus

Allez à la racine du projet "ministere_interieur" et lancez :
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

## Installation (Python)

Créer un venv : installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows
```

installer les dépendances :
```bash
pip install -U pip
pip install -r requirements.txt
```

ou installer les dépendances via poetry :
```bash
poetry install
```

Si installation via poetry, il faut lancer tous les microservices par **poetry run** :
exemple :
```bash
poetry run python ingestor.py
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
## TEI pour Reranker

Installer Rust nativement :
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Git clone ce projet :
```bash
https://github.com/huggingface/text-embeddings-inference.git
```

Dans le projet et lancez :
```bash
cargo install --path router -F metal -F accelerate
```

Lancez le reranker via TEI :
```bash
text-embeddings-router --model-id BAAI/bge-reranker-base --port 8084 --auto-truncate
```


ou par script bash lancez : 

```bash
 chmod +x start_reranker.sh
 ```

 ```bash
 ./start_reranker.sh
 ```bash

## Lancez le service Redis

```bash
docker run -d --name redis-rag -p 6380:6379 redis
```

Surveillez la consommation redis : 
```bash
docker exec -it redis-rag redis-cli info memory
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
uvicorn rag_service:app --host 0.0.0.0 --port 8004 ou export PYTHONPATH=$PYTHONPATH:.                             
uvicorn src.rag_server.rag_service_telemetry:app --host 0.0.0.0 --port 8004
```


## Ingestion (PDF/Words -> chunks -> upsert)

Script CLI : src/ingestor_server/ingestor.py

Exemple :

```bash
python src/ingestor_server/ingestor_service/ingestor.py ./src/ingestor_server/ingestor_service/pdfs \
  --strategy hybrid \
  --max-chars 1200 \
```

ou 

```bash
python src/ingestor_server/ingestor_service/ingestor.py ./src/ingestor_server/ingestor_service/pdfs \ --strategy recursive --batch-size 16
```


