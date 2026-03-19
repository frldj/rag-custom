FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances système (zlib1g-dev pour FlagEmbedding)
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 curl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers depuis la racine du projet
COPY requirements.txt .
COPY .env .env

RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source complet
COPY src/ ./src/

# CONFIGURATION DU CHEMIN PYTHON
# En mettant /app/src, on peut importer "utils", "rag_server", etc. directement
ENV PYTHONPATH="/app/src"
ENV HF_HOME="/app/.cache"