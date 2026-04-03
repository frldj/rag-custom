# FROM python:3.10-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# # Installation des dépendances système (zlib1g-dev pour FlagEmbedding)
# RUN apt-get update && apt-get install -y \
#     build-essential libgl1 libglib2.0-0 curl zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copie des fichiers depuis la racine du projet
# COPY requirements.txt .
# COPY .env .env

# RUN pip install --no-cache-dir -r requirements.txt

# # Copie du code source complet
# COPY src/ ./src/

# # CONFIGURATION DU CHEMIN PYTHON
# # En mettant /app/src, on peut importer "utils", "rag_server", etc. directement
# ENV PYTHONPATH="/app/src"
# ENV HF_HOME="/app/.cache"


# --- Étape 1 : Builder ---
FROM python:3.11-slim as builder

# Configuration de Poetry
ENV POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Installation des dépendances système nécessaires pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copie des fichiers de dépendances uniquement (pour le cache Docker)
COPY pyproject.toml poetry.lock ./

# Installation des dépendances (uniquement main, on exclut le dev)
RUN poetry install --only main --no-root

# --- Étape 2 : Image Finale ---
FROM python:3.11-slim

WORKDIR /app

# Variables d'environnement Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    HF_HOME="/app/.cache"

# Installation des librairies d'exécution (nécessaires pour l'OCR, Magic, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie des packages installés depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source et du .env
COPY src/ ./src/
COPY .env .env

# Création du dossier cache avec les permissions larges (pour les modèles IA)
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Pas de CMD ici : elle est définie individuellement dans tes fichiers docker-compose