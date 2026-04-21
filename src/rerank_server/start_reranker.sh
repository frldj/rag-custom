# #!/bin/bash

# # On s'assure que le chemin vers les binaires Rust est connu
# export PATH="$HOME/.cargo/bin:$PATH"

# echo "Démarrage du Reranker (Natif Metal)..."

# # On lance le service en tâche de fond
# # Les logs iront dans un fichier 'reranker.log' au lieu de polluer ta console
# nohup text-embeddings-router --model-id BAAI/bge-reranker-base --port 8084 --auto-truncate > reranker.log 2>&1 &

# echo "✅ Reranker lancé en arrière-plan !"
# echo "Logs : tail -f reranker.log"
# echo "Pour l'arrêter : pkill text-embeddings-router"


# # text-embeddings-router \
# #   --model-id BAAI/bge-reranker-base \
# #   --port 8084 \
# #   --auto-truncate \
# #   --max-client-batch-size 32 \
# #   --max-concurrent-requests 10


#!/bin/bash
#!/bin/bash

export PATH="$HOME/.cargo/bin:$PATH"

# On nettoie tout ce qui traîne sur 8282
echo "--- 🛠️  Nettoyage du port 8282 ---"
lsof -ti:8282 | xargs kill -9 2>/dev/null
pkill text-embeddings-router 2>/dev/null
sleep 2

echo "🚀 Lancement du Reranker sur le port 8282..."

# L'astuce : on définit des adresses spécifiques pour éviter le "os error 48"
# On demande au router de ne pas essayer de binder les metrics sur 0.0.0.0
nohup text-embeddings-router \
  --model-id BAAI/bge-reranker-base \
  --port 8282 \
  --auto-truncate \
  --disable-spans > reranker.log 2>&1 &

echo "✅ Lancé sur le port 8282."
echo "🔎 Vérification : tail -f reranker.log"