#!/usr/bin/env bash
# Downloads the ArXiv papers used as sample corpus during development.
# Saves them to src/ingestor_server/ingestor_service/pdfs/ (git-ignored).

set -euo pipefail

DEST="src/ingestor_server/ingestor_service/pdfs"
mkdir -p "$DEST"

declare -A PAPERS=(
  ["2310.08560v2"]="SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
  ["2312.03732v1"]="Chain-of-Note: Enhancing Robustness in RAG"
  ["2405.04434v5"]="FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research"
  ["2412.13663v2"]="RAG-Star: Enhancing Deliberative Reasoning with RAG"
  ["2501.17887v1"]="Contextual RAG: Enhancing RAG with Context-Aware Filtering"
  ["2502.12110v10"]="Agentic RAG: A Survey on RAG-based AI Agents"
  ["2506.02153v1"]="Survey on Evaluation of RAG Systems"
  ["2507.13334v2"]="Advances in Embedding Models for Dense Retrieval"
)

for arxiv_id in "${!PAPERS[@]}"; do
  title="${PAPERS[$arxiv_id]}"
  dest_file="$DEST/${arxiv_id}.pdf"
  if [[ -f "$dest_file" ]]; then
    echo "Already exists: $arxiv_id"
    continue
  fi
  echo "Downloading $arxiv_id — $title"
  curl -L --retry 3 -o "$dest_file" "https://arxiv.org/pdf/${arxiv_id}"
done

echo "Done. PDFs saved to $DEST/"
