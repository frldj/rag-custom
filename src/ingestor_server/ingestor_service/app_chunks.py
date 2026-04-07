import logging  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunking_api")

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, Literal
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from src.utils.docling_chunker import (
    MultiFormatDoclingChunker,
    chunks_to_dicts,
)

import os
from datetime import datetime
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Chunking API")

chunker = MultiFormatDoclingChunker(
    min_chars=100,
    max_chars=1200,
    strategy="hybrid" # Stratégie par défaut
)


summarizer_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), 
    base_url=os.getenv("OLLAMA_URL"),
    temperature=0,
    num_ctx=8000,
    num_predict=1000
)

async def generate_real_summary(text: str) -> str:
    # On reste sur 32k, mais on s'assure que le timeout est géré
    truncated_text = text[:8000] 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "/no_think"),
        ("human", """Tu es un expert en indexation sémantique.
            Analyses ce document et génère une fiche d'indexation :
            TITRE : <Nom>
            DOMAINE : <Catégorie | Sous-catégorie>
            MOTS-CLÉS : <8 mots-clés>
            RÉSUMÉ_UTILITÉ : <Paragraphe factuel de 5-6 phrases sur les problèmes résolus et conclusions.>

            Texte : {input_text}""")
        ])
    
    chain = prompt | summarizer_llm | StrOutputParser()
    
    try:
        summary = await asyncio.wait_for(
            chain.ainvoke({"input_text": truncated_text}), 
            timeout=60.0
        )
        return summary.strip()
    except asyncio.TimeoutError:
        logger.error("Timeout Ollama sur le résumé")
        return "ERREUR : Temps de génération trop long."
    except Exception as e:
        logger.error(f"Erreur Ollama : {e}")
        return f"ERREUR_SUMMARIZATION : {type(e).__name__}"
    

def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " "))
    return name.strip() or "uploaded_file"

@app.post("/chunks")
async def create_chunks(
    file: UploadFile = File(...),
    strategy: Literal["hybrid", "recursive"] = Query(
        "hybrid", 
        description="Choix de la méthode de découpage : 'hybrid' (structurel/tableaux) ou 'recursive' (paragraphes)"
    ),
    max_chars: int = Query(1200, description="Taille maximale d'un chunk"),
    min_chars: int = Query(100, description="Taille minimale (fusion avec le précédent)")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    safe_name = sanitize_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()

    if not suffix:
        raise HTTPException(status_code=400, detail="Extension de fichier manquante")

    temp_dir = tempfile.mkdtemp(prefix="chunk_upload_")
    tmp_path = Path(temp_dir) / safe_name

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # A. EXTRACTION DU TEXTE BRUT POUR LE RÉSUMÉ
        conversion_res = chunker.converter.convert(tmp_path)
        full_text = conversion_res.document.export_to_markdown()

        # B. GÉNÉRATION DU VRAI RÉSUMÉ (Async)
        real_summary = await generate_real_summary(full_text)

        # C. DATE DU JOUR (Format ISO pour Milvus)
        today_date = datetime.now().strftime("%Y-%m-%d")

        chunks = chunker.chunk_file(
            tmp_path, 
            strategy=strategy,
            max_chars=max_chars,
            min_chars=min_chars,
            summary=real_summary,
            doc_date=today_date    
        )

        return {
            "filename": file.filename,
            "format": suffix.replace(".", ""),
            "summary_generated": real_summary, 
            "ingestion_date": today_date,
            "strategy_used": strategy,
            "chunk_count": len(chunks),
            "chunks": chunks_to_dicts(chunks),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Erreur lors du chunking : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)