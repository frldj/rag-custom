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

app = FastAPI(title="Chunking API")

chunker = MultiFormatDoclingChunker(
    min_chars=600,
    max_chars=1500,
    strategy="hybrid" # Stratégie par défaut
)

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
    max_chars: int = Query(1500, description="Taille maximale d'un chunk"),
    min_chars: int = Query(600, description="Taille minimale (fusion avec le précédent)")
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

        # On met à jour les paramètres pour cet appel spécifique
        # Note : on passe les paramètres directement à chunk_file 
        # pour éviter de réinitialiser la classe entière
        chunks = chunker.chunk_file(
            tmp_path, 
            strategy=strategy,
            max_chars=max_chars,
            min_chars=min_chars
        )

        return {
            "filename": file.filename,
            "format": suffix.replace(".", ""),
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