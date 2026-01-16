from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Configuration
# =========================
@dataclass
class ChunkingConfig:
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 100
    chars_per_token: int = 4
    
    # Comportement
    drop_bibliography: bool = True
    drop_appendix: bool = True
    stop_after_appendix_start: bool = True 
    min_chunk_chars: int = 100 # PPT a parfois peu de texte par slide
    
    # Marqueurs
    appendix_markers: Tuple[str, ...] = ("annexe", "annexes", "appendix", "appendice", "conclusion")
    bibliography_markers: Tuple[str, ...] = ("bibliographie", "references", "sources")

# =========================
# Modèle Chunk
# =========================
@dataclass
class Chunk:
    id: str
    source: str
    section_title: str
    text: str
    page_no: int
    meta: Dict[str, Any]

# =========================
# Fonctions Utilitaires
# =========================
def _count_tokens(text: str, cfg: ChunkingConfig) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text) // cfg.chars_per_token

def _is_marker_present(title: str, markers: Tuple[str, ...]) -> bool:
    t = title.lower().strip()
    t_clean = re.sub(r'^[\d\.]+\s+', '', t)
    for m in markers:
        if rf"\b{re.escape(m)}\b" in t_clean and len(t_clean.split()) <= 2:
            return True
    return False

# =========================
# Logique de Chunking PPTX
# =========================
# def chunk_pptx_from_res(res: Any, cfg: Optional[ChunkingConfig] = None) -> List[Chunk]:
#     cfg = cfg or ChunkingConfig()
    
#     # Récupération du markdown brut
#     md_raw = res.artifacts.get("markdown_final") or res.artifacts.get("markdown") or ""
#     source = getattr(res, "source", "unknown_pptx")
    
#     # 1. Découpage par Page (Slide)
#     # On utilise le placeholder généré par Docling
#     slides = md_raw.split("<!-- page_break -->")
    
#     chunks: List[Chunk] = []
#     in_excluded_zone = False

#     for i, slide_content in enumerate(slides):
#         page_no = i + 1
#         slide_content = slide_content.strip()
#         if not slide_content:
#             continue

#         # 2. Identification du titre de la slide
#         # Souvent la première ligne ou le premier # 
#         lines = slide_content.split('\n')
#         slide_title = "Sans titre"
#         for line in lines:
#             clean_line = line.strip().lstrip('#').strip()
#             if clean_line:
#                 slide_title = clean_line
#                 break

#         # 3. Filtrage Appendice / Fin de doc
#         if _is_marker_present(slide_title, cfg.appendix_markers) or \
#            _is_marker_present(slide_title, cfg.bibliography_markers):
#             in_excluded_zone = True
            
#         if in_excluded_zone and cfg.stop_after_appendix_start:
#             continue

#         # 4. Nettoyage et préparation
#         # On garde l'intégralité de la slide dans la meta pour le RAG
#         full_slide_text = slide_content.replace("<!-- image_placeholder -->", "[Image]").strip()

#         # 5. Chunking de la slide
#         # Si la slide est très courte, on fait un seul chunk.
#         # Si elle est très longue (rare en PPT), on pourrait la diviser, 
#         # mais ici on privilégie l'unité de la slide.
        
#         tokens = _count_tokens(full_slide_text, cfg)
        
#         # Création du chunk
#         meta = {
#             "type": "pptx_slide",
#             "tokens_est": tokens,
#             "full_page_text": full_slide_text, # Référence totale pour l'IA
#             "slide_title": slide_title
#         }

#         chunks.append(Chunk(
#             id=f"{source}::slide_{page_no:03d}",
#             source=source,
#             section_title=slide_title,
#             text=full_slide_text,
#             page_no=page_no,
#             meta=meta
#         ))

#     return [c for c in chunks if len(c.text) >= cfg.min_chunk_chars]

# =========================
# Logique de Chunking PPTX Corrigée
# =========================
def chunk_pptx_from_res(res: Any, cfg: Optional[ChunkingConfig] = None) -> List[Chunk]:
    cfg = cfg or ChunkingConfig()
    
    # Récupération du markdown brut
    md_raw = res.artifacts.get("markdown_final") or res.artifacts.get("markdown") or ""
    source = getattr(res, "source", "unknown_pptx")
    
    # 1. Découpage par Page (Slide)
    raw_slides = md_raw.split("<!-- page_break -->")
    
    # --- CORRECTION DU DÉCALAGE ---
    # On nettoie et on ne garde que les segments qui contiennent du vrai texte
    # Cela évite que le split crée une "Page 1" vide si le doc commence par un saut de page.
    slides = [s.strip() for s in raw_slides if s.strip()]
    
    chunks: List[Chunk] = []
    in_excluded_zone = False

    for i, slide_content in enumerate(slides):
        # Maintenant, l'index 0 est forcément la première slide réelle.
        page_no = i + 1
        
        # 2. Identification du titre de la slide
        lines = slide_content.split('\n')
        slide_title = "Sans titre"
        for line in lines:
            clean_line = line.strip().lstrip('#').strip()
            if clean_line:
                slide_title = clean_line
                break

        # 3. Filtrage Appendice / Fin de doc
        if _is_marker_present(slide_title, cfg.appendix_markers) or \
           _is_marker_present(slide_title, cfg.bibliography_markers):
            in_excluded_zone = True
            
        if in_excluded_zone and cfg.stop_after_appendix_start:
            continue

        # 4. Nettoyage (Placeholder image -> Tag texte)
        full_slide_text = slide_content.replace("<!-- image_placeholder -->", "[Image]").strip()

        # 5. Calcul des tokens
        tokens = _count_tokens(full_slide_text, cfg)
        
        # 6. Création du chunk
        # On inclut le texte complet de la slide en meta pour la référence RAG
        meta = {
            "type": "pptx_slide",
            "tokens_est": tokens,
            "full_page_text": full_slide_text, 
            "slide_title": slide_title
        }

        chunks.append(Chunk(
            id=f"{source}::slide_{page_no:03d}",
            source=source,
            section_title=slide_title,
            text=full_slide_text,
            page_no=page_no,
            meta=meta
        ))

    return [c for c in chunks if len(c.text) >= cfg.min_chunk_chars]

# =========================
# Exemple d'Usage
# =========================
# factory = UniversalExtractorFactory(config=ExtractConfig())
# res = factory.extract("votre_presentation.pptx")
# chunks = chunk_pptx_from_res(res)