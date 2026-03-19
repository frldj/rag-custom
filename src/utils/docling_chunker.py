# from __future__ import annotations

# import re
# import uuid
# import logging
# import gc
# import html
# import os
# from dataclasses import dataclass, field, asdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# # On essaie d'importer les libs OCR
# try:
#     from onnxtr.models import db_mobilenet_v3_large, crnn_mobilenet_v3_large
#     from ultralytics import YOLO
#     import docling_ocr_onnxtr
#     from docling_ocr_onnxtr import OnnxtrOcrOptions
#     HAS_ONNXTR = True
# except ImportError:
#     HAS_ONNXTR = False

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
# from docling.document_converter import (
#     DocumentConverter,
#     PdfFormatOption,
#     WordFormatOption,
#     PowerpointFormatOption,
#     MarkdownFormatOption,
#     HTMLFormatOption,
#     ExcelFormatOption,
# )
# from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
# from docling.chunking import HybridChunker

# logger = logging.getLogger(__name__)

# @dataclass
# class Chunk:
#     id: str
#     source: str
#     text: str
#     meta: Dict[str, Any] = field(default_factory=dict)

# class MultiFormatDoclingChunker:
#     def __init__(
#         self,
#         min_chars: int = 600,
#         max_chars: int = 1500,
#         device: AcceleratorDevice = AcceleratorDevice.AUTO,
#         artifacts_path: Optional[str] = None 
#     ):
#         self.min_chars = min_chars
#         self.max_chars = max_chars

#         # 1. CONFIGURATION DU PIPELINE
#         pipeline_options = PdfPipelineOptions()
        
#         # --- SOLUTION AU PROBLÈME ---
#         # On autorise explicitement les plugins externes (nécessaire pour onnxtr)
#         pipeline_options.allow_external_plugins = True 
        
#         pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
        
#         # On active l'OCR
#         pipeline_options.do_ocr = True
        
#         if HAS_ONNXTR:
#             # On configure les options OCR avec onnxtr
#             ocr_options = OnnxtrOcrOptions()
#             ocr_options.force_full_page_ocr = True
#             pipeline_options.ocr_options = ocr_options
#             logger.info("OCR : Moteur ONNXTR configuré (allow_external_plugins=True)")
#         else:
#             # Si onnxtr est absent, on laisse Docling utiliser rapidocr (qui est interne)
#             logger.warning("OCR : docling-ocr-onnxtr non installé, utilisation du moteur interne.")

#         pipeline_options.do_table_structure = True
#         pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
#         if artifacts_path:
#             pipeline_options.artifacts_path = artifacts_path

#         # 2. INITIALISATION DU CONVERTISSEUR
#         self.converter = DocumentConverter(
#             allowed_formats=[
#                 InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, 
#                 InputFormat.XLSX, InputFormat.HTML, InputFormat.MD
#             ],
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options, 
#                     backend=DoclingParseV4DocumentBackend
#                 ),
#                 InputFormat.DOCX: WordFormatOption(),
#                 InputFormat.PPTX: PowerpointFormatOption(),
#                 InputFormat.MD: MarkdownFormatOption(),
#                 InputFormat.HTML: HTMLFormatOption(),
#                 InputFormat.XLSX: ExcelFormatOption(),
#             },
#         )

#         self.docling_chunker = HybridChunker(
#             tokenizer="sentence-transformers/all-MiniLM-L6-v2", 
#             merge_peers=True,
#             max_tokens=self.max_chars 
#         )

#     def _deep_clean(self, text: str) -> str:
#         if not text: return ""
#         text = html.unescape(text)
#         # Supprime commentaires Docling mais garde les sauts de page pour le split initial si besoin
#         text = re.sub(r"<!--(?!(?: page_break )).*?-->", "", text)
#         text = text.replace("\u00a0", " ").replace("\xad", "")
#         text = re.sub(r"[ \t]+", " ", text)
        
#         lines = text.split('\n')
#         seen, unique_lines = set(), []
#         for line in lines:
#             clean_l = line.strip()
#             if not clean_l.startswith("|") and clean_l.lower() in seen and len(clean_l) > 15:
#                 continue
#             unique_lines.append(line)
#             if len(clean_l) > 15: seen.add(clean_l.lower())
#         return "\n".join(unique_lines).strip()

#     def _merge_small_chunks(self, raw_chunks: List[str]) -> List[str]:
#         merged = []
#         for c in raw_chunks:
#             c = self._deep_clean(c)
#             if not c or len(c) < 5: continue
#             if not merged:
#                 merged.append(c)
#             elif len(merged[-1]) < self.min_chars:
#                 merged[-1] = f"{merged[-1]}\n\n{c}"
#             else:
#                 merged.append(c)
#         if len(merged) > 1 and len(merged[-1]) < self.min_chars:
#             last = merged.pop()
#             merged[-1] = f"{merged[-1]}\n\n{last}"
#         return merged

#     def chunk_file(self, file_path: str | Path) -> List[Chunk]:
#         file_path = Path(file_path)
#         try:
#             result = self.converter.convert(file_path)
#             doc = result.document
            
#             # Le Chunker Hybride est très performant pour les tableaux
#             docling_chunks = list(self.docling_chunker.chunk(doc))
#             raw_segments = [self.docling_chunker.serialize(c) for c in docling_chunks]

#             if hasattr(result, "input") and hasattr(result.input, "_backend"):
#                 result.input._backend.unload()
#             del result
#             gc.collect()

#             final_texts = self._merge_small_chunks(raw_segments)
            
#             chunks = []
#             for i, text in enumerate(final_texts):
#                 chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}"))
#                 chunks.append(Chunk(
#                     id=chunk_id,
#                     source=str(file_path),
#                     text=text,
#                     meta={
#                         "char_count": len(text),
#                         "format": file_path.suffix.lower(),
#                         "index": i,
#                         "has_table": "|" in text and "---" in text
#                     }
#                 ))
#             return chunks

#         except Exception as e:
#             logger.error(f"Erreur chunking {file_path}: {e}")
#             raise

# def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
#     return [asdict(c) for c in chunks]

from __future__ import annotations

import re
import uuid
import logging
import gc
import html
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Imports Docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption,
)
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.chunking import HybridChunker

import os
from huggingface_hub import login

logger = logging.getLogger(__name__)

# Si la variable d'environnement est présente, on se connecte
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


PAGE_BREAK = "<!-- page_break -->"

@dataclass
class Chunk:
    id: str
    source: str
    text: str
    page_no: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class MultiFormatDoclingChunker:
    def __init__(
        self,
        strategy: Literal["recursive", "hybrid"] = "hybrid",
        min_chars: int = 600,
        max_chars: int = 1500,
        device: AcceleratorDevice = AcceleratorDevice.AUTO
    ):
        # Paramètres par défaut
        self.default_strategy = strategy
        self.default_min_chars = min_chars
        self.default_max_chars = max_chars
        
        # Configuration PIPELINE (VITESSE OPTIMISÉE)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
        pipeline_options.allow_external_plugins = True  # Nécessaire pour charger onnxtr
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # --- CONFIGURATION OCR(ONNXTR + MobileNetV3) ---
        pipeline_options.do_ocr = True
        try:
            from docling_ocr_onnxtr import OnnxtrOcrOptions
            

            ocr_opts = OnnxtrOcrOptions(
                det_arch="db_mobilenet_v3_large",    # Modèle de détection
                reco_arch="crnn_mobilenet_v3_large",  # Modèle de reconnaissance
                force_full_page_ocr=False            # Uniquement si la page n'est pas lisible nativement
            )
            
            pipeline_options.ocr_options = ocr_opts
            logger.info("OCR : Moteur ONNXTR configuré explicitement avec MobileNetV3.")
        except ImportError:
            logger.warning("OCR : docling-ocr-onnxtr non trouvé, passage en mode auto.")
            pass

        # Initialisation du convertisseur
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, 
                    backend=DoclingParseV4DocumentBackend
                ),
                InputFormat.PPTX: PowerpointFormatOption(),
                InputFormat.DOCX: WordFormatOption(),
            }
        )

    def _deep_clean(self, text: str) -> str:
        if not text: return ""
        text = html.unescape(text)
        text = re.sub(r"<!--(?!(?: page_break )).*?-->", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _merge_logic(self, segments: List[str], min_chars: int) -> List[str]:
        merged = []
        for seg in segments:
            seg = self._deep_clean(seg)
            if not seg: continue
            
            if not merged:
                merged.append(seg)
            elif len(merged[-1]) < min_chars:
                merged[-1] = f"{merged[-1]}\n\n{seg}"
            else:
                merged.append(seg)
        
        if len(merged) > 1 and len(merged[-1]) < min_chars:
            last = merged.pop()
            merged[-1] = f"{merged[-1]}\n\n{last}"
        return merged

    def chunk_file(
        self, 
        file_path: str | Path, 
        strategy: Optional[str] = None, 
        max_chars: Optional[int] = None, 
        min_chars: Optional[int] = None
    ) -> List[Chunk]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        current_strategy = strategy or self.default_strategy
        current_max = max_chars or self.default_max_chars
        current_min = min_chars or self.default_min_chars

        try:
            result = self.converter.convert(file_path)
            doc = result.document
            raw_segments = []

            if ext == ".pptx":
                # PPTX : Découpage par slide
                full_md = doc.export_to_markdown(page_break_placeholder=PAGE_BREAK)
                raw_segments = full_md.split(PAGE_BREAK)
            
            elif current_strategy == "hybrid":
                # STRATÉGIE HYBRIDE (DOCLING)
                hybrid_chunker = HybridChunker(
                    tokenizer=os.getenv("EMBEDDING_MODEL_NAME"), #"sentence-transformers/all-MiniLM-L6-v2",
                    merge_peers=True,
                    max_tokens=int(current_max / 3) 
                )
                chunks_gen = hybrid_chunker.chunk(doc)
                raw_segments = [hybrid_chunker.serialize(c) for c in chunks_gen]
            
            else:
                # STRATÉGIE RECURSIVE (LANGCHAIN)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=current_max,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    keep_separator=True
                )
                full_md = doc.export_to_markdown(page_break_placeholder=PAGE_BREAK)
                pages = full_md.split(PAGE_BREAK)
                for page in pages:
                    if not page.strip(): continue
                    raw_segments.extend(text_splitter.split_text(page))

            # Nettoyage mémoire
            if hasattr(result, "input") and hasattr(result.input, "_backend"):
                result.input._backend.unload()
            del result, doc
            gc.collect()

            final_texts = self._merge_logic(raw_segments, current_min)
            
            return [
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}")),
                    source=file_path.name,
                    text=t,
                    meta={"char_count": len(t), "format": ext, "strategy": current_strategy}
                ) for i, t in enumerate(final_texts)
            ]

        except Exception as e:
            logger.error(f"Erreur sur {file_path}: {e}")
            raise

def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in chunks]

# from __future__ import annotations

# import re
# import uuid
# import logging
# import gc
# import html
# from dataclasses import dataclass, field, asdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
# from docling.document_converter import (
#     DocumentConverter,
#     PdfFormatOption,
#     WordFormatOption,
#     PowerpointFormatOption,
#     MarkdownFormatOption,
#     HTMLFormatOption,
#     ExcelFormatOption,
# )
# from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

# # IMPORT DU CHUNKER NATIF DOCLING
# from docling.chunking import HybridChunker

# logger = logging.getLogger(__name__)

# @dataclass
# class Chunk:
#     id: str
#     source: str
#     text: str
#     meta: Dict[str, Any] = field(default_factory=dict)

# class MultiFormatDoclingChunker:
#     def __init__(
#         self,
#         min_chars: int = 600,
#         max_chars: int = 1500,
#         device: AcceleratorDevice = AcceleratorDevice.AUTO
#     ):
#         self.min_chars = min_chars
#         self.max_chars = max_chars

#         # Configuration du pipeline Docling
#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
#         pipeline_options.do_ocr = True 
#         pipeline_options.do_table_structure = True
#         pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

#         # 1. Initialisation du convertisseur
#         self.converter = DocumentConverter(
#             allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, 
#                              InputFormat.XLSX, InputFormat.HTML, InputFormat.MD],
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend),
#                 InputFormat.DOCX: WordFormatOption(),
#                 InputFormat.PPTX: PowerpointFormatOption(),
#                 InputFormat.MD: MarkdownFormatOption(),
#                 InputFormat.HTML: HTMLFormatOption(),
#                 InputFormat.XLSX: ExcelFormatOption(),
#             },
#         )

#         # 2. Initialisation du Hybrid Chunker (Natif Docling)
#         # Il gère nativement le respect des titres et des tableaux.
#         self.docling_chunker = HybridChunker(
#             tokenizer="sentence-transformers/all-MiniLM-L6-v2", # Tokenizer léger pour le calcul de taille
#             merge_peers=True,
#             max_tokens=self.max_chars # Ici max_chars est utilisé comme limite haute
#         )

#     def _deep_clean(self, text: str) -> str:
#         """Nettoyage final avant fusion/export"""
#         if not text: return ""
#         text = html.unescape(text)
#         text = re.sub(r"<!--.*?-->", "", text)
#         text = text.replace("\u00a0", " ").replace("\xad", "")
#         text = re.sub(r"[ \t]+", " ", text)
#         # Déduplication des lignes (important pour PPTX)
#         lines = text.split('\n')
#         seen = set()
#         unique_lines = []
#         for line in lines:
#             clean_l = line.strip()
#             if not clean_l.startswith("|") and clean_l.lower() in seen and len(clean_l) > 10:
#                 continue
#             unique_lines.append(line)
#             if len(clean_l) > 10: seen.add(clean_l.lower())
#         return "\n".join(unique_lines).strip()

#     def _merge_small_chunks(self, raw_chunks: List[str]) -> List[str]:
#         """Fusionne les morceaux structurels s'ils sont < 600 caractères"""
#         merged = []
#         for c in raw_chunks:
#             c = self._deep_clean(c)
#             if not c or len(c) < 5: continue
            
#             if not merged:
#                 merged.append(c)
#             elif len(merged[-1]) < self.min_chars:
#                 merged[-1] = f"{merged[-1]}\n\n{c}"
#             else:
#                 merged.append(c)
        
#         if len(merged) > 1 and len(merged[-1]) < self.min_chars:
#             last = merged.pop()
#             merged[-1] = f"{merged[-1]}\n\n{last}"
#         return merged

#     def chunk_file(self, file_path: str | Path) -> List[Chunk]:
#         file_path = Path(file_path)
#         try:
#             # 1. Conversion Docling
#             result = self.converter.convert(file_path)
#             doc = result.document
            
#             # 2. Chunking Hybride (Structurel)
#             # Cette méthode renvoie des objets 'BaseChunk' qui respectent les tableaux
#             docling_chunks = list(self.docling_chunker.chunk(doc))
            
#             # 3. Extraction du texte des chunks structurels
#             raw_segments = [self.docling_chunker.serialize(c) for c in docling_chunks]

#             # Nettoyage mémoire
#             if hasattr(result, "input") and hasattr(result.input, "_backend"):
#                 result.input._backend.unload()
#             del result
#             gc.collect()

#             # 4. Fusion selon votre règle des 600 caractères
#             final_texts = self._merge_small_chunks(raw_segments)
            
#             # 5. Création des objets finaux
#             chunks = []
#             for i, text in enumerate(final_texts):
#                 chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}"))
#                 chunks.append(Chunk(
#                     id=chunk_id,
#                     source=str(file_path),
#                     text=text,
#                     meta={
#                         "char_count": len(text),
#                         "format": file_path.suffix.lower(),
#                         "index": i
#                     }
#                 ))
#             return chunks

#         except Exception as e:
#             logger.error(f"Erreur chunking {file_path}: {e}")
#             raise

# def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
#     return [asdict(c) for c in chunks]

# from __future__ import annotations

# import re
# import uuid
# import logging
# import gc
# import os
# import html
# from dataclasses import dataclass, field, asdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import pandas as pd
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Imports Docling
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
# from docling.document_converter import (
#     DocumentConverter,
#     PdfFormatOption,
#     WordFormatOption,
#     PowerpointFormatOption,
#     MarkdownFormatOption,
#     HTMLFormatOption,
#     ExcelFormatOption,
# )
# from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

# logger = logging.getLogger(__name__)

# PAGE_BREAK = "<!-- page_break -->"

# @dataclass
# class Chunk:
#     id: str
#     source: str
#     text: str
#     page_no: Optional[int] = None
#     meta: Dict[str, Any] = field(default_factory=dict)

# # =============================================================================
# # Chunker Multi-Format avec Protection des Tableaux
# # =============================================================================

# class MultiFormatDoclingChunker:
#     def __init__(
#         self,
#         min_chars: int = 600,
#         max_chars: int = 1500,
#         device: AcceleratorDevice = AcceleratorDevice.AUTO
#     ):
#         self.min_chars = min_chars
        
#         # Splitter pour le texte normal uniquement
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=max_chars,
#             chunk_overlap=0,
#             separators=["\n\n", "\n", ". ", " ", ""],
#             keep_separator=True
#         )

#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
#         pipeline_options.do_ocr = True 
#         pipeline_options.do_table_structure = True
#         pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
#         pipeline_options.table_structure_options.do_cell_matching = True

#         self.converter = DocumentConverter(
#             allowed_formats=[
#                 InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, 
#                 InputFormat.XLSX, InputFormat.HTML, InputFormat.MD
#             ],
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options, 
#                     backend=DoclingParseV4DocumentBackend
#                 ),
#                 InputFormat.DOCX: WordFormatOption(),
#                 InputFormat.PPTX: PowerpointFormatOption(),
#                 InputFormat.MD: MarkdownFormatOption(),
#                 InputFormat.HTML: HTMLFormatOption(),
#                 InputFormat.XLSX: ExcelFormatOption(),
#             },
#         )

#     def _deep_clean(self, text: str) -> str:
#         if not text:
#             return ""
#         text = html.unescape(text)
#         text = re.sub(r"<!--(?!(?: page_break )).*?-->", "", text) # On garde page_break pour le split
#         text = text.replace("\u00a0", " ").replace("\xad", "")
#         text = re.sub(r"[ \t]+", " ", text)
        
#         # Nettoyage des lignes (sauf pour les tableaux où on garde la structure)
#         lines = text.split('\n')
#         seen_lines = set()
#         unique_lines = []
#         for line in lines:
#             clean_line = line.strip()
#             # Si ce n'est pas une ligne de tableau, on peut dédupliquer
#             if not clean_line.startswith("|"):
#                 if clean_line.lower() not in seen_lines or not clean_line:
#                     unique_lines.append(line)
#                     seen_lines.add(clean_line.lower())
#             else:
#                 unique_lines.append(line) # Garder les lignes de tableaux telles quelles
        
#         text = "\n".join(unique_lines)
#         text = re.sub(r"\n{3,}", "\n\n", text)
#         return text.strip()

#     def _isolate_tables_and_text(self, text: str) -> List[Tuple[bool, str]]:
#         """
#         Sépare le texte en blocs : (True, contenu_tableau) ou (False, texte_normal).
#         """
#         lines = text.splitlines()
#         blocks = []
#         current_block = []
#         in_table = False

#         for line in lines:
#             # Un tableau Markdown commence/finit généralement par |
#             is_table_line = line.strip().startswith("|") and line.strip().endswith("|")
            
#             if is_table_line:
#                 if not in_table:
#                     # On stocke le bloc de texte précédent
#                     if current_block:
#                         blocks.append((False, "\n".join(current_block)))
#                     current_block = []
#                     in_table = True
#                 current_block.append(line)
#             else:
#                 if in_table:
#                     # On stocke le tableau précédent
#                     if current_block:
#                         blocks.append((True, "\n".join(current_block)))
#                     current_block = []
#                     in_table = False
#                 current_block.append(line)
        
#         # Dernier bloc
#         if current_block:
#             blocks.append((in_table, "\n".join(current_block)))
        
#         return blocks

#     def _merge_small_chunks(self, raw_segments: List[str]) -> List[str]:
#         merged = []
#         for s in raw_segments:
#             s = self._deep_clean(s)
#             if not s or len(s) < 5:
#                 continue
            
#             if not merged:
#                 merged.append(s)
#             elif len(merged[-1]) < self.min_chars:
#                 merged[-1] = f"{merged[-1]}\n\n{s}"
#             else:
#                 merged.append(s)
        
#         if len(merged) > 1 and len(merged[-1]) < self.min_chars:
#             last = merged.pop()
#             merged[-1] = f"{merged[-1]}\n\n{last}"
            
#         return merged

#     def chunk_file(self, file_path: str | Path) -> List[Chunk]:
#         file_path = Path(file_path)
#         ext = file_path.suffix.lower()
        
#         try:
#             result = self.converter.convert(file_path)
#             doc = result.document
#             full_md = doc.export_to_markdown(page_break_placeholder=PAGE_BREAK)
            
#             raw_segments = []
            
#             # On découpe d'abord par page
#             pages = full_md.split(PAGE_BREAK)
            
#             for i, page_content in enumerate(pages):
#                 if not page_content.strip(): continue
                
#                 if ext == ".pptx":
#                     # PPTX : On traite la slide entière comme un segment
#                     raw_segments.append(page_content)
#                 else:
#                     # PDF / DOCX / MD : On isole les tableaux
#                     blocks = self._isolate_tables_and_text(page_content)
#                     for is_table, content in blocks:
#                         if is_table:
#                             # Le tableau est ajouté d'un seul bloc, sans split interne
#                             raw_segments.append(content)
#                         else:
#                             # Le texte normal est passé au splitter récursif
#                             split_text = self.text_splitter.split_text(content)
#                             raw_segments.extend(split_text)

#             if hasattr(result, "input") and hasattr(result.input, "_backend"):
#                 result.input._backend.unload()
#             del result
#             gc.collect()

#             # Phase de fusion (respecte min_chars)
#             # Note : un tableau restera entier même s'il dépasse max_chars
#             final_texts = self._merge_small_chunks(raw_segments)
            
#             chunks = []
#             for i, text in enumerate(final_texts):
#                 chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}"))
#                 chunks.append(Chunk(
#                     id=chunk_id,
#                     source=str(file_path),
#                     text=text,
#                     meta={
#                         "char_count": len(text),
#                         "format": ext,
#                         "index": i,
#                         "has_table": "|" in text and "---" in text # Info utile en meta
#                     }
#                 ))
#             return chunks

#         except Exception as e:
#             logger.error(f"Erreur chunking {file_path}: {e}")
#             raise

# def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
#     return [asdict(c) for c in chunks]

