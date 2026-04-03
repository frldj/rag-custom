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
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from huggingface_hub import login
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
from docling_ocr_onnxtr import OnnxtrOcrOptions
from onnxtr.models import db_mobilenet_v3_large, crnn_mobilenet_v3_large
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.chunking import HybridChunker

# Imports pour la sérialisation avancée des tableaux
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

import platform

try:
    if platform.system() == "Darwin":
        import ocrmac
    else:
        ocrmac = None
except ImportError:
    ocrmac = None

# Plus loin dans ton code
if ocrmac:
    # Utiliser l'OCR Mac
    pass
else:
    # Utiliser Docling standard (qui tourne sur ONNX/CPU sous Linux)
    pass

logger = logging.getLogger(__name__)

import logging
# On réduit au silence le warning spécifique des transformers/tokenizers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Connexion HF si nécessaire
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

@dataclass
class Chunk:
    id: str
    source: str
    text: str  # Texte enrichi (avec préfixe E5 et breadcrumbs)
    page_no: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Provider pour forcer le rendu Markdown des tableaux
class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(), 
        )

class MultiFormatDoclingChunker:
    def __init__(
        self,
        strategy: Literal["recursive", "hybrid"] = "hybrid",
        min_chars: int = 200, # Seuil bas pour ne pas perdre de données
        max_chars: int = 1500,
        device: AcceleratorDevice = AcceleratorDevice.AUTO
    ):
        self.default_strategy = strategy
        self.default_min_chars = min_chars
        self.default_max_chars = max_chars
        
        # 1. CONFIGURATION DU PIPELINE HAUTE PRÉCISION
        pipeline_options = PdfPipelineOptions()
        pipeline_options.allow_external_plugins = True
        pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.do_ocr = True # Indispensable pour les PDF scannés

        try:
            from docling_ocr_onnxtr import OnnxtrOcrOptions
            ocr_options = OnnxtrOcrOptions()
            
            # Au lieu d'assigner des objets (qui font planter Pydantic), 
            # on passe les noms des architectures si le plugin le permet, 
            # ou on laisse par défaut en s'assurant que le plugin est chargé.
            
            # Si tu veux vraiment MobileNet V3 Large, la version propre est :
            # ocr_options.force_full_page_ocr = True # Exemple d'option valide
            
            pipeline_options.ocr_options = ocr_options
            logger.info("OCR : Plugin ONNXTR activé.")
        except Exception as e:
            logger.warning(f"Erreur configuration OCR : {e}")

        pdf_options = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseV4DocumentBackend
        )

        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
            format_options={
                InputFormat.PDF: pdf_options,
                InputFormat.PPTX: PowerpointFormatOption(),
                InputFormat.DOCX: WordFormatOption(),
            }
        )

    def chunk_file(
        self, 
        file_path: str | Path, 
        strategy: Optional[str] = None,
        max_chars: Optional[int] = None,
        min_chars: Optional[int] = None,
        summary: Optional[str] = None,  
        doc_date: Optional[str] = None
    ) -> List[Chunk]:
        file_path = Path(file_path)
        current_strategy = strategy or self.default_strategy
        
        # Note : target_max pourrait être converti en tokens pour l'hybrid 
        # (ex: target_max // 3) si tu veux que l'argument max_chars pilote l'hybrid.
        
        try:
            result = self.converter.convert(file_path)
            doc = result.document

            doc_meta = {
                "file_name": file_path.name,
                "doc_summary": summary or "Non disponible",
                "doc_date": doc_date or "unknown",
                "file_ext": file_path.suffix.lower()
            }

            if current_strategy == "hybrid":
                tokenizer_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
                
                # On utilise max_chars pour estimer max_tokens si fourni, 
                # sinon 420 (valeur optimale e5)
                #calculated_max_tokens = (max_chars // 3) if max_chars else 420
                target_tokens = 448

                hybrid_chunker = HybridChunker(
                    tokenizer=tokenizer_name,
                    max_tokens=target_tokens, #min(calculated_max_tokens, 450), # On ne dépasse pas la limite e5
                    merge_peers=True,
                    serializer_provider=MDTableSerializerProvider()
                )
                
                chunks_gen = hybrid_chunker.chunk(doc) 
                final_chunks = []

                for i, c in enumerate(chunks_gen):
                    raw_text = hybrid_chunker.serialize(c)
                    breadcrumb = " > ".join(c.meta.headings) if c.meta.headings else ""
                    
                    if breadcrumb:
                        enriched_text = f"passage: {breadcrumb}\n{raw_text}"
                    else:
                        enriched_text = f"passage: {raw_text}"

                    is_table = any(getattr(item, "label", "") == "Table" for item in c.meta.doc_items)

                    try:
                        token_count = len(hybrid_chunker.tokenizer.tokenizer.encode(enriched_text))
                    except Exception:
                        token_count = 0 
                    
                    page_no = None
                    try:
                        if c.meta.doc_items:
                            page_no = c.meta.doc_items[0].prov[0].page_no
                    except (AttributeError, IndexError):
                        pass

                    # On utilise min_chars ici pour filtrer les chunks trop petits
                    limit_min = min_chars or self.default_min_chars
                    if len(raw_text) < (limit_min // 4) and not is_table:
                        continue

                    final_chunks.append(Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}")),
                    source=file_path.name,
                    text=enriched_text,
                    page_no=page_no,
                    meta={
                        **doc_meta, # <--- On injecte les infos globales
                        "raw_content": raw_text,
                        "breadcrumb": breadcrumb,
                        "token_count": token_count,
                        "is_table": is_table,
                        }
                    ))
                return final_chunks
            
            else:
                # Logique RECURSIVE par défaut
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chars or self.default_max_chars,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                full_md = doc.export_to_markdown()
                raw_segments = text_splitter.split_text(full_md)
                
                return [
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}|{i}")),
                    source=file_path.name,
                    text=f"passage: {t}",
                    meta={**doc_meta, "char_count": len(t), "strategy": "recursive"} # <--- On injecte ici aussi
                ) for i, t in enumerate(raw_segments)
                ]

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}: {e}")
            raise
        finally:
            if 'result' in locals():
                del result
            gc.collect()

def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in chunks]


