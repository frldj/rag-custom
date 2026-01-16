from __future__ import annotations

import base64
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

import requests
from PIL import Image

# ---- Docling ----
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem


# =============================================================================
# Optional notebook display imports (safe)
# =============================================================================
try:
    from IPython.display import Markdown as _IPyMarkdown, display as _ipy_display
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False
    _IPyMarkdown = None
    _ipy_display = None


# =============================================================================
# Ollama summarizers
# =============================================================================
TextSummarizer = Callable[[str, Dict[str, Any]], str]
ImageSummarizer = Callable[[Any, Dict[str, Any]], str]


def make_ollama_text_summarizer(
    model: str = "llama3.2:3b",
    endpoint: str = "http://localhost:11434/api/chat",
) -> TextSummarizer:
    def _summarize(text: str, metadata: Dict[str, Any]) -> str:
        prompt = (
            "Tu es un assistant d'ingestion de documents pour un système RAG.\n"
            "Résume le contenu suivant de façon concise, factuelle et exploitable pour le RAG.\n"
            "Réponse attendue : un paragraphe en français, sans préambule.\n\n"
            f"Métadonnées : {metadata}\n\n"
            f"Texte à résumer :\n{text}"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    return _summarize


def make_ollama_image_summarizer(
    model: str = "qwen2.5vl:3b",
    endpoint: str = "http://localhost:11434/api/chat",
    keep_alive: str = "30m",
    temperature: float = 0.0,
    max_side: int = 768,
    jpeg_quality: int = 58,
    timeout_connect: int = 10,
    timeout_read: int = 60,
) -> ImageSummarizer:
    def _img_to_b64_jpeg(pil_image: Any) -> str:
        if pil_image is None:
            return ""
        img = pil_image.copy()

        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img = img.convert("L")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _summarize_image(pil_image: Any, metadata: Dict[str, Any]) -> str:
        img_b64 = _img_to_b64_jpeg(pil_image)
        if not img_b64:
            return ""

        prompt = (
            "Tu es un assistant d'ingestion de documents pour un système RAG.\n"
            "Donne un résumé concis et descriptif de cette image telle qu'elle apparaît dans un rapport.\n"
            "Mentionne les informations clés (tendances, comparaisons, chiffres importants, etc.).\n"
            "Réponse attendue : un paragraphe en français, sans préambule.\n\n"
            f"Métadonnées : {metadata}\n"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_ctx": 512, "num_predict": 96},
        }

        resp = requests.post(endpoint, json=payload, timeout=(timeout_connect, timeout_read))
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    return _summarize_image


# =============================================================================
# Config & result
# =============================================================================
@dataclass
class ExtractConfig:
    artifacts_path: str = str(Path.home() / ".cache" / "docling" / "models")

    image_placeholder: str = "<!-- image_placeholder -->"
    table_placeholder: str = "<!-- table_placeholder -->"
    page_break_placeholder: str = "<!-- page_break -->"

    # Filtres images
    min_width: int = 300
    min_height: int = 200

    # Filtres tables
    min_rows: int = 2
    min_cols: int = 2
    min_cells: int = 6

    # Tables longues
    table_long_max_rows: int = 20
    table_long_max_md_chars: int = 2500
    table_preview_rows: int = 8

    # Docling pipeline PDF
    images_scale: float = 2.0
    generate_page_images: bool = True
    generate_picture_images: bool = True


@dataclass
class ExtractResult:
    source: str
    input_format: str
    artifacts: dict[str, Any]

    def display_markdown(self, key: str = "markdown_final") -> None:
        if not _HAS_IPY:
            raise RuntimeError("IPython.display indisponible (pas en notebook ?)")

        md_text = self.artifacts.get(key)
        if not isinstance(md_text, str):
            raise ValueError(
                f"Pas de markdown '{key}' dans artifacts. keys={list(self.artifacts.keys())}"
            )

        images_dict = self.artifacts.get("images", {})
        display_markdown_with_images(md_text, images_dict)


def display_markdown_with_images(md_text: str, images_dict: dict) -> None:
    if not _HAS_IPY:
        raise RuntimeError("IPython.display indisponible (pas en notebook ?)")

    parts = md_text.split("<!-- image:")
    _ipy_display(_IPyMarkdown(parts[0]))

    for part in parts[1:]:
        ref, rest = part.split(" -->", 1)
        ref = ref.strip()

        img = images_dict.get(ref, {}).get("pil")
        if img is not None:
            _ipy_display(img)

        _ipy_display(_IPyMarkdown(rest))


# =============================================================================
# Utils
# =============================================================================
def _is_text_item(item) -> bool:
    txt = getattr(item, "text", None)
    return isinstance(txt, str) and txt.strip() != ""


def _df_preview_markdown(df, n=8) -> str:
    head = df.head(n)
    more = max(len(df) - n, 0)
    md = head.to_markdown(index=False)
    if more > 0:
        md += f"\n\n> … +{more} lignes"
    return md


def word_to_pdf(docx_path: str | Path, output_dir: str | Path) -> Path:
    docx_path = Path(docx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(docx_path)],
        check=True,
    )
    return output_dir / (docx_path.stem + ".pdf")


# =============================================================================
# NEW: markdown "descriptifs only" (images & tables)
# =============================================================================
def build_descriptions_markdown(res: ExtractResult) -> str:
    images = res.artifacts.get("images", {}) or {}
    tables = res.artifacts.get("tables", {}) or {}

    parts: list[str] = []

    img_order = res.artifacts.get("pictures_in_doc_order") or list(images.keys())
    img_blocks: list[str] = []
    for ref in img_order:
        info = images.get(ref)
        if not info:
            continue
        summ = (info.get("summary") or "").strip()
        if not summ:
            continue
        img_blocks.append(
            f"<!-- image:{ref} -->\n"
            f"> **Descriptif image ({ref})** : {summ}\n"
        )
    if img_blocks:
        parts.append("## Descriptifs des images\n")
        parts.append("\n".join(img_blocks))

    # ✅ ordre stable des tables gardées
    tab_order = res.artifacts.get("tables_in_doc_order") or list(tables.keys())
    tab_blocks: list[str] = []
    for ref in tab_order:
        info = tables.get(ref)
        if not info:
            continue

        summ = (info.get("summary") or "").strip()
        if not summ:
            summ = (info.get("preview_md") or "").strip()
        if not summ:
            continue

        # num humain cohérent si présent
        idx = info.get("table_idx_in_doc")
        label = f"{idx}" if isinstance(idx, int) else ref

        tab_blocks.append(
            f"<!-- table:{ref} -->\n"
            f"> **Descriptif tableau ({label})** : {summ}\n"
        )
    if tab_blocks:
        parts.append("\n## Descriptifs des tableaux\n")
        parts.append("\n".join(tab_blocks))

    return "\n".join(parts).strip()


# =============================================================================
# Extractor interface
# =============================================================================
class Extractor(Protocol):
    def extract(self, path: str | Path) -> ExtractResult: ...


# =============================================================================
# PDF extractor (FIX ORDER FOR TABLES)
# =============================================================================
class PdfDoclingExtractor:
    def __init__(
        self,
        config: ExtractConfig,
        image_summarizer: Optional[ImageSummarizer] = None,
        text_summarizer: Optional[TextSummarizer] = None,
        swallow_summarizer_errors: bool = True,
    ):
        self.cfg = config
        self.image_summarizer = image_summarizer
        self.text_summarizer = text_summarizer
        self.swallow_summarizer_errors = swallow_summarizer_errors

    def _build_md_base_with_placeholders(self, doc) -> str:
        return doc.export_to_markdown(
            image_placeholder=self.cfg.image_placeholder,
            #table_placeholder=self.cfg.table_placeholder,
            page_break_placeholder=self.cfg.page_break_placeholder,
        )

    def extract(self, path: str | Path) -> ExtractResult:
        pdf_path = Path(path)
        source = str(pdf_path)

        pipeline_options = PdfPipelineOptions(artifacts_path=self.cfg.artifacts_path)
        pipeline_options.images_scale = self.cfg.images_scale
        pipeline_options.generate_page_images = self.cfg.generate_page_images
        pipeline_options.generate_picture_images = self.cfg.generate_picture_images

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        min_area = self.cfg.min_width * self.cfg.min_height

        images_kept: dict[str, dict] = {}
        tables_kept: dict[str, dict] = {}
        pictures_in_doc_order: list[str] = []
        tables_in_doc_order: list[str] = []

        table_idx_in_doc = 0  # ✅ numéro stable des tables gardées

        for item, _level in doc.iterate_items():
            if isinstance(item, PictureItem):
                pictures_in_doc_order.append(item.self_ref)

                pil_img = item.get_image(doc)
                if pil_img is None:
                    continue

                w, h = pil_img.size
                if w < self.cfg.min_width or h < self.cfg.min_height or (w * h) < min_area:
                    continue

                images_kept[item.self_ref] = {
                    "page_no": item.prov[0].page_no if item.prov else None,
                    "bbox": item.prov[0].bbox if item.prov else None,
                    "pil": pil_img,
                    "summary": None,
                    "summary_error": None,
                }

            elif isinstance(item, TableItem):
                # ✅ IMPORTANT: on filtre AVANT de pousser dans tables_in_doc_order
                df = item.export_to_dataframe(doc=doc)
                r, c = df.shape
                if r < self.cfg.min_rows or c < self.cfg.min_cols or (r * c) < self.cfg.min_cells:
                    continue

                ref = item.self_ref

                # ✅ ordre d'apparition des tables GARDÉES
                tables_in_doc_order.append(ref)

                tables_kept[ref] = {
                    "table_idx_in_doc": table_idx_in_doc,  # ✅ num cohérent 0..n
                    "page_no": item.prov[0].page_no if item.prov else None,
                    "bbox": item.prov[0].bbox if item.prov else None,
                    "dataframe": df,
                    "html": item.export_to_html(doc=doc),
                    "is_long": None,
                    "preview_md": None,
                    "summary": None,
                    "summary_error": None,
                }
                table_idx_in_doc += 1

        # --- Résumé images (optionnel) ---
        if self.image_summarizer is not None:
            for ref, info in images_kept.items():
                meta = {
                    "source": source,
                    "kind": "picture",
                    "ref": ref,
                    "page": info.get("page_no"),
                    "bbox": info.get("bbox"),
                }
                try:
                    info["summary"] = self.image_summarizer(info["pil"], meta)
                except Exception as e:
                    info["summary_error"] = repr(e)
                    if not self.swallow_summarizer_errors:
                        raise

        # --- Résumé tables longues (DANS L'ORDRE D'APPARITION DES TABLES GARDÉES) ---
        for ref in tables_in_doc_order:
            info = tables_kept[ref]
            df = info["dataframe"]
            full_md = df.to_markdown(index=False)

            is_long = (len(df) > self.cfg.table_long_max_rows) or (len(full_md) > self.cfg.table_long_max_md_chars)
            info["is_long"] = is_long
            info["preview_md"] = _df_preview_markdown(df, n=self.cfg.table_preview_rows)

            if is_long and self.text_summarizer is not None:
                meta = {
                    "source": source,
                    "kind": "table",
                    "ref": ref,
                    "table_idx_in_doc": info.get("table_idx_in_doc"),
                    "page": info.get("page_no"),
                    "bbox": info.get("bbox"),
                }
                try:
                    info["summary"] = self.text_summarizer(full_md, meta)
                except Exception as e:
                    info["summary_error"] = repr(e)
                    if not self.swallow_summarizer_errors:
                        raise

        # --- Markdown base + placeholders images ---
        md_base = self._build_md_base_with_placeholders(doc)
        md_final = md_base

        for ref in pictures_in_doc_order:
            if ref in images_kept:
                md_final = md_final.replace(self.cfg.image_placeholder, f"<!-- image:{ref} -->", 1)
            else:
                md_final = md_final.replace(self.cfg.image_placeholder, "", 1)

        has_table_placeholders = (self.cfg.table_placeholder in md_final)

        # ✅ Dans le markdown, on garde le bon "numéro" de table via table_idx_in_doc
        def _table_label(ref: str) -> str:
            idx = tables_kept.get(ref, {}).get("table_idx_in_doc")
            return f"{idx}" if isinstance(idx, int) else ref

        if has_table_placeholders:
            for ref in tables_in_doc_order:
                t = tables_kept[ref]
                label = _table_label(ref)
                tag = f"<!-- table:{ref} -->"

                if t.get("is_long"):
                    md_final = md_final.replace(
                        self.cfg.table_placeholder,
                        (
                            f"\n\n{tag}\n\n"
                            f"> **Résumé tableau ({label})** : {t['summary']}\n\n"
                            f"**Aperçu**\n\n{t['preview_md']}\n\n"
                        ),
                        1,
                    )
                else:
                    md_final = md_final.replace(
                        self.cfg.table_placeholder,
                        f"\n\n{tag}\n\n{t['dataframe'].to_markdown(index=False)}\n\n",
                        1,
                    )
            md_final = md_final.replace(self.cfg.table_placeholder, "")
        else:
            if tables_kept:
                md_final += "\n\n## Tables extraites\n"
                for ref in tables_in_doc_order:
                    t = tables_kept[ref]
                    label = _table_label(ref)
                    tag = f"<!-- table:{ref} -->"
                    if t.get("is_long"):
                        md_final += (
                            f"\n\n{tag}\n\n"
                            f"> **Résumé tableau ({label})** : {t['summary']}\n\n"
                            f"**Aperçu**\n\n{t['preview_md']}\n\n"
                        )
                    else:
                        md_final += f"\n\n{tag}\n\n{t['dataframe'].to_markdown(index=False)}\n\n"

        md_final = md_final.replace(self.cfg.image_placeholder, "").replace(self.cfg.table_placeholder, "")

        md_final_with_summaries = md_final
        for ref, info in images_kept.items():
            summ = info.get("summary")
            if summ:
                tag = f"<!-- image:{ref} -->"
                md_final_with_summaries = md_final_with_summaries.replace(
                    tag, tag + f"\n\n> **Résumé image** : {summ}\n", 1
                )

        artifacts: dict[str, Any] = {
            "markdown_base": md_base,
            "markdown_final": md_final_with_summaries,
            "images": images_kept,
            "tables": tables_kept,
            "pictures_in_doc_order": pictures_in_doc_order,
            "tables_in_doc_order": tables_in_doc_order,  # ✅ tables gardées uniquement
        }

        tmp_res = ExtractResult(source=source, input_format="pdf", artifacts=artifacts)
        artifacts["markdown_descriptions_only"] = build_descriptions_markdown(tmp_res)

        return ExtractResult(source=source, input_format="pdf", artifacts=artifacts)


# =============================================================================
# Word extractor (convert -> PDF -> PDF extractor)
# =============================================================================
class WordToPdfExtractor:
    def __init__(self, pdf_extractor: PdfDoclingExtractor, tmp_pdf_dir: str | Path = "pdfs"):
        self.pdf_extractor = pdf_extractor
        self.tmp_pdf_dir = Path(tmp_pdf_dir)

    def extract(self, path: str | Path) -> ExtractResult:
        docx_path = Path(path)
        pdf_path = word_to_pdf(docx_path, self.tmp_pdf_dir)

        res = self.pdf_extractor.extract(pdf_path)
        res.artifacts["converted_from"] = str(docx_path)
        res.artifacts["converted_pdf_path"] = str(pdf_path)
        res.input_format = "word->pdf"
        res.source = str(docx_path)
        return res


# =============================================================================
# PPTX extractor
# =============================================================================
class PptxDoclingExtractor:
    def __init__(self, config: ExtractConfig):
        self.cfg = config

    def extract(self, path: str | Path) -> ExtractResult:
        pptx_path = Path(path)
        converter = DocumentConverter(allowed_formats=[InputFormat.PPTX])
        result = converter.convert(pptx_path)
        doc = result.document

        artifacts = {
            "markdown": doc.export_to_markdown(
                page_break_placeholder=self.cfg.page_break_placeholder,
                image_placeholder=self.cfg.image_placeholder
            ),
            "docling_dict": doc.export_to_dict(),
        }
        return ExtractResult(source=str(pptx_path), input_format="pptx", artifacts=artifacts)


# =============================================================================
# Factory / Router
# =============================================================================
class UniversalExtractorFactory:
    def __init__(
        self,
        config: ExtractConfig,
        image_summarizer: Optional[ImageSummarizer] = None,
        text_summarizer: Optional[TextSummarizer] = None,
        tmp_pdf_dir: str | Path = "pdfs",
    ):
        self.cfg = config

        self._pdf = PdfDoclingExtractor(
            config=self.cfg,
            image_summarizer=image_summarizer,
            text_summarizer=text_summarizer,
            swallow_summarizer_errors=True,
        )
        self._word = WordToPdfExtractor(pdf_extractor=self._pdf, tmp_pdf_dir=tmp_pdf_dir)
        self._pptx = PptxDoclingExtractor(config=self.cfg)

        self._by_ext: dict[str, Extractor] = {
            ".pdf": self._pdf,
            ".docx": self._word,
            ".doc": self._word,
            ".pptx": self._pptx,
        }

    def extract(self, path: str | Path) -> ExtractResult:
        p = Path(path)
        ext = p.suffix.lower()
        extractor = self._by_ext.get(ext)
        if extractor is None:
            raise ValueError(f"Format non supporté: {ext} (fichier={p.name})")
        return extractor.extract(p)
