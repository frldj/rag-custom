from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
@dataclass
class ChunkingConfig:
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 100  # (not used strictly, kept for compatibility)
    chars_per_token: int = 4

    drop_bibliography: bool = True
    drop_appendix: bool = True
    stop_after_appendix_start: bool = True

    min_chunk_chars: int = 200
    remove_docling_placeholders: bool = True

    # markers
    appendix_markers: Tuple[str, ...] = ("annexe", "annexes", "appendix", "appendice")
    bibliography_markers: Tuple[str, ...] = ("bibliographie", "references", "sources", "références", "bibliography")
    conclusion_markers: Tuple[str, ...] = ("conclusion",)
    stop_after_conclusion: bool = True

    allow_numbered_headings: bool = True
    allow_named_headings: bool = True


# =========================
# Tokenizer
# =========================
def _get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_TOKENIZER = _get_tokenizer()


def _count_tokens(text: str, cfg: ChunkingConfig) -> int:
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text))
    return max(1, len(text) // cfg.chars_per_token)


# =========================
# Marker detection
# =========================
def _is_marker_present(title: str, markers: Tuple[str, ...], *, allow_long_title: bool = False) -> bool:
    """
    Detect markers in a heading.
    - default behavior: marker must be present AND heading is short (<=2 words) after removing numbering
    - allow_long_title=True: marker presence is enough (used for Conclusion)
    """
    t = (title or "").lower().strip()
    t_clean = re.sub(r'^[\d\.]+\s*', '', t).strip()

    for m in markers:
        pattern = rf"\b{re.escape(m)}\b"
        if re.search(pattern, t_clean):
            if allow_long_title:
                return True
            words = t_clean.split()
            if len(words) <= 2:
                return True
    return False


def _is_conclusion_heading(title: str, cfg: ChunkingConfig) -> bool:
    # conclusion should match even if the title is long like:
    # "5. Conclusion, Limitation, and Future Work"
    return _is_marker_present(title, cfg.conclusion_markers, allow_long_title=True)


# =========================
# Helpers Structure (Extraction par blocs)
# =========================
_TABLE_RE = re.compile(r'((?:\n|^)\|.*\|(?:\n\|.*\|)+)', re.MULTILINE)


def _extract_blocks(text: str) -> List[Dict[str, Any]]:
    """Sépare une section en blocs (paragraphes/tables) avec leurs offsets (offset relatif à la section)."""
    blocks: List[Dict[str, Any]] = []
    last_idx = 0

    for match in _TABLE_RE.finditer(text):
        prev_text = text[last_idx:match.start()].strip()
        if prev_text:
            for p in re.split(r'\n{2,}', prev_text):
                p_clean = p.strip()
                if p_clean:
                    blocks.append({
                        "type": "paragraph",
                        "content": p_clean,
                        "offset": text.find(p_clean, last_idx),
                    })

        table_title = "Tableau sans titre"
        if blocks and blocks[-1]["type"] == "paragraph":
            lines = blocks[-1]["content"].split('\n')
            if re.search(r'^(Tableau|Table|Tab\.)\s*[:\d]', lines[-1].strip(), re.I):
                table_title = lines[-1].strip()
                rem = "\n".join(lines[:-1]).strip()
                if rem:
                    blocks[-1]["content"] = rem
                else:
                    blocks.pop()

        blocks.append({
            "type": "table",
            "content": match.group(0).strip(),
            "title": table_title,
            "offset": match.start(),
        })
        last_idx = match.end()

    rem = text[last_idx:].strip()
    if rem:
        for p in re.split(r'\n{2,}', rem):
            p_clean = p.strip()
            if p_clean:
                blocks.append({
                    "type": "paragraph",
                    "content": p_clean,
                    "offset": text.find(p_clean, last_idx),
                })

    return blocks


def _extract_sections(md: str, cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    """Extrait les sections avec l'index 'start' global."""
    markers: List[Tuple[int, int, str, int]] = []

    # Markdown headings
    for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", md, re.MULTILINE):
        markers.append((m.start(), m.end(), m.group(2).strip(), len(m.group(1))))

    # Numbered headings like "4.1. Title"
    if cfg.allow_numbered_headings:
        for m in re.finditer(
            r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>[A-ZÀ-Ÿ].{1,120})\s*$",
            md,
            re.MULTILINE,
        ):
            num = m.group("num")
            markers.append((m.start(), m.end(), f"{num} {m.group('title')}", num.count(".") + 1))

    if not markers:
        return [{"title": "document", "level": 1, "text": md, "start": 0}]

    markers.sort(key=lambda x: x[0])

    sections: List[Dict[str, Any]] = []
    for i, (start, _, title, level) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(md)
        sections.append({
            "title": title,
            "level": level,
            "text": md[start:end].strip(),
            "start": start,
        })
    return sections


def _add_section_paths(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stack: List[Tuple[int, str]] = []
    out: List[Dict[str, Any]] = []
    for sec in sections:
        lvl = sec["level"]
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, sec["title"]))
        sec["section_path"] = " > ".join([t for _, t in stack])
        out.append(sec)
    return out


# =========================
# Chunk Model
# =========================
@dataclass
class Chunk:
    id: str
    source: str
    section_path: str
    section_title: str
    text: str
    page_no: int
    meta: Dict[str, Any]


# =========================
# Main
# =========================
def chunk_from_extract_result(res: Any, cfg: Optional[ChunkingConfig] = None) -> List[Chunk]:
    cfg = cfg or ChunkingConfig()
    md = res.artifacts.get("markdown_final") or res.artifacts.get("markdown_base") or ""
    if not isinstance(md, str):
        md = str(md)

    # --- LOGIQUE DE PAGES ---
    pages_raw = md.split("<!-- page_break -->")
    page_map = {i + 1: p.replace("<!-- page_break -->", "").strip() for i, p in enumerate(pages_raw)}

    all_sections = _add_section_paths(_extract_sections(md, cfg))
    chunks: List[Chunk] = []
    source = getattr(res, "source", "unknown")

    # --- cutoff global après Conclusion ---
    conclusion_idx = None
    for i, s in enumerate(all_sections):
        if _is_conclusion_heading(s["title"], cfg):
            conclusion_idx = i
            break

    conclusion_cutoff_pos = None
    if conclusion_idx is not None and cfg.stop_after_conclusion:
        conclusion_cutoff_pos = all_sections[conclusion_idx + 1]["start"] if (conclusion_idx + 1) < len(all_sections) else len(md)

    in_excluded_zone = False

    for si, sec in enumerate(all_sections):
        sec_start_pos = sec["start"]
        path = sec["section_path"]
        title = sec["title"]

        # Stop global après la section Conclusion
        if conclusion_cutoff_pos is not None and sec_start_pos >= conclusion_cutoff_pos:
            break

        # --- Détection Exclusion (bibliography/appendix) ---
        is_bib = _is_marker_present(title, cfg.bibliography_markers)
        is_app = _is_marker_present(title, cfg.appendix_markers) or _is_marker_present(path.split(" > ")[0], cfg.appendix_markers)

        if (is_bib or is_app) and (si / max(1, len(all_sections)) > 0.4):
            if sec["level"] <= 2:
                in_excluded_zone = True

        if in_excluded_zone and cfg.stop_after_appendix_start:
            # "stop" (not just skip) once appendix/biblio zone begins
            break

        if (is_bib and cfg.drop_bibliography) or (is_app and cfg.drop_appendix):
            continue

        # --- Chunking par blocs ---
        blocks = _extract_blocks(sec["text"])

        # buffer stores tuples: (content, abs_pos, page_no)
        buffer: List[Tuple[str, int, int]] = []
        buffer_tokens = 0

        for bi, block in enumerate(blocks):
            rel_off = int(block.get("offset", 0) or 0)
            absolute_pos = sec_start_pos + rel_off

            # page number (1-indexed)
            block_page = md[:absolute_pos].count("<!-- page_break -->") + 1

            clean_content = (block["content"] or "").replace("<!-- page_break -->", "").strip()
            if not clean_content:
                continue

            # If we're past conclusion cutoff (very defensive)
            if conclusion_cutoff_pos is not None and absolute_pos >= conclusion_cutoff_pos:
                break

            if block["type"] == "table":
                # flush buffer before table
                if buffer:
                    chunks.append(_build_chunk_with_page(buffer, si, bi, source, sec, cfg, page_map))
                    buffer, buffer_tokens = [], 0

                table_text = f"SECTION: {path}\nTITRE TABLEAU: {block.get('title','Tableau sans titre')}\n\n{clean_content}"
                chunks.append(Chunk(
                    id=f"{source}::s{si}::t{bi}",
                    source=source,
                    section_path=path,
                    section_title=title,
                    text=table_text,
                    page_no=block_page,
                    meta={
                        "type": "table",
                        "abs_pos": absolute_pos,
                        "page_context": f"Contenu extrait de la page {block_page}",
                        "full_page_text": page_map.get(block_page, ""),
                    }
                ))
                continue

            # paragraph
            p_tokens = _count_tokens(clean_content, cfg)

            if not buffer:
                buffer = [(clean_content, absolute_pos, block_page)]
                buffer_tokens = p_tokens
                continue

            if buffer_tokens + p_tokens > cfg.chunk_size_tokens:
                # flush current buffer
                chunks.append(_build_chunk_with_page(buffer, si, bi, source, sec, cfg, page_map))

                # overlap: keep last paragraph only
                last = buffer[-1]
                buffer = [last, (clean_content, absolute_pos, block_page)]
                buffer_tokens = _count_tokens("\n\n".join([x[0] for x in buffer]), cfg)
            else:
                buffer.append((clean_content, absolute_pos, block_page))
                buffer_tokens += p_tokens

        # flush remaining buffer
        if buffer:
            chunks.append(_build_chunk_with_page(buffer, si, len(blocks), source, sec, cfg, page_map))

    # Final filtering
    out = [c for c in chunks if len(c.text) >= cfg.min_chunk_chars]

    # Hard filter after conclusion cutoff, if any
    if conclusion_cutoff_pos is not None:
        out = [c for c in out if int(c.meta.get("abs_pos", 0)) < conclusion_cutoff_pos]

    return out


def _build_chunk_with_page(
    buffer: List[Tuple[str, int, int]],
    si: int,
    bi: int,
    source: str,
    sec: Dict[str, Any],
    cfg: ChunkingConfig,
    page_map: Dict[int, str],
) -> Chunk:
    texts = [x[0] for x in buffer]
    abs_pos = int(buffer[0][1]) if buffer else int(sec.get("start", 0))
    page_no = int(buffer[0][2]) if buffer else 1

    full_text = "\n\n".join(texts)

    return Chunk(
        id=f"{source}::s{si}::c{si}_{page_no}_{bi}",
        source=source,
        section_path=sec["section_path"],
        section_title=sec["title"],
        text=full_text,
        page_no=page_no,
        meta={
            "type": "text",
            "abs_pos": abs_pos,
            "tokens_est": _count_tokens(full_text, cfg),
            "page_context": f"Contenu extrait de la page {page_no}",
            "full_page_text": page_map.get(page_no, ""),
        },
    )
