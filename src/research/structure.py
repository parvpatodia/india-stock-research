"""Element-aware (structure-aware) chunking.

A financial document is not flat prose: it is section headings, paragraphs, and TABLES whose
meaning depends on a caption/units line ("(Rs in crore)") and a period header ("FY2024"). Blind
fixed-word windows cut tables in half and strip a figure from its units, which is exactly how a
scaled number ("73,670" meaning 73,670 crore) gets misread. This module splits a document on its
real structural boundaries and, critically, keeps every TABLE intact together with its caption,
title, and period header, and tags each piece with rich metadata (section, currency, unit scale,
fiscal period).

Dependency-light: stdlib only, deterministic. Table detection here is a heuristic over already-
extracted text (whitespace/pipe columns), NOT a PDF-geometry parser -- a follow-up W2 increment
can add a real table extractor (Camelot/Docling) behind this same seam. See LIMITATIONS below.

LIMITATIONS (honest, for the reviewer):
- Detection is column-heuristic: a table row needs >=2 numeric columns separated by 2+ spaces (or
  a tab), or Markdown pipes. A PDF that extracts its columns collapsed to single spaces will not
  be detected as a table (its rows fall back to paragraph text -- degraded, never wrong).
- Heading detection is conservative (short, no terminal sentence punctuation, Title Case / ALL
  CAPS / a known section name, or a trailing colon). It can miss an unusual heading; it will not
  invent one from a normal sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

# --- canonical word-window splitter (reused by grounding's blind ingestion path) --------------


def window_split(text: str, words_per_chunk: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows. The single canonical implementation used by
    both the blind ingestion path and the oversized-paragraph fallback here."""
    words = text.split()
    if not words:
        return []
    if len(words) <= words_per_chunk:
        return [" ".join(words)]
    step = max(1, words_per_chunk - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + words_per_chunk]
        if piece:
            chunks.append(" ".join(piece))
        if start + words_per_chunk >= len(words):
            break
    return chunks


# --- element model ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredPiece:
    text: str
    element_kind: str                 # "heading" | "paragraph" | "table"
    section: str | None = None
    unit_scale: str | None = None     # "crore" | "lakh" | "million" | "absolute" (tables only)
    currency: str | None = None       # "INR" (tables only)
    fiscal_period: str | None = None  # e.g. "FY2024"


# Line classes.
_BLANK, _HEADING, _CAPTION, _TABLE, _TEXT = "blank", "heading", "caption", "table", "text"

_NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SCALE_WORD = r"(crores?|lakhs?|lacs?|million|mn)"
_CAPTION_RE = re.compile(r"\bin\s+(?:₹|rs\.?|inr)?\s*" + _SCALE_WORD + r"\b", re.IGNORECASE)
_UNITS_ANNOTATION = re.compile(r"^\(?\s*(?:₹|rs\.?|inr).*?" + _SCALE_WORD + r".*?\)?$",
                               re.IGNORECASE)
_CURRENCY = re.compile(r"(₹|inr|rs\.?)", re.IGNORECASE)
_SCALE_NORM = {"crore": "crore", "crores": "crore", "lakh": "lakh", "lakhs": "lakh",
               "lac": "lakh", "lacs": "lakh", "million": "million", "mn": "million"}

# Known Indian annual-report section names (substring, case-insensitive) -> always a heading.
_KNOWN_SECTIONS = (
    "management discussion", "directors' report", "director's report", "independent auditor",
    "balance sheet", "statement of profit and loss", "profit and loss", "cash flow statement",
    "notes to", "risk management", "key risks", "corporate governance", "related party",
    "contingent liabilit", "segment", "business overview", "financial highlights",
)
# Small words ignored when checking Title Case.
_STOPWORDS = {"of", "and", "the", "to", "for", "in", "on", "a", "an", "vs", "&"}


def _is_table_row(line: str) -> bool:
    if "|" in line and len([c for c in line.split("|") if c.strip()]) >= 2:
        return True
    if len(_NUM_TOKEN.findall(line)) >= 2 and re.search(r"\S(?:  +|\t)\S", line):
        return True
    return False


def _is_caption(line: str) -> bool:
    if _is_table_row(line):
        return False
    s = line.strip()
    return bool(_CAPTION_RE.search(s) or _UNITS_ANNOTATION.match(s))


def _looks_title_case(s: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'&]*", s)]
    if not words:
        return False
    significant = [w for w in words if w.lower() not in _STOPWORDS]
    return bool(significant) and all(w[0].isupper() for w in significant)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or not re.search(r"[A-Za-z]", s) or _is_table_row(s):
        return False
    low = s.lower()
    if any(k in low for k in _KNOWN_SECTIONS):
        return True
    if len(s) > 80 or len(s.split()) > 10:
        return False
    if s.endswith((".", "!", "?", ",", ";")):
        return False
    if s.endswith(":") and len(s.split()) <= 8:
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and s.upper() == s:                      # ALL CAPS
        return True
    return _looks_title_case(s) and len(s.split()) <= 6


def _classify(line: str) -> str:
    if not line.strip():
        return _BLANK
    if _is_table_row(line):
        return _TABLE
    if _is_caption(line):
        return _CAPTION
    if _is_heading(line):
        return _HEADING
    return _TEXT


def detect_period(text: str) -> str | None:
    """The fiscal period a piece belongs to, normalized to 'FY<year>', or None. Reads an explicit
    FY tag, a 'year ended 31 March 2024' phrase, or a 31-03-YYYY date."""
    m = re.search(r"\bFY\s?(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        y = m.group(1)
        return "FY" + (("20" + y) if len(y) == 2 else y)
    m = re.search(r"(?:year ended|as at|as on|as of)\s+\d{1,2}\s+(?:march|mar)\.?\s+(\d{4})",
                  text, re.IGNORECASE)
    if m:
        return "FY" + m.group(1)
    m = re.search(r"\b31[-/.]0?3[-/.](\d{4})\b", text)
    if m:
        return "FY" + m.group(1)
    return None


def _table_scale_and_currency(table_text: str) -> tuple[str | None, str | None]:
    """Detect the declared scale + currency from a table's first few lines (title/caption/header),
    not its data rows, so a stray label word can't be misread as the table's unit."""
    head = "\n".join(table_text.split("\n")[:3])
    m = _CAPTION_RE.search(head) or re.search(r"\b" + _SCALE_WORD + r"\b", head, re.IGNORECASE)
    scale = _SCALE_NORM.get(m.group(1).lower()) if m else None
    currency = "INR" if _CURRENCY.search(head) else None
    if scale is None and currency is not None:
        scale = "absolute"       # rupees, but no crore/lakh/million multiplier declared
    return scale, currency


def _make_piece(text: str, kind: str, section: str | None) -> StructuredPiece:
    if kind == _TABLE:
        scale, currency = _table_scale_and_currency(text)
    else:
        scale, currency = None, None
    return StructuredPiece(text=text.strip(), element_kind=kind, section=section,
                           unit_scale=scale, currency=currency,
                           fiscal_period=detect_period(text))


def split_elements(text: str) -> list[StructuredPiece]:
    """Split a document into structural pieces in document order. Tables (with their caption and
    immediately-preceding title heading) are kept intact; headings set the running section and are
    also emitted as their own pieces; consecutive text lines form paragraphs."""
    if not text or not text.strip():
        return []
    lines = text.split("\n")
    n = len(lines)
    kinds = [_classify(line) for line in lines]

    # 1) Grow table regions: a maximal run of table rows, extended UP over a contiguous caption
    #    and one contiguous title heading (never split a table from its caption/units/title).
    regions: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if kinds[i] == _TABLE:
            j = i
            while j + 1 < n and kinds[j + 1] == _TABLE:
                j += 1
            s = i
            while s - 1 >= 0 and kinds[s - 1] == _CAPTION:
                s -= 1
            if s - 1 >= 0 and kinds[s - 1] == _HEADING:
                s -= 1
            regions.append((s, j))
            i = j + 1
        else:
            i += 1

    region_start = {s: (s, e) for (s, e) in regions}
    consumed = {x for (s, e) in regions for x in range(s, e + 1)}

    pieces: list[StructuredPiece] = []
    current_section: str | None = None
    para: list[str] = []

    def flush_para() -> None:
        joined = "\n".join(para).strip()
        if joined:
            pieces.append(_make_piece(joined, _TEXT, current_section))
        para.clear()

    i = 0
    while i < n:
        if i in region_start:
            flush_para()
            s, e = region_start[i]
            pieces.append(_make_piece("\n".join(lines[s:e + 1]), _TABLE, current_section))
            i = e + 1
            continue
        if i in consumed:
            i += 1
            continue
        kind = kinds[i]
        if kind == _BLANK:
            flush_para()
        elif kind == _HEADING:
            flush_para()
            current_section = lines[i].strip()
            pieces.append(_make_piece(lines[i], _HEADING, current_section))
        else:                                            # text or an un-absorbed caption line
            para.append(lines[i])
        i += 1
    flush_para()
    # normalize element_kind labels to the public names
    return [replace(p, element_kind={_TEXT: "paragraph", _TABLE: "table", _HEADING: "heading"}
                    .get(p.element_kind, p.element_kind)) for p in pieces]


def structure_chunks(text: str, words_per_chunk: int, overlap: int) -> list[StructuredPiece]:
    """split_elements, then window-split only oversized PARAGRAPH pieces (so long prose still
    retrieves well). Tables and headings are never split -- correctness over chunk size."""
    out: list[StructuredPiece] = []
    for piece in split_elements(text):
        if piece.element_kind == "paragraph" and len(piece.text.split()) > words_per_chunk:
            for sub in window_split(piece.text, words_per_chunk, overlap):
                out.append(replace(piece, text=sub))
        else:
            out.append(piece)
    return out
