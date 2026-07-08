"""Annual-report figure extraction: a second, independent PRIMARY source.

The LLM reads the annual report and proposes figures, but it is never trusted on its own.
Each proposed figure must (1) carry a verbatim quote that actually appears in the report
(grounding: a hallucinated number has no real quote and is dropped), and (2) later agree with
another source (yfinance) in the cross-verification step. Unit words (crore/lakh) are converted
in CODE from the unit the model reports, not by the model's arithmetic. A wrong extraction
therefore fails to verify rather than becoming a trusted fact.
"""
from __future__ import annotations

import re
from typing import Callable

from ..llm.client import LLMClient, LiteLLMClient
from ..research.grounded_analyst import _parse_json
from ..research.grounding import DocumentStore
from .figure_sources import FRAMEWORK_FIGURES, FigureSource

# India-specific unit words -> multiplier to absolute rupees. Code does the math, not the LLM.
_UNIT_SCALE = {
    "crore": 1e7, "cr": 1e7, "lakh": 1e5, "lac": 1e5,
    "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
    "absolute": 1.0, "rupees": 1.0, "rs": 1.0, "": 1.0,
}

# Line items that actually appear in an annual report's statements.
# Raw items extracted from the report. EBIT is derived (PBT + interest), matching how the
# other sources define it, because annual reports do not print an "EBIT" line.
_EXTRACT_ITEMS = ("net_profit", "operating_cash_flow", "total_debt", "equity",
                  "profit_before_tax", "interest_expense")
# Framework figures this source exposes (EBIT derived, no standalone profit_before_tax).
_TARGETS = ("net_profit", "operating_cash_flow", "total_debt", "equity", "ebit",
            "interest_expense")

# Per-figure retrieval queries: pull the CONSOLIDATED statement region for each figure, so the
# model sees the actual statement lines rather than narrative text.
_FIGURE_QUERIES = {
    "net_profit": "consolidated profit for the year net profit attributable to owners after tax",
    "operating_cash_flow": "net cash generated from operating activities consolidated cash flow statement",
    "total_debt": "borrowings non-current current lease liabilities consolidated balance sheet",
    "equity": "total equity attributable to owners equity share capital other equity consolidated",
    "profit_before_tax": "profit before tax consolidated statement of profit and loss",
    "interest_expense": "finance costs interest expense consolidated statement of profit and loss",
}

_EXTRACT_SYSTEM = """You extract specific financial figures from an Indian company's annual \
report text. Return ONLY JSON, no prose.

Use the CONSOLIDATED figures (not standalone) and the LATEST reported financial year.

For each requested figure give: the value exactly as printed (a number), the unit word used \
near it ("crore", "lakh", "million", "billion", or "absolute"), and "quote": the exact \
verbatim text (a substring copied from the SOURCE) the number came from. If a figure is not \
present in the SOURCE, set its value to null. NEVER infer, compute, or use outside knowledge; \
copy only what is written.

Also return "fiscal_year": the 4-digit calendar year in which the reported financial year
ended (e.g. 2026 for the year ended 31 March 2026).

Shape:
{"fiscal_year": 2026,
 "net_profit": {"value": 26248, "unit": "crore", "quote": "Profit for the year 26,248"},
 "profit_before_tax": {"value": null, "unit": "", "quote": ""}, ...}
Figures: net_profit, operating_cash_flow, total_debt, equity, profit_before_tax, interest_expense.
"""

# Strict output schema so the model fills THESE keys (small models otherwise invent their own).
_FIGURE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": ["number", "null"]},
                   "unit": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["value", "unit"],
}
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"fiscal_year": {"type": ["integer", "null"]},
                   **{item: _FIGURE_SCHEMA for item in _EXTRACT_ITEMS}},
    "required": ["fiscal_year", *_EXTRACT_ITEMS],
}

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("₹", "").strip()
        f = float(value)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def parse_extraction(payload: dict, source_text: str) -> dict[str, float | None]:
    """Pure: turn the model's JSON into grounded, unit-converted values.

    A figure survives only if its value parses, its unit is known, AND it is grounded: either
    its quote is a real substring of the report, OR the reported number's digits actually appear
    in the report (numeric grounding, robust to garbled PDF-table text). A number absent from the
    report is rejected as a hallucination. Value is converted to absolute rupees in code.
    """
    result: dict[str, float | None] = {}
    norm_source = _norm(source_text)
    digits_source = re.sub(r"\D", "", source_text)  # all digits, comma/space-insensitive
    if not isinstance(payload, dict):
        return {name: None for name in _EXTRACT_ITEMS}
    for name in _EXTRACT_ITEMS:
        obj = payload.get(name)
        if not isinstance(obj, dict):
            result[name] = None
            continue
        value = _num(obj.get("value"))
        quote = str(obj.get("quote", "")).strip()
        unit = str(obj.get("unit", "")).strip().lower()
        scale = _UNIT_SCALE.get(unit)
        if value is None or scale is None:
            result[name] = None
            continue
        quote_grounded = bool(quote) and _norm(quote) in norm_source
        value_digits = re.sub(r"\D", "", str(obj.get("value")))
        numeric_grounded = len(value_digits) >= 3 and value_digits in digits_source
        if not (quote_grounded or numeric_grounded):
            result[name] = None
            continue
        result[name] = value * scale
    return result


_FY_PATTERNS = [
    re.compile(r"year\s+ended[^.]{0,40}?march[,\s]+.{0,6}?(20\d{2})", re.IGNORECASE),
    re.compile(r"march\s+31[,\s]+(20\d{2})", re.IGNORECASE),
    re.compile(r"31\s*(?:st)?\s*march[,\s]+(20\d{2})", re.IGNORECASE),
    re.compile(r"\bFY\s?(20\d{2})", re.IGNORECASE),
]


def _num_year(value) -> int | None:
    n = _num(value)
    if n is None:
        return None
    y = int(n)
    return y if 1990 <= y <= 2100 else None


_FY_RANGE = re.compile(r"20\d{2}\s*[-/]\s*(\d{2})\b")  # e.g. "2025-26" -> ends 2026


def detect_fiscal_year(text: str) -> int | None:
    """Best-effort report fiscal year (the calendar year the financial year ended)."""
    head = text[:20000]
    years: list[int] = []
    for pattern in _FY_PATTERNS:
        for m in pattern.finditer(head):
            y = _num_year(m.group(1))
            if y:
                years.append(y)
    for m in _FY_RANGE.finditer(head):        # "FY 2025-26" style -> the ending year 2026
        y = _num_year("20" + m.group(1))
        if y:
            years.append(y)
    return max(years) if years else None


class AnnualReportFigureSource(FigureSource):
    source_id = "annual_report"

    def __init__(self, text_provider: Callable[[str], str | None],
                 client: LLMClient | None = None, retrieve_k: int = 8):
        self.text_provider = text_provider   # symbol -> full annual-report text (or None)
        self.client = client or LiteLLMClient()
        self.retrieve_k = retrieve_k
        self._cache: dict[str, tuple[dict, int | None]] = {}

    def _extract(self, symbol: str) -> tuple[dict[str, float | None], int | None]:
        """Run the LLM extraction once per symbol (memoized). Returns (figures, fiscal_year)."""
        if symbol in self._cache:
            return self._cache[symbol]
        result: tuple[dict, int | None] = ({}, None)
        text = self.text_provider(symbol)
        if text and text.strip() and self.client.available:
            store = DocumentStore(words_per_chunk=180, overlap=30)
            store.add_document("annual_report", text)
            # Per-figure targeted retrieval: union the best chunks for each figure so the model
            # sees the actual consolidated statement lines, not just narrative text.
            best: dict[str, object] = {}
            queries = ["financial highlights consolidated results profit for the year revenue",
                       *_FIGURE_QUERIES.values()]
            for query in queries:
                for rc in store.retrieve(query, k=3):
                    prev = best.get(rc.chunk.chunk_id)
                    if prev is None or rc.score > prev.score:
                        best[rc.chunk.chunk_id] = rc
            chunks = sorted(best.values(), key=lambda r: -r.score)[:18]
            context = "\n\n".join(rc.chunk.text for rc in chunks)
            if context.strip():
                try:
                    raw = self.client.complete(_EXTRACT_SYSTEM, context, max_tokens=1200,
                                               json_schema=_EXTRACT_SCHEMA)
                    payload = _parse_json(raw)
                except Exception:
                    payload = {}
                parsed = parse_extraction(payload, text)  # ground against the FULL report text
                pbt, interest = parsed.get("profit_before_tax"), parsed.get("interest_expense")
                if pbt is not None and interest is not None:
                    parsed["ebit"] = pbt + interest      # EBIT derived, same as the other sources
                fy = _num_year(payload.get("fiscal_year")) or detect_fiscal_year(text)
                result = (parsed, fy)
        self._cache[symbol] = result
        return result

    def figures(self, symbol: str) -> dict[str, float | None]:
        out: dict[str, float | None] = {name: None for name in FRAMEWORK_FIGURES}
        parsed, _ = self._extract(symbol)
        for name in _TARGETS:
            if parsed.get(name) is not None:
                out[name] = parsed[name]
        return out

    def figures_by_year(self, symbol: str) -> dict[str, dict[int, float]]:
        # WHY: tag the extracted figures with the report's fiscal year so they align with the
        # SAME year from yfinance/Screener, letting the primary filing break a real tie. Without
        # a detected year we cannot align, so we return nothing rather than guess the year.
        parsed, fy = self._extract(symbol)
        if fy is None:
            return {}
        return {name: {fy: parsed[name]} for name in _TARGETS if parsed.get(name) is not None}
