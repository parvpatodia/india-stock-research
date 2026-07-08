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
_TARGETS = ("net_profit", "operating_cash_flow", "total_debt", "equity", "ebit",
            "interest_expense")

_EXTRACT_SYSTEM = """You extract specific financial figures from an Indian company's annual \
report text. Return ONLY JSON, no prose.

For each requested figure give: the value exactly as printed (a number), the unit word used \
near it ("crore", "lakh", "million", "billion", or "absolute"), and "quote": the exact \
verbatim text (a substring copied from the SOURCE) the number came from. If a figure is not \
present in the SOURCE, set its value to null. NEVER infer, compute, or use outside knowledge; \
copy only what is written.

Also return "fiscal_year": the 4-digit calendar year in which the reported financial year
ended (e.g. 2026 for the year ended 31 March 2026).

Shape:
{"fiscal_year": 2026,
 "net_profit": {"value": 80775, "unit": "crore", "quote": "Profit for the year 80,775"},
 "operating_cash_flow": {"value": null, "unit": "", "quote": ""}, ...}
Figures: net_profit, operating_cash_flow, total_debt, equity, ebit, interest_expense.
"""

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

    A figure survives only if its value parses, its quote is a real substring of the source
    (grounding), and its unit is known. Value is converted to absolute rupees in code.
    """
    result: dict[str, float | None] = {}
    norm_source = _norm(source_text)
    if not isinstance(payload, dict):
        return {name: None for name in _TARGETS}
    for name in _TARGETS:
        obj = payload.get(name)
        if not isinstance(obj, dict):
            result[name] = None
            continue
        value = _num(obj.get("value"))
        quote = str(obj.get("quote", "")).strip()
        unit = str(obj.get("unit", "")).strip().lower()
        scale = _UNIT_SCALE.get(unit)
        if value is None or not quote or scale is None:
            result[name] = None
            continue
        if _norm(quote) not in norm_source:   # grounding: the quote must really be in the report
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


def detect_fiscal_year(text: str) -> int | None:
    """Best-effort report fiscal year (the calendar year the financial year ended)."""
    head = text[:8000]
    years: list[int] = []
    for pattern in _FY_PATTERNS:
        for m in pattern.finditer(head):
            y = _num_year(m.group(1))
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
            chunks = store.retrieve(
                "net profit revenue total debt borrowings equity EBIT operating income "
                "operating cash flow interest expense finance costs", k=self.retrieve_k)
            context = "\n\n".join(rc.chunk.text for rc in chunks)
            if context.strip():
                try:
                    raw = self.client.complete(_EXTRACT_SYSTEM, context, max_tokens=900)
                    payload = _parse_json(raw)
                except Exception:
                    payload = {}
                parsed = parse_extraction(payload, text)  # ground against the FULL report text
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
