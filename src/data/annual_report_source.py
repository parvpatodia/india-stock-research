"""Annual-report figure extraction: a second, independent PRIMARY source.

The LLM reads the annual report and proposes figures, but it is never trusted on its own.
Each proposed figure must (1) carry a verbatim quote that actually appears in the report
(grounding: a hallucinated number has no real quote and is dropped), and (2) later agree with
another source (yfinance) in the cross-verification step. Unit words (crore/lakh) are converted
in CODE from the unit the model reports, not by the model's arithmetic. A wrong extraction
therefore fails to verify rather than becoming a trusted fact.
"""
from __future__ import annotations

import math
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
                  "profit_before_tax", "interest_expense", "total_assets")
# Framework figures this source exposes (EBIT derived, no standalone profit_before_tax).
_TARGETS = ("net_profit", "operating_cash_flow", "total_debt", "equity", "ebit",
            "interest_expense", "total_assets")

# Per-figure retrieval queries: pull the CONSOLIDATED statement region for each figure, so the
# model sees the actual statement lines rather than narrative text.
_FIGURE_QUERIES = {
    "net_profit": "consolidated profit for the year net profit attributable to owners after tax",
    "operating_cash_flow": "net cash generated from operating activities consolidated cash flow statement",
    "total_debt": "borrowings non-current current lease liabilities consolidated balance sheet",
    "equity": "total equity attributable to owners equity share capital other equity consolidated",
    "profit_before_tax": "profit before tax consolidated statement of profit and loss",
    "interest_expense": "finance costs interest expense consolidated statement of profit and loss",
    "total_assets": "total assets consolidated balance sheet",
}

_EXTRACT_SYSTEM = """You extract specific financial figures from an Indian company's annual \
report text. Return ONLY JSON, no prose.

Use the CONSOLIDATED figures (not standalone) and the LATEST reported financial year.

For each requested figure give: the value exactly as printed (a number), the unit word used \
near it ("crore", "lakh", "million", "billion", or "absolute"), "quote": the exact verbatim \
text (a substring copied from the SOURCE) the number came from, and "fiscal_year": the 4-digit \
calendar year THIS SPECIFIC value's own financial year ended in. A report often shows the \
CURRENT year next to a PRIOR year for comparison (e.g. "26,248 crore, compared to 22,825 crore \
last year") -- fiscal_year must match whichever year the VALUE you picked actually belongs to, \
not the report's cover year in general. If a figure is not present in the SOURCE, set its value \
to null. NEVER infer, compute, or use outside knowledge; copy only what is written.

Also return a top-level "fiscal_year": the 4-digit calendar year in which the report's own
LATEST financial year ended (e.g. 2026 for the year ended 31 March 2026).

Shape:
{"fiscal_year": 2026,
 "net_profit": {"value": 26248, "unit": "crore", "quote": "Profit for the year 26,248", "fiscal_year": 2026},
 "profit_before_tax": {"value": null, "unit": "", "quote": "", "fiscal_year": null}, ...}
Figures: net_profit, operating_cash_flow, total_debt, equity, profit_before_tax,
interest_expense, total_assets.
"""

# Strict output schema so the model fills THESE keys (small models otherwise invent their own).
_FIGURE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": ["number", "null"]},
                   "unit": {"type": "string"}, "quote": {"type": "string"},
                   "fiscal_year": {"type": ["integer", "null"]}},
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
    """Parse to a finite float, or None. WHY (real money, HIGH severity): Python's json module
    accepts the non-standard "Infinity"/"-Infinity"/"NaN" tokens by default, so a model's raw
    JSON response can legitimately parse into float('inf'). A later int(value) call on an
    infinite value raises OverflowError with no guard anywhere in the call chain -- rejecting
    non-finite values HERE, at the single point every numeric value and year passes through,
    closes that for every caller at once (value parsing here, and _num_year's year parsing,
    which reuses this same function) instead of guarding each call site separately."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("₹", "").strip()
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def parse_extraction(payload: dict, source_text: str) -> dict[str, float | None]:
    """Pure: turn the model's JSON into grounded, unit-converted values.

    A figure survives only if its value parses, its unit is known, AND it is grounded: either
    its quote is a real substring of the report AND itself contains the claimed value's digits,
    OR the reported number's digits actually appear anywhere in the report (numeric grounding,
    robust to garbled PDF-table text). A number absent from the report is rejected as a
    hallucination. Value is converted to absolute rupees in code.

    WHY (real money, HIGH severity): a quote being a real substring of the report is not enough
    on its own -- a model could attach a genuine, numberless narrative sentence (e.g. management
    commentary with no digits at all) to a completely fabricated value, and the old check would
    accept it via quote-grounding alone, since it never verified the quote actually CONTAINS the
    claimed number. Requiring the value's digits to appear WITHIN the quote itself closes that
    gap while still accepting the legitimate case (a real quote that genuinely states the figure).

    WHY (real money, HIGH severity, cross-year mislabeling): a report routinely shows the current
    AND a prior year side by side (e.g. "Net profit for FY2026 was 26,248 crore, compared to
    22,825 crore in FY2025"). Both numbers are genuinely real and appear in the source, so
    neither quote- nor numeric-grounding above can tell a real PRIOR-year figure apart from the
    CURRENT year's figure a field is actually supposed to report. When the model provides an
    optional per-figure "fiscal_year", it is cross-checked against detect_fiscal_year(source_text)
    -- an INDEPENDENT, deterministic, pattern-based reference, not the model's own top-level
    fiscal_year claim (a model confused enough to mislabel one figure's year could just as easily
    be wrong about that too) -- and a disagreement rejects the figure. Best-effort, not a
    complete fix: a model that omits fiscal_year for a mislabeled figure, or confidently
    mislabels both the value and its year the same wrong way, is not caught by this check alone.
    """
    result: dict[str, float | None] = {}
    norm_source = _norm(source_text)
    # WHY: the reported value must equal a real NUMBER TOKEN in the report, not just appear as a
    # substring of the report's concatenated digits. The old substring check was spoofable, e.g.
    # a hallucinated "26248" matches inside "...1,262...48..." -> "1262"+"48" = "...126248...".
    # Matching whole tokens (grouping commas + decimal point stripped to digits) closes that hole.
    report_num_digits = {re.sub(r"\D", "", tok) for tok in _REPORT_NUM.findall(source_text)}
    report_num_digits.discard("")
    if not isinstance(payload, dict):
        return {name: None for name in _EXTRACT_ITEMS}
    expected_fy = detect_fiscal_year(source_text)
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
        figure_year = _num_year(obj.get("fiscal_year"))
        if expected_fy is not None and figure_year is not None and figure_year != expected_fy:
            result[name] = None
            continue
        # WHY: build the digit string from the PARSED value, not the raw JSON token -- the
        # extraction schema types "value" as a generic JSON number, so a model can legitimately
        # emit a whole-number figure as 80775.0 rather than 80775. str(80775.0) is "80775.0",
        # which strips to "807750" (a spurious extra trailing zero from the ".0"), silently
        # rejecting perfectly legitimate data. Normalizing a whole float via int() first avoids
        # that without affecting genuinely fractional values.
        value_digits = re.sub(r"\D", "", str(int(value) if value == int(value) else value))
        # WHY: match whole NUMBER TOKENS within the quote (same spoof-resistant approach as
        # report_num_digits above), not a substring of the quote's concatenated digits -- a
        # quote with multiple numbers could otherwise coincidentally spoof an unrelated value
        # the same way the full-report substring check could before it was fixed.
        quote_number_digits = {re.sub(r"\D", "", tok) for tok in _REPORT_NUM.findall(quote)}
        quote_grounded = (bool(quote) and _norm(quote) in norm_source
                         and bool(value_digits) and value_digits in quote_number_digits)
        numeric_grounded = len(value_digits) >= 3 and value_digits in report_num_digits
        if not (quote_grounded or numeric_grounded):
            result[name] = None
            continue
        result[name] = value * scale
    return result


# A number as written in a report: leading digit, optional grouping, optional decimals.
_REPORT_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


# WHY (real money, adversarial-review finding): these three patterns are structurally
# unambiguous -- "the year ended 31 March YYYY" is a legally-required statement in every Indian
# annual report's financial statements, and by its own grammar can only ever describe a
# COMPLETED period, never a future target. Kept separate from the ambiguous "bare FY mention"
# patterns below (see detect_fiscal_year).
_FY_PATTERNS_RELIABLE = [
    re.compile(r"year\s+ended[^.]{0,40}?march[,\s]+.{0,6}?(20\d{2})", re.IGNORECASE),
    re.compile(r"march\s+31[,\s]+(20\d{2})", re.IGNORECASE),
    re.compile(r"31\s*(?:st)?\s*march[,\s]+(20\d{2})", re.IGNORECASE),
]
# WHY: a bare "FY2026" or short "FY26" mention is genuinely ambiguous -- confirmed live that a
# forward-looking sentence ("management expects FY2028 to be strong") uses the exact same shape
# to describe a FUTURE target, not the report's own year. Only used as a fallback (see
# detect_fiscal_year) when no reliable cover-page-style statement was found at all.
_FY_PATTERNS_FALLBACK = [
    re.compile(r"\bFY\s?(20\d{2})", re.IGNORECASE),
]
_FY_SHORT = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)          # e.g. "FY26", "FY'26"
_FY_RANGE = re.compile(r"20\d{2}\s*[-/]\s*(\d{2})\b")                  # e.g. "2025-26" -> ends 2026


def _num_year(value) -> int | None:
    n = _num(value)
    if n is None:
        return None
    y = int(n)
    return y if 1990 <= y <= 2100 else None


def _years_matching(patterns: list, head: str) -> list[int]:
    years: list[int] = []
    for pattern in patterns:
        for m in pattern.finditer(head):
            y = _num_year(m.group(1))
            if y:
                years.append(y)
    return years


def detect_fiscal_year(text: str) -> int | None:
    """Best-effort report fiscal year (the calendar year the financial year ended).

    Prefers the structurally-unambiguous "year ended 31 March YYYY" style statements; only
    falls back to bare/short "FY" mentions (which can appear in forward-looking prose, e.g. a
    target for a future year, not this report's own year) when none of those are found at all.
    Within each tier, the max of any matches is taken (a real report's own comparative tables
    routinely show a PRIOR year alongside the current one, e.g. "year ended 31 March 2025" next
    to "year ended 31 March 2026"; the current year is always the larger of the two).

    KNOWN RESIDUAL GAP: if the reliable tier finds nothing at all AND the fallback tier's own
    matches include a genuine short-form current-year mention alongside a SEPARATE forward-
    looking short-form mention (e.g. "In FY26 revenue grew... we target FY30 as a milestone"),
    max() still picks the higher, wrong one -- tiering can't disambiguate two matches of the
    SAME ambiguity level. This requires a real annual report to have zero "year ended 31 March"
    statements anywhere in its first 20000 characters, which is unusual (that statement is a
    legally-required part of the audited financial statements), so this is expected to be rare
    in practice. Not fixed here: reliably distinguishing a genuine mention from a forward-looking
    one within the same ambiguous pattern would need keyword/context heuristics ("target",
    "expect", "guidance") that are themselves error-prone without real samples to calibrate
    against, a worse trade than leaving this narrow, safe-direction (over-rejection, not a
    fabricated fact) gap documented.
    """
    head = text[:20000]
    reliable = _years_matching(_FY_PATTERNS_RELIABLE, head)
    if reliable:
        return max(reliable)
    fallback = _years_matching(_FY_PATTERNS_FALLBACK, head)
    for m in _FY_SHORT.finditer(head):
        y = _num_year("20" + m.group(1))
        if y:
            fallback.append(y)
    for m in _FY_RANGE.finditer(head):        # "FY 2025-26" style -> the ending year 2026
        y = _num_year("20" + m.group(1))
        if y:
            fallback.append(y)
    return max(fallback) if fallback else None


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
        # WHY (real money): a report PDF that times out, 403s, or won't parse must make this source
        # ABSTAIN (return no figures), never crash the caller. This is the primary "Research"
        # action; an unguarded fetch here surfaced a full-page stack trace to the parents.
        try:
            text = self.text_provider(symbol)
        except Exception:
            text = None
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
                # WHY prefer the deterministic detect_fiscal_year over the model's top-level claim
                # (real money, cross-verification integrity): parse_extraction already VALIDATES each
                # figure's year against detect_fiscal_year, deliberately distrusting the model's own
                # year claim (a model confused enough to mislabel a figure could be wrong about the
                # top-level year too). The YEAR TAG that aligns these figures with yfinance/Screener
                # must use that SAME authoritative reference, or a figure validated as FY2025 gets
                # tagged FY2026 whenever the model's top-level claim is wrong -- then it cross-verifies
                # against the WRONG year's value. Fall back to the model's claim only when the
                # deterministic detection finds nothing (figures_by_year returns {} if fy is None).
                fy = detect_fiscal_year(text) or _num_year(payload.get("fiscal_year"))
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
