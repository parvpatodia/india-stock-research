"""Rupee display formatting, in the Indian convention. Dependency-free (no I/O, no pandas,
no yfinance), so every layer -- the data figures, the analysis guidance, and the app UI -- can
share ONE source of truth for how money reads to the parents, without pulling heavier modules
into the pure analysis layer.

Two forms:
- format_rupees: exact whole rupees, Indian-grouped (a holding / P&L / allocation amount, where
  the reader wants the precise number, e.g. "₹5,00,000").
- format_rupees_crore_lakh: abbreviated crore/lakh (large company financials, where "₹79,000
  crore" reads far better than a 12-digit absolute-rupee string).
"""
from __future__ import annotations

from .constants import CURRENCY_SYMBOL


def indian_group(int_digits: str) -> str:
    """Group an integer digit-string in the Indian convention: the last 3 digits, then 2 at a
    time (e.g. 1056499 -> '10,56,499'). WHY: the reader is an Indian investor; Western thousands
    grouping of a rupee figure reads wrong to them."""
    if len(int_digits) <= 3:
        return int_digits
    head, last3 = int_digits[:-3], int_digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + last3


def format_rupees(value: float | None) -> str:
    """Exact whole-rupee amount, Indian-grouped (e.g. ₹5,00,000). None -> 'n/a' so a missing
    price/value never renders as a fabricated ₹0. A negative amount (a loss) keeps a clean
    leading sign: -₹50,000."""
    if value is None:
        return "n/a"
    rounded = int(round(value))
    neg = rounded < 0
    grouped = indian_group(str(abs(rounded)))
    return f"{'-' if neg else ''}{CURRENCY_SYMBOL}{grouped}"


def format_rupees_crore_lakh(value: float) -> str:
    """Rupees in the Indian crore (1e7) / lakh (1e5) convention. WHY: company financials are held
    in ABSOLUTE rupees, so a real net profit rendered raw is a 12-digit string a parent has to
    count zeros on. Trailing zeros are stripped so a whole-crore figure reads clean and its digits
    line up with a model's '79,000 crore' phrasing under numbers_grounded."""
    a = abs(value)
    if a >= 1e7:
        num, unit = value / 1e7, " crore"
    elif a >= 1e5:
        num, unit = value / 1e5, " lakh"
    else:
        num, unit = value, ""
    s = f"{num:.2f}".rstrip("0").rstrip(".")   # up to 2 decimals, no trailing zeros
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    int_part, _, dec_part = s.partition(".")
    grouped = indian_group(int_part) + (f".{dec_part}" if dec_part else "")
    return f"{'-' if neg else ''}{CURRENCY_SYMBOL}{grouped}{unit}"
