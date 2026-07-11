"""Plain-English definitions for the terms the app shows.

The users are non-expert parents, so every number on screen should be explainable in one or
two plain sentences. This is curated static text (no LLM), so there is no hallucination risk
here. Keep each definition concrete and jargon-free.
"""
from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "P&L": "Profit and loss: how much your holding is up or down versus what you paid. "
           "It is on paper until you actually sell.",
    "Weight": "How big a slice of your total portfolio this one holding is, as a percent.",
    "Concentration (HHI)": "A score for how much your money is bunched into a few names. "
                           "Higher means more bunched, which means more risk if one falls.",
    "Effective number of holdings": "Roughly how many equal-sized holdings your portfolio "
                                    "behaves like. Lower than your real count means a few names dominate.",
    "Volatility": "How much the value bounces around. Higher volatility means bigger swings "
                  "up and down, not higher or lower returns.",
    "Beta": "How much a holding tends to move when the whole market moves. Beta 1 moves with "
            "the market; above 1 swings more; below 1 swings less.",
    "Maximum drawdown": "The worst drop from a peak to a later low over the period. It shows "
                        "how bad a fall has been, not how bad it could get.",
    "NAV": "Net Asset Value: the per-unit price of a mutual fund. A past NAV does not predict "
           "future returns.",
    "SIP": "Systematic Investment Plan: investing a fixed amount every month into a fund.",
    "Expense ratio": "The yearly fee a mutual fund charges, as a percent of your money. Lower "
                     "is cheaper for you.",
    "Exit load": "A fee charged if you sell fund units before a set time, like one year.",
    "Dividend yield": "The yearly dividend a stock pays, as a percent of its price.",
    "P/E ratio": "Price divided by earnings per share. A rough gauge of how expensive a stock "
                 "is versus its profits. It needs context, not a number to chase.",
    "Confidence": "How much of the data behind this verdict could be cross-checked across two or "
                  "more independent sources, NOT how likely the stock is to go up. High means most "
                  "figures were confirmed; low means much of it could not be, so lean on it less.",
    "Verified fact": "A statement taken straight from a primary source (an annual report, "
                     "filing, or exchange/AMFI data) and cited. The strongest kind of claim here.",
    "Opinion": "An attributed view from an analyst or commentator, not a hard fact. Weigh it, "
               "do not treat it as proven.",
    "Unverified": "Something that could not be tied to a primary source. Treat it as a lead to "
                  "check, never as a fact.",
}


def explain(term: str) -> str | None:
    """Return the plain-English definition for a term, or None if it is not in the glossary."""
    return GLOSSARY.get(term)
