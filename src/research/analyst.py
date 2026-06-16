"""LLM research notes, grounded in fetched data.

The whole real-money risk of an LLM here is a confident wrong number. The defense is
structural: the model is given a FACTS block and told, in the system prompt, to use only
those figures and never to recommend anything. If no API key is set, the analysis still
works and this layer returns a plain "set your key" message instead of crashing.
"""
from __future__ import annotations

import os

from ..constants import DEFAULT_RESEARCH_MODEL
from ..portfolio.models import PortfolioAnalysis, PositionAnalysis

_NO_KEY_MESSAGE = (
    "AI research note unavailable. Set ANTHROPIC_API_KEY in your .env to enable grounded "
    "research notes. Portfolio analysis above does not need a key."
)

_POSITION_SYSTEM = """You are an equity research assistant for Indian stocks. You write a \
short, neutral research note for ONE holding, to help the owner understand the position \
they already hold.

HARD RULES (the user is dealing with real money):
- Use ONLY the figures in the FACTS block. Never introduce any number, ratio, date, or \
price from your own memory or training. If a figure is marked unavailable, say so.
- Give NO buy / sell / hold advice, NO price target, NO recommendation of any kind.
- Make NO prediction about future price or returns.
- Be concise and neutral. Describe what the data shows and what it does not.
- End with a "Source" line naming the data source given in the FACTS block.

Structure:
1. What the company is, using only the supplied name/sector/industry.
2. What the supplied valuation and price figures indicate, in plain terms, each figure cited.
3. What is notable about this position in the portfolio (its weight, its P&L), using only \
the supplied numbers.
4. Gaps: which data is unavailable here that the owner should check elsewhere before acting.
"""

_OVERVIEW_SYSTEM = """You are an equity research assistant for Indian stocks. You write a \
short, neutral overview of a portfolio's structure, to help the owner understand it.

HARD RULES (real money):
- Use ONLY the figures in the FACTS block. Never invent numbers.
- Give NO advice, NO recommendation, NO rebalancing suggestion, NO prediction.
- Describe concentration, sector tilt, and overall P&L strictly from the supplied figures.
- Neutral and concise. End with the "Source" line from the FACTS block.
"""


class ResearchAnalyst:
    """Wraps the Anthropic client. Lazy: no client is built until a note is requested."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("RESEARCH_MODEL") or DEFAULT_RESEARCH_MODEL
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client_or_none(self):
        if not self.available:
            return None
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def research_note(self, position: PositionAnalysis, fundamentals: dict,
                      source_label: str = "yfinance / Yahoo Finance") -> str:
        if not self.available:
            return _NO_KEY_MESSAGE
        facts = _position_facts(position, fundamentals, source_label)
        return self._complete(_POSITION_SYSTEM, facts)

    def portfolio_overview(self, analysis: PortfolioAnalysis,
                           source_label: str = "yfinance / Yahoo Finance") -> str:
        if not self.available:
            return _NO_KEY_MESSAGE
        facts = _portfolio_facts(analysis, source_label)
        return self._complete(_OVERVIEW_SYSTEM, facts)

    def _complete(self, system: str, facts: str) -> str:
        client = self._client_or_none()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": facts}],
            )
        except Exception as exc:  # surface the failure honestly, do not fake a note
            return f"Research note failed: {exc}"
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def _position_facts(position: PositionAnalysis, f: dict, source_label: str) -> str:
    return f"""FACTS (use only these):
Symbol: {position.symbol}
Company name: {f.get('name') or 'unavailable'}
Sector: {f.get('sector') or position.sector or 'unavailable'}
Industry: {f.get('industry') or 'unavailable'}
Current price: {_fmt(position.current_price)}
Trailing P/E: {_fmt(f.get('trailing_pe'))}
Forward P/E: {_fmt(f.get('forward_pe'))}
Price/Book: {_fmt(f.get('price_to_book'))}
Dividend yield (fraction): {_fmt(f.get('dividend_yield'))}
52-week high: {_fmt(f.get('fifty_two_week_high'))}
52-week low: {_fmt(f.get('fifty_two_week_low'))}
Reported beta: {_fmt(f.get('beta'))}
Position quantity: {_fmt(position.quantity)}
Average cost: {_fmt(position.avg_cost)}
Invested: {_fmt(position.invested)}
Market value: {_fmt(position.market_value)}
Unrealized P&L: {_fmt(position.pnl_abs)} ({_fmt(position.pnl_pct, '%')})
Weight in portfolio: {_fmt(position.weight * 100, '%')}
Source: {source_label}
"""


def _portfolio_facts(a: PortfolioAnalysis, source_label: str) -> str:
    sectors = ", ".join(f"{s} {w * 100:.1f}%" for s, w in
                        sorted(a.sector_weights.items(), key=lambda kv: -kv[1])) or "unavailable"
    missing = ", ".join(a.missing_symbols) if a.missing_symbols else "none"
    return f"""FACTS (use only these):
Number of priced holdings: {len(a.positions)}
Total invested: {_fmt(a.total_invested)}
Total market value: {_fmt(a.total_value)}
Total unrealized P&L: {_fmt(a.total_pnl_abs)} ({_fmt(a.total_pnl_pct, '%')})
Largest single holding weight: {_fmt(a.top_holding_weight * 100, '%')}
Herfindahl concentration index (0..1): {_fmt(a.hhi)}
Effective number of holdings (1/HHI): {_fmt(a.effective_holdings)}
Sector weights: {sectors}
Symbols with no price (excluded from totals): {missing}
Source: {source_label}
"""
