"""LLM research notes, grounded in fetched data.

The whole real-money risk of an LLM here is a confident wrong number. The defense is
structural: the model is given a FACTS block and told to use only those figures and never
to recommend anything. The model is provider-agnostic (see src/llm/client.py); if none is
configured, the analysis still works and this layer returns a plain message.
"""
from __future__ import annotations

from ..llm.client import LLMClient, LiteLLMClient
from ..portfolio.models import PortfolioAnalysis, PositionAnalysis

_NO_LLM_MESSAGE = (
    "AI research note unavailable. Configure an LLM (set LLM_MODEL, e.g. an NVIDIA NIM open "
    "model) to enable grounded research notes. Portfolio analysis above does not need one."
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
- End with a "Source" line naming the data source AND the "Data as of" date from the FACTS \
block, so the reader knows how fresh these figures are.

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
- Neutral and concise. End with the "Source" line and the "Data as of" date from the FACTS \
block.
"""


class ResearchAnalyst:
    """Writes grounded portfolio notes through an injected, provider-agnostic LLMClient."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LiteLLMClient()

    @property
    def available(self) -> bool:
        return self.client.available

    def research_note(self, position: PositionAnalysis, fundamentals: dict,
                      source_label: str = "yfinance / Yahoo Finance",
                      data_as_of: str | None = None) -> str:
        if not self.available:
            return _NO_LLM_MESSAGE
        facts = _position_facts(position, fundamentals, source_label, data_as_of)
        return self._complete(_POSITION_SYSTEM, facts)

    def portfolio_overview(self, analysis: PortfolioAnalysis,
                           source_label: str = "yfinance / Yahoo Finance",
                           data_as_of: str | None = None) -> str:
        if not self.available:
            return _NO_LLM_MESSAGE
        facts = _portfolio_facts(analysis, source_label, data_as_of)
        return self._complete(_OVERVIEW_SYSTEM, facts)

    def _complete(self, system: str, facts: str) -> str:
        try:
            return self.client.complete(system, facts, max_tokens=800)
        except Exception as exc:  # surface the failure honestly, do not fake a note
            return f"Research note failed: {exc}"


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def _position_facts(position: PositionAnalysis, f: dict, source_label: str,
                    data_as_of: str | None = None) -> str:
    return f"""FACTS (use only these):
Data as of: {data_as_of or 'unavailable'}
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


def _portfolio_facts(a: PortfolioAnalysis, source_label: str,
                     data_as_of: str | None = None) -> str:
    sectors = ", ".join(f"{s} {w * 100:.1f}%" for s, w in
                        sorted(a.sector_weights.items(), key=lambda kv: -kv[1])) or "unavailable"
    missing = ", ".join(a.missing_symbols) if a.missing_symbols else "none"
    return f"""FACTS (use only these):
Data as of: {data_as_of or 'unavailable'}
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
