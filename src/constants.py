"""Domain constants. WHY: one source of truth (CLAUDE.md). Grep here before adding any."""

CURRENCY_SYMBOL = "₹"  # rupee

# Index symbols (Yahoo Finance)
NIFTY50_SYMBOL = "^NSEI"
SENSEX_SYMBOL = "^BSESN"
DEFAULT_BENCHMARK = NIFTY50_SYMBOL
INDEX_DISPLAY_NAMES = {
    NIFTY50_SYMBOL: "NIFTY 50",
    SENSEX_SYMBOL: "SENSEX",
}

# Yahoo Finance exchange suffixes
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# Risk math
TRADING_DAYS_PER_YEAR = 252
DEFAULT_HISTORY_PERIOD = "1y"

# Concentration flags (advisory display only, never advice)
CONCENTRATION_TOP_HOLDING_WARN = 0.25  # one name > 25% of the book
CONCENTRATION_HHI_WARN = 0.20          # Herfindahl index above this reads as concentrated

# Promoter pledge AT OR ABOVE this % of promoter holding reads as a serious red flag. ONE source of
# truth: shared by the framework's promoter-pledge metric (analysis, >=) and the Screener pledge
# signal's severity wording (data, >=), which must never drift apart on a real-money threshold --
# so BOTH use >= this value (they previously used > vs >=, contradicting each other at the boundary).
PROMOTER_PLEDGE_HIGH_PCT = 25.0

# LLM research layer: model/provider is configured via LLM_MODEL (see src/llm/client.py).

DISCLAIMER = (
    "For personal research only. Not investment advice, not a recommendation, and not a "
    "solicitation to buy or sell. Data may be delayed or incorrect. Verify every figure "
    "before acting. You alone are responsible for your decisions."
)
