"""Generate full 3-source reports (yfinance + Screener + auto annual report) for a list of
symbols and write results incrementally to data/batch_reports.txt (gitignored). Slow: each
symbol fetches and extracts a full annual report via the local LLM.

Symbols come from holdings.csv (gitignored) if present, else the bundled sample, else CLI args,
so no real portfolio list is ever hardcoded in the repo:

    LLM_MODEL=ollama_chat/qwen2.5:7b LLM_API_BASE=http://localhost:11434 \
        ./.venv/bin/python scripts/batch_reports.py [SYMBOL ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.figure_sources import YFinanceFigureSource  # noqa: E402
from src.data.nse_annual_reports import nse_annual_report_source  # noqa: E402
from src.data.screener_source import ScreenerFigureSource  # noqa: E402
from src.llm.client import LiteLLMClient  # noqa: E402
from src.pipeline import build_report_for_symbol  # noqa: E402
from src.portfolio.loader import load_holdings  # noqa: E402


def _format_verdict_lines(v) -> list[str]:
    """All of a verdict's disclosure lines: cross-verified reasons, then any sector-specific
    caveats (bank/NBFC, real-estate, ...). WHY: reading only `reasons` would silently drop a
    caveat -- see Verdict.sector_caveats, which this app's other rendering surfaces (the
    Research tab, the PDF export) already include alongside reasons."""
    lines = [f"    - {r}" for r in v.reasons]
    lines += [f"    - {c}" for c in v.sector_caveats]
    return lines


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:                                  # explicit CLI symbols win
        syms = [s.strip().upper() for s in sys.argv[1:]]
    else:                                                  # else read the portfolio (never hardcoded)
        src = root / "holdings.csv"
        if not src.exists():
            src = root / "sample_data" / "sample_portfolio.csv"
        syms = [h.symbol for h in load_holdings(src)]

    out = root / "data" / "batch_reports.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    def log(msg: str) -> None:
        with out.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    for sym in syms:
        t0 = time.time()
        try:
            report = build_report_for_symbol(
                sym, [YFinanceFigureSource(), ScreenerFigureSource(),
                      nse_annual_report_source(client=LiteLLMClient())])
            v = report.verdict
            verified = [(f.name, round(f.value)) for f in report.figures if f.is_trustworthy]
            conflicts = [f.name for f in report.figures if f.status.value == "conflict"]
            log(f"### {sym}  ({time.time() - t0:.0f}s)")
            log(f"  valuation={v.valuation.value} quality={v.quality.value} "
                f"leaning={v.leaning.value} confidence={v.confidence.value}")
            log(f"  verified: {verified}")
            log(f"  conflicts: {conflicts}")
            for line in _format_verdict_lines(v):
                log(line)
            log("")
        except Exception as exc:
            log(f"### {sym} ERROR ({time.time() - t0:.0f}s): {type(exc).__name__}: {str(exc)[:120]}")
            log("")

    log("DONE")


if __name__ == "__main__":
    main()
