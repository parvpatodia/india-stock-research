"""Generate full 3-source reports (yfinance + Screener + auto annual report) for a list of
symbols and write results incrementally to data/batch_reports.txt (gitignored). Slow: each
symbol fetches and extracts a full annual report via the local LLM.

    LLM_MODEL=ollama_chat/qwen2.5:7b LLM_API_BASE=http://localhost:11434 \
        ./.venv/bin/python scripts/batch_reports.py
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

SYMS = ["MBAPL", "SHAKTIPUMP", "BLS", "ASTRAL", "BRIGADE",
        "SBIN", "VOLTAMP", "YESBANK", "ADANIPOWER", "ICICIBANK"]

OUT = Path(__file__).resolve().parents[1] / "data" / "batch_reports.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("")


def log(msg: str) -> None:
    with OUT.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


for sym in SYMS:
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
        for reason in v.reasons:
            log(f"    - {reason}")
        log("")
    except Exception as exc:
        log(f"### {sym} ERROR ({time.time() - t0:.0f}s): {type(exc).__name__}: {str(exc)[:120]}")
        log("")

log("DONE")
