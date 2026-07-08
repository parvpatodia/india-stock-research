"""India Equity Research - Streamlit app.

Research-only decision support for a family portfolio: no trades, no buy/sell calls. Every
figure is fetched and timestamped, cross-verified across independent sources or withheld, and
the human expert approves a report before it counts. Built mobile-first (the parents use
iPhones): summary first, plain language, evidence one tap away.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# WHY: ensure `src` imports work regardless of the cwd Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.analysis.sizing import (  # noqa: E402
    AllocationCandidate,
    Stance,
    position_sizing,
    stance_from_verdict,
    suggest_allocation,
)
from src.constants import (  # noqa: E402
    CONCENTRATION_HHI_WARN,
    CONCENTRATION_TOP_HOLDING_WARN,
    DEFAULT_BENCHMARK,
    DISCLAIMER,
    INDEX_DISPLAY_NAMES,
    NIFTY50_SYMBOL,
    SENSEX_SYMBOL,
)
from src.data.annual_report_source import AnnualReportFigureSource  # noqa: E402
from src.data.figure_sources import YFinanceFigureSource  # noqa: E402
from src.data.news_source import NewsSource, registry_with_news  # noqa: E402
from src.data.nse_annual_reports import nse_annual_report_source  # noqa: E402
from src.data.screener_source import ScreenerFigureSource  # noqa: E402
from src.data.sheets_backend import (  # noqa: E402
    append_log,
    build_gateway,
    read_holdings,
    read_reports,
    record_from_report,
    save_report,
)
from src.data.yfinance_provider import YFinanceProvider  # noqa: E402
from src.eval.cases import EvalStore  # noqa: E402
from src.eval.harness import evaluate, ground_truth_from_report  # noqa: E402
from src.glossary import GLOSSARY, explain  # noqa: E402
from src.llm.client import LiteLLMClient  # noqa: E402
from src.pipeline import build_report_for_symbol  # noqa: E402
from src.portfolio.analysis import (  # noqa: E402
    analyze_portfolio,
    annualized_volatility,
    beta,
    daily_returns,
    enrich_sectors,
    max_drawdown,
    portfolio_daily_returns,
)
from src.portfolio.loader import load_holdings  # noqa: E402
from src.research.analyst import ResearchAnalyst  # noqa: E402
from src.research.claims import ESTIMATE, FACT, OPINION  # noqa: E402
from src.research.grounded_analyst import GroundedAnalyst  # noqa: E402
from src.research.grounding import DocumentStore  # noqa: E402
from src.research.library import build_library  # noqa: E402
from src.research.report import ReviewStatus  # noqa: E402
from src.sip import sip_future_value  # noqa: E402
from src.sources.adapters import HttpDocumentAdapter, ingest_documents  # noqa: E402
from src.sources.registry import SourceRegistry  # noqa: E402
from src.data.amfi_provider import AMFIProvider  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")  # load the app's own .env regardless of cwd

st.set_page_config(page_title="India Equity Research", layout="wide", page_icon="📊",
                   initial_sidebar_state="collapsed")  # collapsed = mobile-first

_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = _ROOT / "sample_data" / "sample_portfolio.csv"
HOLDINGS_CSV = _ROOT / "holdings.csv"   # the owner's real portfolio (gitignored)
EVAL_STORE = _ROOT / "data" / "eval_cases.jsonl"

# Sources/documents: prefer the owner's real config, else fall back to the bundled sample.
SOURCES_YAML = _ROOT / "config" / "sources.yaml"
DOCS_DIR = _ROOT / "documents"
if not SOURCES_YAML.exists():
    SOURCES_YAML = _ROOT / "sample_data" / "sources.yaml"
    DOCS_DIR = _ROOT / "sample_data" / "documents"

CURRENCY = "₹"

# Plain-language stance rendering (icon, headline). Kept in one place so wording is consistent.
_STANCE_UI = {
    Stance.FAVORABLE: ("🟢", "Evidence leans favorable"),
    Stance.NEUTRAL: ("🟡", "Evidence is mixed / neutral"),
    Stance.UNFAVORABLE: ("🔴", "Evidence leans unfavorable"),
    Stance.INSUFFICIENT_DATA: ("⚪", "Not enough verified data"),
}


# --- cached data access (the provider is the only network boundary) ---

@st.cache_resource
def get_provider() -> YFinanceProvider:
    return YFinanceProvider()


@st.cache_resource
def get_analyst() -> ResearchAnalyst:
    return ResearchAnalyst()


@st.cache_resource
def get_grounded_analyst() -> GroundedAnalyst:
    return GroundedAnalyst()


@st.cache_resource
def get_news_source() -> NewsSource:
    return NewsSource()


@st.cache_resource(show_spinner="Loading AMFI mutual-fund NAVs...")
def get_amfi() -> AMFIProvider | None:
    # WHY: loaded lazily (only when the user searches a fund) so a page load needs no network.
    provider = AMFIProvider()
    try:
        provider.load()
    except Exception:
        return None
    return provider


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(symbols: tuple[str, ...]) -> tuple[dict[str, float | None], str]:
    # WHY: stamp the fetch time inside the cached fn so the displayed "as of" reflects the
    # actual price age on a cache hit, not the current wall clock (real-money guardrail #1).
    prices = get_provider().current_prices(list(symbols))
    return prices, datetime.now().strftime("%Y-%m-%d %H:%M")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_index(symbol: str) -> dict:
    return get_provider().index_quote(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_close(symbol: str) -> pd.Series:
    hist = get_provider().history(symbol, period="1y")
    if "Close" in hist:
        return hist["Close"].dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    return get_provider().fundamentals(symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(symbol: str, company_name: str):
    return get_news_source().fetch(symbol, company_name)


@st.cache_resource
def get_base_registry() -> SourceRegistry | None:
    return SourceRegistry.from_config(SOURCES_YAML) if SOURCES_YAML.exists() else None


def _secret(key: str, default=None):
    """Read a Streamlit secret, tolerating no secrets file at all (local dev)."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _sheet_configured() -> bool:
    return bool(_secret("sheet_key") and _secret("gcp_service_account"))


@st.cache_resource
def get_gateway():
    """The persistence backend: the real Google Sheet if a service account is configured in
    secrets, else a local JSON file (gitignored) so approvals still persist in dev."""
    creds = _secret("gcp_service_account")
    creds_dict = dict(creds) if creds else None
    return build_gateway(creds_dict, _secret("sheet_key"), _ROOT / "data" / "reports.json")


def _persist_review(report, sym: str, stance, action: str, reviewer: str, note: str) -> None:
    """Persist an approve/reject to the gateway (Sheet or local JSON). Best-effort: a
    persistence error must never block the in-session review action."""
    try:
        gw = get_gateway()
        save_report(gw, record_from_report(report, sym, stance.value))
        append_log(gw, action, sym, reviewer, note)
    except Exception:
        pass


def _library_fingerprint() -> str:
    parts = []
    if SOURCES_YAML.exists():
        s = SOURCES_YAML.stat()
        parts.append(f"cfg:{s.st_mtime_ns}:{s.st_size}")
    if DOCS_DIR.is_dir():
        for p in sorted(DOCS_DIR.iterdir()):
            if p.is_file():
                st_ = p.stat()
                parts.append(f"{p.name}:{st_.st_mtime_ns}:{st_.st_size}")
    return "|".join(parts)


@st.cache_resource
def get_curated_library(fingerprint: str):
    """Build the news-inclusive registry + a store of the owner's curated documents (if any).
    Registry always includes the live news feeds so news can be ingested as attributed context."""
    base = get_base_registry()
    registry = registry_with_news(base)
    store = DocumentStore(registry=registry)
    skipped: list[str] = []
    failed: list[str] = []
    if base is not None:
        _, skipped, failed = build_library(base, DOCS_DIR, store=store)
    return registry, store, skipped, failed


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{CURRENCY}{value:,.0f}"


def plain_summary(verdict, stance: Stance) -> str:
    """A one-line, jargon-free read for a non-expert. Honest when data is thin."""
    if verdict is None or stance == Stance.INSUFFICIENT_DATA:
        return ("Not enough independently verified data to form a view. Withheld on purpose, "
                "rather than guessed.")
    val = {"cheap": "looks cheap versus its own history",
           "fair": "looks fairly priced versus its own history",
           "expensive": "looks expensive versus its own history",
           "unknown": "valuation could not be verified"}[verdict.valuation.value]
    qual = {"strong": "a strong balance sheet",
            "mixed": "a mixed balance sheet",
            "weak": "balance-sheet concerns",
            "unknown": "balance-sheet quality unconfirmed"}[verdict.quality.value]
    _, headline = _STANCE_UI[stance]
    return f"It {val}, with {qual}. {headline}."


def build_markdown_report(title: str, report, stance: Stance) -> str:
    """A plain-text report the user can download and read or share. Mirrors what is on screen."""
    v = report.verdict
    lines = [f"# {title}", ""]
    status = ("APPROVED (reviewed)" if report.is_trusted
              else "REJECTED, not for decisions" if report.status == ReviewStatus.REJECTED
              else "DRAFT, not yet reviewed by your expert")
    lines += [f"**Status:** {status}", f"**Generated:** {report.created_at}", ""]
    icon, headline = _STANCE_UI[stance]
    lines += [f"## {icon} {headline}", "", plain_summary(v, stance), ""]
    if v is not None:
        lines += ["## Verdict",
                  f"- Valuation: {v.valuation.value}",
                  f"- Quality: {v.quality.value}",
                  f"- Leaning: {v.leaning.value}",
                  f"- Confidence: {v.confidence.value}", ""]
        if v.reasons:
            lines += ["## Why (each from cross-verified figures)"]
            lines += [f"- {r}" for r in v.reasons]
            lines += [""]
    lines += ["## Figures"]
    for f in report.figures:
        val = f"{f.value:,.2f}" if f.value is not None else "withheld (not cross-verified)"
        srcs = ", ".join(sorted({sv.source_id for sv in f.sources}))
        lines += [f"- {f.name}: {val}  [{f.status.value}; sources: {srcs}]"]
    lines += ["", "---", (v.caveat if v is not None else DISCLAIMER), "", DISCLAIMER]
    return "\n".join(lines)


# --- header ---

st.title("📊 India Equity Research")
st.caption("Understand your investments. Research only, no advice, and never a buy or sell order.")
st.warning(DISCLAIMER, icon="⚠️")

# --- sidebar: input + settings + status ---

with st.sidebar:
    st.header("Settings")
    sheet_on = _sheet_configured()
    use_sheet = st.checkbox("Use my Google Sheet portfolio", value=sheet_on,
                            disabled=not sheet_on,
                            help="Reads holdings live from the linked Google Sheet." if sheet_on
                            else "Not configured. Add gcp_service_account + sheet_key to secrets.")
    uploaded = st.file_uploader("Upload / update portfolio CSV", type=["csv"])
    have_real = HOLDINGS_CSV.exists()
    use_mine = st.checkbox("Use my portfolio (holdings.csv)",
                           value=have_real and not uploaded and not (sheet_on and use_sheet),
                           disabled=not have_real)
    use_sample = st.checkbox("Use sample portfolio",
                             value=not have_real and not uploaded and not (sheet_on and use_sheet))
    st.caption("Columns matched loosely: Symbol, Quantity, Avg Cost, (optional) Sector. "
               "Zerodha/Groww exports work too.")
    st.divider()
    cap_pct = st.slider("Per-stock cap (%)", min_value=5, max_value=40,
                        value=int(CONCENTRATION_TOP_HOLDING_WARN * 100),
                        help="No single stock should exceed this share of the book. Used for the "
                             "'how much' sizing and the lump-sum plan.") / 100.0
    st.divider()
    analyst = get_analyst()
    if analyst.available:
        st.success(f"AI research: on ({analyst.client.model_name})")
    else:
        st.info("AI research: off. Set LLM_MODEL to enable the annual-report tiebreaker and the "
                "research chat. The cross-verified analysis works without it.")

# --- resolve the data source (Google Sheet if selected, else CSV) ---

holdings = None
if sheet_on and use_sheet:
    try:
        holdings = read_holdings(get_gateway()) or None
    except Exception as exc:
        st.error(f"Could not read the Google Sheet: {exc}")

if holdings is None:
    source = None
    if uploaded is not None:
        source = uploaded
    elif use_mine and HOLDINGS_CSV.exists():
        source = HOLDINGS_CSV
    elif use_sample:
        source = SAMPLE_CSV
    if source is None:
        st.info("Upload a portfolio CSV, or tick a portfolio option in the sidebar, to begin.")
        st.stop()
    try:
        holdings = load_holdings(source)
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        st.stop()

if not holdings:
    st.error("No valid holdings found.")
    st.stop()

symbols = tuple(h.symbol for h in holdings)

# Backfill blank sectors from yfinance (cached per symbol; a cold cache pays a one-time cost).
with st.spinner("Looking up sectors…"):
    holdings = enrich_sectors(holdings, fetch_fundamentals)

with st.spinner("Fetching live prices..."):
    prices, prices_as_of = fetch_prices(symbols)
analysis = analyze_portfolio(holdings, prices)
value_by_symbol = {p.symbol: p.market_value for p in analysis.positions}

if "reports" not in st.session_state:
    st.session_state.reports = {}

tab_portfolio, tab_research, tab_invest, tab_ask = st.tabs(
    ["📁 My Portfolio", "🔎 Research a Stock", "💰 Invest a Lump Sum", "💬 Ask"])


# ==================== TAB 1: MY PORTFOLIO ====================

with tab_portfolio:
    st.subheader("Your portfolio")
    st.caption(f"Prices as of {prices_as_of}. Source: yfinance / Yahoo Finance.")

    m = st.columns(2)
    m[0].metric("Invested", money(analysis.total_invested))
    m[1].metric("Market value", money(analysis.total_value))
    m2 = st.columns(2)
    m2[0].metric("Profit / loss", money(analysis.total_pnl_abs),
                 f"{analysis.total_pnl_pct:+.2f}%", help=explain("P&L"))
    m2[1].metric("Holdings priced", f"{len(analysis.positions)} / {len(holdings)}")

    if analysis.missing_symbols:
        st.warning("No price found for: " + ", ".join(analysis.missing_symbols)
                   + ". Excluded from the totals. Check the symbol spelling or exchange.")

    rows = [{
        "Symbol": p.symbol, "Sector": p.sector, "Qty": p.quantity,
        "Avg cost": round(p.avg_cost, 2), "Price": round(p.current_price, 2),
        "Value": round(p.market_value, 2), "P&L %": round(p.pnl_pct, 2),
        "Weight %": round(p.weight * 100, 2),
    } for p in sorted(analysis.positions, key=lambda x: -x.market_value)]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Allocation charts"):
        alloc_df = pd.DataFrame({
            "Symbol": [p.symbol for p in analysis.positions],
            "Weight": [p.weight * 100 for p in analysis.positions],
        })
        st.plotly_chart(px.pie(alloc_df, names="Symbol", values="Weight", hole=0.4,
                               title="By holding"), width="stretch")
        sector_df = pd.DataFrame({
            "Sector": list(analysis.sector_weights.keys()),
            "Weight %": [w * 100 for w in analysis.sector_weights.values()],
        }).sort_values("Weight %", ascending=False)
        st.plotly_chart(px.bar(sector_df, x="Sector", y="Weight %", title="By sector"),
                        width="stretch")

    with st.expander("Concentration"):
        cc = st.columns(3)
        cc[0].metric("Largest holding", f"{analysis.top_holding_weight * 100:.1f}%")
        cc[1].metric("HHI", f"{analysis.hhi:.3f}", help=explain("Concentration (HHI)"))
        cc[2].metric("Effective # holdings", f"{analysis.effective_holdings:.1f}")
        flags = []
        if analysis.top_holding_weight > CONCENTRATION_TOP_HOLDING_WARN:
            flags.append(f"One name is over {CONCENTRATION_TOP_HOLDING_WARN * 100:.0f}% of the book.")
        if analysis.hhi > CONCENTRATION_HHI_WARN:
            flags.append("HHI reads as concentrated (few names drive the book).")
        if flags:
            st.warning(" ".join(flags) + " An observation about structure, not advice.")
        else:
            st.success("No single-name or HHI concentration flag triggered.")

    with st.expander("Risk (1-year history)"):
        if st.button("Compute risk metrics"):
            with st.spinner("Fetching 1-year history..."):
                bench_close = fetch_history_close(DEFAULT_BENCHMARK)
                bench_returns = daily_returns(bench_close)
                close_by_symbol = {p.symbol: fetch_history_close(p.symbol)
                                   for p in analysis.positions}
                weights = {p.symbol: p.weight for p in analysis.positions}
                port_returns = portfolio_daily_returns(close_by_symbol, weights)
            risk_cols = st.columns(3)
            risk_cols[0].metric("Annualized volatility",
                                f"{annualized_volatility(port_returns) * 100:.1f}%",
                                help=explain("Volatility"))
            risk_cols[1].metric(f"Beta vs {INDEX_DISPLAY_NAMES[DEFAULT_BENCHMARK]}",
                                f"{beta(port_returns, bench_returns):.2f}", help=explain("Beta"))
            worst = min((max_drawdown(c) for c in close_by_symbol.values() if not c.empty),
                        default=0.0)
            risk_cols[2].metric("Worst single-name drawdown", f"{worst * 100:.1f}%",
                                help=explain("Maximum drawdown"))

    with st.expander("Market context"):
        ctx_cols = st.columns(2)
        for col, idx_symbol in zip(ctx_cols, (NIFTY50_SYMBOL, SENSEX_SYMBOL)):
            quote = fetch_index(idx_symbol)
            name = INDEX_DISPLAY_NAMES.get(idx_symbol, idx_symbol)
            price = quote.get("price")
            change = quote.get("change_pct")
            col.metric(name, f"{price:,.2f}" if price is not None else "n/a",
                       f"{change:+.2f}%" if change is not None else None)


# ==================== TAB 2: RESEARCH A STOCK ====================

def _run_live(sym: str, ar_override: str = ""):
    sources = [YFinanceFigureSource(), ScreenerFigureSource()]  # both free; cross-verify
    label = "yfinance + screener"
    if ar_override.strip():
        _adapter = HttpDocumentAdapter("annual_report")

        def _ar_text(_symbol, _url=ar_override.strip()):
            docs = _adapter.fetch(_url)
            return docs[0].text if docs else None

        sources.append(AnnualReportFigureSource(_ar_text, client=LiteLLMClient()))
        label += " + annual report"
    elif LiteLLMClient().available:
        sources.append(nse_annual_report_source(client=LiteLLMClient()))
        label += " + annual report (auto)"
    key = f"{sym} (live/{label})"
    with st.spinner(f"Analyzing {sym} ({label})..."):
        st.session_state.reports[key] = build_report_for_symbol(sym, sources)
    st.session_state.active_report = key


with tab_research:
    st.subheader("Research a stock")
    st.caption("Any NSE stock, yours or not. yfinance + Screener are cross-verified; if an LLM is "
               "set, the annual report is auto-fetched as a third source to break ties.")

    port_syms = sorted(symbols)
    pick = st.selectbox("Pick one of your holdings", port_syms)
    if st.button("Research this holding", type="primary"):
        _run_live(pick)
    with st.expander("Or search any other stock"):
        typed = st.text_input("NSE symbol", placeholder="RELIANCE")
        ar_url = st.text_input("Annual report PDF URL (optional override)",
                               placeholder="blank = auto-fetch from NSE")
        if st.button("Research this symbol") and typed.strip():
            _run_live(typed.strip().upper(), ar_url)

    active = st.session_state.get("active_report")
    report = st.session_state.reports.get(active) if active else None
    if report is not None:
        sym = active.split(" ")[0]
        stance = stance_from_verdict(report.verdict)
        icon, headline = _STANCE_UI[stance]

        # status banner
        if report.is_trusted:
            last = report.audit[-1]
            st.success(f"APPROVED by {last.reviewer}. Reviewed.")
        elif report.status == ReviewStatus.REJECTED:
            st.error("REJECTED, sent back for correction. Not for decisions.")
        else:
            st.warning("DRAFT, not yet reviewed by your expert. Not for decisions.")

        # summary-first: one line + stance
        st.markdown(f"### {icon} {sym}: {headline}")
        st.write(plain_summary(report.verdict, stance))

        # how much (transparent cap math), framed by the stance
        held = value_by_symbol.get(sym, 0.0)
        sizing = position_sizing(held, analysis.total_value, cap_pct)
        if stance in (Stance.FAVORABLE, Stance.NEUTRAL):
            if sizing.over_cap:
                st.info(f"You already hold {money(held)}, which is over your "
                        f"{cap_pct:.0%} per-stock cap of {money(sizing.cap_value)}. "
                        "Even though the evidence isn't negative, adding more would concentrate "
                        "the book further.")
            else:
                st.info(f"Your {cap_pct:.0%} per-stock cap is {money(sizing.cap_value)}. "
                        f"You hold {money(held)}, so there is room for about "
                        f"{money(sizing.headroom)} more if you decide to add. This is arithmetic "
                        "on your own limit, not a recommendation.")
        elif stance == Stance.UNFAVORABLE:
            st.info(f"The verified evidence leans unfavorable, so this is not a spot to add. "
                    f"You hold {money(held)}.")

        st.caption(report.verdict.caveat if report.verdict else DISCLAIMER)

        # download
        st.download_button("⬇️ Download full report",
                           data=build_markdown_report(active, report, stance),
                           file_name=f"{sym}_research.md", mime="text/markdown")

        # the evidence, one tap away
        with st.expander("See the evidence (figures, reasons, sources)"):
            if report.verdict is not None:
                vc = st.columns(2)
                vc[0].metric("Valuation", report.verdict.valuation.value)
                vc[1].metric("Quality", report.verdict.quality.value)
                vc2 = st.columns(2)
                vc2[0].metric("Leaning", report.verdict.leaning.value)
                vc2[1].metric("Confidence", report.verdict.confidence.value)
                if report.verdict.reasons:
                    st.markdown("**Why (each from cross-verified figures):**")
                    for reason in report.verdict.reasons:
                        st.markdown(f"- {reason}")
            fig_rows = [{
                "Figure": f.name, "Status": f.status.value,
                # WHY: all-string column; a mixed float/"withheld" column crashes st.dataframe.
                "Value": (f"{f.value:,.2f}" if f.value is not None else "withheld"),
                "Sources": ", ".join(sorted({sv.source_id for sv in f.sources})),
            } for f in report.figures]
            st.dataframe(pd.DataFrame(fig_rows), width="stretch", hide_index=True)
            if report.conflicts:
                st.error(f"{len(report.conflicts)} figure(s) in CONFLICT (independent sources "
                         "disagree); withheld from the verdict.")
            single = [f for f in report.figures if f.status.value == "single_source"]
            if single:
                st.info(f"{len(single)} figure(s) are single-source, so the verdict is "
                        "intentionally low-confidence until a second source cross-verifies them.")

        # recent news (context only, dated, never a verified fact)
        with st.expander("Recent news (context, dated, not verified facts)"):
            company = fetch_fundamentals(sym).get("name") or sym
            items = fetch_news(sym, company)
            if not items:
                st.caption("No recent news found (or the feed was unreachable).")
            for it in items:
                headline_md = f"[{it.title}]({it.url})" if it.url else it.title
                st.markdown(f"- {headline_md}  \n  _{it.publisher or 'source'}, "
                            f"{it.published or 'undated'}_")
            st.caption("News is reporting, attributed and dated. It is NOT cross-verified like a "
                       "figure and does not move the verdict above.")

        # expert review panel (the safety gate) + learning loop
        with st.expander("Expert review panel", expanded=not report.is_trusted):
            reviewer = st.text_input("Reviewer (your name)", key=f"rv_{active}")
            note = st.text_area("Note", key=f"note_{active}")
            ack = False
            if report.conflicts:
                ack = st.checkbox("I checked the conflicting figures by hand and accept them",
                                  key=f"ack_{active}")
            rc = st.columns(2)
            if rc[0].button("Approve", key=f"ap_{active}"):
                try:
                    updated = report.approve(reviewer=reviewer, note=note,
                                             acknowledge_conflicts=ack)
                    st.session_state.reports[active] = updated
                    _persist_review(updated, sym, stance, "approved", reviewer, note)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            corrections = rc[1].text_area("Corrections (one per line, for rejection)",
                                          key=f"corr_{active}")
            if rc[1].button("Reject", key=f"rj_{active}"):
                try:
                    fixes = tuple(c.strip() for c in corrections.splitlines() if c.strip())
                    updated = report.reject(reviewer=reviewer, note=note, corrections=fixes)
                    st.session_state.reports[active] = updated
                    _persist_review(updated, sym, stance, "rejected", reviewer, note)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if report.audit:
                st.markdown("**Review history:**")
                for e in report.audit:
                    st.caption(f"{e.timestamp} — {e.status.value} by {e.reviewer}: {e.note}")

            st.markdown("**Record a corrected figure** (feeds the learning loop; no mistake twice)")
            gt_fig = st.selectbox("Figure", [f.name for f in report.figures], key=f"gtf_{active}")
            gt_val = st.number_input("Correct value (absolute rupees)", value=0.0, step=1.0,
                                     format="%.0f", key=f"gtv_{active}")
            if st.button("Save correction", key=f"gts_{active}") and reviewer.strip():
                EvalStore(EVAL_STORE).add(ground_truth_from_report(
                    report, gt_fig, gt_val, note=note, reviewer=reviewer.strip()))
                st.success(f"Recorded ground truth for {gt_fig}. Checked on every run.")

    cases = EvalStore(EVAL_STORE).load()
    if cases:
        ev = evaluate(cases)
        st.caption(f"Learning loop: {len(cases)} recorded corrections, accuracy {ev.accuracy:.0%} "
                   f"({ev.matches}/{ev.total}), trusted-but-wrong {ev.trusted_wrong} (must be 0).")


# ==================== TAB 3: INVEST A LUMP SUM ====================

with tab_invest:
    st.subheader("Invest a lump sum")
    st.caption("Given an amount, this spreads it across your APPROVED names that the evidence "
               "supports, each kept under your per-stock cap. Not a buy order, you decide.")

    amount = st.number_input(f"Amount to invest ({CURRENCY})", min_value=0, value=0, step=50000)

    # Approved names = persisted (durable, from the Sheet/local store) + this session's approvals
    # (fresher). Only APPROVED qualifies; unreviewed drafts never appear here, on purpose.
    approved_stance: dict[str, Stance] = {}
    try:
        for r in read_reports(get_gateway()):
            if r.status == "approved" and r.stance:
                try:
                    approved_stance[r.symbol] = Stance(r.stance)
                except ValueError:
                    pass
    except Exception:
        pass
    for key, rep in st.session_state.reports.items():
        if rep.is_trusted and rep.verdict is not None:
            approved_stance[key.split(" ")[0]] = stance_from_verdict(rep.verdict)

    if not approved_stance:
        st.info("No approved research yet. Go to **Research a Stock**, review a report, and "
                "click Approve. Only approved names can be suggested here, on purpose.")
    else:
        candidates = [
            AllocationCandidate(symbol=sym, stance=stance,
                                current_value=value_by_symbol.get(sym, 0.0))
            for sym, stance in approved_stance.items()
        ]
        st.markdown("**Approved names considered:**")
        for c in candidates:
            icon, headline = _STANCE_UI[c.stance]
            st.markdown(f"- {icon} **{c.symbol}** — {headline.lower()} "
                        f"(you hold {money(c.current_value)})")

        if amount > 0 and st.button("Suggest how to spread it", type="primary"):
            plan = suggest_allocation(float(amount), candidates, analysis.total_value, cap_pct)
            if plan.allocations:
                st.markdown("**Suggested spread (within your caps):**")
                st.dataframe(pd.DataFrame([{"Stock": a.symbol, "Add": money(a.amount)}
                                           for a in plan.allocations]),
                             width="stretch", hide_index=True)
                for a in plan.allocations:
                    st.caption(f"{a.symbol}: {a.reason}")
                st.metric("Placed", money(plan.invested))
            if plan.uninvested > 0:
                st.warning(f"{money(plan.uninvested)} left unplaced.")
            for n in plan.notes:
                st.info(n)
            st.caption(plan.caveat)


# ==================== TAB 4: ASK ====================

with tab_ask:
    st.subheader("Ask about a stock")
    st.caption("Answered from recent news (attributed, dated context) and any curated sources. "
               "It cites where each answer came from, never gives buy/sell advice, and says so "
               "when it cannot answer.")

    registry, curated_store, skipped, failed = get_curated_library(_library_fingerprint())
    grounded = get_grounded_analyst()
    if not grounded.available:
        st.info("Set LLM_MODEL to ask questions. The sources still load below.")

    ask_sym = st.text_input("Stock (symbol)", placeholder="RELIANCE", key="ask_sym")
    question = st.text_input("Your question", placeholder="What is the recent news about it?")
    if st.button("Ask", disabled=not grounded.available) and question.strip():
        store = DocumentStore(registry=registry)
        # curated primary docs (if the owner added any) + live news for the named stock
        base = get_base_registry()
        if base is not None:
            build_library(base, DOCS_DIR, store=store)
        if ask_sym.strip():
            company = fetch_fundamentals(ask_sym.strip().upper()).get("name") or ask_sym
            with st.spinner("Reading recent news..."):
                items = fetch_news(ask_sym.strip().upper(), company)
                ingest_documents(store, NewsSource.as_documents(items))
        if len(store) == 0:
            st.warning("No sources to answer from. Enter a stock symbol so recent news can load.")
        else:
            with st.spinner("Reading the sources..."):
                result = grounded.answer(question, store, registry,
                                         as_of=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if result.abstained:
                st.warning(f"No verified answer. {result.abstain_reason}")
            else:
                for claim in result.claims:
                    cited = ", ".join(
                        (registry.get(c.source_id).name if registry.get(c.source_id) else c.source_id)
                        for c in claim.citations) or "no source"
                    if claim.kind == FACT and claim.is_verified_fact:
                        st.success(f"✓ {claim.text}")
                    elif claim.kind == OPINION:
                        st.info(f"Reported / opinion: {claim.text}")
                    elif claim.kind == ESTIMATE:
                        st.info(f"Estimate (derived, not a primary figure): {claim.text}")
                    else:
                        st.error(f"⚠ Unverified: {claim.text}")
                    st.caption(f"Source: {cited}")


# --- footer: funds/SIP + glossary + disclaimer ---

with st.expander("Mutual funds & SIP projection"):
    fund_query = st.text_input("Search a mutual fund by name", key="fund_q",
                               placeholder="e.g. bluechip, index, flexi cap")
    if fund_query.strip():
        provider = get_amfi()
        if provider is None:
            st.warning("Could not load AMFI NAV data (network issue). Try again in a moment.")
        else:
            hits = provider.search(fund_query, limit=15)
            if hits:
                st.dataframe(pd.DataFrame([{
                    "Scheme code": h.scheme_code, "Scheme": h.name,
                    "NAV": round(h.nav, 4), "As of": h.date,
                } for h in hits]), width="stretch", hide_index=True)
            else:
                st.info("No scheme matched that search.")
    st.markdown("**SIP projection** (arithmetic on an assumption, not a prediction)")
    sc = st.columns(3)
    sip_monthly = sc[0].number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500,
                                     help=explain("SIP"))
    sip_years = sc[1].number_input("Years", min_value=1, max_value=40, value=10)
    sip_return = sc[2].number_input("Assumed annual return (%)", min_value=0.0, max_value=30.0,
                                    value=10.0, step=0.5)
    proj = sip_future_value(sip_monthly, sip_return, int(sip_years))
    pcols = st.columns(3)
    pcols[0].metric("You invest", money(proj.invested))
    pcols[1].metric(f"Projected at {sip_return:.1f}%", money(proj.projected_value))
    pcols[2].metric("Projected gain", money(proj.gain))
    st.caption("Compound-interest arithmetic on a return YOU assumed. Not a prediction, not advice.")

with st.expander("Glossary"):
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}** — {meaning}")

st.caption(DISCLAIMER)
