"""India Equity Research - Streamlit app.

Research-only decision support for a family portfolio: no trades, no buy/sell calls. Every
figure is fetched and timestamped, cross-verified across independent sources or withheld, and
the human expert approves a report before it counts. Built mobile-first (the parents use
iPhones): summary first, plain language, evidence one tap away.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
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
    long_term_guidance,
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
from src.data.figure_sources import (  # noqa: E402
    PERCENT_FIGURES,
    RATIO_FIGURES,
    YFinanceFigureSource,
    format_figure_value,
)
from src.data.news_source import NewsSource, registry_with_news  # noqa: E402
from src.formatting import format_rupees  # noqa: E402
from src.data.nse_annual_reports import (  # noqa: E402
    fetch_annual_report_text,
    nse_annual_report_source,
)
from src.data.screener_source import ScreenerFigureSource  # noqa: E402
from src.data.sheets_backend import (  # noqa: E402
    AppsScriptGateway,
    append_log,
    build_gateway,
    read_holdings,
    read_reports,
    record_from_report,
    resolve_approved_stances,
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
    historical_cagr,
    max_drawdown,
    portfolio_daily_returns,
)
from src.portfolio.loader import load_holdings  # noqa: E402
from src.research.claims import ESTIMATE, FACT, OPINION  # noqa: E402
from src.research.annual_report_reader import read_filing  # noqa: E402
from src.research.grounded_analyst import GroundedAnalyst  # noqa: E402
from src.research.grounding import DocumentStore  # noqa: E402
from src.research.verified_context import (  # noqa: E402
    CASH_CONVERSION_TREND_SOURCE_ID,
    OTHER_INCOME_SHARE_SOURCE_ID,
    PROMOTER_TREND_SOURCE_ID,
    VERIFIED_FIGURES_SOURCE_ID,
    cash_conversion_trend_document,
    other_income_share_document,
    promoter_trend_document,
    symbol_has_no_data,
    verified_figures_document,
)
from src.research.library import (  # noqa: E402
    build_library,
    parse_demo_enabled_secret,
    resolve_curated_library_paths,
)
from src.research.report import ReviewStatus, most_recent_by_symbol  # noqa: E402
from src.sip import sip_future_value  # noqa: E402
from src.sources.adapters import HttpDocumentAdapter, ingest_documents  # noqa: E402
from src.sources.registry import CredibilityTier, Source, SourceRegistry  # noqa: E402
from src.data.amfi_provider import AMFIProvider  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")  # load the app's own .env regardless of cwd

st.set_page_config(page_title="India Equity Research", layout="wide", page_icon="📊",
                   initial_sidebar_state="collapsed")  # collapsed = mobile-first

_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = _ROOT / "sample_data" / "sample_portfolio.csv"
HOLDINGS_CSV = _ROOT / "holdings.csv"   # the owner's real portfolio (gitignored)
EVAL_STORE = _ROOT / "data" / "eval_cases.jsonl"


def _secret(key: str, default=None):
    """Read a Streamlit secret, tolerating no secrets file at all (local dev)."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# Sources/documents: prefer the owner's real config, else fall back to the bundled sample --
# but ONLY when explicitly opted in (demo_sample_library secret). WHY: config/sources.yaml is
# gitignored, so it can never exist in this app's git-based Streamlit Cloud deployment; without
# this gate every deployed session would silently load synthetic sample data as if it were real
# (live-verified real-money risk -- see resolve_curated_library_paths / sample_data/sources.yaml).
SOURCES_YAML, DOCS_DIR = resolve_curated_library_paths(
    _ROOT / "config" / "sources.yaml", _ROOT / "documents",
    _ROOT / "sample_data" / "sources.yaml", _ROOT / "sample_data" / "documents",
    demo_enabled=parse_demo_enabled_secret(_secret("demo_sample_library", False)))

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
def get_grounded_analyst() -> GroundedAnalyst:
    return GroundedAnalyst()


@st.cache_resource
def get_news_source() -> NewsSource:
    return NewsSource()


@st.cache_resource(ttl=86400, show_spinner="Loading AMFI mutual-fund NAVs...")
def get_amfi() -> AMFIProvider | None:
    # WHY: loaded lazily (only when the user searches a fund) so a page load needs no network.
    # WHY ttl=86400: AMFI publishes an updated NAV once per trading day; without a TTL this
    # cache_resource never expires for the LIFETIME of the deployed process (which can run for
    # days/weeks between redeploys on Streamlit Cloud), unlike every other live data source in
    # this app (all use an explicit ttl on st.cache_data). The 'As of' date shown in the results
    # table would then silently fall further and further behind today with no signal beyond that
    # date column. 24h keeps it no more than a day stale, matching AMFI's actual update cadence.
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_long_history_close(symbol: str) -> pd.Series:
    # WHY: a long, slow-changing window for a real historical-CAGR reference (see
    # historical_cagr); cached for a day since decades of daily bars don't need refetching often.
    hist = get_provider().history(symbol, period="max")
    if "Close" in hist:
        return hist["Close"].dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    return get_provider().fundamentals(symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(symbol: str, company_name: str):
    return get_news_source().fetch(symbol, company_name)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ar_text(symbol: str, url: str = ""):
    return fetch_annual_report_text(symbol, url)


@st.cache_resource(ttl=3600)
def get_screener_source() -> ScreenerFigureSource:
    # WHY ttl MUST match fetch_promoter_trend's own ttl below: ScreenerFigureSource memoizes
    # fetched HTML per symbol internally (self._cache, never expires on its own). Without a ttl
    # HERE, this singleton instance -- and its internal cache -- lives for the whole deployed
    # process, so fetch_promoter_trend's ttl=3600 becomes a no-op for any symbol already looked
    # up once: Streamlit re-calls the function every hour, but the SAME long-lived instance just
    # returns its already-cached (possibly days-old) HTML instead of re-fetching. Live-verified:
    # 3 calls to the same symbol over simulated hours produced exactly 1 real fetch. Expiring
    # this resource on the same cadence forces a fresh instance (empty internal cache) each hour,
    # so the outer ttl's freshness guarantee is actually real, not illusory.
    return ScreenerFigureSource()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_promoter_trend(symbol: str):
    return get_screener_source().promoter_holding_trend(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cash_conversion_trend(symbol: str):
    return get_screener_source().cash_conversion_cycle_trend(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_other_income_share(symbol: str):
    return get_screener_source().other_income_share(symbol)


@st.cache_data(ttl=300, show_spinner="Loading holdings from your Sheet…")
def fetch_published_holdings(url: str):
    """Read holdings from a Google Sheet 'Publish to web -> CSV' link. Keyless: the link is
    public, so no service account is needed. Parsing reuses the CSV loader's column matching."""
    import io
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", "replace")
    return load_holdings(io.StringIO(text))


@st.cache_resource
def get_base_registry() -> SourceRegistry | None:
    return SourceRegistry.from_config(SOURCES_YAML) if SOURCES_YAML.exists() else None


def _sheet_configured() -> bool:
    return bool((_secret("apps_script_url") and _secret("apps_script_token"))
                or (_secret("sheet_key") and _secret("gcp_service_account")))


@st.cache_resource
def get_gateway():
    """The persistence backend: the Apps Script web app (keyless, token-gated) if configured,
    else a service-account Sheet, else a local JSON file (gitignored) so approvals still persist
    in dev. All satisfy the same SheetGateway interface."""
    url, token = _secret("apps_script_url"), _secret("apps_script_token")
    if url and token:
        return AppsScriptGateway(url, token)
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


def _bridge_secrets_to_env() -> None:
    """Copy LLM config from Streamlit secrets into env vars. WHY: on Streamlit Cloud the model
    is set in the Secrets UI, but LiteLLMClient reads os.environ; this bridges the two so the
    hosted model (e.g. Groq) works there. No-op locally with no secrets file."""
    for key in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
        value = _secret(key)
        if value and not os.environ.get(key):
            os.environ[key] = str(value)


def _check_password() -> bool:
    """Shared-password gate. Open when no password is configured (local dev); required once a
    password is set in secrets (deployed). Data is only fetched after this returns True.

    Two ways in so non-technical users never retype the password: a `?key=<password>` in the URL
    (their Home-Screen bookmark carries it -> tapping the icon auto-signs-in), or typing it once.
    The bare URL (no key) still shows the prompt, so a stranger with only the base link is blocked.
    """
    expected = _secret("app_password")
    if not expected:
        return True
    if st.session_state.get("_authed"):
        return True
    if str(st.query_params.get("key", "")) == str(expected):   # bookmarked magic-link auto-login
        st.session_state["_authed"] = True
        return True
    st.title("🔒 India Equity Research")
    st.caption("Enter the password to continue.")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pw == expected:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


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
    Registry always includes the live news feeds so news can be ingested as attributed context,
    plus this app's own cross-verified-figures source (see verified_context.py) so Ask can ground
    financial questions. WHY register it HERE and not in the Ask tab: this function is
    @st.cache_resource, so Streamlit's cache lock guarantees the body runs exactly once even under
    concurrent sessions. Registering it as part of that one-time build avoids a real check-then-act
    race on this process-shared registry (two concurrent first Ask requests could otherwise both
    pass a "not yet registered" check before either added it, and the second add() would raise on
    the duplicate id, crashing that user's request)."""
    base = get_base_registry()
    registry = registry_with_news(base)
    registry.add(Source(
        VERIFIED_FIGURES_SOURCE_ID, "This app's cross-verified figures", CredibilityTier.PRIMARY,
        notes="Only figures independently agreed by >=2 public sources (yfinance + Screener); "
              "see the Research tab for the full evidence."))
    registry.add(Source(
        PROMOTER_TREND_SOURCE_ID, "Promoter shareholding trend (Screener)", CredibilityTier.ANALYST,
        notes="Single-source (Screener only), not cross-verified -- reported context, never a "
              "fact, and never a buy/sell signal on its own."))
    registry.add(Source(
        CASH_CONVERSION_TREND_SOURCE_ID, "Cash conversion cycle trend (Screener)",
        CredibilityTier.ANALYST,
        notes="Single-source (Screener only), not cross-verified -- reported context, never a "
              "fact, and never a buy/sell signal on its own."))
    registry.add(Source(
        OTHER_INCOME_SHARE_SOURCE_ID, "Other income share of profit (Screener)",
        CredibilityTier.ANALYST,
        notes="Single-source (Screener's own P&L), not cross-verified -- reported context, "
              "never a fact, and never a buy/sell signal on its own."))
    store = DocumentStore(registry=registry)
    skipped: list[str] = []
    failed: list[str] = []
    if base is not None:
        _, skipped, failed = build_library(base, DOCS_DIR, store=store)
    return registry, store, skipped, failed


def money(value: float | None) -> str:
    # WHY: the parents' own portfolio/allocation amounts read in the Indian convention
    # (₹5,00,000, not Western ₹500,000), consistent with the crore/lakh research figures. Shared
    # formatter so there is one source of truth for how rupees display (see src/formatting.py).
    return format_rupees(value)


def ask_no_figures_tip(symbol: str, already_researched_this_session: bool) -> str:
    """The right guidance when the Ask tab can't ground a numeric question in cross-verified
    figures for `symbol`. WHY (real money, workflow honesty): the tip used to be the SAME
    "research it in the Research tab first" message whether the stock was never researched this
    session at all, OR it WAS researched but simply produced no cross-verified figure (every
    figure single-source, in genuine CONFLICT across sources, or found by NEITHER source at all)
    -- verified_figures_document returns None in all these cases, so vf_doc is None can't tell
    them apart on its own. Telling a user who already researched the stock to "research it first"
    is a false claim about what they just did, and re-researching will not resolve a genuine
    cross-source disagreement between yfinance and Screener -- the fix there is to open the
    evidence panel, not click Research again."""
    if not already_researched_this_session:
        return (f"Tip: for questions about {symbol}'s numbers (P/E, debt, profit, dividend...), "
                "research it in the 'Research a Stock' tab first — Ask can then ground answers "
                "in its cross-verified figures.")
    return (f"{symbol} was already researched this session, but no figure cross-verified across "
            "sources (each is either single-source, a genuine conflict, or not found by any "
            "source at all). Re-researching won't resolve a real disagreement or a genuine gap "
            "in coverage -- open the Research tab's evidence panel to see which figures conflict, "
            "are single-source, or are simply unavailable.")


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


_STANCE_PDF = {Stance.FAVORABLE: "[+] ", Stance.NEUTRAL: "[~] ",
               Stance.UNFAVORABLE: "[-] ", Stance.INSUFFICIENT_DATA: "[?] "}


def build_pdf_report(title: str, report, stance: Stance, guidance=None,
                     promoter_trend: str | None = None,
                     cash_conversion_trend: str | None = None,
                     other_income_share: str | None = None) -> bytes:
    """A downloadable PDF of the report, mirroring what is on screen. Uses the core Helvetica
    font (Latin-1), so text is sanitized and the rupee sign is written as 'Rs.'.

    WHY (real money, UI honesty): promoter_trend/cash_conversion_trend/other_income_share are
    the same single-source (Screener-only) context signals the Research tab shows in their own
    expanders -- this button is labeled "Download full report", so a parent who saves this PDF
    to review offline, or share with family, must see the SAME signals the live app shows them,
    not a report that is silently missing three of the app's own research signals.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    def s(text) -> str:
        return str(text).replace("₹", "Rs.").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text: str, size: int = 11, style: str = "", h: float = 6, gap: float = 0):
        # WHY: new_x=LMARGIN resets the cursor to the left each line; fpdf's default leaves it at
        # the right margin, so the next multi_cell(0) would compute zero width and raise.
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, h, s(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if gap:
            pdf.ln(gap)

    v = report.verdict
    line(title, size=15, style="B", h=8)
    status = ("APPROVED (reviewed)" if report.is_trusted
              else "REJECTED, not for decisions" if report.status == ReviewStatus.REJECTED
              else "DRAFT, not yet reviewed by your expert")
    pdf.set_text_color(90, 90, 90)
    line(f"Status: {status}    Generated: {report.created_at}", size=10, gap=2)
    pdf.set_text_color(0, 0, 0)

    _, headline = _STANCE_UI[stance]
    line(_STANCE_PDF[stance] + headline, size=13, style="B", h=8)
    line(plain_summary(v, stance), size=11, gap=2)

    if report.insights:
        line("Why, in plain terms", size=12, style="B", h=7)
        for point in report.insights:
            line(f"- {point}", size=11)
        pdf.ln(1)

    if guidance is not None:
        line(f"For a long-term investor: {guidance.headline}", size=12, style="B", h=7)
        for point in guidance.points:
            line(f"- {point}", size=11)
        pdf.ln(1)

    if v is not None:
        line("Verdict", size=12, style="B", h=7)
        line(f"Valuation: {v.valuation.value}    Quality: {v.quality.value}    "
             f"Leaning: {v.leaning.value}    Confidence: {v.confidence.value}", gap=1)
        if v.reasons:
            line("Why (each from cross-verified figures)", size=12, style="B", h=7)
            for reason in v.reasons:
                line(f"- {reason}")
            pdf.ln(1)
        if v.sector_caveats:
            line("Sector context", size=12, style="B", h=7)
            for caveat in v.sector_caveats:
                line(f"- {caveat}")
            pdf.ln(1)

    line("Figures", size=12, style="B", h=7)
    for f in report.figures:
        # format_figure_value shows the figure in its actual unit (ratio/percent/rupees), not a
        # bare number that could read as rupees for a ratio or percentage figure.
        val = format_figure_value(f.name, f.value) if f.value is not None else "withheld (not cross-verified)"
        srcs = ", ".join(sorted({sv.source_id for sv in f.sources}))
        period = next((str(sv.locator) for sv in f.sources
                       if str(getattr(sv, "locator", "") or "").upper().startswith("FY")), "current")
        line(f"- {f.name}: {val}  [{period}; {f.status.value}; {srcs}]", size=10, h=5)

    single_source_points = [p for p in
                           (promoter_trend, cash_conversion_trend, other_income_share) if p]
    if single_source_points:
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)
        line("Additional context (single-source, not cross-verified)", size=12, style="B", h=7)
        for point in single_source_points:
            line(f"- {point}", size=10, h=5)
        pdf.set_text_color(90, 90, 90)
        line("Screener-only signals (not in yfinance), so they cannot cross-verify the way the "
             "figures above do. Context, not a fact, and never a buy/sell signal on their own.",
             size=9, style="I", h=5)

    if v is not None:
        pdf.ln(2)
        pdf.set_text_color(90, 90, 90)
        line(v.caveat, size=9, style="I", h=5)
    return bytes(pdf.output())


# --- auth + hosted-model secrets (both no-ops locally with no secrets file) ---

_bridge_secrets_to_env()
if not _check_password():
    st.stop()

# --- header ---

st.title("📊 India Equity Research")
st.caption("Understand your investments.")

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
    # WHY: reuse the already-cached GroundedAnalyst's client for this status check instead of a
    # separate LLM-wrapper class -- this app has exactly one rigorous LLM research path
    # (GroundedAnalyst: structural citation-tier + numeric-grounding checks, never trusting the
    # model's output as-is). A prior, weaker ResearchAnalyst class (prompt-only guardrails, no
    # structural validation) existed only for this trivial availability check and was removed.
    grounded_status = get_grounded_analyst()
    if grounded_status.available:
        st.success(f"AI research: on ({grounded_status.client.model_name})")
    else:
        st.info("AI research: off. Set LLM_MODEL to enable the annual-report tiebreaker and the "
                "research chat. The cross-verified analysis works without it.")

# --- resolve the data source (Google Sheet if selected, else CSV) ---

holdings = None
pub_url = _secret("holdings_csv_url")
if pub_url and uploaded is None:                      # published-CSV link (keyless, auto-load)
    try:
        holdings = fetch_published_holdings(pub_url) or None
    except Exception as exc:
        st.error(f"Could not read the published Sheet CSV: {exc}")

if holdings is None and sheet_on and use_sheet:       # service-account path (if ever configured)
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
# WHY: analyze_portfolio merges multiple lots of the same symbol into one position (see
# _merge_lots), so len(analysis.positions) can be LESS than len(holdings) even when every row
# priced successfully -- comparing against the raw row count would wrongly read as "N holdings
# didn't price" when really rows just merged. Compare against distinct symbols instead.
distinct_holding_symbols = len({h.symbol for h in holdings}) if holdings else 0

if "reports" not in st.session_state:
    st.session_state.reports = {}

# WHY: short labels so all four tabs fit on an iPhone (375px) without horizontal scroll —
# a parent must see Invest/Ask exist, not have them clipped off-screen.
tab_portfolio, tab_research, tab_invest, tab_ask = st.tabs(
    ["📁 Portfolio", "🔎 Research", "💰 Invest", "💬 Ask"])


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
    m2[1].metric("Holdings priced", f"{len(analysis.positions)} / {distinct_holding_symbols}")

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
        # WHY (honesty): weights/HHI are normalized ONLY over the priced positions
        # (analyze_portfolio's `usable` filter); a missing-price name is silently excluded, not
        # treated as zero. If several holdings fail to price (e.g. a temporary data-source issue),
        # the concentration reading becomes an artifact of whichever subset happened to price, not
        # the real full portfolio, and could over- or under-state concentration risk with no
        # caveat at the point of the warning below. Surface that explicitly when it applies.
        if holdings and len(analysis.positions) < distinct_holding_symbols:
            st.caption(f"Based on the {len(analysis.positions)} of {distinct_holding_symbols} "
                       f"holdings that priced; {distinct_holding_symbols - len(analysis.positions)} "
                       "missing name(s) are excluded, so this does not reflect your full "
                       "portfolio.")
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
            # WHY: on a cloud IP yfinance history can come back empty; the risk fns then return
            # 0.0, which would read as a real "0.00 beta / 0.0% volatility". Say "no history"
            # instead of showing a fabricated zero (real-money display guardrail).
            if port_returns.empty or not any(not c.empty for c in close_by_symbol.values()):
                st.warning("Couldn't fetch enough 1-year price history to compute risk right now. "
                           "Try again later.")
            else:
                # WHY (honesty): portfolio_daily_returns renormalizes weights over only the
                # symbols that returned usable 1-year history (by design, so a missing name
                # isn't silently scored as zero return) -- but that means volatility/beta can be
                # driven by a small subset while still being LABELED as "your portfolio's" risk.
                # Demonstrated: 2 of 3 equally-weighted positions failing to fetch history yields
                # a volatility reading 100% derived from the one remaining name. Name the actual
                # coverage so a thin sample never reads as a full-book risk assessment.
                n_with_history = sum(1 for c in close_by_symbol.values() if not c.empty)
                if n_with_history < len(analysis.positions):
                    st.caption(f"Based on the {n_with_history} of {len(analysis.positions)} "
                               "priced holdings that had usable 1-year history; the rest are "
                               "excluded here (weights renormalized over what's available), so "
                               "this may not reflect your full portfolio's risk.")
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

        # WHY (real money, workflow): distinct from ordinary thin coverage -- when EVERY figure
        # is unverifiable (or there are none at all), that usually means the symbol itself is
        # wrong, not that the business has weak disclosure. Live-verified: Page Industries trades
        # as PAGEIND, not PAGE; typing the natural/common name "PAGE" returns zero data from
        # either source with no other signal why. Give an actionable hint instead of leaving the
        # generic "insufficient data" message to look identical for a real company with
        # genuinely poor data, which offers no way to tell the two situations apart.
        if report.no_data_found:
            st.warning(f"No data at all was found for '{sym}' from either source. This usually "
                       "means the exact NSE trading symbol differs from the company's common "
                       "name (e.g. Page Industries trades as PAGEIND, not PAGE). Double-check "
                       "the exact symbol on NSE, BSE, or Screener.in, then try again.")

        # status banner
        if report.is_trusted:
            last = report.audit[-1]
            st.success(f"APPROVED by {last.reviewer}. Reviewed.")
        elif report.status == ReviewStatus.REJECTED:
            st.error("REJECTED, sent back for correction. Not for decisions.")
        else:
            st.warning("DRAFT, not yet reviewed by your expert. Not for decisions.")

        # summary-first: one line + stance + the 5-6 plain-language reasons
        st.markdown(f"### {icon} {sym}: {headline}")
        st.write(plain_summary(report.verdict, stance))
        if report.insights:
            st.markdown("**Why, in plain terms:**")
            for point in report.insights:
                st.markdown(f"- {point}")

        # what to do — long-term hold/trim/accumulate guidance with thesis-based triggers
        held_value = value_by_symbol.get(sym, 0.0)
        sizing = position_sizing(held_value, analysis.total_value, cap_pct)
        guidance = long_term_guidance(stance, sizing, report.verdict, held=held_value > 0)
        st.info(f"**For a long-term investor: {guidance.headline}**")
        for point in guidance.points:
            st.markdown(f"- {point}")
        if held_value > 0:
            st.caption(f"You currently hold {money(held_value)} in {sym}; your "
                       f"{cap_pct:.0%} per-stock cap is {money(sizing.cap_value)}.")

        st.caption(report.verdict.caveat if report.verdict else DISCLAIMER)

        # download (PDF) -- same cached, already-fetched signals the expanders below show live,
        # so the "full report" PDF a parent saves offline isn't silently missing them.
        st.download_button(
            "⬇️ Download full report (PDF)",
            data=build_pdf_report(active, report, stance, guidance,
                                  promoter_trend=fetch_promoter_trend(sym),
                                  cash_conversion_trend=fetch_cash_conversion_trend(sym),
                                  other_income_share=fetch_other_income_share(sym)),
            file_name=f"{sym}_research.pdf", mime="application/pdf")

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
                if report.verdict.sector_caveats:
                    st.markdown("**Sector context:**")
                    for caveat in report.verdict.sector_caveats:
                        st.markdown(f"- {caveat}")
            def _period(fig):
                # WHY: surface the fiscal year the figure is for (in the source locators, e.g.
                # "FY2024"), so a prior-year figure isn't read as current. Point figures = current.
                for sv in fig.sources:
                    loc = str(getattr(sv, "locator", "") or "")
                    if loc.upper().startswith("FY"):
                        return loc
                return "current"

            fig_rows = [{
                "Figure": f.name, "Status": f.status.value,
                # WHY: all-string column; a mixed float/"withheld" column crashes st.dataframe.
                # format_figure_value shows the figure in its ACTUAL unit (ratio/percent/rupees):
                # a bare "25.00" is genuinely ambiguous between a 25% pledge and Rs.25.
                "Value": (format_figure_value(f.name, f.value) if f.value is not None
                         else "withheld"),
                "Period": _period(f),
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
            # WHY: pass the RAW resolved name (never falling back to the bare symbol) -- some
            # real NSE tickers are common English words (PAGE, IDEA, SAIL, RAIN), so searching
            # news for the bare ticker when the name can't be resolved pulls in unrelated results
            # (live-verified). NewsSource itself skips the Google search when no name is given.
            company = fetch_fundamentals(sym).get("name") or ""
            items = fetch_news(sym, company)
            if not items:
                st.caption("No recent news found (or the feed was unreachable).")
            for it in items:
                headline_md = f"[{it.title}]({it.url})" if it.url else it.title
                st.markdown(f"- {headline_md}  \n  _{it.publisher or 'source'}, "
                            f"{it.published or 'undated'}_")
            st.caption("News is reporting, attributed and dated. It is NOT cross-verified like a "
                       "figure and does not move the verdict above.")

        # promoter shareholding trend (Screener only, single-source by nature; a well-known
        # Indian-investor signal, kept clearly separate from the cross-verified figures above)
        with st.expander("Promoter shareholding trend (context, not cross-verified)"):
            trend = fetch_promoter_trend(sym)
            if trend:
                st.markdown(f"- {trend}")
            else:
                st.caption("No shareholding-pattern data found (or the page was unreachable).")
            st.caption("Shareholding data is published only by Screener (not yfinance), so it "
                       "cannot cross-verify the way the figures above do. Context, not a fact, "
                       "and never a buy/sell signal on its own.")

        # cash conversion cycle trend (Screener only, single-source; a cash-flow-discipline /
        # quality-of-earnings signal -- a lengthening cycle can flag slower collections or rising
        # inventory well before it shows up in reported profit)
        with st.expander("Cash conversion cycle trend (context, not cross-verified)"):
            cc_trend = fetch_cash_conversion_trend(sym)
            if cc_trend:
                st.markdown(f"- {cc_trend}")
            else:
                st.caption("No cash-conversion-cycle data found (or the page was unreachable).")
            st.caption("This ratio is published only by Screener (not yfinance), so it cannot "
                       "cross-verify the way the figures above do. Context, not a fact, and "
                       "never a buy/sell signal on its own.")

        # other income share of profit before tax (Screener only, single-source; a quality-of-
        # earnings signal -- profit propped up by non-operating income is less repeatable than
        # profit driven by the core business)
        with st.expander("Other income share of profit (context, not cross-verified)"):
            oi_share = fetch_other_income_share(sym)
            if oi_share:
                st.markdown(f"- {oi_share}")
            else:
                st.caption("No other-income data found (or the page was unreachable).")
            st.caption("This ratio is computed from Screener's own P&L (not yfinance), so it "
                       "cannot cross-verify the way the figures above do. Context, not a fact, "
                       "and never a buy/sell signal on its own.")

        # grounded annual-report reading (cited to the filing; abstains if it can't read it)
        with st.expander("What the annual report says (read by AI, cited to the filing)"):
            if not LiteLLMClient().available:
                st.caption("Set the AI model to read the annual report.")
            else:
                ar_url = st.text_input("Annual report PDF URL (optional; blank = auto-fetch "
                                       "from NSE)", key=f"arurl_{active}")
                if st.button("Read the annual report", key=f"arread_{active}"):
                    with st.spinner("Fetching and reading the filing…"):
                        text = fetch_ar_text(sym, ar_url.strip())
                        readings = read_filing(text, LiteLLMClient()) if text else []
                        st.session_state.setdefault("ar_readings", {})[active] = (bool(text), readings)
                cached_reading = st.session_state.get("ar_readings", {}).get(active)
                if cached_reading is not None:
                    had_text, readings = cached_reading
                    if not had_text:
                        st.warning("Couldn't read the filing (the report source may block cloud "
                                   "servers). Paste the report's PDF URL above and try again.")
                    for fr in readings:
                        st.markdown(f"**{fr.topic}**")
                        if fr.result.abstained:
                            st.caption("Nothing citable found in the filing for this.")
                        else:
                            for claim in fr.result.claims:
                                # WHY: a filing point is the company's OWN statement, grounded in
                                # the text but self-reported. Use a document marker, NOT the ✓ this
                                # app reserves for cross-verified figures, so parents don't read a
                                # management assertion as independently verified.
                                mark = "📄" if claim.is_verified_fact else "•"
                                st.markdown(f"- {mark} {claim.text}")
                    if readings:
                        st.caption("📄 = the company's own statement in its filing (a primary "
                                   "source), quoted and cited, but self-reported, NOT independently "
                                   "cross-verified like the figures above. Anything not in the "
                                   "filing is left out, not guessed.")

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
            # WHY: the label/step/format must match the SELECTED figure's actual unit. current_pe/
            # median_pe are ratios (e.g. 22.5x) and promoter_pledge_pct/dividend_yield_pct are
            # percentages (e.g. 5.2%), not rupees -- a reviewer entering a correction for one of
            # those trusting a blanket "(absolute rupees)" label could record a wrongly-scaled
            # ground truth into the very mechanism meant to catch the system being wrong.
            if gt_fig in RATIO_FIGURES:
                gt_label, gt_step, gt_fmt = "Correct value (e.g. 22.5 for 22.5x)", 0.1, "%.2f"
            elif gt_fig in PERCENT_FIGURES:
                gt_label, gt_step, gt_fmt = "Correct value (%, e.g. 5.2 for 5.2%)", 0.1, "%.2f"
            else:
                gt_label, gt_step, gt_fmt = "Correct value (absolute rupees)", 1.0, "%.0f"
            gt_val = st.number_input(gt_label, value=0.0, step=gt_step, format=gt_fmt,
                                     key=f"gtv_{active}")
            if st.button("Save correction", key=f"gts_{active}"):
                # WHY: guard the default 0.0 — an accidental save would record a bogus correct
                # value and create a permanent spurious 'trusted-wrong' in the must-be-0 metric.
                if not reviewer.strip():
                    st.error("Enter your reviewer name first.")
                elif gt_val == 0:
                    st.error("Enter the correct value (not 0) before saving.")
                else:
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

    # Today's picks: shown instantly from the Sheet; refreshed on demand by the button below.
    # The refresh runs from THIS app (Streamlit Cloud can reach Screener; a scheduler's datacenter
    # IP can't), so it's full cross-verification. Reading the tab is one fast call.
    if "today_rows" not in st.session_state:
        try:
            st.session_state.today_rows = get_gateway().read("Today")
        except Exception:
            st.session_state.today_rows = []
    today_rows = st.session_state.today_rows
    if today_rows:
        st.markdown(f"**📌 Today's long-term picks** ({len(today_rows)})")
        for r in today_rows:
            st.markdown(f"- **{r.get('symbol', '')}** ({r.get('stance', '')}) — "
                        f"{r.get('reason', '')}")
        # WHY: Sheets coerces the date string into a datetime on round-trip; show only the date.
        as_of = str(today_rows[0].get("date", "")).split("T")[0]
        st.caption(f"As of {as_of}. Cross-verified and within your per-stock cap. Not a buy order.")
    else:
        st.caption("No picks yet. They're prepared automatically each day.")
    # WHY: display-only reload, NOT a re-research. The batch runs on the owner's Mac (residential
    # IP, full cross-verification); running it here from the datacenter IP would come back thin and
    # overwrite the good picks. So the app just pulls the latest the Mac computed.
    if _sheet_configured() and st.button("🔄 Reload latest picks"):
        # WHY: only rerun on success; an unconditional st.rerun() after st.error() immediately
        # wipes the error, so a failed reload would look like the button did nothing.
        try:
            st.session_state.today_rows = get_gateway().read("Today")
            st.rerun()
        except Exception as exc:
            st.error(f"Couldn't reload right now: {exc}")
    st.divider()

    st.caption("Given an amount, this spreads it across your APPROVED names that the evidence "
               "supports, each kept under your per-stock cap. Not a buy order, you decide.")

    amount = st.number_input(f"Amount to invest ({CURRENCY})", min_value=0, value=0, step=50000)

    # Approved names = persisted (durable, from the Sheet/local store) + this session's fresher
    # research merged by resolve_approved_stances: a symbol re-researched THIS session that is
    # NOT (yet) re-approved supersedes and clears an older persisted approval, rather than
    # leaving a stale approval silently feeding suggest_allocation's real rupee math.
    try:
        persisted_records = read_reports(get_gateway())
    except Exception:
        persisted_records = []
    approved_stance = resolve_approved_stances(persisted_records, st.session_state.reports)

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
    st.caption("Answered from recent news (attributed, dated context), any curated sources, and "
               "this app's own cross-verified figures for a stock you've researched this session. "
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
        pinned_source_ids: set[str] = set()
        vf_doc = None
        pt_doc = None
        cc_doc = None
        oi_doc = None
        sym_u = ""
        company = ""
        if ask_sym.strip():
            sym_u = ask_sym.strip().upper()
            # WHY: pass the RAW resolved name, never falling back to the bare symbol -- some
            # real NSE tickers are common English words (PAGE, IDEA, SAIL, RAIN), so searching
            # news for the bare ticker when the name can't be resolved pulls in unrelated results
            # (live-verified). NewsSource itself skips the Google search when no name is given.
            # An empty `company` here is also reused below as a cheap (yfinance-only, no extra
            # Screener load) signal that the symbol itself may not exist at all.
            company = fetch_fundamentals(sym_u).get("name") or ""
            with st.spinner("Reading recent news..."):
                items = fetch_news(sym_u, company)
                ingest_documents(store, NewsSource.as_documents(items))

            # WHY (consolidated after 4 near-identical copies): each of these is a small,
            # authoritative, single-chunk document that must be BOTH ingested AND pinned --
            # without pinning, live-verified repeatedly (verified figures, promoter trend, cash
            # conversion cycle, other income share) that a realistic question's one-sentence
            # chunk can score below the retrieval floor, crowded out by news chunks that merely
            # repeat the company name, so the document this app already fetched would silently
            # never reach the model. Promoter trend was shipped unpinned once and needed a
            # follow-up fix; folding "ingest + pin" into one call makes forgetting the pin step
            # for a future signal much harder, not just something to remember to copy correctly.
            def _ingest_and_pin(doc, source_id: str) -> None:
                if doc is not None:
                    ingest_documents(store, [doc])
                    pinned_source_ids.add(source_id)

            # Reuse this session's already-researched report (if any) so Ask can ground financial
            # questions in the SAME cross-verified figures the Research tab computed, not just
            # news/curated docs (previously the richest data in the app was invisible to Ask).
            # most_recent_by_symbol picks by actual timestamp, not dict-iteration position, so a
            # stale, differently-labeled report can't be grounded as if it were freshly researched.
            cached_report = most_recent_by_symbol(st.session_state.reports, sym_u)
            vf_doc = verified_figures_document(sym_u, cached_report)
            _ingest_and_pin(vf_doc, VERIFIED_FIGURES_SOURCE_ID)
            # WHY: promoter shareholding, cash conversion cycle, and other income share are all
            # core Indian-investor / CA-level signals the Research tab already fetches (single
            # cached Screener page each, same calls the Research tab's own expanders make live) --
            # no heavier than the fetch_fundamentals/fetch_news calls above, unlike a full
            # re-research which Ask deliberately never triggers.
            pt_doc = promoter_trend_document(sym_u, fetch_promoter_trend(sym_u))
            _ingest_and_pin(pt_doc, PROMOTER_TREND_SOURCE_ID)
            cc_doc = cash_conversion_trend_document(sym_u, fetch_cash_conversion_trend(sym_u))
            _ingest_and_pin(cc_doc, CASH_CONVERSION_TREND_SOURCE_ID)
            oi_doc = other_income_share_document(sym_u, fetch_other_income_share(sym_u))
            _ingest_and_pin(oi_doc, OTHER_INCOME_SHARE_SOURCE_ID)
        # WHY (real money, workflow): distinct from "haven't researched yet". `company` alone
        # (yfinance's own name lookup) is a WEAKER signal than Report.no_data_found (which spans
        # figures from both yfinance AND Screener) -- a real, valid symbol can have yfinance's
        # name lookup come back empty (a known Yahoo India-coverage gap) while Screener still has
        # real data. symbol_has_no_data widens the check to all five independent "this symbol is
        # real" signals (name, cross-verified figures, promoter trend, cash conversion cycle,
        # other income share) so this hint is never shown in the same response where real per-
        # symbol data was just fetched and used. Live-verified root cause: Page Industries trades
        # as PAGEIND, not PAGE; typing the natural/common name resolves to nothing from ANY of the
        # five signals. Telling the user to "research it first" would be actively misleading here
        # -- doing so with the SAME wrong symbol fails there too, for the same reason.
        symbol_unresolved = bool(sym_u) and symbol_has_no_data(
            company, vf_doc is not None, pt_doc is not None, cc_doc is not None,
            oi_doc is not None)
        wrong_symbol_hint = (
            f"'{sym_u}' didn't resolve to any company data. This usually means the exact NSE "
            "trading symbol differs from the company's common name (e.g. Page Industries trades "
            "as PAGEIND, not PAGE). Double-check the exact symbol on NSE, BSE, or Screener.in, "
            "then try again.")
        if len(store) == 0:
            if symbol_unresolved:
                st.warning(wrong_symbol_hint)
            else:
                st.warning("No sources to answer from. Enter a stock symbol so recent news can load.")
        else:
            with st.spinner("Reading the sources..."):
                result = grounded.answer(question, store, registry,
                                         pin_source_ids=frozenset(pinned_source_ids),
                                         as_of=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if result.abstained:
                st.warning(f"No verified answer. {result.abstain_reason}")
                # WHY (workflow discoverability): Ask only grounds financial-figure questions in
                # a symbol's cross-verified figures if that stock was already researched this
                # session (verified_figures_document needs >=2-source-verified figures; a fresh
                # fetch here would add MORE Screener load to the deployed app's already
                # rate-limited datacenter IP, on every random Ask query, degrading the Research
                # tab and daily picks for everyone -- so Ask deliberately reuses, never triggers,
                # a live fetch). Point the user at the fix instead of leaving a dead end.
                if symbol_unresolved:
                    st.caption(wrong_symbol_hint)
                elif sym_u and vf_doc is None:
                    st.caption(ask_no_figures_tip(sym_u, cached_report is not None))
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
                        # UNVERIFIED: a claim downgraded either for lacking a primary source, or
                        # for stating a number absent from its cited source (a misquote -- applies
                        # to both FACT and OPINION now). If it's from news/analyst, it's
                        # reporting/context (honest, not alarming); only a claim resting solely on
                        # primary sources (so the issue is a fabricated/misquoted figure, not the
                        # tier) gets the hard warning.
                        from_primary_only_source = all(
                            registry.get(c.source_id) and registry.get(c.source_id).citable_as_fact
                            for c in claim.citations)
                        if from_primary_only_source:
                            st.error(f"⚠ Unverified: {claim.text}")
                        else:
                            st.info(f"Reported, not independently verified: {claim.text}")
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
    # WHY (honesty): the slider allows up to 30%/40 years, which compounds into an absurd,
    # misleading corpus if taken literally (a real risk for a non-expert reading a bare number).
    # Ground the assumption against SENSEX's own real, live-computed long-term price return, not
    # a "typical equity return" claim from memory, so the reader can judge how aggressive their
    # assumption is against actual history rather than an arbitrary cap.
    bench_hist = historical_cagr(fetch_long_history_close(SENSEX_SYMBOL))
    if bench_hist is not None:
        bench_cagr, bench_years = bench_hist
        note = ("well above" if sip_return > bench_cagr + 3 else
                "well below" if sip_return < bench_cagr - 3 else "in line with")
        st.caption(f"For context: SENSEX's own price return over the last {bench_years:.0f} "
                   f"years (live data, price only, excludes dividends) works out to about "
                   f"{bench_cagr:.1f}%/yr. Your {sip_return:.1f}% assumption is {note} that; "
                   "real fund returns vary year to year and can be negative.")

with st.expander("Glossary"):
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}** — {meaning}")
