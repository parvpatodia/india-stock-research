"""India Equity Research - Streamlit app.

Research-only decision support for a family portfolio: no trades, no buy/sell calls. Every
figure is fetched and timestamped, cross-verified across independent sources or withheld, and
the human expert approves a report before it counts. Built mobile-first (the parents use
iPhones): summary first, plain language, evidence one tap away.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
import re
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
    AI_DISCLOSURE,
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
    NseAnnualReportResolver,
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
    sector_concentration_note,
    thin_risk_window_note,
)
from src.portfolio.loader import load_holdings, normalize_symbol  # noqa: E402
from src.research.claims import ESTIMATE, FACT, OPINION  # noqa: E402
from src.research.annual_report_reader import read_filing  # noqa: E402
from src.research.grounded_analyst import GroundedAnalyst  # noqa: E402
from src.research.grounding import DocumentStore  # noqa: E402
from src.research.orchestrator import ResearchOrchestrator  # noqa: E402
from src.freshness.staleness import describe_freshness, freshness  # noqa: E402
from src.research.verified_context import (  # noqa: E402
    CASH_CONVERSION_TREND_SOURCE_ID,
    OTHER_INCOME_SHARE_SOURCE_ID,
    PROMOTER_PLEDGE_SOURCE_ID,
    PROMOTER_TREND_SOURCE_ID,
    VERIFIED_FIGURES_SOURCE_ID,
    cash_conversion_trend_document,
    other_income_share_document,
    promoter_pledge_document,
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
from src.sip import (  # noqa: E402
    DEFAULT_INFLATION_PCT,
    real_value,
    sip_future_value,
    sip_return_context,
)
from src.sources.adapters import HttpDocumentAdapter, ingest_documents  # noqa: E402
from src.sources.registry import CredibilityTier, Source, SourceRegistry  # noqa: E402
from src.data.amfi_provider import AMFIProvider  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")  # load the app's own .env regardless of cwd

st.set_page_config(page_title="India Equity Research", layout="wide", page_icon="📊",
                   initial_sidebar_state="collapsed")  # collapsed = mobile-first

# --- W7 visual system (increment 2): a calm, premium fintech look for non-expert parents. ---
# DESIGN CONTRACT (real money, live app): styling must NEVER break rendering or hide a feature.
#   * The .streamlit/config.toml [theme] block is the STABLE backbone (colours, font). If the CSS
#     below silently no-ops (a Streamlit DOM class changed across versions), the app still renders
#     with the full theme -- just without the extra polish. Nothing here is load-bearing for logic.
#   * The CSS is SCOPED and DEFENSIVE: it adds our own `.ier-*` classes and only touches a few
#     stable Streamlit hooks with SAFE properties (spacing, radius, colour). It never sets
#     display/visibility/height on a Streamlit container, so a bad match can't blank the page.
#   * Dark mode is handled WITHOUT hardcoding text/background: our cards use `color: inherit` (text
#     follows Streamlit's active light/dark text colour) over a neutral translucent `rgba()` wash
#     that reads as a raised surface on ANY background. Streamlit 1.58 does not expose its theme as
#     CSS variables, so we deliberately avoid `var(--...)` guesses. gain/loss accents are chosen to
#     stay legible on both light and dark.
_IER_CSS = """
<style>
/* Roomier, product-like layout; cap width on large desktops so it reads as an app, not a script.
   Mobile (< 640px) keeps the full width -- the cap only kicks in far above 375px. */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.01em; }
h1 { font-weight: 700; }
h2, h3 { font-weight: 650; }
/* Gentle radius on the interactive chrome (safe no-ops if a class ever changes). */
.stButton > button, .stDownloadButton > button, div[data-baseweb="input"] input,
.stTextInput input, .stNumberInput input { border-radius: 10px; }
.stButton > button { font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }

/* Hero: reads as a product header, not an st.title. */
.ier-hero { padding: 0.2rem 0 0.6rem 0; }
.ier-hero .ier-title { font-size: 2.0rem; font-weight: 750; letter-spacing: -0.02em;
    line-height: 1.15; margin: 0; }
.ier-hero .ier-sub { font-size: 1.02rem; opacity: 0.72; margin: 0.35rem 0 0 0; line-height: 1.4; }
.ier-hero .ier-badge { display: inline-block; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase; padding: 0.18rem 0.55rem; border-radius: 999px;
    background: rgba(46,90,172,0.12); color: #2E5AAC; margin-bottom: 0.55rem; }

/* Premium metric tiles. Theme-agnostic: translucent surface + inherited text colour. */
.ier-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 0.2rem 0 0.4rem 0; }
.ier-metric { background: rgba(128,132,150,0.09); border: 1px solid rgba(128,132,150,0.20);
    border-radius: 14px; padding: 0.85rem 0.95rem; color: inherit; min-width: 0; }
.ier-metric .ier-lbl { font-size: 0.76rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; opacity: 0.62; margin: 0 0 0.25rem 0; }
.ier-metric .ier-val { font-size: 1.5rem; font-weight: 700; line-height: 1.15;
    letter-spacing: -0.01em; overflow-wrap: anywhere; }
.ier-metric .ier-delta { font-size: 0.9rem; font-weight: 650; margin-top: 0.2rem; }
.ier-metric .ier-delta.gain { color: #159457; }
.ier-metric .ier-delta.loss { color: #E0483B; }

/* Onboarding / empty-state callout. Same theme-agnostic surface trick. */
.ier-note { background: rgba(46,90,172,0.07); border: 1px solid rgba(46,90,172,0.20);
    border-left: 3px solid #2E5AAC; border-radius: 12px; padding: 0.9rem 1.05rem; color: inherit;
    line-height: 1.5; }
.ier-note .ier-note-title { font-weight: 700; margin-bottom: 0.3rem; }
.ier-note ul { margin: 0.4rem 0 0 0; padding-left: 1.1rem; }
.ier-note li { margin: 0.15rem 0; }
.ier-chip { display: inline-block; font-weight: 700; font-size: 0.82rem; padding: 0.02rem 0.4rem;
    border-radius: 6px; }
.ier-chip.g { background: rgba(21,148,87,0.16); color: #159457; }
.ier-chip.o { background: rgba(224,146,20,0.18); color: #B9770F; }
.ier-chip.r { background: rgba(224,72,59,0.16); color: #E0483B; }
.ier-chip.n { background: rgba(128,132,150,0.18); color: inherit; opacity: 0.85; }

/* Mobile-first: iPhone (~375px). Stack the tiles 2-up and dial the hero down. */
@media (max-width: 640px) {
  .ier-metrics { grid-template-columns: repeat(2, 1fr); gap: 0.55rem; }
  .ier-metric .ier-val { font-size: 1.28rem; }
  .ier-hero .ier-title { font-size: 1.6rem; }
  /* clear Streamlit's fixed top toolbar on mobile (the hero badge tucked under it at a tighter
     value); the desktop rule above is fine as-is. */
  .block-container { padding-top: 3rem; }
}

/* Dark mode: config.toml's [theme.dark] follows the device preference, so this media query tracks
   the same signal and brightens our accent colours for legibility on a dark surface. The
   translucent card/note surfaces and `color: inherit` already adapt on their own; only the fixed
   accent hues need the lift. If a user forces dark against a light OS, the light accents below are
   still chosen to read on dark -- this only makes the common (OS-dark) case pop. */
@media (prefers-color-scheme: dark) {
  .ier-metric .ier-delta.gain { color: #34D399; }
  .ier-metric .ier-delta.loss { color: #F87171; }
  .ier-hero .ier-badge { color: #9DBAF2; background: rgba(122,162,232,0.16); }
  .ier-note { background: rgba(122,162,232,0.10); border-color: rgba(122,162,232,0.28);
      border-left-color: #7AA2E8; }
  .ier-chip.g { color: #34D399; background: rgba(52,211,153,0.18); }
  .ier-chip.o { color: #FBBF77; background: rgba(251,146,60,0.20); }
  .ier-chip.r { color: #F87171; background: rgba(248,113,113,0.18); }
}
</style>
"""


def inject_theme_css() -> None:
    """Inject the visual-system CSS. Called once near the top of EVERY run -- Streamlit rebuilds the
    DOM on each rerun, so a `st.markdown` <style> must be re-emitted every time or the styling would
    vanish after the first interaction (no session_state 'once' guard, on purpose). Guarded so a
    styling failure can never crash the page: on any error we skip the polish and fall back to the
    config.toml [theme]."""
    try:
        st.markdown(_IER_CSS, unsafe_allow_html=True)
    except Exception:  # pragma: no cover - styling must never break the app
        pass


def _metric_tile_html(label: str, value: str, delta: str | None = None,
                      tone: str | None = None, title: str | None = None) -> str:
    """One premium metric tile (label / big value / optional coloured delta). All text is escaped;
    `tone` is 'gain' | 'loss' | None. `title` becomes a hover tooltip (preserves a metric's help
    text when we render a custom tile instead of st.metric)."""
    tip = f' title="{html.escape(title, quote=True)}"' if title else ""
    delta_html = ""
    if delta:
        cls = f" {tone}" if tone in ("gain", "loss") else ""
        delta_html = f'<div class="ier-delta{cls}">{html.escape(delta)}</div>'
    return (f'<div class="ier-metric"{tip}>'
            f'<div class="ier-lbl">{html.escape(label)}</div>'
            f'<div class="ier-val">{html.escape(value)}</div>'
            f'{delta_html}</div>')


def render_metric_tiles(tiles: list[dict]) -> None:
    """Render a responsive row of premium metric tiles from a list of dicts
    ({label, value, delta?, tone?, title?}). Raises on any failure so the caller's try/except can
    fall back to native st.metric -- the numbers must always show, styled or not."""
    cells = "".join(_metric_tile_html(**t) for t in tiles)
    st.markdown(f'<div class="ier-metrics">{cells}</div>', unsafe_allow_html=True)


_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = _ROOT / "sample_data" / "sample_portfolio.csv"
HOLDINGS_CSV = _ROOT / "holdings.csv"   # the owner's real portfolio (gitignored)
EVAL_STORE = _ROOT / "data" / "eval_cases.jsonl"


def _clean_secret(val, default):
    """Trim a string secret (blank/whitespace-only -> the default, i.e. treated as absent); pass a
    non-string secret (the service-account dict, a bool) through unchanged.

    WHY (config hygiene): env vars / .env files / Streamlit TOML secrets very commonly carry a
    trailing space or newline (a quoted-with-space value or a copy-paste). An untrimmed URL / token /
    sheet key / model / password then silently fails its fetch / lookup / compare -- the same class
    that locked the parents out on a padded app_password and broke the Ask tab on a padded LLM_MODEL.
    Centralizing it here means every secret read gets the hygiene, so a new call site can't reopen it."""
    if isinstance(val, str):
        return val.strip() or default
    return val


def _secret(key: str, default=None):
    """Read a Streamlit secret, tolerating no secrets file at all (local dev). String values are
    whitespace-trimmed (blank -> the default); see _clean_secret."""
    try:
        return _clean_secret(st.secrets.get(key, default), default)
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ar_ref(symbol: str):
    """Resolve the symbol's latest NSE annual-report ref (fiscal year + FY-end as-of date) for the
    Research-tab freshness banner. Returns an AnnualReportRef or None.

    DEGRADE-SAFE (real money, live app): NSE blocks datacenter IPs (Streamlit Cloud), so this
    commonly returns None there and the banner then renders NOTHING rather than a fabricated date.
    Cached per symbol (ttl 24h, matching fetch_ar_text) so a report render does at most ONE light
    resolve per symbol per day, never a per-rerun refetch -- the report path itself discards the
    ref it resolves internally (see nse_annual_report_source), so this is the one place that
    surfaces it. The resolver's own fetch already swallows network errors and returns None; the
    extra guard is belt-and-suspenders so a parse/attr error can never crash the report view."""
    try:
        return NseAnnualReportResolver().latest_report(symbol.strip().upper())
    except Exception:
        return None


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_promoter_pledge(symbol: str):
    return get_screener_source().promoter_pledge(symbol)


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
    # Trim the configured password: a Streamlit TOML secret commonly carries a trailing space or
    # newline (app_password = "pw "). Compared untrimmed with ==, that rejected the CORRECT password
    # on both the typed prompt and the ?key= magic-link -- locking the parents out of the deployed app
    # with a password that "looks right". A blank/whitespace-only value still means no password set.
    expected = str(_secret("app_password") or "").strip()
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
    registry.add(Source(
        PROMOTER_PLEDGE_SOURCE_ID, "Promoter pledge (Screener)", CredibilityTier.ANALYST,
        notes="Single-source (Screener only), not cross-verified -- reported context, never a "
              "fact, and never a buy/sell signal on its own."))
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


def conflict_values_line(fig) -> str:
    """The actual per-source values behind a withheld (CONFLICT) figure, each in its proper unit.

    WHY (real money, review workflow + honesty): a conflicting figure is shown only as 'withheld',
    which hides WHAT the sources disagreed on. The expert must acknowledge a conflict before
    approving (real-money gate), and seeing the numbers is what lets them judge a benign
    definitional gap (e.g. yfinance's to-owners net profit vs Screener's consolidated, ~15% on
    minority interest) apart from a real parse/scale error to reject. These are the disagreeing,
    UNVERIFIED values -- labeled as such by the caller, never presented as facts."""
    return ", ".join(f"{sv.source_id} {format_figure_value(fig.name, sv.value)}"
                     for sv in fig.sources if sv.value is not None)


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


_ASK_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def ask_source_caption(citations, registry) -> str:
    """The 'Source:' line for an Ask-tab claim: each cited source's name, and -- for a NEWS item --
    the publisher + article date (or "undated") from the citation locator so the reader can judge how
    recent a news-backed claim is. De-duplicated (a claim citing two chunks of one source shows it
    once). WHY append a news locator but not a figure/filing one: the app's own figure/filing
    documents carry redundant internal locators ("RELIANCE verified figures"); appending those would
    be noise, whereas a news locator ("Reuters, 2026-05-15") adds the freshness signal the reader
    needs. An UNDATED news item ("Reuters, undated") is surfaced too -- keeping its publisher and
    flagging that the date is unknown so a parent doesn't assume it's recent -- since only news
    locators ever carry "undated", it discriminates news from figure/filing docs just as safely."""
    labels: list[str] = []
    for c in citations:
        src = registry.get(c.source_id)
        name = src.name if src else c.source_id
        loc = str(getattr(c, "locator", "") or "").split(" chunk ")[0].strip()
        is_news_locator = bool(loc) and (_ASK_ISO_DATE.search(loc) or "undated" in loc.lower())
        label = f"{name} — {loc}" if is_news_locator else name
        if label not in labels:
            labels.append(label)
    return ", ".join(labels) or "no source"


# --- W7 trust UI (SPEC v4 §4 W7): claim-type badges, click-through spans, freshness banners, and
# a show-the-computation panel. These are PURE helpers (no Streamlit), each rendered at the call
# site inside a try/except so a display failure degrades to a small note, never a page crash
# (LESSONS 2026-07-09: abstain/degrade at every boundary). ---

# A news-backed ("recent news") answer older than this reads as stale and is FLAGGED (never hidden,
# never shown as current). News is the Ask tab's dated, time-sensitive source; a figure/filing
# locator carries no content date and is skipped by claim_freshness_lines (its vintage is surfaced
# by the Research tab's annual-vintage caveat instead), so this threshold only governs dated news.
_ASK_STALE_DAYS = 30

# An annual report older than this reads as stale and is FLAGGED in the Research tab. Indian fiscal
# years end 31 March, so the latest available AR is dated its FY-end and can legitimately be up to
# ~14 months old before the next year's report is filed; 400 days keeps the most recently completed
# FY's report reading "fresh" while flagging a report a full extra year behind (recent results not
# reflected). Tuned for a once-a-year document, unlike the 30-day news window above.
_AR_STALE_DAYS = 400


def claim_badge(claim, registry) -> tuple[str, str]:
    """(label, color) for a claim's visible trust badge, mirroring the Ask tab's own message
    rendering so the badge and the message can never disagree. A GREEN 'Verified fact' ONLY for a
    FACT backed solely by primary sources (is_verified_fact -- the app's one green-tick invariant).
    OPINION and ESTIMATE get neutral badges. An UNVERIFIED claim resting only on primary sources gets
    a hard red 'Unverified' (the issue is a misquoted/absent figure -- the exact real-money failure
    this app guards against); one resting on news/analyst text gets an orange 'Reported, not verified'
    (honest context, not alarming)."""
    if claim.kind == FACT and claim.is_verified_fact:
        return ("Verified fact", "green")
    if claim.kind == OPINION:
        return ("Opinion", "blue")
    if claim.kind == ESTIMATE:
        return ("Estimate (derived, not a primary figure)", "grey")
    # UNVERIFIED (or a FACT that somehow lost verification -- treated identically, never green).
    from_primary_only = bool(claim.citations) and all(
        registry.get(c.source_id) and registry.get(c.source_id).citable_as_fact
        for c in claim.citations)
    if from_primary_only:
        return ("Unverified (figure not confirmed in its source)", "red")
    return ("Reported, not verified", "orange")


def claim_freshness_lines(claim, today, stale_days: int = _ASK_STALE_DAYS):
    """Per-citation freshness verdicts from the content date embedded in each citation's locator (a
    news item's locator is 'Publisher, YYYY-MM-DD' or 'Publisher, undated'). Returns (line, stale,
    unknown) tuples -- built from describe_freshness/freshness -- so the caller can visibly flag a
    stale or undated source and never present old news as current. Only a dated/undated (news-like)
    locator yields a line; a figure/filing locator with no date is skipped (nothing to date it by
    here). De-duplicated so two chunks of one dated source produce one line."""
    out: list[tuple[str, bool, bool]] = []
    seen: set[str] = set()
    for c in claim.citations:
        loc = str(getattr(c, "locator", "") or "").split(" chunk ")[0].strip()
        match = _ASK_ISO_DATE.search(loc)
        is_undated = "undated" in loc.lower()
        if not match and not is_undated:
            continue
        as_of = match.group(0) if match else ""
        subject = loc.split(",")[0].strip() or "source"
        line = describe_freshness(as_of, today, stale_days, subject=subject)
        if line in seen:
            continue
        seen.add(line)
        verdict = freshness(as_of, today, stale_days)
        out.append((line, verdict.stale, not verdict.known))
    return out


def annual_report_freshness_line(ref, today, stale_days: int = _AR_STALE_DAYS):
    """Freshness line + (stale, unknown) flags for the PRIMARY source behind a Research-tab report:
    the latest annual report resolved for the symbol. `ref` is an AnnualReportRef (needs .fiscal_year
    and .as_of) or None.

    Returns None when there is no ref OR no usable as-of date, so the caller renders NOTHING
    (degrade) -- never a fabricated date for an unresolved/undated report. Dates the report by its
    FY-end as_of and reuses describe_freshness/freshness (never reimplements date logic), mirroring
    the Ask tab's dated-source freshness. The subject reads 'FY2026 annual report' when the fiscal
    year is known, else a bare 'annual report'."""
    if ref is None:
        return None
    as_of = str(getattr(ref, "as_of", "") or "")
    verdict = freshness(as_of, today, stale_days)
    if not verdict.known:                     # resolved but undated -> nothing to date it by
        return None
    fy = getattr(ref, "fiscal_year", 0) or 0
    subject = f"FY{fy} annual report" if isinstance(fy, int) and fy > 0 else "annual report"
    line = describe_freshness(as_of, today, stale_days, subject=subject)
    return (line, verdict.stale, not verdict.known)


def format_computed_figure(fig) -> str:
    """One-line rendering of a ComputedFigure for the show-the-computation panel: label, the finished
    value in its unit, the exact source inputs, and the formula -- so a reader sees the number was
    computed deterministically by the system from the cited figures, never generated by the model."""
    unit = getattr(fig, "unit", "")
    if unit == "percent":
        value = f"{fig.value:.2f}%"
    elif unit in ("x", "ratio"):
        value = f"{fig.value:.2f}x"
    else:
        value = f"{fig.value:,.2f}"
    inputs = ", ".join(f"{x:,.2f}" for x in fig.inputs)
    return f"{fig.label}: {value} (computed from {inputs}; {fig.formula})"


def plain_summary(verdict, stance: Stance) -> str:
    """A one-line, jargon-free read for a non-expert. Honest when data is thin."""
    if verdict is None or stance == Stance.INSUFFICIENT_DATA:
        return ("Not enough independently verified data to form a view. Withheld on purpose, "
                "rather than guessed.")
    # WHY every phrasing is a VERB clause: the sentence is built as "It {val}, with {qual}.", so a
    # bare-noun "unknown" value ("valuation could not be verified") produced the broken "It valuation
    # could not be verified..." for a reachable case (median P/E uncomputable but quality verified).
    val = {"cheap": "looks cheap versus its own history",
           "fair": "looks fairly priced versus its own history",
           "expensive": "looks expensive versus its own history",
           "unknown": "has a valuation that couldn't be checked against its own history"
           }[verdict.valuation.value]
    # WHY (real money, sector-aware honesty): a bank/NBFC's quality tier is its RETURNS (ROA), not
    # balance-sheet strength -- and the app cannot assess a lender's actual balance-sheet quality
    # (asset quality/GNPA, capital adequacy) from the free feeds. So a bank must never be summarized
    # as having a "strong balance sheet"; describe its profitability for a lender instead (the full
    # "check the filing" caveat is shown separately as a sector caveat).
    if verdict.is_bank:
        qual = {"strong": "strong profitability for a lender",
                "mixed": "middling profitability for a lender",
                "weak": "weak profitability for a lender",
                "unknown": "profitability that couldn't be confirmed"}[verdict.quality.value]
    else:
        qual = {"strong": "a strong balance sheet",
                "mixed": "a mixed balance sheet",
                "weak": "balance-sheet concerns",
                "unknown": "balance-sheet quality unconfirmed"}[verdict.quality.value]
    _, headline = _STANCE_UI[stance]
    return f"It {val}, with {qual}. {headline}."


def data_vintage_note(figures) -> str | None:
    """A plain-language caveat that the fundamental read rests on the latest ANNUAL figures, so the
    company's results SINCE then (recent quarters) are not reflected.

    WHY (real money, honesty for a non-expert): the quality verdict -- earnings quality, leverage,
    interest cover, ROCE, cash-flow discipline -- is built entirely on annual statement figures,
    which by construction never include the most recent quarter(s). A March-year-end company's last
    annual can be up to ~15 months stale by the time the next one files, so a parent reading
    "Evidence leans favorable" as the headline could act on a view that predates several quarters of
    results. That vintage is currently only implicit in FY tags inside the collapsed evidence panel.
    Names the LATEST cross-verified fiscal year (the real vintage that drove the verdict) and points
    at the actionable step. Uses only trustworthy (cross-verified) figures; returns None when none
    carry a fiscal-year tag (e.g. a valuation-only/point-figure report), where there is no annual
    vintage to caveat."""
    years: list[int] = []
    for f in figures:
        if not getattr(f, "is_trustworthy", False):
            continue
        for sv in getattr(f, "sources", ()):
            loc = str(getattr(sv, "locator", "") or "")
            if loc.upper().startswith("FY"):
                m = re.search(r"(\d{4})", loc)
                if m:
                    years.append(int(m.group(1)))
    if not years:
        return None
    return (f"Heads up: this fundamental read is based on the company's latest ANNUAL results "
            f"(FY{max(years)}); anything reported since then — recent quarters — is not included "
            "here, so check the latest quarterly update or recent news before deciding.")


_STANCE_PDF = {Stance.FAVORABLE: "[+] ", Stance.NEUTRAL: "[~] ",
               Stance.UNFAVORABLE: "[-] ", Stance.INSUFFICIENT_DATA: "[?] "}


# Common non-Latin-1 typographic characters this app's own copy uses, mapped to readable ASCII for
# fpdf's core (Latin-1) font. WHY (real money, the shared PDF's credibility): a bare
# encode("latin-1","replace") turns each of these into "?" -- and the EM DASH appears in nearly
# every insight/caveat (deep_metrics/trends/framework), so an unmapped PDF read "... 18%) ? strong",
# corruption in a document a parent downloads to review or share with family. Mapped BEFORE the lossy
# encode in _pdf_latin1, which keeps the encode only as a final safety net for any other stray glyph.
_PDF_CHAR_MAP = {
    "₹": "Rs.",   # rupee sign
    "—": "-",      # em dash
    "–": "-",      # en dash
    "‘": "'",      # left single quote
    "’": "'",      # right single quote / curly apostrophe
    "“": '"',      # left double quote
    "”": '"',      # right double quote
    "…": "...",    # ellipsis
}


def _pdf_latin1(text) -> str:
    """Sanitize text for fpdf's core Latin-1 font: map the common non-Latin-1 typographic characters
    the app's copy uses (em/en dash, curly quotes, ellipsis, the rupee sign) to readable ASCII, THEN
    latin-1-encode as a final safety net so any other stray glyph still can't crash the PDF build."""
    out = str(text)
    for uni, ascii_ in _PDF_CHAR_MAP.items():
        out = out.replace(uni, ascii_)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(title: str, report, stance: Stance, guidance=None,
                     promoter_trend: str | None = None,
                     cash_conversion_trend: str | None = None,
                     other_income_share: str | None = None,
                     promoter_pledge: str | None = None) -> bytes:
    """A downloadable PDF of the report, mirroring what is on screen. Uses the core Helvetica
    font (Latin-1), so text is sanitized and the rupee sign is written as 'Rs.'.

    WHY (real money, UI honesty): promoter_trend/cash_conversion_trend/other_income_share/
    promoter_pledge are the same single-source (Screener-only) context signals the Research tab
    shows in their own expanders -- this button is labeled "Download full report", so a parent who
    saves this PDF to review offline, or share with family, must see the SAME signals the live app
    shows them, not a report that is silently missing some of the app's own research signals.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    def s(text) -> str:
        return _pdf_latin1(text)

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
    # WHY: the saved/shared PDF must carry the same annual-vintage caveat the app shows, so a parent
    # reviewing it offline sees the fundamentals predate recent quarters (mirrors the live view).
    _vintage = data_vintage_note(report.figures)
    if _vintage:
        line(_vintage, size=9, style="I", h=5, gap=2)

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
                           (promoter_pledge, promoter_trend, cash_conversion_trend,
                            other_income_share) if p]
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

    pdf.ln(2)
    pdf.set_text_color(90, 90, 90)
    if v is not None:
        line(v.caveat, size=9, style="I", h=5)
    # WHY (real money, honesty): the PDF is saved/shared OFFLINE, away from the app's always-visible
    # footer disclaimer, so it must carry the full disclaimer itself. Two gaps this closes: a
    # no-verdict (insufficient/no-data) report previously had NO caveat at all (the app shows the
    # DISCLAIMER for a no-verdict report, so the PDF must too); and the verdict caveat alone omits
    # "verify every figure / data may be delayed or incorrect / you alone are responsible" -- exactly
    # what a document shared with family, without the app around it, needs.
    line(DISCLAIMER, size=9, style="I", h=5)
    # WHY (SEBI, real money): a saved/shared PDF leaves the app entirely, so it must carry the same
    # AI-usage disclosure the live page shows -- AI-assisted, cross-checked, human-reviewed, research
    # only, no return/accuracy claim. The operator owns the AI output (Reg 16C); the document that
    # gets forwarded to family must say so on its face, not only inside the app.
    line(AI_DISCLOSURE, size=9, style="I", h=5)
    return bytes(pdf.output())


# --- auth + hosted-model secrets (both no-ops locally with no secrets file) ---

inject_theme_css()   # style every path, including the login screen and the empty states below
_bridge_secrets_to_env()
if not _check_password():
    st.stop()

# --- header (product hero) ---
# WHY: reads as a real product, not a script. The one-liner states the load-bearing promise up
# front -- research, not advice -- so a non-expert parent sees it before any number. Rendered as a
# custom block for the typographic hierarchy; wrapped so a markup failure falls back to the plain
# title/caption and never leaves the page headerless.
try:
    st.markdown(
        '<div class="ier-hero">'
        '<span class="ier-badge">India · NSE / BSE</span>'
        '<div class="ier-title">📊 India Equity Research</div>'
        '<p class="ier-sub">Cross-verified research on your Indian stocks, in plain language. '
        'Every figure is traced to a dated source. Research only — never buy/sell advice.</p>'
        '</div>',
        unsafe_allow_html=True)
except Exception:  # pragma: no cover - header must never blank the page
    st.title("📊 India Equity Research")
    st.caption("Understand your investments.")

# How-to-read guide: contextual, collapsed by default (discoverable, never nags). Teaches a parent
# what the trust badges mean -- the one thing they must understand to read this app safely.
with st.expander("New here? How to read this (30-second guide)"):
    st.markdown(
        "**What this is.** A research tool for your own Indian stocks. It gathers and "
        "cross-checks figures for you. It does **not** tell you what to buy or sell, and it "
        "does not place trades.\n\n"
        "**How to read the trust badges** on an answer:")
    try:
        st.markdown(
            '<div class="ier-note">'
            '<span class="ier-chip g">Verified fact</span> — the figure was confirmed in a '
            'primary source (an official filing).<br>'
            '<span class="ier-chip o">Reported, not verified</span> — it comes from news or an '
            'analyst; treat as context, not proof.<br>'
            '<span class="ier-chip r">Unverified</span> — a figure we could not confirm in its '
            'source; do not act on it.<br>'
            '<span class="ier-chip n">Estimate / Opinion</span> — derived or a viewpoint, not a '
            'primary figure.'
            '</div>', unsafe_allow_html=True)
    except Exception:  # pragma: no cover
        st.caption("Green = verified fact; orange = reported not verified; red = unverified; "
                   "grey = estimate or opinion.")
    st.caption("Prices can be delayed. Verify every figure before acting — you alone are "
               "responsible for your decisions.")

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
        # Friendly first-run state: a parent opening this for the first time sees what it is and
        # exactly how to start, not a bare one-line prompt. Wrapped so a markup failure degrades to
        # the original st.info (the instruction must always show).
        try:
            st.markdown(
                '<div class="ier-note">'
                '<div class="ier-note-title">👋 Welcome — let\'s load your stocks</div>'
                'This tool researches the Indian stocks <b>you already own</b> and explains them '
                'in plain language. It is research only: it never tells you what to buy or sell.'
                '<ul>'
                '<li><b>Open the sidebar</b> (the <b>›</b> arrow, top-left) and tick a portfolio '
                'option, or upload your holdings CSV.</li>'
                '<li>Just exploring? Tick <b>“Use sample portfolio”</b> to see how it works.</li>'
                '<li>Zerodha / Groww / Google Sheet exports work too.</li>'
                '</ul></div>', unsafe_allow_html=True)
        except Exception:  # pragma: no cover
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

    # No percentage delta (not "+0.00%") when total cost is 0 -- an all-zero-cost book is all gain,
    # so its percent return is undefined; the rupee P&L above still carries it (mirrors pnl_pct None).
    _tpct = analysis.total_pnl_pct
    _pnl_delta = f"{_tpct:+.2f}%" if _tpct is not None else None
    _pnl_tone = "gain" if analysis.total_pnl_abs >= 0 else "loss"
    # Premium metric tiles for the portfolio summary. WHY behavioural-neutral: same four numbers,
    # same labels; only the presentation changes. Wrapped so any markup failure falls back to the
    # exact original st.metric cards -- the numbers must always show. (The custom tile carries the
    # P&L help text as a hover tooltip; the glossary keeps the full definition either way.)
    try:
        render_metric_tiles([
            {"label": "Invested", "value": money(analysis.total_invested)},
            {"label": "Market value", "value": money(analysis.total_value)},
            {"label": "Profit / loss", "value": money(analysis.total_pnl_abs),
             "delta": _pnl_delta, "tone": _pnl_tone, "title": explain("P&L")},
            {"label": "Holdings priced",
             "value": f"{len(analysis.positions)} / {distinct_holding_symbols}"},
        ])
    except Exception:  # pragma: no cover - fall back to native metrics, never lose the numbers
        m = st.columns(2)
        m[0].metric("Invested", money(analysis.total_invested))
        m[1].metric("Market value", money(analysis.total_value))
        m2 = st.columns(2)
        m2[0].metric("Profit / loss", money(analysis.total_pnl_abs), _pnl_delta,
                     help=explain("P&L"))
        m2[1].metric("Holdings priced", f"{len(analysis.positions)} / {distinct_holding_symbols}")

    if analysis.missing_symbols:
        st.warning("No price found for: " + ", ".join(analysis.missing_symbols)
                   + ". Excluded from the totals. Check the symbol spelling or exchange.")

    rows = [{
        "Symbol": p.symbol, "Sector": p.sector, "Qty": p.quantity,
        "Avg cost": round(p.avg_cost, 2), "Price": round(p.current_price, 2),
        "Value": round(p.market_value, 2),
        # None (a zero-cost bonus/IPO lot has an undefined % return) -> blank cell, not a misleading 0.0
        "P&L %": round(p.pnl_pct, 2) if p.pnl_pct is not None else None,
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
        # WHY help= (comprehension): "effective 3.2" when a parent holds 8 names is opaque on its
        # own -- and the glossary already HAS the plain explanation; it just wasn't wired to the card
        # (HHI next to it is). Show it, matching every other technical metric card in the app.
        cc[2].metric("Effective # holdings", f"{analysis.effective_holdings:.1f}",
                     help=explain("Effective number of holdings"))
        flags = []
        if analysis.top_holding_weight > CONCENTRATION_TOP_HOLDING_WARN:
            flags.append(f"One name is over {CONCENTRATION_TOP_HOLDING_WARN * 100:.0f}% of the book.")
        if analysis.hhi > CONCENTRATION_HHI_WARN:
            flags.append("HHI reads as concentrated (few names drive the book).")
        # WHY (sector-aware diversification): a heavy single-SECTOR weight is undiversified even when
        # no single NAME trips the checks above (5 bank names at 14% each = 70% Financials). Excludes
        # 'Unknown' (unresolved sectors, not a real concentration). The same coverage caveat above
        # applies -- weights are of the priced subset -- so a missing-price book doesn't over-read.
        sector_flag = sector_concentration_note(analysis.sector_weights)
        if sector_flag:
            flags.append(sector_flag)
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
                # WHY (honesty): even with every name "having history", the SHARED (intersected)
                # window can be short if one name was recently listed/added -- annualizing so few
                # days into a firm volatility/beta overstates precision, so disclose a thin window.
                thin_note = thin_risk_window_note(len(port_returns))
                if thin_note:
                    st.caption(thin_note)
                risk_cols = st.columns(3)
                # n/a (not "0.0%"/"nan%") when the shared window is too short for a std to be
                # computable -- a fabricated number is worse than an honest "n/a" (mirrors beta below).
                _vol = annualized_volatility(port_returns)
                risk_cols[0].metric("Annualized volatility",
                                    f"{_vol * 100:.1f}%" if _vol is not None else "n/a",
                                    help=explain("Volatility"))
                # beta needs the benchmark's OWN history; the index fetch can fail while stock
                # history succeeds, so beta() returns None (not a fabricated 0.00 that would read as
                # a real market-neutral beta). Show n/a in that case.
                port_beta = beta(port_returns, bench_returns)
                risk_cols[1].metric(f"Beta vs {INDEX_DISPLAY_NAMES[DEFAULT_BENCHMARK]}",
                                    f"{port_beta:.2f}" if port_beta is not None else "n/a",
                                    help=explain("Beta"))
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
            # WHY normalize (real money, workflow honesty): a parent often pastes a symbol in a
            # non-bare format -- "NSE:INFY" (TradingView), "INFY-EQ" (NSE site), "INFY.NS" (Yahoo).
            # Bare strip().upper() left the prefix/suffix on, so yfinance found nothing and the
            # "symbol didn't resolve -- the exact ticker differs from the common name (PAGE->PAGEIND)"
            # hint fired for a VALID symbol, misleading the parent into thinking INFY itself is wrong.
            # normalize_symbol (already used for the uploaded portfolio, unit-tested) strips them.
            _run_live(normalize_symbol(typed), ar_url)

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
        # WHY (real money, honesty): the verdict rests on ANNUAL figures; name that vintage and point
        # at recent quarters so a non-expert doesn't act on a headline that predates months of results.
        _vintage = data_vintage_note(report.figures)
        if _vintage:
            st.caption(_vintage)
        # H6 freshness banner (additive, degrade-safe): the vintage + a fresh/stale flag for the
        # PRIMARY source, the latest NSE annual report on file for this symbol, resolved on demand
        # and cached per symbol (fetch_ar_ref). If NSE didn't resolve one (e.g. it blocks the cloud
        # server) or anything fails, this renders NOTHING -- never a fabricated date, never a crash
        # (LESSONS: abstain at every boundary). Mirrors the Ask tab's ⏳ freshness style: a warning
        # when the newest report is over a year old, a quiet ⏳ caption when it is current.
        try:
            _ar_fresh = annual_report_freshness_line(fetch_ar_ref(sym),
                                                     datetime.now().strftime("%Y-%m-%d"))
            if _ar_fresh is not None:
                _ar_line, _ar_stale, _ar_unknown = _ar_fresh
                if _ar_stale:
                    st.warning(f"⏳ Analysis based on the {_ar_line}. That is over a year old, so "
                               "recent results are not reflected; check the latest annual report "
                               "and recent quarters before deciding.")
                else:
                    st.caption(f"⏳ Analysis based on the {_ar_line}.")
        except Exception:  # pragma: no cover - a freshness banner must never crash the report view
            pass
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
                                  other_income_share=fetch_other_income_share(sym),
                                  promoter_pledge=fetch_promoter_pledge(sym)),
            file_name=f"{sym}_research.pdf", mime="application/pdf")

        # the evidence, one tap away
        with st.expander("See the evidence (figures, reasons, sources)"):
            if report.verdict is not None:
                vc = st.columns(2)
                vc[0].metric("Valuation", report.verdict.valuation.value)
                vc[1].metric("Quality", report.verdict.quality.value)
                vc2 = st.columns(2)
                vc2[0].metric("Leaning", report.verdict.leaning.value)
                # WHY help= (real money, comprehension): a non-expert can read "Confidence: high" as
                # "high chance of gains" when it means how much of the data cross-verified. Explain it
                # the same way every other technical metric card here already does (HHI/Beta/etc.).
                vc2[1].metric("Confidence", report.verdict.confidence.value,
                              help=explain("Confidence"))
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
                # Show the actual disagreeing values so the expert can judge the nature of the gap
                # (a benign definitional difference to acknowledge vs a real error to reject)
                # before approving -- 'withheld' alone hides what the sources actually said.
                for f in report.conflicts:
                    st.caption(f"• {f.name}: {conflict_values_line(f)}")
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

        # promoter pledge (Screener only, single-source; a top-tier Indian red flag -- pledged
        # promoter shares can be sold by lenders on a margin call. Screener flags it only when
        # material, so its ABSENCE means "not flagged", never a confirmed zero pledge.)
        with st.expander("Promoter pledge (context, not cross-verified)"):
            pledge = fetch_promoter_pledge(sym)
            if pledge:
                st.markdown(f"- ⚠ {pledge}")
            else:
                st.caption("Screener does not flag a material promoter pledge for this stock. "
                           "That is not a confirmed zero -- Screener surfaces pledge only when it "
                           "is material; check the latest shareholding filing to be sure.")
            st.caption("Pledge data is published only by Screener (not yfinance), so it cannot "
                       "cross-verify the way the figures above do. Context, not a fact, and never "
                       "a buy/sell signal on its own.")

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
        # WHY (real money, honest metric): headline trusted-but-wrong (the safety number that must
        # stay 0) and precision over what the system actually TRUSTED -- never a bare matches/total
        # "accuracy", which folds SAFE withholds (a figure it correctly refused to trust because
        # sources conflicted) into the denominator and so reads a cautious, correct run as inaccurate.
        trusted = ev.matches + ev.trusted_wrong
        correctness = (f"when it trusted a figure it was right {ev.trusted_accuracy:.0%} "
                       f"({ev.matches}/{trusted})" if trusted
                       else "it trusted none here (all withheld or unavailable)")
        st.caption(
            f"Learning loop over {len(cases)} recorded corrections: trusted-but-wrong "
            f"{ev.trusted_wrong} (must be 0); {correctness}; it safely withheld {ev.withheld} "
            "it couldn't cross-verify (withholding is the safe outcome, not an error).")


# ==================== TAB 3: INVEST A LUMP SUM ====================

with tab_invest:
    st.subheader("Invest a lump sum")

    # Today's research shortlist: shown instantly from the Sheet; refreshed on demand by the button below.
    # The refresh runs from THIS app (Streamlit Cloud can reach Screener; a scheduler's datacenter
    # IP can't), so it's full cross-verification. Reading the tab is one fast call.
    if "today_rows" not in st.session_state:
        try:
            st.session_state.today_rows = get_gateway().read("Today")
        except Exception:
            st.session_state.today_rows = []
    today_rows = st.session_state.today_rows
    if today_rows:
        # WHY "research shortlist", not "picks" (real money, hard "never a buy/sell call"): a bold
        # "Today's picks" header reads as a buy list to a non-expert, drawing the eye before the
        # caveat below -- the same framing concern fixed in the daily push notification. Frame it as
        # a shortlist to research, matching the notification and the app's non-advice stance.
        st.markdown(f"**📌 Today's research shortlist** ({len(today_rows)})")
        for r in today_rows:
            st.markdown(f"- **{r.get('symbol', '')}** ({r.get('stance', '')}) — "
                        f"{r.get('reason', '')}")
        # WHY: Sheets coerces the date string into a datetime on round-trip; show only the date.
        as_of = str(today_rows[0].get("date", "")).split("T")[0]
        st.caption(f"As of {as_of}. Cross-verified and within your per-stock cap — a shortlist to "
                   "research, not a buy or sell call.")
    else:
        st.caption("No shortlist yet. It's prepared automatically each day.")
    # WHY: display-only reload, NOT a re-research. The batch runs on the owner's Mac (residential
    # IP, full cross-verification); running it here from the datacenter IP would come back thin and
    # overwrite the good picks. So the app just pulls the latest the Mac computed.
    if _sheet_configured() and st.button("🔄 Reload latest shortlist"):
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
        # WHY (honesty): source-loading is gated on the (now-disabled) Ask button below, so nothing
        # actually loads here without an LLM -- the old "the sources still load below" was simply
        # false. Say what is true: Ask needs the model; the rest of the app does not.
        st.info("Set LLM_MODEL to ask questions here. The Portfolio, Research, and Invest tabs "
                "work without it.")

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
        pl_doc = None
        sym_u = ""
        company = ""
        if ask_sym.strip():
            # normalize the same way the Research tab and the uploaded portfolio do, so a pasted
            # "NSE:INFY" / "INFY-EQ" / "INFY.NS" resolves instead of falsely reading as a bad ticker.
            sym_u = normalize_symbol(ask_sym)
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
            # promoter pledge: a top-tier Indian red flag ("has the promoter pledged shares?" is a
            # natural question); same single cached Screener page as the signals above.
            pl_doc = promoter_pledge_document(sym_u, fetch_promoter_pledge(sym_u))
            _ingest_and_pin(pl_doc, PROMOTER_PLEDGE_SOURCE_ID)
        # WHY (real money, workflow): distinct from "haven't researched yet". `company` alone
        # (yfinance's own name lookup) is a WEAKER signal than Report.no_data_found (which spans
        # figures from both yfinance AND Screener) -- a real, valid symbol can have yfinance's
        # name lookup come back empty (a known Yahoo India-coverage gap) while Screener still has
        # real data. symbol_has_no_data widens the check to all six independent "this symbol is
        # real" signals (name, cross-verified figures, promoter trend, cash conversion cycle,
        # other income share, promoter pledge) so this hint is never shown in the same response
        # where real per-symbol data was just fetched and used. Live-verified root cause: Page
        # Industries trades as PAGEIND, not PAGE; typing the natural/common name resolves to
        # nothing from ANY of the six signals. Telling the user to "research it first" would be
        # actively misleading here -- doing so with the SAME wrong symbol fails there too.
        symbol_unresolved = bool(sym_u) and symbol_has_no_data(
            company, vf_doc is not None, pt_doc is not None, cc_doc is not None,
            oi_doc is not None, pl_doc is not None)
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
                # WHY retrieval_hint (Ask-tab answer quality): the user entered a specific stock, so
                # scope retrieval to it. A natural question ("what is the recent news?") shares no
                # words with a specific headline, so without this the fetched news scored below the
                # floor and was never retrieved. The resolved company name + symbol (which the
                # company's own news mentions) surfaces it; the model still answers the original Q.
                _ask_as_of = datetime.now().strftime("%Y-%m-%d %H:%M")
                _ask_hint = f"{company} {sym_u}".strip()
                _ask_pins = frozenset(pinned_source_ids)
                # W7 (SPEC v4 §4): route the grounded-answer path through the W4 ResearchOrchestrator
                # (PLAN -> RETRIEVE -> COMPUTE -> VERIFY[gate] -> WRITE) so a growth/CAGR figure is
                # pre-computed in Python and the model only PHRASES it (compute-don't-generate, end to
                # end). It reuses the SAME analyst, retrieval, pins, as-of, and retrieval_hint the
                # direct path used, so inputs/gating are unchanged. DEGRADE-SAFE (real money, live
                # app): if this new orchestration layer raises for any reason, fall back to the proven
                # grounded.answer so a new code path can NEVER take the parents' Ask tab down.
                computed: tuple = ()
                try:
                    _out = ResearchOrchestrator(grounded).run(
                        question, store, registry, pin_source_ids=_ask_pins,
                        as_of=_ask_as_of, retrieval_hint=_ask_hint)
                    result = _out.result
                    computed = _out.computed
                except Exception:
                    result = grounded.answer(question, store, registry,
                                             pin_source_ids=_ask_pins,
                                             as_of=_ask_as_of, retrieval_hint=_ask_hint)
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
                _ask_today = datetime.now().strftime("%Y-%m-%d")
                for claim in result.claims:
                    cited = ask_source_caption(claim.citations, registry)
                    # W7 claim-type badge (additive, degrade-safe): an explicit trust label above the
                    # claim -- a green badge ONLY for a verified fact. A failure here must never drop
                    # the proven message rendering below.
                    try:
                        _badge_label, _badge_color = claim_badge(claim, registry)
                        st.badge(_badge_label, color=_badge_color)
                    except Exception:
                        pass
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
                    # W7 freshness banner (additive, degrade-safe): flag a stale or undated
                    # news-backed claim so a parent never reads old news as current.
                    try:
                        for _fline, _stale, _unknown in claim_freshness_lines(claim, _ask_today):
                            if _stale:
                                st.warning(f"⏳ {_fline}. Verify against the latest before deciding.")
                            elif _unknown:
                                st.caption(f"⏳ {_fline} (freshness unknown)")
                            else:
                                st.caption(f"⏳ {_fline}")
                    except Exception:
                        pass
                    # W7 click-through source spans (additive, degrade-safe): the EXACT supporting
                    # sentence inside each cited chunk, so a reader can see what backs the figure.
                    try:
                        _spans = claim.spans()
                        if _spans:
                            with st.expander("Show the source spans (what backs this)"):
                                for _sp in _spans:
                                    _src = registry.get(_sp.get("source_id", ""))
                                    _name = _src.name if _src else _sp.get("source_id", "source")
                                    _loc = str(_sp.get("locator", "")).split(" chunk ")[0].strip()
                                    _quote = str(_sp.get("quote", "")).strip()
                                    st.caption(f"{_name} · {_loc}" if _loc else _name)
                                    if _quote:
                                        st.markdown(f"> {_quote}")
                                    else:
                                        st.caption("(no exact supporting sentence located in this "
                                                   "chunk)")
                    except Exception:
                        pass
                # W7 show-the-computation (additive, degrade-safe): any figure the orchestrator
                # derived in Python, with its inputs + formula -- computed by the SYSTEM, not the
                # model. Only shown when the orchestrator actually pre-computed a figure.
                try:
                    if computed:
                        with st.expander("Show the computation (done by the system, not the model)"):
                            for _fig in computed:
                                st.markdown(f"- {format_computed_figure(_fig)}")
                            st.caption("These figures were computed deterministically in Python from "
                                       "the cited source numbers, then only phrased by the model. The "
                                       "model never did the arithmetic itself.")
                except Exception:
                    pass
            # W8 AI-usage disclosure (SPEC v4 §6, SEBI): render the disclosure right beside the answer
            # (whether it answered or abstained) so a parent reading an AI-assisted answer always sees
            # what it is -- AI-assisted, cross-checked, human-reviewed, research only, no return claim.
            # Degrade-safe: a failure here must never drop the answer above it.
            try:
                st.caption(AI_DISCLOSURE)
            except Exception:
                pass


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
    # WHY (real money, honesty): the projected corpus is in FUTURE (nominal) rupees. Over a
    # multi-decade SIP, inflation erodes what it actually buys, so a parent planning for retirement
    # must see the real (today's-money) value, not just the large nominal figure.
    real = real_value(proj.projected_value, int(sip_years))
    st.caption(f"In today's money (assuming {DEFAULT_INFLATION_PCT:.0f}%/yr inflation, roughly "
               f"India's long-run average) that {money(proj.projected_value)} is worth about "
               f"{money(real)} — inflation erodes purchasing power over long horizons. An "
               "assumption, not a prediction.")
    # WHY (honesty): the slider allows up to 30%/40 years, which compounds into an absurd,
    # misleading corpus if taken literally (a real risk for a non-expert reading a bare number).
    # The downside disclosure (returns can be negative -> a SIP can LOSE money) is ALWAYS shown by
    # sip_return_context, never gated on the live SENSEX fetch: when that fetch fails (a real
    # network/rate-limit case here), the projection must still carry it, or a parent sees a rosy
    # projected gain with no downside. When the benchmark IS available it grounds the assumption
    # against SENSEX's own real, live-computed long-term price return so the reader can judge how
    # aggressive it is, rather than an arbitrary cap or a "typical equity return" claim from memory.
    bench_hist = historical_cagr(fetch_long_history_close(SENSEX_SYMBOL))
    st.caption(sip_return_context(sip_return, bench_hist))

with st.expander("Glossary"):
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}** — {meaning}")

# --- persistent compliance footer (SPEC v4 §6, SEBI): always-visible AI-usage disclosure ---
# WHY (real money, SEBI Reg 16C / Jan-2025 guidelines): the parents must always be able to see, in
# the app's own calm voice, that this is AI-assisted, cross-checked, human-reviewed research -- NOT
# advice, NOT a buy/sell call, with no return/accuracy/win-rate claim. The operator owns the AI
# output and must DISCLOSE that AI is used; this footer is that disclosure, shown at the bottom of
# every page alongside the standard not-advice/verify-every-figure line. Degrade-safe: a markup
# failure falls back to plain captions, and the whole block is wrapped so it can never blank the page.
try:
    st.divider()
    try:
        st.markdown(
            '<div class="ier-note">'
            '<div class="ier-note-title">About this tool</div>'
            f'{AI_DISCLOSURE}<br>{DISCLAIMER}'
            '</div>', unsafe_allow_html=True)
    except Exception:  # pragma: no cover - the disclosure must always show
        st.caption(AI_DISCLOSURE)
        st.caption(DISCLAIMER)
except Exception:  # pragma: no cover - never let the footer blank the page
    st.caption(AI_DISCLOSURE)
