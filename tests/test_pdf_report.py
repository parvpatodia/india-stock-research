"""build_pdf_report lives in app.py, so it's exercised by importing the module directly (same
env-safe pattern as test_app_cache_ttls.py) and reading the generated PDF's real text back out
via pypdf -- the same library src/research/library.py already uses to read a PDF's content.
"""
import io
import os

from src.analysis.sizing import Stance
from src.research.report import Confidence, Leaning, QualityTier, Report, ValuationTier, Verdict


def _import_app_with_clean_env():
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        import app
        return app
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _report():
    v = Verdict(ValuationTier.FAIR, QualityTier.STRONG, Leaning.NEUTRAL, Confidence.MEDIUM,
               reasons=("debt/equity 0.60 reads moderate.",))
    return Report(company="RELIANCE", verdict=v, insights=("Price: fairly valued.",))


def test_pdf_includes_the_single_source_context_signals():
    # WHY (real money, UI honesty): the download button is labeled "Download full report" -- a
    # parent who saves this PDF to review offline or share with family must see the SAME
    # promoter-trend / cash-conversion-cycle / other-income-share signals the live Research tab
    # shows in their own expanders, not a report silently missing three of the app's own signals.
    app = _import_app_with_clean_env()
    pdf_bytes = app.build_pdf_report(
        "RELIANCE (live)", _report(), Stance.NEUTRAL,
        promoter_trend="Promoter holding has stayed roughly steady near 50.0%.",
        cash_conversion_trend="Cash conversion cycle has lengthened from -2 days to 25 days.",
        other_income_share="27% of FY2026's profit before tax came from non-operating "
                           "\"other income\".",
        promoter_pledge="Screener flags that promoters have pledged 73% of their holding.")
    text = _pdf_text(pdf_bytes)
    assert "Promoter holding has stayed roughly steady near 50.0%." in text
    assert "Cash conversion cycle has lengthened from -2 days to 25 days." in text
    assert "27% of FY2026's profit" in text
    assert "promoters have pledged 73% of their holding" in text   # the red flag must be in the PDF
    assert "Additional context" in text
    assert "cannot cross-verify" in text


def test_pdf_always_carries_the_full_disclaimer_incl_no_data_report():
    # WHY (real money, honesty): the "Download full report" PDF is saved/shared OFFLINE, away from
    # the app's always-visible footer disclaimer. A no-verdict (insufficient/no-data) report carried
    # NO caveat at all in the PDF -- inconsistent with the app, which shows the full DISCLAIMER for a
    # no-verdict report. And even a normal report's verdict caveat omits "verify every figure / data
    # may be delayed or incorrect / you alone are responsible" -- exactly what a shared document needs.
    # Every PDF must carry the full app disclaimer; a report WITH a verdict keeps its verdict caveat too.
    app = _import_app_with_clean_env()

    def norm(pdf_bytes):
        return " ".join(_pdf_text(pdf_bytes).split()).lower()

    no_data = norm(app.build_pdf_report("XYZ (live)", Report(company="XYZ", verdict=None),
                                        Stance.INSUFFICIENT_DATA))
    assert "not investment advice" in no_data and "verify every figure" in no_data
    normal = norm(app.build_pdf_report("RELIANCE (live)", _report(), Stance.NEUTRAL))
    assert "verify every figure" in normal          # the full disclaimer is present
    assert "caveated opinion" in normal             # AND the verdict-specific caveat is still there


def test_pdf_omits_the_section_entirely_when_no_signal_is_available():
    # WHY: don't show an empty/misleading "Additional context" header when nothing was fetched
    # (e.g. Screener was unreachable) -- omission must read as "nothing shown", not "nothing
    # exists", but an empty section header would look like a broken/incomplete report.
    app = _import_app_with_clean_env()
    pdf_bytes = app.build_pdf_report("RELIANCE (live)", _report(), Stance.NEUTRAL)
    text = _pdf_text(pdf_bytes)
    assert "Additional context" not in text


def test_pdf_latin1_maps_typographic_chars_to_readable_ascii_not_question_marks():
    # WHY (real money, the shared PDF's credibility): fpdf's core font is Latin-1, and this app's
    # insights/caveats use the em dash PERVASIVELY (deep_metrics/trends/framework). A bare
    # encode('latin-1','replace') turned every em dash into '?', so a parent who downloads "the full
    # report" to review or share with family saw "... net margin 18%) ? strong" -- reads as
    # corruption in a document about their real money. Map the common typographic characters the
    # app's own copy uses to readable ASCII BEFORE the lossy encode; the encode stays as a safety net.
    app = _import_app_with_clean_env()
    f = app._pdf_latin1
    assert f("net margin 18%) — strong") == "net margin 18%) - strong"   # em dash -> hyphen
    assert "?" not in f("Leverage rose — check it; margins fell –3%")
    assert f("₹1,00,000") == "Rs.1,00,000"                               # rupee sign
    assert f("2019–2024") == "2019-2024"                                 # en dash
    assert f("“quote” ‘x’") == "\"quote\" 'x'"            # curly quotes
    assert f("wait…") == "wait..."                                       # ellipsis
    assert f("ROA 1.1% (strong)") == "ROA 1.1% (strong)"                     # plain ASCII unchanged
    # the latin-1 safety net still catches anything UNmapped, so a stray symbol never crashes the build
    assert "?" in f("★ star")


def test_pdf_renders_em_dash_insights_without_question_mark_garble():
    # end-to-end: an em-dash insight must read back from the real PDF as '-', never '?' garble.
    app = _import_app_with_clean_env()
    rep = Report(company="RELIANCE", verdict=_report().verdict,
                 insights=("It keeps about Rs18 from every Rs100 of sales (net margin 18%) "
                           "— strong.",))
    norm = " ".join(_pdf_text(app.build_pdf_report("RELIANCE (live)", rep, Stance.NEUTRAL)).split())
    assert "18%) - strong" in norm            # em dash rendered as a hyphen
    assert "18%) ? strong" not in norm        # never the '?' garble


def test_pdf_carries_the_annual_data_vintage_caveat_when_figures_are_annual():
    # WHY (real money, honesty): the saved/shared PDF must disclose that the fundamentals rest on
    # the latest ANNUAL figures (recent quarters not included), the same vintage caveat the live
    # Research view shows -- so a parent reviewing the PDF offline isn't misled by a stale headline.
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    app = _import_app_with_clean_env()
    report = Report(company="RELIANCE", verdict=_report().verdict, figures=(
        VerifiedFigure("net_profit", VerificationStatus.VERIFIED, 100.0,
                       (SourcedValue(100.0, "yfinance", locator="FY2025"),
                        SourcedValue(100.0, "screener", locator="FY2025")), "agree"),
    ))
    text = " ".join(_pdf_text(app.build_pdf_report("RELIANCE (live)", report, Stance.NEUTRAL)).split())
    assert "latest ANNUAL results (FY2025)" in text
    assert "recent quarters" in text.lower()
