from src.research.report import Report
from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
from src.research.verified_context import VERIFIED_FIGURES_SOURCE_ID, verified_figures_document


def _verified(name, value, note="2 independent sources agree") -> VerifiedFigure:
    return VerifiedFigure(name, VerificationStatus.VERIFIED, value,
                          (SourcedValue(value, "yfinance"), SourcedValue(value, "screener")), note)


def _single_source(name, value) -> VerifiedFigure:
    return VerifiedFigure(name, VerificationStatus.SINGLE_SOURCE, value,
                          (SourcedValue(value, "yfinance"),), "only one independent source")


def test_document_includes_only_cross_verified_figures():
    # WHY (real money): the Ask tab must ground answers only in numbers that ALREADY passed the
    # >=2-source cross-verification bar. A single-source or conflicting figure must never appear
    # in this document, or the Ask tab could state an unverified number as if it were settled.
    report = Report(company="X", figures=(
        _verified("net_profit", 958000000.0),
        _single_source("total_debt", 500000000.0),   # must be excluded
    ))
    doc = verified_figures_document("RELIANCE", report)
    assert doc is not None
    assert "958,000,000" in doc.text
    assert "500,000,000" not in doc.text            # single-source figure withheld
    assert doc.source_id == VERIFIED_FIGURES_SOURCE_ID


def test_document_carries_insights_and_readable_labels():
    report = Report(company="X", insights=("Track record: sales have been growing 12% a year.",),
                    figures=(_verified("current_pe", 18.2), _verified("promoter_pledge_pct", 0.0)))
    doc = verified_figures_document("RELIANCE", report)
    assert "Track record: sales have been growing 12% a year." in doc.text
    assert "P/E" in doc.text and "18.2" in doc.text          # human label, not raw 'current_pe'
    assert "0.0%" in doc.text or "0%" in doc.text             # pledge formatted as a percentage


def test_document_formats_dividend_yield_as_a_percentage():
    report = Report(company="X", figures=(_verified("dividend_yield_pct", 0.47),))
    doc = verified_figures_document("RELIANCE", report)
    assert "Dividend yield" in doc.text and "0.5%" in doc.text  # human label, percent not rupees


def test_document_discloses_when_the_figures_were_fetched():
    # WHY (real money, honesty): the Ask tab stamps every citation's as_of with the CURRENT time
    # (when the question is asked), not when these figures were actually fetched. A user who
    # researched a stock hours earlier in the same session and asks a question later would see a
    # citation implying today's-figures freshness for data that could be substantially stale. The
    # document text must self-disclose the real fetch time, the same self-disclosure pattern
    # already used for news (dated, attributed) and the annual-report reader (📄, self-reported).
    report = Report(company="X", created_at="2026-07-09T09:00:00Z",
                    figures=(_verified("current_pe", 18.2),))
    doc = verified_figures_document("RELIANCE", report)
    assert "2026-07-09T09:00:00Z" in doc.text


def test_no_document_when_nothing_verified():
    report = Report(company="X", figures=(_single_source("net_profit", 100.0),))
    assert verified_figures_document("RELIANCE", report) is None


def test_no_report_returns_none():
    assert verified_figures_document("RELIANCE", None) is None
