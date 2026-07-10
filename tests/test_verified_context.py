from src.research.report import Report
from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
from src.research.verified_context import (
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
    assert "₹95.8 crore" in doc.text                 # rendered in the Indian crore convention
    assert "50 crore" not in doc.text                # single-source figure withheld entirely
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


def test_promoter_trend_document_carries_the_sentence_and_source_id():
    # WHY (research rigor): promoter behavior/shareholding-pattern signals are a core Indian-
    # investor framework the Research tab already surfaces (Screener-only, single-source, always
    # self-disclosed as "not cross-verified" inline) but the Ask tab had zero access to it, so a
    # question like "has the promoter been selling?" could never be grounded even when the app had
    # already fetched the answer. Wrap it the same way news is wrapped: a citable, attributed
    # document, at ANALYST tier (never PRIMARY/citable_as_fact) so it can only ever be shown as
    # reported context, matching the caveat already embedded in the sentence itself.
    trend = ("Promoter holding has decreased from 55.0% (Mar 2023) to 48.0% (Mar 2026); a falling "
            "promoter stake can reflect a stake sale, a merger/reclassification, or dilution; "
            "check exchange filings or recent news for the actual reason (not cross-verified, "
            "Screener only).")
    doc = promoter_trend_document("RELIANCE", trend)
    assert doc is not None
    assert doc.source_id == PROMOTER_TREND_SOURCE_ID
    assert "RELIANCE" in doc.text
    assert trend in doc.text


def test_promoter_trend_document_none_when_no_trend_available():
    assert promoter_trend_document("RELIANCE", None) is None
    assert promoter_trend_document("RELIANCE", "") is None


def test_cash_conversion_trend_document_carries_the_sentence_and_source_id():
    # WHY (research rigor, cash-flow discipline): mirrors promoter_trend_document exactly --
    # Screener-only, single-source, self-disclosed "not cross-verified" context that the Research
    # tab already surfaces but the Ask tab had no access to at all.
    trend = ("Cash conversion cycle has lengthened from -2 days (FY2015) to 25 days (FY2026); a "
            "lengthening cash cycle can mean slower collections, rising inventory, or weaker "
            "supplier terms; worth checking against sector peers and recent quarters (not "
            "cross-verified, Screener only).")
    doc = cash_conversion_trend_document("RELIANCE", trend)
    assert doc is not None
    assert doc.source_id == CASH_CONVERSION_TREND_SOURCE_ID
    assert "RELIANCE" in doc.text
    assert trend in doc.text


def test_cash_conversion_trend_document_none_when_no_trend_available():
    assert cash_conversion_trend_document("RELIANCE", None) is None
    assert cash_conversion_trend_document("RELIANCE", "") is None


def test_other_income_share_document_carries_the_sentence_and_source_id():
    # WHY (research rigor, quality of earnings): mirrors promoter_trend_document /
    # cash_conversion_trend_document exactly -- Screener-only, single-source, self-disclosed
    # "not cross-verified" context that the Research tab already surfaces but the Ask tab had no
    # access to at all.
    trend = ("27% of FY2026's profit before tax came from non-operating \"other income\" "
            "(investment gains, interest income, or one-off items) rather than the core "
            "business -- worth checking how repeatable that income is (not cross-verified, "
            "Screener only).")
    doc = other_income_share_document("RELIANCE", trend)
    assert doc is not None
    assert doc.source_id == OTHER_INCOME_SHARE_SOURCE_ID
    assert "RELIANCE" in doc.text
    assert trend in doc.text


def test_other_income_share_document_none_when_no_data_available():
    assert other_income_share_document("RELIANCE", None) is None
    assert other_income_share_document("RELIANCE", "") is None


def test_symbol_has_no_data_true_only_when_every_signal_is_empty():
    assert symbol_has_no_data("", verified_figures_found=False, promoter_trend_found=False,
                             cash_conversion_trend_found=False,
                             other_income_share_found=False) is True


def test_symbol_has_no_data_false_when_yfinance_name_resolved():
    assert symbol_has_no_data("Reliance Industries", False, False, False, False) is False


def test_symbol_has_no_data_false_when_only_promoter_trend_resolved():
    # WHY (real money, honesty): the actual bug this guards against -- a real, valid NSE symbol
    # where yfinance's own name lookup comes back empty (a known Yahoo India-coverage gap) but
    # Screener has data. `company` alone (the OLD, sole signal) would wrongly call this symbol
    # unresolved even though real per-symbol data was just fetched and used to answer the question.
    assert symbol_has_no_data("", False, promoter_trend_found=True,
                             cash_conversion_trend_found=False,
                             other_income_share_found=False) is False


def test_symbol_has_no_data_false_when_only_verified_figures_resolved():
    assert symbol_has_no_data("", verified_figures_found=True, promoter_trend_found=False,
                             cash_conversion_trend_found=False,
                             other_income_share_found=False) is False


def test_symbol_has_no_data_false_when_only_cash_conversion_trend_resolved():
    # WHY: same "Screener has data even when yfinance's name lookup is empty" shape as the
    # promoter-trend case, for a sibling per-symbol signal.
    assert symbol_has_no_data("", False, False, cash_conversion_trend_found=True,
                             other_income_share_found=False) is False


def test_symbol_has_no_data_false_when_only_other_income_share_resolved():
    # WHY: same shape, for the newest of the four per-symbol signals.
    assert symbol_has_no_data("", False, False, False, other_income_share_found=True) is False
