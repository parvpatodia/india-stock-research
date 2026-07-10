import json

from src.data.annual_report_source import AnnualReportFigureSource, parse_extraction
from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource
from src.llm.client import LLMClient
from src.pipeline import build_report_for_symbol
from src.research.report import Confidence, QualityTier

AR_TEXT = (
    "Reliance Industries Integrated Annual Report. "
    "Profit for the year was 80,775 crore. "
    "Net cash generated from operating activities was 1,92,113 crore. "
    "Total borrowings stood at 3,98,000 crore. "
    "Total equity was 9,04,030 crore. "
    "Earnings before interest and tax (EBIT) was 1,47,218 crore. "
    "Finance costs (interest expense) were 24,056 crore."
)


class FakeClient(LLMClient):
    def __init__(self, response, available=True):
        self._response = response
        self._available = available

    @property
    def available(self):
        return self._available

    def complete(self, system, user, max_tokens=1000, json_mode=False, json_schema=None):
        return self._response


class FakeYF(FigureSource):
    source_id = "yfinance"

    def __init__(self, data):
        self._data = data

    def figures(self, symbol):
        return {name: self._data.get(name) for name in FRAMEWORK_FIGURES}


class FakeYearSource(FigureSource):
    def __init__(self, source_id, series):
        self.source_id = source_id
        self._series = series

    def figures(self, symbol):
        return {name: None for name in FRAMEWORK_FIGURES}

    def figures_by_year(self, symbol):
        return self._series


# --- pure parse_extraction (grounding + unit conversion) ---

def test_parse_grounded_crore_converts_to_absolute():
    payload = {"net_profit": {"value": 80775, "unit": "crore",
                              "quote": "Profit for the year was 80,775 crore"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] == 80775 * 1e7


def test_parse_rejects_ungrounded_quote():
    payload = {"net_profit": {"value": 99999, "unit": "crore",
                              "quote": "a sentence that is not in the report"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] is None


def test_parse_rejects_unknown_unit():
    payload = {"net_profit": {"value": 80775, "unit": "zorkmids",
                              "quote": "Profit for the year was 80,775 crore"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] is None


def test_parse_handles_comma_string_value():
    payload = {"equity": {"value": "9,04,030", "unit": "crore",
                          "quote": "Total equity was 9,04,030 crore"}}
    assert parse_extraction(payload, AR_TEXT)["equity"] == 904030 * 1e7


def test_parse_null_value_is_none():
    payload = {"net_profit": {"value": None, "unit": "crore", "quote": "x"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] is None


def test_parse_accepts_a_whole_number_value_serialized_as_a_json_float():
    # WHY (adversarial-review finding): the extraction schema types "value" as a generic JSON
    # "number", so a model can legitimately emit a whole-number figure as 80775.0 rather than
    # 80775. str(80775.0) is "80775.0", which digit-strips to "807750" -- a spurious extra
    # trailing zero that would never match the real figure's digits in the quote or the report,
    # silently rejecting perfectly legitimate annual-report data (fails closed, so not a false
    # "verified" fact, but it weakens this source's real contribution to cross-verification).
    text = "Profit for the year was 80,775 crore, up from the prior year."
    payload = {"net_profit": {"value": 80775.0, "unit": "crore",
                              "quote": "Profit for the year was 80,775 crore"}}
    assert parse_extraction(payload, text)["net_profit"] == 80775 * 1e7


def test_numeric_grounding_accepts_real_number_with_bad_quote():
    # The number 80,775 IS in the report; the quote is paraphrased (as garbled PDF text causes).
    payload = {"net_profit": {"value": 80775, "unit": "crore", "quote": "paraphrased, not verbatim"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] == 80775 * 1e7


def test_numeric_grounding_rejects_absent_number():
    payload = {"net_profit": {"value": 12345, "unit": "crore", "quote": "not in report"}}
    assert parse_extraction(payload, AR_TEXT)["net_profit"] is None


def test_numeric_grounding_rejects_concatenated_digit_spoof():
    # WHY (regression): "26248" appears inside the report's concatenated digits ("1,262" + "48"
    # -> "...126248..."), but is NOT a real number token in the report -> must be rejected.
    text = "Revenue grew to 1,262 crore across 48 branches."
    spoof = {"net_profit": {"value": 26248, "unit": "crore", "quote": "not verbatim"}}
    assert parse_extraction(spoof, text)["net_profit"] is None
    real = {"net_profit": {"value": 1262, "unit": "crore", "quote": "not verbatim"}}
    assert parse_extraction(real, text)["net_profit"] == 1262 * 1e7   # a real token is accepted


def test_quote_grounding_rejects_a_real_but_numberless_quote_with_a_fabricated_value():
    # WHY (real money, HIGH severity): quote_grounded only checked that the QUOTE STRING was a
    # real substring of the report -- not that the quoted excerpt actually CONTAINS the claimed
    # number. A model could attach a genuine, numberless narrative sentence from the report (e.g.
    # management commentary with no digits at all) to a completely FABRICATED value, and it would
    # pass as "grounded" via quote_grounded alone, since quote_grounded OR numeric_grounded is
    # sufficient. Confirmed live: this exact shape returned a fabricated 999999cr net_profit
    # instead of None. The report's own cross-verification layer happens to catch this in the
    # common 3-source case (the fabricated value becomes a withheld CONFLICT/outlier against
    # yfinance+Screener agreeing), but this source's OWN grounding check should not depend on a
    # separate layer to catch what it should reject outright.
    text = ("Management commentary: Profit for the year has grown significantly compared to "
            "last year, driven by strong demand.")
    payload = {"net_profit": {"value": 999999, "unit": "crore",
                              "quote": "Profit for the year has grown significantly"}}
    assert parse_extraction(payload, text)["net_profit"] is None


def test_quote_grounding_still_accepts_a_quote_that_genuinely_contains_the_value():
    # The legitimate case this fix must not break: a real quote that DOES contain the number.
    text = "Profit for the year was 80,775 crore, up from the prior year."
    payload = {"net_profit": {"value": 80775, "unit": "crore",
                              "quote": "Profit for the year was 80,775 crore"}}
    assert parse_extraction(payload, text)["net_profit"] == 80775 * 1e7


# --- source behavior ---

def test_ar_source_no_llm_returns_all_none():
    src = AnnualReportFigureSource(lambda s: AR_TEXT, client=FakeClient("", available=False))
    assert all(v is None for v in src.figures("X").values())


def test_ar_source_abstains_when_text_provider_raises():
    # WHY (regression 2026-07-09): a report PDF that times out / 403s / won't parse must abstain,
    # not crash the primary "Research" button with a page-wide stack trace.
    def boom(_symbol):
        raise TimeoutError("PDF download timed out")
    src = AnnualReportFigureSource(boom, client=FakeClient("{}", available=True))
    assert all(v is None for v in src.figures("X").values())   # no exception, no figures
    assert src.figures_by_year("X") == {}


def test_ar_source_extracts_and_converts():
    resp = json.dumps({"net_profit": {"value": 80775, "unit": "crore",
                                      "quote": "Profit for the year was 80,775 crore"}})
    src = AnnualReportFigureSource(lambda s: AR_TEXT, client=FakeClient(resp))
    assert src.figures("X")["net_profit"] == 80775 * 1e7


# --- the payoff: AR extraction + yfinance cross-verify into trusted figures ---

def test_annual_report_and_yfinance_crossverify():
    yf = FakeYF({
        "net_profit": 807.75e9, "operating_cash_flow": 1.92113e12, "total_debt": 3.98e12,
        "equity": 9.0403e12, "ebit": 1.47218e12, "interest_expense": 2.4056e11,
        "current_pe": 21.84,
    })
    ar_json = json.dumps({
        "net_profit": {"value": 80775, "unit": "crore", "quote": "Profit for the year was 80,775 crore"},
        "operating_cash_flow": {"value": 192113, "unit": "crore", "quote": "operating activities was 1,92,113 crore"},
        "total_debt": {"value": 398000, "unit": "crore", "quote": "Total borrowings stood at 3,98,000 crore"},
        "equity": {"value": 904030, "unit": "crore", "quote": "Total equity was 9,04,030 crore"},
        "ebit": {"value": 147218, "unit": "crore", "quote": "(EBIT) was 1,47,218 crore"},
        "interest_expense": {"value": 24056, "unit": "crore", "quote": "interest expense) were 24,056 crore"},
    })
    ar = AnnualReportFigureSource(lambda s: AR_TEXT, client=FakeClient(ar_json))

    r = build_report_for_symbol("RELIANCE", [yf, ar])
    net_profit = next(f for f in r.figures if f.name == "net_profit")
    assert net_profit.is_trustworthy                    # cross-verified across two sources
    assert r.verdict.quality == QualityTier.STRONG      # computed from verified figures
    assert r.verdict.confidence in (Confidence.MEDIUM, Confidence.HIGH)


def test_ar_figures_by_year_tags_fiscal_year():
    resp = json.dumps({"fiscal_year": 2026,
                       "net_profit": {"value": 80775, "unit": "crore",
                                      "quote": "Profit for the year was 80,775 crore"}})
    src = AnnualReportFigureSource(lambda s: AR_TEXT, client=FakeClient(resp))
    assert src.figures_by_year("X")["net_profit"] == {2026: 80775 * 1e7}


def test_annual_report_breaks_a_two_source_conflict():
    # yfinance and Screener disagree on FY2026 net profit; the annual report confirms yfinance.
    yf = FakeYearSource("yfinance", {"net_profit": {2026: 807.75e9}})
    screener = FakeYearSource("screener", {"net_profit": {2026: 957.54e9}})  # the outlier
    ar_json = json.dumps({"fiscal_year": 2026,
                          "net_profit": {"value": 80775, "unit": "crore",
                                         "quote": "Profit for the year was 80,775 crore"}})
    ar = AnnualReportFigureSource(lambda s: AR_TEXT, client=FakeClient(ar_json))

    r = build_report_for_symbol("RELIANCE", [yf, screener, ar])
    net_profit = next(f for f in r.figures if f.name == "net_profit")
    assert net_profit.is_trustworthy                                  # tie broken
    assert abs(net_profit.value - 807.75e9) / 807.75e9 < 0.02         # resolved to yf/AR, not screener
    assert "screener" in net_profit.note                              # screener named as outlier
