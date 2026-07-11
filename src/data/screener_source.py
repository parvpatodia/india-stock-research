"""Screener.in figure source: a free, independent, structured second source.

Screener publishes each company's annual P&L, balance sheet, and cash flow (in Rs crore) on a
public page with no login. This parses those tables into the framework's figures and converts
crore -> absolute rupees IN CODE (Screener is always in crore), so the values cross-check
against yfinance's absolute figures. It is an unofficial source (scraping, ToS grey, can break
when the page changes), which is exactly why every figure still has to agree with yfinance
before it is trusted; a parse error or a page change shows up as a conflict, not a bad fact.
"""
from __future__ import annotations

import io
import re
from typing import Callable

import pandas as pd

from ..constants import PROMOTER_PLEDGE_HIGH_PCT
from .figure_sources import FRAMEWORK_FIGURES, FigureSource

_CRORE = 1e7
# WHY anchor on the value's own <span> (real money, data quality): Screener renders each top ratio
# as "<label> <span ...>value</span>". A non-greedy ".*?(number)" instead grabbed the NEXT number
# ANYWHERE after the label, so a loss-making company showing "Stock P/E" with NO value silently took
# the following ratio (e.g. Book Value) as its P/E -- a fabricated figure. Requiring the number to
# sit inside the span immediately after the label means a valueless label matches nothing, not an
# unrelated field. \s* (which includes newlines) removes the need for DOTALL.
_PE_RE = re.compile(r"Stock P/E\s*<span[^>]*>\s*([\d,]+\.?\d+)", re.IGNORECASE)
_DIV_YIELD_RE = re.compile(r"Dividend Yield\s*<span[^>]*>\s*([\d,]+\.?\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(\d{4})\s*$")
_MONTH_YEAR_RE = re.compile(r"([A-Za-z]{3})\s+(\d{4})\s*$")


def _clean_label(value) -> str:
    return str(value).replace("\xa0", " ").replace("+", "").strip().lower()


def _num(value) -> float | None:
    try:
        s = str(value).replace(",", "").replace("%", "").strip()
        if s in ("", "nan", "-", "none"):
            return None
        f = float(s)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _is_annual(df) -> bool:
    # WHY (resilience): Screener's quarterly results table precedes the annual P&L and shares its
    # row labels, so "distinct years in the first few columns" misfires on a quarter that straddles
    # a calendar year (e.g. Dec 2023 / Mar 2024 / Jun 2024). Annual columns instead all share ONE
    # fiscal-year-end month across >=2 distinct years; a quarterly table cycles through months. Test
    # every dated column so a truncated view can't fool it.
    months: set[str] = set()
    years: set[str] = set()
    for col in list(df.columns)[1:]:
        s = str(col).strip()
        my = _MONTH_YEAR_RE.search(s)
        if my:
            months.add(my.group(1).lower())
            years.add(my.group(2))
        else:
            y = _YEAR_RE.search(s)          # bare "YYYY" / "FY2024": still a yearly period column
            if y:
                years.add(y.group(1))
    # >=2 distinct years across the columns, and at most one fiscal-year-end month. Quarterly
    # tables always cycle through months (>1), so they stay excluded; year-only headers pass.
    return len(years) >= 2 and len(months) <= 1


def _latest_annual(df, label: str) -> float | None:
    if df is None:
        return None
    # WHY: take the rightmost column whose header is a fiscal year (e.g. "Mar 2024"), skipping
    # a "TTM" column. Otherwise a P&L row would be read TTM while yfinance reports the latest
    # full year, a period mismatch that would (correctly but unhelpfully) show as a conflict.
    year_idx = None
    for i, col in enumerate(df.columns):
        if i > 0 and _YEAR_RE.search(str(col)):
            year_idx = i
    if year_idx is None:
        return None
    for _, row in df.iterrows():
        lab = _clean_label(row.iloc[0])
        if lab == label or lab.startswith(label):
            return _num(row.iloc[year_idx])
    return None


def _find_tables(html: str):
    """Return (pnl, balance, cash) DataFrames from Screener HTML, or Nones."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    pnl = balance = cash = None
    for df in tables:
        if df.shape[1] < 2:
            continue
        labels = {_clean_label(v) for v in df.iloc[:, 0]}
        # WHY accept "financing profit" too (real money, data quality; live-verified on HDFCBANK):
        # a bank/NBFC's Screener P&L has no "operating profit" line -- it shows "Financing Profit"
        # instead. Gating only on "operating profit" left every bank/NBFC with an unparsed P&L, so
        # net_profit/pbt/interest came back None, net_profit never cross-verified, and the ROA-based
        # bank framework (the app's core lens for lenders) always read low-confidence. Both labels
        # still require "net profit" + an annual layout, so this can't misidentify a non-P&L table.
        if pnl is None and any("net profit" in l for l in labels) \
                and any("operating profit" in l or "financing profit" in l for l in labels) \
                and _is_annual(df):
            pnl = df
        # WHY "borrowing" not "borrowings" (real money, data quality; live-verified on HDFCBANK):
        # a bank's Screener balance sheet labels the row "Borrowing" (singular), so requiring the
        # plural left every bank/NBFC balance sheet unidentified -> total_assets/equity/total_debt
        # all None -> single-source -> the bank's balance-sheet figures never cross-verified. The
        # singular substring also matches a non-bank's "Borrowings" (the row-level lookups use
        # startswith), so this correctly recognizes both.
        if balance is None and any("borrowing" in l for l in labels) \
                and any("equity capital" in l for l in labels):
            balance = df
        if cash is None and any("cash from operating activity" in l for l in labels):
            cash = df
    return pnl, balance, cash


def _find_shareholding_table(html: str):
    """The 'Shareholding Pattern' table (Promoters/FIIs/DIIs/Public %, quarterly), verified live
    against screener.in/company/RELIANCE/consolidated/. Two such tables exist on a real page
    (quarterly and a longer yearly view); this returns the FIRST match (the quarterly one, as
    Screener renders it first), giving the more granular recent trend."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    for df in tables:
        if df.shape[1] < 2:
            continue
        labels = {_clean_label(v) for v in df.iloc[:, 0]}
        if any(l.startswith("promoters") for l in labels) and any(l.startswith("public") for l in labels):
            return df
    return None


def _annual_series(df, label: str) -> dict[int, float]:
    """{fiscal_year: value} across every year column for the first matching row."""
    out: dict[int, float] = {}
    if df is None:
        return out
    for _, row in df.iterrows():
        lab = _clean_label(row.iloc[0])
        if lab == label or lab.startswith(label):
            for col in list(df.columns)[1:]:
                m = _YEAR_RE.search(str(col))
                n = _num(row[col])
                if m and n is not None:
                    out[int(m.group(1))] = n
            break
    return out


def parse_screener_series(html: str) -> dict[str, dict[int, float]]:
    """Pure: per-fiscal-year series for the statement figures (absolute rupees)."""
    pnl, balance, cash = _find_tables(html)
    net = _annual_series(pnl, "net profit")
    pbt = _annual_series(pnl, "profit before tax")
    interest = _annual_series(pnl, "interest")
    sales = _annual_series(pnl, "sales")
    ocf = _annual_series(cash, "cash from operating activity")
    debt = _annual_series(balance, "borrowing")
    eqcap = _annual_series(balance, "equity capital")
    reserves = _annual_series(balance, "reserves")
    tassets = _annual_series(balance, "total liabilities")
    series = {
        "net_profit": {y: v * _CRORE for y, v in net.items()},
        "operating_cash_flow": {y: v * _CRORE for y, v in ocf.items()},
        "total_debt": {y: v * _CRORE for y, v in debt.items()},
        "interest_expense": {y: abs(v) * _CRORE for y, v in interest.items()},
        "ebit": {y: (pbt[y] + interest[y]) * _CRORE for y in pbt if y in interest},
        "equity": {y: (eqcap[y] + reserves[y]) * _CRORE for y in eqcap if y in reserves},
        "total_assets": {y: v * _CRORE for y, v in tassets.items()},
        "revenue": {y: v * _CRORE for y, v in sales.items()},
    }
    return {name: yearmap for name, yearmap in series.items() if yearmap}


def parse_screener_figures(html: str) -> dict[str, float | None]:
    """Pure: parse Screener HTML into framework figures (absolute rupees). Missing -> None."""
    out: dict[str, float | None] = {name: None for name in FRAMEWORK_FIGURES}
    pnl, balance, cash = _find_tables(html)

    net_profit = _latest_annual(pnl, "net profit")
    pbt = _latest_annual(pnl, "profit before tax")
    interest = _latest_annual(pnl, "interest")
    ocf = _latest_annual(cash, "cash from operating activity")
    debt = _latest_annual(balance, "borrowing")
    equity_capital = _latest_annual(balance, "equity capital")
    reserves = _latest_annual(balance, "reserves")

    if net_profit is not None:
        out["net_profit"] = net_profit * _CRORE
    if ocf is not None:
        out["operating_cash_flow"] = ocf * _CRORE
    if debt is not None:
        out["total_debt"] = debt * _CRORE
    if interest is not None:
        out["interest_expense"] = abs(interest) * _CRORE
    if pbt is not None and interest is not None:
        out["ebit"] = (pbt + interest) * _CRORE  # EBIT = profit before tax + interest
    if equity_capital is not None and reserves is not None:
        out["equity"] = (equity_capital + reserves) * _CRORE  # shareholders' equity
    total_assets = _latest_annual(balance, "total liabilities")  # Screener's grand total = assets
    if total_assets is not None:
        out["total_assets"] = total_assets * _CRORE
    revenue = _latest_annual(pnl, "sales")
    if revenue is not None:
        out["revenue"] = revenue * _CRORE
    m = _PE_RE.search(html)
    if m:
        out["current_pe"] = _num(m.group(1))
    m = _DIV_YIELD_RE.search(html)
    if m:
        out["dividend_yield_pct"] = _num(m.group(1))
    return out


# Screener surfaces a material promoter pledge as a narrative flag, e.g. "Promoters have pledged
# 73.0% of their holding." Live-verified it appears only when the pledge is material (JPPOWER
# 73%), and is absent for unpledged names -- so absence means "not flagged", NOT zero pledge.
_PLEDGE_RE = re.compile(r"promoter[s]?\s+have\s+pledged\s+([\d.]+)\s*%", re.IGNORECASE)


def parse_promoter_pledge(html: str) -> float | None:
    """The % of promoter holding pledged, from Screener's narrative flag, or None if not flagged.
    SINGLE-SOURCE (Screener only): yfinance does not carry Indian promoter-pledge data, so this
    can never cross-verify; callers disclose that (see promoter_pledge_point) and never present it
    as a cross-verified fact. WHY None-not-zero when absent: Screener only shows the flag when the
    pledge is material, so its absence is 'not flagged', never a reassuring 0% we could fabricate."""
    m = _PLEDGE_RE.search(html or "")
    return _num(m.group(1)) if m else None


def promoter_pledge_point(pct: float | None) -> str | None:
    """A self-disclosing plain-language sentence on promoter pledge, or None if unavailable.

    WHY (real money): pledged promoter shares are collateral a lender can SELL on a margin call if
    the price falls, forcing more selling and often signalling promoter cash stress -- a top-tier
    Indian-market red flag. Wording escalates above PROMOTER_PLEDGE_HIGH_PCT (the shared threshold
    the framework's own pledge metric uses) and always self-discloses 'not cross-verified, Screener
    only' so the caveat travels with the text; context for the reader, never a buy/sell signal and
    never mixed into the cross-verified figures."""
    if pct is None:
        return None
    base = (f"Screener flags that promoters have pledged {pct:.0f}% of their holding. Pledged "
            "shares can be sold by the lender if the price falls (a forced-selling risk) and often "
            "reflect promoter cash stress")
    tail = (" -- at this level a serious red flag; verify against the latest shareholding filing"
            if pct >= PROMOTER_PLEDGE_HIGH_PCT
            else "; worth checking against the latest shareholding filing")
    return f"{base}{tail} (not cross-verified, Screener only)."


# Below the promoter-holding threshold a move reads as "roughly steady" rather than overclaiming
# direction from ordinary quarter-to-quarter noise.
_HOLDING_STEADY_BAND = 0.5

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
# A "Mon YYYY" shareholding period label (e.g. "Mar 2024", "Jun 2023") -> a sortable (year, month).
_PERIOD_RE = re.compile(r"([A-Za-z]{3})[A-Za-z]*[\s'\-]*(\d{4})")


def _period_sort_key(label: str) -> tuple[int, int] | None:
    """Parse a shareholding period label like 'Mar 2024' into (year, month) for chronological
    ordering, or None if it doesn't parse. WHY (resilience): the promoter-holding DIRECTION
    (buying vs selling -- a real signal) must be read oldest->newest by actual period date, not by
    the order Screener happens to render its columns; a scrape-layout change that reversed the
    columns would otherwise flip the reported direction into its opposite."""
    m = _PERIOD_RE.search(str(label))
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    return (int(m.group(2)), month) if month is not None else None


def parse_promoter_holding_series(html: str) -> dict[str, float]:
    """{period_label: promoter_holding_pct} from the Shareholding Pattern table, in the SAME
    left-to-right (oldest -> newest) column order Screener renders. Percent as given (e.g. 50.39),
    not crore-scaled, this is a shareholding percentage, not a rupee figure. SINGLE-SOURCE by
    nature: yfinance does not carry historical Indian promoter-holding data, so this can never
    cross-verify the way the framework's financial figures do. Callers must disclose that (see
    promoter_holding_trend_point), never present it as a cross-verified fact."""
    df = _find_shareholding_table(html)
    out: dict[str, float] = {}
    if df is None:
        return out
    for _, row in df.iterrows():
        label = _clean_label(row.iloc[0])
        if label.startswith("promoters"):
            for col in list(df.columns)[1:]:
                n = _num(row[col])
                if n is not None:
                    out[str(col).strip()] = n
            break
    return out


def promoter_holding_trend_point(series: dict[str, float]) -> str | None:
    """A single, self-disclosing plain-language sentence on how promoter holding has moved across
    the available periods (oldest vs. latest), or None if there are fewer than 2 data points.
    Explicitly states 'not cross-verified, Screener only' inline so the caveat travels with the
    text wherever it is shown, this is context for the reader, never a buy/sell signal and never
    mixed into the app's cross-verified 'insights' without that disclosure attached."""
    if len(series) < 2:
        return None
    # Read oldest -> newest by ACTUAL period date, not by dict/column order (see _period_sort_key):
    # a scrape-layout change that reversed the columns must not flip the reported direction. Fall
    # back to insertion order only if a label doesn't parse, preserving the prior behaviour.
    keys = {label: _period_sort_key(label) for label in series}
    if all(k is not None for k in keys.values()):
        periods = sorted(series.items(), key=lambda kv: keys[kv[0]])
    else:
        periods = list(series.items())
    (first_label, first_val), (last_label, last_val) = periods[0], periods[-1]
    delta = last_val - first_val
    # WHY (real money, promoter-behavior rigor; mirrors leverage_trend_point's spike detection): an
    # oldest-vs-latest read hides a promoter stake reduction that RESOLVED -- 50% -> 30% -> 50% reads
    # "steady" on the endpoints, erasing a real temporary sell-down a parent asking "has the promoter
    # been selling?" needs to know. Surface a materially-lower intra-period trough (a MIDDLE period
    # more than the steady band below BOTH endpoints), reusing the same noise band the endpoint read
    # uses, so a promoter exit-and-return is never silently averaged away.
    middle = periods[1:-1]
    dip = ""
    if middle:
        trough_label, trough_val = min(middle, key=lambda lv: lv[1])
        if trough_val < min(first_val, last_val) - _HOLDING_STEADY_BAND:
            dip = f", though it dipped to {trough_val:.1f}% in {trough_label} in between"
    if abs(delta) < _HOLDING_STEADY_BAND:
        return (f"Promoter holding has stayed roughly steady near {last_val:.1f}% "
                f"({first_label} to {last_label}{dip}; not cross-verified, Screener only).")
    if delta > 0:
        read = ("promoters adding to their stake is often read as a positive signal, though it "
                "can also follow a preferential issue or warrant conversion")
    else:
        # WHY (live-verified against HDFC Bank's real data): a decrease can be an ordinary
        # stake sale, but can equally be a merger/reclassification (e.g. HDFC Ltd merging into
        # HDFC Bank left it with no designated promoter) -- neutral wording naming BOTH, not an
        # alarmist "worth watching" that would mislabel a benign structural event as a red flag.
        read = ("a falling promoter stake can reflect a stake sale, a merger/reclassification, "
                "or dilution; check exchange filings or recent news for the actual reason")
    direction = "increased" if delta > 0 else "decreased"
    return (f"Promoter holding has {direction} from {first_val:.1f}% ({first_label}) to "
            f"{last_val:.1f}% ({last_label}){dip}; {read} (not cross-verified, Screener only).")


# Below this many days a year-to-year move reads as ordinary noise, not a genuine multi-year
# drift. Calibrated against live Screener data across 3 real, different-sector companies:
# Reliance (FY2015-FY2026, -46 to +25 days), TCS (67-93 days, a services company with no
# inventory), HUL (-107 to -50 days) -- adjacent-year swings of 10-25 days are routine for all
# three without reflecting a structural change; 15 stays clear of that noise band.
_CCC_STEADY_BAND = 15.0


def _find_ratio_table(html: str):
    """The per-year efficiency-ratio table (Debtor Days, Inventory Days, Days Payable, Cash
    Conversion Cycle, ROCE %), verified live against screener.in/company/RELIANCE and .../TCS (a
    services company with no inventory still reports a valid Cash Conversion Cycle, from debtor
    days alone)."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    for df in tables:
        if df.shape[1] < 2:
            continue
        labels = {_clean_label(v) for v in df.iloc[:, 0]}
        if any(l.startswith("cash conversion cycle") for l in labels):
            return df
    return None


def parse_cash_conversion_cycle_series(html: str) -> dict[int, float]:
    """{fiscal_year: days} for the Cash Conversion Cycle row -- Screener's own computed
    working-capital-efficiency ratio (days debtors + inventory tie up cash, minus days payables
    fund it). SINGLE-SOURCE by nature (yfinance does not carry this), like promoter holding --
    callers must disclose that (see cash_conversion_cycle_trend_point), never present it as a
    cross-verified fact. A NEGATIVE value is normal and can be favorable (e.g. Reliance, HUL):
    it means suppliers effectively fund working capital, not a parsing error."""
    return _annual_series(_find_ratio_table(html), "cash conversion cycle")


def cash_conversion_cycle_trend_point(series: dict[int, float]) -> str | None:
    """A single, self-disclosing plain-language sentence on how the cash conversion cycle has
    moved across the available fiscal years (oldest vs. latest), or None if there are fewer than
    2 data points. A LENGTHENING cycle (more days) can signal slower collections, rising
    inventory, or weaker supplier terms -- worth the reader's attention as a cash-flow-discipline/
    quality-of-earnings signal; a SHORTENING cycle usually reflects tighter working-capital
    management. Explicitly states 'not cross-verified, Screener only' inline so the caveat
    travels with the text wherever it is shown; context for the reader, never a buy/sell signal."""
    if len(series) < 2:
        return None
    years = sorted(series)
    first_year, last_year = years[0], years[-1]
    first_val, last_val = series[first_year], series[last_year]
    delta = last_val - first_val
    # WHY (mirrors leverage_trend_point's spike detection and the promoter-holding dip): an oldest-
    # vs-latest read hides a working-capital STRESS episode that resolved -- CCC 20 -> 90 -> 25 days
    # reads "steady near 25" on the endpoints, erasing a real intra-period cash-cycle blowout (slow
    # collections / inventory pile-up) a CA would flag. HIGH CCC is the concerning direction (cash
    # tied up longer), so surface a materially-higher intra-period PEAK (a MIDDLE year more than the
    # steady band above BOTH endpoints), reusing the same noise band the endpoint read uses, so a
    # temporary squeeze is never silently averaged away.
    middle_years = years[1:-1]
    spike = ""
    if middle_years:
        peak_year = max(middle_years, key=lambda y: series[y])
        if series[peak_year] > max(first_val, last_val) + _CCC_STEADY_BAND:
            spike = f", though it spiked to {series[peak_year]:.0f} days in FY{peak_year} in between"
    if abs(delta) < _CCC_STEADY_BAND:
        return (f"Cash conversion cycle has stayed roughly steady near {last_val:.0f} days "
                f"(FY{first_year} to FY{last_year}{spike}; not cross-verified, Screener only).")
    if delta > 0:
        read = ("a lengthening cash cycle can mean slower collections, rising inventory, or "
                "weaker supplier terms; worth checking against sector peers and recent quarters")
    else:
        read = ("a shortening cash cycle usually reflects faster collections or tighter "
                "inventory/payables discipline, a positive sign for cash-flow quality")
    direction = "lengthened" if delta > 0 else "shortened"
    return (f"Cash conversion cycle has {direction} from {first_val:.0f} days (FY{first_year}) "
            f"to {last_val:.0f} days (FY{last_year}){spike}; {read} (not cross-verified, "
            "Screener only).")


# Above this % a year's other-income share of profit before tax reads as worth checking, not
# routine. Calibrated against live Screener data across 3 real, different-sector companies:
# Reliance (12-40%, a large treasury/investments book keeps its normal baseline higher), TCS
# (4-14%), HUL (2-27%, its own recent spike). 25 sits above every company's normal baseline
# while still catching HUL's and Reliance's real outlier years.
_OTHER_INCOME_NOTABLE_PCT = 25.0


def parse_other_income_share_series(html: str) -> dict[int, float]:
    """{fiscal_year: pct} = Other Income / Profit before tax * 100, computed entirely from
    Screener's own P&L table (both numerator and denominator from the SAME source -- not a
    cross-source comparison, so this stays single-source by construction, like the cash
    conversion cycle). Years with non-positive profit before tax are skipped (the ratio is not
    meaningful for a loss-making year).

    WHY (quality of earnings): a large share of profit coming from non-operating "other income"
    (investment gains, interest income, one-off items) rather than the core Operating Profit is
    a classic CA red flag -- profit driven by treasury gains or one-off items is less repeatable
    than profit driven by the actual business. Cross-verifying "Other Income" itself across
    yfinance and Screener was tried and abandoned: live-verified Reliance FY2024, Screener's
    Other Income (~15,792cr) is ~5x yfinance's narrower "Other Non Operating Income Expenses"
    concept (~3,302cr) -- a real methodology difference, not a parsing bug -- so this is computed
    from Screener's own internally-consistent P&L alone, never blended across sources.
    """
    pnl, _, _ = _find_tables(html)
    oi = _annual_series(pnl, "other income")
    pbt = _annual_series(pnl, "profit before tax")
    return {y: (oi[y] / pbt[y] * 100.0) for y in oi if y in pbt and pbt[y] > 0}


def other_income_share_point(series: dict[int, float]) -> str | None:
    """A single, self-disclosing plain-language sentence on the LATEST fiscal year's other-income
    share of profit before tax, or None if unavailable. Explicitly states 'not cross-verified,
    Screener only' inline so the caveat travels with the text wherever it is shown; context for
    the reader, never a buy/sell signal."""
    if not series:
        return None
    year = max(series)
    pct = series[year]
    # WHY (real money, honesty): a negative share (live-verified against TCS's real data) means
    # "other income" was actually a net EXPENSE that year, not a small positive contribution --
    # rendering it as e.g. "-5% ... came from other income" is a confusing, nonsensical read of a
    # genuinely different situation, and rounds to a bare "-0%" for a small negative value.
    if pct < 0:
        return (f"Other income was actually a net expense in FY{year} (reducing profit before "
                f"tax by about {abs(pct):.0f}%) rather than adding to it; the bulk of profit is "
                "driven by the core operating business (not cross-verified, Screener only).")
    # WHY (quality of earnings, honesty): when other income EXCEEDS profit before tax (share >
    # 100%), the reported profit exists only because of it -- without other income the company would
    # report a pre-tax LOSS. A bare ">100% came from other income" is a confusing figure that buries
    # a serious flag (mirrors the negative-share fix at the other end of the ratio), so say it plainly.
    if pct > 100:
        return (f"In FY{year}, non-operating \"other income\" (investment gains, interest income, "
                "or one-off items) was LARGER than the entire profit before tax -- without it the "
                "company would have reported a pre-tax LOSS, so the reported profit depended "
                "entirely on non-operating income. A serious quality-of-earnings concern; check how "
                "repeatable that income is (not cross-verified, Screener only).")
    if pct >= _OTHER_INCOME_NOTABLE_PCT:
        # WHY (CA-level quality of earnings): the latest year alone can't tell a CHRONIC structural
        # reliance on non-operating income (the business habitually needs treasury/one-off gains to
        # post its profit -- a deep quality-of-earnings problem) from a ONE-OFF spike (a single asset
        # sale). The multi-year series is already parsed, so report how OFTEN the share has been this
        # high and interpret the pattern -- a recurring pattern is a far stronger red flag than a
        # single year. Needs >=3 years to call a pattern; below that the latest-year read stands.
        pattern = ""
        if len(series) >= 3:
            elevated = sum(1 for v in series.values() if v >= _OTHER_INCOME_NOTABLE_PCT)
            if elevated * 2 >= len(series):
                pattern = (f"; and it has topped {_OTHER_INCOME_NOTABLE_PCT:.0f}% of profit before "
                           f"tax in {elevated} of the last {len(series)} years -- a recurring, "
                           "structural reliance on non-operating income rather than a one-off")
            else:
                pattern = (f"; though it has topped {_OTHER_INCOME_NOTABLE_PCT:.0f}% of profit "
                           f"before tax in only {elevated} of the last {len(series)} years, so this "
                           "reads more like a one-off than a recurring pattern")
        return (f"{pct:.0f}% of FY{year}'s profit before tax came from non-operating \"other "
                "income\" (investment gains, interest income, or one-off items) rather than the "
                f"core business{pattern} -- worth checking how repeatable that income is (not "
                "cross-verified, Screener only).")
    return (f"{pct:.0f}% of FY{year}'s profit before tax came from non-operating \"other "
            "income\"; the bulk of profit is driven by the core operating business (not "
            "cross-verified, Screener only).")


class ScreenerFigureSource(FigureSource):
    source_id = "screener"

    def __init__(self, fetcher: Callable[[str], str | None] | None = None):
        self._fetcher = fetcher or self._http_fetch
        # WHY: memoize the page per symbol. build_company_report calls figures() AND
        # figures_by_year() (and the series gather calls it again) for the same symbol, so without
        # this each stock hit Screener ~3x -> a 32-stock batch made ~96 requests and tripped
        # Cloudflare's rate limit. One fetch per symbol keeps the batch to ~32.
        self._cache: dict[str, str | None] = {}

    def _fetch_cached(self, symbol: str) -> str | None:
        key = symbol.strip().upper()
        if key not in self._cache:
            self._cache[key] = self._fetcher(key)
        return self._cache[key]

    @staticmethod
    def _http_fetch(symbol: str) -> str | None:
        # WHY: browser-like headers + retry-with-backoff. Screener sits behind Cloudflare, which
        # rate-limits bursts from datacenter IPs (Streamlit Cloud), starving cross-verification to
        # a single source. Backoff gives a transient rate-limit time to clear; best-effort, not
        # guaranteed past a hard JS challenge.
        import time

        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                      "image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.screener.in/",
            "Connection": "keep-alive",
        }
        session = requests.Session()
        for attempt in range(3):
            for path in (f"https://www.screener.in/company/{symbol}/consolidated/",
                         f"https://www.screener.in/company/{symbol}/"):
                try:
                    resp = session.get(path, headers=headers, timeout=30)
                    if resp.status_code == 200 and resp.text:
                        return resp.text
                except Exception:
                    continue
            if attempt < 2:
                time.sleep(2 * (attempt + 1))   # 2s, 4s backoff before retrying
        return None

    def figures(self, symbol: str) -> dict[str, float | None]:
        html = self._fetch_cached(symbol)
        if not html:
            return {name: None for name in FRAMEWORK_FIGURES}
        return parse_screener_figures(html)

    def figures_by_year(self, symbol: str) -> dict[str, dict[int, float]]:
        html = self._fetch_cached(symbol)
        if not html:
            return {}
        return parse_screener_series(html)

    def promoter_holding_trend(self, symbol: str) -> str | None:
        """A single-source (Screener only), self-disclosing promoter-holding trend sentence for
        `symbol`, or None if unavailable. Not part of the FigureSource interface (it is not a
        cross-verifiable numeric figure); callers opt in explicitly."""
        html = self._fetch_cached(symbol)
        if not html:
            return None
        return promoter_holding_trend_point(parse_promoter_holding_series(html))

    def cash_conversion_cycle_trend(self, symbol: str) -> str | None:
        """A single-source (Screener only), self-disclosing cash-conversion-cycle trend sentence
        for `symbol`, or None if unavailable. Not part of the FigureSource interface (it is not a
        cross-verifiable numeric figure); callers opt in explicitly."""
        html = self._fetch_cached(symbol)
        if not html:
            return None
        return cash_conversion_cycle_trend_point(parse_cash_conversion_cycle_series(html))

    def promoter_pledge(self, symbol: str) -> str | None:
        """A single-source (Screener only), self-disclosing promoter-pledge sentence for `symbol`,
        or None if Screener does not flag a material pledge. Not part of the FigureSource interface
        (it is not a cross-verifiable numeric figure); callers opt in explicitly."""
        html = self._fetch_cached(symbol)
        if not html:
            return None
        return promoter_pledge_point(parse_promoter_pledge(html))

    def other_income_share(self, symbol: str) -> str | None:
        """A single-source (Screener only), self-disclosing other-income-share-of-profit
        sentence for `symbol`, or None if unavailable. Not part of the FigureSource interface (it
        is not a cross-verifiable numeric figure); callers opt in explicitly."""
        html = self._fetch_cached(symbol)
        if not html:
            return None
        return other_income_share_point(parse_other_income_share_series(html))
