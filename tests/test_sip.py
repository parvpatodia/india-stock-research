from src.sip import DEFAULT_INFLATION_PCT, real_value, sip_future_value


def test_real_value_discounts_a_future_corpus_to_todays_money():
    # WHY (real money, honesty): a multi-decade SIP projects a large NOMINAL corpus, but inflation
    # erodes what it buys. ₹1 crore in 30 years at ~6%/yr inflation is worth ~₹17 lakh in today's
    # money (1e7 / 1.06^30 = ~1.74e6) -- a non-expert must not read the bare nominal as its real worth.
    r = real_value(10_000_000, years=30, inflation_pct=6.0)
    assert 1_600_000 < r < 1_850_000
    assert r < 10_000_000                       # always less than nominal for positive inflation


def test_real_value_is_nominal_at_zero_inflation_or_zero_years():
    assert real_value(1_000_000, years=20, inflation_pct=0.0) == 1_000_000
    assert real_value(500_000, years=0, inflation_pct=6.0) == 500_000


def test_default_inflation_is_a_disclosed_constant():
    assert DEFAULT_INFLATION_PCT == 6.0        # India's rough long-run CPI average (an assumption)


def test_zero_return_projects_exactly_invested():
    p = sip_future_value(monthly=10000, annual_return_pct=0, years=10)
    assert p.invested == 1_200_000
    assert p.projected_value == 1_200_000
    assert p.gain == 0


def test_invested_is_monthly_times_months():
    p = sip_future_value(monthly=5000, annual_return_pct=12, years=3)
    assert p.invested == 5000 * 36


def test_positive_return_grows_above_invested():
    p = sip_future_value(monthly=10000, annual_return_pct=12, years=10)
    assert p.projected_value > p.invested
    assert p.gain > 0
    # annuity-due 12%/10y on 10k/month is ~Rs 23.2 lakh; sanity-bound it
    assert 2_200_000 < p.projected_value < 2_500_000


def test_one_year_zero_rate():
    p = sip_future_value(monthly=1000, annual_return_pct=0, years=1)
    assert p.invested == 12000 and p.projected_value == 12000
