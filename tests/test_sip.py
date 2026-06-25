from src.sip import sip_future_value


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
