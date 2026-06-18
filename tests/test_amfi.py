from src.data.amfi_provider import AMFIProvider, parse_navall

SAMPLE = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Aditya Birla Sun Life Mutual Fund

Open Ended Schemes ( Equity Scheme )
120503;INF209KB17W8;INF209KB18W6;Aditya Birla Sun Life Banking Fund - Direct - Growth;100.5;18-Jun-2026
119551;INF209K01XX1;;Aditya Birla Sun Life Frontline Equity Fund - Growth;350.25;18-Jun-2026
999999;;;Some Scheme With NA NAV;N.A.;18-Jun-2026
"""


def test_parse_navall_skips_headers_and_na():
    schemes = parse_navall(SAMPLE)
    codes = {s.scheme_code for s in schemes}
    assert codes == {"120503", "119551"}      # column header + N.A. row excluded
    s = next(x for x in schemes if x.scheme_code == "120503")
    assert s.nav == 100.5 and s.date == "18-Jun-2026"
    assert s.isin == "INF209KB17W8"


def test_amfi_provider_with_injected_fetcher():
    provider = AMFIProvider(fetcher=lambda: SAMPLE)
    assert provider.load() == 2
    assert provider.get_by_code("119551").name.startswith("Aditya Birla Sun Life Frontline")
    assert len(provider.search("frontline")) == 1
    assert provider.search("") == []
