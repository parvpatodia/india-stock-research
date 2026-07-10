from src.formatting import format_rupees, format_rupees_crore_lakh, indian_group


def test_indian_group_places_separators_the_indian_way():
    # last 3 digits, then groups of 2 (e.g. 1,05,64,995) -- the convention every Indian reader
    # expects; Western thousands grouping reads wrong to them.
    assert indian_group("5") == "5"
    assert indian_group("50") == "50"
    assert indian_group("500") == "500"
    assert indian_group("5000") == "5,000"
    assert indian_group("50000") == "50,000"
    assert indian_group("500000") == "5,00,000"
    assert indian_group("5000000") == "50,00,000"
    assert indian_group("10564995") == "1,05,64,995"


def test_format_rupees_exact_amount_indian_grouped():
    # WHY (real money, UI honesty): the parents' OWN portfolio/allocation amounts must read in the
    # Indian convention, consistent with the crore/lakh research figures -- a ₹5 lakh holding is
    # "₹5,00,000", never Western "₹500,000". Exact whole rupees (no crore/lakh abbreviation) so a
    # holding/P&L amount keeps its precision.
    assert format_rupees(500000) == "₹5,00,000"
    assert format_rupees(452300) == "₹4,52,300"
    assert format_rupees(20000000) == "₹2,00,00,000"
    assert format_rupees(9000) == "₹9,000"
    assert format_rupees(33333.7) == "₹33,334"          # rounded to whole rupees
    assert format_rupees(-50000) == "-₹50,000"          # a loss keeps a clean leading sign
    assert format_rupees(0) == "₹0"
    assert format_rupees(None) == "n/a"                 # missing price/value, not a fabricated 0


def test_format_rupees_crore_lakh_matches_the_figure_convention():
    # the abbreviated crore/lakh form used for large company financials (see format_figure_value)
    assert format_rupees_crore_lakh(790000000000.0) == "₹79,000 crore"
    assert format_rupees_crore_lakh(150000.0) == "₹1.5 lakh"
    assert format_rupees_crore_lakh(9000.0) == "₹9,000"
    assert format_rupees_crore_lakh(-500000000.0) == "-₹50 crore"
