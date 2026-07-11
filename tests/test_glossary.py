from src.glossary import GLOSSARY, explain


def test_explain_known_and_unknown():
    assert explain("NAV").startswith("Net Asset Value")
    assert explain("not a term") is None


def test_all_definitions_are_nonempty_plain_text():
    for term, definition in GLOSSARY.items():
        assert isinstance(definition, str) and len(definition.strip()) > 10
        assert "\n" not in definition  # one-liners, readable as a tooltip


def test_covers_terms_the_app_relies_on():
    # WHY: the app passes these exact keys to help= tooltips; a rename must update both.
    required = {"P&L", "Concentration (HHI)", "Beta", "NAV", "SIP",
                "Verified fact", "Opinion", "Unverified", "Confidence"}
    assert required <= set(GLOSSARY)


def test_confidence_glossary_clarifies_data_completeness_not_likelihood_of_gains():
    # WHY (real money, comprehension): "Confidence" is the one verdict-card term whose meaning is NOT
    # conveyed by the stance headline or plain summary, and a non-expert can read "Confidence: high"
    # as "high chance this makes money" (OUTCOME) when it means DATA completeness -- how much of the
    # data cross-verified across independent sources. The tooltip must say it's about the data, and
    # explicitly that it is NOT the likelihood of returns, or it amplifies a wrong real-money read.
    d = explain("Confidence")
    assert d is not None
    assert "cross" in d.lower() or "source" in d.lower()               # it is about the DATA/sources
    assert "not" in d.lower()                                          # ...and an explicit disclaimer
    assert any(w in d.lower() for w in ("go up", "return", "likel", "gain"))  # not the OUTCOME
