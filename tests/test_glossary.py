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
                "Verified fact", "Opinion", "Unverified"}
    assert required <= set(GLOSSARY)
