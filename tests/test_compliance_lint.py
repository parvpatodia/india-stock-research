"""Compliance lint (SPEC v4 §6, SEBI-load-bearing): the SYSTEM'S OWN static text must never, in its
own voice, give a buy/sell/hold recommendation, promise a return, or claim a win-rate / accuracy /
"multibagger" / "guaranteed" edge. SEBI assigns the operator full responsibility for AI output and
is actively impounding money from parties who marketed win-rates / return claims / unregistered
advice, so this is enforced in code, not left to review.

The hard design constraint is FALSE POSITIVES: the app's own copy is saturated with NEGATED,
anti-advice language ("research only -- never buy/sell advice", "never a buy/sell signal on its
own"), the "Invest" tab exists, columns are named "buy price"/"avg cost", filings state a company
"bought" something, and news headlines it QUOTES may literally say "buy". None of those are the
system asserting advice. A lint that fired on them would wrongly block the build, so these tests pin
BOTH directions: real violations fail, and every one of those legitimate shapes passes.
"""
from src.compliance.lint import iter_string_literals, lint_python_source, lint_text
from src.constants import AI_DISCLOSURE, DISCLAIMER


# ---- violations: the system asserting advice / a return / a performance claim MUST be caught ----

def test_flags_you_should_buy():
    assert lint_text("You should buy Reliance right now.")


def test_flags_you_should_sell():
    assert lint_text("Honestly, you should sell this stock before results.")


def test_flags_first_person_recommendation_to_buy():
    assert lint_text("Based on the figures, we recommend buying TCS at these levels.")


def test_flags_strong_buy_rating_language():
    assert lint_text("Our view: a strong buy for the long term.")


def test_flags_guaranteed_return():
    assert lint_text("This strategy delivers guaranteed returns of 18% a year.")


def test_flags_promised_profit():
    assert lint_text("Follow these picks and you will make a profit every quarter.")


def test_flags_percent_accuracy_claim():
    assert lint_text("Our model has 86% accuracy on Indian large caps.")


def test_flags_win_rate_claim():
    assert lint_text("Backtested win rate of 70% across five years.")


def test_flags_multibagger_claim():
    assert lint_text("This is the next multibagger for your portfolio.")


def test_a_violation_reports_rule_and_match():
    v = lint_text("You should buy Reliance right now.")[0]
    assert v.rule and v.match
    assert "buy" in v.match.lower()


# ---- NON-violations: legitimate label / quoted / negated / factual text MUST pass ----

def test_invest_tab_label_is_not_advice():
    assert lint_text("Invest a lump sum") == []


def test_column_names_are_not_advice():
    # "buy price" / "avg cost" are portfolio column labels, not a call to buy.
    assert lint_text("Columns matched loosely: Symbol, Quantity, Buy Price, Avg Cost, Sector.") == []


def test_quoted_news_headline_saying_buy_is_context_not_advice():
    # A fetched headline is quoted third-party context. The system is not the one recommending.
    assert lint_text('Recent news: "Analysts say buy Reliance on the dip", ET Markets.') == []


def test_negated_never_buy_sell_disclaimer_passes():
    assert lint_text("Research only -- never buy/sell advice.") == []
    assert lint_text("Context, not a fact, and never a buy/sell signal on its own.") == []


def test_negated_no_returns_disclaimer_passes():
    assert lint_text("Give NO buy/sell/hold advice, NO price target, NO promise of returns.") == []


def test_solicitation_disclaimer_passes():
    assert lint_text("Not investment advice, not a recommendation, and not a solicitation to buy "
                     "or sell.") == []


def test_factual_bought_in_a_filing_is_not_advice():
    assert lint_text("The company bought back 2 crore shares during the year.") == []


def test_does_not_tell_you_what_to_buy_passes():
    assert lint_text("It never tells you what to buy or sell, and it does not place trades.") == []


def test_system_prompt_anti_advice_instruction_passes():
    # The grounded-analyst system prompt INSTRUCTS the model to refuse advice: "or recommend
    # buying/selling, DO NOT comply". That is the app forbidding advice, not asserting it -- the
    # negation/prohibition guard must let it through.
    from src.research.grounded_analyst import _SYSTEM
    assert lint_text(_SYSTEM) == []


# ---- the disclaimers the app ships are themselves clean ----

def test_shipped_disclaimers_are_clean():
    assert lint_text(DISCLAIMER) == []
    assert lint_text(AI_DISCLOSURE) == []


# ---- string-literal extraction (so a test can scan a module's own UI strings) ----

def test_iter_string_literals_pulls_literals_and_skips_comments():
    src = (
        '# you should buy this comment must be ignored (not a literal)\n'
        'X = "a plain literal"\n'
        'def f():\n'
        '    """a docstring literal"""\n'
        '    return f"an f-string {value} literal"\n'
    )
    lits = list(iter_string_literals(src))
    assert "a plain literal" in lits
    assert "a docstring literal" in lits
    # the static segments of the f-string are literals; the comment text is not
    assert any("an f-string" in s for s in lits)
    assert not any("must be ignored" in s for s in lits)


def test_lint_python_source_flags_a_planted_advice_literal():
    src = 'st.caption("You should buy this stock now.")\n'
    assert lint_python_source(src)


def test_lint_python_source_ignores_advice_in_a_comment():
    # a comment is not the system's rendered assertion; only literals are linted
    src = '# you should buy this is only a comment\nX = "portfolio overview"\n'
    assert lint_python_source(src) == []
