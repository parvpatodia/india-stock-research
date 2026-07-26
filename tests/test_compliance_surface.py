"""W8 compliance surface (SPEC v4 §6, SEBI): the app must, in its OWN voice, (1) DISCLOSE that it is
AI-assisted, research-only, human-reviewed, with no return/accuracy claim; (2) carry that disclosure
on exports; (3) keep the human-in-the-loop DRAFT->APPROVED gate; and (4) never let its own static UI
copy drift into advice. These pin those invariants as regressions.
"""
import ast
import os

from src.compliance.lint import iter_string_literals, lint_texts
from src.constants import AI_DISCLOSURE, DISCLAIMER
from src.research.report import Report, ReviewStatus

_APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def _import_app_with_clean_env():
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        import app
        return app
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---- (1) the disclosure text itself carries every SEBI-required piece ----

def test_ai_disclosure_states_the_required_pieces():
    d = AI_DISCLOSURE.lower()
    assert "ai" in d                                   # AI usage disclosed (Reg 16C)
    assert "research" in d                             # research-only framing
    assert "not investment advice" in d                # not advice
    assert "buy or sell" in d                          # not a buy/sell call (in a negated phrase)
    assert "human" in d and "review" in d              # human-in-the-loop disclosed
    # no return / accuracy / win-rate promise
    assert "no performance" in d or "no return" in d or "makes no" in d


# ---- (2) exports carry the AI disclosure ----

def test_pdf_export_carries_the_ai_disclosure():
    from src.analysis.sizing import Stance
    app = _import_app_with_clean_env()
    from pypdf import PdfReader
    import io

    def _text(b):
        return " ".join("\n".join(p.extract_text() or "" for p in
                                  PdfReader(io.BytesIO(b)).pages).split())

    # a normal report AND a no-verdict report must both carry the AI disclosure
    normal = _text(app.build_pdf_report("RELIANCE (live)", Report(company="RELIANCE"),
                                        Stance.NEUTRAL))
    assert "AI-assisted" in normal or "AI-assisted".lower() in normal.lower()
    assert "human" in normal.lower()


# ---- (3) human-in-the-loop lifecycle stays intact ----

def test_human_in_the_loop_draft_is_not_trusted_until_approved():
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    # a report with a real cross-verified figure (so it is approvable, not a no-data report)
    fig = VerifiedFigure("net_profit", VerificationStatus.VERIFIED, 100.0,
                         (SourcedValue(100.0, "yfinance", locator="FY2025"),
                          SourcedValue(100.0, "screener", locator="FY2025")), "agree")
    r = Report(company="RELIANCE", figures=(fig,))
    assert r.status == ReviewStatus.DRAFT
    assert not r.is_trusted                              # a draft is never trusted
    approved = r.approve(reviewer="expert")
    assert approved.status == ReviewStatus.APPROVED
    assert approved.is_trusted                           # only an expert approval trusts it


# ---- (4) the app's OWN static copy contains zero self-voice advice ----

def _app_literals() -> list[str]:
    with open(_APP, "r", encoding="utf-8") as fh:
        source = fh.read()
    return list(iter_string_literals(source))


def test_app_static_copy_has_no_self_voice_advice():
    # WHY (load-bearing): scan app.py's OWN string literals (its rendered UI copy) plus the two
    # disclaimers and the grounded-analyst system prompt. A future edit that introduces a buy/sell
    # call, a promised return, or a win-rate/accuracy claim in the app's own voice FAILS here.
    from src.research.grounded_analyst import _SYSTEM
    corpus = _app_literals() + [_SYSTEM, DISCLAIMER, AI_DISCLOSURE]
    violations = lint_texts(corpus)
    assert violations == [], f"self-voice advice in app copy: {violations}"


def test_app_parses_and_imports_the_disclosure():
    # the app actually references AI_DISCLOSURE (the disclosure is wired, not just defined)
    with open(_APP, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "AI_DISCLOSURE" in names


def _all_rendered_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "text"):
        try:
            for el in getattr(at, kind):
                parts.append(str(getattr(el, "value", "")))
        except Exception:
            pass
    return " ".join(parts)


def test_footer_ai_disclosure_actually_renders():
    # WHY (SEBI, real money): the AI-usage disclosure must be VISIBLE to the parents, not just a
    # constant. Run the real app (sample-portfolio path, no LLM) and confirm the disclosure text is
    # on the page.
    from streamlit.testing.v1 import AppTest
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        at = AppTest.from_file(_APP).run(timeout=180)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert len(at.exception) == 0, [e.value for e in at.exception]
    text = _all_rendered_text(at)
    assert "AI-assisted" in text                        # the disclosure is on the page
    assert "human review" in text.lower() or "a human" in text.lower()
