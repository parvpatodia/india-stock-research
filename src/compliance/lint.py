"""Self-voice compliance lint (SPEC v4 §6, SEBI-load-bearing).

FAIL if the SYSTEM'S OWN static text asserts, in its own voice, any of:
  - a buy / sell / hold RECOMMENDATION ("you should buy", "we recommend buying", "strong buy"),
  - a promised / guaranteed RETURN ("guaranteed returns", "you will make a profit"),
  - a performance CLAIM ("86% accuracy", "70% win rate", "multibagger", "sure shot").

SEBI regulates buy/sell advice, assigns the operator FULL responsibility for AI output, and is
actively impounding money from parties who marketed win-rates / return claims / unregistered advice.
So this is enforced in CODE (a test scans app.py's own literals + the system prompt + the
disclaimers), not left to a human reading a diff.

THE HARD PART IS FALSE POSITIVES. The scope is the system's OWN ASSERTIONS, never quoted/attributed
source text. Two defenses make that safe:

1. SCOPE. The lint only ever sees the app's STATIC STRING LITERALS (extracted from source) and its
   fixed prompt/disclaimer constants. It never sees runtime data -- so a fetched news headline that
   literally says "buy Reliance", or an LLM answer, is out of scope BY CONSTRUCTION. Comments are
   not literals, so they are not scanned either.

2. TIGHT PATTERNS + A NEGATION GUARD. The patterns match genuine ADVICE phrasing, not the bare
   presence of "buy"/"sell". So "Invest" (a tab label), "Buy Price"/"Avg Cost" (column names), "the
   company bought back shares" (a filing fact), and a bare quoted "buy Reliance" do NOT match. On top
   of that, the app's copy is saturated with NEGATED, anti-advice clauses ("never buy/sell advice",
   "NO promise of returns", the system prompt's "or recommend buying/selling, DO NOT comply"); a
   clause carrying a prohibition cue (never/not/no/ignore/do not/...) is exempted, so a disclaimer
   that mentions a trade word to FORBID it is not read as asserting it.

The bias is deliberately conservative toward false negatives: a wrongly-flagged legitimate label
would BLOCK the build, which is worse here than missing an exotic advice phrasing a human review
would still catch. The patterns cover the four load-bearing shapes SEBI enforces on.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# A window that cannot cross a clause boundary, used inside multi-token rules so a match stays local.
_W = r"[^.?!;\n]{0,20}"


@dataclass(frozen=True)
class Violation:
    rule: str        # which forbidden category ("buy/sell recommendation" / "guaranteed return" / ...)
    match: str       # the exact offending substring
    context: str     # the clause it was found in (for the failure report)


# Each rule: (category, compiled pattern). Patterns encode the ASSERTION, so bare "buy"/"sell" and
# label/quoted/factual uses do not match. See module docstring for the false-positive design.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # --- buy / sell / hold recommendation in the app's own voice ---
    ("buy/sell recommendation",
     re.compile(r"\byou\s+(?:should|shall|must|need\s+to|ought\s+to|have\s+to|had\s+better|"
                r"are\s+advised\s+to|may\s+want\s+to|might\s+want\s+to)\s+"
                r"(?:buy|sell|hold|invest|purchase|book|exit|dump|accumulate|offload|short)\b",
                re.IGNORECASE)),
    ("buy/sell recommendation",
     re.compile(r"\b(?:recommend|suggest|advise)\w*\s+(?:that\s+you\s+|you\s+)?"
                r"(?:buy|sell|hold|buying|selling|holding|purchas\w+|"
                r"to\s+(?:buy|sell|hold|invest))\b", re.IGNORECASE)),
    ("buy/sell recommendation",
     re.compile(r"\b(?:strong|outright)\s+(?:buy|sell)\b", re.IGNORECASE)),
    ("buy/sell recommendation",
     re.compile(r"\bour\s+(?:buy|sell|hold)\s+(?:call|rating|recommendation)\b", re.IGNORECASE)),
    # --- promised / guaranteed return ---
    ("guaranteed return",
     re.compile(r"\b(?:guarantee[sd]?|assure[sd]?|promise[sd]?|ensure[sd]?)\b" + _W +
                r"\b(?:returns?|profits?|gains?|income)\b", re.IGNORECASE)),
    ("guaranteed return",
     re.compile(r"\b(?:returns?|profits?|gains?)\b" + _W +
                r"\b(?:guaranteed|assured|promised|locked\s+in)\b", re.IGNORECASE)),
    ("guaranteed return",
     re.compile(r"\byou(?:'ll|\s+will|\s+are\s+going\s+to)\s+"
                r"(?:earn|make|get|double|triple|gain|profit|be\s+rich|become\s+rich)\b",
                re.IGNORECASE)),
    # --- accuracy / win-rate / multibagger / sure-shot performance claim ---
    ("performance claim", re.compile(r"\bmultibaggers?\b", re.IGNORECASE)),
    ("performance claim", re.compile(r"\bwin[\s-]?rate\b", re.IGNORECASE)),
    ("performance claim", re.compile(r"\bsure[\s-]?shot\b", re.IGNORECASE)),
    ("performance claim",
     re.compile(r"\b\d+(?:\.\d+)?\s*%\s*"
                r"(?:accuracy|accurate|win|success|hit|guaranteed|assured|returns?|profits?)\b",
                re.IGNORECASE)),
    ("performance claim",
     re.compile(r"\b(?:accuracy|success|hit)\s+rate\s+of\s+\d", re.IGNORECASE)),
)

# Prohibition / negation cues. A clause carrying any of these is exempted: the trade word is being
# FORBIDDEN or disclaimed, not asserted (e.g. "never buy/sell advice", "NO promise of returns",
# "or recommend buying/selling, DO NOT comply"). Word-boundaried so "know" != "no", "note" != "not".
_NEGATION = re.compile(
    r"\b(?:never|not|no|nor|neither|none|without|cannot|refuses?|ignores?|avoids?|"
    r"do\s+not|does\s+not|did\s+not|don't|won't|can't|isn't|aren't|shouldn't|wouldn't|"
    r"rather\s+than|instead\s+of)\b", re.IGNORECASE)

# Split into clauses so a prohibition cue only exempts its OWN clause, and a multi-token rule cannot
# span a sentence boundary. Sentence/clause terminators: . ? ! ; and newlines.
_CLAUSE_SPLIT = re.compile(r"[.?!;\n]+")


def _has_negation(clause: str) -> bool:
    return _NEGATION.search(clause) is not None


def lint_text(text: str) -> list[Violation]:
    """Return every self-voice compliance violation in one block of the system's own text.

    A clause carrying a prohibition/negation cue is skipped (it disclaims, it does not assert). Empty
    / non-string input yields no violations.
    """
    if not text or not isinstance(text, str):
        return []
    out: list[Violation] = []
    for clause in _CLAUSE_SPLIT.split(text):
        if not clause.strip() or _has_negation(clause):
            continue
        for rule, pattern in _RULES:
            for m in pattern.finditer(clause):
                out.append(Violation(rule=rule, match=m.group(0).strip(), context=clause.strip()))
    return out


def lint_texts(texts: Iterable[str]) -> list[Violation]:
    """Lint many blocks (e.g. every UI literal in a module); returns all violations, in order."""
    out: list[Violation] = []
    for t in texts:
        out.extend(lint_text(t))
    return out


def iter_string_literals(source: str) -> Iterator[str]:
    """Yield every string-literal constant in a Python source string, via `ast` (so a test can lint a
    module's OWN rendered UI copy). Covers plain literals, docstrings, and the static segments of
    f-strings. Comments are NOT literals, so they are excluded -- only text the system actually
    renders/asserts is scanned. A syntax error yields nothing (the byte-compile gate catches that)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def lint_python_source(source: str) -> list[Violation]:
    """Lint every string literal in a Python source string. The self-voice check for a whole module's
    UI copy: `lint_python_source(open("app.py").read())`."""
    return lint_texts(iter_string_literals(source))
