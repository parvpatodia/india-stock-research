from src.eval.cases import EvalStore
from src.eval.harness import Outcome, evaluate, ground_truth_from_report
from src.pipeline import build_company_report
from src.research.verification import SourcedValue


def _report_with_net_profit(values):
    figs = {"net_profit": [SourcedValue(v, s) for (v, s) in values]}
    return build_company_report("ACME", figs)


def test_match_when_trusted_value_equals_truth():
    report = _report_with_net_profit([(80000, "a"), (80100, "b")])  # verified ~80050
    gt = ground_truth_from_report(report, "net_profit", 80050)
    res = evaluate([gt])
    assert res.results[0].outcome == Outcome.MATCH
    assert res.trusted_wrong == 0 and res.accuracy == 1.0


def test_trusted_wrong_when_system_trusts_a_value_expert_says_is_false():
    report = _report_with_net_profit([(80000, "a"), (80100, "b")])  # system trusts ~80050
    gt = ground_truth_from_report(report, "net_profit", 95000)       # expert: truth is 95000
    res = evaluate([gt])
    assert res.results[0].outcome == Outcome.TRUSTED_WRONG
    assert res.trusted_wrong == 1 and res.accuracy == 0.0


def test_withheld_is_not_a_mistake():
    report = _report_with_net_profit([(80000, "a"), (95000, "b")])  # conflict -> withheld
    gt = ground_truth_from_report(report, "net_profit", 80000)
    res = evaluate([gt])
    assert res.results[0].outcome == Outcome.WITHHELD
    assert res.trusted_wrong == 0


def test_trusted_accuracy_and_withheld_do_not_penalize_safe_withholds():
    # WHY (real money, honest metric): the parent-facing "learning loop" line must not read a SAFE
    # withhold (a figure the system correctly refused to trust because the sources conflicted) as if
    # it were an inaccuracy. accuracy=matches/total folds every safe withhold into the denominator,
    # so a cautious run of mostly-correct WITHHOLDS shows a low "accuracy" that misrepresents the
    # system to a non-expert. trusted_accuracy is precision over the decisions it actually MADE: of
    # the figures it TRUSTED, how many were right -- a withhold is not counted against it, and it is
    # a perfect 1.0 exactly while trusted-but-wrong stays 0 (the real goal).
    matched = _report_with_net_profit([(80000, "a"), (80000, "b")])     # agree -> trusted, matches
    withheld = _report_with_net_profit([(80000, "a"), (95000, "b")])    # conflict -> safely withheld
    res = evaluate([ground_truth_from_report(matched, "net_profit", 80000),
                    ground_truth_from_report(withheld, "net_profit", 80000)])
    assert res.matches == 1 and res.withheld == 1 and res.trusted_wrong == 0
    assert res.accuracy == 0.5              # coverage unchanged: 1 of 2 figures were verifiable
    assert res.trusted_accuracy == 1.0      # of what it TRUSTED, 100% right; a safe withhold is no miss


def test_trusted_accuracy_is_one_when_nothing_was_trusted():
    # A run that withheld everything asserted nothing false -> vacuously perfect trusted-accuracy,
    # never a division error. The withheld count is what tells the reader coverage was zero.
    withheld = _report_with_net_profit([(80000, "a"), (95000, "b")])    # conflict -> withheld
    res = evaluate([ground_truth_from_report(withheld, "net_profit", 80000)])
    assert res.matches == 0 and res.withheld == 1 and res.trusted_accuracy == 1.0


def test_eval_store_skips_a_corrupt_line(tmp_path):
    # WHY (resilience, real money): the store is append-only JSONL; a crash mid-write (writes are not
    # atomic) or a manual edit can leave a corrupt/partial line. load() is called on every Research-
    # tab render (the "learning loop" caption), so it must SKIP a bad line and return the valid cases,
    # never crash the tab a parent is viewing -- matching the app's skip-bad-lines pattern (AMFI).
    p = tmp_path / "cases.jsonl"
    store = EvalStore(p)
    report = _report_with_net_profit([(80000, "a"), (80100, "b")])
    store.add(ground_truth_from_report(report, "net_profit", 80050))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"company": "X", "figure": \n')          # corrupt/partial JSON line
    store.add(ground_truth_from_report(report, "net_profit", 80050, note="second"))
    loaded = store.load()
    assert len(loaded) == 2                                # both valid cases; the corrupt line skipped
    assert all(c.figure == "net_profit" for c in loaded)


def test_store_roundtrip_and_replay(tmp_path):
    store = EvalStore(tmp_path / "cases.jsonl")
    report = _report_with_net_profit([(80000, "a"), (80100, "b")])
    store.add(ground_truth_from_report(report, "net_profit", 80050, note="checked", reviewer="dad"))
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].company == "ACME" and loaded[0].reviewer == "dad"
    assert evaluate(loaded).results[0].outcome == Outcome.MATCH
