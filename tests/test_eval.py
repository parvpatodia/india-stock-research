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


def test_store_roundtrip_and_replay(tmp_path):
    store = EvalStore(tmp_path / "cases.jsonl")
    report = _report_with_net_profit([(80000, "a"), (80100, "b")])
    store.add(ground_truth_from_report(report, "net_profit", 80050, note="checked", reviewer="dad"))
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].company == "ACME" and loaded[0].reviewer == "dad"
    assert evaluate(loaded).results[0].outcome == Outcome.MATCH
