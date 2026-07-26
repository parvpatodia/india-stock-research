"""The run_eval.py CI-style gate: red-team resistance + numeric-exact-match, offline, with a
clear pass/fail and a non-zero exit when either regresses. Tests the gate functions directly so
the check does not depend on the runtime ground-truth store file."""
import importlib.util
from pathlib import Path

from src.eval.numeric_match import GoldenFigure, GoldenFixture
from src.eval.redteam import RedTeamOutcome, RedTeamReport

_RUN_EVAL = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", _RUN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_redteam_gate_passes_offline():
    mod = _load_run_eval()
    ok, summary = mod.redteam_gate()
    assert ok, summary
    assert "red-team" in summary.lower()


def test_redteam_gate_fails_on_a_breach():
    mod = _load_run_eval()
    breached = RedTeamReport((
        RedTeamOutcome("x", "unit_trap", False, "leaked a wrong number"),
    ))
    ok, summary = mod.redteam_gate(report=breached)
    assert not ok
    assert "unit_trap" in summary or "1" in summary


def test_numeric_gate_passes_on_the_golden_set():
    mod = _load_run_eval()
    ok, summary = mod.numeric_gate()
    assert ok, summary
    assert "numeric" in summary.lower()


def test_numeric_gate_fails_when_a_figure_mismatches():
    mod = _load_run_eval()
    bad = GoldenFixture(
        name="bad", text="Net profit for the year was Rs 73,670 crore.", company="ACME",
        figures=(GoldenFigure(label="np", raw_number="73,670",
                              expected_absolute=73670.0 * 1e6),),  # million, not crore -> mismatch
    )
    ok, summary = mod.numeric_gate(fixtures=(bad,))
    assert not ok


def test_main_returns_zero_when_all_gates_pass():
    mod = _load_run_eval()
    # empty/absent ground-truth store -> that gate is vacuously green; red-team + numeric pass.
    assert mod.main() == 0
