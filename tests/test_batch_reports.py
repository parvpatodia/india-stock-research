from scripts.batch_reports import _format_verdict_lines
from src.research.report import Confidence, Leaning, QualityTier, ValuationTier, Verdict


def test_format_verdict_lines_includes_reasons_and_sector_caveats():
    # WHY (adversarial-review regression): this script's ENTIRE disclosure output for a symbol
    # used to be `for reason in v.reasons: log(...)`. When sector caveats (bank/NBFC caveat,
    # real-estate leverage caveat) were moved out of `reasons` into the new `sector_caveats`
    # field, this script silently stopped printing them at all -- the two OTHER rendering
    # surfaces (the Research tab, the PDF export) were updated to also show sector_caveats, but
    # this one was missed. A caveat like "check the filing for GNPA/CRAR" disappearing from the
    # one file an operator actually reads to sanity-check a batch run is a real regression.
    v = Verdict(ValuationTier.FAIR, QualityTier.STRONG, Leaning.NEUTRAL, Confidence.MEDIUM,
               reasons=("debt/equity 0.60 reads moderate.",),
               sector_caveats=("check the filing for GNPA/CRAR.",))
    lines = _format_verdict_lines(v)
    assert any("debt/equity 0.60 reads moderate." in x for x in lines)
    assert any("check the filing for GNPA/CRAR." in x for x in lines)


def test_format_verdict_lines_empty_when_nothing_to_disclose():
    v = Verdict(ValuationTier.UNKNOWN, QualityTier.UNKNOWN, Leaning.UNKNOWN, Confidence.LOW)
    assert _format_verdict_lines(v) == []
