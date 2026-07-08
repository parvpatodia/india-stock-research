import re

from src.llm.client import LLMClient
from src.research.annual_report_reader import read_filing


class FakeClient(LLMClient):
    """Cites the first annual_report chunk actually present in the prompt, so the grounding path
    (retrieve -> cite -> enforce) is exercised end-to-end without a network/provider."""

    def __init__(self, available: bool = True):
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        m = re.search(r"\[(annual_report#\d+)\]", user)
        cid = m.group(1) if m else "annual_report#0"
        return ('{"abstain": false, "claims": [{"text": "Management cites revenue growth and '
                'a strong order book.", "chunk_ids": ["%s"], "kind": "fact"}]}' % cid)


_FILING = (
    "Management commentary: performance and outlook were strong this year, with a healthy order "
    "book. Risks and challenges include competition and input costs. Segment and business trends "
    "improved across capacity. The auditor noted related-party transactions and contingent "
    "liabilities; no promoter shares are pledged."
)


def test_read_filing_empty_text_abstains():
    assert read_filing("") == []
    assert read_filing(None) == []


def test_read_filing_without_llm_abstains():
    assert read_filing(_FILING, client=FakeClient(available=False)) == []


def test_read_filing_produces_cited_verified_readings():
    readings = read_filing(_FILING, client=FakeClient())
    assert len(readings) == 4                                   # one per CA topic
    grounded = [r for r in readings if not r.result.abstained]
    assert grounded, "expected at least one grounded, cited reading"
    claim = grounded[0].result.claims[0]
    # annual report is PRIMARY, so a grounded point is a verified fact cited to the filing.
    assert claim.is_verified_fact
    assert claim.citations[0].source_id == "annual_report"
