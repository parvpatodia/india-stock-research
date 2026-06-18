from src.llm.client import LLMClient, LiteLLMClient
from src.portfolio.models import PositionAnalysis
from src.research.analyst import ResearchAnalyst
from src.research.claims import FACT
from src.research.grounded_analyst import GroundedAnalyst
from src.research.grounding import DocumentStore
from src.sources.registry import CredibilityTier, Source, SourceRegistry


class FakeClient(LLMClient):
    """Canned-response client so the analyst flows can be tested with no network/provider."""

    def __init__(self, response: str, available: bool = True):
        self._response = response
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        return self._response


def test_litellm_client_availability_tracks_model():
    assert LiteLLMClient(model=None).available is False
    configured = LiteLLMClient(model="nvidia_nim/deepseek-ai/deepseek-v3.2")
    assert configured.available is True
    assert configured.model_name == "nvidia_nim/deepseek-ai/deepseek-v3.2"


def test_research_analyst_degrades_without_llm():
    a = ResearchAnalyst(client=FakeClient("", available=False))
    p = PositionAnalysis("RELIANCE", 15, 1180.0, 1331.0, "Energy", 0.3)
    assert "unavailable" in a.research_note(p, {"name": "Reliance"}).lower()


def test_research_analyst_uses_injected_client():
    a = ResearchAnalyst(client=FakeClient("NOTE BODY"))
    p = PositionAnalysis("RELIANCE", 15, 1180.0, 1331.0, "Energy", 0.3)
    assert a.research_note(p, {"name": "Reliance"}) == "NOTE BODY"


def _store_and_registry():
    reg = SourceRegistry([Source("amfi", "AMFI", CredibilityTier.PRIMARY)])
    store = DocumentStore(words_per_chunk=30, overlap=5, registry=reg)
    store.add_document(
        "amfi",
        "A SIP invests a fixed amount every month into a mutual fund scheme. "
        "NAV is the net asset value per unit of the fund.",
    )
    return store, reg


def test_grounded_analyst_end_to_end_with_fake_client():
    store, reg = _store_and_registry()
    # First chunk id is deterministic: source_id + '#0'.
    payload = ('{"abstain": false, "claims": [{"text": "A SIP invests a fixed amount '
               'monthly.", "chunk_ids": ["amfi#0"], "kind": "fact"}]}')
    a = GroundedAnalyst(client=FakeClient(payload))
    res = a.answer("what is a SIP mutual fund", store, reg, as_of="2026-06-18")
    assert not res.abstained
    assert res.claims[0].kind == FACT and res.claims[0].is_verified_fact
    assert res.claims[0].citations[0].source_id == "amfi"


def test_grounded_analyst_abstains_without_llm():
    store, reg = _store_and_registry()
    a = GroundedAnalyst(client=FakeClient("", available=False))
    res = a.answer("what is a SIP", store, reg)
    assert res.abstained and "configured" in (res.abstain_reason or "").lower()
