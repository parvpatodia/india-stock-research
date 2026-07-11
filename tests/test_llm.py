from src.llm.client import LLMClient, LiteLLMClient
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

    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        return self._response


def test_litellm_client_availability_tracks_model():
    assert LiteLLMClient(model=None).available is False
    configured = LiteLLMClient(model="nvidia_nim/deepseek-ai/deepseek-v3.2")
    assert configured.available is True
    assert configured.model_name == "nvidia_nim/deepseek-ai/deepseek-v3.2"


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


def test_answer_uses_the_retrieval_hint_to_surface_company_news():
    # WHY (real money, Ask-tab answer quality): a natural question like "what is the recent news"
    # shares NO words with a specific headline ("Reliance Q3 profit rises..."), so TF-IDF scored the
    # fetched news below the retrieval floor and it was NEVER retrieved -- live-reproduced: 0 chunks
    # for the tab's OWN default question. The user entered a specific stock, so retrieval must be
    # company-aware: augmenting the retrieval query with the resolved company name (which every
    # fetched-by-company headline contains) surfaces it. The MODEL is still asked the ORIGINAL
    # question, so the answer stays on topic and cites a real chunk.
    reg = SourceRegistry([Source("news_google", "Google News", CredibilityTier.ANALYST)])
    store = DocumentStore(registry=reg)
    store.add_document(
        "news_google",
        "[Reuters, 2026-07-01] Reliance Q3 profit rises 12 percent on refining margins.")
    payload = ('{"abstain": false, "claims": [{"text": "Reliance Q3 profit rose 12 percent.", '
               '"chunk_ids": ["news_google#0"], "kind": "opinion"}]}')
    a = GroundedAnalyst(client=FakeClient(payload))
    # Without the hint the question shares no words with the headline -> nothing retrieved -> abstain.
    assert a.answer("what is the recent news about it", store, reg).abstained
    # With the company name as the retrieval hint, the news is surfaced and answered.
    hinted = a.answer("what is the recent news about it", store, reg,
                      retrieval_hint="Reliance Industries")
    assert not hinted.abstained
    assert hinted.claims[0].citations[0].source_id == "news_google"
