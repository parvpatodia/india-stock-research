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


def test_whitespace_only_model_is_not_configured_and_a_real_model_is_trimmed():
    # WHY (real money, Ask-tab reliability): env / .env / Streamlit TOML secrets very commonly carry a
    # trailing space or newline (a quoted-with-space value, or a copy-paste). An unstripped LLM_MODEL
    # made `available` read True for a WHITESPACE-ONLY value (then complete() errored on an empty
    # model), and passed a trailing-space model string straight to litellm, failing every lookup --
    # silently breaking the Ask tab even though a valid model was intended. Whitespace-only -> NOT
    # configured (honest "no LLM" state); a real value is trimmed so the call works.
    import os
    os.environ.pop("LLM_MODEL", None)
    assert LiteLLMClient(model="   ").available is False
    assert LiteLLMClient(model="\n").available is False
    assert LiteLLMClient(model="\t ").available is False
    c = LiteLLMClient(model="  ollama/llama3.1  ")
    assert c.available is True
    assert c.model == "ollama/llama3.1"                # trimmed, not the whitespace-padded string
    assert c.model_name == "ollama/llama3.1"


def test_api_key_and_base_are_trimmed_and_blank_becomes_none():
    # WHY: the same whitespace-in-config gotcha -- a trailing-newline API key fails auth and a padded
    # api_base fails the request. Trim both; a whitespace-only value becomes None (absent), so litellm
    # falls back to the provider's own env (e.g. NVIDIA_NIM_API_KEY) instead of sending blank creds.
    c = LiteLLMClient(model="m", api_key="  sk-abc\n", api_base=" http://localhost:11434 ")
    assert c.api_key == "sk-abc" and c.api_base == "http://localhost:11434"
    c2 = LiteLLMClient(model="m", api_key="   ", api_base="\n")
    assert c2.api_key is None and c2.api_base is None


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


def test_citation_locator_carries_source_provenance_for_freshness():
    # WHY (real money, Ask-tab freshness): a news-backed claim must let the reader judge how recent
    # it is. The citation's locator carries the human-readable provenance -- for a news item, the
    # publisher and article DATE ("Reuters, 2026-05-15") -- not the opaque internal chunk id
    # ("news_google#0"), so the Ask tab can show the date beside the claim.
    reg = SourceRegistry([Source("news_google", "Google News", CredibilityTier.ANALYST)])
    store = DocumentStore(registry=reg)
    store.add_document("news_google", "Reliance Q3 profit rose 12 percent on refining margins.",
                       locator_prefix="Reuters, 2026-05-15")
    payload = ('{"abstain": false, "claims": [{"text": "Reliance Q3 profit rose 12 percent.", '
               '"chunk_ids": ["news_google#0"], "kind": "opinion"}]}')
    res = GroundedAnalyst(client=FakeClient(payload)).answer("Reliance profit", store, reg)
    assert not res.abstained
    assert "2026-05-15" in res.claims[0].citations[0].locator
    assert res.claims[0].citations[0].locator.startswith("Reuters")   # not an opaque chunk id


def test_answer_drops_verbatim_duplicate_claims():
    # WHY (Ask-tab + annual-report-reader quality): a model can restate the SAME fact more than once
    # (it appears in two retrieved chunks), and every duplicate rendered as its own line reads as
    # broken and repetitive to a non-expert. The FIRST valid occurrence is kept; a later verbatim
    # (case/whitespace-insensitive) repeat is dropped; genuinely distinct claims are untouched.
    reg = SourceRegistry([Source("news_google", "Google News", CredibilityTier.ANALYST)])
    store = DocumentStore(registry=reg)
    store.add_document("news_google", "Reliance Q3 profit rose 12 percent on refining margins.")
    payload = ('{"abstain": false, "claims": ['
               '{"text": "Reliance Q3 profit rose 12 percent.", "chunk_ids": ["news_google#0"], "kind": "opinion"},'
               '{"text": "  reliance q3 profit ROSE 12 percent. ", "chunk_ids": ["news_google#0"], "kind": "opinion"},'
               '{"text": "Refining margins drove the rise.", "chunk_ids": ["news_google#0"], "kind": "opinion"}]}')
    res = GroundedAnalyst(client=FakeClient(payload)).answer("Reliance news", store, reg)
    texts = [c.text for c in res.claims]
    assert len(texts) == 2                                       # the case/space-different repeat dropped
    assert "Reliance Q3 profit rose 12 percent." in texts        # first occurrence kept verbatim
    assert "Refining margins drove the rise." in texts           # a distinct claim survives


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
