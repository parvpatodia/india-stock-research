"""Live end-to-end smoke of the grounded pipeline against a real local LLM (Ollama).

Run the ollama server and pull the model first, then:
    ./.venv/bin/python scripts/live_smoke.py

The document text below is SYNTHETIC test fixture data, not real financial figures. The
point is to prove the mechanism: the model answers only from the provided chunks, cites
them, and abstains when the answer is not in the sources. It is not advice.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import LiteLLMClient  # noqa: E402
from src.research.grounded_analyst import GroundedAnalyst  # noqa: E402
from src.research.grounding import DocumentStore  # noqa: E402
from src.sources.registry import CredibilityTier, Source, SourceRegistry  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "ollama_chat/qwen2.5:7b")
API_BASE = os.environ.get("SMOKE_API_BASE", "http://localhost:11434")

# SYNTHETIC fixtures (not real numbers) under registered primary sources.
ACME_AR = (
    "Acme Industries Ltd, Annual Report FY2024. In FY2024 the company reported consolidated "
    "revenue of Rs 974000 crore, up 2.6 percent year on year. Net profit was Rs 79000 crore. "
    "The board recommended a dividend of Rs 10 per equity share."
)
FUND_FACT = (
    "XYZ Bluechip Fund (Direct, Growth) factsheet. The scheme's NAV was Rs 88.42 as of "
    "18-Jun-2026. The expense ratio is 0.62 percent. The fund is benchmarked to the NIFTY 100."
)


def main() -> int:
    registry = SourceRegistry([
        Source("acme_ar_fy24", "Acme Industries Annual Report FY2024", CredibilityTier.PRIMARY),
        Source("xyz_fund_factsheet", "XYZ Bluechip Fund factsheet", CredibilityTier.PRIMARY),
    ])
    store = DocumentStore(words_per_chunk=80, overlap=15, registry=registry)
    store.add_document("acme_ar_fy24", ACME_AR, locator_prefix="FY24 AR")
    store.add_document("xyz_fund_factsheet", FUND_FACT, locator_prefix="factsheet")

    client = LiteLLMClient(model=MODEL, api_base=API_BASE)
    print(f"client available: {client.available} | model: {client.model_name}\n")

    analyst = GroundedAnalyst(client=client)
    probes = [
        ("answerable", "What was Acme Industries' FY2024 revenue and net profit?"),
        ("in-source but not stated", "What is Acme Industries' debt-to-equity ratio?"),
        ("unrelated, no source", "What is the capital of France?"),
    ]
    for label, question in probes:
        print(f"=== [{label}] {question}")
        res = analyst.answer(question, store, registry, as_of="2026-06-18")
        if res.abstained:
            print(f"  ABSTAINED: {res.abstain_reason}\n")
            continue
        for c in res.claims:
            srcs = ", ".join(cit.source_id for cit in c.citations) or "(none)"
            verified = " [VERIFIED FACT]" if c.is_verified_fact else ""
            print(f"  - ({c.kind}{verified}) {c.text}  <cites: {srcs}>")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
