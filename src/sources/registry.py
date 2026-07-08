"""Source registry with credibility tiers.

This is the gate that makes "no unverified claim passes as fact" real. Only a Tier-1
PRIMARY source can back a claim shown as fact. Tier-2 is attributed opinion. Tier-3
(finfluencers) is context only and can never back a fact or a number. The owner declares
his sources in config/sources.yaml; nothing is hard-coded as trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import yaml


class CredibilityTier(IntEnum):
    PRIMARY = 1   # annual reports, AGM, SEBI filings, exchange/AMFI data, audited financials
    ANALYST = 2   # registered analyst/broker research, reputable financial press
    CREATOR = 3   # YouTube/Instagram finfluencers - context only, attributed, never a fact

    @property
    def label(self) -> str:
        return {1: "Primary", 2: "Analyst", 3: "Creator"}[int(self)]


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    tier: CredibilityTier
    url: str | None = None
    notes: str = ""

    @property
    def citable_as_fact(self) -> bool:
        # WHY: only primary sources may back a stated fact. Everything else is opinion/context.
        return self.tier == CredibilityTier.PRIMARY


class SourceRegistry:
    def __init__(self, sources: list[Source] | None = None):
        self._by_id: dict[str, Source] = {}
        for source in sources or []:
            self.add(source)

    def add(self, source: Source) -> None:
        if source.id in self._by_id:
            raise ValueError(f"duplicate source id: {source.id}")
        self._by_id[source.id] = source

    def get(self, source_id: str) -> Source | None:
        return self._by_id.get(source_id)

    def by_tier(self, tier: CredibilityTier) -> list[Source]:
        return [s for s in self._by_id.values() if s.tier == tier]

    def all_sources(self) -> list[Source]:
        """Every registered source. Used to merge registries (e.g. add live news feeds to a
        config-loaded registry) without reaching into private state."""
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    @classmethod
    def from_config(cls, path: str | Path) -> "SourceRegistry":
        """Load sources from a tier-keyed YAML file. Unknown tiers raise, so a typo in the
        config fails loudly instead of silently mislabeling a source's trust level."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        tier_by_name = {
            "primary": CredibilityTier.PRIMARY,
            "analyst": CredibilityTier.ANALYST,
            "creator": CredibilityTier.CREATOR,
        }
        sources: list[Source] = []
        for tier_name, entries in data.items():
            key = str(tier_name).strip().lower()
            if key not in tier_by_name:
                raise ValueError(f"unknown tier '{tier_name}' in {path}; "
                                 f"expected one of {list(tier_by_name)}")
            for entry in entries or []:
                sources.append(Source(
                    id=entry["id"],
                    name=entry["name"],
                    tier=tier_by_name[key],
                    url=entry.get("url"),
                    notes=entry.get("notes", ""),
                ))
        return cls(sources)
