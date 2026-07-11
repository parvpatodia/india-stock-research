"""Ground-truth eval cases captured from expert corrections.

When the expert establishes the true value of a figure, we store it together with the exact
figure snapshot the system used at that moment. Replaying the snapshot lets us check, forever
after, that the system never again TRUSTS a value the expert has flagged as wrong. This is the
"no mistake twice" mechanism, driven by the human reviewer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class GroundTruth:
    company: str
    figure: str
    correct_value: float
    # snapshot: figure_name -> list of [value, source_id, locator] used when the expert corrected.
    snapshot: dict[str, list]
    tolerance: float = 0.02
    note: str = ""
    reviewer: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        return cls(**data)


class EvalStore:
    """Append-only JSONL store of ground-truth cases. Path is injectable (tests use tmp)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, case: GroundTruth) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case.to_dict()) + "\n")

    def load(self) -> list[GroundTruth]:
        if not self.path.exists():
            return []
        cases = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # WHY skip, don't crash (resilience, real money): append-only writes are not atomic, so a
            # crash mid-add (or a manual edit) can leave a corrupt/partial line. load() runs on every
            # Research-tab render, so a single bad line must not crash the tab a parent is viewing --
            # skip it and keep the valid cases (the app's skip-bad-lines pattern, as in the AMFI
            # parser). TypeError/ValueError also cover a line whose fields no longer match the schema.
            try:
                cases.append(GroundTruth.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return cases
