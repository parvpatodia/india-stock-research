#!/usr/bin/env bash
# One-command quality gate for india-stock-research. Exits non-zero on the first failure.
# Runs: byte-compile, the unit suite, and a headless app smoke (renders with no exception).
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3

echo "==> byte-compile"
# shellcheck disable=SC2046
$PY -m py_compile app.py $(find src tests scripts -name '*.py')

echo "==> pytest"
$PY -m pytest -q

echo "==> app smoke (headless render, no exception, no error elements)"
$PY - <<'PYEOF'
import os
for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE"):
    os.environ.pop(k, None)
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py").run(timeout=120)
assert len(at.exception) == 0, f"app raised: {[e.value for e in at.exception]}"
assert len(at.error) == 0, f"app rendered error elements: {[e.value for e in at.error]}"
print("app smoke OK")
PYEOF

echo "ALL GREEN"
