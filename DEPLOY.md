# Deploy (Streamlit Community Cloud, free)

The parents use iPhones, so this deploys as a password-gated web app they open in Safari and
"Add to Home Screen". Local dev keeps using Ollama + `.env`; the deployed app uses a free hosted
model + the Google Sheet, all configured through Streamlit secrets. No secrets are committed.

## 1. Push the repo to GitHub (private)
The app reads real holdings only from the Sheet or an uploaded CSV; `holdings.csv`, `.env`,
`config/sources.yaml`, `data/`, and `.streamlit/secrets.toml` are gitignored. Keep the repo private.

## 2. Free hosted model (Groq)
1. Get a free key at https://console.groq.com/keys
2. Model string: `groq/llama-3.3-70b-versatile` (LiteLLM routes it; better annual-report
   extraction than the local 7B).

## 3. Holdings from the Google Sheet (keyless — recommended)
Google's "Secure by Default" policy blocks service-account key downloads on most accounts, so use
the keyless published-CSV route:
1. Sheet owner: **File → Share → Publish to web →** select the holdings tab **→ CSV → Publish**.
2. Copy the link (looks like `.../pub?gid=0&single=true&output=csv`) and set it as
   `holdings_csv_url` in secrets. Holdings then auto-load; no Google Cloud, no key.
3. Caveat: a published link is readable by anyone who has it. Read-only (approvals don't write
   back to the Sheet). Columns must include Symbol, Quantity, Avg Cost (Sector optional).

Service-account read+write (`sheet_key` + `[gcp_service_account]`) still works IF you can create a
JSON key, but that action is blocked on many Google orgs; prefer the published CSV above.

## 4. Deploy on Streamlit Community Cloud
1. https://share.streamlit.io → New app → point at this repo, branch, `app.py`.
2. Advanced settings → Python 3.12.
3. Paste secrets (see `.streamlit/secrets.toml.example`): `app_password`, `LLM_MODEL`,
   `LLM_API_KEY`, and the `[gcp_service_account]` block + `sheet_key`.
4. Deploy. Open on the iPhone in Safari → Share → **Add to Home Screen** for an app icon.

## 5. Verify after deploy
- Password gate appears and only the shared password gets in.
- Portfolio loads (from the Sheet if configured, else an uploaded CSV).
- Research a holding → verdict + stance + sizing render; approving it writes to the `Reports`
  tab and `Log`.
- `Ask` returns news-grounded, cited answers (proves the hosted model is wired).

## Notes
- The app degrades safely: no model → analysis still works, chat/tiebreaker off; no Sheet →
  falls back to CSV + a local JSON store; a bad Sheet connection falls back to local, never crashes.
- Nothing here is advice. Reports are drafts until the expert (you) approves them.
