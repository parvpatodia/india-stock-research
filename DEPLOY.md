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

## 3. Sheet bridge via Apps Script (keyless, PRIVATE, read+write — recommended)
Google's "Secure by Default" policy blocks service-account key downloads on most accounts. The
Apps Script bridge needs no key, keeps the Sheet private (token-gated, not public), and supports
both auto-loading holdings and writing approvals back.
1. Own the Sheet: make your own copy (File → Make a copy) so you can deploy scripts on it. Your
   holdings must be the first tab (or a tab named `Holdings`) with column headers in **row 1**.
2. In that copy: **Extensions → Apps Script**. Paste `apps_script/Code.gs` from this repo, set
   `TOKEN` to a long random string.
3. **Deploy → New deployment → Web app → Execute as: Me, Who has access: Anyone → Deploy**,
   authorize, and copy the URL ending in `/exec`.
4. In secrets, set `apps_script_url` (the /exec URL) and `apps_script_token` (same as `TOKEN`).
   Holdings auto-load; approvals write to auto-created `Reports` + `Log` tabs. Every request
   carries the token, so "Anyone" access is still private.

Alternatives (only if you skip the Apps Script bridge):
- `holdings_csv_url`: a "Publish to web → CSV" link — auto-loads holdings but is **public** to
  anyone with the link, and read-only.
- Service account (`sheet_key` + `[gcp_service_account]`): read+write, but the JSON key is blocked
  on many Google orgs.

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
