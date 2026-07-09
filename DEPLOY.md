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
4. **Set sharing to Public.** Manage app → Settings → Sharing → "Anyone with the link can view".
   REQUIRED for the parents: a Private app forces a Streamlit/Google login wall (`/-/auth/app`)
   BEFORE our own gate, and the parents' Google accounts aren't authorized viewers. "Public" here
   only removes that Streamlit login wall — the app is still protected by our `app_password` and
   the `?key=` token below, so a stranger with just the base URL still can't get in.
5. Deploy. Open on the iPhone in Safari → Share → **Add to Home Screen** for an app icon.

### 4a. The parents' one-tap link (no password typing)
Give the parents this URL (not the bare one) so tapping their Home-Screen icon auto-signs-in:
`https://<your-app>.streamlit.app/?key=<APP_PASSWORD>` — replace `<APP_PASSWORD>` with the exact
`app_password` you set in secrets. The `?key=` is checked in `_check_password()` (app.py): a match
signs them in with no prompt; the bare URL (no key) still shows the password box, so the base link
alone is useless to a stranger. Have them Add-to-Home-Screen from the `?key=` URL so the token is
baked into the bookmark.

## 4b. Daily suggestions engine (24/7, free)
A GitHub Actions cron (`.github/workflows/daily.yml`) runs the research every morning, ranks
long-term-fit names (favorable/neutral, within your per-stock cap, improving trends), writes a
`Today` tab to the Sheet (shown in the app's Invest tab), and pushes the top picks to ntfy.
1. Repo → Settings → Secrets and variables → Actions → add: `APPS_SCRIPT_URL`, `APPS_SCRIPT_TOKEN`
   (same as the app), `LLM_MODEL`, `LLM_API_KEY` (optional, for the AR tiebreaker), and
   `NTFY_TOPIC` (a long, hard-to-guess topic name you choose, e.g. `parv-stocks-9f3a2c`).
2. Phone push: install the free **ntfy** app (iOS/Android) → Subscribe → enter the same topic.
3. Optional: add a `Watchlist` tab to the Sheet (a `Symbol` column) to research names you don't
   yet own. Trigger a first run from the repo's Actions tab (workflow_dispatch).

Note: GitHub's runners are datacenter IPs; if Screener is blocked there (single-source →
nothing cross-verifies → no picks), switch the second source to an API (e.g. a free FMP key).

## 4c. Daily engine on a Mac (launchd) — the free, full-quality path
Screener rate-limits the 32-stock batch from datacenter IPs (Streamlit Cloud, GitHub → thin
results), but not from a residential IP. So run the batch on the Mac; the app just displays the
`Today` tab. This gives the full ~12 picks for free.
1. Put the engine's config in the gitignored `.env`: `APPS_SCRIPT_URL`, `APPS_SCRIPT_TOKEN`,
   `NTFY_TOPIC`, `POSITION_CAP` (the script `load_dotenv`s it; the daily batch uses only
   yfinance + Screener, no LLM). Test it: `./.venv/bin/python scripts/daily_suggestions.py`.
2. Create `~/Library/LaunchAgents/com.parvpatodia.stockdaily.plist` running the venv python on
   `scripts/daily_suggestions.py`, `StartCalendarInterval` at 09:00, `RunAtLoad` true, logging to
   `data/daily_suggestions.log`. (Change the `Hour` to a time your Mac is usually awake; if it's
   asleep at that time, launchd runs it on the next wake.)
3. Load / reload / pause:
   - load:   `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.parvpatodia.stockdaily.plist`
   - run now: `launchctl kickstart -k gui/$(id -u)/com.parvpatodia.stockdaily`
   - pause:  `launchctl bootout gui/$(id -u)/com.parvpatodia.stockdaily`
Tradeoff: the Mac must be awake for it to run; it's daily-when-on, not truly 24/7. For real
overnight autonomy, use a paid datacenter-reachable data source instead.

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
