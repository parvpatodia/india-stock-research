/**
 * India Equity Research - Sheet bridge (Google Apps Script Web App).
 *
 * Keyless private backend: no service-account key (Google blocks those on most accounts).
 * The web app is deployed "Execute as: Me / Who has access: Anyone", but EVERY request must
 * carry a shared secret TOKEN, so the endpoint is bearer-token private (unlike a published CSV,
 * which has no auth). It reads the holdings tab and reads/writes the Reports + Log tabs only.
 *
 * SETUP
 *  1. Open YOUR copy of the portfolio Sheet -> Extensions -> Apps Script.
 *  2. Replace the default file with this code. Set TOKEN below to a long random string and use
 *     the SAME value as `apps_script_token` in the Streamlit app's secrets.
 *  3. Deploy -> New deployment -> type "Web app" -> Execute as: Me, Who has access: Anyone -> Deploy.
 *  4. Authorize when prompted. Copy the Web app URL that ends in /exec -> that is `apps_script_url`.
 *
 * Your holdings must be the FIRST sheet (or a tab named "Holdings"), with the column headers in
 * ROW 1 (Symbol, Quantity, Avg Cost, and optionally Sector). The "Reports" and "Log" tabs are
 * created automatically the first time an approval is saved.
 */

const TOKEN = 'PASTE_THE_SAME_VALUE_AS_apps_script_token';

function doGet(e) {
  if (!e || e.parameter.token !== TOKEN) return _json_({ error: 'unauthorized' });
  const action = e.parameter.action || 'read';
  if (action === 'read') return _json_(readTab_(e.parameter.tab || 'Holdings'));
  return _json_({ error: 'unknown action' });
}

function doPost(e) {
  let body;
  try { body = JSON.parse(e.postData.contents); } catch (err) { return _json_({ error: 'bad json' }); }
  if (!body || body.token !== TOKEN) return _json_({ error: 'unauthorized' });
  if (body.action === 'write') { writeTab_(body.tab, body.header || [], body.rows || []); return _json_({ ok: true }); }
  if (body.action === 'append') { appendRow_(body.tab, body.header || [], body.row || {}); return _json_({ ok: true }); }
  return _json_({ error: 'unknown action' });
}

function _ss_() { return SpreadsheetApp.getActiveSpreadsheet(); }

function _sheetForRead_(tab) {
  const ss = _ss_();
  let sh = ss.getSheetByName(tab);
  if (!sh && tab === 'Holdings') sh = ss.getSheets()[0];  // holdings = first tab if not named "Holdings"
  return sh;
}

function readTab_(tab) {
  const sh = _sheetForRead_(tab);
  if (!sh) return [];
  const values = sh.getDataRange().getValues();
  if (values.length < 2) return [];
  const header = values[0].map(function (h) { return String(h).trim(); });
  const rows = [];
  for (let i = 1; i < values.length; i++) {
    const o = {};
    for (let j = 0; j < header.length; j++) o[header[j]] = values[i][j];
    rows.push(o);
  }
  return rows;
}

function writeTab_(tab, header, rows) {
  if (!header.length) return;
  const sh = _ss_().getSheetByName(tab) || _ss_().insertSheet(tab);
  sh.clear();
  const grid = [header];
  for (let i = 0; i < rows.length; i++) {
    const r = [];
    for (let j = 0; j < header.length; j++) { const v = rows[i][header[j]]; r.push(v != null ? v : ''); }
    grid.push(r);
  }
  sh.getRange(1, 1, grid.length, header.length).setValues(grid);
}

function appendRow_(tab, header, row) {
  if (!header.length) return;
  const sh = _ss_().getSheetByName(tab) || _ss_().insertSheet(tab);
  if (sh.getLastRow() === 0) sh.appendRow(header);
  const r = [];
  for (let j = 0; j < header.length; j++) { const v = row[header[j]]; r.push(v != null ? v : ''); }
  sh.appendRow(r);
}

function _json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
