"""H5: real PDF TABLE extraction.

The PURE formatter (pdfplumber tables -> chunker-friendly text) is tested deterministically with
fake table data (list-of-rows): its output must be detected as a table by structure._is_table_row
and a crore cell must become a typed NumericRecord through the real chunker + record layer. The
pdfplumber I/O is exercised with a tiny fpdf2-built sample PDF, and its degrade-safe fallback to
pypdf is proven by injection. Offline; no network."""
from src.research.grounding import DocumentStore, numeric_records
from src.research.numeric_records import RUPEES, find_record
from src.research.structure import _is_table_row, split_elements
from src.sources import pdf_tables
from src.sources.adapters import HttpDocumentAdapter
from src.sources.pdf_tables import (extract_pdf_tables_text, format_table,
                                    format_tables)

# A realistic consolidated P&L grid, as pdfplumber returns it (list of rows of cell strings).
PL_ROWS = [["Particulars", "FY2024", "FY2023"],
           ["Revenue from operations", "9,00,000", "7,92,000"],
           ["Profit before tax", "98,000", "88,000"],
           ["Net profit for the year", "73,670", "66,700"]]


# --- pure formatter ---------------------------------------------------------------------------


def test_formatter_rows_are_detected_as_table_rows_by_structure():
    text = format_table(PL_ROWS)
    lines = text.splitlines()
    assert lines, "formatter produced no lines"
    for line in lines:
        assert _is_table_row(line), f"structure._is_table_row missed: {line!r}"


def test_formatter_renders_markdown_pipe_columns_keeping_every_cell():
    text = format_table(PL_ROWS)
    assert "| Net profit for the year | 73,670 | 66,700 |" in text
    assert "| Revenue from operations | 9,00,000 | 7,92,000 |" in text


def test_formatter_skips_empty_rows_and_collapses_wrapped_cells():
    rows = [["Revenue", "9,00,000", "7,92,000"],
            [None, None, None],                       # fully-empty grid row -> dropped
            ["Other\nincome", "1,200", "1,100"]]      # wrapped cell newline collapsed
    text = format_table(rows)
    assert len(text.splitlines()) == 2
    assert "| Other income | 1,200 | 1,100 |" in text
    assert "\nincome" not in text


def test_formatter_empty_inputs_return_empty_string():
    assert format_table([]) == ""
    assert format_table(None) == ""
    assert format_table([[None, None]]) == ""          # only an empty row
    assert format_table([], caption="(Rs in crore)") == ""   # a lone caption is not a table
    assert format_tables([]) == ""


def test_caption_preserved_above_table_and_carries_scale_currency():
    text = format_table(PL_ROWS, caption="(Rs in crore)")
    assert text.splitlines()[0] == "(Rs in crore)"
    tables = [p for p in split_elements(text) if p.element_kind == "table"]
    assert len(tables) == 1
    assert tables[0].unit_scale == "crore"             # scale read from the units line
    assert tables[0].currency == "INR"
    assert "73,670" in tables[0].text                  # grid stays intact with its caption


def test_multiple_tables_stay_separate_regions():
    a = [["Item", "FY2024"], ["Cash", "5,000"]]
    b = [["Item", "FY2024"], ["Debt", "9,000"]]
    tables = [p for p in split_elements(format_tables([a, b])) if p.element_kind == "table"]
    assert len(tables) == 2


# --- end-to-end: a crore cell becomes a typed NumericRecord -----------------------------------


def test_crore_cell_becomes_typed_numeric_record_end_to_end():
    # formatter output -> element-aware chunker (structured ingest) -> typed records, all via the
    # SAME production path (DocumentStore.add_document(structured=True)) the app uses.
    blob = ("Consolidated Statement of Profit and Loss\n"
            + format_table(PL_ROWS, caption="(Rs in crore)"))
    store = DocumentStore(words_per_chunk=200, overlap=20)
    store.add_document("annual_report", blob, structured=True)

    records = numeric_records(store.retrieve("net profit for the year", k=10))
    rec = find_record("73,670", records)
    assert rec is not None, "the crore cell did not become a typed record"
    assert rec.unit == RUPEES
    assert rec.scale == "crore"
    assert rec.currency == "INR"
    assert rec.absolute_value == 73670 * 1e7           # deterministic normalization, not the LLM


# --- pdfplumber I/O on a tiny sample PDF (fpdf2 is already a dependency) -----------------------


def _sample_pdf_bytes() -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "Consolidated Statement of Profit and Loss",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, "(Rs in crore)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    with pdf.table() as table:
        for r in PL_ROWS:
            row = table.row()
            for cell in r:
                row.cell(cell)
    return bytes(pdf.output())


def test_extract_pdf_tables_text_keeps_the_table_intact():
    text = extract_pdf_tables_text(_sample_pdf_bytes())
    assert text is not None
    tables = [p for p in split_elements(text) if p.element_kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "(Rs in crore)" in t.text                   # units line stayed above the grid
    assert t.unit_scale == "crore"
    assert "73,670" in t.text and "66,700" in t.text   # every data row together


def test_adapter_pdf_path_extracts_structured_table():
    raw = _sample_pdf_bytes()
    a = HttpDocumentAdapter("annual_report",
                            fetcher=lambda url: (raw, "application/pdf"))
    doc = a.fetch("http://x/report.pdf")[0]
    tables = [p for p in split_elements(doc.text) if p.element_kind == "table"]
    assert len(tables) == 1
    assert tables[0].unit_scale == "crore"
    assert "Net profit for the year" in tables[0].text


def test_extract_returns_none_on_garbage_never_raises():
    assert extract_pdf_tables_text(b"this is not a pdf at all") is None
    assert extract_pdf_tables_text(b"") is None


def test_adapter_falls_back_to_pypdf_when_table_path_abstains(monkeypatch):
    # Force the pdfplumber path to abstain; the adapter must still return the pypdf text of a real
    # PDF (degrade-safe), never an empty string or a crash.
    monkeypatch.setattr(pdf_tables, "extract_pdf_tables_text", lambda raw: None)
    raw = _sample_pdf_bytes()
    a = HttpDocumentAdapter("annual_report",
                            fetcher=lambda url: (raw, "application/pdf"))
    docs = a.fetch("http://x/report.pdf")
    assert len(docs) == 1
    assert "73,670" in docs[0].text                    # pypdf flat text still carries the figure
