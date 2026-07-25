"""Element-aware chunking: split a document on structural boundaries (headings, paragraphs,
and TABLES kept intact with their caption/units/period line), not blind fixed windows.
Fixtures mirror real Indian annual-report text. Offline; deterministic."""
from src.research.structure import split_elements, structure_chunks, window_split

AR = """Management Discussion and Analysis
The company delivered a strong year with revenue growth across all segments and a healthy
order book. Management remains cautious about input cost inflation for the coming year.

Consolidated Statement of Profit and Loss
(Rs in crore)
Particulars                          FY2024      FY2023
Revenue from operations             9,00,000    7,92,000
Profit before tax                     98,000      88,000
Net profit for the year               73,670      66,700

Key Risks
The company discloses risks from commodity price volatility, foreign exchange exposure, and
regulatory changes affecting the sector.
"""


def test_table_is_one_chunk_with_caption_and_title_intact():
    tables = [p for p in split_elements(AR) if p.element_kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "(Rs in crore)" in t.text                       # units line kept with the table
    assert "Statement of Profit and Loss" in t.text        # title absorbed into the table
    assert "Revenue from operations" in t.text
    assert "Net profit for the year" in t.text
    assert "73,670" in t.text and "66,700" in t.text        # every data row stays together


def test_table_carries_scale_currency_and_period_metadata():
    t = [p for p in split_elements(AR) if p.element_kind == "table"][0]
    assert t.unit_scale == "crore"
    assert t.currency == "INR"
    assert t.fiscal_period == "FY2024"


def test_paragraphs_separated_and_tagged_with_section():
    paras = [p for p in split_elements(AR) if p.element_kind == "paragraph"]
    assert any("order book" in p.text for p in paras)
    risk = [p for p in paras if "commodity price volatility" in p.text][0]
    assert risk.section == "Key Risks"
    # a paragraph is not a table and carries no table scale
    assert risk.unit_scale is None


def test_headings_are_emitted_and_set_section():
    heads = {p.text for p in split_elements(AR) if p.element_kind == "heading"}
    assert "Management Discussion and Analysis" in heads
    assert "Key Risks" in heads


def test_table_never_split_even_when_large():
    rows = "\n".join(f"Line item {i}    {i * 1000:,}    {i * 900:,}" for i in range(1, 40))
    doc = f"Balance Sheet\n(Rs in lakh)\nParticulars   FY2024   FY2023\n{rows}\n"
    tables = [p for p in structure_chunks(doc, words_per_chunk=20, overlap=5)
              if p.element_kind == "table"]
    assert len(tables) == 1                    # NOT fragmented by the word window
    assert "Line item 39" in tables[0].text
    assert tables[0].unit_scale == "lakh"


def test_oversized_paragraph_is_window_split_keeping_section():
    doc = "Business Overview\n" + ("word " * 200) + "\n"
    paras = [p for p in structure_chunks(doc, words_per_chunk=50, overlap=10)
             if p.element_kind == "paragraph"]
    assert len(paras) >= 3                      # long prose still chunked for retrieval
    assert all(p.section == "Business Overview" for p in paras)


def test_markdown_pipe_table_detected():
    doc = "Metrics\n| Particulars | FY2024 | FY2023 |\n| Net profit | 73,670 | 66,700 |\n"
    tables = [p for p in split_elements(doc) if p.element_kind == "table"]
    assert len(tables) == 1
    assert "73,670" in tables[0].text


def test_plain_text_no_structure_is_one_paragraph():
    pieces = split_elements("Just a simple sentence about the business with no structure.")
    assert len(pieces) == 1
    assert pieces[0].element_kind == "paragraph"


def test_empty_text_yields_no_pieces():
    assert split_elements("") == []
    assert split_elements("   \n  ") == []
    assert structure_chunks("", 100, 20) == []


def test_window_split_matches_legacy_blind_behaviour():
    # window_split is the canonical word-window splitter reused by the blind ingestion path;
    # a short text is one piece, a long one overlaps by `overlap`.
    assert window_split("a b c", 10, 2) == ["a b c"]
    pieces = window_split(" ".join(str(i) for i in range(100)), 30, 5)
    assert len(pieces) > 1
    assert pieces[0].split()[0] == "0"
