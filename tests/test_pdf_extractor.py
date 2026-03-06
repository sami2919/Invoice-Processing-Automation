"""Tests for PDF text and table extraction."""

from unittest.mock import MagicMock, patch

from src.tools.pdf_extractor import extract_tables, extract_text


def _mock_page(text="Some invoice text", tables=None):
    page = MagicMock()
    page.extract_text.return_value = text
    page.extract_tables.return_value = tables or []
    return page


# --- extract_text ---

def test_basic_text_extraction():
    page = _mock_page("Invoice #1001\nTotal: $500")
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=[page])
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("fake.pdf")
    assert "Invoice #1001" in text
    assert warnings == []


def test_multi_page_extraction():
    pages = [_mock_page("Page 1 content"), _mock_page("Page 2 content")]
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=pages)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("multi.pdf")
    assert "Page 1" in text
    assert "Page 2" in text
    assert warnings == []


def test_blank_page_warns():
    pages = [_mock_page("Good page"), _mock_page("")]
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=pages)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("partial.pdf")
    assert "Good page" in text
    assert any("Page 2" in w and "blank" in w for w in warnings)


def test_all_blank_pages_warns_no_text():
    pages = [_mock_page(""), _mock_page("   ")]
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=pages)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("scanned.pdf")
    assert text == ""
    assert any("scanned image" in w for w in warnings)


def test_no_pages_warns():
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=[])
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("empty.pdf")
    assert text == ""
    assert any("no pages" in w.lower() for w in warnings)


def test_none_text_treated_as_blank():
    """pdfplumber returns None when a page has no extractable text."""
    page = _mock_page(None)
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=[page])
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        text, warnings = extract_text("none_text.pdf")
    assert text == ""
    assert any("blank" in w.lower() or "scanned" in w.lower() for w in warnings)


def test_corrupted_pdf_returns_error():
    with patch("pdfplumber.open", side_effect=Exception("Invalid PDF structure")):
        text, warnings = extract_text("corrupted.pdf")
    assert text == ""
    assert len(warnings) == 1
    assert "Invalid PDF structure" in warnings[0]


def test_file_not_found():
    with patch("pdfplumber.open", side_effect=FileNotFoundError("nope")):
        text, warnings = extract_text("/bad/path.pdf")
    assert text == ""
    assert len(warnings) == 1


# --- extract_tables ---

def test_basic_table_extraction():
    table_data = [["Item", "Qty", "Price"], ["WidgetA", "10", "250.00"]]
    page = _mock_page(tables=[table_data])
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=[page])
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        tables = extract_tables("invoice.pdf")
    assert len(tables) == 1
    assert tables[0][0][0] == "Item"
    assert tables[0][1][1] == "10"


def test_multiple_tables_across_pages():
    t1 = [["A", "B"], ["1", "2"]]
    t2 = [["C", "D"], ["3", "4"]]
    pages = [_mock_page(tables=[t1]), _mock_page(tables=[t2])]
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=pages)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        tables = extract_tables("multi_table.pdf")
    assert len(tables) == 2


def test_no_tables_returns_empty():
    page = _mock_page(tables=[])
    with patch("pdfplumber.open") as mock_open:
        mock_open.return_value.__enter__ = lambda s: MagicMock(pages=[page])
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        tables = extract_tables("no_tables.pdf")
    assert tables == []


def test_table_extraction_handles_errors():
    with patch("pdfplumber.open", side_effect=Exception("bad pdf")):
        tables = extract_tables("broken.pdf")
    assert tables == []
