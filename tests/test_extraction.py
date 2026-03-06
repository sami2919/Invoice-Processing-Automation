"""Tests for file parsing and extraction helpers."""

import pytest

from src.agents.extraction import _build_user_message, _extract_json_block
from src.tools.file_parser import parse_file
from src.tools.inventory_db import fuzzy_match_item


def test_parse_txt_format(sample_invoices):
    text, file_type = parse_file(sample_invoices["1001"])
    assert file_type == "txt"
    assert "INV-1001" in text
    assert "Widgets Inc." in text


def test_parse_json_format(sample_invoices):
    text, file_type = parse_file(sample_invoices["1004"])
    assert file_type == "json"
    assert "INV-1004" in text


def test_parse_csv_format(sample_invoices):
    text, file_type = parse_file(sample_invoices["1006"])
    assert file_type == "csv"
    assert "INV-1006" in text


def test_parse_xml_format(sample_invoices):
    text, file_type = parse_file(sample_invoices["1014"])
    assert file_type == "xml"
    assert "INV-1014" in text
    assert "EUR" in text


def test_ocr_artifact_normalization():
    """_extract_json_block handles markdown fences from LLM output."""
    llm_response = (
        "```json\n"
        '{"invoice_number": "INV-1012", "vendor_name": "QuickShip Distributers",'
        ' "invoice_date": "2026-01-26", "total_amount": 9975.00,'
        ' "line_items": [{"item_name": "WidgetA", "quantity": 12, "unit_price": 250.0,'
        ' "line_total": 3000.0, "note": null}], "currency": "USD",'
        ' "confidence_scores": {}, "extraction_warnings": ["OCR: 2O26 corrected to 2026"]}\n'
        "```"
    )
    result = _extract_json_block(llm_response)
    assert result["invoice_number"] == "INV-1012"
    assert result["total_amount"] == 9975.00
    assert any("OCR" in w for w in result["extraction_warnings"])


def test_fuzzy_item_matching(test_db):
    assert fuzzy_match_item("Widget A", db_path=test_db) == "WidgetA"


def test_fuzzy_match_gadget_x(test_db):
    assert fuzzy_match_item("Gadget X", db_path=test_db) == "GadgetX"


def test_malformed_input():
    with pytest.raises(FileNotFoundError):
        parse_file("/nonexistent/path/invoice_2003.txt")


def test_unsupported_format(tmp_path):
    bad_file = tmp_path / "invoice.docx"
    bad_file.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_file(str(bad_file))


def test_retry_prompt_includes_feedback():
    msg = _build_user_message("INVOICE TEXT", "vendor_name is missing",
                               {"invoice_number": "INV-X", "vendor_name": "", "total_amount": 0.0})
    assert "SELF-CORRECTION REQUIRED" in msg
    assert "vendor_name is missing" in msg


def test_no_self_correction_without_feedback():
    msg = _build_user_message("INVOICE TEXT", "", None)
    assert "SELF-CORRECTION" not in msg
