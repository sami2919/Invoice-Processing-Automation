"""Tests for the fraud detection agent."""

from src.agents.fraud import fraud_detection_node
from src.tools.inventory_db import record_invoice


def _signals_by_type(fraud_result: dict) -> dict:
    return {s["signal_type"]: s["weight"] for s in fraud_result.get("signals", [])}


def test_clean_invoice_low_risk(clean_state, patch_db, mock_grok):
    result = fraud_detection_node(clean_state)
    fr = result["fraud_result"]
    assert fr["risk_score"] < 30
    assert fr["recommendation"] == "auto_approve"
    signals = _signals_by_type(fr)
    assert "unknown_vendor" not in signals
    assert "urgency_language" not in signals


def test_urgency_language_detection(patch_db, mock_grok):
    state = {
        "file_path": "data/invoices/invoice_1001.txt",
        "raw_text": "INVOICE\nVendor: Widgets Inc.\nNotes: URGENT - Pay immediately!",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-URGENCY",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0,
                            "line_total": 250.0, "note": None}],
            "total_amount": 250.0,
            "currency": "USD",
            "notes": "URGENT - Pay immediately!",
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert signals.get("urgency_language") == 15


def test_fuzzy_vendor_match_no_false_positive(patch_db, mock_grok):
    """A vendor with minor spacing/typo differences from a seeded vendor should NOT
    be flagged as unknown."""
    state = {
        "file_path": "test.txt",
        "raw_text": "INVOICE\nVendor: WidgetsInc.",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-FUZZY-V",
            "vendor_name": "WidgetsInc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0,
                            "line_total": 250.0, "note": None}],
            "total_amount": 250.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert "unknown_vendor" not in signals, (
        "Vendor 'WidgetsInc.' should fuzzy-match 'Widgets Inc.' and not be flagged as unknown"
    )


def test_unknown_vendor_signal(patch_db, mock_grok):
    state = {
        "file_path": "test.txt",
        "raw_text": "INVOICE\nVendor: Unknown Startup LLC",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-UNKN",
            "vendor_name": "Unknown Startup LLC",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0,
                            "line_total": 250.0, "note": None}],
            "total_amount": 250.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert signals.get("unknown_vendor") == 20


def test_duplicate_invoice_signal(patch_db, mock_grok):
    record_invoice("INV-DUP", "Widgets Inc.", 5000.0, "approved", db_path=patch_db)
    state = {
        "file_path": "test.txt", "raw_text": "INVOICE", "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-DUP",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0,
                            "line_total": 250.0, "note": None}],
            "total_amount": 250.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert signals.get("duplicate_invoice") == 25


def test_threshold_manipulation(patch_db, mock_grok):
    state = {
        "file_path": "test.txt", "raw_text": "INVOICE", "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-2005",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [
                {"item_name": "WidgetA", "quantity": 39.0, "unit_price": 250.0,
                 "line_total": 9750.0, "note": None},
                {"item_name": "WidgetB", "quantity": 0.4999, "unit_price": 500.0,
                 "line_total": 249.99, "note": None},
            ],
            "total_amount": 9999.99,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert signals.get("threshold_manipulation") == 10


def test_fuzzy_item_match_no_false_unknown(patch_db, mock_grok):
    """Items with OCR spaces like 'Widget A' should fuzzy-match 'WidgetA' and
    NOT trigger unknown_items signal."""
    state = {
        "file_path": "test.txt",
        "raw_text": "INVOICE\nVendor: Widgets Inc.",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-FUZZY-ITEM",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [
                {"item_name": "Widget A", "quantity": 2.0, "unit_price": 250.0,
                 "line_total": 500.0, "note": None},
                {"item_name": "Widget B", "quantity": 1.0, "unit_price": 500.0,
                 "line_total": 500.0, "note": None},
            ],
            "total_amount": 1000.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert "unknown_items" not in signals, (
        "'Widget A' / 'Widget B' should fuzzy-match inventory items"
    )


def test_math_error_signal_fires_on_discrepancy(patch_db, mock_grok):
    """Line items sum to $1,000 but stated total is $1,200 — math_error should fire."""
    state = {
        "file_path": "test.txt", "raw_text": "INVOICE", "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-MATH-ERR",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [
                {"item_name": "WidgetA", "quantity": 2.0, "unit_price": 250.0,
                 "line_total": 500.0, "note": None},
                {"item_name": "WidgetB", "quantity": 1.0, "unit_price": 500.0,
                 "line_total": 500.0, "note": None},
            ],
            "total_amount": 1200.0,  # should be 1000
            "currency": "USD",
            "notes": None,
            "tax_amount": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    signals = _signals_by_type(result["fraud_result"])
    assert signals.get("math_error") == 10, (
        "math_error signal should fire when line items don't match stated total"
    )


def test_math_error_fires_when_tax_causes_discrepancy(patch_db, mock_grok):
    """INV-1013 scenario: line items sum to $21,040 but total is $22,562.80.
    Tax explains part of the gap, but the fraud agent should NOT account for tax —
    it should flag the mismatch between line items sum and stated total."""
    state = {
        "file_path": "test.txt", "raw_text": "INVOICE", "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-1013",
            "vendor_name": "Atlas Industrial Supply",
            "invoice_date": "2026-01-24",
            "due_date": "2026-03-24",
            "line_items": [
                {"item_name": "WidgetA", "quantity": 15.0, "unit_price": 250.0,
                 "line_total": 3750.0, "note": None},
                {"item_name": "WidgetB", "quantity": 10.0, "unit_price": 500.0,
                 "line_total": 5000.0, "note": None},
                {"item_name": "GadgetX", "quantity": 5.0, "unit_price": 750.0,
                 "line_total": 3750.0, "note": None},
                {"item_name": "WidgetA", "quantity": 5.0, "unit_price": 240.0,
                 "line_total": 1200.0, "note": "Volume discount"},
                {"item_name": "WidgetB", "quantity": 8.0, "unit_price": 480.0,
                 "line_total": 3840.0, "note": "Volume discount"},
                {"item_name": "GadgetX", "quantity": 3.0, "unit_price": 750.0,
                 "line_total": 2250.0, "note": "Expedited"},
                {"item_name": "WidgetA", "quantity": 2.0, "unit_price": 250.0,
                 "line_total": 500.0, "note": "Replacement"},
                {"item_name": "GadgetX", "quantity": 1.0, "unit_price": 750.0,
                 "line_total": 750.0, "note": "Sample"},
            ],
            "subtotal": 21040.0,
            "tax_amount": 1472.80,
            "total_amount": 22562.80,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    result = fraud_detection_node(state)
    fr = result["fraud_result"]
    signals = _signals_by_type(fr)
    assert signals.get("math_error") == 10, (
        "math_error should fire: line items sum $21,040 vs stated $22,562.80"
    )
    assert fr["risk_score"] > 0, "INV-1013 should have non-zero risk score"


def test_no_math_error_on_clean_invoice(clean_state, patch_db, mock_grok):
    """Clean invoice where line items match total exactly — no math_error signal."""
    result = fraud_detection_node(clean_state)
    signals = _signals_by_type(result["fraud_result"])
    assert "math_error" not in signals, (
        "math_error should NOT fire when line items match stated total"
    )


def test_composite_risk_score(fraud_state, patch_db, mock_grok):
    result = fraud_detection_node(fraud_state)
    fr = result["fraud_result"]
    assert fr["risk_score"] >= 70
    assert fr["recommendation"] == "block"
    signals = _signals_by_type(fr)
    assert "unknown_vendor" in signals
    assert "urgency_language" in signals


def test_risk_recommendation_mapping(patch_db, mock_grok):
    def _run(vendor: str, total: float) -> str:
        state = {
            "file_path": "test.txt",
            "raw_text": f"INVOICE\nVendor: {vendor}",
            "file_type": "txt",
            "extracted_invoice": {
                "invoice_number": f"INV-RECO-{int(total)}",
                "vendor_name": vendor,
                "invoice_date": "2026-01-15",
                "due_date": "2026-02-01",
                "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": total,
                                "line_total": total, "note": None}],
                "total_amount": total,
                "currency": "USD",
                "notes": None,
                "confidence_scores": {},
                "extraction_warnings": [],
            },
            "extraction_retries": 0, "extraction_feedback": "",
            "validation_result": None, "fraud_result": None,
            "approval_decision": None, "payment_result": None,
            "audit_trail": [], "error_message": None,
            "current_agent": "fraud", "decision_explanation": "",
        }
        return fraud_detection_node(state)["fraud_result"]["recommendation"]

    assert _run("Widgets Inc.", 200.0) == "auto_approve"

    # block case
    block_state = {
        "file_path": "test.txt",
        "raw_text": "URGENT wire transfer immediately",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-BLOCK",
            "vendor_name": "Fraudster LLC",
            "invoice_date": "2026-01-20",
            "due_date": None,
            "line_items": [{"item_name": "FakeItem", "quantity": 1.0, "unit_price": 1000.0,
                            "line_total": 1000.0, "note": None}],
            "total_amount": 1000.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0, "extraction_feedback": "",
        "validation_result": None, "fraud_result": None,
        "approval_decision": None, "payment_result": None,
        "audit_trail": [], "error_message": None,
        "current_agent": "fraud", "decision_explanation": "",
    }
    r_block = fraud_detection_node(block_state)
    assert r_block["fraud_result"]["recommendation"] == "block"
