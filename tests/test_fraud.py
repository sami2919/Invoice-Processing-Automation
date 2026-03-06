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
    assert signals.get("duplicate_invoice") == 30


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
