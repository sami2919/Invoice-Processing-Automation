"""Tests for the fraud detection agent's individual signals and scoring."""

from unittest.mock import patch

from src.agents.fraud import fraud_detection_node


def _base_state(overrides=None):
    """Minimal state with a clean extracted invoice."""
    inv = {
        "invoice_number": "INV-TEST-001",
        "vendor_name": "Widgets Inc.",
        "invoice_date": "2026-01-15",
        "due_date": "2026-02-01",
        "line_items": [
            {
                "item_name": "WidgetA",
                "quantity": 5.0,
                "unit_price": 250.0,
                "line_total": 1250.0,
                "note": None,
            },
        ],
        "subtotal": 1250.0,
        "tax_amount": None,
        "total_amount": 1250.0,
        "currency": "USD",
        "payment_terms": "Net 30",
        "notes": None,
    }
    if overrides:
        inv.update(overrides)
    return {
        "extracted_invoice": inv,
        "raw_text": "Vendor: Widgets Inc.\nTotal: $1250",
        "file_type": "txt",
    }


def _run(state):
    """Run fraud detection with mocked LLM narrative."""
    with patch("src.agents.fraud.assess", return_value="Mock narrative."):
        return fraud_detection_node(state)


def _signals(result):
    return result["fraud_result"]["signals"]


def _signal_types(result):
    return [s["signal_type"] for s in _signals(result)]


def _score(result):
    return result["fraud_result"]["risk_score"]


def test_clean_invoice_no_signals(patch_db):
    result = _run(_base_state())
    assert _score(result) == 0
    assert _signals(result) == []
    assert result["fraud_result"]["recommendation"] == "auto_approve"


def test_unknown_vendor(patch_db):
    state = _base_state({"vendor_name": "Totally Unknown Corp"})
    result = _run(state)
    assert "unknown_vendor" in _signal_types(result)


def test_elevated_vendor_risk(patch_db):
    state = _base_state({"vendor_name": "Gadgets Co."})
    result = _run(state)
    assert "elevated_vendor_risk" in _signal_types(result)


def test_duplicate_invoice(patch_db):
    from src.tools.inventory_db import record_invoice

    record_invoice("INV-DUP-001", "Widgets Inc.", 500.0, "approved")
    state = _base_state({"invoice_number": "INV-DUP-001"})
    result = _run(state)
    assert "duplicate_invoice" in _signal_types(result)


def test_urgency_language(patch_db):
    state = _base_state()
    state["raw_text"] = "URGENT: Pay immediately via wire transfer to avoid penalty"
    result = _run(state)
    assert "urgency_language" in _signal_types(result)


def test_round_amount(patch_db):
    state = _base_state({"total_amount": 5000.0})
    result = _run(state)
    assert "round_amount" in _signal_types(result)


def test_non_round_amount_no_signal(patch_db):
    state = _base_state({"total_amount": 1250.0})
    result = _run(state)
    assert "round_amount" not in _signal_types(result)


def test_missing_due_date(patch_db):
    state = _base_state({"due_date": None})
    result = _run(state)
    assert "missing_due_date" in _signal_types(result)


def test_unknown_items(patch_db):
    state = _base_state(
        {
            "line_items": [
                {
                    "item_name": "SuperGizmo9000",
                    "quantity": 1.0,
                    "unit_price": 99.0,
                    "line_total": 99.0,
                }
            ],
            "total_amount": 99.0,
        }
    )
    result = _run(state)
    assert "unknown_items" in _signal_types(result)


def test_zero_stock_item(patch_db):
    state = _base_state(
        {
            "line_items": [
                {
                    "item_name": "FakeItem",
                    "quantity": 1.0,
                    "unit_price": 1000.0,
                    "line_total": 1000.0,
                }
            ],
            "total_amount": 1000.0,
        }
    )
    result = _run(state)
    assert "zero_stock_item" in _signal_types(result)


def test_negative_quantity(patch_db):
    state = _base_state(
        {
            "line_items": [
                {
                    "item_name": "WidgetA",
                    "quantity": -3.0,
                    "unit_price": 250.0,
                    "line_total": -750.0,
                }
            ],
            "total_amount": 750.0,
        }
    )
    result = _run(state)
    assert "negative_quantity" in _signal_types(result)


def test_price_variance(patch_db):
    """WidgetA at $500 vs catalog $250 = 100% variance."""
    state = _base_state(
        {
            "line_items": [
                {"item_name": "WidgetA", "quantity": 2.0, "unit_price": 500.0, "line_total": 1000.0}
            ],
            "total_amount": 1000.0,
        }
    )
    result = _run(state)
    assert "price_variance" in _signal_types(result)


def test_math_error(patch_db):
    """Stated total doesn't match line items."""
    state = _base_state({"total_amount": 9999.0})
    result = _run(state)
    assert "math_error" in _signal_types(result)


def test_threshold_manipulation(patch_db):
    state = _base_state({"total_amount": 9500.0})
    result = _run(state)
    assert "threshold_manipulation" in _signal_types(result)


def test_below_threshold_no_signal(patch_db):
    state = _base_state({"total_amount": 8999.0})
    result = _run(state)
    assert "threshold_manipulation" not in _signal_types(result)


def test_future_invoice_date(patch_db):
    state = _base_state({"invoice_date": "2099-01-01"})
    result = _run(state)
    assert "future_invoice_date" in _signal_types(result)


def test_suspicious_vendor_name(patch_db):
    state = _base_state({"vendor_name": "Fraudster LLC"})
    result = _run(state)
    assert "suspicious_vendor_name" in _signal_types(result)


def test_risk_score_capped_at_100(patch_db):
    """Trigger many signals — score should never exceed 100."""
    state = _base_state(
        {
            "vendor_name": "Fraudster Fake Test LLC",
            "invoice_date": "2099-01-01",
            "due_date": None,
            "line_items": [
                {
                    "item_name": "SuperGizmo",
                    "quantity": -1.0,
                    "unit_price": 999.0,
                    "line_total": -999.0,
                }
            ],
            "total_amount": 9500.0,
            "notes": "URGENT pay now wire transfer penalty",
        }
    )
    state["raw_text"] = "URGENT pay now wire transfer penalty"
    result = _run(state)
    assert _score(result) == 100


def test_high_risk_recommends_block(patch_db):
    """Score >= 70 should recommend block."""
    state = _base_state(
        {
            "vendor_name": "Fraudster LLC",
            "due_date": None,
            "invoice_date": "2099-01-01",
            "total_amount": 9500.0,  # threshold manipulation + math error
            "notes": "URGENT wire transfer",
        }
    )
    state["raw_text"] = "URGENT wire transfer"
    result = _run(state)
    assert _score(result) >= 70
    assert result["fraud_result"]["recommendation"] == "block"


def test_medium_risk_recommends_review(patch_db):
    """Score 30-69 should recommend flag_for_review."""
    state = _base_state(
        {
            "vendor_name": "Totally Unknown Corp",
            "due_date": None,
        }
    )
    result = _run(state)
    score = _score(result)
    assert 30 <= score < 70
    assert result["fraud_result"]["recommendation"] == "flag_for_review"


def test_no_extracted_invoice_returns_zero(patch_db):
    state = {"extracted_invoice": None, "raw_text": "", "file_type": "txt"}
    result = fraud_detection_node(state)
    assert _score(result) == 0
    assert result["fraud_result"]["recommendation"] == "auto_approve"
