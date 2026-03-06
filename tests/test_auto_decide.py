"""Tests for the _auto_decide logic used in batch/auto-approve HITL handling.

Session 3 fix: auto-decide must consider validation warnings, not just risk score.
"""

import os
os.environ.setdefault("XAI_API_KEY", "fake-test-key-for-unit-tests")

import pytest

from main import _auto_decide


def _state(risk_score=0, warnings=None, issues=None):
    """Build a minimal state dict for _auto_decide."""
    return {
        "fraud_result": {
            "risk_score": risk_score,
            "recommendation": "auto_approve",
            "signals": [],
            "narrative": "",
        },
        "validation_result": {
            "is_valid": True,
            "issues": issues or [],
            "warnings": warnings or [],
            "stock_checks": {},
        },
    }


# --- High risk -> reject ---

def test_high_risk_rejects():
    decision, reason = _auto_decide(_state(risk_score=75))
    assert decision == "rejected"
    assert "75" in reason


def test_exactly_high_threshold_rejects():
    decision, _ = _auto_decide(_state(risk_score=70))
    assert decision == "rejected"


# --- Medium risk -> escalate ---

def test_medium_risk_escalates():
    decision, _ = _auto_decide(_state(risk_score=45))
    assert decision == "escalated"


def test_exactly_medium_threshold_escalates():
    """risk == 30 (medium_risk_threshold) should escalate, not reject."""
    decision, _ = _auto_decide(_state(risk_score=30))
    assert decision == "escalated"


# --- Substantive warnings -> escalated ---

def test_price_variance_warning_escalates():
    state = _state(risk_score=10, warnings=[
        "Price variance for 'WidgetA': catalog $250.00, invoice $300.00 (20.0%)"
    ])
    decision, reason = _auto_decide(state)
    assert decision == "escalated"
    assert "warning" in reason.lower()


def test_duplicate_warning_escalates():
    state = _state(risk_score=5, warnings=[
        "Duplicate invoice: 'INV-1004' was already processed"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_currency_warning_escalates():
    state = _state(risk_score=5, warnings=[
        "Non-USD currency: 'EUR' (expected USD)"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_unknown_item_warning_escalates():
    state = _state(risk_score=5, warnings=[
        "Item 'WidgetC' not found in inventory"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_ocr_artifact_warning_escalates():
    state = _state(risk_score=5, warnings=[
        "OCR artifacts detected in invoice text"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


# --- Past-due warnings should be ignored ---

def test_past_due_warning_alone_approves():
    """Past-due warnings are normal in AP workflows, should not prevent approval."""
    state = _state(risk_score=10, warnings=[
        "Invoice is past due (due date: 2026-01-15)"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "approved"


def test_overdue_warning_alone_approves():
    state = _state(risk_score=5, warnings=[
        "Invoice is overdue by 30 days"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "approved"


def test_past_the_due_date_warning_alone_approves():
    state = _state(risk_score=5, warnings=[
        "This invoice is past the due date"
    ])
    decision, _ = _auto_decide(state)
    assert decision == "approved"


# --- Mixed past-due + substantive -> escalated ---

def test_past_due_with_substantive_warning_escalates():
    """If there are both past-due and real warnings, escalate for the real ones."""
    state = _state(risk_score=10, warnings=[
        "Invoice is past due (due date: 2026-01-15)",
        "Price variance for 'WidgetA': catalog $250.00, invoice $300.00 (20.0%)",
    ])
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


# --- Clean invoice -> approved ---

def test_clean_invoice_approves():
    decision, reason = _auto_decide(_state(risk_score=10))
    assert decision == "approved"
    assert "approved" in reason.lower()


def test_zero_risk_no_warnings_approves():
    decision, _ = _auto_decide(_state(risk_score=0))
    assert decision == "approved"


# --- Edge cases ---

def test_missing_validation_result_approves_if_low_risk():
    """If validation_result is None/missing, don't crash — just use risk score."""
    state = {
        "fraud_result": {"risk_score": 5},
        "validation_result": None,
    }
    decision, _ = _auto_decide(state)
    assert decision == "approved"


def test_missing_fraud_result_approves():
    """If fraud_result is None/missing, treat risk as 0."""
    state = {
        "fraud_result": None,
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "approved"
