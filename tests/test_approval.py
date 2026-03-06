"""Tests for the approval agent — thresholds, severity classification, human review."""

from unittest.mock import patch

import pytest

from src.agents.approval import approval_node
from main import _auto_decide


def _base_state(**overrides):
    """Quick state builder for approval tests."""
    state = {
        "file_path": "test.txt",
        "raw_text": "INVOICE",
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": "INV-TEST",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 2.0,
                            "unit_price": 250.0, "line_total": 500.0}],
            "total_amount": 500.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0,
        "extraction_feedback": "",
        "validation_result": {"is_valid": True, "issues": [], "warnings": [], "stock_checks": {}},
        "fraud_result": {
            "risk_score": 10,
            "recommendation": "auto_approve",
            "signals": [],
            "narrative": "Looks fine.",
        },
        "approval_decision": None,
        "payment_result": None,
        "audit_trail": [],
        "error_message": None,
        "current_agent": "approval",
        "decision_explanation": "",
    }
    state.update(overrides)
    return state


# --- auto-approve path ---

def test_auto_approve_low_amount_low_risk():
    """$500 invoice with risk 10 -> auto-approved."""
    state = _base_state()
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "auto"


def test_auto_approve_just_under_threshold():
    """$999.99 should still auto-approve if risk is low."""
    state = _base_state()
    state["extracted_invoice"]["total_amount"] = 999.99
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"


def test_auto_approve_5k_under_10k_threshold():
    """$5,000 invoice with low risk should auto-approve (threshold is $10K)."""
    state = _base_state()
    state["extracted_invoice"]["total_amount"] = 5000.0
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "auto"


def test_auto_approve_9999_under_10k_threshold():
    """$9,999.99 should auto-approve when threshold is $10K."""
    state = _base_state()
    state["extracted_invoice"]["total_amount"] = 9999.99
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"


# --- auto-reject via high risk ---

def test_auto_reject_high_risk():
    state = _base_state(fraud_result={
        "risk_score": 85,
        "recommendation": "block",
        "signals": [{"signal_type": "urgency_language", "weight": 15}],
        "narrative": "Suspicious.",
    })
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"
    assert result["approval_decision"]["approver"] == "system"
    assert "85" in result["approval_decision"]["reasoning"]


def test_auto_reject_at_exactly_threshold():
    """risk_score == 70 should trigger reject (>= threshold)."""
    state = _base_state(fraud_result={
        "risk_score": 70,
        "recommendation": "block",
        "signals": [],
        "narrative": "",
    })
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


# --- validation failure: critical vs reviewable ---

def test_critical_validation_failure_rejects():
    """Missing required field is critical -> auto-reject regardless of risk."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["Required field missing: vendor_name"],
            "warnings": [],
            "stock_checks": {},
        },
        fraud_result={"risk_score": 15, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"
    assert result["approval_decision"]["approver"] == "system"


def test_negative_quantity_is_critical():
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["WidgetA: quantity must be > 0 (got -5.0)"],
            "warnings": [],
            "stock_checks": {},
        },
        fraud_result={"risk_score": 5, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_any_stock_violation_rejects():
    """ANY known item exceeding stock is a hard reject, even if other items are fine."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["Insufficient stock for 'WidgetA': requested 20, available 15"],
            "warnings": [],
            "stock_checks": {
                "WidgetA": {"requested": 20, "available": 15, "sufficient": False},
                "WidgetB": {"requested": 2, "available": 10, "sufficient": True},
            },
        },
        fraud_result={"risk_score": 10, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_single_item_stock_violation_rejects():
    """Single item exceeding stock -> reject."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["Insufficient stock for 'WidgetA': requested 100, available 15"],
            "warnings": [],
            "stock_checks": {
                "WidgetA": {"requested": 100, "available": 15, "sufficient": False},
            },
        },
        fraud_result={"risk_score": 10, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_all_items_unknown_is_critical():
    """Every item not found in inventory (available=None) -> critical reject."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["'SuperGizmo' not found in inventory", "'MegaSprocket' not found in inventory"],
            "warnings": [],
            "stock_checks": {
                "SuperGizmo": {"requested": 5, "available": None, "sufficient": False},
                "MegaSprocket": {"requested": 3, "available": None, "sufficient": False},
            },
        },
        fraud_result={"risk_score": 10, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_all_items_unknown_empty_stock_checks():
    """INV-1008 scenario: both items unknown, stock_checks={} (empty) -> rejected.

    Validation skips unknown items in _check_aggregate_stock, so stock_checks
    is empty even though issues list says items aren't found.
    """
    state = _base_state(
        extracted_invoice={
            "invoice_number": "INV-1008",
            "vendor_name": "NoProd Industries",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [
                {"item_name": "SuperGizmo", "quantity": 5.0,
                 "unit_price": 500.0, "line_total": 2500.0},
                {"item_name": "MegaSprocket", "quantity": 3.0,
                 "unit_price": 2000.0, "line_total": 6000.0},
            ],
            "total_amount": 9900.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        validation_result={
            "is_valid": False,
            "issues": [
                "Item 'SuperGizmo' not found in inventory (unknown item)",
                "Item 'MegaSprocket' not found in inventory (unknown item)",
            ],
            "warnings": ["Vendor 'NoProd Industries' not found in approved vendor list"],
            "stock_checks": {},  # empty — unknown items skipped by validation
        },
        fraud_result={"risk_score": 55, "recommendation": "flag_for_review",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_unknown_item_with_valid_items_escalates():
    """INV-1016 scenario: WidgetC unknown but WidgetA/B valid -> escalate, not reject."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["Item 'WidgetC' not found in inventory (unknown item)"],
            "warnings": [],
            "stock_checks": {
                "WidgetA": {"requested": 5, "available": 15, "sufficient": True},
                "WidgetB": {"requested": 3, "available": 10, "sufficient": True},
            },
        },
        fraud_result={"risk_score": 10, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "escalated", "reasoning": "Unknown item needs review"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "escalated"


def test_validation_failure_with_high_risk_always_rejects():
    """Even reviewable issues should reject if risk score is high."""
    state = _base_state(
        validation_result={
            "is_valid": False,
            "issues": ["Item 'WidgetC' not found in inventory (unknown item)"],
            "warnings": [],
            "stock_checks": {
                "WidgetA": {"requested": 5, "available": 15, "sufficient": True},
            },
        },
        fraud_result={"risk_score": 75, "recommendation": "block",
                       "signals": [], "narrative": ""},
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


# --- concerning warnings -> escalate ---

def test_price_variance_warning_escalates():
    """Price variance warning should escalate for review."""
    state = _base_state(
        validation_result={
            "is_valid": True,
            "issues": [],
            "warnings": ["Price variance for 'WidgetA': catalog $250.00, invoice: $300.00 (20.0%)"],
            "stock_checks": {"WidgetA": {"requested": 5, "available": 15, "sufficient": True}},
        },
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "escalated", "reasoning": "Price variance"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "escalated"


def test_currency_warning_escalates():
    """Non-USD currency warning should escalate for review."""
    state = _base_state(
        validation_result={
            "is_valid": True,
            "issues": [],
            "warnings": ["Non-USD currency: 'EUR' (expected USD)"],
            "stock_checks": {},
        },
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "escalated", "reasoning": "EUR currency"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "escalated"


def test_duplicate_warning_escalates():
    """Duplicate invoice warning should escalate for review."""
    state = _base_state(
        validation_result={
            "is_valid": True,
            "issues": [],
            "warnings": ["Duplicate invoice: 'INV-1004' was already processed on 2026-01-15 (status: approved)"],
            "stock_checks": {},
        },
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "escalated", "reasoning": "Duplicate"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "escalated"


def test_past_due_warning_does_not_escalate():
    """Past-due warning alone should NOT prevent auto-approve."""
    state = _base_state(
        validation_result={
            "is_valid": True,
            "issues": [],
            "warnings": ["Invoice is past due (due date: 2026-01-15)"],
            "stock_checks": {},
        },
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"


# --- human review (interrupt) ---

def test_human_review_for_medium_risk_high_amount():
    """$5000 invoice with risk 45 -> needs human review (interrupt)."""
    state = _base_state(
        extracted_invoice={
            "invoice_number": "INV-HITL",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 20.0,
                            "unit_price": 250.0, "line_total": 5000.0}],
            "total_amount": 5000.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        fraud_result={"risk_score": 45, "recommendation": "flag_for_review",
                       "signals": [], "narrative": "Medium risk."},
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "approved", "reasoning": "Looks legit"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "human"
    assert "Looks legit" in result["approval_decision"]["reasoning"]


def test_human_review_reject():
    state = _base_state(
        extracted_invoice={
            "invoice_number": "INV-HITL-2",
            "vendor_name": "Shady Corp.",
            "invoice_date": "2026-01-15",
            "due_date": None,
            "line_items": [{"item_name": "WidgetB", "quantity": 10.0,
                            "unit_price": 500.0, "line_total": 5000.0}],
            "total_amount": 5000.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        fraud_result={"risk_score": 50, "recommendation": "flag_for_review",
                       "signals": [], "narrative": ""},
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "rejected", "reasoning": "Not a real vendor"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"
    assert result["approval_decision"]["approver"] == "human"


def test_human_review_invalid_decision_defaults_to_rejected():
    """If human sends garbage, we default to rejected for safety."""
    state = _base_state(
        extracted_invoice={
            "invoice_number": "INV-BAD-INPUT",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 5.0,
                            "unit_price": 250.0, "line_total": 1250.0}],
            "total_amount": 1250.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        fraud_result={"risk_score": 35, "recommendation": "flag_for_review",
                       "signals": [], "narrative": ""},
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "yolo", "reasoning": "idk"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


# --- high amount triggers HITL ---

def test_high_amount_low_risk_triggers_hitl():
    """$15K invoice with risk 10 -> HITL (above auto-approve threshold)."""
    state = _base_state(
        extracted_invoice={
            "invoice_number": "INV-BIGAMT",
            "vendor_name": "Widgets Inc.",
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [{"item_name": "WidgetA", "quantity": 10.0,
                            "unit_price": 1500.0, "line_total": 15000.0}],
            "total_amount": 15000.0,
            "currency": "USD",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        fraud_result={"risk_score": 10, "recommendation": "auto_approve",
                       "signals": [], "narrative": ""},
    )
    with patch("src.agents.approval.interrupt",
               return_value={"decision": "approved", "reasoning": "Big but clean"}):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "human"


# --- audit trail ---

def test_audit_trail_present():
    result = approval_node(_base_state())
    trail = result["audit_trail"]
    assert len(trail) == 1
    assert trail[0]["agent"] == "approval"


def test_audit_trail_action_matches_decision():
    """auto_approve action in trail when auto-approved."""
    result = approval_node(_base_state())
    assert result["audit_trail"][0]["action"] == "auto_approve"

    # and auto_reject when risk is high
    state = _base_state(fraud_result={
        "risk_score": 80, "recommendation": "block",
        "signals": [], "narrative": "",
    })
    result = approval_node(state)
    assert result["audit_trail"][0]["action"] == "auto_reject"


# --- _auto_decide tests (batch/auto-approve mode) ---

def test_auto_decide_medium_risk_escalates():
    """Medium risk (30-69) should return 'escalated', not 'rejected'."""
    state = {
        "fraud_result": {"risk_score": 35},
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_auto_decide_high_risk_rejects():
    """High risk (>=70) should return 'rejected'."""
    state = {
        "fraud_result": {"risk_score": 80},
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "rejected"


def test_auto_decide_low_risk_with_warnings_escalates():
    """Low risk with substantive warnings should return 'escalated'."""
    state = {
        "fraud_result": {"risk_score": 10},
        "validation_result": {"warnings": ["Price variance: $300 vs catalog $250"]},
    }
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_auto_decide_clean_approves():
    """Low risk, no warnings should return 'approved'."""
    state = {
        "fraud_result": {"risk_score": 5},
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "approved"


def test_auto_decide_past_due_only_approves():
    """Past-due-only warnings are ignorable — should approve."""
    state = {
        "fraud_result": {"risk_score": 5},
        "validation_result": {"warnings": ["Invoice is past due"]},
    }
    decision, _ = _auto_decide(state)
    assert decision == "approved"


def test_auto_decide_at_medium_threshold_boundary():
    """risk_score == 30 (exactly medium threshold) should escalate."""
    state = {
        "fraud_result": {"risk_score": 30},
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "escalated"


def test_auto_decide_at_high_threshold_boundary():
    """risk_score == 70 (exactly high threshold) should reject."""
    state = {
        "fraud_result": {"risk_score": 70},
        "validation_result": {"warnings": []},
    }
    decision, _ = _auto_decide(state)
    assert decision == "rejected"
