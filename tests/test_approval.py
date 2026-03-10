"""Tests for the approval agent's decision routing."""

from unittest.mock import patch

from src.agents.approval import _build_reflection, approval_node


def _base_state(overrides=None):
    """Build a state dict for testing approval routing."""
    state = {
        "extracted_invoice": {
            "invoice_number": "INV-APPROVE-001",
            "vendor_name": "Widgets Inc.",
            "total_amount": 500.0,
            "line_items": [{"item_name": "WidgetA", "quantity": 5.0, "unit_price": 100.0}],
        },
        "validation_result": {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stock_checks": {
                "WidgetA": {"requested": 5, "available": 15, "sufficient": True},
            },
        },
        "fraud_result": {
            "risk_score": 10,
            "recommendation": "auto_approve",
            "signals": [],
            "narrative": "Low risk.",
        },
        "approval_decision": None,
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(state.get(key), dict):
                state[key] = {**state[key], **val}
            else:
                state[key] = val
    return state


def test_auto_approve_clean_low_amount():
    """Clean invoice under $10K with low risk -> auto approve."""
    result = approval_node(_base_state())
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "auto"


def test_reject_stock_violation():
    """Stock violation -> auto reject."""
    state = _base_state(
        {
            "validation_result": {
                "is_valid": False,
                "issues": ["Insufficient stock for 'WidgetA': requested 20, available 15"],
                "warnings": [],
                "stock_checks": {
                    "WidgetA": {"requested": 20, "available": 15, "sufficient": False},
                },
            },
        }
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"
    assert result["approval_decision"]["approver"] == "system"


def test_reject_critical_issue():
    """Critical validation issue like missing required field -> reject."""
    state = _base_state(
        {
            "validation_result": {
                "is_valid": False,
                "issues": ["Required field missing: vendor_name"],
                "warnings": [],
                "stock_checks": {},
            },
        }
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_reject_high_risk_score():
    """Risk score >= 70 -> auto reject."""
    state = _base_state(
        {
            "fraud_result": {
                "risk_score": 75,
                "recommendation": "block",
                "signals": [],
                "narrative": "High risk.",
            },
        }
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_reject_all_items_unknown():
    """All items unknown and no stock checks -> reject."""
    state = _base_state(
        {
            "extracted_invoice": {
                "invoice_number": "INV-UNK-001",
                "vendor_name": "Widgets Inc.",
                "total_amount": 500.0,
                "line_items": [
                    {"item_name": "MysteryWidget", "quantity": 1.0, "unit_price": 500.0}
                ],
            },
            "validation_result": {
                "is_valid": False,
                "issues": ["Item 'MysteryWidget' not found in inventory"],
                "warnings": [],
                "stock_checks": {},
            },
        }
    )
    result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"


def test_escalate_invalid_with_fixable_issues():
    """Invalid but not critical -> escalate for review (HITL interrupt)."""
    state = _base_state(
        {
            "validation_result": {
                "is_valid": False,
                "issues": ["Math mismatch: calculated $450 differs from stated $500"],
                "warnings": [],
                "stock_checks": {"WidgetA": {"requested": 5, "available": 15, "sufficient": True}},
            },
        }
    )
    # The approval node calls interrupt() for escalation, so we mock it
    with patch(
        "src.agents.approval.interrupt",
        return_value={"decision": "approved", "reasoning": "Looks OK"},
    ):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "approved"
    assert result["approval_decision"]["approver"] == "human"


def test_escalate_concerning_warnings():
    """Warnings like price variance -> escalate."""
    state = _base_state(
        {
            "validation_result": {
                "is_valid": True,
                "issues": [],
                "warnings": ["Price variance: WidgetA billed at $350 vs catalog $250"],
                "stock_checks": {"WidgetA": {"requested": 5, "available": 15, "sufficient": True}},
            },
        }
    )
    with patch(
        "src.agents.approval.interrupt",
        return_value={"decision": "rejected", "reasoning": "Too expensive"},
    ):
        result = approval_node(state)
    assert result["approval_decision"]["status"] == "rejected"
    assert result["approval_decision"]["approver"] == "human"


def test_escalate_high_amount():
    """Amount >= $10K with low risk -> needs human review."""
    state = _base_state(
        {
            "extracted_invoice": {
                "invoice_number": "INV-BIG-001",
                "vendor_name": "Widgets Inc.",
                "total_amount": 15000.0,
                "line_items": [{"item_name": "WidgetA", "quantity": 5.0, "unit_price": 3000.0}],
            },
        }
    )
    with patch(
        "src.agents.approval.interrupt",
        return_value={"decision": "approved", "reasoning": "Budget OK"},
    ):
        result = approval_node(state)
    assert result["approval_decision"]["approver"] == "human"


def test_escalate_medium_risk():
    """Risk 30-69 with amount under threshold -> still escalates."""
    state = _base_state(
        {
            "fraud_result": {
                "risk_score": 45,
                "recommendation": "flag_for_review",
                "signals": [],
                "narrative": "Medium risk.",
            },
        }
    )
    with patch(
        "src.agents.approval.interrupt",
        return_value={"decision": "approved", "reasoning": "Reviewed"},
    ):
        result = approval_node(state)
    assert result["approval_decision"]["approver"] == "human"


# --- Reflection / critique loop tests ---


def test_reflection_returns_string():
    """_build_reflection calls Grok and returns a non-empty critique."""
    inv = {"invoice_number": "INV-REF-001", "vendor_name": "Widgets Inc.", "total_amount": 5000.0}
    fraud = {"risk_score": 35, "signals": [{"severity": "medium", "description": "Round amount"}]}
    validation = {"warnings": ["Price variance"], "issues": []}
    with patch("src.agents.approval.assess", return_value="This invoice looks borderline."):
        result = _build_reflection(inv, fraud, validation, "Medium risk requires review")
    assert isinstance(result, str)
    assert len(result) > 0


def test_reflection_graceful_on_llm_failure():
    """If Grok fails, reflection returns empty string instead of crashing."""
    inv = {"invoice_number": "INV-FAIL-001", "vendor_name": "X", "total_amount": 100.0}
    with patch("src.agents.approval.assess", side_effect=Exception("API down")):
        result = _build_reflection(inv, {}, {}, "test reason")
    assert result == ""


def test_escalation_includes_reflection():
    """When escalated, the review context should contain the reflection field."""
    state = _base_state(
        {
            "validation_result": {
                "is_valid": True,
                "issues": [],
                "warnings": ["Price variance: WidgetA $350 vs catalog $250"],
                "stock_checks": {"WidgetA": {"requested": 5, "available": 15, "sufficient": True}},
            },
        }
    )

    captured_ctx = {}

    def mock_interrupt(ctx):
        captured_ctx.update(ctx)
        return {"decision": "approved", "reasoning": "Reviewed"}

    with patch("src.agents.approval.interrupt", side_effect=mock_interrupt):
        with patch(
            "src.agents.approval.assess", return_value="Consider approving — variance is minor."
        ):
            approval_node(state)

    assert "reflection" in captured_ctx
    assert "Consider approving" in captured_ctx["reflection"]
