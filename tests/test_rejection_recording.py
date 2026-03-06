"""Tests for Session 4: invoice recording on rejection + duplicate detection."""

import json
import uuid
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from tests.conftest import _make_state


def test_rejection_node_records_invoice(patch_db):
    """Rejected invoices should be recorded in invoice_history so duplicate detection works."""
    from src.pipeline import rejection_node
    from src.tools.inventory_db import check_duplicate_invoice

    state = _make_state(
        invoice_number="INV-REJECT-001",
        vendor="Widgets Inc.",
        approval_decision={
            "status": "rejected",
            "reasoning": "Stock violation",
            "approver": "system",
        },
    )

    result = rejection_node(state)

    # Verify the rejection was recorded
    is_dup, info = check_duplicate_invoice("INV-REJECT-001")
    assert is_dup, "Rejected invoice should be recorded in invoice_history"
    assert info["status"] == "rejected"
    assert info["vendor"] == "Widgets Inc."

    # Verify audit trail mentions recording
    audit_entries = result.get("audit_trail", [])
    assert any("recorded" in e.get("action", "").lower() or "recorded" in e.get("details", "").lower()
               for e in audit_entries), "Audit trail should mention recording"


def test_rejection_node_preserves_existing_decision(patch_db):
    """rejection_node should keep the existing approval_decision if present."""
    from src.pipeline import rejection_node

    existing_decision = {
        "status": "rejected",
        "reasoning": "High fraud risk score: 85",
        "approver": "system",
    }
    state = _make_state(
        invoice_number="INV-REJECT-002",
        approval_decision=existing_decision,
    )

    result = rejection_node(state)

    # Should NOT overwrite existing decision
    assert "approval_decision" not in result or result.get("approval_decision") == existing_decision


def test_rejection_node_creates_fallback_decision(patch_db):
    """When no approval_decision exists, rejection_node should create a fallback."""
    from src.pipeline import rejection_node

    state = _make_state(
        invoice_number="INV-REJECT-003",
        approval_decision=None,
    )

    result = rejection_node(state)

    fallback = result.get("approval_decision")
    assert fallback is not None
    assert fallback["status"] == "rejected"


def test_rejection_recording_does_not_crash_on_failure(patch_db, monkeypatch):
    """If record_invoice fails, rejection_node should still complete."""
    from src.pipeline import rejection_node

    monkeypatch.setattr(
        "src.pipeline.record_invoice",
        MagicMock(side_effect=Exception("DB write failed")),
    )

    state = _make_state(
        invoice_number="INV-REJECT-004",
        approval_decision={"status": "rejected", "reasoning": "test", "approver": "system"},
    )

    # Should not raise
    result = rejection_node(state)
    assert result is not None


def test_duplicate_detected_after_rejection(patch_db, monkeypatch):
    """Process INV-1004 (rejected), then INV-1004 again — duplicate should be flagged."""
    from src.pipeline import rejection_node
    from src.tools.inventory_db import check_duplicate_invoice

    # First: simulate INV-1004 being rejected
    state1 = _make_state(
        invoice_number="INV-1004",
        vendor="Precision Parts Ltd.",
        approval_decision={"status": "rejected", "reasoning": "test", "approver": "system"},
    )
    state1["extracted_invoice"]["total_amount"] = 1890.0
    rejection_node(state1)

    # Now check: duplicate detection should find it
    is_dup, info = check_duplicate_invoice("INV-1004")
    assert is_dup, "INV-1004 should be found in invoice_history after rejection"
    assert info["status"] == "rejected"


def test_rejection_records_correct_amount(patch_db):
    """Verify the recorded amount matches the invoice total."""
    from src.pipeline import rejection_node
    from src.tools.inventory_db import check_duplicate_invoice

    state = _make_state(invoice_number="INV-AMT-001")
    state["extracted_invoice"]["total_amount"] = 15225.0
    state["approval_decision"] = {"status": "rejected", "reasoning": "stock", "approver": "system"}

    rejection_node(state)

    is_dup, info = check_duplicate_invoice("INV-AMT-001")
    assert is_dup
    assert float(info["amount"]) == 15225.0
