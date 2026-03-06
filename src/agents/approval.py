"""Approval agent — routes invoices to auto-approve, auto-reject, or human review."""

import structlog
from langgraph.types import interrupt

from src.config import get_settings
from src.models.invoice import ApprovalDecision
from src.models.state import InvoiceState

logger = structlog.get_logger(__name__)


def _make_decision(status: str, approver: str, reasoning: str, action: str) -> dict:
    decision = ApprovalDecision(status=status, reasoning=reasoning, approver=approver)
    return {
        "approval_decision": decision.model_dump(),
        "current_agent": "approval",
        "audit_trail": [{"agent": "approval", "action": action, "details": reasoning}],
    }


def approval_node(state: InvoiceState) -> dict:
    settings = get_settings()
    fraud = state.get("fraud_result") or {}
    inv = state.get("extracted_invoice") or {}
    validation = state.get("validation_result") or {}

    risk_score = int(fraud.get("risk_score", 0))
    amount = float(inv.get("total_amount") or 0.0)
    invoice_number = inv.get("invoice_number", "UNKNOWN")
    recommendation = fraud.get("recommendation", "auto_approve")

    is_valid = validation.get("is_valid", True)
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])
    stock_checks = validation.get("stock_checks", {})

    # --- STEP 1: Auto-reject on critical failures ---

    # Any KNOWN item exceeding stock is a hard reject
    has_stock_violation = any(
        v.get("requested", 0) > v.get("available", 0)
        for v in stock_checks.values()
        if v.get("available") is not None
    )

    # Critical issue patterns (always reject)
    critical_patterns = (
        "Required field missing",
        "quantity must be > 0",
        "total_amount must be > 0",
        "total_amount is not a valid number",
        "not approved",
        "elevated risk tier",
        "blocked risk tier",
        "Insufficient stock",
    )
    has_critical_issue = any(
        any(pattern.lower() in issue.lower() for pattern in critical_patterns)
        for issue in issues
    )

    # All items unknown (nothing validates at all)
    # stock_checks may be empty {} when validation skips unknown items entirely
    unknown_item_issues = [i for i in issues if "not found in inventory" in i.lower()]
    line_items = inv.get("line_items") or []
    all_items_unknown = (
        not stock_checks and unknown_item_issues and len(unknown_item_issues) >= len(line_items)
    ) or (
        stock_checks
        and all(v.get("available") is None for v in stock_checks.values())
    )

    if has_stock_violation or has_critical_issue or all_items_unknown:
        reason = (
            f"Critical validation failure: stock_violation={has_stock_violation}, "
            f"critical_issue={has_critical_issue}, all_unknown={all_items_unknown}"
        )
        logger.info("approval.auto_reject", invoice=invoice_number, reason=reason)
        return _make_decision("rejected", "system", reason, "auto_reject")

    # --- STEP 2: Auto-reject on high fraud risk ---
    if risk_score >= settings.high_risk_threshold:
        reason = f"High fraud risk score: {risk_score}"
        logger.warning("approval.auto_reject", invoice=invoice_number, risk=risk_score)
        return _make_decision("rejected", "system", reason, "auto_reject")

    # --- STEP 3: Escalate on non-critical validation failures ---
    if not is_valid:
        reason = f"Validation issues require review: {'; '.join(issues[:3])}"
        logger.info("approval.escalate", invoice=invoice_number, issues=len(issues))
        return _escalate_for_review(state, reason, invoice_number)

    # --- STEP 4: Escalate on concerning warnings ---
    concerning_warning_patterns = (
        "price variance", "price differs", "price mismatch",
        "math", "total mismatch", "calculated total",
        "currency", "non-USD", "EUR",
        "duplicate", "previously processed",
        "unknown item", "not found in inventory",
        "OCR", "artifact",
    )
    concerning_warnings = [
        w for w in warnings
        if any(p.lower() in w.lower() for p in concerning_warning_patterns)
    ]

    if concerning_warnings:
        reason = f"Warnings need review: {'; '.join(concerning_warnings[:3])}"
        logger.info("approval.escalate", invoice=invoice_number, warnings=len(concerning_warnings))
        return _escalate_for_review(state, reason, invoice_number)

    # --- STEP 5: Auto-approve if clean ---
    if amount < settings.auto_approve_threshold and risk_score < settings.medium_risk_threshold:
        reason = f"Clean invoice: amount=${amount:,.2f}, risk={risk_score}"
        logger.info("approval.auto_approve", invoice=invoice_number, amount=amount, risk=risk_score)
        return _make_decision("approved", "auto", reason, "auto_approve")

    # --- STEP 6: Everything else -> HITL ---
    reason = f"Amount=${amount:,.2f} or risk={risk_score} requires human review"
    logger.info("approval.needs_review", invoice=invoice_number, amount=amount, risk=risk_score)
    return _escalate_for_review(state, reason, invoice_number)


def _escalate_for_review(state: dict, reason: str, invoice_number: str = "UNKNOWN") -> dict:
    """Interrupt the graph for human review."""
    fraud = state.get("fraud_result") or {}
    inv = state.get("extracted_invoice") or {}
    recommendation = fraud.get("recommendation", "auto_approve")

    signals = fraud.get("signals", [])
    sig_desc = [s.get("description", "") for s in signals if isinstance(s, dict)]

    label_map = {"auto_approve": "approve", "flag_for_review": "review", "block": "reject"}
    rec_label = label_map.get(recommendation, recommendation)

    review_ctx = {
        "invoice": inv,
        "validation": state.get("validation_result") or {},
        "fraud": fraud,
        "amount": float(inv.get("total_amount") or 0),
        "risk_score": int(fraud.get("risk_score", 0)),
        "recommendation": rec_label,
        "fraud_signals": sig_desc,
        "fraud_narrative": fraud.get("narrative", ""),
        "escalation_reason": reason,
    }

    human_input = interrupt(review_ctx)

    valid = {"approved", "rejected", "escalated", "pending_human_review"}
    human_decision = human_input.get("decision", "rejected")
    if human_decision not in valid:
        human_decision = "rejected"

    reasoning = human_input.get("reasoning", "Human reviewer decision")
    decision = ApprovalDecision(status=human_decision, reasoning=reasoning, approver="human")

    return {
        "approval_decision": decision.model_dump(),
        "current_agent": "approval",
        "audit_trail": [{"agent": "approval", "action": "human_review",
                         "details": f"Human decision: {human_decision} — {reasoning}"}],
    }
