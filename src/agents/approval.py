"""Approval agent — routes invoices to auto-approve, auto-reject, or human review."""

import structlog
from langgraph.types import interrupt

from src.config import get_settings
from src.models.invoice import ApprovalDecision
from src.models.state import InvoiceState

logger = structlog.get_logger(__name__)


def approval_node(state: InvoiceState) -> dict:
    settings = get_settings()
    fraud = state.get("fraud_result") or {}
    inv = state.get("extracted_invoice") or {}

    risk_score = int(fraud.get("risk_score", 0))
    amount = float(inv.get("total_amount") or 0.0)
    invoice_number = inv.get("invoice_number", "UNKNOWN")
    recommendation = fraud.get("recommendation", "auto_approve")

    # validation failures: reject critical, escalate reviewable
    validation = state.get("validation_result") or {}
    if not validation.get("is_valid", True):
        val_issues = validation.get("issues", [])
        stock_checks = validation.get("stock_checks", {})

        critical_patterns = (
            "Required field missing", "quantity must be > 0",
            "total_amount must be > 0", "total_amount is not a valid number",
            "not approved", "elevated risk tier",
        )
        has_critical = any(any(p in issue for p in critical_patterns) for issue in val_issues)

        # all items out of stock = unfillable
        if stock_checks:
            overage = sum(1 for v in stock_checks.values() if not v.get("sufficient", True))
            if overage == len(stock_checks) and overage > 0:
                has_critical = True

        # all items unknown = critical
        unknown_issues = [i for i in val_issues if "not found in inventory" in i]
        if unknown_issues and len(stock_checks) == 0:
            has_critical = True

        if has_critical or risk_score >= settings.high_risk_threshold:
            status, action = "rejected", "auto_reject"
            prefix = "Validation failed (critical)"
        else:
            status, action = "escalated", "auto_escalate"
            prefix = "Flagged for review (validation concerns)"

        decision = ApprovalDecision(
            status=status,
            reasoning=f"{prefix}: {'; '.join(val_issues)}. Risk score: {risk_score}/100.",
            approver="system",
        )
        logger.info(f"approval.{action}", invoice=invoice_number, issues=len(val_issues), risk=risk_score)
        return {
            "approval_decision": decision.model_dump(),
            "current_agent": "approval",
            "audit_trail": [{"agent": "approval", "action": action,
                             "details": f"Validation failed ({len(val_issues)} issue(s)), risk {risk_score}/100"}],
        }

    # auto-reject: high risk
    if risk_score >= settings.high_risk_threshold:
        logger.warning("approval.auto_reject", invoice=invoice_number, risk=risk_score)
        decision = ApprovalDecision(
            status="rejected",
            reasoning=f"Auto-rejected: risk score {risk_score}/100 >= threshold ({settings.high_risk_threshold}).",
            approver="system",
        )
        return {
            "approval_decision": decision.model_dump(),
            "current_agent": "approval",
            "audit_trail": [{"agent": "approval", "action": "auto_reject",
                             "details": f"Risk {risk_score}/100 >= {settings.high_risk_threshold}"}],
        }

    # auto-approve: low amount AND low risk
    if amount < settings.auto_approve_threshold and risk_score < settings.medium_risk_threshold:
        decision = ApprovalDecision(
            status="approved",
            reasoning=(f"Auto-approved: amount ${amount:,.2f} below ${settings.auto_approve_threshold:,.0f} "
                       f"and risk {risk_score}/100 below {settings.medium_risk_threshold}."),
            approver="auto",
        )
        return {
            "approval_decision": decision.model_dump(),
            "current_agent": "approval",
            "audit_trail": [{"agent": "approval", "action": "auto_approve",
                             "details": f"${amount:,.2f}, risk {risk_score}/100"}],
        }

    # human review — interrupt() pauses the graph
    signals = fraud.get("signals", [])
    sig_desc = [s.get("description", "") for s in signals if isinstance(s, dict)]

    # TODO: make this configurable?
    label_map = {"auto_approve": "approve", "flag_for_review": "review", "block": "reject"}
    rec_label = label_map.get(recommendation, recommendation)

    review_ctx = {
        "invoice": inv, "validation": state.get("validation_result") or {},
        "fraud": fraud, "amount": amount, "risk_score": risk_score,
        "recommendation": rec_label, "fraud_signals": sig_desc,
        "fraud_narrative": fraud.get("narrative", ""),
    }

    logger.info("approval.needs_review", invoice=invoice_number, amount=amount, risk=risk_score)
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
