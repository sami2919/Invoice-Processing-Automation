"""Approval agent: routes invoices to auto approve, auto reject, or human review."""

import structlog
from langgraph.types import interrupt

from src.config import get_settings
from src.llm.grok_client import assess
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


def _build_risk_summary(inv: dict, fraud: dict, validation: dict) -> dict:
    risk_score = int(fraud.get("risk_score", 0))
    amount = float(inv.get("total_amount") or 0.0)
    invoice_number = inv.get("invoice_number", "UNKNOWN")

    is_valid = validation.get("is_valid", True)
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])
    stock_checks = validation.get("stock_checks", {})

    # Any KNOWN item exceeding stock is a hard reject
    has_stock_violation = any(
        v["requested"] > v["available"] for v in stock_checks.values()
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
        any(pattern.lower() in issue.lower() for pattern in critical_patterns) for issue in issues
    )

    # All items unknown (nothing validates at all)
    # stock_checks may be empty {} when validation skips unknown items entirely
    unknown_item_issues = [i for i in issues if "not found in inventory" in i.lower()]
    line_items = inv.get("line_items") or []
    all_items_unknown = (
        not stock_checks and unknown_item_issues and len(unknown_item_issues) >= len(line_items)
    ) or (stock_checks and all(v.get("available") is None for v in stock_checks.values()))

    concerning_warning_patterns = (
        "price variance",
        "price differs",
        "price mismatch",
        "math",
        "total mismatch",
        "calculated total",
        "currency",
        "non-USD",
        "EUR",
        "duplicate",
        "previously processed",
        "unknown item",
        "not found in inventory",
        "OCR",
        "artifact",
    )
    concerning_warnings = [
        w for w in warnings if any(p.lower() in w.lower() for p in concerning_warning_patterns)
    ]

    return {
        "risk_score": risk_score,
        "amount": amount,
        "invoice_number": invoice_number,
        "is_valid": is_valid,
        "issues": issues,
        "warnings": warnings,
        "has_stock_violation": has_stock_violation,
        "has_critical_issue": has_critical_issue,
        "all_items_unknown": all_items_unknown,
        "concerning_warnings": concerning_warnings,
    }


def _determine_approval_decision(state: dict, summary: dict, settings) -> dict:
    invoice_number = summary["invoice_number"]
    risk_score = summary["risk_score"]
    amount = summary["amount"]

    if summary["has_stock_violation"] or summary["has_critical_issue"] or summary["all_items_unknown"]:
        reason = (
            f"Critical validation failure: stock_violation={summary['has_stock_violation']}, "
            f"critical_issue={summary['has_critical_issue']}, all_unknown={summary['all_items_unknown']}"
        )
        logger.info("approval.auto_reject", invoice=invoice_number, reason=reason)
        return _make_decision("rejected", "system", reason, "auto_reject")

    if risk_score >= settings.high_risk_threshold:
        reason = f"High fraud risk score: {risk_score}"
        logger.warning("approval.auto_reject", invoice=invoice_number, risk=risk_score)
        return _make_decision("rejected", "system", reason, "auto_reject")

    if not summary["is_valid"]:
        reason = f"Validation issues require review: {'; '.join(summary['issues'][:3])}"
        logger.info("approval.escalate", invoice=invoice_number, issues=len(summary["issues"]))
        return _escalate_for_review(state, reason, invoice_number)

    if summary["concerning_warnings"]:
        reason = f"Warnings need review: {'; '.join(summary['concerning_warnings'][:3])}"
        logger.info("approval.escalate", invoice=invoice_number, warnings=len(summary["concerning_warnings"]))
        return _escalate_for_review(state, reason, invoice_number)

    if amount < settings.auto_approve_threshold and risk_score < settings.medium_risk_threshold:
        reason = f"Clean invoice: amount=${amount:,.2f}, risk={risk_score}"
        logger.info("approval.auto_approve", invoice=invoice_number, amount=amount, risk=risk_score)
        return _make_decision("approved", "auto", reason, "auto_approve")

    reason = f"Amount=${amount:,.2f} or risk={risk_score} requires human review"
    logger.info("approval.needs_review", invoice=invoice_number, amount=amount, risk=risk_score)
    return _escalate_for_review(state, reason, invoice_number)


def approval_node(state: InvoiceState) -> dict:
    settings = get_settings()
    fraud = state.get("fraud_result") or {}
    inv = state.get("extracted_invoice") or {}
    validation = state.get("validation_result") or {}
    summary = _build_risk_summary(inv, fraud, validation)
    return _determine_approval_decision(state, summary, settings)


def _build_reflection(inv: dict, fraud: dict, validation: dict, reason: str) -> str:
    """Generate a devil's advocate counter argument for the human reviewer."""
    signals = fraud.get("signals", [])
    sig_lines = (
        "\n".join(
            f"- [{s.get('severity', '?').upper()}] {s.get('description', '')}"
            for s in signals
            if isinstance(s, dict)
        )
        or "None"
    )
    warnings = validation.get("warnings", [])
    issues = validation.get("issues", [])

    prompt = (
        "You are a senior financial auditor acting as devil's advocate.\n\n"
        f"Invoice: {inv.get('invoice_number', 'UNKNOWN')}\n"
        f"Vendor: {inv.get('vendor_name', 'UNKNOWN')}\n"
        f"Amount: ${float(inv.get('total_amount') or 0):,.2f}\n"
        f"Risk score: {fraud.get('risk_score', 0)}/100\n\n"
        f"Escalation reason: {reason}\n\n"
        f"Fraud signals:\n{sig_lines}\n\n"
        f"Validation issues: {'; '.join(issues[:3]) or 'None'}\n"
        f"Validation warnings: {'; '.join(warnings[:3]) or 'None'}\n\n"
        "In 2-3 sentences, challenge this escalation. If the concerns are "
        "weak, argue why this invoice could safely be approved. If the "
        "concerns are strong, explain what specific risk would materialise "
        "if it were approved anyway. Be concrete and specific."
    )
    try:
        return assess(prompt, temperature=0.5)
    except Exception as exc:
        logger.warning("approval.reflection_failed", error=str(exc)[:200])
        return ""


def _escalate_for_review(state: dict, reason: str, invoice_number: str = "UNKNOWN") -> dict:
    """Interrupt the graph for human review with a devil's advocate reflection."""
    fraud = state.get("fraud_result") or {}
    inv = state.get("extracted_invoice") or {}
    validation = state.get("validation_result") or {}
    recommendation = fraud.get("recommendation", "auto_approve")

    signals = fraud.get("signals", [])
    sig_desc = [s.get("description", "") for s in signals if isinstance(s, dict)]

    label_map = {"auto_approve": "approve", "flag_for_review": "review", "block": "reject"}
    rec_label = label_map.get(recommendation, recommendation)

    # Grok reflection: gives the human reviewer a second opinion
    reflection = _build_reflection(inv, fraud, validation, reason)

    review_ctx = {
        "invoice": inv,
        "validation": validation,
        "fraud": fraud,
        "amount": float(inv.get("total_amount") or 0),
        "risk_score": int(fraud.get("risk_score", 0)),
        "recommendation": rec_label,
        "fraud_signals": sig_desc,
        "fraud_narrative": fraud.get("narrative", ""),
        "escalation_reason": reason,
        "reflection": reflection,
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
        "audit_trail": [
            {
                "agent": "approval",
                "action": "human_review",
                "details": f"Human decision: {human_decision} — {reasoning}",
            }
        ],
    }
