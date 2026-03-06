"""Explanation agent — generates a VP-readable summary of the processing decision."""

import structlog

from src.llm.grok_client import assess
from src.models.state import InvoiceState

logger = structlog.get_logger(__name__)

_INSTRUCTIONS = (
    "You are a concise financial analyst writing for a VP of Finance. "
    "Write exactly 3-4 sentences. Use plain business language — no technical jargon, "
    "no JSON field names, no score numbers unless they add clear business value. "
    "Focus on: what invoice was processed, any concerns, the decision, and why."
)


def _build_prompt(state: InvoiceState) -> str:
    inv = state.get("extracted_invoice") or {}
    validation = state.get("validation_result") or {}
    fraud = state.get("fraud_result") or {}
    approval = state.get("approval_decision") or {}
    payment = state.get("payment_result") or {}

    amount = inv.get("total_amount", 0.0)
    signals = fraud.get("signals", [])
    sig_desc = [s.get("description", "") for s in signals if isinstance(s, dict)]

    lines = [
        _INSTRUCTIONS, "",
        "=== INVOICE SUMMARY ===",
        f"Invoice: {inv.get('invoice_number', 'UNKNOWN')}",
        f"Vendor: {inv.get('vendor_name') or 'UNKNOWN'}",
        f"Amount: {inv.get('currency', 'USD')} {amount:,.2f}" if isinstance(amount, (int, float)) else f"Amount: {amount}",
        "",
        "=== DECISION ===",
        f"Status: {approval.get('status', 'unknown').upper()}",
        f"Decided by: {approval.get('approver', 'system')}",
        f"Reasoning: {approval.get('reasoning', '')}",
    ]

    txn = payment.get("transaction_id", "")
    if txn:
        lines += ["", f"Payment transaction: {txn}"]
    narrative = fraud.get("narrative", "")
    if narrative:
        lines += ["", "=== RISK ASSESSMENT ===", narrative]
    elif sig_desc:
        lines += ["", "=== RISK FLAGS ==="] + [f"- {d}" for d in sig_desc]
    for key, label in [("warnings", "VALIDATION WARNINGS"), ("issues", "VALIDATION ISSUES")]:
        items = validation.get(key, [])
        if items:
            lines += ["", f"=== {label} ==="] + [f"- {x}" for x in items]
    lines += ["", "Write your 3-4 sentence VP-level explanation now:"]
    return "\n".join(lines)


def explanation_node(state: InvoiceState) -> dict:
    inv = state.get("extracted_invoice") or {}
    inv_num = inv.get("invoice_number", "UNKNOWN")
    approval = state.get("approval_decision") or {}

    try:
        explanation = assess(_build_prompt(state), temperature=0.4)
    except Exception:
        vendor = inv.get("vendor_name") or "UNKNOWN"
        try:
            amt = f"${float(inv.get('total_amount') or 0.0):,.2f}"
        except (ValueError, TypeError):
            amt = str(inv.get("total_amount", "unknown"))
        explanation = (
            f"Invoice {inv_num} from {vendor} for {amt} was {approval.get('status', 'unknown')} "
            f"by the {approval.get('approver', 'system')}. "
            f"Reason: {approval.get('reasoning', 'No reasoning provided.')}."
        )

    return {
        "decision_explanation": explanation,
        "current_agent": "explanation",
        "audit_trail": [{"agent": "explanation", "action": "explanation_generated",
                         "details": f"{len(explanation)} chars for {inv_num}"}],
    }
