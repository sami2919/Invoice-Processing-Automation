"""Fraud detection — 14 weighted signals scored deterministically, with an LLM narrative."""

import re
from datetime import date, datetime, timezone
from typing import Optional

import structlog

from src.config import get_settings
from src.llm.grok_client import assess
from src.models.invoice import FraudResult, FraudSignal
from src.models.state import InvoiceState
from src.tools.inventory_db import (
    check_duplicate_invoice,
    check_item_exists,
    check_vendor_approved,
    get_item_price,
    get_item_stock,
)

logger = structlog.get_logger(__name__)

_MATH_TOLERANCE_PCT = 0.01

_URGENCY_RE = re.compile(
    r"\b(URGENT|immediately|wire\s+transfer|penalty|pay\s+now)\b",
    re.IGNORECASE,
)
_SUSPICIOUS_VENDOR_RE = re.compile(
    r"\b(fraudster|fake|noprod|test)\b",
    re.IGNORECASE,
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def fraud_detection_node(state: InvoiceState) -> dict:
    extracted = state.get("extracted_invoice")
    raw_text = state.get("raw_text", "")

    if not extracted:
        result = FraudResult(risk_score=0, signals=[], recommendation="auto_approve",
                             narrative="No extracted invoice data available for fraud analysis.")
        return {
            "fraud_result": result.model_dump(),
            "current_agent": "fraud",
            "audit_trail": [{"agent": "fraud", "action": "skip", "details": "no extracted invoice"}],
        }

    vendor = extracted.get("vendor_name", "")
    total = float(extracted.get("total_amount", 0.0))
    inv_num = extracted.get("invoice_number", "")
    line_items = extracted.get("line_items", [])
    due_date_raw = extracted.get("due_date")
    invoice_date_raw = extracted.get("invoice_date")
    notes = extracted.get("notes", "") or ""
    tax_amount: Optional[float] = extracted.get("tax_amount")

    signals: list[FraudSignal] = []

    # unknown vendor (+20)
    is_approved, vendor_info = check_vendor_approved(vendor)
    if not vendor_info.get("found", False):
        signals.append(FraudSignal(
            signal_type="unknown_vendor", severity="high",
            description=f"Vendor '{vendor}' is not in the approved vendor list.", weight=20))

    # elevated/blocked vendor (+15)
    risk_tier = vendor_info.get("risk_tier", "standard")
    if vendor_info.get("found") and risk_tier in ("elevated", "blocked"):
        signals.append(FraudSignal(
            signal_type="elevated_vendor_risk", severity="medium",
            description=f"Vendor '{vendor}' has risk tier '{risk_tier}'.", weight=15))

    # duplicate invoice (+30)
    is_dup, existing = check_duplicate_invoice(inv_num)
    if is_dup:
        signals.append(FraudSignal(
            signal_type="duplicate_invoice", severity="critical",
            description=f"Invoice '{inv_num}' was already processed on {existing.get('processed_at', 'an earlier date')}.",
            weight=30))

    # amount > 3x vendor average (+15)
    hist_avg = float(vendor_info.get("historical_avg_amount", 0.0) or 0.0)
    if vendor_info.get("found") and hist_avg > 0 and total > 3 * hist_avg:
        signals.append(FraudSignal(
            signal_type="amount_exceeds_3x_avg", severity="high",
            description=f"Invoice amount ${total:,.2f} exceeds 3x vendor average (${hist_avg:,.2f}).",
            weight=15))

    # urgency / social engineering (+15)
    matches = _URGENCY_RE.findall(f"{raw_text} {notes}")
    if matches:
        terms = list(dict.fromkeys(m.lower() for m in matches))
        signals.append(FraudSignal(
            signal_type="urgency_language", severity="high",
            description=f"Social-engineering language detected: {', '.join(terms)}.", weight=15))

    # round amount (+5)
    if total > 0 and total % 1000 == 0:
        signals.append(FraudSignal(
            signal_type="round_amount", severity="low",
            description=f"Invoice total ${total:,.2f} is a suspiciously round number.", weight=5))

    # missing due date (+10)
    if not due_date_raw:
        signals.append(FraudSignal(
            signal_type="missing_due_date", severity="medium",
            description="Invoice has no due date specified.", weight=10))

    # unknown items (+15)
    unknown = [i.get("item_name", "") for i in line_items if not check_item_exists(i.get("item_name", ""))]
    if unknown:
        desc = f"{len(unknown)} item(s) not found in inventory: {', '.join(repr(n) for n in unknown[:3])}"
        if len(unknown) > 3:
            desc += f" (+{len(unknown) - 3} more)"
        signals.append(FraudSignal(signal_type="unknown_items", severity="medium", description=desc, weight=15))

    # per-item signals (fire at most once each)
    zero_stock_fired = neg_qty_fired = price_var_fired = False

    for item in line_items:
        name = item.get("item_name", "")
        qty = float(item.get("quantity", 1))
        price = float(item.get("unit_price", 0.0))

        # zero-stock item (+20) — only for items actually in catalog
        if not zero_stock_fired and check_item_exists(name):
            if get_item_stock(name) == 0:
                signals.append(FraudSignal(
                    signal_type="zero_stock_item", severity="high",
                    description=f"Item '{name}' has zero stock in inventory.", weight=20))
                zero_stock_fired = True

        # negative/zero qty (+15)
        if not neg_qty_fired and qty <= 0:
            signals.append(FraudSignal(
                signal_type="negative_quantity", severity="high",
                description=f"Line item '{name}' has non-positive quantity ({qty}).", weight=15))
            neg_qty_fired = True

        # price variance > 20% (+10)
        if not price_var_fired:
            db_price = get_item_price(name)
            if db_price > 0 and price > 0:
                var = abs(price - db_price) / db_price
                if var > 0.20:
                    signals.append(FraudSignal(
                        signal_type="price_variance", severity="medium",
                        description=f"Item '{name}' billed at ${price:.2f} vs DB ${db_price:.2f} ({var:.0%} variance).",
                        weight=10))
                    price_var_fired = True

    # math errors (+10)
    subtotal = sum(
        float(i.get("quantity", 0)) * float(i.get("unit_price", 0.0))
        for i in line_items if float(i.get("quantity", 0)) > 0
    )
    expected = subtotal + float(tax_amount) if tax_amount is not None else subtotal
    if total > 0 and abs(expected - total) / total > _MATH_TOLERANCE_PCT:
        signals.append(FraudSignal(
            signal_type="math_error", severity="medium",
            description=f"Calculated total ${expected:,.2f} does not match stated ${total:,.2f} ({abs(expected - total) / total:.1%} diff).",
            weight=10))

    # threshold manipulation $9k-$10k (+10)
    if 9000.0 <= total <= 9999.99:
        signals.append(FraudSignal(
            signal_type="threshold_manipulation", severity="medium",
            description=f"Invoice amount ${total:,.2f} falls in the $9,000-$9,999.99 threshold-evasion range.",
            weight=10))

    # future invoice date (+10)
    today = _today()
    if invoice_date_raw:
        try:
            inv_date = date.fromisoformat(invoice_date_raw) if isinstance(invoice_date_raw, str) else invoice_date_raw
            if inv_date > today:
                signals.append(FraudSignal(
                    signal_type="future_invoice_date", severity="medium",
                    description=f"Invoice date {inv_date} is in the future.", weight=10))
        except (ValueError, TypeError):
            pass

    # suspicious vendor name (+10)
    if _SUSPICIOUS_VENDOR_RE.search(vendor):
        signals.append(FraudSignal(
            signal_type="suspicious_vendor_name", severity="high",
            description=f"Vendor name '{vendor}' contains suspicious keywords.", weight=10))

    # aggregate score
    settings = get_settings()
    risk_score = min(sum(s.weight for s in signals), 100)

    if risk_score >= settings.high_risk_threshold:
        recommendation = "block"
    elif risk_score >= settings.medium_risk_threshold:
        recommendation = "flag_for_review"
    else:
        recommendation = "auto_approve"

    # ask Grok for a short narrative
    signal_lines = "\n".join(f"- [{s.severity.upper()}] {s.description}" for s in signals) or "No fraud signals."
    prompt = (
        f"You are a financial compliance analyst reviewing an invoice.\n\n"
        f"Invoice: {inv_num}\nVendor: {vendor}\nAmount: ${total:,.2f}\n"
        f"Risk Score: {risk_score}/100\nRecommendation: {recommendation.replace('_', ' ')}\n\n"
        f"Fraud signals:\n{signal_lines}\n\n"
        f"Write a 2-3 sentence risk assessment. Be concise."
    )
    try:
        narrative = assess(prompt, temperature=0.4)
    except Exception as e:
        logger.warning("fraud.narrative_failed", error=str(e))
        narrative = f"Risk score {risk_score}/100 with {len(signals)} signal(s). Recommendation: {recommendation.replace('_', ' ')}."

    result = FraudResult(risk_score=risk_score, signals=signals, recommendation=recommendation, narrative=narrative)
    logger.info("fraud.done", invoice=inv_num, score=risk_score, rec=recommendation)

    return {
        "fraud_result": result.model_dump(),
        "current_agent": "fraud",
        "audit_trail": [{"agent": "fraud", "action": "fraud_check",
                         "details": f"Score {risk_score}/100 — {recommendation} — {len(signals)} signal(s)"}],
    }
