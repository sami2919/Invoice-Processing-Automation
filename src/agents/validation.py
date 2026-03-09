"""Validation agent — checks extracted invoice data against the inventory DB."""

import time
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Optional

import structlog

from src.models.audit import AuditEntry
from src.models.invoice import ValidationResult
from src.models.state import InvoiceState
from src.tools.inventory_db import (
    check_duplicate_invoice,
    check_item_exists,
    check_vendor_approved,
    fuzzy_match_item,
    get_item_price,
    get_item_stock,
)

logger = structlog.get_logger(__name__)

PRICE_VARIANCE_THRESHOLD = 0.10
MATH_TOLERANCE_PCT = 0.01
FUTURE_DATE_DAYS = 30


def _check_required_fields(inv: dict) -> list[str]:
    issues: list[str] = []
    if not str(inv.get("vendor_name", "")).strip():
        issues.append("Required field missing: vendor_name is empty")
    if not str(inv.get("invoice_number", "")).strip():
        issues.append("Required field missing: invoice_number is empty")
    if not (inv.get("line_items") or []):
        issues.append("Required field missing: no line items found")
    total = inv.get("total_amount") or 0.0
    try:
        if float(total) <= 0:
            issues.append(f"total_amount must be > 0, got {total}")
    except (TypeError, ValueError):
        issues.append(f"total_amount is not a valid number: {total!r}")
    return issues


def _check_negative_quantities(line_items: list[dict]) -> list[str]:
    issues: list[str] = []
    for idx, item in enumerate(line_items):
        try:
            qty = float(item.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            issues.append(
                f"Line item {idx + 1} '{item.get('item_name', '?')}': "
                f"quantity must be > 0, got {item.get('quantity')}"
            )
    return issues


def _resolve_item_names(
    line_items: list[dict], db_path: Optional[str]
) -> tuple[list[str], dict[str, Optional[str]]]:
    """Resolve each item name to its canonical inventory name via exact + fuzzy match."""
    issues: list[str] = []
    name_map: dict[str, Optional[str]] = {}

    for item in line_items:
        original = str(item.get("item_name", "") or "")
        if original in name_map:
            continue
        if check_item_exists(original, db_path):
            name_map[original] = original
        else:
            canonical = fuzzy_match_item(original, db_path)
            name_map[original] = canonical
            if canonical is None:
                issues.append(f"Item '{original}' not found in inventory (unknown item)")

    return issues, name_map


def _check_aggregate_stock(
    line_items: list[dict], name_map: dict[str, Optional[str]], db_path: Optional[str],
) -> tuple[list[str], dict[str, dict]]:
    """Sum quantities per canonical item across ALL line items, then check stock.
    This catches INV-1013 where WidgetA appears 3x (15+5+2=22) vs stock=15."""
    aggregated: dict[str, float] = defaultdict(float)
    for item in line_items:
        original = str(item.get("item_name", "") or "")
        canonical = name_map.get(original)
        if canonical is None:
            continue
        try:
            qty = float(item.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty > 0:
            aggregated[canonical] += qty

    issues: list[str] = []
    stock_checks: dict[str, dict] = {}
    for canonical, total_qty in aggregated.items():
        available = get_item_stock(canonical, db_path)
        sufficient = total_qty <= available
        stock_checks[canonical] = {"requested": total_qty, "available": available, "sufficient": sufficient}
        if not sufficient:
            issues.append(f"Insufficient stock for '{canonical}': requested {total_qty:g}, available {available}")

    return issues, stock_checks


def _check_price_variance(
    line_items: list[dict], name_map: dict[str, Optional[str]], db_path: Optional[str],
) -> list[str]:
    prices_by_item: dict[str, list[tuple[str, float]]] = {}
    for item in line_items:
        original = str(item.get("item_name", "") or "")
        canonical = name_map.get(original)
        if canonical is None:
            continue
        try:
            price = float(item.get("unit_price", 0) or 0)
        except (TypeError, ValueError):
            continue
        prices_by_item.setdefault(canonical, []).append((original, price))

    issues: list[str] = []
    for canonical, observed in prices_by_item.items():
        catalog = get_item_price(canonical, db_path)
        if catalog <= 0:
            continue
        flagged = [(o, p) for o, p in observed if abs(p - catalog) / catalog > PRICE_VARIANCE_THRESHOLD]
        if flagged:
            strs = ", ".join(f"${p:.2f} ({abs(p - catalog) / catalog * 100:.1f}%)" for _, p in flagged)
            issues.append(f"Price variance for '{canonical}': catalog ${catalog:.2f}, invoice: {strs}")
    return issues


def _check_math(line_items: list[dict], total_amount: float, tax_amount: float = 0.0) -> list[str]:
    if total_amount <= 0:
        return []
    computed = 0.0
    for item in line_items:
        try:
            qty = float(item.get("quantity", 0) or 0)
            price = float(item.get("unit_price", 0) or 0)
        except (TypeError, ValueError):
            qty, price = 0.0, 0.0
        if qty > 0:
            computed += qty * price
    expected = computed + tax_amount
    delta = abs(expected - total_amount) / total_amount
    if delta > MATH_TOLERANCE_PCT:
        return [
            f"Math mismatch: calculated ${expected:.2f} (items ${computed:.2f}"
            + (f" + tax ${tax_amount:.2f}" if tax_amount else "")
            + f") differs from stated total ${total_amount:.2f} ({delta * 100:.1f}% difference)"
        ]
    return []


def _check_duplicate(invoice_number: str, db_path: Optional[str]) -> list[str]:
    is_dup, record = check_duplicate_invoice(invoice_number, db_path)
    if is_dup:
        return [
            f"Duplicate invoice: '{invoice_number}' was already processed "
            f"on {record.get('processed_at', 'unknown')} (status: {record.get('status', 'unknown')})"
        ]
    return []


def _check_currency(currency: str) -> list[str]:
    if currency != "USD":
        return [f"Non-USD currency: '{currency}' (expected USD)"]
    return []


def _parse_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _check_dates(inv: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    today = datetime.now(timezone.utc).date()

    inv_date = _parse_date(inv.get("invoice_date"))
    due = _parse_date(inv.get("due_date"))

    if inv_date and inv_date > today and (inv_date - today).days > FUTURE_DATE_DAYS:
        warnings.append(f"Invoice date {inv_date} is {(inv_date - today).days} days in the future")
    if inv_date and due and due < inv_date:
        issues.append(f"Due date {due} is before invoice date {inv_date}")
    return issues, warnings


def _check_vendor(vendor_name: str, db_path: Optional[str]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    if not vendor_name.strip():
        return issues, warnings

    is_approved, info = check_vendor_approved(vendor_name, db_path)
    if not info.get("found"):
        warnings.append(f"Vendor '{vendor_name}' not found in approved vendor list")
        return issues, warnings
    if not is_approved:
        issues.append(f"Vendor '{vendor_name}' is not approved")
    risk = info.get("risk_tier", "standard")
    if risk in ("elevated", "blocked"):
        issues.append(f"Vendor '{vendor_name}' has elevated risk tier: '{risk}'")
    return issues, warnings


def validation_node(state: InvoiceState) -> dict:
    t0 = time.monotonic()
    inv: dict = state.get("extracted_invoice") or {}

    if not inv:
        logger.error("validation.no_invoice")
        result = ValidationResult(is_valid=False, issues=["No extracted invoice — extraction may have failed"])
        return {
            "validation_result": result.model_dump(),
            "current_agent": "validation",
            "audit_trail": [AuditEntry(agent_name="validation", action="validation_failed",
                                       details="No extracted invoice", duration=time.monotonic() - t0).model_dump()],
        }

    invoice_number = str(inv.get("invoice_number", "UNKNOWN") or "UNKNOWN")
    vendor = str(inv.get("vendor_name", "") or "")
    line_items: list[dict] = inv.get("line_items") or []
    currency = str(inv.get("currency", "USD") or "USD")
    try:
        total = float(inv.get("total_amount") or 0.0)
    except (TypeError, ValueError):
        total = 0.0

    db_path: Optional[str] = None
    all_issues: list[str] = []
    all_warnings: list[str] = []

    # run all checks
    all_issues.extend(_check_required_fields(inv))
    existence_issues, name_map = _resolve_item_names(line_items, db_path)
    all_issues.extend(existence_issues)
    stock_issues, stock_checks = _check_aggregate_stock(line_items, name_map, db_path)
    all_issues.extend(stock_issues)
    all_issues.extend(_check_negative_quantities(line_items))

    # warnings (not hard failures)
    all_warnings.extend(_check_price_variance(line_items, name_map, db_path))
    try:
        tax = float(inv.get("tax_amount") or 0.0)
    except (TypeError, ValueError):
        tax = 0.0
    all_warnings.extend(_check_math(line_items, total, tax))
    all_warnings.extend(_check_duplicate(invoice_number, db_path))
    all_warnings.extend(_check_currency(currency))

    date_issues, date_warnings = _check_dates(inv)
    all_issues.extend(date_issues)
    all_warnings.extend(date_warnings)
    vendor_issues, vendor_warnings = _check_vendor(vendor, db_path)
    all_issues.extend(vendor_issues)
    all_warnings.extend(vendor_warnings)

    is_valid = len(all_issues) == 0
    result = ValidationResult(is_valid=is_valid, issues=all_issues, warnings=all_warnings, stock_checks=stock_checks)
    duration = time.monotonic() - t0

    logger.info("validation.done", invoice=invoice_number, valid=is_valid,
                issues=len(all_issues), warnings=len(all_warnings))

    return {
        "validation_result": result.model_dump(),
        "current_agent": "validation",
        "audit_trail": [AuditEntry(
            agent_name="validation", action="validation_complete",
            details=f"{'PASSED' if is_valid else 'FAILED'}: {len(all_issues)} issue(s), {len(all_warnings)} warning(s)",
            duration=duration,
        ).model_dump()],
    }
