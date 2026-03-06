"""Deterministic DB queries for validation — no LLM calls."""

import difflib
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.database import get_db_connection

logger = structlog.get_logger(__name__)

FUZZY_THRESHOLD = 0.8


def check_item_exists(item_name: str, db_path: str | None = None) -> bool:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM inventory WHERE LOWER(item) = LOWER(?)",
            (item_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_item_stock(item_name: str, db_path: str | None = None) -> int:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT stock FROM inventory WHERE LOWER(item) = LOWER(?)",
            (item_name,),
        ).fetchone()
        return int(row["stock"]) if row else 0
    finally:
        conn.close()


def get_item_price(item_name: str, db_path: str | None = None) -> float:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT unit_price FROM inventory WHERE LOWER(item) = LOWER(?)",
            (item_name,),
        ).fetchone()
        return float(row["unit_price"]) if row else 0.0
    finally:
        conn.close()


def fuzzy_match_item(item_name: str, db_path: str | None = None) -> Optional[str]:
    """Try to match an item name against inventory using SequenceMatcher.
    Handles OCR artifacts like 'Widget A' vs 'WidgetA'."""
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute("SELECT item FROM inventory").fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    normalized_query = item_name.replace(" ", "").lower()
    best_match: Optional[str] = None
    best_ratio = 0.0

    for row in rows:
        canonical = row["item"]
        normalized = canonical.replace(" ", "").lower()
        ratio = difflib.SequenceMatcher(None, normalized_query, normalized).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = canonical

    if best_ratio >= FUZZY_THRESHOLD:
        logger.debug("fuzzy_match_found", query=item_name, match=best_match, ratio=round(best_ratio, 3))
        return best_match

    return None


def check_vendor_approved(vendor_name: str, db_path: str | None = None) -> tuple[bool, dict]:
    """Check vendor approval status with fuzzy matching fallback.
    Returns (is_approved, vendor_info)."""
    conn = get_db_connection(db_path)
    try:
        # exact (case-insensitive) match first
        row = conn.execute(
            "SELECT * FROM vendors WHERE LOWER(name) = LOWER(?)",
            (vendor_name,),
        ).fetchone()

        if row is not None:
            info = dict(row)
            info["found"] = True
            info["is_approved"] = bool(info["is_approved"])
            return info["is_approved"], info

        # fuzzy match: normalize spaces/punctuation and compare
        rows = conn.execute("SELECT * FROM vendors").fetchall()
        normalized_query = vendor_name.replace(" ", "").lower()
        best_row = None
        best_ratio = 0.0

        for r in rows:
            normalized = r["name"].replace(" ", "").lower()
            ratio = difflib.SequenceMatcher(None, normalized_query, normalized).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_row = r

        if best_ratio >= FUZZY_THRESHOLD and best_row is not None:
            logger.debug("fuzzy_vendor_match", query=vendor_name,
                         match=best_row["name"], ratio=round(best_ratio, 3))
            info = dict(best_row)
            info["found"] = True
            info["is_approved"] = bool(info["is_approved"])
            return info["is_approved"], info

        return False, {"found": False, "name": vendor_name}
    finally:
        conn.close()


def check_duplicate_invoice(invoice_number: str, db_path: str | None = None) -> tuple[bool, dict]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM invoice_history WHERE invoice_number = ?",
            (invoice_number,),
        ).fetchone()
        if row is None:
            return False, {}
        return True, dict(row)
    finally:
        conn.close()


def record_invoice(
    invoice_number: str,
    vendor: str,
    amount: float,
    status: str,
    file_hash: str | None = None,
    db_path: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO invoice_history "
            "(invoice_number, vendor, amount, processed_at, status, file_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (invoice_number, vendor, amount, now, status, file_hash),
        )
        conn.commit()
        logger.info("invoice_recorded", invoice_number=invoice_number, status=status)
    finally:
        conn.close()
