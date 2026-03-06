"""SQLite database init — schema, seed data, WAL mode."""

import sqlite3

from src.config import get_settings


def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_settings().db_path
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create tables and seed data. Idempotent."""
    conn = get_db_connection(db_path)
    try:
        _create_tables(conn)
        _seed_inventory(conn)
        _seed_vendors(conn)
        conn.commit()
    finally:
        conn.close()


def clear_invoice_history(db_path: str | None = None) -> None:
    """Wipe invoice_history — useful between batch runs."""
    conn = get_db_connection(db_path)
    try:
        conn.execute("DELETE FROM invoice_history")
        conn.commit()
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inventory (
            item        TEXT PRIMARY KEY,
            stock       INTEGER NOT NULL DEFAULT 0,
            unit_price  REAL NOT NULL,
            category    TEXT NOT NULL,
            min_order_qty INTEGER NOT NULL DEFAULT 1,
            max_order_qty INTEGER NOT NULL DEFAULT 9999
        );

        CREATE TABLE IF NOT EXISTS vendors (
            name                    TEXT PRIMARY KEY,
            is_approved             INTEGER NOT NULL DEFAULT 0,
            historical_avg_amount   REAL NOT NULL DEFAULT 0.0,
            total_invoices          INTEGER NOT NULL DEFAULT 0,
            error_rate              REAL NOT NULL DEFAULT 0.0,
            risk_tier               TEXT NOT NULL DEFAULT 'standard'
        );

        CREATE TABLE IF NOT EXISTS invoice_history (
            invoice_number  TEXT NOT NULL,
            vendor          TEXT NOT NULL,
            amount          REAL NOT NULL,
            processed_at    TEXT NOT NULL,
            status          TEXT NOT NULL,
            file_hash       TEXT,
            PRIMARY KEY (invoice_number, vendor)
        );

        CREATE TABLE IF NOT EXISTS audit_trail (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number  TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            agent_name      TEXT NOT NULL,
            action          TEXT NOT NULL,
            details         TEXT,
            confidence      REAL
        );
    """)


def _seed_inventory(conn: sqlite3.Connection) -> None:
    rows = [
        ("WidgetA", 15, 250.00, "widgets", 1, 100),
        ("WidgetB", 10, 500.00, "widgets", 1, 50),
        ("GadgetX", 5, 750.00, "gadgets", 1, 20),
        ("FakeItem", 0, 1000.00, "unknown", 1, 9999),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO inventory "
        "(item, stock, unit_price, category, min_order_qty, max_order_qty) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _seed_vendors(conn: sqlite3.Connection) -> None:
    rows = [
        ("Widgets Inc.", 1, 5000.0, 12, 0.05, "standard"),
        ("Precision Parts Ltd.", 1, 3000.0, 8, 0.02, "standard"),
        ("Gadgets Co.", 1, 12000.0, 6, 0.10, "elevated"),
        ("Summit Manufacturing Co.", 1, 4000.0, 15, 0.01, "standard"),
        ("Atlas Industrial Supply", 1, 20000.0, 4, 0.03, "standard"),
        ("Reliable Components Inc.", 1, 6000.0, 10, 0.02, "standard"),
        ("Consolidated Materials Group", 1, 7000.0, 9, 0.04, "standard"),
        ("Acme Industrial Supplies", 1, 3000.0, 20, 0.01, "standard"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO vendors "
        "(name, is_approved, historical_avg_amount, total_invoices, error_rate, risk_tier) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
