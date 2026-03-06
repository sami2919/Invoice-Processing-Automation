"""Shared pytest fixtures."""

import os
os.environ.setdefault("XAI_API_KEY", "fake-test-key-for-unit-tests")

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.database import get_db_connection as _orig_get_db_connection
from src.database import init_db

DATA_DIR = Path("data/invoices")


@pytest.fixture(scope="function")
def test_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


@pytest.fixture(scope="function")
def patch_db(test_db, monkeypatch):
    """Patch get_db_connection everywhere so agents use the test DB."""
    def _patched(p=None):
        return _orig_get_db_connection(test_db)

    monkeypatch.setattr("src.database.get_db_connection", _patched)
    monkeypatch.setattr("src.tools.inventory_db.get_db_connection", _patched)
    return test_db


@pytest.fixture(scope="function")
def mock_grok(monkeypatch):
    """Suppress all LLM calls."""
    monkeypatch.setattr(
        "src.agents.fraud.assess",
        lambda prompt, temperature=0.4: "Mock risk narrative.",
    )
    try:
        monkeypatch.setattr(
            "src.agents.explanation.assess",
            lambda prompt, temperature=0.4: "Mock explanation.",
        )
    except AttributeError:
        pass


def make_extraction_mock(response_dict: dict, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agents.extraction.get_structured_llm",
        lambda schema: MagicMock(invoke=MagicMock(side_effect=Exception("mocked"))),
    )
    monkeypatch.setattr(
        "src.agents.extraction.assess",
        lambda prompt, temperature=0.0: json.dumps(response_dict),
    )


@pytest.fixture(scope="session")
def sample_invoices():
    base = DATA_DIR
    return {
        "1001": str(base / "invoice_1001.txt"),
        "1003": str(base / "invoice_1003.txt"),
        "1004": str(base / "invoice_1004.json"),
        "1004_revised": str(base / "invoice_1004_revised.json"),
        "1006": str(base / "invoice_1006.csv"),
        "1008": str(base / "invoice_1008.txt"),
        "1009": str(base / "invoice_1009.json"),
        "1010": str(base / "invoice_1010.txt"),
        "1012": str(base / "invoice_1012.txt"),
        "1013": str(base / "invoice_1013.json"),
        "1014": str(base / "invoice_1014.xml"),
    }


@pytest.fixture(scope="function")
def pipeline():
    from src.pipeline import build_pipeline
    return build_pipeline(checkpointer=MemorySaver())


def _make_state(vendor="Widgets Inc.", invoice_number="INV-1001", raw_text="", **overrides):
    """Helper to build an InvoiceState dict without copy-pasting everywhere."""
    base = {
        "file_path": "data/invoices/invoice_1001.txt",
        "raw_text": raw_text or (
            f"INVOICE\n\nVendor: {vendor}\nInvoice Number: {invoice_number}\n"
            "Date: 2026-01-15\nDue Date: 2026-02-01\n\n"
            "Items:\n  WidgetA    qty: 10    unit price: $250.00\n"
            "  WidgetB    qty: 5     unit price: $500.00\n\n"
            "Total Amount: $5,000.00\nPayment Terms: Net 15\n"
        ),
        "file_type": "txt",
        "extracted_invoice": {
            "invoice_number": invoice_number,
            "vendor_name": vendor,
            "invoice_date": "2026-01-15",
            "due_date": "2026-02-01",
            "line_items": [
                {"item_name": "WidgetA", "quantity": 10.0, "unit_price": 250.0,
                 "line_total": 2500.0, "note": None},
                {"item_name": "WidgetB", "quantity": 5.0, "unit_price": 500.0,
                 "line_total": 2500.0, "note": None},
            ],
            "subtotal": 5000.0,
            "tax_amount": 0.0,
            "total_amount": 5000.0,
            "currency": "USD",
            "payment_terms": "Net 15",
            "notes": None,
            "confidence_scores": {},
            "extraction_warnings": [],
        },
        "extraction_retries": 0,
        "extraction_feedback": "",
        "validation_result": None,
        "fraud_result": None,
        "approval_decision": None,
        "payment_result": None,
        "audit_trail": [],
        "error_message": None,
        "current_agent": "validation",
        "decision_explanation": "",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="function")
def clean_state():
    return _make_state()


@pytest.fixture(scope="function")
def fraud_state():
    return _make_state(
        vendor="Fraudster LLC",
        invoice_number="INV-1003",
        raw_text=(
            "INVOICE\n\nVendor: Fraudster LLC\nInvoice Number: INV-1003\n"
            "Date: 2026-01-20\nDue Date: yesterday\n\n"
            "Items:\n  FakeItem   qty: 100   unit price: $1,000.00\n\n"
            "Total Amount: $100,000.00\nPayment Terms: Immediate\n"
            "Notes: URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.\n"
        ),
        current_agent="fraud",
        extracted_invoice={
            "invoice_number": "INV-1003",
            "vendor_name": "Fraudster LLC",
            "invoice_date": "2026-01-20",
            "due_date": None,
            "line_items": [
                {"item_name": "FakeItem", "quantity": 100.0, "unit_price": 1000.0,
                 "line_total": 100000.0, "note": None},
            ],
            "subtotal": 100000.0,
            "tax_amount": None,
            "total_amount": 100000.0,
            "currency": "USD",
            "payment_terms": "Immediate",
            "notes": "URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
            "confidence_scores": {},
            "extraction_warnings": [],
        },
    )
