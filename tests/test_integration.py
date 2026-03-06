"""End-to-end integration tests."""

import json
import uuid
from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver


def _mock_extraction(monkeypatch, response_dict: dict) -> None:
    monkeypatch.setattr(
        "src.agents.extraction.get_structured_llm",
        lambda schema: MagicMock(invoke=MagicMock(side_effect=Exception("mocked"))),
    )
    monkeypatch.setattr(
        "src.agents.extraction.assess",
        lambda prompt, temperature=0.0: json.dumps(response_dict),
    )


def _mock_extraction_sequence(monkeypatch, responses: list[dict]) -> None:
    monkeypatch.setattr(
        "src.agents.extraction.get_structured_llm",
        lambda schema: MagicMock(invoke=MagicMock(side_effect=Exception("mocked"))),
    )
    call_iter = iter(responses)
    monkeypatch.setattr("src.agents.extraction.assess",
                         lambda prompt, temperature=0.0: json.dumps(next(call_iter)))


def _mock_all_llm(monkeypatch) -> None:
    monkeypatch.setattr("src.agents.fraud.assess",
                         lambda prompt, temperature=0.4: "Mock risk narrative.")
    monkeypatch.setattr("src.agents.explanation.assess",
                         lambda prompt, temperature=0.4: "Mock explanation.")


def _clean_invoice(invoice_number: str = "INV-HAPPY-001") -> dict:
    return {
        "invoice_number": invoice_number,
        "vendor_name": "Widgets Inc.",
        "invoice_date": "2026-01-15",
        "due_date": "2026-02-01",
        "line_items": [{"item_name": "WidgetB", "quantity": 1.0, "unit_price": 500.0,
                        "line_total": 500.0, "note": None}],
        "subtotal": 500.0, "tax_amount": None, "total_amount": 500.0,
        "currency": "USD", "payment_terms": "Net 30", "notes": None,
        "confidence_scores": {}, "extraction_warnings": [],
    }


def _fraud_invoice(invoice_number: str = "INV-FRAUD-001") -> dict:
    return {
        "invoice_number": invoice_number,
        "vendor_name": "Fraudster LLC",
        "invoice_date": "2027-01-01",
        "due_date": None,
        "line_items": [
            {"item_name": "WidgetA", "quantity": 4.0, "unit_price": 250.0,
             "line_total": 1000.0, "note": None},
            {"item_name": "WidgetB", "quantity": 2.0, "unit_price": 500.0,
             "line_total": 1000.0, "note": None},
        ],
        "subtotal": 2000.0, "tax_amount": None, "total_amount": 2000.0,
        "currency": "USD", "payment_terms": "Immediate",
        "notes": "URGENT - Pay immediately via wire transfer to avoid penalty.",
        "confidence_scores": {}, "extraction_warnings": [],
    }


def _hitl_invoice(invoice_number: str = "INV-HITL-001") -> dict:
    return {
        "invoice_number": invoice_number,
        "vendor_name": "Widgets Inc.",
        "invoice_date": "2026-01-15",
        "due_date": "2026-02-15",
        "line_items": [{"item_name": "WidgetA", "quantity": 5.0, "unit_price": 3000.0,
                        "line_total": 15000.0, "note": None}],
        "subtotal": 15000.0, "tax_amount": None, "total_amount": 15000.0,
        "currency": "USD", "payment_terms": "Net 30", "notes": None,
        "confidence_scores": {}, "extraction_warnings": [],
    }


def test_happy_path_end_to_end(patch_db, monkeypatch, tmp_path):
    _mock_extraction(monkeypatch, _clean_invoice())
    _mock_all_llm(monkeypatch)

    from src.pipeline import build_pipeline, process_invoice
    pipeline = build_pipeline(checkpointer=MemorySaver())
    invoice_file = tmp_path / "invoice_happy.txt"
    invoice_file.write_text("Invoice: INV-HAPPY-001\nVendor: Widgets Inc.\nTotal: $500")

    state = process_invoice(pipeline, str(invoice_file), thread_id=str(uuid.uuid4()))

    approval = state.get("approval_decision") or {}
    assert approval.get("status") == "approved"
    assert approval.get("approver") == "auto"
    assert (state.get("payment_result") or {}).get("status") == "success"

    agents = {entry.get("agent") for entry in state.get("audit_trail", [])}
    assert "extraction" in agents
    assert "payment" in agents


def test_rejection_path(patch_db, monkeypatch, tmp_path):
    inv = _fraud_invoice()
    raw_text = "URGENT - Pay immediately via wire transfer to avoid penalty.\nVendor: Fraudster LLC"
    _mock_extraction(monkeypatch, inv)
    _mock_all_llm(monkeypatch)
    monkeypatch.setattr("src.agents.extraction.assess",
                         lambda prompt, temperature=0.0: json.dumps(inv))

    from src.pipeline import build_pipeline, process_invoice
    pipeline = build_pipeline(checkpointer=MemorySaver())
    invoice_file = tmp_path / "invoice_fraud.txt"
    invoice_file.write_text(raw_text)

    state = process_invoice(pipeline, str(invoice_file), thread_id=str(uuid.uuid4()))

    assert (state.get("approval_decision") or {}).get("status") == "rejected"
    assert (state.get("fraud_result") or {}).get("risk_score", 0) >= 70
    assert state.get("payment_result") is None


def test_self_correction_loop(patch_db, monkeypatch, tmp_path):
    bad = {**_clean_invoice("INV-RETRY-001"), "vendor_name": ""}
    good = _clean_invoice("INV-RETRY-001")
    _mock_extraction_sequence(monkeypatch, [bad, good])
    _mock_all_llm(monkeypatch)

    from src.pipeline import build_pipeline, process_invoice
    pipeline = build_pipeline(checkpointer=MemorySaver())
    f = tmp_path / "invoice_retry.txt"
    f.write_text("Vendor: Widgets Inc.\nInvoice: INV-RETRY-001\nTotal: $500")

    state = process_invoice(pipeline, str(f), thread_id=str(uuid.uuid4()))
    assert state.get("extraction_retries") == 1
    assert (state.get("approval_decision") or {}).get("status") == "approved"
    assert "retry" in [e.get("action") for e in state.get("audit_trail", [])]


def test_interrupt_and_resume(patch_db, monkeypatch, tmp_path):
    _mock_extraction(monkeypatch, _hitl_invoice())
    _mock_all_llm(monkeypatch)

    from src.pipeline import build_pipeline, process_invoice, resume_after_human_review
    checkpointer = MemorySaver()
    pipeline = build_pipeline(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())

    f = tmp_path / "invoice_hitl.txt"
    f.write_text("Vendor: Widgets Inc.\nInvoice: INV-HITL-001\nTotal: $15000")

    state = process_invoice(pipeline, str(f), thread_id=thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = pipeline.get_state(config)
    assert snapshot.next, "Pipeline should be paused"

    final = resume_after_human_review(pipeline, thread_id=thread_id,
                                       decision="approved", reasoning="Test operator approved")
    assert (final.get("approval_decision") or {}).get("status") == "approved"
    assert (final.get("approval_decision") or {}).get("approver") == "human"
    assert (final.get("payment_result") or {}).get("status") == "success"


def test_batch_processing(patch_db, monkeypatch, tmp_path):
    clean = _clean_invoice("INV-BATCH-001")
    clean2 = {**_clean_invoice("INV-BATCH-002"), "invoice_number": "INV-BATCH-002"}
    fraud = _fraud_invoice("INV-BATCH-003")
    raw_fraud = "URGENT - Pay immediately via wire transfer\nVendor: Fraudster LLC"

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "invoice_b1.txt").write_text("Vendor: Widgets Inc.\nTotal: $500")
    (batch_dir / "invoice_b2.txt").write_text("Vendor: Widgets Inc.\nTotal: $500")
    (batch_dir / "invoice_b3.txt").write_text(raw_fraud)

    _mock_extraction_sequence(monkeypatch, [clean, clean2, fraud])
    _mock_all_llm(monkeypatch)
    monkeypatch.setattr("main.init_db", lambda: None)

    from main import run_batch
    batch = run_batch(str(batch_dir), auto_approve=True)

    assert batch.total_processed == 3
    decisions = {r.invoice_number: r.decision for r in batch.records}
    assert decisions.get("INV-BATCH-001") == "approved"
    assert decisions.get("INV-BATCH-002") == "approved"
    assert decisions.get("INV-BATCH-003") == "rejected"
