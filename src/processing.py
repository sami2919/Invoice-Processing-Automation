"""Shared processing utilities used by both web.py and main.py."""

import glob
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from src.config import get_settings
from src.models.audit import ProcessingRecord
from src.pipeline import process_invoice, resume_after_human_review

logger = logging.getLogger(__name__)

_BATCH_EXTENSIONS = ("*.txt", "*.json", "*.csv", "*.xml", "*.pdf")


def detect_interrupt(pipeline, thread_id: str) -> tuple[bool, dict | None]:
    """Check if pipeline is paused at an interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = pipeline.get_state(config)

    if not snapshot.next:
        return False, None

    for task in snapshot.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return True, task.interrupts[0].value

    return False, None


def _extract_invoice_fields(state: dict) -> dict:
    extracted = state.get("extracted_invoice") or {}
    return {
        "invoice_number": extracted.get("invoice_number") or state.get("file_path", "unknown"),
        "vendor": extracted.get("vendor_name") or "MISSING - No vendor specified",
        "amount": float(extracted.get("total_amount") or 0.0),
    }


def _extract_decision_fields(state: dict) -> dict:
    fraud = state.get("fraud_result") or {}
    approval = state.get("approval_decision") or {}
    return {
        "risk_score": int(fraud.get("risk_score") or 0),
        "decision": approval.get("status") or "unknown",
        "explanation": state.get("decision_explanation") or approval.get("reasoning") or "",
    }


def build_processing_record(state: dict, processing_time: float) -> ProcessingRecord:
    """Build a ProcessingRecord from pipeline state."""
    return ProcessingRecord(
        **_extract_invoice_fields(state),
        **_extract_decision_fields(state),
        processing_time_seconds=processing_time,
    )


def auto_decide_hitl(state: dict) -> tuple[str, str]:
    """Decide on behalf of human reviewer in batch/auto mode."""
    fraud = state.get("fraud_result") or {}
    validation = state.get("validation_result") or {}
    risk_score = int(fraud.get("risk_score") or 0)
    warnings = validation.get("warnings", [])

    settings = get_settings()

    if risk_score >= settings.high_risk_threshold:
        return "rejected", f"Auto reject: risk score {risk_score} >= {settings.high_risk_threshold}"

    if risk_score >= settings.medium_risk_threshold:
        return "escalated", f"Flagged: medium risk {risk_score}, needs human review"

    ignorable = ("past due", "overdue", "past the due date")
    substantive_warnings = [
        w for w in warnings if not any(ign.lower() in w.lower() for ign in ignorable)
    ]

    if substantive_warnings:
        return "escalated", f"Flagged for review: {len(substantive_warnings)} warning(s)"

    return "approved", "Auto approved: low risk, no substantive warnings"


def collect_batch_files(dir_path: Path) -> list[str]:
    """Collect invoice files from a directory, deduplicated by stem (preferring non-PDF)."""
    raw_files: list[str] = []
    for ext in _BATCH_EXTENSIONS:
        raw_files.extend(glob.glob(str(dir_path / ext)))
    return dedup_by_stem(sorted(set(raw_files)))


def dedup_by_stem(file_paths: list[str]) -> list[str]:
    """Deduplicate file paths by stem, preferring non-PDF formats."""
    seen_stems: dict[str, str] = {}
    for fp in sorted(file_paths):
        stem = Path(fp).stem
        if stem not in seen_stems:
            seen_stems[stem] = fp
        elif seen_stems[stem].endswith(".pdf") and not fp.endswith(".pdf"):
            seen_stems[stem] = fp
    return sorted(seen_stems.values())


HitlHandler = Callable[[dict, dict | None], tuple[str, str]]
"""Callback signature: (pipeline_state, review_context) -> (decision, reasoning)."""


def _default_hitl_handler(_state: dict, _ctx: dict | None) -> tuple[str, str]:
    return "escalated", "Batch mode: requires manual review"


def _process_single_file(
    pipeline,
    fp: str,
    auto_approve: bool,
    handler: HitlHandler,
    store: Optional[dict],
    normalise_trail: Optional[Callable[[list, dict], list[dict]]],
) -> ProcessingRecord:
    thread_id = str(uuid.uuid4())
    t0 = time.monotonic()
    try:
        state = process_invoice(pipeline, fp, thread_id=thread_id)
        is_interrupted, review_ctx = detect_interrupt(pipeline, thread_id)

        if is_interrupted:
            if auto_approve:
                decision, reasoning = auto_decide_hitl(state)
            else:
                decision, reasoning = handler(state, review_ctx)
            state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

        elapsed = time.monotonic() - t0
        record = build_processing_record(state, elapsed)
        if store is not None and normalise_trail is not None:
            store["audit_entries"].extend(
                normalise_trail(state.get("audit_trail") or [], state)
            )
    except Exception:
        logger.error("batch.invoice_error", exc_info=True)
        elapsed = time.monotonic() - t0
        record = ProcessingRecord(
            invoice_number=Path(fp).stem,
            vendor="error",
            amount=0.0,
            risk_score=0,
            decision="error",
            processing_time_seconds=elapsed,
            explanation="Processing failed — check server logs for details.",
        )
    return record


def batch_process_files(
    pipeline,
    files: list[str],
    auto_approve: bool,
    store: Optional[dict] = None,
    normalise_trail: Optional[Callable[[list, dict], list[dict]]] = None,
    hitl_handler: Optional[HitlHandler] = None,
):
    """Generator that yields SSE events while processing a batch of invoice files.

    Yields (event_name, payload) tuples for each progress step and the final summary.

    Args:
        hitl_handler: Optional callback for interactive HITL decisions.
            Receives (pipeline_state, review_context) and returns (decision, reasoning).
            Only called when auto_approve is False.
    """
    records: list[ProcessingRecord] = []
    total = len(files)
    handler = hitl_handler or _default_hitl_handler

    for i, fp in enumerate(files):
        fname = Path(fp).name
        yield (
            "progress",
            {
                "current": i + 1,
                "total": total,
                "file": fname,
            },
        )

        record = _process_single_file(pipeline, fp, auto_approve, handler, store, normalise_trail)
        records.append(record)

    if store is not None:
        store["results"].extend(records)

    yield (
        "complete",
        {
            "status": "complete",
            "total": len(records),
            "approved": sum(1 for r in records if r.decision == "approved"),
            "rejected": sum(1 for r in records if r.decision in ("rejected", "error")),
            "flagged": sum(
                1 for r in records if r.decision in ("escalated", "pending_human_review")
            ),
            "records": records,
        },
    )
