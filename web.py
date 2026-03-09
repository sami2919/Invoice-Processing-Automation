"""Flask dashboard for the invoice processing pipeline."""

import argparse
import csv
import glob
import io
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import json

from flask import Flask, Response, jsonify, render_template, request, send_file
from langgraph.checkpoint.memory import MemorySaver

from src.config import get_settings
from src.database import clear_invoice_history, init_db
from src.models.audit import ProcessingRecord
from src.pipeline import build_pipeline, process_invoice, resume_after_human_review

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

_BATCH_EXTENSIONS = ("*.txt", "*.json", "*.csv", "*.xml", "*.pdf")

_store = {
    "pipeline": None,
    "results": [],
    "audit_entries": [],
    "is_interrupted": False,
    "review_context": None,
    "current_thread_id": None,
    "last_state": None,
    "last_record": None,
}


def _ensure_pipeline():
    if _store["pipeline"] is None:
        _store["pipeline"] = build_pipeline(checkpointer=MemorySaver())
        init_db()
        clear_invoice_history()


def _detect_interrupt(pipeline, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = pipeline.get_state(config)
    if not snapshot.next:
        return False, None
    for task in snapshot.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return True, task.interrupts[0].value
    return False, None


def _build_processing_record(state: dict, processing_time: float) -> ProcessingRecord:
    extracted = state.get("extracted_invoice") or {}
    fraud = state.get("fraud_result") or {}
    approval = state.get("approval_decision") or {}
    return ProcessingRecord(
        invoice_number=extracted.get("invoice_number") or state.get("file_path", "unknown"),
        vendor=extracted.get("vendor_name") or "MISSING - No vendor specified",
        amount=float(extracted.get("total_amount") or 0.0),
        risk_score=int(fraud.get("risk_score") or 0),
        decision=approval.get("status") or "unknown",
        processing_time_seconds=processing_time,
        explanation=state.get("decision_explanation") or approval.get("reasoning") or "",
    )


def _normalise_trail(trail: list, state: dict) -> list[dict]:
    inv = state.get("extracted_invoice") or {}
    invoice_number = inv.get("invoice_number") or state.get("file_path", "unknown")
    out = []
    for entry in trail:
        if isinstance(entry, dict):
            out.append({
                "invoice": invoice_number,
                "agent": entry.get("agent", "unknown"),
                "action": entry.get("action", ""),
                "details": entry.get("details", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return out


def _auto_decide_hitl(state: dict) -> tuple[str, str]:
    fraud = state.get("fraud_result") or {}
    validation = state.get("validation_result") or {}
    risk_score = int(fraud.get("risk_score") or 0)
    warnings = validation.get("warnings", [])
    settings = get_settings()

    if risk_score >= settings.high_risk_threshold:
        return "rejected", f"Auto-reject: risk score {risk_score} >= {settings.high_risk_threshold}"
    if risk_score >= settings.medium_risk_threshold:
        return "escalated", f"Flagged: medium risk {risk_score}, needs human review"

    ignorable = ("past due", "overdue", "past the due date")
    substantive_warnings = [
        w for w in warnings
        if not any(ign.lower() in w.lower() for ign in ignorable)
    ]
    if substantive_warnings:
        return "escalated", f"Flagged for review: {len(substantive_warnings)} warning(s)"
    return "approved", "Auto-approved: low risk, no substantive warnings"


def _record_to_dict(r: ProcessingRecord) -> dict:
    return {
        "invoice_number": r.invoice_number,
        "vendor": r.vendor,
        "amount": r.amount,
        "risk_score": r.risk_score,
        "decision": r.decision,
        "processing_time_seconds": round(r.processing_time_seconds, 3),
        "explanation": r.explanation,
        "timestamp": r.timestamp.isoformat(),
    }


def _collect_batch_files(dir_path: Path) -> list[str]:
    raw_files: list[str] = []
    for ext in _BATCH_EXTENSIONS:
        raw_files.extend(glob.glob(str(dir_path / ext)))
    seen_stems: dict[str, str] = {}
    for f in sorted(set(raw_files)):
        stem = Path(f).stem
        if stem not in seen_stems:
            seen_stems[stem] = f
        elif seen_stems[stem].endswith(".pdf") and not f.endswith(".pdf"):
            seen_stems[stem] = f
    return sorted(seen_stems.values())


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/settings", methods=["GET"])
def api_settings():
    settings = get_settings()
    return jsonify({
        "high_risk_threshold": settings.high_risk_threshold,
        "medium_risk_threshold": settings.medium_risk_threshold,
        "auto_approve_threshold": int(settings.auto_approve_threshold),
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    _ensure_pipeline()
    results = _store["results"]
    total = len(results)
    approved = sum(1 for r in results if r.decision == "approved")
    flagged = sum(1 for r in results if r.decision in ("escalated", "pending_human_review"))
    rejected = sum(1 for r in results if r.decision in ("rejected", "error"))
    avg_risk = int(sum(r.risk_score for r in results) / total) if total else 0

    return jsonify({
        "total": total,
        "approved": approved,
        "flagged": flagged,
        "rejected": rejected,
        "avg_risk": avg_risk,
        "is_interrupted": _store["is_interrupted"],
        "review_context": _store["review_context"],
        "results": [_record_to_dict(r) for r in results],
        "audit_entries": _store["audit_entries"],
    })


@app.route("/api/process", methods=["POST"])
def api_process():
    _ensure_pipeline()
    pipeline = _store["pipeline"]

    file_path = None

    # Handle file upload
    if "file" in request.files:
        uploaded = request.files["file"]
        if uploaded.filename:
            tmp = Path(tempfile.gettempdir()) / uploaded.filename
            uploaded.save(str(tmp))
            file_path = str(tmp)

    # Handle path input
    if not file_path:
        data = request.get_json(silent=True) or {}
        file_path = data.get("file_path", "").strip()

    if not file_path:
        return jsonify({"error": "No file uploaded or path provided."}), 400
    if not Path(file_path).exists():
        return jsonify({"error": f"File not found: {file_path}"}), 404

    thread_id = str(uuid.uuid4())
    _store["current_thread_id"] = thread_id

    t0 = time.monotonic()
    state = process_invoice(pipeline, file_path, thread_id=thread_id)
    elapsed = time.monotonic() - t0

    is_interrupted, review_context = _detect_interrupt(pipeline, thread_id)

    if is_interrupted:
        _store["is_interrupted"] = True
        _store["review_context"] = review_context
        _store["last_state"] = state
        return jsonify({
            "status": "interrupted",
            "review_context": review_context,
            "thread_id": thread_id,
            "elapsed": round(elapsed, 2),
        })

    trail = state.get("audit_trail") or []
    _store["audit_entries"].extend(_normalise_trail(trail, state))
    record = _build_processing_record(state, elapsed)
    _store["results"].append(record)
    _store["last_record"] = record
    _store["last_state"] = state

    return jsonify({
        "status": "complete",
        "record": _record_to_dict(record),
        "audit_trail": trail,
    })


@app.route("/api/hitl", methods=["POST"])
def api_hitl():
    _ensure_pipeline()
    if not _store["is_interrupted"]:
        return jsonify({"error": "No pending HITL review."}), 400

    data = request.get_json(silent=True) or {}
    decision = data.get("decision", "").strip()
    reasoning = data.get("reasoning", "").strip()

    if decision not in ("approved", "rejected", "escalated"):
        return jsonify({"error": "Decision must be approved, rejected, or escalated."}), 400

    pipeline = _store["pipeline"]
    thread_id = _store["current_thread_id"]

    t0 = time.monotonic()
    final_state = resume_after_human_review(pipeline, thread_id, decision, reasoning)
    elapsed = time.monotonic() - t0

    _store["audit_entries"].extend(
        _normalise_trail(final_state.get("audit_trail") or [], final_state)
    )
    record = _build_processing_record(final_state, elapsed)
    _store["results"].append(record)
    _store["last_record"] = record
    _store["last_state"] = final_state
    _store["is_interrupted"] = False
    _store["review_context"] = None
    _store["current_thread_id"] = None

    return jsonify({
        "status": "complete",
        "record": _record_to_dict(record),
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    _ensure_pipeline()
    data = request.get_json(silent=True) or {}
    batch_dir = data.get("directory", "data/invoices").strip()
    auto_approve = data.get("auto_approve", True)
    fresh = data.get("fresh", True)

    dir_path = Path(batch_dir)
    if not dir_path.is_dir():
        return jsonify({"error": f"Directory not found: {batch_dir}"}), 404

    files = _collect_batch_files(dir_path)
    if not files:
        return jsonify({"error": "No invoice files found."}), 404

    if fresh:
        clear_invoice_history()

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        pipeline = _store["pipeline"]
        records: list[ProcessingRecord] = []
        total = len(files)

        for i, fp in enumerate(files):
            fname = Path(fp).name
            yield _sse("progress", {
                "current": i + 1,
                "total": total,
                "file": fname,
            })

            thread_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                state = process_invoice(pipeline, fp, thread_id=thread_id)
                is_interrupted, _ = _detect_interrupt(pipeline, thread_id)

                if is_interrupted:
                    if auto_approve:
                        decision, reasoning = _auto_decide_hitl(state)
                    else:
                        decision, reasoning = "escalated", "Batch mode: requires manual review"
                    state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

                elapsed = time.monotonic() - t0
                record = _build_processing_record(state, elapsed)
                records.append(record)
                _store["audit_entries"].extend(
                    _normalise_trail(state.get("audit_trail") or [], state)
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                records.append(ProcessingRecord(
                    invoice_number=Path(fp).stem,
                    vendor="error",
                    amount=0.0,
                    risk_score=0,
                    decision="error",
                    processing_time_seconds=elapsed,
                    explanation=str(exc),
                ))

        _store["results"].extend(records)

        yield _sse("complete", {
            "status": "complete",
            "total": len(records),
            "approved": sum(1 for r in records if r.decision == "approved"),
            "rejected": sum(1 for r in records if r.decision in ("rejected", "error")),
            "flagged": sum(1 for r in records if r.decision in ("escalated", "pending_human_review")),
            "records": [_record_to_dict(r) for r in records],
        })

    return Response(generate(), mimetype="text/event-stream")


_ALLOWED_EXTENSIONS = {".txt", ".json", ".csv", ".xml", ".pdf"}


def _dedup_uploaded_files(file_paths: list[str]) -> list[str]:
    """Deduplicate uploaded files by stem, preferring non-PDF formats."""
    seen_stems: dict[str, str] = {}
    for fp in sorted(file_paths):
        stem = Path(fp).stem
        if stem not in seen_stems:
            seen_stems[stem] = fp
        elif seen_stems[stem].endswith(".pdf") and not fp.endswith(".pdf"):
            seen_stems[stem] = fp
    return sorted(seen_stems.values())


@app.route("/api/batch-upload", methods=["POST"])
def api_batch_upload():
    _ensure_pipeline()

    uploaded_files = request.files.getlist("files")
    if not uploaded_files or all(not f.filename for f in uploaded_files):
        return jsonify({"error": "No files uploaded."}), 400

    auto_approve = request.form.get("auto_approve", "true").lower() == "true"
    fresh = request.form.get("fresh", "true").lower() == "true"

    # Validate extensions and save to temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="batch_upload_"))
    saved_paths: list[str] = []

    for uploaded in uploaded_files:
        if not uploaded.filename:
            continue
        ext = Path(uploaded.filename).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type: {uploaded.filename}. "
                         f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            }), 400
        dest = tmp_dir / uploaded.filename
        uploaded.save(str(dest))
        saved_paths.append(str(dest))

    if not saved_paths:
        return jsonify({"error": "No valid files uploaded."}), 400

    files = _dedup_uploaded_files(saved_paths)

    if fresh:
        clear_invoice_history()

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        pipeline = _store["pipeline"]
        records: list[ProcessingRecord] = []
        total = len(files)

        for i, fp in enumerate(files):
            fname = Path(fp).name
            yield _sse("progress", {
                "current": i + 1,
                "total": total,
                "file": fname,
            })

            thread_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                state = process_invoice(pipeline, fp, thread_id=thread_id)
                is_interrupted, _ = _detect_interrupt(pipeline, thread_id)

                if is_interrupted:
                    if auto_approve:
                        decision, reasoning = _auto_decide_hitl(state)
                    else:
                        decision, reasoning = "escalated", "Batch mode: requires manual review"
                    state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

                elapsed = time.monotonic() - t0
                record = _build_processing_record(state, elapsed)
                records.append(record)
                _store["audit_entries"].extend(
                    _normalise_trail(state.get("audit_trail") or [], state)
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                records.append(ProcessingRecord(
                    invoice_number=Path(fp).stem,
                    vendor="error",
                    amount=0.0,
                    risk_score=0,
                    decision="error",
                    processing_time_seconds=elapsed,
                    explanation=str(exc),
                ))

        _store["results"].extend(records)

        yield _sse("complete", {
            "status": "complete",
            "total": len(records),
            "approved": sum(1 for r in records if r.decision == "approved"),
            "rejected": sum(1 for r in records if r.decision in ("rejected", "error")),
            "flagged": sum(1 for r in records if r.decision in ("escalated", "pending_human_review")),
            "records": [_record_to_dict(r) for r in records],
        })

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/export-csv", methods=["GET"])
def api_export_csv():
    results = _store["results"]
    if not results:
        return jsonify({"error": "No results to export."}), 404

    fieldnames = [
        "invoice_number", "vendor", "amount", "risk_score",
        "decision", "processing_time_seconds", "explanation", "timestamp",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(_record_to_dict(r))

    mem = io.BytesIO(buf.getvalue().encode())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                     download_name=f"batch_results_{ts}.csv")


@app.route("/api/reset-db", methods=["POST"])
def api_reset_db():
    db = Path("inventory.db")
    if db.exists():
        db.unlink()
    init_db()
    _store["results"].clear()
    _store["audit_entries"].clear()
    _store["last_record"] = None
    _store["last_state"] = None
    _store["is_interrupted"] = False
    _store["review_context"] = None
    return jsonify({"status": "ok"})


def main():
    parser = argparse.ArgumentParser(description="Invoice Processing Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    _ensure_pipeline()
    print(f"\n  Invoice Processing Dashboard")
    print(f"  http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
