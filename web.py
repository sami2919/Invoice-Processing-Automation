"""Flask dashboard for the invoice processing pipeline."""

import argparse
import csv
import io
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from langgraph.checkpoint.memory import MemorySaver
from werkzeug.utils import secure_filename

from src.config import get_settings
from src.database import clear_invoice_history, init_db
from src.models.audit import ProcessingRecord
from src.pipeline import build_pipeline, process_invoice, resume_after_human_review
from src.processing import (
    batch_process_files,
    build_processing_record,
    collect_batch_files,
    dedup_by_stem,
    detect_interrupt,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

_PROJECT_ROOT = Path(__file__).resolve().parent
_ALLOWED_BASE = _PROJECT_ROOT / "data" / "invoices"
_ALLOWED_EXTENSIONS = {".txt", ".json", ".csv", ".xml", ".pdf"}

# In-process store — safe for single-worker Flask (dev/demo).
# For multi-worker deployment, replace with Redis or a database.
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


def _normalise_trail(trail: list, state: dict) -> list[dict]:
    inv = state.get("extracted_invoice") or {}
    invoice_number = inv.get("invoice_number") or state.get("file_path", "unknown")
    out = []
    for entry in trail:
        if isinstance(entry, dict):
            out.append(
                {
                    "invoice": invoice_number,
                    "agent": entry.get("agent", "unknown"),
                    "action": entry.get("action", ""),
                    "details": entry.get("details", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    return out


def _sse(event: str, payload: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


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


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/settings", methods=["GET"])
def api_settings():
    settings = get_settings()
    return jsonify(
        {
            "high_risk_threshold": settings.high_risk_threshold,
            "medium_risk_threshold": settings.medium_risk_threshold,
            "auto_approve_threshold": int(settings.auto_approve_threshold),
        }
    )


@app.route("/api/state", methods=["GET"])
def api_state():
    _ensure_pipeline()
    results = _store["results"]
    total = len(results)
    approved = sum(1 for r in results if r.decision == "approved")
    flagged = sum(1 for r in results if r.decision in ("escalated", "pending_human_review"))
    rejected = sum(1 for r in results if r.decision in ("rejected", "error"))
    avg_risk = int(sum(r.risk_score for r in results) / total) if total else 0

    return jsonify(
        {
            "total": total,
            "approved": approved,
            "flagged": flagged,
            "rejected": rejected,
            "avg_risk": avg_risk,
            "is_interrupted": _store["is_interrupted"],
            "review_context": _store["review_context"],
            "results": [_record_to_dict(r) for r in results],
            "audit_entries": _store["audit_entries"],
        }
    )


def _parse_uploaded_file(req):
    file_path = None

    if "file" in req.files:
        uploaded = req.files["file"]
        if uploaded.filename:
            tmp = Path(tempfile.gettempdir()) / secure_filename(uploaded.filename)
            uploaded.save(str(tmp))
            file_path = str(tmp)

    if not file_path:
        data = req.get_json(silent=True) or {}
        file_path = data.get("file_path", "").strip()

    if not file_path:
        return None, (jsonify({"error": "No file uploaded or path provided."}), 400)

    temp_base = Path(tempfile.gettempdir()).resolve()
    resolved = Path(file_path).resolve()
    try:
        resolved.relative_to(_ALLOWED_BASE)
    except ValueError:
        try:
            resolved.relative_to(temp_base)
        except ValueError:
            return None, (jsonify({"error": "File path must be within data/invoices/."}), 403)
    if not resolved.exists():
        return None, (jsonify({"error": "File not found."}), 404)

    return file_path, None


def _format_processing_result(state, elapsed):
    trail = state.get("audit_trail") or []
    _store["audit_entries"].extend(_normalise_trail(trail, state))
    record = build_processing_record(state, elapsed)
    _store["results"].append(record)
    _store["last_record"] = record
    _store["last_state"] = state

    return jsonify(
        {
            "status": "complete",
            "record": _record_to_dict(record),
            "audit_trail": trail,
        }
    )


@app.route("/api/process", methods=["POST"])
def api_process():
    _ensure_pipeline()
    pipeline = _store["pipeline"]

    file_path, error = _parse_uploaded_file(request)
    if error:
        return error

    thread_id = str(uuid.uuid4())
    _store["current_thread_id"] = thread_id

    t0 = time.monotonic()
    state = process_invoice(pipeline, file_path, thread_id=thread_id)
    elapsed = time.monotonic() - t0

    is_interrupted, review_context = detect_interrupt(pipeline, thread_id)

    if is_interrupted:
        _store["is_interrupted"] = True
        _store["review_context"] = review_context
        _store["last_state"] = state
        return jsonify(
            {
                "status": "interrupted",
                "review_context": review_context,
                "thread_id": thread_id,
                "elapsed": round(elapsed, 2),
            }
        )

    return _format_processing_result(state, elapsed)


@app.route("/api/hitl", methods=["POST"])
def api_hitl():
    _ensure_pipeline()
    if not _store["is_interrupted"]:
        return jsonify({"error": "No pending HITL review."}), 400

    data = request.get_json(silent=True) or {}
    decision = data.get("decision", "").strip()
    reasoning = data.get("reasoning", "").strip()[:2000]

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
    record = build_processing_record(final_state, elapsed)
    _store["results"].append(record)
    _store["last_record"] = record
    _store["last_state"] = final_state
    _store["is_interrupted"] = False
    _store["review_context"] = None
    _store["current_thread_id"] = None

    return jsonify(
        {
            "status": "complete",
            "record": _record_to_dict(record),
        }
    )


@app.route("/api/batch", methods=["POST"])
def api_batch():
    _ensure_pipeline()
    data = request.get_json(silent=True) or {}
    batch_dir = data.get("directory", "data/invoices").strip()
    auto_approve = data.get("auto_approve", True)
    fresh = data.get("fresh", True)

    # Restrict batch directory to data/invoices (anchored to project root)
    dir_path = Path(batch_dir).resolve()
    try:
        dir_path.relative_to(_ALLOWED_BASE)
    except ValueError:
        return jsonify({"error": "Batch directory must be within data/invoices/."}), 403
    if not dir_path.is_dir():
        return jsonify({"error": "Directory not found."}), 404

    files = collect_batch_files(dir_path)
    if not files:
        return jsonify({"error": "No invoice files found."}), 404

    if fresh:
        clear_invoice_history()

    return Response(_batch_stream(files, auto_approve), mimetype="text/event-stream")


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
            return jsonify(
                {
                    "error": f"Unsupported file type: {uploaded.filename}. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
                }
            ), 400
        dest = tmp_dir / secure_filename(uploaded.filename)
        uploaded.save(str(dest))
        saved_paths.append(str(dest))

    if not saved_paths:
        return jsonify({"error": "No valid files uploaded."}), 400

    files = dedup_by_stem(saved_paths)

    if fresh:
        clear_invoice_history()

    return Response(_batch_stream(files, auto_approve), mimetype="text/event-stream")


def _batch_stream(files: list[str], auto_approve: bool):
    """Yield SSE frames for a batch run — shared by api_batch and api_batch_upload."""
    pipeline = _store["pipeline"]
    for event, payload in batch_process_files(
        pipeline,
        files,
        auto_approve,
        _store,
        _normalise_trail,
    ):
        if event == "complete" and "records" in payload:
            payload = {
                **payload,
                "records": [_record_to_dict(r) for r in payload["records"]],
            }
        yield _sse(event, payload)


@app.route("/api/export-csv", methods=["GET"])
def api_export_csv():
    results = _store["results"]
    if not results:
        return jsonify({"error": "No results to export."}), 404

    fieldnames = [
        "invoice_number",
        "vendor",
        "amount",
        "risk_score",
        "decision",
        "processing_time_seconds",
        "explanation",
        "timestamp",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(_record_to_dict(r))

    mem = io.BytesIO(buf.getvalue().encode())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return send_file(
        mem, mimetype="text/csv", as_attachment=True, download_name=f"batch_results_{ts}.csv"
    )


@app.route("/api/reset-db", methods=["POST"])
def api_reset_db():
    db = Path(get_settings().db_path)
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
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    _ensure_pipeline()
    print("\n  Invoice Processing Dashboard")
    print(f"  http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
