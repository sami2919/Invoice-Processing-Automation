"""Streamlit dashboard for the invoice processing pipeline."""

import csv
import glob
import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from src.config import get_settings
from src.database import init_db
from src.models.audit import ProcessingRecord
from src.pipeline import (
    build_pipeline,
    get_pipeline_diagram,
    process_invoice,
    resume_after_human_review,
)
from src.theme import (
    COLORS,
    EXECUTIVE_CSS,
    decision_badge,
    decision_color,
    kpi_card,
    page_header,
    risk_badge,
    risk_color,
    section_header,
)

_BATCH_EXTENSIONS = ("*.txt", "*.json", "*.csv", "*.xml", "*.pdf")


def collect_batch_files(dir_path: Path) -> list[str]:
    """Collect invoice files from a directory, deduplicating by stem.

    When both a PDF and a text-based format exist for the same stem,
    the non-PDF version is preferred (text extraction is more reliable).
    """
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


st.set_page_config(
    page_title="Invoice Processing AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)


def _init_session_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = build_pipeline(checkpointer=MemorySaver())
        init_db()
        from src.database import clear_invoice_history
        clear_invoice_history()
    defaults = {
        "results": [],
        "audit_entries": [],
        "is_interrupted": False,
        "review_context": None,
        "current_thread_id": None,
        "last_state": None,
        "last_record": None,
        "auto_approve": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


def _detect_interrupt(pipeline, thread_id: str) -> tuple[bool, Optional[dict]]:
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


def _records_to_csv(records: list[ProcessingRecord]) -> bytes:
    fieldnames = [
        "invoice_number", "vendor", "amount", "risk_score",
        "decision", "processing_time_seconds", "explanation", "timestamp",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        writer.writerow({
            "invoice_number": r.invoice_number,
            "vendor": r.vendor,
            "amount": r.amount,
            "risk_score": r.risk_score,
            "decision": r.decision,
            "processing_time_seconds": round(r.processing_time_seconds, 3),
            "explanation": r.explanation,
            "timestamp": r.timestamp.isoformat(),
        })
    return buf.getvalue().encode()


_AGENT_STEPS = [
    ("extract", "Extraction"),
    ("validate", "Validation"),
    ("fraud_check", "Fraud Detection"),
    ("approve", "Approval"),
    ("payment", "Payment"),
    ("explain", "Explanation"),
]
_AGENT_ORDER = [key for key, _ in _AGENT_STEPS]


# -- KPI row --

def _render_kpi_row() -> None:
    results = st.session_state.results
    total = len(results)
    approved = [r for r in results if r.decision == "approved"]
    flagged = [r for r in results if r.decision in ("escalated", "pending_human_review")]
    avg_risk = int(sum(r.risk_score for r in results) / total) if total else 0
    auto_pct = int(len(approved) / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Processed", str(total), accent=COLORS["accent"]),
                     unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Auto-Approved", f"{auto_pct}%",
                             subtitle=f"{len(approved)} of {total}" if total else "",
                             accent=COLORS["success"]),
                     unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Flagged for Review", str(len(flagged)), accent=COLORS["warning"]),
                     unsafe_allow_html=True)
    with c4:
        risk_accent = COLORS["success"] if avg_risk < 30 else COLORS["warning"] if avg_risk < 70 else COLORS["danger"]
        st.markdown(kpi_card("Avg Risk Score", f"{avg_risk}/100", accent=risk_accent),
                     unsafe_allow_html=True)


# -- Sidebar --

def _render_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.markdown(
            f'<div style="padding:8px 0 16px;">'
            f'<div style="color:#F1F5F9; font-size:1.15rem; font-weight:700; '
            f'letter-spacing:-0.01em;">Settings</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="color:{COLORS["muted"]}; font-size:0.72rem; font-weight:600; '
            f'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Risk Thresholds</div>',
            unsafe_allow_html=True,
        )
        st.slider("High Risk", 50, 100, settings.high_risk_threshold,
                   key="sb_high_risk", help="Risk score >= this -> auto-reject", disabled=True)
        st.slider("Medium Risk", 10, 60, settings.medium_risk_threshold,
                   key="sb_medium_risk", disabled=True)
        st.slider("Auto-Approve Limit ($)", 100, 10000, int(settings.auto_approve_threshold),
                   step=100, key="sb_auto_amount", disabled=True)
        st.caption("Edit `.env` to change thresholds.")

        st.divider()
        auto = st.toggle("Auto-approve HITL", value=st.session_state.auto_approve,
                          help="Skip human review -- auto-decide from risk score")
        st.session_state.auto_approve = auto

        st.divider()
        st.markdown(
            f'<div style="color:{COLORS["muted"]}; font-size:0.72rem; font-weight:600; '
            f'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">System</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
            f'<span style="width:8px; height:8px; border-radius:50%; background:#34D399; '
            f'display:inline-block;"></span>'
            f'<span style="color:#CBD5E1; font-size:0.82rem;">Pipeline ready</span>'
            f'</div>'
            f'<div style="color:#64748B; font-size:0.78rem;">'
            f'{len(st.session_state.results)} invoices processed this session</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if st.button("Reset Database", help="Clear inventory.db invoice history",
                      use_container_width=True):
            db = Path("inventory.db")
            if db.exists():
                db.unlink()
            init_db()
            st.success("Database reset.")

        st.divider()
        st.markdown(
            f'<div style="color:{COLORS["muted"]}; font-size:0.72rem; font-weight:600; '
            f'text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Pipeline</div>',
            unsafe_allow_html=True,
        )
        try:
            diagram_bytes = get_pipeline_diagram(st.session_state.pipeline)
            st.image(diagram_bytes, use_container_width=True)
        except Exception:
            st.caption("Install graphviz to see diagram.")


# -- Tab 1 helpers --

def _render_agent_viz(current_agent: Optional[str], done: bool = False) -> None:
    if current_agent in _AGENT_ORDER:
        current_idx = _AGENT_ORDER.index(current_agent)
    else:
        current_idx = len(_AGENT_ORDER) - 1 if done else -1

    steps_html = ""
    for i, (key, label) in enumerate(_AGENT_STEPS):
        if done or i < current_idx:
            bg = COLORS["success"]
            ring = f"background:{bg}; color:#fff;"
            icon = "&#10003;"
            text_color = COLORS["text"]
            connector_color = COLORS["success"]
        elif i == current_idx:
            bg = COLORS["accent"]
            ring = f"background:{bg}; color:#fff; box-shadow:0 0 0 4px rgba(59,130,246,0.2);"
            icon = str(i + 1)
            text_color = COLORS["text"]
            connector_color = COLORS["border"]
        else:
            ring = (f"background:{COLORS['background']}; color:{COLORS['muted']}; "
                    f"border:2px solid {COLORS['border']};")
            icon = str(i + 1)
            text_color = COLORS["muted"]
            connector_color = COLORS["border"]

        connector = ""
        if i < len(_AGENT_STEPS) - 1:
            connector = (
                f'<div style="width:2px; height:16px; background:{connector_color}; '
                f'margin-left:15px;"></div>'
            )

        steps_html += (
            f'<div style="display:flex; align-items:center; gap:12px;">'
            f'<div style="width:32px; height:32px; border-radius:50%; {ring} '
            f'display:flex; align-items:center; justify-content:center; '
            f'font-size:0.75rem; font-weight:700; flex-shrink:0;">{icon}</div>'
            f'<div style="color:{text_color}; font-size:0.85rem; font-weight:500;">{label}</div>'
            f'</div>'
            f'{connector}'
        )

    st.markdown(
        f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
        f'border-radius:10px; padding:20px;">{steps_html}</div>',
        unsafe_allow_html=True,
    )


def _render_review_panel(review_context: dict) -> None:
    inv = review_context.get("invoice") or {}
    amount = review_context.get("amount", 0)
    risk_score_val = review_context.get("risk_score", 0)
    recommendation = review_context.get("recommendation", "unknown")
    fraud_signals = review_context.get("fraud_signals") or []
    fraud_narrative = review_context.get("fraud_narrative", "")
    validation = review_context.get("validation") or {}

    st.markdown(
        f'<div style="background:{COLORS["warning_bg"]}; border:1px solid {COLORS["warning"]}33; '
        f'border-radius:10px; padding:14px 20px; margin-bottom:16px;">'
        f'<div style="color:{COLORS["warning"]}; font-weight:700; font-size:0.9rem;">'
        f'Human Review Required</div>'
        f'<div style="color:{COLORS["text_secondary"]}; font-size:0.8rem; margin-top:2px;">'
        f'This invoice requires manual approval before proceeding.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    inv_num = inv.get("invoice_number", "N/A")
    vendor = inv.get("vendor_name", "N/A")
    st.markdown(
        f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
        f'border-radius:10px; padding:20px; margin-bottom:16px;">'
        f'<div style="display:flex; gap:24px; flex-wrap:wrap; margin-bottom:16px;">'
        f'<div><div style="color:{COLORS["text_secondary"]}; font-size:0.72rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.06em;">Invoice</div>'
        f'<div style="color:{COLORS["text"]}; font-size:1rem; font-weight:600; margin-top:2px;">'
        f'{inv_num}</div></div>'
        f'<div><div style="color:{COLORS["text_secondary"]}; font-size:0.72rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.06em;">Vendor</div>'
        f'<div style="color:{COLORS["text"]}; font-size:1rem; font-weight:600; margin-top:2px;">'
        f'{vendor}</div></div>'
        f'<div><div style="color:{COLORS["text_secondary"]}; font-size:0.72rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.06em;">Amount</div>'
        f'<div style="color:{COLORS["text"]}; font-size:1rem; font-weight:600; margin-top:2px;">'
        f'${float(amount):,.2f}</div></div>'
        f'</div>'
        f'<div style="display:flex; gap:16px; align-items:center;">'
        f'<span style="color:{COLORS["text_secondary"]}; font-size:0.82rem; font-weight:500;">'
        f'Risk:</span> {risk_badge(risk_score_val)}'
        f'<span style="color:{COLORS["text_secondary"]}; font-size:0.82rem; font-weight:500; '
        f'margin-left:8px;">Recommendation:</span> {decision_badge(recommendation)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if fraud_narrative:
        st.info(fraud_narrative)
    if fraud_signals:
        signals_html = "".join(
            f'<li style="color:{COLORS["text"]}; font-size:0.85rem; margin-bottom:4px;">{s}</li>'
            for s in fraud_signals
        )
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:6px;">Fraud Signals</div>'
            f'<ul style="margin:0; padding-left:20px;">{signals_html}</ul></div>',
            unsafe_allow_html=True,
        )

    val_issues = validation.get("issues") or []
    if val_issues:
        issues_html = "".join(
            f'<li style="color:{COLORS["text"]}; font-size:0.85rem; margin-bottom:4px;">{v}</li>'
            for v in val_issues
        )
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:6px;">Validation Issues</div>'
            f'<ul style="margin:0; padding-left:20px;">{issues_html}</ul></div>',
            unsafe_allow_html=True,
        )

    reasoning = st.text_area("Reasoning (optional)", placeholder="Explain your decision...",
                              key="hitl_reasoning")

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("Approve", type="primary", use_container_width=True, key="hitl_approve"):
            _handle_hitl_decision("approved", reasoning)
    with col_reject:
        if st.button("Reject", type="secondary", use_container_width=True, key="hitl_reject"):
            _handle_hitl_decision("rejected", reasoning)


def _auto_decide_hitl(state: dict) -> tuple[str, str]:
    """Decide on behalf of human reviewer in auto-approve mode."""
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


def _handle_hitl_decision(decision: str, reasoning: str) -> None:
    thread_id = st.session_state.current_thread_id
    pipeline = st.session_state.pipeline
    t0 = time.monotonic()

    with st.spinner(f"Resuming with decision: {decision}..."):
        final_state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

    elapsed = time.monotonic() - t0
    st.session_state.audit_entries.extend(
        _normalise_trail(final_state.get("audit_trail") or [], final_state)
    )
    record = _build_processing_record(final_state, elapsed)
    st.session_state.results.append(record)
    st.session_state.last_record = record
    st.session_state.last_state = final_state
    st.session_state.is_interrupted = False
    st.session_state.review_context = None
    st.session_state.current_thread_id = None
    st.rerun()


def _render_result_banner(record: ProcessingRecord, state: dict) -> None:
    dec = record.decision
    bg = decision_color(dec)
    bg_light = {
        "approved": COLORS["success_bg"],
        "rejected": COLORS["danger_bg"],
    }.get(dec, COLORS["warning_bg"])

    st.markdown(
        f'<div style="background:{bg_light}; border:1px solid {bg}33; border-left:4px solid {bg}; '
        f'border-radius:8px; padding:16px 20px; margin-bottom:16px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'flex-wrap:wrap; gap:12px;">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<span style="font-weight:700; font-size:1rem; color:{COLORS["text"]};">'
        f'{record.invoice_number}</span>'
        f'{decision_badge(dec)}'
        f'</div>'
        f'<div style="display:flex; gap:20px; color:{COLORS["text_secondary"]}; font-size:0.82rem;">'
        f'<span>Risk: {risk_badge(record.risk_score)}</span>'
        f'<span style="font-weight:500;">${record.amount:,.2f}</span>'
        f'<span>{record.processing_time_seconds:.1f}s</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if record.explanation:
        with st.expander("Decision Explanation"):
            st.markdown(
                f'<div style="color:{COLORS["text"]}; font-size:0.85rem; '
                f'line-height:1.6; white-space:pre-wrap;">{record.explanation}</div>',
                unsafe_allow_html=True,
            )

    trail = state.get("audit_trail") or []
    if trail:
        with st.expander("Audit Trail"):
            for entry in trail:
                if isinstance(entry, dict):
                    st.markdown(
                        f"**{entry.get('agent', '?')}** -> `{entry.get('action', '?')}`: "
                        f"{entry.get('details', '')}",
                        unsafe_allow_html=True,
                    )


# -- Tab 1: Single invoice --

def _render_tab_single() -> None:
    col_left, col_right = st.columns([0.6, 0.4])

    with col_left:
        st.markdown(section_header("Invoice Input", "Upload a file or enter a path to process"),
                     unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload invoice file",
                                     type=["txt", "json", "csv", "xml", "pdf"],
                                     key="single_uploader")
        path_input = st.text_input("Or enter file path",
                                    placeholder="data/invoices/invoice_1001.txt",
                                    key="single_path")
        process_clicked = st.button("Process Invoice", type="primary",
                                     disabled=st.session_state.is_interrupted)

    with col_right:
        st.markdown(section_header("Pipeline Progress"), unsafe_allow_html=True)
        if st.session_state.is_interrupted and st.session_state.review_context:
            _render_review_panel(st.session_state.review_context)
            return
        if st.session_state.last_record is not None:
            _render_agent_viz(None, done=True)
        else:
            st.markdown(
                f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
                f'border-radius:10px; padding:32px; text-align:center;">'
                f'<div style="color:{COLORS["muted"]}; font-size:0.9rem;">'
                f'Upload an invoice and click Process to begin.</div></div>',
                unsafe_allow_html=True,
            )

    if process_clicked:
        file_path: Optional[str] = None
        if uploaded is not None:
            tmp = Path(f"/tmp/{uploaded.name}")
            tmp.write_bytes(uploaded.read())
            file_path = str(tmp)
        elif path_input.strip():
            file_path = path_input.strip()

        if not file_path:
            st.error("Please upload a file or enter a file path.")
            return
        if not Path(file_path).exists():
            st.error(f"File not found: `{file_path}`")
            return

        st.session_state.last_record = None
        st.session_state.last_state = None

        thread_id = str(uuid.uuid4())
        st.session_state.current_thread_id = thread_id
        pipeline = st.session_state.pipeline

        with col_right:
            with st.status("Processing invoice...", expanded=True) as status:
                status.update(label="Extracting invoice data...", state="running")
                t0 = time.monotonic()
                state = process_invoice(pipeline, file_path, thread_id=thread_id)
                elapsed = time.monotonic() - t0
                status.update(label=f"Pipeline complete ({elapsed:.1f}s)", state="complete")

        is_interrupted, review_context = _detect_interrupt(pipeline, thread_id)

        if is_interrupted:
            if st.session_state.auto_approve:
                decision, reasoning = _auto_decide_hitl(state)
                _handle_hitl_decision(decision, reasoning)
            else:
                st.session_state.is_interrupted = True
                st.session_state.review_context = review_context
                st.rerun()
        else:
            trail = state.get("audit_trail") or []
            st.session_state.audit_entries.extend(_normalise_trail(trail, state))
            record = _build_processing_record(state, elapsed)
            st.session_state.results.append(record)
            st.session_state.last_record = record
            st.session_state.last_state = state
            st.rerun()

    if st.session_state.last_record is not None and not st.session_state.is_interrupted:
        _render_result_banner(st.session_state.last_record, st.session_state.last_state or {})


# -- Tab 2: Batch --

def _render_tab_batch() -> None:
    st.markdown(section_header("Batch Processing", "Process multiple invoices from a directory"),
                 unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        batch_dir = st.text_input("Invoice directory", value="data/invoices", key="batch_dir")
    with col2:
        auto_batch = st.checkbox("Auto-approve HITL", value=True, key="batch_auto")
    with col3:
        fresh_batch = st.checkbox("Fresh run", value=True, key="batch_fresh",
                                   help="Clear invoice history to avoid false-positive duplicates")

    run_clicked = st.button("Process All", type="primary")

    if run_clicked:
        dir_path = Path(batch_dir.strip())
        if not dir_path.is_dir():
            st.error(f"Directory not found: `{batch_dir}`")
            return

        files = collect_batch_files(dir_path)

        if not files:
            st.warning("No invoice files found in that directory.")
            return

        if fresh_batch:
            from src.database import clear_invoice_history
            clear_invoice_history()

        pipeline = st.session_state.pipeline
        progress_bar = st.progress(0, text="Starting...")
        status_box = st.empty()
        records: list[ProcessingRecord] = []

        for i, fp in enumerate(files):
            fname = Path(fp).name
            status_box.info(f"Processing {i + 1}/{len(files)}: `{fname}`")
            thread_id = str(uuid.uuid4())
            t0 = time.monotonic()
            try:
                state = process_invoice(pipeline, fp, thread_id=thread_id)
                is_interrupted, _ = _detect_interrupt(pipeline, thread_id)

                if is_interrupted:
                    decision, reasoning = _auto_decide_hitl(state)
                    state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

                elapsed = time.monotonic() - t0
                record = _build_processing_record(state, elapsed)
                records.append(record)
                st.session_state.audit_entries.extend(
                    _normalise_trail(state.get("audit_trail") or [], state)
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                records.append(ProcessingRecord(
                    invoice_number=Path(fp).stem, vendor="error", amount=0.0,
                    risk_score=0, decision="error",
                    processing_time_seconds=elapsed, explanation=str(exc),
                ))

            progress_bar.progress((i + 1) / len(files), text=f"{i + 1}/{len(files)} done")

        st.session_state.results.extend(records)
        st.session_state["batch_records"] = records
        status_box.success(f"Batch complete -- {len(records)} invoices processed")

    # results table
    batch_records: list[ProcessingRecord] = st.session_state.get("batch_records", [])
    if batch_records:
        import pandas as pd

        approved_n = sum(1 for r in batch_records if r.decision == "approved")
        rejected_n = sum(1 for r in batch_records if r.decision in ("rejected", "error"))
        flagged_n = sum(1 for r in batch_records if r.decision in ("escalated", "pending_human_review"))

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(kpi_card("Total", str(len(batch_records)), accent=COLORS["accent"]),
                         unsafe_allow_html=True)
        with m2:
            st.markdown(kpi_card("Approved", str(approved_n), accent=COLORS["success"]),
                         unsafe_allow_html=True)
        with m3:
            st.markdown(kpi_card("Rejected", str(rejected_n), accent=COLORS["danger"]),
                         unsafe_allow_html=True)
        with m4:
            st.markdown(kpi_card("Flagged", str(flagged_n), accent=COLORS["warning"]),
                         unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        df_data = [{
            "Invoice": r.invoice_number,
            "Vendor": r.vendor,
            "Amount ($)": f"${r.amount:,.2f}",
            "Risk Score": f"{r.risk_score}/100",
            "Decision": r.decision,
            "Time (s)": f"{r.processing_time_seconds:.1f}",
        } for r in batch_records]
        df = pd.DataFrame(df_data)

        def _row_color(row):
            if row["Decision"] == "approved":
                bg = f"background-color: {COLORS['success']}18"
            elif row["Decision"] in ("rejected", "error"):
                bg = f"background-color: {COLORS['danger']}18"
            else:
                bg = f"background-color: {COLORS['warning']}18"
            return [bg] * len(row)

        def _risk_color_cell(val):
            try:
                score = int(str(val).split("/")[0])
            except (ValueError, IndexError):
                return ""
            c = risk_color(score)
            return f"color: {c}; font-weight: bold"

        styled = df.style.apply(_row_color, axis=1).map(_risk_color_cell, subset=["Risk Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button("Download CSV", data=_records_to_csv(batch_records),
                            file_name=f"batch_results_{ts}.csv", mime="text/csv")


# -- Tab 3: Analytics --

def _render_tab_analytics() -> None:
    st.markdown(section_header("Analytics", "Insights from processed invoices"),
                 unsafe_allow_html=True)
    results = st.session_state.results

    if not results:
        st.markdown(
            f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
            f'border-radius:10px; padding:40px; text-align:center;">'
            f'<div style="color:{COLORS["muted"]}; font-size:0.9rem;">'
            f'No invoices processed yet. Process some invoices to see analytics.</div></div>',
            unsafe_allow_html=True,
        )
        return

    import pandas as pd

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:8px;">Risk Score Distribution</div>',
            unsafe_allow_html=True,
        )
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
        labels = ["0-9", "10-19", "20-29", "30-39", "40-49",
                   "50-59", "60-69", "70-79", "80-89", "90-100"]
        risk_df = pd.DataFrame({"Risk Score": [r.risk_score for r in results]})
        risk_df["Bucket"] = pd.cut(risk_df["Risk Score"], bins=bins, labels=labels, right=False)
        bucket_counts = risk_df["Bucket"].value_counts().sort_index().rename("Count")
        st.bar_chart(bucket_counts, color=COLORS["accent"])

    with col2:
        st.markdown(
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:8px;">Decision Breakdown</div>',
            unsafe_allow_html=True,
        )
        decisions: dict[str, int] = {}
        for r in results:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        dec_df = pd.DataFrame({"Count": decisions}).rename_axis("Decision")
        st.bar_chart(dec_df, color=COLORS["accent"])

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:8px;">Top Vendors by Invoice Count</div>',
            unsafe_allow_html=True,
        )
        vendors: dict[str, int] = {}
        for r in results:
            vendors[r.vendor] = vendors.get(r.vendor, 0) + 1
        vendor_df = (
            pd.DataFrame({"Vendor": list(vendors.keys()), "Invoices": list(vendors.values())})
            .sort_values("Invoices", ascending=False).head(10)
        )
        st.dataframe(vendor_df, use_container_width=True, hide_index=True)

    with col4:
        st.markdown(
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin-bottom:8px;">Cost Savings Calculator</div>',
            unsafe_allow_html=True,
        )
        manual_cost = st.number_input("Manual processing cost per invoice ($)",
                                       min_value=1.0, max_value=500.0, value=15.0, step=1.0)
        auto_approved = sum(1 for r in results if r.decision == "approved")
        savings = auto_approved * manual_cost
        st.markdown(
            kpi_card("Estimated Savings", f"${savings:,.2f}",
                     subtitle=f"{auto_approved} auto-approved x ${manual_cost:.2f}",
                     accent=COLORS["success"]),
            unsafe_allow_html=True,
        )
        st.caption(f"{auto_approved} of {len(results)} invoices required no human review.")

    # rejection keyword analysis
    rejected = [r for r in results if r.decision in ("rejected", "error")]
    if rejected:
        st.markdown(
            f'<div style="color:{COLORS["text"]}; font-weight:600; font-size:0.85rem; '
            f'margin:20px 0 8px;">Rejection Reason Keywords</div>',
            unsafe_allow_html=True,
        )
        stop = {"the", "a", "an", "is", "was", "and", "or", "to", "of", "in",
                "for", "with", "this", "that", "it", "be", "are", "by", "at"}
        word_freq: dict[str, int] = {}
        for r in rejected:
            for word in r.explanation.lower().split():
                w = word.strip(".,;:()[]\"'")
                if len(w) > 3 and w not in stop:
                    word_freq[w] = word_freq.get(w, 0) + 1
        if word_freq:
            top = sorted(word_freq.items(), key=lambda x: -x[1])[:10]
            kw_df = pd.DataFrame(top, columns=["Keyword", "Frequency"])
            st.dataframe(kw_df, use_container_width=True, hide_index=True)


# -- Tab 4: Audit trail --

def _render_tab_audit() -> None:
    st.markdown(section_header("Audit Trail", "Complete history of all pipeline actions"),
                 unsafe_allow_html=True)
    all_entries = st.session_state.audit_entries

    if not all_entries:
        st.markdown(
            f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
            f'border-radius:10px; padding:40px; text-align:center;">'
            f'<div style="color:{COLORS["muted"]}; font-size:0.9rem;">'
            f'No audit entries yet. Process an invoice to see the trail.</div></div>',
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        invoice_filter = st.text_input("Filter by invoice #", key="af_invoice")
    with col2:
        agents = ["(all)"] + sorted(set(e.get("agent", "") for e in all_entries))
        agent_filter = st.selectbox("Filter by agent", agents, key="af_agent")
    with col3:
        actions = ["(all)"] + sorted(set(e.get("action", "") for e in all_entries))
        action_filter = st.selectbox("Filter by action", actions, key="af_action")

    filtered = all_entries
    if invoice_filter.strip():
        filtered = [e for e in filtered
                     if invoice_filter.strip().lower() in e.get("invoice", "").lower()]
    if agent_filter != "(all)":
        filtered = [e for e in filtered if e.get("agent") == agent_filter]
    if action_filter != "(all)":
        filtered = [e for e in filtered if e.get("action") == action_filter]

    st.caption(f"Showing {len(filtered)} of {len(all_entries)} entries")

    _action_colors = {
        "auto_approve": COLORS["success"], "auto_reject": COLORS["danger"],
        "human_review": COLORS["warning"], "retry": COLORS["accent"],
        "rejected": COLORS["danger"], "approved": COLORS["success"],
        "error": COLORS["danger"],
    }

    for entry in reversed(filtered):
        agent = entry.get("agent", "unknown")
        action = entry.get("action", "")
        details = entry.get("details", "")
        invoice = entry.get("invoice", "")
        ts = entry.get("timestamp", "")
        dot_color = _action_colors.get(action, COLORS["muted"])

        st.markdown(
            f'<div style="background:{COLORS["surface"]}; border:1px solid {COLORS["border"]}; '
            f'border-radius:8px; padding:14px 18px; margin-bottom:8px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
            f'<div>'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="width:8px; height:8px; border-radius:50%; background:{dot_color}; '
            f'display:inline-block;"></span>'
            f'<span style="font-weight:600; font-size:0.85rem; color:{COLORS["text"]};">'
            f'{agent}</span>'
            f'<span style="color:{COLORS["text_secondary"]}; font-size:0.82rem;">'
            f'-></span>'
            f'<code style="background:{COLORS["background"]}; padding:2px 8px; '
            f'border-radius:4px; font-size:0.78rem; color:{COLORS["text_secondary"]};">'
            f'{action}</code>'
            f'</div>'
            f'{"<div style=&quot;color:" + COLORS["text_secondary"] + "; font-size:0.78rem; margin-top:4px; margin-left:16px;&quot;>" + invoice + "</div>" if invoice else ""}'
            f'{"<div style=&quot;color:" + COLORS["text"] + "; font-size:0.82rem; margin-top:4px; margin-left:16px;&quot;>" + details + "</div>" if details else ""}'
            f'</div>'
            f'<div style="color:{COLORS["muted"]}; font-size:0.72rem; white-space:nowrap;">'
            f'{ts[:19].replace("T", " ") if ts else ""}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# -- Main layout --

st.markdown(page_header(), unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

_render_kpi_row()
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Process Invoice",
    "Batch Processing",
    "Analytics",
    "Audit Trail",
])

with tab1:
    _render_tab_single()
with tab2:
    _render_tab_batch()
with tab3:
    _render_tab_analytics()
with tab4:
    _render_tab_audit()

_render_sidebar()
