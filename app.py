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

st.set_page_config(
    page_title="Invoice Processing AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = build_pipeline(checkpointer=MemorySaver())
        init_db()
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
    ("extract", "🔎 Extraction"),
    ("validate", "✅ Validation"),
    ("fraud_check", "🛡️ Fraud Detection"),
    ("approve", "👤 Approval"),
    ("payment", "💳 Payment"),
    ("explain", "📝 Explanation"),
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
    c1.metric("Total Processed", total, border=True)
    c2.metric("Auto-Approved Rate", f"{auto_pct}%", border=True)
    c3.metric("Flagged for Review", len(flagged), border=True)
    c4.metric("Avg Risk Score", avg_risk, border=True)


# -- Sidebar --

def _render_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.title("⚙️ Settings")
        st.caption("Current thresholds (from .env config)")

        st.slider("High Risk Threshold", 50, 100, settings.high_risk_threshold,
                   key="sb_high_risk", help="Risk score ≥ this → auto-reject", disabled=True)
        st.slider("Medium Risk Threshold", 10, 60, settings.medium_risk_threshold,
                   key="sb_medium_risk", disabled=True)
        st.slider("Auto-Approve Amount ($)", 100, 10000, int(settings.auto_approve_threshold),
                   step=100, key="sb_auto_amount", disabled=True)
        st.caption("To change thresholds, edit `.env` and restart the app.")

        st.divider()
        auto = st.toggle("⚡ Auto-approve HITL", value=st.session_state.auto_approve,
                          help="Skip human review — auto-decide from risk score")
        st.session_state.auto_approve = auto

        st.divider()
        st.subheader("🔧 System Status")
        st.success("Pipeline ready", icon="✅")
        st.caption(f"Session invoices: {len(st.session_state.results)}")

        if st.button("🗑️ Reset Database", help="Clear inventory.db invoice history"):
            db = Path("inventory.db")
            if db.exists():
                db.unlink()
            init_db()
            st.success("Database reset.")

        st.divider()
        st.subheader("🗺️ Pipeline Diagram")
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

    for i, (key, label) in enumerate(_AGENT_STEPS):
        if done or i < current_idx:
            st.markdown(f"✅ {label}")
        elif i == current_idx:
            st.markdown(f"⏳ **{label}** ← running")
        else:
            st.markdown(f"⬜ {label}")


def _render_review_panel(review_context: dict) -> None:
    inv = review_context.get("invoice") or {}
    amount = review_context.get("amount", 0)
    risk_score = review_context.get("risk_score", 0)
    recommendation = review_context.get("recommendation", "unknown")
    fraud_signals = review_context.get("fraud_signals") or []
    fraud_narrative = review_context.get("fraud_narrative", "")
    validation = review_context.get("validation") or {}

    st.warning("⚠️ Human Review Required", icon="👤")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Invoice #", inv.get("invoice_number", "N/A"))
        c2.metric("Vendor", inv.get("vendor_name", "N/A"))
        c3.metric("Amount", f"${float(amount):,.2f}")

        risk_color = "red" if risk_score >= 70 else "orange" if risk_score >= 30 else "green"
        col_r, col_rec = st.columns(2)
        col_r.markdown(f"**Risk Score**: :{risk_color}[{risk_score}/100]")
        col_rec.markdown(f"**Recommendation**: `{recommendation.upper()}`")

        if fraud_narrative:
            st.info(fraud_narrative)
        if fraud_signals:
            st.markdown("**Fraud Signals:**")
            for sig in fraud_signals:
                st.markdown(f"- {sig}")

        val_issues = validation.get("issues") or []
        if val_issues:
            st.markdown("**Validation Issues:**")
            for issue in val_issues:
                st.markdown(f"- {issue}")

    reasoning = st.text_area("Reasoning (optional)", placeholder="Explain your decision...",
                              key="hitl_reasoning")

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve", type="primary", use_container_width=True, key="hitl_approve"):
            _handle_hitl_decision("approved", reasoning)
    with col_reject:
        if st.button("❌ Reject", type="secondary", use_container_width=True, key="hitl_reject"):
            _handle_hitl_decision("rejected", reasoning)


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
    decision = record.decision
    if decision == "approved":
        st.success(
            f"✅ **{record.invoice_number}** — APPROVED  |  Risk: {record.risk_score}/100  |  "
            f"${record.amount:,.2f}  |  {record.processing_time_seconds:.1f}s"
        )
    elif decision == "rejected":
        st.error(
            f"❌ **{record.invoice_number}** — REJECTED  |  Risk: {record.risk_score}/100  |  "
            f"${record.amount:,.2f}  |  {record.processing_time_seconds:.1f}s"
        )
    else:
        st.info(f"ℹ️ **{record.invoice_number}** — {decision.upper()}  |  Risk: {record.risk_score}/100")

    if record.explanation:
        with st.expander("📝 Decision Explanation"):
            st.write(record.explanation)

    trail = state.get("audit_trail") or []
    if trail:
        with st.expander("📋 Audit Trail"):
            for entry in trail:
                if isinstance(entry, dict):
                    st.markdown(
                        f"**{entry.get('agent', '?')}** → `{entry.get('action', '?')}`: "
                        f"{entry.get('details', '')}"
                    )


# -- Tab 1: Single invoice --

def _render_tab_single() -> None:
    col_left, col_right = st.columns([0.6, 0.4])

    with col_left:
        st.subheader("📄 Invoice Input")
        uploaded = st.file_uploader("Upload invoice file",
                                     type=["txt", "json", "csv", "xml", "pdf"],
                                     key="single_uploader")
        path_input = st.text_input("Or enter file path",
                                    placeholder="data/invoices/invoice_1001.txt",
                                    key="single_path")
        process_clicked = st.button("🚀 Process Invoice", type="primary",
                                     disabled=st.session_state.is_interrupted)

    with col_right:
        st.subheader("🤖 Agent Progress")
        if st.session_state.is_interrupted and st.session_state.review_context:
            _render_review_panel(st.session_state.review_context)
            return
        if st.session_state.last_record is not None:
            _render_agent_viz(None, done=True)
        else:
            st.caption("Upload an invoice and click Process.")

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
            with st.spinner("Running pipeline..."):
                t0 = time.monotonic()
                state = process_invoice(pipeline, file_path, thread_id=thread_id)
                elapsed = time.monotonic() - t0

        is_interrupted, review_context = _detect_interrupt(pipeline, thread_id)

        if is_interrupted:
            if st.session_state.auto_approve:
                fraud = state.get("fraud_result") or {}
                risk = int(fraud.get("risk_score") or 0)
                settings = get_settings()
                decision = "approved" if risk < settings.medium_risk_threshold else "rejected"
                _handle_hitl_decision(decision, f"Auto-decided: risk score {risk}")
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
    st.subheader("📦 Batch Invoice Processing")

    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        batch_dir = st.text_input("Invoice directory", value="data/invoices", key="batch_dir")
    with col2:
        auto_batch = st.checkbox("Auto-approve HITL", value=True, key="batch_auto")
    with col3:
        fresh_batch = st.checkbox("Fresh run", value=True, key="batch_fresh",
                                   help="Clear invoice history to avoid false-positive duplicates")

    run_clicked = st.button("🚀 Process All", type="primary")

    if run_clicked:
        dir_path = Path(batch_dir.strip())
        if not dir_path.is_dir():
            st.error(f"Directory not found: `{batch_dir}`")
            return

        extensions = ("*.txt", "*.json", "*.csv", "*.xml", "*.pdf")
        files: list[str] = []
        for ext in extensions:
            files.extend(glob.glob(str(dir_path / ext)))
        files = sorted(set(files))

        if not files:
            st.warning("No invoice files found in that directory.")
            return

        if fresh_batch:
            from src.database import clear_invoice_history
            clear_invoice_history()

        pipeline = st.session_state.pipeline
        settings = get_settings()
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
                    fraud = state.get("fraud_result") or {}
                    risk = int(fraud.get("risk_score") or 0)
                    if auto_batch:
                        decision = "approved" if risk < settings.medium_risk_threshold else "rejected"
                        reasoning = f"Batch auto: risk {risk}"
                    else:
                        decision, reasoning = "rejected", "Batch HITL skipped"
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
        status_box.success(f"✅ Batch complete — {len(records)} invoices processed")

    # results table
    batch_records: list[ProcessingRecord] = st.session_state.get("batch_records", [])
    if batch_records:
        import pandas as pd

        approved_n = sum(1 for r in batch_records if r.decision == "approved")
        rejected_n = sum(1 for r in batch_records if r.decision in ("rejected", "error"))
        flagged_n = sum(1 for r in batch_records if r.decision in ("escalated", "pending_human_review"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(batch_records))
        m2.metric("Approved", approved_n)
        m3.metric("Rejected", rejected_n)
        m4.metric("Flagged", flagged_n)

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
                bg = "background-color: #d4edda"
            elif row["Decision"] in ("rejected", "error"):
                bg = "background-color: #f8d7da"
            else:
                bg = "background-color: #fff3cd"
            return [bg] * len(row)

        def _risk_color(val):
            try:
                score = int(str(val).split("/")[0])
            except (ValueError, IndexError):
                return ""
            if score >= 70:
                return "color: #dc3545; font-weight: bold"
            elif score >= 30:
                return "color: #856404; font-weight: bold"
            return "color: #155724"

        styled = df.style.apply(_row_color, axis=1).map(_risk_color, subset=["Risk Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button("⬇️ Download CSV", data=_records_to_csv(batch_records),
                            file_name=f"batch_results_{ts}.csv", mime="text/csv")


# -- Tab 3: Analytics --

def _render_tab_analytics() -> None:
    st.subheader("📊 Analytics")
    results = st.session_state.results

    if not results:
        st.info("No invoices processed yet. Process some invoices first.")
        return

    import pandas as pd

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Risk Score Distribution**")
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
        labels = ["0-9", "10-19", "20-29", "30-39", "40-49",
                   "50-59", "60-69", "70-79", "80-89", "90-100"]
        risk_df = pd.DataFrame({"Risk Score": [r.risk_score for r in results]})
        risk_df["Bucket"] = pd.cut(risk_df["Risk Score"], bins=bins, labels=labels, right=False)
        bucket_counts = risk_df["Bucket"].value_counts().sort_index().rename("Count")
        st.bar_chart(bucket_counts)

    with col2:
        st.markdown("**Decision Breakdown**")
        decisions: dict[str, int] = {}
        for r in results:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        dec_df = pd.DataFrame({"Count": decisions}).rename_axis("Decision")
        st.bar_chart(dec_df)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Top Vendors by Invoice Count**")
        vendors: dict[str, int] = {}
        for r in results:
            vendors[r.vendor] = vendors.get(r.vendor, 0) + 1
        vendor_df = (
            pd.DataFrame({"Vendor": list(vendors.keys()), "Invoices": list(vendors.values())})
            .sort_values("Invoices", ascending=False).head(10)
        )
        st.dataframe(vendor_df, use_container_width=True, hide_index=True)

    with col4:
        st.markdown("**Cost Savings Calculator**")
        manual_cost = st.number_input("Manual processing cost per invoice ($)",
                                       min_value=1.0, max_value=500.0, value=15.0, step=1.0)
        auto_approved = sum(1 for r in results if r.decision == "approved")
        savings = auto_approved * manual_cost
        st.metric("Estimated Savings", f"${savings:,.2f}",
                   help=f"{auto_approved} auto-approved × ${manual_cost:.2f}")
        st.caption(f"{auto_approved} of {len(results)} invoices required no human review.")

    # rejection keyword analysis
    rejected = [r for r in results if r.decision in ("rejected", "error")]
    if rejected:
        st.markdown("**Rejection Reason Keywords**")
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
    st.subheader("📋 Audit Trail")
    all_entries = st.session_state.audit_entries

    if not all_entries:
        st.info("No audit entries yet.")
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

    _action_icon = {
        "auto_approve": "✅", "auto_reject": "❌", "human_review": "👤",
        "retry": "🔄", "rejected": "🚫", "approved": "✅", "error": "💥",
    }

    for entry in reversed(filtered):
        agent = entry.get("agent", "unknown")
        action = entry.get("action", "")
        details = entry.get("details", "")
        invoice = entry.get("invoice", "")
        ts = entry.get("timestamp", "")
        icon = _action_icon.get(action, "📌")

        with st.container(border=True):
            left, right = st.columns([0.8, 0.2])
            with left:
                st.markdown(f"{icon} **{agent}** → `{action}`")
                if invoice:
                    st.caption(f"Invoice: {invoice}")
                if details:
                    st.write(details)
            with right:
                if ts:
                    st.caption(ts[:19].replace("T", " "))


# -- Main layout --

_render_kpi_row()
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Process Invoice",
    "📦 Batch Processing",
    "📊 Analytics",
    "📋 Audit Trail",
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
