"""CLI entry point for the invoice processing pipeline.

Usage:
  python main.py --invoice_path=data/invoices/invoice_1001.txt
  python main.py --batch=data/invoices/ --auto-approve --fresh
"""

import argparse
import csv
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langgraph.checkpoint.memory import MemorySaver

from src.database import clear_invoice_history, init_db
from src.models.audit import BatchResult, ProcessingRecord
from src.pipeline import build_pipeline, process_invoice, resume_after_human_review
from src.processing import (
    auto_decide_hitl,
    batch_process_files,
    build_processing_record,
    collect_batch_files,
    detect_interrupt,
)

logger = structlog.get_logger(__name__)


def _print_result_summary(record: ProcessingRecord) -> None:
    symbol = {"approved": "✓", "rejected": "✗", "escalated": "⚠", "pending_human_review": "⚠"}.get(
        record.decision, "?"
    )
    logger.info(
        "invoice.result",
        symbol=symbol,
        invoice=record.invoice_number,
        vendor=record.vendor,
        amount=f"${record.amount:,.2f}",
        risk_score=record.risk_score,
        decision=record.decision,
        time_seconds=f"{record.processing_time_seconds:.1f}s",
    )
    if record.explanation:
        logger.info("invoice.explanation", text=record.explanation)


def _print_batch_summary(batch: BatchResult, csv_path: str) -> None:
    print()
    print("=" * 70)
    print(f"  BATCH SUMMARY — {batch.total_processed} invoices processed")
    print("=" * 70)
    print(f"  {'Invoice':<20} {'Vendor':<28} {'Amount':>10} {'Risk':>4} {'Decision':<10}")
    print("-" * 70)
    for r in batch.records:
        print(
            f"  {r.invoice_number:<20} {r.vendor:<28} ${r.amount:>9,.2f} "
            f"{r.risk_score:>4} {r.decision:<10}"
        )
    print("=" * 70)
    print(f"  Approved : {batch.approved_count}")
    print(f"  Rejected : {batch.rejected_count}")
    print(f"  Flagged  : {batch.flagged_count}")
    print(f"  Avg time : {batch.avg_processing_time:.1f}s")
    print(f"  CSV      : {csv_path}")
    print("=" * 70)


def _export_csv(batch: BatchResult, output_dir: str = ".") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = str(Path(output_dir) / f"batch_results_{ts}.csv")

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

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in batch.records:
            writer.writerow(
                {
                    "invoice_number": record.invoice_number,
                    "vendor": record.vendor,
                    "amount": record.amount,
                    "risk_score": record.risk_score,
                    "decision": record.decision,
                    "processing_time_seconds": round(record.processing_time_seconds, 3),
                    "explanation": record.explanation,
                    "timestamp": record.timestamp.isoformat(),
                }
            )

    logger.info("batch.csv_exported", path=csv_path, records=len(batch.records))
    return csv_path


def _prompt_human_decision(review_context: dict) -> tuple[str, str]:
    """Display review context and prompt for a decision."""
    print()
    print("=" * 60)
    print("  HUMAN REVIEW REQUIRED")
    print("=" * 60)

    inv = review_context.get("invoice_number", "unknown")
    vendor = review_context.get("vendor", "unknown")
    amount = review_context.get("amount", 0)
    risk_score = review_context.get("risk_score", 0)
    recommendation = review_context.get("recommendation", "unknown")
    narrative = review_context.get("risk_narrative", "")
    flags = review_context.get("flags", [])

    print(f"  Invoice   : {inv}")
    print(f"  Vendor    : {vendor}")
    print(f"  Amount    : ${float(amount):,.2f}")
    print(f"  Risk score: {risk_score}/100  (Recommendation: {recommendation})")
    if narrative:
        print(f"  Narrative : {narrative}")
    if flags:
        print("  Flags:")
        for flag in flags:
            print(f"    - {flag}")
    print("=" * 60)

    valid = {"approved", "rejected", "escalated", "pending_human_review"}
    while True:
        decision = input("  Decision [approved/rejected/escalated]: ").strip().lower()
        if decision in valid:
            break
        print(f"  Invalid. Choose from: {', '.join(sorted(valid))}")

    reasoning = input("  Reasoning (optional): ").strip()
    return decision, reasoning


def run_single_invoice(file_path: str, auto_approve: bool = False) -> ProcessingRecord:
    init_db()
    clear_invoice_history()
    checkpointer = MemorySaver()
    pipeline = build_pipeline(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())

    logger.info("single.start", file_path=file_path, thread_id=thread_id)
    t0 = time.monotonic()

    state = process_invoice(pipeline, file_path, thread_id=thread_id)
    is_interrupted, review_context = detect_interrupt(pipeline, thread_id)

    if is_interrupted:
        if auto_approve:
            decision, reasoning = auto_decide_hitl(state)
        else:
            decision, reasoning = _prompt_human_decision(review_context or {})
        state = resume_after_human_review(pipeline, thread_id, decision, reasoning)

    processing_time = time.monotonic() - t0
    record = build_processing_record(state, processing_time)
    _print_result_summary(record)
    return record


def _cli_hitl_handler(state: dict, review_ctx: dict | None) -> tuple[str, str]:
    """Adapter: wraps _prompt_human_decision for the batch_process_files callback."""
    return _prompt_human_decision(review_ctx or {})


def run_batch(directory: str, auto_approve: bool, fresh: bool = False) -> BatchResult:
    init_db()
    if fresh:
        clear_invoice_history()
        logger.info("batch.cleared_invoice_history")

    checkpointer = MemorySaver()
    pipeline = build_pipeline(checkpointer=checkpointer)

    files = collect_batch_files(Path(directory))

    if not files:
        logger.warning("batch.no_files_found", directory=directory)
        return BatchResult()

    logger.info("batch.start", directory=directory, file_count=len(files))

    records: list[ProcessingRecord] = []
    for event, payload in batch_process_files(
        pipeline,
        files,
        auto_approve,
        hitl_handler=None if auto_approve else _cli_hitl_handler,
    ):
        if event == "progress":
            logger.info("batch.processing", file=payload["file"])
        elif event == "complete":
            records = payload["records"]

    approved = [r for r in records if r.decision == "approved"]
    rejected = [r for r in records if r.decision in ("rejected", "error")]
    flagged = [r for r in records if r.decision in ("escalated", "pending_human_review")]
    avg_time = sum(r.processing_time_seconds for r in records) / len(records) if records else 0.0

    return BatchResult(
        records=records,
        total_processed=len(records),
        approved_count=len(approved),
        rejected_count=len(rejected),
        flagged_count=len(flagged),
        avg_processing_time=avg_time,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoice processing pipeline CLI")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--invoice_path", metavar="PATH", help="Single invoice file")
    mode.add_argument("--batch", metavar="DIR", help="Directory of invoices for batch mode")

    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=False,
        help="Skip HITL, auto-decide based on risk score",
    )
    parser.add_argument(
        "--fresh", action="store_true", default=False, help="Clear invoice_history before batch run"
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.invoice_path:
        run_single_invoice(args.invoice_path, auto_approve=args.auto_approve)
    elif args.batch:
        batch = run_batch(args.batch, auto_approve=args.auto_approve, fresh=args.fresh)
        csv_path = _export_csv(batch, output_dir=".")
        _print_batch_summary(batch, csv_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
