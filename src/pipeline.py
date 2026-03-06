"""LangGraph pipeline — assembles agent nodes into a compiled graph.

Flow: extract -> validate -> [retry | fraud_check] -> approve -> [payment | reject] -> explain -> END
"""

import uuid
from typing import Optional

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents.approval import approval_node
from src.agents.explanation import explanation_node
from src.agents.extraction import extraction_node
from src.agents.fraud import fraud_detection_node
from src.agents.payment import payment_node
from src.agents.validation import validation_node
from src.config import get_settings
from src.database import init_db
from src.models.invoice import ApprovalDecision
from src.models.state import InvoiceState
from src.tools.file_parser import parse_file

logger = structlog.get_logger(__name__)


def rejection_node(state: InvoiceState) -> dict:
    """Ensure approval_decision exists and log the rejection."""
    existing = state.get("approval_decision") or {}
    result: dict = {
        "current_agent": "rejection",
        "audit_trail": [
            {
                "agent": "rejection",
                "action": "rejected",
                "details": existing.get("reasoning", "Rejected after validation failure"),
            }
        ],
    }
    if not existing:
        fallback = ApprovalDecision(
            status="rejected",
            reasoning="Rejected: validation failed after all retries were exhausted.",
            approver="system",
        )
        result["approval_decision"] = fallback.model_dump()
    return result


def retry_extraction_node(state: InvoiceState) -> dict:
    """Bump retry counter and build feedback for extraction."""
    retries = state.get("extraction_retries", 0)
    validation = state.get("validation_result") or {}
    issues = validation.get("issues", [])

    feedback = "Previous extraction had these issues:\n"
    feedback += "\n".join(f"- {i}" for i in issues)
    feedback += "\nPlease re-examine the source text and correct these specific problems."

    logger.info("extraction.retry", attempt=retries + 1, issue_count=len(issues))

    return {
        "extraction_retries": retries + 1,
        "extraction_feedback": feedback,
        "audit_trail": [
            {
                "agent": "retry_extraction",
                "action": "retry",
                "details": f"Retry #{retries + 1}",
            }
        ],
    }


# patterns that indicate issues re-extraction can't fix
_UNFIXABLE = (
    "Insufficient stock",
    "Duplicate invoice",
    "not approved",
    "elevated risk tier",
    "quantity must be > 0",
    "not found in approved vendor list",
)


def _has_extraction_fixable_issues(issues: list[str]) -> bool:
    """Are there any issues that re-extraction might actually fix?"""
    for issue in issues:
        if not any(p in issue for p in _UNFIXABLE):
            return True
    return False


def route_after_validation(state: InvoiceState) -> str:
    settings = get_settings()
    validation = state.get("validation_result") or {}

    if validation.get("is_valid"):
        return "fraud_check"

    issues = validation.get("issues", [])
    retries = state.get("extraction_retries", 0)

    if retries < settings.max_extraction_retries and _has_extraction_fixable_issues(issues):
        return "retry"

    # always go through fraud_check so we get a real risk_score
    return "fraud_check"


def route_after_approval(state: InvoiceState) -> str:
    decision = state.get("approval_decision") or {}
    if decision.get("status") == "approved":
        return "payment"
    return "reject"


def build_pipeline(checkpointer=None):
    """Compile the LangGraph pipeline."""
    workflow = StateGraph(InvoiceState)

    workflow.add_node("extract", extraction_node)
    workflow.add_node("retry_extraction", retry_extraction_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("fraud_check", fraud_detection_node)
    workflow.add_node("approve", approval_node)
    workflow.add_node("payment", payment_node)
    workflow.add_node("reject", rejection_node)
    workflow.add_node("explain", explanation_node)

    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"fraud_check": "fraud_check", "retry": "retry_extraction"},
    )
    workflow.add_edge("retry_extraction", "extract")
    workflow.add_edge("fraud_check", "approve")
    workflow.add_conditional_edges(
        "approve",
        route_after_approval,
        {"payment": "payment", "reject": "reject"},
    )
    workflow.add_edge("payment", "explain")
    workflow.add_edge("reject", "explain")
    workflow.add_edge("explain", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


def get_pipeline_diagram(pipeline) -> bytes:
    return pipeline.get_graph().draw_mermaid_png()


def process_invoice(
    pipeline,
    file_path: str,
    thread_id: Optional[str] = None,
) -> InvoiceState:
    """Run a single invoice through the full pipeline."""
    init_db()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    raw_text, file_type = parse_file(file_path)
    logger.info("pipeline.start", file=file_path, thread_id=thread_id)

    initial_state: InvoiceState = {
        "file_path": file_path,
        "raw_text": raw_text,
        "file_type": file_type,
        "extracted_invoice": None,
        "extraction_retries": 0,
        "extraction_feedback": "",
        "validation_result": None,
        "fraud_result": None,
        "approval_decision": None,
        "payment_result": None,
        "audit_trail": [],
        "error_message": None,
        "current_agent": "extract",
        "decision_explanation": "",
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = pipeline.invoke(initial_state, config=config)
    logger.info("pipeline.complete", thread_id=thread_id)
    return result


def resume_after_human_review(
    pipeline,
    thread_id: str,
    decision: str,
    reasoning: str = "",
) -> InvoiceState:
    """Resume a pipeline paused at the HITL interrupt."""
    logger.info("pipeline.resume", thread_id=thread_id, decision=decision)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = pipeline.invoke(
        Command(resume={"decision": decision, "reasoning": reasoning}),
        config=config,
    )
    return result
