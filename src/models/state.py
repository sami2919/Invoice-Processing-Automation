"""LangGraph state definition."""

import operator
from typing import Annotated, Optional, TypedDict


class InvoiceState(TypedDict):
    # input
    file_path: str
    raw_text: str
    file_type: str

    # extraction
    extracted_invoice: Optional[dict]
    extraction_retries: int
    extraction_feedback: str

    # downstream agents
    validation_result: Optional[dict]
    fraud_result: Optional[dict]
    approval_decision: Optional[dict]
    payment_result: Optional[dict]

    # append-only via LangGraph reducer
    audit_trail: Annotated[list, operator.add]

    # routing / meta
    error_message: Optional[str]
    current_agent: str
    decision_explanation: str
