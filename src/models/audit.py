"""Audit and reporting models."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str
    action: str
    details: str
    duration: Optional[float] = None
    confidence: Optional[float] = None


class ProcessingRecord(BaseModel):
    invoice_number: str
    vendor: str
    amount: float
    risk_score: int
    decision: str
    processing_time_seconds: float
    explanation: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchResult(BaseModel):
    records: list[ProcessingRecord] = []
    total_processed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    flagged_count: int = 0
    avg_processing_time: float = 0.0
