"""Pydantic models for invoice data."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class LineItem(BaseModel):
    item_name: str
    quantity: float
    unit_price: float
    line_total: Optional[float] = None
    note: Optional[str] = None

    @field_validator("quantity")
    @classmethod
    def quantity_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"quantity must be > 0, got {v}")
        return v


class ExtractedInvoice(BaseModel):
    invoice_number: str
    vendor_name: str
    invoice_date: Optional[date] = None
    due_date: Optional[date] = None
    line_items: list[LineItem]
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: float
    currency: str = "USD"
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    confidence_scores: dict[str, float] = {}
    extraction_warnings: list[str] = []

    @field_validator("vendor_name")
    @classmethod
    def vendor_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vendor_name must not be empty")
        return v

    @field_validator("line_items")
    @classmethod
    def at_least_one_line_item(cls, v: list[LineItem]) -> list[LineItem]:
        if not v:
            raise ValueError("line_items must contain at least one item")
        return v


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[str] = []
    warnings: list[str] = []
    stock_checks: dict[str, dict] = {}


class FraudSignal(BaseModel):
    signal_type: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    weight: int


class FraudResult(BaseModel):
    risk_score: int  # 0-100
    signals: list[FraudSignal] = []
    recommendation: Literal["auto_approve", "flag_for_review", "block"]
    narrative: str = ""

    @field_validator("risk_score")
    @classmethod
    def risk_score_in_range(cls, v: int) -> int:
        if not (0 <= v <= 100):
            raise ValueError(f"risk_score must be 0-100, got {v}")
        return v


class ApprovalDecision(BaseModel):
    status: Literal["approved", "rejected", "escalated", "pending_human_review"]
    reasoning: str
    approver: Literal["auto", "human", "system"]
    conditions: list[str] = []
