"""Extraction agent — calls Grok to parse raw invoice text into structured data."""

import json
import re
from datetime import date as _date
from datetime import datetime, timezone
from typing import Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.llm.grok_client import assess, get_structured_llm
from src.models.invoice import ExtractedInvoice, LineItem
from src.models.state import InvoiceState
from src.tools.file_parser import parse_file
from src.tools.inventory_db import fuzzy_match_item

logger = structlog.get_logger(__name__)


# Looser extraction schemas — no business-rule validators so the LLM can
# faithfully capture malformed data (negative qty, blank vendor, OCR junk).
# Validation happens downstream.

class _LineItemExtract(BaseModel):
    item_name: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    line_total: Optional[float] = None
    note: Optional[str] = None


class _InvoiceExtract(BaseModel):
    invoice_number: str = "UNKNOWN"
    vendor_name: str = ""
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    line_items: list[_LineItemExtract] = []
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: float = 0.0
    currency: str = "USD"
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    confidence_scores: dict[str, float] = {}
    extraction_warnings: list[str] = []


def _parse_date(v: str | None) -> _date | None:
    if v is None:
        return None
    try:
        return _date.fromisoformat(str(v))
    except (ValueError, TypeError):
        return None


_SYSTEM_PROMPT = """\
You are an expert invoice data extraction system. Extract all fields from the \
provided invoice text and return them as structured JSON.

EXTRACTION RULES:
1. Extract exactly what is written — do not invent or assume missing data.
2. Normalize item names: remove embedded spaces ("Widget A" → "WidgetA", \
"Gadget X" → "GadgetX").
3. Fix OCR artifacts: letter O used as digit 0 ("2O26" → "2026", \
"3,500.O0" → "3500.00"); letter l used as digit 1.
4. Recalculate line_total as qty × unit_price; note any mismatch in \
extraction_warnings.
5. Assign confidence_scores (0.0–1.0) per key field — lower when data is \
ambiguous or missing.
6. List extraction_warnings for: OCR artifacts corrected, missing required \
fields, ambiguous values, total mismatches.
7. Preserve negative quantities exactly as written — do NOT correct them.
8. If vendor name is blank or absent, set vendor_name to an empty string.
9. Dates must be ISO 8601 (YYYY-MM-DD). Set to null if not parseable.
10. All monetary amounts must be plain floats (no currency symbols, no commas).

OUTPUT: valid JSON matching this schema exactly — no markdown, no preamble:
{
  "invoice_number": "string",
  "vendor_name": "string",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "line_items": [
    {
      "item_name": "string",
      "quantity": <number>,
      "unit_price": <number>,
      "line_total": <number or null>,
      "note": "string or null"
    }
  ],
  "subtotal": <number or null>,
  "tax_amount": <number or null>,
  "total_amount": <number>,
  "currency": "USD",
  "payment_terms": "string or null",
  "notes": "string or null",
  "confidence_scores": {
    "invoice_number": <0.0–1.0>,
    "vendor_name": <0.0–1.0>,
    "invoice_date": <0.0–1.0>,
    "due_date": <0.0–1.0>,
    "line_items": <0.0–1.0>,
    "total_amount": <0.0–1.0>
  },
  "extraction_warnings": ["..."]
}\
"""


def _build_user_message(raw_text: str, feedback: str, prior: dict | None) -> str:
    """Build the user prompt, including self-correction context on retries."""
    parts = [f"INVOICE TEXT:\n{raw_text}"]
    if feedback and prior:
        parts.append(
            "\nSELF-CORRECTION REQUIRED:\n"
            f"Your previous extraction had the following issues:\n{feedback}\n\n"
            f"Previous extraction output:\n{json.dumps(prior, indent=2, default=str)}\n\n"
            "Re-examine the source invoice text carefully and correct these "
            "specific problems."
        )
    return "\n".join(parts)


def _coerce_line_item(raw: dict) -> dict:
    return {
        "item_name": str(raw.get("item_name", raw.get("item", "Unknown"))),
        "quantity": float(raw.get("quantity", 0)),
        "unit_price": float(raw.get("unit_price", 0.0)),
        "line_total": (float(raw["line_total"]) if raw.get("line_total") is not None else None),
        "note": raw.get("note"),
    }


def _to_extracted_invoice(raw: _InvoiceExtract, warnings: list[str]) -> ExtractedInvoice:
    """Convert loose schema to strict ExtractedInvoice, falling back to model_construct
    for intentionally invalid data (negative qty, empty vendor, etc)."""
    data = raw.model_dump()
    try:
        return ExtractedInvoice.model_validate(data)
    except ValidationError as exc:
        warnings.append(f"Pydantic validation issues (data extracted as-is): {str(exc)[:300]}")
        raw_items = data.get("line_items", [])
        items = [LineItem.model_construct(**_coerce_line_item(item)) for item in raw_items]
        return ExtractedInvoice.model_construct(
            invoice_number=str(data.get("invoice_number", "UNKNOWN")),
            vendor_name=str(data.get("vendor_name", "")),
            invoice_date=_parse_date(data.get("invoice_date")),
            due_date=_parse_date(data.get("due_date")),
            line_items=items,
            subtotal=data.get("subtotal"),
            tax_amount=data.get("tax_amount"),
            total_amount=float(data.get("total_amount") or 0.0),
            currency=str(data.get("currency", "USD")),
            payment_terms=data.get("payment_terms"),
            notes=data.get("notes"),
            confidence_scores=data.get("confidence_scores", {}),
            extraction_warnings=list(data.get("extraction_warnings", [])) + warnings,
        )


def _extract_json_block(text: str) -> dict:
    """Pull a JSON object out of LLM response text (handles markdown fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON in LLM response: {text[:200]!r}")


def _fallback_extract(raw_text, feedback, prior, warnings):
    """Use assess() for free-form JSON when structured output fails."""
    msg = _build_user_message(raw_text, feedback, prior)
    prompt = f"{_SYSTEM_PROMPT}\n\nReturn ONLY valid JSON — no markdown, no preamble.\n\n{msg}"
    data = _extract_json_block(assess(prompt, temperature=0.0))
    return _InvoiceExtract.model_validate(data)


def _normalize_invoice_number(inv_num: str) -> str:
    """Ensure invoice number has INV- prefix."""
    n = inv_num.strip()
    if not n or n.upper() == "UNKNOWN":
        return n
    upper = n.upper()
    if n.isdigit():
        return f"INV-{n}"
    if upper.startswith("INV-"):
        return "INV-" + n[4:]
    if upper.startswith("INV "):
        return "INV-" + n[4:]
    if re.match(r"(?i)^INV\d", n):
        return "INV-" + n[3:]
    return n


def _fuzzy_match_items(invoice: ExtractedInvoice, warnings: list[str]) -> ExtractedInvoice:
    """Replace item names with canonical DB names when fuzzy match hits."""
    updated = []
    changed = False
    for item in invoice.line_items:
        canonical = fuzzy_match_item(item.item_name)
        if canonical and canonical != item.item_name:
            warnings.append(f"Item normalized: '{item.item_name}' → '{canonical}' (fuzzy match)")
            updated.append(item.model_copy(update={"item_name": canonical}))
            changed = True
        else:
            updated.append(item)

    if changed:
        return invoice.model_copy(update={"line_items": updated})
    return invoice


def _verify_total(invoice: ExtractedInvoice, warnings: list[str]) -> None:
    computed = sum(
        (item.line_total if item.line_total is not None else item.quantity * item.unit_price)
        for item in invoice.line_items
    )
    extracted = invoice.total_amount
    if extracted == 0 and computed != 0:
        warnings.append(
            f"Extracted total is zero but line items sum to {computed:.2f} "
            "— total field may be missing or unextracted"
        )
    elif extracted != 0 and abs(computed - extracted) / abs(extracted) > 0.01:
        warnings.append(
            f"Total mismatch: extracted {extracted:.2f}, computed from line items {computed:.2f}"
        )


def extraction_node(state: InvoiceState) -> dict:
    retries = state.get("extraction_retries", 0)
    feedback = state.get("extraction_feedback", "") or ""
    prior: dict | None = state.get("extracted_invoice")

    raw_text: str = state.get("raw_text") or ""
    file_type: str = state.get("file_type", "unknown") or "unknown"
    if not raw_text:
        raw_text, file_type = parse_file(state["file_path"])

    logger.info("extraction.start", attempt=retries + 1, file_type=file_type, chars=len(raw_text))

    warnings: list[str] = []

    # primary: structured LLM call
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(raw_text, feedback, prior)),
    ]

    raw_invoice: _InvoiceExtract | None = None
    try:
        raw_invoice = get_structured_llm(_InvoiceExtract).invoke(messages)
    except Exception as e:
        logger.warning("extraction.structured_failed", error=str(e)[:200])
        warnings.append(f"Structured output failed ({type(e).__name__}); falling back to raw JSON.")

    # fallback: raw JSON
    if raw_invoice is None:
        try:
            raw_invoice = _fallback_extract(raw_text, feedback, prior, warnings)
        except Exception as e:
            logger.error("extraction.failed", error=str(e))
            return {
                "error_message": f"Extraction failed: {e}",
                "current_agent": "extraction",
                "audit_trail": [{
                    "agent": "extraction", "action": "error",
                    "attempt": retries + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                }],
            }

    invoice = _to_extracted_invoice(raw_invoice, warnings)
    invoice = _fuzzy_match_items(invoice, warnings)

    # normalize invoice number
    norm = _normalize_invoice_number(invoice.invoice_number)
    if norm != invoice.invoice_number:
        warnings.append(f"Invoice number normalized: '{invoice.invoice_number}' -> '{norm}'")
        invoice = invoice.model_copy(update={"invoice_number": norm})

    _verify_total(invoice, warnings)

    all_warnings = list(invoice.extraction_warnings or []) + warnings
    invoice = invoice.model_copy(update={"extraction_warnings": all_warnings})

    logger.info(
        "extraction.done",
        invoice=invoice.invoice_number,
        vendor=invoice.vendor_name,
        total=invoice.total_amount,
        items=len(invoice.line_items),
    )

    return {
        "raw_text": raw_text,
        "file_type": file_type,
        "extracted_invoice": invoice.model_dump(mode="json"),
        "current_agent": "extraction",
        "audit_trail": [{
            "agent": "extraction",
            "action": "extract",
            "attempt": retries + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "invoice_number": invoice.invoice_number,
            "vendor_name": invoice.vendor_name,
            "total_amount": invoice.total_amount,
            "confidence": invoice.confidence_scores,
            "warnings": all_warnings,
        }],
    }
