"""Payment agent — calls mock payment API and records the invoice."""

import structlog

from src.models.state import InvoiceState
from src.tools.inventory_db import record_invoice
from src.tools.payment_api import mock_payment

logger = structlog.get_logger(__name__)


def payment_node(state: InvoiceState) -> dict:
    inv = state.get("extracted_invoice") or {}
    vendor = inv.get("vendor_name") or "UNKNOWN"
    amount = float(inv.get("total_amount") or 0.0)
    inv_num = inv.get("invoice_number", "UNKNOWN")

    result = mock_payment(vendor=vendor, amount=amount, invoice_number=inv_num)

    if result.get("status") == "success":
        try:
            record_invoice(invoice_number=inv_num, vendor=vendor, amount=amount, status="approved")
        except Exception as e:
            logger.error("payment.record_failed", invoice=inv_num, error=str(e))
    else:
        logger.error("payment.failed", invoice=inv_num)

    return {
        "payment_result": result,
        "current_agent": "payment",
        "audit_trail": [{"agent": "payment", "action": "payment_initiated",
                         "details": f"TXN {result.get('transaction_id')} — ${amount:,.2f} to {vendor}"}],
    }
