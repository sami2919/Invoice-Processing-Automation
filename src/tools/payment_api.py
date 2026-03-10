"""Simulated payment API — would hit a real ERP in production."""

import time
from datetime import datetime, timezone


def mock_payment(vendor: str, amount: float, invoice_number: str) -> dict:
    return {
        "status": "success",
        "transaction_id": f"TXN-{invoice_number}-{int(time.time())}",
        "vendor": vendor,
        "amount": amount,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
