"""Tests for the validation agent's deterministic checks."""

from src.agents.validation import (
    _check_aggregate_stock,
    _check_currency,
    _check_duplicate,
    _check_math,
    _check_negative_quantities,
    _check_price_variance,
    _check_required_fields,
    _resolve_item_names,
    validation_node,
)
from src.tools.inventory_db import record_invoice


def test_valid_invoice_passes(clean_state, patch_db):
    result = validation_node(clean_state)
    vr = result["validation_result"]
    assert vr["is_valid"] is True
    assert vr["issues"] == []


def test_aggregate_quantity_check(test_db):
    """WidgetA across 3 lines totals 22 but stock is only 15."""
    line_items = [
        {"item_name": "WidgetA", "quantity": 15.0, "unit_price": 250.0},
        {"item_name": "WidgetA", "quantity": 5.0, "unit_price": 240.0, "note": "Volume discount"},
        {"item_name": "WidgetA", "quantity": 2.0, "unit_price": 250.0, "note": "Replacement"},
    ]
    _, name_map = _resolve_item_names(line_items, test_db)
    issues, stock_checks = _check_aggregate_stock(line_items, name_map, test_db)

    assert stock_checks["WidgetA"]["requested"] == 22.0
    assert stock_checks["WidgetA"]["available"] == 15
    assert stock_checks["WidgetA"]["sufficient"] is False


def test_duplicate_detection(test_db):
    record_invoice("INV-1004", "Precision Parts Ltd.", 1890.0, "approved", db_path=test_db)
    issues = _check_duplicate("INV-1004", test_db)
    assert len(issues) == 1
    assert "INV-1004" in issues[0]


def test_no_duplicate_for_fresh_invoice(test_db):
    assert _check_duplicate("INV-FRESH", test_db) == []


def test_negative_quantity():
    issues = _check_negative_quantities([{"item_name": "WidgetA", "quantity": -5.0, "unit_price": 250.0}])
    assert len(issues) == 1
    assert "-5" in issues[0]


def test_zero_quantity_also_flagged():
    issues = _check_negative_quantities([{"item_name": "WidgetB", "quantity": 0.0, "unit_price": 500.0}])
    assert len(issues) == 1


def test_unknown_item(test_db):
    issues, name_map = _resolve_item_names(
        [{"item_name": "SuperGizmo", "quantity": 1.0, "unit_price": 999.0}], test_db
    )
    assert name_map["SuperGizmo"] is None
    assert any("SuperGizmo" in i for i in issues)


def test_zero_stock_item(test_db):
    line_items = [{"item_name": "FakeItem", "quantity": 1.0, "unit_price": 1000.0}]
    _, name_map = _resolve_item_names(line_items, test_db)
    issues, stock_checks = _check_aggregate_stock(line_items, name_map, test_db)
    assert stock_checks["FakeItem"]["available"] == 0
    assert any("Insufficient stock" in i for i in issues)


def test_currency_flag():
    issues = _check_currency("EUR")
    assert len(issues) == 1
    assert "EUR" in issues[0]


def test_usd_currency_clean():
    assert _check_currency("USD") == []


def test_math_verification():
    """$1000 computed vs $1100 stated = 10% diff -> flagged."""
    issues = _check_math([{"item_name": "WidgetA", "quantity": 10.0, "unit_price": 100.0}], 1100.0)
    assert len(issues) == 1


def test_math_passes_within_tolerance():
    issues = _check_math([{"item_name": "WidgetA", "quantity": 10.0, "unit_price": 100.0}], 1005.0)
    assert issues == []


def test_price_variance(test_db):
    """WidgetA at $350 vs catalog $250 = 40% variance."""
    line_items = [{"item_name": "WidgetA", "quantity": 5.0, "unit_price": 350.0}]
    _, name_map = _resolve_item_names(line_items, test_db)
    issues = _check_price_variance(line_items, name_map, test_db)
    assert len(issues) == 1
    assert "WidgetA" in issues[0]


def test_price_variance_within_tolerance(test_db):
    line_items = [{"item_name": "WidgetA", "quantity": 5.0, "unit_price": 255.0}]
    _, name_map = _resolve_item_names(line_items, test_db)
    assert _check_price_variance(line_items, name_map, test_db) == []


def test_missing_vendor():
    inv = {"vendor_name": "", "invoice_number": "INV-TEST",
           "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0}],
           "total_amount": 250.0}
    issues = _check_required_fields(inv)
    assert any("vendor_name" in i for i in issues)


def test_required_fields_all_present():
    inv = {"vendor_name": "Widgets Inc.", "invoice_number": "INV-GOOD",
           "line_items": [{"item_name": "WidgetA", "quantity": 1.0, "unit_price": 250.0}],
           "total_amount": 250.0}
    assert _check_required_fields(inv) == []
