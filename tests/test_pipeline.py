"""Tests for pipeline structure and routing logic."""

from unittest.mock import MagicMock, patch

from src.pipeline import process_invoice, route_after_approval, route_after_validation


def test_pipeline_compiles(pipeline):
    assert pipeline is not None


def test_pipeline_has_expected_nodes(pipeline):
    expected = {"extract", "retry_extraction", "validate", "fraud_check",
                "approve", "payment", "reject", "explain"}
    actual = set(pipeline.get_graph().nodes.keys())
    assert expected.issubset(actual), f"Missing nodes: {expected - actual}"


def test_pipeline_diagram_generates(pipeline):
    diagram = pipeline.get_graph().draw_mermaid()
    assert isinstance(diagram, str) and len(diagram) > 0
    assert "extract" in diagram


def test_route_after_validation_valid():
    state = {"validation_result": {"is_valid": True}, "extraction_retries": 0}
    assert route_after_validation(state) == "fraud_check"


def test_route_after_validation_retry_with_fixable_issues():
    state = {
        "validation_result": {
            "is_valid": False,
            "issues": ["Math mismatch: calculated $500 differs from stated total $600"],
        },
        "extraction_retries": 0,
    }
    assert route_after_validation(state) == "retry"


def test_route_after_validation_skip_retry_unfixable():
    state = {
        "validation_result": {
            "is_valid": False,
            "issues": ["Insufficient stock for 'WidgetA': requested 22, available 15"],
        },
        "extraction_retries": 0,
    }
    assert route_after_validation(state) == "fraud_check"


def test_route_after_validation_retries_exhausted():
    state = {
        "validation_result": {"is_valid": False, "issues": ["Some issue"]},
        "extraction_retries": 3,
    }
    assert route_after_validation(state) == "fraud_check"


def test_route_after_approval_approved():
    assert route_after_approval({"approval_decision": {"status": "approved"}}) == "payment"


def test_route_after_approval_rejected():
    assert route_after_approval({"approval_decision": {"status": "rejected"}}) == "reject"


def test_recursion_limit():
    mock_pipeline = MagicMock()
    mock_pipeline.invoke.return_value = {
        "audit_trail": [], "current_agent": "explain", "decision_explanation": "done",
    }
    with patch("src.pipeline.parse_file", return_value=("invoice text", "txt")):
        with patch("src.pipeline.init_db"):
            process_invoice(mock_pipeline, "test.txt", "thread-test-123")
    call_kwargs = mock_pipeline.invoke.call_args.kwargs
    assert call_kwargs["config"]["recursion_limit"] == 25
