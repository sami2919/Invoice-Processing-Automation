"""Tests for the Grok LLM client wrapper."""

from unittest.mock import MagicMock, patch

from src.llm.grok_client import assess, get_llm, get_structured_llm


@patch("src.llm.grok_client.ChatXAI")
def test_get_llm_returns_chat_instance(mock_chat_cls):
    get_llm.cache_clear()
    mock_chat_cls.return_value = MagicMock()
    llm = get_llm(model="grok-3-fast", temperature=0.1)
    mock_chat_cls.assert_called_once()
    assert llm is not None


@patch("src.llm.grok_client.ChatXAI")
def test_get_llm_passes_settings(mock_chat_cls):
    get_llm.cache_clear()
    mock_chat_cls.return_value = MagicMock()
    get_llm(model="grok-3-fast", temperature=0.5)
    call_kwargs = mock_chat_cls.call_args.kwargs
    assert call_kwargs["model"] == "grok-3-fast"
    assert call_kwargs["temperature"] == 0.5
    assert "xai_api_key" in call_kwargs


@patch("src.llm.grok_client.ChatXAI")
def test_get_llm_caches_result(mock_chat_cls):
    get_llm.cache_clear()
    mock_chat_cls.return_value = MagicMock()
    a = get_llm(model="grok-3-fast", temperature=0.1)
    b = get_llm(model="grok-3-fast", temperature=0.1)
    assert a is b
    assert mock_chat_cls.call_count == 1


@patch("src.llm.grok_client.ChatXAI")
def test_get_llm_different_params_different_instance(mock_chat_cls):
    get_llm.cache_clear()
    mock_chat_cls.side_effect = [MagicMock(), MagicMock()]
    a = get_llm(model="grok-3-fast", temperature=0.0)
    b = get_llm(model="grok-3-fast", temperature=0.5)
    assert a is not b


@patch("src.llm.grok_client.get_llm")
def test_get_structured_llm_calls_with_structured_output(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    from pydantic import BaseModel

    class FakeSchema(BaseModel):
        name: str

    get_structured_llm(FakeSchema, model="grok-3-fast")
    mock_llm.with_structured_output.assert_called_once_with(FakeSchema)


@patch("src.llm.grok_client.get_llm")
def test_assess_returns_content_string(mock_get_llm):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This invoice looks legitimate."
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    result = assess("Analyze this invoice for fraud risk.")
    assert result == "This invoice looks legitimate."
    mock_llm.invoke.assert_called_once_with("Analyze this invoice for fraud risk.")


@patch("src.llm.grok_client.get_llm")
def test_assess_uses_specified_temperature(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="ok")
    mock_get_llm.return_value = mock_llm

    assess("test prompt", temperature=0.8)
    call_kwargs = mock_get_llm.call_args.kwargs
    assert call_kwargs["temperature"] == 0.8


@patch("src.llm.grok_client.get_llm")
def test_assess_default_temperature(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="ok")
    mock_get_llm.return_value = mock_llm

    assess("test prompt")
    call_kwargs = mock_get_llm.call_args.kwargs
    assert call_kwargs["temperature"] == 0.4
