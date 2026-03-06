"""Grok LLM client via langchain-xai."""

from functools import lru_cache
from typing import Optional, Type

import structlog
from langchain_xai import ChatXAI
from openai import OpenAI
from pydantic import BaseModel

from src.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=4)
def get_llm(model: Optional[str] = None, temperature: float = 0.1) -> ChatXAI:
    settings = get_settings()
    resolved = model or settings.grok_model
    return ChatXAI(
        model=resolved,
        temperature=temperature,
        xai_api_key=settings.xai_api_key,
    )


def get_structured_llm(schema: Type[BaseModel], model: Optional[str] = None):
    """LLM chain that outputs a validated Pydantic model."""
    llm = get_llm(model=model, temperature=0.0)
    return llm.with_structured_output(schema)


def assess(prompt: str, temperature: float = 0.4) -> str:
    """Single prompt -> string response. Higher temp for reasoning tasks."""
    settings = get_settings()
    llm = get_llm(model=settings.grok_model, temperature=temperature)
    response = llm.invoke(prompt)
    return response.content


@lru_cache(maxsize=1)
def _get_raw_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.xai_api_key, base_url=settings.xai_base_url)


raw_client: OpenAI = _get_raw_client()
