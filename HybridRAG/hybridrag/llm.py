from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from hybridrag.schemas import extract_json_object


StructuredT = TypeVar("StructuredT", bound=BaseModel)


def make_chat_model(
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float | None = None,
):
    """Create a LangChain ChatOpenAI model for OpenAI or OpenAI-compatible APIs."""

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatOpenAI(**kwargs)


def invoke_structured(
    llm,
    schema: type[StructuredT],
    messages: list[dict[str, str]],
    method: str = "function_calling",
) -> tuple[StructuredT, str, str]:
    """Invoke a model with structured output, falling back to JSON parsing."""

    if method == "json":
        return invoke_json_fallback(llm, schema, messages)

    try:
        # OpenAI's provider-native structured-output schema subset is stricter
        # than Pydantic's JSON schema for nested/defaulted models. Function
        # calling still gives validated Pydantic output without those schema
        # subset constraints, and also works better with OpenAI-compatible APIs.
        structured_llm = llm.with_structured_output(schema, method="function_calling")
        result = structured_llm.invoke(messages)
        if isinstance(result, schema):
            return result, result.model_dump_json(), "function_calling"
        if isinstance(result, dict):
            parsed = schema.model_validate(result)
            return parsed, json.dumps(result, ensure_ascii=False), "function_calling_dict"
        parsed = schema.model_validate(result)
        return parsed, parsed.model_dump_json(), "function_calling_coerced"
    except Exception:
        return invoke_json_fallback(llm, schema, messages)


def invoke_json_fallback(
    llm,
    schema: type[StructuredT],
    messages: list[dict[str, str]],
) -> tuple[StructuredT, str, str]:
    fallback_messages = build_json_fallback_messages(messages)
    response = llm.invoke(fallback_messages)
    raw_text = message_content_to_text(response)
    payload = extract_json_object(raw_text)
    if payload is None:
        raise ValueError(
            "Model did not return parseable JSON. The response may be truncated; "
            f"try increasing --max-tokens. Response prefix: {raw_text[:500]}"
        )
    return schema.model_validate(payload), raw_text, "json_fallback"


def build_json_fallback_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build fallback messages that satisfy strict user/assistant alternation.

    Some vLLM chat templates reject consecutive user turns. The original
    messages are normally [system, user], so append the JSON-only instruction
    into the user content instead of adding a second user message.
    """

    system_messages = [message["content"] for message in messages if message.get("role") == "system"]
    non_system_messages = [
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
        if message.get("role") != "system"
    ]
    non_system_messages.append(
        "Return ONLY a valid JSON object instance matching the requested output shape. "
        "Do not return a JSON schema, field descriptions, markdown fences, or trailing text."
    )

    fallback: list[dict[str, str]] = []
    if system_messages:
        fallback.append({"role": "system", "content": "\n\n".join(system_messages)})
    fallback.append({"role": "user", "content": "\n\n".join(non_system_messages)})
    return fallback


def message_content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        return "\n".join(parts)
    return str(content)
