from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OpenAIChatClient:
    """Small OpenAI wrapper used by KGCaRe.

    KG extraction intentionally uses raw text completions so the 3-step prompt
    chain remains unchanged. Structured parsing is only used for QA/boundary
    outputs.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        structured_method: Literal["parse", "json", "json_schema"] = "parse",
    ) -> None:
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_method = structured_method

    def chat_text(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return response.choices[0].message.content or ""

    def structured(
        self,
        schema: Type[T],
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[T, str]:
        if self.structured_method == "json":
            return self._structured_json(schema, messages, max_tokens=max_tokens)
        if self.structured_method == "json_schema":
            return self._structured_json_schema(schema, messages, max_tokens=max_tokens)
        try:
            parse = self.client.beta.chat.completions.parse
            response = parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                response_format=schema,
            )
            message = response.choices[0].message
            raw = message.content or ""
            parsed = message.parsed
            if parsed is None:
                raise ValueError("OpenAI parse returned no parsed object")
            return parsed, raw
        except Exception:
            json_messages = [dict(message) for message in messages]
            instruction = "\n\nReturn only a valid JSON object. Do not include markdown fences."
            if json_messages and json_messages[-1].get("role") == "user":
                json_messages[-1]["content"] = json_messages[-1].get("content", "") + instruction
            else:
                json_messages.append({"role": "user", "content": instruction.strip()})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=json_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            return schema.model_validate(json.loads(raw)), raw

    def _structured_json(
        self,
        schema: Type[T],
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[T, str]:
        json_messages = [dict(message) for message in messages]
        instruction = "\n\nReturn only a valid JSON object. Do not include markdown fences."
        if json_messages and json_messages[-1].get("role") == "user":
            json_messages[-1]["content"] = json_messages[-1].get("content", "") + instruction
        else:
            json_messages.append({"role": "user", "content": instruction.strip()})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=json_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw)), raw

    def _structured_json_schema(
        self,
        schema: Type[T],
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[T, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        raw = response.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw)), raw


class OpenAIEmbeddingClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def embed_texts(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([item.embedding for item in ordered])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text], batch_size=1)[0]
