from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConditionalQAAnswer(BaseModel):
    answer_type: Literal["yes_no", "span"] = Field(
        description="Normalized answer family expected by the ConditionalQA formatter."
    )
    answers: list[str] = Field(
        default_factory=list,
        description="Ordered answer strings after normalization.",
    )
    conditions: list[list[str]] = Field(
        default_factory=list,
        description="Parallel condition lists for each answer entry.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional short explanation kept for debugging only.",
    )


class HotpotQAAnswer(BaseModel):
    answer: str = Field(description="Final short answer string.")
    rationale: str | None = Field(
        default=None,
        description="Optional short explanation kept for debugging only.",
    )
