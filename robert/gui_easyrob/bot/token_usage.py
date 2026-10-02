"""Token accounting primitives shared by robBOT backends and the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["MAX_USER_PROMPT_TOKENS", "TokenUsage", "estimate_token_count"]


MAX_USER_PROMPT_TOKENS = 500


def estimate_token_count(text: object) -> int:
    """Return a conservative tokenizer-free estimate for user-facing limits."""
    if type(text) is not str or not text:
        return 0
    ascii_count = sum(1 for character in text if ord(character) < 128)
    non_ascii = [character for character in text if ord(character) >= 128]
    non_ascii_bytes = sum(len(character.encode("utf-8")) for character in non_ascii)
    return math.ceil(ascii_count / 4) + max(len(non_ascii), math.ceil(non_ascii_bytes / 3))


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    estimated_cost_usd: float | None = 0.0
    is_estimated: bool = False
    provider: str = ""
    model: str = ""
    assessment_prompt_tokens: int = 0
    assessment_completion_tokens: int = 0
    search_content_tokens: int = 0
    search_requests: int = 0
    estimated_model_cost_usd: float | None = 0.0
    estimated_search_cost_usd: float | None = 0.0

    def __add__(self, other: object) -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented

        def add_optional(first: float | None, second: float | None) -> float | None:
            if first is None or second is None:
                return None
            return first + second

        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            assessment_prompt_tokens=(
                self.assessment_prompt_tokens + other.assessment_prompt_tokens
            ),
            assessment_completion_tokens=(
                self.assessment_completion_tokens + other.assessment_completion_tokens
            ),
            search_content_tokens=self.search_content_tokens + other.search_content_tokens,
            search_requests=self.search_requests + other.search_requests,
            estimated_model_cost_usd=add_optional(
                self.estimated_model_cost_usd, other.estimated_model_cost_usd
            ),
            estimated_search_cost_usd=add_optional(
                self.estimated_search_cost_usd, other.estimated_search_cost_usd
            ),
            estimated_cost_usd=add_optional(
                self.estimated_cost_usd, other.estimated_cost_usd
            ),
            is_estimated=self.is_estimated or other.is_estimated,
            provider=self.provider if self.provider == other.provider else "",
            model=self.model if self.model == other.model else "",
        )
