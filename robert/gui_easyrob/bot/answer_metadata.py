"""Provider-independent answer evidence and citation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import urlsplit

from .response_quality import calibrate_validation_guarantees

__all__ = [
    "AnswerMetadata",
    "EvidenceOrigin",
    "normalize_provider_answer_text",
    "normalize_provider_citations",
    "SourceCitation",
    "WebSearchMode",
]


_MALFORMED_URL_CITATION_RE = re.compile(
    r"【\[(https?://[^\s\]】]+)】\]\((https?://[^\s\)】]+)】\)",
    flags=re.IGNORECASE,
)
_INTERNAL_REFERENCE_RE = re.compile(
    r"(?:\[\s*|【\s*)(?:easyrob|robert|aqme)[\-_‐‑‒–—―]"
    r"[A-Za-z0-9_\-‐‑‒–—―.]+(?:\s*\]|】)",
    flags=re.IGNORECASE,
)
_EMPTY_MARKDOWN_ARTIFACT_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]*)?\\[ \t]*(?:\n|$)"
)
_COMPLETE_SENTENCE_RE = re.compile(r"[.!?](?=\s|$)")


class EvidenceOrigin(str, Enum):
    """Describe the evidence used to produce a visible answer."""

    LOCAL_SYSTEM = "local_system"
    LOCAL_DOCUMENTATION = "local_documentation"
    WEB_AND_DOCUMENTATION = "web_and_documentation"
    GENERAL_KNOWLEDGE = "general_knowledge"
    INSUFFICIENT = "insufficient"


class WebSearchMode(str, Enum):
    """Control whether a provider may perform its native web search."""

    OFF = "off"
    AUTO = "auto"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A validated, user-visible HTTP source."""

    title: str
    url: str
    start_index: int | None = None
    end_index: int | None = None

    @classmethod
    def create(
        cls,
        title: object,
        url: object,
        **locations: object,
    ) -> "SourceCitation | None":
        clean_url = str(url or "").strip()
        try:
            parsed = urlsplit(clean_url)
        except ValueError:
            return None
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None

        start = locations.get("start_index")
        end = locations.get("end_index")
        clean_start = start if isinstance(start, int) and start >= 0 else None
        clean_end = end if isinstance(end, int) and end >= 0 else None
        if clean_start is not None and clean_end is not None and clean_end < clean_start:
            clean_start = None
            clean_end = None

        clean_title = str(title or "").strip() or clean_url
        return cls(
            title=clean_title[:200],
            url=clean_url[:2000],
            start_index=clean_start,
            end_index=clean_end,
        )


def normalize_provider_citations(text: object) -> str:
    """Repair a known malformed provider URL citation without trusting its target."""
    clean_text = str(text or "")

    def replace(match: re.Match[str]) -> str:
        visible_url, target_url = match.groups()
        if visible_url.casefold().rstrip("/") != target_url.casefold().rstrip("/"):
            return match.group(0)
        citation = SourceCitation.create(visible_url, target_url)
        if citation is None:
            return match.group(0)
        return f"[{citation.url}]({citation.url})"

    return _MALFORMED_URL_CITATION_RE.sub(replace, clean_text)


def _tables_to_labeled_lists(text: str) -> str:
    """Flatten pipe tables for the narrow chat renderer without inventing cells."""
    lines = text.splitlines()
    output = []
    index = 0
    def cells(line):
        return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
    while index < len(lines):
        headers = cells(lines[index])
        separators = cells(lines[index + 1]) if index + 1 < len(lines) else []
        if len(headers) > 1 and len(headers) == len(separators) and all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separators
        ):
            index += 2
            output.append("")
            while index < len(lines) and "|" in lines[index]:
                values = cells(lines[index])
                if len(values) > len(headers):
                    break
                for header, value in zip(headers, values):
                    if value:
                        output.append(f"- **{header}:** {value}")
                output.append("")
                index += 1
        else:
            output.append(lines[index])
            index += 1
    return "\n".join(output)


def normalize_provider_answer_text(text: object, *, token_limited: bool = False) -> str:
    """Clean provider formatting while preserving readable Markdown and web links."""
    cleaned = normalize_provider_citations(text).replace("\r\n", "\n").replace("\r", "\n")
    code_blocks: list[str] = []
    def protect_code(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"@@ROBBOTCODE{len(code_blocks) - 1}@@"
    cleaned = re.sub(r"(?m)^```[^\n]*\n[\s\S]*?(?:^```[ \t]*$|\Z)", protect_code, cleaned)
    cleaned = _tables_to_labeled_lists(cleaned)
    cleaned = cleaned.replace("&#x20;", " ").replace("&#32;", " ")
    cleaned = _INTERNAL_REFERENCE_RE.sub("", cleaned)
    cleaned = _EMPTY_MARKDOWN_ARTIFACT_RE.sub("", cleaned)
    cleaned = re.sub(r"\\[ \t]*\n", "\n", cleaned)
    cleaned = re.sub(r"\\([*_\-])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    lines = cleaned.splitlines()
    next_item_number = 0
    for index, line in enumerate(lines):
        ordered_item = re.match(r"^(\s*)\d+([.)])\s+(.+)$", line)
        if ordered_item:
            next_item_number = next_item_number + 1 if next_item_number else 1
            lines[index] = (
                f"{ordered_item.group(1)}{next_item_number}{ordered_item.group(2)} "
                f"{ordered_item.group(3)}"
            )
        elif line.strip() and not (
            next_item_number and (line[:1].isspace() or re.fullmatch(r"@@ROBBOTCODE\d+@@", line.strip()))
        ):
            next_item_number = 0
    cleaned = "\n".join(lines)
    if token_limited:
        cleaned = re.sub(r"(?m)^\s*\d+\\?[.)]\s*$", "", cleaned).rstrip()
        sentence_ends = list(_COMPLETE_SENTENCE_RE.finditer(cleaned))
        if sentence_ends:
            cleaned = cleaned[: sentence_ends[-1].end()].rstrip()
        elif cleaned:
            cleaned = cleaned.rstrip(" .") + "…"
    cleaned = calibrate_validation_guarantees(cleaned)
    for index, block in enumerate(code_blocks):
        cleaned = cleaned.replace(f"@@ROBBOTCODE{index}@@", block)
    return cleaned


@dataclass(frozen=True, slots=True)
class AnswerMetadata:
    """Normalized evidence details returned by any answer backend."""

    evidence_origin: EvidenceOrigin = EvidenceOrigin.GENERAL_KNOWLEDGE
    sources: tuple[SourceCitation, ...] = ()
    search_error: str = ""
