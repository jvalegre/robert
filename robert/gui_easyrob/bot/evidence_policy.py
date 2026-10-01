"""Deterministic evidence routing and privacy-safe search query policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Sequence

from .bot_rag import SearchResult, normalize_terms
from .prompt_policy import anonymize_paths, redact_sensitive_text
from .question_intent import is_explicit_web_request, requires_fresh_information

__all__ = [
    "EvidenceAssessment",
    "EvidenceDecision",
    "assess_evidence",
    "sanitize_search_query",
]


STRONG_LOCAL_SCORE = 4.0
STRONG_LOCAL_COVERAGE = 0.6
MAX_SEARCH_QUERY_CHARS = 500


def official_documentation_domains(question: str) -> tuple[str, ...]:
    """Select program documentation from explicit program or distinctive module names."""
    domains = []
    if re.search(r"\b(?:aqme|qdescp|csearch|cmin|qcorr|qprep)\b", question, re.I):
        domains.append("aqme.readthedocs.io")
    if re.search(r"\b(?:robert|easyrob|y[- ]shuffle|y[- ]mean)\b", question, re.I):
        domains.append("robert.readthedocs.io")
    return tuple(domains)


def _deep_program_question(question: str) -> bool:
    folded = unicodedata.normalize("NFKD", question.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    if re.search(r"\b(?:popup|dialog|button|tab|displayed|boton|pestana|ventana)\b", folded):
        return False
    return bool(official_documentation_domains(question) and re.search(
        r"\b(?:why|how does|calculat(?:e|ed|ion)|compare|difference|parameters?|defaults?|thresholds?|"
        r"boltzmann|cross.validation|y[- ]shuffle|conformer|solvent|explain|criteria|read.?the.?docs|por que|como calcula|como funciona|"
        r"diferencia|parametros?|umbrales?|validacion cruzada)\b", folded,
    ))

_ROLE_MARKER_RE = re.compile(
    r"(?i)(?:^|\s)(?:system|assistant|developer|tool|user)(?:\s*\(data\))?\s*:\s*"
)
_TEMPLATE_MARKER_RE = re.compile(
    r"(?i)<\|/?(?:im_start|im_end|system|assistant|user|developer|tool)\|>|\[/?INST\]"
)
_URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s]+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\s,;]+")
_POSIX_FILE_PATH_RE = re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s,;]+")
_REDACTION_RE = re.compile(r"(?i)\[(?:redacted|path|home)\]|<path>(?:[/\\][^\s]+)?")
_SPACE_RE = re.compile(r"\s+")
_FOUNDATIONAL_RE = re.compile(
    r"(?i)\b(?:what is|what are|explain|difference between|how does)\b.*\b(?:"
    r"machine learning|random forest|neural network|regression|classification|"
    r"cross validation|descriptor|molecule|chemistry|reaction|catalyst|overfitting|"
    r"feature importance|principal component|clustering|software testing|unit test|"
    r"integration test|one[- ]?shot|zero[- ]?shot|few[- ]?shot|coding|programming|"
    r"algorithm|data structure)\b"
)


class EvidenceDecision(str, Enum):
    """Select the least expensive evidence path that can answer safely."""

    LOCAL_ENOUGH = "local_enough"
    GENERAL_ENOUGH = "general_enough"
    WEB_OPTIONAL = "web_optional"
    WEB_REQUIRED = "web_required"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    decision: EvidenceDecision
    reason: str
    query_coverage: float = 0.0
    top_score: float = 0.0


def _coverage(results: Sequence[SearchResult], question: str) -> float:
    if not results:
        return 0.0
    first = results[0]
    query_terms = set(first.query_terms or normalize_terms(question))
    if not query_terms:
        return 0.0
    return min(1.0, len(set(first.matched_terms) & query_terms) / len(query_terms))


def assess_evidence(
    question: object,
    results: Sequence[SearchResult],
    *,
    intent: str = "tutorial",
) -> EvidenceAssessment:
    """Classify evidence sufficiency without making a paid model request."""
    if type(question) is not str or not normalize_terms(question):
        return EvidenceAssessment(EvidenceDecision.INSUFFICIENT, "empty_or_unsafe_question")
    if _TEMPLATE_MARKER_RE.search(question):
        return EvidenceAssessment(EvidenceDecision.INSUFFICIENT, "prompt_protocol_marker")
    if is_explicit_web_request(question) or requires_fresh_information(question):
        return EvidenceAssessment(EvidenceDecision.WEB_REQUIRED, "current_or_explicit_web")
    if _deep_program_question(question):
        return EvidenceAssessment(EvidenceDecision.WEB_REQUIRED, "advanced_official_documentation")

    ordered = sorted(results, key=lambda item: item.score, reverse=True)
    top_score = ordered[0].score if ordered else 0.0
    coverage = _coverage(ordered, question)
    if top_score >= STRONG_LOCAL_SCORE and coverage >= STRONG_LOCAL_COVERAGE:
        return EvidenceAssessment(
            EvidenceDecision.LOCAL_ENOUGH,
            "strong_local_evidence",
            query_coverage=coverage,
            top_score=top_score,
        )
    if _FOUNDATIONAL_RE.search(question):
        return EvidenceAssessment(
            EvidenceDecision.GENERAL_ENOUGH,
            "stable_foundational_question",
            query_coverage=coverage,
            top_score=top_score,
        )
    return EvidenceAssessment(
        EvidenceDecision.WEB_OPTIONAL,
        f"borderline_{str(intent or 'unknown').strip().lower()}",
        query_coverage=coverage,
        top_score=top_score,
    )


def sanitize_search_query(
    question: object,
    secret_values: Sequence[str] = (),
) -> str:
    """Return a bounded query with private and prompt-control data removed."""
    if type(question) is not str:
        return ""
    value = redact_sensitive_text(question, secret_values)
    value = _URL_RE.sub(" ", value)
    value = _WINDOWS_PATH_RE.sub(" ", value)
    value = _POSIX_FILE_PATH_RE.sub(" ", value)
    value = anonymize_paths(value)
    value = _TEMPLATE_MARKER_RE.sub(" ", value)
    value = _ROLE_MARKER_RE.sub(" ", value)
    value = _REDACTION_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip(" \t\r\n,;:-")
    if not normalize_terms(value):
        return ""
    domains = official_documentation_domains(value)
    if domains:
        scope = " OR ".join(f"site:{domain}" for domain in domains)
        value = f"({scope}) {value}"
    if len(value) <= MAX_SEARCH_QUERY_CHARS:
        return value
    bounded = value[:MAX_SEARCH_QUERY_CHARS].rsplit(" ", 1)[0].strip()
    return bounded or value[:MAX_SEARCH_QUERY_CHARS]
