"""Lexical knowledge base loading and ranking for the EasyROB bot."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .knowledge_schema import (
    CONTEXT_SECTIONS,
    DOMAINS,
    KnowledgeLoadIssue,
    KnowledgeValidationError,
    validate_context_payload,
    validate_context_set,
)
from .question_intent import classify_question_intent

__all__ = ["KnowledgeBase", "KnowledgeChunk", "SearchResult", "normalize_terms"]

FIELD_WEIGHTS = {
    "question_patterns": 4.0,
    "title": 3.5,
    "keywords": 3.0,
    "aliases": 3.0,
    "tab": 2.5,
    "module": 2.5,
    "program": 2.5,
    "entity_type": 2.0,
    "topic": 1.5,
    "kind": 1.5,
    "text": 1.0,
}
BM25_K1 = 1.2
BM25_B = 0.75
MIN_RELEVANCE_SCORE = 1.0

_FINAL_CONTEXT_FILENAMES = (
    "robert_context.json",
    "aqme_context.json",
    "ui_context.json",
)


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_TUTORIAL_TAB_HINTS = (
    ("MolSSI Databases", ("molssi databases", "descriptor libraries", "molssi")),
    ("Advanced Options", ("advanced options", "general", "curate", "generate", "predict settings")),
    ("Results", ("results tab", "results view", "result views")),
    ("Check model", ("check model tab", "evaluate tab", "evaluate my model")),
    ("Predictions", ("predictions tab", "prediction results", "predict workflow", "external test")),
    ("Images", ("images tab", "generated figures", "browse these figures")),
    ("Reports", ("reports tab", "pdf report", "report summarizes")),
    ("AQME", ("aqme tab", "aqme workflow", "run aqme", "descriptors")),
    ("ROBERT", ("robert tab", "run robert", "full workflow", "machine learning workflow")),
)
_AQME_MODULES = {"AQME", "QDESCP", "CSEARCH", "CMIN", "QCORR", "QPREP"}
_ROBERT_ML_MODULES = {"ROBERT", "CURATE", "GENERATE", "VERIFY", "PREDICT", "REPORT", "FULL_WORKFLOW"}
_UI_TAB_NAMES = {"ROBERT", "AQME", "Advanced Options", "Results", "Reports", "Predictions", "Images", "MolSSI Databases", "Check model"}
_LOW_SIGNAL_TITLES = {"overview", "developement", "reference", "module", "main"}
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "do", "does", "for", "how", "i", "in",
    "is", "it", "me", "of", "on", "please", "something", "tell", "the", "this",
    "to", "what", "when", "where", "which", "why", "with", "you",
    "como", "cual", "de", "el", "es", "esta", "explica", "explicas", "funciona",
    "la", "las", "lo", "los", "mi", "necesito", "para", "por", "que", "son",
    "un", "una",
}
_TERM_ALIASES = (
    (re.compile(r"\bflujo\s+de\s+trabajo\b"), "workflow"),
    (re.compile(r"\bdescriptores?\b"), "descriptors"),
    (re.compile(r"\bmodelos?\b"), "model"),
    (re.compile(r"\b(?:prediccion|predicciones)\b"), "predictions"),
    (re.compile(r"\bregresi[oó]n\b"), "regression"),
    (re.compile(r"\bclasificaci[oó]n\b"), "classification"),
    (re.compile(r"\binformes?\b"), "reports"),
    (re.compile(r"\br\s*(?:squared|2)\b"), "r2"),
    (re.compile(r"\bno[\s_-]+pfi\b"), "no_pfi"),
    (re.compile(r"\btarget\s+column\b"), "target_column"),
    (re.compile(r"\badvanced\s+options\b"), "advanced_options"),
    (re.compile(r"\bfull\s+workflow\b"), "full_workflow"),
)


def normalize_terms(value: object) -> tuple[str, ...]:
    """Return English lexical terms after stable Unicode and declared alias normalization."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(term for item in value for term in normalize_terms(item))
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    for pattern, replacement in _TERM_ALIASES:
        normalized = pattern.sub(replacement, normalized)
    return tuple(
        term for term in _TOKEN_RE.findall(normalized)
        if term not in _STOP_WORDS
    )


def _canonical_question_pattern(value: object) -> tuple[str, ...]:
    """Canonicalize exact patterns without collapsing interrogative intent."""
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    for pattern, replacement in _TERM_ALIASES:
        normalized = pattern.sub(replacement, normalized)
    interrogatives = {"what", "when", "where", "which", "who", "why", "how"}
    return tuple(
        term for term in _TOKEN_RE.findall(normalized)
        if term in interrogatives or term not in _STOP_WORDS
    )


def _tokenize(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_tokenize(item))
        return tokens
    return set(_TOKEN_RE.findall(str(value).lower()))


def _ensure_sequence(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_tutorial_text(text: str) -> str:
    cleaned = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ").replace("\r", "\n")
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _infer_tutorial_tab(stem: str, text: str) -> str:
    lowered = f"{stem} {text}".lower()
    for tab_name, hints in _TUTORIAL_TAB_HINTS:
        if any(hint in lowered for hint in hints):
            return tab_name
    return "ROBERT"


def _tutorial_keywords(stem: str, tab: str, text: str) -> tuple[str, ...]:
    stem_label = stem.replace("_", " ").replace("-", " ").strip()
    keywords = [stem_label, tab]
    sentences = [part.strip() for part in re.split(r"[.!?]", text) if part.strip()]
    if sentences:
        keywords.append(sentences[0])
    return tuple(keyword for keyword in keywords if keyword)


def _default_title(text: str, chunk_id: str) -> str:
    sentences = [part.strip() for part in re.split(r"[.!?]", text) if part.strip()]
    if sentences:
        return sentences[0][:120]
    return chunk_id


def _coerce_priority(value: object, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    id: str
    source: str
    topic: str
    tab: str
    keywords: tuple[str, ...]
    text: str
    source_file: str = ""
    program: str = ""
    module: str = ""
    kind: str = "concept"
    title: str = ""
    question_patterns: tuple[str, ...] = ()
    priority: float = 0.5
    audience: str = ""
    source_tier: str = ""
    entity_type: str = ""
    entity_id: str = ""
    aliases: tuple[str, ...] = ()
    related_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float
    matched_terms: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    chunk: KnowledgeChunk
    field_counters: Mapping[str, Counter[str]]
    body_length: int
    content_fingerprint: str


def _build_indexed_chunk(chunk: KnowledgeChunk) -> IndexedChunk:
    counters = {
        field: Counter(normalize_terms(getattr(chunk, field)))
        for field in FIELD_WEIGHTS
    }
    normalized_content = " ".join(
        unicodedata.normalize("NFKC", chunk.text).casefold().split()
    )
    fingerprint = hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()
    return IndexedChunk(
        chunk=chunk,
        field_counters=counters,
        body_length=sum(counters["text"].values()),
        content_fingerprint=fingerprint,
    )


class KnowledgeBase:
    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        issues: Sequence[KnowledgeLoadIssue] = (),
    ):
        self._chunks = list(chunks)
        self._issues = tuple(issues)
        self._index = tuple(_build_indexed_chunk(chunk) for chunk in self._chunks)
        self._document_count = len(self._index)
        self._average_body_length = (
            sum(item.body_length for item in self._index) / self._document_count
            if self._document_count else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for item in self._index:
            document_frequencies.update(
                set().union(*(counter.keys() for counter in item.field_counters.values()))
            )
        self._document_frequencies = document_frequencies

    @property
    def issues(self) -> tuple[KnowledgeLoadIssue, ...]:
        return self._issues

    @property
    def is_degraded(self) -> bool:
        return bool(self._issues)

    def chunks_for_source(self, source: str) -> list[KnowledgeChunk]:
        normalized = _normalize_text(source)
        if not normalized:
            return []
        return [chunk for chunk in self._chunks if _normalize_text(chunk.source) == normalized]

    @classmethod
    def from_records(cls, records: Iterable[dict[str, object]]) -> "KnowledgeBase":
        chunks = [cls._record_to_chunk(record, source_fallback="memory") for record in records]
        return cls(chunks)

    @staticmethod
    def _with_json_path(
        issue: KnowledgeLoadIssue,
        json_path: str,
    ) -> KnowledgeLoadIssue:
        return KnowledgeLoadIssue(
            code=issue.code,
            message=f"{json_path}: {issue.message}",
            source=Path(issue.source).name,
            record_id=issue.record_id,
        )

    @classmethod
    def _validated_context_payloads(
        cls,
        payloads: Mapping[str, object],
        *,
        include_missing_domains: bool,
    ) -> tuple[dict[str, object], list[KnowledgeLoadIssue]]:
        sanitized: dict[str, object] = {}
        issues: list[KnowledgeLoadIssue] = []

        if not isinstance(payloads, Mapping):
            return {}, [
                KnowledgeLoadIssue(
                    "invalid_context_set",
                    "$: Knowledge contexts must be supplied as a domain mapping.",
                    "context set",
                )
            ]

        for domain in DOMAINS:
            if domain not in payloads:
                if include_missing_domains:
                    issues.append(
                        KnowledgeLoadIssue(
                            "missing_context_domain",
                            f"$: Context set is missing domain '{domain}'.",
                            "context set",
                        )
                    )
                continue

            payload = payloads[domain]
            source = f"{domain}_context.json"
            if not isinstance(payload, Mapping):
                issues.extend(
                    cls._with_json_path(issue, "$")
                    for issue in validate_context_payload(payload, domain, source)
                )
                continue

            structural_payload = dict(payload)
            for section in CONTEXT_SECTIONS:
                if isinstance(structural_payload.get(section), list):
                    structural_payload[section] = []
            file_issues = validate_context_payload(
                structural_payload,
                expected_domain=domain,
                source=source,
            )
            if file_issues:
                issues.extend(cls._with_json_path(issue, "$") for issue in file_issues)
                continue

            clean_payload = dict(payload)
            for section in CONTEXT_SECTIONS:
                clean_records: list[object] = []
                section_records = payload[section]
                for index, record in enumerate(section_records):
                    json_path = f"$.{section}[{index}]"
                    if not isinstance(record, Mapping):
                        issues.append(
                            KnowledgeLoadIssue(
                                "invalid_record",
                                f"{json_path}: Context record must be an object.",
                                source,
                            )
                        )
                        continue
                    probe = dict(structural_payload)
                    probe[section] = [record]
                    record_issues = validate_context_payload(
                        probe,
                        expected_domain=domain,
                        source=source,
                    )
                    if record_issues:
                        issues.extend(
                            cls._with_json_path(issue, json_path)
                            for issue in record_issues
                        )
                        continue
                    clean_records.append(dict(record))
                clean_payload[section] = clean_records
            sanitized[domain] = clean_payload

        unexpected_domains = sorted(
            (domain for domain in payloads if domain not in DOMAINS),
            key=str,
        )
        for domain in unexpected_domains:
            issues.append(
                KnowledgeLoadIssue(
                    "unexpected_context_domain",
                    f"$: Context set contains unexpected domain {domain!r}.",
                    "context set",
                )
            )

        cross_issue_codes = {
            "schema_version_mismatch",
            "version_mismatch",
            "duplicate_id",
            "duplicate_entity_id",
            "unresolved_related_id",
        }
        issues.extend(
            cls._with_json_path(issue, "$")
            for issue in validate_context_set(sanitized)
            if issue.code in cross_issue_codes
        )
        return sanitized, issues

    @classmethod
    def _from_payloads(
        cls,
        payloads: Mapping[str, object],
        *,
        strict: bool,
        initial_issues: Sequence[KnowledgeLoadIssue] = (),
        include_missing_domains: bool,
    ) -> "KnowledgeBase":
        sanitized, validation_issues = cls._validated_context_payloads(
            payloads,
            include_missing_domains=include_missing_domains,
        )
        issues = [*initial_issues, *validation_issues]
        chunks: list[KnowledgeChunk] = []
        for domain in DOMAINS:
            payload = sanitized.get(domain)
            if isinstance(payload, dict):
                chunks.extend(
                    cls._context_object_to_chunks(
                        payload,
                        source_fallback=f"{domain}_context.json",
                    )
                )
        if strict and issues:
            raise KnowledgeValidationError(issues)
        return cls(chunks, issues)

    @classmethod
    def from_payloads(
        cls,
        payloads: Mapping[str, object],
        strict: bool = True,
    ) -> "KnowledgeBase":
        return cls._from_payloads(
            payloads,
            strict=strict,
            include_missing_domains=True,
        )

    @staticmethod
    def _exact_final_paths(base_dir: Path) -> dict[str, Path]:
        try:
            entries_by_name = {
                entry.name: entry
                for entry in base_dir.iterdir()
                if entry.is_file() and entry.name in _FINAL_CONTEXT_FILENAMES
            }
        except OSError:
            return {}
        return {
            name: entries_by_name[name]
            for name in _FINAL_CONTEXT_FILENAMES
            if name in entries_by_name
        }

    @classmethod
    def from_directory(
        cls,
        directory: str | Path | None = None,
        tutorials_dir: str | Path | None = None,
        strict: bool = True,
        allow_generic_json: bool = False,
    ) -> "KnowledgeBase":
        root_dir = Path(__file__).resolve().parent
        base_dir = Path(directory) if directory is not None else root_dir / "knowledge"
        tutorials_path = Path(tutorials_dir) if tutorials_dir is not None else None
        final_paths = cls._exact_final_paths(base_dir)
        if final_paths:
            issues: list[KnowledgeLoadIssue] = []
            payloads: dict[str, object] = {}
            for name in _FINAL_CONTEXT_FILENAMES:
                domain = name.removesuffix("_context.json")
                path = final_paths.get(name)
                if path is None:
                    issues.append(
                        KnowledgeLoadIssue(
                            "missing_final_context",
                            f"$: Required final context '{name}' is missing.",
                            name,
                        )
                    )
                    continue
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        payloads[domain] = json.load(handle)
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    issues.append(
                        KnowledgeLoadIssue(
                            "invalid_json",
                            f"$: Could not read valid JSON ({type(exc).__name__}).",
                            path.name,
                        )
                    )
            knowledge_base = cls._from_payloads(
                payloads,
                strict=False,
                initial_issues=issues,
                include_missing_domains=False,
            )
            chunks = list(knowledge_base._chunks)
            issues = list(knowledge_base.issues)
            if not strict and directory is None:
                for domain in DOMAINS:
                    payload = payloads.get(domain)
                    if isinstance(payload, dict) and "schema_version" not in payload:
                        try:
                            chunks.extend(
                                cls._context_object_to_chunks(
                                    payload,
                                    source_fallback=f"{domain}_context.json",
                                )
                            )
                        except (TypeError, ValueError):
                            pass
        elif directory is not None and allow_generic_json:
            chunks, issues = cls._load_generic_json(base_dir)
        else:
            chunks = []
            issues = [
                KnowledgeLoadIssue(
                    "missing_final_context",
                    f"$: Required final context '{name}' is missing.",
                    name,
                )
                for name in _FINAL_CONTEXT_FILENAMES
            ]

        if tutorials_path is not None:
            try:
                chunks.extend(cls._load_tutorial_chunks(tutorials_path))
            except (OSError, UnicodeError) as exc:
                issues.append(
                    KnowledgeLoadIssue(
                        "invalid_tutorial",
                        f"$: Could not load tutorial markdown ({type(exc).__name__}).",
                        tutorials_path.name,
                    )
                )
        if strict and issues:
            raise KnowledgeValidationError(issues)
        return cls(chunks, issues)

    @classmethod
    def _load_generic_json(
        cls,
        base_dir: Path,
    ) -> tuple[list[KnowledgeChunk], list[KnowledgeLoadIssue]]:
        chunks: list[KnowledgeChunk] = []
        issues: list[KnowledgeLoadIssue] = []
        json_paths = sorted(base_dir.glob("*.json"))
        for path in json_paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                issues.append(
                    KnowledgeLoadIssue(
                        "invalid_json",
                        f"{path.name}: invalid JSON at $ ({type(exc).__name__}).",
                        path.name,
                    )
                )
                continue
            if isinstance(payload, dict):
                try:
                    chunks.extend(cls._context_object_to_chunks(payload, source_fallback=path.name))
                except (TypeError, ValueError) as exc:
                    issues.append(
                        KnowledgeLoadIssue(
                            "invalid_payload",
                            f"{path.name}: {exc} at $.",
                            path.name,
                        )
                    )
                continue
            if not isinstance(payload, list):
                issues.append(
                    KnowledgeLoadIssue(
                        "invalid_payload",
                        f"{path.name}: expected JSON object or array payload at $.",
                        path.name,
                    )
                )
                continue
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    issues.append(
                        KnowledgeLoadIssue(
                            "invalid_record",
                            f"{path.name}: record {index} must be an object, got "
                            f"{type(record).__name__} at $[{index}].",
                            path.name,
                        )
                    )
                    continue
                try:
                    chunks.append(cls._record_to_chunk(record, source_fallback=path.name))
                except (TypeError, ValueError) as exc:
                    issues.append(
                        KnowledgeLoadIssue(
                            "invalid_record",
                            f"{path.name}: {exc} at $[{index}].",
                            path.name,
                            str(record.get("id", "")),
                        )
                    )
        return chunks, issues

    @classmethod
    def _context_object_to_chunks(
        cls,
        payload: dict[str, object],
        source_fallback: str,
    ) -> list[KnowledgeChunk]:
        domain = str(payload.get("domain", "")).strip()
        version = str(payload.get("version", "")).strip()
        chunks: list[KnowledgeChunk] = []

        overview = payload.get("overview") or {}
        if isinstance(overview, dict) and overview.get("text"):
            overview_program = "AQME" if domain == "aqme" else domain.upper() or "ROBERT"
            overview_tab = "AQME" if domain == "aqme" else "ROBERT"
            chunks.append(
                cls._record_to_chunk(
                    {
                        "id": f"{domain or 'knowledge'}-overview",
                        "source": source_fallback,
                        "topic": "overview",
                        "tab": overview.get("tab", overview_tab),
                        "program": overview.get("program", overview_program),
                        "module": overview.get("module", "OVERVIEW"),
                        "kind": overview.get("kind", "concept"),
                        "title": overview.get("title", f"{domain.title()} overview".strip()),
                        "keywords": [domain, version, "overview"],
                        "text": overview["text"],
                        "priority": overview.get("priority", 1.0),
                    },
                    source_fallback=source_fallback,
                )
            )

        section_kinds = {
            "workflows": "workflow",
            "modules": "module",
            "interface": "interface",
            "parameters": "parameter",
            "concepts": "concept",
            "inputs_outputs": "output",
            "tutorials": "tutorial",
            "troubleshooting": "diagnostic",
        }
        for section_name, default_kind in section_kinds.items():
            section_records = payload.get(section_name) or []
            if not isinstance(section_records, list):
                raise ValueError(f"{source_fallback}: section {section_name} must be a list")
            for index, section_record in enumerate(section_records):
                if not isinstance(section_record, dict):
                    raise ValueError(
                        f"{source_fallback}: section {section_name} record {index} must be an object"
                    )
                chunks.append(
                    cls._record_to_chunk(
                        {
                            "id": section_record.get("id")
                            or f"{domain or 'knowledge'}-{section_name}-{index + 1}",
                            "source": section_record.get("source", source_fallback),
                            "topic": section_record.get("topic", section_name),
                            "kind": section_record.get("kind", default_kind),
                            "tab": section_record.get("tab", ""),
                            "program": section_record.get("program", domain.upper()),
                            "module": section_record.get("module", section_name.upper()),
                            "title": section_record.get("title", ""),
                            "keywords": section_record.get("keywords", []),
                            "question_patterns": section_record.get("question_patterns", []),
                            "text": section_record.get("text", ""),
                            "priority": section_record.get("priority", 0.7),
                            "source_file": section_record.get("source_file", ""),
                            "audience": section_record.get("audience", ""),
                            "source_tier": section_record.get("source_tier", ""),
                            "entity_type": section_record.get("entity_type", ""),
                            "entity_id": section_record.get("entity_id", ""),
                            "aliases": section_record.get("aliases", []),
                            "related_ids": section_record.get("related_ids", []),
                        },
                        source_fallback=source_fallback,
                    )
                )
        return chunks

    @classmethod
    def _load_tutorial_chunks(cls, tutorials_dir: Path) -> list[KnowledgeChunk]:
        if not tutorials_dir.exists():
            return []

        chunks: list[KnowledgeChunk] = []
        for path in sorted(tutorials_dir.glob("*.md")):
            raw_text = path.read_text(encoding="utf-8")
            blocks = [block.strip() for block in raw_text.split("---") if block.strip()]
            for index, block in enumerate(blocks, start=1):
                text = _clean_tutorial_text(block)
                if not text:
                    continue
                tab = _infer_tutorial_tab(path.stem, text)
                chunk_id = f"tutorial-{path.stem}-{index}"
                chunks.append(
                    KnowledgeChunk(
                        id=chunk_id,
                        source=f"tutorial:{path.name}",
                        topic="tutorial",
                        tab=tab,
                        keywords=_tutorial_keywords(path.stem, tab, text),
                        text=text,
                        source_file=path.name,
                        program="ROBERT",
                        module=tab.upper().replace(" ", "_"),
                        kind="tutorial",
                        title=_default_title(text, chunk_id),
                        question_patterns=(path.stem.replace("_", " ").replace("-", " "), tab),
                        priority=0.65,
                    )
                )
        return chunks

    @staticmethod
    def _record_to_chunk(record: dict[str, object], source_fallback: str) -> KnowledgeChunk:
        chunk_id = str(record.get("id", "")).strip()
        text = str(record.get("text", "")).strip()
        if not chunk_id:
            raise ValueError("knowledge record is missing required field: id")
        if not text:
            raise ValueError("knowledge record is missing required field: text")

        keywords = tuple(str(keyword) for keyword in _ensure_sequence(record.get("keywords")))
        question_patterns = tuple(
            str(pattern) for pattern in _ensure_sequence(record.get("question_patterns"))
        )
        aliases = tuple(str(alias) for alias in _ensure_sequence(record.get("aliases")))
        related_ids = tuple(
            str(related_id) for related_id in _ensure_sequence(record.get("related_ids"))
        )
        source = str(record.get("source") or source_fallback)
        kind = str(record.get("kind") or record.get("topic") or "concept")
        return KnowledgeChunk(
            id=chunk_id,
            source=source,
            topic=str(record.get("topic") or kind),
            tab=str(record.get("tab", "")),
            keywords=keywords,
            text=text,
            source_file=str(record.get("source_file", "")),
            program=str(record.get("program", "")),
            module=str(record.get("module", "")),
            kind=kind,
            title=str(record.get("title") or "").strip() or _default_title(text, chunk_id),
            question_patterns=question_patterns,
            priority=_coerce_priority(record.get("priority"), default=0.5),
            audience=str(record.get("audience", "")),
            source_tier=str(record.get("source_tier", "")),
            entity_type=str(record.get("entity_type", "")),
            entity_id=str(record.get("entity_id", "")),
            aliases=aliases,
            related_ids=related_ids,
        )

    @staticmethod
    def _is_aqme_related(chunk: KnowledgeChunk) -> bool:
        module = str(chunk.module or "").upper()
        text = f"{chunk.title} {chunk.text} {' '.join(chunk.keywords)} {chunk.question_patterns}".lower()
        return (
            module in _AQME_MODULES
            or str(chunk.tab or "") == "AQME"
            or str(chunk.program or "").upper() == "AQME"
            or any(term in text for term in ("aqme", "smiles", "qdescp", "chemdraw", "descriptor generation"))
        )

    @staticmethod
    def _is_robert_related(chunk: KnowledgeChunk) -> bool:
        module = str(chunk.module or "").upper()
        text = f"{chunk.title} {chunk.text} {' '.join(chunk.keywords)} {chunk.question_patterns}".lower()
        return (
            module in _ROBERT_ML_MODULES
            or str(chunk.program or "").upper() == "ROBERT"
            or str(chunk.tab or "") in {"ROBERT", "Advanced Options", "Reports", "Predictions", "Images"}
            or any(term in text for term in ("machine learning", "model", "curate", "generate", "verify", "predict", "report"))
        )

    @staticmethod
    def _is_ui_related(chunk: KnowledgeChunk) -> bool:
        source = str(chunk.source or "")
        text = f"{chunk.title} {chunk.text} {' '.join(chunk.keywords)} {chunk.question_patterns}".lower()
        return (
            source.startswith("tutorial:")
            or source == "easyrob_code_docstrings"
            or str(chunk.tab or "") in _UI_TAB_NAMES
            and any(term in text for term in (" tab", " button", "gui", "interface", "settings", "popup", "window"))
        )

    @staticmethod
    def _lexical_score(
        chunk: KnowledgeChunk,
        question_terms: set[str],
        workflow_terms: set[str],
        console_terms: set[str],
    ) -> float:
        chunk_terms = _tokenize(
            (
                chunk.topic,
                chunk.tab,
                chunk.text,
                chunk.keywords,
                chunk.title,
                chunk.module,
                chunk.program,
                chunk.kind,
                chunk.question_patterns,
            )
        )
        keyword_terms = _tokenize(chunk.keywords)
        topic_terms = _tokenize((chunk.topic, chunk.tab))
        text_terms = _tokenize(chunk.text)

        question_overlap = len(question_terms & chunk_terms)
        keyword_overlap = len(question_terms & keyword_terms)
        workflow_overlap = len(workflow_terms & (chunk_terms | keyword_terms | topic_terms))
        console_overlap = len(console_terms & (chunk_terms | text_terms | topic_terms))

        return (
            question_overlap * 1.0
            + keyword_overlap * 2.0
            + workflow_overlap * 2.5
            + console_overlap * 1.5
        )

    @staticmethod
    def _metadata_score(chunk: KnowledgeChunk, question_terms: set[str]) -> float:
        title_terms = _tokenize(chunk.title)
        module_terms = _tokenize((chunk.module, chunk.program))
        pattern_terms = _tokenize(chunk.question_patterns)
        return (
            len(question_terms & title_terms) * 2.0
            + len(question_terms & module_terms) * 2.0
            + len(question_terms & pattern_terms) * 3.0
            + chunk.priority
        )

    @staticmethod
    def _pattern_phrase_bonus(chunk: KnowledgeChunk, question: str) -> float:
        normalized_question = _normalize_text(question)
        if not normalized_question:
            return 0.0
        bonus = 0.0
        for pattern in chunk.question_patterns:
            normalized_pattern = _normalize_text(pattern)
            if not normalized_pattern:
                continue
            if normalized_question == normalized_pattern:
                bonus = max(bonus, 6.0)
            elif normalized_pattern in normalized_question:
                bonus = max(bonus, 3.0)
        return bonus

    @staticmethod
    def _question_specific_penalty(chunk: KnowledgeChunk, question: str) -> float:
        normalized_question = _normalize_text(question)
        title_text = _normalize_text(f"{chunk.title} {chunk.text}")
        penalty = 0.0
        if "button" in title_text and "button" not in normalized_question:
            penalty -= 3.5
        if "tab" in title_text and "tab" not in normalized_question:
            penalty -= 1.5
        return penalty

    @staticmethod
    def _contextual_bonus(chunk: KnowledgeChunk, active_tab: str | None) -> float:
        active_terms = _tokenize(active_tab)
        tab_terms = _tokenize(chunk.tab)
        return 4.0 if active_terms and active_terms <= tab_terms else 0.0

    @staticmethod
    def _question_tab_bonus(chunk: KnowledgeChunk, question_terms: set[str]) -> float:
        tab_terms = _tokenize(chunk.tab)
        if not tab_terms:
            return 0.0
        overlap = question_terms & tab_terms
        if not overlap:
            return 0.0
        return 5.0 + float(len(overlap) - 1)

    @staticmethod
    def _kind_bonus(chunk: KnowledgeChunk, question_intent: str) -> float:
        kind_weights = {
            "diagnostic": {"diagnostic": 4.0, "workflow": 2.0, "output": 1.0},
            "tutorial": {"tutorial": 4.0, "concept": 2.0, "workflow": 1.5},
            "results": {"results": 4.0, "output": 2.5, "concept": 1.5},
            "parameter": {"parameter": 4.5, "tutorial": 1.5, "concept": 1.0},
            "module-explanation": {"concept": 4.0, "tutorial": 2.0},
        }
        return kind_weights.get(question_intent, {}).get(chunk.kind, 0.0)

    @staticmethod
    def _program_bonus(chunk: KnowledgeChunk, question_terms: set[str]) -> float:
        program_terms = _tokenize(chunk.program)
        if program_terms and question_terms & program_terms:
            return 2.0
        return 0.0

    @staticmethod
    def _source_bonus(chunk: KnowledgeChunk, question_intent: str) -> float:
        if chunk.source.startswith("tutorial:"):
            return 3.0 if question_intent == "tutorial" else -0.25
        if chunk.source == "curated_overrides":
            return 3.0
        if chunk.source in {"robert_docs", "aqme_docs"}:
            return 0.4
        return 0.0

    @staticmethod
    def _quality_penalty(chunk: KnowledgeChunk) -> float:
        penalty = 0.0
        text = chunk.text or ""
        lowered = text.lower()
        if len(text) > 1800:
            penalty -= 1.5
        if any(marker in lowered for marker in (".. ", "image::", "raw::", "<div", "code::")):
            penalty -= 2.0
        if any(marker in lowered for marker in ("<input", "circleci", "codecov", "downloads", "robert_banner", "aqme_banner", ":local:")):
            penalty -= 5.0
        if len(text.strip()) < 18:
            penalty -= 3.0
        if str(chunk.title or "").strip().lower() in _LOW_SIGNAL_TITLES:
            penalty -= 1.5
        if chunk.source in {"robert_code_docstrings", "aqme_code_docstrings"} and str(chunk.title or "").strip().lower() in {"module", "main"}:
            penalty -= 2.5
        return penalty

    @classmethod
    def _focus_bonus(cls, chunk: KnowledgeChunk, focus_tags: set[str]) -> float:
        if not focus_tags:
            return 0.0

        aqme_related = cls._is_aqme_related(chunk)
        robert_related = cls._is_robert_related(chunk)
        ui_related = cls._is_ui_related(chunk)
        bonus = 0.0

        if "aqme" in focus_tags:
            if aqme_related:
                bonus += 3.5
            elif robert_related and "robert" not in focus_tags:
                bonus -= 2.5
        if "robert" in focus_tags:
            if robert_related:
                bonus += 3.5
            elif aqme_related and "aqme" not in focus_tags:
                bonus -= 2.5
        if "ui" in focus_tags:
            if ui_related:
                bonus += 2.0
            elif focus_tags == {"ui"}:
                bonus -= 0.5
        return bonus

    def _bm25_score(self, item: IndexedChunk, query_terms: tuple[str, ...]) -> float:
        if not self._document_count:
            return 0.0
        length_ratio = (
            item.body_length / self._average_body_length
            if self._average_body_length > 0.0 else 1.0
        )
        score = 0.0
        for term in set(query_terms):
            frequency = self._document_frequencies.get(term, 0)
            if not frequency:
                continue
            idf = math.log(
                1.0 + (self._document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for field, weight in FIELD_WEIGHTS.items():
                term_frequency = item.field_counters[field].get(term, 0)
                if not term_frequency:
                    continue
                normalization = (
                    1.0 - BM25_B + BM25_B * length_ratio
                    if field == "text" else 1.0
                )
                saturation = (
                    term_frequency * (BM25_K1 + 1.0)
                    / (term_frequency + BM25_K1 * normalization)
                )
                score += weight * idf * saturation
        return score

    @staticmethod
    def _bounded_metadata_score(
        item: IndexedChunk,
        question_terms: tuple[str, ...],
        active_tab_terms: set[str],
        workflow_terms: set[str],
        console_terms: set[str],
        intent: str,
        focus_tags: set[str],
    ) -> float:
        chunk = item.chunk
        term_set = set(question_terms)
        tab_terms = set(item.field_counters["tab"])
        program_terms = set(item.field_counters["program"])
        module_terms = set(item.field_counters["module"])
        explicit_tab = bool(tab_terms & term_set)
        active_tab_match = bool(tab_terms) and tab_terms <= active_tab_terms

        metadata = 0.0
        if explicit_tab:
            metadata += 2.0
        elif active_tab_match:
            metadata += 1.0

        intent_weights = {
            "diagnostic": {"diagnostic": 2.0, "workflow": 1.0},
            "tutorial": {"tutorial": 1.5, "concept": 0.75, "workflow": 0.5},
            "results": {"results": 2.0, "output": 1.25},
            "parameter": {"parameter": 2.25, "tutorial": 0.5},
            "module-explanation": {"concept": 2.0, "tutorial": 0.75},
        }
        metadata += intent_weights.get(intent, {}).get(chunk.kind.casefold(), 0.0)

        tier = chunk.source_tier.casefold()
        ui_workflow_focus = (
            intent == "tutorial"
            or chunk.kind.casefold() in {"interface", "tutorial", "workflow"}
            or bool(term_set & {"button", "tab", "workflow", "full_workflow"})
            or "ui" in focus_tags
        )
        if tier == "curated" and ui_workflow_focus:
            metadata += 3.0
        elif tier == "tutorial" and intent == "tutorial":
            metadata += 1.5
        elif tier == "reference":
            metadata += 0.25

        if program_terms & term_set:
            metadata += 1.0
        if workflow_terms & (program_terms | module_terms | tab_terms):
            metadata += 0.75
        indexed_terms = set().union(
            *(counter.keys() for counter in item.field_counters.values())
        )
        if intent == "diagnostic" and console_terms & indexed_terms:
            metadata += 0.75
        title_terms = set(item.field_counters["title"])
        if term_set and term_set <= title_terms:
            metadata += 3.0 / (1.0 + len(title_terms - term_set))
        # K5 records used this source marker before source_tier was introduced.
        if chunk.source == "curated_overrides":
            metadata += 2.0
        elif chunk.source.endswith("_code_docstrings"):
            metadata -= 3.0
        if focus_tags:
            related = set(item.field_counters["program"]) | set(item.field_counters["tab"])
            for tag in focus_tags:
                if tag in related and tag not in term_set:
                    metadata += 1.0
                elif tag in {"aqme", "robert"}:
                    metadata -= 0.75

        metadata = max(-3.0, min(6.0, metadata))
        priority = chunk.priority
        if not math.isfinite(priority):
            return math.nan
        return metadata + max(0.0, min(0.5, priority))

    def search(
        self,
        question: str,
        active_tab: str | None = None,
        workflow: str | None = None,
        console_terms: Iterable[str] | None = None,
        limit: int = 5,
        question_intent: str | None = None,
        focus_tags: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []

        question_terms = normalize_terms(question)
        if not question_terms:
            return []
        question_pattern = _canonical_question_pattern(question)
        person_query = bool(question_pattern and question_pattern[0] == "who")
        # Workflow/console context must never flood the explicit-question BM25 query.
        retrieval_terms = question_terms
        active_tab_terms = set(normalize_terms(active_tab)) if active_tab else set()
        workflow_terms = set(normalize_terms(workflow)) if workflow else set()
        console_term_values = tuple(console_terms) if console_terms is not None else ()
        console_query_terms = (
            set(normalize_terms(console_term_values))
            if console_term_values else set()
        )
        intent = str(question_intent or classify_question_intent(question)).strip() or "tutorial"
        focus_tag_set = {str(tag).strip().lower() for tag in (focus_tags or []) if str(tag).strip()}

        scored: list[SearchResult] = []
        for item in self._index:
            if person_query and not any(
                item.field_counters[field].get("who", 0)
                for field in ("question_patterns", "title", "keywords", "aliases")
            ):
                continue
            lexical_score = self._bm25_score(item, retrieval_terms)
            if lexical_score <= 0.0:
                continue
            metadata_score = self._bounded_metadata_score(
                item,
                question_terms,
                active_tab_terms,
                workflow_terms,
                console_query_terms,
                intent,
                focus_tag_set,
            )
            score = lexical_score + metadata_score
            if not math.isfinite(score) or score < MIN_RELEVANCE_SCORE:
                continue
            indexed_terms = {
                term
                for counter in item.field_counters.values()
                for term in counter
            }
            matched_terms = tuple(sorted(set(question_terms) & indexed_terms))
            scored.append(
                SearchResult(
                    chunk=item.chunk,
                    score=score,
                    matched_terms=matched_terms,
                    query_terms=tuple(sorted(set(question_terms))),
                )
            )
        query_signature = question_pattern

        def ranking_key(result: SearchResult) -> tuple[bool, float, str]:
            exact_curated_pattern = (
                result.chunk.source_tier.casefold() == "curated"
                and any(
                    _canonical_question_pattern(pattern) == query_signature
                    for pattern in result.chunk.question_patterns
                )
            )
            return (not exact_curated_pattern, -result.score, result.chunk.id)

        scored.sort(key=ranking_key)
        unique: list[SearchResult] = []
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        fingerprints = {id(item.chunk): item.content_fingerprint for item in self._index}
        for result in scored:
            fingerprint = fingerprints[id(result.chunk)]
            if result.chunk.id in seen_ids or fingerprint in seen_content:
                continue
            seen_ids.add(result.chunk.id)
            seen_content.add(fingerprint)
            unique.append(result)
            if len(unique) >= limit:
                break
        return unique
