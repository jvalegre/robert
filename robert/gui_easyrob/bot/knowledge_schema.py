from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
CONTEXT_VERSION = "2026-07-10"
DOMAINS = ("robert", "aqme", "ui")
CONTEXT_SECTIONS = (
    "workflows",
    "modules",
    "interface",
    "parameters",
    "concepts",
    "inputs_outputs",
    "tutorials",
    "troubleshooting",
)
ALLOWED_KINDS = {
    "concept",
    "workflow",
    "module",
    "interface",
    "parameter",
    "output",
    "tutorial",
    "diagnostic",
    "results",
}
ALLOWED_SOURCE_TIERS = {"curated", "tutorial", "reference", "generated"}
ALLOWED_AUDIENCES = {"end_user", "advanced_user", "developer_reference"}
ALLOWED_ENTITY_TYPES = {
    "program",
    "module",
    "tab",
    "button",
    "selector",
    "checkbox",
    "dialog",
    "status",
    "workflow",
    "tutorial",
}
ALLOWED_TABS = {
    "",
    "ROBERT",
    "AQME",
    "Advanced Options",
    "Results",
    "Reports",
    "Predictions",
    "Evaluate",
    "Check model",
    "Images",
    "MolSSI Databases",
}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


_ALLOWED_PROGRAMS = {"ROBERT", "AQME"}
_CONTEXT_TOP_LEVEL_KEYS = {
    "schema_version",
    "version",
    "domain",
    "overview",
    *CONTEXT_SECTIONS,
}
_GUIDANCE_TOP_LEVEL_KEYS = {"schema_version", "version", "domain", "records"}
_MANIFEST_TOP_LEVEL_KEYS = {
    "schema_version",
    "version",
    "required_entities",
    "tutorial_files",
}
_REQUIRED_RECORD_STRINGS = (
    "id",
    "source",
    "source_file",
    "program",
    "module",
    "kind",
    "topic",
    "tab",
    "title",
)
_GUIDANCE_LIST_FIELDS = (
    "when_to_use",
    "prerequisites",
    "steps",
    "next_steps",
    "common_issues",
)


@dataclass(frozen=True, slots=True)
class KnowledgeLoadIssue:
    code: str
    message: str
    source: str = ""
    record_id: str = ""


class KnowledgeValidationError(ValueError):
    def __init__(self, issues: Sequence[KnowledgeLoadIssue]):
        self.issues = tuple(issues)
        detail = "; ".join(
            f"{issue.code}: {issue.message}" for issue in self.issues
        )
        super().__init__(detail or "knowledge validation failed")


def _add_issue(
    issues: list[KnowledgeLoadIssue],
    code: str,
    message: str,
    source: str,
    record_id: str = "",
) -> None:
    issues.append(
        KnowledgeLoadIssue(
            code=code,
            message=message,
            source=source,
            record_id=record_id,
        )
    )


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(
        _is_nonempty_string(item) for item in value
    )


def _valid_schema_version(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_version(value: object) -> bool:
    return isinstance(value, str) and ISO_DATE_RE.fullmatch(value) is not None


def _record_identifier(record: Mapping[object, object]) -> str:
    value = record.get("id")
    return value if _is_nonempty_string(value) else ""


def _validate_required_string(
    record: Mapping[object, object],
    field: str,
    issues: list[KnowledgeLoadIssue],
    source: str,
    record_id: str,
    *,
    allow_empty: bool = False,
) -> object:
    if field not in record:
        _add_issue(
            issues,
            f"missing_{field}",
            f"Record is missing required field '{field}'.",
            source,
            record_id,
        )
        return None

    value = record[field]
    valid = isinstance(value, str) and (allow_empty or bool(value.strip()))
    if not valid:
        _add_issue(
            issues,
            f"invalid_{field}",
            f"Record field '{field}' must be "
            + ("a string." if allow_empty else "a non-empty string."),
            source,
            record_id,
        )
    return value


def _validate_string_list(
    record: Mapping[object, object],
    field: str,
    issues: list[KnowledgeLoadIssue],
    source: str,
    record_id: str,
    *,
    required: bool,
) -> object:
    if field not in record:
        if required:
            _add_issue(
                issues,
                f"missing_{field}",
                f"Record is missing required list field '{field}'.",
                source,
                record_id,
            )
        return None

    value = record[field]
    if not _is_string_list(value):
        _add_issue(
            issues,
            f"invalid_{field}",
            f"Record field '{field}' must be a list of non-empty strings.",
            source,
            record_id,
        )
    return value


def _validate_user_guidance(
    guidance: object,
    entity_type: object,
    kind: object,
    issues: list[KnowledgeLoadIssue],
    source: str,
    record_id: str,
) -> None:
    if not isinstance(guidance, Mapping):
        _add_issue(
            issues,
            "invalid_user_guidance",
            "Curated record field 'user_guidance' must be an object.",
            source,
            record_id,
        )
        return

    for field in ("purpose", "result"):
        if field not in guidance:
            _add_issue(
                issues,
                f"missing_{field}",
                f"Curated guidance is missing required field '{field}'.",
                source,
                record_id,
            )
        elif not _is_nonempty_string(guidance[field]):
            _add_issue(
                issues,
                f"invalid_{field}",
                f"Curated guidance field '{field}' must be a non-empty string.",
                source,
                record_id,
            )

    valid_lists: dict[str, list[object]] = {}
    for field in _GUIDANCE_LIST_FIELDS:
        if field not in guidance:
            _add_issue(
                issues,
                f"missing_{field}",
                f"Curated guidance is missing required list field '{field}'.",
                source,
                record_id,
            )
            continue
        value = guidance[field]
        if not _is_string_list(value):
            _add_issue(
                issues,
                f"invalid_{field}",
                f"Curated guidance field '{field}' must be a list of "
                "non-empty strings.",
                source,
                record_id,
            )
            continue
        valid_lists[field] = value

    guidance_type = ""
    if isinstance(entity_type, str) and entity_type in {"workflow", "tutorial"}:
        guidance_type = entity_type
    elif isinstance(kind, str) and kind in {"workflow", "tutorial"}:
        guidance_type = kind
    if guidance_type:
        for field in ("steps", "next_steps"):
            if field in valid_lists and not valid_lists[field]:
                _add_issue(
                    issues,
                    f"missing_{field}",
                    f"{guidance_type.title()} guidance requires at least one "
                    f"'{field}' item.",
                    source,
                    record_id,
                )

    if kind == "diagnostic" and "common_issues" in valid_lists:
        if not valid_lists["common_issues"]:
            _add_issue(
                issues,
                "missing_common_issues",
                "Diagnostic guidance requires at least one common issue.",
                source,
                record_id,
            )


def _validate_record(
    record: Mapping[object, object],
    issues: list[KnowledgeLoadIssue],
    source: str,
    *,
    require_text: bool,
    require_curated: bool,
) -> None:
    record_id = _record_identifier(record)

    for field in _REQUIRED_RECORD_STRINGS:
        _validate_required_string(
            record,
            field,
            issues,
            source,
            record_id,
            allow_empty=field == "tab",
        )

    if require_text or "text" in record:
        _validate_required_string(
            record,
            "text",
            issues,
            source,
            record_id,
        )

    program = record.get("program")
    if isinstance(program, str) and program not in _ALLOWED_PROGRAMS:
        _add_issue(
            issues,
            "invalid_program",
            "Record field 'program' must be 'ROBERT' or 'AQME'.",
            source,
            record_id,
        )

    kind = record.get("kind")
    if isinstance(kind, str) and kind not in ALLOWED_KINDS:
        _add_issue(
            issues,
            "invalid_kind",
            f"Unsupported knowledge kind '{kind}'.",
            source,
            record_id,
        )

    tab = record.get("tab")
    if isinstance(tab, str) and tab not in ALLOWED_TABS:
        _add_issue(
            issues,
            "invalid_tab",
            f"Unsupported EasyROB tab '{tab}'.",
            source,
            record_id,
        )

    source_tier = record.get("source_tier")
    if "source_tier" not in record:
        _add_issue(
            issues,
            "missing_source_tier",
            "Record is missing required field 'source_tier'.",
            source,
            record_id,
        )
    elif not isinstance(source_tier, str) or source_tier not in ALLOWED_SOURCE_TIERS:
        _add_issue(
            issues,
            "invalid_source_tier",
            "Record field 'source_tier' is not supported.",
            source,
            record_id,
        )

    audience = record.get("audience")
    if "audience" not in record:
        _add_issue(
            issues,
            "missing_audience",
            "Record is missing required field 'audience'.",
            source,
            record_id,
        )
    elif not isinstance(audience, str) or audience not in ALLOWED_AUDIENCES:
        _add_issue(
            issues,
            "invalid_audience",
            "Record field 'audience' is not supported.",
            source,
            record_id,
        )

    if "priority" not in record:
        _add_issue(
            issues,
            "missing_priority",
            "Record is missing required field 'priority'.",
            source,
            record_id,
        )
    else:
        priority = record["priority"]
        if isinstance(priority, bool):
            valid_priority = False
        elif isinstance(priority, int):
            valid_priority = 0 <= priority <= 1
        elif isinstance(priority, float):
            valid_priority = math.isfinite(priority) and 0.0 <= priority <= 1.0
        else:
            valid_priority = False
        if not valid_priority:
            _add_issue(
                issues,
                "invalid_priority",
                "Record priority must be finite and within 0.0 through 1.0.",
                source,
                record_id,
            )

    for field in ("keywords", "question_patterns"):
        _validate_string_list(
            record,
            field,
            issues,
            source,
            record_id,
            required=True,
        )
    for field in ("aliases", "related_ids"):
        _validate_string_list(
            record,
            field,
            issues,
            source,
            record_id,
            required=False,
        )

    has_curated_metadata = require_curated or source_tier == "curated" or any(
        field in record for field in ("entity_type", "entity_id", "user_guidance")
    )
    if not has_curated_metadata:
        return

    if source_tier != "curated":
        _add_issue(
            issues,
            "invalid_curated_source_tier",
            "Curated guidance records require source_tier='curated'.",
            source,
            record_id,
        )
    if audience != "end_user":
        _add_issue(
            issues,
            "invalid_curated_audience",
            "Curated guidance records require audience='end_user'.",
            source,
            record_id,
        )

    entity_type = record.get("entity_type")
    if "entity_type" not in record:
        _add_issue(
            issues,
            "missing_entity_type",
            "Curated record is missing required field 'entity_type'.",
            source,
            record_id,
        )
    elif not isinstance(entity_type, str) or entity_type not in ALLOWED_ENTITY_TYPES:
        _add_issue(
            issues,
            "invalid_entity_type",
            "Curated record field 'entity_type' is not supported.",
            source,
            record_id,
        )

    _validate_required_string(
        record,
        "entity_id",
        issues,
        source,
        record_id,
    )
    for field in ("aliases", "related_ids"):
        if field not in record:
            _validate_string_list(
                record,
                field,
                issues,
                source,
                record_id,
                required=True,
            )

    if isinstance(entity_type, str) and entity_type in {
        "tab",
        "button",
        "selector",
        "checkbox",
    }:
        if not _is_nonempty_string(tab):
            _add_issue(
                issues,
                "missing_tab",
                f"Curated {entity_type} records require a non-empty tab.",
                source,
                record_id,
            )

    if "user_guidance" not in record:
        _add_issue(
            issues,
            "missing_user_guidance",
            "Curated record is missing required field 'user_guidance'.",
            source,
            record_id,
        )
    else:
        _validate_user_guidance(
            record["user_guidance"],
            entity_type,
            kind,
            issues,
            source,
            record_id,
        )


def validate_context_payload(
    payload: object,
    expected_domain: str,
    source: str,
) -> tuple[KnowledgeLoadIssue, ...]:
    issues: list[KnowledgeLoadIssue] = []
    if not isinstance(payload, Mapping):
        _add_issue(
            issues,
            "invalid_payload",
            "Knowledge context payload must be an object.",
            source,
        )
        return tuple(issues)

    for key in sorted(
        (key for key in payload if key not in _CONTEXT_TOP_LEVEL_KEYS),
        key=str,
    ):
        _add_issue(
            issues,
            "unexpected_top_level_key",
            f"Unexpected context field {key!r}.",
            source,
        )

    if "schema_version" not in payload:
        _add_issue(
            issues,
            "missing_schema_version",
            "Context is missing 'schema_version'.",
            source,
        )
    elif not _valid_schema_version(payload["schema_version"]) or payload[
        "schema_version"
    ] != SCHEMA_VERSION:
        _add_issue(
            issues,
            "invalid_schema_version",
            f"Context schema_version must be {SCHEMA_VERSION}.",
            source,
        )

    if "version" not in payload:
        _add_issue(
            issues,
            "missing_version",
            "Context is missing 'version'.",
            source,
        )
    elif not _valid_version(payload["version"]):
        _add_issue(
            issues,
            "invalid_version",
            "Context version must use ISO YYYY-MM-DD format.",
            source,
        )

    if "domain" not in payload:
        _add_issue(
            issues,
            "missing_domain",
            "Context is missing 'domain'.",
            source,
        )
    else:
        domain = payload["domain"]
        if not isinstance(domain, str) or domain not in DOMAINS:
            _add_issue(
                issues,
                "invalid_domain",
                "Context domain is not supported.",
                source,
            )
        if domain != expected_domain:
            _add_issue(
                issues,
                "domain_mismatch",
                f"Expected domain '{expected_domain}', found {domain!r}.",
                source,
            )

    if "overview" not in payload:
        _add_issue(
            issues,
            "missing_overview",
            "Context is missing 'overview'.",
            source,
        )
    else:
        overview = payload["overview"]
        if not isinstance(overview, Mapping):
            _add_issue(
                issues,
                "invalid_overview",
                "Context overview must be an object.",
                source,
            )
        else:
            for field in ("title", "text"):
                if field not in overview:
                    _add_issue(
                        issues,
                        f"missing_overview_{field}",
                        f"Context overview is missing '{field}'.",
                        source,
                    )
                elif not _is_nonempty_string(overview[field]):
                    _add_issue(
                        issues,
                        f"invalid_overview_{field}",
                        f"Context overview '{field}' must be a non-empty string.",
                        source,
                    )

    for section in CONTEXT_SECTIONS:
        if section not in payload:
            _add_issue(
                issues,
                "missing_section",
                f"Context is missing required list section '{section}'.",
                source,
            )
            continue
        records = payload[section]
        if not isinstance(records, list):
            _add_issue(
                issues,
                "invalid_section",
                f"Context section '{section}' must be a list.",
                source,
            )
            continue
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                _add_issue(
                    issues,
                    "invalid_record",
                    f"Context section '{section}' record {index} must be an object.",
                    source,
                )
                continue
            _validate_record(
                record,
                issues,
                source,
                require_text=True,
                require_curated=False,
            )

    return tuple(issues)


def _context_records(
    payloads: Mapping[str, object],
) -> list[tuple[str, str, Mapping[object, object]]]:
    records: list[tuple[str, str, Mapping[object, object]]] = []
    for domain in DOMAINS:
        payload = payloads.get(domain)
        if not isinstance(payload, Mapping):
            continue
        source = f"{domain}_context.json"
        for section in CONTEXT_SECTIONS:
            section_records = payload.get(section)
            if not isinstance(section_records, list):
                continue
            records.extend(
                (domain, source, record)
                for record in section_records
                if isinstance(record, Mapping)
            )
    return records


def validate_context_set(
    payloads: Mapping[str, object],
) -> tuple[KnowledgeLoadIssue, ...]:
    issues: list[KnowledgeLoadIssue] = []
    if not isinstance(payloads, Mapping):
        _add_issue(
            issues,
            "invalid_context_set",
            "Knowledge contexts must be supplied as a domain mapping.",
            "context set",
        )
        return tuple(issues)

    for domain in DOMAINS:
        if domain not in payloads:
            _add_issue(
                issues,
                "missing_context_domain",
                f"Context set is missing domain '{domain}'.",
                "context set",
            )
            continue
        issues.extend(
            validate_context_payload(
                payloads[domain],
                expected_domain=domain,
                source=f"{domain}_context.json",
            )
        )

    for domain in sorted(
        (domain for domain in payloads if domain not in DOMAINS), key=str
    ):
        _add_issue(
            issues,
            "unexpected_context_domain",
            f"Context set contains unexpected domain {domain!r}.",
            "context set",
        )

    schema_versions = [
        payload["schema_version"]
        for domain in DOMAINS
        if isinstance((payload := payloads.get(domain)), Mapping)
        and "schema_version" in payload
    ]
    if schema_versions and any(
        value != schema_versions[0] for value in schema_versions[1:]
    ):
        _add_issue(
            issues,
            "schema_version_mismatch",
            "All knowledge contexts must share one schema_version.",
            "context set",
        )

    versions = [
        payload["version"]
        for domain in DOMAINS
        if isinstance((payload := payloads.get(domain)), Mapping)
        and "version" in payload
    ]
    if versions and any(value != versions[0] for value in versions[1:]):
        _add_issue(
            issues,
            "version_mismatch",
            "All knowledge contexts must share one version.",
            "context set",
        )

    records = _context_records(payloads)
    identifier_registry: dict[str, tuple[str, str, str]] = {}
    for _domain, record_source, record in records:
        record_id = _record_identifier(record)
        entity_id = record.get("entity_id")
        record_identifiers = (
            ("record id", record_id),
            ("entity_id", entity_id if _is_nonempty_string(entity_id) else ""),
        )
        for identifier_kind, identifier in record_identifiers:
            if not identifier:
                continue
            previous = identifier_registry.get(identifier)
            if previous is not None:
                previous_kind, previous_source, previous_record_id = previous
                code = (
                    "duplicate_entity_id"
                    if previous_kind == identifier_kind == "entity_id"
                    else "duplicate_id"
                )
                _add_issue(
                    issues,
                    code,
                    f"Knowledge identifier '{identifier}' used as {identifier_kind} "
                    f"conflicts with {previous_kind} from record "
                    f"'{previous_record_id}' in '{previous_source}'.",
                    record_source,
                    record_id,
                )
            else:
                identifier_registry[identifier] = (
                    identifier_kind,
                    record_source,
                    record_id,
                )

    resolvable_ids = set(identifier_registry)
    for _domain, record_source, record in records:
        related_ids = record.get("related_ids")
        if not _is_string_list(related_ids):
            continue
        record_id = _record_identifier(record)
        for related_id in related_ids:
            if related_id not in resolvable_ids:
                _add_issue(
                    issues,
                    "unresolved_related_id",
                    f"Related ID '{related_id}' does not resolve to a knowledge record.",
                    record_source,
                    record_id,
                )

    return tuple(issues)


def _validate_guidance_payload(
    payload: object,
    expected_domain: str,
    source: str,
    issues: list[KnowledgeLoadIssue],
) -> list[Mapping[object, object]]:
    if not isinstance(payload, Mapping):
        _add_issue(
            issues,
            "invalid_guidance_payload",
            f"Guidance payload for '{expected_domain}' must be an object.",
            source,
        )
        return []

    for key in sorted(
        (key for key in payload if key not in _GUIDANCE_TOP_LEVEL_KEYS), key=str
    ):
        _add_issue(
            issues,
            "unexpected_top_level_key",
            f"Unexpected guidance source field {key!r}.",
            source,
        )

    if "schema_version" not in payload:
        _add_issue(
            issues,
            "missing_schema_version",
            f"Guidance source '{expected_domain}' is missing 'schema_version'.",
            source,
        )
    elif not _valid_schema_version(payload["schema_version"]) or payload[
        "schema_version"
    ] != SCHEMA_VERSION:
        _add_issue(
            issues,
            "invalid_schema_version",
            f"Guidance schema_version must be {SCHEMA_VERSION}.",
            source,
        )

    if "version" not in payload:
        _add_issue(
            issues,
            "missing_version",
            f"Guidance source '{expected_domain}' is missing 'version'.",
            source,
        )
    elif not _valid_version(payload["version"]):
        _add_issue(
            issues,
            "invalid_version",
            "Guidance version must use ISO YYYY-MM-DD format.",
            source,
        )

    if "domain" not in payload:
        _add_issue(
            issues,
            "missing_domain",
            f"Guidance source '{expected_domain}' is missing 'domain'.",
            source,
        )
    else:
        domain = payload["domain"]
        if not isinstance(domain, str) or domain not in DOMAINS:
            _add_issue(
                issues,
                "invalid_domain",
                "Guidance domain is not supported.",
                source,
            )
        if domain != expected_domain:
            _add_issue(
                issues,
                "domain_mismatch",
                f"Expected guidance domain '{expected_domain}', found {domain!r}.",
                source,
            )

    if "records" not in payload:
        _add_issue(
            issues,
            "missing_records",
            f"Guidance source '{expected_domain}' is missing 'records'.",
            source,
        )
        return []
    payload_records = payload["records"]
    if not isinstance(payload_records, list):
        _add_issue(
            issues,
            "invalid_records",
            f"Guidance records for '{expected_domain}' must be a list.",
            source,
        )
        return []

    records: list[Mapping[object, object]] = []
    for index, record in enumerate(payload_records):
        if not isinstance(record, Mapping):
            _add_issue(
                issues,
                "invalid_record",
                f"Guidance record {index} for '{expected_domain}' must be an object.",
                source,
            )
            continue
        records.append(record)
        _validate_record(
            record,
            issues,
            source,
            require_text=False,
            require_curated=True,
        )
    return records


def validate_guidance_source_set(
    payloads: Mapping[str, object],
    manifest: object,
    source: str,
) -> tuple[KnowledgeLoadIssue, ...]:
    issues: list[KnowledgeLoadIssue] = []
    if not isinstance(payloads, Mapping):
        _add_issue(
            issues,
            "invalid_guidance_source_set",
            "Guidance sources must be supplied as a domain mapping.",
            source,
        )
        return tuple(issues)

    records_by_domain: dict[str, list[Mapping[object, object]]] = {}
    for domain in DOMAINS:
        if domain not in payloads:
            _add_issue(
                issues,
                "missing_guidance_domain",
                f"Guidance source set is missing domain '{domain}'.",
                source,
            )
            records_by_domain[domain] = []
            continue
        records_by_domain[domain] = _validate_guidance_payload(
            payloads[domain], domain, source, issues
        )

    for domain in sorted(
        (domain for domain in payloads if domain not in DOMAINS), key=str
    ):
        _add_issue(
            issues,
            "unexpected_guidance_domain",
            f"Guidance source set contains unexpected domain {domain!r}.",
            source,
        )

    required_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    tutorial_files: dict[str, str] = {}
    if not isinstance(manifest, Mapping):
        _add_issue(
            issues,
            "invalid_manifest",
            "Coverage manifest must be an object.",
            source,
        )
    else:
        for key in sorted(
            (key for key in manifest if key not in _MANIFEST_TOP_LEVEL_KEYS), key=str
        ):
            _add_issue(
                issues,
                "unexpected_manifest_key",
                f"Unexpected coverage manifest field {key!r}.",
                source,
            )

        if "schema_version" not in manifest:
            _add_issue(
                issues,
                "missing_schema_version",
                "Coverage manifest is missing 'schema_version'.",
                source,
            )
        elif not _valid_schema_version(manifest["schema_version"]) or manifest[
            "schema_version"
        ] != SCHEMA_VERSION:
            _add_issue(
                issues,
                "invalid_schema_version",
                f"Coverage manifest schema_version must be {SCHEMA_VERSION}.",
                source,
            )

        if "version" not in manifest:
            _add_issue(
                issues,
                "missing_version",
                "Coverage manifest is missing 'version'.",
                source,
            )
        elif not _valid_version(manifest["version"]):
            _add_issue(
                issues,
                "invalid_version",
                "Coverage manifest version must use ISO YYYY-MM-DD format.",
                source,
            )

        required_entities = manifest.get("required_entities")
        if not isinstance(required_entities, Mapping):
            _add_issue(
                issues,
                "invalid_required_entities",
                "Coverage manifest 'required_entities' must be an object.",
                source,
            )
        else:
            for domain in DOMAINS:
                value = required_entities.get(domain)
                if value is None:
                    _add_issue(
                        issues,
                        "missing_manifest_domain",
                        f"Coverage manifest is missing domain '{domain}'.",
                        source,
                    )
                elif not _is_string_list(value):
                    _add_issue(
                        issues,
                        "invalid_required_entities",
                        f"Required entities for '{domain}' must be a list of "
                        "non-empty strings.",
                        source,
                    )
                else:
                    required_by_domain[domain] = value
            for domain in sorted(
                (domain for domain in required_entities if domain not in DOMAINS),
                key=str,
            ):
                _add_issue(
                    issues,
                    "unexpected_manifest_domain",
                    f"Coverage manifest contains unexpected domain {domain!r}.",
                    source,
                )

        raw_tutorial_files = manifest.get("tutorial_files")
        if not isinstance(raw_tutorial_files, Mapping):
            _add_issue(
                issues,
                "invalid_tutorial_files",
                "Coverage manifest 'tutorial_files' must be an object.",
                source,
            )
        else:
            for entity_id, filename in raw_tutorial_files.items():
                if not _is_nonempty_string(entity_id) or not _is_nonempty_string(
                    filename
                ):
                    _add_issue(
                        issues,
                        "invalid_tutorial_mapping",
                        "Tutorial mappings require non-empty string IDs and filenames.",
                        source,
                    )
                    continue
                tutorial_files[entity_id] = filename

    header_values: list[tuple[object, object]] = []
    for domain in DOMAINS:
        payload = payloads.get(domain)
        if isinstance(payload, Mapping):
            header_values.append(
                (payload.get("schema_version"), payload.get("version"))
            )
    if isinstance(manifest, Mapping):
        header_values.append(
            (manifest.get("schema_version"), manifest.get("version"))
        )
    schema_values = [value[0] for value in header_values if value[0] is not None]
    version_values = [value[1] for value in header_values if value[1] is not None]
    if schema_values and any(
        value != schema_values[0] for value in schema_values[1:]
    ):
        _add_issue(
            issues,
            "schema_version_mismatch",
            "Guidance sources and manifest must share one schema_version.",
            source,
        )
    if version_values and any(
        value != version_values[0] for value in version_values[1:]
    ):
        _add_issue(
            issues,
            "version_mismatch",
            "Guidance sources and manifest must share one version.",
            source,
        )

    entity_records: dict[str, tuple[str, Mapping[object, object]]] = {}
    actual_by_domain: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    for domain in DOMAINS:
        for record in records_by_domain[domain]:
            record_id = _record_identifier(record)
            entity_id = record.get("entity_id")
            if not _is_nonempty_string(entity_id):
                continue
            if entity_id in entity_records:
                _add_issue(
                    issues,
                    "duplicate_entity_id",
                    f"Entity ID '{entity_id}' is not globally unique.",
                    source,
                    record_id,
                )
            else:
                entity_records[entity_id] = (domain, record)
            actual_by_domain[domain].add(entity_id)

    required_domains: dict[str, str] = {}
    for domain in DOMAINS:
        required = set(required_by_domain[domain])
        actual = actual_by_domain[domain]
        for entity_id in sorted(required):
            previous_domain = required_domains.get(entity_id)
            if previous_domain is not None and previous_domain != domain:
                _add_issue(
                    issues,
                    "duplicate_required_entity",
                    f"Entity ID '{entity_id}' is required by multiple domains.",
                    source,
                )
            required_domains[entity_id] = domain
            if entity_id not in actual:
                actual_record = entity_records.get(entity_id)
                if actual_record is not None:
                    _add_issue(
                        issues,
                        "entity_domain_mismatch",
                        f"Entity ID '{entity_id}' belongs to '{actual_record[0]}', "
                        f"not '{domain}'.",
                        source,
                        _record_identifier(actual_record[1]),
                    )
                else:
                    _add_issue(
                        issues,
                        "missing_required_entity",
                        f"Required entity ID '{entity_id}' is missing from '{domain}'.",
                        source,
                    )
        for entity_id in sorted(actual - required):
            _add_issue(
                issues,
                "unexpected_entity",
                f"Entity ID '{entity_id}' is not listed for domain '{domain}'.",
                source,
                _record_identifier(entity_records[entity_id][1]),
            )

    resolvable_entity_ids = set(entity_records)
    for domain in DOMAINS:
        for record in records_by_domain[domain]:
            related_ids = record.get("related_ids")
            if not _is_string_list(related_ids):
                continue
            record_id = _record_identifier(record)
            for related_id in related_ids:
                if related_id not in resolvable_entity_ids:
                    _add_issue(
                        issues,
                        "unresolved_related_id",
                        f"Related entity ID '{related_id}' does not resolve.",
                        source,
                        record_id,
                    )

    tutorial_entity_ids = {
        entity_id
        for entity_id, (_domain, record) in entity_records.items()
        if record.get("entity_type") == "tutorial"
    }
    for entity_id in sorted(tutorial_entity_ids - set(tutorial_files)):
        _add_issue(
            issues,
            "missing_tutorial_mapping",
            f"Tutorial entity ID '{entity_id}' has no Markdown mapping.",
            source,
            _record_identifier(entity_records[entity_id][1]),
        )
    for entity_id in sorted(set(tutorial_files) - tutorial_entity_ids):
        if entity_id not in entity_records:
            code = "unknown_tutorial_entity"
            message = f"Tutorial mapping ID '{entity_id}' does not resolve."
            record_id = ""
        else:
            code = "invalid_tutorial_entity"
            message = f"Tutorial mapping ID '{entity_id}' is not a tutorial entity."
            record_id = _record_identifier(entity_records[entity_id][1])
        _add_issue(issues, code, message, source, record_id)

    return tuple(issues)
