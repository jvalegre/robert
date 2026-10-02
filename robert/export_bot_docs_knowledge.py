"""Export ROBERT and AQME docs into packaged JSON knowledge chunks for the bot."""

from __future__ import annotations

import json
import re
import ast
import hashlib
import math
import tempfile
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gui_easyrob.bot.knowledge_schema import (
    CONTEXT_SECTIONS,
    CONTEXT_VERSION as SCHEMA_CONTEXT_VERSION,
    DOMAINS,
    SCHEMA_VERSION,
    KnowledgeLoadIssue,
    KnowledgeValidationError,
    validate_context_set,
    validate_guidance_source_set,
)


ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT_DIR / "gui_easyrob" / "bot" / "knowledge"
ROBERT_DOCS_DIR = ROOT_DIR.parent / "docs"
AQME_DOCS_DIR = ROOT_DIR.parent.parent / "aqme" / "docs"
ROBERT_CODE_DIR = ROOT_DIR
EASYROB_CODE_DIR = ROOT_DIR / "gui_easyrob"
AQME_CODE_DIR = ROOT_DIR.parent.parent / "aqme" / "aqme"
HEADING_CHARS = {"=", "-", "~", "^", '"'}
GENERIC_SECTION_TITLES = {"parameters", "parameter", "examples", "notes", "options"}
SKIP_SECTION_TITLE_PARTS = {
    "running the tests",
    "tests",
    "developers",
    "development",
    "how to cite",
    "license",
    "special acknowledgments",
    "acknowledgments",
}
KNOWN_MODULES = {
    "AQME",
    "CURATE",
    "GENERATE",
    "PREDICT",
    "VERIFY",
    "REPORT",
    "QDESCP",
    "CSEARCH",
    "CMIN",
    "QCORR",
    "QPREP",
}
CODE_RELEVANCE_TERMS = {
    "aqme",
    "descriptor",
    "descriptors",
    "error",
    "extrapolation",
    "mae",
    "metric",
    "metrics",
    "model",
    "models",
    "parameter",
    "parameters",
    "pfi",
    "prediction",
    "predictions",
    "r2",
    "report",
    "rmse",
    "warning",
}
CONTEXT_VERSION = SCHEMA_CONTEXT_VERSION
UI_TABS = {"ROBERT", "Advanced Options", "Reports", "Predictions", "Images", "MolSSI Databases"}
FINAL_CONTEXT_FILENAMES = {
    "robert_context.json",
    "aqme_context.json",
    "ui_context.json",
}
LEGACY_KNOWLEDGE_FILENAMES = {
    "robert_docs.json",
    "aqme_docs.json",
    "robert_code_docstrings.json",
    "easyrob_code_docstrings.json",
    "aqme_code_docstrings.json",
    "curated_overrides.json",
    "workflow_requirements.json",
    "workflow_outputs.json",
    "troubleshooting_runtime.json",
    "troubleshooting_inputs.json",
    "robert_backend_logic.json",
    "molssi_integration.json",
    "ml_operational_concepts.json",
    "gui_overview.json",
    "aqme_integration.json",
}


@dataclass(frozen=True, slots=True)
class ExportSources:
    robert_docs: Path
    aqme_docs: Path
    robert_code: Path
    easyrob_code: Path
    aqme_code: Path
    tutorials: Path
    user_guidance: Path


DEFAULT_EXPORT_SOURCES = ExportSources(
    robert_docs=ROBERT_DOCS_DIR,
    aqme_docs=AQME_DOCS_DIR,
    robert_code=ROBERT_CODE_DIR,
    easyrob_code=EASYROB_CODE_DIR,
    aqme_code=AQME_CODE_DIR,
    tutorials=EASYROB_CODE_DIR / "tutorials",
    user_guidance=ROOT_DIR / "gui_easyrob" / "bot" / "knowledge_sources",
)
TRUSTED_COVERAGE_MANIFEST = (
    ROOT_DIR
    / "gui_easyrob"
    / "bot"
    / "knowledge_sources"
    / "coverage_manifest.json"
)


def clean_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r":[a-zA-Z0-9_-]+:`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`_?", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"\|([^|]+)\|", r"\1", cleaned)
    kept_lines: list[str] = []
    skip_indented_block = False
    block_indent = 0
    skip_developer_section = False
    developer_indent = 0
    skip_traceback = False
    traceback_indent = 0
    skip_exception_group = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_exception_group:
            if re.match(r"^\+-{5,}\+?$", stripped):
                skip_exception_group = False
            continue
        if skip_traceback:
            # A Python traceback ends with an exception summary dedented to the
            # traceback marker's level.  Using structure rather than class-name
            # suffixes supports arbitrary user-defined exception names.
            if stripped and indent <= traceback_indent:
                skip_traceback = False
                continue
            if not stripped:
                skip_traceback = False
                kept_lines.append("")
            continue
        if stripped in {
            "During handling of the above exception, another exception occurred:",
            "The above exception was the direct cause of the following exception:",
        }:
            continue
        if skip_developer_section:
            if not stripped or indent > developer_indent:
                continue
            skip_developer_section = False
        if re.match(
            r"^(?:args|arguments|attributes|examples|returns|raises|yields):\s*$",
            stripped,
            re.IGNORECASE,
        ):
            skip_developer_section = True
            developer_indent = indent
            continue
        if stripped.casefold().startswith("traceback (most recent call last):"):
            skip_traceback = True
            traceback_indent = indent
            continue
        group_marker = stripped.lstrip("+| ")
        if group_marker.casefold().startswith(
            "exception group traceback (most recent call last):"
        ):
            skip_exception_group = True
            continue
        if skip_indented_block:
            if not stripped or indent > block_indent:
                continue
            skip_indented_block = False
        if not stripped:
            kept_lines.append("")
            continue
        directive = re.match(r"^\.\.\s+(?:\S+\s+)?([a-zA-Z0-9_-]+)::", stripped)
        if directive:
            if directive.group(1).lower() in {"code", "code-block", "raw"}:
                skip_indented_block = True
                block_indent = indent
            continue
        if stripped.startswith(".. "):
            continue
        if re.match(r"^:[a-zA-Z0-9_-]+(?:\s+[^:]+)?:(?:\s|$)", stripped):
            continue
        lowered = stripped.lower()
        if lowered.startswith("<input"):
            continue
        if any(
            marker in lowered
            for marker in (
                "badge.svg",
                "shields.io/",
                "travis-ci.",
                "codecov.io/",
                "actions/workflows/",
                "build status",
            )
        ):
            continue
        if set(stripped) <= {"=", "-", "+", "~", "^"}:
            continue
        without_html = re.sub(
            r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^>]*)?/?>",
            "",
            line,
        )
        if without_html.strip():
            kept_lines.append(without_html)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\.{2,}\s+", ". ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "chunk"


def stable_chunk_id(source_name: str, source_file: str, identity: str) -> str:
    normalized_file = Path(source_file.replace("\\", "/")).as_posix().lower()
    raw_identity = f"{source_name}:{normalized_file}:{identity}"
    digest = hashlib.sha1(raw_identity.encode("utf-8")).hexdigest()[:8]
    path_label = slugify(str(Path(normalized_file).with_suffix("")))
    private_identity = ".".join(
        f"private-{part[1:]}" if part.startswith("_") else part
        for part in identity.split(".")
    )
    return "-".join(
        (
            slugify(source_name),
            path_label,
            slugify(private_identity),
            digest,
        )
    )


def iter_documented_nodes(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    nodes: list[tuple[str, ast.AST]] = [("module", tree)]

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        occurrences: dict[str, int] = {}

        def visit_node(node: ast.AST) -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                occurrences[node.name] = occurrences.get(node.name, 0) + 1
                occurrence = occurrences[node.name]
                local_identity = node.name if occurrence == 1 else f"{node.name}#{occurrence}"
                qualified = f"{prefix}.{local_identity}" if prefix else local_identity
                nodes.append((qualified, node))
                visit(getattr(node, "body", []), qualified)
                return
            for child in ast.iter_child_nodes(node):
                visit_node(child)

        for node in body:
            visit_node(node)

    visit(getattr(tree, "body", []))
    return nodes


def should_skip_section(title: str, body: str) -> bool:
    lowered_title = title.strip().lower()
    lowered_body = body.strip().lower()
    if any(part == lowered_title or part in lowered_title for part in SKIP_SECTION_TITLE_PARTS):
        return True
    if "pytest" in lowered_body or "tests-end" in lowered_body:
        return True
    if not lowered_body:
        return True
    return False


def split_rst_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_body: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        stripped = line.strip()
        next_stripped = next_line.strip()
        if stripped and next_stripped and set(next_stripped) <= HEADING_CHARS and len(next_stripped) >= len(stripped):
            if current_title or "".join(current_body).strip():
                sections.append((current_title or "Overview", clean_text("\n".join(current_body))))
            current_title = stripped
            current_body = []
            index += 2
            continue
        current_body.append(line)
        index += 1

    if current_title or "".join(current_body).strip():
        sections.append((current_title or "Overview", clean_text("\n".join(current_body))))
    return [(title, body) for title, body in sections if title or body]


def infer_module(source_file: str, title: str, program: str, fallback_module: str = "") -> str:
    candidates = [title.strip().upper(), fallback_module.strip().upper()]
    candidates.extend(part.upper() for part in Path(source_file).parts)
    for candidate in candidates:
        matches = [module for module in KNOWN_MODULES if module in candidate]
        if matches:
            return min(matches, key=lambda module: (candidate.index(module), module))
    return fallback_module or program.upper()


def infer_kind(source_file: str, title: str, body: str) -> str:
    lowered = f"{source_file} {title} {body}".lower()
    title_lowered = title.strip().lower()
    if title_lowered in {"parameters", "parameter"} or " parameter " in f" {lowered} ":
        return "parameter"
    if any(token in lowered for token in ("rmse", "mae", "r2", "score", "metric", "report", "reproducibility")):
        return "results"
    if any(token in lowered for token in ("file", "folder", "output", "pdf", "csv", "image", "plot")):
        return "output"
    if "workflow" in lowered or "step" in lowered or "before running" in lowered:
        return "workflow"
    if "error" in lowered or "warning" in lowered or "missing" in lowered or "required" in lowered:
        return "diagnostic"
    if any(part in lowered for part in ("tutorial", "quickstart", "example", "installation")):
        return "tutorial"
    return "concept"


def infer_tab(program: str, source_file: str, kind: str) -> str:
    lowered = source_file.lower()
    if "report" in lowered or kind == "results":
        return "Reports"
    if "predict" in lowered:
        return "Predictions"
    if program == "AQME":
        return "AQME"
    if kind == "parameter":
        return "Advanced Options"
    return "ROBERT"


def build_question_patterns(title: str, module: str, kind: str) -> list[str]:
    title_lower = title.strip().lower()
    patterns = [title_lower] if title_lower else []
    if module:
        module_lower = module.lower()
        if kind == "parameter":
            patterns.append(f"what does {title_lower} do")
            patterns.append(f"what is {title_lower}")
        elif kind == "results":
            patterns.append(f"what does {title_lower} mean")
            patterns.append(f"how should i interpret {title_lower}")
        else:
            patterns.append(f"what is {module_lower}")
            patterns.append(f"how does {module_lower} work")
    return [pattern for pattern in patterns if pattern]


def infer_priority(source_file: str, kind: str) -> float:
    lowered = source_file.lower()
    if kind in {"parameter", "results"}:
        return 0.95
    if "tutorial" in lowered or "quickstart" in lowered or "modules" in lowered:
        return 0.85
    if "example" in lowered:
        return 0.7
    if "api" in lowered:
        return 0.45
    return 0.6


def is_user_relevant_python_doc(path: str, name: str, docstring: str) -> bool:
    terms = set(re.findall(r"[a-z0-9_]+", f"{path} {name} {docstring}".lower()))
    if terms & CODE_RELEVANCE_TERMS:
        return True
    return any(term in f"{path} {name} {docstring}".lower() for term in ("r squared", "mean absolute error"))


def infer_kind_from_python_doc(path: str, name: str, docstring: str) -> str:
    lowered = f"{path} {name} {docstring}".lower()
    if any(term in lowered for term in ("rmse", "mae", "r2", "r squared", "metric", "prediction", "extrapolation")):
        return "results"
    if any(term in lowered for term in ("parameter", "option", "setting", "threshold", "pfi")):
        return "parameter"
    if any(term in lowered for term in ("error", "warning", "failed", "missing", "identical")):
        return "diagnostic"
    if any(term in lowered for term in ("report", "output", "pdf", "csv", "image")):
        return "output"
    return "concept"


def build_chunks_from_python_source(
    text: str,
    source_name: str,
    source_file: str,
    program: str,
) -> list[dict[str, object]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return []

    chunks: list[dict[str, object]] = []
    for identity, node in iter_documented_nodes(tree):
        docstring = ast.get_docstring(node, clean=True)
        name = identity.rsplit(".", 1)[-1]
        if not docstring or not is_user_relevant_python_doc(source_file, name, docstring):
            continue
        kind = infer_kind_from_python_doc(source_file, name, docstring)
        module = infer_module(source_file, name, program)
        doc_text = clean_text(docstring)
        keyword_terms = sorted(
            term for term in CODE_RELEVANCE_TERMS
            if term in f"{source_file} {name} {doc_text}".lower()
        )
        title = name.replace("_", " ").strip()
        chunk_id = stable_chunk_id(source_name, source_file, identity)
        chunks.append(
            {
                "id": chunk_id,
                "source": source_name,
                "source_file": source_file.replace("\\", "/"),
                "program": program,
                "module": module,
                "kind": kind,
                "topic": kind,
                "tab": infer_tab(program, source_file, kind),
                "title": title,
                "question_patterns": build_question_patterns(title, module, kind),
                "keywords": keyword_terms + [title, module.lower(), kind],
                "text": doc_text,
                "priority": 0.9 if kind in {"results", "parameter", "diagnostic"} else 0.7,
            }
        )
    return chunks


def build_chunks_from_rst_text(
    text: str,
    source_name: str,
    source_file: str,
    program: str,
) -> list[dict[str, object]]:
    sections = split_rst_sections(text)
    if not sections:
        return []

    fallback_module = infer_module(source_file, sections[0][0], program)
    chunks: list[dict[str, object]] = []
    title_occurrences: dict[str, int] = {}
    for title, body in sections:
        if should_skip_section(title, body):
            continue
        if not body:
            continue
        module = infer_module(source_file, title, program, fallback_module=fallback_module)
        kind = infer_kind(source_file, title, body)
        title_value = title if title.strip().lower() not in GENERIC_SECTION_TITLES else f"{module} {title}".strip()
        canonical_title = re.sub(r"\s+", " ", title.strip()).casefold()
        title_occurrences[canonical_title] = title_occurrences.get(canonical_title, 0) + 1
        identity = f"{canonical_title}#{title_occurrences[canonical_title]}"
        chunk_id = stable_chunk_id(source_name, source_file, identity)
        chunks.append(
            {
                "id": chunk_id,
                "source": source_name,
                "source_file": source_file.replace("\\", "/"),
                "program": program,
                "module": module,
                "kind": kind,
                "topic": kind,
                "tab": infer_tab(program, source_file, kind),
                "title": title_value,
                "question_patterns": build_question_patterns(title_value, module, kind),
                "keywords": [module.lower(), kind, title_value.lower()],
                "text": body,
                "priority": infer_priority(source_file, kind),
            }
        )
    return chunks


def iter_doc_files(base_dir: Path, include_prefixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".rst", ".md"}:
            continue
        relative = path.relative_to(base_dir).as_posix()
        if relative.startswith("API/") or relative.startswith("Misc/"):
            continue
        if include_prefixes and not any(relative.startswith(prefix) for prefix in include_prefixes):
            continue
        files.append(path)
    return sorted(files)


def export_tree(base_dir: Path, source_name: str, program: str, include_prefixes: tuple[str, ...]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for path in iter_doc_files(base_dir, include_prefixes):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(
            build_chunks_from_rst_text(
                text=text,
                source_name=source_name,
                source_file=path.relative_to(base_dir).as_posix(),
                program=program,
            )
        )
    return chunks


def iter_python_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    skipped_parts = {"__pycache__", ".git", ".pytest_cache", "tests", "docs", "knowledge"}
    files: list[Path] = []
    for path in base_dir.rglob("*.py"):
        relative_parts = set(path.relative_to(base_dir).parts)
        if relative_parts & skipped_parts:
            continue
        files.append(path)
    return sorted(files)


def export_python_tree(base_dir: Path, source_name: str, program: str) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for path in iter_python_files(base_dir):
        source_file = path.relative_to(base_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(
            build_chunks_from_python_source(
                text=text,
                source_name=source_name,
                source_file=source_file,
                program=program,
            )
        )
    return chunks


def build_curated_overrides() -> list[dict[str, object]]:
    return [
        {
            "id": "robert-overview-curated",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "ROBERT",
            "kind": "concept",
            "topic": "concept",
            "tab": "ROBERT",
            "title": "ROBERT overview",
            "question_patterns": [
                "what is robert",
                "what does robert do",
                "how does robert work",
            ],
            "keywords": ["robert", "machine learning", "workflow", "models", "curate", "generate", "predict"],
            "text": "ROBERT is the main machine-learning workflow in easyROB. It is used to curate datasets, train and compare models, validate them, run predictions, and assemble result reports once the input data are ready.",
            "priority": 1.0,
        },
        {
            "id": "easyrob-robert-vs-aqme",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "FULL_WORKFLOW",
            "kind": "concept",
            "topic": "concept",
            "tab": "ROBERT",
            "title": "ROBERT versus AQME in easyROB",
            "question_patterns": [
                "what is the difference between robert and aqme",
                "robert vs aqme",
                "is aqme the same as robert",
            ],
            "keywords": ["robert", "aqme", "difference", "machine learning", "descriptors", "smiles"],
            "text": "In easyROB, ROBERT is the main machine-learning workflow. It handles dataset curation, model generation, validation, prediction, and reporting. AQME is a separate chemistry workflow that is only needed when you want to generate descriptors from molecular structures such as SMILES before or during the ROBERT workflow.",
            "priority": 1.0,
        },
        {
            "id": "aqme-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "AQME",
            "module": "AQME",
            "kind": "concept",
            "topic": "concept",
            "tab": "AQME",
            "title": "AQME overview",
            "question_patterns": [
                "what is aqme",
                "what does aqme do",
                "how does aqme work",
            ],
            "keywords": ["aqme", "descriptors", "smiles", "conformers", "qdescp", "workflow"],
            "text": "AQME is the chemistry workflow used by easyROB to process molecular structures and generate descriptors when the dataset starts from structure information such as SMILES. In the easyROB GUI it is mainly used before ROBERT modelling so the machine-learning workflow has descriptor columns to train on.",
            "priority": 1.0,
        },
        {
            "id": "robert-curate-stage-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "CURATE",
            "kind": "concept",
            "topic": "concept",
            "tab": "ROBERT",
            "title": "CURATE stage",
            "question_patterns": ["what does curate do", "what is curate", "curate stage"],
            "keywords": ["curate", "dataset", "prepares", "clean", "filter", "preprocessing"],
            "text": "CURATE prepares the input dataset before model generation. It checks and cleans the data, handles columns that should not be model descriptors, and produces a curated dataset that later ROBERT stages can use more reliably.",
            "priority": 1.0,
        },
        {
            "id": "robert-generate-stage-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "GENERATE",
            "kind": "concept",
            "topic": "concept",
            "tab": "ROBERT",
            "title": "GENERATE stage",
            "question_patterns": ["what does generate do", "what is generate", "generate stage"],
            "keywords": ["generate", "trains", "models", "machine learning", "pfi", "descriptors"],
            "text": "GENERATE trains ROBERT machine-learning models from the curated dataset. This is where model candidates are built and compared, including workflows with descriptor reduction such as PFI.",
            "priority": 1.0,
        },
        {
            "id": "robert-predict-stage-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "PREDICT",
            "kind": "concept",
            "topic": "concept",
            "tab": "Predictions",
            "title": "PREDICT stage",
            "question_patterns": ["what does predict do", "what is predict", "predict stage"],
            "keywords": ["predict", "external test", "inference", "predictions", "trained model"],
            "text": "PREDICT applies a trained ROBERT model to a dataset for inference, typically an external test CSV. It uses the saved model outputs and generates predicted values, usually together with uncertainty information when available.",
            "priority": 1.0,
        },
        {
            "id": "robert-verify-stage-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "VERIFY",
            "kind": "concept",
            "topic": "concept",
            "tab": "ROBERT",
            "title": "VERIFY stage",
            "question_patterns": ["what does verify do", "what is verify", "verify stage"],
            "keywords": ["verify", "validates", "checks", "model", "robustness", "quality"],
            "text": "VERIFY validates and stress-checks the modelling results after model generation. It helps assess whether the selected models look reliable before the final report is interpreted.",
            "priority": 1.0,
        },
        {
            "id": "robert-report-stage-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "REPORT",
            "kind": "results",
            "topic": "results",
            "tab": "Reports",
            "title": "REPORT stage",
            "question_patterns": ["what does report do", "what is report", "report stage"],
            "keywords": ["report", "pdf", "ROBERT_report.pdf", "metrics", "summary", "results"],
            "text": "REPORT builds the final ROBERT PDF summary. The PDF brings together the workflow settings, selected models, metrics, validation information, warnings, plots, and result interpretation material generated by the run.",
            "priority": 1.0,
        },
        {
            "id": "robert-r2-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "REPORT",
            "kind": "results",
            "topic": "results",
            "tab": "Reports",
            "title": "R2 or R squared",
            "question_patterns": [
                "what is r2",
                "what is r squared",
                "what does r2 mean",
                "how should i interpret r squared",
            ],
            "keywords": ["r2", "r squared", "coefficient of determination", "variance", "metric", "report"],
            "text": "R2, also called R squared, estimates how much variance in the target values is explained by the model. Higher R2 is generally better for regression, but it must be interpreted together with RMSE, MAE, validation behavior, outliers, and external-test performance.",
            "priority": 1.0,
        },
        {
            "id": "robert-mae-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "REPORT",
            "kind": "results",
            "topic": "results",
            "tab": "Reports",
            "title": "MAE",
            "question_patterns": [
                "what is mae",
                "what does mae mean",
                "how should i interpret mae",
            ],
            "keywords": ["mae", "mean absolute error", "prediction error", "metric", "target", "report"],
            "text": "MAE means Mean Absolute Error. It is the average absolute difference between predicted and observed target values, expressed in the same units as the target. Lower MAE means smaller typical prediction errors.",
            "priority": 1.0,
        },
        {
            "id": "robert-pfi-vs-no-pfi",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "PREDICT",
            "kind": "results",
            "topic": "results",
            "tab": "Predictions",
            "title": "PFI versus No_PFI predictions",
            "question_patterns": [
                "what is pfi vs no pfi",
                "why are there pfi and no pfi results",
                "what is no pfi",
            ],
            "keywords": ["pfi", "no_pfi", "predictions", "descriptor reduction", "feature importance"],
            "text": "ROBERT can report two model families: PFI and No_PFI. PFI uses permutation feature importance to reduce descriptors to the most relevant variables, while No_PFI keeps the unreduced descriptor set. Comparing both helps detect whether descriptor reduction improves robustness or changes predictions strongly.",
            "priority": 1.0,
        },
        {
            "id": "robert-identical-predictions",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "PREDICT",
            "kind": "results",
            "topic": "results",
            "tab": "Predictions",
            "title": "Identical predictions",
            "question_patterns": [
                "why are all my predictions identical",
                "why predictions are identical",
                "all predictions identical",
            ],
            "keywords": ["identical predictions", "predictions", "model", "constant", "external test"],
            "text": "If all predictions are identical or nearly identical, the model may be over-regularized, using uninformative descriptors, receiving mismatched external-test descriptors, or applying a model outside its reliable domain. Check PFI and No_PFI agreement, descriptor columns, extrapolation warnings, and whether the external CSV matches the training descriptor format.",
            "priority": 1.0,
        },
        {
            "id": "easyrob-enable-aqme-workflow",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "FULL_WORKFLOW",
            "kind": "workflow",
            "topic": "workflow",
            "tab": "ROBERT",
            "title": "Enable AQME Workflow",
            "question_patterns": [
                "should i use the enable aqme workflow",
                "what does enable aqme workflow do",
                "when do i need aqme workflow",
            ],
            "keywords": ["enable aqme workflow", "smiles", "descriptors", "optional", "robert", "aqme"],
            "text": "Enable AQME Workflow only when your dataset starts from molecular structures such as SMILES and you need easyROB to generate descriptors automatically. If your CSV already contains descriptor columns, you usually do not need AQME and can run the ROBERT machine-learning workflow directly.",
            "priority": 1.0,
        },
        {
            "id": "easyrob-smiles-aqme-robert-path",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "FULL_WORKFLOW",
            "kind": "workflow",
            "topic": "workflow",
            "tab": "ROBERT",
            "title": "SMILES to descriptors to ROBERT workflow",
            "question_patterns": [
                "how do smiles work in easyrob",
                "how do i go from smiles to a model",
                "workflow from smiles to robert",
            ],
            "keywords": ["smiles", "aqme", "descriptors", "robert", "workflow", "target column"],
            "text": "When the dataset contains SMILES instead of ready-made descriptors, the usual path is: load the CSV, choose the target column and prediction type in ROBERT, enable AQME Workflow so descriptors can be generated from SMILES, optionally tune AQME settings, and then run ROBERT so the descriptor-generation step feeds the machine-learning workflow.",
            "priority": 1.0,
        },
    ]


def empty_context(domain: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "version": CONTEXT_VERSION,
        "overview": {},
        "workflows": [],
        "modules": [],
        "interface": [],
        "parameters": [],
        "concepts": [],
        "inputs_outputs": [],
        "tutorials": [],
        "troubleshooting": [],
    }


def add_entry(context: dict[str, object], section: str, entry: dict[str, object]) -> None:
    section_entries = context[section]
    assert isinstance(section_entries, list)
    section_entries.append(entry)


def content_fingerprint(text: object) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", str(text or "")).casefold().split()
    )
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def deduplicate_contexts(
    contexts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Keep one deterministic, user-preferred record for each rendered body."""
    tier_order = {"curated": 0, "tutorial": 1, "reference": 2, "generated": 3}
    candidates: dict[
        str,
        list[tuple[tuple[object, ...], tuple[str, str, int], dict[str, object]]],
    ] = {}
    copied: dict[str, dict[str, object]] = {}

    for domain in sorted(contexts):
        payload = contexts[domain]
        copied[domain] = dict(payload)
        for section in CONTEXT_SECTIONS:
            records = payload[section]
            assert isinstance(records, list)
            copied[domain][section] = []
            for index, record in enumerate(records):
                assert isinstance(record, dict)
                priority_value = record.get("priority")
                priority = (
                    float(priority_value)
                    if isinstance(priority_value, (int, float))
                    and not isinstance(priority_value, bool)
                    and math.isfinite(float(priority_value))
                    else float("-inf")
                )
                record_id = str(record.get("id", ""))
                rank = (
                    tier_order.get(str(record.get("source_tier", "")), len(tier_order)),
                    -priority,
                    record_id,
                    domain,
                    section,
                    index,
                )
                location = (domain, section, index)
                fingerprint = content_fingerprint(record.get("text"))
                candidates.setdefault(fingerprint, []).append(
                    (rank, location, record)
                )

    winners = {
        min(group, key=lambda candidate: candidate[0])[1]
        for group in candidates.values()
    }
    for domain in sorted(contexts):
        payload = contexts[domain]
        for section in CONTEXT_SECTIONS:
            records = payload[section]
            assert isinstance(records, list)
            copied[domain][section] = [
                record
                for index, record in enumerate(records)
                if (domain, section, index) in winners
            ]
    return copied


def overview_blocks() -> dict[str, dict[str, str]]:
    return {
        "robert": {
            "title": "ROBERT overview",
            "text": "ROBERT is the machine-learning workflow in easyROB for curation, model generation, validation, prediction, and reporting.",
        },
        "aqme": {
            "title": "AQME overview",
            "text": "AQME is the chemistry workflow used to turn molecular structure information such as SMILES into descriptors for downstream modeling.",
        },
        "ui": {
            "title": "easyROB UI overview",
            "text": "easyROB is the graphical interface that guides users through ROBERT, AQME, reporting, images, prediction, and tutorial-driven workflows.",
        },
    }


def load_tutorial_records(tutorials_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not tutorials_dir.exists():
        return records
    for path in sorted(tutorials_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = [block.strip() for block in raw_text.split("---") if block.strip()]
        for index, block in enumerate(blocks, start=1):
            block_text = re.sub(r"<[^>]+>", " ", block)
            text = clean_text(block_text)
            if not text:
                continue
            title = text.split(".")[0].splitlines()[0][:120].strip() or path.stem.replace("_", " ")
            records.append(
                {
                    "id": f"tutorial-{path.stem}-{index}",
                    "source": f"tutorial:{path.name}",
                    "kind": "tutorial",
                    "topic": "tutorial",
                    "title": title,
                    "text": text,
                    "keywords": [path.stem.replace("_", " "), "tutorial"],
                    "source_file": path.name,
                    "tab": "ROBERT",
                    "program": "ROBERT",
                    "module": "TUTORIALS",
                    "priority": 0.95,
                }
            )
    return records


def _validation_error(code: str, message: str, source: Path | str) -> KnowledgeValidationError:
    return KnowledgeValidationError(
        (KnowledgeLoadIssue(code=code, message=message, source=str(source)),)
    )


def load_curated_guidance(
    source_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    filenames = {
        "robert": "robert_user_guidance.json",
        "aqme": "aqme_user_guidance.json",
        "ui": "ui_user_guidance.json",
    }
    payloads: dict[str, object] = {}
    manifest_path = source_dir / "coverage_manifest.json"
    paths = [source_dir / filename for filename in filenames.values()] + [manifest_path]
    for path in paths:
        if not path.is_file():
            raise _validation_error(
                "missing_source",
                f"Required curated guidance source is missing: {path}",
                path,
            )

    try:
        for domain, filename in filenames.items():
            payloads[domain] = json.loads(
                (source_dir / filename).read_text(encoding="utf-8")
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _validation_error(
            "invalid_source",
            f"Could not load curated guidance: {exc}",
            source_dir,
        ) from exc

    issues = list(
        validate_guidance_source_set(
            payloads,
            manifest,
            source=str(source_dir),
        )
    )
    try:
        trusted_manifest = json.loads(
            TRUSTED_COVERAGE_MANIFEST.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _validation_error(
            "invalid_trusted_contract",
            f"Could not load the packaged curated coverage contract: {exc}",
            TRUSTED_COVERAGE_MANIFEST,
        ) from exc

    required_entities = (
        manifest.get("required_entities") if isinstance(manifest, Mapping) else None
    )
    trusted_required = trusted_manifest.get("required_entities")
    if not isinstance(required_entities, Mapping) or not isinstance(
        trusted_required, Mapping
    ):
        issues.append(
            KnowledgeLoadIssue(
                code="curated_contract_mismatch",
                message="Curated required-entity coverage does not match the packaged contract.",
                source=str(source_dir),
            )
        )
    else:
        for domain in DOMAINS:
            injected_ids = required_entities.get(domain)
            packaged_ids = trusted_required.get(domain)
            if not isinstance(injected_ids, list) or not isinstance(packaged_ids, list):
                continue
            if set(injected_ids) != set(packaged_ids):
                issues.append(
                    KnowledgeLoadIssue(
                        code="curated_contract_mismatch",
                        message=f"Curated entity coverage for '{domain}' does not match the packaged contract.",
                        source=str(source_dir),
                    )
                )
            payload = payloads.get(domain)
            records = payload.get("records") if isinstance(payload, Mapping) else None
            if isinstance(records, list):
                actual_ids = {
                    record.get("entity_id")
                    for record in records
                    if isinstance(record, Mapping)
                    and isinstance(record.get("entity_id"), str)
                }
                if actual_ids != set(packaged_ids):
                    issues.append(
                        KnowledgeLoadIssue(
                            code="curated_contract_mismatch",
                            message=f"Curated records for '{domain}' do not satisfy the packaged contract.",
                            source=str(source_dir),
                        )
                    )
    for domain, filename in filenames.items():
        payload = payloads.get(domain)
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if record.get("source_file") != filename:
                issues.append(
                    KnowledgeLoadIssue(
                        code="curated_domain_mismatch",
                        message=(
                            f"Curated record from '{domain}' must claim source_file "
                            f"'{filename}'."
                        ),
                        source=str(source_dir / filename),
                        record_id=str(record.get("id", "")),
                    )
                )
    if issues:
        raise KnowledgeValidationError(issues)

    records: list[dict[str, object]] = []
    for domain in DOMAINS:
        payload = payloads[domain]
        assert isinstance(payload, Mapping)
        domain_records = payload["records"]
        assert isinstance(domain_records, list)
        records.extend(dict(record) for record in domain_records)
    assert isinstance(manifest, dict)
    return records, manifest


def render_user_guidance(record: Mapping[str, object]) -> str:
    guidance = record.get("user_guidance")
    if not isinstance(guidance, Mapping):
        raise _validation_error(
            "invalid_user_guidance",
            "Curated record user_guidance must be an object.",
            str(record.get("source_file", "curated guidance")),
        )

    lines = [f"Purpose: {guidance['purpose']}"]
    list_fields = (
        ("when_to_use", "When to use", False),
        ("prerequisites", "Prerequisites", False),
        ("steps", "Steps", True),
    )
    for field, label, numbered in list_fields:
        values = guidance.get(field, [])
        if values:
            lines.append(f"{label}:")
            lines.extend(
                f"{index}. {value}" if numbered else f"- {value}"
                for index, value in enumerate(values, start=1)
            )
    lines.append(f"Expected result: {guidance['result']}")
    for field, label in (
        ("next_steps", "Next steps"),
        ("common_issues", "Common issues"),
    ):
        values = guidance.get(field, [])
        if values:
            lines.append(f"{label}:")
            lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def route_chunk(chunk: dict[str, object]) -> tuple[str, str]:
    program = str(chunk.get("program", "")).upper()
    source = str(chunk.get("source", ""))
    source_file = str(chunk.get("source_file", "")).replace("\\", "/")
    title = str(chunk.get("title", ""))
    text = str(chunk.get("text", ""))
    tab = str(chunk.get("tab", ""))
    kind = str(chunk.get("kind", "concept"))
    ui_signal = f"{title} {text}".lower()

    aqme_like_module = str(chunk.get("module", "")).upper() in {"AQME", "QDESCP", "CSEARCH", "CMIN", "QCORR", "QPREP"}

    if program == "AQME" or tab == "AQME":
        domain = "aqme"
    elif source == "robert_code_docstrings" and aqme_like_module:
        domain = "ui"
    elif (
        source == "easyrob_code_docstrings"
        or source_file.lower().startswith("tutorials/")
        or source_file.lower().endswith("readme.rst")
        or (tab in UI_TABS and any(term in ui_signal for term in (" tab", " button", "easyrob", "interface", "gui")))
    ):
        domain = "ui"
    else:
        domain = "robert"

    if kind == "workflow":
        return domain, "workflows"
    if kind == "parameter":
        return domain, "parameters"
    if kind == "diagnostic":
        return domain, "troubleshooting"
    if kind == "output":
        return domain, "inputs_outputs"
    if kind == "tutorial":
        return domain, "tutorials"
    if kind == "results":
        return domain, "concepts"
    return domain, "modules"


def chunk_to_context_entry(chunk: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(chunk.get("id", "")),
        "source": str(chunk.get("source", "")),
        "kind": str(chunk.get("kind", "")),
        "topic": str(chunk.get("topic", "")),
        "title": str(chunk.get("title", "")),
        "text": str(chunk.get("text", "")),
        "keywords": list(chunk.get("keywords", [])),
        "question_patterns": list(chunk.get("question_patterns", [])),
        "source_file": str(chunk.get("source_file", "")),
        "tab": str(chunk.get("tab", "")),
        "program": str(chunk.get("program", "")),
        "module": str(chunk.get("module", "")),
        "priority": float(chunk.get("priority", 0.7)),
    }


def write_context_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def remove_legacy_knowledge_files(output_dir: Path) -> None:
    for filename in LEGACY_KNOWLEDGE_FILENAMES:
        legacy_path = output_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()


def _legacy_export_docs_knowledge(output_dir: Path = KNOWLEDGE_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    robert_chunks = export_tree(
        ROBERT_DOCS_DIR,
        source_name="robert_docs",
        program="ROBERT",
        include_prefixes=("Modules/", "Report/", "Tutorials/", "Examples/", "Install/", "README.rst"),
    )
    aqme_chunks = export_tree(
        AQME_DOCS_DIR,
        source_name="aqme_docs",
        program="AQME",
        include_prefixes=("Quickstart/", "Examples/", "README.rst", "video_tutorials.rst"),
    )
    robert_code_chunks = export_python_tree(
        ROBERT_CODE_DIR,
        source_name="robert_code_docstrings",
        program="ROBERT",
    )
    easyrob_code_chunks = export_python_tree(
        EASYROB_CODE_DIR,
        source_name="easyrob_code_docstrings",
        program="ROBERT",
    )
    aqme_code_chunks = export_python_tree(
        AQME_CODE_DIR,
        source_name="aqme_code_docstrings",
        program="AQME",
    )

    curated_overrides = build_curated_overrides() + [
        {
            "id": "easyrob-robert-run-prerequisites",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "FULL_WORKFLOW",
            "kind": "workflow",
            "topic": "workflow",
            "tab": "ROBERT",
            "title": "Before running ROBERT",
            "question_patterns": [
                "what do i need before running robert",
                "what do i need before starting robert",
                "how can i start a machine learning workflow",
                "what should i check before running robert"
            ],
            "keywords": ["robert", "run", "before", "main csv", "target", "prediction type", "workflow"],
            "text": "Before running ROBERT in easyROB, load the main CSV for the training dataset, choose the target column, choose the prediction type, and confirm the workflow you want to run. A separate test CSV is optional unless you are running PREDICT on an external dataset. Enable the AQME workflow only when you need descriptors generated from molecular structures such as SMILES.",
            "priority": 1.0,
        },
        {
            "id": "aqme-qdescp-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "AQME",
            "module": "QDESCP",
            "kind": "concept",
            "topic": "concept",
            "tab": "AQME",
            "title": "What QDESCP does",
            "question_patterns": ["what is qdescp", "how does qdescp work"],
            "keywords": ["qdescp", "aqme", "descriptors"],
            "text": "QDESCP generates quantum mechanical descriptors in AQME so they can later be used in downstream workflows such as descriptor-based modeling.",
            "priority": 0.98,
        },
        {
            "id": "easyrob-advanced-options-tab",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "ADVANCED_OPTIONS",
            "kind": "concept",
            "topic": "concept",
            "tab": "Advanced Options",
            "title": "Advanced Options tab",
            "question_patterns": [
                "what is advanced options tab",
                "what is the advanced options tab",
                "advanced options tab",
                "which parameters can i use"
            ],
            "keywords": ["advanced options", "parameters", "general", "curate", "generate", "predict"],
            "text": "The Advanced Options tab contains detailed ROBERT workflow parameters grouped by GENERAL, CURATE, GENERATE, and PREDICT. Most users can keep the defaults, but this tab is where advanced users tune dataset curation, model generation, prediction settings, and general workflow behavior.",
            "priority": 1.0,
        },
        {
            "id": "easyrob-run-robert-button",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "FULL_WORKFLOW",
            "kind": "workflow",
            "topic": "workflow",
            "tab": "ROBERT",
            "title": "Run ROBERT button",
            "question_patterns": [
                "what is run robert button",
                "what does run robert do",
                "how do i run robert",
                "run robert button"
            ],
            "keywords": ["run robert", "button", "launch", "workflow", "configured"],
            "text": "The Run ROBERT button launches the selected ROBERT workflow using the inputs and options currently configured in the GUI. In Full Workflow it runs the automated pipeline; in individual modes it runs the selected ROBERT stage.",
            "priority": 1.0,
        },
        {
            "id": "robert-pfi-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "GENERATE",
            "kind": "parameter",
            "topic": "parameter",
            "tab": "Advanced Options",
            "title": "PFI",
            "question_patterns": [
                "what is pfi",
                "what does pfi mean",
                "what is pfi in robert",
                "what is pfi filter"
            ],
            "keywords": ["pfi", "permutation feature importance", "descriptor", "feature selection", "generate"],
            "text": "PFI means Permutation Feature Importance. In ROBERT it estimates which descriptors contribute most to model predictions, and it can be used to build reduced models with only the most relevant descriptors.",
            "priority": 1.0,
        },
        {
            "id": "robert-rmse-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "REPORT",
            "kind": "results",
            "topic": "results",
            "tab": "Reports",
            "title": "RMSE",
            "question_patterns": [
                "what is rmse",
                "what does rmse mean",
                "what is rmse in robert",
                "how should i interpret rmse"
            ],
            "keywords": ["rmse", "root mean squared error", "prediction error", "metric", "target"],
            "text": "RMSE means Root Mean Squared Error. It measures the typical prediction error magnitude in the same units as the target value; lower RMSE generally indicates more accurate predictions.",
            "priority": 1.0,
        },
        {
            "id": "robert-report-metrics-overview",
            "source": "curated_overrides",
            "source_file": "manual",
            "program": "ROBERT",
            "module": "REPORT",
            "kind": "results",
            "topic": "results",
            "tab": "Reports",
            "title": "How to interpret ROBERT metrics",
            "question_patterns": ["what does rmse mean", "how should i interpret r2"],
            "keywords": ["rmse", "r2", "report", "metrics"],
            "text": "ROBERT report metrics summarize model quality. RMSE reflects prediction error magnitude in the target units, while R2 reflects how much target variance the model explains.",
            "priority": 0.99,
        },
    ]

    contexts = {
        "robert": empty_context("robert"),
        "aqme": empty_context("aqme"),
        "ui": empty_context("ui"),
    }
    for domain, overview in overview_blocks().items():
        contexts[domain]["overview"] = overview

    all_chunks = (
        robert_chunks
        + aqme_chunks
        + robert_code_chunks
        + easyrob_code_chunks
        + aqme_code_chunks
        + curated_overrides
    )
    for chunk in all_chunks:
        domain, section = route_chunk(chunk)
        add_entry(contexts[domain], section, chunk_to_context_entry(chunk))

    tutorials_dir = EASYROB_CODE_DIR / "tutorials"
    for tutorial_record in load_tutorial_records(tutorials_dir):
        add_entry(contexts["ui"], "tutorials", tutorial_record)

    remove_legacy_knowledge_files(output_dir)
    write_context_json(output_dir / "robert_context.json", contexts["robert"])
    write_context_json(output_dir / "aqme_context.json", contexts["aqme"])
    write_context_json(output_dir / "ui_context.json", contexts["ui"])


def _section_for_kind(kind: object) -> str:
    return {
        "workflow": "workflows",
        "module": "modules",
        "interface": "interface",
        "parameter": "parameters",
        "output": "inputs_outputs",
        "tutorial": "tutorials",
        "diagnostic": "troubleshooting",
        "results": "concepts",
        "concept": "concepts",
    }.get(str(kind), "concepts")


def _generated_entry(
    chunk: Mapping[str, object],
    *,
    source_tier: str,
    audience: str,
) -> dict[str, object]:
    entry = chunk_to_context_entry(dict(chunk))
    entry["source_tier"] = source_tier
    entry["audience"] = audience
    return entry


def build_contexts(sources: ExportSources) -> dict[str, dict[str, object]]:
    roots = {
        "robert_docs": sources.robert_docs,
        "aqme_docs": sources.aqme_docs,
        "robert_code": sources.robert_code,
        "easyrob_code": sources.easyrob_code,
        "aqme_code": sources.aqme_code,
        "tutorials": sources.tutorials,
        "user_guidance": sources.user_guidance,
    }
    root_issues = [
        KnowledgeLoadIssue(
            code="missing_source",
            message=f"Required export source root '{name}' is missing or not a directory.",
            source=str(path),
        )
        for name, path in roots.items()
        if not path.is_dir()
    ]
    if root_issues:
        raise KnowledgeValidationError(root_issues)

    curated_records, manifest = load_curated_guidance(sources.user_guidance)
    tutorial_files = manifest.get("tutorial_files")
    assert isinstance(tutorial_files, Mapping)
    tutorial_root = sources.tutorials.resolve()
    tutorial_issues: list[KnowledgeLoadIssue] = []
    for entity_id, filename in tutorial_files.items():
        relative_path = Path(str(filename))
        candidate = (tutorial_root / relative_path).resolve()
        if relative_path.is_absolute() or not candidate.is_relative_to(tutorial_root):
            tutorial_issues.append(
                KnowledgeLoadIssue(
                    code="unsafe_tutorial_path",
                    message=f"Tutorial source '{filename}' escapes the tutorial root.",
                    source=str(filename),
                    record_id=str(entity_id),
                )
            )
        elif not candidate.is_file():
            tutorial_issues.append(
                KnowledgeLoadIssue(
                    code="missing_tutorial_source",
                    message=f"Required tutorial source '{filename}' is missing.",
                    source=str(candidate),
                    record_id=str(entity_id),
                )
            )
    if tutorial_issues:
        raise KnowledgeValidationError(tutorial_issues)

    extracted = {
        "robert_docs": export_tree(
            sources.robert_docs,
            "robert_docs",
            "ROBERT",
            ("Modules/", "Report/", "Tutorials/", "Examples/", "Install/", "README.rst"),
        ),
        "aqme_docs": export_tree(
            sources.aqme_docs,
            "aqme_docs",
            "AQME",
            ("Quickstart/", "Examples/", "README.rst", "video_tutorials.rst"),
        ),
        "robert_code_docstrings": export_python_tree(
            sources.robert_code, "robert_code_docstrings", "ROBERT"
        ),
        "easyrob_code_docstrings": export_python_tree(
            sources.easyrob_code, "easyrob_code_docstrings", "ROBERT"
        ),
        "aqme_code_docstrings": export_python_tree(
            sources.aqme_code, "aqme_code_docstrings", "AQME"
        ),
        "tutorials": load_tutorial_records(sources.tutorials),
    }
    empty_categories = [
        KnowledgeLoadIssue(
            code="empty_source_category",
            message=f"Required export category '{name}' produced no records.",
            source=str(roots.get(name, sources.tutorials)),
        )
        for name, records in extracted.items()
        if not records
    ]
    if empty_categories:
        raise KnowledgeValidationError(empty_categories)

    contexts = {domain: empty_context(domain) for domain in DOMAINS}
    for domain, overview in overview_blocks().items():
        contexts[domain]["overview"] = overview
        contexts[domain]["version"] = manifest["version"]

    generated_metadata = {
        "robert_docs": ("reference", "advanced_user"),
        "aqme_docs": ("reference", "advanced_user"),
        "robert_code_docstrings": ("generated", "developer_reference"),
        "easyrob_code_docstrings": ("generated", "developer_reference"),
        "aqme_code_docstrings": ("generated", "developer_reference"),
        "tutorials": ("tutorial", "end_user"),
    }
    for category, chunks in extracted.items():
        tier, audience = generated_metadata[category]
        for chunk in chunks:
            domain, section = route_chunk(dict(chunk))
            if category == "tutorials":
                domain, section = "ui", "tutorials"
            add_entry(
                contexts[domain],
                section,
                _generated_entry(chunk, source_tier=tier, audience=audience),
            )

    curated_domains = {
        "robert_user_guidance.json": "robert",
        "aqme_user_guidance.json": "aqme",
        "ui_user_guidance.json": "ui",
    }
    for record in curated_records:
        entry = dict(record)
        entry["text"] = render_user_guidance(entry)
        source_file = str(entry.get("source_file", ""))
        domain = curated_domains[source_file]
        add_entry(contexts[domain], _section_for_kind(entry.get("kind")), entry)

    contexts = deduplicate_contexts(contexts)

    for payload in contexts.values():
        for section in CONTEXT_SECTIONS:
            records = payload[section]
            assert isinstance(records, list)
            records.sort(key=lambda record: str(record["id"]))

    validation_issues = validate_context_set(contexts)
    if validation_issues:
        raise KnowledgeValidationError(validation_issues)
    return contexts


def export_docs_knowledge(
    output_dir: Path = KNOWLEDGE_DIR,
    sources: ExportSources = DEFAULT_EXPORT_SOURCES,
) -> tuple[Path, Path, Path]:
    contexts = build_contexts(sources)
    ordered_outputs = (
        ("robert", "robert_context.json"),
        ("aqme", "aqme_context.json"),
        ("ui", "ui_context.json"),
    )
    serialized = {
        filename: json.dumps(
            contexts[domain],
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
        for domain, filename in ordered_outputs
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    original_finals: dict[str, bytes | None] = {}
    for _domain, filename in ordered_outputs:
        final_path = output_dir / filename
        original_finals[filename] = (
            final_path.read_bytes() if final_path.exists() else None
        )

    temporary_paths: dict[str, Path] = {}
    try:
        for _domain, filename in ordered_outputs:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_dir,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_paths[filename] = Path(handle.name)
                handle.write(serialized[filename])
        try:
            for _domain, filename in ordered_outputs:
                temporary_paths[filename].replace(output_dir / filename)
        except BaseException:
            for _domain, filename in ordered_outputs:
                final_path = output_dir / filename
                original = original_finals[filename]
                if original is None:
                    if final_path.exists():
                        final_path.unlink()
                else:
                    final_path.write_bytes(original)
            raise
    finally:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()

    remove_legacy_knowledge_files(output_dir)
    return tuple(output_dir / filename for _domain, filename in ordered_outputs)


if __name__ == "__main__":
    export_docs_knowledge()
