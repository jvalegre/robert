"""Rule-based diagnosis and answer formatting for the EasyROB bot."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import unicodedata
from typing import Sequence

from .bot_context import GuiSnapshot
from .error_parser import infer_target_column_failure_hint

__all__ = ["Diagnosis", "diagnose_snapshot", "format_heuristic_answer"]


FAILURE_MARKERS = {
    "error",
    "errors",
    "failed",
    "failure",
    "warning",
    "warnings",
    "exception",
    "traceback",
    "cannot",
    "blocked",
}

_WORKFLOWS_REQUIRING_MAIN_CSV = {"FULL WORKFLOW"}
_WORKFLOWS_REQUIRING_TEST_CSV = {"PREDICT"}
_WORKFLOWS_REQUIRING_CSV_CONTEXT = {"REPORT"}
_WORKFLOWS_NOT_REQUIRING_TARGET = {"PREDICT", "REPORT"}
_TAB_ALIASES = {
    "interactive plots": "Interactive plots",
    "interactive plot": "Interactive plots",
    "aqme": "AQME",
    "report": "Reports",
    "reports": "Reports",
    "result": "Results",
    "results": "Results",
    "images": "Images",
    "image": "Images",
    "prediction": "Predictions",
    "predictions": "Predictions",
    "molssi": "MolSSI Databases",
    "robert": "ROBERT",
}
_POPUP_QUESTION_RE = re.compile(
    r"\b(pop[- ]?up|dialog|warning|message box|message|alert|appeared|appears|window)\b",
    re.IGNORECASE,
)
_CURRENT_CONTEXT_REFERENCE_RE = re.compile(
    r"\b(this|that|it|current|currently|just|now|happened|failed|failure|error|warning)\b",
    re.IGNORECASE,
)
_AQME_WORKFLOW_QUESTION_RE = re.compile(
    r"\b(enable\s+aqme\s+workflow|aqme\s+workflow|aqme\s+checkbox)\b",
    re.IGNORECASE,
)
_DESCRIPTOR_GENERATION_QUESTION_RE = re.compile(
    r"(?:\b(?:need|generate|create|make|use|require)\b[^?.!]{0,80}\bdescriptors?\b|"
    r"\bdescriptors?\b[^?.!]{0,80}\b(?:need|generate|create|make|use|require)\b)",
    re.IGNORECASE,
)
_AQME_TAB_QUESTION_RE = re.compile(
    r"\baqme\s+tab\b",
    re.IGNORECASE,
)
_AQME_LIVE_CONTEXT_RE = re.compile(
    r"\b(common substructure|fmcs|mcs|max(?:imum)? common substructure|smarts|selected atoms?|atomic descriptors?|aqme warning|aqme message|this aqme)\b",
    re.IGNORECASE,
)
_DOCS_LABEL = "Documentation"
_AQME_ADVANCED_HINT_RE = re.compile(
    r"\b(tab|fmcs|smarts|atomic|atom|descriptor level|denovo|interpret|full|chemdraw|parameter|setting|configure)\b",
    re.IGNORECASE,
)
_TRACEBACK_LINE_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n\r]+)', re.IGNORECASE)
_EXCEPTION_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(?P<message>.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    summary: str
    reason: str
    next_step: str


def _has_failure_markers(snapshot: GuiSnapshot) -> bool:
    if any(term in FAILURE_MARKERS for term in snapshot.console_terms):
        return True
    console = snapshot.recent_console.lower()
    if "not found" in console:
        return True
    return any(marker in console for marker in FAILURE_MARKERS)


def _console_specific_diagnosis(snapshot: GuiSnapshot) -> Diagnosis | None:
    console = snapshot.recent_console.strip()
    lowered = console.lower()
    if not lowered:
        return None

    report_match = re.search(r"\bROBERT_report(?:_[A-Za-z0-9_]+)?\.pdf\b", console, re.IGNORECASE)
    if report_match and ("not found" in lowered or "filenotfounderror" in lowered):
        report_name = report_match.group(0)
        return Diagnosis(
            summary=f"{report_name} was not found.",
            reason="The report PDF is missing from the run directory, so the GUI cannot open or display the final ROBERT report.",
            next_step=f"Run the REPORT step or Full Workflow again, then check that `{report_name}` was created in the same folder as the selected CSV.",
        )

    column_match = re.search(r"column missing:\s*([^\n\r.]+)", console, flags=re.IGNORECASE)
    if column_match:
        column_name = column_match.group(1).strip().strip("'\"")
        return Diagnosis(
            summary=f"The required `{column_name}` column is missing.",
            reason="The current workflow expected that column in the loaded CSV or generated AQME table, but it was not present.",
            next_step=f"Check the input table and add or map the `{column_name}` column before saving or running the workflow again.",
        )

    if "target column must be numeric" in lowered:
        return Diagnosis(
            summary="The target column must be numeric.",
            reason="Regression workflows need numerical target values so ROBERT can train and evaluate the model.",
            next_step="Edit the target column to contain only numeric values, then reload or save the CSV again.",
        )

    if "target column is empty" in lowered:
        return Diagnosis(
            summary="The target column is empty.",
            reason="ROBERT needs target values for the rows used to train or validate the model.",
            next_step="Fill the target column with the measured response values before saving or running the workflow.",
        )

    if "no valid molecules" in lowered:
        return Diagnosis(
            summary="No valid molecules were found in the input file.",
            reason="AQME/easyROB could not read usable molecular structures from the selected file.",
            next_step="Check that the structures are chemically valid, the file format is correct, and SMILES or ChemDraw entries are not empty.",
        )

    if "please select a csv file" in lowered:
        return Diagnosis(
            summary="No CSV file was selected.",
            reason="The workflow needs a CSV path before it can load data or determine the run directory.",
            next_step="Select the main CSV or the required test CSV, depending on the selected workflow.",
        )

    if "aqme" in lowered and "expected output csv" in lowered and ("not found" in lowered or "missing" in lowered):
        return Diagnosis(
            summary="The expected AQME output CSV was not found.",
            reason="AQME appears to have finished, but easyROB could not locate the descriptor CSV needed for the ROBERT workflow.",
            next_step="Check the AQME output folder and file names, then rerun AQME if the descriptor CSV was not created.",
        )

    if "could not delete folder" in lowered or ("permissionerror" in lowered and "folder" in lowered):
        return Diagnosis(
            summary="easyROB could not delete a previous output folder.",
            reason="A previous run folder may be open in another program, locked by the operating system, or missing write permissions.",
            next_step="Close files or folders from the previous run, then retry. If needed, remove the old output folder manually after checking you do not need its contents.",
        )

    return None


def _normalize_workflow(workflow: str) -> str:
    return workflow.strip().upper()


def _has_csv_context(snapshot: GuiSnapshot) -> bool:
    return bool(snapshot.main_csv_path.strip() or snapshot.test_csv_path.strip())


def _requires_target_column(workflow: str) -> bool:
    normalized = _normalize_workflow(workflow)
    return bool(normalized) and normalized not in _WORKFLOWS_NOT_REQUIRING_TARGET


def _mentioned_tab(question: str) -> str | None:
    lowered = question.lower()
    for token, tab_name in _TAB_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return tab_name
    return None


def _is_explicit_tab_question(question: str) -> bool:
    lowered = question.lower()
    return bool(re.search(r"\b(tab|view|enabled|disabled|blocked|locked|available|open|unlock|where)\b", lowered))


def _tab_gating_diagnosis(snapshot: GuiSnapshot, mentioned_tab: str) -> Diagnosis | None:
    result_view = "Report" if mentioned_tab == "Reports" else mentioned_tab
    if result_view in snapshot.result_disabled_views:
        return Diagnosis(
            summary=f"The {result_view} view in Results is unavailable for the selected model.",
            reason=f"Results enables {result_view} only when its files exist for the selected model and variant.",
            next_step="Check the selected model in Results and the files generated for this run, then refresh the result views.",
        )
    if result_view in snapshot.result_enabled_views:
        selected_model = snapshot.result_model or "the current selection"
        return Diagnosis(
            summary=f"The {result_view} view in Results is available for {selected_model}.",
            reason=f"The GUI detected matching {result_view} files for this selection.",
            next_step=f"Open Results and select {result_view} to inspect the available outputs.",
        )
    if mentioned_tab not in snapshot.disabled_tabs:
        return None

    if mentioned_tab == "AQME":
        if not snapshot.aqme_workflow_enabled:
            return Diagnosis(
                summary="The AQME tab is disabled because the AQME workflow path is not enabled.",
                reason="AQME is optional in the GUI, so the tab stays gated until `Enable AQME Workflow` is checked.",
                next_step="Check `Enable AQME Workflow`. If you then want to run AQME, load the main CSV and confirm the dataset includes structure information such as a SMILES column.",
            )
        if not snapshot.main_csv_path.strip():
            return Diagnosis(
                summary="The AQME workflow path is enabled, but AQME still cannot run without a main CSV.",
                reason="AQME descriptor generation needs a dataset to process after the tab is unlocked.",
                next_step="Load the main CSV first, then return to AQME and verify the structure columns needed for descriptor generation.",
            )
        return Diagnosis(
            summary="The AQME tab is still unavailable even though the AQME workflow path is enabled.",
            reason="That usually means the AQME branch is enabled but some required AQME-specific context is still missing.",
            next_step="Check the loaded dataset, especially whether it includes the structure information AQME expects, such as a SMILES column.",
        )

    if mentioned_tab == "Results":
        return Diagnosis(
            summary="The Results tab is disabled because no report, prediction, or image output has been detected yet.",
            reason="Results opens when at least one of its Report, Predictions, or Images views has output for this run.",
            next_step="Run a workflow that creates a report, prediction CSV, or figures, then refresh the run context.",
        )

    if mentioned_tab == "Reports":
        return Diagnosis(
            summary="The Reports tab is disabled because no report output has been detected yet.",
            reason="The Reports tab is unlocked by the generated ROBERT PDF rather than by selecting the workflow alone.",
            next_step="Run the workflow until the reporting stage completes and confirm that a `ROBERT_report_*.pdf` file is created successfully.",
        )

    if mentioned_tab == "Images":
        return Diagnosis(
            summary="The Images tab is disabled because no image outputs have been detected yet.",
            reason="The GUI enables Images only after workflow figure folders are generated and discovered.",
            next_step="Complete a workflow step that writes figures, then refresh or reopen that run context.",
        )

    if mentioned_tab == "Predictions":
        return Diagnosis(
            summary="The Predictions tab is disabled because no prediction outputs are available yet.",
            reason="That tab becomes relevant only after a run produces external-test prediction artifacts.",
            next_step="Provide a test dataset if needed, run prediction, and then check whether the prediction outputs were generated.",
        )

    return None


def diagnose_snapshot(snapshot: GuiSnapshot, question: str) -> Diagnosis:
    workflow = _normalize_workflow(snapshot.workflow)
    mentioned_tab = _mentioned_tab(question)
    if (mentioned_tab == "Results" and "Results" not in snapshot.enabled_tabs
            and "Results" not in snapshot.disabled_tabs
            and "Reports" in snapshot.disabled_tabs):
        # Older snapshots expose the three result pages as separate top-level tabs.
        mentioned_tab = "Reports"

    if mentioned_tab and _is_explicit_tab_question(question):
        tab_diagnosis = _tab_gating_diagnosis(snapshot, mentioned_tab)
        if tab_diagnosis is not None:
            return tab_diagnosis

    if workflow in _WORKFLOWS_REQUIRING_MAIN_CSV and not snapshot.main_csv_path.strip():
        return Diagnosis(
            summary="The main CSV input is not set.",
            reason="The workflow cannot start without the primary dataset path.",
            next_step="Select the main CSV before running the workflow.",
        )

    if workflow in _WORKFLOWS_REQUIRING_TEST_CSV and not snapshot.test_csv_path.strip():
        return Diagnosis(
            summary="The test CSV input is not set.",
            reason="The PREDICT workflow needs an external dataset to score.",
            next_step="Select the test CSV before running prediction.",
        )

    if workflow in _WORKFLOWS_REQUIRING_CSV_CONTEXT and not _has_csv_context(snapshot):
        return Diagnosis(
            summary="No CSV context is available for the report workflow.",
            reason="REPORT needs at least one loaded CSV path so the GUI can resolve the run directory.",
            next_step="Load the training CSV or test CSV that belongs to the run you want to report.",
        )

    if _requires_target_column(workflow) and not snapshot.target_column.strip():
        return Diagnosis(
            summary="The target column is not selected.",
            reason="This workflow needs an explicit target column before ROBERT can build the command.",
            next_step="Choose the target column in the Y dropdown.",
        )

    if snapshot.process_running:
        return Diagnosis(
            summary="A process is already running.",
            reason="EasyROB is already running a workflow and cannot start a second run yet.",
            next_step="Wait for the current run to finish before starting another one.",
        )

    console_diagnosis = _console_specific_diagnosis(snapshot)
    if console_diagnosis is not None:
        return console_diagnosis

    if _has_failure_markers(snapshot):
        return Diagnosis(
            summary="The console shows a failure marker.",
            reason="The console text includes an error-like marker such as `failed`, `warning`, or `not found`.",
            next_step="Inspect the console output for the first error or warning and fix that issue first.",
        )

    return Diagnosis(
        summary="No specific block was detected from the GUI snapshot.",
        reason="The current snapshot does not point to a single obvious missing input or active failure.",
        next_step="Check the selected workflow, inputs, and console output, then retry.",
    )


def _format_candidates(candidates: Sequence[object]) -> str:
    if not candidates:
        return ""

    labels: list[str] = []
    for candidate in candidates:
        candidate_obj = getattr(candidate, "chunk", candidate)
        for attr_name in ("id", "topic", "tab", "source"):
            value = getattr(candidate_obj, attr_name, None)
            if value:
                label = str(value).strip()
                if label and label not in labels:
                    labels.append(label)
                break
        if len(labels) == 1:
            break

    if labels:
        return f"\nRelevant retrieval hints: {', '.join(labels)}."
    return ""


def _best_chunk(candidates: Sequence[object]) -> object | None:
    if not candidates:
        return None
    return getattr(candidates[0], "chunk", candidates[0])


def _clean_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned.strip()


def _short_answer_text(text: str, limit: int = 420) -> str:
    cleaned = _clean_sentence(text)
    cleaned = re.sub(r"\.\.\s+[\w-]+(?:\s+\w+)?::\s*\S*", "", cleaned)
    cleaned = re.sub(r"\.\.\s+[\w-]+(?:-start|-end)?", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = _clean_sentence(cleaned)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if sentences:
        candidate = " ".join(sentences[:2]).strip()
    else:
        candidate = cleaned
    if len(candidate) <= limit:
        return candidate
    return candidate[: limit - 3].rstrip() + "..."


def _title_prefixed_text(title: str, text: str, fallback: str) -> str:
    clean_title = _clean_sentence(title)
    clean_text = _clean_sentence(text)
    if not clean_text:
        return clean_title or fallback
    if clean_title and clean_text.lower().startswith(clean_title.lower()):
        return clean_text
    if clean_title:
        return f"{clean_title}: {clean_text}"
    return clean_text or fallback


def _format_guidance_answer(
    question: str,
    snapshot: GuiSnapshot,
    candidates: Sequence[object],
    intent: str,
) -> str:
    best = _best_chunk(candidates)
    if best is None:
        fallback = "I cannot confirm a specific answer from the packaged EasyROB guidance."
        if snapshot.active_tab:
            fallback += (
                f" The current tab is {snapshot.active_tab}, but I need the exact visible label, "
                "error, or popup text to guide you."
            )
        else:
            fallback += " Please share the exact visible label, error, or popup text."
        return fallback

    full_text = str(getattr(best, "text", "") or "")
    text = _short_answer_text(full_text)
    tab = str(getattr(best, "tab", "") or "").strip()
    source = str(getattr(best, "source", "") or "").strip()
    title = str(getattr(best, "title", "") or "").strip()
    kind = str(getattr(best, "kind", "") or "").strip()

    if intent == "parameter" or kind == "parameter":
        parts = [_title_prefixed_text(title, text, "Parameter")]
        if tab:
            parts.append(f"You would normally adjust this in the {tab} tab.")
        return " ".join(part for part in parts if part)

    if intent == "results" or kind == "results":
        parts = [_title_prefixed_text(title, text, "Result")]
        if tab:
            parts.append(f"This is usually interpreted in the {tab} tab.")
        return " ".join(part for part in parts if part)

    parts = [text]
    if kind == "workflow" and str(getattr(best, "source_tier", "") or "").casefold() == "curated":
        steps_match = re.search(
            r"\bSteps:\s*(.*?)\s*Expected result:", full_text, re.IGNORECASE | re.DOTALL
        )
        if steps_match:
            steps = _clean_sentence(steps_match.group(1))
            if steps:
                limited_steps = steps if len(steps) <= 320 else steps[:317].rstrip() + "..."
                parts.append(f"Steps: {limited_steps}")
    if source == "curated_overrides":
        return " ".join(part for part in parts if part)
    if intent == "tutorial":
        if tab == "Advanced Options":
            parts.append(
                f"For a general explanation, start from the {_DOCS_LABEL} button or ReadTheDocs."
            )
        elif tab:
            parts.append(f"This is mainly covered in the {tab} tab.")
    elif intent == "results" and tab:
        parts.append(f"This is mainly interpreted from the {tab} outputs.")
    return " ".join(part for part in parts if part)


def _question_mentions_popup(question: str) -> bool:
    return bool(_POPUP_QUESTION_RE.search(question or ""))


def _popup_is_relevant(question: str) -> bool:
    if _question_mentions_popup(question):
        return True
    return bool(_CURRENT_CONTEXT_REFERENCE_RE.search(question or ""))


def _question_is_about_aqme_workflow(question: str) -> bool:
    return bool(_AQME_WORKFLOW_QUESTION_RE.search(question or ""))


def _question_is_about_descriptor_generation(question: str) -> bool:
    return bool(_DESCRIPTOR_GENERATION_QUESTION_RE.search(question or ""))


def _question_mentions_aqme_advanced_controls(question: str) -> bool:
    return bool(_AQME_ADVANCED_HINT_RE.search(question or ""))


def _question_is_about_aqme_tab(question: str) -> bool:
    return bool(_AQME_TAB_QUESTION_RE.search(question or ""))


def _question_is_about_aqme_live_context(
    question: str,
    snapshot: GuiSnapshot | None = None,
    intent: str = "diagnostic",
) -> bool:
    if _AQME_LIVE_CONTEXT_RE.search(question or ""):
        return True
    return bool(
        snapshot is not None
        and snapshot.active_tab == "AQME"
        and intent == "diagnostic"
        and _CURRENT_CONTEXT_REFERENCE_RE.search(question or "")
    )


def _format_aqme_workflow_answer(question: str) -> str:
    if _question_mentions_aqme_advanced_controls(question):
        return (
            "The AQME tab is the advanced area for descriptor generation in easyROB. "
            "After you enable `Enable AQME Workflow`, that tab lets you tune AQME settings, "
            "inspect common-structure/FMCS detection, and optionally choose atom-based descriptors. "
            "Most users do not need to change that tab unless they want to customize descriptor generation."
        )
    return (
        "Use `Enable AQME Workflow` when easyROB needs to generate descriptors from molecular structures "
        "such as SMILES. For the common case, checking that option is enough and you do not need to open "
        "the AQME tab unless you want to tune descriptor settings, inspect FMCS/common-structure detection, "
        "or choose atomic descriptors. If your CSV already contains the descriptors you want to model, AQME "
        "is usually not required."
    )


def _format_descriptor_generation_options() -> str:
    return (
        "If your CSV already contains useful molecular descriptor columns for the model, AQME is not needed. "
        "If it contains SMILES but lacks model-relevant descriptors, choose between two paths: press `Run AQME` "
        "to generate descriptor CSV files only, or check `Enable AQME Workflow` and press `Run ROBERT` to generate "
        "descriptors automatically and train the model in one run. The AQME tab is optional and mainly for advanced "
        "settings such as atom-specific descriptors from a common substructure."
    )


def _format_aqme_tab_answer() -> str:
    return (
        "The AQME tab is the advanced area for structure-based preprocessing and descriptor generation in easyROB. "
        "After `Enable AQME Workflow` is checked, that tab lets you configure AQME, inspect common-structure/FMCS "
        "detection, and optionally choose atom-based descriptors. Most users only need to enable AQME when they want "
        "descriptors from SMILES or other structures; opening the AQME tab is mainly for advanced customization."
    )


def _format_aqme_live_context_answer(snapshot: GuiSnapshot) -> str | None:
    if snapshot.aqme_message_text:
        parts = [f"In the AQME tab, the current message is: {snapshot.aqme_message_text}"]
        if snapshot.aqme_info_text:
            parts.append(snapshot.aqme_info_text)
        if snapshot.aqme_multiple_matches_detected:
            parts.append(
                "This means the detected common substructure matches more than one place in at least one molecule, "
                "so atom-specific selection can become ambiguous."
            )
        elif snapshot.aqme_smarts_pattern:
            parts.append(
                f"The current common substructure/FMCS pattern is `{snapshot.aqme_smarts_pattern}`."
            )
        return " ".join(part for part in parts if part)

    if snapshot.aqme_smarts_pattern:
        parts = [
            f"The AQME tab currently detected the common substructure/FMCS pattern `{snapshot.aqme_smarts_pattern}`."
        ]
        if snapshot.aqme_info_text:
            parts.append(snapshot.aqme_info_text)
        if snapshot.aqme_selected_atoms:
            parts.append(
                f"Selected atom positions in that pattern: {', '.join(str(idx) for idx in snapshot.aqme_selected_atoms)}."
            )
        if snapshot.aqme_descriptor_level:
            parts.append(f"Descriptor level: {snapshot.aqme_descriptor_level}.")
        return " ".join(part for part in parts if part)

    return None


def _normalized_question(question: str) -> str:
    value = unicodedata.normalize("NFKD", str(question or "").casefold())
    return " ".join(
        "".join(character for character in value if not unicodedata.combining(character)).split()
    )


def _format_live_gui_state_answer(question: str, snapshot: GuiSnapshot) -> str | None:
    """Answer explicit questions about the current GUI without retrieval guesses."""
    normalized = _normalized_question(question)
    spanish = bool(re.search(r"\b(que|cual|tengo|esta|ahora|pestana|seleccionad[oa])\b", normalized))
    current_hint = bool(re.search(
        r"\b(current|currently|now|right now|selected|active|open|loaded|running|"
        r"actual|ahora|seleccionad[oa]|activa|abierta|cargad[oa]|ejecutando|tengo)\b",
        normalized,
    ))

    if re.search(r"\b(tab|pestana)\b", normalized) and (
        current_hint or re.search(r"\b(am i in|donde estoy|which tab|what tab)\b", normalized)
    ):
        value = snapshot.active_tab or "unknown"
        return f"La pestaña activa es **{value}**." if spanish else f"The active tab is **{value}**."

    if "workflow" in normalized and current_hint:
        value = snapshot.workflow or "unknown"
        return (
            f"El workflow seleccionado es **{value}**."
            if spanish else f"The selected workflow is **{value}**."
        )

    if re.search(r"\b(what is running|what's running|que se esta ejecutando|que esta corriendo)\b", normalized):
        if snapshot.process_running:
            value = snapshot.active_process or "A workflow"
            return f"**{value}** se está ejecutando ahora." if spanish else f"**{value} is running** now."
        return "No hay ningún proceso ejecutándose ahora." if spanish else "No process is running now."

    if re.search(r"\b(columns?|columnas?)\b", normalized) and current_hint:
        parts = []
        if snapshot.target_column:
            parts.append(f"target={snapshot.target_column}")
        if snapshot.name_column:
            parts.append(f"names={snapshot.name_column}")
        if snapshot.ignored_columns:
            parts.append(f"ignored={', '.join(snapshot.ignored_columns)}")
        values = "; ".join(parts) or "none"
        return f"Columnas seleccionadas: **{values}**." if spanish else f"Selected columns: **{values}**."

    if re.search(r"\b(files?|csv|archivos?)\b", normalized) and current_hint:
        main_name = os.path.basename(snapshot.main_csv_path.replace("\\", "/")) if snapshot.main_csv_path else "none"
        test_name = os.path.basename(snapshot.test_csv_path.replace("\\", "/")) if snapshot.test_csv_path else "none"
        if spanish:
            return f"CSV principal: **{main_name}**. CSV de test: **{test_name}**."
        return f"Main CSV: **{main_name}**. Test CSV: **{test_name}**."

    if re.search(r"\b(advanced settings|advanced options|ajustes avanzados|opciones avanzadas)\b", normalized) and current_hint:
        if not snapshot.advanced_settings:
            return "No se detectaron ajustes avanzados." if spanish else "No advanced settings were detected."
        group_by_name = {
            "auto_type": "General", "seed": "General", "kfold": "General",
            "repeat_kfolds": "General", "split": "General",
            "categorical": "CURATE", "corr_filter_x": "CURATE", "corr_filter_y": "CURATE",
            "desc_thres": "CURATE", "thres_x": "CURATE", "thres_y": "CURATE",
            "models": "GENERATE", "error_type": "GENERATE", "init_points": "GENERATE",
            "n_iter": "GENERATE", "expect_improv": "GENERATE", "pfi_filter": "GENERATE",
            "pfi_epochs": "GENERATE", "pfi_threshold": "GENERATE", "pfi_max": "GENERATE",
            "auto_test": "GENERATE", "test_set": "GENERATE",
            "t_value": "PREDICT", "shap_show": "PREDICT", "pfi_show": "PREDICT",
        }
        grouped: dict[str, list[str]] = {}
        for setting in snapshot.advanced_settings:
            name = setting.partition("=")[0]
            grouped.setdefault(group_by_name.get(name, "Other"), []).append(setting)
        heading = "Ajustes actuales de Advanced Options:" if spanish else "Current Advanced Options:"
        order = ("General", "CURATE", "GENERATE", "PREDICT", "Other")
        lines = [f"- {group}: {', '.join(grouped[group])}" for group in order if group in grouped]
        return f"{heading}\n\n" + "\n".join(lines)

    return None


def _popup_context(snapshot: GuiSnapshot, question: str) -> dict[str, object] | None:
    if snapshot.popup_text.strip():
        return {
            "title": snapshot.popup_title,
            "text": snapshot.popup_text,
            "buttons": snapshot.popup_buttons,
            "kind": snapshot.popup_kind,
            "source": snapshot.popup_source,
            "active": bool(snapshot.popup_active),
            "recent": not bool(snapshot.popup_active),
            "age_seconds": snapshot.popup_age_seconds,
        }
    return None


def _format_popup_answer(popup: dict[str, object]) -> str:
    title = _clean_sentence(str(popup.get("title", "") or "Popup"))
    text = _clean_sentence(str(popup.get("text", "") or ""))
    buttons = [str(button) for button in popup.get("buttons", []) or []]
    lowered = text.lower()

    parts = [f"{title}: {text}" if title else text]
    age_seconds = popup.get("age_seconds")
    if popup.get("recent") and isinstance(age_seconds, (int, float)):
        parts.insert(0, f"This popup appeared about {age_seconds:.1f} seconds ago.")
    if {"Yes", "No"}.issubset(set(buttons)) and any(term in lowered for term in ("delete", "overwritten", "overwrite")):
        parts.append(
            "`Yes` continues and allows easyROB to delete or overwrite those previous-run folders. "
            "`No` cancels that action so you can keep or inspect the old outputs first."
        )
    elif buttons:
        parts.append(f"Available choices: {', '.join(buttons)}.")
    return " ".join(part for part in parts if part)


def _extract_console_failure_detail(console_text: str) -> str:
    text = str(console_text or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    exception_line = ""
    traceback_line = ""
    for line in reversed(lines):
        if not exception_line and _EXCEPTION_LINE_RE.search(line):
            exception_line = line
        if not traceback_line:
            match = _TRACEBACK_LINE_RE.search(line)
            if match:
                traceback_line = f'{match.group(1)}, line {match.group(2)}, in {match.group(3).strip()}'
        if exception_line and traceback_line:
            break

    if exception_line and traceback_line:
        return f"{exception_line}. It appears around {traceback_line}."
    if exception_line:
        return exception_line
    if traceback_line:
        return f"The traceback points to {traceback_line}."
    return ""


def _format_process_failure_popup_answer(snapshot: GuiSnapshot, popup: dict[str, object]) -> str:
    title = _clean_sentence(str(popup.get("title", "") or "WARNING!"))
    text = _clean_sentence(str(popup.get("text", "") or ""))
    failure_detail = _extract_console_failure_detail(snapshot.recent_console)

    if failure_detail:
        return (
            f"{title}: {text} "
            f"The concrete error in the console is: {failure_detail} "
            "This is the detail to diagnose, not just the popup itself."
        )

    return _format_popup_answer(popup)


def _structured_failure_answer(snapshot: GuiSnapshot, popup: dict[str, object]) -> str:
    if not snapshot.failure_type and not snapshot.failure_message:
        return ""

    parts = [
        f"{popup.get('title', 'WARNING!')}: {popup.get('text', '')}".strip(),
        (
            f"The concrete error is {snapshot.failure_type}: {snapshot.failure_message}."
            if snapshot.failure_type and snapshot.failure_message
            else ""
        ),
        (
            f"It happens around {snapshot.failure_location} in {snapshot.failure_function}."
            if snapshot.failure_location and snapshot.failure_function
            else ""
        ),
        (
            f"The failing operation is `{snapshot.failure_operation}`."
            if snapshot.failure_operation
            else ""
        ),
        (
            f"Most likely cause: {snapshot.likely_cause}."
            if snapshot.likely_cause
            else ""
        ),
        infer_target_column_failure_hint(
            snapshot.target_column,
            snapshot.prediction_type,
            snapshot.failure_type,
            snapshot.failure_operation,
            snapshot.likely_cause,
        ),
    ]
    return " ".join(part for part in parts if part)


def format_heuristic_answer(
    question: str,
    snapshot: GuiSnapshot,
    diagnosis: Diagnosis,
    candidates: Sequence[object],
    intent: str = "diagnostic",
) -> str:
    live_state_answer = _format_live_gui_state_answer(question, snapshot)
    if live_state_answer:
        return live_state_answer

    popup = _popup_context(snapshot, question)
    if popup is not None and (
        _popup_is_relevant(question)
    ):
        if str(popup.get("source", "") or "").endswith("_finish_failure"):
            structured = _structured_failure_answer(snapshot, popup)
            if structured:
                return structured
            return _format_process_failure_popup_answer(snapshot, popup)
        return _format_popup_answer(popup)

    if _question_is_about_aqme_live_context(question, snapshot, intent):
        aqme_answer = _format_aqme_live_context_answer(snapshot)
        if aqme_answer:
            return aqme_answer

    if intent != "diagnostic":
        if _question_is_about_aqme_tab(question):
            return _format_aqme_tab_answer()
        if _question_is_about_aqme_workflow(question):
            return _format_aqme_workflow_answer(question)
        if _question_is_about_descriptor_generation(question):
            return _format_descriptor_generation_options()
        return _format_guidance_answer(question, snapshot, candidates, intent)

    parts = [
        f"What is happening: {diagnosis.summary}",
        f"Why: {diagnosis.reason}",
        f"What to check next: {diagnosis.next_step}",
    ]

    return "\n".join(parts)
