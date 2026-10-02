"""Snapshot extraction helpers for the EasyROB GUI bot."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Final

from .error_parser import parse_python_failure
from .workflow_results import WorkflowResultSnapshot

__all__ = ["POPUP_CONTEXT_TTL_SECONDS", "GuiSnapshot", "build_gui_snapshot"]


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_RECENT_CONSOLE_LIMIT = 2000
POPUP_CONTEXT_TTL_SECONDS: Final[float] = 30.0
_POPUP_TITLE_LIMIT = 240
_POPUP_TEXT_LIMIT = 4000
_POPUP_BUTTON_LIMIT = 120
_POPUP_BUTTON_COUNT_LIMIT = 12
_POPUP_TAG_LIMIT = 120
_GUI_LIST_ITEM_LIMIT = 160
_GUI_LIST_COUNT_LIMIT = 40
_CONSOLE_PHRASE_MARKERS = (
    "column missing",
    "target column must be numeric",
    "target column is empty",
    "no valid molecules",
    "please select a csv file",
    "expected output csv",
    "could not delete folder",
)


def _maybe_call(value: Any, *args: Any) -> Any:
    return value(*args) if callable(value) else value


def _safe_maybe_call(value: Any, *args: Any, default: Any = None) -> Any:
    """Call one explicitly selected GUI accessor once, treating failures as absent."""
    try:
        return _maybe_call(value, *args)
    except Exception:
        return default


def _get_attr(obj: Any, name: str, default: Any = "") -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _text_from_widget(widget: Any, method_name: str, default: str = "", *args: Any) -> str:
    method = _get_attr(widget, method_name, None)
    if not callable(method):
        return default
    value = _safe_maybe_call(method, *args, default=None)
    if value is None:
        return default
    return str(value)


def _tokenize(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_tokenize(item))
        return tokens
    return _TOKEN_RE.findall(str(value).lower())


def _append_console_phrase_markers(console_text: str, terms: list[str]) -> list[str]:
    lowered = console_text.lower()
    enriched = list(terms)
    for marker in _CONSOLE_PHRASE_MARKERS:
        if marker in lowered and marker not in enriched:
            enriched.append(marker)
    if "not found" in lowered and "not found" not in enriched:
        enriched.append("not found")
    return enriched


def _popup_value(popup: Any, key: str, default: Any = "") -> Any:
    if popup is None:
        return default
    if isinstance(popup, dict):
        return popup.get(key, default)
    return _get_attr(popup, key, default)


def _bounded_popup_text(value: Any, limit: int) -> str:
    if callable(value) or value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        return ""
    raw_text = str(value)
    safe_text = "".join(
        " " if ord(character) < 32 or 127 <= ord(character) <= 159 else character
        for character in raw_text
    )
    text = " ".join(safe_text.split())
    return text[:limit]


def _popup_buttons(popup: Any) -> list[str]:
    value = _popup_value(popup, "buttons", [])
    if isinstance(value, str):
        item = _bounded_popup_text(value, _POPUP_BUTTON_LIMIT)
        return [item] if item else []
    if isinstance(value, (list, tuple)):
        buttons: list[str] = []
        for raw_item in list(value)[:_POPUP_BUTTON_COUNT_LIMIT]:
            item = _bounded_popup_text(raw_item, _POPUP_BUTTON_LIMIT)
            if item:
                buttons.append(item)
        return buttons
    return []


def _recent_console_slice(value: str, limit: int = _RECENT_CONSOLE_LIMIT) -> str:
    if limit <= 0:
        return ""
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    normalized: list[int] = []
    for item in items:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_text_items(value: Any) -> tuple[str, ...]:
    """Return a bounded, detached tuple from GUI-owned text values."""
    if not isinstance(value, (list, tuple)):
        return ()
    normalized: list[str] = []
    for item in value[:_GUI_LIST_COUNT_LIMIT]:
        text = _bounded_popup_text(item, _GUI_LIST_ITEM_LIMIT)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _collect_list_widget_text(widget: Any) -> tuple[str, ...]:
    count_accessor = _get_attr(widget, "count", None)
    count_value = _safe_maybe_call(count_accessor, default=0) if callable(count_accessor) else 0
    try:
        count = max(0, min(int(count_value or 0), _GUI_LIST_COUNT_LIMIT))
    except (TypeError, ValueError):
        return ()
    values: list[str] = []
    item_accessor = _get_attr(widget, "item", None)
    if not callable(item_accessor):
        return ()
    for index in range(count):
        item = _safe_maybe_call(item_accessor, index, default=None)
        text = _bounded_popup_text(_text_from_widget(item, "text", ""), _GUI_LIST_ITEM_LIMIT)
        if text:
            values.append(text)
    return tuple(values)


def _line_edit_setting(widget: Any) -> str:
    text = _text_from_widget(widget, "text", "").strip()
    if text:
        return text
    placeholder = _text_from_widget(widget, "placeholderText", "").strip()
    return f"default ({placeholder})" if placeholder else ""


def _checked(widget: Any) -> bool:
    accessor = _get_attr(widget, "isChecked", None)
    return bool(_safe_maybe_call(accessor, default=False)) if callable(accessor) else False


def _collect_advanced_settings(options: Any) -> tuple[str, ...]:
    if options is None:
        return ()
    values: list[str] = []
    checkbox_fields = (
        ("auto_type", "auto_type"),
        ("corr_filter_xbool", "corr_filter_x"),
        ("corr_filter_ybool", "corr_filter_y"),
        ("pfi_filter", "pfi_filter"),
        ("auto_test", "auto_test"),
    )
    combo_fields = (
        ("split", "split"),
        ("categoricalstr", "categorical"),
        ("error_type", "error_type"),
    )
    line_fields = (
        ("seed", "seed"),
        ("kfold", "kfold"),
        ("repeat_kfolds", "repeat_kfolds"),
        ("desc_thresfloat", "desc_thres"),
        ("thres_xfloat", "thres_x"),
        ("thres_yfloat", "thres_y"),
        ("init_points", "init_points"),
        ("n_iter", "n_iter"),
        ("expect_improv", "expect_improv"),
        ("pfi_epochs", "pfi_epochs"),
        ("pfi_threshold", "pfi_threshold"),
        ("pfi_max", "pfi_max"),
        ("test_set", "test_set"),
        ("t_value", "t_value"),
        ("shap_show", "shap_show"),
        ("pfi_show", "pfi_show"),
    )
    for attr_name, label in checkbox_fields:
        widget = _get_attr(options, attr_name, None)
        if widget is not None:
            values.append(f"{label}={_checked(widget)}")
    for attr_name, label in combo_fields:
        text = _text_from_widget(_get_attr(options, attr_name, None), "currentText", "").strip()
        if text:
            values.append(f"{label}={text}")
    for attr_name, label in line_fields:
        text = _line_edit_setting(_get_attr(options, attr_name, None))
        if text:
            values.append(f"{label}={text}")
    model_widgets = _get_attr(options, "modellist", {})
    if isinstance(model_widgets, dict):
        selected_models = [
            str(name) for name, widget in sorted(model_widgets.items(), key=lambda item: str(item[0]))
            if _checked(widget)
        ]
        if selected_models:
            values.append(f"models={', '.join(selected_models)}")
    return _normalize_text_items(values)


def _popup_age(value: Any) -> float | None:
    if type(value) is int:
        if value < 0 or value >= POPUP_CONTEXT_TTL_SECONDS:
            return None
        return float(value)
    if type(value) is not float:
        return None
    if not math.isfinite(value) or value < 0.0 or value >= POPUP_CONTEXT_TTL_SECONDS:
        return None
    return value


def _is_enabled(widget: Any) -> bool:
    if widget is None:
        return False
    if hasattr(widget, "isEnabled"):
        accessor = _get_attr(widget, "isEnabled", None)
        if not callable(accessor):
            return False
        value = _safe_maybe_call(accessor, default=False)
        return bool(value)
    return bool(widget)


def _is_process_running(window: Any, run_enabled: bool, stop_enabled: bool) -> bool:
    if stop_enabled and not run_enabled:
        return True
    if run_enabled and not stop_enabled:
        return False

    current_process = str(_get_attr(window, "current_process", "") or "").strip()
    if current_process and current_process.lower() not in {"idle", "stopped", "none"}:
        return True
    return False


def _collect_tab_states(tab_widget: Any) -> tuple[list[str], list[str]]:
    count_method = _get_attr(tab_widget, "count", None)
    count_value = _safe_maybe_call(count_method, default=0) if callable(count_method) else 0
    try:
        count = int(count_value or 0)
    except (TypeError, ValueError):
        count = 0

    enabled_tabs: list[str] = []
    disabled_tabs: list[str] = []
    for index in range(max(count, 0)):
        tab_name = _text_from_widget(tab_widget, "tabText", "", index).strip()
        if not tab_name:
            continue
        is_enabled_method = _get_attr(tab_widget, "isTabEnabled", None)
        is_enabled = bool(_safe_maybe_call(is_enabled_method, index, default=False)) if callable(is_enabled_method) else True
        if is_enabled:
            enabled_tabs.append(tab_name)
        else:
            disabled_tabs.append(tab_name)
    return enabled_tabs, disabled_tabs


@dataclass(frozen=True, slots=True)
class GuiSnapshot:
    active_tab: str
    workflow: str
    main_csv_path: str
    test_csv_path: str
    target_column: str
    prediction_type: str
    process_running: bool
    run_enabled: bool
    stop_enabled: bool
    recent_console: str
    console_terms: list[str]
    name_column: str = ""
    active_process: str = ""
    run_aqme_enabled: bool = False
    ignored_columns: tuple[str, ...] = field(default_factory=tuple)
    advanced_settings: tuple[str, ...] = field(default_factory=tuple)
    aqme_workflow_enabled: bool = False
    enabled_tabs: list[str] = field(default_factory=list)
    disabled_tabs: list[str] = field(default_factory=list)
    popup_active: bool = False
    popup_title: str = ""
    popup_text: str = ""
    popup_buttons: tuple[str, ...] = field(default_factory=tuple)
    popup_kind: str = ""
    popup_source: str = ""
    popup_age_seconds: float | None = None
    failure_type: str = ""
    failure_message: str = ""
    failure_location: str = ""
    failure_function: str = ""
    failure_operation: str = ""
    likely_cause: str = ""
    aqme_smarts_pattern: str = ""
    aqme_message_text: str = ""
    aqme_message_tooltip: str = ""
    aqme_info_text: str = ""
    aqme_selected_atoms: list[int] = field(default_factory=list)
    aqme_descriptor_level: str = ""
    aqme_solvent: str = ""
    aqme_atoms_text: str = ""
    aqme_multiple_matches_detected: bool = False
    aqme_metal_found: bool = False
    aqme_unified_smiles_count: int = 0
    workflow_results: WorkflowResultSnapshot | None = None
    result_model: str = ""
    result_models: tuple[str, ...] = field(default_factory=tuple)
    result_selected_variants: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    result_active_view: str = ""
    result_enabled_views: tuple[str, ...] = field(default_factory=tuple)
    result_disabled_views: tuple[str, ...] = field(default_factory=tuple)
    all_models_enabled: bool = False

    def __post_init__(self) -> None:
        """Keep the public snapshot valid even when tests construct it directly."""
        object.__setattr__(self, "ignored_columns", _normalize_text_items(self.ignored_columns))
        object.__setattr__(self, "advanced_settings", _normalize_text_items(self.advanced_settings))
        object.__setattr__(self, "popup_title", _bounded_popup_text(self.popup_title, _POPUP_TITLE_LIMIT))
        object.__setattr__(self, "popup_text", _bounded_popup_text(self.popup_text, _POPUP_TEXT_LIMIT))
        object.__setattr__(self, "popup_buttons", tuple(_popup_buttons({"buttons": self.popup_buttons})))
        object.__setattr__(self, "popup_kind", _bounded_popup_text(self.popup_kind, _POPUP_TAG_LIMIT).lower())
        object.__setattr__(self, "popup_source", _bounded_popup_text(self.popup_source, _POPUP_TAG_LIMIT).lower())
        if self.popup_active:
            object.__setattr__(self, "popup_age_seconds", None)
            return
        age = _popup_age(self.popup_age_seconds)
        if age is not None:
            object.__setattr__(self, "popup_age_seconds", age)
            return
        object.__setattr__(self, "popup_title", "")
        object.__setattr__(self, "popup_text", "")
        object.__setattr__(self, "popup_buttons", ())
        object.__setattr__(self, "popup_kind", "")
        object.__setattr__(self, "popup_source", "")
        object.__setattr__(self, "popup_age_seconds", None)


def build_gui_snapshot(window: Any) -> GuiSnapshot:
    tab_widget = _get_attr(window, "tab_widget", None)
    current_index_accessor = _get_attr(tab_widget, "currentIndex", None)
    current_index = _safe_maybe_call(current_index_accessor, default=0) if callable(current_index_accessor) else 0
    active_tab = _text_from_widget(tab_widget, "tabText", "", current_index)
    if not active_tab and _get_attr(window, "current_tab", None):
        active_tab = str(_get_attr(window, "current_tab", ""))
    elif not active_tab and _get_attr(tab_widget, "currentText", None):
        active_tab = _text_from_widget(tab_widget, "currentText", "")

    result_view = active_tab in {"Results", "Reports", "Images", "Predictions"}
    result_source = str(_get_attr(window, "_result_view_source_path", "") or "") if result_view else ""
    evaluate_candidate = _get_attr(window, "evaluate_tab", None)
    evaluate_tab = evaluate_candidate if (
        active_tab == "Check model"
        or (result_source and result_source == str(_get_attr(evaluate_candidate, "csv_path", "") or ""))
    ) else None
    input_view = evaluate_tab if evaluate_tab is not None else window
    workflow_selector = _get_attr(window, "workflow_selector", None)
    workflow = "EVALUATE" if evaluate_tab is not None else _text_from_widget(workflow_selector, "currentText", "")

    main_csv_path = result_source or str(
        _get_attr(input_view, "csv_path", None)
        or _get_attr(input_view, "file_path", None)
        or _get_attr(input_view, "main_csv_path", None)
        or _get_attr(window, "_result_view_source_path", None)
        or ""
    )
    test_csv_path = str(
        _get_attr(input_view, "csv_test_path", None)
        or _get_attr(input_view, "test_csv_path", None)
        or ""
    )

    target_column = _text_from_widget(_get_attr(input_view, "y_dropdown", None), "currentText", "")
    if not target_column:
        target_column = str(_get_attr(input_view, "target_column", "") or "")

    prediction_type = _text_from_widget(_get_attr(input_view, "type_dropdown", None), "currentText", "")
    if not prediction_type:
        prediction_type = str(_get_attr(input_view, "prediction_type", "") or "")

    name_column = _text_from_widget(_get_attr(input_view, "names_dropdown", None), "currentText", "")

    run_enabled = _is_enabled(_get_attr(input_view, "run_button", None))
    stop_enabled = _is_enabled(_get_attr(input_view, "stop_button", None))
    process_running = _is_process_running(input_view, run_enabled, stop_enabled)
    active_process = (
        "EVALUATE" if evaluate_tab is not None else str(_get_attr(window, "current_process", "") or "").strip()
    ) if process_running else ""
    run_aqme_enabled = _is_enabled(_get_attr(window, "run_aqme_button", None))
    ignored_columns = _collect_list_widget_text(_get_attr(window, "ignore_list", None))
    advanced_settings = _collect_advanced_settings(_get_attr(window, "options_tab", None))
    enabled_tabs, disabled_tabs = _collect_tab_states(tab_widget)
    result_model = ""
    result_models: tuple[str, ...] = ()
    result_selected_variants: tuple[tuple[str, str], ...] = ()
    result_active_view = ""
    result_enabled_views: tuple[str, ...] = ()
    result_disabled_views: tuple[str, ...] = ()
    catalog = _get_attr(window, "_result_catalog", None)
    workspace = _get_attr(window, "results_workspace", None)
    if catalog is not None and workspace is not None and main_csv_path:
        try:
            same_run = Path(main_csv_path).resolve().parent == Path(catalog.root).resolve()
        except (OSError, ValueError, TypeError):
            same_run = False
        if same_run:
            result_models = tuple(str(model) for model in _get_attr(catalog, "models", ())) if _get_attr(catalog, "is_all_models", False) else ()
            current_model = _safe_maybe_call(_get_attr(workspace, "current_model", None), default=None)
            result_model = str(current_model) if current_model else "Best models" if result_models else ""
            selected = _safe_maybe_call(_get_attr(catalog, "selected_variants", None), current_model, default={}) if result_models else {}
            if isinstance(selected, dict):
                result_selected_variants = tuple((str(variant), str(model)) for variant, model in selected.items())
            buttons = _get_attr(workspace, "buttons", {})
            if isinstance(buttons, dict):
                result_enabled_views = tuple(name for name, button in buttons.items() if _is_enabled(button))
                result_disabled_views = tuple(name for name, button in buttons.items() if not _is_enabled(button))
                result_active_view = next((name for name, button in buttons.items() if _checked(button)), "")

    console_output = _get_attr(input_view, "console_output", None)
    recent_console = _recent_console_slice(_text_from_widget(console_output, "toPlainText", ""))
    console_terms = _append_console_phrase_markers(recent_console, _tokenize(recent_console))
    parsed_failure = parse_python_failure(recent_console)

    aqme_workflow = _get_attr(window, "aqme_workflow", None)
    aqme_checked_accessor = _get_attr(aqme_workflow, "isChecked", None)
    aqme_workflow_enabled = bool(_safe_maybe_call(aqme_checked_accessor, default=False)) if callable(aqme_checked_accessor) else False
    active_popup = _get_attr(window, "bot_active_popup", None)
    popup_context = _get_attr(window, "bot_popup_context", None)
    popup_context_age_seconds = _popup_age(_get_attr(window, "bot_popup_context_age_seconds", None))
    has_active_popup = active_popup is not None
    effective_popup = active_popup if has_active_popup else None
    effective_popup_age_seconds = None
    if not has_active_popup and popup_context is not None:
        is_fresh = popup_context_age_seconds is not None
        if is_fresh:
            effective_popup = popup_context
            effective_popup_age_seconds = popup_context_age_seconds
    aqme_tab = _get_attr(window, "tab_widget_aqme", None)
    smarts_targets = _get_attr(aqme_tab, "smarts_targets", [])
    aqme_smarts_pattern = ""
    if isinstance(smarts_targets, (list, tuple)) and smarts_targets:
        aqme_smarts_pattern = str(smarts_targets[0] or "")
    elif smarts_targets:
        aqme_smarts_pattern = str(smarts_targets or "")
    aqme_message_text = str(_get_attr(aqme_tab, "bot_context_message", "") or "")
    aqme_message_tooltip = str(_get_attr(aqme_tab, "bot_context_tooltip", "") or "")
    aqme_info_text = str(_get_attr(aqme_tab, "bot_context_info", "") or "")
    if not aqme_info_text:
        aqme_info_text = _text_from_widget(_get_attr(aqme_tab, "mol_info_label", None), "text", "")
    aqme_descriptor_level = _text_from_widget(_get_attr(aqme_tab, "descriptor_level", None), "currentText", "")
    aqme_solvent = _text_from_widget(_get_attr(aqme_tab, "solvent", None), "currentText", "")
    aqme_atoms_text = _text_from_widget(_get_attr(aqme_tab, "atoms", None), "text", "")
    aqme_selected_atoms = _normalize_int_list(_get_attr(aqme_tab, "selected_atoms", []))
    aqme_multiple_matches_detected = bool(_get_attr(aqme_tab, "multiple_matches_detected", False))
    aqme_metal_found = bool(_get_attr(aqme_tab, "metal_found", False))
    unified_smiles = _get_attr(aqme_tab, "unified_smiles", [])
    aqme_unified_smiles_count = len(unified_smiles) if isinstance(unified_smiles, (list, tuple, set)) else 0

    return GuiSnapshot(
        active_tab=active_tab,
        workflow=workflow,
        main_csv_path=main_csv_path,
        test_csv_path=test_csv_path,
        target_column=target_column,
        prediction_type=prediction_type,
        process_running=process_running,
        run_enabled=run_enabled,
        stop_enabled=stop_enabled,
        recent_console=recent_console,
        console_terms=console_terms,
        name_column=name_column,
        active_process=active_process,
        run_aqme_enabled=run_aqme_enabled,
        ignored_columns=ignored_columns,
        advanced_settings=advanced_settings,
        aqme_workflow_enabled=aqme_workflow_enabled,
        enabled_tabs=enabled_tabs,
        disabled_tabs=disabled_tabs,
        popup_active=has_active_popup,
        popup_title=_bounded_popup_text(_popup_value(effective_popup, "title", ""), _POPUP_TITLE_LIMIT),
        popup_text=_bounded_popup_text(_popup_value(effective_popup, "text", ""), _POPUP_TEXT_LIMIT),
        popup_buttons=_popup_buttons(effective_popup),
        popup_kind=_bounded_popup_text(_popup_value(effective_popup, "kind", ""), _POPUP_TAG_LIMIT).lower(),
        popup_source=_bounded_popup_text(_popup_value(effective_popup, "source", ""), _POPUP_TAG_LIMIT).lower(),
        popup_age_seconds=effective_popup_age_seconds,
        failure_type=parsed_failure.failure_type,
        failure_message=parsed_failure.failure_message,
        failure_location=parsed_failure.failure_location,
        failure_function=parsed_failure.failure_function,
        failure_operation=parsed_failure.failure_operation,
        likely_cause=parsed_failure.likely_cause,
        aqme_smarts_pattern=aqme_smarts_pattern,
        aqme_message_text=aqme_message_text,
        aqme_message_tooltip=aqme_message_tooltip,
        aqme_info_text=aqme_info_text,
        aqme_selected_atoms=aqme_selected_atoms,
        aqme_descriptor_level=aqme_descriptor_level,
        aqme_solvent=aqme_solvent,
        aqme_atoms_text=aqme_atoms_text,
        aqme_multiple_matches_detected=aqme_multiple_matches_detected,
        aqme_metal_found=aqme_metal_found,
        aqme_unified_smiles_count=aqme_unified_smiles_count,
        workflow_results=(
            _get_attr(window, "workflow_result_snapshot", None)
            if isinstance(_get_attr(window, "workflow_result_snapshot", None), WorkflowResultSnapshot)
            else None
        ),
        result_model=result_model,
        result_models=result_models,
        result_selected_variants=result_selected_variants,
        result_active_view=result_active_view,
        result_enabled_views=result_enabled_views,
        result_disabled_views=result_disabled_views,
        all_models_enabled=_checked(_get_attr(window, "all_models_toggle", None)),
    )
