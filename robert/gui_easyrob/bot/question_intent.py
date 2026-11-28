"""Question intent classification for the EasyROB bot."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Sequence

__all__ = [
    "classify_question_intent",
    "is_gui_question",
    "is_explicit_web_request",
    "is_generic_question",
    "is_follow_up_question",
    "is_workflow_result_question",
    "requires_fresh_information",
    "select_inference_history",
    "last_completed_exchange",
]


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_DIAGNOSTIC_PHRASES = (
    "why is",
    "why are",
    "why can",
    "why can't",
    "cannot run",
    "can't run",
    "not working",
    "doesn't work",
    "disabled",
    "blocked",
    "error",
    "failed",
    "failure",
    "warning",
    "not found",
    "locked",
    "por que",
    "no funciona",
    "deshabilitad",
    "bloquead",
    "fallo",
    "advertencia",
)
_TUTORIAL_PHRASES = (
    "how do i",
    "how can i",
    "how to",
    "what do i need",
    "which parameters",
    "what parameters",
    "how it works",
    "how does",
    "what is",
    "where can i",
    "what should i do next",
)
_RESULTS_TERMS = {
    "resultado", "resultados", "informe", "informes", "metrica", "metricas",
    "prediccion", "predicciones", "puntuacion", "puntuaciones",
    "result",
    "results",
    "metric",
    "metrics",
    "rmse",
    "mae",
    "r2",
    "squared",
    "accuracy",
    "score",
    "scores",
    "report",
    "reports",
    "prediction",
    "predictions",
    "no_pfi",
    "interpret",
    "interpretation",
    "model",
    "models",
}
_METRIC_TERMS = {
    "rmse",
    "mae",
    "r2",
    "accuracy",
    "score",
    "scores",
    "metric",
    "metrics",
}
_PARAMETER_TERMS = {
    "parameter",
    "parameters",
    "option",
    "options",
    "threshold",
    "cutoff",
    "setting",
    "settings",
    "pfi",
}
_MODULE_TERMS = {
    "generate",
    "curate",
    "predict",
    "verify",
    "qdescp",
    "csearch",
    "cmin",
    "qcorr",
    "qprep",
}
_NAMED_SUBJECTS = _MODULE_TERMS | _METRIC_TERMS | {"aqme", "robert", "pfi", "shap"}
_TUTORIAL_TERMS = {
    "how",
    "use",
    "using",
    "start",
    "setup",
    "configure",
    "parameter",
    "parameters",
    "option",
    "options",
    "need",
    "before",
    "workflow",
    "works",
}
_FOLLOW_UP_TERMS = {
    "it",
    "that",
    "this",
    "those",
    "these",
    "them",
    "there",
    "and",
    "also",
    "then",
    "enabled",
    "disabled",
    "tab",
}
_FOLLOW_UP_PHRASES = (
    "what should i check next",
    "what do i check next",
    "what next",
    "next step",
    "explicamelo",
    "explicalo",
    "explicarlo mas",
    "mas detalle",
    "more detail",
)
_OPAQUE_FOLLOW_UPS = {
    "what",
    "what?",
    "excuse me",
    "excuse me?",
    "sorry",
    "sorry?",
    "huh",
    "huh?",
    "why",
    "why?",
    "por que",
    "por que?",
    "como",
    "como?",
}
_LIVE_CONTEXT_HINT_RE = re.compile(
    r"\b(current|currently|here|now|just|popup|dialog|warning|message|alert|error|failed|failure|blocked|disabled|loaded|selected|shown|displayed|this|that|these|it|my|missing|overlap|ahora|actualmente|aqui|pestana|deshabilitada|deshabilitado|bloqueada|bloqueado|cargada|cargado|seleccionada|seleccionado|mi|mis|este|esta|estos|estas|falta|faltan|ausente|ausentes|solapamiento)\b",
    re.IGNORECASE,
)
_GENERIC_TOPIC_HINT_RE = re.compile(
    r"\b(?:aqme|robert|descriptors?|descriptores?|features?|machine learning|models?|modelos?|"
    r"molecules?|predictions?|predicciones?|concept|meaning|difference|rmse|mae|r2|accuracy|"
    r"metrics?|metricas?|pfi|no_pfi|shap|overfitting|cross[- ]validation|regression|classification)\b",
)
_EXPLANATION_RE = re.compile(
    r"^(?:(?:and|y)\s+)?(?:what (?:is|are|does|do)\b|what['’]s\b|what .+ means?\b|"
    r"how (?:does|do|to)\b|how many .+ (?:does|do)\b|why (?:does|do|is|are)\b|"
    r"(?:can you )?(?:explain|define)\b|que (?:es|son|significa|hace|hacen|muestra)\b|"
    r"como (?:funciona|funcionan)\b|"
    r"(?:puedes )?explic(?:a|ar|ame)\b|define\b|difference between\b|(?:que )?diferencia (?:hay )?entre\b)",
)
_GUI_CONTROL_RE = re.compile(
    r"\b(?:buttons?|tabs?|pop[- ]?ups?|dialogs?|checkbox(?:es)?|drop[- ]?downs?|toggles?|"
    r"message box|boton(?:es)?|pestanas?|ventanas?|dialogos?|casillas?|desplegables?)\b|"
    r"\b(?:views?|vistas?)\s+(?:is\s+|esta\s+)?(?:enabled|disabled|available|unavailable|"
    r"habilitada|deshabilitada|disponible)\b",
)
_GUI_MESSAGE_RE = re.compile(
    r"\b(?:this|that|the|este|esta|ese|esa)\s+(?:warning|error|alert|message|advertencia|aviso|mensaje)\b|"
    r"\b(?:warning|error|alert|message|advertencia|aviso|mensaje)\s+(?:on screen|displayed|en pantalla)\b",
)
_GUI_PROCEDURE_RE = re.compile(
    r"^(?:(?:how (?:do|can|should) i|how to|where (?:do|can) i)\s+"
    r"(?:run|start|load|open|find|view|configure|set|activate|enable|export|save|train)\b|"
    r"(?:como|donde)\s+(?:ejecut\w+|inici\w+|carg\w+|abr\w+|encuentr\w+|ver|"
    r"configur\w+|activ\w+|export\w+|guard\w+|entren\w+)\b)",
)
_SUMMARY_COMMAND_RE = re.compile(
    r"^(?:(?:please|por favor)[, ]+)?(?:summari[sz]e|resumen|resume|resumir)"
    r"(?:[ .?!]*$|\s+(?:(?:my|our|the|this|el|los|mi|mis|este)\s+)?"
    r"(?:(?:robert|aqme)\s+)?(?:results?|resultados?|reports?|informes?|run|"
    r"ejecucion|workflow|flujo|all models|todos los modelos)\b)",
)
_RESULT_SUBJECT_RE = re.compile(
    r"\b(?:results?|resultados?|reports?|informes?|metrics?|metricas?|rmse|mae|r2|r\^2|"
    r"roc[- _]?auc|auc|accuracy|f1|mcc|scores?|puntuaciones?|test|validation|validacion|"
    r"predictions?|predicciones?|outliers?|warnings?|advertencias?|avisos?|errores?|"
    r"shap|pfi|no_pfi|features?|importance|importancia|descriptors?|descriptores?|"
    r"uncertaint(?:y|ies)|incertidumbres?|overlap|solapamiento|leakage|"
    r"outputs?|salidas?|files?|archivos?|curate|generate|verify|predict|aqme|qdescp|csearch|robert|"
    r"workflow|flujo|run|ejecucion|models?|modelos?|solvent|disolvente|solvente)\b",
)
_RUN_REFERENCE_RE = re.compile(
    r"\b(?:my|our|this|the last|mi|mis|nuestros?|esta|este|ultima|ultimo)\s+"
    r"(?:robert\s+|aqme\s+)?(?:results?|resultados?|run|ejecucion|workflow|flujo|reports?|informes?|"
    r"predictions?|predicciones?|metrics?|metricas?|rmse|mae|r2|score|puntuacion|pfi|shap|csvs?)\b|"
    r"\b(?:in|from|of|en|del)\s+(?:the|el|los)\s+(?:results?|resultados?|run|reports?|informes?)\b",
)
_RUN_OUTCOME_RE = re.compile(
    r"\b(?:did|was|were|passed|failed|finished|completed|happened|generated|produced|created|"
    r"obtained|recorded|fue|fueron|dio|dieron|paso|aprobo|fallo|termino|ocurrio|se ejecuto|"
    r"generad[oa]s?|genero|generaron|produjo|cread[oa]s?|obtuve|obtuvimos)\b",
)
_RESULT_REQUEST_RE = re.compile(
    r"\b(?:show|compare|interpret|analy[sz]e|muestra|muestran|compara|interpreta|analiza|"
    r"faltan|missing|best|mejores?)\b",
)
_FOCUSED_RESULT_RE = re.compile(
    r"\b(?:shap|pfi|no_pfi|rmse|mae|r2)\s+(?:for|para)\s+[a-z0-9_]{2,4}\s*[?.!]*$|"
    r"\b(?:test|train|validation|validacion)\s+(?:rmse|mae|r2|accuracy|f1|mcc|score)\b|"
    r"\b(?:the|los|el)\s+(?:results?|resultados?|metrics?|metricas?|best model|mejor modelo)\s*[?.!]*$",
)
_OPERATIONAL_HINT_RE = re.compile(
    r"\b(where|file|folder|path|project|csv|smiles|load|open|select|enable|disable|run|click|button|tab|start|configure|set|archivo|carpeta|ruta|proyecto|cargar|abrir|seleccionar|activar|desactivar|ejecutar|pulsar|boton|pestana|configurar)\b",
    re.IGNORECASE,
)
_GREETING_TERMS = {
    "hi",
    "hello",
    "hey",
    "heyy",
    "ey",
    "yo",
    "hola",
    "buenas",
    "hello!",
    "hi!",
    "hey!",
}
_GREETING_PHRASES = (
    "how are you",
    "how are u",
    "how ar eu",
    "how r u",
    "how's it going",
    "how is it going",
)
_GREETING_PHRASE_TERMS = _GREETING_TERMS | {
    "how", "are", "you", "u", "ar", "eu", "doing", "today", "is", "it", "going", "s",
}
_EXPLICIT_WEB_RE = re.compile(
    r"\b(?:search|look up|browse|buscar|busca|consulta)\b.*\b(?:web|internet|online|en linea)\b",
    re.IGNORECASE,
)
_FRESH_INFORMATION_RE = re.compile(
    r"\b(?:latest|newest|recent|today|this week|this month|currently available|"
    r"current (?:[a-z0-9_.-]+ ){0,3}(?:version|release|documentation|docs|price|status|recommendation)|"
    r"ultima version|última versión|mas reciente|más reciente|hoy|esta semana|actualmente disponible)\b",
    re.IGNORECASE,
)


def _normalize_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(normalized.casefold().split())


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize_text(value)))


def _last_user_message(conversation_history: Sequence[Mapping[str, str]] | None) -> str:
    if not conversation_history:
        return ""
    for message in reversed(conversation_history):
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return ""


def _looks_like_follow_up(question: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    named_subject = re.match(
        r"^(?:and|y) (?:what (?:does|do|is|are)|how (?:does|do)|"
        r"que (?:hace|es|son)|como funciona) ([a-z0-9_]+)\b",
        question,
    )
    if named_subject and named_subject.group(1) in _NAMED_SUBJECTS | {"a", "an", "un", "una"}:
        return False
    if any(phrase in question for phrase in _FOLLOW_UP_PHRASES):
        return True
    if not tokens & {"rmse", "mae", "r2", "accuracy", "mcc", "f1", "auc"} and re.search(
        r"\b(?:lower|higher|better|worse)\b.*\b(?:value|score|number)\b|"
        r"\b(?:value|score|number)\b.*\b(?:lower|higher|better|worse)\b",
        question,
    ):
        return True
    if tokens & {"each", "either"} and tokens & {"use", "choose"}:
        return True
    if len(tokens) <= 8 and tokens & {
        "it", "that", "this", "those", "these", "them",
        "eso", "esto", "esos", "estos", "esas", "estas",
    }:
        return True
    if re.search(r"\b(?:como|donde) (?:lo|la|los|las)\b", question):
        return True
    if question.startswith(("y ", "entonces ", "and ", "what about ", "what if ")):
        return True
    if question.startswith(("what is ", "what are ", "how does ", "how do i ", "how can i ", "which ", "where ")):
        return False
    if len(tokens) <= 5 and tokens & _FOLLOW_UP_TERMS:
        return True
    return question.startswith(("and ", "what about ", "what if ", "it ", "that "))


def is_follow_up_question(question: str) -> bool:
    """Return whether *question* conservatively depends on prior conversation."""

    clean_question = _normalize_text(question).lstrip("¿¡")
    return clean_question in _OPAQUE_FOLLOW_UPS or _looks_like_follow_up(
        clean_question, _tokenize(clean_question)
    )


def is_explicit_web_request(question: str) -> bool:
    """Return whether the user explicitly requested an online lookup."""
    return type(question) is str and bool(_EXPLICIT_WEB_RE.search(question))


def requires_fresh_information(question: str) -> bool:
    """Return whether a question depends on time-sensitive external facts."""
    return type(question) is str and bool(_FRESH_INFORMATION_RE.search(question))


def is_gui_question(question: str) -> bool:
    """Identify requests about interactive controls or an on-screen message."""
    text = _normalize_text(question).lstrip("¿¡")
    return bool(_GUI_CONTROL_RE.search(text) or _GUI_PROCEDURE_RE.search(text) or (
        _GUI_MESSAGE_RE.search(text) and not _RUN_REFERENCE_RE.search(text)
    ))


def is_workflow_result_question(
    question: str,
    intent: str | None = None,
    *,
    named_model: bool = False,
) -> bool:
    """Identify questions that depend on a particular workflow execution."""
    text = _normalize_text(question).lstrip("¿¡")
    if _SUMMARY_COMMAND_RE.match(text):
        return True
    if not text or is_gui_question(text):
        return False
    if _RUN_REFERENCE_RE.search(text) or re.search(r"\b(?:what happened|que ocurrio|que paso)\b", text):
        return True
    result_subject = bool(_RESULT_SUBJECT_RE.search(text))
    if _FOCUSED_RESULT_RE.search(text) or (result_subject and named_model):
        return True
    if _EXPLANATION_RE.match(text):
        return False
    if _RUN_OUTCOME_RE.search(text):
        return result_subject
    return (intent or classify_question_intent(text)) == "results" or (
        result_subject and bool(_RESULT_REQUEST_RE.search(text))
    )


def is_generic_question(
    question: str,
    intent: str | None = None,
    *,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
) -> bool:
    """Return whether a self-contained explanation should omit live GUI context."""
    if type(question) is not str:
        return False
    clean_question = _normalize_text(question).lstrip("¿¡")
    resolved_intent = (
        str(intent or "").strip().lower()
        or classify_question_intent(clean_question)
    )
    if is_gui_question(clean_question) or is_workflow_result_question(clean_question, resolved_intent):
        return False
    if any(marker in clean_question for marker in ("<", ">", "\x00")):
        return False
    if is_follow_up_question(clean_question):
        previous = _last_user_message(conversation_history)
        return bool(previous and is_generic_question(previous))
    if _LIVE_CONTEXT_HINT_RE.search(clean_question):
        return False
    if _OPERATIONAL_HINT_RE.search(clean_question):
        return False
    return bool(_EXPLANATION_RE.match(clean_question) or _GENERIC_TOPIC_HINT_RE.search(clean_question))


def _canonical_history_record(record: object) -> dict[str, str] | None:
    if not isinstance(record, Mapping):
        return None
    role = record.get("role")
    content = record.get("content")
    if not isinstance(role, str) or role not in {"user", "assistant"} or not isinstance(content, str):
        return None
    clean_content = content.strip()
    if not clean_content or record.get("memory_eligible", True) is not True:
        return None
    return {"role": role, "content": clean_content}


def select_inference_history(
    question: str,
    conversation_history: Sequence[Mapping[str, object]] | None,
) -> tuple[dict[str, str], ...]:
    """Return the nearest complete standalone exchange behind a follow-up chain."""

    if not is_follow_up_question(question):
        return ()
    records = tuple(conversation_history) if isinstance(conversation_history, (list, tuple)) else ()
    selected = last_completed_exchange(records)
    while (selected and len(records) >= 4
           and _canonical_history_record(records[-2]) == selected[0]
           and _canonical_history_record(records[-1]) == selected[1]
           and is_follow_up_question(selected[0]["content"])):
        previous_records = records[:-2]
        previous = last_completed_exchange(previous_records)
        if (not previous or _canonical_history_record(previous_records[-2]) != previous[0]
                or _canonical_history_record(previous_records[-1]) != previous[1]):
            break
        selected = previous
        records = previous_records
    return selected


def last_completed_exchange(
    conversation_history: Sequence[Mapping[str, object]] | None,
) -> tuple[dict[str, str], ...]:
    """Select adjacent eligible turns without joining across invalid records."""
    records = tuple(conversation_history) if isinstance(conversation_history, (list, tuple)) else ()
    if len(records) < 2:
        return ()
    user_record = _canonical_history_record(records[-2])
    assistant_record = _canonical_history_record(records[-1])
    if (
        user_record is not None
        and assistant_record is not None
        and user_record["role"] == "user"
        and assistant_record["role"] == "assistant"
    ):
        return (user_record, assistant_record)
    return ()


def classify_question_intent(
    question: str,
    conversation_history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    clean_question = _normalize_text(question)
    tokens = _tokenize(clean_question)
    if not clean_question:
        return "tutorial"
    if clean_question in _GREETING_TERMS:
        return "greeting"
    if tokens and len(tokens) <= 3 and tokens <= _GREETING_TERMS:
        return "greeting"
    if (
        any(phrase in clean_question for phrase in _GREETING_PHRASES)
        and tokens <= _GREETING_PHRASE_TERMS
    ):
        return "greeting"
    if is_gui_question(clean_question):
        if _GUI_MESSAGE_RE.search(clean_question) or any(phrase in clean_question for phrase in _DIAGNOSTIC_PHRASES):
            return "diagnostic"
        return "tutorial"
    explanation_follow_up = bool(re.search(
        r"\b(?:important|matter|importa)\b", clean_question,
    )) and not bool(re.search(
        r"\b(?:error|warning|failed|failure|blocked|disabled|fallo|"
        r"advertencia|bloquead\w*|deshabilitad\w*)\b", clean_question,
    ))
    if is_follow_up_question(clean_question) and (
        explanation_follow_up
        or not any(phrase in clean_question for phrase in _DIAGNOSTIC_PHRASES)
    ):
        previous_user_question = _last_user_message(conversation_history)
        if previous_user_question and previous_user_question.strip().lower() != clean_question:
            previous_intent = classify_question_intent(previous_user_question, conversation_history=None)
            if previous_intent != "greeting":
                return previous_intent

    diagnostic_score = sum(3 for phrase in _DIAGNOSTIC_PHRASES if phrase in clean_question)
    tutorial_score = sum(2 for phrase in _TUTORIAL_PHRASES if phrase in clean_question)
    results_score = len(tokens & _RESULTS_TERMS) * 2
    if (tokens & {"mi", "mis", "este", "esta", "estos", "estas"}
            and tokens & {"modelo", "modelos"}
            and re.search(r"\b(?:mejor\w*|peor\w*|confi\w*|fiab\w*|rendim\w*|compar\w*|resultados?)\b", clean_question)):
        results_score += 4
    parameter_score = len(tokens & _PARAMETER_TERMS) * 2
    module_score = len(tokens & _MODULE_TERMS) * 2

    if {"run", "csv"} <= tokens:
        diagnostic_score += 2
    if {"tab", "enabled"} <= tokens or {"tab", "disabled"} <= tokens or {"tab", "locked"} <= tokens:
        diagnostic_score += 2
    if "r squared" in clean_question:
        results_score += 6
    if "predictions" in tokens and ("identical" in tokens or "same" in tokens):
        results_score += 6
        diagnostic_score = max(0, diagnostic_score - 3)
    if "pfi" in tokens and ("no_pfi" in tokens or "no pfi" in clean_question or "vs" in tokens):
        results_score += 6
        parameter_score = max(0, parameter_score - 2)
    if {"parameter", "robert"} <= tokens or {"parameters", "robert"} <= tokens:
        tutorial_score += 3
    if {"need", "run"} <= tokens or {"need", "workflow"} <= tokens:
        tutorial_score += 2
    if {"what", "does"} <= tokens and tokens & _PARAMETER_TERMS:
        parameter_score += 2
    if {"how", "interpret"} <= tokens or {"what", "mean"} <= tokens:
        results_score += 2
    if "what" in tokens and tokens & _METRIC_TERMS:
        results_score += 2
    if tokens & _MODULE_TERMS and ("what" in tokens or "how" in tokens):
        module_score += 1

    tutorial_score += len(tokens & _TUTORIAL_TERMS)

    if diagnostic_score == 0 and results_score == 0 and tutorial_score == 0 and parameter_score == 0 and module_score == 0:
        return "tutorial"
    if diagnostic_score >= max(tutorial_score, results_score, parameter_score, module_score):
        return "diagnostic"
    if parameter_score >= max(results_score, tutorial_score, module_score) and parameter_score > 0:
        return "parameter"
    if results_score > max(tutorial_score, module_score):
        return "results"
    if module_score > tutorial_score:
        return "module-explanation"
    return "tutorial"
