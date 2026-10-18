"""Answer orchestration for the EasyROB GUI bot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import re
from typing import Callable, Final, Mapping, Sequence
import unicodedata

from .answer_metadata import AnswerMetadata, EvidenceOrigin, WebSearchMode
from .bot_context import GuiSnapshot
from .workflow_results import _requested_variants, render_workflow_answer, scope_workflow_result_snapshot
from .bot_rag import KnowledgeBase, SearchResult, normalize_terms
from .heuristics import diagnose_snapshot, format_heuristic_answer
from .evidence_policy import EvidenceDecision, assess_evidence, sanitize_search_query
from .llm_providers import (
    LLMProviderError,
    LLMProviderErrorKind,
    _response_language,
    build_llm_prompts,
    build_local_llm_prompts,
    build_provider_registry,
)
from .local_llm import LocalLLMManager
from .knowledge_schema import KnowledgeLoadIssue
from .question_intent import classify_question_intent, is_generic_question, select_inference_history
from .prompt_policy import select_answer_depth
from .response_quality import result_summary_issues
from .token_usage import MAX_USER_PROMPT_TOKENS, TokenUsage, estimate_token_count

__all__ = ["BotAnswerResult", "BotEngine", "MAX_USER_QUESTION_TOKENS"]


_TUTORIALS_BY_ID = {
    "overview": ("tutorial:overview.md", "Overview"),
    "csv": ("tutorial:csv.md", "From CSV"),
    "chemdraw": ("tutorial:chemdraw.md", "From ChemDraw"),
    "predictions": ("tutorial:predictions.md", "New Predictions"),
    "descriptors": ("tutorial:descriptors.md", "Descriptors"),
    "check_model": ("tutorial:check_model.md", "Check ML"),
    "robbot": ("tutorial:robbot.md", "robBOT"),
}
_EXACT_TUTORIAL_QUESTIONS = {
    "how does the gui overview work": "overview",
    "how do i start from a csv file": "csv",
    "how do i start from chemdraw": "chemdraw",
    "how do i make predictions for new molecules": "predictions",
    "how do i generate descriptors without training a model": "descriptors",
    "how do i check my own machine learning model": "check_model",
    "how do i use robbot": "robbot",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _result_selection_note(snapshot: GuiSnapshot, question: str) -> str:
    """Explain the single overall-best report retained in the run folder."""
    if len(snapshot.result_root_reports) != 1 or not snapshot.result_archived_report_count:
        return ""
    filename = snapshot.result_root_reports[0]
    selected = ", ".join(
        f"{variant}={model}" for variant, model in snapshot.result_selected_variants
    )
    if _response_language(question) == "Spanish":
        return (
            f"**Selección de Results:** ROBERT dejó un único informe final en la carpeta "
            f"principal, `{filename}` ({selected}). Hay {snapshot.result_archived_report_count} "
            "informes en `REPORT_models`; selecciona un modelo concreto para consultar sus "
            "otras variantes."
        )
    return (
        f"**Results selection:** ROBERT kept one final report in the run folder, "
        f"`{filename}` ({selected}). There are {snapshot.result_archived_report_count} "
        "reports in `REPORT_models`; choose a named model to inspect its other variants."
    )


def _result_provider_fallback(
    snapshot: GuiSnapshot, question: str, *, full: bool, reason: str = "",
) -> "BotAnswerResult":
    explanation = (
        "Cloud AI no pudo responder esta vez. Este resumen se ha creado directamente a partir "
        "de los datos registrados del workflow, sin interpretación del modelo de IA.\n\n"
        if _response_language(question) == "Spanish" else
        "Cloud AI could not answer this time. This summary comes directly from recorded "
        "workflow data, without AI interpretation.\n\n"
    )
    if reason:
        if _response_language(question) == "Spanish":
            translated = {
                "the generated answer conflicted with the workflow evidence": "la respuesta generada contradecía los datos del workflow",
                "rate limit": "límite de uso del proveedor",
                "connection": "problema de conexión",
                "timeout": "tiempo de espera agotado",
                "authentication": "problema de autenticación",
                "invalid request": "solicitud rechazada",
                "invalid response": "respuesta no válida del proveedor",
                "unavailable service": "servicio no disponible",
                "unexpected bot error": "error interno del bot",
            }.get(reason, "problema del proveedor")
            explanation += f"Motivo de la respuesta local: {translated}.\n\n"
        else:
            explanation += f"Reason for local answer: {reason}.\n\n"
    return BotAnswerResult(
        explanation + "\n\n".join(part for part in (
            _result_selection_note(snapshot, question) if snapshot.result_model == "Best models" else "",
            render_workflow_answer(snapshot.workflow_results, question, full=full),
        ) if part),
        memory_eligible=False,
        metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.LOCAL_SYSTEM),
    )
_TUTORIAL_OVERVIEW_LIMIT = 1
_TUTORIAL_BULLET_LIMIT = 4
_POPUP_QUESTION_RE = re.compile(r"\b(pop[- ]?up|dialog|warning|message box|message|alert|window)\b", re.IGNORECASE)
_CURRENT_DATE_QUESTIONS = {
    "current date",
    "cual es la fecha de hoy",
    "fecha actual",
    "fecha de hoy",
    "que dia es hoy",
    "que fecha es hoy",
    "todays date",
    "what date is it",
    "what day is it today",
    "what day is today",
    "what is todays date",
}
_SPANISH_DATE_QUESTIONS = {
    "cual es la fecha de hoy",
    "fecha actual",
    "fecha de hoy",
    "que dia es hoy",
    "que fecha es hoy",
}
_CURRENT_TIME_QUESTIONS = {
    "current time",
    "hora actual",
    "que hora es",
    "que hora es ahora",
    "what is the current time",
    "what time is it",
}
_SPANISH_TIME_QUESTIONS = {
    "hora actual",
    "que hora es",
    "que hora es ahora",
}
_SPANISH_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_SPANISH_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
MAX_USER_QUESTION_TOKENS: Final[int] = MAX_USER_PROMPT_TOKENS


@dataclass(frozen=True, slots=True)
class TutorialSummary:
    source: str
    label: str
    overview: str
    bullets: tuple[str, ...]
    candidates: tuple[SearchResult, ...]


@dataclass(frozen=True, slots=True)
class BotAnswerResult:
    text: str
    auto_reset_context: bool = False
    memory_eligible: bool = True
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: AnswerMetadata = field(default_factory=AnswerMetadata)


class BotEngine:
    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        provider_registry: Mapping[str, object] | None = None,
        local_manager: LocalLLMManager | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        tutorials_dir = Path(__file__).resolve().parents[1] / "tutorials"
        self._knowledge_base = knowledge_base or KnowledgeBase.from_directory(
            tutorials_dir=tutorials_dir,
            strict=False,
        )
        self._provider_registry = dict(provider_registry or build_provider_registry())
        self._local_manager = local_manager or LocalLLMManager()
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(self._provider_registry)

    @property
    def knowledge_issues(self) -> tuple[KnowledgeLoadIssue, ...]:
        return self._knowledge_base.issues

    @property
    def knowledge_is_degraded(self) -> bool:
        return self._knowledge_base.is_degraded

    def answer(
        self,
        question: str,
        snapshot: GuiSnapshot,
        mode: str,
        provider: str,
        api_key: str,
        model: str | None = None,
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        tutorial_id: str | None = None,
        web_search_enabled: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        return self.answer_with_metadata(
            question=question,
            snapshot=snapshot,
            mode=mode,
            provider=provider,
            api_key=api_key,
            model=model,
            conversation_history=conversation_history,
            tutorial_id=tutorial_id,
            web_search_enabled=web_search_enabled,
            progress_callback=progress_callback,
        ).text

    def answer_with_metadata(
        self,
        question: str,
        snapshot: GuiSnapshot,
        mode: str,
        provider: str,
        api_key: str,
        model: str | None = None,
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        tutorial_id: str | None = None,
        web_search_enabled: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> BotAnswerResult:
        clean_question = question.strip() if type(question) is str else ""
        if not clean_question:
            return BotAnswerResult("Please enter a question for the bot.", memory_eligible=False)
        if estimate_token_count(clean_question) > MAX_USER_QUESTION_TOKENS:
            return BotAnswerResult(
                "Please shorten your question to 500 estimated tokens or fewer.",
                memory_eligible=False,
            )

        requested_variants = _requested_variants(clean_question)
        named_model = any(re.search(
            rf"(?<![\w]){re.escape(model)}(?![\w])", clean_question, re.IGNORECASE,
        ) for model in snapshot.result_models)
        asks_all_models = bool(re.search(
            r"\b(?:all models|todos los modelos|cada modelo)\b", clean_question.casefold(),
        ))
        selected_variants = {variant for variant, _ in snapshot.result_selected_variants}
        if (requested_variants and snapshot.result_models and selected_variants
                and not named_model and not asks_all_models
                and not set(requested_variants).issubset(selected_variants)):
            selected = ", ".join(
                f"{variant}={model}" for variant, model in snapshot.result_selected_variants
            )
            archive_note = (
                f" Hay {snapshot.result_archived_report_count} informes archivados en REPORT_models."
                if snapshot.result_archived_report_count else ""
            )
            if _response_language(clean_question) == "Spanish":
                reply = (
                    f"La selección actual de Results ({snapshot.result_model or 'Best models'}) "
                    f"solo contiene {selected}. La variante que preguntas no está seleccionada."
                    f"{archive_note} Elige un modelo concreto en el selector Model para consultar "
                    "sus variantes disponibles."
                )
            else:
                archive_note = (
                    f" There are {snapshot.result_archived_report_count} archived reports in REPORT_models."
                    if snapshot.result_archived_report_count else ""
                )
                reply = (
                    f"The current Results selection ({snapshot.result_model or 'Best models'}) "
                    f"contains only {selected}. The requested variant is not selected."
                    f"{archive_note} Choose a named model in the Model selector to inspect "
                    "its available variants."
                )
            return BotAnswerResult(
                reply, memory_eligible=False,
                metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.LOCAL_SYSTEM),
            )

        if (len(snapshot.result_root_reports) == 1 and snapshot.result_archived_report_count
                and re.search(r"\b(?:report|informe|pdf)\b", clean_question, re.IGNORECASE)
                and re.search(r"\b(?:only|single|one|solo|unico|único|uno)\b", clean_question, re.IGNORECASE)):
            return BotAnswerResult(
                _result_selection_note(snapshot, clean_question), memory_eligible=False,
                metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.LOCAL_SYSTEM),
            )

        if snapshot.workflow_results is not None and snapshot.result_models:
            snapshot = replace(snapshot, workflow_results=scope_workflow_result_snapshot(
                snapshot.workflow_results,
                clean_question,
                selected_variants=snapshot.result_selected_variants,
                available_models=snapshot.result_models,
                prefer_root_report=(
                    snapshot.result_model == "Best models" and len(snapshot.result_root_reports) == 1
                    and not named_model and not asks_all_models
                ),
            ))

        local_clock_answer = self._build_local_clock_answer(clean_question)
        if local_clock_answer:
            return BotAnswerResult(
                local_clock_answer,
                metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.LOCAL_SYSTEM),
            )

        self._emit_progress(progress_callback, "checking_documentation")

        selected_history = select_inference_history(clean_question, conversation_history)
        question_intent = classify_question_intent(
            clean_question,
            conversation_history=selected_history,
        )
        result_evidence_question = snapshot.workflow_results is not None and bool(re.search(
            r"\b(?:results?|resultados?|summari[sz]e|summary|resumen|resume|resumir|curate|generate|"
            r"verify|predict|report|aqme|qdescp|csearch|rmse|mae|r2|roc[- _]?auc|auc|accuracy|f1|mcc|"
            r"metric|metrics|metrica|metricas|"
            r"prediction|predictions|prediccion|predicciones|warning|warnings|outlier|outliers|failed|fallo|fallaron)\b",
            clean_question.casefold(),
        ))
        if snapshot.workflow_results is not None and not result_evidence_question:
            result_evidence_question = bool(re.search(
                r"\b(?:how many|cu[aá]ntos?|which|qu[eé]|what)\b.*\b(?:descriptors?|descriptores?|"
                r"files?|archivos?|outputs?|salidas?|uncertaint(?:y|ies)|incertidumbre(?:s)?)\b.*"
                r"\b(?:generated|created|produced|obtain(?:ed)?|generad[oa]s?|cread[oa]s?|obtenid[oa]s?)\b|"
                r"\b(?:descriptors?|descriptores?|files?|archivos?|outputs?|salidas?)\b.*"
                r"\b(?:were|was|se)\s+(?:generated|created|produced|generaron|crearon|obtuvieron)\b",
                clean_question.casefold(),
            ))
        result_summary_request = bool(re.search(
            r"\b(?:summari[sz]e|summary|resumen|resume|resumir)\b",
            clean_question.casefold(),
        ))
        if snapshot.workflow_results is not None and result_summary_request and re.search(
            r"\b(?:workflow|report|results?|informe|resultados?|flujo)\b", clean_question, re.IGNORECASE
        ):
            question_intent = "results"
        if question_intent == "greeting":
            return BotAnswerResult(self._build_greeting_reply(clean_question, snapshot))
        if (self._is_low_information_question(clean_question) and not selected_history
                and not (snapshot.workflow_results is not None and result_evidence_question)):
            return BotAnswerResult(
                "Please ask a more specific question about the GUI, workflow, results, or the step you want help with.",
                memory_eligible=False,
            )
        tutorial_summary = self._build_suggested_tutorial_summary(clean_question, tutorial_id=tutorial_id)
        diagnosis = diagnose_snapshot(snapshot, clean_question)
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode == "heuristic" and self._should_force_popup_answer(clean_question, snapshot):
            return BotAnswerResult(
                format_heuristic_answer(
                    clean_question,
                    snapshot,
                    diagnosis,
                    [],
                    intent=question_intent,
                )
            )
        if (
            question_intent == "tutorial"
            and "check next" in clean_question.lower()
            and not diagnosis.summary.startswith("No specific block")
        ):
            question_intent = "diagnostic"
        answer_depth = select_answer_depth(clean_question, question_intent)
        generic_question = is_generic_question(clean_question, question_intent)
        if snapshot.workflow_results is not None and result_summary_request and question_intent == "results":
            generic_question = False
        if generic_question:
            result_evidence_question = False
        # Resolve retrieval references using user evidence, never generated claims.
        retrieval_question = clean_question
        if selected_history:
            retrieval_question = f"{clean_question}\nPrevious user question: {selected_history[0]['content'][:2000]}"
        candidates = self._knowledge_base.search(
            question=retrieval_question,
            active_tab=None if generic_question or question_intent == "tutorial" else snapshot.active_tab,
            workflow=None if generic_question else snapshot.workflow,
            console_terms=snapshot.console_terms if question_intent == "diagnostic" and not generic_question else [],
            question_intent=question_intent,
            focus_tags=() if generic_question else self._infer_focus_tags(retrieval_question, snapshot),
        )
        if tutorial_summary is not None:
            candidates = list(tutorial_summary.candidates)
        candidates = self._prioritize_candidates(candidates, question_intent)

        if normalized_mode == "heuristic":
            if tutorial_summary is not None:
                return BotAnswerResult(self._render_tutorial_summary(tutorial_summary))
            if snapshot.workflow_results is not None and not generic_question and (question_intent == "results" or result_evidence_question):
                result_answer = render_workflow_answer(
                    snapshot.workflow_results,
                    clean_question,
                    full=result_summary_request,
                )
                if (result_summary_request and snapshot.result_model == "Best models"
                        and not named_model and not asks_all_models):
                    note = _result_selection_note(snapshot, clean_question)
                    if note:
                        result_answer = f"{note}\n\n{result_answer}"
                return BotAnswerResult(result_answer)
            return BotAnswerResult(
                format_heuristic_answer(
                    clean_question,
                    snapshot,
                    diagnosis,
                    candidates,
                    intent=question_intent,
                )
            )
        if normalized_mode == "local ai":
            if tutorial_summary is not None:
                deterministic_summary = self._render_tutorial_summary(tutorial_summary)
                try:
                    system_prompt, user_prompt = build_local_llm_prompts(
                        clean_question,
                        snapshot,
                        (),
                        conversation_history=selected_history,
                        question_intent=question_intent,
                        tutorial_summary=self._format_tutorial_summary_for_prompt(tutorial_summary),
                        tutorial_label=tutorial_summary.label,
                        answer_depth=answer_depth,
                    )
                    return self._generate_local_answer(system_prompt, user_prompt)
                except Exception:
                    return BotAnswerResult(deterministic_summary)
            system_prompt, user_prompt = build_local_llm_prompts(
                clean_question,
                snapshot,
                candidates,
                conversation_history=selected_history,
                question_intent=question_intent,
                answer_depth=answer_depth,
            )
            try:
                return self._generate_local_answer(system_prompt, user_prompt)
            except Exception as exc:
                reset_context = self._is_local_bad_request_error(exc)
                try:
                    retry_question = clean_question
                    if selected_history:
                        retry_question += f"\nPrevious topic: {selected_history[0]['content'][:240]}"
                    retry_system_prompt, retry_user_prompt = build_local_llm_prompts(
                        retry_question,
                        snapshot,
                        candidates,
                        conversation_history=(),
                        question_intent=question_intent,
                        minimal_context_retry=True,
                        answer_depth=answer_depth,
                    )
                    retry_result = self._generate_local_answer(retry_system_prompt, retry_user_prompt)
                    retry_answer = retry_result.text
                    if reset_context:
                        retry_answer = (
                            "The local model rejected the first prompt, so I retried using a minimal prompt "
                            "with only the current question and GUI state.\n\n"
                            f"{retry_answer}"
                        )
                    return BotAnswerResult(
                        retry_answer,
                        auto_reset_context=reset_context,
                        usage=retry_result.usage,
                    )
                except Exception as retry_exc:
                    if reset_context:
                        return BotAnswerResult(
                            "Local AI error: the minimal prompt retry could not complete. Try asking again, "
                            "or switch to Heuristic if you want the most stable fallback.",
                            auto_reset_context=True,
                            memory_eligible=False,
                        )
                    return BotAnswerResult(
                        "Local AI could not complete the request. Try asking again, or switch to Heuristic "
                        "if you want the most stable fallback.",
                        memory_eligible=False,
                    )
        if normalized_mode not in {"cloud ai", "llm"}:
            return BotAnswerResult("Unsupported bot mode.", memory_eligible=False)

        provider_name = str(provider or "").strip() or "OpenAI"
        if not str(api_key or "").strip():
            return BotAnswerResult(f"Cloud AI mode requires an API key for {provider_name}.", memory_eligible=False)

        adapter = self._resolve_provider(provider_name)
        if adapter is None:
            return BotAnswerResult(f"Unsupported LLM provider: {provider_name}.", memory_eligible=False)

        assessment = assess_evidence(retrieval_question, candidates, intent=question_intent)
        if assessment.decision is EvidenceDecision.INSUFFICIENT:
            return BotAnswerResult(
                "I do not have enough safe information to answer that question. Please add a little more detail.",
                memory_eligible=False,
                metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.INSUFFICIENT),
            )

        if tutorial_summary is not None and assessment.decision is not EvidenceDecision.WEB_REQUIRED:
            web_search_mode = WebSearchMode.OFF
            intended_origin = EvidenceOrigin.LOCAL_DOCUMENTATION
        else:
            mode_by_decision = {
                EvidenceDecision.LOCAL_ENOUGH: WebSearchMode.OFF,
                EvidenceDecision.GENERAL_ENOUGH: WebSearchMode.OFF,
                EvidenceDecision.WEB_OPTIONAL: WebSearchMode.AUTO,
                EvidenceDecision.WEB_REQUIRED: WebSearchMode.REQUIRED,
            }
            web_search_mode = mode_by_decision[assessment.decision]
            intended_origin = (
                EvidenceOrigin.LOCAL_DOCUMENTATION
                if assessment.decision is EvidenceDecision.LOCAL_ENOUGH
                else EvidenceOrigin.GENERAL_KNOWLEDGE
            )
        limitation = ""
        if web_search_mode is not WebSearchMode.OFF and web_search_enabled is not True:
            web_search_mode = WebSearchMode.OFF
            limitation = "Automatic web search is disabled, so current information was not verified."

        spec = getattr(adapter, "spec", None)
        supports_web_search = getattr(spec, "supports_web_search", False) is True
        if web_search_mode is not WebSearchMode.OFF and not supports_web_search:
            web_search_mode = WebSearchMode.OFF
            limitation = (
                f"{provider_name} does not support native web search in this configuration, "
                "so current information was not verified."
            )

        search_query = ""
        if web_search_mode is not WebSearchMode.OFF:
            search_query = sanitize_search_query(
                retrieval_question,
                secret_values=(str(api_key).strip(),),
            )
            if not search_query:
                return BotAnswerResult(
                    "I could not create a privacy-safe web query, so I cannot verify the current information.",
                    memory_eligible=False,
                    metadata=AnswerMetadata(
                        evidence_origin=EvidenceOrigin.INSUFFICIENT,
                        search_error="External verification was skipped because no safe search query remained.",
                    ),
                )
            self._emit_progress(progress_callback, "searching_current_information")

        self._emit_progress(progress_callback, "generating_answer")

        try:
            generation_kwargs = dict(
                question=clean_question,
                gui_context=snapshot,
                retrieved_chunks=candidates,
                api_key=str(api_key).strip(),
                model=model,
                conversation_history=selected_history,
                question_intent=question_intent,
                tutorial_summary=(
                    self._format_tutorial_summary_for_prompt(tutorial_summary)
                    if tutorial_summary is not None
                    else None
                ),
                tutorial_label=tutorial_summary.label if tutorial_summary is not None else None,
            )
            if hasattr(adapter, "generate_answer_with_usage"):
                documentation_attempted = False
                if web_search_mode is not WebSearchMode.OFF and assessment.reason == "advanced_official_documentation":
                    documentation_attempted = True
                    documented = self._answer_from_online_docs(adapter, generation_kwargs, search_query, answer_depth)
                    if documented is not None:
                        return documented
                try:
                    provider_result = adapter.generate_answer_with_usage(
                        **generation_kwargs,
                        web_search_mode=web_search_mode,
                        search_query=search_query,
                        evidence_origin=(
                            EvidenceOrigin.WEB_AND_DOCUMENTATION
                            if web_search_mode is not WebSearchMode.OFF
                            else intended_origin
                        ),
                        answer_depth=answer_depth,
                    )
                except LLMProviderError as exc:
                    if exc.status_code == 413 and web_search_mode is not WebSearchMode.OFF and not documentation_attempted:
                        fallback = self._answer_from_online_docs(adapter, generation_kwargs, search_query, answer_depth)
                        if fallback is not None:
                            return fallback
                    if web_search_mode is not WebSearchMode.AUTO or exc.status_code != 413:
                        raise
                    provider_result = adapter.generate_answer_with_usage(
                        **generation_kwargs,
                        web_search_mode=WebSearchMode.OFF,
                        search_query="",
                        evidence_origin=intended_origin,
                        answer_depth=answer_depth,
                    )
                    web_search_mode = WebSearchMode.OFF
                    limitation = (
                        "Optional web search was rejected because the provider reported an oversized request; "
                        "the answer was generated without web search."
                    )
                usage = provider_result.usage
                if result_evidence_question and question_intent == "results" and snapshot.workflow_results is not None:
                    result_data = snapshot.workflow_results
                    has_unresolved_verify_checks = any(
                        check[1] in {"FAILED", "UNCLEAR"}
                        for stage in result_data.stages for check in stage.checks
                    )
                    unchanged_test_set_warning = any(
                        "test_set" in warning and "0.2" in warning
                        and bool(re.search(r"(?:0\.2).*?(?:0\.2)", warning))
                        for warning in result_data.warnings
                    )
                    issues = result_summary_issues(
                        provider_result.text,
                        metrics=result_data.metrics,
                        unchanged_test_set_warning=unchanged_test_set_warning,
                        has_unresolved_verify_checks=has_unresolved_verify_checks,
                    )
                    if issues:
                        correction = "\n".join(f"- {issue}" for issue in issues)
                        retry_kwargs = dict(generation_kwargs)
                        retry_kwargs.update(
                            question=(
                                f"{clean_question}\n\nThe previous draft contradicted the recorded evidence. "
                                f"Write a fresh summary and correct these specific errors:\n{correction}"
                            ),
                            retrieved_chunks=(),
                            conversation_history=(),
                        )
                        try:
                            revised = adapter.generate_answer_with_usage(
                                **retry_kwargs,
                                web_search_mode=WebSearchMode.OFF,
                                search_query="",
                                evidence_origin=intended_origin,
                                answer_depth=answer_depth,
                            )
                            usage += revised.usage
                            if not result_summary_issues(
                                revised.text,
                                metrics=result_data.metrics,
                                unchanged_test_set_warning=unchanged_test_set_warning,
                                has_unresolved_verify_checks=has_unresolved_verify_checks,
                            ):
                                provider_result = revised
                            else:
                                return replace(
                                    _result_provider_fallback(
                                        snapshot, clean_question, full=result_summary_request,
                                        reason="the generated answer conflicted with the workflow evidence",
                                    ),
                                    usage=usage,
                                )
                        except LLMProviderError as exc:
                            return replace(
                                _result_provider_fallback(
                                    snapshot, clean_question, full=result_summary_request,
                                    reason=exc.kind.value.replace("_", " "),
                                ),
                                usage=usage,
                            )
                if usage.search_requests > 1:
                    raise LLMProviderError(
                        "The provider exceeded the one-search policy."
                    )
                provider_metadata = getattr(provider_result, "metadata", None)
                metadata = (
                    provider_metadata
                    if isinstance(provider_metadata, AnswerMetadata)
                    else AnswerMetadata(evidence_origin=intended_origin)
                )
                if limitation:
                    metadata = replace(metadata, evidence_origin=intended_origin, search_error=limitation)
                elif usage.search_requests > 0:
                    if not metadata.sources:
                        return BotAnswerResult(
                            "The provider searched the web but returned no usable source citations, "
                            "so I cannot present the result as verified.",
                            memory_eligible=False,
                            usage=usage,
                            metadata=replace(
                                metadata,
                                evidence_origin=EvidenceOrigin.INSUFFICIENT,
                                search_error="Web search returned no usable source citations.",
                            ),
                        )
                    metadata = replace(
                        metadata,
                        evidence_origin=EvidenceOrigin.WEB_AND_DOCUMENTATION,
                    )
                elif web_search_mode is WebSearchMode.REQUIRED:
                    fallback = None if documentation_attempted else self._answer_from_online_docs(adapter, generation_kwargs, search_query, answer_depth, usage)
                    if fallback is not None:
                        return fallback
                    return BotAnswerResult(
                        "I could not verify the current information because the provider performed no web search.",
                        memory_eligible=False,
                        usage=usage,
                        metadata=replace(
                            metadata,
                            evidence_origin=EvidenceOrigin.INSUFFICIENT,
                            search_error="The required web search was not performed.",
                        ),
                    )
                else:
                    metadata = replace(metadata, evidence_origin=intended_origin)
                return BotAnswerResult(provider_result.text, usage=usage, metadata=metadata)
            return BotAnswerResult(adapter.generate_answer(**generation_kwargs))
        except LLMProviderError as exc:
            if result_evidence_question:
                return _result_provider_fallback(
                    snapshot, clean_question, full=result_summary_request,
                    reason=exc.kind.value.replace("_", " "),
                )
            if exc.kind is LLMProviderErrorKind.RATE_LIMIT:
                message = (
                    f"{provider_name} ha alcanzado su límite de uso. Espera un poco y vuelve a enviar la pregunta. "
                    "Si sigue ocurriendo, revisa los límites de tu cuenta del proveedor. Este aviso corresponde al chat, no al flujo de ROBERT."
                    if _response_language(clean_question) == "Spanish" else
                    f"{provider_name} has reached its usage limit. Wait a little and try again. "
                    "If this continues, check your provider account limits. This message concerns the chat, not the ROBERT workflow."
                )
                return BotAnswerResult(message, memory_eligible=False)
            if tutorial_summary is not None:
                return BotAnswerResult(self._render_tutorial_summary(tutorial_summary))
            return BotAnswerResult(f"LLM provider error ({provider_name}): {exc}", memory_eligible=False)
        except Exception as exc:
            if result_evidence_question:
                return _result_provider_fallback(
                    snapshot, clean_question, full=result_summary_request,
                    reason="unexpected bot error",
                )
            if tutorial_summary is not None:
                return BotAnswerResult(self._render_tutorial_summary(tutorial_summary))
            return BotAnswerResult(
                f"Unexpected LLM error ({provider_name}). Please try again.",
                memory_eligible=False,
            )

    @staticmethod
    def _answer_from_online_docs(adapter, generation_kwargs, query, answer_depth, previous_usage=None):
        """Recover an unavailable native search using bounded official page reads."""
        from .online_docs import fetch_documentation
        chunks, sources = fetch_documentation(query)
        if not chunks or not sources:
            return None
        kwargs = dict(generation_kwargs)
        kwargs.update(retrieved_chunks=chunks, tutorial_summary=None, tutorial_label=None)
        result = adapter.generate_answer_with_usage(
            **kwargs, web_search_mode=WebSearchMode.OFF, search_query="",
            evidence_origin=EvidenceOrigin.WEB_AND_DOCUMENTATION, answer_depth=answer_depth,
        )
        usage = previous_usage + result.usage if previous_usage and previous_usage.total_tokens else result.usage
        return BotAnswerResult(
            result.text, usage=usage,
            metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.WEB_AND_DOCUMENTATION, sources=sources),
        )

    @staticmethod
    def _emit_progress(
        callback: Callable[[str], None] | None,
        stage: str,
    ) -> None:
        if not callable(callback):
            return
        try:
            callback(stage)
        except Exception:
            return

    def _resolve_provider(self, provider_name: str) -> object | None:
        if provider_name in self._provider_registry:
            return self._provider_registry[provider_name]
        lowered = provider_name.lower()
        for name, adapter in self._provider_registry.items():
            if name.lower() == lowered:
                return adapter
        return None

    def _build_local_clock_answer(self, question: str) -> str:
        normalized = self._normalize_clock_question(question)
        if normalized not in _CURRENT_DATE_QUESTIONS | _CURRENT_TIME_QUESTIONS:
            return ""
        now = self._now_provider()
        if normalized in _SPANISH_TIME_QUESTIONS:
            return f"Son las {now:%H:%M} (hora local)."
        if normalized in _CURRENT_TIME_QUESTIONS:
            return f"It is {now:%H:%M} local time."
        if normalized in _SPANISH_DATE_QUESTIONS:
            weekday = _SPANISH_WEEKDAYS[now.weekday()]
            month = _SPANISH_MONTHS[now.month - 1]
            return f"Hoy es {weekday}, {now.day} de {month} de {now.year}."
        return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}."

    @staticmethod
    def _normalize_clock_question(question: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(question or ""))
        normalized = "".join(character for character in normalized if not unicodedata.combining(character))
        normalized = normalized.casefold().replace("'", "").replace("’", "")
        return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())

    def _generate_local_answer(self, system_prompt: str, user_prompt: str) -> BotAnswerResult:
        if hasattr(self._local_manager, "generate_response_with_usage"):
            local_result = self._local_manager.generate_response_with_usage(system_prompt, user_prompt)
            return BotAnswerResult(local_result.text, usage=local_result.usage)
        return BotAnswerResult(self._local_manager.generate_response(system_prompt, user_prompt))

    @staticmethod
    def _build_greeting_reply(question: str, snapshot: GuiSnapshot) -> str:
        active_tab = str(getattr(snapshot, "active_tab", "") or "").strip()
        tab_hint = f" I can see you're currently in the {active_tab} tab." if active_tab else ""
        normalized = " ".join(str(question or "").casefold().split())
        wellbeing = any(phrase in normalized for phrase in ("how are", "how r u", "how ar eu"))
        opening = "Hi! I'm doing well, thanks for asking." if wellbeing else "Hi!"
        return f"{opening} How can I help?{tab_hint}"

    @staticmethod
    def _is_low_information_question(question: str) -> bool:
        lowered = str(question or "").strip().lower()
        raw_tokens = re.findall(r"[a-z0-9_]+", lowered)
        meaningful_tokens = set(normalize_terms(lowered))
        if not raw_tokens:
            return True
        if lowered in {"what?", "what", "excuse me?", "excuse me", "sorry?", "sorry", "huh?", "huh"}:
            return True
        domain_terms = {
            "error", "warning", "csv", "aqme", "robert", "predict", "prediction",
            "predictions", "report", "reports", "descriptor", "descriptors", "model",
            "regression", "classification",
        }
        return len(raw_tokens) <= 2 and not bool(meaningful_tokens & domain_terms)

    @staticmethod
    def _should_force_popup_answer(question: str, snapshot: GuiSnapshot) -> bool:
        popup_available = bool(str(snapshot.popup_text or "").strip())
        if not popup_available:
            return False
        return bool(_POPUP_QUESTION_RE.search(str(question or "")))

    @staticmethod
    def _is_local_bad_request_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return "400" in text and "bad request" in text

    def _build_suggested_tutorial_summary(
        self,
        question: str,
        *,
        tutorial_id: str | None = None,
    ) -> TutorialSummary | None:
        resolved_id = self._resolve_tutorial_id(question, tutorial_id=tutorial_id)
        tutorial = _TUTORIALS_BY_ID.get(resolved_id)
        if tutorial is None:
            return None

        source, label = tutorial
        chunks = sorted(
            self._knowledge_base.chunks_for_source(source),
            key=self._tutorial_chunk_order,
        )
        if not chunks:
            return None

        section_points: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            sentences = self._sentences_from_tutorial_text(chunk.text)
            if not sentences:
                continue
            sentence = " ".join(sentences[:2])
            normalized = " ".join(sentence.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            section_points.append(sentence)
            if len(section_points) >= _TUTORIAL_OVERVIEW_LIMIT + _TUTORIAL_BULLET_LIMIT:
                break

        if not section_points:
            return None

        overview = section_points[0]
        bullets = tuple(section_points[1 : 1 + _TUTORIAL_BULLET_LIMIT]) or (overview,)
        candidates = tuple(
            SearchResult(chunk=chunk, score=float(len(chunks) - index))
            for index, chunk in enumerate(chunks)
        )
        return TutorialSummary(
            source=source,
            label=label,
            overview=overview,
            bullets=bullets,
            candidates=candidates,
        )

    @staticmethod
    def _render_tutorial_summary(summary: TutorialSummary) -> str:
        bullets = "\n".join(f"- {bullet}" for bullet in summary.bullets[:_TUTORIAL_BULLET_LIMIT])
        return (
            f"{summary.label} tutorial summary:\n"
            f"{summary.overview}\n\n"
            "Key points:\n"
            f"{bullets}\n\n"
            "If you want the full click-by-click tutorial, open the Tutorial panel "
            f"and select {summary.label}."
        )

    @staticmethod
    def _format_tutorial_summary_for_prompt(summary: TutorialSummary) -> str:
        bullet_lines = "\n".join(
            f"- {bullet}" for bullet in summary.bullets[:_TUTORIAL_BULLET_LIMIT]
        )
        return f"Purpose: {summary.overview}\nKey points:\n{bullet_lines}"

    @staticmethod
    def _tutorial_chunk_order(chunk: object) -> tuple[int, int | str]:
        chunk_id = str(getattr(chunk, "id", "") or "")
        match = re.search(r"-(\d+)$", chunk_id)
        if match:
            return 0, int(match.group(1))
        return 1, chunk_id

    @classmethod
    def _resolve_tutorial_id(cls, question: str, *, tutorial_id: str | None = None) -> str:
        explicit_id = tutorial_id.strip().lower() if type(tutorial_id) is str else ""
        if explicit_id in _TUTORIALS_BY_ID:
            return explicit_id

        normalized = cls._normalize_question_key(question)
        exact_id = _EXACT_TUTORIAL_QUESTIONS.get(normalized)
        if exact_id:
            return exact_id

        if "chemdraw" in normalized and re.search(
            r"\b(start|starting|begin|use|using|import|load|convert|from|guide)\b",
            normalized,
        ):
            return "chemdraw"
        if re.search(r"\b(?:i have|i already have|already loaded|tengo|ya tengo)\b.*\bcsv\b", normalized):
            # Existing data needs workflow guidance, not a CSV-creation summary.
            return ""
        if re.search(r"\bcsv\b", normalized) and re.search(
            r"\b(start|starting|begin|from|load|loading|use|using|guide)\b",
            normalized,
        ):
            return "csv"
        if re.search(r"\b(new|unknown|unseen)\b", normalized) and re.search(
            r"\b(predict|prediction|predictions|molecule|molecules)\b",
            normalized,
        ):
            return "predictions"
        if "overview" in normalized and re.search(r"\b(gui|easyrob|workflow|work)\b", normalized):
            return "overview"
        if "descriptor" in normalized and (
            re.search(r"\bwithout\b.*\b(train|training|run|running|build|building)\b.*\b(model|robert)\b", normalized)
            or re.search(r"\b(?:only|just)\b.*\b(?:generate|create|make)\b.*\bdescriptors?\b", normalized)
            or re.search(r"\bdescriptors?\s+(?:only|alone)\b", normalized)
            or re.search(r"\brun\s+aqme\s+(?:only|alone)\b", normalized)
        ):
            return "descriptors"
        return ""

    @staticmethod
    def _normalize_question_key(question: str) -> str:
        lowered = str(question or "").strip().lower()
        lowered = re.sub(r"[?.!]+$", "", lowered)
        return " ".join(lowered.split())

    @staticmethod
    def _sentences_from_tutorial_text(text: str) -> list[str]:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return []
        parts = [part.strip(" -") for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
        return [part for part in parts if len(part) > 20]

    @staticmethod
    def _infer_focus_tags(question: str, snapshot: GuiSnapshot) -> tuple[str, ...]:
        lowered = str(question or "").lower()
        focus: list[str] = []

        aqme_terms = (
            "aqme",
            "smiles",
            "descriptor generation",
            "descriptors from smiles",
            "qdescp",
            "csearch",
            "cmin",
            "qcorr",
            "qprep",
            "chemdraw",
        )
        robert_terms = (
            "robert",
            "curate",
            "generate",
            "verify",
            "predict",
            "report",
            "pfi",
            "rmse",
            "mae",
            "r squared",
            "r2",
            "machine learning",
            "target column",
            "main csv",
        )
        ui_terms = (
            "tab",
            "button",
            "gui",
            "interface",
            "settings",
            "popup",
            "window",
            "documentation",
            "tutorial",
            "dropdown",
            "checkbox",
        )

        if any(term in lowered for term in aqme_terms):
            focus.append("aqme")
        if any(term in lowered for term in robert_terms):
            focus.append("robert")
        if any(term in lowered for term in ui_terms):
            focus.append("ui")

        active_tab = str(getattr(snapshot, "active_tab", "") or "")
        if active_tab == "AQME":
            focus.extend(["aqme", "ui"])
        elif active_tab in {"ROBERT", "Advanced Options", "Results", "Reports", "Predictions", "Images"}:
            focus.extend(["robert", "ui"])

        if getattr(snapshot, "aqme_workflow_enabled", False) and "aqme" in lowered and "ui" not in focus:
            focus.append("ui")

        if not focus:
            focus.extend(["robert", "ui"])

        deduped: list[str] = []
        for tag in focus:
            if tag not in deduped:
                deduped.append(tag)
        return tuple(deduped)

    @staticmethod
    def _prioritize_candidates(candidates: list[object], question_intent: str) -> list[object]:
        if question_intent == "diagnostic":
            return candidates

        def sort_key(candidate: object) -> tuple[int]:
            chunk = getattr(candidate, "chunk", candidate)
            source = str(getattr(chunk, "source", "") or "")
            source_tier = str(getattr(chunk, "source_tier", "") or "").casefold()
            topic = str(getattr(chunk, "topic", "") or "").lower()
            tab = str(getattr(chunk, "tab", "") or "")

            if source_tier == "curated" or source == "curated_overrides":
                preferred = 0
            elif question_intent == "tutorial":
                preferred = 1 if source.startswith("tutorial:") or topic in {"tutorial", "concept"} else 2
            elif question_intent == "results":
                preferred_tabs = {"Results", "Reports", "Predictions", "Images"}
                preferred = 1 if tab in preferred_tabs or "result" in topic else 2
            elif question_intent == "parameter":
                preferred = 1 if topic == "parameter" else 2
            elif question_intent == "module-explanation":
                preferred = 1 if topic in {"concept", "tutorial"} else 2
            else:
                preferred = 1
            # `KnowledgeBase.search` already provides deterministic lexical order,
            # including exact curated-pattern precedence.  This stable grouping
            # must not re-rank candidates within an intent category.
            return (preferred,)

        return sorted(candidates, key=sort_key)
