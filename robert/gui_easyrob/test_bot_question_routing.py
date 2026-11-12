"""Regression tests for routing user-specific model questions."""

from types import SimpleNamespace

import pytest

from gui_easyrob.bot.answer_metadata import EvidenceOrigin, WebSearchMode
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.llm_providers import ProviderAnswer, build_llm_prompts
from gui_easyrob.bot.question_intent import (
    classify_question_intent,
    is_follow_up_question,
    is_generic_question,
    select_inference_history,
)
from gui_easyrob.bot.token_usage import TokenUsage
from gui_easyrob.bot.workflow_results import (
    WorkflowReportAssessment,
    WorkflowResultSnapshot,
    render_workflow_answer,
)


@pytest.mark.parametrize(
    "question",
    (
        "¿Cuál de mis modelos es mejor?",
        "¿Cuál es el mejor modelo de mis resultados?",
        "¿Puedo confiar en este modelo?",
        "Which of my models is best?",
    ),
)
def test_personal_model_questions_use_results_context(question):
    assert classify_question_intent(question) == "results"
    assert not is_generic_question(question)


def test_general_model_definition_does_not_use_current_results():
    assert is_generic_question("What is a machine learning model?")


def test_generic_metric_definition_omits_current_workflow_evidence():
    question = "What is RMSE?"
    assert is_generic_question(question)

    system_prompt, user_prompt = build_llm_prompts(
        question, _result_snapshot(), (), question_intent="results",
    )

    assert "<WORKFLOW_RESULTS_DATA>" not in user_prompt
    assert "only from supplied workflow evidence" not in system_prompt


@pytest.mark.parametrize("question", ("What was the test R2?", "¿Cuál fue el RMSE del test?"))
def test_past_run_metric_question_uses_current_results(question):
    assert not is_generic_question(question)


def test_personal_model_question_includes_current_workflow_evidence():
    _, user_prompt = build_llm_prompts(
        "¿Cuál de mis modelos es mejor?", _result_snapshot(), (), question_intent="results",
    )

    assert "RMSE=4.2" in user_prompt


@pytest.mark.parametrize(
    "question",
    ("¿Qué es un modelo de ML?", "¿Cómo entreno un modelo en EasyROB?"),
)
def test_general_spanish_model_questions_are_not_result_questions(question):
    assert classify_question_intent(question) == "tutorial"


def test_spanish_explanation_request_is_a_follow_up():
    assert is_follow_up_question("¿Puedes explicarlo más?")


def test_long_demonstrative_question_is_a_follow_up():
    assert is_follow_up_question("Why does that matter for my model?")


def test_follow_up_chain_retains_original_question_for_retrieval():
    history = [
        {"role": "user", "content": "What does VERIFY check in ROBERT?"},
        {"role": "assistant", "content": "VERIFY checks predictive ability."},
        {"role": "user", "content": "Why is that important?"},
        {"role": "assistant", "content": "It gives validation evidence."},
        {"role": "user", "content": "Can you explain it further?"},
        {"role": "assistant", "content": "It checks whether performance is robust."},
    ]

    selected = select_inference_history("And what should I check next?", history)

    assert selected[0]["content"] == "What does VERIFY check in ROBERT?"


def test_follow_up_does_not_cross_an_ineligible_exchange():
    history = [
        {"role": "user", "content": "What does VERIFY check in ROBERT?"},
        {"role": "assistant", "content": "VERIFY checks predictive ability."},
        {"role": "user", "content": "What is the RMSE of my model?"},
        {"role": "assistant", "content": "No result is available.", "memory_eligible": False},
    ]

    assert select_inference_history("Why is that important?", history) == ()


@pytest.mark.parametrize(
    "question",
    ("And what does CSEARCH do in AQME?", "¿Y qué hace CSEARCH en AQME?"),
)
def test_explicit_new_subject_does_not_inherit_previous_module(question):
    history = [
        {"role": "user", "content": "What does VERIFY check in ROBERT?"},
        {"role": "assistant", "content": "VERIFY checks predictive ability."},
    ]

    assert not is_follow_up_question(question)
    assert select_inference_history(question, history) == ()


@pytest.mark.parametrize("question", ("Why is that important?", "¿Y por qué importa?"))
def test_verify_explanation_follow_up_answers_the_module(question):
    history = [
        {"role": "user", "content": "What does VERIFY check in ROBERT?"},
        {"role": "assistant", "content": "VERIFY checks predictive ability."},
    ]
    snapshot = GuiSnapshot(
        "ROBERT", "Full Workflow", "example.csv", "", "yield", "regression",
        False, True, False, "", [],
    )

    answer = BotEngine().answer(
        question, snapshot, "Heuristic", "Test", "", conversation_history=history,
    )

    assert "VERIFY" in answer
    assert "robustness" in answer


def test_spanish_verify_chain_does_not_answer_about_the_target_column():
    history = [
        {"role": "user", "content": "¿Qué comprueba VERIFY en ROBERT?"},
        {"role": "assistant", "content": "VERIFY comprueba la capacidad predictiva."},
        {"role": "user", "content": "¿Y por qué importa?"},
        {"role": "assistant", "content": "Aporta evidencia de validación."},
        {"role": "user", "content": "¿Puedes explicarlo más?"},
        {"role": "assistant", "content": "Revisa si el rendimiento es robusto."},
    ]
    snapshot = GuiSnapshot(
        "ROBERT", "Full Workflow", "example.csv", "", "yield", "regression",
        False, True, False, "", [],
    )

    answer = BotEngine().answer(
        "¿Y qué reviso después?", snapshot, "Heuristic", "Test", "",
        conversation_history=history,
    )

    assert "VERIFY" in answer
    assert "target" not in answer.lower()


def test_error_follow_up_keeps_diagnostic_intent():
    history = (
        {"role": "user", "content": "What does VERIFY check in ROBERT?"},
        {"role": "assistant", "content": "VERIFY checks predictive ability."},
    )

    assert classify_question_intent("Why is that error important?", history) == "diagnostic"


class _RecordingProvider:
    spec = SimpleNamespace(supports_web_search=True)

    def __init__(self):
        self.search_modes = []

    def generate_answer_with_usage(self, **kwargs):
        self.search_modes.append(kwargs["web_search_mode"])
        return ProviderAnswer("The recorded metric needs context.", TokenUsage())


def _result_snapshot():
    workflow = WorkflowResultSnapshot(
        kind="ROBERT",
        status="completed",
        root_name="example",
        summary_available=True,
        metrics=(("RMSE", "4.2", "Test", "PREDICT/PREDICT_data.dat"),),
    )
    return GuiSnapshot(
        "Results", "Full Workflow", "example.csv", "", "yield", "regression",
        False, True, False, "", [], workflow_results=workflow,
    )


def test_local_model_result_question_does_not_trigger_optional_web_search():
    provider = _RecordingProvider()
    engine = BotEngine(knowledge_base=KnowledgeBase([]), provider_registry={"Test": provider})

    result = engine.answer_with_metadata(
        "¿Cuál de mis modelos es mejor?", _result_snapshot(), "Cloud AI", "Test", "test-key",
    )

    assert provider.search_modes == [WebSearchMode.OFF]
    assert result.metadata.evidence_origin == EvidenceOrigin.LOCAL_SYSTEM


def test_generic_metric_definition_is_not_labeled_as_local_result():
    provider = _RecordingProvider()
    engine = BotEngine(knowledge_base=KnowledgeBase([]), provider_registry={"Test": provider})

    result = engine.answer_with_metadata(
        "What is RMSE?", _result_snapshot(), "Cloud AI", "Test", "test-key",
    )

    assert provider.search_modes == [WebSearchMode.AUTO]
    assert result.metadata.evidence_origin == EvidenceOrigin.GENERAL_KNOWLEDGE


def test_explicit_web_request_keeps_web_search_for_model_question():
    provider = _RecordingProvider()
    engine = BotEngine(knowledge_base=KnowledgeBase([]), provider_registry={"Test": provider})

    engine.answer_with_metadata(
        "Busca en internet si este modelo es fiable", _result_snapshot(),
        "Cloud AI", "Test", "test-key",
    )

    assert provider.search_modes == [WebSearchMode.REQUIRED]


@pytest.mark.parametrize("question", ("Can I trust this model?", "¿Puedo confiar en este modelo?"))
def test_model_trust_question_explains_limits_of_recorded_scores(question):
    workflow = WorkflowResultSnapshot(
        kind="ROBERT",
        status="completed",
        root_name="example",
        summary_available=True,
        report_assessments=(WorkflowReportAssessment(
            variant="PFI", model="RF", prediction_type="regression",
            interpolation_score=7, boundary_score=5,
        ),),
    )

    answer = render_workflow_answer(workflow, question)

    assert "Prediction suitability" in answer
    assert "acceptable for your use" in answer


def test_metric_followups_keep_the_original_topic_and_answer_the_new_question():
    engine = BotEngine()
    snapshot = GuiSnapshot(
        "ROBERT", "Full Workflow", "example.csv", "", "yield", "regression",
        False, True, False, "", [],
    )
    first_question = "What does RMSE mean?"
    first_answer = engine.answer(first_question, snapshot, "Heuristic", "Test", "")
    history = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first_answer},
    ]

    second_question = "Is a lower value always better?"
    second_answer = engine.answer(
        second_question, snapshot, "Heuristic", "Test", "", conversation_history=history,
    )
    assert "RMSE" in second_answer
    assert "lower" in second_answer.lower()
    assert "same target" in second_answer.lower()
    assert "QCORR" not in second_answer

    history.extend((
        {"role": "user", "content": second_question},
        {"role": "assistant", "content": second_answer},
    ))
    example_answer = engine.answer(
        "Explain it with a short example.", snapshot, "Heuristic", "Test", "",
        conversation_history=history,
    )
    assert "RMSE" in example_answer
    assert "5" in example_answer
    assert "percentage points" in example_answer


def test_named_metric_comparison_does_not_inherit_an_unrelated_topic():
    assert not is_follow_up_question("Is a lower RMSE value always better?")
