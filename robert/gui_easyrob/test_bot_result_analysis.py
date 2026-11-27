"""Regression tests for grounded analysis of user workflow results."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gui_easyrob.bot.result_insights import (
    extract_aqme_insight,
    extract_dataset_overlap,
    extract_log_insights,
    extract_prediction_insight,
)
from gui_easyrob.bot.workflow_results import (
    WorkflowResultStore,
    render_workflow_answer,
    render_workflow_evidence,
    scope_workflow_result_snapshot,
)
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.result_insights import ResultInsight
from gui_easyrob.bot.workflow_results import WorkflowResultSnapshot
from gui_easyrob.bot.workflow_results import WorkflowReportAssessment
from gui_easyrob.bot.workflow_results import WorkflowStageResult
from gui_easyrob.bot.prompt_policy import _is_workflow_result_question
from gui_easyrob.bot.question_intent import is_generic_question
from gui_easyrob.bot.llm_providers import ProviderAnswer, TokenUsage, _response_language


def test_log_insights_keep_feature_values_and_variants_separate():
    log = """------- Starting model with all variables (No PFI) -------
   o  SHAP plot saved in PREDICT/SHAP_GB_No_PFI.png
      -  Fukui+ = min: -2.1, max: 2.6
   o  PFI plot saved in PREDICT/PFI_GB_No_PFI.png
      Influence on RMSE
      -  Fukui+ = 1.9 +- 0.21
------- Starting model with PFI filter (only important descriptors used) -------
   o  PFI plot saved in PREDICT/PFI_GB_PFI.png
      Influence on RMSE
      -  Dipole = 0.8 +- 0.10
"""

    insights = extract_log_insights(log, "GB", "PREDICT/PREDICT_GB_data.dat")

    assert [(item.variant, item.category) for item in insights] == [
        ("No_PFI", "shap"), ("No_PFI", "pfi"), ("PFI", "pfi")
    ]
    assert "Fukui+" in insights[0].summary and "-2.1" in insights[0].summary
    assert "1.9 ± 0.21" in insights[1].summary
    assert "for RMSE" in insights[1].summary
    assert "Dipole" not in insights[1].summary


def test_prediction_insight_reports_observed_errors_and_missing_targets(tmp_path):
    path = tmp_path / "GB_No_PFI.csv"
    path.write_text(
        "code_name,target,target_pred,target_pred_sd,Set\n"
        "acid-a,1,2,0.4,Test\n"
        "acid-b,3,1,0.5,Test\n"
        "acid-c,,4,0.6,Test\n"
        "acid-d,4,nan,0.3,Test\n",
        encoding="utf-8",
    )

    insight = extract_prediction_insight(path, "GB", "No_PFI", "PREDICT/GB_No_PFI.csv")

    assert insight.model == "GB" and insight.variant == "No_PFI"
    assert "2 evaluated" in insight.summary
    assert "MAE=1.5" in insight.summary
    assert "acid-b" in insight.summary and "absolute error=2" in insight.summary
    assert "2 rows without comparable observed/predicted values" in insight.summary
    assert "acid-c" not in insight.summary


def test_prediction_insight_without_observations_does_not_invent_errors(tmp_path):
    path = tmp_path / "new_acids.csv"
    path.write_text(
        "code_name,target_pred,target_pred_sd\nnew-a,2.0,0.4\nnew-b,5.0,0.7\n",
        encoding="utf-8",
    )

    insight = extract_prediction_insight(path, "GB", "PFI", "PREDICT/csv_test/new_acids.csv")

    assert "2 finite predictions" in insight.summary
    assert "predicted value range=2 to 5" in insight.summary
    assert "observed target column absent" in insight.summary
    assert "MAE" not in insight.summary


def test_workflow_evidence_scopes_feature_and_error_insights(tmp_path):
    input_path = tmp_path / "acids.csv"
    input_path.write_text("code_name,target\na,1\n", encoding="utf-8")
    predict_dir = tmp_path / "PREDICT"
    predict_dir.mkdir()
    for model, feature in (("GB", "Fukui+"), ("RF", "Dipole")):
        (predict_dir / f"PREDICT_{model}_data.dat").write_text(
            "------- Starting model with all variables (No PFI) -------\n"
            f"   o  PFI plot saved in PREDICT/PFI_{model}_No_PFI.png\n"
            "      Influence on RMSE\n"
            f"      -  {feature} = 1.9 +- 0.21\n"
            "      -  Test : R2 = 0.5, RMSE = 1.2\n",
            encoding="utf-8",
        )
        (predict_dir / f"{model}_No_PFI.csv").write_text(
            "code_name,target,target_pred,Set\n"
            f"{model.lower()}-acid,1,3,Test\n",
            encoding="utf-8",
        )

    snapshot = WorkflowResultStore().refresh(str(input_path), "")
    scoped = scope_workflow_result_snapshot(
        snapshot, "GB No_PFI descriptor importance and errors",
        available_models=("GB", "RF"),
    )
    evidence = render_workflow_evidence(scoped, "GB No_PFI descriptor importance and errors", max_chars=5000)
    answer = render_workflow_answer(scoped, "GB No_PFI descriptor importance and errors")

    assert len(snapshot.insights) == 4
    assert len(scoped.insights) == 2
    assert "Fukui+" in evidence and "gb-acid" in evidence
    assert "Dipole" not in evidence and "rf-acid" not in evidence
    assert "Fukui+" in answer and "gb-acid" in answer


def test_external_prediction_question_excludes_internal_rows(tmp_path):
    input_path = tmp_path / "acids.csv"
    input_path.write_text("code_name,target\na,1\n", encoding="utf-8")
    predict_dir = tmp_path / "PREDICT"
    external_dir = predict_dir / "csv_test"
    external_dir.mkdir(parents=True)
    (predict_dir / "PREDICT_GB_data.dat").write_text(
        "------- Starting model with all variables (No PFI) -------\n"
        "      -  Test : R2 = 0.5, RMSE = 1.2\n"
        "   o  SHAP plot saved in PREDICT/SHAP_GB_No_PFI.png\n"
        "      -  Fukui+ = min: -2.1, max: 2.6\n",
        encoding="utf-8",
    )
    (predict_dir / "GB_No_PFI.csv").write_text(
        "code_name,target,target_pred,Set\ninternal-acid,1,3,Test\n", encoding="utf-8"
    )
    (external_dir / "input_GB_No_PFI.csv").write_text(
        "code_name,target,target_pred\nexternal-acid,1,4\n", encoding="utf-8"
    )

    snapshot = WorkflowResultStore().refresh(str(input_path), "")
    question = "GB No_PFI: analiza las predicciones externas"
    evidence = render_workflow_evidence(snapshot, question)
    answer = render_workflow_answer(snapshot, question)

    assert "external-acid" in evidence and "internal-acid" not in evidence
    assert "external-acid" in answer and "internal-acid" not in answer
    assert "Fukui+" not in evidence and "Fukui+" not in answer


def test_aqme_insight_counts_missing_descriptor_values(tmp_path):
    path = tmp_path / "AQME-ROBERT_full_acids.csv"
    path.write_text(
        "SMILES,code_name,target,charge,dipole\n"
        "CCO,a,1,0.1,2.0\n"
        "CCC,b,2,,nan\n"
        "CCN,c,3,0.2,3.0\n",
        encoding="utf-8",
    )

    insight = extract_aqme_insight(path, 3, path.name)

    assert insight.category == "aqme_completeness"
    assert "3 rows" in insight.summary
    assert "2 descriptor columns" in insight.summary
    assert "charge: 1/3 missing" in insight.summary
    assert "dipole: 1/3 missing" in insight.summary


def test_aqme_question_prioritizes_original_descriptor_output(tmp_path):
    input_path = tmp_path / "acids.csv"
    input_path.write_text("code_name,target\na,1\n", encoding="utf-8")
    (tmp_path / "AQME-ROBERT_full_acids.csv").write_text(
        "code_name,target,charge\na,1,\n", encoding="utf-8"
    )
    predict_dir = tmp_path / "PREDICT"
    predict_dir.mkdir()
    (predict_dir / "PREDICT_GB_data.dat").write_text(
        "------- Starting model with all variables (No PFI) -------\n"
        "   o  SHAP plot saved in PREDICT/SHAP_GB_No_PFI.png\n"
        "      -  Fukui+ = min: -2.1, max: 2.6\n",
        encoding="utf-8",
    )
    (predict_dir / "csv_test").mkdir()
    (predict_dir / "csv_test" / "AQME-ROBERT_full_acids_GB_No_PFI.csv").write_text(
        "code_name,target,target_pred\na,1,2\n", encoding="utf-8"
    )

    snapshot = WorkflowResultStore().refresh(str(input_path), "")
    evidence = render_workflow_evidence(snapshot, "Que valores AQME faltan en los descriptores?", max_chars=1200)

    assert len([item for item in snapshot.insights if item.category == "aqme_completeness"]) == 1
    assert "charge: 1/1 missing" in evidence
    assert "Fukui+" not in evidence


def test_identical_train_and_external_smiles_are_flagged(tmp_path):
    train = tmp_path / "acids.csv"
    external = tmp_path / "error.csv"
    train.write_text("SMILES,code_name,target\nCCO,a,1\nCCC,b,2\n", encoding="utf-8")
    external.write_text("SMILES,code_name,target\nCCO,x,9\nCCC,y,8\n", encoding="utf-8")
    (tmp_path / "PREDICT").mkdir()
    (tmp_path / "PREDICT" / "PREDICT_data.dat").write_text("PREDICT completed\n", encoding="utf-8")

    insight = extract_dataset_overlap(train, external)
    snapshot = WorkflowResultStore().refresh(str(train), str(external))
    evidence = render_workflow_evidence(snapshot, "Are external predictions independent?", max_chars=1500)

    assert "2/2 external rows" in insight.summary
    assert "not independent by exact SMILES" in insight.summary
    assert "2/2 external rows" in evidence


def test_heuristic_bot_routes_feature_and_overlap_questions_to_user_results():
    workflow = WorkflowResultSnapshot(
        kind="AQME+ROBERT", status="completed", root_name="acids", summary_available=True,
        insights=(
            ResultInsight("GB", "No_PFI", "shap", "Fukui+: min=-2.1, max=2.6", "PREDICT/PREDICT_GB_data.dat"),
            ResultInsight("", "", "dataset_overlap", "19/19 external rows match the main input", "acids.csv, error.csv"),
        ),
    )
    snapshot = GuiSnapshot(
        "Results", "Full Workflow", "acids.csv", "error.csv", "target", "regression",
        False, True, False, "", [], workflow_results=workflow,
    )
    engine = BotEngine(knowledge_base=KnowledgeBase([]))

    shap_answer = engine.answer("¿Qué muestra SHAP para GB?", snapshot, "Heuristic", "", "")
    overlap_answer = engine.answer("¿Hay solapamiento entre mis CSV?", snapshot, "Heuristic", "", "")

    assert "Fukui+" in shap_answer
    assert "19/19 external rows" in overlap_answer


def test_cloud_result_questions_keep_workflow_context():
    assert _is_workflow_result_question("¿Qué muestra SHAP para GB?")
    assert _is_workflow_result_question("¿Hay solapamiento entre mis CSV?")
    assert not is_generic_question("¿Qué valores AQME faltan en los descriptores?")


def test_cloud_bot_marks_shap_question_as_local_results():
    class RecordingProvider:
        spec = SimpleNamespace(supports_web_search=True)

        def __init__(self):
            self.calls = []

        def generate_answer_with_usage(self, **kwargs):
            self.calls.append(kwargs)
            return ProviderAnswer("The SHAP values are in the workflow files.", TokenUsage())

    provider = RecordingProvider()
    workflow = WorkflowResultSnapshot(
        kind="ROBERT", status="completed", root_name="acids", summary_available=True,
        insights=(ResultInsight("GB", "No_PFI", "shap", "Fukui+: min=-2.1, max=2.6", "PREDICT/PREDICT_GB_data.dat"),),
    )
    snapshot = GuiSnapshot(
        "Results", "Full Workflow", "acids.csv", "", "target", "regression",
        False, True, False, "", [], workflow_results=workflow,
    )
    engine = BotEngine(knowledge_base=KnowledgeBase([]), provider_registry={"Test": provider})

    engine.answer_with_metadata("¿Qué muestra SHAP para GB?", snapshot, "Cloud AI", "Test", "test-key")

    assert len(provider.calls) == 1
    assert provider.calls[0]["question_intent"] == "results"


def test_spanish_scientific_result_question_keeps_spanish_response_language():
    question = (
        "Con GB No_PFI, analiza el test externo de acids: "
        "¿es independiente del conjunto principal? Da los errores observados más grandes "
        "y cita los archivos de origen. No inventes puntuaciones."
    )

    assert _response_language(question) == "Spanish"


def test_result_discovery_survives_optional_report_scoring_error(tmp_path, monkeypatch):
    from gui_easyrob.bot import workflow_results

    input_path = tmp_path / "input.csv"
    input_path.write_text("target,descriptor\n1,2\n", encoding="utf-8")
    predict_dir = tmp_path / "PREDICT"
    verify_dir = tmp_path / "VERIFY"
    predict_dir.mkdir()
    verify_dir.mkdir()
    (predict_dir / "PREDICT_GB_data.dat").write_text(
        "------- Starting model with all variables (No PFI) -------\n"
        "      -  Test : R2 = 0.5, RMSE = 1.2\n",
        encoding="utf-8",
    )
    (verify_dir / "VERIFY_GB_data.dat").write_text(
        "------- Starting model with all variables (No PFI) -------\n"
        "      Original RMSE (10x 5-fold CV) 1.2 + 15% & 30% threshold = 1.4 & 1.6\n",
        encoding="utf-8",
    )

    def broken_score(*_args):
        raise UnboundLocalError("legacy score field missing")

    monkeypatch.setattr(workflow_results, "_report_score_function", lambda: broken_score)

    snapshot = WorkflowResultStore().refresh(str(input_path), "")

    assert snapshot is not None
    assert len(snapshot.report_assessments) == 1
    assert snapshot.report_assessments[0].interpolation_score is None
    assert snapshot.report_assessments[0].score_unavailable_reason == "scoring_error"


def test_external_error_evidence_prioritizes_external_metric_over_report_score():
    snapshot = WorkflowResultSnapshot(
        kind="ROBERT", status="completed", root_name="acids", summary_available=True,
        metrics=(("RMSE", "2.3", "External test (No_PFI)", "PREDICT/PREDICT_GB_data.dat"),),
        report_assessments=(WorkflowReportAssessment(
            variant="No PFI", model="GB", prediction_type="regression",
            interpolation_score=None, boundary_score=None, score_unavailable_reason="incomplete_log",
        ),),
        insights=(ResultInsight(
            "GB", "No_PFI", "prediction_errors", "19 evaluated; MAE=1.65", "PREDICT/csv_test/input_GB_No_PFI.csv"
        ),),
    )

    evidence = render_workflow_evidence(
        snapshot, "GB No_PFI: analiza los errores del test externo", max_chars=1500
    )

    assert "Metric External test (No_PFI) RMSE=2.3" in evidence
    assert "Report-derived assessment" not in evidence


def test_focused_prediction_evidence_omits_unrelated_stage_details():
    snapshot = WorkflowResultSnapshot(
        kind="AQME+ROBERT", status="completed", root_name="acids", summary_available=True,
        stages=tuple(WorkflowStageResult(
            name=name, status="completed", source=f"{name}/{name}_data.dat"
        ) for name in ("CURATE", "GENERATE", "VERIFY", "PREDICT", "AQME", "CSEARCH", "QDESCP")),
        metrics=(("RMSE", "2.3", "External test (No_PFI)", "PREDICT/PREDICT_GB_data.dat"),),
        insights=(ResultInsight(
            "GB", "No_PFI", "prediction_errors", "19 evaluated; MAE=1.65", "PREDICT/csv_test/input_GB_No_PFI.csv"
        ),),
    )

    evidence = render_workflow_evidence(snapshot, "GB No_PFI: analiza los errores externos", max_chars=1000)

    assert "Stage CURATE" not in evidence
    assert "Stage PREDICT" in evidence
    assert "Metric External test (No_PFI) RMSE=2.3" in evidence
