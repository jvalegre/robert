"""Checks for request-local inspection of selected user CSV files."""

from gui_easyrob.bot.input_csv_audit import inspect_selected_csvs
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.llm_providers import ProviderAnswer, TokenUsage, build_llm_prompts
from types import SimpleNamespace


def test_selected_main_csv_reports_concrete_data_problems(tmp_path):
    main = tmp_path / "input.csv"
    main.write_text(
        "SMILES,code_name,target,charge,empty\n"
        "CCO,a,1,0.1,\n"
        "CCO,a,2,invalid,\n"
        "CCC,c,,0.3,\n",
        encoding="utf-8",
    )

    insights = inspect_selected_csvs(
        str(main), target_column="target", name_column="code_name", prediction_type="regression"
    )

    assert len(insights) == 1
    summary = insights[0].summary
    assert insights[0].source == "input.csv"
    assert "3 rows scanned" in summary
    assert "1 missing target value" in summary
    assert "1 nonnumeric value in descriptor charge" in summary
    assert "1 repeated exact SMILES" in summary
    assert "1 repeated code_name" in summary
    assert "empty: 3/3 missing" in summary


def test_external_csv_can_lack_target_but_missing_features_are_reported(tmp_path):
    main = tmp_path / "train.csv"
    external = tmp_path / "new.csv"
    main.write_text("SMILES,code_name,target,charge,dipole\nCCO,a,1,0.1,2\n", encoding="utf-8")
    external.write_text("SMILES,code_name,charge\nCCC,b,0.2\n", encoding="utf-8")

    insights = inspect_selected_csvs(
        str(main), str(external), target_column="target", name_column="code_name",
        prediction_type="regression",
    )

    assert "observed target column absent" in insights[1].summary
    assert "missing descriptor column: dipole" in insights[1].summary
    assert "target column missing" not in insights[1].summary


def test_malformed_rows_and_duplicate_headers_are_bounded_findings(tmp_path):
    main = tmp_path / "broken.csv"
    main.write_text("code_name,target,target\na,1,2\nb,3\n", encoding="utf-8")

    insight = inspect_selected_csvs(str(main), target_column="target")[0]

    assert "duplicate header: target" in insight.summary
    assert "1 row with a different column count" in insight.summary


def test_csv_scan_limit_is_explicit(tmp_path, monkeypatch):
    from gui_easyrob.bot import input_csv_audit

    monkeypatch.setattr(input_csv_audit, "_MAX_SCAN_ROWS", 2)
    main = tmp_path / "long.csv"
    main.write_text("target,desc\n1,1\n2,2\n3,3\n", encoding="utf-8")

    insight = inspect_selected_csvs(str(main), target_column="target")[0]

    assert "2 rows scanned" in insight.summary
    assert "scan limited to first 2 rows" in insight.summary


def test_unreadable_external_csv_does_not_generate_schema_claims(tmp_path):
    main = tmp_path / "train.csv"
    main.write_text("code_name,target,charge\na,1,0.1\n", encoding="utf-8")

    insights = inspect_selected_csvs(str(main), str(tmp_path / "missing.csv"), target_column="target")

    assert "not readable" in insights[1].summary
    assert "missing descriptor column" not in insights[1].summary


def test_semicolon_csv_and_empty_csv_have_explicit_profiles(tmp_path):
    semicolon = tmp_path / "semicolon.csv"
    empty = tmp_path / "empty.csv"
    semicolon.write_text("code_name;target;charge\na;1;0.1\n", encoding="utf-8")
    empty.write_text("", encoding="utf-8")

    assert "1 rows scanned, 3 columns" in inspect_selected_csvs(str(semicolon), target_column="target")[0].summary
    assert "is empty" in inspect_selected_csvs(str(empty))[0].summary


def _gui_snapshot(csv_path):
    return GuiSnapshot(
        "Input", "Full Workflow", str(csv_path), "", "target", "regression",
        False, True, False, "", [], name_column="code_name",
    )


def test_heuristic_bot_inspects_selected_csv_before_workflow(tmp_path):
    main = tmp_path / "input.csv"
    main.write_text("code_name,target,charge\na,1,0.1\nb,,invalid\n", encoding="utf-8")
    engine = BotEngine(knowledge_base=KnowledgeBase([]))

    answer = engine.answer("¿Qué problemas hay en mi CSV?", _gui_snapshot(main), "Heuristic", "", "")

    assert "1 missing target value" in answer
    assert "1 nonnumeric value in descriptor charge" in answer
    assert "input.csv" in answer
    assert "Workflow result" not in answer


def test_cloud_bot_receives_selected_csv_facts_without_workflow(tmp_path):
    class RecordingProvider:
        spec = SimpleNamespace(supports_web_search=True)

        def __init__(self):
            self.calls = []

        def generate_answer_with_usage(self, **kwargs):
            self.calls.append(kwargs)
            return ProviderAnswer("The selected CSV has a missing target value.", TokenUsage())

    main = tmp_path / "input.csv"
    main.write_text("code_name,target,charge\na,1,0.1\nb,,0.2\n", encoding="utf-8")
    provider = RecordingProvider()
    engine = BotEngine(knowledge_base=KnowledgeBase([]), provider_registry={"Test": provider})

    engine.answer_with_metadata("Analiza mi CSV antes de entrenar", _gui_snapshot(main), "Cloud AI", "Test", "test-key")

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["question_intent"] == "results"
    assert call["gui_context"].workflow_results.kind == "INPUT CSV"
    system_prompt, user_prompt = build_llm_prompts(
        "Analiza mi CSV antes de entrenar", call["gui_context"], [], question_intent="results"
    )
    assert "1 missing target value" in user_prompt
    assert "before a workflow run" in system_prompt


def test_generic_question_does_not_expose_selected_csv(tmp_path):
    main = tmp_path / "input.csv"
    main.write_text("code_name,target\na,\n", encoding="utf-8")
    engine = BotEngine(knowledge_base=KnowledgeBase([]))

    answer = engine.answer("What is RMSE?", _gui_snapshot(main), "Heuristic", "", "")

    assert "missing target" not in answer
    assert "input.csv" not in answer
