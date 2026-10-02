"""Representative bot regressions chosen by measured line coverage.

Each source group has its own namespace so its helpers cannot shadow those
from another group. The exported checks remain ordinary pytest tests.
"""

import sys
from types import ModuleType


def _load_case_group(label, source, checks, fixtures):
    module = ModuleType(f"{__name__}.{label}")
    module.__file__ = __file__
    sys.modules[module.__name__] = module
    exec(compile(source, __file__, "exec"), module.__dict__)
    for name in fixtures:
        globals()[name] = module.__dict__[name]
    for name in checks:
        if name.startswith("test_"):
            public_name = f"test_{label}_{name[5:]}"
        else:
            public_name = f"Test{label.title()}{name[4:]}"
        globals()[public_name] = module.__dict__[name]

# Selected checks from test_bot_context.py
_load_case_group('context', """import dataclasses
import math
import unittest
from types import SimpleNamespace
from gui_easyrob.bot.bot_context import POPUP_CONTEXT_TTL_SECONDS, GuiSnapshot, _maybe_call, build_gui_snapshot
from gui_easyrob.bot.workflow_results import WorkflowResultSnapshot

class TestBotContext(unittest.TestCase):

    @staticmethod
    def _popup_window(*, active=None, closed=None, age=None):
        return SimpleNamespace(tab_widget=SimpleNamespace(tabText=lambda index: 'ROBERT', currentIndex=lambda: 0), workflow_selector=SimpleNamespace(currentText=lambda: 'Full Workflow'), y_dropdown=SimpleNamespace(currentText=lambda: 'yield'), type_dropdown=SimpleNamespace(currentText=lambda: 'Regression'), run_button=SimpleNamespace(isEnabled=lambda: True), stop_button=SimpleNamespace(isEnabled=lambda: False), console_output=SimpleNamespace(toPlainText=lambda: ''), bot_active_popup=active, bot_popup_context=closed, bot_popup_context_age_seconds=age)

    def test_build_gui_snapshot_extracts_structured_python_failure_fields(self):
        fake_window = SimpleNamespace(file_path='train.csv', csv_test_path='', current_process='idle', tab_widget=SimpleNamespace(tabText=lambda index: 'ROBERT', currentIndex=lambda: 0), workflow_selector=SimpleNamespace(currentText=lambda: 'Full Workflow'), y_dropdown=SimpleNamespace(currentText=lambda: 'yield'), type_dropdown=SimpleNamespace(currentText=lambda: 'Regression'), run_button=SimpleNamespace(isEnabled=lambda: True), stop_button=SimpleNamespace(isEnabled=lambda: False), console_output=SimpleNamespace(toPlainText=lambda: 'File "utils.py", line 743, in correlation_filter\\nres_y = stats.linregress(csv_df_filtered[col], csv_df_filtered[self.args.y])\\nTypeError: ufunc \\'divide\\' not supported for the input types'), bot_active_popup=None, bot_popup_context=None, bot_popup_context_age_seconds=None)
        snapshot = build_gui_snapshot(fake_window)
        self.assertEqual(snapshot.failure_type, 'TypeError')
        self.assertIn('divide', snapshot.failure_message)
        self.assertEqual(snapshot.failure_location, 'utils.py:743')
        self.assertEqual(snapshot.failure_function, 'correlation_filter')
        self.assertIn('stats.linregress', snapshot.failure_operation)""", ('TestBotContext',), ())

# Selected checks from test_bot_engine.py
_load_case_group('engine', """import dataclasses
import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from types import SimpleNamespace
from gui_easyrob.bot.answer_metadata import AnswerMetadata, EvidenceOrigin, SourceCitation, WebSearchMode
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.evidence_policy import EvidenceAssessment, EvidenceDecision
from gui_easyrob.bot.llm_providers import LLMProviderError, LLMProviderErrorKind, ProviderAnswer
from gui_easyrob.bot.local_llm import LocalLLMError
from gui_easyrob.bot.bot_rag import KnowledgeBase, SearchResult
from gui_easyrob.bot.token_usage import TokenUsage
from gui_easyrob.bot.workflow_results import WorkflowArtifact, WorkflowResultSnapshot, WorkflowStageResult

def _candidate(record_id, *, score, source='reference', source_tier='', topic='concept', tab='ROBERT'):
    chunk = KnowledgeBase.from_records([{'id': record_id, 'text': f'Knowledge for {record_id}.', 'source': source, 'source_tier': source_tier, 'topic': topic, 'tab': tab}])._chunks[0]
    return SearchResult(chunk=chunk, score=score)

class FakeProvider:

    def __init__(self, response='stubbed llm answer'):
        self.response = response
        self.calls = []

    def generate_answer(self, question, gui_context, retrieved_chunks, api_key, model=None, conversation_history=None, question_intent=None, tutorial_summary=None, tutorial_label=None):
        self.calls.append({'question': question, 'gui_context': gui_context, 'retrieved_chunks': list(retrieved_chunks), 'api_key': api_key, 'model': model, 'conversation_history': list(conversation_history or []), 'question_intent': question_intent, 'tutorial_summary': tutorial_summary, 'tutorial_label': tutorial_label})
        return self.response

class UsageProvider(FakeProvider):

    def generate_answer_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderAnswer(text=self.response, usage=TokenUsage(100, 20, 120, estimated_cost_usd=2.7e-05, provider='OpenAI'))

class FailingUsageProvider(UsageProvider):

    def generate_answer_with_usage(self, **kwargs):
        raise LLMProviderError(LLMProviderErrorKind.INVALID_REQUEST, 'Provider unavailable')

class RevisingUsageProvider(UsageProvider):

    def generate_answer_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        answer = 'The raw test RMSE is not available.' if len(self.calls) == 1 else 'The recorded raw test RMSE is 0.15; compare it with an acceptable error for your use.'
        return ProviderAnswer(text=answer, usage=TokenUsage(100, 20, 120, provider='OpenAI'))

class SearchRecordingProvider(UsageProvider):

    def __init__(self, *, search_requests=0, sources=()):
        super().__init__(response='Grounded provider answer')
        self.spec = SimpleNamespace(supports_web_search=True)
        self.search_requests = search_requests
        self.sources = tuple(sources)

    def generate_answer_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderAnswer(text=self.response, usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, search_requests=self.search_requests, provider='OpenAI', model='gpt-4.1-mini'), metadata=AnswerMetadata(evidence_origin=kwargs.get('evidence_origin', EvidenceOrigin.GENERAL_KNOWLEDGE), sources=self.sources))

class OversizedOptionalSearchProvider(UsageProvider):

    def __init__(self):
        super().__init__(response='Fallback answer without web search.')
        self.spec = SimpleNamespace(supports_web_search=True)

    def generate_answer_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get('web_search_mode') is not WebSearchMode.OFF:
            raise LLMProviderError(LLMProviderErrorKind.INVALID_REQUEST, 'The AI provider rejected this request. Detail: Request Entity Too Large', status_code=413)
        return ProviderAnswer(text=self.response, usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, estimated_cost_usd=2.7e-05, provider='OpenAI', model='gpt-4.1-mini'), metadata=AnswerMetadata(evidence_origin=EvidenceOrigin.GENERAL_KNOWLEDGE))

class FakeLocalManager:

    def __init__(self, response='stubbed local answer', errors=None):
        self.response = response
        self.errors = list(errors or [])
        self.calls = []

    def generate_response(self, system_prompt, user_prompt):
        self.calls.append({'system_prompt': system_prompt, 'user_prompt': user_prompt})
        if self.errors:
            raise self.errors.pop(0)
        return self.response

class TestBotEngine(unittest.TestCase):

    def test_result_summary_retries_evidence_contradiction_once(self):
        provider = RevisingUsageProvider()
        snapshot = dataclasses.replace(self.snapshot, workflow_results=WorkflowResultSnapshot(kind='ROBERT', status='completed', root_name='run', summary_available=True, metrics=(('RMSE', '0.15', 'Test (No_PFI)', 'PREDICT/PREDICT_data.dat'),)))
        answer = BotEngine(self.kb, provider_registry={'OpenAI': provider}).answer_with_metadata('Summarize this workflow report', snapshot, 'Cloud AI', 'OpenAI', 'test-key', web_search_enabled=False)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn('0.15', answer.text)
        self.assertEqual(answer.usage.total_tokens, 240)

    def setUp(self):
        records = [{'id': 'csv-rule', 'topic': 'workflow', 'tab': 'ROBERT', 'keywords': ['csv', 'run'], 'text': 'ROBERT needs a main CSV before the workflow can run.', 'source': 'memory'}, {'id': 'tutorial-advanced-options', 'topic': 'tutorial', 'tab': 'Advanced Options', 'keywords': ['parameters', 'settings', 'robert'], 'text': 'The Advanced Options tab provides GENERAL, CURATE, GENERATE, and PREDICT settings for the ROBERT workflow.', 'source': 'tutorial:overview.md'}, {'id': 'results-report-metrics', 'topic': 'results', 'tab': 'Reports', 'keywords': ['results', 'metrics', 'rmse', 'r2'], 'text': 'The report summarizes model quality with metrics such as RMSE and R2 together with the selected workflow settings.', 'source': 'tutorial:reports.md'}]
        self.kb = KnowledgeBase.from_records(records)
        self.snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=False, run_enabled=False, stop_enabled=False, recent_console='', console_terms=[])

    def test_cloud_failure_falls_back_to_grounded_workflow_summary(self):

        class FailingProvider(FakeProvider):

            def generate_answer(self, *args, **kwargs):
                raise RuntimeError('provider unavailable')
        result = WorkflowResultSnapshot(kind='ROBERT', status='completed', root_name='run', summary_available=True, stages=(WorkflowStageResult('REPORT', 'completed', source='ROBERT_report.pdf'),), sources=('ROBERT_report.pdf',))
        snapshot = dataclasses.replace(self.snapshot, workflow_results=result)
        engine = BotEngine(knowledge_base=self.kb, provider_registry={'OpenAI': FailingProvider()})
        answer = engine.answer('Summarize this workflow', snapshot, 'Cloud AI', 'OpenAI', 'test-key', web_search_enabled=False)
        self.assertIn('**Workflow result:** Completed', answer)
        self.assertIn('**REPORT:** Completed', answer)
        self.assertNotIn('Unexpected LLM error', answer)

    def test_local_prompt_uses_history_only_for_follow_up(self):
        history = [{'role': 'user', 'content': 'Why is the Reports tab locked?', 'memory_eligible': True}, {'role': 'assistant', 'content': 'It unlocks after report output exists.', 'memory_eligible': True}]
        local = FakeLocalManager(response='follow-up answer')
        engine = BotEngine(self.kb, local_manager=local)
        engine.answer('And when does it unlock?', self.snapshot, 'Local AI', 'OpenAI', '', conversation_history=history)
        self.assertIn('Why is the Reports tab locked?', local.calls[0]['user_prompt'])
        self.assertIn('It unlocks after report output exists.', local.calls[0]['user_prompt'])
        local.calls.clear()
        engine.answer('What does the AQME tab do?', self.snapshot, 'Local AI', 'OpenAI', '', conversation_history=history)
        self.assertNotIn('Why is the Reports tab locked?', local.calls[0]['user_prompt'])

    def test_heuristic_mode_uses_parameter_builder_for_parameter_questions(self):
        kb = KnowledgeBase.from_records([{'id': 'param-pfi-filter', 'source': 'robert_docs', 'source_file': 'Modules/generate.rst', 'program': 'ROBERT', 'module': 'GENERATE', 'kind': 'parameter', 'tab': 'Advanced Options', 'title': 'PFI filter threshold', 'question_patterns': ['what does pfi filter do'], 'keywords': ['pfi', 'filter', 'threshold'], 'text': 'The PFI filter controls feature selection after model fitting.', 'priority': 0.9}])
        engine = BotEngine(kb)
        answer = engine.answer(question='What does the PFI filter parameter do?', snapshot=self.snapshot, mode='Heuristic', provider='OpenAI', api_key='')
        self.assertIn('PFI filter', answer)
        self.assertIn('controls feature selection', answer)
        self.assertNotIn('What is happening:', answer)

    def test_local_ai_mode_retries_once_with_ultracompact_prompt_on_failure(self):
        local_manager = FakeLocalManager(response='local answer after retry', errors=[LocalLLMError('400 Client Error: Bad Request')])
        engine = BotEngine(self.kb, local_manager=local_manager)
        answer = engine.answer(question='an error happened in my robert process, what does it mean?', snapshot=self.snapshot, mode='Local AI', provider='OpenAI', api_key='', conversation_history=[{'role': 'user', 'content': 'hi ' * 40}, {'role': 'assistant', 'content': 'hello ' * 50}, {'role': 'user', 'content': 'please explain ' * 60}])
        self.assertIn('retried using a minimal prompt', answer)
        self.assertIn('local answer after retry', answer)
        self.assertEqual(len(local_manager.calls), 2)
        self.assertLessEqual(len(local_manager.calls[1]['user_prompt']), len(local_manager.calls[0]['user_prompt']))
        self.assertIn('minimal context retry', local_manager.calls[1]['system_prompt'].lower())
        self.assertNotIn('hi hi', local_manager.calls[0]['user_prompt'])
        self.assertNotIn('please explain', local_manager.calls[1]['user_prompt'])

    def test_heuristic_mode_uses_console_details_for_process_finish_failure_popup(self):
        engine = BotEngine(self.kb)
        snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='File "utils.py", line 743, in correlation_filter\\nres_y = stats.linregress(csv_df_filtered[col], csv_df_filtered[self.args.y])\\nTypeError: ufunc \\'divide\\' not supported for the input types', console_terms=['typeerror', 'linregress', 'divide', 'not', 'supported'], popup_active=True, popup_title='WARNING!', popup_text='ROBERT encountered an issue while finishing. Please check the logs.', popup_buttons=['OK'], popup_kind='warning', popup_source='robert_finish_failure')
        answer = engine.answer(question='what is this popup?', snapshot=snapshot, mode='Heuristic', provider='OpenAI', api_key='')
        self.assertIn('TypeError', answer)
        self.assertIn('divide', answer)
        self.assertIn('correlation_filter', answer)
        self.assertNotIn('CSV file is still there', answer)

    def test_heuristic_mode_explains_smiles_target_as_likely_root_cause(self):
        engine = BotEngine(self.kb)
        snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='SMILES', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='File "utils.py", line 743, in correlation_filter\\nres_y = stats.linregress(csv_df_filtered[col], csv_df_filtered[self.args.y])\\nTypeError: ufunc \\'divide\\' not supported for the input types', console_terms=['typeerror', 'linregress', 'divide'], popup_active=True, popup_title='WARNING!', popup_text='ROBERT encountered an issue while finishing. Please check the logs.', popup_buttons=['OK'], popup_kind='warning', popup_source='robert_finish_failure', failure_type='TypeError', failure_message="ufunc 'divide' not supported for the input types", failure_location='utils.py:743', failure_function='correlation_filter', failure_operation='stats.linregress(csv_df_filtered[col], csv_df_filtered[self.args.y])', likely_cause='one or both inputs to the regression step are not numeric or have incompatible dtypes')
        answer = engine.answer(question='how i solve this error?', snapshot=snapshot, mode='Heuristic', provider='OpenAI', api_key='')
        self.assertIn('SMILES', answer)
        self.assertIn('target', answer.lower())
        self.assertNotIn('Test CSV', answer)

    def test_local_ai_selected_tutorial_uses_bounded_data_blocks(self):
        local_manager = FakeLocalManager(response='Load the CSV first.')
        kb = KnowledgeBase.from_records([{'id': 'tutorial-csv-1', 'source': 'tutorial:csv.md', 'kind': 'tutorial', 'tab': 'ROBERT', 'text': 'Start from a CSV file and select the target column.'}])
        engine = BotEngine(kb, local_manager=local_manager)
        engine.answer(question='How do I start from a CSV file?', snapshot=self.snapshot, mode='Local AI', provider='OpenAI', api_key='')
        self.assertEqual(len(local_manager.calls), 1)
        prompt = local_manager.calls[0]['user_prompt']
        self.assertNotIn('Tutorial summary to rewrite', prompt)
        self.assertIn('<USER_QUESTION_DATA>', prompt)
        self.assertIn('<RETRIEVED_KNOWLEDGE_DATA>', prompt)
        self.assertNotIn('GUI_STATE_DATA', prompt)

    def test_optional_search_413_retries_once_without_web_search(self):
        provider = OversizedOptionalSearchProvider()
        engine = BotEngine(self.kb, provider_registry={'OpenAI': provider})
        assessment = EvidenceAssessment(EvidenceDecision.WEB_OPTIONAL, 'borderline')
        with patch('gui_easyrob.bot.bot_engine.assess_evidence', return_value=assessment):
            result = engine.answer_with_metadata('Explain one-shot testing', self.snapshot, 'Cloud AI', 'OpenAI', 'secret')
        self.assertEqual(result.text, 'Fallback answer without web search.')
        self.assertEqual([call['web_search_mode'] for call in provider.calls], [WebSearchMode.AUTO, WebSearchMode.OFF])
        self.assertIn('without web search', result.metadata.search_error.lower())
        self.assertEqual(result.usage.total_tokens, 120)
if __name__ == '__main__':
    unittest.main()""", ('TestBotEngine',), ())

# Selected checks from test_bot_export_docs_knowledge.py
_load_case_group('export_docs_knowledge', """import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unicodedata
import hashlib
from dataclasses import replace
from pathlib import Path
import pytest
SOURCE_GUIDANCE_DIR = Path(__file__).resolve().parents[1] / 'robert' / 'gui_easyrob' / 'bot' / 'knowledge_sources'

def make_export_sources(tmp_path: Path):
    from export_bot_docs_knowledge import ExportSources
    return ExportSources(robert_docs=tmp_path / 'robert-docs', aqme_docs=tmp_path / 'aqme-docs', robert_code=tmp_path / 'robert-code', easyrob_code=tmp_path / 'easyrob-code', aqme_code=tmp_path / 'aqme-code', tutorials=tmp_path / 'tutorials', user_guidance=tmp_path / 'knowledge-sources')

def make_complete_fixture_sources(tmp_path: Path):
    sources = make_export_sources(tmp_path)
    fixture_files = {sources.robert_docs / 'Modules' / 'generate.rst': 'GENERATE\\n========\\n\\nGENERATE builds machine learning models.\\n', sources.aqme_docs / 'Quickstart' / 'qdescp.rst': 'QDESCP\\n=======\\n\\nQDESCP generates molecular descriptors.\\n', sources.robert_code / 'model.py': 'def evaluate_model():\\n    \"\"\"Evaluate ROBERT model metrics.\"\"\"\\n', sources.easyrob_code / 'window.py': 'def show_report():\\n    \"\"\"Show the ROBERT report output.\"\"\"\\n', sources.aqme_code / 'descriptors.py': 'def generate_descriptors():\\n    \"\"\"Generate AQME molecular descriptors.\"\"\"\\n'}
    for path, content in fixture_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    manifest = json.loads((SOURCE_GUIDANCE_DIR / 'coverage_manifest.json').read_text(encoding='utf-8'))
    sources.tutorials.mkdir(parents=True)
    for filename in manifest['tutorial_files'].values():
        (sources.tutorials / filename).write_text(f'# {Path(filename).stem.title()}\\n\\nFollow this tutorial workflow.', encoding='utf-8')
    sources.user_guidance.mkdir(parents=True)
    for filename in ('robert_user_guidance.json', 'aqme_user_guidance.json', 'ui_user_guidance.json', 'coverage_manifest.json'):
        shutil.copyfile(SOURCE_GUIDANCE_DIR / filename, sources.user_guidance / filename)
    return sources

class TestExportBotDocsKnowledge(unittest.TestCase):

    def test_clean_text_removes_html_badges_roles_and_local_markers(self):
        from export_bot_docs_knowledge import clean_text
        raw = '\\n.. image:: badge.svg\\n   :target: https://example.invalid\\n<input type="checkbox">\\n:local:\\n:class: InternalThing\\nUseful user sentence with :doc:`the guide <https://docs.example.org/guide>`.\\n'
        self.assertEqual(clean_text(raw), 'Useful user sentence with the guide <https://docs.example.org/guide>.')

    def test_curated_overrides_include_core_result_concepts(self):
        from export_bot_docs_knowledge import build_curated_overrides
        chunks = build_curated_overrides()
        chunk_by_id = {str(chunk['id']): chunk for chunk in chunks}
        self.assertIn('robert-r2-overview', chunk_by_id)
        self.assertIn('robert-mae-overview', chunk_by_id)
        self.assertIn('robert-identical-predictions', chunk_by_id)
        self.assertNotIn('robert-score-interpretation', chunk_by_id)

@pytest.mark.parametrize('failed_replacement', [2, 3])
def test_replacement_failure_restores_all_final_and_legacy_files(tmp_path, monkeypatch, failed_replacement):
    import export_bot_docs_knowledge as exporter
    output = tmp_path / 'knowledge'
    output.mkdir()
    original = {'robert_context.json': b'original-robert\\n', 'ui_context.json': b'original-ui\\n'}
    for filename, content in original.items():
        (output / filename).write_bytes(content)
    legacy = output / 'robert_docs.json'
    legacy.write_bytes(b'legacy\\n')
    sources = make_complete_fixture_sources(tmp_path / 'sources')
    real_replace = Path.replace
    replacement_calls = 0

    def fail_selected_final(self, target):
        nonlocal replacement_calls
        target = Path(target)
        if target.name in exporter.FINAL_CONTEXT_FILENAMES:
            replacement_calls += 1
            if replacement_calls == failed_replacement:
                raise OSError(f'replace {failed_replacement} failed')
        return real_replace(self, target)
    monkeypatch.setattr(Path, 'replace', fail_selected_final)
    with pytest.raises(OSError, match=f'replace {failed_replacement} failed'):
        exporter.export_docs_knowledge(output, sources=sources)
    assert (output / 'robert_context.json').read_bytes() == original['robert_context.json']
    assert not (output / 'aqme_context.json').exists()
    assert (output / 'ui_context.json').read_bytes() == original['ui_context.json']
    assert legacy.read_bytes() == b'legacy\\n'
    assert not list(output.glob('*.tmp'))
    assert not list(output.glob('*.bak'))

def _normalized_content_fingerprint(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()
if __name__ == '__main__':
    unittest.main()""", ('TestExportBotDocsKnowledge', 'test_replacement_failure_restores_all_final_and_legacy_files'), ())

# Selected checks from test_bot_heuristics.py
_load_case_group('heuristics', """import unittest
from gui_easyrob.bot.heuristics import diagnose_snapshot, format_heuristic_answer
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_rag import KnowledgeChunk, SearchResult

class TestHeuristics(unittest.TestCase):

    def test_direct_gui_state_questions_use_snapshot_without_documentation(self):
        snapshot = GuiSnapshot(active_tab='Advanced Options', workflow='GENERATE', main_csv_path='C:/private/train.csv', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=True, run_enabled=False, stop_enabled=True, recent_console='Optimizing models', console_terms=['optimizing', 'models'], name_column='code_name', active_process='ROBERT', run_aqme_enabled=False, ignored_columns=('SMILES',), advanced_settings=('seed=7', 'models=RF, GB'))
        cases = {'Which tab am I in?': 'Advanced Options', 'What workflow is selected right now?': 'GENERATE', 'What columns are selected?': 'code_name', 'What is running now?': 'ROBERT is running', 'What advanced settings are selected?': '- General:'}
        for question, expected in cases.items():
            with self.subTest(question=question):
                diagnosis = diagnose_snapshot(snapshot, question)
                answer = format_heuristic_answer(question, snapshot, diagnosis, [], intent='tutorial')
                self.assertIn(expected, answer)""", ('TestHeuristics',), ())

# Selected checks from test_bot_knowledge_schema.py
_load_case_group('knowledge_schema', """from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from gui_easyrob.bot.knowledge_schema import ALLOWED_AUDIENCES, ALLOWED_ENTITY_TYPES, ALLOWED_KINDS, ALLOWED_SOURCE_TIERS, ALLOWED_TABS, CONTEXT_SECTIONS, CONTEXT_VERSION, DOMAINS, ISO_DATE_RE, SCHEMA_VERSION, KnowledgeValidationError, validate_context_payload, validate_context_set, validate_guidance_source_set
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_CONTEXT_VERSION = '2026-07-10'
FIXTURE_CONTEXT_SECTIONS = ('workflows', 'modules', 'interface', 'parameters', 'concepts', 'inputs_outputs', 'tutorials', 'troubleshooting')
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'robert'
SOURCES = PACKAGE_ROOT / 'gui_easyrob' / 'bot' / 'knowledge_sources'
SOURCE_FILENAMES = {'robert': 'robert_user_guidance.json', 'aqme': 'aqme_user_guidance.json', 'ui': 'ui_user_guidance.json'}
EXPECTED_REQUIRED_ENTITIES = {'robert': ['program-robert', 'workflow-robert-full', 'workflow-robert-external-prediction', 'module-curate', 'module-generate', 'module-verify', 'module-predict', 'module-report', 'concept-robert-target-column', 'concept-robert-prediction-type', 'results-robert-report', 'results-robert-rmse', 'results-robert-mae', 'results-robert-r2', 'results-robert-pfi', 'diagnostic-robert-prerequisites'], 'aqme': ['program-aqme', 'concept-aqme-when-not-needed', 'workflow-aqme-descriptors-to-robert', 'module-aqme-qdescp', 'module-aqme-csearch', 'module-aqme-cmin', 'module-aqme-qprep', 'module-aqme-qcorr', 'input-aqme-smiles', 'control-aqme-descriptor-level', 'control-aqme-solvent', 'control-aqme-atom-selection', 'diagnostic-aqme-multiple-matches', 'diagnostic-aqme-metals', 'output-aqme-descriptor-csv'], 'ui': ['tab-robert', 'tab-aqme', 'tab-advanced-options', 'tab-reports', 'tab-predictions', 'tab-evaluate', 'tab-images', 'tab-molssi-databases', 'control-main-csv', 'control-test-csv', 'control-target-column', 'control-prediction-type', 'control-workflow-selector', 'module-generate-hyperoptimization', 'control-enable-aqme', 'action-run-robert', 'action-run-aqme', 'action-stop-process', 'action-clear-test-csv', 'action-open-tutorials', 'action-open-documentation', 'action-open-robbot', 'control-robbot-mode', 'control-robbot-provider', 'action-robbot-send', 'action-robbot-clear-history', 'action-robbot-load-local-ai', 'control-aqme-input', 'action-aqme-map-atoms', 'action-open-chemdraw', 'status-console', 'status-popup', 'status-running-workflow', 'workflow-ui-start-csv', 'workflow-ui-start-chemdraw', 'workflow-ui-generate-descriptors', 'workflow-ui-new-predictions', 'workflow-ui-molssi-data', 'workflow-ui-inspect-results', 'workflow-ui-diagnose-failure', 'tutorial-overview', 'tutorial-csv', 'tutorial-chemdraw', 'tutorial-descriptors', 'tutorial-predictions']}
EXPECTED_TUTORIAL_FILES = {'tutorial-overview': 'overview.md', 'tutorial-csv': 'csv.md', 'tutorial-chemdraw': 'chemdraw.md', 'tutorial-descriptors': 'descriptors.md', 'tutorial-predictions': 'predictions.md'}

def valid_record(record_id: str='fixture-record', kind: str='concept', priority: float=0.5) -> dict[str, object]:
    return {'id': record_id, 'source': 'fixture_source', 'source_file': 'fixture.md', 'program': 'ROBERT', 'module': 'GENERATE', 'kind': kind, 'topic': 'model building', 'tab': 'ROBERT', 'title': 'Fixture record', 'text': 'This record explains a user-facing ROBERT concept.', 'keywords': ['ROBERT', 'model'], 'question_patterns': ['what does this ROBERT concept do'], 'priority': priority, 'audience': 'end_user', 'source_tier': 'generated', 'aliases': ['fixture concept'], 'related_ids': []}

def valid_context(domain: str) -> dict[str, object]:
    payload: dict[str, object] = {'schema_version': FIXTURE_SCHEMA_VERSION, 'version': FIXTURE_CONTEXT_VERSION, 'domain': domain, 'overview': {'title': f'{domain.upper()} overview', 'text': f'User-facing guidance for the {domain} knowledge domain.'}}
    payload.update({section: [] for section in FIXTURE_CONTEXT_SECTIONS})
    payload['concepts'] = [valid_record(record_id=f'{domain}-fixture-record')]
    return payload

def valid_context_set() -> dict[str, dict[str, object]]:
    return {domain: valid_context(domain) for domain in ('robert', 'aqme', 'ui')}

def valid_guidance_record(record_id: str='robert-curated-program', entity_id: str='program-robert') -> dict[str, object]:
    record = valid_record(record_id=record_id)
    record.pop('text')
    record.update({'source': 'curated_user_guidance', 'source_file': 'robert_user_guidance.json', 'source_tier': 'curated', 'entity_type': 'program', 'entity_id': entity_id, 'user_guidance': {'purpose': 'Explain when ROBERT is the appropriate modelling tool.', 'when_to_use': ['Use ROBERT when a descriptor table is ready.'], 'prerequisites': ['Prepare a CSV with a target column.'], 'steps': [], 'result': 'The user can choose the correct ROBERT workflow.', 'next_steps': [], 'common_issues': []}})
    return record

def valid_guidance_set() -> tuple[dict[str, object], dict[str, object]]:
    payloads: dict[str, object] = {'robert': {'schema_version': FIXTURE_SCHEMA_VERSION, 'version': FIXTURE_CONTEXT_VERSION, 'domain': 'robert', 'records': [valid_guidance_record()]}, 'aqme': {'schema_version': FIXTURE_SCHEMA_VERSION, 'version': FIXTURE_CONTEXT_VERSION, 'domain': 'aqme', 'records': []}, 'ui': {'schema_version': FIXTURE_SCHEMA_VERSION, 'version': FIXTURE_CONTEXT_VERSION, 'domain': 'ui', 'records': []}}
    manifest: dict[str, object] = {'schema_version': FIXTURE_SCHEMA_VERSION, 'version': FIXTURE_CONTEXT_VERSION, 'required_entities': {'robert': ['program-robert'], 'aqme': [], 'ui': []}, 'tutorial_files': {}}
    return (payloads, manifest)

def valid_tutorial_guidance_record() -> dict[str, object]:
    record = valid_guidance_record(record_id='ui-curated-tutorial-overview', entity_id='tutorial-overview')
    record.update({'source_file': 'ui_user_guidance.json', 'module': 'EASYROB_UI', 'kind': 'tutorial', 'topic': 'getting started', 'tab': 'ROBERT', 'title': 'EasyROB overview tutorial', 'entity_type': 'tutorial'})
    record['user_guidance']['steps'] = ['Open the ROBERT tab and review the main workflow controls.']
    record['user_guidance']['next_steps'] = ['Continue with the CSV tutorial when a descriptor table is ready.']
    return record

def test_context_schema_validates_sections_record_fields_and_string_lists():
    payload = valid_context('robert')
    payload.pop('interface')
    payload['overview'] = {'title': '', 'text': 7}
    record = payload['concepts'][0]
    record['title'] = ''
    record['keywords'] = ['valid', 7]
    record['tab'] = 'Settings'
    record['audience'] = 'everyone'
    issues = validate_context_payload(payload, 'robert', 'fixture.json')
    assert {issue.code for issue in issues} >= {'missing_section', 'invalid_overview_title', 'invalid_overview_text', 'invalid_title', 'invalid_keywords', 'invalid_tab', 'invalid_audience'}

def test_guidance_source_set_enforces_curated_relationship_and_type_rules():
    payloads, manifest = valid_guidance_set()
    record = payloads['robert']['records'][0]
    record['source_tier'] = 'reference'
    record['audience'] = 'advanced_user'
    record['entity_type'] = 'workflow'
    record['tab'] = ''
    record['related_ids'] = ['missing-entity']
    record['user_guidance']['steps'] = []
    record['user_guidance']['next_steps'] = []
    issues = validate_guidance_source_set(payloads, manifest, 'fixtures')
    assert {issue.code for issue in issues} >= {'invalid_curated_source_tier', 'invalid_curated_audience', 'missing_steps', 'missing_next_steps', 'unresolved_related_id'}

def load_source(name: str) -> object:
    return json.loads((SOURCES / name).read_text(encoding='utf-8'))

def load_curated_payloads() -> dict[str, object]:
    return {domain: load_source(filename) for domain, filename in SOURCE_FILENAMES.items()}

def records_by_entity(payload: object) -> dict[str, dict[str, object]]:
    return {record['entity_id']: record for record in payload['records']}

def rendered_guidance(record: dict[str, object]) -> str:
    guidance = record['user_guidance']
    parts = [guidance['purpose']]
    for field in ('when_to_use', 'prerequisites', 'steps'):
        parts.extend(guidance[field])
    parts.append(guidance['result'])
    for field in ('next_steps', 'common_issues'):
        parts.extend(guidance[field])
    return ' '.join(parts)""", ('test_context_schema_validates_sections_record_fields_and_string_lists', 'test_guidance_source_set_enforces_curated_relationship_and_type_rules'), ())

# Selected checks from test_bot_llm_providers.py
_load_case_group('llm_providers', """import unittest
from unittest.mock import Mock
import requests
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.answer_metadata import EvidenceOrigin, WebSearchMode, normalize_provider_answer_text
from gui_easyrob.bot.prompt_policy import AnswerDepth
from gui_easyrob.bot.llm_providers import LLMProviderError, LLMProviderErrorKind, GEMINI_SPEC, OPENAI_SPEC, PROVIDER_SPECS, GeminiAdapter, OpenAICompatibleAdapter, _build_prompts, build_llm_prompts, build_local_llm_prompts, build_provider_registry

class FakeResponse:

    def __init__(self, payload=None, text='', status_error=None):
        self._payload = payload or {}
        self.text = text
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload

class TestLlmProviders(unittest.TestCase):

    def setUp(self):
        self.snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='', console_terms=[], aqme_workflow_enabled=False, enabled_tabs=['ROBERT'], disabled_tabs=['AQME'])

    def test_openai_required_search_uses_responses_and_normalizes_sources(self):
        captured = {}
        payload = {'id': 'resp-test', 'model': 'gpt-4.1-mini', 'output': [{'type': 'web_search_call', 'status': 'completed', 'action': {'type': 'search', 'query': 'latest relevant change', 'sources': [{'title': 'Official source', 'url': 'https://example.org/source'}]}}, {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Grounded answer.', 'annotations': [{'type': 'url_citation', 'title': 'Official source', 'url': 'https://example.org/source', 'start_index': 0, 'end_index': 8}]}]}], 'usage': {'input_tokens': 1000, 'output_tokens': 100, 'total_tokens': 1100, 'input_tokens_details': {'cached_tokens': 200}}}

        def transport(method, url, **kwargs):
            captured.update(method=method, url=url, **kwargs)
            return FakeResponse(payload=payload)
        result = build_provider_registry(transport=transport)['OpenAI'].generate_answer_with_usage('What changed today?', self.snapshot, [], 'secret', web_search_mode=WebSearchMode.REQUIRED, search_query='latest relevant change', evidence_origin=EvidenceOrigin.WEB_AND_DOCUMENTATION)
        self.assertEqual(captured['url'], 'https://api.openai.com/v1/responses')
        self.assertEqual(captured['json']['tools'], [{'type': 'web_search', 'search_context_size': 'low'}])
        self.assertEqual(captured['json']['tool_choice'], 'required')
        self.assertEqual(captured['json']['max_tool_calls'], 1)
        self.assertEqual(result.text, 'Grounded answer.')
        self.assertEqual(result.usage.search_requests, 1)
        self.assertEqual(result.usage.search_content_tokens, 8000)
        self.assertAlmostEqual(result.usage.estimated_search_cost_usd, 0.01)
        self.assertEqual(result.metadata.sources[0].url, 'https://example.org/source')

    def test_all_cloud_protocols_hide_incomplete_trailing_sentence_when_token_limited(self):
        complete = 'R², MAE and RMSE measure regression performance.'
        partial = f'{complete} **Next useful check:** Before judging the model'
        cases = (('Anthropic', {'content': [{'type': 'text', 'text': partial}], 'stop_reason': 'max_tokens'}), ('Gemini', {'candidates': [{'content': {'parts': [{'text': partial}]}, 'finishReason': 'MAX_TOKENS'}]}), ('OpenAI', {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': partial}]}]}))
        for provider, payload in cases:
            with self.subTest(provider=provider):
                transport = lambda *args, _payload=payload, **kwargs: FakeResponse(payload=_payload)
                adapter = GeminiAdapter(GEMINI_SPEC, transport=transport) if provider == 'Gemini' else build_provider_registry(transport=transport)[provider]
                result = adapter.generate_answer('Explain these metrics', self.snapshot, [], 'key')
                self.assertEqual(result, complete)

    def test_provider_maps_failures_to_safe_categories(self):
        cases = ((requests.Timeout('timeout'), LLMProviderErrorKind.TIMEOUT), (requests.ConnectionError('down'), LLMProviderErrorKind.CONNECTION), (401, LLMProviderErrorKind.AUTHENTICATION), (403, LLMProviderErrorKind.AUTHENTICATION), (429, LLMProviderErrorKind.RATE_LIMIT), (400, LLMProviderErrorKind.INVALID_REQUEST), (503, LLMProviderErrorKind.UNAVAILABLE_SERVICE))
        for exception_or_status, expected_kind in cases:
            with self.subTest(value=exception_or_status):

                def transport(method, url, **kwargs):
                    if isinstance(exception_or_status, requests.RequestException):
                        raise exception_or_status
                    response = Mock()
                    response.status_code = exception_or_status
                    response.text = '{"error":{"message":"provider rejected request"}}'
                    response.json.return_value = {'error': {'message': 'provider rejected request'}}
                    response.raise_for_status.side_effect = requests.HTTPError(f'status {exception_or_status}', response=response)
                    return response
                adapter = OpenAICompatibleAdapter(OPENAI_SPEC, transport=transport)
                with self.assertRaises(LLMProviderError) as caught:
                    adapter.generate_answer('question', self.snapshot, [], 'fake-key')
                self.assertIs(caught.exception.kind, expected_kind)
                self.assertNotIn('fake-key', str(caught.exception))

    def test_invalid_json_empty_answer_and_bad_shape_are_invalid_response(self):
        invalid_json = Mock()
        invalid_json.raise_for_status.return_value = None
        invalid_json.json.side_effect = ValueError('not json')
        payloads = (invalid_json, {}, {'choices': []}, {'choices': [{'message': {'content': ''}}]})
        for payload in payloads:
            with self.subTest(payload=repr(payload)):
                adapter = OpenAICompatibleAdapter(OPENAI_SPEC, transport=lambda *args, value=payload, **kwargs: value)
                with self.assertRaises(LLMProviderError) as caught:
                    adapter.generate_answer('question', self.snapshot, [], 'fake-key')
                self.assertIs(caught.exception.kind, LLMProviderErrorKind.INVALID_RESPONSE)

    def test_local_llm_prompts_are_more_compact_than_cloud_prompts(self):
        snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console=''.join((f'line {idx} error details ' for idx in range(200))), console_terms=['error', 'traceback', 'csv', 'missing', 'runtime', 'aqme'], aqme_workflow_enabled=False, enabled_tabs=['ROBERT', 'AQME', 'PREDICT'], disabled_tabs=['Results'])
        retrieved = [type('Chunk', (), {'id': f'chunk-{idx}', 'topic': 'diagnostic', 'tab': 'ROBERT', 'text': ('useful diagnostic context ' * 40).strip()})() for idx in range(6)]
        conversation = [{'role': 'user', 'content': f'question {idx} ' + 'details ' * 30} if idx % 2 == 0 else {'role': 'assistant', 'content': f'answer {idx} ' + 'context ' * 30} for idx in range(8)]
        _, cloud_user_prompt = _build_prompts(question='what does this error mean?', gui_context=snapshot, retrieved_chunks=retrieved, conversation_history=conversation, question_intent='diagnostic')
        _, local_user_prompt = build_local_llm_prompts(question='what does this error mean?', gui_context=snapshot, retrieved_chunks=retrieved, conversation_history=conversation, question_intent='diagnostic')
        self.assertLess(len(local_user_prompt), len(cloud_user_prompt))
        self.assertNotIn('Conversation so far', local_user_prompt)
        self.assertIn('Recent console tail', local_user_prompt)
if __name__ == '__main__':
    unittest.main()""", ('TestLlmProviders',), ())

# Selected checks from test_bot_local_llm.py
_load_case_group('local_llm', """import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import gui_easyrob.bot.local_llm as local_llm
from gui_easyrob.bot.local_llm import DEFAULT_LOCAL_BATCH_SIZE, DEFAULT_LOCAL_MAX_TOKENS, DEFAULT_LOCAL_N_CTX, DEFAULT_LOCAL_REPEAT_PENALTY, DEFAULT_LOCAL_TEMPERATURE, DEFAULT_LOCAL_TOP_P, DEFAULT_LOCAL_UBATCH_SIZE, LOCAL_MODEL_DISPLAY_NAME, LOCAL_MODEL_FILENAME, LOCAL_MODEL_REPO_ID, LocalLLMError, LocalLLMManager

class FakeResponse:

    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'http {self.status_code}')

    def json(self):
        return self._payload

class FakeProcess:

    def __init__(self, poll_result=None):
        self._poll_result = poll_result
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return self._poll_result

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)

    def kill(self):
        self.killed = True

class TestLocalLLMManager(unittest.TestCase):

    def test_generate_response_strips_leaked_intro_and_keeps_real_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir)
            runtime_binary = runtime_root / 'llama-server.exe'
            runtime_binary.write_text('exe', encoding='utf-8')
            process = FakeProcess()
            transport = Mock(side_effect=[FakeResponse({'status': 'ok'}), FakeResponse({'choices': [{'message': {'content': 'Okay, the user is asking for a sandwich.\\nLet me think about the best answer.\\n\\nYou can make a simple sandwich with bread, cheese, and tomato.'}}]})])
            manager = LocalLLMManager(downloader=Mock(return_value='C:/cache/model.gguf'), runtime_root=runtime_root, runtime_downloader=Mock(return_value=str(runtime_binary)), process_factory=Mock(return_value=process), transport=transport, sleep_fn=lambda _: None, system_name='Windows', machine_name='AMD64')
            answer = manager.generate_response('system', 'user')
            self.assertEqual(answer, 'You can make a simple sandwich with bread, cheese, and tomato.')
if __name__ == '__main__':
    unittest.main()""", ('TestLocalLLMManager',), ())

# Selected checks from test_bot_local_model_worker.py
_load_case_group('local_model_worker', """import unittest
from gui_easyrob.bot.local_llm import LocalLLMError
from gui_easyrob.bot.local_model_worker import LocalModelPrepareWorker

class FakeLocalManager:

    def __init__(self, exists=False, runtime_exists=True, download_error=None, runtime_error=None, load_error=None):
        self.exists = exists
        self.runtime_ready = runtime_exists
        self.download_error = download_error
        self.runtime_error = runtime_error
        self.load_error = load_error
        self.calls = []

    def model_exists(self):
        self.calls.append('model_exists')
        return self.exists

    def download_model(self, progress_callback=None):
        self.calls.append('download_model')
        if progress_callback is not None:
            progress_callback('downloading', 'Downloading local model (about 2.1 GB)...')
        if self.download_error is not None:
            raise self.download_error

    def runtime_exists(self):
        self.calls.append('runtime_exists')
        return self.runtime_ready

    def download_runtime(self, progress_callback=None):
        self.calls.append('download_runtime')
        if progress_callback is not None:
            progress_callback('downloading', 'Downloading local AI runtime...')
        if self.runtime_error is not None:
            raise self.runtime_error

    def load_model(self):
        self.calls.append('load_model')
        if self.load_error is not None:
            raise self.load_error

class TestLocalModelPrepareWorker(unittest.TestCase):

    def test_worker_emits_statuses_and_prepared_signal_when_download_is_needed(self):
        worker = LocalModelPrepareWorker(FakeLocalManager(exists=False))
        statuses = []
        prepared = []
        worker.status_changed.connect(lambda status, detail: statuses.append((status, detail)))
        worker.prepared.connect(lambda: prepared.append(True))
        worker.run()
        self.assertEqual([status for status, _ in statuses], ['Preparing...', 'Downloading...', 'Loading...', 'Ready'])
        self.assertEqual(prepared, [True])""", ('TestLocalModelPrepareWorker',), ())

# Selected checks from test_bot_online_docs.py
_load_case_group('online_docs', """\"\"\"Offline checks for bounded official-documentation fallback.\"\"\"
from unittest.mock import patch
from gui_easyrob.bot.online_docs import documentation_urls, extract_document_text, fetch_documentation

def test_two_programs_both_contribute_excerpts_and_sources():
    with patch('gui_easyrob.bot.online_docs._read_page', return_value='descriptors ' * 1000):
        chunks, sources = fetch_documentation('Compare AQME and ROBERT descriptors')
    assert len(chunks) <= 3
    assert len(sources) == 2
    for source in sources:
        assert any((source.url in chunk.text for chunk in chunks))

def test_advanced_document_reads_respect_the_web_setting():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from gui_easyrob.bot.bot_engine import BotEngine
    from gui_easyrob.bot.bot_context import GuiSnapshot
    from gui_easyrob.bot.llm_providers import ProviderAnswer
    from gui_easyrob.bot.token_usage import TokenUsage
    from gui_easyrob.bot.answer_metadata import SourceCitation, WebSearchMode
    from gui_easyrob.bot.bot_rag import KnowledgeChunk
    adapter = Mock(spec=['spec', 'generate_answer_with_usage'])
    adapter.spec = SimpleNamespace(supports_web_search=True)
    adapter.generate_answer_with_usage.return_value = ProviderAnswer('Grounded answer.', TokenUsage(20, 10, 30))
    engine = BotEngine(provider_registry={'OpenAI': adapter})
    snapshot = GuiSnapshot('ROBERT', 'Full Workflow', 'train.csv', '', 'yield', 'regression', False, True, False, '', [])
    source = SourceCitation.create('AQME', 'https://aqme.readthedocs.io/en/latest/Modules/qdescp.html')
    chunk = KnowledgeChunk('online-0', source.url, 'QDESCP', '', (), 'Boltzmann descriptors.')
    with patch('gui_easyrob.bot.online_docs.fetch_documentation', return_value=([chunk], (source,))) as fetch:
        result = engine.answer_with_metadata('How does AQME calculate QDESCP descriptors?', snapshot, 'Cloud AI', 'OpenAI', 'test-key', web_search_enabled=True)
        assert fetch.call_count == 1
        assert result.metadata.sources == (source,)
        assert adapter.generate_answer_with_usage.call_args.kwargs['web_search_mode'] is WebSearchMode.OFF
        fetch.reset_mock()
        engine.answer_with_metadata('How does AQME calculate QDESCP descriptors?', snapshot, 'Cloud AI', 'OpenAI', 'test-key', web_search_enabled=False)
        fetch.assert_not_called()""", ('test_two_programs_both_contribute_excerpts_and_sources', 'test_advanced_document_reads_respect_the_web_setting'), ())

# Selected checks from test_bot_output_quality.py
_load_case_group('output_quality', """\"\"\"Repeatable support scenarios and output-format regressions, without paid calls.\"\"\"
import pytest
from gui_easyrob.bot.bot_panel import _render_markdown
from gui_easyrob.bot.answer_metadata import normalize_provider_answer_text
from gui_easyrob.bot.bot_engine import BotEngine

def test_provider_table_becomes_readable_labeled_rows():
    answer = 'What needs attention\\n\\n| Observation | Why it matters | What to check next |\\n| --- | --- | --- |\\n| Score **4/10** | Review validation. | Open the report. |\\n| Uncertainty | Not recorded. | |'
    cleaned = normalize_provider_answer_text(answer)
    assert '| ---' not in cleaned
    assert '**Observation:** Score **4/10**' in cleaned
    assert '**Why it matters:** Review validation.' in cleaned
    assert '**What to check next:** Open the report.' in cleaned
    assert '<ul>' in _render_markdown(cleaned)""", ('test_provider_table_becomes_readable_labeled_rows',), ())

# Selected checks from test_bot_rag.py
_load_case_group('rag', """import unittest
from unittest import mock
import json
from pathlib import Path
import tempfile
from gui_easyrob.bot import bot_rag
from gui_easyrob.bot.bot_rag import KnowledgeBase
from gui_easyrob.bot.knowledge_schema import KnowledgeValidationError
_SECTIONS = ('workflows', 'modules', 'interface', 'parameters', 'concepts', 'inputs_outputs', 'tutorials', 'troubleshooting')
_GOLDEN_FIXTURE = Path(__file__).parent / 'fixtures' / 'bot_rag_golden.json'

def _valid_record(record_id='valid-record', **overrides):
    record = {'id': record_id, 'source': 'test_source', 'source_file': 'test.md', 'program': 'ROBERT', 'module': 'TEST', 'kind': 'concept', 'topic': 'testing', 'tab': 'ROBERT', 'title': 'Valid record', 'text': 'Valid knowledge text.', 'source_tier': 'generated', 'audience': 'end_user', 'keywords': ['valid'], 'question_patterns': ['what is valid'], 'priority': 0.7, 'aliases': ['valid alias'], 'related_ids': []}
    record.update(overrides)
    return record

def _valid_context(domain, records=()):
    payload = {'schema_version': 1, 'version': '2026-07-10', 'domain': domain, 'overview': {'title': f'{domain.upper()} overview', 'text': f'{domain.upper()} overview text.'}}
    payload.update({section: [] for section in _SECTIONS})
    payload['concepts'] = list(records)
    return payload

def _valid_context_set(records_by_domain=None):
    records_by_domain = records_by_domain or {}
    return {domain: _valid_context(domain, records_by_domain.get(domain, ())) for domain in ('robert', 'aqme', 'ui')}

def _write_context(path, domain, records=()):
    path.write_text(json.dumps(_valid_context(domain, records)), encoding='utf-8')

class TestKnowledgeBase(unittest.TestCase):

    def test_lenient_generic_loading_keeps_valid_records_and_aggregates_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            (base / 'records.json').write_text(json.dumps([{'id': 'valid-generic', 'text': 'Valid generic text.'}, 'not an object', {'id': 'missing-text'}]), encoding='utf-8')
            kb = KnowledgeBase.from_directory(base, strict=False, allow_generic_json=True)
        self.assertEqual([chunk.id for chunk in kb.chunks_for_source('records.json')], ['valid-generic'])
        self.assertEqual([issue.code for issue in kb.issues], ['invalid_record', 'invalid_record'])

    def test_lenient_loading_reports_duplicate_ids_and_keeps_valid_records(self):
        payloads = _valid_context_set({'robert': [_valid_record('duplicate-record', title='First record'), _valid_record('duplicate-record', title='Second record')]})
        kb = KnowledgeBase.from_payloads(payloads, strict=False)
        self.assertEqual([chunk.id for chunk in kb.chunks_for_source('test_source')], ['duplicate-record', 'duplicate-record'])
        self.assertIn('duplicate_id', {issue.code for issue in kb.issues})

    def test_from_directory_loads_raw_gui_tutorial_markdown_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            tutorials_dir = base / 'tutorials'
            tutorials_dir.mkdir()
            _write_context(base / 'ui_context.json', 'ui')
            (tutorials_dir / 'csv.md').write_text('Load the CSV in the ROBERT tab. Then select the target column and run the workflow.', encoding='utf-8')
            kb = KnowledgeBase.from_directory(base, tutorials_dir=tutorials_dir, strict=False)
        tutorial_chunks = [chunk for chunk in kb._chunks if chunk.source == 'tutorial:csv.md']
        self.assertEqual(len(tutorial_chunks), 1)
        self.assertEqual(tutorial_chunks[0].kind, 'tutorial')
        self.assertIn('target column', tutorial_chunks[0].text)
if __name__ == '__main__':
    unittest.main()""", ('TestKnowledgeBase',), ())

# Selected checks from test_bot_worker.py
_load_case_group('worker', """import unittest
from PySide6.QtWidgets import QApplication
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.answer_metadata import AnswerMetadata
from gui_easyrob.bot.bot_worker import BotReplyWorker
from gui_easyrob.bot.token_usage import TokenUsage

class FakeEngine:

    def __init__(self, answer_text=None, error=None, auto_reset_context=False, memory_eligible=True, usage=None):
        self.answer_text = answer_text
        self.error = error
        self.auto_reset_context = auto_reset_context
        self.memory_eligible = memory_eligible
        self.usage = usage or TokenUsage()
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.answer_text

    def answer_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {'text': self.answer_text, 'auto_reset_context': self.auto_reset_context, 'memory_eligible': self.memory_eligible, 'usage': self.usage, 'metadata': AnswerMetadata()}

class TestBotReplyWorker(unittest.TestCase):

    def setUp(self):
        self.snapshot = GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='yield', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='', console_terms=[])

    def _worker(self, engine, *, request_id=7, generation=3, api_key='', history=None, tutorial_id=None):
        return BotReplyWorker(engine=engine, request_id=request_id, generation=generation, question='What should I check next?', snapshot=self.snapshot, mode='Local AI', provider='OpenAI', api_key=api_key, conversation_history=history or [], tutorial_id=tutorial_id)

    def test_success_carries_request_generation_reset_and_memory_metadata(self):
        usage = TokenUsage(100, 20, 120, provider='OpenAI')
        engine = FakeEngine(answer_text='worker answer', usage=usage)
        worker = self._worker(engine, history=[{'role': 'user', 'content': 'Previous question'}])
        payloads = []
        worker.succeeded.connect(lambda *args: payloads.append(args))
        worker.run()
        self.assertEqual(payloads, [(7, 3, 'worker answer', False, True, usage, AnswerMetadata())])
        self.assertEqual(engine.calls[0]['question'], 'What should I check next?')
        self.assertEqual(engine.calls[0]['mode'], 'Local AI')
        self.assertEqual(engine.calls[0]['conversation_history'][0]['content'], 'Previous question')
        self.assertEqual(worker.api_key, '')
if __name__ == '__main__':
    unittest.main()""", ('TestBotReplyWorker',), ())

# Selected checks from test_bot_workflow_result_questions.py
_load_case_group('workflow_result_questions', """from __future__ import annotations
import json
from pathlib import Path
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.prompt_policy import CLOUD_PROMPT_BUDGET, build_bounded_user_prompt
from gui_easyrob.bot.workflow_results import WorkflowArtifact, WorkflowResultSnapshot, WorkflowStageResult, render_workflow_evidence
FIXTURE = Path(__file__).parent / 'fixtures' / 'workflow_result_questions.json'

def _snapshot() -> WorkflowResultSnapshot:
    return WorkflowResultSnapshot(kind='AQME+ROBERT', status='completed_with_warnings', root_name='example-run', summary_available=True, stages=(WorkflowStageResult('GENERATE', 'completed', 2.5, (), 'GENERATE/GENERATE_data.dat', ('Best model: RandomForestRegressor',)), WorkflowStageResult('VERIFY', 'completed_with_warnings', 0.75, (('flawed', 'PASSED', 'RMSE', '0.44'), ('sorted', 'UNCLEAR', 'RMSE', '0.39')), 'VERIFY/VERIFY_data.dat'), WorkflowStageResult('PREDICT', 'completed_with_warnings', 1.1, (), 'PREDICT/PREDICT_data.dat'), WorkflowStageResult('QDESCP', 'completed', 4.2, (), 'AQME/QDESCP_data.dat')), metrics=(('R2', '0.81', 'Cross-validation', 'PREDICT/PREDICT_data.dat'), ('R2', '0.74', 'Test', 'PREDICT/PREDICT_data.dat')), warnings=('PREDICT/PREDICT_data.dat: mol-2 is an outlier',), artifacts=(WorkflowArtifact('PREDICT/predictions.csv', 'csv', 2, 4, None, ('y_pred',), ('y_pred_sd',)), WorkflowArtifact('AQME-ROBERT_interpret_train.csv', 'csv', 2, 5, 2)), sources=('GENERATE/GENERATE_data.dat', 'VERIFY/VERIFY_data.dat', 'PREDICT/PREDICT_data.dat', 'AQME/QDESCP_data.dat'))

def _gui(result: WorkflowResultSnapshot) -> GuiSnapshot:
    return GuiSnapshot(active_tab='ROBERT', workflow='Full Workflow', main_csv_path='train.csv', test_csv_path='', target_column='y', prediction_type='Regression', process_running=False, run_enabled=True, stop_enabled=False, recent_console='', console_terms=[], workflow_results=result)

def test_every_matrix_question_builds_a_bounded_anti_invention_prompt():
    cases = json.loads(FIXTURE.read_text(encoding='utf-8'))
    gui = _gui(_snapshot())
    for case in cases:
        prompt = build_bounded_user_prompt(case['question'], gui, [], [], 'results', profile='cloud')
        assert len(prompt) <= CLOUD_PROMPT_BUDGET.total_chars
        assert '<WORKFLOW_RESULTS_DATA>' in prompt
        assert 'Do not infer molecule failures' in prompt
        assert 'explicitly state when requested information is absent' in prompt""", ('test_every_matrix_question_builds_a_bounded_anti_invention_prompt',), ())

# Selected checks from test_bot_workflow_results.py
_load_case_group('workflow_results', """from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
from gui_easyrob.bot.workflow_results import WorkflowResultStore, render_workflow_answer, render_workflow_evidence

def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path

def _complete_robert_tree(root: Path) -> Path:
    train = _write(root / 'train.csv', 'code_name,y,d1,d2\\nmol-1,1.2,3,4\\nmol-2,2.5,5,6\\n')
    _write(root / 'CURATE' / 'CURATE_data.dat', 'Starting CURATE with 2 datapoints and 2 descriptors\\no 1 duplicate datapoint was removed\\nTime CURATE: 1.25 seconds\\n')
    _write(root / 'GENERATE' / 'GENERATE_data.dat', 'Best model: RandomForestRegressor\\nTarget metric: RMSE = 0.31\\nTime GENERATE: 2.50 seconds\\n')
    _write(root / 'GENERATE' / 'Best_model' / 'No_PFI' / 'model_params.csv', 'model,error_type\\nRF,rmse\\n')
    _write(root / 'VERIFY' / 'VERIFY_data.dat', 'o flawed: PASSED, RMSE = 0.44, higher than thresholds\\n- sorted: UNCLEAR, RMSE = 0.39, higher than original, but close to fail\\nTime VERIFY: 0.75 seconds\\n')
    _write(root / 'PREDICT' / 'PREDICT_data.dat', 'Summary of results\\n- 5-fold CV : R2 = 0.81, MAE = 0.22, RMSE = 0.30\\n- Test : R2 = 0.74, MAE = 0.28, RMSE = 0.38\\nWARNING! 1 outlier was detected: mol-2\\nTime PREDICT: 1.10 seconds\\n')
    _write(root / 'PREDICT' / 'predictions.csv', 'code_name,y,y_pred,y_pred_sd\\nmol-1,1.2,1.1,0.1\\nmol-2,2.5,2.2,0.3\\n')
    _write(root / 'ROBERT_report.pdf', '%PDF-1.4 synthetic test file')
    return train

def _report_score_tree(root: Path) -> Path:
    train = _write(root / 'train.csv', 'code_name,y,d1\\nmol-1,1.2,3\\n')
    _write(root / 'PREDICT' / 'PREDICT_data.dat', '------- Starting model with all variables (No PFI) -------\\n   -  Test : R2 = 0.96\\n')
    _write(root / 'VERIFY' / 'VERIFY_data.dat', 'VERIFY completed\\n')
    _write(root / 'ROBERT_report.pdf', '%PDF-1.4')
    return train

def test_discovers_aqme_outputs_and_descriptor_counts(tmp_path: Path):
    source = _write(tmp_path / 'molecules.csv', 'code_name,SMILES,target\\na,CC,1\\nb,CCC,2\\n')
    _write(tmp_path / 'AQME' / 'QDESCP_data.dat', 'QDESCP processed 2 molecules\\nWARNING! Molecule b used fallback geometry\\nTime QDESCP: 4.2 seconds\\n')
    _write(tmp_path / 'AQME-ROBERT_interpret_molecules.csv', 'code_name,SMILES,target,xtb_HOMO,rdkit_MW\\na,CC,1,-4.2,30\\nb,CCC,2,-4.0,44\\n')
    snapshot = WorkflowResultStore().refresh(str(source), '')
    assert snapshot is not None
    assert snapshot.kind == 'AQME'
    assert snapshot.status == 'completed_with_warnings'
    output = next((artifact for artifact in snapshot.artifacts if artifact.path.startswith('AQME-ROBERT_')))
    assert output.rows == 2
    assert output.columns == 5
    assert output.descriptor_count == 2
    assert 'fallback geometry' in ' '.join(snapshot.warnings)
    reloaded = WorkflowResultStore().refresh(str(tmp_path / 'AQME-ROBERT_interpret_molecules.csv'), '')
    reloaded_output = next((artifact for artifact in reloaded.artifacts if artifact.path.startswith('AQME-ROBERT_')))
    assert reloaded_output.descriptor_count == 2

def test_snapshot_reuses_report_score_logic_instead_of_reading_pdf(tmp_path: Path):
    score = {'ML_model': 'RF', 'interp_score_No PFI': 3, 'extrap_score_No PFI': 0, 'scaled_rmse_cv_No PFI': 11.2, 'flawed_mod_score_No PFI': -2, 'diff_scaled_rmse_score_No PFI': 2}
    with patch('gui_easyrob.bot.workflow_results._report_score_function', return_value=lambda *args: score):
        snapshot = WorkflowResultStore().refresh(str(_report_score_tree(tmp_path)), '')
    assert len(snapshot.report_assessments) == 1
    assessment = snapshot.report_assessments[0]
    assert assessment.variant == 'No PFI'
    assert assessment.model == 'RF'
    assert assessment.prediction_type == 'regression'
    assert assessment.interpolation_score == 3
    assert assessment.boundary_score == 0
    assert any((label == 'Scaled CV RMSE (% of target range)' for label, value in assessment.details))
    assert ('VERIFY tests', -2, 0) in assessment.component_scores
    assert ('Test vs CV', 2, 2) in assessment.component_scores
    evidence = render_workflow_evidence(snapshot, 'Summarize this workflow', full=True)
    assert 'Report-derived assessment No PFI: Interpolation score=3/10; Boundary robustness=0/10' in evidence
    assert 'The PDFs themselves were not parsed or sent' in evidence""", ('test_discovers_aqme_outputs_and_descriptor_counts', 'test_snapshot_reuses_report_score_logic_instead_of_reading_pdf'), ())
