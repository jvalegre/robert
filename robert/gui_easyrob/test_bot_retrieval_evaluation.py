"""Tests for the offline evidence retrieval benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from evaluate_bot_retrieval import evaluate_cases
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_rag import KnowledgeChunk, SearchResult


def test_evaluation_reports_ranked_cases_and_language_metrics():
    cases = [
        {
            "id": "english-hit",
            "language": "en",
            "question": "What does QDESCP do in AQME?",
            "expected_evidence": ["module-aqme-qdescp"],
        },
        {
            "id": "spanish-hit",
            "language": "es",
            "question": "¿Qué hace QDESCP en AQME?",
            "expected_evidence": ["module-aqme-qdescp"],
        },
        {
            "id": "unmatched",
            "language": "en",
            "question": "What does QDESCP do in AQME?",
            "expected_evidence": ["intentionally-missing-evidence"],
        },
    ]

    report = evaluate_cases(cases)

    assert report["summary"]["count"] == 3
    assert report["summary"]["hits_at_3"] == 2
    assert report["summary"]["hit_rate_at_3"] == 2 / 3
    assert report["summary"]["hit_rate_at_1"] == 2 / 3
    assert report["by_language"]["es"]["hit_rate_at_3"] == 1.0
    assert report["cases"][0]["rank"] == 1
    assert report["cases"][2]["rank"] is None
    assert report["cases"][0]["top_evidence"]


def test_cli_writes_reviewable_results_without_provider_configuration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cases = root / "tests" / "fixtures" / "bot_retrieval_cases.json"
    output = tmp_path / "retrieval.json"

    completed = subprocess.run(
        [sys.executable, str(root / "evaluate_bot_retrieval.py"), "--cases", str(cases), "--output", str(output)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["count"] == 27
    assert set(report["by_language"]) == {"en", "es"}
    assert {case["id"] for case in report["cases"]} == {
        case["id"] for case in json.loads(cases.read_text(encoding="utf-8"))
    }


def test_strong_technical_match_is_not_discarded_by_generic_curated_evidence():
    technical = SearchResult(
        KnowledgeChunk("technical", "code_docstrings", "technical", "AQME", (), "Descriptor algorithm"),
        score=64.0,
    )
    generic = SearchResult(
        KnowledgeChunk("generic", "curated_overrides", "concept", "AQME", (), "AQME overview"),
        score=41.0,
    )

    ordered = BotEngine._prioritize_candidates([technical, generic], "tutorial")

    assert ordered[0].chunk.id == "technical"


def test_near_tied_curated_evidence_keeps_priority():
    technical = SearchResult(
        KnowledgeChunk("technical", "code_docstrings", "technical", "AQME", (), "Descriptor algorithm"),
        score=44.0,
    )
    curated = SearchResult(
        KnowledgeChunk("curated", "curated_overrides", "concept", "AQME", (), "Descriptor guidance"),
        score=41.0,
    )

    ordered = BotEngine._prioritize_candidates([technical, curated], "tutorial")

    assert ordered[0].chunk.id == "curated"


def test_labeled_retrieval_cases_rank_relevant_evidence_first():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads(
        (root / "tests" / "fixtures" / "bot_retrieval_cases.json").read_text(encoding="utf-8")
    )

    report = evaluate_cases(cases)

    misses = [case["id"] for case in report["cases"] if not case["matched_at_1"]]
    assert misses == []
