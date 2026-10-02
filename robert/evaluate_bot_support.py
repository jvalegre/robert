"""Run bounded support conversations and export reviewable answers and formatting.

Offline mode exercises the heuristic backend. --live uses the real cloud backend,
requires GROQ_API_KEY (or --saved-key), and can incur provider charges.
"""

from __future__ import annotations

import argparse
import getpass
from dataclasses import asdict
from html import escape
import json
import os
from pathlib import Path
import time

from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.bot_panel import _render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--web", action="store_true", help="Enable provider-billed web search for documentation evaluations.")
    parser.add_argument("--key-stdin", action="store_true", help="Read a transient provider key without saving it.")
    parser.add_argument("--saved-key", action="store_true", help="Use the API key saved in easyROB settings.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/bot_support"))
    parser.add_argument("--interval", type=float, default=25.0, help="Seconds between live calls to respect provider limits.")
    parser.add_argument("--cases", type=Path, help="Alternative conversation fixture.")
    args = parser.parse_args()
    key = os.environ.get("GROQ_API_KEY", "") if args.live else ""
    if args.live and args.key_stdin:
        key = getpass.getpass("Groq API key: ").strip()
    if args.live and args.saved_key:
        from PySide6.QtCore import QSettings
        settings = QSettings("easyROB", "easyROB")
        saved_provider = str(settings.value("bot/provider", "Groq"))
        if saved_provider != "Groq":
            parser.error("The saved provider is not Groq; use GROQ_API_KEY instead.")
        key = str(settings.value("bot/api_key", ""))
    if args.live and not key:
        parser.error("Set GROQ_API_KEY or use --saved-key after saving a Groq key in easyROB.")
    cases_path = args.cases or Path(__file__).parent / "tests/fixtures/bot_support_conversations.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    engine = BotEngine()
    rows = []
    for case in cases:
        history = []
        snapshot = GuiSnapshot(
            "ROBERT", "Full Workflow", "train.csv", "", "yield", "regression",
            False, True, False, case.get("failure", ""), [],
            failure_message=case.get("failure", ""),
        )
        for question in case["questions"]:
            if args.live and rows:
                time.sleep(max(0, args.interval))
            started = time.perf_counter()
            answer = engine.answer_with_metadata(
                question, snapshot, "Cloud AI" if args.live else "Heuristic", "Groq", key,
                conversation_history=history, web_search_enabled=args.web,
            )
            elapsed = time.perf_counter() - started
            text = answer.text.replace(key, "[redacted]") if key else answer.text
            warnings = []
            if not answer.memory_eligible:
                warnings.append("Request failed or answer excluded from memory; inspect manually.")
            if any(len(paragraph.split()) > 120 for paragraph in text.split("\n\n")):
                warnings.append("Long paragraph: consider shorter prose or numbered actions.")
            if "<script" in text.lower():
                warnings.append("Raw HTML returned; verify escaped rendering.")
            rows.append(dict(case=case["id"], question=question, answer=text,
                             seconds=round(elapsed, 3), usage=asdict(answer.usage), metadata=asdict(answer.metadata),
                             warnings=warnings, review=case["review"]))
            history += [dict(role="user", content=question),
                        dict(role="assistant", content=text, memory_eligible=answer.memory_eligible)]
            print(f"{case['id']}: {len(text.split())} words, {elapsed:.2f}s, {len(warnings)} review flags", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    mode = "live-groq" if args.live else "offline-heuristic"
    (args.output / f"{mode}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    cards = []
    for row in rows:
        sources = " ".join(
            f'<a href="{escape(source["url"], quote=True)}">{escape(source["title"])}</a>'
            for source in row.get("metadata", {}).get("sources", [])
        )
        cards.append(f"<article><h2>{escape(row['case'])}</h2><p><b>User:</b> {escape(row['question'])}</p>"
                     f"<section>{_render_markdown(row['answer'])}</section>"
                     f"<aside>Sources: {sources}<br>Review: {escape(row['review'])}<br>{escape('; '.join(row['warnings']))}</aside></article>")
    html = ('<!doctype html><meta charset="utf-8"><title>robBOT support evaluation</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:850px;margin:40px auto;background:#f5f7fa;color:#172033}'
            'article{background:white;padding:24px;margin:20px 0;border:1px solid #dce2e9;border-radius:12px}'
            'aside{color:#596579;font-size:13px;margin-top:20px}pre{white-space:pre-wrap;background:#edf1f5;padding:16px}'
            'li{margin:6px 0}h2{font-size:16px}</style>'
            f'<h1>robBOT: {mode}</h1><p>Human review required: automated flags do not establish factual correctness.</p>'
            + "".join(cards))
    (args.output / f"{mode}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
