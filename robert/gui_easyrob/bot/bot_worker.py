"""Background worker for non-blocking EasyROB bot replies."""

from __future__ import annotations

from typing import Mapping, Sequence

from PySide6.QtCore import QThread, Signal

from .bot_context import GuiSnapshot
from .answer_metadata import AnswerMetadata
from .token_usage import TokenUsage

__all__ = ["BotReplyWorker"]


class BotReplyWorker(QThread):
    succeeded = Signal(int, int, str, bool, bool, object, object)
    failed = Signal(int, int, str)
    progressed = Signal(int, int, str)

    def __init__(
        self,
        engine,
        *,
        request_id: int,
        generation: int,
        question: str,
        snapshot: GuiSnapshot,
        mode: str,
        provider: str,
        api_key: str,
        model: str | None = None,
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        tutorial_id: str | None = None,
        web_search_enabled: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self._interruption_requested = False
        self.request_id = request_id if type(request_id) is int else 0
        self.generation = generation if type(generation) is int else 0
        self.question = question if type(question) is str else ""
        self.snapshot = snapshot
        self.mode = mode
        self.provider = provider
        self.api_key = api_key if type(api_key) is str else ""
        self.model = model
        self.tutorial_id = tutorial_id.strip() if type(tutorial_id) is str else ""
        self.web_search_enabled = web_search_enabled is True
        records = conversation_history if type(conversation_history) in {list, tuple} else ()
        self.conversation_history = tuple(
            {"role": record["role"], "content": record["content"], **(
                {"memory_eligible": record["memory_eligible"]}
                if "memory_eligible" in record else {}
            )}
            for record in records
            if type(record) is dict
            and type(record.get("role")) is str
            and type(record.get("content")) is str
        )

    def requestInterruption(self) -> None:
        self._interruption_requested = True
        super().requestInterruption()

    def _is_interrupted(self) -> bool:
        return self._interruption_requested or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            if self._is_interrupted():
                return
            if hasattr(self.engine, "answer_with_metadata"):
                answer_kwargs = dict(
                    question=self.question,
                    snapshot=self.snapshot,
                    mode=self.mode,
                    provider=self.provider,
                    api_key=self.api_key,
                    model=self.model,
                    conversation_history=self.conversation_history,
                    web_search_enabled=self.web_search_enabled,
                    progress_callback=self._report_progress,
                )
                if self.tutorial_id:
                    answer_kwargs["tutorial_id"] = self.tutorial_id
                answer_result = self.engine.answer_with_metadata(**answer_kwargs)
                if type(answer_result) is dict:
                    raw_answer = answer_result.get("text", "")
                    answer = raw_answer if type(raw_answer) is str else ""
                    auto_reset_context = answer_result.get("auto_reset_context", False) is True
                    memory_eligible = answer_result.get("memory_eligible", True) is True
                    raw_usage = answer_result.get("usage")
                    raw_metadata = answer_result.get("metadata")
                else:
                    raw_answer = getattr(answer_result, "text", "")
                    answer = raw_answer if type(raw_answer) is str else ""
                    auto_reset_context = getattr(answer_result, "auto_reset_context", False) is True
                    memory_eligible = getattr(answer_result, "memory_eligible", True) is True
                    raw_usage = getattr(answer_result, "usage", None)
                    raw_metadata = getattr(answer_result, "metadata", None)
                usage = raw_usage if isinstance(raw_usage, TokenUsage) else TokenUsage()
                metadata = raw_metadata if isinstance(raw_metadata, AnswerMetadata) else AnswerMetadata()
            else:
                answer_kwargs = dict(
                    question=self.question,
                    snapshot=self.snapshot,
                    mode=self.mode,
                    provider=self.provider,
                    api_key=self.api_key,
                    model=self.model,
                    conversation_history=self.conversation_history,
                )
                if self.tutorial_id:
                    answer_kwargs["tutorial_id"] = self.tutorial_id
                raw_answer = self.engine.answer(**answer_kwargs)
                answer = raw_answer if type(raw_answer) is str else ""
                auto_reset_context = False
                memory_eligible = True
                usage = TokenUsage()
                metadata = AnswerMetadata()
        except Exception:
            if not self._is_interrupted():
                self.failed.emit(
                    self.request_id,
                    self.generation,
                    "The bot could not complete the request.",
                )
            return
        finally:
            self.api_key = ""

        if self._is_interrupted():
            return
        self.succeeded.emit(
            self.request_id,
            self.generation,
            answer,
            auto_reset_context,
            memory_eligible,
            usage,
            metadata,
        )

    def _report_progress(self, stage: str) -> None:
        if self._is_interrupted() or type(stage) is not str:
            return
        self.progressed.emit(self.request_id, self.generation, stage)
