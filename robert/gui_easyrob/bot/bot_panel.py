"""Right-side conversational panel for robBOT."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
from decimal import Decimal, ROUND_HALF_UP

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QMovie, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .llm_providers import provider_names
from .token_usage import MAX_USER_PROMPT_TOKENS, TokenUsage, estimate_token_count

__all__ = ["API_SETUP_GIF_PATH", "BotPanel", "SUGGESTED_QUESTIONS"]


API_SETUP_GIF_PATH = Path(__file__).resolve().parents[1] / "assets" / "gif_easyrob.gif"
_SESSION_COST_NOTICE_USD = 0.05
_SESSION_COST_HIGH_USD = 0.25


SUGGESTED_QUESTIONS = (
    "How does the GUI overview work?",
    "How do I start from a CSV file?",
    "How do I start from ChemDraw?",
    "How do I make predictions for new molecules?",
    "How do I generate descriptors without training a model?",
    "How do I check my own machine learning model?",
    "How do I use robBOT?",
)
SUGGESTED_TUTORIAL_IDS = (
    "overview",
    "csv",
    "chemdraw",
    "predictions",
    "descriptors",
    "check_model",
    "robbot",
)

from .answer_metadata import AnswerMetadata, EvidenceOrigin, SourceCitation

class _AnimatedGifLabel(QLabel):
    """Display native GIF frames with smooth, aspect-preserving scaling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._movie: QMovie | None = None
        self.setAlignment(Qt.AlignCenter)

    def set_movie(self, movie: QMovie) -> None:
        movie.setParent(self)
        self._movie = movie
        movie.frameChanged.connect(self._update_frame)
        self._update_frame()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._movie is not None and self._movie.isValid():
            if self._movie.state() != QMovie.Running:
                self._movie.start()

    def hideEvent(self, event) -> None:
        if self._movie is not None and self._movie.state() == QMovie.Running:
            self._movie.stop()
        super().hideEvent(event)

    def _update_frame(self, _frame_number: int = -1) -> None:
        if self._movie is None:
            return
        frame = self._movie.currentPixmap()
        if frame.isNull() or self.size().isEmpty():
            return
        self.setPixmap(frame.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_frame()


_MARKDOWN_CODE_RE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!\w)_([^_\n]+)_(?!\w)")
_MARKDOWN_UNORDERED_ITEM_RE = re.compile(r"^\s*[-+*]\s+(.+)$")
_MARKDOWN_ORDERED_ITEM_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")


def _render_markdown(text: str) -> str:
    """Render a small safe Markdown subset for provider and user messages."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    fenced_code: list[str] = []
    def stash_fence(match: re.Match[str]) -> str:
        fenced_code.append(match.group(1))
        return f"\n@@EASYROB_FENCE_{len(fenced_code) - 1}@@\n"
    normalized = re.sub(r"(?m)^```[^\n]*\n([\s\S]*?)(?:^```[ \t]*$|\Z)", stash_fence, normalized)
    normalized = re.sub(r"(?m)^[ \t]*-[ \t]*\\[ \t]*(?:\n|$)", "", normalized)
    normalized = re.sub(r"\\[ \t]*\n", "\n", normalized)
    normalized = re.sub(r"\\(?=[*_-])", "", normalized)
    normalized = re.sub(r"&#(?:x0*20|32);", " ", normalized, flags=re.IGNORECASE)
    safe_lines = [escape(line, quote=False) for line in normalized.split("\n")]

    code_fragments: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        token = f"@@EASYROB_CODE_{len(code_fragments)}@@"
        code_fragments.append(match.group(1))
        return token

    def render_inline(value: str) -> str:
        rendered = _MARKDOWN_CODE_RE.sub(stash_code, value)
        rendered = _MARKDOWN_BOLD_RE.sub(
            lambda match: f"<strong>{match.group(1) or match.group(2)}</strong>",
            rendered,
        )
        rendered = _MARKDOWN_ITALIC_RE.sub(
            lambda match: f"<em>{match.group(1) or match.group(2)}</em>",
            rendered,
        )
        def render_link(match: re.Match[str]) -> str:
            from html import unescape
            citation = SourceCitation.create(unescape(match.group(1)), unescape(match.group(2)))
            if citation is None:
                return match.group(1)
            return f'<a href="{escape(citation.url, quote=True)}">{match.group(1)}</a>'
        rendered = re.sub(r"【(https?://[^\s】]+)】", r" [Source](\1)", rendered)
        rendered = re.sub(r"\[([^\]\n]+)\]\(([^\s)]+)\)", render_link, rendered)
        for index, fragment in enumerate(code_fragments):
            rendered = rendered.replace(
                f"@@EASYROB_CODE_{index}@@",
                f"<code>{fragment}</code>",
            )
        return rendered

    rendered_lines: list[str] = []
    open_list: str | None = None
    paragraph: list[str] = []
    def flush_paragraph() -> None:
        if paragraph:
            rendered_lines.append('<p style="margin: 0 0 8px 0;">' + "<br>".join(paragraph) + "</p>")
            paragraph.clear()
    for line in safe_lines:
        fence = re.fullmatch(r"@@EASYROB_FENCE_(\d+)@@", line)
        if fence and int(fence.group(1)) < len(fenced_code):
            flush_paragraph()
            if open_list:
                rendered_lines.append(f"</{open_list}>")
                open_list = None
            rendered_lines.append("<pre>" + escape(fenced_code[int(fence.group(1))], quote=False).rstrip("\n") + "</pre>")
            continue
        unordered = _MARKDOWN_UNORDERED_ITEM_RE.match(line)
        ordered = _MARKDOWN_ORDERED_ITEM_RE.match(line)
        if unordered or ordered:
            flush_paragraph()
            list_name = "ul" if unordered else "ol"
            item_text = (unordered or ordered).group(1)
            if open_list != list_name:
                if open_list:
                    rendered_lines.append(f"</{open_list}>")
                start_number = int(re.match(r"\s*(\d+)", line).group(1)) if ordered else 1
                start_attribute = f' start="{start_number}"' if ordered and start_number != 1 else ""
                rendered_lines.append(f"<{list_name}{start_attribute}>")
                open_list = list_name
            rendered_lines.append(f"<li>{render_inline(item_text)}</li>")
            continue
        if not line.strip():
            flush_paragraph()
            continue
        if open_list:
            rendered_lines.append(f"</{open_list}>")
            open_list = None
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            flush_paragraph()
            rendered_lines.append('<p style="margin: 8px 0 4px 0;"><strong>' + render_inline(heading.group(1)) + "</strong></p>")
        elif line:
            paragraph.append(render_inline(line))
        else:
            flush_paragraph()

    if open_list:
        rendered_lines.append(f"</{open_list}>")
    flush_paragraph()
    return "".join(rendered_lines)


class BotPanel(QFrame):
    send_requested = Signal(str, str)
    workflow_summary_requested = Signal()
    close_requested = Signal()
    clear_requested = Signal()
    local_prepare_requested = Signal()
    api_save_requested = Signal()
    api_delete_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._applying_styles = False
        self._busy = False
        self._workflow_summary_available = False
        self._selected_tutorial_id = ""
        self._messages: list[str] = []
        self._message_records: list[tuple[str, str]] = []
        self._theme_colors: dict[str, str] = {}
        self._session_usage = TokenUsage()
        self.source_links: list[QLabel] = []

        self.setObjectName("easyrobBotPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumWidth(520)
        # Let the panel consume the available width when the floating window is
        # maximized; the minimum keeps the normal compact layout usable.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()
        self._apply_styles()
        self._update_mode_ui(self.mode_combo.currentText())
        self._update_send_enabled()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        self.chrome = QFrame()
        self.chrome.setObjectName("botChrome")
        chrome_layout = QVBoxLayout(self.chrome)
        chrome_layout.setContentsMargins(0, 0, 0, 0)
        chrome_layout.setSpacing(0)
        root_layout.addWidget(self.chrome)

        header = QFrame()
        header.setObjectName("botHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 12)
        header_layout.setSpacing(10)

        identity_layout = QVBoxLayout()
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.setSpacing(2)
        self.title_label = QLabel("robBOT")
        self.title_label.setObjectName("botTitle")
        subtitle_label = QLabel("Ask robBOT about workflows, results, and AQME/ROBERT documentation.")
        subtitle_label.setObjectName("botSubtitle")
        identity_layout.addWidget(self.title_label)
        identity_layout.addWidget(subtitle_label)
        header_layout.addLayout(identity_layout, 1)

        self.clear_button = QPushButton("Clear")
        self.close_button = QPushButton("Close")
        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.close_button.clicked.connect(self.close_requested.emit)
        header_layout.addWidget(self.clear_button)
        header_layout.addWidget(self.close_button)

        chrome_layout.addWidget(header)

        self.content_tabs = QTabWidget()
        self.content_tabs.setObjectName("botContentTabs")
        self.chat_tab = QWidget()
        self.chat_tab.setObjectName("botChatTab")
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName("botSettingsTab")
        self.content_tabs.addTab(self.chat_tab, "Chat")
        self.content_tabs.addTab(self.settings_tab, "Settings")

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setObjectName("botSettingsScroll")

        self.settings_body = QWidget()
        settings_body_layout = QVBoxLayout(self.settings_body)
        settings_body_layout.setContentsMargins(0, 4, 0, 0)
        settings_body_layout.setSpacing(10)

        self.settings_help_button = self._help_button(
            "Settings help",
            "Use Settings to choose which robBOT mode answers your questions.\n\n"
            "Local AI uses the packaged offline model in this ROBERT environment.\n\n"
            "Cloud AI is the default and uses an external provider with your API key.\n\n"
            "Heuristic gives stable rule-based help from the GUI state, popups, logs, and packaged docs.",
        )
        settings_header = QWidget()
        settings_header_layout = QHBoxLayout(settings_header)
        settings_header_layout.setContentsMargins(0, 0, 0, 0)
        settings_header_layout.setSpacing(6)
        settings_header_label = QLabel("robBOT settings")
        settings_header_label.setObjectName("botSectionLabel")
        settings_header_layout.addWidget(settings_header_label)
        settings_header_layout.addWidget(self.settings_help_button)
        settings_header_layout.addStretch(1)
        settings_body_layout.addWidget(settings_header)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Local AI", "Cloud AI", "Heuristic"])
        self.mode_combo.setCurrentText("Cloud AI")
        self.mode_combo.currentTextChanged.connect(self._update_mode_ui)
        self.mode_help_button = self._help_button(
            "Mode help",
            "Local AI runs the packaged local model in this ROBERT environment.\n\n"
            "Cloud AI is the default and uses an external provider with your API key.\n\n"
            "Heuristic is deterministic and uses only GUI state, popups, logs, and packaged knowledge.",
        )
        settings_body_layout.addWidget(self._labeled_row("Mode", self.mode_combo, self.mode_help_button))

        self.local_controls = QWidget()
        local_layout = QVBoxLayout(self.local_controls)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(10)

        self.local_status_label = QLabel("Not downloaded")
        self.local_detail_label = QLabel("")
        self.local_detail_label.setWordWrap(True)
        self.local_progress_bar = QProgressBar()
        self.local_progress_bar.setTextVisible(False)
        self.local_progress_bar.setRange(0, 0)
        self.local_progress_bar.hide()
        self.local_action_button = QPushButton("Download model")
        local_layout.addWidget(self._labeled_row("Local model", self.local_status_label))
        local_layout.addWidget(self.local_detail_label)
        local_layout.addWidget(self.local_progress_bar)
        local_layout.addWidget(self.local_action_button)
        self.local_action_button.clicked.connect(self.local_prepare_requested.emit)
        settings_body_layout.addWidget(self.local_controls)

        self.llm_controls = QWidget()
        llm_layout = QVBoxLayout(self.llm_controls)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(10)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(provider_names())
        groq_index = self.provider_combo.findText("Groq")
        if groq_index >= 0:
            self.provider_combo.setCurrentIndex(groq_index)
        self.provider_combo.currentTextChanged.connect(
            lambda _text: self._update_mode_indicator(self.mode_combo.currentText())
        )
        self.provider_help_button = self._help_button(
            "Provider help",
            "Groq is a fast cloud option for trying robBOT with an API key.\n\n"
            "Other providers use their own accounts and billing rules.",
        )
        llm_layout.addWidget(self._labeled_row("Provider", self.provider_combo, self.provider_help_button))

        self.web_search_checkbox = QCheckBox(
            "Allow automatic web search when information is missing or current"
        )
        self.web_search_checkbox.setChecked(True)
        self.web_search_checkbox.setToolTip(
            "Supported cloud providers may perform at most one billed search per question."
        )
        llm_layout.addWidget(self.web_search_checkbox)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Enter API key for the selected provider")
        self.api_key_help_button = self._help_button(
            "API key help",
            "API key means the secret token from the selected provider account.\n\n"
            "easyROB uses it only to send this robBOT question to that provider.",
        )
        llm_layout.addWidget(self._labeled_row("API key", self.api_key_edit, self.api_key_help_button))

        api_actions = QWidget()
        api_actions_layout = QHBoxLayout(api_actions)
        api_actions_layout.setContentsMargins(0, 0, 0, 0)
        api_actions_layout.setSpacing(8)
        self.save_api_button = QPushButton("Save API")
        self.delete_api_button = QPushButton("Delete API")
        self.save_api_button.clicked.connect(self.api_save_requested.emit)
        self.delete_api_button.clicked.connect(self.api_delete_requested.emit)
        api_actions_layout.addWidget(self.save_api_button)
        api_actions_layout.addWidget(self.delete_api_button)
        api_actions_layout.addStretch(1)
        llm_layout.addWidget(api_actions)

        self.api_setup_guide = QTextBrowser()
        self.api_setup_guide.setObjectName("botApiGuide")
        self.api_setup_guide.setOpenExternalLinks(True)
        self.api_setup_guide.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.api_setup_guide.setMaximumHeight(132)
        self.api_setup_guide.setHtml(
            "<b>Groq quick setup</b><br>"
            "1. Open <a href='https://console.groq.com/keys'>console.groq.com/keys</a>.<br>"
            "2. Sign in or create a Groq account.<br>"
            "3. Create an API key in <b>API Keys</b>.<br>"
            "4. Paste it here, choose <b>Groq</b>, and ask again."
        )

        api_setup_media = QWidget()
        self.api_setup_media = api_setup_media
        api_setup_media_layout = QVBoxLayout(api_setup_media)
        api_setup_media_layout.setContentsMargins(0, 0, 0, 0)
        api_setup_media_layout.setSpacing(10)
        self.api_setup_gif_label = _AnimatedGifLabel()
        self.api_setup_gif_label.setObjectName("botApiSetupGif")
        self.api_setup_gif_label.setMinimumHeight(180)
        self.api_setup_gif_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.api_setup_movie = QMovie(str(API_SETUP_GIF_PATH))
        if self.api_setup_movie.isValid():
            self.api_setup_gif_label.set_movie(self.api_setup_movie)
        else:
            self.api_setup_gif_label.setText("Groq setup animation unavailable.\nUse the written steps.")
            self.api_setup_gif_label.setWordWrap(True)
        api_setup_media_layout.addWidget(self.api_setup_guide)
        api_setup_media_layout.addWidget(self.api_setup_gif_label)
        self.api_setup_open_button = QPushButton("Open visual guide")
        self.api_setup_open_button.clicked.connect(self._show_api_setup_dialog)
        api_setup_media_layout.addWidget(self.api_setup_open_button, alignment=Qt.AlignCenter)
        llm_layout.addWidget(api_setup_media)

        settings_body_layout.addWidget(self.llm_controls)
        settings_body_layout.addStretch(1)

        self.settings_scroll.setWidget(self.settings_body)
        settings_tab_layout = QVBoxLayout(self.settings_tab)
        settings_tab_layout.setContentsMargins(12, 12, 12, 12)
        settings_tab_layout.setSpacing(10)
        settings_tab_layout.addWidget(self.settings_scroll)

        conversation_shell = QFrame()
        conversation_shell.setObjectName("botConversationShell")
        conversation_layout = QVBoxLayout(conversation_shell)
        conversation_layout.setContentsMargins(12, 12, 12, 12)
        conversation_layout.setSpacing(10)

        conversation_label = QLabel("Conversation")
        conversation_label.setObjectName("botSectionLabel")
        conversation_layout.addWidget(conversation_label)

        self.chat_local_status_shell = QFrame()
        self.chat_local_status_shell.setObjectName("botLocalStatusShell")
        chat_local_status_layout = QVBoxLayout(self.chat_local_status_shell)
        chat_local_status_layout.setContentsMargins(10, 10, 10, 10)
        chat_local_status_layout.setSpacing(6)
        self.chat_local_status_title = QLabel("Local AI status")
        self.chat_local_status_title.setObjectName("botSectionLabel")
        self.chat_local_status_text = QLabel("")
        self.chat_local_status_text.setWordWrap(True)
        self.chat_local_progress_bar = QProgressBar()
        self.chat_local_progress_bar.setTextVisible(False)
        self.chat_local_progress_bar.setRange(0, 0)
        self.chat_local_progress_bar.hide()
        chat_local_status_layout.addWidget(self.chat_local_status_title)
        chat_local_status_layout.addWidget(self.chat_local_status_text)
        chat_local_status_layout.addWidget(self.chat_local_progress_bar)
        self.chat_local_status_shell.hide()
        conversation_layout.addWidget(self.chat_local_status_shell)

        self.history_view = QTextBrowser()
        self.history_view.setOpenExternalLinks(True)
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("The bot conversation will appear here.")
        self.history_view.setObjectName("botHistoryView")
        self.history_view.setMinimumHeight(320)
        conversation_layout.addWidget(self.history_view, stretch=1)

        workflow_summary_shell = QFrame()
        workflow_summary_shell.setObjectName("botComposerShell")
        workflow_summary_layout = QHBoxLayout(workflow_summary_shell)
        workflow_summary_layout.setContentsMargins(10, 8, 10, 8)
        workflow_summary_layout.setSpacing(10)
        self.workflow_summary_status = QLabel("No workflow results detected")
        self.workflow_summary_status.setWordWrap(True)
        self.workflow_summary_button = QPushButton("Summarize workflow")
        self.workflow_summary_button.setEnabled(False)
        self.workflow_summary_button.setToolTip(
            "This action becomes available when easyROB detects an available workflow result."
        )
        self.workflow_summary_button.clicked.connect(self.workflow_summary_requested.emit)
        workflow_summary_layout.addWidget(self.workflow_summary_status, stretch=1)
        workflow_summary_layout.addWidget(self.workflow_summary_button)
        conversation_layout.addWidget(workflow_summary_shell)

        composer_shell = QFrame()
        composer_shell.setObjectName("botComposerShell")
        composer_layout = QHBoxLayout(composer_shell)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        composer_layout.setSpacing(10)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Ask about your workflow, results, or AQME/ROBERT")
        self.input_edit.textChanged.connect(self._update_send_enabled)
        self.input_edit.textEdited.connect(self._clear_selected_tutorial)
        self.input_edit.returnPressed.connect(self._emit_send_requested)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._emit_send_requested)
        composer_layout.addWidget(self.input_edit, stretch=1)
        composer_layout.addWidget(self.send_button)
        conversation_layout.addWidget(composer_shell)

        self.prompt_usage_label = QLabel()
        self.prompt_usage_label.setObjectName("botPromptUsageLabel")
        conversation_layout.addWidget(self.prompt_usage_label)
        self.request_stage_label = QLabel("")
        self.request_stage_label.setObjectName("botPromptUsageLabel")
        conversation_layout.addWidget(self.request_stage_label)

        self.advanced_options_toggle = QToolButton(self.chat_tab)
        self.advanced_options_toggle.setObjectName("botAdvancedOptionsToggle")
        self.advanced_options_toggle.setText("Advanced options")
        self.advanced_options_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_options_toggle.setArrowType(Qt.DownArrow)
        self.advanced_options_toggle.setCheckable(True)
        self.advanced_options_toggle.setChecked(True)
        self.advanced_options_toggle.setCursor(Qt.PointingHandCursor)
        conversation_layout.addWidget(self.advanced_options_toggle)

        self.advanced_options_body = QWidget(self.chat_tab)
        self.advanced_options_body.setObjectName("botAdvancedOptionsBody")
        advanced_options_layout = QVBoxLayout(self.advanced_options_body)
        advanced_options_layout.setContentsMargins(0, 0, 0, 0)
        advanced_options_layout.setSpacing(10)

        self.usage_shell = QFrame(self.advanced_options_body)
        self.usage_shell.setObjectName("botUsageShell")
        usage_layout = QVBoxLayout(self.usage_shell)
        usage_layout.setContentsMargins(10, 8, 10, 8)
        usage_layout.setSpacing(4)
        usage_header = QHBoxLayout()
        usage_header.setContentsMargins(0, 0, 0, 0)
        usage_title = QLabel("Token usage")
        usage_title.setObjectName("botSectionLabel")
        self.reset_usage_button = QPushButton("Reset usage")
        self.reset_usage_button.clicked.connect(self.reset_usage)
        usage_header.addWidget(usage_title)
        usage_header.addStretch(1)
        usage_header.addWidget(self.reset_usage_button)
        self.last_usage_label = QLabel("Last request: no usage yet")
        self.session_usage_label = QLabel("Session: 0 tokens · $0.0000")
        self.last_usage_label.setWordWrap(True)
        self.session_usage_label.setWordWrap(True)
        self.session_usage_warning = QLabel("")
        self.session_usage_warning.setObjectName("botUsageWarning")
        self.session_usage_warning.setWordWrap(True)
        self.session_usage_warning.hide()
        usage_layout.addLayout(usage_header)
        usage_layout.addWidget(self.last_usage_label)
        usage_layout.addWidget(self.session_usage_label)
        usage_layout.addWidget(self.session_usage_warning)
        self.sources_title_label = QLabel("Sources")
        self.sources_title_label.setObjectName("botSectionLabel")
        self.sources_title_label.hide()
        self.sources_layout = QVBoxLayout()
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_layout.setSpacing(2)
        self.sources_scroll = QScrollArea()
        self.sources_scroll.setWidgetResizable(True)
        self.sources_scroll.setFrameShape(QFrame.NoFrame)
        self.sources_scroll.setMaximumHeight(120)
        source_container = QWidget()
        source_container.setLayout(self.sources_layout)
        self.sources_scroll.setWidget(source_container)
        self.sources_scroll.hide()
        usage_layout.addWidget(self.sources_title_label)
        usage_layout.addWidget(self.sources_scroll)
        advanced_options_layout.addWidget(self.usage_shell)

        self.suggested_label = QLabel("Suggested questions", self.advanced_options_body)
        self.suggested_label.setObjectName("botSectionLabel")
        advanced_options_layout.addWidget(self.suggested_label)

        self.suggested_combo = QComboBox(self.advanced_options_body)
        self.suggested_combo.setObjectName("botSuggestedQuestionCombo")
        self.suggested_combo.addItem("Choose a suggested question...")
        for question, tutorial_id in zip(SUGGESTED_QUESTIONS, SUGGESTED_TUTORIAL_IDS):
            self.suggested_combo.addItem(question, tutorial_id)
        self.suggested_combo.currentIndexChanged.connect(self._apply_suggested_question_from_index)
        advanced_options_layout.addWidget(self.suggested_combo)
        conversation_layout.addWidget(self.advanced_options_body)
        self.advanced_options_toggle.toggled.connect(self._set_advanced_options_visible)
        self._set_advanced_options_visible(True)

        chat_tab_layout = QVBoxLayout(self.chat_tab)
        chat_tab_layout.setContentsMargins(12, 12, 12, 12)
        chat_tab_layout.setSpacing(10)
        chat_tab_layout.addWidget(conversation_shell, stretch=1)

        chrome_layout.addWidget(self.content_tabs, stretch=1)

    def _set_advanced_options_visible(self, expanded: bool) -> None:
        """Show or hide the optional questions and token accounting controls."""
        expanded = bool(expanded)
        self.advanced_options_body.setVisible(expanded)
        self.advanced_options_toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _resize_api_setup_movie(self) -> None:
        if not hasattr(self, "api_setup_movie") or not self.api_setup_movie.isValid():
            return
        available_width = self.api_setup_media.width()
        if available_width <= 0:
            return
        width = min(600, available_width)
        height = round(width * 9 / 16)
        if self.api_setup_gif_label.height() != height:
            self.api_setup_gif_label.setFixedHeight(height)
        self.api_setup_gif_label.update()

    def _show_api_setup_dialog(self) -> None:
        if not self.api_setup_movie.isValid():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Groq quick setup")
        dialog.resize(820, 520)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(16, 16, 16, 16)
        dialog_layout.setSpacing(10)

        gif_label = _AnimatedGifLabel()
        gif_label.setMinimumSize(720, 405)
        gif_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        movie = QMovie(str(API_SETUP_GIF_PATH))
        gif_label.set_movie(movie)
        dialog_layout.addWidget(gif_label, stretch=1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        dialog_layout.addWidget(close_button, alignment=Qt.AlignRight)
        dialog._api_setup_movie = movie
        movie.start()
        dialog.exec()

    def _apply_styles(self) -> None:
        if self._applying_styles:
            return
        self._applying_styles = True
        palette = self.palette()
        is_dark = palette.color(QPalette.Window).lightness() < 128

        def css(color: QColor) -> str:
            return color.name(QColor.HexRgb)

        window = palette.color(QPalette.Window)
        base = palette.color(QPalette.Base)
        text = palette.color(QPalette.Text)
        button = palette.color(QPalette.Button)
        button_text = palette.color(QPalette.ButtonText)
        mid = palette.color(QPalette.Mid)
        light = palette.color(QPalette.Light)
        highlighted_text = palette.color(QPalette.HighlightedText)
        robert_purple = QColor("#6A0DAD")
        robert_purple_hover = QColor("#7B2CBF")

        chrome_bg_start = base.lighter(108) if is_dark else base.lighter(102)
        chrome_bg_end = window.lighter(112) if is_dark else window.darker(102)
        header_start = robert_purple_hover
        header_end = robert_purple
        subtitle = highlighted_text if is_dark else light.lighter(105)
        section = text.lighter(125) if is_dark else text.darker(115)
        conversation_bg = base.lighter(112) if is_dark else base.lighter(102)
        composer_bg = base.lighter(106) if is_dark else base.lighter(100)
        suggestion_bg = button.lighter(112) if is_dark else button.lighter(104)
        suggestion_hover = suggestion_bg.darker(108) if is_dark else suggestion_bg.darker(104)
        disabled_text = mid.lighter(120) if is_dark else mid.darker(110)
        user_bubble = conversation_bg.lighter(104) if is_dark else conversation_bg
        user_border = mid.lighter(115) if is_dark else QColor("#DEE1E6")
        user_role = text.lighter(145) if is_dark else text.darker(120)
        bot_bubble = base.lighter(120) if is_dark else QColor("#F7F2FC")
        bot_border = robert_purple.lighter(120) if is_dark else QColor("#DACBE8")
        bot_role = robert_purple.lighter(155) if is_dark else robert_purple

        self._theme_colors = {
            "user_bubble": css(user_bubble),
            "user_text": css(text),
            "user_border": css(user_border),
            "user_role": css(user_role),
            "bot_bubble": css(bot_bubble),
            "bot_text": css(text),
            "bot_border": css(bot_border),
            "bot_role": css(bot_role),
        }

        try:
            self.setStyleSheet(
                f"""
            #easyrobBotPanel {{
                background: {css(window)};
            }}
            #botChrome {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {css(chrome_bg_start)}, stop:1 {css(chrome_bg_end)});
                border: 1px solid {css(mid)};
                border-radius: 20px;
            }}
            #botHeader {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {css(header_start)}, stop:1 {css(header_end)});
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
            }}
            #botTitle {{
                color: {css(highlighted_text)};
                font-size: 18px;
                font-weight: 700;
            }}
            #botSubtitle {{
                color: {css(subtitle)};
                font-size: 11px;
            }}
            #botSettingsTab, #botChatTab, #botConversationShell {{
                background: transparent;
            }}
            #botContentTabs::pane {{
                border: none;
                background: transparent;
            }}
            #botContentTabs QTabBar::tab {{
                background: {css(button)};
                color: {css(button_text)};
                border: 1px solid {css(mid)};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 18px;
                margin-right: 4px;
            }}
            #botContentTabs QTabBar::tab:selected {{
                background: {css(base)};
                color: {css(text)};
            }}
            #botSectionLabel {{
                color: {css(section)};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            #botHistoryView {{
                background: {css(conversation_bg)};
                color: {css(text)};
                border: 1px solid {css(mid)};
                border-radius: 16px;
                padding: 6px;
            }}
            #botLocalStatusShell {{
                background: {css(base.lighter(108) if is_dark else base.lighter(102))};
                color: {css(text)};
                border: 1px solid {css(mid)};
                border-radius: 14px;
            }}
            #botUsageShell {{
                background: {css(base.lighter(108) if is_dark else base.lighter(102))};
                color: {css(text)};
                border: 1px solid {css(mid)};
                border-radius: 14px;
            }}
            #botPromptUsageLabel {{
                color: {css(disabled_text)};
                font-size: 11px;
            }}
            #botPromptUsageLabel[warning="true"] {{
                color: #B7791F;
                font-weight: 600;
            }}
            #botPromptUsageLabel[limit_exceeded="true"] {{
                color: #C53030;
                font-weight: 700;
            }}
            #botUsageWarning {{
                color: #B7791F;
                font-weight: 600;
            }}
            #botComposerShell {{
                background: {css(composer_bg)};
                border: 1px solid {css(mid)};
                border-radius: 16px;
                padding: 8px;
            }}
            #botApiGuide {{
                background: {css(base.lighter(108) if is_dark else base.lighter(102))};
                color: {css(text)};
                border: 1px solid {css(mid)};
                border-radius: 12px;
                padding: 8px;
            }}
            QLineEdit, QComboBox {{
                background: {css(base)};
                border: 1px solid {css(mid)};
                border-radius: 10px;
                padding: 8px 10px;
                color: {css(text)};
            }}
            QPushButton {{
                background: {css(button)};
                border: 1px solid {css(mid)};
                border-radius: 10px;
                padding: 8px 12px;
                color: {css(button_text)};
            }}
            QPushButton:hover {{
                background: {css(button.darker(108) if is_dark else button.darker(104))};
            }}
            QPushButton:disabled {{
                color: {css(disabled_text)};
                background: {css(button)};
            }}
            #botSuggestionButton {{
                text-align: left;
                background: {css(suggestion_bg)};
                border: 1px solid {css(mid)};
                border-radius: 12px;
                padding: 10px 12px;
            }}
            #botSuggestionButton:hover {{
                background: {css(suggestion_hover)};
            }}
            QToolButton {{
                background: transparent;
                border: none;
                color: {css(text)};
                font-weight: 700;
                padding: 4px 0;
            }}
            #botHelpButton {{
                border: 1px solid {css(mid)};
                border-radius: 9px;
                min-width: 18px;
                min-height: 18px;
                max-width: 18px;
                max-height: 18px;
                padding: 0;
                font-weight: 700;
            }}
            """
            )
            self._rerender_history()
        finally:
            self._applying_styles = False

    def _help_button(self, title: str, message: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("botHelpButton")
        button.setText("?")
        button.setCursor(Qt.PointingHandCursor)
        button.setProperty("help_title", title)
        button.setProperty("help_message", message)
        button.clicked.connect(lambda checked=False, b=button: self._show_help_dialog(b))
        return button

    def _show_help_dialog(self, button: QToolButton) -> None:
        title = str(button.property("help_title") or "Assistant help")
        message = str(button.property("help_message") or "")
        QMessageBox.information(self, title, message)

    def _labeled_row(self, label_text: str, widget: QWidget, help_button: QWidget | None = None) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(6)
        label = QLabel(label_text)
        label.setObjectName("botSectionLabel")
        label_row.addWidget(label)
        if help_button is not None:
            label_row.addWidget(help_button)
        label_row.addStretch(1)
        layout.addLayout(label_row)
        layout.addWidget(widget)
        return wrapper

    def _apply_suggested_question(self, question: str) -> None:
        self.input_edit.setText(question)
        self.input_edit.setFocus(Qt.OtherFocusReason)
        self.input_edit.selectAll()

    def _apply_suggested_question_from_index(self, index: int) -> None:
        if index <= 0:
            return
        self._selected_tutorial_id = str(self.suggested_combo.itemData(index) or "")
        self._apply_suggested_question(self.suggested_combo.itemText(index))
        self.suggested_combo.blockSignals(True)
        self.suggested_combo.setCurrentIndex(0)
        self.suggested_combo.blockSignals(False)

    def _clear_selected_tutorial(self, _text: str = "") -> None:
        self._selected_tutorial_id = ""

    def _update_mode_indicator(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized == "cloud ai":
            provider_label = str(self.provider_combo.currentText()).strip()
            label = provider_label or "Cloud AI"
        elif normalized == "heuristic":
            label = "Heuristic"
        else:
            label = "Local AI"
        self.title_label.setText(f"robBOT ({label})")
        if self._message_records:
            self._rerender_history()

    def _display_role_label(self, role: str) -> str:
        normalized_role = str(role).strip().lower()
        if normalized_role == "user":
            return str(role).strip() or "User"
        normalized_mode = str(self.mode_combo.currentText()).strip().lower()
        if normalized_mode == "cloud ai":
            active_label = str(self.provider_combo.currentText()).strip() or "Cloud AI"
        elif normalized_mode == "heuristic":
            active_label = "Heuristic"
        else:
            active_label = "Local AI"
        return f"robBOT ({active_label})"

    def _update_mode_ui(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        is_cloud = normalized == "cloud ai"
        is_local = normalized == "local ai"
        self._update_mode_indicator(mode)
        self.local_controls.setHidden(not is_local)
        self.llm_controls.setHidden(not is_cloud)
        self.local_action_button.setEnabled(not self._busy and is_local)
        self.provider_combo.setEnabled(not self._busy and is_cloud)
        self.web_search_checkbox.setEnabled(not self._busy and is_cloud)
        self.api_key_edit.setEnabled(not self._busy and is_cloud)
        self.save_api_button.setEnabled(not self._busy and is_cloud)
        self.delete_api_button.setEnabled(not self._busy and is_cloud and bool(self.delete_api_button.property("has_saved_api")))
        self.api_setup_guide.setHidden(not is_cloud)
        self.api_setup_gif_label.setHidden(not is_cloud)
        self.api_setup_open_button.setHidden(not is_cloud)
        self.api_setup_open_button.setEnabled(not self._busy and self.api_setup_movie.isValid())
        if is_cloud and self.api_setup_movie.isValid() and self.api_setup_gif_label.isVisible():
            if self.api_setup_movie.state() != QMovie.Running:
                self.api_setup_movie.start()
        elif self.api_setup_movie.state() == QMovie.Running:
            self.api_setup_movie.stop()

    def _update_send_enabled(self) -> None:
        has_text = bool(self.input_edit.text().strip())
        token_count = estimate_token_count(self.input_edit.text())
        warning = token_count >= int(MAX_USER_PROMPT_TOKENS * 0.75)
        limit_exceeded = token_count > MAX_USER_PROMPT_TOKENS
        self.prompt_usage_label.setText(
            f"Estimated question size: {token_count:,} / {MAX_USER_PROMPT_TOKENS:,} tokens"
        )
        self.prompt_usage_label.setProperty("warning", warning)
        self.prompt_usage_label.setProperty("limit_exceeded", limit_exceeded)
        self.prompt_usage_label.style().unpolish(self.prompt_usage_label)
        self.prompt_usage_label.style().polish(self.prompt_usage_label)
        self.send_button.setEnabled(has_text and not self._busy and not limit_exceeded)
        self.input_edit.setEnabled(not self._busy)

    def _emit_send_requested(self) -> None:
        if self._busy:
            return
        question = self.input_edit.text().strip()
        if not question:
            return
        tutorial_id = self._selected_tutorial_id
        self._selected_tutorial_id = ""
        self.send_requested.emit(question, tutorial_id)

    def _message_html(self, role: str, text: str) -> str:
        is_user = role.strip().lower() == "user"
        bubble_color = self._theme_colors["user_bubble"] if is_user else self._theme_colors["bot_bubble"]
        text_color = self._theme_colors["user_text"] if is_user else self._theme_colors["bot_text"]
        border_color = self._theme_colors["user_border"] if is_user else self._theme_colors["bot_border"]
        role_color = self._theme_colors["user_role"] if is_user else self._theme_colors["bot_role"]
        safe_role = escape(self._display_role_label(role))
        safe_text = _render_markdown(text)
        # QTextBrowser supports table cells and padding, but not CSS bubble layouts.
        return (
            f"<table width='100%' cellspacing='0' cellpadding='12' "
            f"style='margin-top: 10px; margin-bottom: 12px; border: 1px solid {border_color};'>"
            f"<tr><td bgcolor='{bubble_color}' style='color: {text_color};'>"
            f"<p style='font-size: 11px; font-weight: 700; color: {role_color}; margin: 0 0 8px 0;'>{safe_role}</p>"
            f"<div style='font-size: 14px;'>{safe_text}</div>"
            f"</td></tr></table><p style='margin: 0; font-size: 4px;'>&nbsp;</p>"
        )

    def append_message(self, role: str, text: str) -> None:
        self._message_records.append((role, text))
        self._messages.append(self._message_html(role, text))
        self.history_view.setHtml("".join(self._messages))
        scrollbar = self.history_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_history(self) -> None:
        self._messages.clear()
        self._message_records.clear()
        self.history_view.clear()

    @staticmethod
    def _format_cost(cost: float | None) -> str:
        if cost is None:
            return "cost unavailable"
        rounded = Decimal(str(cost)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return f"~${rounded:.4f}"

    def record_usage(self, usage: TokenUsage) -> TokenUsage:
        if not isinstance(usage, TokenUsage):
            usage = TokenUsage(
                prompt_tokens=max(0, int(getattr(usage, "prompt_tokens", 0) or 0)),
                completion_tokens=max(0, int(getattr(usage, "completion_tokens", 0) or 0)),
                total_tokens=max(0, int(getattr(usage, "total_tokens", 0) or 0)),
                cached_prompt_tokens=max(0, int(getattr(usage, "cached_prompt_tokens", 0) or 0)),
                assessment_prompt_tokens=max(0, int(getattr(usage, "assessment_prompt_tokens", 0) or 0)),
                assessment_completion_tokens=max(0, int(getattr(usage, "assessment_completion_tokens", 0) or 0)),
                search_content_tokens=max(0, int(getattr(usage, "search_content_tokens", 0) or 0)),
                search_requests=max(0, int(getattr(usage, "search_requests", 0) or 0)),
                estimated_model_cost_usd=getattr(usage, "estimated_model_cost_usd", None),
                estimated_search_cost_usd=getattr(usage, "estimated_search_cost_usd", None),
                estimated_cost_usd=getattr(usage, "estimated_cost_usd", None),
                is_estimated=getattr(usage, "is_estimated", True) is True,
                provider=str(getattr(usage, "provider", "") or ""),
                model=str(getattr(usage, "model", "") or ""),
            )
        self._session_usage = self._session_usage + usage
        estimate_prefix = "Estimated " if usage.is_estimated else ""
        cached = (
            f", {usage.cached_prompt_tokens:,} cached"
            if usage.cached_prompt_tokens else ""
        )
        source = f" ({usage.provider})" if usage.provider else ""
        self.last_usage_label.setText(
            f"{estimate_prefix}last request{source}: {usage.total_tokens:,} tokens "
            f"({usage.prompt_tokens:,} input, {usage.completion_tokens:,} output{cached}) · "
            f"{self._format_cost(usage.estimated_cost_usd)}"
        )
        session = self._session_usage
        session_search_label = (
            f"{session.search_requests} search"
            if session.search_requests == 1
            else f"{session.search_requests} searches"
        )
        cost_breakdown = ""
        if (
            session.estimated_model_cost_usd is not None
            and session.estimated_search_cost_usd is not None
        ):
            cost_breakdown = (
                f" (model {self._format_cost(session.estimated_model_cost_usd)} + "
                f"search {self._format_cost(session.estimated_search_cost_usd)})"
            )
        self.session_usage_label.setText(
            f"Session: {session.total_tokens:,} tokens "
            f"({session.prompt_tokens:,} input, {session.completion_tokens:,} output) | "
            f"{session_search_label} | total {self._format_cost(session.estimated_cost_usd)}"
            f"{cost_breakdown}"
        )
        if session.estimated_cost_usd is not None and session.estimated_cost_usd >= _SESSION_COST_HIGH_USD:
            self.session_usage_warning.setText(
                f"High usage: estimated session cost has reached {self._format_cost(session.estimated_cost_usd)}. "
                "Check your provider dashboard and project limits."
            )
            self.session_usage_warning.show()
        elif session.estimated_cost_usd is not None and session.estimated_cost_usd >= _SESSION_COST_NOTICE_USD:
            self.session_usage_warning.setText(
                f"Usage notice: estimated session cost has reached {self._format_cost(session.estimated_cost_usd)}."
            )
            self.session_usage_warning.show()
        elif session.estimated_cost_usd is not None:
            self.session_usage_warning.hide()
        elif session.total_tokens >= 50_000:
            milestone = (session.total_tokens // 50_000) * 50_000
            self.session_usage_warning.setText(
                f"High usage: this session has passed {milestone:,} tokens and reliable cost is unavailable. "
                "Check your provider dashboard and project limits."
            )
            self.session_usage_warning.show()
        elif session.total_tokens >= 10_000:
            self.session_usage_warning.setText(
                "Usage notice: this session has passed 10,000 tokens, but reliable cost is unavailable. "
                "You can reset this counter without deleting the chat."
            )
            self.session_usage_warning.show()
        else:
            self.session_usage_warning.hide()
        return usage

    def record_answer_metadata(self, metadata: AnswerMetadata, usage: TokenUsage) -> None:
        """Record usage and render normalized evidence details and safe links."""
        if isinstance(metadata, AnswerMetadata):
            safe_metadata = metadata
        else:
            raw_origin = getattr(metadata, "evidence_origin", "general_knowledge")
            origin_value = getattr(raw_origin, "value", raw_origin)
            try:
                normalized_origin = EvidenceOrigin(str(origin_value))
            except ValueError:
                normalized_origin = EvidenceOrigin.INSUFFICIENT
            normalized_sources = []
            raw_sources = getattr(metadata, "sources", ())
            for raw_source in raw_sources if type(raw_sources) in {list, tuple} else ():
                source = SourceCitation.create(
                    getattr(raw_source, "title", ""),
                    getattr(raw_source, "url", ""),
                    start_index=getattr(raw_source, "start_index", None),
                    end_index=getattr(raw_source, "end_index", None),
                )
                if source is not None:
                    normalized_sources.append(source)
            safe_metadata = AnswerMetadata(
                evidence_origin=normalized_origin,
                sources=tuple(normalized_sources),
                search_error=str(getattr(metadata, "search_error", "") or "")[:240],
            )
        safe_usage = self.record_usage(usage)
        origin_labels = {
            EvidenceOrigin.LOCAL_SYSTEM: "Local system",
            EvidenceOrigin.LOCAL_DOCUMENTATION: "Local documentation",
            EvidenceOrigin.WEB_AND_DOCUMENTATION: "Web + documentation",
            EvidenceOrigin.GENERAL_KNOWLEDGE: "General knowledge",
            EvidenceOrigin.INSUFFICIENT: "Insufficient evidence",
        }
        origin = origin_labels.get(safe_metadata.evidence_origin, "Evidence unavailable")
        provider_model = " / ".join(
            value for value in (safe_usage.provider, safe_usage.model) if value
        ) or "Offline"
        search_label = (
            f"{safe_usage.search_requests} search"
            if safe_usage.search_requests == 1
            else f"{safe_usage.search_requests} searches"
        )
        summary = (
            f"{origin} | {provider_model} | {safe_usage.total_tokens:,} tokens | "
            f"{search_label} | {self._format_cost(safe_usage.estimated_cost_usd)}"
        )
        details = [
            f"{safe_usage.prompt_tokens:,} input",
            f"{safe_usage.completion_tokens:,} output",
        ]
        if safe_usage.cached_prompt_tokens:
            details.append(f"{safe_usage.cached_prompt_tokens:,} cached")
        assessment_tokens = (
            safe_usage.assessment_prompt_tokens + safe_usage.assessment_completion_tokens
        )
        if assessment_tokens:
            details.append(f"{assessment_tokens:,} assessment")
        if safe_usage.search_content_tokens:
            details.append(f"{safe_usage.search_content_tokens:,} search-content tokens")
        if safe_usage.estimated_model_cost_usd not in {None, 0.0}:
            details.append(f"model {self._format_cost(safe_usage.estimated_model_cost_usd)}")
        if safe_usage.estimated_search_cost_usd not in {None, 0.0}:
            details.append(f"search {self._format_cost(safe_usage.estimated_search_cost_usd)}")
        if safe_metadata.search_error:
            details.append(safe_metadata.search_error[:240])
        self.last_usage_label.setText(f"{summary}\n({', '.join(details)})")
        self._render_sources(safe_metadata.sources)

    def _render_sources(self, sources: tuple[SourceCitation, ...]) -> None:
        for label in self.source_links:
            self.sources_layout.removeWidget(label)
            label.deleteLater()
        self.source_links.clear()
        seen: set[str] = set()
        for raw_source in sources:
            if not isinstance(raw_source, SourceCitation):
                continue
            source = SourceCitation.create(raw_source.title, raw_source.url)
            if source is None:
                continue
            key = source.url.casefold().rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            label = QLabel(
                f'<a href="{escape(source.url, quote=True)}">'
                f'{escape(source.title[:90])}</a>'
            )
            label.setToolTip(source.url)
            label.setOpenExternalLinks(True)
            label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            label.setWordWrap(True)
            self.sources_layout.addWidget(label)
            self.source_links.append(label)
        self.sources_title_label.setVisible(bool(self.source_links))
        self.sources_scroll.setFixedHeight(min(120, 40 * len(self.source_links)))
        self.sources_scroll.setVisible(bool(self.source_links))

    def set_request_stage(self, stage: str) -> None:
        """Show a bounded progress message for the active request."""
        messages = {
            "checking_documentation": "Checking documentation...",
            "searching_current_information": "Searching current information...",
            "generating_answer": "Generating answer...",
        }
        self.request_stage_label.setText(messages.get(str(stage or ""), ""))

    def reset_usage(self) -> None:
        self._session_usage = TokenUsage()
        self.last_usage_label.setText("Last request: no usage yet")
        self.session_usage_label.setText("Session: 0 tokens · $0.0000")
        self.session_usage_warning.clear()
        self.session_usage_warning.hide()
        self._render_sources(())

    def _rerender_history(self) -> None:
        if not self._message_records:
            return
        self._messages = [self._message_html(role, text) for role, text in self._message_records]
        self.history_view.setHtml("".join(self._messages))
        scrollbar = self.history_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in {QEvent.PaletteChange, QEvent.ApplicationPaletteChange}:
            self._apply_styles()
        super().changeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_api_setup_movie()

    def set_local_status(self, status: str, detail: str | None = None) -> None:
        self.local_status_label.setText(status)
        if detail is not None:
            self.local_detail_label.setText(detail)
        detail_text = str(detail if detail is not None else self.local_detail_label.text()).strip()
        normalized = str(status).strip().lower()
        if normalized == "ready":
            self.local_action_button.setText("Model ready")
        elif normalized == "downloaded":
            self.local_action_button.setText("Load model")
        elif normalized == "error":
            self.local_action_button.setText("Retry")
        elif normalized == "not downloaded":
            self.local_action_button.setText("Download model")
        else:
            self.local_action_button.setText("Preparing model...")

        show_chat_status = normalized == "preparing..." or self.chat_local_progress_bar.isVisible()
        self.chat_local_status_text.setText(f"{status}. {detail_text}".strip().strip("."))
        self.chat_local_status_shell.setVisible(show_chat_status and bool(self.chat_local_status_text.text().strip()))

    def set_local_progress_active(self, is_active: bool) -> None:
        self.local_progress_bar.setVisible(bool(is_active))
        self.chat_local_progress_bar.setVisible(bool(is_active))
        if is_active:
            self.local_progress_bar.setRange(0, 0)
            self.chat_local_progress_bar.setRange(0, 0)
            self.chat_local_status_shell.setVisible(True)
        else:
            self.local_progress_bar.setRange(0, 1)
            self.local_progress_bar.setValue(0)
            self.chat_local_progress_bar.setRange(0, 1)
            self.chat_local_progress_bar.setValue(0)
            self.chat_local_status_shell.setVisible(False)

    def set_busy(self, is_busy: bool) -> None:
        self._busy = bool(is_busy)
        self.suggested_combo.setEnabled(not self._busy)
        self.mode_combo.setEnabled(not self._busy)
        self.content_tabs.setTabEnabled(1, not self._busy)
        self.clear_button.setEnabled(not self._busy)
        self.workflow_summary_button.setEnabled(
            self._workflow_summary_available and not self._busy
        )
        self._update_mode_ui(self.mode_combo.currentText())
        self._update_send_enabled()

    def set_workflow_summary_available(
        self,
        available: bool,
        *,
        status: str = "",
        workflow_kind: str = "",
    ) -> None:
        """Update the summary action from the current result snapshot."""
        self._workflow_summary_available = bool(available)
        self.workflow_summary_button.setEnabled(self._workflow_summary_available and not self._busy)
        if not self._workflow_summary_available:
            self.workflow_summary_status.setText("No workflow results detected")
            self.workflow_summary_button.setText("Summarize workflow")
            self.workflow_summary_button.setToolTip(
                "This action becomes available when easyROB detects an available workflow result."
            )
            return
        status_label = {
            "completed": "completed",
            "completed_with_warnings": "completed with warnings",
            "failed": "failed",
            "partial": "partial results",
            "running": "running",
        }.get(str(status or "").strip().lower(), str(status or "result").replace("_", " "))
        kind_label = str(workflow_kind or "Workflow").strip()
        is_robert_report = "ROBERT" in kind_label.upper()
        self.workflow_summary_button.setText(
            "Summarize report" if is_robert_report else "Summarize workflow"
        )
        self.workflow_summary_status.setText(f"{kind_label}: {status_label}")
        self.workflow_summary_button.setToolTip(
            "Explain the ROBERT report from its structured DAT and CSV evidence without uploading the PDF."
            if is_robert_report else
            "Create a grounded summary from the detected workflow logs and CSV outputs."
        )

    def set_saved_api_state(self, has_saved_api: bool) -> None:
        self.delete_api_button.setProperty("has_saved_api", bool(has_saved_api))
        self.delete_api_button.setEnabled(
            not self._busy
            and str(self.mode_combo.currentText()).strip().lower() == "cloud ai"
            and bool(has_saved_api)
        )
