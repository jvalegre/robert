"""Local LLM backend powered by a managed llama.cpp server process."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable

import requests

from .answer_metadata import normalize_provider_answer_text

from .token_usage import TokenUsage, estimate_token_count

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

LOCAL_MODEL_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LOCAL_MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_MODEL_DISPLAY_NAME = "Qwen2.5-3B-Instruct (Q4_K_M)"
LOCAL_MODEL_DOWNLOAD_SIZE_DISPLAY = "2.1 GB"
LOCAL_RUNTIME_VERSION = "b9856"
LOCAL_RUNTIME_RELEASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LOCAL_RUNTIME_VERSION}"
)
DEFAULT_LOCAL_MAX_TOKENS = 112
DEFAULT_LOCAL_N_CTX = 1280
DEFAULT_LOCAL_BATCH_SIZE = 256
DEFAULT_LOCAL_UBATCH_SIZE = 64
DEFAULT_LOCAL_TEMPERATURE = 0.1
DEFAULT_LOCAL_TOP_P = 0.9
DEFAULT_LOCAL_REPEAT_PENALTY = 1.1

__all__ = [
    "LOCAL_MODEL_REPO_ID",
    "LOCAL_MODEL_FILENAME",
    "LOCAL_MODEL_DISPLAY_NAME",
    "LOCAL_MODEL_DOWNLOAD_SIZE_DISPLAY",
    "LOCAL_RUNTIME_VERSION",
    "DEFAULT_LOCAL_MAX_TOKENS",
    "LocalLLMError",
    "LocalLLMManager",
    "RuntimeAssetSpec",
]


class LocalLLMError(RuntimeError):
    """Raised when the local model runtime cannot be prepared or queried."""


@dataclass(frozen=True)
class RuntimeAssetSpec:
    platform_key: str
    archive_name: str
    executable_name: str


@dataclass(frozen=True, slots=True)
class LocalAnswer:
    text: str
    usage: TokenUsage


def _missing_huggingface_download(*args: Any, **kwargs: Any) -> str:
    raise LocalLLMError(
        "Local model download failed: huggingface_hub is not installed. "
        "Install it in the active ROBERT environment with "
        "`pip install huggingface_hub`."
    )


try:
    from huggingface_hub import hf_hub_download as _hf_hub_download
except Exception:
    _hf_hub_download = _missing_huggingface_download


def _default_runtime_root() -> Path:
    return Path(sys.prefix).resolve() / "easyrob" / "local_ai"


def _default_model_cache_root() -> Path:
    return _default_runtime_root() / "hf_cache"


def _default_model_download_root() -> Path:
    return _default_runtime_root() / "models"


def _default_thread_count() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(cpu_count - 1, 8))


class LocalLLMManager:
    _RUNTIME_ASSETS: dict[tuple[str, str], RuntimeAssetSpec] = {
        ("windows", "amd64"): RuntimeAssetSpec(
            platform_key="windows-x64",
            archive_name="llama-b9856-bin-win-cpu-x64.zip",
            executable_name="llama-server.exe",
        ),
        ("windows", "x86_64"): RuntimeAssetSpec(
            platform_key="windows-x64",
            archive_name="llama-b9856-bin-win-cpu-x64.zip",
            executable_name="llama-server.exe",
        ),
        ("windows", "arm64"): RuntimeAssetSpec(
            platform_key="windows-arm64",
            archive_name="llama-b9856-bin-win-cpu-arm64.zip",
            executable_name="llama-server.exe",
        ),
        ("darwin", "arm64"): RuntimeAssetSpec(
            platform_key="macos-arm64",
            archive_name="llama-b9856-bin-macos-arm64.zip",
            executable_name="llama-server",
        ),
        ("darwin", "x86_64"): RuntimeAssetSpec(
            platform_key="macos-x64",
            archive_name="llama-b9856-bin-macos-x64.zip",
            executable_name="llama-server",
        ),
        ("linux", "x86_64"): RuntimeAssetSpec(
            platform_key="linux-x64",
            archive_name="llama-b9856-bin-ubuntu-x64.zip",
            executable_name="llama-server",
        ),
        ("linux", "amd64"): RuntimeAssetSpec(
            platform_key="linux-x64",
            archive_name="llama-b9856-bin-ubuntu-x64.zip",
            executable_name="llama-server",
        ),
        ("linux", "aarch64"): RuntimeAssetSpec(
            platform_key="linux-arm64",
            archive_name="llama-b9856-bin-ubuntu-arm64.zip",
            executable_name="llama-server",
        ),
        ("linux", "arm64"): RuntimeAssetSpec(
            platform_key="linux-arm64",
            archive_name="llama-b9856-bin-ubuntu-arm64.zip",
            executable_name="llama-server",
        ),
    }

    def __init__(
        self,
        repo_id: str = LOCAL_MODEL_REPO_ID,
        filename: str = LOCAL_MODEL_FILENAME,
        downloader: Callable[..., str] = _hf_hub_download,
        runtime_root: Path | None = None,
        model_cache_root: Path | None = None,
        model_download_root: Path | None = None,
        runtime_downloader: Callable[[RuntimeAssetSpec, Path, Callable[[str, str], None] | None], str] | None = None,
        transport: Callable[..., Any] = requests.request,
        process_factory: Callable[..., Any] = subprocess.Popen,
        sleep_fn: Callable[[float], None] = time.sleep,
        system_name: str | None = None,
        machine_name: str | None = None,
        port: int = 11435,
        host: str = "127.0.0.1",
        startup_timeout: float = 30.0,
        request_timeout: float = 180.0,
        n_ctx: int = DEFAULT_LOCAL_N_CTX,
        max_tokens: int = DEFAULT_LOCAL_MAX_TOKENS,
        threads: int | None = None,
        batch_size: int = DEFAULT_LOCAL_BATCH_SIZE,
        ubatch_size: int = DEFAULT_LOCAL_UBATCH_SIZE,
        temperature: float = DEFAULT_LOCAL_TEMPERATURE,
        top_p: float = DEFAULT_LOCAL_TOP_P,
        repeat_penalty: float = DEFAULT_LOCAL_REPEAT_PENALTY,
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self._downloader = downloader
        self._runtime_root = Path(runtime_root) if runtime_root is not None else _default_runtime_root()
        self._model_cache_root = (
            Path(model_cache_root)
            if model_cache_root is not None
            else self._runtime_root / "hf_cache"
        )
        self._model_download_root = (
            Path(model_download_root)
            if model_download_root is not None
            else self._runtime_root / "models"
        )
        self._runtime_downloader = runtime_downloader or self._download_runtime_archive
        self._transport = transport
        self._process_factory = process_factory
        self._sleep_fn = sleep_fn
        self._system_name = (system_name or platform.system()).strip()
        self._machine_name = (machine_name or platform.machine()).strip()
        self._port = int(port)
        self._host = host
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._n_ctx = int(n_ctx)
        self._max_tokens = int(max_tokens)
        self._threads = int(threads) if threads is not None else _default_thread_count()
        self._batch_size = int(batch_size)
        self._ubatch_size = int(ubatch_size)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._repeat_penalty = float(repeat_penalty)

        self._model_path: str | None = None
        self._runtime_binary_path: str | None = None
        self._server_process: Any | None = None
        self._ready = False

    @property
    def server_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def runtime_asset_spec(self) -> RuntimeAssetSpec:
        system_key = self._system_name.strip().lower()
        machine_key = self._machine_name.strip().lower()
        spec = self._RUNTIME_ASSETS.get((system_key, machine_key))
        if spec is None:
            raise LocalLLMError(
                f"Local AI runtime is not available for {self._system_name}/{self._machine_name}."
            )
        return spec

    def runtime_exists(self) -> bool:
        runtime_path = self._runtime_binary_path
        if runtime_path and self._is_runtime_complete(Path(runtime_path)):
            return True

        spec = self.runtime_asset_spec()
        expected_path = self._runtime_target_root(spec) / spec.executable_name
        if self._is_runtime_complete(expected_path):
            self._runtime_binary_path = str(expected_path)
            return True

        return False

    def model_exists(self) -> bool:
        if self._model_path:
            return True
        try:
            self._model_download_root.mkdir(parents=True, exist_ok=True)
            self._model_cache_root.mkdir(parents=True, exist_ok=True)
            self._model_path = self._downloader(
                repo_id=self.repo_id,
                filename=self.filename,
                local_files_only=True,
                local_dir=self._model_download_root,
                cache_dir=self._model_cache_root,
            )
        except Exception:
            return False
        return bool(self._model_path)

    def ensure_model_path(self) -> str:
        if self._model_path:
            return self._model_path
        try:
            self._model_download_root.mkdir(parents=True, exist_ok=True)
            self._model_cache_root.mkdir(parents=True, exist_ok=True)
            self._model_path = self._downloader(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=self._model_download_root,
                cache_dir=self._model_cache_root,
            )
        except Exception as exc:
            raise LocalLLMError(f"Local model download failed: {exc}") from exc
        return self._model_path

    def download_model(self, progress_callback: Callable[[str, str], None] | None = None) -> str:
        if progress_callback is not None:
            progress_callback(
                "downloading",
                (
                    "Downloading local model "
                    f"(about {LOCAL_MODEL_DOWNLOAD_SIZE_DISPLAY}). This can take several minutes."
                ),
            )
        expected_path = self._model_download_root / self.filename
        if progress_callback is not None and expected_path.exists():
            try:
                size_mb = expected_path.stat().st_size // (1024 * 1024)
                progress_callback(
                    "downloading",
                    f"Resuming local model download... {size_mb} MB already present.",
                )
            except OSError:
                pass
        return self.ensure_model_path()

    def download_runtime(self, progress_callback: Callable[[str, str], None] | None = None) -> str:
        if self.runtime_exists():
            return str(self._runtime_binary_path)

        spec = self.runtime_asset_spec()
        if progress_callback is not None:
            progress_callback(
                "downloading",
                "Downloading the local AI runtime for this platform...",
            )
        try:
            self._runtime_binary_path = self._runtime_downloader(
                spec,
                self._runtime_target_root(spec),
                progress_callback,
            )
        except Exception as exc:
            raise LocalLLMError(f"Local runtime download failed: {exc}") from exc
        return str(self._runtime_binary_path)

    def load_model(self, force_reload: bool = False) -> Any:
        if self.is_model_ready() and not force_reload:
            return self._server_process
        if force_reload:
            self.shutdown()

        model_path = self.ensure_model_path()
        runtime_path = self.download_runtime()

        try:
            self._server_process = self._start_server(runtime_path, model_path)
            self._wait_until_ready()
        except Exception as exc:
            self.shutdown()
            if isinstance(exc, LocalLLMError):
                raise
            raise LocalLLMError(f"Local model loading failed: {exc}") from exc

        self._ready = True
        return self._server_process

    def is_model_ready(self) -> bool:
        return bool(
            self._ready
            and self._server_process is not None
            and self._server_process.poll() is None
        )

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_response_with_usage(system_prompt, user_prompt).text

    def generate_response_with_usage(self, system_prompt: str, user_prompt: str) -> LocalAnswer:
        self.load_model()
        effective_user_prompt = self._prepare_user_prompt(user_prompt)
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": effective_user_prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "repeat_penalty": self._repeat_penalty,
            "max_tokens": self._max_tokens,
            "stream": False,
            "stop": ["\nUser:", "\nAssistant:", "<think>", "</think>"],
        }
        try:
            response = self._transport(
                "POST",
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = self._sanitize_response_text(self._extract_response_text(data))
        except Exception as exc:
            raise LocalLLMError(f"Local model generation failed: {exc}") from exc
        if not text:
            raise LocalLLMError("Local model returned an empty answer.")
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        has_reported_usage = (
            type(prompt_tokens) is int
            and prompt_tokens >= 0
            and type(completion_tokens) is int
            and completion_tokens >= 0
        )
        if not has_reported_usage:
            prompt_tokens = estimate_token_count(system_prompt) + estimate_token_count(effective_user_prompt)
            completion_tokens = estimate_token_count(text)
            total_tokens = prompt_tokens + completion_tokens
        elif type(total_tokens) is not int or total_tokens < 0:
            total_tokens = prompt_tokens + completion_tokens
        return LocalAnswer(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=0.0,
                is_estimated=not has_reported_usage,
                provider="Local AI",
                model=str(data.get("model") or LOCAL_MODEL_DISPLAY_NAME),
            ),
        )

    def shutdown(self) -> None:
        process = self._server_process
        self._server_process = None
        self._ready = False
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _runtime_target_root(self, spec: RuntimeAssetSpec) -> Path:
        return self._runtime_root / "llama.cpp" / LOCAL_RUNTIME_VERSION / spec.platform_key

    def _is_runtime_complete(self, executable_path: Path) -> bool:
        if not executable_path.exists():
            return False
        if self._system_name.strip().lower() != "windows":
            return True
        return (executable_path.parent / "llama-server-impl.dll").exists()

    def _start_server(self, runtime_path: str, model_path: str) -> Any:
        command = [
            runtime_path,
            "-m",
            model_path,
            "-c",
            str(self._n_ctx),
            "-t",
            str(self._threads),
            "-b",
            str(self._batch_size),
            "-ub",
            str(self._ubatch_size),
            "--host",
            self._host,
            "--port",
            str(self._port),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if self._system_name.lower() == "windows" else 0
        return self._process_factory(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + max(0.0, self._startup_timeout)
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            if self._server_process is not None and self._server_process.poll() is not None:
                raise LocalLLMError("Local runtime failed to start and exited before reporting ready state.")
            try:
                response = self._transport(
                    "GET",
                    f"{self.server_url}/health",
                    timeout=min(5.0, self._request_timeout),
                )
                if getattr(response, "status_code", 200) == 200:
                    response.raise_for_status()
                    return
            except Exception as exc:
                last_error = exc
            self._sleep_fn(0.25)

        if last_error is not None:
            raise LocalLLMError(f"Local runtime health check failed: {last_error}") from last_error
        raise LocalLLMError("Local runtime failed to start before the health check timeout.")

    def _download_runtime_archive(
        self,
        spec: RuntimeAssetSpec,
        target_root: Path,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> str:
        archive_url = f"{LOCAL_RUNTIME_RELEASE_URL}/{spec.archive_name}"
        target_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / spec.archive_name
            with requests.get(archive_url, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", "0") or "0")
                downloaded = 0
                with archive_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            if total > 0:
                                progress_callback(
                                    "downloading",
                                    f"Downloading local AI runtime... {downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB",
                                )
                            else:
                                progress_callback("downloading", "Downloading local AI runtime...")

            extract_root = Path(tmpdir) / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(archive_path), str(extract_root))

            executable_path = self._find_executable(extract_root, spec.executable_name)
            return self._install_runtime_tree(
                source_root=executable_path.parent,
                target_root=target_root,
                executable_name=spec.executable_name,
            )

    def _extract_response_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        text = self._coerce_message_text(message.get("content"))
        if text:
            return text
        text = self._coerce_message_text(message.get("reasoning_content"))
        if text:
            return text
        text = self._coerce_message_text(choices[0].get("text"))
        if text:
            return text
        delta = choices[0].get("delta") or {}
        text = self._coerce_message_text(delta.get("content"))
        if text:
            return text
        text = self._coerce_message_text(delta.get("reasoning_content"))
        return text

    def _prepare_user_prompt(self, user_prompt: str) -> str:
        prompt = str(user_prompt or "").strip()
        if self._supports_no_think() and "/no_think" not in prompt:
            prompt = f"{prompt}\n\n/no_think"
        return prompt

    def _supports_no_think(self) -> bool:
        repo = str(self.repo_id or "").lower()
        return "qwen3-" in repo or "qwen3." in repo

    def _sanitize_response_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        lines = cleaned.splitlines()
        intro_window = " ".join(line.strip().lower() for line in lines[:2])[:220]
        if intro_window and self._contains_reasoning_leak(intro_window):
            cleaned = self._strip_leaked_intro(lines)
            if not cleaned:
                return ""
        return normalize_provider_answer_text(cleaned)

    def _contains_reasoning_leak(self, text: str) -> bool:
        leaked_markers = (
            "okay, the user is asking",
            "let me check",
            "let me think",
            "make sure to mention",
            "keep it concise",
            "avoid jargon",
            "since the user is new",
            "key retrieved knowledge",
            "recent conversation summary",
        )
        lowered = str(text or "").strip().lower()
        return any(marker in lowered for marker in leaked_markers)

    def _strip_leaked_intro(self, lines: list[str]) -> str:
        remaining: list[str] = []
        dropping = True
        for raw_line in lines:
            line = raw_line.strip()
            lowered = line.lower()
            if dropping:
                if not line:
                    continue
                if self._contains_reasoning_leak(lowered):
                    continue
                if lowered in {"sure.", "sure!", "okay.", "ok.", "alright.", "certainly."}:
                    continue
                dropping = False
            remaining.append(line if dropping is False else raw_line)
        return "\n".join(part for part in remaining if part).strip()

    def _coerce_message_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    part = item.strip()
                elif isinstance(item, dict):
                    part = str(
                        item.get("text")
                        or item.get("content")
                        or item.get("value")
                        or ""
                    ).strip()
                else:
                    part = str(item or "").strip()
                if part:
                    parts.append(part)
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            return str(
                value.get("text")
                or value.get("content")
                or value.get("value")
                or ""
            ).strip()
        return str(value or "").strip()

    def _install_runtime_tree(self, source_root: Path, target_root: Path, executable_name: str) -> str:
        if target_root.exists():
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        for item in source_root.iterdir():
            destination = target_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        final_path = target_root / executable_name
        if not final_path.exists():
            raise LocalLLMError(f"Local runtime install is missing {executable_name}.")
        if not executable_name.endswith(".exe"):
            final_path.chmod(final_path.stat().st_mode | 0o111)
        return str(final_path)

    @staticmethod
    def _find_executable(search_root: Path, executable_name: str) -> Path:
        for candidate in search_root.rglob(executable_name):
            if candidate.is_file():
                return candidate
        raise LocalLLMError(f"Local runtime archive did not contain {executable_name}.")
