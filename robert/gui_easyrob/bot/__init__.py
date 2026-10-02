"""Lazy bot package exports used while the scaffold is still incomplete."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["BotEngine", "BotPanel", "BotWindow"]

_EXPORTS = {
    "BotEngine": "bot_engine",
    "BotPanel": "bot_panel",
    "BotWindow": "bot_window",
}


if TYPE_CHECKING:
    from .bot_engine import BotEngine
    from .bot_panel import BotPanel
    from .bot_window import BotWindow


def _load_export(name: str) -> Any:
    module_name = _EXPORTS[name]
    try:
        module = import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{module_name}":
            raise ImportError(
                f"{name} is not available yet; {__name__}.{module_name} has not been added."
            ) from exc
        raise

    return getattr(module, name)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = _load_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
