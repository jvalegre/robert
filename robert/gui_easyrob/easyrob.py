"""
Factory for retrieving the main application window class.

This module provides a lightweight abstraction to load the GUI entry class
without triggering Qt initialization at import time.

Execution modes:
- Installed package mode loads `robert.gui_easyrob.main.window` first.
- Portable mode falls back to the local `main.window` module.

Rationale:
- Keeps GUI imports deferred to avoid issues with QApplication initialization.
- Supports both development (running as a script) and installed usage
  without requiring changes to the codebase.

Note:
- This module assumes a consistent environment; import errors are not masked
  beyond selecting the appropriate execution context.

"""

def get_main_window_class():
    """Factory function to retrieve the main application window class."""
    try:
        from robert.gui_easyrob.main.window import EasyROB
    except ImportError:
        from main.window import EasyROB

    return EasyROB
