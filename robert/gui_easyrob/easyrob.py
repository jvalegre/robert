"""
Factory for retrieving the main application window class.

This module provides a lightweight abstraction to load the GUI entry class
without triggering Qt initialization at import time.

Execution modes:
- Portable mode (local execution):
  The GUI is loaded from the local folder structure (`main.window`).
- Installed package mode:
  The GUI is loaded from the installed package
  (`robert.gui_easyrob.main.window`).

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
        from main.window import EasyROB
    except ImportError:
        from robert.gui_easyrob.main.window import EasyROB

    return EasyROB
