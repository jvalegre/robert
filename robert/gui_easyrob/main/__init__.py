"""Main window package for easyROB.

This package exposes the top-level application window so the entry point can
instantiate the interface without importing internal implementation details.
"""

from .window import EasyROB

__all__ = ["EasyROB"]
