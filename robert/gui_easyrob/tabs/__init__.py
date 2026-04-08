"""Public tab components used by the easyROB main window.

The package re-exports each tab widget so the main window can assemble the
interface from a small, stable import surface.
"""

from .predictions import PredictionsTab
from .aqme import AQMETab
from .advanced_options import AdvancedOptionsTab
from .molssi import MolSSIDatabasesTab
from .results import ResultsTab
from .images import ImagesTab

__all__ = [
    "PredictionsTab",
    "AQMETab",
    "AdvancedOptionsTab",
    "MolSSIDatabasesTab",
    "ResultsTab",
    "ImagesTab",
]
