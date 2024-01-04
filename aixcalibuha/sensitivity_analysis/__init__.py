"""
This package contains classes to perform
sensitivity analysis with.
"""
from .sensitivity_analyzer import SenAnalyzer, _del_duplicates, _rename_tuner_names
from .sobol import SobolAnalyzer
from .morris import MorrisAnalyzer
from .fast import FASTAnalyzer
from .pawn import PAWNAnalyzer
