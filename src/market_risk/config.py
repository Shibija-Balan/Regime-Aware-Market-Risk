"""Project defaults."""

from collections import OrderedDict

DEFAULT_WEIGHTS = OrderedDict(
    {
        "SPY": 0.20,
        "TLT": 0.20,
        "LQD": 0.20,
        "GLD": 0.20,
        "FXB": 0.20,
    }
)
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_VAR_WINDOW = 250
DEFAULT_REGIME_WINDOW = 60
DEFAULT_EWMA_LAMBDA = 0.94
