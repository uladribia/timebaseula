"""TimeBaseUla public package exports."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from timebaseula.models.timebase import (
    AutoTimeBase,
    AutoTimeBaseTrend,
    TimeBase,
    TimeBaseTrend,
)

__all__ = [
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "TimeBase",
    "TimeBaseTrend",
    "__version__",
]
