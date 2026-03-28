"""TimeBaseUla public package exports."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from timebaseula.models.auto import AutoTimeBase, AutoTimeBaseTrend
from timebaseula.models.timebase import TimeBase, TimeBaseTrend

__all__ = [
    "TimeBase",
    "TimeBaseTrend",
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "__version__",
]
