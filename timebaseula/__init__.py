"""TimeBaseUla."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from timebaseula.models.timebase import TimeBase, TimeBaseTrend

__all__ = ["TimeBase", "TimeBaseTrend", "__version__"]
