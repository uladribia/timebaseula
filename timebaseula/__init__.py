"""TimeBaseUla public package exports."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from timebaseula.models.timebase import TimeBase, TimeBaseTrend

_AUTO_EXPORTS = {
    "AutoTimeBase": "AutoTimeBase",
    "AutoTimeBaseTrend": "AutoTimeBaseTrend",
}


def __getattr__(name: str) -> Any:
    """Lazily load Ray-backed auto wrappers on first access."""
    if name in _AUTO_EXPORTS:
        value = getattr(import_module("timebaseula.models.auto"), _AUTO_EXPORTS[name])
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "TimeBase",
    "TimeBaseTrend",
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "__version__",
]
