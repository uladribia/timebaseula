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
from timebaseula.recommend import (
    DatasetProfile,
    profile_dataset,
    recommend_timebase_kwargs,
    recommend_timebase_trend_kwargs,
)
from timebaseula.synthetic import make_synthetic_series

__all__ = [
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "DatasetProfile",
    "TimeBase",
    "TimeBaseTrend",
    "__version__",
    "make_synthetic_series",
    "profile_dataset",
    "recommend_timebase_kwargs",
    "recommend_timebase_trend_kwargs",
]
