"""Test the public package surface."""

import subprocess
import sys
from importlib import reload
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

EXPECTED_EXPORTS = {
    "TimeBase",
    "TimeBaseTrend",
    "AutoTimeBase",
    "AutoTimeBaseTrend",
    "__version__",
}


class TestInit:
    """Test the module initialization and version information."""

    @patch("importlib.metadata.version", side_effect=PackageNotFoundError)
    def test_version_not_found(self, mock_version: MagicMock) -> None:
        """Test `__version__` when the package is not found."""
        del mock_version
        if "timebaseula" in sys.modules:
            del sys.modules["timebaseula"]

        import timebaseula

        reload(timebaseula)
        assert timebaseula.__version__ == "unknown"

    @patch("importlib.metadata.version")
    def test_version_found(self, mock_version: MagicMock) -> None:
        """Test `__version__` when the package is found."""
        mock_version.return_value = "0.1.0"
        if "timebaseula" in sys.modules:
            del sys.modules["timebaseula"]

        import timebaseula

        reload(timebaseula)
        assert timebaseula.__version__ == "0.1.0"

    def test_public_exports_match_supported_api(self) -> None:
        """The package should expose only the supported public API."""
        if "timebaseula" in sys.modules:
            del sys.modules["timebaseula"]

        import timebaseula

        assert set(timebaseula.__all__) == EXPECTED_EXPORTS
        assert hasattr(timebaseula, "AutoTimeBase")
        assert hasattr(timebaseula, "AutoTimeBaseTrend")
        assert not hasattr(timebaseula, "make_synthetic_series")
        assert not hasattr(timebaseula, "profile_dataset")
        assert not hasattr(timebaseula, "recommend_timebase_kwargs")

    def test_package_import_does_not_eagerly_import_auto_module(self) -> None:
        """Importing the package should not import Ray-backed auto wrappers."""
        code = (
            "import sys; import timebaseula; "
            "print('timebaseula.models.auto' in sys.modules)"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            text=True,
        )

        assert result.stdout.strip() == "False"
