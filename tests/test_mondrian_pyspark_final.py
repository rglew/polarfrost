"""Mock tests for PySpark implementation of Mondrian k-anonymity."""

from typing import Any, Dict, Optional


class MockSparkConf:
    """Mock implementation of SparkConf for testing."""

    def __init__(self) -> None:
        """Initialize a new MockSparkConf with an empty configuration."""
        self._options: Dict[str, Any] = {}

    def set(self, key: str, value: str) -> None:
        """Set a configuration option.

        Args:
            key: The configuration key
            value: The configuration value
        """
        self._options[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration option.

        Args:
            key: The configuration key
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        return self._options.get(key, default)


def test_mock_spark_conf_set_get() -> None:
    """Test setting and getting configuration values."""
    conf = MockSparkConf()
    conf.set("spark.app.name", "test")
    assert conf.get("spark.app.name") == "test"


def test_mock_spark_conf_default_value() -> None:
    """Test getting a non-existent key returns the default value."""
    conf = MockSparkConf()
    assert conf.get("nonexistent", "default") == "default"


def test_mock_spark_conf_none_default() -> None:
    """Test getting a non-existent key with no default returns None."""
    conf = MockSparkConf()
    assert conf.get("nonexistent") is None
