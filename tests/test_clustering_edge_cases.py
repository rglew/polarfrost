"""Edge case tests for the clustering k-anonymity implementation."""

import polars as pl
import pytest

from polarfrost.clustering import clustering_k_anonymity


def test_clustering_empty_dataframe() -> None:
    """Test clustering with an empty DataFrame."""
    # Create an empty DataFrame
    df = pl.DataFrame({"age": [], "gender": [], "condition": []})

    # Test with empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2,
            method="kmeans",
        )


def test_clustering_invalid_k() -> None:
    """Test clustering with invalid k values."""
    # Create a test DataFrame
    df = pl.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "gender": ["M", "F", "M", "F"],
            "condition": ["A", "B", "A", "B"],
        }
    )

    # Test with k = 0
    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=0,
            method="kmeans",
        )

    # Test with k = -1
    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=-1,
            method="kmeans",
        )

    # Test with k as a string that can't be converted to int
    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k="invalid",  # type: ignore
            method="kmeans",
        )


def test_clustering_missing_columns() -> None:
    """Test clustering with missing columns."""
    # Create a test DataFrame
    df = pl.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "gender": ["M", "F", "M", "F"],
            "condition": ["A", "B", "A", "B"],
        }
    )

    # Test with non-existent quasi-identifier
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "nonexistent"],
            sensitive_column="condition",
            k=2,
            method="kmeans",
        )

    # Test with non-existent sensitive column
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="nonexistent",
            k=2,
            method="fcbg",  # Using a valid method
        )


def test_clustering_unsupported_method() -> None:
    """Test clustering with an unsupported method."""
    # Create a test DataFrame
    df = pl.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "gender": ["M", "F", "M", "F"],
            "condition": ["A", "B", "A", "B"],
        }
    )

    # Test with unsupported method
    with pytest.raises(
        ValueError, match="Unsupported clustering method: unsupported_method"
    ):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2,
            method="unsupported_method",
        )


def test_clustering_with_lazyframe() -> None:
    """Test that clustering works with LazyFrame input."""
    # Create a test LazyFrame
    df = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35, 40, 40],
            "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
            "condition": ["A", "B"] * 4,
        }
    ).lazy()

# This should raise NotImplementedError since clustering is not implemented yet
    with pytest.raises(NotImplementedError):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2,
            # Using a valid method that's not implemented yet
            method="fcbg"
        )


def test_clustering_with_none_values() -> None:
    """Test clustering with None/NA values in the data."""
    # Create a test DataFrame with None values
    df = pl.DataFrame(
        {
            "age": [25, None, 35, 40, 45, 50, 55, 60],
            "gender": ["M", "F", None, "F", "M", "F", "M", None],
            "income": [
                50000, 60000, 70000, None, 90000, 100000, 110000, 120000  # noqa: E501
            ],
            "condition": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

# This should raise NotImplementedError since clustering is not implemented yet
    with pytest.raises(NotImplementedError):
        clustering_k_anonymity(
            df=df,
            quasi_identifiers=["age", "gender", "income"],
            sensitive_column="condition",
            k=2,
            method="rsc",  # Using a valid method that's not implemented yet
        )
