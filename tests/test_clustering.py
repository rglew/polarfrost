"""Tests for the clustering-based k-anonymity implementation."""

import polars as pl
import pytest

from polarfrost import clustering_k_anonymity


def test_clustering_import() -> None:
    """Test that the clustering function is properly imported."""
    assert callable(clustering_k_anonymity)


def test_clustering_not_implemented() -> None:
    """Test that the clustering function raises NotImplementedError."""
    df = pl.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "gender": ["M", "F", "M", "F"],
            "income": [50000, 60000, 70000, 80000],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Clustering k-anonymity will be implemented soon"
    ):
        clustering_k_anonymity(
            df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="income",
            k=2
        )


def test_clustering_invalid_method() -> None:
    """Test that an invalid method raises a ValueError."""
    df = pl.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30]})

    with pytest.raises(ValueError, match="Unsupported clustering method"):
        clustering_k_anonymity(
            df,
            quasi_identifiers=["age"],
            sensitive_column="income",
            k=2,
            method="invalid_method",
        )


def test_clustering_empty_dataframe() -> None:
    """Test that an empty DataFrame raises a ValueError."""
    df = pl.DataFrame({"age": [], "income": []})

    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        clustering_k_anonymity(
            df,
            quasi_identifiers=["age"],
            sensitive_column="income",
            k=2
        )


def test_clustering_invalid_k() -> None:
    """Test that invalid k values raise appropriate errors."""
    df = pl.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30]})

    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(df, ["age"], "income", k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(df, ["age"], "income", k=-1)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        clustering_k_anonymity(
            df,
            ["age"],
            "income",
            k="not_an_integer"  # type: ignore[arg-type]
        )


def test_clustering_missing_columns() -> None:
    """Test that missing columns raise appropriate errors."""
    df = pl.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30]})

    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        clustering_k_anonymity(df, ["nonexistent"], "income", k=2)

    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        clustering_k_anonymity(df, ["age"], "nonexistent", k=2)


def test_clustering_invalid_input_type() -> None:
    """Test that invalid input types raise appropriate errors."""
    # Test invalid input type
    # Test with intentionally wrong type to trigger the error
    # Intentional type error test - we're checking the runtime validation
    with pytest.raises(
        ValueError,
        match="Input must be a Polars DataFrame or LazyFrame"
    ):
        # Intentional type error test - we're checking the runtime validation
        # mypy: disable-error-code=arg-type,call-arg
        clustering_k_anonymity(
            # Test runtime validation with invalid input type
            "not a dataframe",  # type: ignore[arg-type]
            ["age"],
            "income",
            k=2
        )

    # Test empty quasi_identifiers list
    with pytest.raises(
        ValueError,
        match="quasi_identifiers must be a non-empty list"
    ):
        clustering_k_anonymity(
            pl.DataFrame({"age": [1, 2, 3]}),
            [],  # Empty list should raise error
            "income",
            k=2
        )


def test_clustering_with_lazyframe() -> None:
    """Test that the function works with LazyFrame input."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 35, 35],
            "gender": ["M", "M", "F", "F"],
            "income": [50000, 51000, 60000, 61000],
        }
    ).lazy()

    with pytest.raises(NotImplementedError):
        clustering_k_anonymity(
            df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="income",
            k=2
        )
