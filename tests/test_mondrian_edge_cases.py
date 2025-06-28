"""Tests for edge cases in the Mondrian k-anonymity implementation."""

from typing import List, cast

import polars as pl
import pytest

from polarfrost.mondrian import (
    mondrian_k_anonymity,
    mondrian_k_anonymity_polars,
)


def test_mondrian_empty_dataframe() -> None:
    """Test handling of empty DataFrames."""
    df = pl.DataFrame({"age": [], "gender": [], "income": []})

    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        mondrian_k_anonymity(
            df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="income",
            k=2
        )


def test_mondrian_k_larger_than_dataset() -> None:
    """Test when k is larger than the dataset size."""
    df = pl.DataFrame(
        {
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "income": [50000, 60000, 70000],
        }
    )

    # Larger than dataset size
    result = mondrian_k_anonymity(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="income",
        k=5,
    )

    # Should return a single group with all records
    # Using shape[0] for DataFrame compatibility
    assert result.shape[0] == 1
    assert result["count"][0] == 3


def test_mondrian_single_column() -> None:
    """Test with a single quasi-identifier column."""
    df: pl.DataFrame = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35],
            "income": [50000, 55000, 60000, 65000, 70000, 75000],
        }
    )

    result = cast(pl.DataFrame, mondrian_k_anonymity(
        df,
        quasi_identifiers=["age"],
        sensitive_column="income",
        k=2
    ))

    # Should have at least one group
    # Using shape[0] for DataFrame compatibility
    assert result.shape[0] > 0
    # Each group should have at least k records
    count_values: List[int] = result["count"].to_list()
    assert all(count >= 2 for count in count_values)


def test_mondrian_all_identical() -> None:
    """Test with all records being identical."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 25, 25],
            "gender": ["M", "M", "M", "M"],
            "income": [50000, 50000, 50000, 50000],
        }
    )

    result = mondrian_k_anonymity(
        df, quasi_identifiers=["age", "gender"], sensitive_column="income", k=2
    )

    # Should return a single group with all records
    # Using shape[0] for DataFrame compatibility
    assert result.shape[0] == 1
    assert result["count"][0] == 4


def test_mondrian_with_nulls() -> None:
    """Test handling of null values in the data."""
    df: pl.DataFrame = pl.DataFrame(
        {
            "age": [25, None, 30, 30, None, 35],
            "gender": ["M", "F", None, "F", "M", None],
            "income": [50000, 55000, 60000, 65000, 70000, 75000],
        }
    )

    result = cast(pl.DataFrame, mondrian_k_anonymity(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="income",
        k=2,
        categorical=["gender"],
    ))

    # Should complete without errors
    # Using shape[0] for DataFrame compatibility
    assert result.shape[0] > 0
    # All groups should satisfy k-anonymity
    count_values: List[int] = result["count"].to_list()
    assert all(count >= 2 for count in count_values)


def test_mondrian_lazyframe_input() -> None:
    """Test that the function works with LazyFrame input."""
    df: pl.LazyFrame = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35],
            "gender": ["M", "M", "F", "F", "M", "M"],
            "income": [50000, 55000, 60000, 65000, 70000, 75000],
        }
    ).lazy()

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="income",
        k=2,
        categorical=["gender"]
    )

    # Should complete without errors
    # Using shape[0] for DataFrame compatibility
    assert result.shape[0] > 0
    # All groups should satisfy k-anonymity
    count_values: List[int] = result["count"].to_list()
    assert all(count >= 2 for count in count_values)


def test_mondrian_invalid_k() -> None:
    """Test with invalid k values."""
    df = pl.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity(df, ["age"], "income", k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity(df, ["age"], "income", k=-1)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity(
            df,
            ["age"],
            "income",
            k="not_an_integer"  # type: ignore[arg-type]
        )
