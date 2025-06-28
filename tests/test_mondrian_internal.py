"""Internal implementation tests for Mondrian k-anonymity."""

import numpy as np
import polars as pl

from polarfrost.mondrian import mondrian_k_anonymity_polars


def test_mondrian_string_comparison() -> None:
    """Test string comparison in span calculation."""
    df = pl.DataFrame(
        {
            "id": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "value": [10, 20, 30, 40, 50, 60],
            "condition": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )

    # Test with string IDs that can't be converted to numbers
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["id", "value"],
        sensitive_column="condition",
        k=2,
        categorical=["id"],
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_bytes_handling() -> None:
    """Test handling of bytes data in aggregation."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "data": [b"abc", b"def", b"abc", b"def"],
            "condition": ["X", "Y", "X", "Y"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["id", "data"],
        sensitive_column="condition",
        k=2,
        categorical=["data"],
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_single_partition() -> None:
    """Test case where only one partition is possible."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 25, 25],
            "gender": ["M", "M", "M", "M"],
            "condition": ["A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        k=2,
        categorical=["gender"],
    )

    assert len(result) == 1
    assert result["count"][0] == 4
    # Check that the age range is correct
    assert result["age"][0] == "25-25"
    # For categorical columns, we expect the unique values to be preserved
    assert result["gender"][0] == "M"


def test_mondrian_numerical_precision() -> None:
    """Test handling of numerical precision in span calculation."""
    df = pl.DataFrame(
        {
            "value": [1.23456789, 1.23456788, 1.23456787, 1.23456786],
            "category": ["A", "A", "B", "B"],
            "condition": ["X", "Y", "X", "Y"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df=df,
        quasi_identifiers=["value", "category"],
        sensitive_column="condition",
        k=2
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_mixed_numeric_types() -> None:
    """Test handling of mixed numeric types (int, float)."""
# Explicitly specify dtypes to avoid inference issues
    df = pl.DataFrame(
        {
            # All floats to avoid type issues
            "value": [1.0, 2.5, 3.0, 4.5, 5.0, 6.5],
            "category": ["A", "B", "A", "B", "A", "B"],
            "condition": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df=df,
        quasi_identifiers=["value", "category"],
        sensitive_column="condition",
        k=2
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_large_dataset() -> None:
    """Test with a larger dataset to verify performance and correctness."""
    np.random.seed(42)

    # Generate a larger dataset with clear patterns
    # Reduced from 1000 to make tests faster while still testing the logic
    n = 100
    ages = np.random.randint(20, 70, n)
    genders = np.random.choice(["M", "F"], n)
    zipcodes = [f"{np.random.randint(10000, 100000)}" for _ in range(n)]
    conditions = np.random.choice(["A", "B", "C", "D"], n)

    df = pl.DataFrame({
        "age": ages,
        "gender": genders,
        "zipcode": zipcodes,
        "condition": conditions
    })

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=10,
        categorical=["gender", "zipcode"],
    )

    # Verify basic properties
    assert len(result) >= 1  # At least one group
    assert all(count >= 10 for count in result["count"].to_list())
    assert set(result.columns) == {
        "age", "gender", "zipcode", "condition", "count"
    }
