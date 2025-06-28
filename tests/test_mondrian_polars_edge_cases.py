"""Additional edge case tests.

Tests for Polars implementation of Mondrian k-anonymity.
"""

import numpy as np
import polars as pl

from polarfrost.mondrian import mondrian_k_anonymity_polars


def test_mixed_data_types() -> None:
    """Test with mixed data types including None values."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 35, 35, None, None, 45, 45],
            "gender": ["M", "M", "F", None, "F", "F", None, None],
            "income": [50000, 51000, 60000, 61000, 70000, 71000, 80000, 81000],
            "condition": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        k=2,
        categorical=["gender"],
    )

    # Verify the result has the expected columns
    assert set(result.columns) == {"age", "gender", "condition", "count"}

    # Verify all groups have at least k=2 records
    assert all(count >= 2 for count in result["count"].to_list())


def test_single_record_partitions() -> None:
    """Test behavior with k=1 where each record could be its own partition."""
    df = pl.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "gender": ["M", "F", "M", "F"],
            "income": [50000, 60000, 70000, 80000],
            "condition": ["A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df=df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        k=1
    )

    # With k=1, should get same number of groups as input rows
    assert len(result) == len(df)
    assert all(count == 1 for count in result["count"].to_list())


def test_all_identical_records() -> None:
    """Test when all records are identical."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 25, 25],
            "gender": ["M", "M", "M", "M"],
            "condition": ["A", "A", "B", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df=df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        k=2
    )

    # Should have exactly one group with all records
    assert len(result) == 1
    assert result["count"][0] == 4


def test_numerical_precision() -> None:
    """Test handling of floating point precision in numerical data."""
    df = pl.DataFrame({
        "value": [1.1, 1.1000001, 2.2, 2.2000001],
        "category": ["A", "A", "B", "B"]
    })

    result = mondrian_k_anonymity_polars(
        df=df,
        quasi_identifiers=["value"],
        sensitive_column="category",
        k=2
    )

    # Should have two groups, each with 2 records
    assert len(result) == 2
    assert all(count >= 2 for count in result["count"].to_list())


def test_large_k_value() -> None:
    """Test behavior when k is larger than the number of records."""
    df = pl.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "gender": ["M", "F", "M", "F"],
            "condition": ["A", "B", "A", "B"],
        }
    )

# When k is larger than the number of records,
    # all records will be in a single group
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        # Larger than number of rows (4)
        k=5,
    )

    # Should have exactly one group with all records
    assert len(result) == 1
    assert result["count"][0] == 4


def test_performance_large_dataset() -> None:
    """Test performance with a larger dataset."""
    # Generate a larger dataset with 1000 records
    np.random.seed(42)
    n = 1000

    df = pl.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "gender": np.random.choice(["M", "F"], n),
            "zipcode": [
                f"{np.random.randint(10000, 99999):05d}"
                for _ in range(n)
            ],
            "income": np.random.normal(50000, 15000, n).astype(int),
            "condition": np.random.choice(["A", "B", "C", "D"], n),
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=10,
        categorical=["gender", "zipcode"],
    )

    # Basic validation
    assert len(result) > 0
    assert all(count >= 10 for count in result["count"].to_list())
    assert set(result.columns) == {
        "age", "gender", "zipcode", "condition", "count"
    }
