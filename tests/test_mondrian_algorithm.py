"""Tests for the core Mondrian algorithm implementation."""

import polars as pl
from polarfrost.mondrian import mondrian_k_anonymity_polars


def test_mondrian_numerical_attributes() -> None:
    """Test Mondrian with numerical attributes."""
    # Create a test DataFrame with numerical quasi-identifiers
    df = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35, 40, 40],
            "income": [
                50000, 55000, 60000, 65000,
                70000, 75000, 80000, 85000
            ],
            "condition": ["A", "B"] * 4,
        }
    )

    # Apply k-anonymity with k=2
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "income"],
        sensitive_column="condition",
        k=2
    )

    # Verify the result has the expected columns
    assert set(result.columns) == {"age", "income", "condition", "count"}

    # Verify all groups have at least k=2 records
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_categorical_attributes() -> None:
    """Test Mondrian with categorical attributes."""
# Create a test DataFrame with categorical quasi-identifiers
    df = pl.DataFrame(
        {
            "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
            "zipcode": [
                "12345",
                "12345",
                "12345",
                "12345",
                "67890",
                "67890",
                "67890",
                "67890",
            ],
            "condition": ["A", "B"] * 4,
        }
    )

    # Apply k-anonymity with k=2 and mark gender and zipcode as categorical
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["gender", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["gender", "zipcode"],
    )

    # Verify the result has the expected columns
    assert set(result.columns) == {"gender", "zipcode", "condition", "count"}

    # Verify all groups have at least k=2 records
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_mixed_attributes() -> None:
    """Test Mondrian with mixed numerical and categorical attributes."""
# Create a test DataFrame with mixed attribute types
    df = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35, 40, 40],
            "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
            "income": [
                50000, 55000, 60000, 65000,
                70000, 75000, 80000, 85000
            ],
            "zipcode": [
                "12345",
                "12345",
                "12345",
                "12345",
                "67890",
                "67890",
                "67890",
                "67890",
            ],
            "condition": ["A", "B"] * 4,
        }
    )

    # Apply k-anonymity with k=2 and mark gender and zipcode as categorical
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "income", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["gender", "zipcode"],
    )

    # Verify the result has the expected columns
    assert set(result.columns) == {
        "age",
        "gender",
        "income",
        "zipcode",
        "condition",
        "count",
    }

    # Verify all groups have at least k=2 records
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_small_k() -> None:
    """Test Mondrian with k=1 (minimum group size)."""
    # Create a small test DataFrame
    df = pl.DataFrame({
        "age": [25, 30, 35, 40],
        "condition": ["A", "B", "A", "B"]
    })

    # Apply k-anonymity with k=1 (minimum group size)
    result = mondrian_k_anonymity_polars(
        df, quasi_identifiers=["age"], sensitive_column="condition", k=1
    )

    # With k=1, each record can be its own group
    assert len(result) == len(df)
    assert all(count == 1 for count in result["count"].to_list())


def test_mondrian_large_k() -> None:
    """Test Mondrian with k larger than the number of records."""
    # Create a small test DataFrame
    df = pl.DataFrame({
        "age": [25, 30, 35, 40],
        "condition": ["A", "B", "A", "B"]
    })

    # Apply k-anonymity with k larger than the number of records
    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age"],
        sensitive_column="condition",
        # Larger than number of records (4)
        k=5,
    )

    # Should have exactly one group with all records
    assert len(result) == 1
    assert result["count"][0] == len(df)
