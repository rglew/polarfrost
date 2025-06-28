"""Advanced edge case tests for Mondrian k-anonymity implementation."""

import polars as pl

from polarfrost.mondrian import mondrian_k_anonymity_polars


def test_mondrian_single_record() -> None:
    """Test with a single record - should return as is."""
    df = pl.DataFrame(
        {
            "age": [30],
            "gender": ["M"],
            "zipcode": ["12345"],
            "condition": ["A"]
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=1,
    )

    assert len(result) == 1
    assert result["count"][0] == 1


def test_mondrian_all_identical_records() -> None:
    """Test with all records being identical."""
    df = pl.DataFrame(
        {
            "age": [30, 30, 30, 30],
            "gender": ["M", "M", "M", "M"],
            "zipcode": ["12345"] * 4,
            "condition": ["A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=2,
    )

    assert len(result) == 1
    assert result["count"][0] == 4


def test_mondrian_mixed_data_types() -> None:
    """Test with mixed data types including None values."""
    df = pl.DataFrame(
        {
            "age": [25, None, 35, 40, 45, None, 55, 60],
            "gender": ["M", "F", None, "F", "M", "F", "M", None],
            "income": [
                50000, 60000, 70000, None, 90000, 100000, 110000, 120000
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
            "condition": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender", "income", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["gender", "zipcode"],
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_special_characters() -> None:
    """Test with special characters in string fields."""
    df = pl.DataFrame(
        {
            "name": ["John", "Jane", "Jake", "Jill"],
            "age": [25, 25, 30, 30],
            "address": [
                "123 Main St",
                "456 Oak Ave",
                "123 Main St",
                "789 Pine Rd"
            ],
            "condition": ["A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["name", "age", "address"],
        sensitive_column="condition",
        k=2,
        categorical=["name", "address"],
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())


def test_mondrian_large_k_value() -> None:
    """Test with k larger than the number of records."""
    df = pl.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "gender": ["M", "F", "M", "F"],
            "condition": ["A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="condition",
        k=5,  # Larger than number of records (4)
    )

    # Should have exactly one group with all records
    assert len(result) == 1
    assert result["count"][0] == len(df)


def test_mondrian_numeric_as_categorical() -> None:
    """Test treating numeric columns as categorical."""
    df = pl.DataFrame(
        {
            "age": [25, 25, 30, 30, 35, 35, 40, 40],
            "zipcode": [
                12345, 12345, 12345, 12345,
                67890, 67890, 67890, 67890
            ],
            "condition": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    result = mondrian_k_anonymity_polars(
        df,
        quasi_identifiers=["age", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["age", "zipcode"],  # Numeric as categorical
    )

    assert len(result) >= 1
    assert all(count >= 2 for count in result["count"].to_list())
    # Verify that zipcode was treated as categorical
    # (should be a string in output)
    assert all(
        isinstance(zipcode, str)
        for zipcode in result["zipcode"].to_list()
    )
