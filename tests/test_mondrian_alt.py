"""Tests for the alternative Mondrian k-anonymity implementation.

This module tests the implementation that preserves row count.
"""

import polars as pl
import pytest

from polarfrost.mondrian import mondrian_k_anonymity_alt


def test_mondrian_alt_basic() -> None:
    """Test basic functionality with a simple dataset."""
    # Create test data
    data = {
        "org_id": [1, 1, 1, 1, 1, 1],
        "age": [25, 30, 35, 40, 45, 50],
        "gender": ["M", "F", "M", "F", "M", "F"],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000],
        "department": ["HR", "IT", "IT", "HR", "IT", "HR"],
        "survey_response": [
            "Happy", "Sad", "Happy", "Neutral", "Happy", "Sad"
        ],
    }
    df = pl.DataFrame(data).lazy()

    # Apply anonymization
    result: pl.DataFrame = mondrian_k_anonymity_alt(
        df=df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="survey_response",
        k=2,
        categorical=["gender", "department"],
        group_columns=["org_id"],
    ).collect()

    # Verify results
    assert len(result) == len(df.collect())  # Same number of rows
    assert "org_id" in result.columns  # Group column preserved
    assert "salary" in result.columns  # Non-QI column preserved
    assert "survey_response" in result.columns  # Sensitive column preserved

    # Check that QI columns were modified
    # Create a DataFrame from the original data for comparison
    original_df = pl.DataFrame(data)

    # Get lists for comparison
    result_ages = result["age"].to_list()
    result_genders = result["gender"].to_list()
    original_ages = original_df["age"].to_list()
    original_genders = original_df["gender"].to_list()

    # Verify that at least one value was modified
    assert any(
        result_ages[i] != original_ages[i]
        for i in range(len(result_ages))
    )
    assert any(
        result_genders[i] != original_genders[i]
        for i in range(len(result_genders))
    )


def test_mondrian_alt_with_hierarchy() -> None:
    """Test with hierarchical organization data."""
    # Create test data with parent-child org structure
    data = {
        "org_id": [1, 1, 1, 2, 2, 2],  # Two orgs
        "age": [25, 30, 35, 40, 45, 50],
        "gender": ["M", "F", "M", "F", "M", "F"],
        "survey_response": [
            "Happy", "Sad", "Happy", "Neutral", "Happy", "Sad"
        ],
        "department": ["HR", "IT", "IT", "HR", "IT", "HR"],
    }
    df = pl.DataFrame(data).lazy()

    # Apply anonymization with k=4
    # (will require masking for org_id=2 which has 3 rows)
    result: pl.DataFrame = mondrian_k_anonymity_alt(
        df=df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="survey_response",
        k=4,
        categorical=["gender", "department"],
        group_columns=["org_id"],
    ).collect()

    # Verify results
    assert len(result) == len(df.collect())

    # Check that org_id values are preserved
    assert sorted(result["org_id"].unique().to_list()) == [1, 2]

    # Check that small groups are masked
    org_2_responses = result.filter(
        pl.col("org_id") == 2
    )["survey_response"].to_list()
    assert all(
        r == "masked"
        for r in org_2_responses
    )


def test_mondrian_alt_preserve_columns() -> None:
    """Test that non-QI columns are preserved exactly."""
    data = {
        "org_id": [1, 1, 1, 1],
        "age": [25, 30, 35, 40],
        "gender": ["M", "F", "M", "F"],
        # Non-QI column
        "salary": [50000, 60000, 70000, 80000],
        "survey_response": ["Happy", "Sad", "Happy", "Sad"],
    }
    df = pl.DataFrame(data).lazy()

    result: pl.DataFrame = mondrian_k_anonymity_alt(
        df=df,
        quasi_identifiers=["age", "gender"],
        sensitive_column="survey_response",
        k=2,
        group_columns=["org_id"],
    ).collect()

    # Check that non-QI columns are preserved exactly
    assert result["salary"].to_list() == data["salary"]
    assert result["org_id"].to_list() == data["org_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
