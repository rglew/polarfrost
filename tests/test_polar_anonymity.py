"""
Tests for the anonymization module.
"""
import pytest
import polars as pl
from polarfrost.polar_anonymity import polar_k_anonymize

def test_basic_anonymization():
    """Test basic k-anonymity functionality."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "zip": ["12345", "12345", "12345", "54321", "54321", "54321"],
        "age": [25, 30, 35, 40, 45, 50],
        "condition": ["A", "B", "A", "B", "A", "B"]
    }).lazy()
    
    result = polar_k_anonymize(
        df,
        group_columns=["zip", "age"],
        sensitive_columns="condition",
        k=2
    ).collect()
    
    # Verify all groups of size < k are masked
    assert result.filter(pl.col("zip") == "12345")["condition"].n_unique() == 1
    assert result.filter(pl.col("zip") == "54321")["condition"].n_unique() == 1

def test_demographics_anonymization():
    """Test anonymization of demographic data."""
    def anonymize_demographics(df, k=10, debug=False):
        group_columns = [
            "client_flat_org_id",
            "appointment_month",
            "demographic_category_id",
            "demographic_response_id"
        ]
        return polar_k_anonymize(
            df=df,
            group_columns=group_columns,
            sensitive_columns="demographic_response",
            k=k,
            debug=debug
        )
    
    # Test data
    df = pl.DataFrame({
        "client_flat_org_id": [1, 1, 2, 2, 3, 3],
        "appointment_month": ["2023-01", "2023-01", "2023-01", "2023-01", "2023-02", "2023-02"],
        "demographic_category_id": [1, 1, 1, 2, 2, 2],
        "demographic_response_id": [100, 100, 101, 200, 200, 201],
        "demographic_response": ["A", "A", "B", "C", "C", "D"],
        "other_data": [10, 20, 30, 40, 50, 60]
    }).lazy()
    
    result = anonymize_demographics(df, k=2).collect()
    assert result.filter(pl.col("demographic_response_id") == 201)["demographic_response"][0] == "masked"

def test_edge_cases():
    """Test edge cases."""
    # Test empty dataframe
    df = pl.DataFrame(schema={"id": pl.Int64, "group": pl.Utf8, "value": pl.Utf8}).lazy()
    result = polar_k_anonymize(df, group_columns=["group"], sensitive_columns="value", k=1).collect()
    assert result.is_empty()
    
    # Test k=1 (should never mask anything)
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "group": ["A", "A", "B"],
        "value": ["X", "Y", "Z"]
    }).lazy()
    result = polar_k_anonymize(df, group_columns=["group"], sensitive_columns="value", k=1).collect()
    assert result["value"].to_list() == ["X", "Y", "Z"]