"""
Test the Mondrian k-anonymity implementation with sample data.
"""
import os
import polars as pl
import pytest
from polarfrost import mondrian_k_anonymity

def test_mondrian_basic():
    """Test basic Mondrian k-anonymity with a small dataset."""
    # Create a small test dataset
    data = {
        "age": [25, 25, 35, 35, 45, 45, 55, 55],
        "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
        "zipcode": ["12345", "12345", "12345", "12345", "67890", "67890", "67890", "67890"],
        "income": [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
    }
    df = pl.DataFrame(data)
    
    # Apply k-anonymity with k=2
    quasi_identifiers = ["age", "gender", "zipcode"]
    sensitive_column = "income"
    categorical = ["gender", "zipcode"]
    k = 2
    
    anon_df = mondrian_k_anonymity(
        df,
        quasi_identifiers=quasi_identifiers,
        sensitive_column=sensitive_column,
        k=k,
        categorical=categorical
    )
    
    # Verify the output
    assert isinstance(anon_df, pl.DataFrame)
    assert len(anon_df) > 0  # Should have at least one group
    assert all(count >= k for count in anon_df["count"])  # All groups should satisfy k-anonymity
    
    # Check that all quasi-identifiers are generalized
    for col in quasi_identifiers:
        assert col in anon_df.columns
    
    # Check that the sensitive column is included
    assert sensitive_column in anon_df.columns
    assert "count" in anon_df.columns

def test_mondrian_with_lazyframe():
    """Test that the function works with LazyFrames."""
    data = {
        "age": [25, 25, 35, 35, 45, 45],
        "gender": ["M", "M", "F", "F", "M", "M"],
        "zipcode": ["12345", "12345", "12345", "12345", "67890", "67890"],
        "income": [50000, 55000, 60000, 65000, 70000, 75000]
    }
    df = pl.LazyFrame(data)
    
    anon_df = mondrian_k_anonymity(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="income",
        k=2,
        categorical=["gender", "zipcode"]
    )
    
    assert isinstance(anon_df, pl.DataFrame)
    assert len(anon_df) > 0

def test_mondrian_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    df = pl.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30]})
    
    # Test with k larger than dataset size - should return a single group
    result = mondrian_k_anonymity(df, ["age"], "income", k=5)
    assert len(result) == 1  # Should return a single group with all records
    assert result["count"][0] == 3  # All records should be in one group
    
    # Test with invalid column names
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        mondrian_k_anonymity(df, ["invalid"], "income", k=2)
    
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        mondrian_k_anonymity(df, ["age"], "invalid", k=2)

if __name__ == "__main__":
    test_mondrian_basic()
    test_mondrian_with_lazyframe()
    test_mondrian_invalid_input()
    print("All tests passed!")
