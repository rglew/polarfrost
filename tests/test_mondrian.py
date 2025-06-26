import os
import polars as pl
from polarfrost.mondrian import mondrian_k_anonymity

def test_mondrian_basic():
    """Test basic k-anonymity functionality."""
    # Sample data
    data = {
        "age": [25, 25, 35, 35, 45, 45],
        "gender": ["M", "M", "F", "F", "M", "M"],
        "zipcode": ["12345", "12345", "12345", "12345", "67890", "67890"],
        "income": [50000, 55000, 60000, 65000, 70000, 75000]
    }
    df = pl.DataFrame(data)
    
    # Apply k-anonymity with k=2
    result = mondrian_k_anonymity(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="income",
        k=2,
        categorical=["gender", "zipcode"]
    )
    
    # Basic assertions
    assert len(result) > 0
    assert "count" in result.columns
    assert all(result["count"] >= 2)
