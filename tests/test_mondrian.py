from typing import List, cast

import polars as pl
from polarfrost.mondrian import mondrian_k_anonymity


def test_mondrian_basic() -> None:
    """Test basic k-anonymity functionality."""
    # Sample data
    data = {
        "age": [25, 25, 35, 35, 45, 45],
        "gender": ["M", "M", "F", "F", "M", "M"],
        "zipcode": ["12345", "12345", "12345", "12345", "67890", "67890"],
        "income": [50000, 55000, 60000, 65000, 70000, 75000],
    }
    df = pl.DataFrame(data)

    # Apply k-anonymity with k=2 and type the result
    result: pl.DataFrame = cast(pl.DataFrame, mondrian_k_anonymity(
        df,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="income",
        k=2,
        categorical=["gender", "zipcode"],
    ))

    # Basic assertions for Polars DataFrame
    assert result.shape[0] > 0, "Result should not be empty"
    assert "count" in result.columns, "Result should have a 'count' column"

    # Get count values and verify k-anonymity
    count_values: List[int] = result["count"].to_list()
    assert all(count >= 2 for count in count_values), \
        "k-anonymity constraint not satisfied"
