"""
Modern data anonymization functions using Polars for efficient processing.
"""

import time
from typing import List, Union
import polars as pl

def polar_k_anonymize(
    df: pl.LazyFrame,
    group_columns: List[str],
    sensitive_columns: Union[str, List[str]],
    k: int = 10,
    mask_value: str = "masked",
    debug: bool = False
) -> pl.LazyFrame:
    """
    Efficient k-anonymity implementation using modern data processing techniques.
    
    Args:
        df: Input LazyFrame containing the data to be anonymized
        group_columns: List of column names that form the quasi-identifiers
        sensitive_columns: Column name(s) to be masked when group size < k
        k: Minimum group size required (default: 10)
        mask_value: Value to use for masking sensitive data (default: "masked")
        debug: If True, prints performance metrics (default: False)
    """
    start_time = time.time() if debug else None
    
    # Convert single column to list if needed
    if isinstance(sensitive_columns, str):
        sensitive_columns = [sensitive_columns]
    
    # Validate input columns
    all_columns = set(df.columns)
    missing_columns = set(group_columns + sensitive_columns) - all_columns
    if missing_columns:
        raise ValueError(f"Columns not found in dataframe: {missing_columns}")
    
    # Find small groups
    small_groups = (
        df
        .select(group_columns)
        .group_by(group_columns)
        .agg(pl.len().alias("__count__"))
        .filter(pl.col("__count__") < k)
        .select(group_columns)
        .with_columns(pl.lit(True).alias("__mask__"))
    )
    
    # Apply masking
    result = (
        df
        .join(small_groups, on=group_columns, how="left")
        .with_columns([
            pl.when(pl.col("__mask__").fill_null(False))
            .then(pl.lit(mask_value))
            .otherwise(pl.col(col))
            .alias(col)
            for col in sensitive_columns
        ])
        .drop("__mask__")
    )
    
    if debug:
        elapsed = time.time() - start_time
        row_count = df.collect().height
        print(f"Anonymization completed in {elapsed:.2f} seconds")
        print(f"Processed {row_count:,} rows at {row_count/elapsed:,.1f} rows/sec")
    
    return result