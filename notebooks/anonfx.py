from typing import Optional
import time
import polars as pl

def calculate_small_groups(
    lazy_df: pl.LazyFrame,
    group_cols: list[str],
    k: int = 10
) -> pl.LazyFrame:
    """
    Calculate small groups that need to be masked based on k-anonymity.
    
    Args:
        lazy_df: Input data as a Polars LazyFrame
        group_cols: List of column names to group by
        k: Minimum group size for k-anonymity
        
    Returns:
        pl.LazyFrame: Small groups that need masking
    """
    return (
        lazy_df
        .select(group_cols)
        .group_by(group_cols)
        .agg(pl.len().alias("__count__"))
        .filter(pl.col("__count__") < k)
        .select(group_cols)
    )

def anonymize_demographics(
    lazy_df: pl.LazyFrame,
    group_cols: list[str],
    k: int = 10,
    debug: bool = False,
    schema: Optional[dict] = None
) -> pl.LazyFrame:
    """
    Apply k-anonymity to demographics data.
    
    Args:
        lazy_df: Input data as a Polars LazyFrame
        group_cols: List of column names to group by for k-anonymity
        k: Minimum group size for k-anonymity (default: 10)
        debug: Enable debug logging and timing information
        schema: Optional schema to check for required columns
        
    Returns:
        pl.LazyFrame: Anonymized data with small groups masked
    """
    if debug:
        start_time = time.time()
        initial_count = lazy_df.collect().height
        print(f"\n=== Initial row count: {initial_count:,} ===")
    
    # Validate required columns
    if schema is not None:
        required_cols = set(group_cols + ["appointment_date"])
        missing_cols = required_cols - set(schema.keys())
        if missing_cols:
            raise ValueError(f"Missing required columns in schema: {missing_cols}")
            
        # Add appointment_month if needed and not in group_cols
        if 'appointment_month' not in schema and 'appointment_month' in group_cols:
            lazy_df = lazy_df.with_columns(
                pl.col("appointment_date").dt.strftime("%Y-%m").alias("appointment_month")
            )
    
    # Calculate small groups
    small_groups = calculate_small_groups(lazy_df, group_cols, k).with_columns(
        pl.lit(True).alias("__mask__")
    )
    
    # Join and apply masking
    result = (
        lazy_df
        .join(
            small_groups,
            on=group_cols,
            how="left"
        )
        .with_columns(
            pl.when(pl.col("__mask__").fill_null(False))
            .then(pl.lit("masked"))
            .otherwise(pl.col("demographic_response"))
            .alias("demographic_response")
        )
        .drop("__mask__")  # Clean up the temporary column
    )
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Anonymization completed in {elapsed:.2f} seconds")
        print(f"Processed {initial_count:,} rows at {initial_count/elapsed:,.1f} rows/sec")
    
    return result

def anonymize_and_measure_performance(
    lazy_df: pl.LazyFrame,
    group_cols: list[str],
    k: int = 10,
    debug: bool = False,
    schema: Optional[dict] = None
) -> tuple[pl.LazyFrame, float, float, int]:
    """
    Apply k-anonymity to demographics data and measure performance.
    
    Args:
        lazy_df: Input data as a Polars LazyFrame
        group_cols: List of column names to group by for k-anonymity
        k: Minimum group size for k-anonymity (default: 10)
        debug: Enable debug logging and timing information
        schema: Optional schema to check for required columns
        
    Returns:
        tuple: (result, elapsed_time, memory_used, row_count)
    """
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    # Run the anonymization
    result = anonymize_demographics(
        lazy_df=lazy_df,
        group_cols=group_cols,
        k=k,
        debug=debug,
        schema=schema
    )
    
    # Collect to force computation and get row count
    df = result.collect()
    row_count = df.height
    
    # Calculate metrics
    elapsed = time.time() - start_time
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = end_mem - start_mem
    
    if debug:
        print(f"\n=== Performance Metrics ===")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Memory used: {mem_used:.2f} MB")
        print(f"Rows processed: {row_count:,}")
        print(f"Processing speed: {row_count/elapsed:,.1f} rows/sec")
    
    return result, elapsed, mem_used, row_count