"""
Efficient Mondrian k-Anonymity implementation using Polars and PySpark.
Compatible with local (Polars) and Databricks/Spark (PySpark) environments.
"""

from typing import List, Optional, Union, Dict, Any, cast, TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType


# ------------------------- POLARS VERSION -------------------------
def mondrian_k_anonymity_polars(
    df: "pl.DataFrame | pl.LazyFrame",
    quasi_identifiers: List[str],
    sensitive_column: str,
    k: int,
    categorical: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Perform Mondrian k-anonymity using Polars LazyFrame for local processing.
    Accepts either DataFrame or LazyFrame as input.
    """
    if categorical is None:
        categorical = []

    # Input validation
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise ValueError("Input must be a Polars DataFrame or LazyFrame")

    # Convert to LazyFrame if not already
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
        
    # Check for empty DataFrame by collecting a sample
    if df.select(pl.len()).collect().item(0, 0) == 0:
        raise ValueError("Input DataFrame cannot be empty")

    # Validate k is a positive integer
    if not isinstance(k, (int, str)) or (isinstance(k, str) and not k.isdigit()) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)  # Convert to int if it's a string of digits

    # Initialize partitions with the full dataset
    partitions = [df]
    result = []

    # Process partitions until none left
    while partitions:
        part = partitions.pop()

        # Get partition size (lazy evaluation)
        n_rows = part.select(pl.len()).collect().item(0, 0)

        # If partition is too small to split, add to results
        if n_rows < 2 * k:
            result.append(part)
            continue

        # Compute spans for each quasi-identifier
        spans: Dict[str, Any] = {}
        for col in quasi_identifiers:
            if col in categorical:
                # For categorical, use number of unique values as span
                n_unique = part.select(pl.col(col).n_unique()).collect().item()
                spans[col] = n_unique
            else:
                # For numerical, use range as span
                stats = part.select(
                    [pl.col(col).min().alias("min"), pl.col(col).max().alias("max")]
                ).collect()
                col_min = stats[0, "min"]
                col_max = stats[0, "max"]

                # Handle string comparison by converting to float if possible
                if col_min is not None and col_max is not None:
                    try:
                        # Try to convert to float for comparison
                        min_val = float(col_min) if not isinstance(col_min, (int, float)) else col_min
                        max_val = float(col_max) if not isinstance(col_max, (int, float)) else col_max
                        spans[col] = max_val - min_val
                    except (ValueError, TypeError):
                        # If conversion fails, use string length difference
                        spans[col] = abs(len(str(col_max)) - len(str(col_min)))
                else:
                    spans[col] = 0

        # Find the attribute with maximum span
        split_col = max(spans, key=spans.get)  # type: ignore

        # If no split possible, add to results
        if spans[split_col] == 0:
            result.append(part)
            continue

        # Split the partition
        if split_col in categorical:
            # For categorical, split on unique values
            uniq_vals = (
                part.select(pl.col(split_col).unique()).collect().to_series().to_list()
            )
            mid = len(uniq_vals) // 2
            left_vals = set(uniq_vals[:mid])
            right_vals = set(uniq_vals[mid:])
            left = part.filter(pl.col(split_col).is_in(left_vals))
            right = part.filter(pl.col(split_col).is_in(right_vals))
        else:
            # For numerical, split on median
            median = part.select(pl.col(split_col).median()).collect().item()
            left = part.filter(pl.col(split_col) <= median)
            right = part.filter(pl.col(split_col) > median)

        # Check if both partitions satisfy k-anonymity
        left_n = left.select(pl.len()).collect().item(0, 0)
        right_n = right.select(pl.len()).collect().item(0, 0)

        if left_n >= k and right_n >= k:
            # Both partitions are valid, continue splitting
            partitions.extend([left, right])
        else:
            # At least one partition is too small, keep as is
            result.append(part)

    # Aggregate each partition
    agg_rows = []
    for part in result:
        # Collect only the columns we need
        part_df = part.select(quasi_identifiers + [sensitive_column]).collect()
        row = {}

        # Generalize quasi-identifiers
        for col in quasi_identifiers:
            if col in categorical:
                # For categorical, use set of unique values
                unique_vals = part_df[col].unique()
                row[col] = ",".join(sorted(str(v) for v in unique_vals))
            else:
                # For numerical, use range
                min_val = part_df[col].min()
                max_val = part_df[col].max()
                if isinstance(min_val, bytes) or isinstance(max_val, bytes):
                    min_val = (
                        min_val.decode("utf-8")
                        if isinstance(min_val, bytes)
                        else str(min_val)
                    )
                    max_val = (
                        max_val.decode("utf-8")
                        if isinstance(max_val, bytes)
                        else str(max_val)
                    )
                row[col] = f"{min_val}-{max_val}"

        # Add sensitive values and count
        sensitive_vals = part_df[sensitive_column].unique()
        row[sensitive_column] = ",".join(sorted(str(v) for v in sensitive_vals))
        # Store count as integer
        row["count"] = int(part_df.height)
        agg_rows.append(row)

    return pl.DataFrame(agg_rows)


# ------------------------- PYSPARK VERSION -------------------------
def mondrian_k_anonymity_spark(
    df: "SparkDataFrame",
    quasi_identifiers: List[str],
    sensitive_column: str,
    k: int,
    categorical: Optional[List[str]] = None,
    schema: Optional["StructType"] = None,
) -> "SparkDataFrame":
    """
    Perform Mondrian k-anonymity using PySpark for distributed processing.
    
    Args:
        df: Input PySpark DataFrame
        quasi_identifiers: List of column names that are quasi-identifiers
        sensitive_column: Name of the sensitive column
        k: Anonymity parameter (minimum group size), must be a positive integer
        categorical: List of categorical column names
        schema: Schema for the output DataFrame
        
    Returns:
        Anonymized DataFrame with generalized quasi-identifiers
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    
    # Validate k parameter first
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
        
    # Validate schema
    if schema is None:
        raise ValueError("Schema must be provided for PySpark UDF")
        
    # Check for empty DataFrame
    if df.rdd.isEmpty():
        raise ValueError("Input DataFrame cannot be empty")
        
    if categorical is None:
        categorical = []

    # Define the UDF with proper type hints
    @pandas_udf(returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
    def mondrian_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        partitions = [pdf]
        result = []

        while partitions:
            part = partitions.pop()

            # If partition is too small to split, add to results
            if len(part) < 2 * k:
                result.append(part)
                continue

            # Compute spans for each quasi-identifier
            spans = {}
            for col in quasi_identifiers:
                if col in categorical:
                    spans[col] = part[col].nunique()
                else:
                    col_min = part[col].min()
                    col_max = part[col].max()
                    spans[col] = (
                        col_max - col_min
                        if pd.notnull(col_max) and pd.notnull(col_min)
                        else 0
                    )

            # Find the attribute with maximum span
            split_col = max(spans.items(), key=lambda x: x[1])[0]  # type: ignore

            # If no split possible, add to results
            if spans.get(split_col, 0) <= 0:
                result.append(part)
                continue

            # Split on the chosen column
            if split_col in categorical:
                # For categorical, split on median value
                value_counts = part[split_col].value_counts()
                if len(value_counts) > 0:
                    split_val = value_counts.index[len(value_counts) // 2]
                    mask = part[split_col] == split_val
                    left = part[mask]
                    right = part[~mask]
                else:
                    result.append(part)
                    continue
            else:
                # For numerical, split on median
                median_val = part[split_col].median()
                if pd.notna(median_val):
                    left = part[part[split_col] <= median_val]
                    right = part[part[split_col] > median_val]
                else:
                    result.append(part)
                    continue

            # Check if both partitions satisfy k-anonymity
            if len(left) >= k and len(right) >= k:
                # Both partitions are valid, continue splitting
                partitions.extend([left, right])
            else:
                # At least one partition is too small, keep as is
                result.append(part)

        # Aggregate the results
        agg_rows = []
        for part in result:
            row = {}

            # Generalize quasi-identifiers
            for col in quasi_identifiers:
                if col in categorical:
                    # For categorical, use set of unique values
                    row[col] = ",".join(sorted(map(str, part[col].unique())))
                else:
                    # For numerical, use range
                    row[col] = f"{part[col].min()}-{part[col].max()}"

            # Add sensitive values and count
            row[sensitive_column] = ",".join(
                sorted(str(v) for v in part[sensitive_column].unique())
            )
            # Store count as integer
            row["count"] = int(len(part))
            agg_rows.append(row)

        return pd.DataFrame(agg_rows)

    # Apply the UDF with explicit schema
    result_df = df.groupBy().applyInPandas(
        mondrian_partition, schema=schema  # type: ignore
    )
    return result_df


# ------------------------- DISPATCHER -------------------------
def mondrian_k_anonymity(
    df: Union[pl.DataFrame, pl.LazyFrame, "SparkDataFrame"],
    quasi_identifiers: List[str],
    sensitive_column: str,
    k: int,
    categorical: Optional[List[str]] = None,
    schema: Optional["StructType"] = None,
) -> Union[pl.DataFrame, "SparkDataFrame"]:
    """
    Dispatcher: Use Polars or PySpark Mondrian k-anonymity depending on input type.

    Args:
        df: Input DataFrame (Polars or PySpark)
        quasi_identifiers: List of column names that are quasi-identifiers
        sensitive_column: Name of the sensitive column
        k: Anonymity parameter (minimum group size)
        categorical: List of categorical column names
        schema: Schema for PySpark output (required for PySpark)

    Returns:
        Anonymized DataFrame with generalized quasi-identifiers
    """
    try:
        from pyspark.sql import DataFrame as SparkDataFrame

        if isinstance(df, SparkDataFrame):
            return mondrian_k_anonymity_spark(
                df, quasi_identifiers, sensitive_column, k, categorical, schema
            )
    except ImportError:
        pass

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return mondrian_k_anonymity_polars(
            df, quasi_identifiers, sensitive_column, k, categorical
        )

    raise ValueError(
        "Input df must be a polars.DataFrame, polars.LazyFrame, or pyspark.sql.DataFrame"
    )
