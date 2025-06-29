"""
Efficient Mondrian k-Anonymity implementation using Polars and PySpark.
Compatible with local (Polars) and Databricks/Spark (PySpark) environments.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, TypeVar

import pandas as pd
import polars as pl
from pyspark.sql.functions import pandas_udf, PandasUDFType

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType

# Type variables for pandas_udf return types
T = TypeVar('T', bound=pd.DataFrame)
PandasDataFrameT = TypeVar('PandasDataFrameT', bound=pd.DataFrame)
ResultDataFrameT = TypeVar('ResultDataFrameT', bound=pd.DataFrame)


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
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

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
                stats = part.select([
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max")
                ]).collect()
                col_min = stats[0, "min"]
                col_max = stats[0, "max"]

                # Handle string comparison by converting to float if possible
                if col_min is not None and col_max is not None:
                    try:
                        # Try to convert to float for comparison
                        min_float = (
                            float(col_min)
                            if not isinstance(col_min, (int, float))
                            else col_min
                        )
                        max_float = (
                            float(col_max)
                            if not isinstance(col_max, (int, float))
                            else col_max
                        )
                        spans[col] = max_float - min_float
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
            uniq_vals = part.select(
                pl.col(split_col).unique()
            ).collect().to_series().to_list()
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
        row: Dict[str, Union[str, int]] = {}

        # Generalize quasi-identifiers
        for col in quasi_identifiers:
            if col in categorical:
                # For categorical, use set of unique values
                unique_vals = part_df[col].unique()
                sorted_vals = sorted(str(v) for v in unique_vals)
                row[col] = ",".join(sorted_vals)
            else:
                # For numerical, use range
                min_val: Any = part_df[col].min()
                max_val: Any = part_df[col].max()

                # Ensure we have valid numeric values
                if min_val is None or max_val is None:
                    # Handle null values
                    row[col] = "*"
                else:
                    # Convert to string, handling bytes and other types
                    # Handle different types for string conversion
                    if isinstance(min_val, bytes):
                        min_str = min_val.decode("utf-8")
                    elif isinstance(min_val, (int, float)):
                        min_str = str(min_val)
                    else:
                        min_str = str(min_val)

                    if isinstance(max_val, bytes):
                        max_str = max_val.decode("utf-8")
                    elif isinstance(max_val, (int, float)):
                        max_str = str(max_val)
                    else:
                        max_str = str(max_val)

                    # Store as string range
                    row[col] = f"{min_str}-{max_str}"

        # Add sensitive values and count
        sensitive_vals = part_df[sensitive_column].unique()
        sorted_sensitive = sorted(str(v) for v in sensitive_vals)
        row[sensitive_column] = ",".join(sorted_sensitive)
        # Store count as integer to match the expected type
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
    @pandas_udf(
        returnType=schema,
        functionType=PandasUDFType.GROUPED_MAP,
    )
    def mondrian_partition(pdf: 'pd.DataFrame') -> 'pd.DataFrame':
        """Process a partition of data using Mondrian k-anonymity.

        Args:
            pdf: Input pandas DataFrame partition

        Returns:
            Processed DataFrame with k-anonymity applied
        """
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
            split_col = max(
                spans.items(),
                key=lambda x: x[1]
            )[0]

            # If no split possible, add to results
            if spans.get(split_col, 0) <= 0:
                result.append(part)
                continue

            # Split on the chosen column
            left = None
            right = None

            if split_col in categorical:
                # For categorical, split on median value
                value_counts = part[split_col].value_counts()
                if len(value_counts) > 0:
                    split_val = value_counts.index[len(value_counts) // 2]
                    mask = part[split_col] == split_val
                    left = part[mask]
                    right = part[~mask]
            else:
                # For numerical, split on median
                median_val = part[split_col].median()
                if pd.notna(median_val):
                    left = part[part[split_col] <= median_val]
                    right = part[part[split_col] > median_val]

            # If we couldn't split the partition, add it to results
            if (left is None or right is None or  # noqa: W503,W504
                    len(left) == 0 or len(right) == 0):
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
                    unique_vals = sorted(map(str, part[col].unique()))
                    row[col] = ",".join(unique_vals)
                else:
                    # For numerical, use range
                    min_val = part[col].min()
                    max_val = part[col].max()
                    range_str = f"{min_val}-{max_val}"
                    row[col] = range_str

            # Add sensitive values and count
            unique_sensitive = sorted(
                str(v)
                for v in part[sensitive_column].unique()
            )
            row[sensitive_column] = ",".join(unique_sensitive)
            # Store count as string to match the expected type
            row["count"] = str(len(part))
            agg_rows.append(row)

        return pd.DataFrame(agg_rows)

    # Apply the UDF with explicit schema
    result_df = df.groupBy().applyInPandas(
        mondrian_partition,
        schema=schema
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
    """Dispatcher for Mondrian k-anonymity.

    Uses Polars or PySpark implementation based on input type.

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
        "Input df must be a polars.DataFrame, "
        "polars.LazyFrame, or pyspark.sql.DataFrame"
    )


def _generalize_partition(
    partition: pl.DataFrame,
    quasi_identifiers: List[str],
    categorical: List[str],
    mask_value: str = "masked",
) -> pl.DataFrame:
    """Generalize a partition by applying Mondrian-style generalization.

    Args:
        partition: Input DataFrame to generalize
        quasi_identifiers: List of column names that are quasi-identifiers
        categorical: List of categorical column names
        mask_value: Value to use for masking categorical values

    Returns:
        Generalized DataFrame with quasi-identifiers masked or ranged
    """
    result = partition.clone()

    def to_str(val: Any) -> str:
        """Convert a value to string, handling None and bytes."""
        if val is None:
            return ""
        if isinstance(val, bytes):
            return val.decode('utf-8', errors='replace')
        return str(val)

    for col in quasi_identifiers:
        is_cat = col in categorical
        if is_cat:
            # For categoricals, use mask if multiple values exist
            if result[col].n_unique() > 1:
                result = result.with_columns(pl.lit(mask_value).alias(col))
        else:
            # For numerical, create a range
            min_val = result[col].min()
            max_val = result[col].max()

            min_str = to_str(min_val)
            max_str = to_str(max_val)

            if min_val == max_val:
                result = result.with_columns(pl.lit(min_str).alias(col))
            else:
                result = result.with_columns(
                    pl.lit(f"[{min_str}-{max_str}]").alias(col)
                )

    return result


def mondrian_k_anonymity_alt(
    df: pl.LazyFrame,
    quasi_identifiers: List[str],
    sensitive_column: str,
    k: int,
    categorical: Optional[List[str]] = None,
    mask_value: str = "masked",
    group_columns: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """
    Alternative Mondrian k-anonymity that preserves the original row count.

    Args:
        df: Input LazyFrame
        quasi_identifiers: List of column names that are quasi-identifiers
        sensitive_column: Name of the sensitive column
        k: Anonymity parameter (minimum group size)
        categorical: List of categorical column names
        mask_value: Value to use for masking small groups
        group_columns: Additional columns to use for grouping but keep
            unchanged

    Returns:
        Anonymized LazyFrame with same row count as input
    """
    if not isinstance(df, pl.LazyFrame):
        raise ValueError("Input must be a Polars LazyFrame")

    # Get schema to preserve column order
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    all_columns = list(schema.keys())

    # Initialize parameters
    categorical = categorical or []
    group_columns = group_columns or []

    # Validate inputs
    if k < 1:
        msg = "k must be a positive integer"
        raise ValueError(msg)

    # Check if all specified columns exist
    all_columns_to_check = (
        quasi_identifiers + [sensitive_column] + (
            group_columns or []) + (categorical or [])
    )
    for col in set(all_columns_to_check):
        if col not in schema:
            raise ValueError(f"Column {col!r} not found in DataFrame")

    # Ensure no overlap between group_columns and QIs
    if any(col in quasi_identifiers for col in group_columns):
        raise ValueError("group_columns cannot overlap with quasi_identifiers")

    # Collect the data once
    df_collected = df.collect()

    # Process each group
    if group_columns:
        # Get unique group combinations
        groups = df_collected.select(group_columns).unique()

        results = []

        for group in groups.rows(named=True):
            # Filter current group
            condition = pl.lit(True)
            for col, val in group.items():
                condition = condition & (pl.col(col) == val)

            group_df = df_collected.filter(condition)
            group_size = len(group_df)

            if group_size < k:
                # Mask QIs and sensitive column for small groups
                masked_cols = {}
                # Mask all QIs
                for col in quasi_identifiers:
                    if col in categorical:
                        masked_cols[col] = pl.lit(mask_value)
                # Always mask the sensitive column for small groups
                masked_cols[sensitive_column] = pl.lit(mask_value)

                if masked_cols:
                    group_df = group_df.with_columns(**masked_cols)

                results.append(group_df)
            else:
                # Apply generalization to QIs
                if quasi_identifiers:
                    group_df = _generalize_partition(
                        group_df,
                        quasi_identifiers,
                        categorical or [],
                        mask_value
                    )
                results.append(group_df)

        # Combine results
        result_df = pl.concat(results)
    else:
        # Process entire dataset as one group
        if len(df_collected) < k:
            # Mask all QIs and sensitive column
            masked_cols = {}
            # Mask all QIs
            for col in quasi_identifiers:
                if col in categorical:
                    masked_cols[col] = pl.lit(mask_value)
            # Always mask the sensitive column for small groups
            masked_cols[sensitive_column] = pl.lit(mask_value)

            if masked_cols:
                result_df = df_collected.with_columns(**masked_cols)
            else:
                result_df = df_collected
        else:
            # Apply generalization to QIs
            if quasi_identifiers:
                result_df = _generalize_partition(
                    df_collected,
                    quasi_identifiers,
                    categorical or [],
                    mask_value
                )
            else:
                result_df = df_collected

    # Ensure original column order and return as LazyFrame
    return result_df.select(all_columns).lazy()
