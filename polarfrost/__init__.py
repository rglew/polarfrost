"""
PolarFrost: Fast k-anonymity implementation using Polars and PySpark.

This package provides efficient implementations of k-anonymity algorithms,
including the Mondrian algorithm, with support for both local (Polars)
and distributed (PySpark) processing.
"""

__version__ = "0.1.1"

# Import main functions
try:
    from .mondrian import (
        mondrian_k_anonymity,
        mondrian_k_anonymity_polars,
        mondrian_k_anonymity_spark,
    )
    from .clustering import clustering_k_anonymity

    __all__ = [
        "mondrian_k_anonymity",
        "mondrian_k_anonymity_polars",
        "mondrian_k_anonymity_spark",
        "clustering_k_anonymity",
    ]
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import all dependencies: {e}")

    # Define dummy functions for type checking
    from typing import Any, List, Optional, Union
    from typing_extensions import Literal
    from pyspark.sql.types import StructType
    from pyspark.sql import DataFrame as SparkDataFrame
    from polars import DataFrame as PolarsDataFrame, LazyFrame
    
    def mondrian_k_anonymity(
        df: Union[PolarsDataFrame, LazyFrame, SparkDataFrame],
        quasi_identifiers: List[str],
        sensitive_column: str,
        k: int,
        categorical: Optional[List[str]] = None,
        schema: Optional[StructType] = None
    ) -> Union[PolarsDataFrame, SparkDataFrame]:
        """Dummy function for type checking when dependencies are missing."""
        raise ImportError("Mondrian k-anonymity is not available due to missing dependencies")

    def mondrian_k_anonymity_polars(
        df: Union[PolarsDataFrame, LazyFrame],
        quasi_identifiers: List[str],
        sensitive_column: str,
        k: int,
        categorical: Optional[List[str]] = None
    ) -> PolarsDataFrame:
        """Dummy function for type checking when dependencies are missing."""
        raise ImportError("Mondrian k-anonymity (Polars) is not available due to missing dependencies")

    def mondrian_k_anonymity_spark(
        df: SparkDataFrame,
        quasi_identifiers: List[str],
        sensitive_column: str,
        k: int,
        categorical: Optional[List[str]] = None,
        schema: Optional[StructType] = None
    ) -> SparkDataFrame:
        """Dummy function for type checking when dependencies are missing."""
        raise ImportError("Mondrian k-anonymity (PySpark) is not available due to missing dependencies")

    def clustering_k_anonymity(
        df: Union[PolarsDataFrame, LazyFrame],
        quasi_identifiers: List[str],
        sensitive_column: str,
        k: int,
        categorical: Optional[List[str]] = None,
        method: str = "kmeans"
    ) -> PolarsDataFrame:
        """Dummy function for type checking when dependencies are missing."""
        raise ImportError("Clustering k-anonymity is not available due to missing dependencies")

    __all__ = []
