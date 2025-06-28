"""
PolarFrost: Fast k-anonymity implementation using Polars and PySpark.

This package provides efficient implementations of k-anonymity algorithms,
including the Mondrian algorithm, with support for both local (Polars)
and distributed (PySpark) processing.
"""

from typing import Any, TypeVar, Union

__version__ = "0.2.0"

# Import main functions
try:
    from .clustering import clustering_k_anonymity
    from .mondrian import (
        mondrian_k_anonymity,
        mondrian_k_anonymity_polars,
        mondrian_k_anonymity_spark,
    )

    __all__ = [
        "mondrian_k_anonymity",
        "mondrian_k_anonymity_alt",
        "mondrian_k_anonymity_polars",
        "mondrian_k_anonymity_spark",
        "clustering_k_anonymity",
    ]
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import all dependencies: {e}", stacklevel=2)

    # Dummy types for when dependencies are not available
    T = TypeVar('T')

    class DummyType:
        def __getattr__(self, name: str) -> Any:
            return Any

    class LazyFrame(DummyType):
        pass

    class SparkDataFrame(DummyType):
        pass

    class StructType(DummyType):
        pass

    PolarsDataFrame = Union[DummyType, LazyFrame]

    def _raise_import_error(pkg: str) -> None:
        raise ImportError(f"This function requires {pkg} to be installed")

    def __getattr__(name: str) -> Any:
        if name in {
            'mondrian_k_anonymity',
            'mondrian_k_anonymity_polars',
            'mondrian_k_anonymity_spark',
            'clustering_k_anonymity',
        }:
            _raise_import_error("polars and/or pyspark")
        msg = (f"No module named {name!r}. "
               "Make sure to install the required dependencies.")
        raise AttributeError(msg)

    __all__ = [
        "mondrian_k_anonymity",
        "mondrian_k_anonymity_polars",
        "mondrian_k_anonymity_spark",
        "clustering_k_anonymity",
    ]
