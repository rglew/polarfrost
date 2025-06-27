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
    def mondrian_k_anonymity(*args, **kwargs):
        raise ImportError("Mondrian k-anonymity is not available due to missing dependencies")

    def mondrian_k_anonymity_polars(*args, **kwargs):
        raise ImportError("Mondrian k-anonymity (Polars) is not available due to missing dependencies")

    def mondrian_k_anonymity_spark(*args, **kwargs):
        raise ImportError("Mondrian k-anonymity (PySpark) is not available due to missing dependencies")

    def clustering_k_anonymity(*args, **kwargs):
        raise ImportError("Clustering k-anonymity is not available due to missing dependencies")

    __all__ = []
