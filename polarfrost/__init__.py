"""
PolarFrost: Fast k-anonymity implementation using Polars and PySpark.

This package provides efficient implementations of k-anonymity algorithms,
including the Mondrian algorithm, with support for both local (Polars)
and distributed (PySpark) processing.
"""

__version__ = "0.1.0"

# Import main functions
try:
    from .mondrian import (
        mondrian_k_anonymity,
        mondrian_k_anonymity_polars,
        mondrian_k_anonymity_spark
    )
    __all__ = [
        'mondrian_k_anonymity',
        'mondrian_k_anonymity_polars',
        'mondrian_k_anonymity_spark'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import mondrian: {e}")
    __all__ = []
