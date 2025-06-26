"""
Frost ❄️ - Fast k-anonymity implementation using Polars
"""

__version__ = "0.1.0"

# Import core functionality
try:
    from .mondrian import mondrian_k_anonymity
    from .clustering import clustering_k_anonymity
    __all__ = ['mondrian_k_anonymity', 'clustering_k_anonymity']
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all modules: {e}")
    __all__ = []
