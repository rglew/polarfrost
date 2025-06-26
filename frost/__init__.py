"""
Frost ❄️ - Fast k-anonymity implementation using Polars
"""

from .mondrian import mondrian_k_anonymity
from .clustering import clustering_k_anonymity

__version__ = "0.1.0"
__all__ = ['mondrian_k_anonymity', 'clustering_k_anonymity']
