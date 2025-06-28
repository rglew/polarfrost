# This is a minimal PySpark type stub file
# It's used to provide type information for mypy

from typing import Any, List, Dict, Optional, Union, Tuple, Callable, TypeVar, Generic, overload

# Basic types
T = TypeVar('T')

# Add any PySpark specific types you need here
class DataFrame:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def show(self, *args: Any, **kwargs: Any) -> None: ...
    # Add other DataFrame methods as needed

class SparkSession:
    @staticmethod
    def builder() -> 'Builder': ...
    def createDataFrame(self, data: Any, *args: Any, **kwargs: Any) -> DataFrame: ...
    def stop(self) -> None: ...
    # Add other SparkSession methods as needed

class Builder:
    def appName(self, name: str) -> 'Builder': ...
    def getOrCreate(self) -> SparkSession: ...
    # Add other Builder methods as needed

# Add any other PySpark classes you need

# This allows 'import pyspark' to work in type-checked code
__all__ = ['DataFrame', 'SparkSession', 'Builder']
