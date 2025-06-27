"""Tests for PySpark implementation of Mondrian k-anonymity."""

import pytest
import polars as pl
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, 
    StructField, 
    StringType, 
    IntegerType,
    FloatType
)

# Skip all tests in this module if PySpark is not available
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType
    pyspark_available = True
except ImportError:
    pyspark_available = False

pytestmark = pytest.mark.skipif(
    not pyspark_available, 
    reason="PySpark not available"
)

# Fixture to create a Spark session
@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder
        .appName("polarfrost-test")
        .master("local[2]")
        .getOrCreate()
    )
    yield spark
    spark.stop()

# Fixture to create test data
@pytest.fixture
def test_data_pyspark(spark):
    schema = StructType([
        StructField("age", IntegerType()),
        StructField("gender", StringType()),
        StructField("zipcode", StringType()),
        StructField("income", IntegerType()),
        StructField("condition", StringType())
    ])
    
    data = [
        (25, "M", "12345", 50000, "A"),
        (25, "M", "12345", 55000, "B"),
        (35, "F", "12345", 60000, "A"),
        (35, "F", "12345", 65000, "B"),
        (45, "M", "67890", 70000, "A"),
        (45, "M", "67890", 75000, "B"),
        (55, "F", "67890", 80000, "A"),
        (55, "F", "67890", 85000, "B")
    ]
    
    return spark.createDataFrame(data, schema)

def test_pyspark_basic_anonymization(spark, test_data_pyspark):
    """Test basic PySpark anonymization."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    # Define output schema for the UDF
    output_schema = StructType([
        StructField("age", StringType()),
        StructField("gender", StringType()),
        StructField("zipcode", StringType()),
        StructField("condition", StringType()),
        StructField("count", IntegerType())
    ])
    
    result = mondrian_k_anonymity_spark(
        df=test_data_pyspark,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["gender", "zipcode"],
        schema=output_schema
    )
    
    # Verify the result has the expected schema
    assert set(result.columns) == {"age", "gender", "zipcode", "condition", "count"}
    
    # Verify the count is at least k for each group
    counts = result.select("count").collect()
    for row in counts:
        assert row["count"] >= 2

def test_pyspark_empty_dataframe(spark):
    """Test PySpark with empty DataFrame."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    schema = StructType([
        StructField("age", IntegerType()),
        StructField("gender", StringType()),
        StructField("condition", StringType())
    ])
    
    empty_df = spark.createDataFrame([], schema)
    output_schema = schema.add("count", IntegerType())
    
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        mondrian_k_anonymity_spark(
            df=empty_df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2,
            schema=output_schema
        )

def test_pyspark_invalid_k(spark, test_data_pyspark):
    """Test PySpark with invalid k values."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    output_schema = StructType([
        StructField("age", StringType()),
        StructField("gender", StringType()),
        StructField("condition", StringType()),
        StructField("count", IntegerType())
    ])
    
    # Test k = 0
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=0,
            schema=output_schema
        )
    
    # Test k = -1
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=-1,
            schema=output_schema
        )

def test_pyspark_missing_columns(spark, test_data_pyspark):
    """Test PySpark with missing columns."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    output_schema = StructType([
        StructField("age", StringType()),
        StructField("gender", StringType()),
        StructField("nonexistent", StringType()),
        StructField("count", IntegerType())
    ])
    
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "nonexistent"],
            sensitive_column="gender",
            k=2,
            schema=output_schema
        )

def test_pyspark_categorical_handling(spark, test_data_pyspark):
    """Test PySpark handling of categorical variables."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    output_schema = StructType([
        StructField("age", StringType()),
        StructField("gender", StringType()),
        StructField("zipcode", StringType()),
        StructField("condition", StringType()),
        StructField("count", IntegerType())
    ])
    
    result = mondrian_k_anonymity_spark(
        df=test_data_pyspark,
        quasi_identifiers=["age", "gender", "zipcode"],
        sensitive_column="condition",
        k=2,
        categorical=["gender", "zipcode"],
        schema=output_schema
    )
    
    # Verify that categorical columns are properly handled
    gender_values = [row["gender"] for row in result.collect()]
    for val in gender_values:
        # Should be either a single value or a comma-separated list
        assert "," in val or val in ["M", "F"]
