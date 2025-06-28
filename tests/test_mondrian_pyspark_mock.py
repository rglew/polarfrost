"""Mock tests for PySpark implementation of Mondrian k-anonymity."""

import pytest
import polars as pl
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from typing import Dict, Any, List, Optional, Union
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock

# Skip if PySpark is not available
pyspark = pytest.importorskip("pyspark")
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, 
    ArrayType, MapType, BooleanType
)
from pyspark.sql import DataFrame as SparkDataFrame

# Import SparkContext and SparkConf at module level
from pyspark import SparkContext, SparkConf

# Disable logging for cleaner test output
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

# Mock SparkConf
class MockSparkConf:
    def __init__(self) -> None:
        """Initialize a new MockSparkConf with an empty configuration."""
        self._conf: Dict[str, Any] = {}
    
    def setAppName(self, name: str) -> 'MockSparkConf':
        """Set the application name.
        
        Args:
            name: The application name
            
        Returns:
            This MockSparkConf instance
        """
        self._conf['spark.app.name'] = name
        return self
    
    def setMaster(self, master: str) -> 'MockSparkConf':
        """Set the master URL.
        
        Args:
            master: The master URL
            
        Returns:
            This MockSparkConf instance
        """
        self._conf['spark.master'] = master
        return self
    
    def set(self, key: str, value: Any) -> 'MockSparkConf':
        """Set a configuration property.
        
        Args:
            key: The configuration key
            value: The configuration value
            
        Returns:
            This MockSparkConf instance
        """
        self._conf[key] = value
        return self
    
    def get(self, key: str, defaultValue: Optional[Any] = None) -> Any:
        """Get a configuration property.
        
        Args:
            key: The configuration key
            defaultValue: Default value if key is not found
            
        Returns:
            The configuration value or defaultValue if key is not found
        """
        return self._conf.get(key, defaultValue)
    
    def getAll(self) -> List[Tuple[str, Any]]:
        """Get all configuration properties as a list of key-value pairs.
        
        Returns:
            A list of (key, value) tuples
        """
        return list(self._conf.items())
    
    def setIfMissing(self, key: str, value: Any) -> 'MockSparkConf':
        """Set a configuration property if it's not already set.
        
        Args:
            key: The configuration key
            value: The configuration value
            
        Returns:
            This MockSparkConf instance
        """
        if key not in self._conf:
            self._conf[key] = value
        return self

# Mock SparkContext
class MockSparkContext:
    """Mock implementation of PySpark's SparkContext for testing purposes."""
    _active_spark_context: Optional['MockSparkContext'] = None
    _jvm: Optional[MagicMock] = None
    _gateway: Optional[MagicMock] = None
    
    def __init__(self, master: Optional[str] = None, appName: Optional[str] = None, 
                 conf: Optional['MockSparkConf'] = None, **kwargs: Any) -> None:
        """Initialize a new MockSparkContext.
        
        Args:
            master: Cluster URL to connect to (e.g., 'local[2]')
            appName: A name for the application
            conf: Optional Spark configuration
            **kwargs: Additional keyword arguments (ignored)
        """
        self.master = master or 'local[2]'
        self.appName = appName or 'polarfrost-test'
        self._conf = conf or MockSparkConf()
        self._jsc = MagicMock()
        self._jvm = MagicMock()
        self._gateway = MagicMock()
        self._calls: List[Any] = []
        self._active_spark_context = self
        
        # Set default configurations
        self._conf.setAppName(self.appName)
        self._conf.setMaster(self.master)
        self._conf.setIfMissing('spark.driver.memory', '1g')
        self._conf.setIfMissing('spark.executor.memory', '1g')
        self._conf.setIfMissing('spark.sql.shuffle.partitions', '2')
        self._conf.setIfMissing('spark.default.parallelism', '2')
        self._conf.setIfMissing('spark.driver.extraJavaOptions', 
                               '-Dio.netty.tryReflectionSetAccessible=true')
        self._conf.setIfMissing('spark.executor.extraJavaOptions', 
                               '-Dio.netty.tryReflectionSetAccessible=true')
        
        # Set up JVM and gateway mocks
        self._jvm = MagicMock()
        self._gateway = MagicMock()
        self._gateway.jvm = self._jvm
        
    @property
    def version(self) -> str:
        """Get the version of Spark.
        
        Returns:
            The Spark version string
        """
        return '3.4.0'
        
    def getConf(self) -> 'MockSparkConf':
        """Get the Spark configuration.
        
        Returns:
            The Spark configuration object
        """
        return self._conf
        
    def stop(self) -> None:
        """Stop the SparkContext."""
        MockSparkContext._active_spark_context = None
        
    @classmethod
    def getOrCreate(cls, *args: Any, **kwargs: Any) -> 'MockSparkContext':
        """Get the active SparkContext or create a new one if none exists.
        
        Args:
            *args: Positional arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor
            
        Returns:
            The active MockSparkContext instance
        """
        if cls._active_spark_context is None:
            cls._active_spark_context = cls(*args, **kwargs)
        return cls._active_spark_context

# Mock SparkSession
class MockSparkSession:
    _instantiatedSession = None
    _activeSession = None
    
    class Builder:
        def __init__(self):
            self.appName = None
            self.master = None
            self.config = {}
        
        def appName(self, name: str) -> 'MockSparkSession.Builder':
            """Set the application name."""
            self.appName = name
            return self
            
        def master(self, master: str) -> 'MockSparkSession.Builder':
            """Set the master URL."""
            self.master = master
            return self
            
        def config(self, key: Optional[str] = None, value: Optional[Any] = None) -> 'MockSparkSession':
            if key is not None and value is not None:
                self.config[key] = value
            return self
            
        def getOrCreate(self) -> 'MockSparkSession':
            if MockSparkSession._instantiatedSession is None:
                MockSparkSession._instantiatedSession = MockSparkSession()
            return MockSparkSession._instantiatedSession
    
    def __init__(self, sparkContext=None, appName=None, master=None, **kwargs):
        self._sc = sparkContext or MockSparkContext(master=master, appName=appName)
        self._jvm = self._sc._jvm
        self._jsparkSession = MagicMock()
        self._conf = self._sc._conf
        self._catalog = MagicMock()
        self._udf = MagicMock()
        self._version = '3.4.0'
        self._calls = []
        
        # Set up default configurations
        self._conf.setAppName(appName or 'polarfrost-test')
        self._conf.setMaster(master or 'local[2]')
        
        # Set up mock methods
        self.createDataFrame = MagicMock(side_effect=self._mock_create_dataframe)
        self.sql = MagicMock(return_value=MockSparkDataFrame())
        self.stop = MagicMock()
        self.catalog = MagicMock()
        self.catalog.listDatabases.return_value = []
        self.catalog.listTables.return_value = []
        
        # Set as active session
        MockSparkSession._activeSession = self
        
    def _mock_create_dataframe(self, data: Any, schema: Optional[Any] = None, samplingRatio: Optional[float] = None, verifySchema: bool = True) -> 'MockSparkDataFrame':
        """Mock implementation of createDataFrame"""
        if hasattr(data, 'toPandas'):  # If it's a DataFrame
            return data
        elif isinstance(data, list):
            if not data:  # Empty list
                return MockSparkDataFrame(pandas_df=pd.DataFrame())
            elif isinstance(data[0], (list, tuple)):
                if schema and hasattr(schema, 'fieldNames'):
                    columns = schema.fieldNames()
                else:
                    columns = [f'col_{i}' for i in range(len(data[0]))]
                return MockSparkDataFrame(pd.DataFrame(data, columns=columns))
            else:  # Single column
                return MockSparkDataFrame(pd.DataFrame({'value': data}))
        elif hasattr(data, 'toPandas'):  # Handle RDDs or other DataFrame-like objects
            return MockSparkDataFrame(pandas_df=data.toPandas())
        return MockSparkDataFrame(pandas_df=pd.DataFrame(data))
        
    @property
    def _wrapped(self):
        return self
        
    @property
    def sparkContext(self) -> 'MockSparkContext':
        """Get the SparkContext associated with this SparkSession."""
        return self._sc
    
    def createDataFrame(
        self, 
        data: Any, 
        schema: Optional[Any] = None, 
        samplingRatio: Optional[float] = None, 
        verifySchema: bool = True
    ) -> 'MockSparkDataFrame':
        """Create a DataFrame from an RDD, a list or a pandas.DataFrame.
        
        Args:
            data: The input data to create the DataFrame from
            schema: Optional schema for the DataFrame
            samplingRatio: Not used, for API compatibility
            verifySchema: Not used, for API compatibility
            
        Returns:
            A new MockSparkDataFrame instance
        """
        return MockSparkDataFrame(data, schema)
    
    def newSession(self) -> 'MockSparkSession':
        """Return a new session with the same configuration as this session.
        
        Returns:
            A new MockSparkSession instance with the same configuration
        """
        return self
    
    @classmethod
    def getActiveSession(cls) -> 'MockSparkSession':
        """Get the currently active SparkSession, or create a new one if none exists.
        
        Returns:
            The active MockSparkSession instance
        """
        if cls._activeSession is None:
            cls._activeSession = cls()
        return cls._activeSession
    
    @classmethod
    def builder(cls) -> 'MockSparkSession.Builder':
        """Create a builder for constructing a SparkSession.
        
        Returns:
            A new Builder instance for creating MockSparkSession
        """
        return cls.Builder()
    
    def stop(self) -> None:
        """Stop the SparkSession."""
        MockSparkSession._instantiatedSession = None
        MockSparkSession._activeSession = None

# Set up active SparkSession for testing
@pytest.fixture(scope="function")
def spark_session():
    """Fixture to provide a SparkSession for testing."""
    # Save any existing SparkSession
    from pyspark.sql import SparkSession
    from pyspark import SparkContext
    
    # Create a new mock session
    mock_session = MockSparkSession()
    
    # Patch the SparkSession and SparkContext classes
    with patch('pyspark.sql.SparkSession', autospec=True) as mock_spark_session_class, \
         patch('pyspark.SparkContext', autospec=True) as mock_spark_context_class, \
         patch('pyspark.sql.SparkSession.builder', new_callable=MockSparkSession.Builder), \
         patch('pyspark.SparkContext.getOrCreate', return_value=mock_session.sparkContext):
        
        # Configure the mock SparkSession class
        mock_spark_session_class.getActiveSession.return_value = mock_session
        mock_spark_session_class._instantiatedSession = mock_session
        mock_spark_session_class.builder.return_value.getOrCreate.return_value = mock_session
        
        # Configure the mock SparkContext class
        mock_spark_context_class._active_spark_context = mock_session.sparkContext
        
        # Set up the active session
        MockSparkSession._instantiatedSession = mock_session
        MockSparkSession._activeSession = mock_session
        
        yield mock_session
        
        # Cleanup
        MockSparkSession._instantiatedSession = None
        MockSparkSession._activeSession = None
        MockSparkContext._active_spark_context = None

# Enhanced Mock PySpark DataFrame for testing
class MockSparkDataFrame:
    """Mock implementation of PySpark DataFrame for testing purposes."""
    
    def __init__(
        self, 
        data: Optional[Union[pd.DataFrame, List, Tuple]] = None, 
        schema: Optional[Any] = None, 
        pandas_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Initialize a new MockSparkDataFrame.
        
        Args:
            data: Input data as a pandas DataFrame, list, or tuple
            schema: Optional schema for the DataFrame
            pandas_df: Optional pandas DataFrame to use directly
        """
        # Initialize pandas DataFrame
        if pandas_df is not None:
            self.pandas_df = pandas_df.copy()
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                self.pandas_df = data.copy()
            else:
                # Handle list of rows
                if data and isinstance(data[0], (list, tuple)):
                    if schema and hasattr(schema, 'fieldNames'):
                        columns = schema.fieldNames()
                        self.pandas_df = pd.DataFrame(data, columns=columns)
                    else:
                        self.pandas_df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(len(data[0]))])
                else:
                    # Single column data
                    self.pandas_df = pd.DataFrame({"value": data})
        else:
            # Create empty DataFrame without triggering pandas config recursion
            self.pandas_df = pd.DataFrame({})
        
        # Set schema if provided
        self._schema = schema
        self._is_empty: bool = self.pandas_df.empty
        
        # Initialize columns and data types
        self._columns: List[str] = list(self.pandas_df.columns) if not self.pandas_df.empty else []
        self._dtypes: Dict[str, str] = {col: str(dtype) for col, dtype in self.pandas_df.dtypes.items()} if not self.pandas_df.empty else {}
        
        # Mock rdd.isEmpty()
        self.rdd: MagicMock = MagicMock()
        self.rdd.isEmpty.return_value = self._is_empty
        
        # Mock count() and __len__
        self.count: MagicMock = MagicMock(return_value=len(self.pandas_df))
        
        # Define __len__ as a proper method with type hints
        def __len__(self) -> int:
            return len(self.pandas_df)
            
        self.__len__ = __len__.__get__(self)
        
        # Initialize sparkSession as None to avoid recursion
        self._spark_session: Optional[MockSparkSession] = None
        
        # Mock Java API
        self._jdf: MagicMock = MagicMock()
        self._jdf.schema.return_value.jsonValue.return_value = {
            'type': 'struct',
            'fields': [{'name': col, 'type': 'string', 'nullable': True} for col in self._columns]
        } if self._columns else {'type': 'struct', 'fields': []}
        
        # Mock Java sequence
        self._jseq: MagicMock = MagicMock()
        self._jseq.isEmpty.return_value = self._is_empty
        
    @property
    def sparkSession(self) -> 'MockSparkSession':
        """Get the SparkSession associated with this DataFrame.
        
        Returns:
            The associated MockSparkSession instance, creating one if it doesn't exist
        """
        if self._spark_session is None:
            self._spark_session = MockSparkSession()
        return self._spark_session
        
    @sparkSession.setter
    def sparkSession(self, value: 'MockSparkSession') -> None:
        """Set the SparkSession associated with this DataFrame.
        
        Args:
            value: The MockSparkSession instance to associate with this DataFrame
        """
        self._spark_session = value
        
    def schema(self) -> Any:
        """Get the schema of this DataFrame.
        
        Returns:
            The schema object or None if not set
        """
        return self._schema
    
    def withColumn(self, name: str, col: Any) -> 'MockSparkDataFrame':
        """Mock implementation of withColumn.
        
        Args:
            name: Name of the new column
            col: Column expression
            
        Returns:
            This MockSparkDataFrame instance (for method chaining)
        """
        # Simple mock for withColumn - just return self for chaining
        return self
    
    def groupBy(self, *cols: Any) -> 'MockGroupedData':
        """Group the DataFrame by the specified columns.
        
        Args:
            *cols: Column names or expressions to group by
            
        Returns:
            A MockGroupedData instance for the grouped data
        """
        # Create a mock GroupedData object
        class MockGroupedData:
            def __init__(self, parent_df: 'MockSparkDataFrame') -> None:
                self.parent_df = parent_df
            
            def applyInPandas(self, func: Callable[[pd.DataFrame], pd.DataFrame], 
                           schema: Any) -> 'MockSparkDataFrame':
                """Apply a function to each group and return the result as a DataFrame.
                
                Args:
                    func: Function to apply to each group's pandas DataFrame
                    schema: Schema of the resulting DataFrame
                    
                Returns:
                    A new MockSparkDataFrame containing the results
                    
                Raises:
                    Exception: If there's an error applying the function
                """
                try:
                    # Get the pandas DataFrame from the parent
                    pdf = self.parent_df.pandas_df
                    
                    # If DataFrame is empty, return an empty DataFrame with the expected schema
                    if pdf.empty:
                        return MockSparkDataFrame(pd.DataFrame(columns=schema.fieldNames()))
                        
                    # Call the function with the pandas DataFrame
                    result_df = func(pdf)
                    
                    # Convert back to mock DataFrame with the provided schema
                    return MockSparkDataFrame(pandas_df=result_df, schema=schema)
                except Exception as e:
                    # Re-raise the exception with a more informative message
                    raise type(e)(f"Error in applyInPandas: {str(e)}") from e
        
        return MockGroupedData(self)
    
    def select(self, *cols: Any) -> 'MockSparkDataFrame':
        """Select columns from the DataFrame.
        
        Args:
            *cols: Column names or column expressions to select
            
        Returns:
            A new MockSparkDataFrame with the selected columns
        """
        if not cols:
            return self
            
        # Handle both column names and column objects
        col_names: List[str] = []
        for col in cols:
            if hasattr(col, '_jc'):  # Column object
                col_names.append(str(col))
            else:  # String column name
                col_names.append(col)
        
        # Create a new DataFrame with selected columns
        if self.pandas_df.empty:
            return MockSparkDataFrame(pd.DataFrame(columns=col_names))
        else:
            return MockSparkDataFrame(pandas_df=self.pandas_df[col_names])
    
    def collect(self) -> List[MagicMock]:
        """Collect all records from the DataFrame as a list.
        
        Returns:
            A list of Row objects as MagicMock instances
        """
        return [MagicMock(asDict=lambda row=row: row) for row in self.pandas_df.to_dict('records')]
    
    def columns(self) -> List[str]:
        """Get the list of column names.
        
        Returns:
            A list of column names
        """
        return self._columns
    
    def dtypes(self) -> List[Tuple[str, str]]:
        """Get the data types of each column.
        
        Returns:
            A list of (column_name, type_string) tuples
        """
        return [(col, self._dtypes.get(col, 'string')) for col in self._columns]
    
    def createOrReplaceTempView(self, name: str) -> None:
        """Create or replace a temporary view with the given name.
        
        Args:
            name: Name of the temporary view
        """
        # No-op for testing
        pass
        
    def isEmpty(self) -> bool:
        """Check if the DataFrame is empty.
        
        Returns:
            True if the DataFrame is empty, False otherwise
        """
        return self._is_empty
        
    def rdd(self) -> Any:
        """Convert the DataFrame to an RDD.
        
        Returns:
            A mock RDD object
        """
        return self._rdd
        
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying pandas DataFrame if it exists.
        
        Args:
            name: Name of the attribute to get
            
        Returns:
            The attribute value from the pandas DataFrame
            
        Raises:
            AttributeError: If the attribute doesn't exist on either the mock or pandas DataFrame
        """
        if hasattr(self.pandas_df, name):
            return getattr(self.pandas_df, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Helper function to create test data
def create_test_data() -> pd.DataFrame:
    """Create test data with various data types and edge cases."""
    return pd.DataFrame({
        'age': [25, 25, 35, 35, 45, 45, 55, 55],
        'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
        'zipcode': ['12345', '12345', '12345', '12345', '67890', '67890', '67890', '67890'],
        'income': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000],
        'condition': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'has_children': [True, False, True, False, True, False, True, False],
        'score': [0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    })

@pytest.fixture(scope='function')
def mock_spark_session() -> MockSparkSession:
    """Fixture to mock SparkSession with proper context setup."""
    # Reset class variables to ensure clean state
    MockSparkSession._activeSession = None
    MockSparkSession._instantiatedSession = None
    
    # Create a mock SparkConf with test configurations
    mock_conf = MockSparkConf()
    mock_conf.setAppName('polarfrost-test')
    mock_conf.setMaster('local[2]')
    mock_conf.set('spark.driver.memory', '1g')
    mock_conf.set('spark.executor.memory', '1g')
    mock_conf.set('spark.sql.shuffle.partitions', '2')
    mock_conf.set('spark.default.parallelism', '2')
    
    # Create a mock SparkContext with the configuration
    mock_sc = MockSparkContext(conf=mock_conf)
    
    # Create a mock SparkSession
    mock_session = MockSparkSession(sparkContext=mock_sc)
    
    # Set up test data
    data = [
        (25, "M", "12345", 50000, "Flu"),
        (30, "F", "12345", 60000, "Cold"),
        (35, "M", "54321", 70000, "Flu"),
        (40, "F", "54321", 80000, "Cold")
    ]
    
    schema = StructType([
        StructField("age", IntegerType()),
        StructField("gender", StringType()),
        StructField("zipcode", StringType()),
        StructField("income", IntegerType()),
        StructField("condition", StringType())
    ])
    
    # Create a test DataFrame
    test_df = MockSparkDataFrame(pd.DataFrame(
        data, 
        columns=[f.name for f in schema.fields]
    ))
    
    # Configure mock methods
    mock_session.createDataFrame.return_value = test_df
    mock_session.sql.return_value = test_df
    mock_session.catalog = MagicMock()
    mock_session.catalog.listDatabases.return_value = []
    mock_session.catalog.listTables.return_value = []
    
    # Set up the active session
    MockSparkSession._instantiatedSession = mock_session
    MockSparkSession._activeSession = mock_session
    
    # Patch the SparkSession builder to return our mock session
    with patch('pyspark.sql.SparkSession.builder') as mock_builder, \
         patch('pyspark.SparkContext', return_value=mock_sc), \
         patch('pyspark.conf.SparkConf', return_value=mock_conf):
        
        # Configure the mock builder
        mock_builder.getOrCreate.return_value = mock_session
        mock_builder.appName.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        
        yield mock_session
    
    # Clean up after test
    mock_session.stop()
    MockSparkSession._activeSession = None
    MockSparkSession._instantiatedSession = None
    MockSparkContext._active_spark_context = None

@pytest.fixture
def test_data_pyspark() -> MockSparkDataFrame:
    """Create a test PySpark DataFrame."""
    # Define test data
    data = [
        (25, "M", "12345", 50000, "A", True, 0.8),
        (25, "M", "12345", 55000, "C", True, 0.95),
        (35, "F", "12345", 60000, "B", False, 0.7),
        (35, "F", "12345", 65000, "A", True, 0.85),
        (45, "M", "67890", 70000, "B", False, 0.65),
        (45, "M", "67890", 75000, "A", True, 0.9),
        (40, "F", "54321", 70000, "B", False, 0.65),
        (40, "F", "54321", 70000, "B", False, 0.65),
    ]
    
    # Define schema with proper nullability
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("zipcode", StringType(), True),
        StructField("income", IntegerType(), True),
        StructField("condition", StringType(), True),
        StructField("has_children", BooleanType(), True),
        StructField("score", FloatType(), True)
    ])
    
    # Create pandas DataFrame with proper column names
    columns = [field.name for field in schema.fields]
    pdf = pd.DataFrame(data, columns=columns)
    
    # Create mock DataFrame
    mock_df = MockSparkDataFrame(pandas_df=pdf, schema=schema)
    
    # Configure mock behavior
    mock_df.rdd.isEmpty.return_value = False
    mock_df.count.return_value = len(pdf)
    
    # Mock the groupBy method
    def mock_group_by(*cols):
        class MockGroupedData:
            def __init__(self, df):
                self.df = df
                
            def applyInPandas(self, func, schema):
                # For testing, just apply the function to the entire DataFrame
                result_df = func(self.df.pandas_df)
                return MockSparkDataFrame(pandas_df=result_df, schema=schema)
                
        return MockGroupedData(mock_df)
    
    mock_df.groupBy = MagicMock(side_effect=mock_group_by)
    
    return mock_df

@pytest.fixture
def output_schema() -> StructType:
    """Define output schema for testing."""
    return StructType([
        StructField("age", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("condition", StringType(), True),
        StructField("count", IntegerType(), False)
    ])

def test_pyspark_mock_basic(test_data_pyspark: SparkDataFrame, output_schema: StructType) -> None:
    """Test basic functionality with mock PySpark DataFrame."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    # Test with mock
    with patch('pyspark.sql.functions.pandas_udf') as mock_pandas_udf, \
         patch('pyspark.sql.functions.col') as mock_col, \
         patch('pyspark.sql.functions.lit') as mock_lit:
        
        # Mock pandas_udf
        mock_pandas_udf.side_effect = lambda *args, **kwargs: (lambda f: f)
        
        # Mock col() and lit()
        mock_col.side_effect = lambda x: x
        mock_lit.side_effect = lambda x: x
        
        # Call the function
        result = mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender", "zipcode"],
            sensitive_column="condition",
            k=2,
            categorical=["gender", "zipcode"],
            schema=output_schema
        )
        
        # Verify the result is not None and has the expected schema
        assert result is not None
        assert hasattr(result, 'schema')
        
        # Verify the schema matches the expected output schema
        if hasattr(result, '_schema') and result._schema is not None:
            actual_fields = {f.name: str(f.dataType) for f in result._schema.fields}
            expected_fields = {f.name: str(f.dataType) for f in output_schema.fields}
            assert actual_fields == expected_fields
        
        # Verify the UDF was registered
        mock_pandas_udf.assert_called()
        
        # Verify the result has the expected columns
        result_columns = set(result.columns())
        expected_columns = {"age", "gender", "zipcode", "condition", "count"}
        assert result_columns.issuperset(expected_columns)


def test_pyspark_mock_with_none_values(test_data_pyspark: SparkDataFrame, output_schema: StructType) -> None:
    """Test handling of None/NA values in PySpark DataFrame."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    # Add some None values to the test data
    pdf = test_data_pyspark.pandas_df.copy()
    pdf.loc[0, 'age'] = None
    pdf.loc[1, 'gender'] = None
    
    # Create a new mock DataFrame with the updated data
    test_data = MockSparkDataFrame(pdf, schema=test_data_pyspark._schema)
    
    with patch('pyspark.sql.functions.pandas_udf') as mock_pandas_udf:
        mock_pandas_udf.side_effect = lambda *args, **kwargs: (lambda f: f)
        
        result = mondrian_k_anonymity_spark(
            df=test_data,
            quasi_identifiers=["age", "gender", "zipcode"],
            sensitive_column="condition",
            k=2,
            categorical=["gender", "zipcode"],
            schema=output_schema
        )
        
        assert result is not None
        assert hasattr(result, 'schema')


def test_pyspark_mock_with_single_partition(test_data_pyspark: SparkDataFrame, output_schema: StructType) -> None:
    """Test with data that should result in a single partition."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    # Create data where all records are identical in quasi-identifiers
    pdf = test_data_pyspark.pandas_df.copy()
    pdf['age'] = 30
    pdf['gender'] = 'M'
    pdf['zipcode'] = '12345'
    
    test_data = MockSparkDataFrame(pdf, schema=test_data_pyspark._schema)
    
    with patch('pyspark.sql.functions.pandas_udf') as mock_pandas_udf:
        mock_pandas_udf.side_effect = lambda *args, **kwargs: (lambda f: f)
        
        result = mondrian_k_anonymity_spark(
            df=test_data,
            quasi_identifiers=["age", "gender", "zipcode"],
            sensitive_column="condition",
            k=2,
            categorical=["gender", "zipcode"],
            schema=output_schema
        )
        
        assert result is not None
        assert hasattr(result, 'schema')


def test_pyspark_mock_with_large_k(test_data_pyspark: SparkDataFrame, output_schema: StructType) -> None:
    """Test with k larger than the number of records."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    with patch('pyspark.sql.functions.pandas_udf') as mock_pandas_udf:
        mock_pandas_udf.side_effect = lambda *args, **kwargs: (lambda f: f)
        
        # k is larger than the number of records (8)
        result = mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender", "zipcode"],
            sensitive_column="condition",
            k=10,
            categorical=["gender", "zipcode"],
            schema=output_schema
        )
        
        assert result is not None
        assert hasattr(result, 'schema')
        
        # Should have a single group with all records
        result_pdf = result.pandas_df
        assert len(result_pdf) == 1
        assert result_pdf['count'].iloc[0] == len(test_data_pyspark.pandas_df)

def test_pyspark_mock_empty_dataframe(mock_spark_session: MockSparkSession) -> None:
    """Test with empty PySpark DataFrame."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark

    # Create an empty mock PySpark DataFrame
    empty_pdf = pd.DataFrame(columns=["age", "gender", "condition"])
    mock_df = MockSparkDataFrame(empty_pdf)

    # Mock schema
    output_schema = StructType([
        StructField("age", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("condition", StringType(), True),
        StructField("count", IntegerType(), False)
    ])

    # Test with empty DataFrame - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        mondrian_k_anonymity_spark(
            df=mock_df,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2,
            schema=output_schema
        )
    
    # Check that the error message is correct
    assert "Input DataFrame cannot be empty" in str(exc_info.value)
    
    # Verify rdd.isEmpty() was called
    mock_df.rdd.isEmpty.assert_called_once()

def test_pyspark_mock_invalid_k(test_data_pyspark: SparkDataFrame, mock_spark_session: MockSparkSession) -> None:
    """Test with invalid k values."""
    from polarfrost.mondrian import mondrian_k_anonymity_spark
    
    # Mock schema
    output_schema = StructType([
        StructField("age", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("condition", StringType(), True),
        StructField("count", IntegerType(), False)
    ])
    
    # Test with k = 0
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=0,
            schema=output_schema
        )
    
    # Test with negative k
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=-1,
            schema=output_schema
        )
    
    # Test with non-integer k
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k="invalid",
            schema=output_schema
        )
        
    # Test with k as a float
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=2.5,  # Invalid: float
            schema=output_schema
        )
    
    # Test with k = -1
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mondrian_k_anonymity_spark(
            df=test_data_pyspark,
            quasi_identifiers=["age", "gender"],
            sensitive_column="condition",
            k=-1,
            schema=output_schema
        )
