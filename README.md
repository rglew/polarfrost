# Polarfrost ❄️

[![PyPI](https://img.shields.io/pypi/v/polarfrost)](https://pypi.org/project/polarfrost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/polarfrost)](https://pypi.org/project/polarfrost/)
[![CI](https://github.com/rglew/polarfrost/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rglew/polarfrost/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rglew/polarfrost/branch/main/graph/badge.svg)](https://codecov.io/gh/rglew/polarfrost)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/github/actions/workflow/status/rglew/polarfrost/ci.yml?branch=main&label=tests)](https://github.com/rglew/polarfrost/actions/workflows/ci.yml)

A high-performance k-anonymity implementation using Polars, featuring the Mondrian algorithm for efficient privacy-preserving data analysis.

## ✨ Features

- 🚀 **Blazing Fast**: Leverages Polars for high-performance data processing
- 🚀 **High Performance**: Optimized for speed with Polars' lazy execution
- 📊 **Data Utility**: Preserves data utility while ensuring privacy
- 🐍 **Pythonic API**: Simple and intuitive interface
- 🔒 **Privacy-Preserving**: Implements k-anonymity to protect sensitive information
- 🛡 **Robust Input Validation**: Comprehensive validation of input parameters
- 🧪 **High Test Coverage**: 80%+ test coverage with comprehensive edge case testing
- 📦 **Production Ready**: Well-tested and ready for production use
- 🔄 **Flexible Input**: Works with both eager and lazy Polars DataFrames
- 📈 **Scalable**: Efficiently handles both small and large datasets

## 📦 Installation

```bash
# Basic installation
pip install polarfrost

# With PySpark support
pip install "polarfrost[spark]"

# For development
git clone https://github.com/rglew/polarfrost.git
cd polarfrost
pip install -e ".[dev]"
```

## 🧪 Testing

To run the test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests (excluding PySpark tests that require Java)
pytest -k "not test_mondrian_pyspark.py"

# Run mock PySpark tests only
pytest tests/test_mondrian_pyspark_mock.py
```

### PySpark Testing Notes
- The test suite includes both real PySpark tests and mock PySpark tests
- Real PySpark tests require Java 8 or 11 to be installed
- Mock PySpark tests run without Java and are used in CI
- To run real PySpark tests, ensure Java is installed and set `JAVA_HOME`
- The mock tests provide equivalent test coverage without Java dependencies

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/test_mondrian.py

# Run only edge case tests
pytest tests/test_mondrian_edge_cases.py

# Run with coverage report
pytest --cov=polarfrost --cov-report=term-missing
```

## 🚀 Quick Start

### Basic Usage with Polars (Mondrian Algorithm)

#### Standard Mondrian k-Anonymity

The standard implementation groups records and returns one representative row per group:

```python
import polars as pl
from polarfrost import mondrian_k_anonymity

# Sample data
data = {
    "age": [25, 25, 35, 35, 45, 45, 55, 55],
    "gender": ["M", "M", "F", "F", "M", "M", "F", "F"],
    "zipcode": ["12345", "12345", "12345", "12345", "67890", "67890", "67890", "67890"],
    "income": [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000],
    "medical_condition": ["A", "B", "A", "B", "A", "B", "A", "B"]
}
df = pl.DataFrame(data)

# Apply k-anonymity with k=2
anonymized = mondrian_k_anonymity(
    df,
    quasi_identifiers=["age", "gender", "zipcode"],
    sensitive_column="medical_condition",
    k=2,
    categorical=["gender", "zipcode"]
)

print(anonymized)

### Alternative Implementation with Row Preservation

For use cases where you need to preserve the original number of rows (1:1 input-output mapping), use `mondrian_k_anonymity_alt`:

```python
from polarfrost import mondrian_k_anonymity_alt

# Apply k-anonymity while preserving row count
anonymized = mondrian_k_anonymity_alt(
    df.lazy(),  # Must be a LazyFrame
    quasi_identifiers=["age", "gender", "zipcode"],
    sensitive_column="medical_condition",
    k=2,
    categorical=["gender", "zipcode"],
    group_columns=["org_id"]  # Optional: group by organization
)

# Collect the results (since we started with a LazyFrame)
anonymized_df = anonymized.collect()
print(anonymized_df)
```

#### Key Differences from Standard Implementation

1. **Row Preservation**: Maintains original row count (1:1 input-output mapping)
2. **In-Place Anonymization**: Modifies QI columns directly instead of creating new ones
3. **Group Processing**: Supports hierarchical data through `group_columns`
4. **Small Group Handling**: Masks sensitive data in groups smaller than k
5. **LazyFrame Requirement**: Input must be a Polars LazyFrame for efficiency

#### When to Use Which Version

- Use `mondrian_k_anonymity` when you need grouped results and don't need to maintain row order
- Use `mondrian_k_anonymity_alt` when you need to:
  - Preserve the original number of rows
  - Maintain relationships with other tables through foreign keys
  - Process hierarchical data with different k-values per group
  - Keep non-QI columns unchanged
```

### Using PySpark for Distributed Processing (Mondrian Algorithm)

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from polarfrost import mondrian_k_anonymity

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PolarFrostExample") \
    .getOrCreate()

# Sample schema
schema = StructType([
    StructField("age", IntegerType()),
    StructField("gender", StringType()),
    StructField("zipcode", StringType()),
    StructField("income", IntegerType()),
    StructField("medical_condition", StringType())
])

# Sample data
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

# Create Spark DataFrame
df = spark.createDataFrame(data, schema)

# Apply k-anonymity with PySpark
anonymized = mondrian_k_anonymity(
    df,
    quasi_identifiers=["age", "gender", "zipcode"],
    sensitive_column="medical_condition",
    k=2,
    categorical=["gender", "zipcode"],
    schema=df.schema  # Required for PySpark
)

anonymized.show()
```

## 📚 API Reference

### `mondrian_k_anonymity`

```python
def mondrian_k_anonymity(
    df: Union[pl.DataFrame, pl.LazyFrame, "pyspark.sql.DataFrame"],
    quasi_identifiers: List[str],
    sensitive_column: str,
    k: int,
    categorical: Optional[List[str]] = None,
    schema: Optional["pyspark.sql.types.StructType"] = None,
) -> Union[pl.DataFrame, "pyspark.sql.DataFrame"]:
    """
    Apply Mondrian k-anonymity to the input data.

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
```

## 🔍 Development Notes

### Testing Strategy

- **Unit Tests**: Core functionality of all modules
- **Mock Tests**: PySpark functionality without Java dependencies
- **Edge Cases**: Handling of boundary conditions and unusual inputs
- **Input Validation**: Comprehensive validation of all function parameters
- **Backend Compatibility**: Tests for both Polars and PySpark backends

### PySpark Implementation

The PySpark implementation includes mock versions of key classes for testing:
- `MockSparkConf`: Mocks Spark configuration
- `MockSparkContext`: Mocks the Spark context
- `MockSparkSession`: Mocks the Spark session
- `MockSparkDataFrame`: Mocks Spark DataFrames with pandas backend

These mocks allow testing PySpark functionality without requiring a Java runtime.

## 🔍 Algorithms

### Mondrian k-Anonymity Algorithm

The Mondrian algorithm is a multidimensional partitioning approach that recursively splits the data along attribute values to create anonymized groups. Here's how it works in detail:

#### Algorithm Steps:

1. **Initialization**: Start with the entire dataset and the list of quasi-identifiers (QIs).

2. **Partitioning**:
   - Find the dimension (QI) with the widest range of values
   - Find the median value of that dimension
   - Split the data into two partitions at the median

3. **Anonymity Check**:
   - For each partition, check if it contains at least k records
   - If any partition has fewer than k records, undo the split
   - If all partitions have at least k records, keep the split

4. **Recursion**:
   - Recursively apply the partitioning to each new partition
   - Stop when no more valid splits can be made

5. **Generalization**:
   - For each final partition, replace QI values with their range or category
   - Keep sensitive attributes as-is but ensure k-anonymity is maintained

#### Example: Patient Data Anonymization

**Original Data (k=2):**

| Age | Gender | Zipcode | Condition       |
|-----|--------|---------|-----------------|
| 28  | M      | 10001   | Heart Disease   |
| 29  | M      | 10002   | Cancer          |
| 30  | F      | 10003   | Diabetes        |
| 31  | F      | 10004   | Heart Disease   |
| 32  | M      | 10005   | Asthma          |
| 33  | M      | 10006   | Diabetes        |
| 34  | F      | 10007   | Cancer          |
| 35  | F      | 10008   | Asthma          |

**After Mondrian k-Anonymization (k=2):**

| Age      | Gender | Zipcode | Condition       | Count |
|----------|--------|---------|-----------------|-------|
| [28-29]  | M      | 1000*   | Heart Disease   | 2     |
| [28-29]  | M      | 1000*   | Cancer          | 2     |
| [30-31]  | F      | 1000*   | Diabetes        | 1     |
| [30-31]  | F      | 1000*   | Heart Disease   | 1     |
| [32-33]  | M      | 1000*   | Asthma          | 1     |
| [32-33]  | M      | 1000*   | Diabetes        | 1     |
| [34-35]  | F      | 1000*   | Cancer          | 1     |
| [34-35]  | F      | 1000*   | Asthma          | 1     |

**Final Anonymized Groups (k=2):**

| Age      | Gender | Zipcode | Conditions              | Count |
|----------|--------|---------|-------------------------|-------|
| [28-29]  | M      | 1000*   | {Heart Disease, Cancer} | 2     |
| [30-31]  | F      | 1000*   | {Diabetes, Heart Disease}| 2     |
| [32-33]  | M      | 1000*   | {Asthma, Diabetes}      | 2     |
| [34-35]  | F      | 1000*   | {Cancer, Asthma}        | 2     |

#### Key Observations:

1. **k=2 Anonymity**: Each group contains exactly 2 records
2. **Generalization**:
   - Ages are generalized to ranges
   - Zipcodes are truncated to 4 digits (1000*)
   - Sensitive conditions are preserved but grouped
3. **Privacy**: No individual can be uniquely identified by the quasi-identifiers
4. **Utility**: The data remains useful for analysis (e.g., "2 males aged 28-29 in zip 1000* have heart disease or cancer")

### Clustering-Based k-Anonymity (Upcoming)

Coming soon: Support for clustering-based k-anonymity with multiple algorithms:
- **FCBG (Fast Clustering-Based Generalization)**: Groups similar records using clustering
- **RSC (Randomized Single-Clustering)**: Uses a single clustering pass with randomization
- **Random Clustering**: Random assignment while maintaining k-anonymity

### Choosing the Right Algorithm

- **Mondrian**: Best for datasets with clear partitioning dimensions and when you need to preserve the utility of numerical ranges
- **Clustering-based**: Better for datasets where natural clusters exist in the data
- **Random**: Provides basic k-anonymity with minimal computational overhead but may have lower data utility

## 🛡 Input Validation

PolarFrost performs comprehensive input validation to ensure data integrity:

### DataFrame Validation
- Validates input is a Polars or PySpark DataFrame
- Handles both eager and lazy evaluation modes
- Verifies DataFrame is not empty
- Validates column existence and types

### Parameter Validation
- `k` must be a positive integer
- `quasi_identifiers` must be a non-empty list of existing columns
- `sensitive_column` must be a single existing column
- `categorical` columns must be a subset of quasi-identifiers

### Edge Cases Handled
- Empty DataFrames
- Missing or NULL values
- Single record partitions
- k larger than dataset size
- Mixed data types in columns
- Duplicate column names

### Error Messages
Clear, descriptive error messages help identify and fix issues quickly:
```python
# Example error for invalid k value
ValueError: k must be a positive integer, got 'invalid'

# Example error for missing columns
ValueError: Columns not found in DataFrame: ['nonexistent_column']
```

## 🧪 Testing

PolarFrost includes extensive test coverage with over 80% code coverage:

### Test Categories
- ✅ **Unit Tests**: Core functionality of all modules
- 🔍 **Edge Cases**: Handling of boundary conditions and unusual inputs
- 🛡 **Input Validation**: Comprehensive validation of all function parameters
- 🔄 **Backend Compatibility**: Tests for both Polars and PySpark backends
- 🐛 **Error Handling**: Proper error messages and exception handling

### Running Tests

```bash
# Run all tests
pytest --cov=polarfrost --cov-report=term-missing tests/

# Run tests matching a specific pattern
pytest -k "test_mondrian" --cov=polarfrost --cov-report=term-missing

# Run with detailed coverage report
pytest --cov=polarfrost --cov-report=html && open htmlcov/index.html
```

### Test Coverage
Current test coverage includes:
- 96% coverage for clustering module
- 54% coverage for mondrian module (improving)
- Comprehensive input validation tests
- Edge case coverage for all public APIs

## 📈 Performance

PolarFrost is optimized for performance across different workloads:

### Performance Features
- **Lazy Evaluation**: Leverages Polars' lazy evaluation for optimal query planning
- **Minimal Data Copying**: Efficient memory management with minimal data duplication
- **Parallel Processing**: Utilizes multiple cores for faster computation
- **Distributed Processing**: Scales to large datasets with PySpark backend
- **Smart Partitioning**: Efficient data partitioning for balanced workloads

### Performance Tips
1. **Use LazyFrames** for multi-step operations to enable query optimization
   ```python
   # Good: Uses lazy evaluation
   df.lazy()\
     .filter(pl.col('age') > 30)\
     .collect()
   ```

2. **Specify Categorical Columns** for better performance with string data
   ```python
   mondrian_k_anonymity(df, ..., categorical=['gender', 'zipcode'])
   ```

3. **Batch Processing** for large datasets
   - Process data in chunks when possible
   - Use PySpark for distributed processing of very large datasets

4. **Monitor Performance**
   - Use Polars' built-in profiling
   - Enable query plans with `df.explain()` (Polars) or `df.explain(True)` (PySpark)

## 🔄 Dependency Management

This project uses [Dependabot](https://docs.github.com/en/code-security/dependabot) to keep dependencies up to date. Dependabot will automatically create pull requests for dependency updates.

### Update Schedule
- **Python Dependencies**: Checked weekly (Mondays at 9:00 AM AEST)
- **GitHub Actions**: Checked monthly

### Configuration
Dependabot is configured via [.github/dependabot.yml](.github/dependabot.yml). By default:
- Only patch and minor version updates are automatically created
- Major version updates are ignored by default
- Dependencies are grouped by name
- Pull requests are automatically labeled with `dependencies` and `automated`

To update the configuration, modify the [.github/dependabot.yml](.github/dependabot.yml) file.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛠 Development

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or pip
- [pre-commit](https://pre-commit.com/)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rglew/polarfrost.git
   cd polarfrost
   ```

2. **Install dependencies**
   ```bash
   # Using Poetry (recommended)
   poetry install

   # Or using pip
   pip install -e .[dev]
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

3. Run tests locally:
   ```bash
   pytest tests/ -v
   ```

4. Push your changes and create a pull request

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- We use `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

All these checks are automatically run via pre-commit hooks and CI.

### Testing

- Write tests for new features
- Run tests with `pytest`
- Ensure test coverage remains high
- Document any new features or changes

## 📄 Changelog

### 0.1.0 (2025-06-26)
- Initial release with Mondrian k-anonymity implementation
- Support for both Polars and PySpark backends
- Comprehensive test suite
