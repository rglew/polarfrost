# Polarfrost ❄️

[![PyPI](https://img.shields.io/pypi/v/polarfrost)](https://pypi.org/project/polarfrost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/polarfrost)](https://pypi.org/project/polarfrost/)

A high-performance k-anonymity implementation using Polars and PySpark, featuring the Mondrian algorithm for efficient privacy-preserving data analysis.

## ✨ Features

- 🚀 **Blazing Fast**: Leverages Polars for high-performance data processing
- 🔄 **Dual Backend**: Supports both local (Polars) and distributed (PySpark) processing
- 📊 **Data Utility**: Preserves data utility while ensuring privacy
- 🐍 **Pythonic API**: Simple and intuitive interface
- 🔒 **Privacy-Preserving**: Implements k-anonymity to protect sensitive information
- 📦 **Production Ready**: Well-tested and ready for production use

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

## 🚀 Quick Start

### Basic Usage with Polars

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
```

### Using PySpark for Distributed Processing

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

## 🔍 How It Works

PolarFrost implements the Mondrian algorithm for k-anonymity, which works by:

1. **Partitioning** the data based on quasi-identifiers
2. **Generalizing** values within each partition
3. **Ensuring** each group contains at least k records
4. **Preserving** the utility of the data while protecting privacy

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
