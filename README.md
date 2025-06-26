# Polarfrost

A fast k-anonymity implementation using Polars, featuring both Mondrian and Clustering algorithms for efficient privacy-preserving data analysis.

## Features

- 🚀 Blazing fast k-anonymity using Polars
- 🧊 Supports both local (Polars) and distributed (PySpark) processing
- 📊 Preserves data utility while ensuring privacy
- 🐍 Simple Python API

## Installation

```bash
pip install polarfrost
```

## Quick Start

```python
import polars as pl
from polarfrost import mondrian_k_anonymity

# Load your data
df = pl.read_csv("your_data.csv")

# Apply k-anonymity
anonymized = mondrian_k_anonymity(
    df,
    quasi_identifiers=["age", "gender", "zipcode"],
    sensitive_column="income",
    k=3,
    categorical=["gender", "zipcode"]
)

print(anonymized)
```

## License

MIT
