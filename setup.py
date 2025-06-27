from setuptools import setup, find_packages
import pathlib

# Read the contents of README.md
this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="polarfrost",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "polars>=0.13.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "spark": ["pyspark>=3.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
    },
    author="Richard Glew",
    author_email="richard.glew@hotmail.com",
    description="A fast k-anonymity implementation using Polars and PySpark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rglew/polarfrost",
    keywords=["anonymization", "privacy", "polars", "k-anonymity", "data-privacy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_data={
        "polarfrost": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)
