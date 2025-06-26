from setuptools import setup, find_packages

setup(
    name="frost",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "polars>=0.13.0",
        "pandas",
        "numpy"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A fast k-anonymity implementation using Polars",
    url="https://github.com/yourusername/frost",
    keywords=["anonymization", "privacy", "polars", "k-anonymity"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
