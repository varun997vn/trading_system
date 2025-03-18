from setuptools import setup, find_packages

setup(
    name="trading_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "pyyaml",
        "pytest",
        "alpaca-trade-api",
        "ib_insync",
        "scikit-learn",
        "statsmodels",
        "jupyter",
        "pyarrow",
        "requests",
        "tqdm"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python-based trading analysis engine",
    keywords="trading, finance, analysis, algorithmic trading",
    python_requires=">=3.8",
)