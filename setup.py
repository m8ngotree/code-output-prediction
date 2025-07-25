"""
Setup configuration for code-output-prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="code-output-prediction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python project for generating synthetic datasets where LLMs predict code execution outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/code-output-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-output-prediction=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="ai, code generation, synthetic datasets, llm, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/code-output-prediction/issues",
        "Source": "https://github.com/yourusername/code-output-prediction",
        "Documentation": "https://github.com/yourusername/code-output-prediction#readme",
    },
)