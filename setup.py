from setuptools import setup, find_packages
import io

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="robert",
    packages=find_packages(exclude=["tests"]),
    version="0.0.1",
    license="MIT",
    description="Refiner and Optimizer of a Bunch of Existing Regression Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juan V. Alegre Requena, David Dalmau",
    author_email="jv.alegre@csic.es",
    keywords=[
        "workflows",
        "machine learning",
        "cheminformatics",
        "data curation",
        "prediction",
        "automated",
    ],
    url="https://github.com/jvalegre/robert",
    download_url="https://github.com/jvalegre/robert/archive/refs/tags/0.0.1.tar.gz",
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",  # Specify which python versions you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "PyYAML",
        "pandas",
        "progress",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "sklearn",
        "hyperopt",
        "shap",
    ],
    python_requires=">=3.0",
    include_package_data=True,
)
