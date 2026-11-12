import runpy
from pathlib import Path

from setuptools import setup, find_packages

version_file = Path(__file__).resolve().parent / "robert" / "_version.py"
version = runpy.run_path(str(version_file))["ROBERT_VERSION"]

setup(
    name="robert",
    version=version,
    packages=find_packages(exclude=["tests"]),
    package_data={
        "robert": [
            "report/*",
            "icons/*",
        ],
        "robert.gui_easyrob": [
            "assets/*.gif",
            "web/*.html",
            "web/*.css",
            "web/*.js",
            "web/*.gif",
            "web/*.png",
            "tutorials/*.md",
            "tutorials/tutorial_images/*/*.png",
        ],
        "robert.gui_easyrob.bot": [
            "knowledge/*.json",
            "knowledge_sources/*.json",
        ],
    },
    license="MIT",
    description="Refiner and Optimizer of a Bunch of Existing Regression Tools",
    long_description="Documentation in Read The Docs: https://robert.readthedocs.io",
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
    download_url=f"https://github.com/jvalegre/robert/archive/refs/tags/{version}.tar.gz",
    classifiers=[
        "Development Status :: 5 - Production/Stable", # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers", # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12", # scipy==1.18.0 needs >=3.12, PySide6==6.9.2 needs <3.14
        "Programming Language :: Python :: 3.13",
    ],
    install_requires=[
        "PyYAML==6.0.3",
        "matplotlib==3.11.1",
        "pandas==2.3.3",
        "numpy==2.3.4",
        "progress==1.6.1",
        "seaborn==0.13.2",
        "scipy==1.18.0",
        "scikit-learn==1.9.0",
        "bayesian-optimization==3.3.0",
        "numba==0.62.1",
        "shap==0.52.0",
        "weasyprint==63.1",
        "PySide6==6.9.2",
        "keyring==25.7.0; sys_platform != 'win32'",
        "PyMuPDF==1.27.2.3",
        "pdfplumber==0.11.10",
        "openpyxl==3.1.5",
        "ansi2html==1.9.5",
        "rdkit==2025.9.1",
        "psutil==7.2.2",
        "requests==2.34.2",
    ],
    python_requires=">=3.12,<3.14",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "easyrob=robert.gui_easyrob.easyrob_launcher:main",
            "checkml=robert.gui_easyrob.checkml_launcher:main",
        ],
    },
)
