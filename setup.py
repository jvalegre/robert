from setuptools import setup, find_packages
version="1.0.0"
setup(
    name="robert",
    packages=find_packages(exclude=["tests"]),
    package_data={"robert": ["model_params/*"]},
    version=version,
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
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
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
        "pandas>=2.0",
        "progress",
        "numpy>=1.23,<1.24"
        "matplotlib>=3.7.1",
        "seaborn",
        "scipy",
        "scikit-learn",
        "hyperopt",
        "numba",
        "shap",
        "glib",
        "weasyprint",
        "scikit-learn-intelex",
        # requires also "conda install -c conda-forge gtk3" in Windows
        # requires also "conda install -c conda-forge shap"
    ],
    python_requires=">=3.0",
    include_package_data=True,
)
