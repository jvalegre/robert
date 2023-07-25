from setuptools import setup, find_packages
version="1.0.2"
setup(
    name="robert",
    packages=find_packages(exclude=["tests"]),
    package_data={"robert": ["model_params/*","report/*"]},
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
        "pandas>=2.0.3,<2.1",
        "progress",
        "numpy>=1.24,<1.25",
        "matplotlib>=3.7,<3.8",
        "seaborn",
        "scipy",
        "scikit-learn==1.3.0", # fixed to make ROBERT reproducible when using the same version
        "hyperopt",
        "numba>=0.57,<0.58",
        "shap>=0.42,<0.43",
        # requires also "conda install -c conda-forge gtk3 glib weasyprint pango" for report.py
        # optionally, the intelex accelerator might be installed "pip install scikit-learn-intelex"
    ],
    python_requires=">=3.0",
    include_package_data=True,
)
