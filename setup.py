from setuptools import setup, find_packages
version="2.0.1"
setup(
    name="robert",
    packages=find_packages(exclude=["tests"]),
    package_data={"robert": ["model_params/*","report/*","icons/*"]},
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
        "Development Status :: 5 - Production/Stable",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",  # Specify which python versions you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "PyYAML==6.0.2",
        "matplotlib==3.10.0",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "progress==1.6",
        "seaborn==0.13.2",
        "scipy==1.15.0",
        "scikit-learn==1.6.0",
        "bayesian-optimization==3.0.0b1",
        "numba==0.60.0",
        "shap==0.46.0",
        "weasyprint==63.1"
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
