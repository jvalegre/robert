.. robert-banner-start

.. |robert_banner| image:: ../Logos/Robert_logo.jpg

|robert_banner|

.. robert-banner-end

.. badges-start

.. |CircleCI| image:: https://img.shields.io/circleci/build/github/jvalegre/robert?label=Circle%20CI&logo=circleci
   :target: https://app.circleci.com/pipelines/github/jvalegre/robert

.. |Codecov| image:: https://img.shields.io/codecov/c/github/jvalegre/robert?label=Codecov&logo=codecov
   :target: https://anaconda.org/conda-forge/robert

.. |Downloads| image:: https://img.shields.io/conda/dn/conda-forge/robert?label=Downloads&logo=Anaconda
   :target: https://anaconda.org/conda-forge/robert

.. |ReadtheDocs| image:: https://img.shields.io/readthedocs/robert?label=Read%20the%20Docs&logo=readthedocs
   :target: https://robert.readthedocs.io
   :alt: Documentation Status

|CircleCI|
|Codecov|
|Downloads|
|ReadtheDocs|

.. badges-end

.. checkboxes-start

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. *  raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">

.. checkboxes-end

======================================================================
ROBERT (Refiner and Optimizer of a Bunch of Existing Regression Tools)
======================================================================

.. contents::
   :local:

Introduction
------------

.. introduction-start

The code is an ensemble of automated machine learning protocols that can be run sequentially
through a single command line. The complete workflow has been designed to meet
state-of-the-art requirements from cheminformatics studies, including:

   *  RDKit-based conformer generation from SMILES databases in CSV files, 
      followed by the generation of 200+ molecular and atomic descriptors using RDKit, 
      xTB and DBSTEP. Requires the `AQME program <https://aqme.readthedocs.io>`__.  
   *  Data curation, including filters for correlated descriptors, noise, and duplicates, 
      as well as conversion of categorical descriptors.  
   *  Model selection, including comparison of multiple hyperoptimized models from 
      scikit-learn and training sizes.  
   *  Prediction of external test sets, as well as SHAP and PFI feature analysis.  
   *  Statistical tests to asses the predictive ability of the models, including y-shuffle
      and y-mean tests, k-fold cross-validation, and predictions with one-hot features.  

Don't miss out the latest hands-on tutorials from our 
`YouTube channel <https://www.youtube.com/channel/UCHRqI8N61bYxWV9BjbUI4Xw>`_.  

.. introduction-end

.. installation-start

Installation
++++++++++++

In a nutshell, ROBERT and its dependencies are installed as follows:

1. Install ROBERT using conda-forge and the intelex accelerator (only if your system is compatible with intelex) (preferred):  

.. code-block:: shell 
   
   conda install -c conda-forge robert
   pip install scikit-learn-intelex

2. In some cases, conda-forge might be slow and users might install ROBERT using pip instead. Then, install the libraries required for report.py and the intelex accelerator (only if your system is compatible with intelex):  

.. code-block:: shell

   pip install robert
   pip install weasyprint
   conda install -c conda-forge glib gtk3 pango
   pip install scikit-learn-intelex

.. installation-end 

.. update-start 

Update to the latest version
++++++++++++++++++++++++++++

1. Update to the latest version with pip (preferred):  

.. code-block:: shell

   pip install robert --upgrade

2. Download the code from GitHub, go to the main robert folder in your terminal (contains the setup.py file), and:  

.. code-block:: shell

   pip install .

.. update-end 

.. note-start 

Quick note for users with no Python experience
++++++++++++++++++++++++++++++++++++++++++++++

You need a terminal with Python to install and run ROBERT. These are some suggested first steps:  

**For Windows users:**

1. Install `Anaconda with Python 3 <https://docs.anaconda.com/free/anaconda/install/windows/>`__.  

2. Open an Anaconda prompt.

3. Install ROBERT as defined above (:code:`conda install -c conda-forge robert`).

4. Go to the folder with your CSV database (using the "cd" command, i.e. :code:`cd C:/Users/test_robert`).

5. Run ROBERT as explained in the Examples section.

**For macOS and Linux users:**

1. Open a terminal with Python.

2. Install ROBERT as defined above (:code:`conda install -c conda-forge robert`).

3. Go to the folder with your CSV database (using the "cd" command, i.e. :code:`cd C:/Users/test_robert`).

4. Run ROBERT as explained in the Examples section.

.. note-end 

.. requirements-start

Requirements
------------

Python and Python libraries
+++++++++++++++++++++++++++

*These libraries are installed during the initial conda-forge installation.*  

*  Python >= 3.6
*  matplotlib-base >=3.7.1
*  pandas >=2.0
*  numpy >=1.23,<1.24
*  progress
*  pyyaml
*  seaborn
*  scipy
*  scikit-learn >=1.2,<1.3
*  hyperopt
*  numba
*  shap
*  glib
*  weasyprint
*  gtk3
*  pango

.. requirements-end

.. tests-start

Running the tests
-----------------

Requires the pytest library. 

.. code-block:: shell

   cd path/to/robert/source/code
   pytest -v

.. tests-end

Developement
------------

.. developers-start 

List of main developers and contact emails:  

*  Juan V. Alegre-Requena [
   `ORCID <https://orcid.org/0000-0002-0769-7168>`__ , 
   `Github <https://github.com/jvalegre>`__ , 
   `email <jv.alegre@csic.es>`__ ]
*  David Dalmau Ginesta [
   `ORCID <https://orcid.org/0000-0002-2506-6546>`__ , 
   `Github <https://github.com/ddgunizar>`__ , 
   `email <ddalmau@unizar.es>`__]

For suggestions and improvements of the code (greatly appreciated!), please 
reach out through the issues and pull requests options of `Github <https://github.com/jvalegre/robert>`__.

.. developers-end

License
-------

.. license-start 

ROBERT is freely available under an `MIT License <https://opensource.org/licenses/MIT>`__  

.. license-end

Reference
---------

.. reference-start

If you use any of the ROBERT modules, please include this citation:  

* `ROBERT v1.0, Alegre-Requena, J. V.; Dalmau, D. 2023. https://github.com/jvalegre/robert <https://github.com/jvalegre/robert>`__  
  
Additionally, please include the corresponding reference for Scikit-learn and SHAP:  

* Pedregosa et al., Scikit-learn: Machine Learning in Python, J. Mach. Learn. Res. 2011, 12, 2825-2830.  
* Lundberg et al., From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell. 2020, 2, 56–67.  

.. reference-end

Acknowledgment
--------------

.. acknowledgment-start

J.V.A.R. - The acronym ROBERT is dedicated to Prof. ROBERT Paton, who was a mentor to me throughout my years at Colorado State University and who introduced me to the field of cheminformatics. Cheers mate!

D.D.G. - The style of the ROBERT_report.pdf file was created with the help of Oliver Lee (University of St Andrews, 2023).

.. acknowledgment-end
