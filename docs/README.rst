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

ROBERT is an ensemble of automated machine learning protocols that can be run sequentially 
through a single command line. The program works for **regression and classification problems**.
Comprehensive workflows have been designed to meet state-of-the-art standards for cheminformatics 
studies, including:

   *  **Atomic and molecular descriptor generation from SMILES**, including an RDKit conformer sampling and 
      the generation of 200+ steric, electronic and structural descriptors using RDKit, xTB and DBSTEP. 
      Requires the `AQME program <https://aqme.readthedocs.io>`__.  
   *  **Data curation**, including filters for correlated descriptors, noise, and duplicates, 
      as well as conversion of categorical descriptors.  
   *  **Model selection**, including comparison of multiple hyperoptimized models from 
      scikit-learn and training sizes.  
   *  **Prediction** of external test sets, as well as SHAP and PFI feature analysis.  
   *  **VERIFY tests** to asses the predictive ability of the models, including y-shuffle
      and y-mean tests, k-fold cross-validation, and predictions with one-hot features.  

The code has been designed for:

   *  **Inexperienced researchers** in the field of ML. ROBERT workflows are fully automated, and provide 
      users with comprehensive explanations of the resulting models and their prediction reliability. 
      Moreover, ready-to-use examples and tutorials can be accessed on ReadtheDocs and YouTube. 
   *  **ML experts** aiming to automate workflows, enhance reproducibility, or save time. Entire workflows 
      can be executed using a single command line while following modern standards of reproducibility and 
      transparency. Additionally, individual ROBERT modules can be integrated into customized ML workflows. 

Don't miss out the latest hands-on tutorials from our 
`YouTube channel <https://www.youtube.com/channel/UCHRqI8N61bYxWV9BjbUI4Xw>`_.  

.. introduction-end

.. installation-start

Installation
++++++++++++

In a nutshell, ROBERT and its dependencies are installed as follows:

**1.** Create and activate the conda environment where you want to install the program. If you are not sure of what 
this point means, check out the "Users with no Python experience" section. This is an example for Python 3.10, but 
it also works for other Python versions (i.e., 3.7, 3.9 and 3.11):

.. code-block:: shell 
   
   conda create -n robert python=3.10
   conda activate robert

**2.** Install ROBERT using conda-forge and the intelex accelerator with pip (only if your system is compatible with intelex) (preferred):  

.. code-block:: shell 
   
   conda install -c conda-forge robert
   pip install scikit-learn-intelex

**3.** Update ROBERT to the latest version (do not skip this step!):

.. code-block:: shell 
   
   pip install robert --upgrade

**4.** If conda-forge is too slow, users might install ROBERT using pip instead. Then, install the libraries required for report.py and the intelex accelerator (only if your system is compatible with intelex):  

.. code-block:: shell

   pip install robert
   pip install weasyprint
   conda install -c conda-forge glib gtk3 pango
   pip install scikit-learn-intelex

.. warning::

   In some HPCs, the Helvetica/Arial font used to create the report might not be installed. If the report PDF 
   looks messy, install the fonts with :code:`conda install -c conda-forge mscorefonts`.

.. installation-end 

.. note-start 

Users with no Python experience
+++++++++++++++++++++++++++++++

You need a terminal with Python to install and run ROBERT. These are some suggested first steps:  

.. |br| raw:: html

   <br />

**1.** Install `Anaconda with Python 3 <https://docs.anaconda.com/free/anaconda/install>`__ for your 
operating system (Windows, macOS or Linux). Alternatively, if you're familiar with conda installers, 
you can install `Miniconda with Python 3 <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`__ 
(requires less space than Anaconda).  


**2.** Open an Anaconda prompt (Windows users) or a terminal (macOS and Linux).


**3.** Create a conda environment called "robert" with Python (:code:`conda create -n robert python=3.10`). 
|br|
*You only need to do this once.*
|br|
*This is an example for Python 3.10, but it also works for other Python versions (i.e., 3.7, 3.9 and 3.11).*


**4.** Activate the conda environment called "robert" (:code:`conda activate robert`).


**5.** Install ROBERT as defined in the "Installation" section (:code:`conda install -c conda-forge robert`).


**6.** Install the intelex code accelerator with pip (only if your system is compatible with intelex) (:code:`pip install scikit-learn-intelex`).


**7.** Update ROBERT to the latest version (do not skip this step!) (:code:`pip install robert --upgrade`).


**8.** Go to the folder with your CSV database (using the "cd" command, i.e. :code:`cd C:/Users/test_robert`).


**9.** Run ROBERT as explained in the Examples section.

.. note-end 

.. gui-start 

Graphical User Interface (GUI): easyROB
+++++++++++++++++++++++++++++++++++++++

You need a terminal with Python to run easyROB, the GUI of ROBERT. This GUI simplifies the setup 
of ROBERT workflows, enabling users to select files and configure options easily. To run easyROB follow
these steps: 

**1.** Install ROBERT as defined in the "Installation" section.

.. warning::

   The GUI only works with ROBERT version 1.0.5 or later (check your version!). 

**2.** Open an Anaconda prompt (Windows users) or a terminal (macOS and Linux).


**3.** Activate the conda environment called "robert" (:code:`conda activate robert`).


.. |easyrob| image:: /Modules/images/Robert_icon.png
   :target: https://github.com/jvalegre/robert/tree/master/GUI_easyROB/easyrob.py
   :width: 50

.. |download| image:: /Modules/images/download.png
   :width: 200  

**4.** Download `easyrob.py: <https://github.com/jvalegre/robert/tree/master/GUI_easyROB/easyrob.py>`__ |easyrob|, tapping on this button on GitHub |download|


**5.** Go to the folder with the easyrob.py file (using the "cd" command, i.e. :code:`cd C:/Users/test_robert`).


**6.** Run easyROB with the following command line (:code:`python easyrob.py`).

.. |easyrob_interface| image:: /Modules/images/easyROB.png
   :width: 500
  
.. centered:: |easyrob_interface|

.. gui-end 

.. requirements-start

Requirements
------------

Python and Python libraries
+++++++++++++++++++++++++++

*These libraries are installed during the initial conda-forge installation.*  

*  Python >= 3.6
*  matplotlib-base
*  pandas
*  numpy
*  progress
*  pyyaml
*  seaborn
*  scipy
*  scikit-learn
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

* Dalmau, D.; Alegre Requena, J. V. ChemRxiv, 2023, DOI: 10.26434/chemrxiv-2023-k994h. 

If you use the AQME module, please include this citation:  

* Alegre-Requena et al., AQME: Automated Quantum Mechanical Environments for Researchers and Educators. Wiley Interdiscip. Rev. Comput. Mol. Sci. 2023, 13, e1663.

Additionally, please include the corresponding reference for Scikit-learn and SHAP:  

* Pedregosa et al., Scikit-learn: Machine Learning in Python, J. Mach. Learn. Res. 2011, 12, 2825-2830.  
* Lundberg et al., From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell. 2020, 2, 56–67.  

.. reference-end

Acknowledgment
--------------

.. acknowledgment-start

J.V.A.R. - The acronym ROBERT is dedicated to **ROBERT Paton**, who was a mentor to me throughout my years at Colorado State University and who introduced me to the field of cheminformatics. Cheers mate!

D.D.G. - The style of the ROBERT_report.pdf file was created with the help of **Oliver Lee** (2023, Zysman-Colman group at University of St Andrews).

We really THANK all the testers for their feedback and for participating in the reproducibility tests, including:

* **David Valiente** (2022-2023, Universidad Miguel Hernández)
* **Heidi Klem** (2023, Paton group at Colorado State University)
* **Iñigo Iribarren** (2023, Trujillo group at Trinity College Dublin)
* **Guilian Luchini** (2023, Paton group at Colorado State University)
* **Alex Platt** (2023, Paton group at Colorado State University)
* **Oliver Lee** (2023, Zysman-Colman group at University of St Andrews)
* **Xinchun Ran** (2023, Yang group at Vanderbilt University)

.. acknowledgment-end
