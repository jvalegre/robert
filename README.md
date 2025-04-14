![](Logos/Robert_logo.jpg)
#
## <p align="center"> ROBERT (Refiner and Optimizer of a Bunch of Existing Regression Tools)</p>


[![CircleCI](https://img.shields.io/circleci/build/github/jvalegre/robert?label=Circle%20CI&logo=circleci)](https://app.circleci.com/pipelines/github/jvalegre/robert)
[![Codecov](https://img.shields.io/codecov/c/github/jvalegre/robert?label=Codecov&logo=codecov)](https://codecov.io/gh/jvalegre/robert)
[![Downloads](https://img.shields.io/pepy/dt/robert?label=Downloads&logo=pypi)](https://www.pepy.tech/projects/robert)
[![Read the Docs](https://img.shields.io/readthedocs/robert?label=Read%20the%20Docs&logo=readthedocs)](https://robert.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/robert)](https://pypi.org/project/robert/)

## Documentation  
Full documentation with installation instructions, technical details and examples can be found in [Read the Docs](https://robert.readthedocs.io).  

Don't miss out the latest hands-on tutorials from our [YouTube channel](https://www.youtube.com/channel/UCHRqI8N61bYxWV9BjbUI4Xw)!

## Recommended installation
1. (Only once) Create new conda environment: `conda create -n robert python=3.10`  
2. Activate conda environment: `conda activate robert`  
3. Install ROBERT using pip: `pip install robert`
4. Install RDKit (only if you plan to use easyROB): `conda install conda-forge::rdkit`
5. Install libraries for the PDF report `conda install -y -c conda-forge glib gtk3 pango mscorefonts`
6. (Only for compatible devices) Install Intelex accelerator: `pip install scikit-learn-intelex==2025.2.0`  
* Inexperienced users should visit the *Users with no Python experience* section in [Read the Docs](https://robert.readthedocs.io).
## Update the program
1. Update to the latest version: `pip install robert --upgrade`

## Developers and help desk  
List of main developers and contact emails:  
  - [ ] [Juan V. Alegre-Requena](https://orcid.org/0000-0002-0769-7168). Contact: [jv.alegre@csic.es](mailto:jv.alegre@csic.es)  
  - [ ] [David Dalmau Ginesta](https://orcid.org/0000-0002-2506-6546). Contact: [ddalmau@unizar.es](mailto:ddalmau@unizar.es)
  - [ ] [Miguel Martinez Fernandez](https://orcid.org/0009-0002-8538-7250). Contact [miguel.martinez@csic.es](mailto:miguel.martinez@csic.es)
  - [ ] [Luis Giner Tendero](https://github.com/LlGinerT/). Contact [lginertendero@gmail.com](mailto:lginertendero@gmail.com)

For suggestions and improvements of the code (greatly appreciated!), please reach out through the issues and pull requests options of Github.  

## License
ROBERT is freely available under an [MIT](https://opensource.org/licenses/MIT) License  

## Special acknowledgements
J.V.A.R. - The acronym ROBERT is dedicated to **ROBERT Paton**, who was a mentor to me throughout my years at Colorado State University and who introduced me to the field of cheminformatics. Cheers mate!

D.D.G. - The style of the ROBERT_report.pdf file was created with the help of **Oliver Lee** (2023, Zysman-Colman group at University of St Andrews).

J.V.A.R. and D.D.G. - The improvements from v1.0 to v1.2 are largely the result of insightful discussions with **Matthew Sigman** and his students, **Jamie Cadge** and **Simone Gallarati** (2024, University of Utah).

We really THANK all the testers for their feedback and for participating in the reproducibility tests, including:

* **David Valiente** (2022-2023, Universidad Miguel Hernández)
* **Heidi Klem** (2023, Paton group at Colorado State University)
* **Iñigo Iribarren** (2023, Trujillo group at Trinity College Dublin)
* **Guilian Luchini** (2023, Paton group at Colorado State University)
* **Alex Platt** (2023, Paton group at Colorado State University)
* **Oliver Lee** (2023, Zysman-Colman group at University of St Andrews)
* **Xinchun Ran** (2023, Yang group at Vanderbilt University)

## How to cite ROBERT
If you use any of the ROBERT modules, please include this citation:  
* Dalmau, D.; Alegre Requena, J. V. ROBERT: Bridging the Gap between Machine Learning and Chemistry. *Wiley Interdiscip. Rev. Comput. Mol. Sci.* **2024**, *14*, e1733.

If you use the AQME module, please include this citation:  
* Alegre-Requena et al., AQME: Automated Quantum Mechanical Environments for Researchers and Educators. *Wiley Interdiscip. Rev. Comput. Mol. Sci.* **2023**, *13*, e1663.

Additionally, please include the corresponding reference for Scikit-learn, SHAP and BayesianOptimization:  
* Pedregosa et al., Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825-2830.  
* Lundberg et al., From local explanations to global understanding with explainable AI for trees. *Nat. Mach. Intell.* **2020**, *2*, 56–67.
* Fernando Nogueira, {Bayesian Optimization}: Open source constrained global optimization tool for {Python}, **2014**, https://github.com/bayesian-optimization/BayesianOptimization  