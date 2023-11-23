![](Logos/Robert_logo.jpg)
#
## <p align="center"> ROBERT (Refiner and Optimizer of a Bunch of Existing Regression Tools)</p>


[![CircleCI](https://img.shields.io/circleci/build/github/jvalegre/robert?label=Circle%20CI&logo=circleci)](https://app.circleci.com/pipelines/github/jvalegre/robert)
[![Codecov](https://img.shields.io/codecov/c/github/jvalegre/robert?label=Codecov&logo=codecov)](https://codecov.io/gh/jvalegre/robert)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/robert?label=Downloads&logo=Anaconda)](https://anaconda.org/conda-forge/robert)
[![Read the Docs](https://img.shields.io/readthedocs/robert?label=Read%20the%20Docs&logo=readthedocs)](https://robert.readthedocs.io/)

ROBERT is an ensemble of automated machine learning protocols that can be run sequentially 
through a single command line. The program works for **regression and classification problems**.
Comprehensive workflows have been designed to meet state-of-the-art standards for cheminformatics 
studies.

## Documentation  
Full documentation with installation instructions, technical details and examples can be found in [Read the Docs](https://robert.readthedocs.io).  

Don't miss out the latest hands-on tutorials from our [YouTube channel](https://www.youtube.com/channel/UCHRqI8N61bYxWV9BjbUI4Xw)!  

## Recommended installation and update guide  
In a nutshell, ROBERT and its dependencies are installed/updated as follows:  
1. Install ROBERT using conda-forge and the intelex accelerator (only if your system is compatible with intelex) (preferred):  
`conda install -c conda-forge robert`  
`pip install scikit-learn-intelex`  
2. In some cases, conda-forge might be slow and users might install ROBERT using pip instead. Then, install the libraries required for report.py and the intelex accelerator (only if your system is compatible with intelex):  
`pip install robert`  
`pip install weasyprint`  
`conda install -c conda-forge glib gtk3 pango`  
`pip install scikit-learn-intelex`  
3. Update to the latest version:  
`pip install robert --upgrade`  

## Developers and help desk  
List of main developers and contact emails:  
  - [ ] [Juan V. Alegre-Requena](https://orcid.org/0000-0002-0769-7168). Contact: [jv.alegre@csic.es](mailto:jv.alegre@csic.es)  
  - [ ] [David Dalmau Ginesta](https://orcid.org/0000-0002-2506-6546). Contact: [ddalmau@unizar.es](mailto:ddalmau@unizar.es)  

For suggestions and improvements of the code (greatly appreciated!), please reach out through the issues and pull requests options of Github.  

## License
ROBERT is freely available under an [MIT](https://opensource.org/licenses/MIT) License  

## Special acknowledgements
J.V.A.R. - The acronym ROBERT is dedicated to Prof. ROBERT Paton, who was a mentor to me throughout my years at Colorado State University and who introduced me to the field of cheminformatics. Cheers mate!

D.D.G. - The style of the ROBERT_report.pdf file was created with the help of Oliver Lee (University of St Andrews, 2023).

## How to cite ROBERT
If you use any of the ROBERT modules, please include this citation:  
* Dalmau, D.; Alegre Requena, J. V. ROBERT: Bridging the Gap between Machine Learning and Chemistry; preprint; Chemistry, 2023. https://doi.org/10.26434/chemrxiv-2023-k994h.  

If you use the AQME module, please include this citation:  
* Alegre-Requena et al., AQME: Automated Quantum Mechanical Environments for Researchers and Educators. Wiley Interdiscip. Rev. Comput. Mol. Sci. 2023, 13, e1663.

Additionally, please include the corresponding reference for Scikit-learn and SHAP:  
* Pedregosa et al., Scikit-learn: Machine Learning in Python, J. Mach. Learn. Res. 2011, 12, 2825-2830.  
* Lundberg et al., From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell. 2020, 2, 56–67.  