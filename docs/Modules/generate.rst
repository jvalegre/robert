.. generate-modules-start

Screening of ML models
----------------------

Overview of the GENERATE module
+++++++++++++++++++++++++++++++

.. |generate_fig| image:: images/GENERATE.jpg
   :width: 600

.. centered:: |generate_fig|

Input required
++++++++++++++

This module uses a curated CSV (preferrably coming from a CURATE job).

Automated protocols
+++++++++++++++++++

*  Hyperoptimizes multiple ML models.  
*  Filters off descriptors with low permutation feature importance (PFI). New models are generated parallely, (1) No PFI and (2) PFI-filtered models.  
*  Creates a heatmap plot with different model types and partition sizes. One plot is generated for No PFI models and another for PFI-filtered models.  
*  Selects and stores the best No PFI and PFI models.  

Technical information
+++++++++++++++++++++

The GENERATE module performs an exploration of various combinations ML algorithms and partition sizes. It uses built-in ML models from scikit-learn [1] or its code accelerator, scikit-learn-intelex. These models are hyperoptimized using the hyperopt Python module to find their optimal parameters. The determination of the number of epochs for hyperoptimization and random seeds of the models was achieved through exhaustive benchmarking.
The software automatically generates a heatmap displaying RMSE values obtained from hyperoptimized algorithms. Furthermore, it performs permutation feature importance (PFI) analysis to identify the most influential descriptors and generate new models with only those descriptors. Users have the flexibility to fine-tune the PFI filter threshold using the "PFI_threshold" parameter. By default, this threshold removes features that contribute less than 4% to the model's R2. While this filter is activated by default, users can deactivate it by setting the "pfi_filter" option to False.
The model screening involves testing various partition sizes, which can be changed using the "train" keyword. Moreover, users can choose between two different modes for data splitting using the "split" option: "KN" (k-neighbors clustering-based) and "RND" (random). The selection of ML algorithms during screening is tuned through the "model" parameter, offering a range of popular options such as Random Forests (RF), Multivariate Linear Models (MVL), Gradient Boosting (GB), Gaussian Process (GP), AdaBoost Regressor (AdaB), MLP Regressor Neural Network (NN), and Voting Regressor (VR). 
This module is designed to handle both regression and classification problems, optimizing either the RMSE (regression) or accuracy (classification) of the validation set during the hyperoptimization process. Users can adjust the error type for optimization with the "error_type" keyword, which offers the following options: RMSE, MAE, and R2 for regression tasks, and Matthew's correlation coefficient (MCC), F1 score, and accuracy for classification tasks.

* [1] `Scikit-learn: Machine Learning in Python <https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`__

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. generate-modules-end
