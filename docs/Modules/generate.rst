.. generate-modules-start

Screening of ML models
----------------------

Overview of the GENERATE module
+++++++++++++++++++++++++++++++

.. |generate_fig| image:: images/GENERATE.jpg
   :width: 600

.. centered:: |generate_fig|

BO hyperoptimization
+++++++++++++++++++++

.. |bo_fig| image:: images/BO.jpg
   :width: 600

.. centered:: |bo_fig|

Input required
++++++++++++++

This module uses a curated CSV (preferrably coming from a CURATE job).

Automated protocols
+++++++++++++++++++

*  Splits the dataset into training and test sets using various strategies (KN, RND, STRATIFIED, EVEN, EXTRA_Q1, EXTRA_Q5). By default (``split = auto``), the "EVEN" strategy is used for regression, which creates bins across the full y-value range and selects central points, and the "STRATIFIED" strategy is used for classification, which keeps the proportion of each class in the test set.
*  Hyperoptimizes multiple ML models using a Bayesian Optimization approach, employing an objective function that combines the error from a 10-times 5-fold CV and a sorted CV to minimize overfitting.
*  Filters off descriptors with low permutation feature importance (PFI). New models are generated parallely, (1) No PFI and (2) PFI-filtered models.  
*  Creates a heatmap plot with different model types. One plot is generated for No PFI models and another for PFI-filtered models.  
*  Selects and stores the best No PFI and PFI models.  

Technical information
+++++++++++++++++++++

The GENERATE module performs an exploration of various ML algorithms. It uses built-in ML models from scikit-learn [1]. These models are hyperoptimized using Bayesian Optimization [2] to find their optimal parameters.

The software automatically generates a heatmap displaying RMSE values obtained from hyperoptimized algorithms. Furthermore, it performs permutation feature importance (PFI) analysis to identify the most influential descriptors and generate new models with only those descriptors. Users have the flexibility to fine-tune the PFI filter threshold using the "pfi_threshold" parameter. By default (0.2), descriptors are removed when their PFI is lower than 20% of the model's score or 20% of the PFI of the most important descriptor (whichever is larger). While this filter is activated by default, users can deactivate it by setting the "pfi_filter" option to False.

Users can choose between different modes for data splitting using the "split" option: (KN, RND, STRATIFIED, EVEN, EXTRA_Q1, EXTRA_Q5). The selection of ML algorithms during screening is tuned through the "model" parameter, offering a range of popular options such as Random Forests (RF), Multivariate Linear Models (MVL), Gradient Boosting (GB), Gaussian Process (GP), AdaBoost (ADAB), MLP Neural Network (NN), ridge regression (RIDGE, regression only) and logistic regression (LOGISTIC, classification only). By default, RF, GB, NN and MVL are screened (LOGISTIC replaces MVL in classification).

20% of the data (with a minimum of 4 points) is used as a test set before hyperoptimization. With the default split, this ensures an even distribution of test points across the range of y values (regression) or across the classes (classification), facilitating a balanced evaluation of the predictions.

This module is designed to handle both regression and classification problems, optimizing either the RMSE (regression) or MCC (classification) during the hyperoptimization process. The objective function averages the error of the 10x 5-fold CV and the error of a sorted CV (in regression, the worse of the folds with the lowest and highest y values), so models that only work in the middle of the y range are penalized. Users can adjust the error type for optimization with the "error_type" keyword, which offers the following options: RMSE, MAE, and R2 for regression tasks, and Matthew's correlation coefficient (MCC), F1 score, and accuracy for classification tasks.

* [1] `Scikit-learn: Machine Learning in Python <https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`__
* [2] `Bayesian Optimization: Open source constrained global optimization tool for Python <https://github.com/bayesian-optimization/BayesianOptimization>`

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. generate-modules-end
