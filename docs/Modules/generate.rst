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

*  Splits the dataset into training and test sets using various strategies (KN, RND, STRATIFIED, EVEN, EXTRA_Q1, EXTRA_Q5). By default, the "EVEN" strategy is used, which creates bins across the full y-value range and selects central points. For unbalanced data, a stratified approach is applied.
*  Hyperoptimizes multiple ML models using a Bayesian Optimization approach, employing an objective function that combines the error from a 10-times 5-fold CV and a sorted CV to minimize overfitting.
*  Filters off descriptors with low permutation feature importance (PFI). New models are generated parallely, (1) No PFI and (2) PFI-filtered models.  
*  Creates a heatmap plot with different model types. One plot is generated for No PFI models and another for PFI-filtered models.  
*  Selects and stores the best No PFI and PFI models.  

Technical information
+++++++++++++++++++++

The GENERATE module performs an exploration of various ML algorithms. It uses built-in ML models from scikit-learn [1]. These models are hyperoptimized using Bayesian Optimization [2] to find their optimal parameters.

The software automatically generates a heatmap displaying RMSE values obtained from hyperoptimized algorithms. Furthermore, it performs permutation feature importance (PFI) analysis to identify the most influential descriptors and generate new models with only those descriptors. Users have the flexibility to fine-tune the PFI filter threshold using the "PFI_threshold" parameter. By default, this threshold removes features that contribute less than 4% to the model's R2. While this filter is activated by default, users can deactivate it by setting the "pfi_filter" option to False.

Users can choose between different modes for data splitting using the "split" option: (KN, RND, STRATIFIED, EVEN, EXTRA_Q1, EXTRA_Q5). The selection of ML algorithms during screening is tuned through the ``model`` parameter. By default, ROBERT screens RF, GB, NN, and MVL (regression) or RF, GB, NN, and AdaB (classification). Additional options include Gaussian Process (GP), Voting Regressor/Classifier (VR), and **XGB** (XGBoost regressor/classifier). XGB is opt-in (for example ``--model "[XGB]"`` or ``--model "[RF,XGB]"``); it is hyperoptimized with in-code Bayesian bounds rather than a packaged YAML file. **VR** combines RF, GB, and NN base estimators; Bayesian optimization tunes ensemble weights and member-model hyperparameters (prefixed ``rf_``, ``gb_``, and ``nn_`` bounds). The same ``model`` list is accepted by the :doc:`Python API <../API/robert.api>` via :class:`~robert.api.RobertModel`, where uncertainty kwargs are documented in detail.

20% of the data is used as a test set before hyperoptimization. This algorithm ensures an even distribution of data points across the range of y values, facilitating a balanced evaluation of predictions across the low, mid, and high y-value ranges.

This module is designed to handle both regression and classification problems, optimizing either the RMSE (regression) or MCC (classification) of the validation set during the hyperoptimization process. Users can adjust the error type for optimization with the "error_type" keyword, which offers the following options: RMSE, MAE, and R2 for regression tasks, and Matthew's correlation coefficient (MCC), F1 score, and accuracy for classification tasks.

* [1] `Scikit-learn: Machine Learning in Python <https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`__
* [2] `Bayesian Optimization: Open source constrained global optimization tool for Python <https://github.com/bayesian-optimization/BayesianOptimization>`

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. generate-modules-end
