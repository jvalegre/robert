.. _versions:

========
Versions
========

Version 1.3.1 [`url <https://github.com/jvalegre/robert/releases/tag/1.3.1>`__]
   -  Fixed a bug in one-hot encoding in the one-hot test
   -  Adding the possibility to disable the automatic standarization of descriptors (--std False)
   -  Changing CV_test (now it standardizes the descriptors in each fold)
   -  Fixing a bug with the sklearn-intelex accelerator
   -  Fixing a threading bug with matplotlib in SHAP
   -  Sorting the training points when using all the split methods to match GENERATE models with PREDICT/VERIFY

Version 1.3.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.3.0>`__]
   -  Fixing a bug in the KNN imputer (it was incorrectly placing values in the target variable)
   -  Adding a new way of splitting data (stratified) to ensure that the validation points are taken throughout the range of the target values
   -  Fixing bug to work with spaces in descriptor names
   -  Changing the way of selecting the best model (now using a combined error metric, not only the validation error)
   -  Fixing bug in GENERATE when plotting the models' heatmap in case the model had infinite values
   -  Auto_test is now done by default if the database has more than 100 datapoints
   -  90% training size disables for datasets with less than 100 datapoints and 80% for less than 50 datapoints
   -  Changing models paramaters to avoid overgitting in small datasets
   -  Fixing bug (ROBERT was not reading some CSV files correctly when saved as UTF-8)
   -  Fixed bug in the report module when the Target_values had spaces
   -  MVL is replaced with AdaB when ROBERT assigns automated classification problems
   -  Adding automatic checks to ensure compatible classification problems
   -  ROBERT score is printed in the section title in the report to save space
   -  Kmeans clustering is applied individually to the different target values in classification problems to allow for a more compensated training selection

Version 1.2.1 [`url <https://github.com/jvalegre/robert/releases/tag/1.2.1>`__]
   -  NN solver are now set to 'lbfgs' by default in the MLPRegressor to work with small datasets
   -  Thres_x is now set to 0.7 by default in the CURATE module
   -  Fixing bug in the PREDICT module when using EVALUATE module (it was not showing the linear model equation)
   -  Adding linear model equation in the REPORT module
   -  Changing the threshold for correlated features in predict_utils to adjust to the new thres_x
   -  Changing the way missing values are treated (previously filled with 0s, now using KNN imputer)
   -  Adding .csv in --csv_test in case the user forgets to add it
   -  Adding ROBERT score number in the REPORT module
   -  Creating --descp_lvl to select which descriptors to use in the AQME-ROBERT workflow (interpret/denovo/full)
   -  The AQME-ROBERT workflow now uses interpretable descriptors by default (--descp_lvl interpret)

Version 1.2.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.2.0>`__]
   -  Changing cross-validation (CV) in VERIFY to LOOCV for datasets with less than 50 points
   -  Changing MAPIE in PREDICT to LOOCV for datasets with less than 50 points
   -  By default, RFECV uses LOOCV for small datasets and 5-fold CV for larger datasets
   -  The external test set is chosen more evenly along the range of y values (not fully random)
   -  Changing the format of the VERIFY plot, from donut to bar plots
   -  Automatic KN data splitting for databases with less than 250 datapoints
   -  Change CV_test from ShuffleSplit to Kfold
   -  Predictions from CV are now represented in a graph and stored in a CSV
   -  Changing the ROBERT score to depend more heavily on results from CV
   -  Fixing auto_test (now it works as specified in the documentation)
   -  Adding clas predictions to report PDF
   -  Adding new pytests that cover the ROBERT score section from the report PDF
   -  Adding the EVALUATE module to evaluate linear models with user-defined descriptors and partitions
   -  Adding Pearson heatmap in PREDICT for the two models, with individual variable correlation analysis
   -  Adding y-distribution graphs and analysis of uniformity
   -  Major changes to the report PDF file to include sections rather than modules
   -  Improving explanation of the ROBERT score on Read The Docs
   -  Printing coefficients in MVL models inside PREDICT.dat
   -  Fixing bug in RFECV for classification problems, now it uses RandomForestClassifier()
   -  Automatic recognition of classification problems

Version 1.1.2 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.2>`__]
   -  Fixing conda-forge install and making pip install the preferred installation method in ReadtheDocs

Version 1.1.1 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.1>`__]
   -  Hotfix of 1.1.0 in the installation
   -  Add documentation of AQME with versions >=1.6.0, in which SMILES workflows are fully reproducible

Version 1.1.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.0>`__]
   -  Adding RFECV in CURATE to fix the maximum number of descriptors to 1/3 of datapoints
   -  Added the possibility to use more than 1 SMILES column in the AQME module
   -  Change the scoring criteria in the PFI workflow (from R2 to RMSE)
   -  Fixing models where R2 in validation is much better than in training (if the validation set is very small or unrepresentative, the model may appear to perform excellently simply by chance)
   -  Fixing PFI_plot bug (now takes all the features into account)
   -  Fixing a bad allocation memory issue in GENERATE
   -  Fixing bug in classification models when more than 2 classes of the target variable are present
   -  Fixing reproducibility when using a specific seed in GENERATE module
   -  Change CV_test from Kfold to ShuffleSplit and adding a random_state to ensure reproducibility
   -  Allows CSV inputs that use ; as separator
   -  Fixing CV_test bug in VERIFY (now it uses equal test size to the model tested)
   -  Adding variability in the prediction with MAPIE python library
   -  Adding sd in the predictions table when using external test set
   -  Fixing error_type bug for classification models
   -  MCC as default metric for classification models (better to check performance in unbalanced datasets)
   -  PFI workflow now uses the same metric as error_type

Version 1.0.5 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.5>`__]
   -  Fixing some overfitted models with train and validation R2 0.99-1
   -  Including the easyROB graphical user interface (GUI)

Version 1.0.4 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.4>`__]
   -  Fixing outlier bug for negative t-values
   -  csv_test is treated separately from the test set from GENERATE
   -  Table of score thresholds in ROBERT_report.pdf
   -  Showing predictions at the end of the PREDICT section of ROBERT_report.pdf
   -  Adding --csv_test to AQME workflows
   -  Adding the --crest option to AQME workflows
   -  Auto adjusting the convergence criteria and xTB accuracy of QDESCP based on number 
      of datapoints

Version 1.0.3 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.3>`__]
   -  Changing default split to RND
   -  Adding the scikit-learn-intelex accelerator (now it's compatible for scikit-learn 1.3)
   -  Changing the thres_test default value to 0.25 (before: 0.20)
   -  Automatic KN data splitting for databases with less than 100 datapoints
   -  Droping 90% and 80% training sizes for small databases (less than 50 and 30 datapoints)
   -  Better print for command lines (more reproducible commands)
   -  Adding more information in the --help option
   -  Introducing SCORE and REPRODUBILITY to ROBERT_report.pdf
   -  Added the auto_test option
   -  Fixed empty spaces in heatmaps from GENERATE
   -  Mantain the ordering of GENERATE heatmaps across No_PFI and PFI 
   -  Added pytest to full workflows with classification and tests
   -  Fixed " separators in command lines with options that had more than one word (i.e. 
      --qdescp_keywords)
   -  Fixed length of outlier names for long words

Version 1.0.2 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.2>`__]
   -  Adding the REPORT module
   -  Adding the ReadTheDocs documentation

Version 1.0.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.0>`__]
   -  First estable version of the program
