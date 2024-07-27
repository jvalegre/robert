.. _versions:

========
Versions
========
Version 1.1.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.0>`__]
   -  Adding RFECV in CURATE to fix the maximum number of descriptors to 1/3 of datapoints
   -  Added the possibility to use more than 1 SMILES column in the AQME module
   -  Change the scoring criteria in the PFI workflow (from R2 to RMSE)
   -  Fixing models where R2 in validation is much better than in training (if the validation set is very small or unrepresentative, the model may appear to perform excellently simply by chance)
   -  Fixing PFI_plot bug (now takes all the features into account)
   -  Fixing a bad allocation memory issue in GENERATE
   -  Fixing bug in classification models when more than 2 classes of the target variable are present
   -  Fixing reproducibility when using a specific seed in GENERATE module
   -  Change CV_test from Kflod to ShuffleSplit and adding a random_state to ensure reproducibility
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
