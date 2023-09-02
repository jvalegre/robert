.. _versions:

========
Versions
========

Version 1.0.4 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.4>`__]
   -  Fixing outlier bug for negative t-values
   -  csv_test is treated separately from the test set from GENERATE
   -  Table of score thresholds in ROBERT_report.pdf
   -  Showing predictions at the end of the PREDICT section of ROBERT_report.pdf

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
