![](Logos/Robert_logo.jpg)
#
## <p align="center"> Robert (Refiner and Optimizer of a Bunch of Existing Regression Tools)</p>

The code contains two jupyter notebooks with automated ML protocols that start from databases in csv files and produce publication-quality results, including 1) R2, MAE, RMSE of training, validation and test sets, 2) SHAP analysis, 3) Cross validation and x/y shuffle tests, 4) and more! 
Since random seeds are used, the same results can be reproduced when using the same csv file and input options.
The code works for regression and classification models.

## Quickstart
Edit the first cell in the code section of the notebooks and run all the cells. If some python modules are missing, the user needs to install them (i.e. matplot, pandas, etc)

## Description of the notebooks
### Robert generator (GEN, compatible with reggression and classification models)
Jupyter notebook that cleans and processes the database and generates the ML model. Some of the most relevant features included are:
* Correlation filter of X variables. Discards variables that are highly correlated (user-defined R2 threshold, default = 0.85)
* Filter of noise. Discard variables that correlate very poorly with the y value (user defined R2 threshold, default = 0.02)
* Standardizes the data
* Splitting of the data intro training and validation sets. The options available are:
  1. k-neighbors-based splitting ('KN'). Runs a k-neighbors clustering protocol to select training points that are as different as possible (always including the max and min response values in the training set by default)
  2. Random splitting ('RND'). Splits the data randomly using the predefined random seed variable (random_init)
* Runs a hyperoptimizer to select optimal parameters for the ML models. The models available at this point are: 
  1. Random forests ('RF')
  2. Multivariate lineal models ('MVL')
  3. Gradient boosting ('GB')
  4. AdaBoost regressor ('AdaB')
  5. MLP regressor neural network ('NN')
  6. Voting regressor combining RF, GB and NN ('VR')
* Calculates R2, MAE, RMSE and k-fold crossvalidation R2 for training and validation sets
* Calculates the permutation feature importance (PFI) of the descriptors in the model, generating a PNG image with the PFI graph
* Filter descriptors that are not important based on PFI (user-threshold, default = 4% of the model's score)
* Calculates R2, MAE, RMSE and k-fold crossvalidation R2 for training and validation sets using the PFI-filtered variables
* Saves the database as a CSV called "FILENAME_final_dataset.csv" and a PNG image with the predicted vs measured values

### Robert analyzer (ANA)
* Loads the model previously generated in Robert_GEN. Optionally, an external test set might be provided
* Runs a SHAP analysis, generating a PNG image with the SHAP results
* If an external test set was provided, calculates R2, MAE, RMSE and k-fold crossvalidation R2 for the external test set
* If an external test set was provided, Saves the database as a CSV called "FILENAME_final_dataset_with_test.csv" and a PNG image with the predicted vs measured values

## Working example
In the example provided (Robert_example.csv, with variables as x1, x2, x3..., file located in the Examples folder), the main options in the input cell of the Robert_GEN notebook were:
```
csv_name = 'Robert_example'
response_value = 'Target_values'
fixed_descriptors = ['Name']
training_size = 80
model_type = 'RF'
prediction_type = 'reg'
```

All the options that weren't mentioned kept their default values. To run the Robert_GEN notebook and generate results:

1. Move Robert_example.csv to the same folder containing the notebooks.

2. Open Robert_GEN, modify the input cell (first cell after the imports) and run the complete notebook. After completing this step, you get:

  a. Predicted vs database values.png . File containing the final representation of predicted vs target values.

  <p align="center"><img src="Examples/Results/Predicted%20vs%20database%20values.png" width="50%" height="50%"></p>
  
  b. RF permutation feature importances (PFI).png . File containing the PFI analysis with the most significant variables and their importance.
  
  <p align="center"><img src="Examples/Results/RF%20permutation%20feature%20importances%20(PFI).png" width="50%" height="50%"></p>

  c. Robert_results.txt . File containing R2, MAE and RMSE of training and validation sets, as well as R2 of k-fold crossvalidation.
  ```
  Model: RF
  k-neighbours-based training, validation and test sets have been created with this distribution:
  Training points: 29
  Validation points: 8

  k-neighbours-based training: R2 = 0.98; MAE = 0.06; RMSE = 0.09
  5-fold cross validation: 0.75 ± 0.15
  k-neighbours-based validation: R2 = 0.96; MAE = 0.17; RMSE = 0.21
  ```

  d. Robert_results_x-shuffle.txt and Robert_results_y-shuffle.txt. File containing the same results when shuffling x or y values (to test prediction ability).
  ```
  -- x-shuffle --
  Model: RF
  k-neighbours-based training, validation and test sets have been created with this distribution:
  Training points: 29
  Validation points: 8

  k-neighbours-based training: R2 = 0.85; MAE = 0.17; RMSE = 0.27
  5-fold cross validation: -1.18 ± 1.88
  k-neighbours-based validation: R2 = 0.4; MAE = 0.34; RMSE = 0.51
  ```
  
  ```
  -- y-shuffle --
  Model: RF
  k-neighbours-based training, validation and test sets have been created with this distribution:
  Training points: 29
  Validation points: 8

  k-neighbours-based training: R2 = 0.8; MAE = 0.28; RMSE = 0.38
  5-fold cross validation: -1.69 ± 0.94
  k-neighbours-based validation: R2 = 0.58; MAE = 1.19; RMSE = 1.27
  ```
  
  e. Robert_example_final_dataset.csv . Contains the final database after PFI-filtering, including a column showing how the sets were created
  
  f. Predictor_parameters.csv . Contains details about the optimized ML model

Then, run the Robert_ANA notebook:

3. Open Robert_ANA, modify the input cell (first cell after the imports) as follows:
```
csv_training = 'Robert_example_final_dataset'
csv_test = 'Robert_example_test'
response_value = 'Target_values'
fixed_descriptors = ['Name']
prediction_type = 'reg'
```

4. Run all the cells to get:

  a. RF SHAP importances.png . File containing the SHAP feature importance analysis.
  
  <p align="center"><img src="Examples/Results/RF%20SHAP%20importances.png" width="60%" height="60%"></p>
  
  b. Robert_results_test_set.txt . Results including the external test set.
  Model: RF
  ```
  k-neighbours-based training, validation and test sets have been created with this distribution:
  Training points: 29
  Test points: 9

  k-neighbours-based training: R2 = 0.98; MAE = 0.06; RMSE = 0.09
  k-neighbours-based test: R2 = 0.98; MAE = 0.08; RMSE = 0.12
  ```
  
  c. Predicted vs database values with test.png . File containing the final representation of predicted vs target values including the external test set.
  
  <p align="center"><img src="Examples/Results/Predicted%20vs%20database%20values%20with%20test.png" width="50%" height="50%"></p>
  
  d. Robert_example_final_dataset_with_test.csv . Contains the final database with results including the external set
  
## Reference
Robert v1.0, Alegre-Requena, J. V. 2022. https://github.com/jvalegre/robert
