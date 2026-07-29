"""
Parameters
----------

    csv_name : str, default=''
        Name of the CSV file containing the database. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv'). 
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility'). 
    discard : list, default=[]
        List containing the columns of the input CSV file that will not be included as descriptors
        in the curated CSV file (i.e. ['name','SMILES']).
    ignore : list, default=[]
        List containing the columns of the input CSV file that will be ignored during the curation process
        (i.e. ['name','SMILES']). The descriptors will be included in the curated CSV file. The y value
        is automatically ignored.
    destination : str, default=None
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).  
    auto_type : bool, default=True
        If there are only two y values, the program automatically changes the type of problem to classification.
    model : list, default=['RF','GB','NN','MVL'] (regression) and default=['RF','GB','NN','AdaB'] (classification) 
        ML models available: 
        1. 'RF' (Random forest)
        2. 'MVL' (Multivariate lineal models)
        3. 'GB' (Gradient boosting)
        4. 'NN' (MLP neural network)
        5. 'GP' (Gaussian Process)
        6. 'AdaB' (AdaBoost)
    custom_params : str, default=None
        Define new parameters for the ML models used in the hyperoptimization workflow. The path
        to the folder containing all the yaml files should be specified (i.e. custom_params='YAML_FOLDER')
    type : str, default='reg'
        Type of the pedictions. Options: 
        1. 'reg' (Regressor)
        2. 'clas' (Classifier)
    seed : int, default=0
        Random seed used in the ML predictor models and other protocols.
    error_type : str, default: rmse (regression), mcc (classification)
        Target value used during the hyperopt optimization. Options:
        Regression:
        1. rmse (root-mean-square error)
        2. mae (mean absolute error)
        3. r2 (R-squared, not recommended since R2 might be good even with high errors in small datasets)
        Classification:
        1. mcc (Matthew's correlation coefficient)
        2. f1 (F1 score)
        3. acc (accuracy, fraction of correct predictions)
    init_points : int, default=10
        Number of initial points for Bayesian optimization (exploration)
    n_iter : int, default=10
        Number of iterations for Bayesian optimization (exploitation)
    expect_improv : int, default=0.05
        Expected improvement for Bayesian optimization
    pfi_filter : bool, default=True
        Activate the PFI filter of descriptors.
    pfi_epochs : int, default=5
        Sets the number of times a feature is randomly shuffled during the PFI analysis
        (standard from sklearn webpage: 5).
    pfi_threshold : float, default=0.2
        The PFI filter is X% of the model's score (% adjusted, 0.2 = 20% of the total score during PFI).
    pfi_max : int, default=0
        Number of features to keep after the PFI filter. If pfi_max is 0, all the features that pass the PFI
        filter are used.
    auto_test : bool, default=True
        Raises % of test points to 20% if test_set is lower than that.
    test_set : float, default=0.2
        Amount of datapoints to separate as external test set (0.2 = 20%). These points will not be used during the
        hyperoptimization, and PREDICT will use the points as test set during ROBERT workflows. Select
        --test_set 0 to use only training and validation.
    kfold : int, default=5
        Number of random data splits for the cross-validation of the models. 
    repeat_kfolds : int, default=10
        Number of repetitions for the k-fold cross-validation of the models.
    split : str, default= 'even' (regression) or 'rnd' (classification)
        Specifies how the data is split into training and test sets. Options:
        1. 'even': splits the data evenly into training and test sets.
        2. 'RND': randomly splits the data.
        3. 'stratified': splits the data while preserving the distribution of the target variable.
        4. 'KN': uses a k-means approach to select representative samples for training (good for intrapolation, bad for extrapolation).
        5. 'extra_q1': selects the 20% lowest values.
        6. 'extra_q5': selects the 20% highest values.
        
"""
#####################################################.
#        This file stores the GENERATE class        #
#             used in model generation              #
#####################################################.

import os
import time
import glob
from robert.json_output_for_agent import (
    init_module_audit,
    audit_set,
    audit_event,
    finalize_module_audit,
    agent_json_path,
    write_json_output_audit,
)
from robert.utils import (
    load_variables, 
    finish_print,
    load_database,
    check_clas_problem,
    prepare_sets
)
from robert.generate_utils import (
    BO_workflow,
    PFI_workflow,
    heatmap_workflow,
    detect_best
)


class generate:
    """
    Class containing all the functions from the GENERATE module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the GENERATE module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):

        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "generate")

        # Initialize GENERATE runtime audit capture (additive and fail-soft).
        self.args.generate_audit = init_module_audit(
            module="GENERATE",
            artifact_type="generate_audit",
            source_files=[str(self.args.csv_name)],
            command_line=None,
        )
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "source_csv", self.args.csv_name)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "target_column", self.args.y)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "names_column", self.args.names)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "ignored_columns", getattr(self.args, "ignore", []))
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "discarded_columns_requested", getattr(self.args, "discard", []))
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "model_list", list(self.args.model))
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "prediction_type", self.args.type)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "seed", self.args.seed)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "error_type", self.args.error_type)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "init_points", self.args.init_points)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "n_iter", self.args.n_iter)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "pfi_filter", self.args.pfi_filter)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "pfi_epochs", self.args.pfi_epochs)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "pfi_threshold", self.args.pfi_threshold)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "pfi_max", self.args.pfi_max)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "test_set", self.args.test_set)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "kfold", self.args.kfold)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "repeat_kfolds", self.args.repeat_kfolds)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "split", self.args.split)
        self.args.generate_audit = audit_set(self.args.generate_audit, "inputs", "destination", str(self.args.destination))

        generate_audit_path = agent_json_path("generate_audit.json")
        json_audit_path = agent_json_path("generate_json_output_audit.json")

        # load database, discard user-defined descriptors and perform data checks
        csv_df, _, _ = load_database(self,self.args.csv_name,"generate")
        # load_database() already records canonical load counts in generate_audit.
        self.args.generate_audit = audit_set(self.args.generate_audit, "load_database", "source_csv_used", self.args.csv_name)
        self.args.generate_audit = audit_set(self.args.generate_audit, "load_database", "source_csv_role", "initial_source_csv")

        # adjust classification labels and auto-detect binary classification
        if self.args.type.lower() == 'clas' or (self.args.type.lower() == 'reg' and self.args.auto_type):
            self = check_clas_problem(self,csv_df)
        
        # scan different ML models
        txt_heatmap = f"\no  Starting heatmap scan with {len(self.args.model)} ML models ({self.args.model})."

        # scan different training partition sizes
        cycle = 1
        txt_heatmap += f'\n   Heatmap generation:'
        self.args.log.write(txt_heatmap)

        # scan different ML models
        bo_target_txt = f'''   o Starting BO-based hyperoptimization using the combined target:
                    \n     1. 50% = {self.args.error_type.upper()} from a {self.args.repeat_kfolds}x repeated {self.args.kfold}-fold CV (interpoplation)
                    \n     2. 50% = {self.args.error_type.upper()} from the bottom or top (worst performing) fold in a sorted {self.args.kfold}-fold CV (extrapolation)
                \n'''
        self.args.log.write(bo_target_txt)
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "number_of_ml_models", len(self.args.model))
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "model_list", list(self.args.model))
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "error_type", self.args.error_type)
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "repeated_kfold_setting", self.args.repeat_kfolds)
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "kfold_setting", self.args.kfold)
        self.args.generate_audit = audit_set(self.args.generate_audit, "model_scan", "bo_target_explanation", bo_target_txt.strip())

        for ML_model in self.args.model:

            self.args.log.write(f'   - {cycle}/{len(self.args.model)} - ML model: {ML_model} ')

            # Try to load model-specific curated CSV first, fall back to general CSV
            # Get the base name from the original csv_name (remove path if any)
            if 'CURATE' in str(self.args.csv_name):
                # If csv_name is already a CURATE file, extract the original base name
                csv_basename = os.path.basename(f'{self.args.csv_name}').replace('_CURATE.csv', '').replace('.csv', '')
            else:
                csv_basename = os.path.basename(f'{self.args.csv_name}').split('.')[0]
            
            curate_folder = self.args.initial_dir.joinpath('CURATE')
            csv_model_specific = curate_folder.joinpath(f'{csv_basename}_CURATE_{ML_model}.csv')
            
            # Store the original csv_name temporarily
            original_csv_name = self.args.csv_name
            
            model_specific_found = os.path.exists(csv_model_specific)
            if model_specific_found:
                csv_to_load = csv_model_specific
                # Temporarily update csv_name to the model-specific CSV
                self.args.csv_name = str(csv_model_specific)
                self.args.log.write(f'      o Using model-specific curated database: {os.path.basename(csv_model_specific)}')
            else:
                csv_to_load = self.args.csv_name
                self.args.log.write(f'      x Using general database (model-specific not found): {os.path.basename(self.args.csv_name)}')

            self.args.generate_audit = audit_event(
                self.args.generate_audit,
                event_type="model_run_start",
                payload={
                    "cycle_number": int(cycle),
                    "model_name": str(ML_model),
                    "model_specific_curate_csv_found": bool(model_specific_found),
                    "model_specific_csv_path_checked": str(csv_model_specific),
                    "csv_path_used": str(csv_to_load),
                    "fell_back_to_general_csv": bool(not model_specific_found),
                },
                evidence_level="direct",
            )
            
            # load database, discard user-defined descriptors and perform data checks
            csv_df, csv_X, csv_y = load_database(self,csv_to_load,"generate",print_info=False)
            if self.args.type.lower() == 'clas':
                self = check_clas_problem(self,csv_df)
                csv_y = csv_df[self.args.y]

            descriptor_count_after_load = len([col for col in csv_df.columns if col not in self.args.ignore and col != self.args.y])
            self.args.generate_audit = audit_event(
                self.args.generate_audit,
                event_type="model_csv_loaded",
                payload={
                    "model_name": str(ML_model),
                    "descriptor_count_after_loading_model_csv": int(descriptor_count_after_load),
                    "source_csv_role": "model_specific_curate_csv" if model_specific_found else "general_csv_fallback",
                    "source_csv_used": str(csv_to_load),
                },
                evidence_level="direct",
            )


            # standardizes and separates an external test set
            Xy_data = prepare_sets(self,csv_df,csv_X,csv_y,None,self.args.names,None,None,None,BO_opt=True)
            train_count = len(Xy_data['y_train']) if 'y_train' in Xy_data else 0
            test_count = len(Xy_data['test_points']) if 'test_points' in Xy_data else 0
            total_count = train_count + test_count
            test_fraction = float(test_count / total_count) if total_count > 0 else 0.0
            self.args.generate_audit = audit_event(
                self.args.generate_audit,
                event_type="prepare_sets",
                payload={
                    "model_name": str(ML_model),
                    "train_count": int(train_count),
                    "test_count": int(test_count),
                    "test_fraction": test_fraction,
                    "descriptor_count": int(len(Xy_data.get('X_descriptors', []))),
                    "split_method": str(self.args.split),
                },
                evidence_level="derived",
            )

            # hyperopt process for ML models
            _ = BO_workflow(self, Xy_data, csv_df, ML_model)

            # apply the PFI descriptor filter if it's activated
            if self.args.pfi_filter:
                # load database, discard user-defined descriptors and perform data checks
                csv_df, csv_X, csv_y = load_database(self,csv_to_load,"generate",print_info=False)
                if self.args.type.lower() == 'clas':
                    self = check_clas_problem(self,csv_df)
                    csv_y = csv_df[self.args.y]

                # standardizes and separates an external test set
                Xy_data = prepare_sets(self,csv_df,csv_X,csv_y,None,self.args.names,None,None,None,BO_opt=True)

                _ = PFI_workflow(self, csv_df, ML_model, Xy_data)
            
            # Restore the original csv_name
            self.args.csv_name = original_csv_name

            cycle += 1

        # detects best combinations
        dir_csv = self.args.destination.joinpath(f"Raw_data")
        best_no_pfi = detect_best(f'{dir_csv}/No_PFI')
        if best_no_pfi is not None:
            self.args.generate_audit = audit_event(
                self.args.generate_audit,
                event_type="best_model_selection",
                payload=best_no_pfi,
                evidence_level="direct",
            )

        # create heatmap plot(s)
        hm_no_pfi = heatmap_workflow(self,"No_PFI")
        if hm_no_pfi is not None:
            self.args.generate_audit = audit_event(
                self.args.generate_audit,
                event_type="image_artifact",
                payload=hm_no_pfi,
                evidence_level="direct",
            )

        # detect best and create heatmap for PFI models
        if self.args.pfi_filter:
            try: # if no models were found
                best_pfi = detect_best(f'{dir_csv}/PFI')
                if best_pfi is not None:
                    self.args.generate_audit = audit_event(
                        self.args.generate_audit,
                        event_type="best_model_selection",
                        payload=best_pfi,
                        evidence_level="direct",
                    )
                hm_pfi = heatmap_workflow(self,"PFI")
                if hm_pfi is not None:
                    self.args.generate_audit = audit_event(
                        self.args.generate_audit,
                        event_type="image_artifact",
                        payload=hm_pfi,
                        evidence_level="direct",
                    )
            except UnboundLocalError:
                pass

        raw_no_pfi_paths = sorted(glob.glob(self.args.destination.joinpath("Raw_data/No_PFI/*.csv").as_posix()))
        raw_pfi_paths = sorted(glob.glob(self.args.destination.joinpath("Raw_data/PFI/*.csv").as_posix()))
        best_no_pfi_paths = sorted(glob.glob(self.args.destination.joinpath("Best_model/No_PFI/*.csv").as_posix()))
        best_pfi_paths = sorted(glob.glob(self.args.destination.joinpath("Best_model/PFI/*.csv").as_posix()))
        heatmap_paths = sorted(glob.glob(self.args.destination.joinpath("Raw_data/Heatmap_ML_models*.png").as_posix()))
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "raw_data_no_pfi_csv_paths", raw_no_pfi_paths)
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "raw_data_pfi_csv_paths", raw_pfi_paths)
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "best_model_no_pfi_paths", best_no_pfi_paths)
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "best_model_pfi_paths", best_pfi_paths)
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "heatmap_image_paths", heatmap_paths)
        self.args.generate_audit = audit_set(self.args.generate_audit, "outputs", "generated_audit_path", str(generate_audit_path))
        self.args.generate_audit = audit_set(self.args.generate_audit, "runtime", "module_runtime_seconds", round(time.time() - start_time, 2))

        try:
            audit_write_ok = finalize_module_audit(self.args.generate_audit, generate_audit_path, status="completed")
            if not audit_write_ok:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="GENERATE",
                    attempted_output_path=generate_audit_path,
                    attempted=True,
                    succeeded=False,
                    artifact="generate_audit.json",
                    error=RuntimeError("generate_audit_json_write_returned_false"),
                )
            else:
                _ = write_json_output_audit(
                    json_audit_path,
                    module="GENERATE",
                    attempted_output_path=generate_audit_path,
                    attempted=True,
                    succeeded=True,
                    artifact="generate_audit.json",
                    error=None,
                )
        except Exception as json_error:
            _ = write_json_output_audit(
                json_audit_path,
                module="GENERATE",
                attempted_output_path=generate_audit_path,
                attempted=True,
                succeeded=False,
                artifact="generate_audit.json",
                error=json_error,
            )

        _ = finish_print(self,start_time,'GENERATE')
