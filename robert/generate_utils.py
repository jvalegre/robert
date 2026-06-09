#####################################################.
#     This file stores functions from GENERATE      #
#####################################################.

import os
import shutil
import pandas as pd
import numpy as np
import glob
import json
from robert.json_output_for_agent import audit_event, audit_set
from robert.utils import (
    load_params,
    PFI_filter,
    create_heatmap,
    BO_optimizer,
    BO_metrics,
    model_adjust_params
    )


# hyperopt workflow
def BO_workflow(self, Xy_data, csv_df, ML_model):
    '''
    Load hyperparameter space and perform a Bayesian optimization
    '''

    bo_data = {'model': ML_model.upper(),
                'type': self.args.type.lower(),
                'kfold': self.args.kfold,
                'repeat_kfolds': self.args.repeat_kfolds,
                'seed': self.args.seed,
                'error_type': self.args.error_type.lower(),
                'y': self.args.y,
                'names': self.args.names,
                'X_descriptors': Xy_data['X_descriptors']}

    bo_ran = ML_model.upper() != 'MVL'

    if ML_model.upper() != 'MVL':
        bo_data['params'], bo_data[f"combined_{bo_data['error_type']}"] = BO_optimizer(self,bo_data,Xy_data)
        bo_data['params'] = model_adjust_params(self, bo_data['model'], bo_data['params'])

    else:
        bo_data['params'] = {} # no need to format params
        bo_data = BO_metrics(self, bo_data, Xy_data)
        metric_combined = bo_data[f"combined_{bo_data['error_type']}"]
        self.args.log.write(f"   o Combined {bo_data['error_type'].upper()} for {bo_data['model']} (no BO needed) (no PFI filter): {metric_combined:.2}")

    # include the Set column to differentiate between train and test sets (and external test, if any)
    csv_df = set_sets(csv_df,Xy_data)

    # save csv files with model params and with Xy datapoints
    db_name = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model}_db")
    params_name = self.args.destination.joinpath(f"Raw_data/No_PFI/{ML_model.upper()}")
    _ = csv_df.to_csv(f'{db_name}.csv', index = None, header=True)
    
    # Convert params dict to string to avoid serialization issues
    bo_data_to_save = bo_data.copy()
    if 'params' in bo_data_to_save:
        bo_data_to_save['params'] = json.dumps(bo_data_to_save['params'])
    if 'X_descriptors' in bo_data_to_save:
        bo_data_to_save['X_descriptors'] = json.dumps(bo_data_to_save['X_descriptors'])
    
    # Save class label mapping if it exists (for classification with string labels)
    if hasattr(self.args, 'class_0_label'):
        bo_data_to_save['class_0_label'] = self.args.class_0_label
        bo_data_to_save['class_1_label'] = self.args.class_1_label

    # Save split type
    bo_data_to_save['split'] = self.args.split
 
    bo_data_df = pd.DataFrame([bo_data_to_save])
    _ = bo_data_df.to_csv(f'{params_name}.csv', index = None, header=True)

    if hasattr(self.args, 'generate_audit'):
        self.args.generate_audit = audit_event(
            self.args.generate_audit,
            event_type="bo_workflow",
            payload={
                "model": bo_data['model'],
                "type": bo_data['type'],
                "error_type": bo_data['error_type'],
                "kfold": bo_data['kfold'],
                "repeat_kfolds": bo_data['repeat_kfolds'],
                "seed": bo_data['seed'],
                "y": bo_data['y'],
                "names": bo_data['names'],
                "descriptor_count": len(bo_data['X_descriptors']),
                "descriptor_list": bo_data['X_descriptors'],
                "bo_was_run": bool(bo_ran),
                "bo_skipped_reason": None if bo_ran else "model_is_mvl",
                "optimized_parameters": bo_data.get('params', {}),
                "combined_metric_value": bo_data.get(f"combined_{bo_data['error_type']}", None),
                "no_pfi_parameter_csv_path": f'{params_name}.csv',
                "no_pfi_database_csv_path": f'{db_name}.csv',
                "class_label_mapping": {
                    "class_0_label": getattr(self.args, 'class_0_label', None),
                    "class_1_label": getattr(self.args, 'class_1_label', None),
                },
                "split_type": self.args.split,
            },
            evidence_level="direct",
        )

    return bo_data


def PFI_workflow(self, csv_df, ML_model, Xy_data):
    '''
    Filter off parameters with low PFI (not relevant in the model)
    '''

    # convert df to dict, then adjust params to a valid format
    name_csv_hyperopt = f"Raw_data/No_PFI/{ML_model}"
    path_csv = self.args.destination.joinpath(f'{name_csv_hyperopt}.csv')
    PFI_dict = load_params(self,path_csv)

    PFI_discard_cols,descp_cols_pfi = PFI_filter(self, Xy_data, PFI_dict)

    desc_keep = calc_desc_keep(self,Xy_data,PFI_discard_cols)

    discard_idx, descriptors_PFI = [],[]
    fallback_reason = None
    # if no descriptors pass the filter, just choose them based on importance until having the number of descps from desc_keep
    if len(PFI_discard_cols) == len(descp_cols_pfi):
        fallback_reason = "all_descriptors_below_pfi_threshold_using_desc_keep_fallback"
        PFI_discard_cols = []

    for _,column in enumerate(descp_cols_pfi):
        if column not in PFI_discard_cols and len(descriptors_PFI) < desc_keep:
            descriptors_PFI.append(column)
        else:
            discard_idx.append(column)

    # only use the descriptors that passed the PFI filter
    Xy_data_PFI = Xy_data.copy()
    Xy_data_PFI['X_train_scaled'] = Xy_data['X_train_scaled'].drop(discard_idx, axis=1)

    PFI_dict['X_descriptors'] = descriptors_PFI

    # updates the model's error and descriptors used from the corresponding No_PFI CSV file 
    # (the other parameters remain the same)
    PFI_dict = BO_metrics(self, PFI_dict, Xy_data_PFI)
    metric_combined = PFI_dict[f"combined_{PFI_dict['error_type']}"]
    self.args.log.write(f"   o Combined {PFI_dict['error_type'].upper()} for {PFI_dict['model']} (with PFI filter): {metric_combined:.2}")

    # save CSV file
    _ = save_pfi_csv(self,csv_df,name_csv_hyperopt,PFI_dict,Xy_data_PFI,ML_model)

    if hasattr(self.args, 'generate_audit'):
        pfi_params_csv_path = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_PFI.csv")
        pfi_db_csv_path = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_PFI_db.csv")
        self.args.generate_audit = audit_event(
            self.args.generate_audit,
            event_type="pfi_workflow",
            payload={
                "pfi_activated": True,
                "model": str(ML_model),
                "pfi_threshold": self.args.pfi_threshold,
                "pfi_max": self.args.pfi_max,
                "pfi_epochs": self.args.pfi_epochs,
                "descriptor_count_before_pfi": len(descp_cols_pfi),
                "descriptors_discarded_by_pfi": PFI_discard_cols,
                "descriptor_count_discarded": len(PFI_discard_cols),
                "descriptors_retained_after_pfi": descriptors_PFI,
                "descriptor_count_after_pfi": len(descriptors_PFI),
                "desc_keep": int(desc_keep),
                "fallback_reason": fallback_reason,
                "combined_metric_after_pfi": PFI_dict.get(f"combined_{PFI_dict['error_type']}", None),
                "pfi_parameter_csv_path": str(pfi_params_csv_path),
                "pfi_database_csv_path": str(pfi_db_csv_path),
            },
            evidence_level="direct",
        )


def calc_desc_keep(self,Xy_data,PFI_discard_cols):
    '''
    Calculate number of descriptors to keep in the PFI model
    '''
    
    # generate new X datasets and store the descriptors used for the PFI-filtered model
    desc_keep = len(Xy_data['X_train_scaled'].columns)
    
    # if the filter does not remove any descriptors based on the PFI threshold, or the 
    # proportion of descriptors:total datapoints is higher than 1:3, then the filter takes
    # the minimum value of 1 and 2:
    #   1. 25% less descriptors than the No PFI original model
    #   2. Proportion of 1:3 of descriptors:total datapoints (training + validation)
    total_points = len(Xy_data['y_train'])
    n_descp_PFI = desc_keep-len(PFI_discard_cols)

    # determine how many points will be kept
    if desc_keep > 1:
        if self.args.pfi_max > 0:
            desc_keep = self.args.pfi_max
        elif n_descp_PFI > 0.2*total_points or n_descp_PFI >= (0.75*desc_keep) or n_descp_PFI == 0:
            option_one = round(0.75*len(Xy_data['X_train_scaled'].columns))
            option_two = round(0.2*total_points)
            option_three = round(len(Xy_data['X_train_scaled'].columns)-1) # for databases with two or three descriptors
            desc_keep = min(option_one,option_two,option_three)

    return desc_keep


def save_pfi_csv(self,csv_df,name_csv_hyperopt,PFI_dict,Xy_data_PFI,ML_model):
    '''
    Saves CSV files with PFI models and information    
    '''

    name_csv_hyperopt_PFI = name_csv_hyperopt.replace('No_PFI','PFI')
    path_csv_PFI = self.args.destination.joinpath(f'{name_csv_hyperopt_PFI}_PFI')
    
    # Save class label mapping if it exists (for classification with string labels)
    if hasattr(self.args, 'class_0_label'):
        PFI_dict['class_0_label'] = self.args.class_0_label
        PFI_dict['class_1_label'] = self.args.class_1_label
    
    csv_PFI_df = pd.DataFrame([PFI_dict])
    _ = csv_PFI_df.to_csv(f'{path_csv_PFI}.csv', index = None, header=True)

    # include the Set column to differentiate between train and test sets (and external test, if any)
    csv_df = set_sets(csv_df,Xy_data_PFI)

    # save the csv file
    if os.path.exists(self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_PFI.csv")):
        db_name = self.args.destination.joinpath(f"Raw_data/PFI/{ML_model}_PFI_db")
        _ = csv_df.to_csv(f'{db_name}.csv', index = None, header=True)


def set_sets(csv_df,Xy_data):
    """
    Set a new column for the sets, including test set (if any)
    """

    set_column = []
    n_points = len(csv_df[csv_df.columns[0]])
    for i in range(0,n_points):
        if i in Xy_data['test_points']:
            set_column.append('Test')
        else:
            set_column.append('Training')

    csv_df['Set'] = set_column

    return csv_df


def detect_best(folder):
    '''
    Check which combination led to the best results
    '''

    # detect files
    file_list = glob.glob(f'{folder}/*.csv')
    candidate_param_csv_files = []
    candidate_metric_values = {}
    errors = []
    for file in file_list:
        if '_db' not in file:
            results_model = pd.read_csv(f'{file}', encoding='utf-8')
            training_error = results_model[f"combined_{results_model['error_type'][0]}"][0]
            errors.append(training_error)
            candidate_param_csv_files.append(file)
            candidate_metric_values[file] = float(training_error)
        else:
            errors.append(np.nan)
    # detect best result and copy files to the Best_model folder
    if results_model['error_type'][0].lower() in ['mae','rmse']:
        min_idx = errors.index(np.nanmin(errors))
    else:
        min_idx = errors.index(np.nanmax(errors))
    best_name = file_list[min_idx]
    best_db = f'{os.path.dirname(file_list[min_idx])}/{os.path.basename(file_list[min_idx]).split(".csv")[0]}_db.csv'

    shutil.copyfile(f'{best_name}', f'{best_name}'.replace('Raw_data','Best_model'))
    shutil.copyfile(f'{best_db}', f'{best_db}'.replace('Raw_data','Best_model'))

    return {
        "folder_scanned": str(folder),
        "candidate_parameter_csv_files": candidate_param_csv_files,
        "candidate_metric_values": candidate_metric_values,
        "error_type": str(results_model['error_type'][0]),
        "selection_rule": "minimum" if results_model['error_type'][0].lower() in ['mae', 'rmse'] else "maximum",
        "best_parameter_csv_path": str(best_name),
        "best_database_csv_path": str(best_db),
        "copied_destination_parameter_path": str(best_name).replace('Raw_data', 'Best_model'),
        "copied_destination_database_path": str(best_db).replace('Raw_data', 'Best_model'),
        "best_metric_value": float(np.nanmin(errors)) if results_model['error_type'][0].lower() in ['mae', 'rmse'] else float(np.nanmax(errors)),
        "pfi_status": "PFI" if '/PFI' in str(folder).replace('\\', '/') else "No_PFI",
    }


def heatmap_workflow(self,folder_hm):
    """
    Create matrix of ML models, training sizes and errors/precision
    """

    path_raw = self.args.destination.joinpath(f"Raw_data")
    csv_data, model_list = {},[]
    for csv_file in glob.glob(path_raw.joinpath(f"{folder_hm}/*.csv").as_posix()):
        if '_db' not in csv_file:
            basename = os.path.basename(csv_file)
            csv_model = basename.replace('.','_').split('_')[0]
            if csv_model not in model_list:
                csv_value = pd.read_csv(csv_file, encoding='utf-8')
                csv_data[csv_model] = csv_value[f"combined_{self.args.error_type}"][0]

    # pass dictionary into a dataframe, and sort the models alphabetically
    csv_df = pd.DataFrame([csv_data])
    
    # sort columns in the same order as the optimization
    df_cols = []
    for model in self.args.model:
        df_cols.append(model.upper())
    csv_df = csv_df[df_cols]

    # plot heatmap
    if folder_hm == "No_PFI":
        suffix = 'No PFI'
    elif folder_hm == "PFI":
        suffix = 'PFI'
    _ = create_heatmap(self,csv_df,suffix,path_raw)

    title_fig = f'Heatmap ML models {suffix}'
    name_fig = '_'.join(title_fig.split())
    image_path = path_raw.joinpath(f'{name_fig}.png')

    return {
        "plot_type": "heatmap",
        "pfi_status": "No_PFI" if folder_hm == "No_PFI" else "PFI",
        "source_folder": str(path_raw.joinpath(folder_hm)),
        "model_scores_included": {str(k): float(v) for k, v in csv_data.items()},
        "heatmap_suffix": suffix,
        "image_path": str(image_path),
        "image_generated": bool(os.path.exists(image_path)),
        "fail_soft_reason": None if os.path.exists(image_path) else "heatmap_output_not_detected_after_creation",
        "short_interpretation": "model-screening heatmap comparing optimized models",
    }

