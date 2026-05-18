#####################################################.
#     This file stores functions from PREDICT       #
#####################################################.

import os
from pathlib import Path
import pandas as pd
import numpy as np
from robert.utils import (
    get_graph_style,
    pearson_map,
    graph_reg,
    graph_clas,
    get_error_labels,
    PARAMS_DIR_BEST_MODEL_MARK,
)


def iter_best_model_dirs(params_dir):
    """
    Yield (params_dir, suffix, suffix_title) for default Best_model layout or a
    single custom folder.
    """
    if PARAMS_DIR_BEST_MODEL_MARK in str(params_dir).replace("\\", "/"):
        params_dirs = [
            f"{params_dir}/No_PFI",
            f"{params_dir}/PFI",
        ]
        suffixes = ["(with no PFI filter)", "(with PFI filter)"]
        suffix_titles = ["No_PFI", "PFI"]
    else:
        params_dirs = [params_dir]
        suffixes = ["(custom)"]
        suffix_titles = ["custom"]
    return zip(params_dirs, suffixes, suffix_titles)


def _uq_columns_for_split(Xy_data, y_col, split):
    """Optional meta-UQ and auto-UQ columns for a train/test/external split."""
    prefix = f"y_pred_{split}"
    out = {}
    for suffix in ("uq_model", "uq_meta", "uq_total"):
        key = f"{prefix}_{suffix}"
        if key in Xy_data:
            out[f"{y_col}_pred_{suffix}"] = Xy_data[key]
    auto_key = f"{prefix}_uq_auto"
    if auto_key in Xy_data:
        out[f"{y_col}_pred_uq_auto"] = Xy_data[auto_key]
    return out


def _uq_auto_source_column(Xy_data):
    """Constant source label for the selected auto uncertainty candidate."""
    selected = Xy_data.get("uq_auto_selected")
    if not selected:
        return None
    return str(selected)


def _reconvert_values(values, reconvert_labels, class_mapping_reverse):
    if not reconvert_labels:
        return values
    return [class_mapping_reverse[int(y)] for y in values]


def _append_split_columns(
    df,
    Xy_data,
    model_data,
    split,
    *,
    reconvert_labels,
    class_mapping_reverse,
    hw_scalar,
    auto_src,
):
    """Add y, predictions, SD, UQ, and conformal columns for one split."""
    y_col = model_data["y"]
    y_key = f"y_{split}"
    pred_key = f"y_pred_{split}"
    sd_key = f"y_pred_{split}_sd"

    y_values = _reconvert_values(
        Xy_data[y_key].tolist(), reconvert_labels, class_mapping_reverse
    )
    pred_values = _reconvert_values(
        Xy_data[pred_key], reconvert_labels, class_mapping_reverse
    )

    df[y_col] = y_values
    df[f"{y_col}_pred"] = pred_values
    df[f"{y_col}_pred_sd"] = Xy_data[sd_key]
    for col_name, col_vals in _uq_columns_for_split(Xy_data, y_col, split).items():
        df[col_name] = col_vals
    df[f"{y_col}_pred_conformal_hw"] = [hw_scalar] * len(df)
    if auto_src is not None:
        df[f"{y_col}_pred_uq_auto_source"] = [auto_src] * len(df)
    return df


def plot_predictions(self, params_dict, Xy_data, path_n_suffix):
    """
    Plot graphs of predicted vs actual values for train, validation and test sets
    """

    set_types = [
        f"{params_dict['repeat_kfolds']}x {params_dict['kfold']}-fold CV",
        "test",
    ]

    graph_style = get_graph_style()

    self.args.log.write("\n   o  Saving graphs in:")

    if params_dict["type"].lower() == "reg":
        # Plot graph with all sets
        _ = graph_reg(self, Xy_data, params_dict, set_types, path_n_suffix, graph_style)
        # Plot CV average ± SD graph of validation or test set
        _ = graph_reg(
            self,
            Xy_data,
            params_dict,
            set_types,
            path_n_suffix,
            graph_style,
            sd_graph=True,
        )
        if (
            "y_external" in Xy_data
            and not Xy_data["y_external"].isnull().values.any()
            and len(Xy_data["y_external"]) > 0
        ):
            # Plot CV average ± SD graph of external set
            set_type = "external"
            _ = graph_reg(
                self,
                Xy_data,
                params_dict,
                set_type,
                path_n_suffix,
                graph_style,
                csv_test=True,
                sd_graph=True,
            )

    elif params_dict["type"].lower() == "clas":
        for set_type in set_types:
            _ = graph_clas(self, Xy_data, params_dict, set_type, path_n_suffix)
        if (
            "y_external" in Xy_data
            and not Xy_data["y_external"].isnull().values.any()
            and len(Xy_data["y_external"]) > 0
        ):
            set_type = "external"
            _ = graph_clas(
                self, Xy_data, params_dict, set_type, path_n_suffix, csv_test=True
            )

    return graph_style


def save_predictions(self, Xy_data, model_data, suffix_title):
    """
    Saves CSV files with the different sets and their predicted results
    """

    # Check if we need to reconvert class labels (for classification with string labels)
    reconvert_labels = False
    class_mapping_reverse = None
    if "class_0_label" in model_data and "class_1_label" in model_data:
        reconvert_labels = True
        class_mapping_reverse = {
            0: model_data["class_0_label"],
            1: model_data["class_1_label"],
        }

    # save CV and test results as a single df
    Xy_train, Xy_test = (
        pd.DataFrame(Xy_data["names_train"]),
        pd.DataFrame(Xy_data["names_test"]),
    )
    for col in Xy_data["X_train"]:
        Xy_train[col] = Xy_data["X_train"][col].tolist()
        Xy_test[col] = Xy_data["X_test"][col].tolist()

    # Store y values and predictions, reconverting if needed
    hw_scalar = float(Xy_data.get("conformal_half_width", float("nan")))
    if model_data["type"].lower() != "reg":
        hw_scalar = float("nan")
    auto_src = _uq_auto_source_column(Xy_data)

    Xy_train = _append_split_columns(
        Xy_train,
        Xy_data,
        model_data,
        "train",
        reconvert_labels=reconvert_labels,
        class_mapping_reverse=class_mapping_reverse,
        hw_scalar=hw_scalar,
        auto_src=auto_src,
    )
    Xy_test = _append_split_columns(
        Xy_test,
        Xy_data,
        model_data,
        "test",
        reconvert_labels=reconvert_labels,
        class_mapping_reverse=class_mapping_reverse,
        hw_scalar=hw_scalar,
        auto_src=auto_src,
    )

    df_results = pd.concat([Xy_train, Xy_test], axis=0)

    # add column with sets
    train_list = ["CV" for _ in Xy_data["y_train"]]
    test_list = ["Test" for _ in Xy_data["y_test"]]
    col_set = train_list + test_list
    df_results["Set"] = col_set

    # save results as CSV
    base_csv_name = f"PREDICT/{model_data['model']}_{suffix_title}"
    base_csv_path = f"{Path(os.getcwd()).joinpath(base_csv_name)}"
    path_n_suffix = f"{base_csv_path}"
    _ = df_results.to_csv(f"{base_csv_path}.csv", index=None, header=True)

    # also save results for performance of individual folds (useful for t-tests and Wilcoxon tests between the folds)
    error1, error2, error3 = get_error_labels(model_data["type"])

    # df_folds = pd.DataFrame()
    # df_folds['Fold'] = [f'{i+1}' for i in range(len(Xy_data['idx_valid']))]
    # df_folds['idx_valid'] = Xy_data['idx_valid']
    # df_folds[f'{error1}_valid'] = Xy_data[f'fold_{error1}_valid']
    # df_folds[f'{error2}_valid'] = Xy_data[f'fold_{error2}_valid']
    # df_folds[f'{error3}_valid'] = Xy_data[f'fold_{error3}_valid']
    # df_folds[f'{error1}_test'] = Xy_data[f'fold_{error1}_test']
    # df_folds[f'{error2}_test'] = Xy_data[f'fold_{error2}_test']
    # df_folds[f'{error3}_test'] = Xy_data[f'fold_{error3}_test']

    # path_folds = f'{base_csv_path}_CV_folds'
    # _ = df_folds.to_csv(f'{path_folds}.csv', index = None, header=True)

    # prints
    print_preds = "   o  Saving CSV databases with predictions and their SD in:"
    print_preds += (
        f"\n      -  Predicted results of starting dataset: {base_csv_name}.csv"
    )

    if self.args.csv_test != "":
        # saves prediction for external test in --csv_test
        Xy_external = pd.DataFrame(Xy_data["names_external"])

        for col in Xy_data["X_external"]:
            Xy_external[col] = Xy_data["X_external"][col].tolist()

        if "y_external" in Xy_data:
            y_external_values = _reconvert_values(
                Xy_data["y_external"].tolist(),
                reconvert_labels,
                class_mapping_reverse,
            )
            Xy_external[model_data["y"]] = y_external_values

        pred_values = _reconvert_values(
            Xy_data["y_pred_external"], reconvert_labels, class_mapping_reverse
        )
        Xy_external[f"{model_data['y']}_pred"] = pred_values
        Xy_external[f"{model_data['y']}_pred_sd"] = Xy_data["y_pred_external_sd"]
        for col_name, col_vals in _uq_columns_for_split(
            Xy_data, model_data["y"], "external"
        ).items():
            Xy_external[col_name] = col_vals
        Xy_external[f"{model_data['y']}_pred_conformal_hw"] = [hw_scalar] * len(
            Xy_external
        )
        if auto_src is not None:
            Xy_external[f"{model_data['y']}_pred_uq_auto_source"] = [auto_src] * len(
                Xy_external
            )

        path_external = Path(os.getcwd()).joinpath("PREDICT/csv_test/")
        Path(path_external).mkdir(exist_ok=True, parents=True)
        csv_name_external = f"{os.path.basename(self.args.csv_test).split('.csv')[0]}_{model_data['model']}_{suffix_title}.csv"
        name_external = f"{path_external}/{csv_name_external}"

        _ = Xy_external.to_csv(name_external, index=None, header=True)
        print_preds += f"\n      -  External set with predicted results: PREDICT/csv_test/{csv_name_external}"

    self.args.log.write(print_preds)

    # store the names of the datapoints
    name_points = {}
    if model_data["names"] != "":
        if (
            model_data["names"].lower() in Xy_train
        ):  # accounts for upper/lowercase mismatches
            model_data["names"] = model_data["names"].lower()
        if model_data["names"].upper() in Xy_train:
            model_data["names"] = model_data["names"].upper()
        if model_data["names"] in Xy_train:
            name_points["train"] = df_results[model_data["names"]][
                df_results.Set == "CV"
            ]
            name_points["test"] = df_results[model_data["names"]][
                df_results.Set == "Test"
            ]

    return path_n_suffix, name_points, Xy_data


def _ensure_pred_range_stats(Xy_data):
    """Set y-range summary keys when diagnostic plots were skipped."""
    if "pred_min" in Xy_data:
        return
    pred_min = min(min(Xy_data["y_train"]), min(Xy_data["y_test"]))
    pred_max = max(max(Xy_data["y_train"]), max(Xy_data["y_test"]))
    Xy_data["pred_min"] = pred_min
    Xy_data["pred_max"] = pred_max
    Xy_data["pred_range"] = float(np.abs(pred_max - pred_min))


def print_predict(self, Xy_data, model_data, suffix_title):
    """
    Prints results of the predictions for all the sets
    """
    _ensure_pred_range_stats(Xy_data)

    print_results = f"\n   o  Summary of results {model_data['model']}_{suffix_title}:"

    # get number of points and proportions
    n_train = len(Xy_data["y_train"])
    n_test = len(Xy_data["y_test"])
    print_results += (
        "\n      -  Point counts: CV (train+valid.) = "
        f"{n_train}, held-out test = {n_test}"
    )

    total_points = n_train + n_test
    prop_train = round(n_train * 100 / total_points)
    prop_test = round(n_test * 100 / total_points)
    print_results += (
        f"\n      -  Proportion CV (train+valid.):test = {prop_train}:{prop_test}"
    )

    n_descps = len(Xy_data["X_train"].keys())
    print_results += f"\n      -  Number of descriptors = {n_descps}"
    print_results += (
        "\n      -  Proportion (train+valid.) points:descriptors = "
        f"{n_train}:{n_descps}"
    )

    # print results and save dat file
    CV_type = f"{model_data['repeat_kfolds']}x {model_data['kfold']}-fold CV"
    if model_data["type"].lower() == "reg":
        print_results += f"\n      -  {CV_type} : R2 = {Xy_data['r2_train']:.2}, MAE = {Xy_data['mae_train']:.2}, RMSE = {Xy_data['rmse_train']:.2}"
        print_results += f"\n      -  Test : R2 = {Xy_data['r2_test']:.2}, MAE = {Xy_data['mae_test']:.2}, RMSE = {Xy_data['rmse_test']:.2}"
        print_results += f"\n      -  Average SD in test set = {np.mean(Xy_data['y_pred_test_sd']):.2}"
        print_results += f"\n      -  y range of dataset (train+valid.) = {float(Xy_data['pred_min']):.2} to {float(Xy_data['pred_max']):.2}, total {float(Xy_data['pred_range']):.2}"
        if (
            "y_external" in Xy_data
            and not Xy_data["y_external"].isnull().values.any()
            and len(Xy_data["y_external"]) > 0
        ):
            print_results += f"\n      -  External test : R2 = {Xy_data['r2_external']:.2}, MAE = {Xy_data['mae_external']:.2}, RMSE = {Xy_data['rmse_external']:.2}"

    elif model_data["type"].lower() == "clas":
        print_results += f"\n      -  {CV_type} : Accur. = {Xy_data['acc_train']:.2}, F1 score = {Xy_data['f1_train']:.2}, MCC = {Xy_data['mcc_train']:.2}"
        if (
            "y_pred_test" in Xy_data
            and not Xy_data["y_test"].isnull().values.any()
            and len(Xy_data["y_test"]) > 0
        ):
            print_results += f"\n      -  Test : Accur. = {Xy_data['acc_test']:.2}, F1 score = {Xy_data['f1_test']:.2}, MCC = {Xy_data['mcc_test']:.2}"
        if (
            "y_external" in Xy_data
            and not Xy_data["y_external"].isnull().values.any()
            and len(Xy_data["y_external"]) > 0
        ):
            print_results += f"\n      -  External test : Accur. = {Xy_data['acc_external']:.2}, F1 score = {Xy_data['f1_external']:.2}, MCC = {Xy_data['mcc_external']:.2}"

    self.args.log.write(print_results)


def pearson_map_predict(self, Xy_data, params_dir):
    """
    Plots the Pearson map and analyzes correlation of descriptors.
    """

    X_combined = pd.concat(
        [Xy_data["X_train"], Xy_data["X_test"]], axis=0, ignore_index=True
    )
    corr_matrix = pearson_map(self, X_combined, "predict", params_dir=params_dir)

    corr_dict = {"descp_1": [], "descp_2": [], "r": []}
    for i, descp in enumerate(corr_matrix.columns):
        for j, val in enumerate(corr_matrix[descp]):
            if i < j and np.abs(val) > 0.8:
                corr_dict["descp_1"].append(corr_matrix.columns[i])
                corr_dict["descp_2"].append(corr_matrix.columns[j])
                corr_dict["r"].append(val)

    print_corr = "      Ideally, variables should show low correlations."  # no initial \n, it's a new log.write
    if len(corr_dict["descp_1"]) == 0:
        print_corr += "\n      o  Correlations between variables are acceptable"
    else:
        abs_r_list = list(np.abs(corr_dict["r"]))
        abs_max_r = max(abs_r_list)
        max_r = corr_dict["r"][abs_r_list.index(abs_max_r)]
        max_descp_1 = corr_dict["descp_1"][abs_r_list.index(abs_max_r)]
        max_descp_2 = corr_dict["descp_2"][abs_r_list.index(abs_max_r)]
        if abs_max_r > 0.84:
            print_corr += f"\n      x  WARNING! High correlations observed (up to r = {max_r:.2} or R2 = {max_r * max_r:.2}, for {max_descp_1} and {max_descp_2})"
        elif abs_max_r > 0.71:
            print_corr += f"\n      x  WARNING! Noticeable correlations observed (up to r = {max_r:.2} or R2 = {max_r * max_r:.2}, for {max_descp_1} and {max_descp_2})"

    self.args.log.write(print_corr)
