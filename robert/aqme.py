"""
Parameters
----------

    csv_name : str, default=''
        Name of the CSV file containing the database with SMILES and code_name columns. A path can be provided (i.e. 'C:/Users/FOLDER/FILE.csv').
    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).
    y : str, default=''
        Name of the column containing the response variable in the input CSV file (i.e. 'solubility').
    qdescp_keywords : str, default=''
        Add extra keywords to the AQME-QDESCP run (i.e. qdescp_keywords="--qdescp_atoms ['Ir']")
    descp_lvl : str, default='interpret'
        Type of descriptor to be used in the AQME-ROBERT workflow. Options are 'interpret', 'denovo' or 'full'.

"""
#####################################################.
#         This file stores the AQME class           #
#     used to perform the AQME-ROBERT workflow      #
#####################################################.

import os
import glob
import subprocess
import time
import shutil
import sys
import shlex
from pathlib import Path
import pandas as pd
from robert.utils import load_variables, finish_print, load_database

# list of potential arguments from CSV inputs in AQME
aqme_args = [
    "charge",
    "mult",
    "complex_type",
    "geom",
    "constraints_atoms",
    "constraints_dist",
    "constraints_angle",
    "constraints_dihedral",
    "sample",
]


def _expected_robert_csv(descp_lvl, aqme_indv_name):
    return f"AQME-ROBERT_{descp_lvl}_{aqme_indv_name}.csv"


def _append_aqme_job_logs(log, tail=4000):
    for log_path in (
        Path("QDESCP/QDESCP_data.dat"),
        Path("CSEARCH/CSEARCH_data.dat"),
    ):
        if log_path.is_file():
            log.write(
                f"\n--- tail {log_path} ---\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-tail:]
            )


class aqme:
    """
    Class containing all the functions from the AQME module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the AQME module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):
        start_time = time.time()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "aqme")

        # check if AQME is installed (required for this module)
        _ = self.init_aqme()

        # run an AQME workflow, which includes CSEARCH and QDESCP jobs
        self = self.run_csearch_qdescp(self.args.csv_name)

        # run an AQME workflow for the test set (if any)
        if self.args.csv_test != "":
            _ = self.run_csearch_qdescp(self.args.csv_test, aqme_test=True)

        # move AQME output files (remove previous runs as well)
        _ = move_aqme()

        # finish the printing of the AQME info file
        _ = finish_print(self, start_time, "AQME")

    def run_csearch_qdescp(self, csv_target, aqme_test=False):
        """
        Runs CSEARCH and QDESCP jobs in AQME
        """

        # load database just to perform data checks (i.e. no need to run AQME if the specified y is not
        # in the database, since the program would crush in the subsequent CURATE job)
        job_type = "aqme"
        path_sdf = Path(f"{os.getcwd()}/CSEARCH/sdf_temp")
        if aqme_test:
            job_type = "aqme_test"
            if not path_sdf.exists():
                path_sdf.mkdir(exist_ok=True, parents=True)

        # move the SDF files from the csv_name run (otherwise, the xTB calcs are repeated in csv_test)
        else:
            if path_sdf.exists():
                shutil.rmtree(path_sdf)
            path_sdf.mkdir(exist_ok=True, parents=True)
            for sdf_file in glob.glob(f"{os.getcwd()}/CSEARCH/*.sdf"):
                new_sdf = path_sdf.joinpath(os.path.basename(sdf_file))
                if os.path.exists(new_sdf):
                    os.remove(new_sdf)
                shutil.move(sdf_file, new_sdf)

        # Load database
        csv_df, _, _ = load_database(self, csv_target, job_type, print_info=False)

        # avoid running calcs with special signs (i.e. *)
        for name_csv_indiv in csv_df["code_name"]:
            if "*" in f"{name_csv_indiv}":
                self.args.log.write(
                    f"\nx  WARNING! The names provided in the CSV contain * (i.e. {name_csv_indiv}). Please, remove all the * characters."
                )
                self.args.log.finalize()
                sys.exit(1)

        # find if there is more than one SMILES column in the CSV file
        for column in csv_df.columns:
            if "SMILES" == column.upper() or "SMILES_" in column.upper():
                self.args.ignore.append(column)

                # create individual csv file for each SMILES column
                csv_temp = csv_df[
                    ["code_name", column]
                    + [col for col in csv_df.columns if col.lower() in aqme_args]
                ]
                csv_temp.columns = ["code_name", "SMILES"] + [
                    col for col in csv_temp.columns if col.lower() in aqme_args
                ]

                if column.upper() == "SMILES":
                    smi_suffix = None
                    csv_temp.to_csv("AQME_indiv.csv", index=False)
                    aqme_indv_name = "AQME_indiv"
                else:
                    smi_suffix = column.split("_")[1]
                    csv_temp["code_name"] = (
                        csv_temp["code_name"].astype(str) + "_" + smi_suffix
                    )
                    csv_temp.to_csv(f"AQME_indiv_{smi_suffix}.csv", index=False)
                    aqme_indv_name = f"AQME_indiv_{smi_suffix}"

                # run AQME-QDESCP to generate descriptors
                cmd_qdescp = [
                    sys.executable,
                    "-u",
                    "-m",
                    "aqme",
                    "--qdescp",
                    "--input",
                    f"{aqme_indv_name}.csv",
                    "--program",
                    "xtb",
                    "--csv_name",
                    f"{aqme_indv_name}.csv",
                    "--nprocs",
                    f"{self.args.nprocs}",
                    "--sample",
                    "3",
                    "--robert",
                ]
                expected_csv = _expected_robert_csv(self.args.descp_lvl, aqme_indv_name)
                aqme_success = self.run_aqme(
                    cmd_qdescp,
                    self.args.qdescp_keywords,
                    expected_csv=expected_csv,
                )
                if not aqme_success:
                    self._fail_aqme_job(
                        "x  ROBERT stopped because AQME did not create descriptor "
                        "CSV output. Please, check the previous AQME warnings."
                    )

                if smi_suffix is not None:
                    # Change column names by adding suffix
                    df_temp = pd.read_csv(expected_csv, encoding="utf-8")
                    df_temp.columns = [
                        f"{col}_{smi_suffix}"
                        if col not in ["code_name", "SMILES"] and col not in aqme_args
                        else col
                        for col in df_temp.columns
                    ]
                    df_temp.to_csv(
                        f"AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv",
                        index=False,
                    )

                    # Check if there are missing rows in the AQME-ROBERT_{aqme_indv_name}.csv
                    if len(df_temp) < len(csv_temp):
                        missing_rows = csv_temp.loc[
                            ~csv_temp["code_name"].isin(df_temp["code_name"])
                        ]
                        missing_rows[["code_name", "SMILES"]].to_csv(
                            f"AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv",
                            mode="a",
                            header=False,
                            index=False,
                        )

                    # Get the order of code_name in aqme_indv_name
                    order = csv_temp["code_name"].tolist()

                    # Sort the rows in 'AQME-ROBERT_{aqme_indv_name}.csv' based on the order
                    df_temp = df_temp.sort_values(
                        by="code_name",
                        key=lambda x: x.map({v: i for i, v in enumerate(order)}),
                    )

                    # Fill missing values with corresponding SMILES row
                    df_temp = df_temp.fillna(
                        df_temp.groupby("SMILES").transform("first")
                    )

                    df_temp.to_csv(
                        f"AQME-ROBERT_{self.args.descp_lvl}_{aqme_indv_name}.csv",
                        index=False,
                    )

                # return SDF files after csv_test
                if aqme_test:
                    for sdf_file in glob.glob(f"{path_sdf}/*.sdf"):
                        new_sdf = Path(f"{os.getcwd()}/CSEARCH").joinpath(
                            os.path.basename(sdf_file)
                        )
                        shutil.move(sdf_file, new_sdf)
                        shutil.rmtree(path_sdf)

        # if AQME-ROBERT_AQME_indiv_n.csv >0 in folder:
        if len(glob.glob(f"AQME-ROBERT_{self.args.descp_lvl}_AQME_indiv*.csv")) > 0:
            df_concat = pd.DataFrame()

            # Read and concatenate CSV files
            for file in sorted(
                glob.glob(f"AQME-ROBERT_{self.args.descp_lvl}_AQME_indiv*.csv"),
                key=os.path.getmtime,
                reverse=True,
            ):
                columns_to_drop = ["code_name", "SMILES"] + aqme_args
                df_temp = pd.read_csv(file, encoding="utf-8")
                columns_to_drop = [
                    col for col in columns_to_drop if col in df_temp.columns
                ]
                df_temp = df_temp.drop(columns=columns_to_drop)
                df_concat = pd.concat([df_temp, df_concat], axis=1)
            df_concat = pd.concat([csv_df, df_concat], axis=1)
            df_concat.to_csv(
                f"AQME-ROBERT_{self.args.descp_lvl}_{csv_target}", index=False
            )

        # if no qdesc_atom is set, only keep molecular properties and discard atomic properties
        aqme_db = f"AQME-ROBERT_{self.args.descp_lvl}_{csv_target}"

        # ensure that the AQME database was successfully created
        if not os.path.exists(aqme_db):
            self.args.log.write(
                "\nx  The initial AQME descriptor protocol did not create any CSV output!"
            )
            _append_aqme_job_logs(self.args.log)
            self.args.log.finalize()
            sys.exit(1)

        # remove atomic properties if no SMARTS patterns were selected in qdescp,
        # and drop AQME argument columns from CSV inputs (single read/write)
        if "qdescp_atoms" not in self.args.qdescp_keywords:
            _ = filter_atom_prop_and_aqme_args(aqme_db, csv_df, strip_atom_lists=True)
        else:
            _ = filter_atom_prop_and_aqme_args(aqme_db, csv_df, strip_atom_lists=False)
        # delete AQME_indiv*.csv files
        for file in glob.glob("*QME_indiv*.csv"):
            os.remove(file)

        # this returns stores options just in case csv_test is included
        return self

    def _fail_aqme_job(self, message):
        self.args.log.write(f"\n{message}")
        _append_aqme_job_logs(self.args.log)
        self.args.log.finalize()
        sys.exit(1)

    def run_aqme(self, command, extra_keywords, *, expected_csv=None):
        """
        Function that runs the AQME jobs
        """

        if extra_keywords != "":
            split_args = shlex.split(extra_keywords, posix=(os.name != "nt"))
            command.extend(split_args)

        env = os.environ.copy()
        py_bin = os.path.dirname(sys.executable)
        if os.name == "nt":
            win_paths = [
                py_bin,
                os.path.join(sys.prefix, "Library", "bin"),
                os.path.join(sys.prefix, "Scripts"),
            ]
        else:
            win_paths = [py_bin]
        current_path = env.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        for path_dir in win_paths:
            if os.path.isdir(path_dir) and path_dir not in path_entries:
                path_entries.insert(0, path_dir)
        env["PATH"] = os.pathsep.join(path_entries)

        lib = os.path.join(sys.prefix, "lib")
        if os.path.isdir(lib):
            if sys.platform == "darwin":
                var_name = "DYLD_FALLBACK_LIBRARY_PATH"
            else:
                var_name = "LD_LIBRARY_PATH"
            previous = env.get(var_name, "")
            entries = previous.split(os.pathsep) if previous else []
            if lib not in entries:
                env[var_name] = lib + (os.pathsep + previous if previous else "")

        # Avoid pipe deadlock: nested AQME/CSEARCH can be very verbose on stdout.
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stderr_tail = (result.stderr or "")[-4000:]
        missing_csv = expected_csv is not None and not os.path.isfile(expected_csv)
        if result.returncode != 0 or missing_csv:
            if result.returncode != 0:
                self.args.log.write(
                    f"\nx  AQME subprocess failed with exit code {result.returncode}."
                )
            if missing_csv:
                self.args.log.write(
                    f"\nx  Expected AQME output not found: {expected_csv}"
                )
            self.args.log.write(f"   Command: {' '.join(command)}")
            if stderr_tail:
                self.args.log.write(f"   stderr (tail):\n{stderr_tail}")
            if (
                "full_level_boltz" in stderr_tail
                and "TypeError" in stderr_tail
                and "NoneType" in stderr_tail
            ):
                self.args.log.write(
                    "   x AQME failed while computing Boltzmann properties (None energies). "
                    "This usually indicates an AQME-side qdescp issue for one or more structures."
                )
            _append_aqme_job_logs(self.args.log)
            return False
        return True

    def init_aqme(self):
        """
        Checks whether AQME is installed
        """

        import importlib.util

        if importlib.util.find_spec("aqme.qprep") is None:
            self.args.log.write(
                "x  AQME is not installed (required for the --aqme option)! The program is typically installed within 2-5 minutes (https://aqme.readthedocs.io, see the Installation section)"
            )
            self.args.log.finalize()
            sys.exit(1)


def filter_atom_prop_and_aqme_args(aqme_db, csv_df, *, strip_atom_lists):
    """
    Drop atomic list descriptors when no --qdescp_atoms was used, and remove
    columns that duplicate AQME CSV inputs (single pass over the dataframe).
    """
    aqme_df = pd.read_csv(aqme_db, encoding="utf-8")
    if strip_atom_lists:
        for column in list(aqme_df.columns):
            if column == "DBSTEP_Vbur":
                aqme_df = aqme_df.drop(column, axis=1)
            # remove lists of atomic properties (skip columns from AQME arguments)
            elif aqme_df[column].dtype == object and column.lower() not in aqme_args:
                first_cell = aqme_df[column].iloc[0] if len(aqme_df) else None
                if (
                    first_cell is not None
                    and "[" in str(first_cell)
                    and column not in csv_df.columns
                ):
                    aqme_df = aqme_df.drop(column, axis=1)
    for column in list(aqme_df.columns):
        if column.lower() in aqme_args:
            aqme_df = aqme_df.drop(column, axis=1)
    os.remove(aqme_db)
    aqme_df.to_csv(f"{aqme_db}", index=None, header=True)


def filter_atom_prop(aqme_db, csv_df):
    """
    Function that filters off atomic properties if no atom was selected in the --qdescp_atoms option
    """

    filter_atom_prop_and_aqme_args(aqme_db, csv_df, strip_atom_lists=True)


def filter_aqme_args(aqme_db):
    """
    Function that filters off AQME arguments in CSV inputs
    """
    filter_atom_prop_and_aqme_args(aqme_db, pd.DataFrame(), strip_atom_lists=False)


def move_aqme():
    """
    Move raw data from AQME-CSEARCH and -QDESCP runs into the AQME folder
    """

    for file in glob.glob("*"):
        if "CSEARCH" in file or "QDESCP" in file:
            if os.path.exists(f"AQME/{file}"):
                if len(os.path.basename(Path(file)).split(".")) == 1:
                    shutil.rmtree(f"AQME/{file}")
                else:
                    os.remove(f"AQME/{file}")
            shutil.move(file, f"AQME/{file}")
    for dat_file in ["AQME/CSEARCH_data.dat", "AQME/QDESCP_data.dat"]:
        if not os.path.exists(dat_file):
            Path(dat_file).write_text("", encoding="utf-8")
