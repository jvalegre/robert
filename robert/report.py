"""
Parameters
----------

    destination : str, default=None,
        Directory to create the output file(s).
    varfile : str, default=None
        Option to parse the variables using a yaml file (specify the filename, i.e. varfile=FILE.yaml).
    report_modules : list of str, default=['CURATE','GENERATE','VERIFY','PREDICT']
        List of the modules to include in the report.
    debug_report : bool, default=False
        Debug mode using during the pytests of report.py

"""
#####################################################.
#        This file stores the REPORT class          #
#    used for generating the final PDF report       #
#####################################################.

import os
import shutil
import re
import sys
import glob
import json
import platform
import pandas as pd
from pathlib import Path
from robert.utils import (load_variables,
    pd_to_dict,
    create_score_heatmap,
)
from robert.report_utils import (
    get_csv_names,
    get_col_score,
    calc_score,
    adv_flawed,
    adv_predict,
    adv_test,
    adv_diff_test,
    adv_cv_sd,
    adv_sorted_cv,
    adv_train_val_gap,
    adv_sorted_cv_high,
    adv_sorted_cv_low,
    adv_spearman,
    adv_bound_sd,
    adv_applicability_domain,
    adv_bound_diagnostic,
    get_col_text,
    repro_info,
    make_report,
    css_content,
    format_lines,
    combine_cols,
    get_metrics,
    get_boundary_metrics,
    get_col_transpa,
    get_outliers,
    detect_predictions,
    get_csv_metrics,
    get_csv_pred
)

# suffixes -> output PDF filenames, one PDF per model
PDF_NAMES = {'No PFI': 'ROBERT_report_No_PFI.pdf', 'PFI': 'ROBERT_report_PFI.pdf'}


class report:
    """
    Class containing all the functions from the REPORT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the REPORT module (for a complete list of variables, visit the ROBERT documentation)
    """

    def __init__(self, **kwargs):
        # check if there is a problem with weasyprint (required for this module)
        # Suppress fontconfig warnings during import on Windows
        if platform.system() == 'Windows':
            import tempfile
            temp_stderr = tempfile.TemporaryFile(mode='w+')
            old_stderr = os.dup(2)
            os.dup2(temp_stderr.fileno(), 2)

        try:
            from weasyprint import HTML
        except (OSError, ModuleNotFoundError):
            if platform.system() == 'Windows':
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                temp_stderr.close()
            print(f"\nx The REPORT module requires some libraries that are missing, the PDF with the summary of the results has not been created. Try installing the libraries with 'conda install -y -c conda-forge glib gtk3 pango mscorefonts'")
            sys.exit()
        finally:
            if platform.system() == 'Windows':
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                temp_stderr.close()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "report")

        eval_only = False
        # if EVALUATE is activated, no PFI models are generated
        path_eval = Path(f'{os.getcwd()}/EVALUATE/EVALUATE_data.dat')
        if os.path.exists(path_eval):
            eval_only = True

        # spacing used for the Boundary robustness (right) column in Sections A and B
        self.spacing_PFI = f'{("&nbsp;")*4}'

        # --all_models: VERIFY/PREDICT split their log into one VERIFY_{MODEL}_data.dat /
        # PREDICT_{MODEL}_data.dat per model (see verify.py/predict.py) instead of a single
        # shared file, since a merged log gives REPORT no reliable way to tell which lines
        # belong to which model. Discover which models actually have both logs present, and
        # generate a full set of PDFs (No PFI + PFI) per model instead of just one set.
        if getattr(self.args, 'all_models', False):
            model_names = sorted(
                os.path.basename(f)[len('VERIFY_'):-len('_data.dat')]
                for f in glob.glob(f'{os.getcwd()}/VERIFY/VERIFY_*_data.dat')
                if os.path.exists(f"{os.getcwd()}/PREDICT/PREDICT_{os.path.basename(f)[len('VERIFY_'):-len('_data.dat')]}_data.dat")
            )
        else:
            model_names = [None]

        # collects each model's final Interpolation/Boundary robustness score, keyed by suffix
        # ('No PFI'/'PFI') then model name - used to build the score-based Section F
        # heatmap (create_score_heatmap()), which every model's own PDF embeds (each PDF
        # shows the SAME heatmap, comparing all models). Since every model's PDF needs the
        # finished heatmap, scores are collected here in a pre-pass over every model BEFORE
        # the main per-model PDF loop below starts - the main loop re-derives its own
        # data_score again per suffix as a side effect of building that PDF's HTML
        # (print_score()), so this duplicates a bit of parsing, but keeps the heatmap logic
        # independent of the (already long) main loop
        model_scores = {'No PFI': {}, 'PFI': {}}

        if getattr(self.args, 'all_models', False) and model_names != [None]:
            for score_model_name in model_names:
                self.model_suffix = f'_{score_model_name}'
                _, _, dat_files_pre, _, _ = self.get_repro(eval_only)

                for suffix in ['No PFI', 'PFI']:
                    if eval_only and suffix == 'PFI':
                        continue

                    _, params_df_pre = self.get_transparency(suffix)
                    pred_type_pre = params_df_pre['type'][0].lower()
                    data_score_pre = calc_score(dat_files_pre, suffix, pred_type_pre, {})
                    # rmse_bound: average of the Low/High sorted-CV scaled RMSEs (regression
                    # only) - used only as the 3rd tie-break criterion for best_models_text()
                    rmse_low = data_score_pre.get(f'scaled_rmse_low_{suffix}')
                    rmse_high = data_score_pre.get(f'scaled_rmse_high_{suffix}')
                    rmse_bound = (rmse_low+rmse_high)/2 if rmse_low is not None and rmse_high is not None else None
                    model_scores[suffix][score_model_name] = (
                        data_score_pre.get(f'interp_score_{suffix}'),
                        data_score_pre.get(f'extrap_score_{suffix}'),
                        data_score_pre.get(f'scaled_rmse_cv_{suffix}'),
                        rmse_bound,
                    )

            for suffix in ['No PFI', 'PFI']:
                if eval_only and suffix == 'PFI':
                    continue
                if model_scores[suffix]:
                    self.save_score_heatmap(model_scores[suffix], suffix)

        # kept on self (not just local) so print_generate() can build the informative
        # "best model for Interpolation / Boundary robustness" line in Section F
        self.model_scores = model_scores

        for model_name in model_names:
            self.model_suffix = f'_{model_name}' if model_name is not None else ''

            # Reproducibility section (these functions only gather information, the sections
            # will be print later in the report) - shared by both PDFs of this model
            citation_dat, repro_dat, dat_files, csv_name, robert_version = self.get_repro(eval_only)

            # generate one PDF per model (No PFI / PFI)
            for suffix in ['No PFI', 'PFI']:
                if eval_only and suffix == 'PFI':
                    continue

                suffix_title = '_'.join(suffix.split())

                # Transparency section
                transpa_dat, params_df = self.get_transparency(suffix)
                pred_type = params_df['type'][0].lower()

                # print header
                report_html = self.print_header(citation_dat)

                # print ROBERT score section
                score_dat, data_score = self.print_score(dat_files, pred_type, suffix, suffix_title)
                report_html += score_dat

                # print warnings in ROBERT score section
                warnings_dat, warnings_dict = self.print_warnings(pred_type, data_score, suffix)
                report_html += warnings_dat

                # print advanced score analysis
                report_html += self.print_adv_anal(pred_type, data_score, suffix, suffix_title)

                # print y distribution
                report_html += self.print_y_distrib(warnings_dict, suffix, suffix_title)

                # print feature importances
                report_html += self.print_features(warnings_dict, suffix, suffix_title)

                # print outlier analysis
                report_html += self.print_outliers(pred_type, suffix, suffix_title)

                # print model screening
                report_html += self.print_generate(eval_only, suffix_title)

                # print reproducibility section
                report_html += repro_dat

                # print transparency section
                report_html += transpa_dat

                # print abbreviation section
                report_html += self.get_abbrev()

                # print new predictions
                report_html += self.print_predictions(pred_type, suffix, suffix_title)

                # print miscellaneous section
                report_html += self.print_misc()

                if self.args.debug_report:
                    with open(f"report_debug{self.model_suffix}_{suffix_title}.txt", "w", encoding="utf-8") as debug_text:
                        debug_text.write(report_html)

                # create css
                with open("report.css", "w", encoding="utf-8") as cssfile:
                    cssfile.write(css_content(csv_name,robert_version))

                pdf_name = PDF_NAMES[suffix] if model_name is None else f'ROBERT_report_{model_name}_{suffix_title}.pdf'

                # Suppress fontconfig warnings from WeasyPrint on Windows
                # These warnings come from the C library level, so we need to redirect at OS level
                if platform.system() == 'Windows':
                    import tempfile

                    # Create a temporary file to redirect stderr
                    temp_stderr = tempfile.TemporaryFile(mode='w+')
                    old_stderr = os.dup(2)  # Duplicate stderr file descriptor
                    os.dup2(temp_stderr.fileno(), 2)  # Redirect stderr to temp file

                    try:
                        _ = make_report(report_html,HTML,pdf_name)
                    finally:
                        os.dup2(old_stderr, 2)  # Restore stderr
                        os.close(old_stderr)
                        temp_stderr.close()
                else:
                    _ = make_report(report_html,HTML,pdf_name)

                # Remove report.css file
                os.remove("report.css")

                print(f'\no  {pdf_name} was created successfully in the working directory!')

        # --all_models: the working directory would otherwise end up with 8 PDFs (4 models x
        # 2 PFI variants) - move all of them into REPORT_models/ and copy back only the 2 that
        # matter most, so the working directory stays uncluttered by default
        if getattr(self.args, 'all_models', False) and model_names != [None]:
            self.organize_all_models_pdfs()


    @staticmethod
    def _pick_best_model(items,primary_idx,secondary_idx,rmse_idx):
        """
        Shared --all_models tie-break cascade (own score -> other axis's score -> lower RMSE),
        used by both organize_all_models_pdfs() and best_models_text() so the two don't each
        keep their own hand-indexed copy of the same cascade against differently-shaped
        tuples (previously risked drifting out of sync if one tuple's shape ever changed
        without mirroring the index shift in the other). items: list of
        (key, interp, bound, rmse_cv, rmse_bound) tuples, always in this order regardless of
        caller. Returns the single best tuple.
        """
        def sort_key(item):
            primary = item[primary_idx] if item[primary_idx] is not None else -1
            secondary = item[secondary_idx] if item[secondary_idx] is not None else -1
            rmse = item[rmse_idx] if item[rmse_idx] is not None else float('inf')
            return (-primary,-secondary,rmse)
        return sorted(items, key=sort_key)[0]


    def organize_all_models_pdfs(self):
        """
        Moves every --all_models PDF (one per model x PFI variant) into a REPORT_models/
        subfolder, then copies back into the working directory only the single best PDF for
        Interpolation and the single best PDF for Boundary robustness - picked across BOTH
        models AND PFI variants together (unlike best_models_text(), which picks per suffix),
        since there should be exactly one "best" PDF per axis in the working directory. Same
        tie-break cascade as best_models_text(): own score -> other score -> lower RMSE. If
        the same (model, suffix) wins both axes, only that one PDF ends up back in the
        working directory
        """

        candidates = []
        for suffix, suffix_scores in getattr(self,'model_scores',{}).items():
            suffix_title = '_'.join(suffix.split())
            for model,vals in suffix_scores.items():
                interp,bound,rmse_cv,rmse_bound = vals
                pdf_name = f'ROBERT_report_{model}_{suffix_title}.pdf'
                if os.path.exists(pdf_name):
                    candidates.append((pdf_name,interp,bound,rmse_cv,rmse_bound))

        if not candidates:
            return

        best_interp_pdf = self._pick_best_model(candidates,1,2,3)[0]
        best_bound_pdf = self._pick_best_model(candidates,2,1,4)[0]

        dest_dir = Path('REPORT_models')
        dest_dir.mkdir(exist_ok=True)
        for pdf_name,*_ in candidates:
            shutil.move(pdf_name, str(dest_dir / pdf_name))

        for best_pdf in {best_interp_pdf, best_bound_pdf}:
            shutil.copy(str(dest_dir / best_pdf), best_pdf)

        print(f'\no  All {len(candidates)} model PDFs were moved to REPORT_models/')
        if best_interp_pdf == best_bound_pdf:
            print(f'o  {best_interp_pdf} (best for both Interpolation and Boundary robustness) was kept in the working directory')
        else:
            print(f'o  {best_interp_pdf} (best for Interpolation) and {best_bound_pdf} (best for Boundary robustness) were kept in the working directory')


    def print_header(self,citation_dat):
        """
        Retrieves the header for the HTML string
        """

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self._posix_uri(self.args.path_icons)}/Robert_logo.jpg" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            </h1>
            {citation_dat}
            """

        return header_lines


    def print_score(self,dat_files,pred_type,suffix,suffix_title):
        """
        Generates the ROBERT score section (left = Interpolation, right = Boundary robustness)
        """

        # starts with the icon of ROBERT score
        score_dat = ''
        score_dat = self.module_lines('score',score_dat)

        # calculates the ROBERT score (R2 is analogous for accuracy in classification)
        data_score = {}
        data_score = calc_score(dat_files,suffix,pred_type,data_score)

        # the score's thresholds were calibrated assuming the standard test_set=0.2 split (see
        # calc_score()/get_predict_scores() for how score_available is derived, and the note in
        # score.rst) - with a non-standard internal split (including 0, i.e. no held-out test
        # set), the score isn't meaningful, so it's replaced with a short notice instead of a
        # misleadingly low value. --csv_test does NOT affect this: it no longer suppresses the
        # internal test split (see prepare_sets() in utils.py) precisely so the score keeps
        # working normally even when an external test set is also provided - an external file
        # may not even have known y values (e.g. compounds not yet made), so it can't be relied
        # on as a substitute for the internal, score-calibrated test set. Defaults to available
        # (fail open) if the underlying line wasn't found for some reason, rather than hiding
        # the score by mistake
        if not data_score.get(f'score_available_{suffix}', True):
            n_test = data_score.get(f'n_test_{suffix}')
            score_dat += f"""<p style="text-align: justify;">Score not available: this model's test set has {n_test} point(s), not the standard ~20% split the ROBERT score was calibrated for (see <a href="https://robert.readthedocs.io/en/latest/Report/score.html">the ROBERT score documentation</a>). This happens with a custom --test_set value, including 0 (i.e. no held-out test set). The rest of this report (feature importances, outlier analysis, reproducibility, etc.) is unaffected.</p>"""
            return score_dat,data_score

        columns_score = []
        for col in ['interpolation','boundary']:
            spacing = '' if col == 'interpolation' else self.spacing_PFI
            score_key = 'interp_score' if col == 'interpolation' else 'extrap_score'
            score_val = data_score.get(f'{score_key}_{suffix}', 0)
            # only integer score_N.jpg icons exist - both interp_score and extrap_score are
            # already plain integer sums of 0/2 sub-scores (see calc_score()), this round()
            # is just a defensive cast
            score_icon = int(round(score_val))
            score_info = f"""<img src="file:///{self._posix_uri(self.args.path_icons)}/score_{score_icon}.jpg" style="width: 100%; margin-top:7px; margin-bottom:-18px;"></p>"""
            columns_score.append(get_col_score(score_info,data_score,suffix,col,spacing))

        # Combine both columns
        score_dat += combine_cols(columns_score)

        # add corresponding images (left: Results, right: Results_boundary_high)
        # the saved PNGs keep their in-image title (matplotlib plt.text) - height 218 is
        # deliberately LESS than what the full titled image needs at 270px width (~237px,
        # aspect 0.878), so object-fit:cover (anchored bottom, see print_img_tag) crops
        # exactly the title strip off the top and nothing else. 218 = the image's own
        # plot-only height at 270px width (aspect 0.806, measured directly with the title
        # excluded) - don't "fix" this by raising the height to fit the title in, that brings
        # the title back; don't lower it either, that starts cropping into the plot itself
        diff_height = 25 # account for different graph sizes in reg and clas
        height = 218
        if pred_type == 'clas':
            height += diff_height
        score_dat += self.print_score_images(suffix_title,height)

        # metrics text row beneath the images
        module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        columns_summary = [
            get_metrics(module_file,suffix,''),
            get_boundary_metrics(data_score,suffix,pred_type,self.spacing_PFI),
        ]

        # Combine both columns
        score_dat += combine_cols(columns_summary)

        return score_dat,data_score


    def print_score_images(self,suffix_title,height):
        """
        Places the interpolation (Results_*.png) and boundary robustness
        (Results_boundary_high_*.png) images for this model side by side (original fixed-size
        image row layout)
        """

        module_path = Path(f'{os.getcwd()}/PREDICT')

        # glob (not rglob): avoids picking up files from the PREDICT/csv_test subfolder
        interp_images = [str(fp) for fp in module_path.glob('Results_*.png')
                          if self.matches_suffix(str(fp),suffix_title) and 'Results_boundary' not in str(fp)]
        # top 20% (High) plot only - the low/ad/williams variants have their own filenames now
        # (Results_boundary_low_..., Results_boundary_ad_..., Results_boundary_williams_...)
        bound_images = [str(fp) for fp in module_path.glob('Results_boundary_high_*.png')
                          if self.matches_suffix(str(fp),suffix_title)]

        interp_images = self._filter_by_model(interp_images)
        bound_images = self._filter_by_model(bound_images)

        interp_image = interp_images[0] if interp_images else ''
        bound_image = bound_images[0] if bound_images else ''

        left_tag = self.print_img_tag(interp_image,height)
        right_tag = self.print_img_tag(bound_image,height)

        return self.print_img_row_indent(left_tag,right_tag,-5)


    def print_warnings(self,pred_type,data_score,suffix):
        """
        Generates the warning box in the ROBERT score section
        """

        # load spacing, colors, and line and table formats
        space,color_dict,style_lines,warnings_dat = self.get_warning_params()

        # gather the lines from PREDICT where the potential warnings are print
        warnings_dict = self.get_warning_lines(pred_type)

        warnings_dict[f'severe_warnings_{suffix}'] = []
        warnings_dict[f'moderate_warnings_{suffix}'] = []

        # analyze and append warnings
        warnings_dict = self.analyze_warnings(data_score,suffix,warnings_dict,pred_type)

        # add box (full width now that it's a single box, not paired with a PFI column)
        warning_print = f'''
        <div style="width:100%; box-sizing:border-box; border:0.5px solid Gray; padding:4px 4px 4px 4px; margin-top: 20px; min-height:270px; text-align: justify;">'''

        # add severe warnings
        warning_print += f'''
        <p style="margin-bottom: -10px; margin-top: 5px;"><strong>{space}Severe warnings</strong></p>'''
        if len(warnings_dict[f'severe_warnings_{suffix}']) == 0:
            warning_print += self.print_line_warning(
                'No severe warnings detected',
                style_lines,color_dict['blue'],space)
        else:
            for sev_warning in warnings_dict[f'severe_warnings_{suffix}']:
                warning_print += self.print_line_warning(
                    sev_warning,
                    style_lines,color_dict['red'],space)

        # add moderate warnings
        warning_print += f'''
        <p style="margin-bottom: -10px; margin-top: 35px;"><strong>{space}Moderate warnings</strong></p>'''
        if len(warnings_dict[f'moderate_warnings_{suffix}']) == 0:
            warning_print += self.print_line_warning(
                'No moderate warnings detected',
                style_lines,color_dict['blue'],space)
        else:
            for mode_warning in warnings_dict[f'moderate_warnings_{suffix}']:
                warning_print += self.print_line_warning(
                    mode_warning,
                    style_lines,color_dict['yellow'],space)

        # add overall assessment
        warning_print += self.print_assessment(space,suffix,data_score,style_lines,warnings_dict,color_dict,pred_type)

        # end box
        warning_print += '''</div>'''

        warnings_dat += warning_print

        # page break
        warnings_dat += f"""<p style="page-break-after: always;"></p>"""

        return warnings_dat,warnings_dict


    def get_warning_params(self):
        '''
        Load spacing, colors, and line and table formats
        '''

        space = '&nbsp;'
        color_dict = {
            'red': '#c56666',
            'yellow': '#c5c57d',
            'blue': '#9ba5e3'
        }
        style_lines = '<p style="margin-bottom: -10px;">'

        warnings_dat = ''

        return space,color_dict,style_lines,warnings_dat

    def analyze_warnings(self,data_score,suffix,warnings_dict,pred_type):
        '''
        Analyze and append warnings
        '''

        # tests from flawed models
        if data_score[f'flawed_mod_score_{suffix}'] < 0:
            if data_score[f'failed_tests_{suffix}'] > 0:
                warnings_dict[f'severe_warnings_{suffix}'].append('Failing required tests (Section B.1)')
            else:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Some tests are unclear (Section B.1)')

        # variation in CV
        if pred_type == 'reg':
            if data_score[f'cv_sd_score_{suffix}'] == 0:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Imprecise predictions (Section B.6)')
        elif pred_type == 'clas':
            if data_score[f'diff_mcc_score_{suffix}'] == 0:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Imprecise predictions (Section B.5)')

        # y distribution
        if 'WARNING! Your data is not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Uneven y distribution (Section C)')
            elif pred_type == 'clas': # it's severe in clasification
                warnings_dict[f'severe_warnings_{suffix}'].append('Very uneven class distribution (Section C)')
        elif 'WARNING! Your data is slightly not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Slightly uneven y distribution (Section C)')
            elif pred_type == 'clas':
                warnings_dict[f'moderate_warnings_{suffix}'].append('Uneven class distribution (Section C)')

        # feature correlation
        if 'WARNING! High correlations' in warnings_dict[f'pearson_info_{suffix}']:
            warnings_dict[f'moderate_warnings_{suffix}'].append('Highly correlated features (Section D)')
        elif 'WARNING! Noticeable correlations' in warnings_dict[f'pearson_info_{suffix}']:
            warnings_dict[f'moderate_warnings_{suffix}'].append('Moderately correlated features (Section D)')

        # outliers (threshold is set above 6.5 SD, around 99.9 CI)
        if pred_type == 'reg':
            if warnings_dict[f'max_sd_{suffix}'] > 6.5:
                warnings_dict[f'moderate_warnings_{suffix}'].append('Potential "faulty" outliers (Section E)')

        return warnings_dict


    def get_warning_lines(self,pred_type):
        '''
        Gather the lines from PREDICT where the potential warnings are print
        '''

        warnings_dict = {}

        # get lines with warnings from PREDICT
        file_pred = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        with open(file_pred, 'r', encoding='utf-8') as datfile:
            lines = datfile.readlines()
            pfi_section_pearson = False # to get both No PFI and PFI information
            pfi_section_y_dist = False
            pfi_section_outlier = False
            for i,line in enumerate(lines):
                if 'Ideally, variables should show low' in line and not pfi_section_pearson:
                    warnings_dict['pearson_info_No PFI'] = lines[i+1][6:]
                    pfi_section_pearson = True # the next line found will correspond to the PFI section
                elif 'Ideally, variables should show low' in line and pfi_section_pearson:
                    warnings_dict['pearson_info_PFI'] = lines[i+1][6:]
                if 'Ideally, the number of datapoints in' in line and not pfi_section_y_dist:
                    warnings_dict['y_dist_info_No PFI'] = lines[i+2][6:]
                    pfi_section_y_dist = True
                elif 'Ideally, the number of datapoints in' in line and pfi_section_y_dist:
                    warnings_dict['y_dist_info_PFI'] = lines[i+2][6:]
                if pred_type == 'reg':
                    if 'Outliers plot saved' in line and not pfi_section_outlier:
                        max_SD = 0
                        for j in range(i,len(lines)):
                            if '-------' in lines[j]:
                                break
                            elif 'SDs' in lines[j]:
                                # regex instead of a fixed split index: the outlier name (from
                                # --names) can itself contain spaces, which would otherwise
                                # shift where the SD value lands after lines[j].split()
                                sd_match = re.search(r'\(([\d.]+)\s*SDs\)', lines[j])
                                if sd_match:
                                    sd_line = float(sd_match.group(1))
                                    if sd_line > max_SD:
                                        max_SD = sd_line
                        warnings_dict['max_sd_No PFI'] = max_SD
                        pfi_section_outlier = True
                    elif 'Outliers plot saved' in line and pfi_section_outlier:
                        max_SD = 0
                        for j in range(i,len(lines)):
                            if '-------' in lines[j]:
                                break
                            elif 'SDs' in lines[j]:
                                # regex instead of a fixed split index: the outlier name (from
                                # --names) can itself contain spaces, which would otherwise
                                # shift where the SD value lands after lines[j].split()
                                sd_match = re.search(r'\(([\d.]+)\s*SDs\)', lines[j])
                                if sd_match:
                                    sd_line = float(sd_match.group(1))
                                    if sd_line > max_SD:
                                        max_SD = sd_line
                        warnings_dict['max_sd_PFI'] = max_SD

            return warnings_dict


    def print_line_warning(self,message,style_lines,color,space):
        '''
        Add line with warning
        '''

        return f'''
        {style_lines}<span style='font-size:15px; color: {color};'>{space}&#9673;</span>
        {space}{message}</p>'''


    def print_assessment(self,space,suffix,data_score,style_lines,warnings_dict,color_dict,pred_type):
        '''
        Add overall assessment to the ROBERT score section. Interpolation (max 10) and
        Boundary robustness (max 10 for reg, 2 for clas) now have independent, differently-scaled
        scores, so the assessment uses whichever fraction of its own max is worse (a model
        that isn't robust at the boundaries is unreliable even if it interpolates well).
        '''

        assessment_print = f'''
<p style="margin-bottom: -10px; margin-top: 35px;"><strong>{space}Overall assessment</strong></p>'''

        # the verdict below leans on interp_score/extrap_score, which fold in the (unavailable)
        # test-set score component - see print_score() for how score_available is derived
        if not data_score.get(f'score_available_{suffix}', True):
            assessment_print += self.print_line_warning(
                'Not available (no standard test set)',
                style_lines,color_dict['blue'],space)
            return assessment_print

        interp_score = data_score.get(f'interp_score_{suffix}', 0)
        bound_score = data_score.get(f'extrap_score_{suffix}', 0)
        interp_max = 10
        bound_max = 10 if pred_type == 'reg' else 2
        interp_pct = interp_score / interp_max
        bound_pct = bound_score / bound_max if bound_max else 0
        overall_pct = min(interp_pct,bound_pct)

        if len(warnings_dict[f'severe_warnings_{suffix}']) > 0 or overall_pct < 0.5:
            assessment_print += self.print_line_warning(
                'The model is unreliable',
                style_lines,color_dict['red'],space)

        elif overall_pct >= 0.9:
            if pred_type == 'reg' and len(warnings_dict[f'moderate_warnings_{suffix}']) >= 3:
                assessment_print += self.print_line_warning(
                    'Reliable model, but examine warnings',
                    style_lines,color_dict['yellow'],space)
            elif pred_type == 'clas' and len(warnings_dict[f'moderate_warnings_{suffix}']) >= 2:
                assessment_print += self.print_line_warning(
                    'Reliable model, but examine warnings',
                    style_lines,color_dict['yellow'],space)
            else:
                assessment_print += self.print_line_warning(
                    f'The model seems reliable',
                    style_lines,color_dict['blue'],space)

        elif overall_pct >= 0.7:
            assessment_print += self.print_line_warning(
                'Decent model, but it has limitations',
                style_lines,color_dict['yellow'],space)

        elif overall_pct >= 0.5:
            assessment_print += self.print_line_warning(
                'Moderate model, with important limitations',
                style_lines,color_dict['yellow'],space)

        return assessment_print


    def print_adv_anal(self,pred_type,data_score,suffix,suffix_title):
        """
        Generates the advanced score analysis section (left = Interpolation, right = Boundary robustness)
        """

        # Section B is entirely a breakdown of the score computed in Section A - skip it the
        # same way when there's no standard test set to base that score on (see print_score())
        if not data_score.get(f'score_available_{suffix}', True):
            return ''

        adv_score_dat = ''

        adv_score_dat += self.module_lines('adv_anal',adv_score_dat)

        # Text sub-metrics pair up row by row via combine_cols (proven fine for text
        # everywhere else in the report). Images are NOT put in flex/table columns - they use
        # the exact same fixed-270px print_img_row()/print_img_tag() mechanism as Section C's
        # y_distribution image, which is the only approach that sizes images correctly.
        if pred_type == 'reg':
            # items 5 and 6 share the last row (instead of item 6 getting its own trailing
            # row) since boundary robustness's last row (item 5, applicability domain) now carries
            # two stacked images and is much taller - giving item 6 its own row would strand
            # it waiting for that whole tall row to finish, leaving interpolation's column
            # with a large blank gap in between. This doesn't eliminate every such gap (a flex
            # row's height is always set by its tallest child), but three other approaches
            # tried for a full fix - floats, a single continuous flex column, inline-block+
            # <br> stacking - each hit a different WeasyPrint bug shrinking the fixed-width
            # images or breaking pagination, confirmed by direct measurement each time. This
            # row-based version is the one confirmed to render at the correct size with
            # correct pagination, so it's what's shipped despite the residual gap.
            interp_rows = [
                adv_flawed(suffix,data_score,''),
                adv_predict(self,suffix,data_score,'',pred_type),
                adv_test(self,suffix,data_score,'',pred_type),
                adv_train_val_gap(self,suffix,data_score,''),
                adv_diff_test(self,suffix,data_score,'',pred_type) + adv_cv_sd(self,suffix,data_score,''),
            ]

            bound_rows = [
                adv_sorted_cv_high(self,suffix,data_score,self.spacing_PFI),
                adv_sorted_cv_low(self,suffix,data_score,self.spacing_PFI),
                adv_spearman(self,suffix,data_score,self.spacing_PFI),
                adv_bound_sd(self,suffix,data_score,self.spacing_PFI),
                adv_applicability_domain(self,suffix,data_score,self.spacing_PFI),
            ]

        else:
            # classification: Low/High/degradation/bias still aren't defined for MCC, so
            # Boundary robustness keeps only the existing fold-consistency score (see
            # calc_score). Interpolation item 6 (prediction stability) is defined for
            # classification too - see the 'clas' branch in get_predict_scores()
            interp_rows = [
                adv_flawed(suffix,data_score,''),
                adv_predict(self,suffix,data_score,'',pred_type),
                adv_test(self,suffix,data_score,'',pred_type),
                adv_diff_test(self,suffix,data_score,'',pred_type) + adv_cv_sd(self,suffix,data_score,'',pred_type),
            ]

            bound_rows = [
                adv_sorted_cv(self,suffix,data_score,self.spacing_PFI,pred_type),
            ]

        # images are interleaved right after the row they illustrate (not all dumped at the
        # end) so each one stays visually attached to its sub-metric: VERIFY_tests + High plot
        # after row 0 (item 1 on both sides), Low plot after row 1 (item 2, boundary robustness
        # side only), CV_variability (SD) after the last interpolation row (item 6)
        # the saved PNGs all keep their in-image title (matplotlib plt.text) - every height
        # below is deliberately the image's OWN plot-only height at 270px width (title
        # excluded), which is LESS than what the full titled image needs, so
        # object-fit:cover (anchored bottom, see print_img_tag) crops exactly the title strip
        # off the top of each one and nothing else. Each image type has a very slightly
        # different aspect ratio (VERIFY_tests' bar chart especially, and Williams has one
        # fewer legend entry than the others), so this only works if every declared height
        # matches its own image type - reusing one height for images with a different actual
        # aspect ratio brings back a visible (partial or full) title, or crops into the plot
        verify_height = 228 if pred_type == 'reg' else 232

        if pred_type == 'reg':
            interp_tag1 = self.print_img_tag(self.find_img('VERIFY_tests','VERIFY',suffix_title),verify_height)
            bound_tag1 = self.print_img_tag(self.find_img('Results_boundary_high','PREDICT',suffix_title),218)
            # applicability domain gets both plots stacked in the boundary robustness (right)
            # column: the actual-vs-predicted scatter (ties to the Scaled RMSE score) on top, the
            # Williams plot (leverage vs standardized residual, flags which points are
            # structurally suspicious) below it
            williams_tag = self.print_img_tag(self.find_img('Results_boundary_williams','PREDICT',suffix_title),219)
            ad_tag = self.print_img_tag(self.find_img('Results_boundary_ad','PREDICT',suffix_title),218)
            cv_variability_tag = self.print_img_tag(self.find_img('CV_variability','PREDICT',suffix_title,exclude=['CV_variability_boundary','CV_variability_cv']),218)
            # item 6's 2nd SD plot: out-of-fold CV predictions in train+validation (vs the
            # existing cv_variability_tag, which is the test set) - see graph_reg(sd_set='train')
            cv_variability_cv_tag = self.print_img_tag(self.find_img('CV_variability_cv','PREDICT',suffix_title),218)
            # two separate stacked rows (not one row with both images wrapped together) -
            # wrapping both in a shared inline-block/span container shrank the fixed-width
            # images below their declared size (the same WeasyPrint quirk flex/position:
            # absolute hit earlier); concatenating two independent print_img_row_indent calls
            # reuses the already-proven single-image mechanism twice instead. CV_variability
            # (interpolation's item 6, test SD) pairs with the first AD image since both are
            # item 5's row - the 2nd row now pairs interpolation's new CV-variability (train)
            # image with boundary robustness's Williams plot, since both columns have 2 images here
            ad_stacked = (
                self.print_img_row_indent(cv_variability_tag,ad_tag,10) +
                self.print_img_row_indent(cv_variability_cv_tag,williams_tag,10)
            )
            img_after_row = {
                0: self.print_img_row_indent(interp_tag1,bound_tag1,13),
                1: self.print_img_row_indent('',self.print_img_tag(self.find_img('Results_boundary_low','PREDICT',suffix_title),218),10),
                4: ad_stacked,
            }
        else:
            interp_tag1 = self.print_img_tag(self.find_img('VERIFY_tests','VERIFY',suffix_title),verify_height)
            img_after_row = {0: self.print_img_row_indent(interp_tag1,'',13)}

        for i in range(max(len(interp_rows),len(bound_rows))):
            left = interp_rows[i] if i < len(interp_rows) else ''
            right = bound_rows[i] if i < len(bound_rows) else ''
            adv_score_dat += combine_cols([left,right],align_top=True)
            adv_score_dat += img_after_row.get(i,'')

        if pred_type == 'reg':
            # item 6 (unscored diagnostics) goes after the applicability domain plots, not
            # merged into item 5's row - so it reads as "here's extra context once you've seen
            # the plots" rather than interrupting the text-then-image flow of item 5
            bound_diagnostic = adv_bound_diagnostic(suffix,data_score,self.spacing_PFI)
            adv_score_dat += combine_cols(['',bound_diagnostic],align_top=True)

        adv_score_dat += '<p style="margin-bottom: 50px;"></p>'

        return adv_score_dat


    def print_misc(self):
        """
        Generates the miscellaneous section
        """

        misc_dat = ''
        misc_dat += self.module_lines('misc',misc_dat)

        # get some tips
        style_line = '<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation
        misc_dat += f"""<p style="text-align: justify; margin-top: -3px; margin-bottom: -3px;"><u>Some general tips to improve the score</u></p>"""
        misc_dat += f'<p style="text-align: justify;">1. Adding meaningful datapoints might help to improve the model. Also, using a uniform population of datapoints across the whole range of y values usually helps to obtain reliable predictions across the whole range. More information about the range of y values used is available in Section C.</p>'
        misc_dat += f'{style_line}2. Adding meaningful descriptors or replacing/deleting the least useful descriptors used might help. Feature importances are gathered in Section D.</p>'


        # how to predict new values
        misc_dat += f"""
        <br><p style="text-align: justify; margin-top: -3px; margin-bottom: -3px;"><u>How to predict new values with these models?</u></p>
<p style="text-align: justify;">1. Create a CSV database with the new points, including the necessary descriptors.</p>
{style_line}2. Place the CSV file in the parent folder (i.e., where the module folders were created)</p>
{style_line}3. Run the PREDICT module as 'python -m robert --predict --csv_test FILENAME.csv'.</p>
{style_line}4. The predictions will be shown at the end of the resulting PDF report and will be stored in the last column of two CSV files called MODEL_SIZE_test(_No)_PFI.csv, which are in the PREDICT folder.</p>"""

        # add separator line
        misc_dat += '<hr style="margin-top: 20px;">'

        return misc_dat


    def print_outliers(self,pred_type,suffix,suffix_title):
        """
        Generates the outliers section (2 columns, same content on both sides for now)
        """

        # starts with the icon of outliers
        outlier_dat = ''
        outlier_dat = self.module_lines('outliers',outlier_dat,pred_type=pred_type)

        if pred_type == 'reg':
            # get information about outliers
            module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
            columns_outlier = []
            for col in ['interpolation','boundary']:
                spacing = '' if col == 'interpolation' else self.spacing_PFI
                columns_outlier.append(get_outliers(module_file,suffix,spacing))
            outlier_dat += combine_cols(columns_outlier)

            # add corresponding image
            height = 217
            outlier_dat += self.print_img('Outliers',-5,height,'PREDICT',suffix_title)

        # add separator line and page break
        outlier_dat += '<hr style="margin-top: 20px;">'
        outlier_dat += f"""<p style="page-break-after: always;"></p>"""

        return outlier_dat


    def print_y_distrib(self,warnings_dict,suffix,suffix_title):
        """
        Generates the y distribution section (2 columns, same content on both sides for now)
        """

        # starts with the icon of outliers
        distrib_dat = ''
        distrib_dat = self.module_lines('y_distrib',distrib_dat)

        # add corresponding image
        height = 220
        distrib_dat += self.print_img('y_distribution',-5,height,'PREDICT',suffix_title)

        columns_y_distrib = []
        for col in ['interpolation','boundary']:
            spacing = '' if col == 'interpolation' else self.spacing_PFI

            # split the sentence into 1 column size and add spacing line by line
            y_distrib_sentence = format_lines(warnings_dict[f'y_dist_info_{suffix}'],max_width=55,one_column=True,spacing=spacing)

            column = f"""
            <p style='margin-top:25px; margin-bottom:-6px'><span style="font-weight:bold;">{spacing*3}y distribution analysis</span></p>
            {y_distrib_sentence}
            <p style='margin-bottom:-15px'></p>
            """
            columns_y_distrib.append(column)

        distrib_dat += combine_cols(columns_y_distrib)

        distrib_dat += '<p style="margin-bottom: 30px;"></p>'

        # add separator line and page break
        distrib_dat += '<hr style="margin-top: 20px;">'
        distrib_dat += f"""<p style="page-break-after: always;"></p>"""

        return distrib_dat


    def print_features(self,warnings_dict,suffix,suffix_title):
        """
        Generates the feature analysis section (2 columns, same content on both sides for now)
        """

        # starts with the icon of feature importances
        feature_dat = ''
        feature_dat = self.module_lines('features',feature_dat)

        # Add the linear model equation (only shown when the model is MVL). Located within this
        # suffix's own "Summary of results" section instead of picking the Nth equation found in
        # the whole file by position, since only one of No_PFI/PFI may actually be an MVL model
        equation = None
        with open(f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if 'o  Summary of results' in line and (('No_PFI:' in line) == (suffix == 'No PFI')):
                for j in range(i, len(lines)):
                    if 'o  SHAP' in lines[j]:
                        break
                    if 'o  Linear model equation' in lines[j]:
                        equation = lines[j + 1].strip().lstrip('- ')
                        break
                break

        if equation:
            feature_dat += "<p style='margin-top:-5px; margin-bottom:-6px'><span style='font-weight:bold;'>Linear model equation</span></p>"
            feature_dat += f"<p style='margin-top:10px; margin-bottom:35px'>{equation}</p>"

        # add corresponding images
        module_path = Path(f'{os.getcwd()}/PREDICT')

        shap_images = self._filter_by_model([img for img in glob.glob(f'{module_path}/SHAP_*.png') if self.matches_suffix(img,suffix_title)])
        pfi_images = self._filter_by_model([img for img in glob.glob(f'{module_path}/PFI_*.png') if self.matches_suffix(img,suffix_title)])
        pearson_images = self._filter_by_model([img for img in glob.glob(f'{module_path}/Pearson_*.png') if self.matches_suffix(img,suffix_title)])

        margin_top, margin_bottom = -10,30
        missing_messages = {
            'SHAP': 'SHAP plot not found.',
            'PFI': 'PFI plot not found.',
            # Pearson maps aren't created when >30 descriptors - the one case with a known cause
            'Pearson': 'Pearson maps not created if >30 descriptors.',
        }
        for label, images in [('SHAP', shap_images), ('PFI', pfi_images), ('Pearson', pearson_images)]:
            if len(images) == 1:
                pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px"><img src="file:///{self._posix_uri(images[0])}" style="margin: 0; width: 100%;"/>'
                pair_list += f'{("&nbsp;")*22}'
                pair_list += f'<img src="file:///{self._posix_uri(images[0])}" style="margin: 0; width: 100%;"/></p>'
            else:
                pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px">{missing_messages[label]}</p>'
            feature_dat += pair_list

        columns_pearson = []
        for col in ['interpolation','boundary']:
            spacing = '' if col == 'interpolation' else self.spacing_PFI

            # split the sentence into 1 column size and add spacing line by line
            pearson_sentence = format_lines(warnings_dict[f'pearson_info_{suffix}'],max_width=55,one_column=True,spacing=spacing)

            column = f"""
            <p style='margin-top:-10px; margin-bottom:-6px'><span style="font-weight:bold;">{spacing*3}Correlation analysis</span></p>
            {pearson_sentence}
            """
            columns_pearson.append(column)

        feature_dat += combine_cols(columns_pearson)

        # add separator line and page break
        feature_dat += '<hr style="margin-top: 10px;">'
        feature_dat += f"""<p style="page-break-after: always;"></p>"""

        return feature_dat


    def print_generate(self,eval_only,suffix_title):
        """
        Generates the GENERATE hyperoptimization section
        """

        # starts with the icon of feature importances
        generate_dat = ''
        generate_dat = self.module_lines('generate',generate_dat,eval_only=eval_only)

        # add corresponding image - with --all_models, every model's PDF embeds the SAME pair
        # of score-based heatmaps (Interpolation per model on the left, Boundary robustness on
        # the right, built in __init__'s pre-pass via save_score_heatmap()) instead of GENERATE's
        # raw combined-RMSE one, since the RMSE used to pick BO hyperparameters isn't the most
        # useful basis to compare models against each other once the full VERIFY/PREDICT
        # scores are known. Kept as two separate images (not one combined heatmap) so the
        # left/right split matches the Interpolation/Boundary robustness separation used
        # everywhere else in the report
        if not eval_only:
            height = 236
            if getattr(self.args, 'all_models', False):
                # NOTE: this deliberately does NOT use combine_cols() (flex divs) - WeasyPrint
                # silently ignores an explicit <img> width when the image sits inside a flex
                # item (same issue print_img_row_indent works around elsewhere in this file),
                # so both images rendered at a tiny, uncontrollable size (~150px) under
                # combine_cols. Plain inline flow (like print_img_row/print_img_row_indent) is
                # the only layout WeasyPrint sizes correctly here. Reuses the exact same
                # 270px/61.2px width+gap every other paired-image row in the report already
                # uses (confirmed by measuring print_score_images' rendered PDF output) -
                # widening beyond 270px was tried and overflows the page's content width once
                # the gap needed to keep the right column's start aligned is added back in
                img_w = 270
                gap_px = 61.2
                spacer = f'<span style="display: inline-block; width: {gap_px}px;"></span>'

                interp_path = self.find_img('ScoreHeatmapInterp','GENERATE',suffix_title)
                bound_path = self.find_img('ScoreHeatmapBound','GENERATE',suffix_title)
                interp_tag = f'<img src="file:///{self._posix_uri(interp_path)}" style="margin: 0; width: {img_w}px;"/>' if interp_path else ''
                bound_tag = f'<img src="file:///{self._posix_uri(bound_path)}" style="margin: 0; width: {img_w}px;"/>' if bound_path else ''

                interp_cap = f'<span style="display: inline-block; width: {img_w}px; text-align: center; font-weight:bold;">Interpolation</span>'
                bound_cap = f'<span style="display: inline-block; width: {img_w}px; text-align: center; font-weight:bold;">Boundary robustness</span>'

                generate_dat += f'<p style="margin-bottom: 6px;">{interp_cap}{spacer}{bound_cap}</p>'
                generate_dat += f'<p style="margin-top: -5px;">{interp_tag}{spacer}{bound_tag}</p>'
                generate_dat += self.best_models_text(suffix_title)
            else:
                generate_dat += self.print_img('Heatmap',-5,height,'GENERATE',suffix_title)

        generate_dat += '<p style="margin-bottom: 50px;"></p>'

        return generate_dat


    def best_models_text(self,suffix_title):
        """
        Informative-only line (doesn't change which model(s) run through VERIFY/PREDICT/
        REPORT) showing, for this PFI variant, the best model for Interpolation and the
        best model for Boundary robustness, picked from the --all_models pre-pass scores
        collected in self.model_scores (see __init__). Tie-break cascade: own score ->
        the other score -> lower RMSE (scaled_rmse_cv for Interpolation, average of the
        Low/High sorted-CV scaled RMSEs for Boundary robustness)
        """

        suffix = suffix_title.replace('_',' ')
        scores = getattr(self,'model_scores',{}).get(suffix,{})
        if not scores:
            return ''

        # canonical (key, interp, bound, rmse_cv, rmse_bound) shape - same as
        # organize_all_models_pdfs()'s candidates, so _pick_best_model() can use the exact same
        # index arguments there instead of a second hand-shifted copy
        items = [(model,vals[0],vals[1],vals[2],vals[3]) for model,vals in scores.items()]

        interp_best = self._pick_best_model(items,1,2,3)
        bound_best = self._pick_best_model(items,2,1,4)
        interp_model,interp_vals = interp_best[0],interp_best[1:]
        bound_model,bound_vals = bound_best[0],bound_best[1:]

        return (f'<p style="margin-top: 10px; margin-bottom: 0px; font-size: 12.5px;"><i>'
                f'Best for Interpolation: <b>{interp_model}</b> (Interpolation {interp_vals[0]}, Boundary robustness {interp_vals[1]})'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'Best for Boundary robustness: <b>{bound_model}</b> (Boundary robustness {bound_vals[1]}, Interpolation {bound_vals[0]})'
                f'</i></p>')


    def save_score_heatmap(self,model_scores_suffix,suffix):
        """
        Builds Section F's Interpolation and Boundary robustness score heatmaps (one
        model-per-model image each, 0-10) for one PFI variant, from the interp/extrap scores
        collected across every model in the --all_models pre-pass (model_scores_suffix:
        {model_name: (interp_score, extrap_score)}). Two separate single-row images instead of
        one combined 2-row image, so Interpolation can be placed in the report's left column
        and Boundary robustness in the right, keeping the same left/right separation as the
        rest of the report
        """

        # keep the same model order used everywhere else in the report (self.args.model)
        model_cols = [model.upper() for model in self.args.model if model.upper() in model_scores_suffix]

        save_dir = Path(f'{os.getcwd()}/GENERATE/Raw_data')
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix_title = '_'.join(suffix.split())

        for score_idx,(label,file_name) in enumerate([('Interpolation','ScoreHeatmapInterp'),('Boundary robustness','ScoreHeatmapBound')]):
            csv_df = pd.DataFrame(
                {model: [model_scores_suffix[model][score_idx]] for model in model_cols},
                index=[label],
            )
            _ = create_score_heatmap(csv_df, save_dir / f'{file_name}_{suffix_title}.png')


    def get_repro(self,eval_only):
        """
        Generates the reproducibility section
        """

        version_n_date, citation, command_line, python_version, total_time, dat_files = repro_info(self.args.report_modules,self.model_suffix)
        robert_version = version_n_date.split()[2]

        if self.args.csv_name == '' or self.args.csv_test == '':
            self = get_csv_names(self,command_line)

        repro_dat,citation_dat = '',''

        # version, date and citation
        citation_dat += f"""<p style="text-align: justify; margin-top: -9px;"><br>{version_n_date}</p>
        <p style="text-align: justify;  margin-top: -10px;"><span style="font-weight:bold;">How to cite:</span> {citation}</p>"""

        aqme_workflow,aqme_updated = False,True
        crest_workflow = False
        if '--aqme' in command_line:
            original_command = command_line
            aqme_workflow = True
            command_line = command_line.replace('AQME-ROBERT_','')
            self.args.csv_name = f'{self.args.csv_name}'.replace('AQME-ROBERT_','')
            if self.args.csv_test != '':
                self.args.csv_test = f'{self.args.csv_test}'.replace('AQME-ROBERT_','')

        if '--program crest' in command_line.lower():
            crest_workflow = True

        # make the text more compact if --aqme is used (more lines are included)
        if aqme_workflow:
            first_line = f'<p style="text-align: justify; margin-bottom: 10px; margin-top: -16px;">' # reduces line separation separation
        else:
            first_line = f'<p style="text-align: justify; margin-bottom: 10px; margin-top: -8px;">' # reduces line separation separation
        reduced_line = f'<p style="text-align: justify; margin-top: -5px;">' # reduces line separation separation
        space = ('&nbsp;')*4

        # just in case the command lines are so long
        command_line = format_lines(command_line,cmd_line=True)

        # reproducibility section, starts with the icon of reproducibility
        repro_dat += f"""{first_line}<br><strong>1. Download these files <i>(the authors should have uploaded the files as supporting information!)</i>:</strong></p>"""
        repro_dat += f"""{reduced_line}{space}- CSV database ({self.args.csv_name})</p>"""
        if self.args.csv_test != '':
            repro_dat += f"""{reduced_line}{space}- External test set ({self.args.csv_test})</p>"""

        if aqme_workflow:
            try:
                path_aqme = Path(f'{os.getcwd()}/AQME/CSEARCH_data.dat')
                with open(path_aqme, 'r', errors="replace") as datfile:
                    outlines = datfile.readlines()
                aqme_version = outlines[0].split()[2]
                find_aqme = True
            except:
                find_aqme = False
                aqme_version = '0.0' # dummy number
            if int(aqme_version.split('.')[0]) in [0,1] and int(aqme_version.split('.')[1]) < 6:
                aqme_updated = False
                repro_dat += f"""{reduced_line}{space}<i>Warning! This workflow might not be exactly reproducible, update to AQME v1.6.0+ (pip install aqme --upgrade)</i></p>"""
                repro_dat += f"""{reduced_line}{space}To obtain the same results, download the descriptor database (AQME-ROBERT_{self.args.csv_name}) and run:</p>"""
                repro_line = []
                original_command = original_command.replace(self.args.csv_name,f'AQME-ROBERT_{self.args.csv_name}')
                for i,keyword in enumerate(original_command.split('"')):
                    if i == 0:
                        if '--aqme' not in keyword and '--qdescp_keywords' not in keyword and '--csearch_keywords' not in keyword:
                                repro_line.append(keyword)
                        else:
                            repro_line.append('python -m robert ')
                    if i > 0:
                        if '--qdescp_keywords' not in original_command.split('"')[i-1] and '--csearch_keywords' not in original_command.split('"')[i-1]:
                            if '--aqme' not in keyword and '--qdescp_keywords' not in keyword and '--csearch_keywords' not in keyword and keyword != '\n':
                                repro_line.append(keyword)
                repro_line = '"'.join(repro_line)
                repro_line += '"'
                if '--names ' not in repro_line:
                    repro_line += ' --names "code_name"'
                repro_line = f'{reduced_line}{space}- Run: {repro_line}'
                repro_line = format_lines(repro_line,cmd_line=True)
                repro_dat += f"""{reduced_line}{repro_line}</p>"""

        if aqme_workflow and not aqme_updated:
            # I use a very reduced line in this title because the formatted command_line comes with an extra blank line
            # (if AQME is not updated the PDF contains a reproducibility warning)
            repro_dat += f"""<p style="text-align: justify; margin-top: -44px;"><br><strong>2. Install and adjust the versions of the following Python modules:</strong></p>"""
        else:
            repro_dat += f"""{first_line}<br><strong>2. Install and adjust the versions of the following Python modules:</strong></p>"""
        repro_dat += f"""{reduced_line}{space}- Install ROBERT and its dependencies: conda install -y -c conda-forge robert</p>"""
        repro_dat += f"""{reduced_line}{space}- Adjust ROBERT version: pip install robert=={robert_version}</p>"""

        if aqme_workflow:
            if not find_aqme:
                repro_dat += f"""{reduced_line}{space}- AQME is required, but no version was found:</p>"""
                repro_dat += f"""{reduced_line}{space}- Install AQME and its dependencies: pip install aqme==VERSION_USED</p>"""
            if find_aqme:
                repro_dat += f"""{reduced_line}{space}- Install or adjust AQME version: pip install aqme=={aqme_version}</p>"""

            try:
                path_xtb = Path(f'{os.getcwd()}/AQME/QDESCP')
                xtb_json = glob.glob(f'{path_xtb}/*.json')[0]
                with open(xtb_json, "r") as f:  # Opening JSON file
                    data = json.loads(f.read())  # read file
                xtb_version = data['xtb version'].split()[0]
                find_xtb = True
            except:
                find_xtb = False
            if not find_xtb:
                repro_dat += f"""{reduced_line}{space}- xTB is required, but no version was found:</p>"""
            repro_dat += f"""{reduced_line}{space}- Install xTB: conda install -y -c conda-forge xtb</p>"""
            if find_xtb:
                repro_dat += f"""{reduced_line}{space}- Adjust xTB version (if possible): conda install -y -c conda-forge xtb={xtb_version}</p>"""

        if crest_workflow:
            try:
                from importlib.metadata import PackageNotFoundError, version as importlib_version
                crest_version = importlib_version("crest")
                find_crest = True
            except PackageNotFoundError:
                find_crest = False
            if not find_crest:
                repro_dat += f"""{reduced_line}{space}- CREST is required, but no version was found:</p>"""
            repro_dat += f"""{reduced_line}{space}- Install CREST: conda install -c conda-forge crest</p>"""
            if find_crest:
                repro_dat += f"""{reduced_line}{space}- Adjust CREST version: conda install -c conda-forge crest={crest_version})</p>"""

        character_line = ''
        if self.args.csv_test != '':
            character_line += 's'

        repro_dat += f"""{first_line}<br><strong>3. Run ROBERT using this command line in the folder with the CSV database{character_line}:</strong></p>{reduced_line}{command_line}</p>"""

        # I use a very reduced line in this title because the formatted command_line comes with an extra blank line
        if aqme_workflow:
            repro_dat += f"""<p style="text-align: justify; margin-top: -44px;"><br><strong>4. Execution time, Python version and OS:</strong></p>"""
        else:
            repro_dat += f"""<p style="text-align: justify; margin-top: -37px;"><br><strong>4. Execution time, Python version and OS:</strong></p>"""

        # add total execution time
        repro_dat += f"""{reduced_line}Originally run in Python {python_version} using {platform.system()} {platform.version()}</p>"""
        repro_dat += f"""{reduced_line}Total execution time: {total_time} seconds <i>(the number of processors should be specified by the user)</i></p>"""

        # add separator line and page break
        repro_dat += '<hr style="margin-top: 20px;">'
        repro_dat += f"""<p style="page-break-after: always;"></p>"""

        repro_dat = self.module_lines('repro',repro_dat)

        return citation_dat, repro_dat, dat_files, self.args.csv_name, robert_version


    def get_transparency(self,suffix):
        """
        Generates the transparency section
        """

        transpa_dat = ''
        titles_line = f'<p style="text-align: justify; margin-top: -12px; margin-bottom: 3px">' # reduces line separation separation

        # add params of the model
        transpa_dat += f"""{titles_line}<br><strong>1. Parameters of the scikit-learn model (same keywords as used in scikit-learn):</strong></p>"""

        model_dat, params_df = self.transpa_model_misc('model_section',suffix)
        transpa_dat += model_dat

        # add misc params
        transpa_dat += f"""<p style="text-align: justify; margin-top: -95px; margin-bottom: 3px;"><br><strong>2. ROBERT options, including prediction type (REG or CLAS), folds and repeats used for CV, etc:</strong></p>"""

        section_dat, params_df = self.transpa_model_misc('misc_section',suffix)
        transpa_dat += section_dat

        transpa_dat = self.module_lines('transpa',transpa_dat)


        return transpa_dat,params_df


    def transpa_model_misc(self,section,suffix):
        """
        Collects the data for model parameters and misc options in the Reproducibility section
        (2 columns, same content on both sides for now)
        """

        # set the parameters for the ML model
        params_dir = f'{self.args.params_dir}/{"_".join(suffix.split())}'
        files_param = glob.glob(f'{params_dir}/*.csv')
        for file_param in files_param:
            if '_db' not in file_param:
                params_df = pd.read_csv(file_param, encoding='utf-8')
        params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

        columns_repro = []
        for col in ['interpolation','boundary']:
            spacing = '' if col == 'interpolation' else self.spacing_PFI
            columns_repro.append(get_col_transpa(params_dict,suffix,section,spacing))
        section_dat = combine_cols(columns_repro)
        section_dat += '<p style="text-align: justify; margin-top: -70px;">'

        return section_dat,params_df


    def get_abbrev(self):
        """
        Generates the abbreviations section
        """

        # starts with the icon of abbreviation
        abbrev_dat = ''
        abbrev_dat = self.module_lines('abbrev',abbrev_dat)

        columns_abbrev = []
        columns_abbrev.append(get_col_text('abbrev_1'))
        columns_abbrev.append(get_col_text('abbrev_2'))
        columns_abbrev.append(get_col_text('abbrev_3'))

        abbrev_dat += combine_cols(columns_abbrev)
        abbrev_dat +=f'<hr style="margin-top: 15px;">'

        abbrev_dat += f"""<p style="page-break-after: always;"></p>"""

        return abbrev_dat


    def print_predictions(self,pred_type,suffix,suffix_title):
        """
        Generates the new predictions section (2 columns, same content on both sides for now)
        """

        # detects whether there are predictions from an external test set
        module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        csv_test_exists, y_value, names, path_csv_test = detect_predictions(module_file)

        if csv_test_exists:
            pred_dat = ''
            pred_dat = self.module_lines('pred',pred_dat,pred_type=pred_type)

            columns_metrics, columns_pred = [], []
            for col in ['interpolation','boundary']:
                spacing = '' if col == 'interpolation' else self.spacing_PFI

                # add metrics
                columns_metrics.append(get_csv_metrics(module_file,suffix,spacing))

                # add predictions table
                columns_pred.append(get_csv_pred(suffix,path_csv_test,y_value,names,spacing))

            pred_dat += combine_cols(columns_metrics)
            pred_dat += combine_cols(columns_pred)

            # add corresponding image
            height = 217
            if pred_type == 'reg':
                prefix_img = 'CV_variability'
            elif pred_type == 'clas':
                prefix_img = 'Results'
                height += 17
            if len(glob.glob(f'{os.getcwd()}/PREDICT/csv_test/{prefix_img}*.png')) > 0:
                pred_dat += self.print_img(prefix_img,-5,height,'PREDICT/csv_test',suffix_title)

            # add separator line and page break
            pred_dat += '<hr style="margin-top: 20px;">'
            pred_dat += f"""<p style="page-break-after: always;"></p>"""

            return pred_dat

        else:
            return ''


    def module_lines(self,module,module_data,pred_type='reg',eval_only=False):
        """
        Returns the line with icon and module for section titles
        """

        if module == 'score':
            module_name = 'Section A. ROBERT Score'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This score is designed to evaluate the models using different metrics. Interpolation measures how reliably the model predicts within the range of data it was trained on; Boundary robustness measures how well it holds up at the edges of that range, where predictions are hardest to trust.</i>'
        elif module == 'adv_anal':
            module_name = 'Section B. Advanced Score Analysis'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section explains each component that comprises the ROBERT score. <a href="https://robert.readthedocs.io/en/latest/Report/score.html">More details here.</a></i>'
        elif module == 'y_distrib':
            module_name = 'Section C. Distribution of y Values'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section shows the distribution of y values within the training and validation sets.</i>'
        elif module == 'features':
            module_name = 'Section D. Feature Importances'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section presents feature importances measured using the validation set.</i>'
        elif module == 'outliers':
            module_name = 'Section E. Outlier Analysis'
            if pred_type == 'clas':
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This feature is disabled in classification problems.</i>'
            else:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section detects outliers using the standard deviation (SD) of errors from the training set.</i>'
        elif module == 'generate':
            module_name = 'Section F. Model Screening'
            if eval_only:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">The screening of models is disabled when using the EVALUATE module.</i>'
            else:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section compares different combinations of hyperoptimized algorithms and partition sizes. The combined error is calculated as the product of the training error, validation error, and cross-validation error.</i>'
        elif module == 'repro':
            module_name = 'Section G. Reproducibility'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section provides all the instructions to reproduce the results presented.</i>'
        elif module == 'transpa':
            module_name = 'Section H. Transparency'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section contains important parameters used in scikit-learn models and ROBERT.</i>'
        elif module == 'abbrev':
            module_name = 'Section I. Abbreviations'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Reference section for the abbreviations used.</i>'
        elif module == 'pred':
            module_name = 'Section J. New Predictions'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Predictions of the external test set added with the csv_test option.</i>'
        elif module == 'misc':
            module_name = 'Miscellaneous'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">General tips to improve the models and instructions to predict new values.</i>'

        if module not in ['repro','transpa','misc']:
            module_data = format_lines(module_data)
        module_data = '<div class="aqme-content"><pre>' + module_data + '</pre></div>'

        separator_section = '<hr><p style="margin-top:25px;"></p>'

        title_line = f"""
            {separator_section}
            <p><span style="font-weight:bold;">
                <img src="file:///{self._posix_uri(self.args.path_icons)}/{module}.png" alt="" style="width:20px; height:20px; margin-right:5px;">
                {module_name}
            </span></p>{section_explain}
            {module_data}
            </p>
            """

        return title_line


    @staticmethod
    def matches_suffix(filepath,suffix_title):
        """
        Checks whether a file belongs to the No_PFI or PFI model. Plain substring matching isn't
        enough since 'No_PFI' ends in 'PFI', so a 'PFI' file would also match 'No_PFI' filenames.
        """

        if suffix_title == 'PFI':
            return 'PFI' in filepath and 'No_PFI' not in filepath

        return suffix_title in filepath


    def _filter_by_model(self,file_list):
        """
        --all_models: every model's images live in the SAME VERIFY/PREDICT folder with the
        SAME suffix_title (e.g. Results_GB_No_PFI.png, Results_NN_No_PFI.png, ...), so a
        suffix-only filter matches every model at once - narrow down to this model's own
        file. Some images (Section F's score heatmap in GENERATE) are intentionally SHARED
        across every model's PDF and have no model name in their filename at all, so only
        narrow when it actually finds a match - otherwise keep the list as given
        """

        model_name = self.model_suffix[1:] if getattr(self, 'model_suffix', '') else None
        if not model_name:
            return file_list

        model_matches = [f for f in file_list
                          if f'_{model_name}_' in os.path.basename(f) or os.path.basename(f).endswith(f'_{model_name}.png')]
        if model_matches:
            return model_matches

        # empty result: only fall back to the unfiltered list for genuinely shared files (no
        # model name in ANY of their filenames, e.g. GENERATE's score heatmap) - if other
        # models' names do appear in file_list, this model's own file is actually missing, and
        # falling back to the unfiltered list would silently show a DIFFERENT model's image as
        # if it were this model's own (a garbled/misleading PDF with no warning)
        other_models = [m for m in getattr(self.args, 'model', []) if m.upper() != model_name.upper()]
        is_shared = not any(
            f'_{m}_' in os.path.basename(f) or os.path.basename(f).endswith(f'_{m}.png')
            for f in file_list for m in other_models
        )
        return file_list if is_shared else []


    def find_img(self,file_name,module,suffix_title,exclude=None):
        """
        Finds the single image (matching suffix_title) for a given module/file_name prefix.
        'exclude' filters out filenames containing that substring, or any substring in a
        list/tuple (e.g. so a search for 'CV_variability' doesn't also match
        'CV_variability_boundary_...' files)
        """

        module_path = Path(f'{os.getcwd()}/{module}')
        if exclude is None:
            exclude = []
        elif isinstance(exclude,str):
            exclude = [exclude]

        # rglob (matches original behavior): GENERATE's heatmap lives in a Raw_data subfolder
        results_images = [str(file_path) for file_path in module_path.rglob(f'{file_name}_*.png')
                           if self.matches_suffix(str(file_path),suffix_title) and not any(ex in str(file_path) for ex in exclude)]

        results_images = self._filter_by_model(results_images)

        if not results_images:
            return ''

        return results_images[0]


    @staticmethod
    def _posix_uri(path) -> str:
        """
        POSIX-ify a path for embedding in a file:// URI. file:// URIs are only
        well-formed with forward slashes; str(Path) uses backslashes on Windows,
        which some file:// URI parsers don't tolerate.
        """

        return str(path).replace('\\', '/')

    def print_img_tag(self,image_path,height):
        """
        Generates an <img> tag with the original fixed size (270px wide)
        """

        if not image_path:
            return ''

        return f'<img src="file:///{self._posix_uri(image_path)}" style="margin: 0; width: 270px; height: {height}px; object-fit: cover; object-position: 0 100%;"/>'


    def print_img_row_indent(self,left_tag,right_tag,margin_top):
        """
        Places two DIFFERENT images (interpolation left, boundary robustness right) in a
        single flowing paragraph, like print_img_row, but with a precisely calculated spacer
        (not 22 nbsp) so the right image starts at the same x as the indented Boundary
        robustness caption/text above it. Flex columns and position:absolute were both tried
        first, but WeasyPrint silently shrinks a fixed-width <img> below its declared width
        whenever it's wrapped in a flex item or an absolutely positioned span - plain inline
        flow (like the original print_img_row) is the only layout that renders it at full size.
        331.2px = row half-width (234.9pt, confirmed empirically via the width:100% score-bar
        image) + the 18px BOUNDARY_INDENT used for the text above, converted to px; 61.2px is
        that same offset minus the 270px-wide left image.
        """

        spacer_px = 61.2 if left_tag else 331.2
        spacer = f'<span style="display: inline-block; width: {spacer_px}px;"></span>'

        return f'<p style="width: 100%; margin-bottom: -2px; margin-top: {margin_top}px">{left_tag}{spacer}{right_tag}</p>'


    def print_img_row(self,left_tag,right_tag,margin_top):
        """
        Places two image tags side by side in a single full-width paragraph (matches the
        original report layout: fixed-width images separated by a fixed nbsp gap)
        """

        pair_list = f'<p style="width: 100%; margin-bottom: -2px; margin-top: {margin_top}px">{left_tag}{("&nbsp;")*22}{right_tag}</p>'

        return pair_list


    def print_img(self,file_name,margin_top,height,module,suffix_title):
        """
        Generates the row for a single image (matching suffix_title), duplicated on both sides
        """

        tag = self.print_img_tag(self.find_img(file_name,module,suffix_title),height)

        return self.print_img_row(tag,tag,margin_top)


