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
from robert.checkml_score import round_checkml_score
from robert.report_utils import (
    BOUNDARY_INDENT,
    get_csv_names,
    get_col_score,
    calc_score,
    adv_flawed,
    adv_predict,
    adv_test,
    adv_diff_test,
    adv_cv_sd,
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
    score_badge_2,
    get_col_transpa,
    get_outliers,
    detect_predictions,
    get_csv_metrics,
    get_csv_pred
)

# suffixes -> output PDF filenames, one PDF per model
PDF_NAMES = {'No PFI': 'ROBERT_report_No_PFI.pdf', 'PFI': 'ROBERT_report_PFI.pdf'}

# grey section separators always CLOSE a section (never open one), so the line can't end up
# at the top of a page. Only Section A opens with one, below the header on page 1. Sections
# whose own HTML doesn't already end with a line + page break get one in the main loop
# (see _close_section())
def _close_section(section_html):
    '''
    Closes a section that doesn't end with its own line/page break with a grey line. The line
    is the bottom border of a wrapper around the section (not a separate <hr>), so it's always
    attached to the section's last line and can't end up alone at the top of the next page.
    The 20px gap above it is a margin, which is dropped automatically at a page break
    '''
    tail = section_html[-400:]
    if 'page-break-after: always' in tail or '<hr' in tail:
        return section_html
    # trailing empty spacer paragraphs could be pushed alone onto the next page, taking the
    # border with them - the gap above the line is the wrapper's padding instead
    section_html = re.sub(r'(\s*(<p[^>]*>\s*</p>|</p>))+\s*$', '', section_html)
    return (f'<div style="border-bottom: 1px solid #a0a0a0; padding-bottom: 15px; margin-bottom: 25px;">'
            f'{section_html}</div>')

# a descriptor whose importance is this small a fraction of the top descriptor's (or below
# this absolute value) contributes essentially nothing - used by _has_low_relevance_descp()
# below to flag when trimming near-irrelevant descriptors is worth trying (fewer, more
# relevant descriptors often generalize better than many diluted ones)
LOW_RELEVANCE_RELATIVE_THRESHOLD = 0.04
LOW_RELEVANCE_ABSOLUTE_THRESHOLD = 0.05



def _has_low_relevance_descp(lines,start_idx,shap=False):
    '''
    Scans the descriptor-importance lines right after a "PFI plot saved"/"SHAP plot saved"
    marker (see PFI_plot()/shap_analysis() in utils.py for the exact format) and returns True
    if any descriptor other than the top one is negligible: below
    LOW_RELEVANCE_RELATIVE_THRESHOLD of the top descriptor's magnitude, or below
    LOW_RELEVANCE_ABSOLUTE_THRESHOLD in absolute terms. Both PFI and SHAP list descriptors
    sorted from most to least important already, so the first value found is the top one.
    '''

    values = []
    for line in lines[start_idx+1:]:
        stripped = line.strip()
        if not stripped.startswith('-'):
            break
        if shap:
            # "-  {desc} = min: {min}, max: {max}" - use the larger-magnitude side as the
            # importance proxy, matching how these lines are already sorted (utils.py)
            min_match = re.search(r'min:\s*(-?[\d.]+)', stripped)
            max_match = re.search(r'max:\s*(-?[\d.]+)', stripped)
            if not (min_match and max_match):
                continue
            values.append(max(abs(float(min_match.group(1))), abs(float(max_match.group(1)))))
        else:
            # "-  {desc} = {value} +- {sd}" - PFI values can be negative (a genuinely useless/
            # noisy descriptor can shuffle-permute to a *better* score by chance), so compare
            # magnitudes, not signed values
            val_match = re.search(r'=\s*(-?[\d.]+)\s*\+-', stripped)
            if not val_match:
                continue
            values.append(abs(float(val_match.group(1))))

    if len(values) < 2:
        return False

    top = values[0]
    if top <= 0:
        return False
    for val in values[1:]:
        if val < LOW_RELEVANCE_RELATIVE_THRESHOLD * top or val < LOW_RELEVANCE_ABSOLUTE_THRESHOLD:
            return True
    return False


def _scan_outlier_block(lines,start_idx):
    '''
    Scans one "Outliers plot saved" block (see outlier_analysis() in utils.py) and returns
    (max_sd, outlier_pct): the single largest SD value found (as before), and the combined
    outlier percentage across every "{label}: {N} outliers out of {M} datapoints ({pct}%)"
    summary line in the block (e.g. Train + Test combined), so "too many outliers overall"
    can be flagged separately from "one very extreme outlier" (the existing 6.5 SD check).
    '''

    max_sd = 0.0
    total_outliers, total_points = 0, 0
    for line in lines[start_idx:]:
        if '-------' in line:
            break
        sd_match = re.search(r'\(([\d.]+)\s*SDs\)', line)
        if sd_match:
            sd_line = float(sd_match.group(1))
            if sd_line > max_sd:
                max_sd = sd_line
            continue
        pct_match = re.search(r'(\d+) outliers out of (\d+) datapoints', line)
        if pct_match:
            total_outliers += int(pct_match.group(1))
            total_points += int(pct_match.group(2))

    outlier_pct = (total_outliers / total_points * 100) if total_points > 0 else 0.0
    return max_sd, outlier_pct


class report:
    """
    Class containing all the functions from the REPORT module.

    Parameters
    ----------
    kwargs : argument class
        Specify any arguments from the REPORT module (for a complete list of variables, visit the ROBERT documentation)
    """

    # lines available in the Section A warnings box on page 1 (see print_warnings())
    WARNING_LINE_BUDGET = 6

    def __init__(self, **kwargs):
        # check if there is a problem with weasyprint (required for this module)
        # Suppress fontconfig warnings during import on Windows
        if platform.system() == 'Windows':
            import tempfile
            temp_stderr = tempfile.TemporaryFile(mode='w+')
            old_stderr = os.dup(2)
            os.dup2(temp_stderr.fileno(), 2)

        weasyprint_missing = False
        try:
            from weasyprint import HTML
        except (OSError, ModuleNotFoundError):
            weasyprint_missing = True
        finally:
            # stderr is restored only here (it used to be restored in the except block too, so
            # the finally block then called os.dup2() on the already-closed descriptor and the
            # run crashed with "OSError: Bad file descriptor" instead of the message below)
            if platform.system() == 'Windows':
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                temp_stderr.close()
        if weasyprint_missing:
            print(f"\nx The REPORT module requires some libraries that are missing, the PDF with the summary of the results has not been created. Try installing the libraries with 'conda install -y -c conda-forge glib gtk3 pango mscorefonts'")
            sys.exit()

        # load default and user-specified variables
        self.args = load_variables(kwargs, "report")

        # if CHECKML is activated, no PFI models are ever generated (it always scores the
        # model as-is on the descriptors already in the input CSV, never re-curates them) - so
        # the PFI PDF loop below is skipped whenever CHECKML was involved
        path_eval = Path(f'{os.getcwd()}/CHECKML/CHECKML_data.dat')
        eval_only = os.path.exists(path_eval)
        # with --pfi_filter False, GENERATE never creates a Best_model/PFI (or All_models/.../PFI)
        # folder, but the suffix loops below unconditionally tried 'PFI' too - get_transparency()
        # found no CSV there and crashed with UnboundLocalError instead of just skipping it, the
        # same way it already does for CHECKML (see eval_only above)
        skip_pfi = eval_only or not self.args.pfi_filter
        # stored on self too (not just the local var) so module_lines() can add a provenance
        # note to Section A without needing this threaded through print_score()'s own call chain
        self.eval_only = eval_only

        # spacing used for the Boundary robustness (right) column in Sections A and B
        self.spacing_PFI = f'{("&nbsp;")*4}'

        # detects whether CHECKML evaluated a model with user-supplied hyperparameters
        # (model_obj/model_file/model_params) rather than the trivial 'MVL' default - only
        # those carry the optimistic-scores risk flagged in print_warnings() below, since 'MVL'
        # has no hyperparameters that could have been tuned on data ROBERT now scores against.
        # The model name is read off the actual GENERATE output (not re-derived from CLI args,
        # which aren't available here) - CHECKML always narrows GENERATE to a single model, so
        # the first non-'_db' CSV in Best_model/No_PFI names it
        self.custom_model_used = False
        if eval_only:
            no_pfi_files = [f for f in glob.glob(f'{os.getcwd()}/GENERATE/Best_model/No_PFI/*.csv')
                             if not os.path.basename(f).endswith('_db.csv')]
            if no_pfi_files:
                resolved_model_name = os.path.splitext(os.path.basename(no_pfi_files[0]))[0]
                self.custom_model_used = resolved_model_name != 'MVL'

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
            # A REPORT-only run can follow a workflow that was generated with
            # --all_models False. Those runs have shared VERIFY_data.dat and
            # PREDICT_data.dat logs rather than per-model logs. If no paired
            # per-model logs were found, process the shared logs as the single
            # model report instead of silently producing no reports.
            if not model_names and all(
                os.path.exists(f'{os.getcwd()}/{module}/{module}_data.dat')
                for module in ('VERIFY', 'PREDICT')
            ):
                model_names = [None]
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
        self.all_models_clas = False

        if getattr(self.args, 'all_models', False) and model_names != [None]:
            for score_model_name in model_names:
                self.model_suffix = f'_{score_model_name}'
                _, _, dat_files_pre, _, _ = self.get_repro(eval_only)

                for suffix in ['No PFI', 'PFI']:
                    if skip_pfi and suffix == 'PFI':
                        continue

                    _, params_df_pre = self.get_transparency(suffix)
                    pred_type_pre = params_df_pre['type'][0].lower()
                    self.all_models_clas = pred_type_pre == 'clas'
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
                if skip_pfi and suffix == 'PFI':
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
                if skip_pfi and suffix == 'PFI':
                    continue

                suffix_title = '_'.join(suffix.split())

                # Transparency section
                transpa_dat, params_df = self.get_transparency(suffix)
                pred_type = params_df['type'][0].lower()
                # model of this PDF, used by analyze_warnings() for the tree-based model warning
                self.report_model_name = str(params_df['model'][0]) if 'model' in params_df.columns else ''

                # print header
                report_html = self.print_header(citation_dat)

                # print CheckML score section
                score_dat, metrics_dat, data_score = self.print_score(dat_files, pred_type, suffix, suffix_title)
                report_html += score_dat

                # Model metrics sit below Section A's score bars.
                _, warnings_dict = self.print_warnings(dat_files, pred_type, data_score, suffix)
                report_html += metrics_dat

                # page 2: results summary with every warning (see print_summary_page())
                report_html += self.print_summary_page(pred_type, data_score, warnings_dict, suffix, params_df)

                # sections B onwards, each one closed by its own grey line (see _close_section())
                sections = [
                    self.print_adv_anal(pred_type, data_score, suffix, suffix_title),  # B
                    self.print_y_distrib(warnings_dict, suffix, suffix_title),  # C
                    self.print_features(warnings_dict, suffix, suffix_title),  # D
                    self.print_outliers(pred_type, suffix, suffix_title),  # E
                    self.print_generate(eval_only, suffix_title),  # F
                    repro_dat,  # G
                    transpa_dat,  # H
                    self.get_abbrev(),  # I
                    self.print_predictions(pred_type, suffix, suffix_title),  # J (if any)
                    self.print_misc(),  # Miscellaneous
                ]
                for section_html in sections:
                    if section_html:
                        report_html += _close_section(section_html)

                if self.args.debug_report:
                    with open(f"report_debug{self.model_suffix}_{suffix_title}.txt", "w", encoding="utf-8") as debug_text:
                        debug_text.write(report_html)

                # create css
                with open("report.css", "w", encoding="utf-8") as cssfile:
                    cssfile.write(css_content(csv_name,robert_version,getattr(self,'eval_only',False)))

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
        # 2 PFI variants) - move all of them into REPORT_models/ and copy back only the single
        # overall-best one (see _pick_best_overall()), so the working directory stays
        # uncluttered by default
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


    # A candidate whose displayed integer tier is below this is "Very weak" on ROBERT's own
    # scale (see checkml_score_tier()) - too weak on that axis to be a sensible
    # "best overall" pick no matter how good its other axis is.
    OVERALL_FLOOR = 3

    def _pick_best_overall(self,items):
        """
        Picks a single overall-best (key, interp, bound, rmse_cv, rmse_bound) candidate,
        combining Interpolation and Boundary robustness instead of ranking them separately
        (which could point at two different models/suffixes). A candidate with either score
        below OVERALL_FLOOR is disqualified first, regardless of how good its other axis is -
        a model that can't be trusted at all on one axis isn't a safe overall pick just
        because it excels at the other. Only if NO candidate clears that floor does the floor
        get dropped entirely and every candidate is ranked by the normal Interpolation-primary
        cascade (_pick_best_model) - a model that's merely weak on both axes is still a more
        informative pick than refusing to choose. Classification has no Boundary robustness
        score at all, so the floor never applies there - callers route that case straight to
        _pick_best_model() instead of calling this.
        """
        survivors = [item for item in items
                     if item[1] is not None and round_checkml_score(item[1]) >= self.OVERALL_FLOOR
                     and item[2] is not None and round_checkml_score(item[2]) >= self.OVERALL_FLOOR]
        pool = survivors if survivors else items
        return self._pick_best_model(pool,1,2,3)


    def organize_all_models_pdfs(self):
        """
        Moves every --all_models PDF (one per model x PFI variant) into a REPORT_models/
        subfolder, then copies back into the working directory only the single overall-best
        PDF - picked across BOTH models AND PFI variants together (unlike best_models_text(),
        which picks per suffix) via _pick_best_overall(), so there's always exactly one PDF
        left in the working directory, never two
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

        # Boundary robustness isn't defined in classification, so Interpolation decides alone
        # (no floor to apply - there's nothing to disqualify a candidate on)
        if self.all_models_clas:
            best_pdf = self._pick_best_model(candidates,1,2,3)[0]
        else:
            best_pdf = self._pick_best_overall(candidates)[0]

        dest_dir = Path('REPORT_models')
        dest_dir.mkdir(exist_ok=True)
        for pdf_name,*_ in candidates:
            shutil.move(pdf_name, str(dest_dir / pdf_name))

        shutil.copy(str(dest_dir / best_pdf), best_pdf)

        print(f'\no  All {len(candidates)} model PDFs were moved to REPORT_models/')
        print(f'o  {best_pdf} (best overall model) was kept in the working directory')


    def print_header(self,citation_dat):
        """
        Retrieves the header for the HTML string
        """

        # CHECKML scores a model the user already chose (not one ROBERT's own Bayesian search
        # picked), so its reports use the CheckML logo instead of ROBERT's, to visually tell
        # the two apart at a glance
        logo_name = 'CheckML_logo.jpg' if getattr(self,'eval_only',False) else 'Robert_logo.jpg'

        # combines the top image with the other sections of the header
        header_lines = f"""
            <h1 style="text-align: center; margin-bottom: 0.5em;">
                <img src="file:///{self._posix_uri(self.args.path_icons)}/{logo_name}" alt="" style="display: block; margin-left: auto; margin-right: auto; width: 50%; margin-top: -12px;" />
                <span style="font-weight:bold;"></span>
            </h1>
            {citation_dat}
            """

        return header_lines


    @staticmethod
    def print_radar_subheadings():
        """Show the score breakdown heading above the retained interpolation radar."""

        return (
            '<p style="width:100%; margin:4px 0 2px 0; white-space:nowrap;">'
            '<span style="display:inline-block; width:270px; margin-left:0; text-decoration:underline;">'
            'Score breakdown (see <a href="#section-b">Section B, left</a>)</span>'
            '</p>'
        )


    def print_score(self,dat_files,pred_type,suffix,suffix_title):
        """
        Generates the CheckML score section (left = Interpolation, right = Boundary robustness)
        """

        # calculates the CheckML score (R2 is analogous for accuracy in classification)
        data_score = {}
        data_score = calc_score(dat_files,suffix,pred_type,data_score)

        # a test set picked by the user (CHECKML's 'Set' column) is never the systematic split
        # the score was calibrated for, even if it happens to have ~20% of the points (in which
        # case the size check in calc_score() alone would still give a score)
        path_eval_log = Path(f'{os.getcwd()}/CHECKML/CHECKML_data.dat')
        if path_eval_log.exists() and "Using the 'Set' column" in path_eval_log.read_text(encoding='utf-8'):
            data_score[f'score_available_{suffix}'] = False
            data_score[f'score_unavailable_reason_{suffix}'] = 'user_set'

        # the score's thresholds were calibrated assuming the standard test_set=0.2 split (see
        # calc_score()/get_predict_scores() for how score_available is derived, and the note in
        # score.rst) - with a non-standard internal split (including 0, i.e. no held-out test
        # set), the score isn't calibrated, so its bars/tier labels are replaced with grayed-out,
        # "N/A"-labeled versions instead of a misleading STRONG/WEAK claim (see score_gray_N.jpg/
        # score_w_2_gray_N.jpg below and in report_utils.py's score_badge_2()) - the underlying
        # numbers, images and warnings are still real and shown exactly as usual. --csv_test does
        # NOT affect this: it no longer suppresses the internal test split (see prepare_sets() in
        # utils.py) precisely so the score keeps working normally even when an external test set
        # is also provided. Defaults to available (fail open) if the underlying line wasn't found
        # for some reason, rather than hiding the score by mistake
        score_available = data_score.get(f'score_available_{suffix}', True)

        # starts with the icon of CheckML score - when the score isn't calibrated, the standing
        # intro paragraph is replaced (not just dropped) with the shorter "Score not available"
        # notice, in the exact same slot/spacing (see explain_override in module_lines()) so the
        # Interpolation/Boundary robustness columns right below keep the same tight spacing they
        # always have under that slot, instead of sitting oddly close under a separately-appended
        # paragraph
        explain_override = None
        if not score_available:
            reason = data_score.get(f'score_unavailable_reason_{suffix}')
            if reason == 'cv':
                cv_type = data_score.get(f'cv_type_{suffix}', '')
                reason_txt = f"this model used {cv_type} instead of the standard 10x repeated 5-fold CV the CheckML score was calibrated for. If you want a score, use the default --kfold 5 --repeat_kfolds 10"
            elif reason == 'user_set':
                reason_txt = "the test set was chosen by the user (Set column) instead of CheckML's systematic ~20% split the CheckML score was calibrated for. If you want a score, remove the Set column"
            else:
                n_test = data_score.get(f'n_test_{suffix}')
                reason_txt = f"this model's test set has {n_test} point(s), not the standard ~20% split the CheckML score was calibrated for. If you want a score, use the default --test_set 0.2"
            explain_override = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Score not available: {reason_txt} (see <a href="https://robert.readthedocs.io/en/latest/Report/score.html">the CheckML score documentation</a>).</i>'

        score_dat = ''
        score_dat = self.module_lines('score',score_dat,pred_type=pred_type,explain_override=explain_override)

        # Boundary robustness isn't defined for classification (no Low/High/applicability-
        # domain analysis exists for a discrete MCC-based model - see boundary_plot(), which
        # only runs for pred_type=='reg'), so only the Interpolation column is shown
        score_cols = ['interpolation'] if pred_type == 'clas' else ['interpolation','boundary']

        columns_score = []
        boundary_sections = ''
        if pred_type != 'clas':
            _,high_image = score_badge_2(
                self,data_score,suffix,data_score.get(f'sorted_cv_high_score_{suffix}',0)
            )
            boundary_sections = self.boundary_score_sections(high_image)
        else:
            boundary_sections = ''
        for col in score_cols:
            spacing = '' if col == 'interpolation' else self.spacing_PFI
            score_key = 'interp_score' if col == 'interpolation' else 'extrap_score'
            score_val = data_score.get(f'{score_key}_{suffix}', 0)
            # Only integer score_N.jpg icons exist. Use the same half-up integer tier for the
            # icon, assessment phrase, model-selection floor, and CheckML app display.
            score_icon = round_checkml_score(score_val)
            icon_name = f'score_{score_icon}' if score_available else f'score_gray_{score_icon}'
            # The original half-page score image renders at 234.9pt = 313.2 CSS px.
            # Keep that fixed size while the right-column text wrapper has its 10px inset.
            score_image_width = '313.2px' if col == 'boundary' else '100%'
            score_info = f"""<img src="file:///{self._posix_uri(self.args.path_icons)}/{icon_name}.jpg" style="width: {score_image_width}; margin-top:7px; margin-bottom:-18px;"></p>"""
            columns_score.append(get_col_score(score_info,data_score,suffix,col,spacing))
        if pred_type == 'clas':
            # no Boundary robustness column (see comment above) - a short note instead of a
            # blank column, so it reads as intentionally disabled rather than missing content,
            # matching the wording used for the same reason in Section E (see module_lines())
            indent = f'padding-left: {BOUNDARY_INDENT};' if self.spacing_PFI else ''
            columns_score.append(f'''<div style="{indent} box-sizing: border-box;">
            <p style="margin-top:-18px;"><span style="font-weight:bold;">Boundary robustness</span></p>
            <p style="margin-top: -10px;"><i>Disabled in classification problems.</i></p>
            </div>''')

        # Combine both columns
        score_dat += combine_cols(columns_score,right_offset=25)

        # add corresponding images (left: Results, right: Results_boundary_high - clas has no
        # boundary image, see above, so the right side stays blank)
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
        # Keep the interpolation plot on the left and place the extrapolation details and
        # Section A callouts together in the right column.
        interp_tag,_ = self.get_score_image_tags(
            suffix_title,height,pred_type,interp_width=285
        )
        section_headings = combine_cols([
            '<p style="margin:0; text-decoration:underline;">Model metrics</p>',
            f'<p style="box-sizing:border-box; padding-left:{BOUNDARY_INDENT}; margin:0; text-decoration:underline;">'
            'Score breakdown (see <a href="#section-b">Section B, right</a>)</p>',
        ],right_offset=25)
        metrics_dat = (
            f'<div style="margin-bottom:12px;">{section_headings}</div>'
            + self.print_img_row_indent(
                interp_tag,boundary_sections,3,right_offset=25,left_width=285
            )
        )

        # metrics text row beneath the images
        module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        interp_radar,_ = self.save_radar_plots(data_score,suffix,pred_type)
        interp_radar_tag = (
            f'<img src="file:///{self._posix_uri(interp_radar)}" '
            'style="display:block; flex:none; width:270px; height:auto; max-width:none; margin:0;"/>'
            if interp_radar else ''
        )
        left_summary = f'''<div style="margin-left:0;">{get_metrics(module_file,suffix,'')}</div>
        <div style="page-break-inside:avoid; margin-top:23px;">
        {self.print_radar_subheadings()}
        <div style="width:270px; margin-top:17px;">{interp_radar_tag}</div>
        </div>'''
        columns_summary = [left_summary]
        columns_summary.append(self.section_a_callouts())
        metrics_dat += combine_cols(columns_summary,right_offset=25)

        return score_dat,metrics_dat,data_score


    def boundary_score_sections(self,high_image):
        """Build the three temporary title/badge rows for the first-page boundary panel."""

        icon_dir = f'file:///{self._posix_uri(self.args.path_icons)}'
        icon_style = 'width:25px; height:25px; margin-right:13px; vertical-align:middle;'
        image = f'<img src="file:///{high_image}" alt="CheckML score" style="width: 13%; vertical-align: middle;">'
        first_row_style = 'display:block; margin:0 0 14px 0;'
        next_row_style = 'display:block; padding-top:5px; margin:0 0 14px 0;'
        text_style = 'display:inline-block; width:232px; vertical-align:middle;'
        first_text_style = f'{text_style} margin-top:3px;'
        score_line_style = 'display:block; margin-top:4px;'
        rows = (
            f'<span style="{first_row_style}"><img src="{icon_dir}/robust_y_high.png" alt="" style="{icon_style} margin-top:3px;">'
            f'<span style="{first_text_style}">1. Extrapolation to higher y values'
            f'<span style="{score_line_style}">9.9 / 9.9&nbsp&nbsp;{image}</span></span></span>'
        )
        rows += (
            f'<span style="{next_row_style}"><img src="{icon_dir}/robust_y_low.png" alt="" style="{icon_style}">'
            f'<span style="{text_style}">2. Extrapolation to lower y values'
            f'<span style="{score_line_style}">9.9 / 9.9&nbsp&nbsp;{image}</span></span></span>'
        )
        rows += (
            f'<span style="{next_row_style}"><img src="{icon_dir}/robust_x.png" alt="" style="{icon_style}">'
            f'<span style="{text_style}">3. Descriptor-space extrapolation'
            f'<span style="{score_line_style}">9.9 / 9.9&nbsp&nbsp;{image}</span></span></span>'
        )

        # Offset the evaluations visually past the overlay rule. Relative positioning keeps
        # their original layout width and leaves the two-column row geometry unchanged.
        return f'<span style="position:relative; left:4px; display:inline-block; width:270px; vertical-align:top;">{rows}</span>'


    @staticmethod
    def section_a_callouts():
        """Build the explanatory callouts shown beneath the right-column score details."""

        def card(title,content,margin_top=8):
            return f'''<div style="box-sizing:border-box; width:100%; background:#eeeeee; border-left:4px solid #999999; border-radius:4px; padding:4px 8px 5px 8px; margin:{margin_top}px 0 5px 0; color:#222222; page-break-inside:avoid;">
<div style="font-size:11px; font-weight:bold; letter-spacing:0.3px; color:#000000;">{title}</div>
<div style="font-size:11px; line-height:1.5;">{content}</div></div>'''

        influence = 'Separating a 20% test set changes the scaled RMSE by only XX% compared to the full dataset. CheckML scores seem to capture the original model\'s performance. <a href="https://robert.readthedocs.io/en/latest/Report/score.html">More details here.</a>'
        disclaimer = (
            '<u>User-defined train/test splits are not allowed</u>'
            '<br>No single train/test split, performance metric or validation strategy can fully establish the reliability of a model. CheckML addresses this limitation through a standardized, multidimensional assessment that shows potential weaknesses and makes model evaluations easier and transparent.'
            '<br><u>CheckML scores are indicative</u>'
            '<br>CheckML scores are indicative rather than rigid thresholds of model quality or usefulness, which ultimately depend on the intended application. Before using CheckML, please read <a href="https://robert.readthedocs.io/en/latest/Report/score.html">its documentation</a> carefully, especially the <i>Considerations</i> section.'
        )
        return (
            f'<div style="position:relative; top:-5px; width:300px; margin-left:-10px; margin-top:-34px;">'
            f'{card("Influence of CV\u2013Test Split",influence,3)}'
            f'{card("Considerations when using CheckML",disclaimer,18)}'
            '</div>'
        )


    def get_score_image_tags(self,suffix_title,height,pred_type,interp_width=270):
        """
        Locates the interpolation (Results_*.png) and, for regression, boundary robustness
        (Results_boundary_high_*.png) images for this model and returns them as ready-to-embed
        <img> tags (right tag is '' for classification, where that analysis isn't defined - see
        print_score())
        """

        module_path = Path(f'{os.getcwd()}/PREDICT')

        # glob (not rglob): avoids picking up files from the PREDICT/csv_test subfolder
        interp_images = [str(fp) for fp in module_path.glob('Results_*.png')
                          if self.matches_suffix(str(fp),suffix_title) and 'Results_boundary' not in str(fp)]

        interp_images = self._filter_by_model(interp_images)
        interp_image = interp_images[0] if interp_images else ''
        interp_height = round(height * interp_width / 270)
        left_tag = self.print_img_tag(interp_image,interp_height,width=interp_width)

        if pred_type == 'clas':
            return left_tag,''

        # top 20% (High) plot only - the low/ad/williams variants have their own filenames now
        # (Results_boundary_low_..., Results_boundary_ad_..., Results_boundary_williams_...)
        bound_images = [str(fp) for fp in module_path.glob('Results_boundary_high_*.png')
                          if self.matches_suffix(str(fp),suffix_title)]
        bound_images = self._filter_by_model(bound_images)
        bound_image = bound_images[0] if bound_images else ''
        right_tag = self.print_img_tag(bound_image,height)

        return left_tag,right_tag


    def print_score_images(self,suffix_title,height,pred_type):
        """
        Places the interpolation image for this model - alongside the boundary robustness image
        for regression (see get_score_image_tags())
        """

        left_tag,right_tag = self.get_score_image_tags(suffix_title,height,pred_type)

        # no boundary-robustness image for classification - keep the original two-column row
        # layout/proportions with the right side blank
        return self.print_img_row_indent(left_tag,right_tag,-5)


    def print_warnings(self,dat_files,pred_type,data_score,suffix):
        """
        Analyzes warnings for the page-2 summary and returns the retained interpolation radar.
        """

        # load spacing, colors, and line and table formats
        space,color_dict,style_lines,warnings_dat = self.get_warning_params()

        # gather the lines from PREDICT (and VERIFY, for the cluster-test check) where the
        # potential warnings are printed
        warnings_dict = self.get_warning_lines(pred_type,dat_files.get('VERIFY',[]))

        warnings_dict[f'severe_warnings_{suffix}'] = []
        warnings_dict[f'moderate_warnings_{suffix}'] = []

        # analyze and append warnings, already in priority order (most decision-critical first
        # - see analyze_warnings() for the ordering rationale)
        warnings_dict = self.analyze_warnings(data_score,suffix,warnings_dict,pred_type)

        # Every warning is shown in the results summary on page 2 (see print_summary_page()).

        # a big, impossible-to-miss banner (separate from the bulleted severe-warnings list
        # below, which uses a small 15px bullet) for the one risk ROBERT cannot itself measure
        # or correct: hyperparameters supplied by the user (CHECKML's model_obj/model_file/
        # model_params) rather than searched by ROBERT's own BO within a held-out split. If
        # those hyperparameters were chosen using a process that already saw data ROBERT now
        # scores against (CV folds or the test set), both scores are optimistic to an unknown
        # degree - this can't be detected from the CSV alone, so it's surfaced instead of silently
        # trusted. Doesn't apply to the plain 'MVL' default, which has nothing to tune
        #
        # disabled on request - commented out, not removed, so it's easy to bring back
        # if getattr(self,'custom_model_used',False):
        #     warning_print += f'''
        # <div style="width:100%; box-sizing:border-box; background-color:{color_dict['red']}; padding:8px; margin-top:2px; margin-bottom:10px; text-align:center;">
        # <span style="font-size:17px; font-weight:bold; color:white;">&#9888; SCORES MAY BE OPTIMISTIC: user-provided hyperparameters</span><br>
        # <span style="font-size:11.5px; color:white;">This model's hyperparameters came from the user (CHECKML), not from ROBERT's own search within a held-out split. If they were tuned using a process that already saw this data (or part of it), <strong>both the CV and Test scores below may be optimistic</strong> - ROBERT cannot detect or correct for this. Only trust these results if the hyperparameters were chosen on data fully independent from this dataset.</span>
        # </div>'''

        return '',warnings_dict


    # displayed size (px) of each radar plot on page 1 - 270px wide like every other report image
    RADAR_SIZE = (270, 215)

    def save_radar_plots(self,data_score,suffix,pred_type):
        '''
        Saves the radar plots of the 0-2 components behind the Interpolation and Boundary
        robustness scores (same values as Section B). The flawed-model tests are a penalty
        (<= 0) rather than a 0-2 component, so they're summarized in text instead (see
        print_warnings()). Classification has no Boundary robustness, so only one radar is made.
        Returns (interpolation_png, boundary_png or '')
        '''

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        diff_key = 'diff_scaled_rmse_score' if pred_type == 'reg' else 'diff_mcc_score'
        radars = [('interpolation', f'interp_score_{suffix}', [
            # short names (numbered as in Section B) so the labels don't crowd the plot
            ('1. Flawed models', f'flawed_mod_score_{suffix}'),
            ('2. CV', f'cv_score_combined_{suffix}'),
            ('3. Test', f'test_score_combined_{suffix}'),
            ('4. Train-val gap', f'train_val_gap_score_{suffix}'),
            ('5. CV vs test', f'{diff_key}_{suffix}'),
            ('6. Stability', f'cv_sd_score_{suffix}'),
        ])]
        if pred_type == 'reg':
            radars.append(('boundary', f'extrap_score_{suffix}', [
                ('1. High (sorted CV)', f'sorted_cv_high_score_{suffix}'),
                ('2. Low (sorted CV)', f'sorted_cv_low_score_{suffix}'),
                ('3. Spearman', f'spearman_score_{suffix}'),
                ('4. Degradation', f'degradation_score_{suffix}'),
                ('5. Applic. domain', f'applicability_domain_score_{suffix}'),
            ]))

        score_available = data_score.get(f'score_available_{suffix}', True)
        out_folder = Path(f'{os.getcwd()}/PREDICT')
        out_folder.mkdir(exist_ok=True)
        suffix_title = '_'.join(suffix.split())
        paths = {'interpolation': '', 'boundary': ''}
        label_font_size = 8.928  # another 20% larger than the previous 7.44 pt
        value_font_size = 7.2    # another 20% larger than the previous 6 pt

        for key,total_key,components in radars:
            labels = [label for label,_ in components]
            raw_values = [float(data_score.get(score_key,0) or 0) for _,score_key in components]
            flawed_score_key = f'flawed_mod_score_{suffix}'
            # The flawed-model component can be negative. Keep its bar pinned at zero,
            # while retaining the raw score for the value badge.
            values = [max(0.0,min(2.0,value)) for value in raw_values]
            display_values = [
                raw_value if score_key == flawed_score_key else value
                for (_,score_key),raw_value,value in zip(components,raw_values,values)
            ]
            # each bar coloured by its own 0-2 value with the score-card palette: red (<0.5),
            # orange (0.5-1), light blue (1-1.5), blue (>=1.5); grey when the score isn't
            # calibrated for this run (non-standard CV/test, see print_score())
            # flawed tests are colored differently: red (<=-1.5), orange (-1.5 to -1), 
            # light blue (-1 to -0.5), blue (>-0.5)
            from robert.checkml_score import checkml_tier_color
            def bar_colour(value, is_flawed=False):
                if not score_available:
                    return '#9e9e9e'
                if is_flawed:
                    tier = (
                        'VERY STRONG' if value > -0.5 else
                        'STRONG' if value > -1 else
                        'WEAK' if value > -1.5 else
                        'VERY WEAK'
                    )
                else:
                    tier = 'VERY WEAK' if value < 0.5 else 'WEAK' if value < 1 else 'STRONG' if value < 1.5 else 'VERY STRONG'
                return checkml_tier_color(tier,'hex')

            n = len(values)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            slot = 2*np.pi/n
            # figure inches = displayed px / 96, so font sizes in pt read as in the rest of the PDF
            fig = plt.figure(figsize=(self.RADAR_SIZE[0]/96, self.RADAR_SIZE[1]/96), dpi=300)
            # Use most of the original 270 x 215 canvas while leaving enough room for the
            # labels around the outer ring. Keeping the canvas size unchanged preserves the
            # PDF image aspect ratio and prevents object-fit from cropping the radar.
            ax = fig.add_axes([0.07, 0.08, 0.86, 0.82], polar=True)
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            ax.set_facecolor('#fbfcfe')
            ax.set_ylim(0, 2.1)
            grid_color = '#b8c1cc'
            # Quiet reference rings keep attention on the bars and their value badges.
            ax.set_axisbelow(True)
            ax.set_yticks([0.5, 1, 1.5, 2.0])
            ax.set_yticklabels([])
            ax.yaxis.grid(True, color=grid_color, linewidth=0.45, linestyle='-')
            ax.set_xticks(angles)
            ax.set_xticklabels([])
            ax.xaxis.grid(False)
            # Black outlines separate adjacent bars and end at each bar's tip.
            ax.bar(angles, values, width=slot*0.62,
                   color=[bar_colour(display_value, score_key == flawed_score_key)
                          for (_,score_key),display_value in zip(components,display_values)],
                   edgecolor='black', linewidth=0.5, zorder=2.5)
            # Radial dividers retain the original pizza-slice separation between sections.
            for edge in angles + slot/2:
                ax.plot([edge, edge], [0, 2.1], color=grid_color, linewidth=0.45, zorder=3)
            # Small outward ticks align each section's outer edge with its label.
            for angle in angles:
                ax.plot([angle, angle], [2.1, 2.18], color='black', linewidth=0.55,
                        zorder=3.2, clip_on=False)
            ax.spines['polar'].set_color('black')
            ax.spines['polar'].set_linewidth(0.65)
            ax.spines['polar'].set_zorder(2)
            # Value badges use the same colored fill as their bars.
            for angle,value,display_value,(_,score_key) in zip(angles,values,display_values,components):
                color = bar_colour(display_value, score_key == flawed_score_key)
                ax.text(angle, max(value - 0.18, 0.22), f'{display_value:.1f}', ha='center', va='center',
                        fontsize=value_font_size, color='#0f172a', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                                  edgecolor='black', linewidth=0.4))
            # component names around the circle, rotated along it (flipped on the bottom half
            # so they never read upside down); place each character along the arc so the
            # full label follows the circular edge instead of sitting on a straight baseline.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            from matplotlib.font_manager import FontProperties
            label_font = FontProperties(size=label_font_size)
            label_radius = 2.38
            radius_px = ax.get_window_extent().width * 0.5 * label_radius / 2
            for label,angle in zip(labels,angles):
                char_widths = [
                    renderer.get_text_width_height_descent(char, label_font, False)[0]
                    for char in label
                ]
                arc_width = sum(char_widths) + max(len(label) - 1, 0) * 0.7
                offset = -arc_width / 2
                normalized_angle = angle % (2*np.pi)
                flip = np.pi/2 < normalized_angle < 3*np.pi/2
                direction = -1 if flip else 1
                for char,char_width in zip(label,char_widths):
                    char_angle = angle + direction * (offset + char_width / 2) / radius_px
                    rotation = -np.degrees(char_angle) + (180 if flip else 0)
                    ax.text(char_angle, label_radius, char, ha='center', va='center',
                            rotation=rotation, rotation_mode='anchor', fontsize=label_font_size,
                            color='#1e293b')
                    offset += char_width + 0.7
            radar_path = out_folder / f'CheckML_radar_{key}{getattr(self,"model_suffix","")}_{suffix_title}.png'
            # Keep the original canvas ratio so the PDF's fixed-size image box cannot crop
            # the labels. The title is omitted so only the radar and component labels show.
            fig.savefig(radar_path, dpi=300)
            plt.close(fig)
            paths[key] = str(radar_path)

        return paths['interpolation'],paths['boundary']


    SUMMARY_MODEL_NAMES = {
        'RF': 'RandomForest', 'GB': 'GradientBoosting', 'NN': 'MLP',
        'GP': 'GaussianProcess', 'ADAB': 'AdaBoost',
    }
    # report palette used by the summary page (same colours as the VERIFY bar chart and the
    # warning bullets used elsewhere in the report)
    SUMMARY_BLUE = '#1f77b4'
    SUMMARY_WARNING_STYLES = {
        # kind: (stripe colour, background tint, label colour)
        'severe': ('#c56666', '#f7e6e6', '#9e3a3a'),
        'moderate': ('#c5c57d', '#f5f5e6', '#6f6f2c'),
        'none': ('#9ba5e3', '#eceefb', '#4a5596'),
    }

    def _summary_score_style(self,score):
        '''
        Fill colour, text colour and rating of a 0-10 score - the same five tiers and colours
        as the score bars of Section A (see checkml_score.py)
        '''
        from robert.checkml_score import checkml_score_tier, checkml_tier_color
        if score is None:
            return '#9e9e9e','#666666','Not available'
        tier = checkml_score_tier(max(0,min(10,score)))
        return checkml_tier_color(tier,'hex'),checkml_tier_color(tier,'text_hex'),tier.title()


    def _summary_model_label(self,params_df,pred_type):
        '''Full scikit-learn name of the model (ROBERT short codes translated)'''
        model = str(params_df['model'][0]) if 'model' in params_df.columns else 'Not available'
        if model.upper() == 'MVL':
            return 'LinearRegression (MVL)'
        if model.upper() in ('RIDGE','LOGISTIC'):
            return {'RIDGE': 'Ridge', 'LOGISTIC': 'LogisticRegression'}[model.upper()]
        if model.upper() in self.SUMMARY_MODEL_NAMES:
            model_type = 'Regressor' if pred_type == 'reg' else 'Classifier'
            return f'{self.SUMMARY_MODEL_NAMES[model.upper()]}{model_type}'
        return model


    def print_summary_page(self,pred_type,data_score,warnings_dict,suffix,params_df):
        '''
        Results summary page (page 2): same content as the CheckML web summary sheet - evaluation
        setup, target distribution, hyperparameters, both scores and EVERY
        warning (the Section A box on page 1 may truncate moderate warnings with "..."), with
        each warning placed under the score it describes (Interpolation / Boundary robustness)
        '''

        from html import escape
        from robert.utils import deserialize_model_params

        navy, muted = self.SUMMARY_BLUE, '#555555'
        y_name = str(params_df['y'][0]) if 'y' in params_df.columns else ''
        names_col = str(params_df['names'][0]) if 'names' in params_df.columns else ''

        # dataset information (from the database saved with the model, which has the Set column)
        dataset = pd.DataFrame()
        db_file = getattr(self,'report_db_file',None)
        if db_file and os.path.exists(db_file):
            try:
                dataset = pd.read_csv(db_file, encoding='utf-8')
            except Exception:
                dataset = pd.DataFrame()
        descriptors = []
        if 'X_descriptors' in params_df.columns:
            try:
                descriptors = json.loads(params_df['X_descriptors'][0])
            except (json.JSONDecodeError, TypeError):
                import ast
                try:
                    descriptors = ast.literal_eval(params_df['X_descriptors'][0])
                except (ValueError, SyntaxError):
                    descriptors = []
        set_txt = ''
        if 'Set' in dataset.columns:
            set_counts = dataset['Set'].value_counts()
            set_txt = ' | '.join(f'{name}: {count:,}' for name,count in set_counts.items())

        # target distribution (classification labels restored with the saved class mapping)
        distribution = 'No target values available'
        if y_name in dataset.columns:
            if pred_type == 'clas':
                class_map = {}
                if 'class_mapping_reverse' in params_df.columns and pd.notna(params_df['class_mapping_reverse'][0]):
                    try:
                        class_map = {int(k): v for k,v in json.loads(params_df['class_mapping_reverse'][0]).items()}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        class_map = {}
                counts = dataset[y_name].map(lambda v: class_map.get(int(v), v) if pd.notna(v) else v).value_counts()
                distribution = '; '.join(f'{label}: {count:,}' for label,count in counts.items())
            else:
                values = pd.to_numeric(dataset[y_name], errors='coerce').dropna()
                if not values.empty:
                    distribution = (f'n = {len(values):,}; range = {values.min():.4g} to {values.max():.4g}; '
                                    f'mean = {values.mean():.4g}; SD = {values.std(ddof=0):.4g}')

        # hyperparameters actually used by the model
        hyper_rows = ''
        if 'params' in params_df.columns:
            try:
                hyperparams = deserialize_model_params(params_df['params'][0])
            except Exception:
                hyperparams = {}
            if isinstance(hyperparams, dict):
                for name,value in hyperparams.items():
                    value_txt = str(value)
                    if len(value_txt) > 90:
                        value_txt = value_txt[:87] + '...'
                    hyper_rows += f'<tr><td>{escape(str(name))}</td><td>{escape(value_txt)}</td></tr>'

        # no written verdict ("decent model", "weak model"...): it flipped on tiny score changes
        # (e.g. 6.9 vs 7.0), so the page relies on the scores and the warnings below instead
        severe_list = warnings_dict[f'severe_warnings_{suffix}']
        moderate_list = warnings_dict[f'moderate_warnings_{suffix}']

        # score cards (same tier colours as Section A's score bars)
        def score_card(title,score,disabled=False):
            if disabled:
                return f'''<td style="width:50%; border:2px solid #b0b0b0; border-radius:9px; background:white; text-align:center; padding:10px 6px;">
<div style="font-weight:bold; color:#222222; font-size:13px;">{title}</div>
<div style="font-weight:bold; color:{muted}; font-size:12px; margin:22px 0;">Disabled for classification</div></td>'''
            fill,text_col,rating = self._summary_score_style(score)
            score_txt = f'{score:.1f}' if score is not None else '-'
            return f'''<td style="width:50%; border:2px solid {fill}; border-radius:9px; background:{fill}22; text-align:center; padding:8px 6px;">
<div style="font-weight:bold; color:#222222; font-size:13px;">{title}</div>
<div style="font-weight:bold; color:{text_col}; font-size:30px; line-height:1.2;">{score_txt}</div>
<div style="font-size:9px; color:{muted};">out of 10</div>
<div style="font-weight:bold; color:{text_col}; font-size:11px; letter-spacing:0.5px;">{rating.upper()}</div></td>'''

        interp_score = data_score.get(f'interp_score_{suffix}')
        extrap_score = data_score.get(f'extrap_score_{suffix}')
        cards = score_card('Interpolation',interp_score)
        cards += '<td style="width:12px; border:none;"></td>'
        cards += score_card('Boundary robustness<br><span style="font-size:10px; font-weight:normal;">(related to extrapolation)</span>',
                            extrap_score,disabled=pred_type != 'reg')

        # warnings, each one under the score it describes (all of them, never truncated)
        def warning_area(warning):
            text = warning.lower()
            is_boundary = 'boundary robustness' in text or 'extrapolat' in text
            # a warning about both scores (i.e. "...in Interpolation and Boundary robustness")
            # goes in both columns
            if is_boundary and 'interpolation' in text:
                return ['interpolation','boundary']
            return ['boundary'] if is_boundary else ['interpolation']

        groups = {'interpolation': [], 'boundary': []}
        for warning in severe_list:
            for area in warning_area(warning):
                groups[area].append(('Severe warning',warning,'severe'))
        for warning in moderate_list:
            for area in warning_area(warning):
                groups[area].append(('Moderate warning',warning,'moderate'))

        def warning_cards(items,empty_txt):
            if not items:
                stripe,bg,label_col = self.SUMMARY_WARNING_STYLES['none']
                return f'''<div style="background:{bg}; border-left:4px solid {stripe}; border-radius:4px; padding:6px 8px; font-size:10px; color:{label_col};">{empty_txt}</div>'''
            cards_html = ''
            for kind,warning,style_key in items:
                stripe,bg,label_col = self.SUMMARY_WARNING_STYLES[style_key]
                # the "(Section X)"/"(Boundary robustness)" pointers are kept: they tell the
                # reader where in the report each warning is explained
                cards_html += f'''<div style="background:{bg}; border-left:4px solid {stripe}; border-radius:4px; padding:4px 8px 5px 8px; margin-bottom:5px; color:#222222; page-break-inside:avoid;">
<div style="font-size:8px; font-weight:bold; letter-spacing:0.3px; color:{label_col};">{kind.upper()}</div>
<div style="font-size:10px; line-height:1.3;">{escape(warning)}</div></div>'''
            return cards_html

        interp_col_html = warning_cards(groups['interpolation'],'No interpolation warnings detected.')
        if pred_type == 'reg':
            bound_col_html = warning_cards(groups['boundary'],'No boundary robustness warnings detected.')
        else:
            bound_col_html = f'<div style="font-size:10px; color:{muted};">Boundary robustness is not evaluated for classification.</div>'

        section_title = lambda txt: f'<p style="font-weight:bold; color:{navy}; font-size:14px; margin:12px 0 5px 0;">{txt}</p>'
        problem_txt = 'Regression' if pred_type == 'reg' else 'Classification'

        # model hyperparameters, right after the evaluation setup they belong to
        hyper_html = ''
        if hyper_rows:
            hyper_html = f'''
{section_title('Model hyperparameters')}
<table style="width:100%; border-collapse:collapse; font-size:9.5px;">
<tr style="background:{navy}; color:white;"><th style="text-align:left; padding:3px 6px; width:40%;">Parameter</th><th style="text-align:left; padding:3px 6px;">Value</th></tr>
{hyper_rows.replace('<td>', '<td style="border:1px solid #c8c8c8; padding:2px 6px;">')}
</table>'''
        dataset_name = os.path.basename(str(getattr(self.args,'csv_name','')))

        summary = f'''
<div style="page-break-inside:avoid;">
<div style="background:{navy}; color:white; border-radius:9px; padding:10px 14px; margin-top:4px;">
<div style="font-size:19px; font-weight:bold;">Results summary</div>
<div style="font-size:10px;">All scores and warnings of this report at a glance. Details of each item are in the sections that follow.</div>
</div>

{section_title('Evaluation setup')}
<div style="background:#eef4fa; border-left:4px solid {navy}; border-radius:4px; padding:7px 14px; font-size:11px; line-height:1.6;">
<b>Dataset:</b> {escape(dataset_name) or 'Not available'}<br>
<b>Datapoints:</b> {len(dataset):,}{f' ({escape(set_txt)})' if set_txt else ''} | <b>Descriptors:</b> {len(descriptors):,}<br>
<b>Target:</b> {escape(y_name) or 'Not available'} | <b>Name / ID:</b> {escape(names_col) or 'Not available'} | <b>Problem type:</b> {problem_txt}<br>
<b>Model:</b> {escape(self._summary_model_label(params_df,pred_type))}
</div>
{hyper_html}
{section_title('Target distribution')}
<div style="border:1px solid #c8c8c8; border-radius:4px; padding:5px 10px; font-size:11px; color:{muted};">{escape(distribution)}</div>

{section_title('Scores')}
<table style="width:100%; border-collapse:separate; border-spacing:0;"><tr>{cards}</tr></table>

{section_title('Warnings')}
<table style="width:100%; border-collapse:collapse;"><tr>
<td style="width:50%; vertical-align:top; padding-right:6px;">
<div style="font-weight:bold; color:{navy}; font-size:11px; margin-bottom:4px;">Interpolation warnings</div>{interp_col_html}</td>
<td style="width:50%; vertical-align:top; padding-left:6px;">
<div style="font-weight:bold; color:{navy}; font-size:11px; margin-bottom:4px;">Boundary robustness warnings</div>{bound_col_html}</td>
</tr></table>
</div>'''

        summary += '<p style="page-break-after: always;"></p>'
        return summary


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
        # explicit, equal, small top/bottom margins (rather than relying on the browser's
        # default ~12px <p> margin collapsing against a negative margin-bottom) so consecutive
        # warning lines pack predictably tight, instead of depending on margin-collapsing
        # arithmetic that's easy to get wrong when tuning spacing
        style_lines = '<p style="font-size: 11px; margin: 2px 0;">'

        warnings_dat = ''

        return space,color_dict,style_lines,warnings_dat


    def analyze_warnings(self,data_score,suffix,warnings_dict,pred_type):
        '''
        Analyze and append warnings. Appended in priority order (most decision-critical for an
        inexperienced user first) - all of them are shown in the Section A box (see
        print_warnings()), but the order still matters for how a reader scans the list.
        '''

        severe = warnings_dict[f'severe_warnings_{suffix}']
        moderate = warnings_dict[f'moderate_warnings_{suffix}']

        # hyperparameters given as the integer 1 where scikit-learn also accepts the fraction 1.0
        # (one sample/feature instead of all of them) - a likely input error that can make the
        # predictions nearly constant, so it's listed first (see validate_params() in checkml.py)
        for param in warnings_dict.get('ambiguous_params', []):
            moderate.append(f'{param} = 1 (integer) means ONE sample/feature in scikit-learn, use 1.0 if all of them were intended (Section H)')

        # tests from flawed models
        if data_score[f'flawed_mod_score_{suffix}'] < 0:
            if data_score[f'failed_tests_{suffix}'] > 0:
                severe.append('Failing required tests (Section B.1)')
            else:
                moderate.append('Some tests are unclear or close to the thresholds (Section B.1)')

        # train-vs-validation and CV-vs-test gaps at 0/2 both point to the same underlying
        # problem an inexperienced user is unlikely to recognize from the number alone: the
        # model looks fine on the data it was fit to, but that doesn't carry over - i.e.
        # overfitting. Severe (not moderate) because this directly undermines the model's main
        # purpose (predicting new points), regardless of how good the raw CV/test numbers look
        if data_score.get(f'train_val_gap_score_{suffix}', 1) == 0:
            severe.append('Overfitting observed between train and validation (Section B.4)')
        if pred_type == 'reg':
            diff_test_zero = data_score.get(f'diff_scaled_rmse_score_{suffix}', 1) == 0
        else:
            diff_test_zero = data_score.get(f'diff_mcc_score_{suffix}', 1) == 0
        if diff_test_zero:
            severe.append('Inconsistent results between CV and test (Section B.5)')

        # variation in CV (existing - unstable predictions across repeated CV runs)
        if pred_type == 'reg':
            if data_score[f'cv_sd_score_{suffix}'] <= 0: # due to the penalty, it might get lowwer than 0 (but in the final score it only adds 0-2 points)
                moderate.append('Imprecise predictions (Section B.6)')
        elif pred_type == 'clas':
            if data_score[f'diff_mcc_score_{suffix}'] == 0:
                moderate.append('Imprecise predictions (Section B.6)')

        # weak CV / test-set predictions on their own (0/2, Sections B.2/B.3) - distinct from
        # the overfitting checks above, which compare two numbers against each other; these
        # flag a single score that's just low
        if data_score.get(f'cv_score_combined_{suffix}', 1) == 0:
            moderate.append('Weak cross-validation predictions (Section B.2)')
        if data_score.get(f'test_score_combined_{suffix}', 1) == 0:
            moderate.append('Weak test-set predictions (Section B.3)')

        # low-relevance descriptor found in PFI and/or SHAP - suggests the model might improve
        # by trimming near-irrelevant descriptors rather than keeping them all (see
        # LOW_RELEVANCE_RELATIVE_THRESHOLD/LOW_RELEVANCE_ABSOLUTE_THRESHOLD above)
        if warnings_dict.get(f'low_relevance_descp_{suffix}', False):
            moderate.append('Low-relevance descriptor(s) detected, try removing them (Section D)')

        # bimodal y-values and/or a failed VERIFY cluster test both hint the response might
        # really be two distinct groups rather than one continuous variable - classification
        # could describe the data better than regression here (regression only: the cluster
        # test is N/A for classification, and "bimodal" only means something for a continuous y)
        if pred_type == 'reg' and (warnings_dict.get(f'bimodal_y_{suffix}', False) or warnings_dict.get(f'cluster_test_failed_{suffix}', False)):
            moderate.append('Response may have two separate groups, classification might fit better (Section C)')

        # Boundary robustness / extrapolation (regression only - not defined for classification)
        if pred_type == 'reg':
            # shown first in the Boundary robustness group and regardless of the boundary
            # metrics: a tree-based model can score well on one split's edges and still be
            # structurally unable to predict y values outside the training range
            # tree-based and nearest-neighbour models (see extrapolation_limitation())
            from robert.report_utils import extrapolation_limitation
            limitation = extrapolation_limitation(getattr(self, 'report_model_name', ''))
            if limitation:
                moderate.append(f'{limitation[0]} algorithm, not designed to extrapolate outside the training range (Boundary robustness)')

            if data_score.get(f'applicability_domain_score_{suffix}', 1) == 0:
                moderate.append('Poor extrapolation to descriptors very different from training (Boundary robustness)')

            high_zero = data_score.get(f'sorted_cv_high_score_{suffix}', 1) == 0
            low_zero = data_score.get(f'sorted_cv_low_score_{suffix}', 1) == 0
            if high_zero or low_zero:
                if high_zero and low_zero:
                    zone_txt = 'the upper and lower zones'
                elif high_zero:
                    zone_txt = 'the upper zone'
                else:
                    zone_txt = 'the lower zone'
                moderate.append(f'Difficult to extrapolate new predictions in {zone_txt} (Boundary robustness)')

            if data_score.get(f'degradation_score_{suffix}', 1) == 0:
                moderate.append("Extrapolation error may be much larger than the model's own error (Boundary robustness)")

            spearman_low = data_score.get(f'spearman_low_{suffix}')
            spearman_high = data_score.get(f'spearman_high_{suffix}')
            if spearman_low is not None and spearman_high is not None:
                low_fail = spearman_low < 0.5
                high_fail = spearman_high < 0.5
                if low_fail or high_fail:
                    if low_fail and high_fail:
                        zone_txt = 'the upper and lower zones'
                    elif high_fail:
                        zone_txt = 'the upper zone'
                    else:
                        zone_txt = 'the lower zone'
                    moderate.append(f'Difficult to prioritize new predictions (maximize/minimize) in {zone_txt} (Boundary robustness)')

        # simpler MVL model as good as this one: from Section F when several models were
        # screened (MVL included), otherwise from VERIFY's MVL baseline comparison (Section B.1)
        # (classification: logistic regression, ROBERT's LOGISTIC, instead of MVL)
        base_txt,base_code = ('MVL','MVL') if pred_type == 'reg' else ('logistic regression','LOGISTIC')
        mvl_warning = self._mvl_screening_warning(suffix,pred_type)
        if mvl_warning is None:
            mvl_status = warnings_dict.get(f'mvl_baseline_{suffix}')
            if mvl_status == 'similar':
                mvl_warning = f'A simpler {base_txt} model performs similarly (Section B.1), consider trying {base_code} with the same descriptors'
            elif mvl_status == 'baseline_better':
                mvl_warning = f'A simpler {base_txt} model performs better (Section B.1), consider trying {base_code} with the same descriptors'
        if mvl_warning:
            moderate.append(mvl_warning)

        # y distribution (existing)
        if 'WARNING! Your data is not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                moderate.append('Uneven y distribution (Section C)')
            elif pred_type == 'clas': # it's severe in clasification
                severe.append('Very uneven class distribution (Section C)')
        elif 'WARNING! Your data is slightly not uniform' in warnings_dict[f'y_dist_info_{suffix}']:
            if pred_type == 'reg':
                moderate.append('Slightly uneven y distribution (Section C)')
            elif pred_type == 'clas':
                moderate.append('Uneven class distribution (Section C)')

        # feature correlation (existing)
        if 'WARNING! High correlations' in warnings_dict[f'pearson_info_{suffix}']:
            moderate.append('Highly correlated features (Section D)')
        elif 'WARNING! Noticeable correlations' in warnings_dict[f'pearson_info_{suffix}']:
            moderate.append('Moderately correlated features (Section D)')

        # outliers - existing check (one very extreme point, > 6.5 SD, ~99.9% CI) plus a new,
        # separate check for too many outliers overall (> 5% of the dataset), regardless of
        # how extreme any single one is
        if pred_type == 'reg':
            if warnings_dict[f'max_sd_{suffix}'] > 6.5:
                moderate.append('Potential "faulty" outliers (Section E)')
            if warnings_dict.get(f'outlier_pct_{suffix}', 0) > 5:
                moderate.append('More than 5% of points are outliers, check and explain them (Section E)')

        return warnings_dict


    # an MVL within this fraction of the chosen model's error in the model screening counts as
    # "similar" (same tolerance as VERIFY's MVL baseline comparison, thres_mvl_baseline)
    MVL_SCREENING_TOLERANCE = 0.15

    def _mvl_screening_warning(self,suffix,pred_type='reg'):
        '''
        When the linear baseline (MVL, or LOGISTIC for classification) was screened together
        with other models (Section F), compares it with the model of this report: with
        --all_models, through their Interpolation/Boundary robustness scores; otherwise,
        through GENERATE's combined error. Returns the warning text, '' when the baseline is
        clearly worse (no warning), or None when Section F has no baseline to compare with
        (then VERIFY's baseline comparison is used instead)
        '''

        from robert.verify import _is_linear_model, linear_baseline_code

        base_code,_ = linear_baseline_code(pred_type)
        base_txt = 'MVL' if pred_type == 'reg' else 'logistic regression'
        model_name = str(getattr(self, 'report_model_name', '') or '')
        if not model_name or _is_linear_model(model_name):
            return None

        # --all_models: compare the final CheckML scores
        scores = getattr(self, 'model_scores', {}).get(suffix, {})
        current = getattr(self, 'model_suffix', '')[1:] or model_name
        mvl_key = next((key for key in scores if str(key).upper() == base_code), None)
        if mvl_key is not None and current in scores:
            mvl_scores,cur_scores = scores[mvl_key],scores[current]
            better_in = []
            if mvl_scores[0] is not None and cur_scores[0] is not None and mvl_scores[0] >= cur_scores[0]:
                better_in.append('Interpolation')
            # Boundary robustness only exists in regression (it's 0 for every classifier)
            if (pred_type == 'reg' and mvl_scores[1] is not None and cur_scores[1] is not None
                    and mvl_scores[1] >= cur_scores[1]):
                better_in.append('Boundary robustness')
            if not better_in:
                return ''
            return f'A simpler {base_txt} model scores as high or higher in {" and ".join(better_in)} (Section F)'

        # default mode: GENERATE's combined error of each screened model
        raw_dir = Path(f'{os.getcwd()}/GENERATE/Raw_data/{"_".join(suffix.split())}')
        error_type = str(getattr(self.args, 'error_type', 'rmse')).lower()
        model_errors = {}
        for csv_file in glob.glob(str(raw_dir / '*.csv')):
            if '_db' in os.path.basename(csv_file):
                continue
            try:
                params_df = pd.read_csv(csv_file, encoding='utf-8')
                model_errors[str(params_df['model'][0]).upper()] = float(params_df[f'combined_{error_type}'][0])
            except (KeyError, IndexError, ValueError, pd.errors.EmptyDataError):
                continue
        if base_code not in model_errors or model_name.upper() not in model_errors:
            return None
        mvl_err,cur_err = model_errors[base_code],model_errors[model_name.upper()]
        if error_type in ['mae','rmse']:
            similar_or_better = mvl_err <= (1+self.MVL_SCREENING_TOLERANCE)*cur_err
        else:
            similar_or_better = mvl_err >= cur_err - self.MVL_SCREENING_TOLERANCE*abs(cur_err)
        if not similar_or_better:
            return ''
        return f'A simpler {base_txt} model has a similar or lower error in the model screening (Section F)'


    def get_warning_lines(self,pred_type,dat_verify=None):
        '''
        Gather the lines from PREDICT (and VERIFY, for the cluster-test check) where the
        potential warnings are printed
        '''

        warnings_dict = {}
        for suf_key in ('No PFI','PFI'):
            warnings_dict[f'bimodal_y_{suf_key}'] = False
            warnings_dict[f'outlier_pct_{suf_key}'] = 0.0
            warnings_dict[f'low_relevance_descp_{suf_key}'] = False

        # get lines with warnings from PREDICT
        file_pred = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        with open(file_pred, 'r', encoding='utf-8') as datfile:
            lines = datfile.readlines()
            pfi_section_pearson = False # to get both No PFI and PFI information
            pfi_section_outlier = False
            pfi_section_pfi = False
            pfi_section_shap = False
            # the bimodal marker line only appears conditionally (unlike Pearson/y-dist/PFI/
            # SHAP/outliers, which always print something) - a plain "first vs second
            # occurrence" toggle would misattribute it to No PFI if it only fires for PFI (or
            # vice versa), so track which suffix we're in via the always-present y-dist marker
            # instead, and just read that state when (if) the bimodal line shows up
            y_dist_suffix = 'No PFI'
            seen_first_y_dist = False
            for i,line in enumerate(lines):
                if 'Ideally, variables should show low' in line and not pfi_section_pearson:
                    warnings_dict['pearson_info_No PFI'] = lines[i+1][6:]
                    pfi_section_pearson = True # the next line found will correspond to the PFI section
                elif 'Ideally, variables should show low' in line and pfi_section_pearson:
                    warnings_dict['pearson_info_PFI'] = lines[i+1][6:]

                if 'Ideally, the number of datapoints in' in line:
                    if seen_first_y_dist:
                        y_dist_suffix = 'PFI'
                    warnings_dict[f'y_dist_info_{y_dist_suffix}'] = lines[i+2][6:]
                    seen_first_y_dist = True

                if 'WARNING! Your data seems to have two separate y-value groups' in line:
                    warnings_dict[f'bimodal_y_{y_dist_suffix}'] = True

                # PFI relative importance - flags a descriptor other than the top one whose
                # influence is negligible (< 4% of the top descriptor's, or < 0.05 in absolute
                # terms) once "PFI plot saved" is found, its descriptor lines are all
                # "-  {desc} = {value} +- {sd}" until the next blank/section line (see
                # PFI_plot() in utils.py) - sorted descending, so the first value is the max
                if 'PFI plot saved' in line and not pfi_section_pfi:
                    warnings_dict['low_relevance_descp_No PFI'] = _has_low_relevance_descp(lines,i)
                    pfi_section_pfi = True
                elif 'PFI plot saved' in line and pfi_section_pfi:
                    warnings_dict['low_relevance_descp_PFI'] = _has_low_relevance_descp(lines,i)

                # SHAP relative importance - same idea, using max(|min|,|max|) per descriptor
                # as the magnitude proxy (see shap_analysis() in utils.py, "-  {desc} = min:
                # {min}, max: {max}" lines, also sorted descending by that same magnitude)
                if 'SHAP plot saved' in line and not pfi_section_shap:
                    if _has_low_relevance_descp(lines,i,shap=True):
                        warnings_dict['low_relevance_descp_No PFI'] = True
                    pfi_section_shap = True
                elif 'SHAP plot saved' in line and pfi_section_shap:
                    if _has_low_relevance_descp(lines,i,shap=True):
                        warnings_dict['low_relevance_descp_PFI'] = True

                if pred_type == 'reg':
                    if 'Outliers plot saved' in line and not pfi_section_outlier:
                        max_SD, outlier_pct = _scan_outlier_block(lines,i)
                        warnings_dict['max_sd_No PFI'] = max_SD
                        warnings_dict['outlier_pct_No PFI'] = outlier_pct
                        pfi_section_outlier = True
                    elif 'Outliers plot saved' in line and pfi_section_outlier:
                        max_SD, outlier_pct = _scan_outlier_block(lines,i)
                        warnings_dict['max_sd_PFI'] = max_SD
                        warnings_dict['outlier_pct_PFI'] = outlier_pct

        # VERIFY's cluster test (regression only - N/A for classification, see
        # cluster_test() in verify.py) is the 4th of 4 fixed-order flawed-model tests
        # (y_mean, y_shuffle, onehot, cluster) printed right after the "Original {RMSE/MCC}
        # (" line - same fixed relative position get_verify_scores() relies on in
        # report_utils.py (dat_verify[i+4])
        for suf_key in ('No PFI','PFI'):
            warnings_dict[f'cluster_test_failed_{suf_key}'] = False
        if pred_type == 'reg' and dat_verify:
            error_keyword = 'RMSE'
            active_suffix = None
            for i,line in enumerate(dat_verify):
                if '------- ' in line and '(No PFI)' in line:
                    active_suffix = 'No PFI'
                elif '------- ' in line and 'with PFI' in line:
                    active_suffix = 'PFI'
                if active_suffix and f"Original {error_keyword} (" in line and i+4 < len(dat_verify):
                    cluster_line = dat_verify[i+4]
                    if 'cluster' in cluster_line and 'FAILED' in cluster_line:
                        warnings_dict[f'cluster_test_failed_{active_suffix}'] = True

        # ambiguous integer hyperparameters flagged by CHECKML (i.e. max_samples = 1 meaning ONE
        # sample instead of all of them, see validate_params() in checkml.py)
        warnings_dict['ambiguous_params'] = []
        checkml_log = Path(f'{os.getcwd()}/CHECKML/CHECKML_data.dat')
        if checkml_log.exists():
            for line in checkml_log.read_text(encoding='utf-8', errors='replace').splitlines():
                if 'WARNING! Ambiguous hyperparameter:' in line:
                    param = line.split('Ambiguous hyperparameter:')[1].split('=')[0].strip()
                    if param and param not in warnings_dict['ambiguous_params']:
                        warnings_dict['ambiguous_params'].append(param)

        # VERIFY's unscored MVL baseline comparison (see mvl_baseline_test() in verify.py),
        # logged after the Sorted CV line: SIMILAR / MVL_BETTER trigger a warning
        for suf_key in ('No PFI','PFI'):
            warnings_dict[f'mvl_baseline_{suf_key}'] = None
        active_suffix = None
        for line in (dat_verify or []):
            if '------- ' in line and '(No PFI)' in line:
                active_suffix = 'No PFI'
            elif '------- ' in line and 'with PFI' in line:
                active_suffix = 'PFI'
            # "MVL baseline" (regression) or "Logistic baseline" (classification)
            if active_suffix and 'baseline (not scored):' in line:
                status = line.split('baseline (not scored):')[1].split(',')[0].strip().lower()
                warnings_dict[f'mvl_baseline_{active_suffix}'] = status

        return warnings_dict


    def print_line_warning(self,message,style_lines,color,space):
        '''
        Add line with warning
        '''

        return f'''
        {style_lines}<span style='font-size:15px; color: {color};'>{space}&#9673;</span>
        {space}{message}</p>'''


    def print_adv_anal(self,pred_type,data_score,suffix,suffix_title):
        """
        Generates the advanced score analysis section (left = Interpolation, right = Boundary robustness)
        """

        adv_score_dat = ''

        adv_score_dat += self.module_lines('adv_anal',adv_score_dat,pred_type=pred_type)

        # Section B is a breakdown of the score computed in Section A - when there's no standard
        # CV/test split to calibrate it (already noted in Section A, no need to repeat it here),
        # each sub-metric badge below shows a grayed-out "N/A" version instead of a colored 0-2
        # bar (see score_badge_2() in report_utils.py), but the real underlying metrics, graphs
        # and any warnings they trigger (see analyze_warnings(), not gated on score_available)
        # are still shown as usual

        # Text sub-metrics pair up row by row via combine_cols (proven fine for text
        # everywhere else in the report). Images are NOT put in flex/table columns - they use
        # the exact same fixed-270px print_img_row()/print_img_tag() mechanism as Section C's
        # y_distribution image, which is the only approach that sizes images correctly.
        if pred_type == 'reg':
            interp_rows = [
                adv_flawed(suffix,data_score,'',pred_type),
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
            # classification: Boundary robustness isn't defined for a discrete MCC-based model
            # (no Low/High/degradation/applicability-domain analysis - see print_score()), so
            # only the Interpolation column is shown. Interpolation items 4 (train vs
            # validation gap) and 6 (prediction stability) are defined for classification too -
            # see the 'clas' branch in get_predict_scores() - so the row structure/order now
            # matches regression's exactly (5 rows)
            interp_rows = [
                adv_flawed(suffix,data_score,'',pred_type),
                adv_predict(self,suffix,data_score,'',pred_type),
                adv_test(self,suffix,data_score,'',pred_type),
                adv_train_val_gap(self,suffix,data_score,'',pred_type),
                adv_diff_test(self,suffix,data_score,'',pred_type) + adv_cv_sd(self,suffix,data_score,'',pred_type),
            ]

            bound_rows = []

        # images are interleaved right after the row they illustrate (not all dumped at the
        # end) so each one stays visually attached to its sub-metric: VERIFY_tests + High plot
        # after row 0 (item 1 on both sides), Low plot after row 1 (item 2, boundary robustness
        # side only), and the two item 5/6 plot pairs after the applicability-domain text
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
            bound_plot_offset = -175
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
            # reuses the already-proven single-image mechanism twice instead. The CV-variability
            # plots pair with the applicability-domain and Williams plots, respectively. Shift
            # only the right images up to sit below item 5; the left images stay below item 6.
            ad_stacked = (
                self.print_img_row_indent(
                    cv_variability_tag,ad_tag,10,right_offset=25,
                    right_top_offset=bound_plot_offset
                ) +
                self.print_img_row_indent(
                    cv_variability_cv_tag,williams_tag,10,right_offset=25,
                    right_top_offset=bound_plot_offset
                )
            )
            img_after_row = {
                0: self.print_img_row_indent(interp_tag1,bound_tag1,13,right_offset=25),
                1: self.print_img_row_indent('',self.print_img_tag(self.find_img('Results_boundary_low','PREDICT',suffix_title),218),10,right_offset=25),
                4: ad_stacked,
            }
        else:
            interp_tag1 = self.print_img_tag(self.find_img('VERIFY_tests','VERIFY',suffix_title),verify_height)
            img_after_row = {0: self.print_img_row_indent(interp_tag1,'',13)}

        for i in range(max(len(interp_rows),len(bound_rows))):
            left = interp_rows[i] if i < len(interp_rows) else ''
            # bound_rows is empty for clas (no Boundary robustness column) - right stays
            # blank and the two-column layout/proportions are kept as-is
            right = bound_rows[i] if i < len(bound_rows) else ''
            if i == 0:
                left = (
                    '<p style="text-align:justify; font-weight:bold; font-size:12px; '
                    'margin-top:-33px; margin-bottom:5px;">'
                    'Interpolation</p>' + left
                )
                if pred_type == 'reg':
                    right = (
                        '<p style="text-align:justify; font-weight:bold; font-size:12px; '
                        'margin-top:-33px; margin-bottom:5px;">'
                        f'{self.spacing_PFI}Boundary robustness</p>' + right
                    )
            # One out-of-flow grey rule spans each page's Boundary robustness rows. The
            # heights follow the known page grouping: rows 1-3 on the first page, then rows
            # 4-6 on the second. They do not take part in the flex layout or page flow.
            right_rule = (-4,875) if i == 0 and pred_type == 'reg' else (
                (0,945) if i == 3 and pred_type == 'reg' else None
            )
            adv_score_dat += combine_cols(
                [left,right],align_top=True,right_offset=25,right_rule=right_rule
            )
            adv_score_dat += img_after_row.get(i,'')

        if pred_type == 'reg':
            bound_diagnostic = adv_bound_diagnostic(suffix,data_score,self.spacing_PFI)
            if bound_diagnostic:
                # Keep the diagnostic text just below the right-column plots after their
                # visual shift. Relative positioning leaves the left column's flow untouched.
                bound_diagnostic = (
                    f'<div style="position:relative; top:{bound_plot_offset}px;">'
                    f'{bound_diagnostic}</div>'
                )
            adv_score_dat += combine_cols(['',bound_diagnostic],align_top=True,right_offset=25)

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
        Generates the outliers section - shown once, in the left column's position (the
        content is the same regardless of suffix, there's no separate interpolation/boundary
        version of it, so repeating it on the right was pure duplication)
        """

        # starts with the icon of outliers
        outlier_dat = ''
        outlier_dat = self.module_lines('outliers',outlier_dat,pred_type=pred_type)

        if pred_type == 'reg':
            # get information about outliers
            module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
            column_outlier = get_outliers(module_file,suffix,'')
            outlier_dat += combine_cols([column_outlier,''])

            # add corresponding image
            height = 217
            outlier_dat += self.print_img('Outliers',-5,height,'PREDICT',suffix_title)

        # add separator line and page break
        outlier_dat += '<hr style="margin-top: 20px;">'
        outlier_dat += f"""<p style="page-break-after: always;"></p>"""

        return outlier_dat


    def print_y_distrib(self,warnings_dict,suffix,suffix_title):
        """
        Generates the y distribution section - shown once, in the left column's position (the
        content is the same regardless of suffix, there's no separate interpolation/boundary
        version of it, so repeating it on the right was pure duplication)
        """

        # starts with the icon of outliers
        distrib_dat = ''
        distrib_dat = self.module_lines('y_distrib',distrib_dat)

        # add corresponding image
        height = 220
        distrib_dat += self.print_img('y_distribution',-5,height,'PREDICT',suffix_title)

        # split the sentence into 1 column size and add spacing line by line
        y_distrib_sentence = format_lines(warnings_dict[f'y_dist_info_{suffix}'],max_width=55,one_column=True,spacing='')

        column_y_distrib = f"""
        <p style='margin-top:25px; margin-bottom:-6px'><span style="font-weight:bold;">y distribution analysis</span></p>
        {y_distrib_sentence}
        <p style='margin-bottom:-15px'></p>
        """

        distrib_dat += combine_cols([column_y_distrib,''])

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
        # PREDICT logs it twice: with standardized descriptors (z = (x - mean)/SD, what the model
        # is fitted on) and in the original units of the descriptors (see linear_equation())
        equations = []
        with open(f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if 'o  Summary of results' in line and (('No_PFI:' in line) == (suffix == 'No PFI')):
                for j in range(i, len(lines)):
                    if 'o  SHAP' in lines[j]:
                        break
                    if 'o  Linear model equation' in lines[j] and j + 1 < len(lines):
                        header = lines[j].strip().replace('o  Linear model equation','').strip()
                        equations.append((header, lines[j + 1].strip().lstrip('- ')))
                break

        if equations:
            feature_dat += "<p style='margin-top:-5px; margin-bottom:-6px'><span style='font-weight:bold;'>Linear model equation</span></p>"
            for k,(header,equation) in enumerate(equations):
                # older logs only have the standardized equation, without a header
                label = f'<i>{header.strip("()")}:</i><br>' if header else ''
                bottom = 35 if k == len(equations)-1 else 0
                feature_dat += f"<p style='margin-top:10px; margin-bottom:{bottom}px'>{label}{equation}</p>"

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
                # shown once, in the left column's position - this suffix's PDF has only this
                # one plot, not a second, different one to pair it with, so showing it twice
                # (or centering it, which reads as its own third position) was duplication
                pair_list = f'<p style="width: 91%; text-align: left; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px"><img src="file:///{self._posix_uri(images[0])}" style="margin: 0; width: 55%;"/></p>'
            else:
                pair_list = f'<p style="width: 91%; margin-bottom: {margin_bottom}px; margin-top: {margin_top}px">{missing_messages[label]}</p>'
            feature_dat += pair_list

        # split the sentence into 1 column size and add spacing line by line - shown once, in
        # the left column's position (same content regardless of suffix, so repeating it on
        # the right was pure duplication)
        pearson_sentence = format_lines(warnings_dict[f'pearson_info_{suffix}'],max_width=55,one_column=True,spacing='')

        column_pearson = f"""
        <p style='margin-top:-10px; margin-bottom:-6px'><span style="font-weight:bold;">Correlation analysis</span></p>
        {pearson_sentence}
        """

        feature_dat += combine_cols([column_pearson,''])

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
                # the only layout WeasyPrint sizes correctly here. Reuses the same image
                # width and gap as the other fixed-width paired-image rows in the report.
                img_w = 270
                gap_px = 86.2
                spacer = f'<span style="display: inline-block; width: {gap_px}px;"></span>'

                interp_path = self.find_img('ScoreHeatmapInterp','GENERATE',suffix_title)
                bound_path = self.find_img('ScoreHeatmapBound','GENERATE',suffix_title)
                interp_tag = f'<img src="file:///{self._posix_uri(interp_path)}" style="margin: 0; width: {img_w}px;"/>' if interp_path else ''
                bound_tag = f'<img src="file:///{self._posix_uri(bound_path)}" style="margin: 0; width: {img_w}px;"/>' if bound_path else ''
                if self.all_models_clas:
                    bound_tag = f'<span style="display: inline-block; width: {img_w}px; text-align: center; color: #808080; font-size: 12px;">Disabled in classification problems.</span>'

                interp_cap = f'<span style="display: inline-block; width: {img_w}px; text-align: center; font-weight:bold;">Interpolation</span>'
                bound_cap = f'<span style="display: inline-block; width: {img_w}px; text-align: center; font-weight:bold;">Boundary robustness</span>'

                generate_dat += f'<p style="margin-bottom: 6px;">{interp_cap}{spacer}{bound_cap}</p>'
                generate_dat += f'<p style="margin-top: -5px;">{interp_tag}{spacer}{bound_tag}</p>'
                generate_dat += self.best_models_text(suffix_title)
            else:
                generate_dat += self.print_img('Heatmap',-5,height,'GENERATE',suffix_title)
                generate_dat += self.default_best_model_text(suffix_title)

        generate_dat += '<p style="margin-bottom: 50px;"></p>'

        return generate_dat


    def default_best_model_text(self,suffix_title):
        """
        Informative-only line (doesn't change which model runs through VERIFY/PREDICT/REPORT)
        showing which of the screened algorithms GENERATE picked as the best for this PFI
        variant, and its combined error. Cheap equivalent of best_models_text() for the
        default (non --all_models) case: reads the same Raw_data CSVs GENERATE's own
        heatmap_workflow()/detect_best() already produce, instead of requiring the full
        --all_models pipeline (every model through VERIFY/PREDICT) just to report this.
        Skipped when only one model was screened - there's nothing to compare against.
        """

        if len(self.args.model) <= 1:
            return ''

        raw_dir = Path(f'{os.getcwd()}/GENERATE/Raw_data/{suffix_title}')
        error_type = self.args.error_type.lower()

        model_errors = {}
        for csv_file in glob.glob(str(raw_dir / '*.csv')):
            if '_db' in os.path.basename(csv_file):
                continue
            try:
                params_df = pd.read_csv(csv_file, encoding='utf-8')
                model_name = str(params_df['model'][0]).upper()
                model_errors[model_name] = params_df[f'combined_{error_type}'][0]
            except (KeyError, IndexError, pd.errors.EmptyDataError):
                continue

        if len(model_errors) <= 1:
            return ''

        best_model = min(model_errors, key=model_errors.get) if error_type in ['mae','rmse'] else max(model_errors, key=model_errors.get)
        screened = ', '.join(model.upper() for model in self.args.model if model.upper() in model_errors)

        return (f'<p style="margin-top: 10px; margin-bottom: 0px; font-size: 12.5px;"><i>'
                f'Best of [{screened}]: <b>{best_model}</b> (combined {error_type.upper()} = {model_errors[best_model]:.2})'
                f'</i></p>')


    def best_models_text(self,suffix_title):
        """
        Informative-only line (doesn't change which model(s) run through VERIFY/PREDICT/
        REPORT) showing, for this PFI variant, the single overall-best model - same
        combined-score pick as organize_all_models_pdfs() (see _pick_best_overall()), so this
        text always names the same model as the PDF that actually gets kept
        """

        suffix = suffix_title.replace('_',' ')
        scores = getattr(self,'model_scores',{}).get(suffix,{})
        if not scores:
            return ''

        # canonical (key, interp, bound, rmse_cv, rmse_bound) shape - same as
        # organize_all_models_pdfs()'s candidates, so _pick_best_model()/_pick_best_overall()
        # can use the exact same index arguments there instead of a second hand-shifted copy
        items = [(model,vals[0],vals[1],vals[2],vals[3]) for model,vals in scores.items()]

        # Boundary robustness isn't defined in classification, so only Interpolation is ranked
        if self.all_models_clas:
            best_model = self._pick_best_model(items,1,2,3)[0]
            return (f'<p style="margin-top: 10px; margin-bottom: 0px; font-size: 12.5px;"><i>'
                    f'Best overall model: <b>{best_model}</b></i></p>')

        best_model,interp_val,bound_val,*_ = self._pick_best_overall(items)

        return (f'<p style="margin-top: 10px; margin-bottom: 0px; font-size: 12.5px;"><i>'
                f'Best overall model: <b>{best_model}</b> (Interpolation {interp_val}, Boundary robustness {bound_val})'
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
            # Boundary robustness isn't defined in classification (always 0), so no heatmap
            if score_idx == 1 and self.all_models_clas:
                continue
            csv_df = pd.DataFrame(
                {model: [model_scores_suffix[model][score_idx]] for model in model_cols},
                index=[label],
            )
            _ = create_score_heatmap(csv_df, save_dir / f'{file_name}_{suffix_title}.png')


    def get_repro(self,eval_only):
        """
        Generates the reproducibility section
        """

        version_n_date, citation_main, citation_second, command_line, python_version, total_time, dat_files = repro_info(self.args.report_modules,self.model_suffix)
        citation_main = (citation_main
                         .replace('WIREs Comput. Mol. Sci.', '<i>WIREs Comput. Mol. Sci.</i>')
                         .replace('2024', '<b>2024</b>')
                         .replace(', 14,', ', <i>14</i>,'))
        robert_version = version_n_date.split()[2]

        if self.args.csv_name == '' or self.args.csv_test == '':
            self = get_csv_names(self,command_line)

        repro_dat,citation_dat = '',''

        # version, date and citation - the main ROBERT paper and the low-data-regimes one each
        # get their own row (still 2 lines total, same as before) instead of one long paragraph
        citation_dat += f"""<p style="text-align: justify; margin-top: -9px;"><br>{version_n_date}</p>
        <p style="text-align: justify; margin-top: -10px; margin-bottom: 0px;"><span style="font-weight:bold;">Please cite ROBERT:</span> {citation_main}</p>"""
        citation_dat += f"""<p style="text-align: justify; margin-top: 0px; margin-bottom: 15px;"><span style="font-weight:bold;">and CheckML score:</span> {citation_second}</p>"""

        # QDESCP's own validation paper - only relevant when AQME generated the descriptors
        # (command_line already has the full "python -m robert ..." line at this point, so the
        # --aqme flag is checked here directly instead of waiting for aqme_workflow below)
        # kept on self so print_warnings() can leave room for this extra (2-line) header row
        self.qdescp_citation = '--aqme' in command_line
        if '--aqme' in command_line:
            qdescp_citation = "Dalmau, D.; Jacot-Descombes, L.; Kalikadien, A.; Manzanilla, B.; Pidko, E. A.; Jorner, K.; Sigman, M. S.; Alegre-Requena, J. V. ACS Catal. 2026, 16, 12565-12574."
            citation_dat += f"""
        <p style="text-align: justify; margin-top: 0px;"><span style="font-weight:bold;">QDESCP descriptors:</span> {qdescp_citation}</p>"""

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
        Collects the data for model parameters and misc options in the Reproducibility section -
        shown once, in the left column's position (same content regardless of suffix, so
        repeating it on the right was pure duplication)
        """

        # set the parameters for the ML model
        params_dir = f'{self.args.params_dir}/{"_".join(suffix.split())}'

        # --all_models: each model's PDF shows its own parameters (from the per-model folders
        # staged by GENERATE), not the ones from the best model in Best_model
        model_name = getattr(self,'model_suffix','')[1:]
        if getattr(self.args,'all_models',False) and model_name:
            params_dir_model = f'{self.args.params_dir.replace("Best_model","All_models")}/{"_".join(suffix.split())}/{model_name}'
            if os.path.isdir(params_dir_model):
                params_dir = params_dir_model
        files_param = glob.glob(f'{params_dir}/*.csv')
        for file_param in files_param:
            if '_db' not in file_param:
                params_df = pd.read_csv(file_param, encoding='utf-8')
            else:
                # database (with its Set column) of this model, used by print_summary_page()
                self.report_db_file = file_param
        params_dict = pd_to_dict(params_df) # (using a dict to keep the same format of load_model)

        column_repro = get_col_transpa(params_dict,suffix,section,'')
        section_dat = combine_cols([column_repro,''])
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
        Generates the new predictions section - shown once, in the left column's position
        (same content regardless of suffix, so repeating it on the right was pure duplication)
        """

        # detects whether there are predictions from an external test set
        module_file = f'{os.getcwd()}/PREDICT/PREDICT{self.model_suffix}_data.dat'
        csv_test_exists, y_value, names, path_csv_test = detect_predictions(module_file)
        if csv_test_exists:
            pred_dat = ''
            pred_dat = self.module_lines('pred',pred_dat,pred_type=pred_type)

            # add metrics
            column_metrics = get_csv_metrics(module_file,suffix,'')

            # add predictions table
            column_pred = get_csv_pred(suffix,path_csv_test,y_value,names,'')

            pred_dat += combine_cols([column_metrics,''])
            pred_dat += combine_cols([column_pred,''])

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


    def module_lines(self,module,module_data,pred_type='reg',eval_only=False,explain_override=None):
        """
        Returns the line with icon and module for section titles. explain_override replaces the
        standing intro paragraph with different text in the exact same slot (only used for
        'score', when the score isn't calibrated for this run - print_score() passes a shorter
        "Score not available" notice instead, so the Interpolation/Boundary robustness columns
        right below keep their normal spacing instead of sitting under a separately-appended
        paragraph)
        """

        if module == 'score':
            module_name = 'Section A. CheckML Score'
            if explain_override is not None:
                section_explain = explain_override
            elif pred_type == 'clas':
                # Boundary robustness isn't defined for classification (see print_score())
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Evaluation of interpolation (within the range of training data).</i>'
            else:
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">Evaluation of interpolation (within the range of training data) and boundary robustness (related to extrapolation).</i>'
        elif module == 'adv_anal':
            module_name = 'Section B. Advanced Score Analysis'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section explains each component that comprises the CheckML score. <a href="https://robert.readthedocs.io/en/latest/Report/score.html">More details here.</a></i>'
        elif module == 'y_distrib':
            module_name = 'Section C. Distribution of y Values'
            section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">This section shows the distribution of y values within the training and validation sets. In regression, Q1-Q4 are four equal-width intervals of the y range (from the lowest to the highest y value), not quartiles of the data: a uniform dataset would have about 25% of the points in each.</i>'
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
                section_explain = f'<p style="margin-top:-7px;"><i style="text-align: justify;">The screening of models is disabled when using the CHECKML module.</i>'
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

        # only Section A opens with a line (below the header on page 1); every other section
        # is separated by the closing line of the previous one (see _close_section())
        if module == 'score':
            separator_section = '<hr><p style="margin-top:25px;"></p>'
        else:
            separator_section = '<p style="margin-top:0px;"></p>'

        # page-break-after: avoid keeps the title (and its explanation line) on the same page
        # as the start of the section, instead of stranding it at the bottom of a page
        section_explain = section_explain.replace('<p style="', '<p style="page-break-after: avoid; ', 1)
        title_position = ''
        if module == 'adv_anal':
            # Lift Section B's header and explanation while leaving its content in place,
            # creating a little more visual space before the Interpolation/Boundary headings.
            title_position = 'position:relative; top:-10px; '
            section_explain = section_explain.replace(
                '<p style="', '<p style="position:relative; top:-10px; ', 1
            )
        section_anchor = ' id="section-b"' if module == 'adv_anal' else ''
        title_line = f"""
            {separator_section}
            <p{section_anchor} style="page-break-after: avoid; {title_position}"><span style="font-weight:bold;">
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
        list/tuple (e.g. so a search for ``CV_variability`` doesn't also match
        ``CV_variability_boundary_...`` files)
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

    def print_img_tag(self,image_path,height,width=270):
        """
        Generates an <img> tag at the requested width and height.
        """

        if not image_path:
            return ''

        return f'<img src="file:///{self._posix_uri(image_path)}" style="margin: 0; width: {width}px; height: {height}px; object-fit: cover; object-position: 0 100%;"/>'


    def print_img_row_indent(self,left_tag,right_tag,margin_top,right_offset=0,left_width=270,right_top_offset=0):
        """
        Places two DIFFERENT images (interpolation left, boundary robustness right) in a
        single flowing paragraph, like print_img_row, but with a precisely calculated spacer
        (not 22 nbsp) so the right image starts at the same x as the indented Boundary
        robustness caption/text above it. Flex columns and position:absolute were both tried
        first, but WeasyPrint silently shrinks a fixed-width <img> below its declared width
        whenever it's wrapped in a flex item or an absolutely positioned span - plain inline
        flow (like the original print_img_row) is the only layout that renders it at full size.
        331.2px = row half-width (234.9pt, confirmed empirically via the width:100% score-bar
        image) + the 18px BOUNDARY_INDENT used for the text above, converted to px. The spacer
        is reduced by the displayed left image width so the right column keeps the same x
        position when that image is enlarged. right_top_offset shifts the right image visually
        without changing the left image's position or the row's flow height.
        """

        spacer_px = (331.2 - left_width if left_tag else 331.2) + (right_offset if right_tag else 0)
        spacer = f'<span style="display: inline-block; width: {spacer_px}px;"></span>'
        if right_tag and right_top_offset:
            right_tag = right_tag.replace(
                'style="',f'style="position:relative; top:{right_top_offset}px; ',1
            )

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
        Generates the row for a single image (matching suffix_title), shown once in the left
        column's position (not centered, not duplicated) - each suffix's PDF only ever has
        this one plot (not a second, different one to pair it with), so showing it twice side
        by side was pure visual duplication rather than an actual left/right comparison
        """

        tag = self.print_img_tag(self.find_img(file_name,module,suffix_title),height)
        if not tag:
            return ''

        return self.print_img_row(tag,'',margin_top)


