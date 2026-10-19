'''
Created on 22 April 2026

@author: thomasgumbricht
'''

# Standard library imports
import os
import json
from math import ceil

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Application package imports
from src.postgres import Get_schema_table

from src.postgres.pg_ai4sh import PG_manage_AI4SH

from src.lib.pilot import Get_project_path

from src.ai4sh.chemometrics import (apply_transformations, apply_standardisation,
                                    apply_chemometrics, apply_scatter_correction,
                                    apply_scaling)

from src.ai4sh.filter import apply_filter, apply_multi_filter, _load_filter_config

from src.ai4sh.feature_symbols import Load_feature_symbols

from src.ai4sh.parquet_units import Read_parquet_with_units, Find_selected_parquet

from src.ai4sh.spectral_columns import Band_value, Is_band_col, Signal_type_of, SIGNAL_TYPE_LABELS

def _spectra_x_axis(columns):
    '''Return (x_values, x_label) appropriate for the given spectral column list.'''
    if not columns:
        return [], 'x'
    first = str(columns[0])
    if first.startswith('pc'):
        return list(range(1, len(columns) + 1)), 'Component'
    try:
        vals = [int(str(c).lstrip('d')) for c in columns]
        return vals, 'Wavelength (nm)'
    except (ValueError, TypeError):
        return list(range(len(columns))), 'Band'

def _step_to_abbrev(step_label):
    '''Return a short abbreviation for an individual chemometric step label.

    See ai4sh/default/chemometric/chemometric_abbreviations.md for the full table.
    '''
    if step_label == 'raw':
        return 'raw'
    if step_label.startswith('derivative:'):
        if 'order ' in step_label:
            n = step_label.split('order ')[-1].rstrip(')').strip()
            return 'd%s' % n
        return 'd1'
    if step_label.startswith('scatter_correction:'):
        scaler = step_label.split(':', 1)[-1].strip()
        return scaler.replace('+', '')
    if step_label.startswith('scaling:'):
        method = step_label.split(':', 1)[-1].strip()
        return {'meancentring': 'mc', 'autoscaling': 'as',
                'paretoscaling': 'ps', 'poissonscaling': 'poi'}.get(method, method[:4])
    if step_label.startswith('decomposition:'):
        if 'pca' in step_label:
            parts = step_label.split('(')
            n = parts[1].split(' ')[0] if len(parts) > 1 else '?'
            return 'pca%s' % n
        return 'decomp'
    if step_label.startswith('filter:'):
        method = step_label.split(':', 1)[-1]
        return {'moving_average': 'ma', 'gauss': 'gf', 'savitzky-golay': 'sg',
                'lowess': 'lw', 'multi_filter': 'mf'}.get(method, method[:4])
    return ''.join(c if c.isalnum() else '' for c in step_label.lower())[:6]

class Process_plot(Get_schema_table):
    '''Plot indicator distributions (boxplot, histogram) from a saved Parquet dataset.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):

        self.verbose = process_S.process.verbose

        self.process_S = process_S

        self.pg_session_C = pg_session_C

        self.project_root_FP = project_root_FP

        self.pg_ai4sh_C = PG_manage_AI4SH(pg_session_C)

        # y-axis label for spectra, set from header row 2 when the Parquet is loaded
        self.signal_label = SIGNAL_TYPE_LABELS['refl']

    def _Sub_process(self, _json_file_key):

        if self.process_S.process.process == 'plot_indicators':

            self._Plot_indicators()

        elif self.process_S.process.process == 'plot_spectra':

            self._Plot_spectra()

        elif self.process_S.process.process == 'plot_spectra_scattercorrection_scalers':

            self._Plot_spectra_scattercorrection_scalers()

        elif self.process_S.process.process == 'plot_spectra_scaling_methods':

            self._Plot_spectra_scaling_methods()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _Parse_indicator_array(self):
        '''Return list of indicator and covariate names from the indicator_array and covariate_array
        parameters (in that order, without duplicates).

        Handles JSON list, CSV string, or postgres "{a,b,c}" format.
        '''
        names = []

        for key in ('indicator_array', 'covariate_array'):

            for name in self._Parse_chemometrics_array(getattr(self.process_S.process.parameters, key, '')):

                if name not in names:
                    names.append(name)

        return names

    def _Load_parquet_data(self, project_root_fp):
        '''Find and load the selected {name}.parquet + {name}.json from project_root_fp
        (see Find_selected_parquet).

        Returns (DataFrame, params_dict). Raises FileNotFoundError if not found.
        '''
        parquet_fpn, params_fpn = Find_selected_parquet(project_root_fp)

        df, units_D = Read_parquet_with_units(parquet_fpn)

        params_D = {}
        if params_fpn:
            with open(params_fpn) as f:
                params_D = json.load(f)

        # Units embedded in the Parquet's column MultiIndex take priority over the (legacy)
        # JSON copy, which is kept only as a fallback for older files.
        if units_D:
            params_D['indicator_units'] = units_D

        self.signal_label = SIGNAL_TYPE_LABELS[
            Signal_type_of(units_D, [c for c in df.columns if Is_band_col(c)])]

        if self.verbose >= 1:
            n_samples = params_D.get('n_samples', len(df))
            print('    Loaded %d rows (%d expected) from %s' % (
                len(df), n_samples, os.path.basename(parquet_fpn)))

        return df, params_D

    def _Load_feature_symbols(self):
        '''Load targetfeaturesymbols JSON. Returns the inner dict keyed by indicator name.

        "default" → loads from <project>/default/plot/targetfeaturesymbols.json.
        Any other value is treated as a file path.
        '''
        raw = getattr(self.process_S.process.parameters, 'targetfeaturesymbols', 'default')

        return Load_feature_symbols(raw, self.verbose)

    def _Build_plot_output_path(self, project_root_fp):
        '''Create and return the plot output directory under project_root_fp/plot/.'''

        plot_dir = os.path.join(project_root_fp, 'plot')

        os.makedirs(plot_dir, exist_ok=True)

        return plot_dir

    def _Get_symbol(self, symbols, indicator, applied_units_D=None):
        '''Return (color, label, unit) for an indicator, with fallback to indicator name.

        If the symbol's unit is "native", resolve to the actual DB-recorded unit saved in
        applied_units_D (params_D['indicator_units']) instead of the literal "native".
        '''

        sym = symbols.get(indicator, {})

        color = sym.get('color', 'steelblue')
        label = sym.get('label', indicator)
        unit = sym.get('unit', '')

        if unit == 'native' and applied_units_D:
            unit = applied_units_D.get(indicator, unit)

        return color, label, unit

    def _build_transform_info(self, indicator, transform_dict, standard_dict):
        '''Return (annotation_str, filename_suffix) for an individual indicator.'''

        transform = transform_dict.get(indicator, 'linear')
        standardised = standard_dict.get(indicator, 'none') != 'none'

        transform_label = 'none' if transform == 'linear' else transform
        ann = 'standard: %s  transform: %s' % (str(standardised).lower(), transform_label)

        parts = []
        if standardised:
            parts.append('z')
        if transform != 'linear':
            parts.append(transform)
        suffix = ('_' + '_'.join(parts)) if parts else ''

        return ann, suffix

    def _build_combined_suffix(self, indicators, transform_dict, standard_dict):
        '''Return filename suffix for an all-indicators plot combining all applied operations.'''

        has_z = any(standard_dict.get(i, 'none') != 'none' for i in indicators)
        transforms = sorted(set(
            transform_dict.get(i, 'linear') for i in indicators
            if transform_dict.get(i, 'linear') != 'linear'
        ))

        parts = []
        if has_z:
            parts.append('z')
        parts.extend(transforms)

        return ('_' + '_'.join(parts)) if parts else ''

    def _Parse_chemometrics_array(self, raw):
        '''Return list of chemometric step names from array parameter.'''
        if not raw:
            return []
        if isinstance(raw, list):
            return [x.strip() for x in raw if x.strip()]
        s = str(raw).strip()
        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]
        return [x.strip() for x in s.split(',') if x.strip()]

    def _Plot_spectra_panel(self, df_step, step_cols, colors, step_label, ann,
                            chain_ann, file_suffix, show, save, plot_dir):
        '''Render one spectral panel and optionally save/show it.

        chain_ann: accumulated preprocessing history shown upper-right (e.g. "snv→mc").
        file_suffix: filename stem derived from chain_ann (e.g. "snv_mc").
        '''
        x_vals, x_label = _spectra_x_axis(step_cols)
        y_label = 'Score' if step_cols and str(step_cols[0]).startswith('pc') else self.signal_label

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (_, row) in enumerate(df_step.iterrows()):
            ax.plot(x_vals, row[step_cols].values, color=colors[i], linewidth=0.8)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(step_label)
        ax.text(0.02, 0.04, ann, transform=ax.transAxes, fontsize=8, verticalalignment='bottom')
        ax.text(0.98, 0.98, chain_ann, transform=ax.transAxes, fontsize=8,
                ha='right', va='top', color='gray')
        fig.tight_layout()

        if save:
            fpn = os.path.join(plot_dir, 'spectra_%s.png' % file_suffix)
            fig.savefig(fpn, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Saved: %s' % fpn)

        if show:
            plt.show()

        plt.close(fig)

    def _Histogram_plot(self, df, indicators, symbols, bins, show, save, plot_dir,
                        transform_dict=None, standard_dict=None,
                        individual_canvas_plot=True, single_canvas_plot=False, applied_units_D=None):
        '''Plot histograms for each indicator — individual canvases then multi-column grid canvas.'''

        n_cols = 3

        transform_dict = transform_dict or {}
        standard_dict = standard_dict or {}

        # --- Individual canvases ---
        if individual_canvas_plot:
            for indicator in indicators:

                if indicator not in df.columns:
                    if self.verbose >= 1:
                        print('    Warning: indicator "%s" not in DataFrame — skipping.' % indicator)
                    continue

                series = df[indicator].dropna()
                color, label, unit = self._Get_symbol(symbols, indicator, applied_units_D)
                ann, suffix = self._build_transform_info(indicator, transform_dict, standard_dict)

                fig, ax = plt.subplots(figsize=(6, 4))
                series.plot.hist(bins=bins, color=color, ax=ax)
                ax.set_title(label)
                ax.set_xlabel(unit)
                ax.text(0.98, 0.98, ann, transform=ax.transAxes, fontsize=7,
                        ha='right', va='top', color='gray')

                if save:
                    fpn = os.path.join(plot_dir, 'histogram_%s%s.png' % (indicator, suffix))
                    fig.savefig(fpn, bbox_inches='tight')
                    if self.verbose >= 1:
                        print('    Saved: %s' % fpn)

                if show:
                    plt.show()

                plt.close(fig)

        # --- Multi-column grid (separate pass so individual plots are fully closed first) ---
        if len(indicators) > 1 and single_canvas_plot:

            n_rows = ceil(len(indicators) / n_cols)
            multi_fig, multi_axs = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 4, n_rows * 3)
            )
            axs_flat = np.array(multi_axs).flatten()

            f = 0
            for indicator in indicators:

                if indicator not in df.columns:
                    continue

                series = df[indicator].dropna()
                color, label, unit = self._Get_symbol(symbols, indicator, applied_units_D)
                ann, _ = self._build_transform_info(indicator, transform_dict, standard_dict)
                ax = axs_flat[f]
                series.plot.hist(bins=bins, color=color, ax=ax)
                ax.set_title(label)
                ax.set_xlabel(unit)
                ax.text(0.98, 0.98, ann, transform=ax.transAxes, fontsize=6,
                        ha='right', va='top', color='gray')
                if f % n_cols != 0:
                    ax.set_ylabel('')
                f += 1

            for idx in range(f, len(axs_flat)):
                axs_flat[idx].set_visible(False)

            multi_fig.tight_layout()

            if save:
                combined_suffix = self._build_combined_suffix(indicators, transform_dict, standard_dict)
                fpn = os.path.join(plot_dir, 'histogram_all-indicators%s.png' % combined_suffix)
                multi_fig.savefig(fpn, bbox_inches='tight')
                if self.verbose >= 1:
                    print('    Saved: %s' % fpn)

            if show:
                plt.show()

            plt.close(multi_fig)

    def _Class_style(self, symbols, name):
        '''Return bar style kwargs for a nominal covariate from its symbol: color, alpha, hatch
        (e.g. "//", "xx", ".."), fill (false = outline only) and edgecolor (default = color).'''

        sym = symbols.get(name, {})

        color = sym.get('color', 'steelblue')

        fill = sym.get('fill', True)

        return dict(color=color if fill else 'none', alpha=sym.get('alpha', 1.0), hatch=sym.get('hatch'),
                    edgecolor=sym.get('edgecolor', color), linewidth=0.8)

    def _Class_counts(self, series, max_classes):
        '''Return class counts (largest first), the classes beyond max_classes summed as "other".'''

        counts = series.dropna().astype(str).value_counts()

        if max_classes and len(counts) > max_classes:

            other = counts.iloc[max_classes:].sum()

            counts = counts.iloc[:max_classes]

            counts['other (%s classes)' % (len(series.dropna().unique()) - max_classes)] = other

        return counts

    def _Relabel_classes(self, df, nominal, params_D, class_label):
        '''Return df with the nominal class values (tags) shown as class_label: tag, display_name, name or alias.

        The lookup is the class_labels of the selection params JSON; values without a lookup (older
        selections, nominal indicators) are shown as stored.
        '''
        lookup_D = (params_D or {}).get('class_labels', {})

        if class_label == 'tag' or not lookup_D:
            return df

        df = df.copy()

        for name in nominal:

            classes_D = lookup_D.get(name, {})

            mapping = {tag: info.get(class_label) or tag for tag, info in classes_D.items()}

            df[name] = df[name].map(lambda v: mapping.get(v, v))

        return df

    def _Class_bar_axes(self, ax, counts, label, style, fontsize):
        '''Draw class counts as horizontal bars (largest on top) on ax.'''

        names = [n if len(n) <= 40 else n[:39] + '…' for n in counts.index]

        ax.barh(names[::-1], counts.values[::-1], height=0.7, **style)

        # Same bar thickness for few classes: room for at least 5 bars, bars at the top
        n_bars = len(counts)

        ax.set_ylim(n_bars - max(n_bars, 5) - 0.6, n_bars - 0.4)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(label)

        ax.set_xlabel('samples')

        ax.tick_params(axis='y', labelsize=fontsize)

        ax.text(0.98, 0.02, '%s class%s, n = %s' % (len(counts), '' if len(counts) == 1 else 'es', int(counts.sum())), transform=ax.transAxes,
                fontsize=fontsize, ha='right', va='bottom', color='gray')

    def _Class_bar_plot(self, df, covariates, symbols, max_classes, show, save, plot_dir,
                        individual_canvas_plot=True, single_canvas_plot=False, ranks_D=None):
        '''Plot class counts for each nominal covariate — individual canvases then a grid canvas.
        ranks_D: {covariate: rank} of hierarchical covariates, added to the title unless lowest.'''

        def _Title(name):
            rank = (ranks_D or {}).get(name, 'lowest')
            label = self._Get_symbol(symbols, name)[1]
            return label if rank == 'lowest' else '%s (%s)' % (label, rank)

        if individual_canvas_plot:

            for name in covariates:

                counts = self._Class_counts(df[name], max_classes)

                fig, ax = plt.subplots(figsize=(7, 0.3 * max(len(counts), 5) + 1.5))

                self._Class_bar_axes(ax, counts, _Title(name), self._Class_style(symbols, name), 8)

                fig.tight_layout()

                if save:
                    fpn = os.path.join(plot_dir, 'classes_%s.png' % name)
                    fig.savefig(fpn, bbox_inches='tight')
                    if self.verbose >= 1:
                        print('    Saved: %s' % fpn)

                if show:
                    plt.show()

                plt.close(fig)

        if len(covariates) > 1 and single_canvas_plot:

            n_cols = 2

            n_rows = ceil(len(covariates) / n_cols)

            all_counts = [self._Class_counts(df[name], max_classes) for name in covariates]

            row_height = 0.3 * max(max(len(c) for c in all_counts), 5) + 1.5

            multi_fig, multi_axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * row_height))

            axs_flat = np.array(multi_axs).flatten()

            for f, (name, counts) in enumerate(zip(covariates, all_counts)):

                self._Class_bar_axes(axs_flat[f], counts, _Title(name), self._Class_style(symbols, name), 7)

            for idx in range(len(covariates), len(axs_flat)):
                axs_flat[idx].set_visible(False)

            multi_fig.tight_layout()

            if save:
                fpn = os.path.join(plot_dir, 'classes_all-covariates.png')
                multi_fig.savefig(fpn, bbox_inches='tight')
                if self.verbose >= 1:
                    print('    Saved: %s' % fpn)

            if show:
                plt.show()

            plt.close(multi_fig)

    def _Boxplot_plot(self, df, indicators, symbols, show, save, plot_dir,
                      transform_dict=None, standard_dict=None,
                      individual_canvas_plot=True, single_canvas_plot=False, applied_units_D=None):
        '''Plot boxplots for each indicator — individual canvases then multi-column grid canvas.'''

        n_cols = 3
        transform_dict = transform_dict or {}
        standard_dict = standard_dict or {}

        # --- Individual canvases ---
        if individual_canvas_plot:
            for indicator in indicators:

                if indicator not in df.columns:
                    if self.verbose >= 1:
                        print('    Warning: indicator "%s" not in DataFrame — skipping.' % indicator)
                    continue

                color, label, unit = self._Get_symbol(symbols, indicator, applied_units_D)
                ann, suffix = self._build_transform_info(indicator, transform_dict, standard_dict)

                fig, ax = plt.subplots(figsize=(4, 5))
                df.boxplot(
                    column=[indicator],
                    patch_artist=True,
                    boxprops=dict(facecolor=color),
                    ax=ax
                )
                ax.set_title(label)
                ax.set_xlabel(unit)
                ax.text(0.98, 0.98, ann, transform=ax.transAxes, fontsize=7,
                        ha='right', va='top', color='gray')

                if save:
                    fpn = os.path.join(plot_dir, 'boxplot_%s%s.png' % (indicator, suffix))
                    fig.savefig(fpn, bbox_inches='tight')
                    if self.verbose >= 1:
                        print('    Saved: %s' % fpn)

                if show:
                    plt.show()

                plt.close(fig)

        # --- Multi-column grid (separate pass so individual canvases are fully closed first) ---
        if len(indicators) > 1 and single_canvas_plot:

            n_rows = ceil(len(indicators) / n_cols)
            multi_fig, multi_axs = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 3, n_rows * 4)
            )
            axs_flat = np.array(multi_axs).flatten()

            f = 0
            for indicator in indicators:

                if indicator not in df.columns:
                    continue

                color, label, unit = self._Get_symbol(symbols, indicator, applied_units_D)
                ann, _ = self._build_transform_info(indicator, transform_dict, standard_dict)
                ax = axs_flat[f]
                df.boxplot(
                    column=[indicator],
                    patch_artist=True,
                    boxprops=dict(facecolor=color),
                    ax=ax
                )
                ax.set_title(label)
                ax.set_xlabel(unit)
                ax.text(0.98, 0.98, ann, transform=ax.transAxes, fontsize=6,
                        ha='right', va='top', color='gray')
                if f % n_cols != 0:
                    ax.set_ylabel('')
                f += 1

            for idx in range(f, len(axs_flat)):
                axs_flat[idx].set_visible(False)

            multi_fig.tight_layout()

            if save:
                combined_suffix = self._build_combined_suffix(indicators, transform_dict, standard_dict)
                fpn = os.path.join(plot_dir, 'boxplot_all-indicators%s.png' % combined_suffix)
                multi_fig.savefig(fpn, bbox_inches='tight')
                if self.verbose >= 1:
                    print('    Saved: %s' % fpn)

            if show:
                plt.show()

            plt.close(multi_fig)

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def _Plot_indicators(self):
        '''Read saved Parquet data and plot indicator distributions.'''

        p = self.process_S.process.parameters

        # Resolve project root path
        project_root_fp = str(p.project_root_fp).strip()

        # Support relative paths via Full_path_locate if not absolute
        if not os.path.isabs(project_root_fp):
            # Resolve relative to repo root
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)

        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        # Load data
        try:
            df, params_D = self._Load_parquet_data(project_root_fp)
        except FileNotFoundError as e:
            print('    ERROR: %s' % e)
            return

        if self.verbose >= 2 and params_D:
            provision = params_D.get('provision_name', '')
            bw = params_D.get('output_bandwidth', '')
            beg = params_D.get('begin_wavelength', '')
            end = params_D.get('end_wavelength', '')
            print('    Dataset: %s  provision: %s  range: %s-%s nm  bw: %s nm' % (
                params_D.get('dataset_name', ''), provision, beg, end, bw))

        # Determine which indicators to plot
        indicators = self._Parse_indicator_array()

        if not indicators:
            # Use all non-spectral, non-metadata columns
            skip_cols = {'sample_id', 'sample_name', 'campaign_tag', 'sampling_log_tag', 'campaign_name', 'latitude_dd',
                         'longitude_dd', 'profile_min', 'profile_max'}
            indicators = [
                c for c in df.columns
                if c not in skip_cols and not c.startswith('class_') and not Is_band_col(c)
            ]

        # Keep only indicators that exist in the DataFrame
        indicators = [i for i in indicators if i in df.columns]

        if not indicators:
            print('    No indicators found in the data. Available columns: %s' % list(df.columns))
            return

        # Nominal (class) indicators/covariates are plotted as class counts, numeric ones as
        # histograms and boxplots
        nominal = [i for i in indicators if not pd.api.types.is_numeric_dtype(df[i])]

        indicators = [i for i in indicators if i not in nominal]

        if self.verbose >= 1:
            print('    Plotting %d numeric: %s and %d nominal: %s' % (len(indicators), indicators, len(nominal), nominal))

        # Apply transformation and standardisation before plotting
        transformation_param = getattr(p, 'transformation', 'default')
        standardisation_param = getattr(p, 'standardisation', 'default')
        df, transform_dict = apply_transformations(df, indicators, transformation_param, self.verbose)
        df, standard_dict = apply_standardisation(df, indicators, standardisation_param, self.verbose)

        if params_D and 'indicator_units' not in params_D:
            print('    Warning: this Parquet was built before unit tracking was added to select_spectra — '
                  'units are not guaranteed to match targetfeaturesymbols.json. Re-run select to be sure.')

        # Load symbols
        symbols = self._Load_feature_symbols()

        # Read plot control flags
        do_histogram = getattr(p, 'histogram', True)
        do_boxplot = getattr(p, 'boxplot', True)
        show = getattr(p, 'show', True)
        save = getattr(p, 'save', False)
        bins = getattr(p, 'bins', 10)
        single_canvas_plot = getattr(p, 'single_canvas_plot', False)
        individual_canvas_plot = getattr(p, 'individual_canvas_plot', True)

        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None

        applied_units_D = params_D.get('indicator_units', {}) if params_D else {}

        if nominal and (do_histogram or do_boxplot):
            class_label = (getattr(p, 'class_label', 'display_name') or 'display_name').strip().lower()

            if class_label not in ('tag', 'display_name', 'name', 'alias'):

                print('    ⚠️  Unknown class_label "%s" (tag, display_name, name or alias); using display_name' % class_label)

                class_label = 'display_name'

            self._Class_bar_plot(self._Relabel_classes(df, nominal, params_D, class_label), nominal, symbols,
                                 getattr(p, 'max_classes', 15), show, save, plot_dir,
                                 individual_canvas_plot, single_canvas_plot, (params_D or {}).get('covariate_rank'))

        if not indicators:
            return

        if do_histogram:
            self._Histogram_plot(df, indicators, symbols, bins, show, save, plot_dir,
                                 transform_dict, standard_dict,
                                 individual_canvas_plot, single_canvas_plot, applied_units_D)

        if do_boxplot:
            self._Boxplot_plot(df, indicators, symbols, show, save, plot_dir,
                               transform_dict, standard_dict,
                               individual_canvas_plot, single_canvas_plot, applied_units_D)

    def _Plot_spectra(self):
        '''Plot spectra through each chemometric step from a saved Parquet dataset.'''

        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        try:
            df, params_D = self._Load_parquet_data(project_root_fp)
        except FileNotFoundError as e:
            print('    ERROR: %s' % e)
            return

        spectral_cols = [c for c in df.columns if Is_band_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns (wl_*, wn_*) found in the data.')
            return

        # Subsample rows
        n_total = len(df)
        max_spectra = int(getattr(p, 'max_spectra', 0))
        skip = max(1, ceil(n_total / max_spectra)) if max_spectra > 0 else 1
        df_sub = df.iloc[::skip].reset_index(drop=True)
        n_plot = len(df_sub)

        if self.verbose >= 1:
            print('    Plotting %d of %d spectra (%d bands, skip=%d)' % (
                n_plot, n_total, len(spectral_cols), skip))

        colormap = str(getattr(p, 'colormap', 'jet')).strip() or 'jet'
        colors = plt.get_cmap(colormap)(np.linspace(0, 1, n_plot))

        show = getattr(p, 'show', True)
        save = getattr(p, 'save', False)
        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None

        # Build spectral-only DataFrame with integer wavelength column names
        wavelengths = [Band_value(c) for c in spectral_cols]
        spec_df = df_sub[spectral_cols].copy()
        spec_df.columns = wavelengths

        # Parse filter parameters
        filter_param       = str(getattr(p, 'filter', 'none')).strip() or 'none'
        multi_filter_param = str(getattr(p, 'multi_filter', 'none')).strip() or 'none'

        # Parse chemometrics parameters
        chemometrics_param = str(getattr(p, 'chemometrics', 'default')).strip() or 'default'
        chemometrics_array = self._Parse_chemometrics_array(getattr(p, 'chemometrics_array', []))

        # Build panels list: always start with raw
        panels   = [('raw', spec_df, wavelengths)]
        base_df  = spec_df
        base_cols = wavelengths

        # Apply filter (multi_filter takes priority)
        if multi_filter_param.lower() not in ('none', 'no', ''):
            multi_cfg = _load_filter_config(multi_filter_param)
            if multi_cfg is not None:
                df_f, cols_f, flabel = apply_multi_filter(base_df, base_cols, multi_cfg)
                panels.append((flabel, df_f, cols_f))
                base_df, base_cols = df_f, cols_f
        elif filter_param.lower() not in ('none', 'no', ''):
            filter_cfg = _load_filter_config(filter_param)
            if filter_cfg is not None:
                df_f, cols_f, flabel = apply_filter(base_df, base_cols, filter_cfg)
                panels.append((flabel, df_f, cols_f))
                base_df, base_cols = df_f, cols_f

        # Append chemometric steps (apply_chemometrics prepends 'raw'; skip it)
        chem_steps = apply_chemometrics(base_df, base_cols, chemometrics_array,
                                        chemometrics_param, self.verbose)
        panels.extend(chem_steps[1:])

        bw = params_D.get('output_bandwidth', '')
        ann = 'n=%d  shown=%d' % (n_total, n_plot)
        if bw:
            ann += '  bw=%s nm' % bw

        chain_abbrevs = []
        for step_label, df_step, step_cols in panels:
            abbrev = _step_to_abbrev(step_label)
            if step_label != 'raw':
                chain_abbrevs.append(abbrev)
            chain_ann = '→'.join(chain_abbrevs) if chain_abbrevs else 'raw'
            file_suffix = '_'.join(chain_abbrevs) if chain_abbrevs else 'raw'
            self._Plot_spectra_panel(df_step, step_cols, colors, step_label, ann,
                                     chain_ann, file_suffix, show, save, plot_dir)

    def _Load_spec_df(self, p):
        '''Load parquet, subsample, and return (spec_df, wavelengths, colors, ann, params_D, show, save, plot_dir).

        spec_df has integer wavelength column names. Returns None as first element on error.
        '''
        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return None, None, None, None, None, None, None, None

        try:
            df, params_D = self._Load_parquet_data(project_root_fp)
        except FileNotFoundError as e:
            print('    ERROR: %s' % e)
            return None, None, None, None, None, None, None, None

        spectral_cols = [c for c in df.columns if Is_band_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns (wl_*, wn_*) found in the data.')
            return None, None, None, None, None, None, None, None

        n_total = len(df)
        max_spectra = int(getattr(p, 'max_spectra', 0))
        skip = max(1, ceil(n_total / max_spectra)) if max_spectra > 0 else 1
        df_sub = df.iloc[::skip].reset_index(drop=True)
        n_plot = len(df_sub)

        if self.verbose >= 1:
            print('    Loaded %d spectra, plotting %d (skip=%d)' % (n_total, n_plot, skip))

        colormap = str(getattr(p, 'colormap', 'jet')).strip() or 'jet'
        colors = plt.get_cmap(colormap)(np.linspace(0, 1, n_plot))

        bw = params_D.get('output_bandwidth', '')
        ann = 'n=%d  shown=%d' % (n_total, n_plot)
        if bw:
            ann += '  bw=%s nm' % bw

        wavelengths = [Band_value(c) for c in spectral_cols]
        spec_df = df_sub[spectral_cols].copy()
        spec_df.columns = wavelengths

        show = getattr(p, 'show', True)
        save = getattr(p, 'save', False)
        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None

        return spec_df, wavelengths, colors, ann, params_D, show, save, plot_dir

    def _Plot_spectra_comparison(self, panels, colors, ann, title, filename_stem,
                                  show, save, plot_dir):
        '''Render a vertically-stacked comparison multiplot.

        panels: list of (abbrev, label, df_out, cols_out).
        Each panel is 2 inches tall, x-axis shared, y-axis independent.
        Saved as spectra_{filename_stem}.png.
        '''
        n = len(panels)
        panel_h = 2.0
        fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(10, panel_h * n),
                                 sharex=True, sharey=False)
        if n == 1:
            axs = [axs]

        for ax, (abbrev, label, df_step, step_cols) in zip(axs, panels):
            x_vals, _ = _spectra_x_axis(step_cols)
            y_label = 'Score' if step_cols and str(step_cols[0]).startswith('pc') else self.signal_label
            for i, (_, row) in enumerate(df_step.iterrows()):
                ax.plot(x_vals, row[step_cols].values, color=colors[i], linewidth=0.5)
            ax.set_ylabel(y_label, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.text(0.98, 0.95, abbrev, transform=ax.transAxes, fontsize=8,
                    ha='right', va='top', color='gray')

        axs[-1].set_xlabel('Wavelength (nm)', fontsize=8)
        axs[0].text(0.02, 0.95, ann, transform=axs[0].transAxes, fontsize=7,
                    va='top', color='gray')
        fig.suptitle(title, fontsize=9, y=1.0)
        fig.tight_layout()

        if save:
            fpn = os.path.join(plot_dir, 'spectra_%s.png' % filename_stem)
            fig.savefig(fpn, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Saved: %s' % fpn)

        if show:
            plt.show()

        plt.close(fig)

    def _Plot_spectra_scattercorrection_scalers(self):
        '''Compare all scatter correction methods in a stacked multiplot.'''
        p = self.process_S.process.parameters
        spec_df, wavelengths, colors, ann, _, show, save, plot_dir = self._Load_spec_df(p)
        if spec_df is None:
            return

        scalers = ['l1', 'l2', 'max', 'snv', 'msc']
        panels = [('raw', 'raw', spec_df, wavelengths)]

        for sc in scalers:
            try:
                df_out, cols_out, _ = apply_scatter_correction(
                    spec_df, wavelengths, {'scaler': [sc]})
                panels.append((sc, 'scatter_correction: %s' % sc, df_out, cols_out))
            except Exception as exc:
                if self.verbose >= 1:
                    print('    Warning: scatter correction "%s" failed: %s' % (sc, exc))

        self._Plot_spectra_comparison(
            panels, colors, ann,
            title='Scatter correction comparison',
            filename_stem='compare_scattercorrection',
            show=show, save=save, plot_dir=plot_dir)

    def _Plot_spectra_scaling_methods(self):
        '''Compare all scaling methods in a stacked multiplot.'''
        p = self.process_S.process.parameters
        spec_df, wavelengths, colors, ann, _, show, save, plot_dir = self._Load_spec_df(p)
        if spec_df is None:
            return

        methods = ['meancentring', 'autoscaling', 'paretoscaling', 'poissonscaling']
        panels = [('raw', 'raw', spec_df, wavelengths)]

        for method in methods:
            abbrev = {'meancentring': 'mc', 'autoscaling': 'as',
                      'paretoscaling': 'ps', 'poissonscaling': 'poi'}[method]
            try:
                df_out, cols_out, _ = apply_scaling(
                    spec_df, wavelengths, {'method': method})
                panels.append((abbrev, 'scaling: %s' % method, df_out, cols_out))
            except Exception as exc:
                if self.verbose >= 1:
                    print('    Warning: scaling "%s" failed: %s' % (method, exc))

        self._Plot_spectra_comparison(
            panels, colors, ann,
            title='Scaling method comparison',
            filename_stem='compare_scaling',
            show=show, save=save, plot_dir=plot_dir)
