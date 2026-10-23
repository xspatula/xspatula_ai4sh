'''
Created on 8 September 2026

@author: thomasgumbricht

Shared loader for <project>/default/plot/targetfeaturesymbols.json (see project_paths.py), used by both
src/ai4sh/select.py (to know which unit each indicator must be translated into
before saving a Parquet subset) and src/ai4sh/plot.py (to label plot axes).
'''

# Standard library imports
import os
import json
from src.ai4sh.project_paths import Default_fpn
from src.ai4sh.ml_setting import Load_settings

DEFAULT_SYMBOLS_FPN = Default_fpn('plot', 'targetfeaturesymbols.json')


def Load_feature_symbols(raw='default', verbose=0):
    '''Load the target feature symbols. Returns the inner dict keyed by indicator name.

    "default" (or empty) → the plot_symbol parameter set "targetfeaturesymbols" (ml_setting; DEFAULT_SYMBOLS_FPN until
    set up). Any other value is a parameter set name or a file path.
    '''
    try:
        data = Load_settings('plot_symbol', raw or 'default', DEFAULT_SYMBOLS_FPN,
                             default_name='targetfeaturesymbols', verbose=verbose) or {}
    except (FileNotFoundError, ValueError) as e:
        if verbose >= 1:
            print('    Warning: %s — using fallback labels.' % e)
        return {}

    return data.get('targetFeatureSymbols', {})


def Load_target_units(raw='default', verbose=0):
    '''Return {indicator_name: unit} from targetfeaturesymbols JSON, skipping indicators with no unit set.'''

    symbols = Load_feature_symbols(raw, verbose)

    return {indicator: sym['unit'] for indicator, sym in symbols.items() if sym.get('unit')}
