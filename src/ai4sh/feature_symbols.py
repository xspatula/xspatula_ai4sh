'''
Created on 8 September 2026

@author: thomasgumbricht

Shared loader for ai4sh/default/plot/targetfeaturesymbols.json, used by both
src/ai4sh/select.py (to know which unit each indicator must be translated into
before saving a Parquet subset) and src/ai4sh/plot.py (to label plot axes).
'''

# Standard library imports
import os
import json

# Repo root: src/ai4sh/feature_symbols.py → up 2 levels
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DEFAULT_SYMBOLS_FPN = os.path.join(_REPO_ROOT, 'ai4sh', 'default', 'plot', 'targetfeaturesymbols.json')


def Load_feature_symbols(raw='default', verbose=0):
    '''Load targetfeaturesymbols JSON. Returns the inner dict keyed by indicator name.

    "default" (or empty) → loads DEFAULT_SYMBOLS_FPN. Any other value is treated as a file path.
    '''
    if not raw or str(raw).strip().lower() == 'default':
        fpn = DEFAULT_SYMBOLS_FPN
    else:
        fpn = str(raw).strip()

    if not os.path.exists(fpn):
        if verbose >= 1:
            print('    Warning: targetfeaturesymbols file not found: %s — using fallback labels.' % fpn)
        return {}

    with open(fpn) as f:
        data = json.load(f)

    return data.get('targetFeatureSymbols', {})


def Load_target_units(raw='default', verbose=0):
    '''Return {indicator_name: unit} from targetfeaturesymbols JSON, skipping indicators with no unit set.'''

    symbols = Load_feature_symbols(raw, verbose)

    return {indicator: sym['unit'] for indicator, sym in symbols.items() if sym.get('unit')}
