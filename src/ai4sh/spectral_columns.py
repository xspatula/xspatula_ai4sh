'''
Created on 25 September 2026

@author: thomasgumbricht

Naming convention for spectral band columns and their signal type.

Header row 1 (the flat Parquet column name): {domain}_{unit}_{value}
    wl_nm_1350      wavelength in nanometre
    wl_um_2.5       wavelength in micrometre
    wn_cm-1_4000    wavenumber in reciprocal centimetre

Header row 2 (the unit slot in the Parquet units metadata, see parquet_units.py) holds the
signal type of every spectral column: refl, absorb, transm, flrs, libs or raman.

Two spectral datasets can only be joined or used together when both header rows are
identical.
'''

# Standard library imports
import re

SIGNAL_TYPES = ('refl', 'absorb', 'transm', 'flrs', 'libs', 'raman')

DEFAULT_SIGNAL_TYPE = 'refl'

# Long names accepted as input (e.g. the output_unit process parameter)
_SIGNAL_TYPE_ALIASES = {
    'reflectance': 'refl',
    'absorbance': 'absorb',
    'transmittance': 'transm',
    'fluorescence': 'flrs',
}

SIGNAL_TYPE_LABELS = {
    'refl': 'Reflectance',
    'absorb': 'Absorbance',
    'transm': 'Transmittance',
    'flrs': 'Fluorescence',
    'libs': 'LIBS intensity',
    'raman': 'Raman intensity',
}

_BAND_COL_RE = re.compile(r'^(wl|wn)_(nm|um|cm-1)_(\d+(?:\.\d+)?)$')


def Normalize_signal_type(signal_type):
    '''Return the short signal type code; raise ValueError if unknown.'''
    s = str(signal_type).strip().lower()
    s = _SIGNAL_TYPE_ALIASES.get(s, s)
    if s not in SIGNAL_TYPES:
        raise ValueError('unknown signal type "%s", use one of %s' % (signal_type, SIGNAL_TYPES))
    return s


def Band_col_name(value, domain='wl', unit='nm'):
    '''Return the column name for one spectral band, e.g. Band_col_name(1350) -> "wl_nm_1350".'''
    v = float(value)
    v_str = '%d' % v if v.is_integer() else ('%f' % v).rstrip('0')
    return '%s_%s_%s' % (domain, unit, v_str)


def Parse_band_col(col):
    '''Return (domain, unit, value) for a band column, or None if col is not one.'''
    m = _BAND_COL_RE.match(str(col))
    if not m:
        return None
    v = float(m.group(3))
    return m.group(1), m.group(2), int(v) if v.is_integer() else v


def Is_band_col(col):
    return bool(_BAND_COL_RE.match(str(col)))


def Band_value(col):
    '''Return the numeric wavelength/wavenumber of a band column (int when integral).'''
    parsed = Parse_band_col(col)
    if parsed is None:
        raise ValueError('not a spectral band column: %s' % col)
    return parsed[2]


def Signal_type_of(units_D, cols):
    '''Return the signal type recorded for cols in units_D (DEFAULT_SIGNAL_TYPE if none).

    Raises ValueError if the columns carry more than one signal type.
    '''
    found = {units_D.get(c) for c in cols if units_D.get(c) in SIGNAL_TYPES}
    if len(found) > 1:
        raise ValueError('spectral columns carry mixed signal types: %s' % sorted(found))
    return found.pop() if found else DEFAULT_SIGNAL_TYPE


def Set_signal_type(units_D, cols, signal_type):
    '''Return a copy of units_D with signal_type as the unit of every column in cols.'''
    signal_type = Normalize_signal_type(signal_type)
    out = dict(units_D or {})
    for c in cols:
        out[str(c)] = signal_type
    return out


def Header_mismatches(cols_a, units_a, cols_b, units_b):
    '''Compare the two header rows of two spectral column sets.

    Returns a list of human-readable mismatches; an empty list means the data can be joined.
    '''
    issues = []
    only_a = [c for c in cols_a if c not in set(cols_b)]
    only_b = [c for c in cols_b if c not in set(cols_a)]
    if only_a or only_b:
        issues.append('header row 1 differs: %d columns only in A, %d only in B'
                      % (len(only_a), len(only_b)))
    sig_a = {units_a.get(c, '') for c in cols_a}
    sig_b = {units_b.get(c, '') for c in cols_b}
    if sig_a != sig_b:
        issues.append('header row 2 differs: %s vs %s' % (sorted(sig_a), sorted(sig_b)))
    return issues
