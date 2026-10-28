'''
Created on 9 September 2026

@author: thomasgumbricht

Round-trip per-indicator units through Parquet files.

Units are stored as custom Parquet file metadata (a JSON-encoded {column: unit} dict),
NOT as a column MultiIndex — Parquet has no native support for column MultiIndex, and
writing one (via pandas/pyarrow) silently degrades to flat columns literally named after
the str() of each tuple (e.g. "('sample_name', '')"), breaking every plain column-name
lookup downstream. File metadata keeps the on-disk columns exactly as before (flat,
single-level strings), so no other code needs to change.
'''

# Standard library imports
import os
import re
import glob
import json

# Third-party imports
import pyarrow as pa
import pyarrow.parquet as pq

# Application package imports
from src.ai4sh.previous_dataframe import Read_previous_history

_UNITS_METADATA_KEY = b'ai4sh_indicator_units'

# Any known machine learning preprocessing suffix at the end of a parquet stem (derived files).
# Scatter corrections in sequence are saved as their concatenation (snv+snv → snvsnv, l1+msc → l1msc).
DERIVED_STEM_RE = re.compile(
    r'_(?:pred|ol|vt|ma|gf|sg|lw|mf|wc|uv(?:-[\w-]+)?|d\d+(?:app)?|(?:l1|l2|max|snv|msc)+|mc|as|ps|poi|pca\d+|(?:[\w-]+-)?(?:pms|rfe|tree))$'
)


def Is_derived_parquet(parquet_fpn):
    '''True if the parquet is the output of a preprocessing step, not a selected (raw) dataset.

    A derived file has a preprocessing suffix (DERIVED_STEM_RE) or a companion JSON with a non-empty
    preprocessing_chain. The companion check catches any suffix the regex does not know.
    '''
    stem = os.path.splitext(os.path.basename(parquet_fpn))[0]
    if DERIVED_STEM_RE.search(stem):
        return True
    json_fpn = os.path.join(os.path.dirname(parquet_fpn), stem + '.json')
    if os.path.exists(json_fpn):
        try:
            with open(json_fpn) as fh:
                return bool(json.load(fh).get('preprocessing_chain'))
        except (OSError, ValueError, AttributeError):
            return False
    return False


def Find_selected_parquet(project_root_fp):
    '''Return (parquet_fpn, json_fpn or None) of the selected (not derived) dataset in project_root_fp.

    Selections are saved as {name}.parquet + {name}.json. Files with a preprocessing suffix
    (Is_derived_parquet) are skipped. The latest selected file registered in .previous_dataframe is used;
    without one, the most recently written candidate is used and the others are reported.
    Raises FileNotFoundError if there is none.
    '''
    for record in reversed(Read_previous_history(project_root_fp)):

        for entry in record.get('entries', []):

            fpn = os.path.join(project_root_fp, entry.get('dataframe', ''))

            stem = os.path.splitext(os.path.basename(fpn))[0]

            if fpn.endswith('.parquet') and os.path.exists(fpn) and not Is_derived_parquet(fpn):

                json_fpn = os.path.join(project_root_fp, stem + '.json')

                return fpn, json_fpn if os.path.exists(json_fpn) else None

    candidates = [fpn for fpn in glob.glob(os.path.join(project_root_fp, '*.parquet'))
                  if not Is_derived_parquet(fpn)]

    if not candidates:
        raise FileNotFoundError('No selected .parquet file found in: %s' % project_root_fp)

    candidates.sort(key=os.path.getmtime, reverse=True)

    if len(candidates) > 1:
        print('    ⚠️  %s selected .parquet files in %s, using the newest: %s\n'
              '        ignored (delete if outdated): %s'
              % (len(candidates), project_root_fp, os.path.basename(candidates[0]),
                 ', '.join(os.path.basename(fpn) for fpn in candidates[1:])))

    parquet_fpn = candidates[0]

    stem = os.path.splitext(os.path.basename(parquet_fpn))[0]

    json_fpn = os.path.join(project_root_fp, stem + '.json')

    if not os.path.exists(json_fpn):
        json_fpn = None

    return parquet_fpn, json_fpn


def Read_parquet_with_units(fpn, **kwargs):
    '''Read a Parquet file, returning (df, {column_name: unit}).

    Returns an empty units dict if the file has no embedded units (e.g. legacy files, or
    files not written via Save_parquet_with_units).
    '''
    table = pq.read_table(fpn, **kwargs)
    df = table.to_pandas()

    units_D = {}
    meta = table.schema.metadata or {}
    raw = meta.get(_UNITS_METADATA_KEY)
    if raw:
        try:
            units_D = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            units_D = {}

    return df, units_D


def Save_parquet_with_units(df, fpn, units_D, index=False):
    '''Write df to Parquet, embedding {column_name: unit} as custom file metadata.'''
    table = pa.Table.from_pandas(df, preserve_index=index)
    new_meta = dict(table.schema.metadata or {})
    new_meta[_UNITS_METADATA_KEY] = json.dumps(units_D or {}).encode('utf-8')
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, fpn)

