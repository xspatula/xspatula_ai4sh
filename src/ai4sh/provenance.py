'''
Created on 21 October 2026

@author: thomasgumbricht

Provenance of the process that is running: its parameters, the content of every settings file it loaded
(default/ JSON files, filter, chemometric, selector and regressor files) and the software versions.

Run_process calls Begin_provenance(process_S) before each process. Settings files are loaded with
Load_config_json, which records the file's path, sha256 and full content. Write_previous_df (and the
regression results JSON) attach Provenance_snapshot(), so a .previous_dataframe record keeps what was
actually used even if a settings file is later edited or deleted:

    "provenance": {
        "process": "spectra_scaling", "process_file": ".../scaling.json",
        "parameters": {...},
        "configs": {"/abs/path/chemometric_default.json": {"sha256": "...", "content": {...}}},
        "environment": {"python": "3.12.4", "scikit-learn": "1.5.1", ..., "git_commit": "2c9b551", "git_dirty": true}
    }
'''

# Standard library imports
import os
import sys
import json
import hashlib
import subprocess

from importlib import metadata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'matplotlib')

_STATE = {'process': None, 'process_file': None, 'parameters': {}, 'configs': {}}

_ENVIRONMENT = {}


def _Jsonable(obj):
    '''Return obj as plain JSON types (unknown types as str).'''

    return json.loads(json.dumps(obj, default=str))


def _Git_state():
    '''Return (commit, dirty) of the repository, or (None, None) if git is not available.'''

    try:

        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=_REPO_ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip() or None

        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=_REPO_ROOT,
                                capture_output=True, text=True, timeout=5).stdout

        return commit, bool(status.strip())

    except (OSError, subprocess.SubprocessError):

        return None, None


def _Environment():
    '''Python, package versions and git state (git state read once per process).'''

    if not _ENVIRONMENT:

        _ENVIRONMENT['python'] = sys.version.split()[0]

        for pkg in _PACKAGES:

            try:
                _ENVIRONMENT[pkg] = metadata.version(pkg)

            except metadata.PackageNotFoundError:
                _ENVIRONMENT[pkg] = None

    _ENVIRONMENT['git_commit'], _ENVIRONMENT['git_dirty'] = _Git_state()

    return dict(_ENVIRONMENT)


def Begin_provenance(process_S, process_file=None):
    '''Start the provenance of a new process: its name and parameters; clears the configs of the previous one.'''

    _STATE['process'] = process_S.process.process

    _STATE['process_file'] = process_file

    _STATE['parameters'] = _Jsonable(dict(process_S.process.parameters.__dict__))

    _STATE['configs'] = {}

    _STATE['environment'] = _Environment()


def Load_config_json(fpn):
    '''Load a settings JSON file and record its path, sha256 and content in the provenance.'''

    with open(fpn, 'rb') as fh:
        raw = fh.read()

    data = json.loads(raw)

    _STATE['configs'][os.path.abspath(fpn)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'content': data}

    return data


def Record_config(key, content, sha256=None, **extra):
    '''Record settings that did not come from a file (e.g. a database parameter set) in the provenance.'''

    if sha256 is None:
        sha256 = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    _STATE['configs'][key] = dict(extra, sha256=sha256, content=content)


def File_sha256(fpn, chunk=1 << 20):
    '''sha256 of a file (read in chunks).'''

    h = hashlib.sha256()

    with open(fpn, 'rb') as fh:

        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)

    return h.hexdigest()


def Provenance_snapshot():
    '''Return the provenance of the running process as a JSON-ready dict.'''

    return _Jsonable({'process': _STATE['process'], 'process_file': _STATE['process_file'],
                      'parameters': _STATE['parameters'], 'configs': _STATE['configs'],
                      'environment': _STATE.get('environment') or _Environment()})
