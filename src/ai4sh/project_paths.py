'''
Created on 19 October 2026

@author: thomasgumbricht

Paths to the project folder of the repository (e.g. lucas/ in xspatula_lucas, ai4sh/ in xspatula_ai4sh), so that
the modules under src/ are identical in every repository built on this package.

The project folder is the one top level folder of the repository that holds a default/ folder (default settings
for plotting, transformation, chemometrics, filters, regressors and selectors).
'''

# Standard library imports
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _Find_project_folder():
    '''Return the name of the top level folder of the repository that holds default/.'''

    candidates = sorted(d for d in os.listdir(REPO_ROOT)
                        if not d.startswith('.') and os.path.isdir(os.path.join(REPO_ROOT, d, 'default')))

    if len(candidates) != 1:

        raise RuntimeError('Expected one project folder with a default/ folder in %s, found: %s'
                           % (REPO_ROOT, candidates or 'none'))

    return candidates[0]


PROJECT_FOLDER = _Find_project_folder()

PROJECT_FP = os.path.join(REPO_ROOT, PROJECT_FOLDER)


def Default_fpn(*parts):
    '''Return the path of a file under the project's default/ folder, e.g. Default_fpn('plot', 'x.json').'''

    return os.path.join(PROJECT_FP, 'default', *parts)
