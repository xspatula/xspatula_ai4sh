'''
Created on 21 October 2026

@author: thomasgumbricht

Scikit-learn transformers for the machine learning preprocessing steps, so that a model can refit the whole
preprocessing chain on the training samples only and keep every fitted state (MSC reference, scaling
mean/scale, PCA loadings, retained bands, Box-Cox lambda ...) together with the regressor.

The exploration processes in machine_learning_preprocess.py fit each step on all samples and save a parquet
plus a companion JSON with the preprocessing_chain. Regression modeling replays that chain with
Chain_to_steps(): every step is refitted inside a Pipeline, so nothing is learned from the test samples.

Every transformer takes and returns a DataFrame. On input the column names are converted to their numeric
index (wl_nm_550 → 550, d555 → 555, pc1 → 1), the same convention as the exploration steps.

Holdout_test_mask() defines the test samples from a hash of the sample id. It does not depend on row order or on
which rows are missing for an indicator, so the supervised band selection steps and the model use the same test
samples as long as they read the same set of samples.
'''

# Standard library imports
import hashlib

import numpy as np
import pandas as pd

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                   PowerTransformer, QuantileTransformer)

from src.ai4sh.chemometrics import apply_derivative, _snv, _msc, Scaling_parameters
from src.ai4sh.filter import apply_filter, apply_multi_filter
from src.ai4sh.spectral_columns import Is_band_col, Band_value

HOLDOUT_SEED = 42

HOLDOUT_KEYS = ('sample_id', 'sample_name')


# ------------------------------------------------------------------ columns and holdout

def Column_index(col):
    '''Numeric index of a spectral column: wl_nm_550 → 550, d555 → 555, pc1 → 1, wc3 → 3, 550 → 550.'''

    if isinstance(col, (int, float, np.integer, np.floating)):
        return col

    s = str(col)

    if Is_band_col(s):
        return Band_value(s)

    if s[:2] in ('pc', 'wc'):
        return int(s[2:])

    if s.startswith('d'):
        return int(s[1:])

    return int(s)


def _Numeric_frame(X):
    '''Return X as a float DataFrame with numeric column names.'''

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X, dtype=float))

    out = X.astype(float)

    out.columns = [Column_index(c) for c in X.columns]

    return out


def Holdout_test_mask(df, test_size, seed=HOLDOUT_SEED):
    '''Boolean array, True for test samples: the round(test_size * n) samples with the lowest hash of the sample id
    (row index if there is none). Independent of row order and of which indicator values are missing.

    Returns (mask, key_column).
    '''
    key_col = next((k for k in HOLDOUT_KEYS if k in df.columns), None)

    keys = df[key_col].astype(str).tolist() if key_col else [str(i) for i in df.index]

    hashes = np.array([int(hashlib.md5(('%s:%s' % (seed, k)).encode()).hexdigest()[:12], 16) for k in keys])

    mask = np.zeros(len(keys), dtype=bool)

    n_test = int(round(float(test_size) * len(keys)))

    if n_test > 0:
        mask[np.argsort(hashes, kind='stable')[:n_test]] = True

    return mask, key_col or 'index'


def Holdout_info(df, test_size, seed=HOLDOUT_SEED):
    '''The holdout description recorded in the preprocessing_chain of supervised selection steps.'''

    _, key_col = Holdout_test_mask(df.head(1), test_size, seed)

    return {'test_size': float(test_size), 'seed': seed, 'key': key_col}


# ------------------------------------------------------------------ spectral transformers

class _Stateless:
    '''Mixin for transformers that learn nothing: always fitted (so a sliced fitted Pipeline can transform).'''

    def __sklearn_is_fitted__(self):
        return True


class BandColumns(BaseEstimator, TransformerMixin):
    '''Keep the spectral columns seen at fit, in that order, with numeric names.'''

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        self.columns_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        missing = [c for c in self.columns_in_ if c not in X.columns]
        if missing:
            raise ValueError('%d spectral band(s) missing in input, e.g. %s' % (len(missing), missing[:5]))
        return X[self.columns_in_]


class SpectralFilter(_Stateless, BaseEstimator, TransformerMixin):
    '''Single smoothing filter (moving average, gauss, Savitzky-Golay, lowess); stateless.'''

    def __init__(self, filter_cfg=None):
        self.filter_cfg = filter_cfg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        out, _, _ = apply_filter(X, list(X.columns), self.filter_cfg or {})
        return out


class MultiFilter(_Stateless, BaseEstimator, TransformerMixin):
    '''Per-region filters; bands between the regions are dropped; stateless.'''

    def __init__(self, multi_filter_cfg=None):
        self.multi_filter_cfg = multi_filter_cfg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        out, cols, _ = apply_multi_filter(X, list(X.columns), self.multi_filter_cfg)
        return out[cols]


class Derivative(_Stateless, BaseEstimator, TransformerMixin):
    '''Finite-difference derivative of order `derivative`; append=True keeps the input bands as well.'''

    def __init__(self, derivative=1, append=False):
        self.derivative = derivative
        self.append = append

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        out, _, _ = apply_derivative(X, list(X.columns), {'derivative': int(self.derivative)})
        if self.append:
            out = pd.concat([X, out], axis=1)
        return out


class ScatterCorrection(BaseEstimator, TransformerMixin):
    '''l1 / l2 / max / snv / msc in series; msc keeps the mean training spectrum as reference.'''

    def __init__(self, scaler=('snv',)):
        self.scaler = scaler

    def _Scalers(self):
        return [self.scaler] if isinstance(self.scaler, str) else list(self.scaler)

    def _Run(self, X, fit):
        from sklearn.preprocessing import normalize
        cols, idx = list(X.columns), X.index
        A = X.values.astype(float)
        if fit:
            self.references_ = []
        n_msc = 0
        for sc in self._Scalers():
            if sc in ('l1', 'l2', 'max'):
                A = normalize(A, norm=sc)
            elif sc == 'snv':
                A = _snv(A)
            elif sc == 'msc':
                if fit:
                    A, ref = _msc(A)
                    self.references_.append(ref)
                else:
                    A, _ = _msc(A, self.references_[n_msc])
                n_msc += 1
        return pd.DataFrame(A, columns=cols, index=idx)

    def fit(self, X, y=None):
        self._Run(_Numeric_frame(X), fit=True)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self._Run(_Numeric_frame(X), fit=True)

    def transform(self, X):
        return self._Run(_Numeric_frame(X), fit=False)


class SpectralScaling(BaseEstimator, TransformerMixin):
    '''meancentring | autoscaling | paretoscaling | poissonscaling, mean and scale per band from training.'''

    def __init__(self, method='meancentring'):
        self.method = method

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        self.mean_, self.scale_ = Scaling_parameters(X.values, self.method)
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        return pd.DataFrame((X.values - self.mean_) / self.scale_, columns=X.columns, index=X.index)


class Decomposition(BaseEstimator, TransformerMixin):
    '''PCA; output columns pc1 .. pcn.'''

    def __init__(self, method='pca', n_components=5):
        self.method = method
        self.n_components = n_components

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        n = min(int(self.n_components), X.shape[0], X.shape[1])
        self.pca_ = PCA(n_components=n).fit(X.values)
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        A = self.pca_.transform(X.values)
        return pd.DataFrame(A, columns=['pc%d' % (i + 1) for i in range(A.shape[1])], index=X.index)


class VarianceThresholdSelect(BaseEstimator, TransformerMixin):
    '''Drop bands whose variance (after minmax / standard / robust scaling) is below threshold.'''

    def __init__(self, scaler='minmax', threshold=0.025):
        self.scaler = scaler
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        scl = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}[self.scaler]
        support = VarianceThreshold(threshold=self.threshold).fit(scl.fit_transform(X.values)).get_support()
        self.columns_out_ = [c for c, keep in zip(X.columns, support) if keep]
        return self

    def transform(self, X):
        return _Numeric_frame(X)[self.columns_out_]


class WardAgglomeration(BaseEstimator, TransformerMixin):
    '''Ward feature agglomeration; output columns wc1 .. wcn.'''

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        n = min(int(self.n_clusters), X.shape[1] - 1)
        self.ward_ = FeatureAgglomeration(n_clusters=n, linkage='ward').fit(X.values)
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        A = self.ward_.transform(X.values)
        return pd.DataFrame(A, columns=['wc%d' % (i + 1) for i in range(A.shape[1])], index=X.index)


class UnivariateSelect(BaseEstimator, TransformerMixin):
    '''SelectKBest(f_regression) refitted on the training samples of the modelled indicator.'''

    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        k = min(int(self.k), X.shape[1])
        support = SelectKBest(f_regression, k=k).fit(X.values, np.ravel(y)).get_support()
        self.columns_out_ = [c for c, keep in zip(X.columns, support) if keep]
        return self

    def transform(self, X):
        return _Numeric_frame(X)[self.columns_out_]


class FixedBands(BaseEstimator, TransformerMixin):
    '''Keep a band selection made in an earlier (exploration) step, matched on the numeric band index.'''

    def __init__(self, bands=()):
        self.bands = bands

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        wanted = {float(b) for b in self.bands}
        self.columns_out_ = [c for c in X.columns if float(c) in wanted]
        if len(self.columns_out_) != len(wanted):
            raise ValueError('FixedBands: %d of %d selected bands not found in the input'
                             % (len(wanted) - len(self.columns_out_), len(wanted)))
        return self

    def transform(self, X):
        return _Numeric_frame(X)[self.columns_out_]


class RegressorFrame(_Stateless, BaseEstimator, TransformerMixin):
    '''Last step before the regressor: string column names f_<index> (cubist needs string names).'''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        return X.set_axis(['f_%s' % c for c in X.columns], axis=1)


# ------------------------------------------------------------------ indicator (target) transformer

TARGET_TRANSFORMS = ('linear', 'log', 'sqrt', 'reciprocal', 'boxcox', 'yeojohnson', 'quantile')


def Target_domain_mask(y, transform):
    '''True where y can be transformed (log/boxcox: > 0, sqrt: >= 0, reciprocal: != 0).'''

    y = np.asarray(y, dtype=float)

    if transform in ('log', 'boxcox'):
        return y > 0

    if transform == 'sqrt':
        return y >= 0

    if transform == 'reciprocal':
        return y != 0

    return np.ones(len(y), dtype=bool)


class IndicatorTransformer(BaseEstimator, TransformerMixin):
    '''Transform (log, sqrt, reciprocal, boxcox, yeojohnson, quantile) and/or z-score standardise the indicator.

    Used as the transformer of a TransformedTargetRegressor: fitted on the training values only, and
    inverse_transform returns predictions in the native unit of the indicator.
    '''

    def __init__(self, method='linear', standardise=False):
        self.method = method
        self.standardise = standardise

    def _Forward(self, y):
        t = self.method
        if t == 'log':
            return np.log(y)
        if t == 'sqrt':
            return np.sqrt(y)
        if t == 'reciprocal':
            return 1.0 / y
        if t == 'boxcox':
            return boxcox(y, lmbda=self.lambda_)
        if t in ('yeojohnson', 'quantile'):
            return self.transformer_.transform(y.reshape(-1, 1)).ravel()
        return y

    def fit(self, y, _=None):
        y = np.asarray(y, dtype=float).ravel()
        if self.method not in TARGET_TRANSFORMS:
            raise ValueError('unknown indicator transformation: %s' % self.method)
        if self.method == 'boxcox':
            _, self.lambda_ = boxcox(y)
        elif self.method == 'yeojohnson':
            self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=False).fit(y.reshape(-1, 1))
        elif self.method == 'quantile':
            self.transformer_ = QuantileTransformer(output_distribution='normal', random_state=0,
                                                    n_quantiles=min(1000, len(y))).fit(y.reshape(-1, 1))
        t = self._Forward(y)
        self.mean_, self.std_ = (float(np.mean(t)), float(np.std(t))) if self.standardise else (0.0, 1.0)
        if self.std_ == 0:
            self.std_ = 1.0
        return self

    def transform(self, y):
        shape = np.shape(y)
        t = self._Forward(np.asarray(y, dtype=float).ravel())
        return ((t - self.mean_) / self.std_).reshape(shape)

    def inverse_transform(self, t):
        shape = np.shape(t)
        t = np.asarray(t, dtype=float).ravel() * self.std_ + self.mean_
        tr = self.method
        if tr == 'log':
            y = np.exp(t)
        elif tr == 'sqrt':
            y = np.square(t)
        elif tr == 'reciprocal':
            y = 1.0 / t
        elif tr == 'boxcox':
            y = inv_boxcox(t, self.lambda_)
        elif tr in ('yeojohnson', 'quantile'):
            y = self.transformer_.inverse_transform(t.reshape(-1, 1)).ravel()
        else:
            y = t
        return y.reshape(shape)

    def Describe(self):
        '''Fitted parameters as a JSON-ready dict.'''
        d = {'transform': self.method, 'standardise': bool(self.standardise),
             'mean': getattr(self, 'mean_', None), 'std': getattr(self, 'std_', None)}
        if self.method == 'boxcox' and hasattr(self, 'lambda_'):
            d['lambda'] = float(self.lambda_)
        if self.method == 'yeojohnson' and hasattr(self, 'transformer_'):
            d['lambda'] = float(self.transformer_.lambdas_[0])
        return d


def Indicator_target_settings(indicator, transformation_raw, standardise_raw):
    '''(transform, standardise) for indicator from loaded transformation / standardise JSON dicts (or None).'''

    transform = 'linear'

    if transformation_raw:
        col_cfg = transformation_raw.get('transformation', {}).get(indicator, {})
        transform = next((t for t in TARGET_TRANSFORMS[1:] if col_cfg.get(t)), 'linear')

    standardise = bool(standardise_raw.get('standardise', {}).get(indicator, False)) if standardise_raw else False

    return transform, standardise


# ------------------------------------------------------------------ preprocessing_chain → pipeline steps

_SUPERVISED = ('spectra_indicator_univariate_selection', 'spectra_indicator_permutation_selection')


def Chain_to_steps(chain, indicator, test_size):
    '''Pipeline steps [(name, transformer)] that refit the preprocessing_chain of a companion JSON.

    indicator: the indicator being modelled (univariate selection for that indicator is refitted).
    test_size: the model's holdout; a fixed (supervised) band selection made on another holdout is flagged.

    Returns (steps, notes); notes is a list of strings describing anything that is not refitted.
    '''
    steps = [('bands', BandColumns())]
    notes = []

    for i, entry in enumerate(chain):
        proc = entry.get('process', '')
        par = entry.get('parameters', {})
        name = '%02d_%s' % (i + 1, entry.get('id', proc))

        if proc == 'detect_outliers':
            notes.append('%s: indicator outliers were set to missing using all samples (not refitted)' % name)
            continue

        if proc == 'select_variance_threshold':
            step = VarianceThresholdSelect(par.get('scaler', 'minmax'), float(par.get('threshold', 0.025)))

        elif proc == 'filter_spectra':
            if 'filter' not in entry:
                raise ValueError('%s: the filter settings are not recorded in the chain — rerun filter_spectra' % name)
            step = SpectralFilter(entry['filter'])

        elif proc in ('multi_filter_spectra', 'mulit_filter_spectra'):
            if 'multi_filter' not in entry:
                raise ValueError('%s: the multi-filter settings are not recorded in the chain — rerun it' % name)
            step = MultiFilter(entry['multi_filter'])

        elif proc == 'spectra_derivative':
            step = Derivative(int(par.get('derivative', 1)), bool(par.get('append', False)))

        elif proc == 'spectra_scatter_correction':
            step = ScatterCorrection(par.get('scaler', ['snv']))

        elif proc == 'spectra_scaling':
            step = SpectralScaling(par.get('method', 'meancentring'))

        elif proc == 'spectra_decomposition':
            step = Decomposition(par.get('method', 'pca'), int(par.get('n_components', 5)))

        elif proc == 'ward_clustering':
            step = WardAgglomeration(int(par['n_clusters_applied']))

        elif proc in _SUPERVISED:
            if proc == 'spectra_indicator_univariate_selection' and par.get('indicator') == indicator:
                step = UnivariateSelect(int(par.get('n_features_selected', 20)))
            else:
                if 'selected_bands' not in entry:
                    raise ValueError('%s: the selected bands are not recorded in the chain — rerun the selection' % name)
                step = FixedBands(entry['selected_bands'])
                holdout = entry.get('holdout')
                if not holdout or abs(holdout.get('test_size', -1) - test_size) > 1e-9:
                    notes.append('%s: band selection was made with test samples included (holdout %s, model test_size %s)'
                                 % (name, holdout.get('test_size') if holdout else 'none', test_size))
                else:
                    notes.append('%s: band selection fixed from the training samples (k-fold metrics still include it)' % name)

        else:
            raise ValueError('%s: process "%s" can not be replayed in a pipeline' % (name, proc))

        steps.append((name, step))

    return steps, notes


def Output_columns(preprocessing, X_sample):
    '''Numeric names of the features that reach the regressor (preprocessing = fitted Pipeline without regressor).'''

    return list(preprocessing.transform(X_sample).columns)
