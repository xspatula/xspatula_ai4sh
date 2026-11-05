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

Covariates (other than spectra) enter a model through CovariateBlocks: the spectral columns pass through the
refitted chain, numeric covariates are kept as they are and nominal covariates (classes, e.g. soil texture) are
one-hot encoded, optionally after pooling classes into groups (class_grouping). Numeric covariates are standardised
with the mean and standard deviation of the training samples, so a model does not depend on their unit (% or g/kg)
and a high-variance covariate does not dominate a plsr component or a knn/svr/mlp distance. Covariate columns are named
cv:<label> and cv:<label>=<class>; Column_index leaves these names as they are.

CappedPLSRegression is the plsr regressor: n_components is capped at fit to what the training samples and the bands
left after the chain allow, the rank of the centred X (at most n_samples - 1), instead of failing. It prints nothing
(k-fold and RFE fit it many times); Pls_cap_text() gives the note that regression modeling prints per indicator and
mode.
'''

# Standard library imports
import re
import json
import hashlib

import numpy as np
import pandas as pd

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import FeatureAgglomeration
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                   PowerTransformer, QuantileTransformer)

from src.ai4sh.chemometrics import apply_derivative, _snv, _msc, _detrend, Scaling_parameters
from src.ai4sh.filter import apply_filter, apply_multi_filter
from src.ai4sh.spectral_columns import Is_band_col, Band_value

HOLDOUT_SEED = 42

HOLDOUT_KEYS = ('sample_id', 'sample_name')

# Prefix of the covariate columns that CovariateBlocks adds to the spectral features
COVARIATE_PREFIX = 'cv:'


# ------------------------------------------------------------------ columns and holdout

def Column_index(col):
    '''Numeric index of a spectral column: wl_nm_550 → 550, d555 → 555, pc1 → 1, wc3 → 3, 550 → 550.
    Covariate columns (cv:<label>, cv:<label>=<class>) are returned as they are.'''

    if isinstance(col, (int, float, np.integer, np.floating)):
        return col

    s = str(col)

    if s.startswith(COVARIATE_PREFIX):
        return s

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
    '''l1 / l2 / max / snv / msc / detrend in series; msc keeps the mean training spectrum as reference.'''

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
            elif sc == 'detrend':
                A = _detrend(A)
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
    '''Drop bands whose variance (after minmax scaling) is below threshold; scaler is kept for models fitted before
    2026-10-31 (standard / robust).'''

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


def Ward_fit(X, n_clusters, connectivity=False):
    '''Fit Ward feature agglomeration on X (samples × bands, bands in wavelength order).

    connectivity=True only merges neighbouring bands, so every cluster is a contiguous wavelength range.
    Returns (ward, order): order lists the sklearn cluster labels sorted by the mean band position of their
    members, so that wc1 is the cluster at the shortest wavelengths.
    '''
    conn = None
    if connectivity:
        from scipy.sparse import diags
        n_bands = X.shape[1]
        conn = diags([np.ones(n_bands - 1), np.ones(n_bands - 1)], [-1, 1])
    ward = FeatureAgglomeration(n_clusters=n_clusters, linkage='ward', connectivity=conn).fit(X)
    positions = np.arange(X.shape[1])
    order = np.argsort([positions[ward.labels_ == k].mean() for k in range(ward.n_clusters_)])
    return ward, order


class WardAgglomeration(BaseEstimator, TransformerMixin):
    '''Ward feature agglomeration; output columns wc1 .. wcn, ordered by wavelength.'''

    def __init__(self, n_clusters=10, connectivity=False):
        self.n_clusters = n_clusters
        self.connectivity = connectivity

    def fit(self, X, y=None):
        X = _Numeric_frame(X)
        n = min(int(self.n_clusters), X.shape[1] - 1)
        self.ward_, self.order_ = Ward_fit(X.values, n, getattr(self, 'connectivity', False))
        return self

    def transform(self, X):
        X = _Numeric_frame(X)
        A = self.ward_.transform(X.values)
        # Models saved before cluster ordering was added have no order_
        order = getattr(self, 'order_', None)
        if order is not None:
            A = A[:, order]
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
        return X.set_axis(['f_%s' % (re.sub(r'[^0-9A-Za-z]+', '_', c) if isinstance(c, str) else c)
                           for c in X.columns], axis=1)


class CappedPLSRegression(PLSRegression):
    '''PLSRegression with n_components capped at fit to the rank of the centred X (at most n_samples - 1).

    The rank, not the number of features: the one-hot columns of a nominal covariate sum to 1, so after centring they
    lose one dimension. A component beyond the rank is numerical noise and gives wild coefficients.
    The requested n_components stays the parameter (clone, get_params, manifest); the number used is n_components_,
    fit_shape_ the (samples, features) it was fitted on and fit_rank_ the rank of the centred X.
    '''

    def fit(self, X, y=None, **kwargs):
        requested = self.n_components
        self.fit_shape_ = tuple(int(n) for n in np.shape(X)[:2])
        Xa = np.asarray(X, dtype=float)
        self.fit_rank_ = int(np.linalg.matrix_rank(Xa - Xa.mean(axis=0))) if Xa.size else 0
        cap = max(1, min(self.fit_shape_[0] - 1, self.fit_rank_))   # centring takes one degree of freedom
        self.n_components_ = min(requested, cap)
        self.n_components = self.n_components_
        try:
            return super().fit(X, y, **kwargs)
        finally:
            self.n_components = requested


def Pls_cap_text(regressor):
    '''"n_components 10 → 5 (120 samples, 5 features)" if a fitted CappedPLSRegression was capped, else "".
    The rank is added when it is below the number of features, e.g. "(130 samples, 8 features, rank 7)".'''
    used = getattr(regressor, 'n_components_', None)
    if used is None or used >= regressor.n_components:
        return ''
    n, p = regressor.fit_shape_
    rank = getattr(regressor, 'fit_rank_', p)
    return 'n_components %d → %d (%d samples, %d features%s)' % (
        regressor.n_components, used, n, p, ', rank %d' % rank if rank < p else '')


# ------------------------------------------------------------------ covariates

def Parse_class_grouping(raw):
    '''class_grouping parameter → {covariate label: {class: group name}}. Raises ValueError if invalid.

    raw: array of objects {"covariate": <label>, "groups": [[class, class ...], [class ...] ...]} (a JSON string of
    that array is also accepted). A group is named by its classes joined with "|".
    '''
    if hasattr(raw, '__dict__') and not isinstance(raw, type):
        raw = vars(raw)

    if isinstance(raw, str):
        s = raw.strip()
        while len(s) >= 2 and s[0] + s[-1] == '{}' and not s.startswith('{"'):
            s = s[1:-1].strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError('class_grouping is not valid JSON: %s' % e)

    if isinstance(raw, dict):
        raw = [raw]

    grouping = {}

    for item in raw or []:
        if hasattr(item, '__dict__') and not isinstance(item, type):
            item = vars(item)
        if not isinstance(item, dict) or 'covariate' not in item or 'groups' not in item:
            raise ValueError('class_grouping items must be {"covariate": <label>, "groups": [[classes] ...]}, got %s'
                             % (item,))
        class_D = {}
        for group in item['groups']:
            members = [str(c) for c in (group if isinstance(group, (list, tuple)) else [group])]
            for c in members:
                if c in class_D:
                    raise ValueError('class_grouping %s: class "%s" in more than one group' % (item['covariate'], c))
                class_D[c] = '|'.join(members)
        grouping[str(item['covariate'])] = class_D

    return grouping


def Group_classes(values, class_D=None):
    '''Class values as strings (missing → None), pooled into groups by class_D {class: group}; a class not in
    class_D is kept as it is.'''

    out = pd.Series(values).astype(object)

    out = out.where(out.notna(), None).map(lambda v: None if v is None else str(v))

    return out.map(lambda v: v if v is None or not class_D else class_D.get(v, v))


class CovariateBlocks(BaseEstimator, TransformerMixin):
    '''Spectra + covariates as one feature frame.

    spectra: Pipeline refitting the preprocessing chain, applied to every column that is not a covariate.
    numeric: covariates standardised with the mean and standard deviation of the training samples (cv:<label>);
    models saved before the standardisation have no numeric_mean_ and keep the values as they are.
    nominal: class covariates, pooled by grouping {label: {class: group}} and one-hot encoded with the classes of
    the training samples (cv:<label>=<class>). A class not seen in training gives all zeros; the model and predict
    processes skip such samples.
    '''

    def __init__(self, spectra=None, numeric=(), nominal=(), grouping=None):
        self.spectra = spectra
        self.numeric = numeric
        self.nominal = nominal
        self.grouping = grouping

    def _Classes(self, X, label):
        return Group_classes(X[label].values, (self.grouping or {}).get(label)).values

    def fit(self, X, y=None):
        covariates = list(self.numeric) + list(self.nominal)
        self.spectral_columns_ = [c for c in X.columns if c not in covariates]
        self.spectra_ = clone(self.spectra).fit(X[self.spectral_columns_], y)
        self.numeric_mean_, self.numeric_std_ = {}, {}
        for label in self.numeric:
            values = X[label].astype(float).values
            self.numeric_mean_[label] = float(np.nanmean(values))
            self.numeric_std_[label] = float(np.nanstd(values))   # 0: constant, only centred (Constant_covariates)
        self.levels_ = {label: sorted({c for c in self._Classes(X, label) if c is not None})
                        for label in self.nominal}
        return self

    def transform(self, X):
        parts = [self.spectra_.transform(X[self.spectral_columns_]).set_axis(X.index, axis=0)]
        if len(self.numeric):
            mean, std = getattr(self, 'numeric_mean_', {}), getattr(self, 'numeric_std_', {})
            parts.append(pd.DataFrame({COVARIATE_PREFIX + label: (X[label].astype(float).values - mean.get(label, 0.0))
                                       / (std.get(label, 1.0) or 1.0) for label in self.numeric}, index=X.index))
        for label in self.nominal:
            classes = self._Classes(X, label)
            parts.append(pd.DataFrame({'%s%s=%s' % (COVARIATE_PREFIX, label, c): (classes == c).astype(float)
                                       for c in self.levels_[label]}, index=X.index))
        return pd.concat(parts, axis=1)

    def Constant_covariates(self):
        '''Covariates that did not vary in the training samples (numeric std 0, nominal one class): no effect.'''
        return ([label for label, std in getattr(self, 'numeric_std_', {}).items() if std == 0]
                + [label for label in self.nominal if len(self.levels_[label]) < 2])

    def Unseen(self, X):
        '''Boolean array, True for rows with a missing covariate or a class (after grouping) not seen in training.'''
        bad = np.zeros(len(X), dtype=bool)
        for label in self.numeric:
            bad |= pd.isna(X[label]).values
        for label in self.nominal:
            levels = set(self.levels_[label])
            bad |= np.array([c is None or c not in levels for c in self._Classes(X, label)], dtype=bool)
        return bad


def Model_blocks(model):
    '''The fitted CovariateBlocks step of a saved model, or None for a spectra-only model.'''

    pipe = getattr(model, 'regressor_', model)

    return getattr(pipe, 'named_steps', {}).get('blocks')


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

        if proc == 'indicator_range':   # fixed limits, nothing learned from the samples
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
            step = WardAgglomeration(int(par['n_clusters_applied']), bool(par.get('connectivity', False)))

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
