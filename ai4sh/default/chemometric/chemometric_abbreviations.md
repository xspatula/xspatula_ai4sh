# Chemometric Abbreviations

Abbreviations added by the machine learning pre-processes to the names of the saved dataframes (parquet + companion JSON), and used in spectral plot annotations (upper-right corner) and plot file names.

Every pre-process adds its abbreviation to the name of its input dataframe, separated by a dot (`.`); a dot is safe on every platform. Steps are shown as a chain separated by `→` in plot annotations.

**Example** — scatter correction (snv), Savitzky-Golay derivative, meancentring, Ward clustering and PCA, run one by one or as one `chemometric_chain`:
- `1350-2500_10.parquet` — the selected (raw) dataset
- `1350-2500_10.snv.parquet`
- `1350-2500_10.snv.sg5.d1.parquet`
- `1350-2500_10.snv.sg5.d1.mc.parquet`
- `1350-2500_10.snv.sg5.d1.mc.wc40.pca5.parquet` — annotation: `snv→sg5.d1→mc→wc40→pca5`

**Plot example** — `plot_spectra` with `chemometrics_array: ["snv", "meancentring"]`:
- `spectra_raw.png` — annotation: `raw`
- `spectra_snv.png` — annotation: `snv`
- `spectra_snv.mc.png` — annotation: `snv→mc`

## Abbreviation table

`{n}` is a number taken from the step settings: decimals are written with `p` instead of a dot (sigma 1.5 → `gf1p5`), and a leading `0.` is dropped (threshold 0.028 → `vtm028`).

| Abbreviation | Method | Category |
|---|---|---|
| `raw` | Raw (unprocessed) spectra (plot annotation only) | — |
| `olif` | Isolation forest outlier detection | Indicator outliers |
| `olee` | Elliptic envelope outlier detection | Indicator outliers |
| `ollof` | Local outlier factor outlier detection | Indicator outliers |
| `ol1csvm` | One-class SVM outlier detection | Indicator outliers |
| `ma{n}` | Moving average, n = kernel length (e.g. `ma5`) | Filter |
| `gf{n}` | Gaussian filter, n = sigma (e.g. `gf2`, `gf1p5`) | Filter |
| `sg{n}` | Savitzky-Golay smoothing, n = window length (e.g. `sg7`) | Filter |
| `sg{n}.d1`, `sg{n}.d2` | Savitzky-Golay derivative (deriv 1 or 2) | Filter / derivative |
| `lw{n}` | LOWESS, n = number of points (e.g. `lw7`) | Filter |
| `mf` | Multi-filter (region-wise) | Filter |
| `l1` | L1 norm (sum normalisation) | Scatter correction |
| `l2` | L2 norm (Euclidean normalisation) | Scatter correction |
| `max` | Max norm (peak normalisation) | Scatter correction |
| `snv` | Standard Normal Variate | Scatter correction |
| `msc` | Multiplicative Scatter Correction | Scatter correction |
| `snvsnv`, `l1snv` ... | Scatter corrections in sequence (concatenated) | Scatter correction |
| `d1`, `d2` ... | First, second ... order finite-difference derivative | Derivative |
| `d1app` ... | Derivative appended to the original bands | Derivative |
| `mc` | Meancentring | Variable scaling |
| `as` | Autoscaling (meancentring + unit variance) | Variable scaling |
| `ps` | Pareto scaling (divide by √std) | Variable scaling |
| `poi` | Poisson scaling (divide by √mean) | Variable scaling |
| `vt{s}{n}` | Variance threshold, s = scaler (`m` minmax, `s` standard, `r` robust), n = threshold (e.g. `vtm028`) | Band reduction |
| `wc{n}` | Ward agglomerative clustering, n = number of clusters (e.g. `wc40`) | Band reduction |
| `pca{n}` | PCA, n = number of components (e.g. `pca5`) | Decomposition |
| `uv` | Univariate selection — combined across indicators | Feature selection |
| `uv-{indicator}` | Univariate selection for one indicator (spaces → hyphens) | Feature selection |
| `{indicator}-{regressor}-pms` | Permutation importance selection | Feature selection |
| `{indicator}-{regressor}-rfe` | Recursive feature elimination | Feature selection |
| `{indicator}-{regressor}-tree` | Tree-based importance selection | Feature selection |

## User name extension

Outlier detection, filters, the multi-filter, variance threshold, Ward clustering and the chemometric chain accept the parameter `name_extension` (1-6 letters, digits or `-`), added after a hashtag: `1350-2500_10.ma5#12321` for a moving average with the kernel weights [1, 2, 3, 2, 1]. Use it to tell apart outputs whose settings the abbreviation does not show.

## Chaining rules

Chained scatter corrections (two or more in the `scaler` list) concatenate their abbreviations without separator, e.g. `snv` + `snv` → `snvsnv`.

Every other step adds its own abbreviation after a dot, applied left-to-right; each step uses the previous step's output as input. A `chemometric_chain` gives the same name as running its steps one by one.

Files written before 2026-10-29 used `_` instead of `.`; they are still recognised as pre-processed (not raw) dataframes.
