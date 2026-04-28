# Chemometric Abbreviations

Abbreviations used in spectral plot annotations (upper-right corner) and saved filenames.

Steps are shown as a chain separated by `→` in annotations and `_` in filenames.

**Example** — `chemometrics_array: ["scatter_correction", "scaling"]`:
- `spectra_raw.png` — annotation: `raw`
- `spectra_snv.png` — annotation: `snv`
- `spectra_snv_mc.png` — annotation: `snv→mc`

## Abbreviation table

| Abbreviation | Method | Category |
|---|---|---|
| `raw` | Raw (unprocessed) spectra | — |
| `d1` | First-order finite-difference derivative | Derivative |
| `d2` | Second-order finite-difference derivative | Derivative |
| `l1` | L1 norm (sum normalisation) | Scatter correction |
| `l2` | L2 norm (Euclidean normalisation) | Scatter correction |
| `max` | Max norm (range normalisation) | Scatter correction |
| `snv` | Standard Normal Variate | Scatter correction |
| `msc` | Multiplicative Scatter Correction | Scatter correction |
| `l1snv` | L1 norm followed by SNV (chained) | Scatter correction |
| `snvmsc` | SNV followed by MSC (chained) | Scatter correction |
| `mc` | Mean centring | Scaling |
| `as` | Autoscaling (mean centring + unit variance) | Scaling |
| `ps` | Pareto scaling (divide by √std) | Scaling |
| `poi` | Poisson scaling (divide by √mean) | Scaling |
| `pca{N}` | PCA, N components (e.g. `pca5`) | Decomposition |
| `ma` | Moving average | Filter |
| `gf` | Gaussian filter | Filter |
| `sg` | Savitzky-Golay (smoothing or derivative) | Filter |
| `lw` | LOWESS | Filter |
| `mf` | Multi-filter (region-wise) | Filter |
| `wc` | Ward agglomerative clustering | Agglomeration |
| `uv` | Univariate selection — combined across indicators | Feature selection |
| `uv-{indicator}` | Univariate selection for a specific indicator (hyphen separator, spaces → hyphens) | Feature selection |

## Chaining rules

Chained scatter corrections (two scalers in `scaler` list) concatenate their abbreviations
without separator, e.g. `l1` + `snv` → `l1snv`.

Multiple steps in `chemometrics_array` chain with `→` in annotations and `_` in filenames,
applied left-to-right; each step uses the previous step's output as input.
