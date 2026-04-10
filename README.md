# bioharmonize

Normalize and validate biological study metadata before downstream analysis.

## Install

```bash
pip install bioharmonize
pip install bioharmonize[anndata]   # AnnData support
pip install bioharmonize[cli]       # CLI support
```

## Quickstart

```python
import bioharmonize as bh

report = bh.clean_obs(df, profile="single_cell_human")
print(report.summary())
```

With AnnData:

```python
import scanpy as sc
import bioharmonize as bh

ad = sc.read_h5ad("study.h5ad")
report = bh.clean_obs(ad.obs, profile="single_cell_human")
print(report.summary())
ad.obs = report.cleaned
```

## What it does

1. **Maps messy columns to canonical names** (e.g. `celltype` -> `cell_type`)
2. **Normalizes common values safely** (e.g. `m` -> `male`, `ctrl` -> `control`)
3. **Shows exactly what changed and what still needs attention** (audit trail)

## CLI

```bash
bioharmonize clean study_obs.csv --profile single_cell_human
bioharmonize validate study_obs.csv --profile single_cell_human
bioharmonize patch study.h5ad --profile single_cell_human
```
